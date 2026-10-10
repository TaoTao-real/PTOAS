// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details.
// You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Shared compile-time UB access model for the VMI transform passes.
//
// Pure analysis primitives extracted verbatim from PTOVmiLoadStoreElision.cpp
// so the function-level DSE pass reasons with the SAME base identity, affine
// address, alias, and lane-set model as the region-local elision:
//   - getCanonicalTrackedValue / AffineAddr / parseAffineAddr: the UB alias
//     chain and the affine (iv, baseOffset, coeff) address decomposition;
//   - BaseIdentity / basesMustAlias / basesMayAlias: must/may-alias over
//     storage roots and affine keys;
//   - LaneRange / resolveMaskLanes / getVRegWidth: the prefix-interval
//     lane-set model for masks and vreg widths;
//   - isContinuousSingleVLoad/VStore: the first-version shape guard;
//   - isEscapeReadOfUB / isEscapeWriteToUB: the MTE escape classification.
//
// Code motion only: the elision pass includes this header and its observable
// behavior is unchanged.

#ifndef PTO_TRANSFORMS_VMIUBACCESSMODEL_H
#define PTO_TRANSFORMS_VMIUBACCESSMODEL_H

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/VmiMemoryLocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {

// Trace a value through the vmi UB alias chain to its canonical root: a
// pto.castptr (memref->ptr) whose memref is a pto.pointer_cast of a constant
// address. For values not on this chain, return the value itself.
inline Value getCanonicalTrackedValue(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def) {
      break;
    }
    if (auto cp = dyn_cast<pto::CastPtrOp>(def)) {
      value = cp->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

// ----------------------------------------------------------------------------
// Affine UB address model.
//
// A vload/vstore base may resolve to a COMPILE-TIME-DERIVABLE affine address
// instead of a bare constant: `pointer_cast(addi(muli(%iv, cK), cB))` from a
// loop-carried induction variable. Such an address is a deterministic function
// of the iteration, so two accesses built from the SAME induction variable and
// the SAME (coeff, base) compare equal — they must alias. This lets a
// store->load round trip on the same affine UB be folded, and lets us reason
// about whether a dynamic store can alias a tracked constant store.
//
// The affine key is (iv, baseOffset, coeff); two affine addresses alias iff
// they share the same iv and (baseOffset, coeff) differ by a multiple of the
// vector-interval width (so the intervals on this iteration overlap). Because
// the model is conservative (an affine address may alias ANY constant UB on
// some iteration), affine keys are used for MUST-ALIAS (same iv+offset) but
// never for must-not-alias against a constant.
// ----------------------------------------------------------------------------
struct AffineAddr {
  Value iv;                 // loop induction variable (null => pure constant)
  int64_t baseOffset = 0;   // constant part
  int64_t coeff = 0;        // iv multiplier
  bool isAffine = false;    // valid affine form (not a bare runtime ptr)
};

// Peel type-preserving/unary casts that wrap an affine expression or its IV:
// arith.index_cast is the common wrapper in the VMI path (loop IV is index,
// the address is i64). The peeled value is the affine operand.
inline Value peelAffineWrappers(Value v) {
  // Bound the peel to avoid pathological cycles in the def chain.
  unsigned steps = 0;
  while (v && steps++ < 16) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      break;
    }
    if (auto ic = dyn_cast<arith::IndexCastOp>(def)) {
      Value in = ic.getIn();
      if (in == v) {
        break;
      }
      v = in;
      continue;
    }
    break;
  }
  return v;
}

// The constant multiplier operand of a muli and, in `iv`, the opposite
// operand (the induction-variable side). Either operand may be the constant;
// when the rhs is not a constant the lhs is tried and, failing that, the
// returned op is null (which the caller rejects).
inline arith::ConstantOp getMuliConstantOperand(Value lhs, Value rhs,
                                                Value &iv) {
  if (auto cst = rhs.getDefiningOp<arith::ConstantOp>()) {
    iv = lhs;
    return cst;
  }
  iv = rhs;
  return lhs.getDefiningOp<arith::ConstantOp>();
}

// addi(muli(%iv, c), b) -> {iv, baseOffset=b, coeff=c}. Returns a non-affine
// AffineAddr when the form is not recognized (the caller then tries the
// addi(%iv, b) form).
inline AffineAddr parseMuliAddAffineAddr(Value lhs, Value rhs) {
  AffineAddr result;
  auto muli = dyn_cast_or_null<arith::MulIOp>(rhs.getDefiningOp());
  if (!muli) {
    return result;
  }
  Value mulLhs = peelAffineWrappers(muli.getLhs());
  Value mulRhs = peelAffineWrappers(muli.getRhs());
  // The multiplier operand must be a constant; the other is the IV.
  Value iv;
  arith::ConstantOp cst = getMuliConstantOperand(mulLhs, mulRhs, iv);
  arith::ConstantOp b = lhs.getDefiningOp<arith::ConstantOp>();
  if (!cst || !b) {
    return result;
  }
  auto cval = dyn_cast<IntegerAttr>(cst.getValue());
  auto bval = dyn_cast<IntegerAttr>(b.getValue());
  if (!cval || !bval) {
    return result;
  }
  result.isAffine = true;
  result.iv = iv;
  result.coeff = cval.getInt();
  result.baseOffset = bval.getInt();
  return result;
}

// addi(%iv, b) -> {iv, baseOffset=b, coeff=1}. Returns a non-affine AffineAddr
// when the rhs is not an integer constant.
inline AffineAddr parseAddIAffineAddr(arith::AddIOp addi) {
  AffineAddr result;
  Value lhs = addi.getLhs();
  Value rhs = addi.getRhs();
  // addi(muli(%iv, c), b): affine in RHS.
  AffineAddr muliForm = parseMuliAddAffineAddr(lhs, rhs);
  if (muliForm.isAffine) {
    return muliForm;
  }
  // addi(%iv, b): affine coeff 1.
  if (auto b = rhs.getDefiningOp<arith::ConstantOp>()) {
    if (auto bval = dyn_cast<IntegerAttr>(b.getValue())) {
      result.isAffine = true;
      result.iv = peelAffineWrappers(lhs);
      result.coeff = 1;
      result.baseOffset = bval.getInt();
    }
  }
  return result;
}

// muli(%iv, c) -> {iv, baseOffset=0, coeff=c}. Returns a non-affine AffineAddr
// when no operand side is an integer constant.
inline AffineAddr parseMulIAffineAddr(arith::MulIOp muli) {
  AffineAddr result;
  Value mulLhs = peelAffineWrappers(muli.getLhs());
  Value mulRhs = peelAffineWrappers(muli.getRhs());
  // The multiplier operand must be a constant; the other is the IV.
  Value iv;
  arith::ConstantOp cst = getMuliConstantOperand(mulLhs, mulRhs, iv);
  if (cst) {
    if (auto cval = dyn_cast<IntegerAttr>(cst.getValue())) {
      result.isAffine = true;
      result.iv = iv;
      result.coeff = cval.getInt();
      result.baseOffset = 0;
    }
  }
  return result;
}

// Parse an address Value into an AffineAddr. Handles:
//   constant              -> {iv=null, baseOffset=c}
//   addi(muli(%iv, c), b) -> {iv, baseOffset=b, coeff=c}
//   addi(%iv, b)          -> {iv, baseOffset=b, coeff=1}
//   muli(%iv, c)          -> {iv, baseOffset=0, coeff=c}
// The IV and the addr may be wrapped in index_cast (peeled). The IV is stored
// as its peeled value; equality is decided by areEquivalentValues (two casts of
// the same induction value are equivalent).
// Anything else (block arg, untraceable) -> isAffine=false.
inline AffineAddr parseAffineAddr(Value addr) {
  AffineAddr result;
  if (!addr) {
    return result;
  }
  addr = peelAffineWrappers(addr);
  if (auto c = addr.getDefiningOp<arith::ConstantOp>()) {
    if (auto iv = dyn_cast<IntegerAttr>(c.getValue())) {
      result.isAffine = true;
      result.coeff = 0;
      result.baseOffset = iv.getInt();
    }
    return result;
  }
  // A block argument (function parameter, outer-region iter arg) or any value
  // without a defining op is a runtime address: it has no affine decomposition.
  // dyn_cast_or_null keeps that case on the untrackable path instead of
  // asserting on a null Operation.
  if (auto addi = dyn_cast_or_null<arith::AddIOp>(addr.getDefiningOp())) {
    return parseAddIAffineAddr(addi);
  }
  if (auto muli = dyn_cast_or_null<arith::MulIOp>(addr.getDefiningOp())) {
    return parseMulIAffineAddr(muli);
  }
  return result;
}

// A canonical, comparable identity for a vload/vstore base: either a constant
// pointer_cast address, or an affine (iv, baseOffset, coeff) key. Two bases
// compare equal (must-alias) iff their constants are equal, or their affine key
// is equal (same iv, same baseOffset, same coeff). This is the compile-time
// identity used for forwarding matches.
struct BaseIdentity {
  Value base;          // canonical base (pointer_cast result), or null if untrackable
  AffineAddr affine;   // affine decomposition (isAffine==false => untrackable)
  bool isConstantAddr; // resolved to a bare constant pointer_cast
  std::optional<pto::VmiStorageRoot> storageRoot;
};

// Resolve a vload/vstore base to a BaseIdentity. Returns an untrackable identity
// (base==null) for block args / untraceable / non-affine bases.
inline BaseIdentity resolveBaseIdentity(Value base) {
  if (auto root = pto::resolveVmiStorageRoot(base)) {
    return {base, {}, true, root};
  }
  if (auto cast = base.getDefiningOp<pto::CastPtrOp>()) {
    Value input = cast.getInput();
    if (isa<IntegerType>(input.getType()) || input.getType().isIndex()) {
      AffineAddr affine = parseAffineAddr(input);
      if (affine.isAffine) {
        return {base, affine, affine.iv == nullptr, std::nullopt};
      }
    }
  }
  Value canon = getCanonicalTrackedValue(base);
  if (!canon) {
    return {};
  }
  if (isa<BlockArgument>(canon)) {
    return {};
  }
  return {};
}
inline bool isTrackableIdentity(const BaseIdentity &id) {
  return id.base != nullptr;
}

inline bool areEquivalentValues(Value lhs, Value rhs) {
  Value cl = getCanonicalTrackedValue(lhs);
  Value cr = getCanonicalTrackedValue(rhs);
  if (cl == cr) {
    return true;
  }
  if (!cl || !cr) {
    return false;
  }
  if (cl.getType() != cr.getType()) {
    return false;
  }
  Operation *ld = cl.getDefiningOp();
  Operation *rd = cr.getDefiningOp();
  if (!ld || !rd) {
    return false;
  }
  // Two identical pure ops (same name, operands, attrs, result type) — e.g.
  // two pto.vmi.create_mask with the same active_lanes constant. This is what
  // makes masks produced by distinct create_mask ops compare equal.
  if (ld->getName() == rd->getName() && ld->getNumRegions() == 0 &&
      rd->getNumRegions() == 0 &&
      ld->getNumOperands() == rd->getNumOperands() &&
      ld->getAttrDictionary() == rd->getAttrDictionary() &&
      llvm::equal(ld->getOperandTypes(), rd->getOperandTypes())) {
    for (auto [a, b] :
         llvm::zip(ld->getOperands(), rd->getOperands())) {
      if (!areEquivalentValues(a, b)) {
        return false;
      }
    }
    return true;
  }
  // Two identical pure constants.
  if (isa<arith::ConstantOp>(ld) && isa<arith::ConstantOp>(rd)) {
    return ld->getAttrDictionary() == rd->getAttrDictionary();
  }
  return cl == cr;
}

inline bool areEquivalentMaskValues(Value lhs, Value rhs) {
  return areEquivalentValues(lhs, rhs);
}

// Whether two base identities reference the SAME UB on this iteration (must
// alias). This is the identity used for content forwarding: two accesses with
// must-alias bases read/write the same bytes, so a store->load forward is safe.
//   - two constant pointer_casts: must-alias iff the same address;
//   - two affine keys: must-alias iff the same iv and the same (baseOffset,
//     coeff) — they compute the identical address on every iteration;
//   - constant vs affine, or any untrackable: NOT must-alias (may differ).
inline bool basesMustAlias(const BaseIdentity &lhs, const BaseIdentity &rhs) {
  if (!isTrackableIdentity(lhs) || !isTrackableIdentity(rhs)) {
    return false;
  }
  if (lhs.storageRoot && rhs.storageRoot) {
    return *lhs.storageRoot == *rhs.storageRoot;
  }
  if (areEquivalentValues(lhs.base, rhs.base)) {
    return true;
  }
  if (lhs.affine.isAffine && rhs.affine.isAffine) {
    return areEquivalentValues(lhs.affine.iv, rhs.affine.iv) &&
           lhs.affine.baseOffset == rhs.affine.baseOffset &&
           lhs.affine.coeff == rhs.affine.coeff;
  }
  return false;
}

// Whether two base identities MAY alias on some iteration. Conservative:
//   - two constant pointer_casts: alias iff the same address (a constant never
//     sweeps a range). Note `isAffine` is also true for a bare constant (iv is
//     null), so the constant/constant case must be decided by `isConstantAddr`
//     BEFORE the affine sweep rule, or two distinct constants would be
//     misclassified as may-alias.
//   - an affine base may alias any constant UB on some iteration (its address
//     sweeps a range as the loop runs), so it is treated as may-alias with a
//     constant;
//   - any untrackable base (block arg / non-affine) may alias anything.
// This drives the nonErasable marking: a store whose UB a later load MAY read
// must not be deleted by overwrite-DSE even after its value is forwarded.
inline bool basesMayAlias(const BaseIdentity &lhs, const BaseIdentity &rhs) {
  if (!isTrackableIdentity(lhs) || !isTrackableIdentity(rhs)) {
    return true; // unknown may alias anything
  }
  if (lhs.storageRoot && rhs.storageRoot) {
    return pto::mayAliasVmiStorageRoot(*lhs.storageRoot, *rhs.storageRoot);
  }
  // Both bare constants: alias iff the same address. This must be checked
  // first, because a bare constant carries isAffine==true (iv==null) and would
  // otherwise fall through to the affine-sweep rule below.
  if (lhs.isConstantAddr && rhs.isConstantAddr) {
    return areEquivalentValues(lhs.base, rhs.base);
  }
  if (lhs.affine.isAffine || rhs.affine.isAffine) {
    return true; // affine sweeps a range -> may hit a constant
  }
  return false;
}

// ----------------------------------------------------------------------------
// Lane-set modeling (prefix interval [0,N)).
//
// create_mask %N is a prefix predicate: lanes [0,N) are active. Lane sets are
// therefore representable as a prefix interval [0,N). A vload reads the full
// vreg [0,VL) (VL = vreg element count) unless its inferred consumer mask bounds
// the read to [0,N) <= [0,VL). A vstore writes its mask's prefix [0,N); under
// pmode=merge only those lanes are written (inactive lanes keep prior UB
// content), under pmode=zero the whole region is defined (inactive lanes store
// 0). Masks that cannot be statically resolved (constant_mask, masked
// combinations, non-constant active_lanes) yield Unknown and the elision
// conservatively skips the affected vload/vstore.
// ----------------------------------------------------------------------------
struct LaneRange {
  // Inclusive upper bound of the prefix [0, upperBound). std::nullopt means the
  // full set [0, VL) (i.e. a mask-free / full vreg read). isUnknown marks a
  // mask we cannot reason about — nothing involving it is forwardable.
  std::optional<unsigned> upperBound;
  bool isUnknown = false;

  static LaneRange full() { return {std::nullopt, false}; }
  static LaneRange unknown() { return {std::nullopt, true}; }
  static LaneRange prefix(unsigned n) { return {n, false}; }

  bool isFull() const { return !isUnknown && !upperBound.has_value(); }
  bool isUnknownSet() const { return isUnknown; }

  // Does this lane set contain (cover) `other`? Unknown never covers or is
  // covered (conservatively not a subset/superset).
  bool contains(const LaneRange &other) const {
    if (isUnknown || other.isUnknown) {
      return false;
    }
    if (isFull()) {
      return true;
    }
    if (other.isFull()) {
      return false;
    }
    return *upperBound >= *other.upperBound;
  }
  // Do the two lane sets intersect? Unknown => conservatively intersects.
  // Two non-empty prefix intervals [0,A) and [0,B) always share lane 0.
  bool intersects(const LaneRange &other) const {
    if (isUnknown || other.isUnknown) {
      return true;
    }
    if (isFull() || other.isFull()) {
      return true;
    }
    return *upperBound > 0 && *other.upperBound > 0;
  }
};

// Resolve a mask Value to a prefix LaneRange. create_mask %constN -> [0,N).
// Anything else (constant_mask, mask_and, non-const active_lanes) -> unknown.
inline LaneRange resolveMaskLanes(Value mask) {
  if (!mask) {
    return LaneRange::full(); // no mask operand => full predicate
  }
  Operation *def = mask.getDefiningOp();
  if (auto cm = dyn_cast<pto::VMICreateMaskOp>(def)) {
    if (auto c =
            cm.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      if (auto iv = dyn_cast<IntegerAttr>(c.getValue())) {
        int64_t n = iv.getInt();
        if (n >= 0) {
          return LaneRange::prefix(static_cast<unsigned>(n));
        }
      }
    }
  }
  return LaneRange::unknown();
}

// The vreg width (VL) of a vload result, for bounding a full read. Returns 0
// if not a vmi.vreg type.
inline unsigned getVRegWidth(Type t) {
  if (auto vt = dyn_cast<pto::VMIVRegType>(t)) {
    return static_cast<unsigned>(vt.getElementCount());
  }
  return 0;
}

// First-version shape guard: this model covers ONLY the continuous, single
// result (load) / single value (store) vmi.vload/vmi.vstore, with no stride,
// block_stride or group. Other memory operations, including internal
// interleaved and broadcast loads, grouped load/store and block-strided
// load/store, are left untouched: loads are recorded as
// non-matchable (unknown read set) and flush the table; stores flush the
// table (we cannot soundly model which lanes they define).
inline bool isContinuousSingleVLoad(pto::VMIvLoadOp op) {
  if (op.getStride() || op.getBlockStride()) {
    return false;
  }
  if (op.getGroup()) {
    return false;
  }
  return true;
}
inline bool isContinuousSingleVStore(pto::VMIvStoreOp op) {
  if (op.getStride() || op.getBlockStride()) {
    return false;
  }
  if (op.getGroup()) {
    return false;
  }
  return true;
}

// A store whose value is an element-storage-width-changing conversion may be
// lowered to a SUBLANE-COMPACTING store: the f32->bf16 narrowing form packs each
// 16-bit payload into the low half of a 32-bit lane (a PK_B32 store), so the UB
// image written by the store is the logical bf16 lane stream rather than the
// register's bit pattern. Forwarding a later UB load to that value would hand
// the load's consumers the packed register instead — a different value than the
// load would read. A width-changing one-operand/one-result vreg conversion (any
// of truncf/extf/fptosi/trunci/..., in either direction) therefore must not act
// as a forward source.
inline bool isRepresentationChangingStoreValue(Value value) {
  Operation *def = value.getDefiningOp();
  bool isSingleOperandVRegConversion =
      def && def->getNumOperands() == 1 && def->getNumResults() == 1;
  if (!isSingleOperandVRegConversion) {
    return false;
  }
  auto source = dyn_cast<VMIVRegType>(def->getOperand(0).getType());
  auto result = dyn_cast<VMIVRegType>(def->getResult(0).getType());
  if (!source || !result) {
    return false;
  }
  return getPTOStorageElemBitWidth(source.getElementType()) !=
         getPTOStorageElemBitWidth(result.getElementType());
}

// A region-escaping op reads a UB and exports it out of the region (mte_ub_gm
// writes UB->GM, mte_gm_ub writes GM->UB). For elision correctness: an escape
// READ of a store's UB means the store is observable and must NOT be erased
// even after its value is forwarded to a load (the escape re-reads the UB).
// mte_gm_ub is an escape WRITE: it redefines the UB from GM, so any prior
// tracked content of that UB is stale.
inline bool isEscapeReadOfUB(Operation *op, Value &ubRead) {
  if (auto mte = dyn_cast<pto::MteUbGmOp>(op)) {
    ubRead = mte.getSource();
    return true;
  }
  return false;
}
inline bool isEscapeWriteToUB(Operation *op, Value &ubWritten) {
  if (auto mte = dyn_cast<pto::MteGmUbOp>(op)) {
    ubWritten = mte.getDestination();
    return true;
  }
  return false;
}

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VMIUBACCESSMODEL_H
