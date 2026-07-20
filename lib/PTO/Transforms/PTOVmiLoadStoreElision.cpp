// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root directory of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// PTOVmiLoadStoreElision.cpp - elide vmi load/store round trips
//===----------------------------------------------------------------------===//
//
// Adapted from PTOFusionLoadStoreElision for the unified VMI path. Inside each
// pto.fusion_region, a TWO-PASS scan over the single-layer scf.for leaf body
// (fused by PTOVmiLoopFusion) and the top-level straight-line segments between
// for's builds a content-version table of every vmi.vload / vmi.vstore, then
// eliminates redundant ones in reverse order.
//
// Canonical base resolution traces pto.castptr -> memref -> pto.pointer_cast
// -> addr constant so that distinct castptr chains to the same compile-time UB
// address compare equal. A vmi.vload has no mask operand, so its read lane set
// is inferred from its consuming op: if all consumers share one mask, that mask
// bounds the read set; if all are mask-free (e.g. vcvt) the read set is the full
// vreg; otherwise the vload is left alone. A vmi.vstore carries its own mask and
// a pmode ("zero" default | "merge"): the store's write lane set is the mask's
// prefix [0,N); under pmode=merge only those lanes are written (inactive lanes
// keep the prior UB content), under pmode=zero the whole region is defined
// (inactive lanes store 0).
//
// Lane sets are modeled as prefix intervals [0,N) (create_mask %N is a prefix
// predicate); masks that cannot be statically resolved (constant_mask, masked
// combinations, non-constant active_lanes) are treated as "unknown" and the
// elision conservatively skips any vload/vstore whose lane set is unknown.
//
// Two passes:
//   Pass 1 (build, forward scan): record each load/store with its (base,
//     offset, lane-set, source value) and mark forward targets:
//       - a vload whose read set is fully covered by a preceding store's write
//         set, with no intervening intersecting write, forwards to that store's
//         value (store->load elision, the store is erased only if dead);
//       - a vload whose read set equals a preceding vload's read set, with no
//         intervening intersecting write, forwards to that load's result
//         (vload->vload dedup);
//       - a store fully overwritten by a later same-base/offset store whose
//         write set covers it is marked dead-store-erase.
//     A merge store invalidates only the lane interval it writes among the
//     preceding entries (a preceding entry fully covered by the merge write
//     set is dead; a partially intersecting one is marked stale so it no
//     longer participates in matching, but is retained so a later vload of the
//     mixed content correctly does NOT forward). A store whose UB is read by a
//     region-escaping op (mte_ub_gm/mte_gm_ub) is marked non-erasable.
//   Pass 2 (eliminate, reverse): for each marked entry, replace the load's
//     uses with the recorded source value and erase the dead loads/stores in
//     reverse order (so a value consumed by a later-forwarded op is replaced
//     before that op is erased). erase is guarded by use_empty().
//
// Runs in the VMI semantic pipeline AFTER PTOVmiLoopFusion + CSE (so cross-for
// UB round trips have become same-block straight-line pairs inside the fused
// loop) and before VMILowerUnifiedToLegacy.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMILOADSTOREELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Trace a value through the vmi UB alias chain to its canonical root: a
// pto.castptr (memref->ptr) whose memref is a pto.pointer_cast of a constant
// address. For values not on this chain, return the value itself.
static Value getCanonicalTrackedValue(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;
    if (auto cp = dyn_cast<pto::CastPtrOp>(def)) {
      value = cp->getOperand(0);
      continue;
    }
    if (auto pc = dyn_cast<pto::PointerCastOp>(def)) {
      if (!pc.getOperands().empty()) {
        Value addr = pc.getOperands()[0];
        if (auto c = addr.getDefiningOp<arith::ConstantOp>())
          return pc.getResult(); // canonicalize to the pointer_cast result
      }
      break;
    }
    break;
  }
  return value;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  Value cl = getCanonicalTrackedValue(lhs);
  Value cr = getCanonicalTrackedValue(rhs);
  if (cl == cr)
    return true;
  if (!cl || !cr)
    return false;
  if (cl.getType() != cr.getType())
    return false;
  Operation *ld = cl.getDefiningOp();
  Operation *rd = cr.getDefiningOp();
  if (!ld || !rd)
    return false;
  // Two pointer_cast of the same constant address with the same memref type
  // are the same UB.
  if (auto lp = dyn_cast<pto::PointerCastOp>(ld)) {
    auto rp = dyn_cast<pto::PointerCastOp>(rd);
    if (!rp)
      return false;
    if (lp.getOperands().empty() || rp.getOperands().empty())
      return false;
    if (lp.getOperands()[0] == rp.getOperands()[0] &&
        lp.getResult().getType() == rp.getResult().getType())
      return true;
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
      if (!areEquivalentValues(a, b))
        return false;
    }
    return true;
  }
  // Two identical pure constants.
  if (isa<arith::ConstantOp>(ld) && isa<arith::ConstantOp>(rd))
    return ld->getAttrDictionary() == rd->getAttrDictionary();
  return cl == cr;
}

static bool areEquivalentMaskValues(Value lhs, Value rhs) {
  return areEquivalentValues(lhs, rhs);
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
    if (isUnknown || other.isUnknown)
      return false;
    if (isFull())
      return true;
    if (other.isFull())
      return false;
    return *upperBound >= *other.upperBound;
  }
  // Do the two lane sets intersect? Unknown => conservatively intersects.
  // Two non-empty prefix intervals [0,A) and [0,B) always share lane 0.
  bool intersects(const LaneRange &other) const {
    if (isUnknown || other.isUnknown)
      return true;
    if (isFull() || other.isFull())
      return true;
    return *upperBound > 0 && *other.upperBound > 0;
  }
};

// Resolve a mask Value to a prefix LaneRange. create_mask %constN -> [0,N).
// Anything else (constant_mask, mask_and, non-const active_lanes) -> unknown.
static LaneRange resolveMaskLanes(Value mask) {
  if (!mask)
    return LaneRange::full(); // no mask operand => full predicate
  Operation *def = mask.getDefiningOp();
  if (auto cm = dyn_cast<pto::VMICreateMaskOp>(def)) {
    if (auto c =
            cm.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      if (auto iv = dyn_cast<IntegerAttr>(c.getValue())) {
        int64_t n = iv.getInt();
        if (n >= 0)
          return LaneRange::prefix(static_cast<unsigned>(n));
      }
    }
  }
  return LaneRange::unknown();
}

// The vreg width (VL) of a vload result, for bounding a full read. Returns 0
// if not a vmi.vreg type.
static unsigned getVRegWidth(Type t) {
  if (auto vt = dyn_cast<pto::VMIVRegType>(t))
    return static_cast<unsigned>(vt.getElementCount());
  return 0;
}

// A vmi.vload has no mask operand. Infer the mask constraint from its
// consuming op(s): a vmi compute op (vmax/vmuls/vsub/vexp/vadd/...) carries a
// mask operand, while a few (e.g. vcvt) take no mask. The result is an
// optional constraint:
//   std::nullopt             -> cannot infer (mixed/ambiguous masks, or a
//                              region-bearing consumer): conservatively do not
//                              forward.
//   some(Value{}) [empty]    -> all consumers are mask-free: any tracked store
//                              matches (forward is safe regardless of store
//                              mask).
//   some(Value{nonEmpty})    -> every consumer shares this one mask: only a
//                              tracked store with an equivalent mask matches.
static std::optional<Value>
inferVMILoadUserMask(pto::VMIvLoadOp load) {
  // Whether at least one consuming op has been seen (vs. a load with no users,
  // which is conservatively not forwardable).
  bool seenConsumer = false;
  // The inferred mask, if any consumer carries one. Empty Value means
  // "no mask constraint so far".
  Value inferred;
  bool hasMaskConstraint = false;
  for (OpOperand &use : load->getResult(0).getUses()) {
    Operation *owner = use.getOwner();
    if (!owner || owner->getNumRegions() != 0)
      return std::nullopt;
    seenConsumer = true;
    Value opMask;
    for (Value operand : owner->getOperands()) {
      if (!isa<pto::VMIMaskType>(operand.getType()))
        continue;
      if (!opMask)
        opMask = operand;
      else if (!areEquivalentMaskValues(opMask, operand))
        return std::nullopt;
    }
    if (!opMask)
      continue; // mask-free consumer (e.g. vcvt): no constraint contributed
    if (!hasMaskConstraint) {
      inferred = opMask;
      hasMaskConstraint = true;
    } else if (!areEquivalentMaskValues(inferred, opMask)) {
      return std::nullopt;
    }
  }
  if (!seenConsumer)
    return std::nullopt;
  return inferred; // empty if all consumers mask-free, else the shared mask
}

// Whether `op` is safe to step over without invalidating tracked UB content:
// it neither writes/reads a tracked UB (only vmi.vload/vstore do) nor is a
// sync/DMA (mte_*/set_flag/...) that could alias the buffer. VMI compute ops
// (vmuls/vmax/...), pointer_cast/castptr/create_mask, and arith are all Pure
// in the VMI dialect but do not all implement MemoryEffectOpInterface, so
// MLIR's isMemoryEffectFree rejects them; the explicit name guard is both
// narrower and correct for the VMI path.
static bool isTransparentToTrackedStores(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  StringRef name = op->getName().getStringRef();
  if (isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(op))
    return false;
  static const llvm::StringLiteral kImpure[] = {
      "pto.mte_gm_ub",  "pto.mte_ub_gm",   "pto.set_flag",
      "pto.wait_flag",  "pto.mem_bar",      "pto.pipe_barrier",
      "pto.vecscope",   "pto.strict_vecscope"};
  for (auto n : kImpure)
    if (name == n)
      return false;
  return true;
}

// A region-escaping op reads a UB and exports it out of the region (mte_ub_gm
// writes UB->GM, mte_gm_ub writes GM->UB). For elision correctness: an escape
// READ of a store's UB means the store is observable and must NOT be erased
// even after its value is forwarded to a load (the escape re-reads the UB).
// mte_gm_ub is an escape WRITE: it redefines the UB from GM, so any prior
// tracked content of that UB is stale.
static bool isEscapeReadOfUB(Operation *op, Value &ubRead) {
  if (auto mte = dyn_cast<pto::MteUbGmOp>(op)) {
    ubRead = mte.getSource();
    return true;
  }
  return false;
}
static bool isEscapeWriteToUB(Operation *op, Value &ubWritten) {
  if (auto mte = dyn_cast<pto::MteGmUbOp>(op)) {
    ubWritten = mte.getDestination();
    return true;
  }
  return false;
}

// A content-version table entry for one vload or vstore. Built in Pass 1 and
// consumed (mutated by marking) in Pass 2.
struct ContentEntry {
  Operation *op = nullptr;
  Value base;          // canonical UB (pointer_cast result, traced from dest/src)
  Value offset;
  LaneRange lanes;     // read set (load) / write set (store)
  bool isLoad = false;
  Value sourceValue;    // store.value or load.result (forward target value)

  // Pass 1 marks:
  int forwardToIdx = -1; // >=0: this load forwards to entries[forwardToIdx].sourceValue
  bool eraseMark = false;     // this op should be erased in Pass 2 (dead load/store)
  bool escapeMark = false;    // a store whose UB is read by a region-escaping op: keep
  bool stale = false;         // no longer participates as a forward target (merge partial write)
};

// Two-pass elision over a straight-line range (a fused scf.for body, or the
// top-level ops of a fusion_region between two for's). Pass 1 builds a content
// table and marks forward targets / dead stores; Pass 2 applies replacements
// and erases in reverse order. A scf.for in the range (only at the top level)
// is not transparent (it has a region), so it flushes the table — correct, as
// a for body may read/write tracked UBs.
template <typename OpRange>
static bool elideOpRange(OpRange ops) {
  SmallVector<ContentEntry, 8> entries;
  bool changed = false;

  // ---- Pass 1: build + mark (forward scan, no IR mutation) ----
  // Match helpers operating on the live entry set (stale entries skipped).
  auto sameLoc = [&](const ContentEntry &e, Value base, Value offset) {
    return areEquivalentValues(e.base, base) &&
           areEquivalentValues(e.offset, offset);
  };

  for (Operation &op : ops) {
    if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      // Resolve the vload read lane set from its consumer mask.
      std::optional<Value> inferredMask = inferVMILoadUserMask(load);
      LaneRange readLanes;
      if (!inferredMask) {
        readLanes = LaneRange::unknown();
      } else if (!*inferredMask) {
        // all consumers mask-free: full vreg read
        unsigned vl = getVRegWidth(load->getResult(0).getType());
        readLanes = vl ? LaneRange::prefix(vl) : LaneRange::full();
      } else {
        // Consumers share one mask: the read set is bounded by its prefix
        // [0,N). An unresolvable mask yields unknown (skip).
        readLanes = resolveMaskLanes(*inferredMask);
      }
      Value base = load.getSource();
      Value offset = load.getOffset();

      if (readLanes.isUnknownSet()) {
        // Cannot reason; record a non-matchable entry so later vloads of the
        // same loc see it (and, lacking a cover, do not forward to it).
        entries.push_back({load, base, offset, readLanes, true,
                           load->getResult(0), -1, false, false, false});
        continue;
      }

      // Look for a preceding entry that fully covers readLanes with no
      // intervening intersecting write. Scan from nearest backwards.
      int matchIdx = -1;
      for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
        ContentEntry &e = entries[i];
        if (e.stale || e.eraseMark)
          continue;
        if (!sameLoc(e, base, offset))
          continue;
        // Need e.lanes to fully cover readLanes.
        if (!e.lanes.contains(readLanes))
          continue;
        // For a store match: any intervening write to the same loc between e
        // and this load would have invalidated e (it would be stale/erased or
        // a newer entry). Because stale entries are skipped and a later write
        // to intersecting lanes marks prior entries stale, reaching here means
        // no intervening write touched readLanes -> safe to forward.
        matchIdx = i;
        break;
      }
      if (matchIdx >= 0) {
        entries.push_back({load, base, offset, readLanes, true,
                           load->getResult(0), matchIdx, true, false, false});
        changed = true; // load will be forwarded + erased in Pass 2
      } else {
        entries.push_back({load, base, offset, readLanes, true,
                           load->getResult(0), -1, false, false, false});
      }
      continue;
    }

    if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
      Value base = store.getDestination();
      Value offset = store.getOffset();
      Value mask = store.getMask().empty() ? Value() : store.getMask().front();
      LaneRange writeLanes = resolveMaskLanes(mask);
      // pmode: "merge" => only writeLanes written; "zero"(default)/absent =>
      // whole region defined (inactive lanes store 0 -> treat as full cover).
      bool pmodeMerge = false;
      if (auto pma = store.getPmodeAttr())
        pmodeMerge = pma.getValue().equals_insensitive("merge");
      if (!pmodeMerge)
        writeLanes = LaneRange::full(); // zero: entire UB defined

      // Mark preceding same-loc entries by how this write touches their lanes:
      //  - fully covered -> a store is dead (eraseMark, unless it escapes); the
      //    entry stops matching (stale).
      //  - partial overlap (merge) -> the entry no longer fully represents the
      //    current UB content, so it must not be a forward target anymore
      //    (stale), but it is neither dead nor erasable (other lanes may still
      //    be read / escape).
      for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
        ContentEntry &e = entries[i];
        if (!sameLoc(e, base, offset))
          continue;
        if (writeLanes.contains(e.lanes)) {
          if (!e.isLoad && !e.escapeMark)
            e.eraseMark = true;
          e.stale = true;
        } else if (writeLanes.intersects(e.lanes)) {
          e.stale = true;
        }
      }
      entries.push_back({store, base, offset, writeLanes, false,
                         store.getValues().front(), -1, false, false, false});
      continue;
    }

    // Non-load/store ops.
    if (!isTransparentToTrackedStores(&op)) {
      // Region-escaping or aliasing op. mte_ub_gm reads a UB (escape: keep its
      // store); mte_gm_ub writes a UB (redefines: invalidate prior entries);
      // other impure ops conservatively invalidate everything.
      Value esc;
      if (isEscapeReadOfUB(&op, esc)) {
        // mte_ub_gm reads a UB out of the region: its store is observable and
        // must survive even after forwarding. The read does not redefine the
        // UB, so entries keep matching.
        Value canon = getCanonicalTrackedValue(esc);
        for (auto &e : entries)
          if (!e.isLoad && areEquivalentValues(e.base, canon))
            e.escapeMark = true;
        continue;
      }
      if (isEscapeWriteToUB(&op, esc)) {
        // mte_gm_ub redefines the UB from GM: prior tracked content is invalid.
        Value canon = getCanonicalTrackedValue(esc);
        for (auto &e : entries)
          if (areEquivalentValues(e.base, canon))
            e.stale = true;
        continue;
      }
      // Other impure (set_flag/mem_bar/scf.for body that may alias tracked
      // UBs): mark every existing entry stale. In a two-pass design entries
      // cannot be dropped mid-scan — stale preserves any already-recorded
      // forward marks for Pass 2 while preventing further matching against
      // these (possibly-aliased) entries. This is the two-pass analog of the
      // old single-pass `trackedStores.clear()`.
      for (auto &e : entries)
        e.stale = true;
    }
  }

  // ---- Pass 2: eliminate (reverse order) ----
  // Replace forwarded loads' uses first (reverse so a value consumed by a
  // later-forwarded op is replaced before that op is erased), then erase dead
  // loads/stores guarded by use_empty.
  for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
    ContentEntry &e = entries[i];
    if (e.forwardToIdx >= 0 && e.isLoad) {
      Value target = entries[e.forwardToIdx].sourceValue;
      e.op->getResult(0).replaceAllUsesWith(target);
    }
  }
  for (int i = static_cast<int>(entries.size()) - 1; i >= 0; --i) {
    ContentEntry &e = entries[i];
    if (e.eraseMark && e.op->use_empty())
      e.op->erase();
    else if (e.forwardToIdx >= 0 && e.isLoad && e.op->use_empty())
      e.op->erase();
  }
  return changed;
}

// Run the two-pass elision over each fusion_region in two scopes:
//  1. the top-level ops of the region body (the prologue/between/epilogue
//     straight-line segments separated by scf.for's) — this catches UB
//     round trips that live OUTSIDE the fused for's, e.g. a reduce's final
//     vstore to UB followed by a vload of the same UB for the next stage
//     (ColMax -> tmuls -> ColExpand-sub broadcast). A scf.for in this range
//     marks existing entries stale (its body may read/write tracked UBs).
//  2. each scf.for body nested in the region (the fused leaf body) — this
//     catches same-iteration UB round trips inside the loop.
static bool elideInRegion(pto::FusionRegionOp region) {
  bool changed = false;
  Block &body = region.getBody().front();
  // Top-level: walk all ops except the region's pto.yield terminator.
  changed |= elideOpRange(body.without_terminator());
  // Each nested scf.for body.
  region.getBody().walk([&](scf::ForOp loop) {
    if (loop->getParentOfType<pto::FusionRegionOp>() == region)
      changed |= elideOpRange(loop.getBody()->without_terminator());
    return WalkResult::advance();
  });
  return changed;
}

struct PTOVmiLoadStoreElisionPass
    : public mlir::pto::impl::PTOVmiLoadStoreElisionBase<
          PTOVmiLoadStoreElisionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    bool changed = false;
    func.walk([&](pto::FusionRegionOp region) {
      changed |= elideInRegion(region);
    });
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiLoadStoreElisionPass() {
  return std::make_unique<PTOVmiLoadStoreElisionPass>();
}
