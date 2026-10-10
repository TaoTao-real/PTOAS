// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details.
// You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Function-level dead-store elimination for VMI UB stores.
//
// PTOVmiLoadStoreElision reasons strictly inside one straight-line range
// (a fused scf.for body, a vecscope body, the top level of a fusion_region)
// and deliberately has no terminal "no later reader" rule: whether a store is
// observable is a FUNCTION-scope question, because the readers that matter
// (later loops, the post-region mte_ub_gm drains) live outside the range.
// This pass complements it with exactly that rule, at function scope:
//
//   A pto.vmi.vstore to a UB destination is erased iff no potential reader of
//   its written bytes executes after it anywhere in the function.
//
// Reader model (closed set, conservative outside it):
//   - pto.vmi.vload: a bounded read [base+offset, +VL*elemBytes) when its base
//     is trackable and its shape is the continuous single-result form;
//     otherwise a root-wide read of its base (an untrackable base reads
//     anywhere).
//   - pto.mte_ub_gm: a root-wide read of its source (the escape drain).
//   - func.call / func.call_indirect: reads anywhere.
//   - any other op consuming a UB pointer operand: a root-wide read of that
//     operand unless the op declares no Read effect (pto.addptr is Pure; a
//     vstore declares Write only). Ops without MemoryEffectOpInterface must
//     be mlir::isPure to be skipped — same audited closed-set policy as the
//     elision pass.
//
// Ordering is the function's sequential program order over the region tree:
// sibling subtrees complete in block order; inside one block, later ops see
// earlier writes. The one refinement over plain statement order is the loop
// case: a read BEFORE a store in the same scf.for body also runs in every
// LATER iteration, so it observes the store's write from iteration i when the
// read runs at iteration j > i. Cross-iteration observation is disproved only
// when both offsets share one affine key (same induction variable, base and
// coefficient) and the per-iteration stride is at least the store's write
// width, which makes the byte intervals of distinct iterations disjoint.
// Everything else — different keys, unknown shapes, untrackable bases, calls,
// unordered region branches — keeps the store.
//
// Tile-level memory ops are gone by construction: this pass runs in the VMI
// semantic pipeline, after lowerPTOToVPTOBackend has expanded every tload/
// tstore, so UB is reachable only through !pto.ptr<_, ub> values.
//
// Runs after pto-vmi-load-store-elision (so it only sees stores the elision
// decided to keep) and before VMILowerUnifiedToLegacy (so vstores have no
// post-update results to preserve when erased).

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "VmiUBAccessModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMIFUNCTIONDSE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// "ub" is an assembly spelling of AddressSpace::VEC (PTOLowPrecisionTypes.cpp:
// parsePtrAddressSpaceKeyword maps both "vec" and "ub" to VEC).
static bool isUBPtrType(Type t) {
  if (auto ptr = dyn_cast<pto::PtrType>(t)) {
    return ptr.getMemorySpace().getAddressSpace() == pto::AddressSpace::VEC;
  }
  return false;
}

static std::optional<uint64_t> ubPtrElemBytes(Type t) {
  auto ptr = dyn_cast<pto::PtrType>(t);
  if (!ptr) {
    return std::nullopt;
  }
  unsigned bytes = pto::getPTOStorageElemByteSize(ptr.getElementType());
  if (bytes == 0) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(bytes);
}

// Peel a chain of pto.addptr ops with CONSTANT element offsets, returning the
// underlying base and the accumulated byte offset. resolveVmiStorageRoot does
// not trace addptr, yet the pipeline routinely materializes views as
// `addptr(castptr(c), 0)`; without this peel those bases look untrackable and
// every reader through them degrades to a global read.
static Value peelConstAddPtr(Value v, uint64_t elemBytes, int64_t &byteOffset) {
  int64_t elements = 0;
  while (v) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      break;
    }
    auto addptr = dyn_cast<pto::AddPtrOp>(def);
    if (!addptr) {
      break;
    }
    auto cst = addptr.getOffset().getDefiningOp<arith::ConstantOp>();
    if (!cst) {
      break; // dynamic offset: stop, caller treats the base as untrackable
    }
    auto val = dyn_cast<IntegerAttr>(cst.getValue());
    if (!val) {
      break;
    }
    if (__builtin_add_overflow(elements, val.getInt(), &elements)) {
      break;
    }
    v = addptr.getPtr();
  }
  if (__builtin_mul_overflow(elements, static_cast<int64_t>(elemBytes),
                             &byteOffset)) {
    return Value();
  }
  return v;
}

// Resolve a UB access base to a trackable identity, folding constant addptr
// offsets into the storage root's address so the root interval stays where
// the access really is. Returns an untrackable identity when the chain
// bottoms out anywhere unresolvable.
static BaseIdentity resolveAccessBase(Value base) {
  auto eb = ubPtrElemBytes(base.getType());
  if (!eb) {
    return resolveBaseIdentity(base);
  }
  int64_t byteOffset = 0;
  Value peeled = peelConstAddPtr(base, *eb, byteOffset);
  if (!peeled) {
    return {};
  }
  BaseIdentity id = resolveBaseIdentity(peeled);
  if (isTrackableIdentity(id) && id.storageRoot && byteOffset != 0) {
    pto::VmiStorageRoot shifted = *id.storageRoot;
    if (__builtin_add_overflow(shifted.address, byteOffset, &shifted.address)) {
      return {};
    }
    id.storageRoot = shifted;
  }
  return id;
}

// One potential read of UB memory. Exactly one of `global` / `rootWide` /
// bounded-interval (`offsetAddr` + `widthBytes`) describes the read set.
struct ReadEvent {
  Operation *op = nullptr;
  BaseIdentity base;        // untrackable => global
  Value offset;             // continuous vload offset (element units)
  AffineAddr offsetAddr;    // affine decomposition of `offset`
  uint64_t widthBytes = 0;  // bounded-interval width
  uint64_t elemBytes = 0;   // element size of the accessing view
  bool rootWide = false;    // reads somewhere within `base`'s storage root
  bool global = false;      // reads anywhere at all
};

// One candidate store. `rootWide` marks an unbounded write set (unknown mask
// or an unmodeled store shape); erasing it requires that nothing later may
// alias its whole storage root.
struct StoreCandidate {
  pto::VMIvStoreOp store;
  Operation *op = nullptr; // store.getOperation(), cached for const lookups
  BaseIdentity base;
  Value offset;
  AffineAddr offsetAddr;
  uint64_t widthBytes = 0;
  uint64_t elemBytes = 0;
  bool rootWide = false;
};

// Read event for one pto.vmi.vload: a bounded interval when the base is
// trackable and the shape/width are modeled, root-wide when only the base is
// trackable, global otherwise.
static void collectLoadReadEvent(pto::VMIvLoadOp load,
                                 SmallVectorImpl<ReadEvent> &events) {
  ReadEvent ev;
  ev.op = load;
  ev.base = resolveAccessBase(load.getSource());
  unsigned vl = getVRegWidth(load->getResult(0).getType());
  auto eb = ubPtrElemBytes(load.getSource().getType());
  if (isTrackableIdentity(ev.base) && isContinuousSingleVLoad(load) && vl &&
      eb) {
    ev.offset = load.getOffset();
    ev.offsetAddr = parseAffineAddr(load.getOffset());
    ev.widthBytes = static_cast<uint64_t>(vl) * *eb;
    ev.elemBytes = *eb;
  } else if (isTrackableIdentity(ev.base)) {
    // Untrackable shape or width: the read cannot be bounded to an
    // interval, but it stays inside its storage root.
    ev.rootWide = true;
  } else {
    ev.global = true;
  }
  events.push_back(ev);
}

// Read event for one pto.mte_ub_gm: the escape drain reads its whole source
// root (root-wide when trackable, global otherwise).
static void collectMteReadEvent(pto::MteUbGmOp mte,
                                SmallVectorImpl<ReadEvent> &events) {
  ReadEvent ev;
  ev.op = mte;
  ev.base = resolveAccessBase(mte.getSource());
  if (isTrackableIdentity(ev.base)) {
    ev.rootWide = true;
  } else {
    ev.global = true;
  }
  events.push_back(ev);
}

// Read events for any other op consuming a UB pointer operand: it may read
// through it unless it is known to be free of read effects. This admits the
// pure address arithmetic (pto.addptr) and the write-only ops (vstore,
// mte_gm_ub) and conservatively reads through the operand for everything
// else.
static void collectOperandReadEvents(Operation *op,
                                     SmallVectorImpl<ReadEvent> &events) {
  for (Value operand : op->getOperands()) {
    if (!isUBPtrType(operand.getType())) {
      continue;
    }
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!iface.hasEffect<MemoryEffects::Read>()) {
        continue;
      }
    } else if (isPure(op)) {
      continue;
    }
    ReadEvent ev;
    ev.op = op;
    ev.base = resolveAccessBase(operand);
    if (!isTrackableIdentity(ev.base)) {
      ev.global = true;
    } else {
      ev.rootWide = true;
    }
    events.push_back(ev);
  }
}

static void collectReadEvents(func::FuncOp func,
                              SmallVectorImpl<ReadEvent> &events) {
  func.walk([&](Operation *op) {
    if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      collectLoadReadEvent(load, events);
      return;
    }
    if (auto mte = dyn_cast<pto::MteUbGmOp>(op)) {
      collectMteReadEvent(mte, events);
      return;
    }
    if (isa<func::CallOp, func::CallIndirectOp>(op)) {
      ReadEvent ev;
      ev.op = op;
      ev.global = true;
      events.push_back(ev);
      return;
    }
    collectOperandReadEvents(op, events);
  });
}

static void collectStoreCandidates(func::FuncOp func,
                                   SmallVectorImpl<StoreCandidate> &cands) {
  func.walk([&](pto::VMIvStoreOp store) {
    // Defensive: a future post-update form would return the advanced base;
    // erasing it would have to materialize the pointer bump.
    if (!store->getResults().empty()) {
      return;
    }
    // Only UB destinations: GM is not function-private, so a GM store can be
    // observed by the host regardless of any in-function reader.
    if (!isUBPtrType(store.getDestination().getType())) {
      return;
    }
    StoreCandidate c;
    c.store = store;
    c.op = store.getOperation();
    c.base = resolveAccessBase(store.getDestination());
    if (!isTrackableIdentity(c.base)) {
      return;
    }
    if (!isContinuousSingleVStore(store)) {
      c.rootWide = true;
      cands.push_back(c);
      return;
    }
    Value mask = store.getMask().empty() ? Value() : store.getMask().front();
    LaneRange lanes = resolveMaskLanes(mask);
    unsigned vl = getVRegWidth(store.getValue().getType());
    auto eb = ubPtrElemBytes(store.getDestination().getType());
    if (!vl || !eb || lanes.isUnknownSet()) {
      c.rootWide = true;
    } else {
      uint64_t n = lanes.isFull() ? static_cast<uint64_t>(vl)
                                  : static_cast<uint64_t>(*lanes.upperBound);
      c.offset = store.getOffset();
      c.offsetAddr = parseAffineAddr(store.getOffset());
      c.widthBytes = n * *eb;
      c.elemBytes = *eb;
    }
    cands.push_back(c);
  });
}

// Whether `a` may execute after `b` in the function's sequential program
// order. Sibling subtrees of one block are ordered by block position and a
// subtree completes before a later sibling starts; branches of one region
// (scf.if) are unordered and reported as "after" (conservative).
static bool isPotentiallyAfter(Operation *a, Operation *b) {
  if (a == b) {
    return false;
  }
  SmallVector<Operation *, 8> ca, cb;
  for (Operation *cur = a; cur; cur = cur->getParentOp()) {
    ca.push_back(cur);
  }
  for (Operation *cur = b; cur; cur = cur->getParentOp()) {
    cb.push_back(cur);
  }
  std::reverse(ca.begin(), ca.end());
  std::reverse(cb.begin(), cb.end());
  size_t i = 0;
  while (i < ca.size() && i < cb.size() && ca[i] == cb[i]) {
    ++i;
  }
  if (i == 0) {
    return true; // different trees (not reachable within one function)
  }
  if (i == ca.size() || i == cb.size()) {
    return true; // one is an ancestor of the other: defensive
  }
  Operation *na = ca[i];
  Operation *nb = cb[i];
  if (na->getBlock() != nb->getBlock()) {
    return true; // unordered branches of one region: conservative
  }
  return !na->isBeforeInBlock(nb);
}

// Whether a read R positioned after a store S may touch S's written bytes.
static bool readAfterStoreMayOverlap(const StoreCandidate &s,
                                     const ReadEvent &r) {
  if (r.rootWide || s.rootWide) {
    return basesMayAlias(r.base, s.base);
  }
  if (!basesMayAlias(r.base, s.base)) {
    return false;
  }
  // Identical element offsets always overlap when the widths are positive
  // (conservative across differing element sizes: same element index under
  // different units may be disjoint, but claiming overlap only keeps stores).
  if (areEquivalentValues(s.offset, r.offset)) {
    return true;
  }
  // Both offsets constant: exact byte-interval check.
  if (s.offsetAddr.isAffine && r.offsetAddr.isAffine &&
      s.offsetAddr.iv == nullptr && r.offsetAddr.iv == nullptr) {
    int64_t sStart, rStart, sEnd, rEnd;
    if (__builtin_mul_overflow(s.offsetAddr.baseOffset,
                               static_cast<int64_t>(s.elemBytes), &sStart) ||
        __builtin_mul_overflow(r.offsetAddr.baseOffset,
                               static_cast<int64_t>(r.elemBytes), &rStart) ||
        __builtin_add_overflow(sStart, static_cast<int64_t>(s.widthBytes),
                               &sEnd) ||
        __builtin_add_overflow(rStart, static_cast<int64_t>(r.widthBytes),
                               &rEnd)) {
      return true;
    }
    return rStart < sEnd && sStart < rEnd;
  }
  return true; // affine vs constant / different keys: conservative
}

// Whether a read R positioned BEFORE a store S in the same scf.for body may
// still observe S: it also runs at every later iteration. Disproved only for
// identical affine keys driven by this loop's induction variable with a
// per-iteration stride of at least the store's write width.
static bool crossIterationMayObserve(const StoreCandidate &s,
                                     const ReadEvent &r, scf::ForOp forOp) {
  if (r.rootWide || s.rootWide) {
    return true;
  }
  if (!basesMustAlias(r.base, s.base)) {
    return basesMayAlias(r.base, s.base);
  }
  if (!r.offsetAddr.isAffine || !s.offsetAddr.isAffine ||
      r.offsetAddr.baseOffset != s.offsetAddr.baseOffset ||
      r.offsetAddr.coeff != s.offsetAddr.coeff || r.elemBytes != s.elemBytes) {
    return true;
  }
  // The driver must be this loop's induction variable (an outer-loop key is
  // constant across these iterations, i.e. stride 0). Both null covers equal
  // constant offsets: stride 0 always re-observes.
  if (r.offsetAddr.iv != s.offsetAddr.iv) {
    return true;
  }
  if (r.offsetAddr.iv && r.offsetAddr.iv != forOp.getInductionVar()) {
    return true;
  }
  int64_t magnitude = s.offsetAddr.coeff < 0 ? -s.offsetAddr.coeff
                                             : s.offsetAddr.coeff;
  int64_t stride;
  if (__builtin_mul_overflow(magnitude, static_cast<int64_t>(s.elemBytes),
                             &stride) ||
      stride < 0) {
    return true;
  }
  return static_cast<uint64_t>(stride) < s.widthBytes;
}

// Whether read `r` keeps candidate store `s` alive:
//   - R at/after S (or later in S's own block): a global read observes
//     anything; a bounded or root-wide read observes what it may touch;
//   - R before S in one block: only a repeating region can reorder them
//     across iterations — an scf.for re-observes unless the cross-iteration
//     proof applies, an scf.while is unmodeled (conservative), any other
//     parent cannot reorder;
//   - R definitively before S's whole subtree (different block): no.
static bool readKeepsStoreAlive(const StoreCandidate &s, const ReadEvent &r) {
  if (r.op == s.op) {
    return false;
  }
  if (r.op->getBlock() == s.store->getBlock()) {
    if (!r.op->isBeforeInBlock(s.store)) {
      return r.global || readAfterStoreMayOverlap(s, r);
    }
    Operation *parent = r.op->getBlock()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      return crossIterationMayObserve(s, r, forOp);
    }
    return isa<scf::WhileOp>(parent);
  }
  if (!isPotentiallyAfter(r.op, s.op)) {
    return false; // R definitively before S's whole subtree
  }
  return r.global || readAfterStoreMayOverlap(s, r);
}

struct PTOVmiFunctionDSEPass
    : public mlir::pto::impl::PTOVmiFunctionDSEBase<PTOVmiFunctionDSEPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    SmallVector<ReadEvent, 16> events;
    collectReadEvents(func, events);
    SmallVector<StoreCandidate, 8> cands;
    collectStoreCandidates(func, cands);

    SmallVector<pto::VMIvStoreOp, 4> toErase;
    for (const StoreCandidate &s : cands) {
      bool dead = true;
      for (const ReadEvent &r : events) {
        if (readKeepsStoreAlive(s, r)) {
          dead = false;
          break;
        }
      }
      if (dead) {
        toErase.push_back(s.store);
      }
    }
    for (pto::VMIvStoreOp store : toErase) {
      store->erase();
    }
    if (toErase.empty()) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiFunctionDSEPass() {
  return std::make_unique<PTOVmiFunctionDSEPass>();
}
