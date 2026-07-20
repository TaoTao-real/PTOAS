// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root directory of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// PTOVmiLoopFusion.cpp - fuse same-header scf.for inside pto.fusion_region
//===----------------------------------------------------------------------===//
//
// VMI tile-library compute is always a single scf.for layer (the inner VL
// loop). This pass fuses adjacent same-header scf.for ops inside each
// pto.fusion_region into one fused scf.for. Two for's can be fused only if the
// ops sitting between them can be legally relocated after fusion:
//   - hoisted above the fused for (loop-invariant: no SSA/UB input produced by
//     any member's result or UB write),
//   - sunk below the fused for (its SSA results / UB writes not read inside
//     any member).
// Between-ops form dependency-connected components (SSA def-use, or same-UB
// store->load); each component must move as a whole. A component that can
// neither hoist nor sink — e.g. the tmuls(scale ColMax) chain, which reads
// the ColMax final UB (cannot hoist, the reduce is complete only after the
// loop) and whose store is read by the ColExpand-sub loop (cannot sink) —
// blocks fusion: the run stops there, so a reduce and the loop that consumes
// its final result stay separate for's.
//
// The fused scf.for's init args concatenate each member's init args (reduce
// carry); the fused body clones each member's body (without scf.yield) in
// source order; the fused scf.yield concatenates each member's yield operands
// mapped through the fused iter-args. Between-components hoisted above /
// sunk below the fused for are moved there (not cloned). The fused loop is
// built with a body-builder callback so the yield is created in place (no
// post-hoc setOperands on iter-arg/result linkage).
//
// The pass only touches scf.for ops directly nested inside a pto.fusion_region
// body. It does not perform mem2reg (UB roundtrip elimination) and does not
// build pto.vecscope.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMILOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Resolve a value to the comparison key used for header equivalence: trace the
// pointer_cast/castptr/arith.constant alias chain so that two references to the
// same compile-time UB address or the same constant are treated as identical.
static Value canonicalizeForCompare(Value v) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return v;
  Operation *def = v.getDefiningOp();
  if (!def)
    return v;

  if (auto pc = dyn_cast<pto::PointerCastOp>(def)) {
    if (!pc.getOperands().empty())
      return canonicalizeForCompare(pc.getOperands()[0]);
  }
  if (auto cp = dyn_cast<pto::CastPtrOp>(def))
    return canonicalizeForCompare(cp->getOperand(0));
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    if (auto lhs = dyn_cast<arith::ConstantOp>(
            add.getLhs().getDefiningOp()))
      return canonicalizeForCompare(add.getRhs());
    if (auto rhs = dyn_cast<arith::ConstantOp>(
            add.getRhs().getDefiningOp()))
      return canonicalizeForCompare(add.getLhs());
  }
  return v;
}

static bool sameHeader(scf::ForOp a, scf::ForOp b) {
  if (canonicalizeForCompare(a.getStep()) !=
      canonicalizeForCompare(b.getStep()))
    return false;
  if (canonicalizeForCompare(a.getLowerBound()) !=
      canonicalizeForCompare(b.getLowerBound()))
    return false;
  if (canonicalizeForCompare(a.getUpperBound()) !=
      canonicalizeForCompare(b.getUpperBound()))
    return false;
  if (a->getAttrs() != b->getAttrs())
    return false;
  return true;
}

// --- UB (tile buffer) identity: which compile-time address+type a vmi
// load/store accesses. Two ops touch the same UB iff they resolve to the same
// (addr-constant, memref-type) pair. Traced through pto.castptr -> memref ->
// pto.pointer_cast(addr-const). Returns std::nullopt if the base is not a
// compile-time constant address (then we conservatively cannot reason).
struct UBId {
  Value addrConst;      // the arith.constant feeding pointer_cast
  Type memrefType;      // the pointer_cast result type (shape+dtype)
  bool operator==(const UBId &o) const {
    return addrConst == o.addrConst && memrefType == o.memrefType;
  }
};

static std::optional<UBId> resolvePtrUB(Value base) {
  while (base) {
    if (auto cp = base.getDefiningOp<pto::CastPtrOp>()) {
      base = cp->getOperand(0);
      continue;
    }
    if (auto pc = base.getDefiningOp<pto::PointerCastOp>()) {
      if (!pc.getOperands().empty() &&
          isa<arith::ConstantOp>(pc.getOperands()[0].getDefiningOp()))
        return UBId{pc.getOperands()[0], pc.getResult().getType()};
    }
    break;
  }
  return std::nullopt;
}

static std::optional<UBId> getVLoadUB(pto::VMIvLoadOp op) {
  return resolvePtrUB(op.getSource());
}
static std::optional<UBId> getVStoreUB(pto::VMIvStoreOp op) {
  return resolvePtrUB(op.getDestination());
}

// Collect every UB a loop's body loads (reads) and stores (writes).
static void collectLoopUBs(scf::ForOp loop, SmallVectorImpl<UBId> &reads,
                           SmallVectorImpl<UBId> &writes) {
  loop.getBody()->walk([&](Operation *op) {
    if (auto v = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (auto id = getVLoadUB(v))
        reads.push_back(*id);
    } else if (auto v = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (auto id = getVStoreUB(v))
        writes.push_back(*id);
    }
    return WalkResult::advance();
  });
}

static bool ubListContains(ArrayRef<UBId> list, const UBId &x) {
  return llvm::any_of(list, [&](const UBId &u) { return u == x; });
}

// A member of a fusion run: the scf.for plus the ops sitting between the
// previous member's for and this one, split by where they can legally land
// after fusion:
//   hoisted -> move before the fused for (loop-invariant: inputs available
//              before the run; UB reads not produced by any member)
//   sunk    -> move after the fused for (outputs not read inside any member)
struct Member {
  scf::ForOp loop;
  SmallVector<Operation *, 8> hoisted; // before fused for
  SmallVector<Operation *, 8> sunk;   // after fused for
};

static SmallVector<scf::ForOp, 8> membersAsLoops(ArrayRef<Member> members) {
  SmallVector<scf::ForOp, 8> loops;
  for (const Member &m : members)
    loops.push_back(m.loop);
  return loops;
}

// Can `op` be hoisted above the fused for (run before any member executes)?
// Inputs must be available before the run: no SSA use of any member's result,
// and no UB read of an address that some member writes (that would read a
// loop-produced value).
static bool canHoistAboveRun(Operation *op, ArrayRef<scf::ForOp> runLoops,
                             ArrayRef<UBId> runWrites) {
  SmallPtrSet<Value, 16> loopResults;
  for (scf::ForOp l : runLoops)
    for (Value r : l.getResults())
      loopResults.insert(r);
  for (Value opnd : op->getOperands())
    if (loopResults.count(opnd))
      return false;
  if (auto v = dyn_cast<pto::VMIvLoadOp>(op))
    if (auto id = getVLoadUB(v))
      if (ubListContains(runWrites, *id))
        return false;
  return true;
}

// Can `op` be sunk below the fused for (run after all members execute)? Its
// UB writes must not be read inside any member (members run per iteration and
// would need the value). SSA outputs consumed inside members also block sink.
static bool canSinkBelowRun(Operation *op, ArrayRef<scf::ForOp> runLoops,
                            ArrayRef<UBId> runReads) {
  for (Value res : op->getResults())
    for (OpOperand &use : res.getUses())
      for (scf::ForOp l : runLoops)
        if (l->isAncestor(use.getOwner()))
          return false;
  if (auto v = dyn_cast<pto::VMIvStoreOp>(op))
    if (auto id = getVStoreUB(v))
      if (ubListContains(runReads, *id))
        return false;
  return true;
}

// Partition between-ops into dependency-connected components. Two between-ops
// are in the same component if data flows between them within the between
// region: either SSA (one's result is used by another), or UB (a vstore's
// written UB is read by a later vload). Each component must be placed AS A
// WHOLE after fusion (all hoisted above the fused for, or all sunk below it).
// Components are returned in source order; each component's ops are in source
// order.
static SmallVector<SmallVector<Operation *, 4>, 8>
partitionBetween(ArrayRef<Operation *> between) {
  unsigned n = between.size();
  SmallVector<unsigned, 8> parent(n);
  for (unsigned i = 0; i < n; ++i)
    parent[i] = i;
  auto find = [&](unsigned x) -> unsigned {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](unsigned a, unsigned b) {
    unsigned ra = find(a), rb = find(b);
    if (ra != rb)
      parent[ra] = rb;
  };
  DenseMap<Operation *, unsigned> idx;
  for (unsigned i = 0; i < n; ++i)
    idx[between[i]] = i;
  SmallVector<std::optional<UBId>, 8> writes(n);
  for (unsigned i = 0; i < n; ++i)
    if (auto v = dyn_cast<pto::VMIvStoreOp>(between[i]))
      writes[i] = getVStoreUB(v);
  for (unsigned j = 0; j < n; ++j) {
    Operation *opj = between[j];
    for (Value opnd : opj->getOperands()) {
      Operation *def = opnd.getDefiningOp();
      if (!def)
        continue;
      auto it = idx.find(def);
      if (it != idx.end() && it->second < j)
        unite(it->second, j);
    }
    if (auto v = dyn_cast<pto::VMIvLoadOp>(opj)) {
      if (auto id = getVLoadUB(v)) {
        for (unsigned i = 0; i < j; ++i)
          if (writes[i] && *writes[i] == *id)
            unite(i, j);
      }
    }
  }
  SmallVector<SmallVector<unsigned>, 8> byRoot(n);
  for (unsigned i = 0; i < n; ++i)
    byRoot[find(i)].push_back(i);
  SmallVector<unsigned, 8> roots;
  for (unsigned i = 0; i < n; ++i)
    if (find(i) == i)
      roots.push_back(i);
  llvm::sort(roots, [&](unsigned a, unsigned b) {
    return byRoot[a].front() < byRoot[b].front();
  });
  SmallVector<SmallVector<Operation *, 4>, 8> comps;
  for (unsigned r : roots) {
    SmallVector<Operation *, 4> comp;
    for (unsigned i : byRoot[r])
      comp.push_back(between[i]);
    comps.push_back(std::move(comp));
  }
  return comps;
}

// Split the region body's op list into members. A run grows by adding the
// next same-header for ONLY IF every op between the previous member's for and
// the candidate for can be legally placed after fusion — hoisted above the
// fused for, or sunk below it. Between-ops sit outside any for body
// originally, so they are not per-iteration and cannot be cloned into the
// fused body (that would change their execution count). If any between-op is
// stuck (can hoist neither above nor below — e.g. it reads a preceding
// reduce's final UB result AND its output is read inside a following member),
// the run stops: the stuck op and the following for start a separate run, so
// two for's separated by a stuck op are not fused into one iteration.
static SmallVector<Member, 8> collectRun(Block &body,
                                         SmallVectorImpl<scf::ForOp> &loops,
                                         unsigned firstLoopIdx) {
  SmallVector<Member, 8> members;
  scf::ForOp first = loops[firstLoopIdx];
  members.push_back(Member{first, {}, {}});

  Operation *betweenStart = first->getNextNode();
  for (unsigned i = firstLoopIdx + 1; i < loops.size(); ++i) {
    scf::ForOp cand = loops[i];
    if (!sameHeader(first, cand))
      break;

    // Between-ops: [betweenStart, cand).
    SmallVector<Operation *, 8> between;
    for (Operation *op = betweenStart; op && op != cand;
         op = op->getNextNode())
      between.push_back(op);

    // UB read/written by the full run if cand joins (members + cand).
    SmallVector<UBId, 8> runReads, runWrites;
    for (Member &m : members)
      collectLoopUBs(m.loop, runReads, runWrites);
    SmallVector<UBId, 8> candReads, candWrites;
    collectLoopUBs(cand, candReads, candWrites);
    runReads.append(candReads.begin(), candReads.end());
    runWrites.append(candWrites.begin(), candWrites.end());

    SmallVector<scf::ForOp, 8> fullLoops = membersAsLoops(members);
    fullLoops.push_back(cand);

    // Partition between-ops into dependency components (SSA def-use or same-UB
    // store->load). Each component must be placed AS A WHOLE: all hoisted above
    // the fused for, or all sunk below it (splitting a component would break its
    // internal dataflow). A component is stuck if it can neither hoist (some op
    // reads a member-produced UB / result) nor sink (some op's UB write / result
    // is used inside a member). If any component is stuck, the run stops here.
    SmallVector<SmallVector<Operation *, 4>, 8> comps =
        partitionBetween(between);
    bool stuck = false;
    for (const SmallVector<Operation *, 4> &comp : comps) {
      bool compHoist = true, compSink = true;
      for (Operation *op : comp) {
        if (!canHoistAboveRun(op, fullLoops, runWrites))
          compHoist = false;
        if (!canSinkBelowRun(op, fullLoops, runReads))
          compSink = false;
      }
      if (!compHoist && !compSink) {
        stuck = true;
        break;
      }
    }
    if (stuck)
      break;

    // Commit cand. Each component goes to the bucket it can: hoist if
    // compHoist, else sink (compSink must hold here).
    Member &last = members.back();
    for (const SmallVector<Operation *, 4> &comp : comps) {
      bool compHoist = true;
      for (Operation *op : comp)
        if (!canHoistAboveRun(op, fullLoops, runWrites)) {
          compHoist = false;
          break;
        }
      if (compHoist) {
        for (Operation *op : comp)
          last.hoisted.push_back(op);
      } else {
        for (Operation *op : comp)
          last.sunk.push_back(op);
      }
    }
    members.push_back(Member{cand, {}, {}});
    betweenStart = cand->getNextNode();
  }
  return members;
}

// Build the fused scf.for for a run of members. Members are erased by caller.
static scf::ForOp buildFusedLoop(OpBuilder &builder,
                                 MutableArrayRef<Member> members) {
  scf::ForOp firstLoop = members.front().loop;
  Location loc = firstLoop.getLoc();

  // Fused init args = concatenation of each member's init args.
  SmallVector<Value, 8> fusedInitArgs;
  for (Member &m : members)
    fusedInitArgs.append(m.loop.getInitArgs().begin(),
                         m.loop.getInitArgs().end());

  SmallVector<IRMapping, 8> mappings(members.size());

  auto bodyBuilder = [&](OpBuilder &b, Location bl, Value iv,
                         ValueRange iterArgs) {
    unsigned iterOffset = 0;
    for (auto [idx, m] : llvm::enumerate(members)) {
      mappings[idx].map(m.loop.getInductionVar(), iv);
      unsigned nArgs = m.loop.getRegionIterArgs().size();
      for (unsigned k = 0; k < nArgs; ++k)
        mappings[idx].map(m.loop.getRegionIterArgs()[k],
                          iterArgs[iterOffset + k]);
      iterOffset += nArgs;
    }

    // Per member: clone only its body (without scf.yield) in source order.
    // Between-ops that were loop-invariant (hoisted bucket) are moved before
    // the fused for below; those whose output no member reads (sunk bucket)
    // are moved after it. Body uses of hoisted values resolve to the top-level
    // originals via lookupOrDefault.
    for (auto [idx, m] : llvm::enumerate(members)) {
      Block &mbody = *m.loop.getBody();
      for (Operation &op : mbody.without_terminator())
        b.clone(op, mappings[idx]);
    }

    // Fused yield = concatenation of each member's yield operands, mapped.
    SmallVector<Value, 8> fusedYield;
    for (auto [idx, m] : llvm::enumerate(members)) {
      auto y = cast<scf::YieldOp>(m.loop.getBody()->getTerminator());
      for (Value v : y.getOperands())
        fusedYield.push_back(mappings[idx].lookupOrDefault(v));
    }
    b.create<scf::YieldOp>(bl, fusedYield);
  };

  auto fused = builder.create<scf::ForOp>(
      loc, firstLoop.getLowerBound(), firstLoop.getUpperBound(),
      firstLoop.getStep(), fusedInitArgs, bodyBuilder);
  fused->setAttrs(firstLoop->getAttrs());

  // Map each member's results to the corresponding slice of the fused loop's
  // results so external (top-level) users can be rewired.
  unsigned resOffset = 0;
  for (auto [idx, m] : llvm::enumerate(members)) {
    for (Value r : m.loop.getResults())
      mappings[idx].map(r, fused.getResults()[resOffset++]);
  }

  // Rewire external uses of each member's results to the fused results.
  resOffset = 0;
  for (auto [idx, m] : llvm::enumerate(members)) {
    for (auto [res, fusedRes] :
         llvm::zip(m.loop.getResults(),
                   fused.getResults().slice(
                       resOffset, m.loop.getNumResults()))) {
      res.replaceAllUsesWith(fusedRes);
    }
    resOffset += m.loop.getNumResults();
  }

  // Place between-ops and init-arg producers:
  //  - hoisted bucket (loop-invariant) and init-arg producers -> move before
  //    the fused for so they dominate the body / init args.
  //  - sunk bucket (outputs not read by any member) -> move after the fused
  //    for.
  // These ops are NOT cloned, so each UB materialization stays materialized
  // once. A later CSE dedups remaining duplicates.
  SmallVector<Operation *, 16> hoistOrder, sinkOrder;
  SmallPtrSet<Operation *, 32> seen;
  auto gather = [&](Operation *op) {
    if (!op || op == fused || op->getParentOp() != fused->getParentOp())
      return;
    if (seen.insert(op).second)
      hoistOrder.push_back(op);
  };
  auto gatherSink = [&](Operation *op) {
    if (!op || op == fused || op->getParentOp() != fused->getParentOp())
      return;
    if (seen.insert(op).second)
      sinkOrder.push_back(op);
  };
  for (Member &m : members) {
    for (Operation *pre : m.hoisted)
      gather(pre);
    for (Operation *sop : m.sunk)
      gatherSink(sop);
    for (Value ia : m.loop.getInitArgs())
      if (Operation *def = ia.getDefiningOp())
        gather(def);
  }
  for (Operation *op : hoistOrder)
    if (!op->isBeforeInBlock(fused))
      op->moveBefore(fused);
  for (Operation *op : sinkOrder)
    if (op->isBeforeInBlock(fused))
      op->moveAfter(fused);

  // Erase the member for ops (between-ops are kept, only the for ops go away).
  for (Member &m : llvm::reverse(members))
    m.loop.erase();

  return fused;
}

// Fuse one maximal run of same-header scf.for starting at firstLoopIdx.
// collectRun stops the run at a between-op that can neither hoist above nor
// sink below the run (a reduce-final UB dependency), so the fused run only
// spans for's whose between-ops are all placeable. Returns true if a fusion
// happened (>=2 members).
static bool fuseRun(Block &body, SmallVectorImpl<scf::ForOp> &loops,
                    unsigned firstLoopIdx) {
  SmallVector<Member, 8> members = collectRun(body, loops, firstLoopIdx);
  if (members.size() < 2)
    return false;
  OpBuilder builder(members.front().loop);
  buildFusedLoop(builder, members);
  return true;
}

struct PTOVmiLoopFusionPass
    : public mlir::pto::impl::PTOVmiLoopFusionBase<PTOVmiLoopFusionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](pto::FusionRegionOp region) {
      bool progressed = true;
      while (progressed) {
        progressed = false;
        Block &body = region.getBody().front();
        SmallVector<scf::ForOp, 16> loops;
        for (Operation &op : body.getOperations())
          if (auto f = dyn_cast<scf::ForOp>(op))
            loops.push_back(f);

        for (unsigned i = 0; i < loops.size();) {
          if (fuseRun(body, loops, i)) {
            progressed = true;
            break; // re-collect after mutation
          }
          ++i;
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiLoopFusionPass() {
  return std::make_unique<PTOVmiLoopFusionPass>();
}
