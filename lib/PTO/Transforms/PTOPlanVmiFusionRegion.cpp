// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root directory of the software repository for the full text of the License.

//===----------------------------------------------------------------------===//
// PTOPlanVmiFusionRegion.cpp - merge adjacent pto.fusion_region into groups
//===----------------------------------------------------------------------===//
//
// `pto.fusion_region` is wrapped around each expanded compute TileOp body by
// OP-Lib inlining (one region per TileOp). This pass merges adjacent regions
// into VMI fusion groups following the VF Fusion rules:
//   - F1 (same region): fusible VMI ops may share one fusion_region.
//   - F2 (ColMax data dependency): a ColReduce scf.for result is complete only
//     after its whole region; the reduce loop and a consumer element-wise loop
//     stay as separate for-row but may share one fusion_region.
//   - F3 (sync break): mte_*/set_flag/wait_flag/mem_bar/pipe_barrier/vecscope/
//     unknown-effect ops close the current group.
//   - rule 1 (UB overlap): a candidate region's UB set (alloc_tile values
//     referenced via tile_buf_addr) must be either identical to or disjoint
//     from every UB already in the group; partial overlap rejects.
//
// Merging splices the candidate region's body (minus its pto.yield) into the
// tail of the current group region's body (before its pto.yield) and erases
// the candidate region. fa's compute regions carry an empty pto.yield (DPS
// in-place UB, nothing escapes), so the merged group stays empty-yield;
// non-empty yields are conservatively left un-merged (kept separate) until a
// later change promotes escaping values to region results.
//
// The pass runs in the VMI semantic pipeline, before VMILowerUnifiedToLegacy
// (so ops are still vmi.vload/vstore/...) and before PTOFlattenFusionRegion.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPLANVMIFUSIONREGION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// A UB identity after FoldTileBufIntrinsics is a `pto.pointer_cast` of a
// compile-time address into a memref<shape x dtype, vec>. Two references are
// the *same* UB iff they share the address constant value AND the memref type
// (shape + dtype); distinct addresses are disjoint at alloc level.
struct UBIdentity {
  int64_t address = 0;
  Type memrefType;

  bool operator==(const UBIdentity &rhs) const {
    return address == rhs.address && memrefType == rhs.memrefType;
  }
};

// Resolve a `pointer_cast` address operand to a concrete int64 if it is a
// compile-time constant; return std::nullopt for unknown provenance (the
// planner then conservatively rejects merging that region).
static std::optional<int64_t> resolveConstAddress(Value addr) {
  auto constOp = addr.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  if (auto iv = dyn_cast<IntegerAttr>(constOp.getValue()))
    return iv.getInt();
  return std::nullopt;
}

// Ops whose presence in a region body closes the current fusion group (F3).
static bool isFusionBoundaryOpName(StringRef name) {
  return name == "pto.mte_gm_ub" || name == "pto.mte_ub_gm" ||
         name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.mem_bar" || name == "pto.pipe_barrier" ||
         name == "pto.vecscope" || name == "pto.strict_vecscope";
}

// A region is a sync/boundary group (F3) if its body contains any DMA/sync op
// or any op the planner cannot reason about. VMI compute ops (vload/vmax/...
// scf.for/scf.yield/arith/castptr/pointer_cast) are fine: scf.for has a region
// but is a legal compute carrier, so it is not a barrier.
static bool regionIsFusionBoundary(FusionRegionOp region) {
  bool boundary = false;
  region.getBody().walk([&](Operation *op) {
    if (boundary)
      return WalkResult::interrupt();
    // scf.for/scf.if/scf.yield are legal compute carriers, and pto.yield is the
    // region's own terminator (not a sync). Skip them before the generic
    // terminator check below, which would otherwise flag pto.yield as a
    // boundary.
    if (isa<scf::ForOp, scf::IfOp, scf::YieldOp, pto::YieldOp>(op))
      return WalkResult::advance();
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      boundary = true;
      return WalkResult::interrupt();
    }
    if (isa<CallOpInterface>(op) || !op->getRegions().empty()) {
      boundary = true;
      return WalkResult::interrupt();
    }
    if (isFusionBoundaryOpName(op->getName().getStringRef())) {
      boundary = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return boundary;
}

// Collect every UB identity referenced inside a region body, via
// `pto.pointer_cast %addr : memref<shape x dtype, vec>`.
static void collectRegionUBs(FusionRegionOp region,
                             SmallVectorImpl<UBIdentity> &out) {
  region.getBody().walk([&](pto::PointerCastOp pcOp) {
    if (pcOp.getOperands().empty())
      return WalkResult::advance();
    auto addr = resolveConstAddress(pcOp.getOperands()[0]);
    if (addr) {
      UBIdentity ub{*addr, pcOp.getResult().getType()};
      if (!llvm::is_contained(out, ub))
        out.push_back(ub);
    }
    return WalkResult::advance();
  });
}

// Rule 1: candidate UBs must each be identical-to or disjoint-from every UB
// already in the group.
static bool canMergeWithGroup(const SmallVectorImpl<UBIdentity> &candidateUBs,
                              const SmallVectorImpl<UBIdentity> &groupUBs) {
  for (const UBIdentity &ub : candidateUBs) {
    for (const UBIdentity &gub : groupUBs) {
      if (ub.address == gub.address) {
        if (ub.memrefType != gub.memrefType)
          return false;
        continue;
      }
    }
  }
  return true;
}

// A region is mergeable into a group only if its yield is empty (nothing
// escapes). Non-empty yields would need region-result extension, which is not
// supported yet; keep them as separate groups.
static bool hasEmptyYield(FusionRegionOp region) {
  auto yieldOp = dyn_cast<pto::YieldOp>(
      region.getBody().front().getTerminator());
  return yieldOp && yieldOp.getValues().empty();
}

// Splice the candidate region's body (minus its pto.yield) into the tail of
// the group region's body, just before the group's pto.yield, then erase the
// candidate region. Both regions must have an empty yield here.
static void spliceRegionIntoGroup(FusionRegionOp group,
                                  FusionRegionOp candidate) {
  Block &dst = group.getBody().front();
  Block &src = candidate.getBody().front();

  Operation *dstYield = dst.getTerminator();
  assert(dstYield && "group region must terminate with pto.yield");

  // Move every src op except its terminator (pto.yield) to just before the
  // group's yield. Both yields are empty so no operand/result rewiring is
  // needed; SSA values defined inside a region implicitly escape to sibling
  // ops in the same parent block once spliced into a shared block.
  for (Operation &op :
       llvm::make_early_inc_range(src.getOperations())) {
    if (&op == src.getTerminator())
      continue;
    op.moveBefore(dstYield);
  }

  candidate.erase();
}

struct PTOPlanVmiFusionRegionPass
    : public mlir::pto::impl::PTOPlanVmiFusionRegionBase<
          PTOPlanVmiFusionRegionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([](func::FuncOp func) { planInFunc(func); });
  }

  static void planInFunc(func::FuncOp func) {
    if (func.getBody().empty())
      return;
    Block &entry = func.getBody().front();

    FusionRegionOp currentGroup = nullptr;
    SmallVector<UBIdentity> groupUBs;

    SmallVector<FusionRegionOp, 16> regions;
    for (Operation &op : entry.getOperations())
      if (auto fr = dyn_cast<FusionRegionOp>(op))
        regions.push_back(fr);

    for (FusionRegionOp region : regions) {
      // F3: a region containing a DMA/sync/unknown op is its own group and
      // closes any open group.
      if (regionIsFusionBoundary(region)) {
        currentGroup = nullptr;
        continue;
      }

      SmallVector<UBIdentity> candidateUBs;
      collectRegionUBs(region, candidateUBs);

      // Only merge empty-yield regions; non-empty yields stay separate.
      const bool mergeable = hasEmptyYield(region);

      if (currentGroup && mergeable &&
          canMergeWithGroup(candidateUBs, groupUBs)) {
        spliceRegionIntoGroup(currentGroup, region);
        for (const UBIdentity &ub : candidateUBs)
          if (!llvm::is_contained(groupUBs, ub))
            groupUBs.push_back(ub);
      } else {
        currentGroup = mergeable ? region : nullptr;
        groupUBs.clear();
        for (const UBIdentity &ub : candidateUBs)
          groupUBs.push_back(ub);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPlanVmiFusionRegionPass() {
  return std::make_unique<PTOPlanVmiFusionRegionPass>();
}
