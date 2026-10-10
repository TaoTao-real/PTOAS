// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details.
// You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Heuristic unroll-factor provider for the VMI fusion pipeline.
//
// pto.fusion.row/col_unroll_factor has no producer on the VMI route in a
// default build: the VfSim planner (PTO_ENABLE_VFSIM_IR_PLANNER) plans VMI
// unroll at VPTO emission with pto.vfsim.unroll_factor instead, which is not
// what --enable-unroll-after-loop-fusion consumes.  This pass fills the gap
// conservatively: for every fusion_region that carries NEITHER factor
// attribute it derives one factor from the region's leaf loops and sets
// pto.fusion.row_unroll_factor (the streaming leaf of a fused VMI region is
// the row loop; the consumer prefers col only when both exist).
//
// Rules (deliberately conservative; the factor is a hint, the consumer still
// applies its own gates):
//   - A region that already carries either factor attribute is left alone:
//     an explicit decision (a hand annotation or a future planner) always
//     wins.
//   - Every leaf scf.for must have a compile-time-constant trip count, and
//     the candidate factor must divide it (the consumer skips indivisible
//     leaves; requiring divisibility of all leaves keeps the region's
//     behavior uniform and predictable).
//   - Resource guard: the unrolled body may not exceed kMaxUnrolledVLoads
//     vmi.vload results per leaf (live vregs are what unrolling multiplies).
//   - Candidates are tried largest first (kFactorCandidates = {4, 2}); a
//     region where none fits keeps no attribute and is not unrolled.
//
// Runs in the VMI semantic pipeline after load/store elision and function DSE
// (so dead stores are removed before their body is replicated) and right
// before pto-unroll-after-loop-fusion consumes the attribute.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVMIUNROLLHEURISTIC
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kRowUnrollFactorAttr =
    "pto.fusion.row_unroll_factor";
constexpr llvm::StringLiteral kColUnrollFactorAttr =
    "pto.fusion.col_unroll_factor";

// Candidate factors, tried largest first.  4 is the experimental candidate
// from the stage-4 optimization plan; 2 is the conservative fallback.
constexpr int64_t kFactorCandidates[] = {4, 2};
// Upper bound on vmi.vload results in one unrolled leaf body (live-vreg
// pressure proxy).  4 loads x factor 4 == 16 sits exactly at the bound.
constexpr int64_t kMaxUnrolledVLoads = 16;

static bool hasChildForOp(scf::ForOp forOp) {
  bool hasChild = false;
  forOp.getBody()->walk([&](scf::ForOp) {
    hasChild = true;
    return WalkResult::interrupt();
  });
  return hasChild;
}

static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
    return std::nullopt;
  }
  return (*ub - *lb + *step - 1) / *step;
}

static int64_t countVLoads(scf::ForOp forOp) {
  int64_t count = 0;
  forOp.getBody()->walk([&](pto::VMIvLoadOp) {
    ++count;
    return WalkResult::advance();
  });
  return count;
}

struct PTOVmiUnrollHeuristicPass
    : public mlir::pto::impl::PTOVmiUnrollHeuristicBase<
          PTOVmiUnrollHeuristicPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    bool changed = false;
    func.walk([&](pto::FusionRegionOp region) {
      // Fill only what is missing: an explicit factor (VfSim or any prior
      // decision, including a consumed marker of 1) always wins.
      if (region->getAttr(kRowUnrollFactorAttr) ||
          region->getAttr(kColUnrollFactorAttr)) {
        return WalkResult::advance();
      }

      SmallVector<scf::ForOp, 4> leaves;
      region.getBody().walk([&](scf::ForOp forOp) {
        if (!hasChildForOp(forOp)) {
          leaves.push_back(forOp);
        }
        return WalkResult::advance();
      });
      if (leaves.empty()) {
        return WalkResult::advance();
      }

      for (scf::ForOp leaf : leaves) {
        if (!getConstantTripCount(leaf)) {
          return WalkResult::advance(); // non-constant trip: leave unset
        }
      }

      for (int64_t factor : kFactorCandidates) {
        bool feasible = true;
        for (scf::ForOp leaf : leaves) {
          int64_t trip = *getConstantTripCount(leaf);
          if (trip % factor != 0 ||
              countVLoads(leaf) * factor > kMaxUnrolledVLoads) {
            feasible = false;
            break;
          }
        }
        if (feasible) {
          region->setAttr(kRowUnrollFactorAttr,
                          Builder(region->getContext())
                              .getI64IntegerAttr(factor));
          changed = true;
          break;
        }
      }
      return WalkResult::advance();
    });
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVmiUnrollHeuristicPass() {
  return std::make_unique<PTOVmiUnrollHeuristicPass>();
}
