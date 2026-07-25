// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIAnalyzeFusionReady.cpp - Canonical VMI fusion readiness --------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIANALYZEFUSIONREADY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kPrincipalAttr = "pto.vmi.tilelib.principal";
constexpr llvm::StringLiteral kFusionCandidateAttr =
    "pto.vmi.fusion_candidate";

enum class FusionReadyStatus {
  StructuralCandidate,
  LayoutPhase,
  BoundaryOnly,
  Unsupported,
};

struct LoopAnalysis {
  FusionReadyStatus status = FusionReadyStatus::StructuralCandidate;
  StringRef reason = "canonical_logical_row";
  int64_t rows = 0;
  int64_t logicalLanes = 0;
};

static StringRef stringifyStatus(FusionReadyStatus status) {
  switch (status) {
  case FusionReadyStatus::StructuralCandidate:
    return "structural_fusion_candidate";
  case FusionReadyStatus::LayoutPhase:
    return "layout_phase";
  case FusionReadyStatus::BoundaryOnly:
    return "boundary_only";
  case FusionReadyStatus::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown VMI fusion-ready status");
}

static bool containsAddressType(Operation *op) {
  auto isAddress = [](Type type) {
    return isa<PtrType, MemRefType, UnrankedMemRefType>(type);
  };
  return llvm::any_of(op->getOperandTypes(), isAddress) ||
         llvm::any_of(op->getResultTypes(), isAddress);
}

static bool isIndirectMemoryOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.vmi.gather" || name == "pto.vmi.vgather" ||
         name == "pto.vmi.vgatherb" || name == "pto.vmi.scatter" ||
         name == "pto.vmi.vscatter";
}

static bool isExplicitFusionBoundary(Operation *op) {
  if (isa<func::CallOp>(op))
    return true;
  if (isa<VecScopeOp, StrictVecScopeOp>(op))
    return true;

  if (auto pipeOp = dyn_cast<OpPipeInterface>(op))
    if (pipeOp.getPipe() != PIPE::PIPE_V)
      return true;

  StringRef name = op->getName().getStringRef();
  return name == "pto.barrier" || name == "pto.barrier_sync" ||
         name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.record_event" || name == "pto.wait_event" ||
         name == "pto.tpush" || name == "pto.tpop" || name == "pto.tfree";
}

static std::optional<StringRef> checkLoopDomain(scf::ForOp loop,
                                                int64_t &rows) {
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step)
    return "dynamic_iteration_domain";
  if (*lower != 0 || *step != 1 || *upper <= *lower)
    return "noncanonical_row_domain";
  rows = *upper - *lower;
  return std::nullopt;
}

static std::optional<StringRef>
collectLogicalTypeFacts(scf::ForOp loop, int64_t &logicalLanes) {
  llvm::SmallSet<int64_t, 4> nonScalarLaneCounts;
  bool sawVMIType = false;
  bool missingMemoryEffects = false;

  loop.walk([&](Operation *op) {
    auto inspectType = [&](Type type) {
      if (auto vreg = dyn_cast<VMIVRegType>(type)) {
        sawVMIType = true;
        int64_t lanes = vreg.getElementCount();
        if (lanes > 1)
          nonScalarLaneCounts.insert(lanes);
        return;
      }
      if (auto mask = dyn_cast<VMIMaskType>(type)) {
        sawVMIType = true;
        if (mask.getElementCount() > 1)
          nonScalarLaneCounts.insert(mask.getElementCount());
      }
    };

    for (Type type : op->getOperandTypes())
      inspectType(type);
    for (Type type : op->getResultTypes())
      inspectType(type);

    StringRef name = op->getName().getStringRef();
    if (name.starts_with("pto.vmi.") && containsAddressType(op)) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects) {
        missingMemoryEffects = true;
        return;
      }
      SmallVector<MemoryEffects::EffectInstance, 2> instances;
      effects.getEffects(instances);
      if (instances.empty())
        missingMemoryEffects = true;
    }
  });

  if (!sawVMIType)
    return "no_vmi_dataflow";
  if (missingMemoryEffects)
    return "incomplete_memory_effects";
  if (nonScalarLaneCounts.size() > 1)
    return "inconsistent_logical_lanes";
  logicalLanes = nonScalarLaneCounts.empty() ? 1 : *nonScalarLaneCounts.begin();
  return std::nullopt;
}

static LoopAnalysis analyzeLoop(scf::ForOp loop) {
  LoopAnalysis result;

  bool hasNestedLoop = false;
  bool hasVecScope = false;
  bool hasBoundary = false;
  bool hasIndirectMemory = false;
  loop.walk([&](Operation *op) {
    if (op != loop.getOperation() && isa<scf::ForOp>(op))
      hasNestedLoop = true;
    if (isa<VecScopeOp, StrictVecScopeOp>(op))
      hasVecScope = true;
    if (op != loop.getOperation() && isExplicitFusionBoundary(op))
      hasBoundary = true;
    if (isIndirectMemoryOp(op))
      hasIndirectMemory = true;
  });

  if (hasNestedLoop) {
    result.status = FusionReadyStatus::Unsupported;
    result.reason = "nested_loop";
    return result;
  }
  if (hasVecScope) {
    result.status = FusionReadyStatus::Unsupported;
    result.reason = "prephysicalization_vecscope";
    return result;
  }
  if (auto reason = checkLoopDomain(loop, result.rows)) {
    result.status = FusionReadyStatus::Unsupported;
    result.reason = *reason;
    return result;
  }
  if (auto reason = collectLogicalTypeFacts(loop, result.logicalLanes)) {
    result.status = FusionReadyStatus::Unsupported;
    result.reason = *reason;
    return result;
  }
  if (hasBoundary) {
    result.status = FusionReadyStatus::BoundaryOnly;
    result.reason = "non_vector_or_unknown_call_boundary";
    return result;
  }
  if (hasIndirectMemory) {
    result.status = FusionReadyStatus::LayoutPhase;
    result.reason = "indirect_memory_access";
    return result;
  }
  return result;
}

static void attachAnalysis(scf::ForOp loop, const LoopAnalysis &analysis) {
  Builder builder(loop.getContext());
  SmallVector<NamedAttribute, 8> fields;
  fields.push_back(builder.getNamedAttr(
      "status", builder.getStringAttr(stringifyStatus(analysis.status))));
  fields.push_back(
      builder.getNamedAttr("reason", builder.getStringAttr(analysis.reason)));
  if (analysis.rows > 0) {
    fields.push_back(builder.getNamedAttr(
        "iteration_domain", builder.getStringAttr("rows")));
    fields.push_back(
        builder.getNamedAttr("rows", builder.getI64IntegerAttr(analysis.rows)));
  }
  if (analysis.logicalLanes > 0)
    fields.push_back(builder.getNamedAttr(
        "logical_lanes", builder.getI64IntegerAttr(analysis.logicalLanes)));
  loop->setAttr(kFusionCandidateAttr,
                DictionaryAttr::get(loop.getContext(), fields));
}

struct VMIAnalyzeFusionReadyPass
    : pto::impl::VMIAnalyzeFusionReadyBase<VMIAnalyzeFusionReadyPass> {
  void runOnOperation() override {
    SmallVector<scf::ForOp, 16> principalLoops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kPrincipalAttr))
        principalLoops.push_back(loop);
    });

    for (scf::ForOp loop : principalLoops)
      attachAnalysis(loop, analyzeLoop(loop));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIAnalyzeFusionReadyPass() {
  return std::make_unique<VMIAnalyzeFusionReadyPass>();
}
