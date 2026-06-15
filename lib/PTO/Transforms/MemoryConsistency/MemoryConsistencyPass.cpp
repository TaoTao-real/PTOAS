// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
// LICENSE in the root of the software repository for the full text of the
// License.

//===- MemoryConsistencyPass.cpp -----------------------------------------===//
// Annotate PTO memory consistency actions before lowering.
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/MemoryConsistency/MemoryAccessDesc.h"
#include "PTO/Transforms/MemoryConsistency/MemoryConsistencyAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMEMORYCONSISTENCY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct TNotifyReleaseState {
  bool drainMte2 = false;
  bool drainMte3 = false;
  bool drainFix = false;
  bool needsDsbDdr = false;
  bool needsCleanGmCache = false;

  void merge(const TNotifyReleaseState &other) {
    drainMte2 |= other.drainMte2;
    drainMte3 |= other.drainMte3;
    drainFix |= other.drainFix;
    needsDsbDdr |= other.needsDsbDdr;
    needsCleanGmCache |= other.needsCleanGmCache;
  }

  void applyBarrier(pto::PIPE pipe) {
    switch (pipe) {
    case pto::PIPE::PIPE_MTE2:
      drainMte2 = false;
      break;
    case pto::PIPE::PIPE_MTE3:
      drainMte3 = false;
      break;
    case pto::PIPE::PIPE_FIX:
      drainFix = false;
      break;
    case pto::PIPE::PIPE_ALL:
      drainMte2 = false;
      drainMte3 = false;
      drainFix = false;
      break;
    default:
      break;
    }
  }
};

struct SignalAcquireState {
  bool needsInvalidateGmCache = false;

  void merge(const SignalAcquireState &other) {
    needsInvalidateGmCache |= other.needsInvalidateGmCache;
  }
};

static SignalAcquireState makeAcquireInvalidateState() {
  SignalAcquireState state;
  state.needsInvalidateGmCache = true;
  return state;
}

static void markPipeDrainForDesc(const pto::MemoryAccessDesc &desc,
                                 TNotifyReleaseState &state,
                                 bool requireDsbDdr) {
  if (!desc.pipe)
    return;

  switch (*desc.pipe) {
  case pto::PIPE::PIPE_MTE2:
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B:
    state.drainMte2 = true;
    break;
  case pto::PIPE::PIPE_MTE3:
  case pto::PIPE::PIPE_MTE4:
  case pto::PIPE::PIPE_MTE5:
    state.drainMte3 = true;
    break;
  case pto::PIPE::PIPE_FIX:
    state.drainFix = true;
    break;
  case pto::PIPE::PIPE_ALL:
    state.drainMte2 = true;
    state.drainMte3 = true;
    state.drainFix = true;
    break;
  default:
    break;
  }

  if (requireDsbDdr)
    state.needsDsbDdr = true;
}

static TNotifyReleaseState getFallbackReleaseStateForPipe(pto::PIPE pipe) {
  TNotifyReleaseState state;
  switch (pipe) {
  case pto::PIPE::PIPE_MTE2:
    state.drainMte2 = true;
    break;
  case pto::PIPE::PIPE_MTE3:
    state.drainMte3 = true;
    break;
  case pto::PIPE::PIPE_FIX:
    state.drainFix = true;
    state.needsDsbDdr = true;
    break;
  case pto::PIPE::PIPE_ALL:
    state.drainMte2 = true;
    state.drainMte3 = true;
    state.drainFix = true;
    state.needsDsbDdr = true;
    break;
  default:
    break;
  }
  return state;
}

static TNotifyReleaseState getDirectReleaseState(Operation *op) {
  TNotifyReleaseState state;
  if (isa<pto::BarrierOp>(op))
    return state;

  SmallVector<pto::MemoryAccessDesc, 4> descs =
      pto::collectMemoryAccessDescs(op);
  for (const pto::MemoryAccessDesc &desc : descs) {
    if (pto::needsPipeDrainBeforePublish(desc)) {
      markPipeDrainForDesc(desc, state, /*requireDsbDdr=*/true);
      continue;
    }

    if (pto::isPayloadAccess(desc) &&
        desc.kind == pto::MemoryConsistencyAccessKind::Write &&
        desc.cachePolicy == pto::MemoryConsistencyCachePolicy::NonCache) {
      markPipeDrainForDesc(desc, state, /*requireDsbDdr=*/true);
      state.needsDsbDdr = true;
      continue;
    }

    if (pto::mayNeedCleanBeforeNonCacheConsumer(desc)) {
      state.needsCleanGmCache = true;
      state.needsDsbDdr = true;
      continue;
    }

    if (pto::isPayloadAccess(desc) &&
        desc.kind == pto::MemoryConsistencyAccessKind::Read &&
        desc.cachePolicy == pto::MemoryConsistencyCachePolicy::NonCache)
      markPipeDrainForDesc(desc, state, /*requireDsbDdr=*/false);
  }

  if (!descs.empty())
    return state;

  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op))
    return getFallbackReleaseStateForPipe(pipeOp.getPipe());
  return state;
}

static TNotifyReleaseState collectReleaseState(Operation *op) {
  TNotifyReleaseState state = getDirectReleaseState(op);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectReleaseState(&nested));
  return state;
}

static bool isLoopLikeOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp>(op);
}

static bool needsAcquireInvalidateBefore(Operation *op) {
  if (!isa<pto::LoadScalarOp>(op))
    return false;

  for (const pto::MemoryAccessDesc &desc : pto::collectMemoryAccessDescs(op)) {
    if (pto::mayNeedInvalidateBeforeCacheableConsumer(desc))
      return true;
  }
  return false;
}

static std::optional<bool> getBoolConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;

  Attribute attr = constant.getValue();
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto intTy = dyn_cast<IntegerType>(value.getType());
        intTy && intTy.getWidth() == 1)
      return !intAttr.getValue().isZero();
  }
  return std::nullopt;
}

static bool isTTestResult(Value value) {
  return value && value.getDefiningOp<pto::TTestOp>();
}

// Returns whether acquire is valid when `condition` evaluates to true. For a
// direct TTest result, true means signal-ready; for a simple inversion, false
// means signal-ready.
static std::optional<bool> getAcquireWhenConditionTrue(Value condition) {
  if (isTTestResult(condition))
    return true;

  if (auto xorOp = condition.getDefiningOp<arith::XOrIOp>()) {
    auto lhsPolarity = getAcquireWhenConditionTrue(xorOp.getLhs());
    auto rhsConst = getBoolConstant(xorOp.getRhs());
    if (lhsPolarity && rhsConst)
      return *rhsConst ? !*lhsPolarity : *lhsPolarity;

    auto rhsPolarity = getAcquireWhenConditionTrue(xorOp.getRhs());
    auto lhsConst = getBoolConstant(xorOp.getLhs());
    if (rhsPolarity && lhsConst)
      return *lhsConst ? !*rhsPolarity : *rhsPolarity;
  }

  if (auto cmpOp = condition.getDefiningOp<arith::CmpIOp>()) {
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (pred != arith::CmpIPredicate::eq &&
        pred != arith::CmpIPredicate::ne)
      return std::nullopt;

    auto lhsPolarity = getAcquireWhenConditionTrue(cmpOp.getLhs());
    auto rhsConst = getBoolConstant(cmpOp.getRhs());
    if (lhsPolarity && rhsConst) {
      bool valueWhenConditionTrue =
          pred == arith::CmpIPredicate::eq ? *rhsConst : !*rhsConst;
      return valueWhenConditionTrue ? *lhsPolarity : !*lhsPolarity;
    }

    auto rhsPolarity = getAcquireWhenConditionTrue(cmpOp.getRhs());
    auto lhsConst = getBoolConstant(cmpOp.getLhs());
    if (rhsPolarity && lhsConst) {
      bool valueWhenConditionTrue =
          pred == arith::CmpIPredicate::eq ? *lhsConst : !*lhsConst;
      return valueWhenConditionTrue ? *rhsPolarity : !*rhsPolarity;
    }
  }

  return std::nullopt;
}

static bool valueFeedsRecognizedAcquireCondition(Value value,
                                                unsigned depth = 0) {
  if (!value || depth > 1)
    return false;

  for (Operation *user : value.getUsers()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
      if (ifOp.getCondition() == value)
        return true;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      if (conditionOp.getCondition() == value)
        return true;
    }
    if (isa<arith::XOrIOp, arith::CmpIOp>(user) &&
        valueFeedsRecognizedAcquireCondition(user->getResult(0), depth + 1))
      return true;
  }
  return false;
}

static bool hasRecognizedControlUse(pto::TTestOp op) {
  return valueFeedsRecognizedAcquireCondition(op.getResult());
}

static SignalAcquireState getDirectAcquireState(Operation *op) {
  SignalAcquireState state;
  // TWAIT is a blocking acquire of the signal. The signal cache maintenance is
  // owned by the comm primitive itself; PTOAS only needs to invalidate stale
  // payload cache lines before the following cacheable GM read.
  if (isa<pto::TWaitOp>(op))
    state.needsInvalidateGmCache = true;
  if (auto test = dyn_cast<pto::TTestOp>(op);
      test && !hasRecognizedControlUse(test))
    state.needsInvalidateGmCache = true;
  return state;
}

static SignalAcquireState collectAcquireState(Operation *op) {
  SignalAcquireState state = getDirectAcquireState(op);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectAcquireState(&nested));
  return state;
}

static void markAcquireConsumer(Operation *op) {
  op->setAttr(kAcquireInvalidateGmCacheAttrName,
              UnitAttr::get(op->getContext()));
}

static void markNestedAcquireConsumers(Operation *op,
                                       const SignalAcquireState &state) {
  if (!state.needsInvalidateGmCache)
    return;

  op->walk([&](Operation *nested) {
    if (needsAcquireInvalidateBefore(nested))
      markAcquireConsumer(nested);
  });
}

static void clearTNotifyReleaseAttrs(pto::TNotifyOp op) {
  op->removeAttr(kTNotifyDrainMte2AttrName);
  op->removeAttr(kTNotifyDrainMte3AttrName);
  op->removeAttr(kTNotifyDrainFixAttrName);
  op->removeAttr(kTNotifyDsbDdrAttrName);
  op->removeAttr(kTNotifyCleanGmCacheAttrName);
}

static void setTNotifyReleaseAttrs(pto::TNotifyOp op,
                                   const TNotifyReleaseState &state) {
  clearTNotifyReleaseAttrs(op);
  if (state.drainMte2)
    op->setAttr(kTNotifyDrainMte2AttrName, UnitAttr::get(op.getContext()));
  if (state.drainMte3)
    op->setAttr(kTNotifyDrainMte3AttrName, UnitAttr::get(op.getContext()));
  if (state.drainFix)
    op->setAttr(kTNotifyDrainFixAttrName, UnitAttr::get(op.getContext()));
  if (state.needsDsbDdr)
    op->setAttr(kTNotifyDsbDdrAttrName, UnitAttr::get(op.getContext()));
  if (state.needsCleanGmCache)
    op->setAttr(kTNotifyCleanGmCacheAttrName, UnitAttr::get(op.getContext()));
}

static void markNestedTNotifyWithReleaseState(Operation *op,
                                              const TNotifyReleaseState &state) {
  op->walk(
      [&](pto::TNotifyOp notify) { setTNotifyReleaseAttrs(notify, state); });
}

static TNotifyReleaseState
annotateTNotifyReleaseForBlock(Block &block,
                               const TNotifyReleaseState &entryPendingState,
                               const TNotifyReleaseState &loopCarriedState) {
  TNotifyReleaseState pendingState = entryPendingState;
  for (Operation &op : block) {
    if (auto notify = dyn_cast<pto::TNotifyOp>(op)) {
      TNotifyReleaseState notifyState = pendingState;
      notifyState.merge(loopCarriedState);
      setTNotifyReleaseAttrs(notify, notifyState);
      pendingState = {};
    }

    pendingState.merge(getDirectReleaseState(&op));

    TNotifyReleaseState regionEntryState = pendingState;
    TNotifyReleaseState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      TNotifyReleaseState nestedLoopCarriedState = loopCarriedState;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedState.merge(collectReleaseState(&op));

      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(annotateTNotifyReleaseForBlock(
            region.front(), regionEntryState, nestedLoopCarriedState));
      } else {
        TNotifyReleaseState regionState = collectReleaseState(&op);
        TNotifyReleaseState nestedNotifyState = regionEntryState;
        nestedNotifyState.merge(nestedLoopCarriedState);
        nestedNotifyState.merge(regionState);
        markNestedTNotifyWithReleaseState(&op, nestedNotifyState);
        TNotifyReleaseState regionExitState = regionEntryState;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
    }
    pendingState.merge(combinedRegionExitState);

    if (auto barrier = dyn_cast<pto::BarrierOp>(op))
      pendingState.applyBarrier(barrier.getPipe().getPipe());
  }
  return pendingState;
}

static void annotateTNotifyRelease(ModuleOp module) {
  module.walk([](pto::TNotifyOp notify) { clearTNotifyReleaseAttrs(notify); });

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getBody().hasOneBlock()) {
      (void)annotateTNotifyReleaseForBlock(func.getBody().front(),
                                           /*entryPendingState=*/{},
                                           /*loopCarriedState=*/{});
      continue;
    }

    TNotifyReleaseState funcState = collectReleaseState(func.getOperation());
    markNestedTNotifyWithReleaseState(func.getOperation(), funcState);
  }
}

static SignalAcquireState
annotateSignalAcquireForBlock(Block &block,
                              const SignalAcquireState &entryPendingState,
                              const SignalAcquireState &loopCarriedState);

static SignalAcquireState
annotateSignalAcquireForRegion(Region &region,
                               const SignalAcquireState &entryPendingState,
                               const SignalAcquireState &loopCarriedState) {
  if (region.empty())
    return entryPendingState;

  if (region.hasOneBlock()) {
    return annotateSignalAcquireForBlock(region.front(), entryPendingState,
                                         loopCarriedState);
  }

  SignalAcquireState regionState;
  for (Block &block : region)
    for (Operation &nested : block)
      regionState.merge(collectAcquireState(&nested));

  SignalAcquireState nestedConsumerState = entryPendingState;
  nestedConsumerState.merge(loopCarriedState);
  nestedConsumerState.merge(regionState);
  for (Block &block : region)
    for (Operation &nested : block)
      markNestedAcquireConsumers(&nested, nestedConsumerState);

  SignalAcquireState regionExitState = entryPendingState;
  regionExitState.merge(regionState);
  return regionExitState;
}

static std::optional<SignalAcquireState>
annotateSignalAcquireForIf(scf::IfOp ifOp,
                           const SignalAcquireState &entryPendingState,
                           const SignalAcquireState &loopCarriedState) {
  auto acquireWhenConditionTrue =
      getAcquireWhenConditionTrue(ifOp.getCondition());
  if (!acquireWhenConditionTrue)
    return std::nullopt;

  SignalAcquireState thenEntry = entryPendingState;
  SignalAcquireState elseEntry = entryPendingState;
  if (*acquireWhenConditionTrue)
    thenEntry.merge(makeAcquireInvalidateState());
  else
    elseEntry.merge(makeAcquireInvalidateState());

  SignalAcquireState exitState =
      annotateSignalAcquireForRegion(ifOp.getThenRegion(), thenEntry,
                                     loopCarriedState);
  if (ifOp.getElseRegion().empty()) {
    exitState.merge(elseEntry);
  } else {
    exitState.merge(annotateSignalAcquireForRegion(ifOp.getElseRegion(),
                                                   elseEntry,
                                                   loopCarriedState));
  }
  return exitState;
}

static std::optional<SignalAcquireState>
annotateSignalAcquireForWhile(scf::WhileOp whileOp,
                              const SignalAcquireState &entryPendingState,
                              const SignalAcquireState &loopCarriedState) {
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  auto acquireWhenConditionTrue =
      getAcquireWhenConditionTrue(conditionOp.getCondition());
  if (!acquireWhenConditionTrue)
    return std::nullopt;

  SignalAcquireState nestedLoopCarriedState = loopCarriedState;
  nestedLoopCarriedState.merge(collectAcquireState(whileOp.getOperation()));

  SignalAcquireState beforeExitState = annotateSignalAcquireForRegion(
      whileOp.getBefore(), entryPendingState, nestedLoopCarriedState);

  SignalAcquireState bodyEntryState = beforeExitState;
  SignalAcquireState loopExitState = beforeExitState;
  if (*acquireWhenConditionTrue)
    bodyEntryState.merge(makeAcquireInvalidateState());
  else
    loopExitState.merge(makeAcquireInvalidateState());

  SignalAcquireState bodyExitState = annotateSignalAcquireForRegion(
      whileOp.getAfter(), bodyEntryState, nestedLoopCarriedState);
  loopExitState.merge(bodyExitState);
  return loopExitState;
}

static SignalAcquireState
annotateSignalAcquireForBlock(Block &block,
                              const SignalAcquireState &entryPendingState,
                              const SignalAcquireState &loopCarriedState) {
  SignalAcquireState pendingState = entryPendingState;
  for (Operation &op : block) {
    SignalAcquireState consumerState = pendingState;
    consumerState.merge(loopCarriedState);
    if (consumerState.needsInvalidateGmCache &&
        needsAcquireInvalidateBefore(&op)) {
      markAcquireConsumer(&op);
      pendingState = {};
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (auto ifExitState =
              annotateSignalAcquireForIf(ifOp, pendingState, loopCarriedState)) {
        pendingState = *ifExitState;
        pendingState.merge(getDirectAcquireState(&op));
        continue;
      }
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      if (auto whileExitState = annotateSignalAcquireForWhile(
              whileOp, pendingState, loopCarriedState)) {
        pendingState = *whileExitState;
        pendingState.merge(getDirectAcquireState(&op));
        continue;
      }
    }

    SignalAcquireState regionEntryState = pendingState;
    SignalAcquireState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      SignalAcquireState nestedLoopCarriedState = loopCarriedState;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedState.merge(collectAcquireState(&op));

      combinedRegionExitState.merge(annotateSignalAcquireForRegion(
          region, regionEntryState, nestedLoopCarriedState));
    }
    pendingState.merge(combinedRegionExitState);
    pendingState.merge(getDirectAcquireState(&op));
  }
  return pendingState;
}

static void annotateSignalAcquire(ModuleOp module) {
  module.walk([](Operation *op) {
    op->removeAttr(kAcquireInvalidateGmCacheAttrName);
  });

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getBody().hasOneBlock()) {
      (void)annotateSignalAcquireForBlock(func.getBody().front(),
                                          /*entryPendingState=*/{},
                                          /*loopCarriedState=*/{});
      continue;
    }

    SignalAcquireState funcState = collectAcquireState(func.getOperation());
    markNestedAcquireConsumers(func.getOperation(), funcState);
  }
}

struct PTOMemoryConsistencyPass
    : public mlir::pto::impl::PTOMemoryConsistencyBase<
          PTOMemoryConsistencyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    annotateTNotifyRelease(module);
    annotateSignalAcquire(module);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMemoryConsistencyPass() {
  return std::make_unique<PTOMemoryConsistencyPass>();
}
