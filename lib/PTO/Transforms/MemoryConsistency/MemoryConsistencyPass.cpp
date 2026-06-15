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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

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
  for (const pto::MemoryAccessDesc &desc : pto::collectMemoryAccessDescs(op)) {
    if (pto::mayNeedInvalidateBeforeCacheableConsumer(desc))
      return true;
  }
  return false;
}

static SignalAcquireState getDirectAcquireState(Operation *op) {
  SignalAcquireState state;
  // TWAIT is a blocking acquire of the signal. The signal cache maintenance is
  // owned by the comm primitive itself; PTOAS only needs to invalidate stale
  // payload cache lines before the following cacheable GM read.
  if (isa<pto::TWaitOp>(op))
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

    SignalAcquireState regionEntryState = pendingState;
    SignalAcquireState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      SignalAcquireState nestedLoopCarriedState = loopCarriedState;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedState.merge(collectAcquireState(&op));

      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(annotateSignalAcquireForBlock(
            region.front(), regionEntryState, nestedLoopCarriedState));
      } else {
        SignalAcquireState regionState = collectAcquireState(&op);
        SignalAcquireState nestedConsumerState = regionEntryState;
        nestedConsumerState.merge(nestedLoopCarriedState);
        nestedConsumerState.merge(regionState);
        markNestedAcquireConsumers(&op, nestedConsumerState);
        SignalAcquireState regionExitState = regionEntryState;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
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
