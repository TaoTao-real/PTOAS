// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/MemoryConsistencyAttrs.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMEMORYCONSISTENCY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isGmAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

static Value getPayloadAliasRoot(Value value) {
  for (unsigned depth = 0; value && depth < 64; ++depth) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;

    Value next;
    if (auto part = dyn_cast<pto::PartitionViewOp>(def)) {
      next = part.getSource();
    } else if (auto make = dyn_cast<pto::MakeTensorViewOp>(def)) {
      next = make.getPtr();
    } else if (auto addPtr = dyn_cast<pto::AddPtrOp>(def)) {
      next = addPtr.getPtr();
    } else if (auto castPtr = dyn_cast<pto::CastPtrOp>(def)) {
      next = castPtr.getInput();
    } else if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (result && result.getResultNumber() < unrealized.getNumOperands())
        next = unrealized.getOperand(result.getResultNumber());
    } else if (auto alias = pto::getOperationAliasInfo(def)) {
      if (alias->first == value)
        next = alias->second;
    }

    if (!next || next == value)
      return value;
    value = next;
  }
  return value;
}

static bool payloadMayAlias(Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return true;
  Value lhsRoot = getPayloadAliasRoot(lhs);
  Value rhsRoot = getPayloadAliasRoot(rhs);
  return lhsRoot == rhsRoot;
}

struct PendingReleaseAccess {
  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  Value payload;
  bool drainPending = false;
  bool needsDsbDdr = false;
  bool needsGmCacheCmo = false;
  bool dsbCovered = false;
  bool cmoCovered = false;
  bool selectedByWholeCacheCmo = false;
};

struct TNotifyReleaseState {
  bool drainMte2 = false;
  bool drainMte3 = false;
  bool drainFix = false;
  bool needsDsbDdr = false;
  bool needsGmCacheCmo = false;
  bool hasAddressedCmo = false;
  bool hasWholeCacheCmo = false;
  bool addressedDrainMte2 = false;
  bool addressedDrainMte3 = false;
  bool addressedDrainFix = false;
  bool addressedNeedsDsbDdr = false;
  bool addressedNeedsGmCacheCmo = false;
  SmallVector<PendingReleaseAccess, 8> pendingAccesses;
  SmallVector<Value, 8> addressedCmoPayloads;

  void merge(const TNotifyReleaseState &other) {
    drainMte2 |= other.drainMte2;
    drainMte3 |= other.drainMte3;
    drainFix |= other.drainFix;
    needsDsbDdr |= other.needsDsbDdr;
    needsGmCacheCmo |= other.needsGmCacheCmo;
    hasAddressedCmo |= other.hasAddressedCmo;
    hasWholeCacheCmo |= other.hasWholeCacheCmo;
    addressedDrainMte2 |= other.addressedDrainMte2;
    addressedDrainMte3 |= other.addressedDrainMte3;
    addressedDrainFix |= other.addressedDrainFix;
    addressedNeedsDsbDdr |= other.addressedNeedsDsbDdr;
    addressedNeedsGmCacheCmo |= other.addressedNeedsGmCacheCmo;
    pendingAccesses.append(other.pendingAccesses.begin(),
                           other.pendingAccesses.end());
    addressedCmoPayloads.append(other.addressedCmoPayloads.begin(),
                                other.addressedCmoPayloads.end());
    refreshNeedsDsbDdr();
    refreshNeedsGmCacheCmo();
    if (hasReleaseCmoMarker())
      recomputeAddressedState();
  }

  void clear() {
    drainMte2 = false;
    drainMte3 = false;
    drainFix = false;
    needsDsbDdr = false;
    needsGmCacheCmo = false;
    hasAddressedCmo = false;
    hasWholeCacheCmo = false;
    addressedDrainMte2 = false;
    addressedDrainMte3 = false;
    addressedDrainFix = false;
    addressedNeedsDsbDdr = false;
    addressedNeedsGmCacheCmo = false;
    pendingAccesses.clear();
    addressedCmoPayloads.clear();
  }

  bool hasReleaseCmoMarker() const {
    return hasAddressedCmo || hasWholeCacheCmo;
  }

  bool accessMatchesAddressedCmo(const PendingReleaseAccess &access) const {
    for (Value payload : addressedCmoPayloads)
      if (payloadMayAlias(access.payload, payload))
        return true;
    return false;
  }

  bool accessMatchesReleaseCmo(const PendingReleaseAccess &access) const {
    return access.selectedByWholeCacheCmo || accessMatchesAddressedCmo(access);
  }

  void refreshNeedsGmCacheCmo() {
    needsGmCacheCmo = false;
    for (const PendingReleaseAccess &access : pendingAccesses) {
      if (access.needsGmCacheCmo && !access.cmoCovered) {
        needsGmCacheCmo = true;
        return;
      }
    }
  }

  void refreshNeedsDsbDdr() {
    needsDsbDdr = false;
    for (const PendingReleaseAccess &access : pendingAccesses) {
      if (access.needsDsbDdr && !access.dsbCovered) {
        needsDsbDdr = true;
        return;
      }
    }
  }

  void recomputeAddressedState() {
    addressedDrainMte2 = false;
    addressedDrainMte3 = false;
    addressedDrainFix = false;
    addressedNeedsDsbDdr = false;
    addressedNeedsGmCacheCmo = false;

    for (const PendingReleaseAccess &access : pendingAccesses) {
      if (!accessMatchesReleaseCmo(access))
        continue;
      if (access.drainPending) {
        if (access.pipe == pto::PIPE::PIPE_MTE2)
          addressedDrainMte2 = true;
        if (access.pipe == pto::PIPE::PIPE_MTE3)
          addressedDrainMte3 = true;
        if (access.pipe == pto::PIPE::PIPE_FIX)
          addressedDrainFix = true;
      }
      addressedNeedsDsbDdr |= access.needsDsbDdr && !access.dsbCovered;
      if (access.needsGmCacheCmo && !access.cmoCovered)
        addressedNeedsGmCacheCmo = true;
    }
  }

  void addAccess(pto::PIPE pipe, Value payload, bool needsDsb,
                 bool needsCmo) {
    PendingReleaseAccess access;
    access.pipe = pipe;
    access.payload = payload;
    access.drainPending = pipe == pto::PIPE::PIPE_MTE2 ||
                          pipe == pto::PIPE::PIPE_MTE3 ||
                          pipe == pto::PIPE::PIPE_FIX;
    access.needsDsbDdr = needsDsb;
    access.needsGmCacheCmo = needsCmo;
    pendingAccesses.push_back(access);

    switch (pipe) {
    case pto::PIPE::PIPE_MTE2:
      drainMte2 = true;
      break;
    case pto::PIPE::PIPE_MTE3:
      drainMte3 = true;
      break;
    case pto::PIPE::PIPE_FIX:
      drainFix = true;
      break;
    default:
      break;
    }
    needsDsbDdr |= needsDsb;
    needsGmCacheCmo |= needsCmo;
    if (hasReleaseCmoMarker())
      recomputeAddressedState();
  }

  void markPipeDrained(pto::PIPE pipe) {
    auto pipeMatches = [&](pto::PIPE accessPipe) {
      return pipe == pto::PIPE::PIPE_ALL || accessPipe == pipe;
    };
    for (PendingReleaseAccess &access : pendingAccesses)
      if (pipeMatches(access.pipe))
        access.drainPending = false;

    if (pipe == pto::PIPE::PIPE_MTE2 || pipe == pto::PIPE::PIPE_ALL) {
      drainMte2 = false;
      addressedDrainMte2 = false;
    }
    if (pipe == pto::PIPE::PIPE_MTE3 || pipe == pto::PIPE::PIPE_ALL) {
      drainMte3 = false;
      addressedDrainMte3 = false;
    }
    if (pipe == pto::PIPE::PIPE_FIX || pipe == pto::PIPE::PIPE_ALL) {
      drainFix = false;
      addressedDrainFix = false;
    }
    if (hasReleaseCmoMarker())
      recomputeAddressedState();
  }

  void applyBarrier(pto::PIPE pipe) {
    markPipeDrained(pipe);
  }

  void applyFenceBarrierAll(pto::FenceScope scope) {
    if (scope != pto::FenceScope::GM && scope != pto::FenceScope::All)
      return;

    for (PendingReleaseAccess &access : pendingAccesses) {
      if (!access.needsDsbDdr)
        continue;
      if (hasReleaseCmoMarker() && !accessMatchesReleaseCmo(access))
        continue;
      if (access.drainPending)
        continue;
      if (access.needsGmCacheCmo && !access.cmoCovered)
        continue;
      access.dsbCovered = true;
    }

    refreshNeedsDsbDdr();
    if (hasReleaseCmoMarker())
      recomputeAddressedState();
  }

  void applyCmoCacheInvalid(pto::CmoCacheInvalidOp cmo) {
    pto::AddressSpace space = cmo.getSpace().getAddressSpace();
    if (!isGmAddressSpace(space))
      return;
    Value addr = cmo.getAddr();
    if (!addr) {
      hasWholeCacheCmo = true;
      for (PendingReleaseAccess &access : pendingAccesses) {
        access.selectedByWholeCacheCmo = true;
        if (access.needsGmCacheCmo)
          access.cmoCovered = true;
      }
      refreshNeedsGmCacheCmo();
      recomputeAddressedState();
      cmo->removeAttr(kCmoCacheInvalidSkipLoweringAttrName);
      return;
    }

    hasAddressedCmo = true;
    addressedCmoPayloads.push_back(addr);
    bool needsRealCmo = false;
    for (PendingReleaseAccess &access : pendingAccesses) {
      if (!payloadMayAlias(access.payload, addr))
        continue;
      if (access.needsGmCacheCmo && !access.cmoCovered) {
        access.cmoCovered = true;
        needsRealCmo = true;
      }
    }
    refreshNeedsGmCacheCmo();
    recomputeAddressedState();

    if (needsRealCmo) {
      cmo->removeAttr(kCmoCacheInvalidSkipLoweringAttrName);
    } else {
      cmo->setAttr(kCmoCacheInvalidSkipLoweringAttrName,
                   UnitAttr::get(cmo.getContext()));
    }
  }

  void remapPayloads(llvm::function_ref<Value(Value)> mapper) {
    for (PendingReleaseAccess &access : pendingAccesses)
      if (access.payload)
        access.payload = mapper(access.payload);
    for (Value &payload : addressedCmoPayloads)
      if (payload)
        payload = mapper(payload);
    if (hasReleaseCmoMarker())
      recomputeAddressedState();
  }
};

struct SignalAcquireState {
  bool pendingInvalidateGmCache = false;
  bool dirtyGmCache = false;

  void merge(const SignalAcquireState &other) {
    pendingInvalidateGmCache |= other.pendingInvalidateGmCache;
    dirtyGmCache |= other.dirtyGmCache;
  }

  void consumeAcquire() {
    pendingInvalidateGmCache = false;
    dirtyGmCache = false;
  }

  void applyCmoCacheInvalid(pto::AddressSpace space,
                            pto::CmoCacheInvalidOp cmo = nullptr) {
    if (!isGmAddressSpace(space))
      return;
    if (cmo && pendingInvalidateGmCache)
      cmo->removeAttr(kCmoCacheInvalidSkipLoweringAttrName);
    dirtyGmCache = false;
    pendingInvalidateGmCache = false;
  }
};

static bool isGmScalarMemory(Type type) {
  if (auto ptrTy = dyn_cast<pto::PtrType>(type)) {
    pto::AddressSpace space = ptrTy.getMemorySpace().getAddressSpace();
    return isGmAddressSpace(space);
  }

  if (auto memTy = dyn_cast<MemRefType>(type)) {
    auto spaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memTy.getMemorySpace());
    return !spaceAttr || isGmAddressSpace(spaceAttr.getAddressSpace());
  }

  return false;
}

static TNotifyReleaseState getMte2PayloadReadReleaseState(Value payload) {
  TNotifyReleaseState state;
  state.addAccess(pto::PIPE::PIPE_MTE2, payload, /*needsDsb=*/false,
                  /*needsCmo=*/false);
  return state;
}

static TNotifyReleaseState getMte3GmWriteReleaseState(Value payload) {
  TNotifyReleaseState state;
  state.addAccess(pto::PIPE::PIPE_MTE3, payload, /*needsDsb=*/true,
                  /*needsCmo=*/false);
  return state;
}

static TNotifyReleaseState getFixGmWriteReleaseState(Value payload) {
  TNotifyReleaseState state;
  state.addAccess(pto::PIPE::PIPE_FIX, payload, /*needsDsb=*/true,
                  /*needsCmo=*/false);
  return state;
}

static TNotifyReleaseState getCacheableGmStoreReleaseState(Value payload) {
  TNotifyReleaseState state;
  state.addAccess(pto::PIPE::PIPE_UNASSIGNED, payload, /*needsDsb=*/true,
                  /*needsCmo=*/true);
  return state;
}

static TNotifyReleaseState getReleaseStateForMacroModel(Operation *op) {
  TNotifyReleaseState state;
  auto model = getSyncMacroModel(op);
  if (!model)
    return state;

  for (const SyncMacroPhase &phase : model->phases) {
    // Macro MTE3 phases write GM payloads internally. A following TNotify must
    // publish its signal only after those stores are drained and DDR-visible.
    if (phase.pipe == PipelineType::PIPE_MTE3) {
      if (phase.defValues.empty()) {
        state.merge(getMte3GmWriteReleaseState(Value{}));
      } else {
        for (Value value : phase.defValues)
          state.merge(getMte3GmWriteReleaseState(value));
      }
    }
  }
  return state;
}

static func::FuncOp lookupCallee(func::CallOp call) {
  return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call.getOperation(), call.getCalleeAttr());
}

static TNotifyReleaseState
collectTNotifyReleaseState(Region &region,
                           llvm::DenseSet<Operation *> &activeCallees);

static Value remapCalleePayloadToCallOperand(Value payload, func::FuncOp callee,
                                             func::CallOp call) {
  if (!payload)
    return payload;

  Value root = getPayloadAliasRoot(payload);
  auto arg = dyn_cast<BlockArgument>(root);
  if (!arg)
    return payload;
  if (arg.getOwner() != &callee.getBody().front())
    return payload;
  if (arg.getArgNumber() >= call.getNumOperands())
    return payload;
  return call.getOperand(arg.getArgNumber());
}

static TNotifyReleaseState
getReleaseStateForCall(func::CallOp call,
                       llvm::DenseSet<Operation *> &activeCallees) {
  func::FuncOp callee = lookupCallee(call);
  if (!callee || callee.isExternal())
    return {};
  if (!activeCallees.insert(callee.getOperation()).second)
    return {};

  TNotifyReleaseState state =
      collectTNotifyReleaseState(callee.getBody(), activeCallees);
  activeCallees.erase(callee.getOperation());
  state.remapPayloads([&](Value payload) {
    return remapCalleePayloadToCallOperand(payload, callee, call);
  });
  return state;
}

static TNotifyReleaseState
getDirectTNotifyReleaseState(Operation *op,
                             llvm::DenseSet<Operation *> &activeCallees) {
  if (isa<pto::BarrierOp, pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp>(op))
    return {};

  if (auto call = dyn_cast<func::CallOp>(op))
    return getReleaseStateForCall(call, activeCallees);

  if (auto store = dyn_cast<pto::StoreScalarOp>(op)) {
    if (isGmScalarMemory(store.getPtr().getType()))
      return getCacheableGmStoreReleaseState(store.getPtr());
  }

  if (auto tload = dyn_cast<pto::TLoadOp>(op))
    return getMte2PayloadReadReleaseState(tload.getSrc());

  if (auto prefetch = dyn_cast<pto::TPrefetchOp>(op))
    return getMte2PayloadReadReleaseState(prefetch.getSrc());

  if (auto tstore = dyn_cast<pto::TStoreOp>(op)) {
    if (tstore.getPipe() == pto::PIPE::PIPE_MTE3)
      return getMte3GmWriteReleaseState(tstore.getDst());
    if (tstore.getPipe() == pto::PIPE::PIPE_FIX)
      return getFixGmWriteReleaseState(tstore.getDst());
    return {};
  }

  if (isa<pto::TStoreFPOp>(op))
    return getFixGmWriteReleaseState(cast<pto::TStoreFPOp>(op).getDst());

  TNotifyReleaseState macroState = getReleaseStateForMacroModel(op);
  if (macroState.drainMte3 || macroState.drainFix ||
      macroState.needsDsbDdr || macroState.needsGmCacheCmo)
    return macroState;

  return {};
}

static TNotifyReleaseState getDirectTNotifyReleaseState(Operation *op) {
  llvm::DenseSet<Operation *> activeCallees;
  return getDirectTNotifyReleaseState(op, activeCallees);
}

static TNotifyReleaseState
collectTNotifyReleaseState(Operation *op,
                           llvm::DenseSet<Operation *> &activeCallees) {
  TNotifyReleaseState state = getDirectTNotifyReleaseState(op, activeCallees);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectTNotifyReleaseState(&nested, activeCallees));
  return state;
}

static TNotifyReleaseState
collectTNotifyReleaseState(Region &region,
                           llvm::DenseSet<Operation *> &activeCallees) {
  TNotifyReleaseState state;
  for (Block &block : region)
    for (Operation &nested : block)
      state.merge(collectTNotifyReleaseState(&nested, activeCallees));
  return state;
}

static TNotifyReleaseState collectTNotifyReleaseState(Region &region) {
  llvm::DenseSet<Operation *> activeCallees;
  return collectTNotifyReleaseState(region, activeCallees);
}

static void applyFenceBarrierAllForSummary(pto::FenceBarrierAllOp fence,
                                           TNotifyReleaseState &state) {
  if (fence.getScope().getScope() != pto::FenceScope::GM &&
      fence.getScope().getScope() != pto::FenceScope::All)
    return;

  // The real annotation pass inserts the pending GM-write pipe drain before a
  // barrier_all.  Loop summaries must model that transfer without mutating IR,
  // otherwise already-released loop-carried writes are reported again at the
  // next iteration's TNotify.
  state.markPipeDrained(pto::PIPE::PIPE_MTE3);
  state.markPipeDrained(pto::PIPE::PIPE_FIX);
  state.applyFenceBarrierAll(fence.getScope().getScope());
}

static TNotifyReleaseState getTNotifyReleaseExitStateForBlock(
    Block &block, TNotifyReleaseState pendingState);

static TNotifyReleaseState
getTNotifyReleaseExitState(Operation *op,
                           TNotifyReleaseState pendingState = {}) {
  if (isa<pto::TNotifyOp>(op))
    pendingState.clear();

  pendingState.merge(getDirectTNotifyReleaseState(op));

  TNotifyReleaseState regionEntryState = pendingState;
  TNotifyReleaseState combinedRegionExitState;
  for (Region &region : op->getRegions()) {
    if (region.hasOneBlock()) {
      combinedRegionExitState.merge(
          getTNotifyReleaseExitStateForBlock(region.front(), regionEntryState));
      continue;
    }

    TNotifyReleaseState regionExitState = regionEntryState;
    regionExitState.merge(collectTNotifyReleaseState(region));
    combinedRegionExitState.merge(regionExitState);
  }
  pendingState.merge(combinedRegionExitState);

  if (auto barrier = dyn_cast<pto::BarrierOp>(op))
    pendingState.applyBarrier(barrier.getPipe().getPipe());
  if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
    pendingState.applyCmoCacheInvalid(cmo);
  if (auto fence = dyn_cast<pto::FenceBarrierAllOp>(op))
    applyFenceBarrierAllForSummary(fence, pendingState);
  return pendingState;
}

static TNotifyReleaseState getTNotifyReleaseExitStateForBlock(
    Block &block, TNotifyReleaseState pendingState) {
  for (Operation &op : block)
    pendingState = getTNotifyReleaseExitState(&op, pendingState);
  return pendingState;
}

static bool isLoopLikeOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp>(op);
}

static bool isMemoryConsistencyBoundaryOp(Operation *op) {
  return isa<pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp, pto::TNotifyOp,
             pto::TWaitOp, pto::TTestOp>(op);
}

static bool calleeContainsMemoryConsistencyBoundaryOps(
    func::FuncOp callee, llvm::DenseSet<Operation *> &activeCallees) {
  if (!callee || callee.isExternal())
    return false;
  if (!activeCallees.insert(callee.getOperation()).second)
    return false;

  WalkResult result = callee.walk([&](Operation *op) -> WalkResult {
    if (op == callee.getOperation())
      return WalkResult::advance();

    if (auto nestedCall = dyn_cast<func::CallOp>(op)) {
      func::FuncOp nestedCallee = lookupCallee(nestedCall);
      if (calleeContainsMemoryConsistencyBoundaryOps(nestedCallee,
                                                    activeCallees))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (isMemoryConsistencyBoundaryOp(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  activeCallees.erase(callee.getOperation());
  return result.wasInterrupted();
}

static bool diagnoseNonInlinedMemoryConsistencyCalls(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    // Entry wrappers are launch/orchestration functions in EmitC tests.  They
    // can call several kernel functions that are analyzed independently by this
    // module pass, so rejecting those calls would incorrectly forbid the normal
    // multi-kernel entry shape.  The unsafe case is a non-inlined call from
    // inside an actual kernel body, where the caller-side signal/fence cannot
    // see payload actions hidden in the callee.
    if (func->hasAttr("pto.entry"))
      continue;

    func.walk([&](func::CallOp call) {
      func::FuncOp callee = lookupCallee(call);
      if (!callee || callee.isExternal())
        return;

      llvm::DenseSet<Operation *> activeCallees;
      if (!calleeContainsMemoryConsistencyBoundaryOps(callee, activeCallees))
        return;

      call.emitOpError()
          << "calls @" << callee.getSymName()
          << ", which contains PTO memory consistency boundary operations; "
             "inline the callee before `pto-memory-consistency` or keep "
             "CMO, fence, and signal operations in the caller";
      hasFailure = true;
    });
  }
  return hasFailure;
}

static void setTNotifyReleaseAttrs(pto::TNotifyOp op,
                                   const TNotifyReleaseState &state) {
  op->removeAttr(kTNotifyDrainMte2AttrName);
  op->removeAttr(kTNotifyDrainMte3AttrName);
  if (state.drainMte2)
    op->setAttr(kTNotifyDrainMte2AttrName, UnitAttr::get(op.getContext()));
  if (state.drainMte3)
    op->setAttr(kTNotifyDrainMte3AttrName, UnitAttr::get(op.getContext()));
}

static void setTNotifyPipeDrainAttrs(pto::TNotifyOp op,
                                     const TNotifyReleaseState &state) {
  TNotifyReleaseState emitState;
  emitState.drainMte2 =
      state.hasReleaseCmoMarker() && state.addressedDrainMte2;
  emitState.drainMte3 =
      state.hasReleaseCmoMarker() && state.addressedDrainMte3;
  setTNotifyReleaseAttrs(op, emitState);
}

static void diagnoseTNotifyRelease(pto::TNotifyOp op,
                                   const TNotifyReleaseState &state,
                                   bool &hasFailure) {
  if (!state.hasReleaseCmoMarker())
    return;

  if (state.addressedNeedsGmCacheCmo) {
    op.emitOpError()
        << "requires explicit `pto.cmo.cacheinvalid %addr "
           "single_cache_line` or `pto.cmo.cacheinvalid all "
           "#pto.address_space<gm>` after matching cacheable GM scalar stores "
           "and before `pto.fence.barrier_all #pto.fence_scope<gm>`";
    hasFailure = true;
    return;
  }
  if (state.addressedNeedsDsbDdr) {
    op.emitOpError()
        << "requires explicit `pto.fence.barrier_all #pto.fence_scope<gm>` "
           "after the matching `pto.cmo.cacheinvalid` release marker and "
           "before publishing the signal";
    hasFailure = true;
  }
}

static void insertDrainsBeforeBarrierAll(pto::FenceBarrierAllOp fence,
                                         TNotifyReleaseState &state) {
  if (fence.getScope().getScope() != pto::FenceScope::GM &&
      fence.getScope().getScope() != pto::FenceScope::All)
    return;
  OpBuilder builder(fence);
  auto insertBarrier = [&](pto::PIPE pipe) {
    builder.create<pto::BarrierOp>(
        fence.getLoc(), pto::PipeAttr::get(fence.getContext(), pipe));
  };
  const bool drainMte3 =
      state.hasReleaseCmoMarker() && state.addressedDrainMte3;
  const bool drainFix =
      state.hasReleaseCmoMarker() && state.addressedDrainFix;
  if (drainMte3) {
    insertBarrier(pto::PIPE::PIPE_MTE3);
    state.markPipeDrained(pto::PIPE::PIPE_MTE3);
  }
  if (drainFix) {
    insertBarrier(pto::PIPE::PIPE_FIX);
    state.markPipeDrained(pto::PIPE::PIPE_FIX);
  }
}

static void markNestedTNotifyWithState(Operation *op,
                                       const TNotifyReleaseState &state,
                                       bool &hasFailure) {
  op->walk([&](pto::TNotifyOp notify) {
    diagnoseTNotifyRelease(notify, state, hasFailure);
    setTNotifyPipeDrainAttrs(notify, state);
  });
}

static void markNestedTNotifyWithState(Region &region,
                                       const TNotifyReleaseState &state,
                                       bool &hasFailure) {
  for (Block &block : region) {
    for (Operation &nested : block)
      markNestedTNotifyWithState(&nested, state, hasFailure);
  }
}

static TNotifyReleaseState
annotateTNotifyReleaseForBlock(Block &block,
                               TNotifyReleaseState entryPendingState,
                               TNotifyReleaseState loopCarriedState,
                               bool &hasFailure) {
  TNotifyReleaseState pendingState = entryPendingState;
  for (Operation &op : block) {
    if (auto notify = dyn_cast<pto::TNotifyOp>(op)) {
      TNotifyReleaseState notifyState = pendingState;
      notifyState.merge(loopCarriedState);
      diagnoseTNotifyRelease(notify, notifyState, hasFailure);
      setTNotifyPipeDrainAttrs(notify, notifyState);
      pendingState.clear();
    }

    pendingState.merge(getDirectTNotifyReleaseState(&op));

    TNotifyReleaseState regionEntryState = pendingState;
    TNotifyReleaseState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      TNotifyReleaseState nestedLoopCarriedState = loopCarriedState;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedState.merge(getTNotifyReleaseExitState(&op));

      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(annotateTNotifyReleaseForBlock(
            region.front(), regionEntryState, nestedLoopCarriedState,
            hasFailure));
      } else {
        TNotifyReleaseState regionState = collectTNotifyReleaseState(region);
        TNotifyReleaseState nestedNotifyState = regionEntryState;
        nestedNotifyState.merge(nestedLoopCarriedState);
        nestedNotifyState.merge(regionState);
        markNestedTNotifyWithState(region, nestedNotifyState, hasFailure);

        TNotifyReleaseState regionExitState = regionEntryState;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
    }
    pendingState.merge(combinedRegionExitState);

    if (auto barrier = dyn_cast<pto::BarrierOp>(op))
      pendingState.applyBarrier(barrier.getPipe().getPipe());
    if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
      pendingState.applyCmoCacheInvalid(cmo);
    if (auto fence = dyn_cast<pto::FenceBarrierAllOp>(op)) {
      insertDrainsBeforeBarrierAll(fence, pendingState);
      pendingState.applyFenceBarrierAll(fence.getScope().getScope());
    }
  }
  return pendingState;
}

static bool annotateTNotifyRelease(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    if (func.getBody().hasOneBlock()) {
      (void)annotateTNotifyReleaseForBlock(func.getBody().front(),
                                           TNotifyReleaseState{},
                                           TNotifyReleaseState{},
                                           hasFailure);
      continue;
    }

    // Be conservative for pre-existing CFG: without a path-sensitive CFG data
    // flow here, every TNotify may observe any release-relevant work in the
    // function.
    TNotifyReleaseState funcState = collectTNotifyReleaseState(func.getBody());
    markNestedTNotifyWithState(func.getBody(), funcState, hasFailure);
  }
  return hasFailure;
}

static void clearAcquireAttrs(pto::LoadScalarOp op) {
  op->removeAttr(kAcquireInvalidateGmCacheAttrName);
}

static void diagnoseAcquireLoad(pto::LoadScalarOp op,
                                const SignalAcquireState &state,
                                bool &hasFailure) {
  if (!state.pendingInvalidateGmCache ||
      !isGmScalarMemory(op.getPtr().getType()))
    return;
  if (state.dirtyGmCache) {
    op.emitOpError()
        << "cannot perform a cacheable GM load after signal acquire while "
           "dirty GM cache may exist; insert explicit "
           "`pto.cmo.cacheinvalid all #pto.address_space<gm>` before the load";
    hasFailure = true;
    return;
  }
  op.emitOpError()
      << "requires explicit `pto.cmo.cacheinvalid all #pto.address_space<gm>` "
         "before a cacheable GM load after `pto.comm.twait` or successful "
         "`pto.comm.ttest`";
  hasFailure = true;
}

static void consumeAcquireAfterDiagnostic(SignalAcquireState &state) {
  if (state.pendingInvalidateGmCache)
    state.consumeAcquire();
}

static SignalAcquireState collectSignalAcquireState(Operation *op) {
  SignalAcquireState state;
  if (isa<pto::TWaitOp, pto::TTestOp>(op))
    state.pendingInvalidateGmCache = true;
  if (auto store = dyn_cast<pto::StoreScalarOp>(op);
      store && isGmScalarMemory(store.getPtr().getType()))
    state.dirtyGmCache = true;
  if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
    state.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace());

  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectSignalAcquireState(&nested));
  return state;
}

static SignalAcquireState collectSignalAcquireState(Region &region) {
  SignalAcquireState state;
  for (Block &block : region)
    for (Operation &nested : block)
      state.merge(collectSignalAcquireState(&nested));
  return state;
}

static void markNestedAcquireLoadsWithState(Operation *op,
                                            SignalAcquireState state,
                                            bool &hasFailure) {
  op->walk([&](pto::LoadScalarOp load) {
    clearAcquireAttrs(load);
    diagnoseAcquireLoad(load, state, hasFailure);
    consumeAcquireAfterDiagnostic(state);
  });
}

static void markNestedAcquireLoadsWithState(Region &region,
                                            SignalAcquireState state,
                                            bool &hasFailure) {
  for (Block &block : region) {
    for (Operation &nested : block)
      markNestedAcquireLoadsWithState(&nested, state, hasFailure);
  }
}

static SignalAcquireState
annotateSignalAcquireForBlock(Block &block, SignalAcquireState entryState,
                              bool &hasFailure) {
  SignalAcquireState state = entryState;
  for (Operation &op : block) {
    if (auto load = dyn_cast<pto::LoadScalarOp>(op)) {
      clearAcquireAttrs(load);
      diagnoseAcquireLoad(load, state, hasFailure);
      consumeAcquireAfterDiagnostic(state);
    }

    if (auto store = dyn_cast<pto::StoreScalarOp>(op);
        store && isGmScalarMemory(store.getPtr().getType()))
      state.dirtyGmCache = true;

    if (isa<pto::TWaitOp, pto::TTestOp>(op))
      state.pendingInvalidateGmCache = true;

    if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
      state.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace(), cmo);

    SignalAcquireState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(
            annotateSignalAcquireForBlock(region.front(), state, hasFailure));
      } else {
        markNestedAcquireLoadsWithState(region, state, hasFailure);
        SignalAcquireState regionState = collectSignalAcquireState(region);
        SignalAcquireState regionExitState = state;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
    }

    if (isLoopLikeOp(&op))
      combinedRegionExitState.merge(state);
    state.merge(combinedRegionExitState);
  }
  return state;
}

static bool annotateSignalAcquire(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    if (func.getBody().hasOneBlock()) {
      (void)annotateSignalAcquireForBlock(func.getBody().front(),
                                          SignalAcquireState{}, hasFailure);
      continue;
    }

    SignalAcquireState funcState = collectSignalAcquireState(func.getBody());
    markNestedAcquireLoadsWithState(func.getBody(), funcState, hasFailure);
  }
  return hasFailure;
}

static void clearMemoryConsistencyInternalAttrs(ModuleOp module) {
  module.walk([](pto::CmoCacheInvalidOp cmo) {
    cmo->removeAttr(kCmoCacheInvalidSkipLoweringAttrName);
  });
}

struct PTOMemoryConsistencyPass
    : public mlir::pto::impl::PTOMemoryConsistencyBase<
          PTOMemoryConsistencyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    clearMemoryConsistencyInternalAttrs(module);
    bool callFailed = diagnoseNonInlinedMemoryConsistencyCalls(module);
    bool releaseFailed = annotateTNotifyRelease(module);
    bool acquireFailed = annotateSignalAcquire(module);
    if (callFailed || releaseFailed || acquireFailed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMemoryConsistencyPass() {
  return std::make_unique<PTOMemoryConsistencyPass>();
}
