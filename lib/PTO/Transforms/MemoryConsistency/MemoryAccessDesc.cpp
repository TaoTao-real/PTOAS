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

//===- MemoryAccessDesc.cpp ----------------------------------------------===//
// Basic memory consistency access descriptors for PTO ops.
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/MemoryConsistency/MemoryAccessDesc.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

MemoryAccessDesc
makeDesc(Operation *op, Value value, MemoryConsistencyAccessKind kind,
         std::optional<AddressSpace> memorySpace, std::optional<PIPE> pipe,
         MemoryConsistencyCachePolicy cachePolicy, bool isSignal = false) {
  MemoryAccessDesc desc;
  desc.op = op;
  desc.value = value;
  desc.kind = kind;
  desc.memorySpace = memorySpace;
  desc.pipe = pipe;
  if (pipe)
    desc.component = getMemoryConsistencyComponent(*pipe);
  desc.cachePolicy = cachePolicy;
  desc.isSignal = isSignal;
  desc.mayCreateDirtyLine =
      cachePolicy == MemoryConsistencyCachePolicy::Cacheable &&
      kind == MemoryConsistencyAccessKind::Write;
  desc.mayReadStaleLine =
      cachePolicy == MemoryConsistencyCachePolicy::Cacheable &&
      kind == MemoryConsistencyAccessKind::Read;
  return desc;
}

MemoryAccessDesc makeDescForComponent(Operation *op, Value value,
                                      MemoryConsistencyAccessKind kind,
                                      std::optional<AddressSpace> memorySpace,
                                      MemoryConsistencyComponent component,
                                      MemoryConsistencyCachePolicy cachePolicy,
                                      MemoryConsistencySignalRole signalRole =
                                          MemoryConsistencySignalRole::None) {
  MemoryAccessDesc desc =
      makeDesc(op, value, kind, memorySpace, std::nullopt, cachePolicy,
               signalRole != MemoryConsistencySignalRole::None);
  desc.component = component;
  desc.signalRole = signalRole;
  if (signalRole != MemoryConsistencySignalRole::None) {
    // The comm primitive implementation owns signal cache maintenance. The
    // descriptor marks publish/consume boundaries; it must not be treated as a
    // payload cache hazard by later scans.
    desc.mayCreateDirtyLine = false;
    desc.mayReadStaleLine = false;
  }
  return desc;
}

std::optional<AddressSpace> getTileBufferAddressSpace(Value value) {
  if (!value)
    return std::nullopt;

  if (auto tileBufType = dyn_cast<TileBufType>(value.getType())) {
    auto addressSpace =
        dyn_cast_or_null<AddressSpaceAttr>(tileBufType.getMemorySpace());
    if (addressSpace)
      return addressSpace.getAddressSpace();
  }

  return std::nullopt;
}

MemoryConsistencyCachePolicy getScalarGMCachePolicy(Value value,
                                                    L1Cache l1Cache) {
  auto memorySpace = getMemoryAccessAddressSpace(value);
  if (!memorySpace)
    return MemoryConsistencyCachePolicy::Unknown;
  if (*memorySpace != AddressSpace::GM)
    return MemoryConsistencyCachePolicy::NotApplicable;
  return l1Cache == L1Cache::Uncache ? MemoryConsistencyCachePolicy::NonCache
                                     : MemoryConsistencyCachePolicy::Cacheable;
}

MemoryConsistencyCachePolicy getDefaultScalarGMCachePolicy(Value value) {
  auto memorySpace = getMemoryAccessAddressSpace(value);
  if (!memorySpace)
    return MemoryConsistencyCachePolicy::Unknown;
  if (*memorySpace != AddressSpace::GM)
    return MemoryConsistencyCachePolicy::NotApplicable;
  return MemoryConsistencyCachePolicy::Cacheable;
}

std::optional<PIPE> getPipeForPipeline(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return PIPE::PIPE_S;
  case PipelineType::PIPE_V:
    return PIPE::PIPE_V;
  case PipelineType::PIPE_M:
    return PIPE::PIPE_M;
  case PipelineType::PIPE_MTE1:
    return PIPE::PIPE_MTE1;
  case PipelineType::PIPE_MTE2:
    return PIPE::PIPE_MTE2;
  case PipelineType::PIPE_MTE3:
    return PIPE::PIPE_MTE3;
  case PipelineType::PIPE_ALL:
    return PIPE::PIPE_ALL;
  case PipelineType::PIPE_MTE4:
    return PIPE::PIPE_MTE4;
  case PipelineType::PIPE_MTE5:
    return PIPE::PIPE_MTE5;
  case PipelineType::PIPE_V2:
    return PIPE::PIPE_V2;
  case PipelineType::PIPE_FIX:
    return PIPE::PIPE_FIX;
  case PipelineType::VIRTUAL_PIPE_MTE2_L1A:
    return PIPE::VIRTUAL_PIPE_MTE2_L1A;
  case PipelineType::VIRTUAL_PIPE_MTE2_L1B:
    return PIPE::VIRTUAL_PIPE_MTE2_L1B;
  case PipelineType::PIPE_NUM:
    return PIPE::PIPE_NUM;
  case PipelineType::PIPE_UNASSIGNED:
    return std::nullopt;
  }
  return std::nullopt;
}

MemoryConsistencyCachePolicy
getMacroCachePolicy(std::optional<AddressSpace> memorySpace,
                    std::optional<PIPE> pipe) {
  if (!memorySpace)
    return MemoryConsistencyCachePolicy::Unknown;
  if (*memorySpace != AddressSpace::GM)
    return MemoryConsistencyCachePolicy::NotApplicable;
  if (!pipe)
    return MemoryConsistencyCachePolicy::Unknown;

  switch (*pipe) {
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_MTE4:
  case PIPE::PIPE_MTE5:
  case PIPE::PIPE_FIX:
  case PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return MemoryConsistencyCachePolicy::NonCache;
  case PIPE::PIPE_S:
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
    return MemoryConsistencyCachePolicy::Cacheable;
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_ALL:
  case PIPE::PIPE_NUM:
  case PIPE::PIPE_UNASSIGNED:
    return MemoryConsistencyCachePolicy::Unknown;
  }
  return MemoryConsistencyCachePolicy::Unknown;
}

MemoryAccessDesc makeMacroDesc(Operation *op, Value value,
                               MemoryConsistencyAccessKind kind,
                               std::optional<PIPE> pipe) {
  auto memorySpace = getMemoryAccessAddressSpace(value);
  return makeDesc(op, value, kind, memorySpace, pipe,
                  getMacroCachePolicy(memorySpace, pipe));
}

SmallVector<MemoryAccessDesc, 4>
collectSyncMacroAccessDescs(Operation *op, const SyncMacroModel &model) {
  SmallVector<MemoryAccessDesc, 4> descs;
  for (const SyncMacroPhase &phase : model.phases) {
    std::optional<PIPE> pipe = getPipeForPipeline(phase.pipe);
    for (Value value : phase.useValues)
      descs.push_back(
          makeMacroDesc(op, value, MemoryConsistencyAccessKind::Read, pipe));
    for (Value value : phase.defValues)
      descs.push_back(
          makeMacroDesc(op, value, MemoryConsistencyAccessKind::Write, pipe));
  }
  return descs;
}

SmallVector<MemoryAccessDesc, 4> collectTLoadAccessDescs(TLoadOp op) {
  SmallVector<MemoryAccessDesc, 4> descs;

  // TLOAD reads GM through MTE2 and writes a local tile buffer.
  constexpr PIPE pipe = PIPE::PIPE_MTE2;
  descs.push_back(makeDesc(op.getOperation(), op.getSrc(),
                           MemoryConsistencyAccessKind::Read, AddressSpace::GM,
                           pipe, MemoryConsistencyCachePolicy::NonCache));
  descs.push_back(makeDesc(op.getOperation(), op.getDst(),
                           MemoryConsistencyAccessKind::Write,
                           getMemoryAccessAddressSpace(op.getDst()), pipe,
                           MemoryConsistencyCachePolicy::NotApplicable));
  return descs;
}

SmallVector<MemoryAccessDesc, 4> collectTStoreAccessDescs(TStoreOp op) {
  SmallVector<MemoryAccessDesc, 4> descs;

  // TSTORE uses FIX for ACC/L0C sources and MTE3 for UB/L1 sources. Keep the
  // pipe from the op so memory consistency actions match the emitted ISA path.
  PIPE pipe = op.getPipe();
  descs.push_back(makeDesc(op.getOperation(), op.getSrc(),
                           MemoryConsistencyAccessKind::Read,
                           getMemoryAccessAddressSpace(op.getSrc()), pipe,
                           MemoryConsistencyCachePolicy::NotApplicable));
  descs.push_back(makeDesc(op.getOperation(), op.getDst(),
                           MemoryConsistencyAccessKind::Write, AddressSpace::GM,
                           pipe, MemoryConsistencyCachePolicy::NonCache));
  return descs;
}

SmallVector<MemoryAccessDesc, 4> collectLoadScalarAccessDescs(LoadScalarOp op) {
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Read,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Scalar,
                               getDefaultScalarGMCachePolicy(op.getPtr()))};
}

SmallVector<MemoryAccessDesc, 4>
collectStoreScalarAccessDescs(StoreScalarOp op) {
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Write,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Scalar,
                               getDefaultScalarGMCachePolicy(op.getPtr()))};
}

SmallVector<MemoryAccessDesc, 4> collectPTOLoadAccessDescs(PTOLoadOp op) {
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Read,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Simt,
                               getDefaultScalarGMCachePolicy(op.getPtr()))};
}

SmallVector<MemoryAccessDesc, 4> collectPTOStoreAccessDescs(PTOStoreOp op) {
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Write,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Simt,
                               getDefaultScalarGMCachePolicy(op.getPtr()))};
}

SmallVector<MemoryAccessDesc, 4> collectPTOLdgAccessDescs(PTOLdgOp op) {
  L1Cache l1Cache =
      op.getL1cacheAttr() ? op.getL1cacheAttr().getValue() : L1Cache::Cache;
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Read,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Simt,
                               getScalarGMCachePolicy(op.getPtr(), l1Cache))};
}

SmallVector<MemoryAccessDesc, 4> collectPTOStgAccessDescs(PTOStgOp op) {
  L1Cache l1Cache =
      op.getL1cacheAttr() ? op.getL1cacheAttr().getValue() : L1Cache::Cache;
  return {makeDescForComponent(op.getOperation(), op.getPtr(),
                               MemoryConsistencyAccessKind::Write,
                               getMemoryAccessAddressSpace(op.getPtr()),
                               MemoryConsistencyComponent::Simt,
                               getScalarGMCachePolicy(op.getPtr(), l1Cache))};
}

SmallVector<MemoryAccessDesc, 4> collectTNotifyAccessDescs(TNotifyOp op) {
  return {makeDescForComponent(
      op.getOperation(), op.getSignal(), MemoryConsistencyAccessKind::Write,
      AddressSpace::GM, MemoryConsistencyComponent::Scalar,
      MemoryConsistencyCachePolicy::NotApplicable,
      MemoryConsistencySignalRole::Publish)};
}

SmallVector<MemoryAccessDesc, 4> collectTWaitAccessDescs(TWaitOp op) {
  return {makeDescForComponent(
      op.getOperation(), op.getSignal(), MemoryConsistencyAccessKind::Read,
      AddressSpace::GM, MemoryConsistencyComponent::Scalar,
      MemoryConsistencyCachePolicy::NotApplicable,
      MemoryConsistencySignalRole::Consume)};
}

SmallVector<MemoryAccessDesc, 4> collectTTestAccessDescs(TTestOp op) {
  return {makeDescForComponent(
      op.getOperation(), op.getSignal(), MemoryConsistencyAccessKind::Read,
      AddressSpace::GM, MemoryConsistencyComponent::Scalar,
      MemoryConsistencyCachePolicy::NotApplicable,
      MemoryConsistencySignalRole::Consume)};
}

} // namespace

SmallVector<MemoryAccessDesc, 4>
mlir::pto::collectMemoryAccessDescs(Operation *op) {
  if (!op)
    return {};

  if (auto tload = dyn_cast<TLoadOp>(op))
    return collectTLoadAccessDescs(tload);
  if (auto tstore = dyn_cast<TStoreOp>(op))
    return collectTStoreAccessDescs(tstore);
  if (auto loadScalar = dyn_cast<LoadScalarOp>(op))
    return collectLoadScalarAccessDescs(loadScalar);
  if (auto storeScalar = dyn_cast<StoreScalarOp>(op))
    return collectStoreScalarAccessDescs(storeScalar);
  if (auto load = dyn_cast<PTOLoadOp>(op))
    return collectPTOLoadAccessDescs(load);
  if (auto store = dyn_cast<PTOStoreOp>(op))
    return collectPTOStoreAccessDescs(store);
  if (auto ldg = dyn_cast<PTOLdgOp>(op))
    return collectPTOLdgAccessDescs(ldg);
  if (auto stg = dyn_cast<PTOStgOp>(op))
    return collectPTOStgAccessDescs(stg);
  if (auto notify = dyn_cast<TNotifyOp>(op))
    return collectTNotifyAccessDescs(notify);
  if (auto wait = dyn_cast<TWaitOp>(op))
    return collectTWaitAccessDescs(wait);
  if (auto test = dyn_cast<TTestOp>(op))
    return collectTTestAccessDescs(test);
  if (auto model = getSyncMacroModel(op))
    return collectSyncMacroAccessDescs(op, *model);

  return {};
}

std::optional<AddressSpace>
mlir::pto::getMemoryAccessAddressSpace(Value value) {
  if (!value)
    return std::nullopt;

  if (auto tileBufferSpace = getTileBufferAddressSpace(value))
    return tileBufferSpace;

  if (isa<TensorViewType, PartitionTensorViewType>(value.getType()))
    return AddressSpace::GM;

  if (auto addressSpace = getPTOAddressSpaceAttr(value.getType()))
    return addressSpace.getAddressSpace();

  return std::nullopt;
}

MemoryConsistencyComponent mlir::pto::getMemoryConsistencyComponent(PIPE pipe) {
  switch (pipe) {
  case PIPE::PIPE_S:
    return MemoryConsistencyComponent::Scalar;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
    return MemoryConsistencyComponent::Vector;
  case PIPE::PIPE_M:
    return MemoryConsistencyComponent::Cube;
  case PIPE::PIPE_MTE1:
    return MemoryConsistencyComponent::MTE1;
  case PIPE::PIPE_MTE2:
  case PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return MemoryConsistencyComponent::MTE2;
  case PIPE::PIPE_MTE3:
    return MemoryConsistencyComponent::MTE3;
  case PIPE::PIPE_MTE4:
    return MemoryConsistencyComponent::MTE4;
  case PIPE::PIPE_MTE5:
    return MemoryConsistencyComponent::MTE5;
  case PIPE::PIPE_FIX:
    return MemoryConsistencyComponent::Fix;
  case PIPE::PIPE_ALL:
  case PIPE::PIPE_NUM:
  case PIPE::PIPE_UNASSIGNED:
    return MemoryConsistencyComponent::Unknown;
  }
  return MemoryConsistencyComponent::Unknown;
}

bool mlir::pto::isGMMemoryAccess(const MemoryAccessDesc &desc) {
  return desc.memorySpace && *desc.memorySpace == AddressSpace::GM;
}

bool mlir::pto::isPayloadAccess(const MemoryAccessDesc &desc) {
  return isGMMemoryAccess(desc) &&
         desc.signalRole == MemoryConsistencySignalRole::None && !desc.isSignal;
}

bool mlir::pto::mayNeedCleanBeforeNonCacheConsumer(
    const MemoryAccessDesc &desc) {
  return isPayloadAccess(desc) &&
         desc.kind == MemoryConsistencyAccessKind::Write &&
         desc.mayCreateDirtyLine;
}

bool mlir::pto::mayNeedInvalidateBeforeCacheableConsumer(
    const MemoryAccessDesc &desc) {
  return isPayloadAccess(desc) &&
         desc.kind == MemoryConsistencyAccessKind::Read &&
         desc.mayReadStaleLine;
}

bool mlir::pto::needsPipeDrainBeforePublish(const MemoryAccessDesc &desc) {
  if (!isPayloadAccess(desc) || desc.kind != MemoryConsistencyAccessKind::Write)
    return false;

  if (desc.cachePolicy != MemoryConsistencyCachePolicy::NonCache)
    return false;

  if (!desc.pipe)
    return false;

  switch (*desc.pipe) {
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_MTE4:
  case PIPE::PIPE_MTE5:
  case PIPE::PIPE_FIX:
  case PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return true;
  case PIPE::PIPE_S:
  case PIPE::PIPE_V:
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_ALL:
  case PIPE::PIPE_V2:
  case PIPE::PIPE_NUM:
  case PIPE::PIPE_UNASSIGNED:
    return false;
  }
  return false;
}

StringRef mlir::pto::stringifyMemoryConsistencyAccessKind(
    MemoryConsistencyAccessKind kind) {
  switch (kind) {
  case MemoryConsistencyAccessKind::Read:
    return "read";
  case MemoryConsistencyAccessKind::Write:
    return "write";
  }
  return "unknown";
}

StringRef mlir::pto::stringifyMemoryConsistencyCachePolicy(
    MemoryConsistencyCachePolicy policy) {
  switch (policy) {
  case MemoryConsistencyCachePolicy::Unknown:
    return "unknown";
  case MemoryConsistencyCachePolicy::NotApplicable:
    return "n/a";
  case MemoryConsistencyCachePolicy::NonCache:
    return "non-cache";
  case MemoryConsistencyCachePolicy::Cacheable:
    return "cacheable";
  }
  return "unknown";
}

StringRef mlir::pto::stringifyMemoryConsistencyComponent(
    MemoryConsistencyComponent component) {
  switch (component) {
  case MemoryConsistencyComponent::Unknown:
    return "unknown";
  case MemoryConsistencyComponent::Scalar:
    return "scalar";
  case MemoryConsistencyComponent::Simt:
    return "simt";
  case MemoryConsistencyComponent::Vector:
    return "vector";
  case MemoryConsistencyComponent::Cube:
    return "cube";
  case MemoryConsistencyComponent::MTE1:
    return "mte1";
  case MemoryConsistencyComponent::MTE2:
    return "mte2";
  case MemoryConsistencyComponent::MTE3:
    return "mte3";
  case MemoryConsistencyComponent::MTE4:
    return "mte4";
  case MemoryConsistencyComponent::MTE5:
    return "mte5";
  case MemoryConsistencyComponent::Fix:
    return "fix";
  }
  return "unknown";
}

StringRef mlir::pto::stringifyMemoryConsistencySignalRole(
    MemoryConsistencySignalRole signalRole) {
  switch (signalRole) {
  case MemoryConsistencySignalRole::None:
    return "none";
  case MemoryConsistencySignalRole::Publish:
    return "publish";
  case MemoryConsistencySignalRole::Consume:
    return "consume";
  }
  return "unknown";
}

StringRef mlir::pto::stringifyMemoryConsistencyAddressSpace(
    std::optional<AddressSpace> memorySpace) {
  if (!memorySpace)
    return "unknown";

  switch (*memorySpace) {
  case AddressSpace::Zero:
    return "zero";
  case AddressSpace::GM:
    return "gm";
  case AddressSpace::MAT:
    return "mat";
  case AddressSpace::LEFT:
    return "left";
  case AddressSpace::RIGHT:
    return "right";
  case AddressSpace::ACC:
    return "acc";
  case AddressSpace::VEC:
    return "vec";
  case AddressSpace::BIAS:
    return "bias";
  case AddressSpace::SCALING:
    return "scaling";
  }
  return "unknown";
}
