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

} // namespace

SmallVector<MemoryAccessDesc, 4>
mlir::pto::collectMemoryAccessDescs(Operation *op) {
  if (!op)
    return {};

  if (auto tload = dyn_cast<TLoadOp>(op))
    return collectTLoadAccessDescs(tload);
  if (auto tstore = dyn_cast<TStoreOp>(op))
    return collectTStoreAccessDescs(tstore);

  return {};
}

std::optional<AddressSpace>
mlir::pto::getMemoryAccessAddressSpace(Value value) {
  if (!value)
    return std::nullopt;

  if (auto tileBufferSpace = getTileBufferAddressSpace(value))
    return tileBufferSpace;

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
