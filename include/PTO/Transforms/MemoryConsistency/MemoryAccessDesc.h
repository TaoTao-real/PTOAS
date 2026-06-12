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

//===- MemoryAccessDesc.h ---------------------------------------*- C++ -*-===//
// Basic memory consistency access descriptors for PTO ops.
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYACCESSDESC_H
#define PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYACCESSDESC_H

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::pto {

enum class MemoryConsistencyAccessKind {
  Read,
  Write,
};

enum class MemoryConsistencyCachePolicy {
  Unknown,
  NotApplicable,
  NonCache,
  Cacheable,
};

enum class MemoryConsistencyComponent {
  Unknown,
  Scalar,
  Simt,
  Vector,
  Cube,
  MTE1,
  MTE2,
  MTE3,
  MTE4,
  MTE5,
  Fix,
};

enum class MemoryConsistencySignalRole {
  None,
  Publish,
  Consume,
};

struct MemoryAccessDesc {
  Operation *op = nullptr;
  Value value;
  MemoryConsistencyAccessKind kind;
  std::optional<AddressSpace> memorySpace;
  std::optional<PIPE> pipe;
  MemoryConsistencyComponent component = MemoryConsistencyComponent::Unknown;
  MemoryConsistencyCachePolicy cachePolicy =
      MemoryConsistencyCachePolicy::Unknown;
  MemoryConsistencySignalRole signalRole = MemoryConsistencySignalRole::None;
  bool isSignal = false;
  bool mayCreateDirtyLine = false;
  bool mayReadStaleLine = false;
};

SmallVector<MemoryAccessDesc, 4> collectMemoryAccessDescs(Operation *op);

std::optional<AddressSpace> getMemoryAccessAddressSpace(Value value);
MemoryConsistencyComponent getMemoryConsistencyComponent(PIPE pipe);
bool isGMMemoryAccess(const MemoryAccessDesc &desc);
bool isPayloadAccess(const MemoryAccessDesc &desc);
bool mayNeedCleanBeforeNonCacheConsumer(const MemoryAccessDesc &desc);
bool mayNeedInvalidateBeforeCacheableConsumer(const MemoryAccessDesc &desc);
bool needsPipeDrainBeforePublish(const MemoryAccessDesc &desc);

StringRef
stringifyMemoryConsistencyAccessKind(MemoryConsistencyAccessKind kind);
StringRef
stringifyMemoryConsistencyCachePolicy(MemoryConsistencyCachePolicy policy);
StringRef
stringifyMemoryConsistencyComponent(MemoryConsistencyComponent component);
StringRef
stringifyMemoryConsistencySignalRole(MemoryConsistencySignalRole signalRole);
StringRef
stringifyMemoryConsistencyAddressSpace(std::optional<AddressSpace> memorySpace);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYACCESSDESC_H
