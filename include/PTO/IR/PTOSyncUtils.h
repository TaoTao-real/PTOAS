//===- PTOSyncUtils.h - Shared sync mapping helpers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_
#define MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

/// Parse a sync endpoint-like attribute used by high-level sync operations.
/// Supported attribute kinds are:
///   - pto.pipe_event_type<...>
///   - pto.sync_op_type<...>
FailureOr<SyncOpType> parseSyncOpTypeLikeAttr(Attribute attr);

/// Map high-level sync operation type to concrete hardware PIPE.
PIPE mapSyncOpTypeToPipe(SyncOpType opType);

/// True if the pipe is a concrete endpoint pipe (not PIPE_ALL/UNASSIGNED).
bool isConcreteSyncPipe(PIPE pipe);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTOSYNCUTILS_H_
