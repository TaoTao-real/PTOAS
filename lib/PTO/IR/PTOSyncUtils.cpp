//===- PTOSyncUtils.cpp - Shared sync mapping helpers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOSyncUtils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<SyncOpType> mlir::pto::parseSyncOpTypeLikeAttr(Attribute attr) {
  if (auto a = dyn_cast_or_null<PipeEventTypeAttr>(attr))
    return a.getOpType();
  if (auto a = dyn_cast_or_null<SyncOpTypeAttr>(attr))
    return a.getOpType();
  return failure();
}

PIPE mlir::pto::mapSyncOpTypeToPipe(SyncOpType opType) {
  switch (opType) {
  case SyncOpType::TLOAD:
    return PIPE::PIPE_MTE2;
  case SyncOpType::TSTORE_VEC:
    return PIPE::PIPE_MTE3;
  case SyncOpType::TSTORE_ACC:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2L:
  case SyncOpType::TMOV_M2B:
    return PIPE::PIPE_MTE1;
  case SyncOpType::TMOV_M2S:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2V:
    return PIPE::PIPE_V;
  case SyncOpType::TMOV_V2M:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMATMUL:
    return PIPE::PIPE_M;
  case SyncOpType::TVEC:
  case SyncOpType::TVECWAIT_EVENT:
    return PIPE::PIPE_V;
  default:
    return PIPE::PIPE_UNASSIGNED;
  }
}

bool mlir::pto::isConcreteSyncPipe(PIPE pipe) {
  return pipe != PIPE::PIPE_UNASSIGNED && pipe != PIPE::PIPE_ALL;
}
