// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "OptMemPlanForPipeline.h"

#include "TileBufferSemantics.h"
#include "Utils.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isDmaLikeOp(Operation *op) {
  return isa<pto::TLoadOp, pto::TStoreOp, pto::TPrefetchOp,
             pto::BuildAsyncSessionOp, pto::TPutAsyncOp,
             pto::TGetAsyncOp>(op);
}

static bool isScalarLikeOp(Operation *op) {
  return isa<pto::LoadScalarOp, pto::StoreScalarOp, pto::TSetValOp,
             pto::TGetValOp>(op);
}

static void collectLocalRoots(DenseSet<Value> &set, ValueRange operands) {
  for (Value operand : operands) {
    auto memorySpaceAttr = GetBufferSpaceAttr(operand);
    if (!isLocalBuffer(memorySpaceAttr))
      continue;
    set.insert(tracebackBufferRoot(operand));
  }
}

} // namespace

void OptMemPlanForDma::build(func::FuncOp func) {
  func.walk([&](Operation *op) {
    if (isDmaLikeOp(op)) {
      updateDmaBuffers(op->getOperands());
      return;
    }
    if (isScalarLikeOp(op))
      updateScalarBuffers(op->getOperands());
  });
}

void OptMemPlanForDma::updateDmaBuffers(ValueRange operands) {
  collectLocalRoots(dmaBuffers, operands);
}

void OptMemPlanForDma::updateScalarBuffers(ValueRange operands) {
  collectLocalRoots(scalarBuffers, operands);
}

bool OptMemPlanForDma::IsDmaBuffer(Value buf) const {
  return dmaBuffers.contains(buf);
}

bool OptMemPlanForDma::IsScalarBuffer(Value buf) const {
  return scalarBuffers.contains(buf);
}

bool OptMemPlanForDma::BufferPipeConflict(Value buf1, Value buf2) const {
  if (IsScalarBuffer(buf1) && IsScalarBuffer(buf2))
    return false;
  if (IsScalarBuffer(buf1) || IsScalarBuffer(buf2))
    return true;
  if (IsDmaBuffer(buf1) || IsDmaBuffer(buf2))
    return true;
  return false;
}
