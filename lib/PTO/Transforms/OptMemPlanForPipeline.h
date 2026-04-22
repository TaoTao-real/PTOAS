// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef OPT_MEM_PLAN_FOR_PIPELINE_H
#define OPT_MEM_PLAN_FOR_PIPELINE_H

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace pto {

class OptMemPlanForDma {
public:
  void build(func::FuncOp func);
  bool BufferPipeConflict(Value buf1, Value buf2) const;
  bool IsDmaBuffer(Value buf) const;
  bool IsScalarBuffer(Value buf) const;

private:
  void updateDmaBuffers(ValueRange operands);
  void updateScalarBuffers(ValueRange operands);

  DenseSet<Value> dmaBuffers;
  DenseSet<Value> scalarBuffers;
};

} // namespace pto
} // namespace mlir

#endif // OPT_MEM_PLAN_FOR_PIPELINE_H
