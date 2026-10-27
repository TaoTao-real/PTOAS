// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef PTO_TRANSFORMS_CVVF_H
#define PTO_TRANSFORMS_CVVF_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::pto {
/// Verified effects of an address-parameterized VF. The body, not an external
/// annotation, is authoritative. Unknown effects are an error.
struct CVVFEffects {
  SmallVector<bool> reads;
  SmallVector<bool> writes;
};
FailureOr<CVVFEffects> getCVVFEffects(func::FuncOp function);
FailureOr<CVVFEffects> getCVVFEffects(func::CallOp call);
Value getCVStorageRoot(Value value);
std::unique_ptr<Pass> createCVOutlineVFPass();
std::unique_ptr<Pass> createCVValidateVFPass();
std::unique_ptr<Pass> createCVApplySchedulePass(bool inferMinimumSlots = false,
                                              bool drainVFCalls = true);
std::unique_ptr<Pass> createCVBindSharedStoragePass();
} // namespace mlir::pto
#endif
