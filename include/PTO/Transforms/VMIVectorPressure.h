// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#ifndef PTO_TRANSFORMS_VMIVECTORPRESSURE_H
#define PTO_TRANSFORMS_VMIVECTORPRESSURE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::pto {

/// Conservative VMI register-pressure estimate measured in A5 256-byte
/// physical vector chunks.  `isExact == false` is a hard rejection for
/// transformations that would extend a live range.
struct VMIVectorPressureEstimate {
  int64_t peakVectorChunks = 0;
  int64_t peakVectorValues = 0;
  int64_t persistentVectorChunks = 0;
  int64_t temporaryVectorChunks = 0;
  int64_t loopCarriedVectorChunks = 0;
  bool isExact = true;
};

FailureOr<int64_t> getVMIVectorChunks(Type type);

/// Estimates the live VMI vectors in one loop body.  Values captured from the
/// preheader and vector iter_args are persistent for the whole body.
VMIVectorPressureEstimate estimateVMILoopPressure(scf::ForOp loop);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMIVECTORPRESSURE_H
