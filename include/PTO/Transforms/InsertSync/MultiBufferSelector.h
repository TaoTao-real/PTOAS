// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pto {

/// Build a round-robin slot selector across a loop nest.
///
/// - The selector is inserted at the beginning of `baseLoop`'s body.
/// - The computed slot id is based on a flattened linear index across
///   `baseLoop` and all its parent `scf.for` loops, supporting non-unit steps.
/// - The returned value is guaranteed to be in `[0, factor)`.
/// - Returns a null Value if `baseLoop` is invalid.
Value buildLoopNestRoundRobinSlotIndex(IRRewriter &rewriter, scf::ForOp baseLoop,
                                       int factor);

/// Legacy helper for 2-slot ping/pong paths that still want an even/odd
/// predicate. New slot-aware code should prefer
/// `buildLoopNestRoundRobinSlotIndex`.
Value buildLoopNestParityCond(IRRewriter &rewriter, scf::ForOp baseLoop);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H
