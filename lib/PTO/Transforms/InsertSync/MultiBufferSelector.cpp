// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/MultiBufferSelector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace mlir {
namespace pto {

Value buildLoopNestRoundRobinSlotIndex(IRRewriter &rewriter, scf::ForOp baseLoop,
                                       int factor) {
  if (!baseLoop || factor <= 1)
    return nullptr;

  Location loc = baseLoop.getLoc();

  // Insert at the beginning of the base loop body so it dominates all uses
  // within the loop nest.
  rewriter.setInsertionPointToStart(baseLoop.getBody());

  Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value nElems = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Collect loop nest from inner to outer (baseLoop, parent, ...).
  SmallVector<scf::ForOp> loops;
  for (scf::ForOp cur = baseLoop; cur; cur = cur->getParentOfType<scf::ForOp>())
    loops.push_back(cur);

  for (scf::ForOp loop : loops) {
    Value iv = loop.getInductionVar();
    Value lb = loop.getLowerBound();
    Value ub = loop.getUpperBound();
    Value step = loop.getStep();

    // iter = (iv - lb) / step
    Value iter = rewriter.create<arith::DivUIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, iv, lb), step);
    idx = rewriter.create<arith::AddIOp>(
        loc, idx, rewriter.create<arith::MulIOp>(loc, iter, nElems));

    // tripCount = ceilDiv(ub - lb, step) = (ub - lb + step - 1) / step
    Value span = rewriter.create<arith::SubIOp>(loc, ub, lb);
    Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, step, one);
    Value num = rewriter.create<arith::AddIOp>(loc, span, stepMinusOne);
    Value tripCount = rewriter.create<arith::DivUIOp>(loc, num, step);
    nElems = rewriter.create<arith::MulIOp>(loc, nElems, tripCount);
  }

  Value factorValue = rewriter.create<arith::ConstantIndexOp>(loc, factor);
  return rewriter.create<arith::RemUIOp>(loc, idx, factorValue);
}

Value buildLoopNestParityCond(IRRewriter &rewriter, scf::ForOp baseLoop) {
  Value slotIndex = buildLoopNestRoundRobinSlotIndex(rewriter, baseLoop, 2);
  if (!slotIndex)
    return nullptr;

  Location loc = baseLoop.getLoc();
  rewriter.setInsertionPointAfter(slotIndex.getDefiningOp());
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, slotIndex,
                                        zero);
}

} // namespace pto
} // namespace mlir
