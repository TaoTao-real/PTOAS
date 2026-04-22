// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- AllocToPointerCast.cpp - convert alloc_tile to pto.pointer_cast. ---===//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"

#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

LogicalResult AllocTileOpToPointerCastOpPattern::matchAndRewrite(
    pto::AllocTileOp op, PatternRewriter &rewriter) const {
  // Manual-address alloc_tile is already fully bound and must not be remapped.
  if (op.getAddr())
    return failure();

  auto tileType = dyn_cast<pto::TileBufType>(op.getResult().getType());
  if (!tileType)
    return failure();

  // Keep config from the tile descriptor so lowering can generate the exact
  // Tile<...> type token (layout/fractal/pad) without reconstructing it later.
  TileBufConfigAttr configAttr = tileType.getConfigAttr();

  constexpr uint64_t kAlign = 4096;
  auto iter = buffer2Offsets.find(op.getResult());
  SmallVector<uint64_t> offsets;
  if (iter != buffer2Offsets.end())
    offsets = iter->second;

  // If MemPlan didn't assign an address, synthesize a unique, aligned offset so
  // downstream PointerCast lowering won't crash on empty addrs.
  if (offsets.empty()) {
    uint64_t bytes = kAlign;
    uint64_t elemBytes = 0;
    Type elemTy = tileType.getElementType();
    if (elemTy.isF16() || elemTy.isBF16())
      elemBytes = 2;
    else if (elemTy.isF32())
      elemBytes = 4;
    else if (auto it = dyn_cast<IntegerType>(elemTy))
      elemBytes = it.getWidth() / 8;

    if (elemBytes != 0) {
      uint64_t numel = 1;
      bool allStatic = true;
      for (int64_t d : tileType.getShape()) {
        if (d == ShapedType::kDynamic) {
          allStatic = false;
          break;
        }
        numel *= static_cast<uint64_t>(d);
      }
      if (allStatic && numel != 0)
        bytes = numel * elemBytes;
    }

    uint64_t stride = ((bytes + kAlign - 1) / kAlign) * kAlign;
    uint64_t off = fallbackNextOffset;
    fallbackNextOffset += std::max<uint64_t>(stride, kAlign);
    offsets.push_back(off);
  }

  SmallVector<Value> addrs;
  addrs.reserve(offsets.size());
  for (uint64_t offset : offsets) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }

  // Preserve valid-shape contract:
  // - dynamic valid dims: forward alloc_tile operands
  // - static valid dims: materialize constants from TileBufType
  // This keeps semantics identical to alloc_tile across PlanMemory rewrite.
  Value vRow = op.getValidRow();
  Value vCol = op.getValidCol();
  auto validShape = tileType.getValidShape();
  if (validShape.size() >= 2) {
    auto indexType = rewriter.getIndexType();
    Location loc = op.getLoc();
    if (!vRow && validShape[0] >= 0) {
      vRow = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[0]));
    }
    if (!vCol && validShape[1] >= 0) {
      vCol = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[1]));
    }
  }

  auto pointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), tileType, ValueRange(addrs), vRow ? vRow : Value(),
      vCol ? vCol : Value(), configAttr);

  rewriter.replaceOp(op, pointerCastOp->getResults());
  return success();
}
