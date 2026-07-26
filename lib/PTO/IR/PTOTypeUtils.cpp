// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTOTypeUtils.h"

#include "PTO/IR/PTO.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kBitsPerByte = 8;
} // namespace

bool mlir::pto::isPTOFloat8Type(Type t) {
  return isPTOFloat8E4M3LikeType(t) || isPTOFloat8E5M2LikeType(t);
}

bool mlir::pto::isPTOFloat8E4M3LikeType(Type t) {
  return isa<Float8E4M3Type, Float8E4M3FNType, Float8E4M3FNUZType,
             Float8E4M3B11FNUZType>(t);
}

bool mlir::pto::isPTOFloat8E5M2LikeType(Type t) {
  return isa<Float8E5M2Type, Float8E5M2FNUZType>(t);
}

bool mlir::pto::isPTOHiFloat8Type(Type t) { return isa<HiF8Type>(t); }

bool mlir::pto::isPTOF8E8M0Type(Type t) { return isa<F8E8M0Type>(t); }

bool mlir::pto::isPTOHiFloat8x2Type(Type t) { return isa<HiF8x2Type>(t); }

bool mlir::pto::isPTOFloat4PackedType(Type t) {
  return isa<F4E1M2x2Type, F4E2M1x2Type>(t);
}

bool mlir::pto::isPTOPackedLdgStgVectorType(Type t) {
  // !pto.hif8x2 is a 2-byte packed hif8 value type (not a VectorType).
  if (isPTOHiFloat8x2Type(t))
    return true;
  auto vecType = dyn_cast<VectorType>(t);
  if (!vecType || vecType.isScalable() || vecType.getRank() != 1 || vecType.getDimSize(0) != 2)
    return false;
  Type elemType = vecType.getElementType();
  bool validElem =
      elemType.isF16() || elemType.isBF16() || elemType.isF32() ||
      isPTOFloat8Type(elemType);
  if (!validElem) {
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      unsigned w = intTy.getWidth();
      validElem = (w == 8 || w == 16 || w == 32);
    }
  }
  if (!validElem)
    return false;
  unsigned totalBits =
      vecType.getDimSize(0) * getPTOStorageElemBitWidth(elemType);
  return totalBits == 16 || totalBits == 32 || totalBits == 64;
}

unsigned mlir::pto::getPTOPackedLdgStgTotalBits(Type t) {
  if (isPTOHiFloat8x2Type(t))
    return getPTOStorageElemBitWidth(t); // 16
  auto vecType = cast<VectorType>(t);
  return vecType.getDimSize(0) *
         getPTOStorageElemBitWidth(vecType.getElementType());
}

bool mlir::pto::isPTOLowPrecisionType(Type t) {
  return isPTOFloat8Type(t) || isPTOHiFloat8Type(t) || isPTOF8E8M0Type(t) ||
         isPTOHiFloat8x2Type(t) || isPTOFloat4PackedType(t);
}

unsigned mlir::pto::getPTOStorageElemBitWidth(Type t) {
  if (isPTOHiFloat8x2Type(t))
    return 16;
  if (isPTOLowPrecisionType(t))
    return kBitsPerByte;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth();
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  return 0;
}

unsigned mlir::pto::getPTOStorageElemByteSize(Type t) {
  unsigned bitWidth = getPTOStorageElemBitWidth(t);
  return bitWidth == 0 ? 0 : bitWidth / kBitsPerByte;
}

namespace {

static SmallVector<int64_t, 2> getTypeValidShape(TileBufType tileType) {
  ArrayRef<int64_t> validShape = tileType.getValidShape();
  if (validShape.empty())
    validShape = tileType.getShape();
  return SmallVector<int64_t, 2>(validShape);
}

static bool isStaticShape(ArrayRef<int64_t> shape) {
  return llvm::none_of(shape, ShapedType::isDynamic);
}

static SmallVector<int64_t, 2> resolveExplicitValidShape(Value validRow,
                                                         Value validCol,
                                                         TileBufType tileType) {
  if (!validRow && !validCol)
    return getTypeValidShape(tileType);

  SmallVector<int64_t, 2> typeValidShape = getTypeValidShape(tileType);
  if (typeValidShape.size() != 2)
    return SmallVector<int64_t, 2>(typeValidShape.size(),
                                   ShapedType::kDynamic);

  auto resolveDimension = [&](Value explicitValue, unsigned dimension) {
    if (!explicitValue)
      return typeValidShape[dimension];
    return getConstantIntValue(explicitValue).value_or(ShapedType::kDynamic);
  };
  return {resolveDimension(validRow, 0), resolveDimension(validCol, 1)};
}

static SmallVector<int64_t, 2> getDynamicValidShape(TileBufType tileType) {
  return SmallVector<int64_t, 2>(getTypeValidShape(tileType).size(),
                                 ShapedType::kDynamic);
}

static SmallVector<int64_t, 2>
resolveStaticTileValidShapeImpl(Value tile, Operation *at,
                                DominanceInfo &dominance,
                                llvm::DenseSet<Value> &visiting);

/// Resolve scalar valid-shape operands produced by `pto.get_validshape` in
/// addition to ordinary index constants. This lets a layout-only treshape
/// preserve statically known valid extents when the A5 TMOV normalization
/// swaps rows and columns through get/set_validshape.
static std::optional<int64_t>
resolveStaticValidShapeDim(Value dim, DominanceInfo &dominance,
                           llvm::DenseSet<Value> &visiting) {
  if (std::optional<int64_t> constant = getConstantIntValue(dim))
    return constant;

  auto getValidShape = dim.getDefiningOp<GetValidShapeOp>();
  if (!getValidShape)
    return std::nullopt;

  SmallVector<int64_t, 2> sourceShape =
      resolveStaticTileValidShapeImpl(getValidShape.getSource(), getValidShape,
                                      dominance, visiting);
  if (sourceShape.size() != 2 || !isStaticShape(sourceShape))
    return std::nullopt;
  if (dim == getValidShape.getValidRow())
    return sourceShape[0];
  if (dim == getValidShape.getValidCol())
    return sourceShape[1];
  return std::nullopt;
}

enum class StaticValidShapeUpdate {
  None,
  Static,
  Dynamic,
};

/// set_validshape is an in-place mutation. Resolve only the latest update that
/// dominates the consuming operation; later or branch-local updates cannot
/// describe the tile state at this program point.
static StaticValidShapeUpdate
resolveStaticValidShapeUpdates(Value tile, Operation *at,
                               DominanceInfo &dominance,
                               llvm::DenseSet<Value> &visiting,
                               SmallVectorImpl<int64_t> &resolvedShape) {
  Operation *latestUpdate = nullptr;
  for (Operation *user : tile.getUsers()) {
    auto setValidShape = dyn_cast<SetValidShapeOp>(user);
    if (!setValidShape || setValidShape.getSource() != tile)
      continue;
    if (!dominance.properlyDominates(user, at)) {
      // If neither operation dominates the other, the update is branch-local
      // or otherwise control-flow dependent with respect to the consumer.
      if (!dominance.properlyDominates(at, user))
        return StaticValidShapeUpdate::Dynamic;
      continue;
    }

    if (!latestUpdate || dominance.properlyDominates(latestUpdate, user)) {
      latestUpdate = user;
      continue;
    }
    if (!dominance.properlyDominates(user, latestUpdate))
      return StaticValidShapeUpdate::Dynamic;
  }

  if (!latestUpdate)
    return StaticValidShapeUpdate::None;
  auto setValidShape = cast<SetValidShapeOp>(latestUpdate);
  std::optional<int64_t> row = resolveStaticValidShapeDim(
      setValidShape.getValidRow(), dominance, visiting);
  std::optional<int64_t> col = resolveStaticValidShapeDim(
      setValidShape.getValidCol(), dominance, visiting);
  if (!row || !col)
    return StaticValidShapeUpdate::Dynamic;
  resolvedShape.assign({*row, *col});
  return StaticValidShapeUpdate::Static;
}

static SmallVector<int64_t, 2>
resolveStaticTileValidShapeImpl(Value tile, Operation *at,
                                DominanceInfo &dominance,
                                llvm::DenseSet<Value> &visiting) {
  auto tileType = dyn_cast<TileBufType>(tile.getType());
  if (!tileType)
    return {};

  if (!visiting.insert(tile).second)
    return getDynamicValidShape(tileType);
  auto finish = [&](SmallVector<int64_t, 2> result) {
    visiting.erase(tile);
    return result;
  };

  SmallVector<int64_t, 2> updateShape;
  switch (resolveStaticValidShapeUpdates(tile, at, dominance, visiting,
                                         updateShape)) {
  case StaticValidShapeUpdate::Static:
    return finish(updateShape);
  case StaticValidShapeUpdate::Dynamic:
    return finish(getDynamicValidShape(tileType));
  case StaticValidShapeUpdate::None:
    break;
  }

  SmallVector<int64_t, 2> typeValidShape = getTypeValidShape(tileType);
  if (isStaticShape(typeValidShape))
    return finish(typeValidShape);

  if (auto result = dyn_cast<OpResult>(tile)) {
    if (auto fusionRegion =
            dyn_cast<FusionRegionOp>(result.getOwner())) {
      auto yield = dyn_cast<YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      unsigned resultIndex = result.getResultNumber();
      if (yield && resultIndex < yield.getNumOperands())
        return finish(resolveStaticTileValidShapeImpl(
            yield.getOperand(resultIndex), yield, dominance, visiting));
    }
  }

  if (auto alloc = tile.getDefiningOp<AllocTileOp>())
    return finish(resolveExplicitValidShape(alloc.getValidRow(),
                                            alloc.getValidCol(), tileType));
  if (auto bind = tile.getDefiningOp<BindTileOp>())
    return finish(resolveExplicitValidShape(bind.getValidRow(),
                                            bind.getValidCol(), tileType));
  if (auto materialize = tile.getDefiningOp<MaterializeTileOp>())
    return finish(resolveExplicitValidShape(
        materialize.getValidRow(), materialize.getValidCol(), tileType));
  if (auto subview = tile.getDefiningOp<SubViewOp>())
    return finish(resolveExplicitValidShape(subview.getValidRow(),
                                            subview.getValidCol(), tileType));
  if (auto bitcast = tile.getDefiningOp<BitcastOp>())
    return finish(resolveStaticTileValidShapeImpl(
        bitcast.getSrc(), bitcast, dominance, visiting));
  if (auto reshape = tile.getDefiningOp<TReshapeOp>()) {
    auto sourceType = dyn_cast<TileBufType>(reshape.getSrc().getType());
    SmallVector<int64_t, 2> sourceValid = resolveStaticTileValidShapeImpl(
        reshape.getSrc(), reshape, dominance, visiting);
    if (sourceType && isStaticShape(sourceValid) &&
        sourceValid == sourceType.getShape())
      return finish(SmallVector<int64_t, 2>(tileType.getShape()));
  }

  return finish(typeValidShape);
}

} // namespace

SmallVector<int64_t, 2>
mlir::pto::resolveStaticTileValidShape(Value tile, Operation *at) {
  if (!at)
    return {};
  Operation *root = at;
  while (root->getParentOp())
    root = root->getParentOp();
  DominanceInfo dominance(root);
  llvm::DenseSet<Value> visiting;
  return resolveStaticTileValidShapeImpl(tile, at, dominance, visiting);
}
