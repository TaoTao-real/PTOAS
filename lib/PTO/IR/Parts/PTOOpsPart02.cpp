// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static std::optional<int64_t> getArgReductionTmpMinStride(Type elemTy,
                                                          int64_t srcValidCols) {
  if (srcValidCols == ShapedType::kDynamic || srcValidCols < 0) {
    return std::nullopt;
  }
  auto repeatElems = getVectorRepeatElements(elemTy);
  auto blockElems = getVectorBlockElements(elemTy);
  if (!repeatElems || !blockElems) {
    return std::nullopt;
  }
  int64_t repeats = ceilDivInt64(srcValidCols, *repeatElems);
  return (ceilDivInt64(repeats * 2, *blockElems) +
          ceilDivInt64(repeats, *blockElems)) *
         *blockElems;
}

static bool hasExactKnownValidShape(Type lhsTy, Type rhsTy) {
  return getValidShapeVec(lhsTy) == getValidShapeVec(rhsTy);
}

static LogicalResult verifyTColArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to have rank-2 valid_shape");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTColArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyTColArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColSumTmpStride(Operation *op, Type srcTy,
                                            Type tmpTy, bool isBinary) {
  if (!isBinary) {
    return success();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpShape.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  int64_t srcValidCols = srcValid[1];
  int64_t tmpStride = tmpShape[1];
  if (srcValidCols != ShapedType::kDynamic && tmpStride != ShapedType::kDynamic &&
      tmpStride < srcValidCols) {
    return op->emitOpError()
           << "expects tmp shape[1] to be at least src valid_shape[1] when "
              "isBinary is true; got "
           << tmpStride << " vs " << srcValidCols;
  }
  return success();
}

static LogicalResult verifyTRowArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcShape = getShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcShape.size() != 2 || tmpShape.size() != 2 || srcValid.size() != 2 ||
      tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  auto repeatElems = getVectorRepeatElements(getElemTy(srcTy));
  if (!repeatElems) {
    return op->emitOpError("failed to infer A2/A3 tmp contract from src element type");
  }

  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] <= *repeatElems) {
    auto tmpTile = dyn_cast<pto::TileBufType>(tmpTy);
    auto layout = tmpTile ? getTileBufLogicalLayout(tmpTile) : std::nullopt;
    if (layout && *layout == pto::Layout::DN) {
      if (tmpShape[1] != ShapedType::kDynamic && tmpShape[1] != 1) {
        return op->emitOpError("expects A2/A3 tmp DN layout to have shape[1] == 1");
      }
      if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] != 1) {
        return op->emitOpError(
            "expects A2/A3 tmp DN layout to have valid_shape[1] == 1");
      }
      if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
          tmpValid[0] < srcValid[0] * 2) {
        return op->emitOpError()
               << "expects A2/A3 tmp DN layout to have valid_shape[0] >= "
               << (srcValid[0] * 2);
      }
      return verifyTmpCapacityAtLeast(op, tmpTy, 32);
    }

    if (!layout || *layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects A2/A3 tmp to use DN 1-col or ND 2-col layout when src valid_shape[1] fits in one repeat");
    }
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
        tmpValid[0] < srcValid[0]) {
      return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < 2) {
      return op->emitOpError(
          "expects A2/A3 tmp valid_shape[1] to be at least 2 in the small-col ND path");
    }
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (srcShape[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] != srcShape[0]) {
    return op->emitOpError("expects A2/A3 tmp shape[0] to match src shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
      tmpValid[0] < srcValid[0]) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTRowArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyTRowArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, ty, name, allowLowPrecision))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    }
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
    }
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
  // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
  // first, so they stay strict without needing a flag here (unlike the row
  // path, whose dst-layout check does not constrain valid).
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    }
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
    }
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name))) {
    return failure();
  }
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2) {
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  }
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1) {
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  }
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

LogicalResult mlir::pto::SectionSimtOp::verify() {
  func::FuncOp func = getOperation()->getParentOfType<func::FuncOp>();
  if (!func) {
    return emitOpError("must be nested under a func.func");
  }

  if (getDimXAttr().getInt() < 0 || getDimYAttr().getInt() < 0 ||
      getDimZAttr().getInt() < 0) {
    return emitOpError("requires non-negative i32 launch dimensions");
  }

  if (func->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return emitOpError("must not appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "'";
  }

  WalkResult nested = getBody().walk([&](SectionSimtOp nestedOp) {
    nestedOp.emitOpError("nested pto.section.simt is not allowed");
    return WalkResult::interrupt();
  });
  if (nested.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult mlir::pto::FusionRegionOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>()) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  if (yield.getValues().size() != getOutputs().size()) {
    return emitOpError() << "expects pto.yield to return "
                         << getOutputs().size() << " values, got "
                         << yield.getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

LogicalResult mlir::pto::YieldOp::verify() {
  auto parent = dyn_cast_or_null<FusionRegionOp>(getOperation()->getParentOp());
  if (!parent) {
    return emitOpError("expects parent op to be pto.fusion_region");
  }

  if (getValues().size() != parent.getOutputs().size()) {
    return emitOpError() << "expects " << parent.getOutputs().size()
                         << " yielded values to match parent results, got "
                         << getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getValues(), parent.getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy) {
    return emitOpError("result must be pto.tensor_view<...>");
  }

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  Type ptrElemTy = ptrTy.getElementType();

  if (ptrElemTy != tvTy.getElementType()) {
    return emitOpError() << "ptr element type must match tensor_view element "
                            "type, but got ptr="
                         << ptrElemTy << " view=" << tvTy.getElementType();
  }

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank) {
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;
  }

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy) {
    return emitOpError("expects tensor_view source and partition_tensor_view result");
  }

  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();
  }

  int64_t srcRank = srcTy.getRank();
  if ((int64_t)getOffsets().size() != srcRank) {
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";
  }

  if ((int64_t)getSizes().size() != srcRank) {
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";
  }

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;

  for (int64_t i = 0; i < srcRank; ++i) {
    auto offVal = getConstIndexValue(getOffsets()[i]);
    auto sizeVal = getConstIndexValue(getSizes()[i]);

    if (offVal && *offVal < 0) {
      return emitOpError() << "offset at dim " << i
                           << " must be non-negative, got " << *offVal;
    }

    if (sizeVal && *sizeVal <= 0) {
      return emitOpError() << "size at dim " << i
                           << " must be positive, got " << *sizeVal;
    }

    if (sameRank && sizeVal) {
      int64_t resDim = resShape[i];
      if (resDim != ShapedType::kDynamic && *sizeVal != resDim) {
        return emitOpError() << "size/result mismatch at dim " << i
                             << ": size operand=" << *sizeVal
                             << " result type dim=" << resDim;
      }
    }

    int64_t srcDim = srcShape[i];
    if (srcDim == ShapedType::kDynamic) {
      continue;
    }

    if (sizeVal && *sizeVal > srcDim) {
      return emitOpError() << "size at dim " << i << " (" << *sizeVal
                           << ") exceeds static source dim (" << srcDim << ")";
    }

    if (offVal && sizeVal && (*offVal + *sizeVal > srcDim)) {
      return emitOpError() << "offset+size at dim " << i << " ("
                           << (*offVal + *sizeVal)
                           << ") exceeds static source dim (" << srcDim << ")";
    }
  }

  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  if (ptrTy != resTy) {
    return emitOpError("result type must match ptr operand type");
  }

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrTy.getElementType();
  }
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type) {
    return false;
  }
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64()) {
    return true;
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  }
  if (mlir::pto::isPTOFloat8Type(type)) {
    return true;
  }
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type)) {
    return true;
  }
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64) {
    return emitOpError("result must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getPtr().getType())) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  return success();
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64) {
    return emitOpError("address operand must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getResult().getType())) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem)) {
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;
  }

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getResult().getType() != arrayTy.getElementType()) {
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  }
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getValue().getType() != arrayTy.getElementType()) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  }
  return success();
}

// Resolve the field type reached by following a constant `path` of field
// indices from `root`, descending through nested structs. Emits an actionable
// op error and returns failure on an empty path, an out-of-range index, or a
// descent into a non-struct field. On success writes the terminal field type to
// `fieldTyOut`.
static LogicalResult walkStructPath(Operation *op, mlir::pto::StructType root,
                                    llvm::ArrayRef<int64_t> path,
                                    Type &fieldTyOut) {
  if (path.empty()) {
    return op->emitOpError() << "struct path must have at least one index";
  }
  Type cur = root;
  for (auto [depth, idx] : llvm::enumerate(path)) {
    auto st = dyn_cast<mlir::pto::StructType>(cur);
    if (!st) {
      return op->emitOpError()
             << "struct path index " << depth
             << " descends into non-struct field of type " << cur;
    }
    if (idx < 0 || idx >= static_cast<int64_t>(st.getNumFields())) {
      return op->emitOpError()
             << "struct path index " << depth << " (" << idx
             << ") is out of range for " << st << " with " << st.getNumFields()
             << " field(s)";
    }
    cur = st.getFieldType(static_cast<unsigned>(idx));
  }
  fieldTyOut = cur;
  return success();
}

// The declared struct is stack storage owned by the enclosing scope, and the
// value lowers to a pointer to that storage. Letting it reach a terminator
// would publish that address outside the owning scope: `return %s` hands the
// caller a pointer into a dead frame, and `scf.yield %s` carries it out of the
// region that owns it. Both are rejected here rather than emitted as C++ that
// looks fine and is undefined at run time.
LogicalResult mlir::pto::DeclareStructOp::verify() {
  for (Operation *user : getResult().getUsers()) {
    if (!user->hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    return emitOpError()
           << "stack-local struct must not escape the scope that declares it, "
              "but its value is passed to '"
           << user->getName()
           << "', which would expose the address of storage that is about to "
              "die; declare the struct in the outer scope and mutate it from "
              "the nested region instead (pto.struct_set mutates in place, "
              "so a struct never needs to be returned or yielded)";
  }
  return success();
}

// Both accessors bottom out at a scalar. A path ending on a nested !pto.struct
// is rejected: the member chain lowers to `emitc.member`, which yields an
// lvalue, and handing a whole aggregate back as an SSA value would mean copying
// it out of the struct — so reaching into a nested struct is spelled as a longer
// path instead.
static LogicalResult verifyStructLeafIsScalar(Operation *op, Type fieldTy) {
  if (!fieldTy.isIntOrFloat()) {
    return op->emitOpError()
           << "struct path must end at a scalar field, but ends at " << fieldTy
           << "; extend the path to reach a scalar inside it";
  }
  return success();
}

LogicalResult mlir::pto::StructGetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(
          getOperation(),
          cast<mlir::pto::StructType>(getOperation()->getOperand(0).getType()),
          getPath(), fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "result type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::StructSetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(
          getOperation(),
          cast<mlir::pto::StructType>(getOperation()->getOperand(0).getType()),
          getPath(), fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::CastPtrOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
  auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
  auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
  bool inputIsInteger = isa<IntegerType>(inputType);
  bool resultIsInteger = isa<IntegerType>(resultType);

  if (!inputPtrType && !inputMemRefType && !inputIsInteger) {
    return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
  }
  if (!resultPtrType && !resultIsInteger) {
    return emitOpError("result must be an integer or !pto.ptr<...>");
  }

  if (inputIsInteger && resultIsInteger) {
    return emitOpError("integer-to-integer cast is not a ptr cast");
  }

  if (inputMemRefType && resultIsInteger) {
    return emitOpError("memref-to-integer cast is unsupported");
  }

  if (inputMemRefType && resultPtrType) {
    auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(
        inputMemRefType.getMemorySpace());
    auto resultSpace = resultPtrType.getMemorySpace();
    if (memrefSpace && memrefSpace != resultSpace) {
      return emitOpError(
          "memref-to-ptr cast must stay within the same PTO memory space");
    }
  }

  if (inputPtrType && resultPtrType &&
      inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
    return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
  }

  return success();
}




void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
      >();

  addInterfaces<PTOInlinerInterface>();
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace();
  }
  return {};
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  if (isPTOEntryFunction(func)) {
    return true;
  }
  auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
  if (!attr) {
    return false;
  }
  return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func,
                                              bool isExternal) {
  if (!func) {
    return;
  }
  if (isExternal) {
    func->setAttr(kPTOVisibilityAttrName,
                  StringAttr::get(func.getContext(),
                                  kPTOVisibilityExternalValue));
    return;
  }
  func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module) {
    return success();
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func)) {
      continue;
    }
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func)) {
      continue;
    }
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

// A !pto.struct is represented as a pointer to stack storage. Its provenance
// must therefore remain explicit: the value comes directly from
// pto.declare_struct in the owning function. Function arguments/results and
// operations such as arith.select and scf.if must not manufacture or relay a
// struct-typed SSA value, because that alias hides the declaration from
// DeclareStructOp's direct-use escape check. CFG block arguments cannot make a
// declaration safe to forward either: the branch is a terminator and is
// rejected by DeclareStructOp::verify.
LogicalResult mlir::pto::validateStructProvenance(ModuleOp module) {
  if (!module) {
    return success();
  }

  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      for (auto [i, inputTy] :
           llvm::enumerate(func.getFunctionType().getInputs())) {
        if (!isa<StructType>(inputTy)) {
          continue;
        }
        func.emitOpError()
            << "argument " << i << " has type " << inputTy
            << ", but a stack-local struct must not be a function argument; "
               "structs must originate from 'pto.declare_struct' in the same "
               "function";
        return WalkResult::interrupt();
      }
      for (auto [i, resultTy] :
           llvm::enumerate(func.getFunctionType().getResults())) {
        if (!isa<StructType>(resultTy)) {
          continue;
        }
        func.emitOpError()
            << "result " << i << " has type " << resultTy
            << ", but a stack-local struct must not be returned: the value is "
               "a pointer into the callee's frame, and returning it (even "
               "when it merely passes an argument back through) launders its "
               "provenance; keep the struct in its declaring function "
               "(pto.struct_set mutates in place, so a result is never needed)";
        return WalkResult::interrupt();
      }
    }

    if (!isa<DeclareStructOp>(op)) {
      for (auto [i, opResult] : llvm::enumerate(op->getResults())) {
        if (!isa<StructType>(opResult.getType())) {
          continue;
        }
        op->emitOpError()
            << "result " << i << " has type " << opResult.getType()
            << ", but only 'pto.declare_struct' may produce a !pto.struct "
               "result; derived results hide the stack-storage lifetime and "
               "can escape their declaring scope";
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  (void)module;
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//===----------------------------------------------------------------------===//

static std::optional<uint64_t>
getLocalAddressAlignmentBytes(Attribute memorySpace) {
  auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  if (!addrSpace) {
    return std::nullopt;
  }

  // Keep this verifier as a conservative front-line guard for explicit local
  // tile addresses. PTO-ISA's buffer_limits.hpp defines the baseline
  // TASSIGN<Addr> alignment as 32 bytes for local tile memories. For L0 tile
  // bases, PTOAS level3/manual IR historically uses a 4096-bit (512-byte)
  // granularity; fuller per-arch/per-layout bounds checks belong in PTO-ISA.
  switch (addrSpace.getAddressSpace()) {
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return 32;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
    return 512;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return std::nullopt;
  }
  return std::nullopt;
}

static LogicalResult verifyConstantLocalAddress(Operation *op, Value addr,
                                                Attribute memorySpace,
                                                int addrIndex = -1) {
  std::optional<uint64_t> alignment =
      getLocalAddressAlignmentBytes(memorySpace);
  if (!alignment || *alignment == 0) {
    return success();
  }

  std::optional<int64_t> constantAddr = mlir::getConstantIntValue(addr);
  if (!constantAddr) {
    return success();
  }

  auto emitAddrError = [&]() {
    InFlightDiagnostic diag = op->emitOpError();
    if (addrIndex >= 0) {
      diag << "addr[" << addrIndex << "]";
    } else {
      diag << "addr";
}
    return diag;
  };

  if (*constantAddr < 0) {
    return emitAddrError() << " must be non-negative, got " << *constantAddr;
  }

  uint64_t unsignedAddr = static_cast<uint64_t>(*constantAddr);
  if ((unsignedAddr % *alignment) != 0) {
    return emitAddrError()
           << " must be aligned to " << *alignment
           << " bytes for local tile memory space, got " << unsignedAddr;
  }

  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        ty.getMemorySpace()))) {
    return failure();
  }

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("result tile_buf must have rank-2 validShape");
  }

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }

  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocMultiTileOp / MultiTileGetOp
//===----------------------------------------------------------------------===//

LogicalResult AllocMultiTileOp::verify() {
  auto mtbTy = getResult().getType();
  if (!mtbTy) {
    return emitOpError("result must be `!pto.multi_tile_buf`");
  }

  TileBufType slotTy = mtbTy.getSlotType();
  if (!slotTy) {
    return emitOpError("multi_tile_buf slot type must be non-null");
  }

  // Reuse the AllocTileOp valid_row/valid_col contract on the slot type.
  Type elemTy = slotTy.getElementType();
  if (isPTOLowPrecisionType(elemTy)) {
    return emitOpError() << "slot dtype " << elemTy
                         << " is not supported by pto.alloc_multi_tile yet";
  }

  if (failed(verifyTileBufLayoutConstraints(*this, slotTy, "slot"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        slotTy.getMemorySpace()))) {
    return failure();
  }

  // Multi-buffer slots are placed at product(shape) * element_size byte
  // intervals -- both level3 validation and PTOPlanMemory size them that way.
  // `row_plus_one` compaction inflates the
  // major stride by one element per row, so the slot's physical strided
  // footprint exceeds product(shape) and adjacent slots would silently overlap
  // (data corruption). Reject it until the slot stride is derived from the true
  // strided footprint. Non-boxed compact/`normal` and boxed fractal slayouts
  // pack densely (footprint == product(shape)), so they stay supported.
  if (slotTy.getCompactModeI32() ==
      static_cast<int32_t>(mlir::pto::CompactMode::RowPlusOne)) {
    return emitOpError()
           << "multi_tile_buf slot uses row_plus_one compaction, whose padded "
              "storage footprint exceeds product(shape) and would overlap "
              "adjacent multi-buffer slots; use a compact (non-row_plus_one) "
              "slot layout or a single pto.alloc_tile";
  }

  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;
  auto vs = slotTy.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("slot tile_buf must have rank-2 validShape");
  }

  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because slot v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }
  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because slot v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  // Count bounds are also enforced by MultiTileBufType::verify, but repeat
  // here so the error points at the alloc op the user wrote.
  uint32_t count = mtbTy.getCount();
  if (count < kPtoMultiBufferMinNum || count > kPtoMultiBufferMaxNum) {
    return emitOpError() << "multi_tile_buf count must be in ["
                         << kPtoMultiBufferMinNum << ", "
                         << kPtoMultiBufferMaxNum << "] (got " << count << ")";
  }

  if (Attribute rawAddrs = (*this)->getAttr(kPtoMultiBufferAddrsAttrName)) {
    auto addrs = dyn_cast<DenseI64ArrayAttr>(rawAddrs);
    if (!addrs) {
      return emitOpError() << "expects internal '"
                           << kPtoMultiBufferAddrsAttrName
                           << "' to be a dense i64 array";
    }
    if (getAddr()) {
      return emitOpError() << "cannot carry both base 'addr' and internal '"
                           << kPtoMultiBufferAddrsAttrName << "'";
    }
    if (addrs.size() != count) {
      return emitOpError() << "expects " << count << " planned slot addresses, got "
                           << addrs.size();
    }

    uint64_t elemBytes = getPTOStorageElemByteSize(slotTy.getElementType());
    uint64_t slotBytes = elemBytes;
    for (int64_t dim : slotTy.getShape()) {
      if (dim == ShapedType::kDynamic) {
        return emitOpError(
            "planned multi-buffer addresses require a static slot shape");
      }
      slotBytes *= static_cast<uint64_t>(dim);
    }
    for (auto [lhsIdx, lhs] : llvm::enumerate(addrs.asArrayRef())) {
      if (lhs < 0) {
        return emitOpError("planned slot addresses must be non-negative");
      }
      uint64_t lhsBegin = static_cast<uint64_t>(lhs);
      uint64_t lhsEnd = lhsBegin + slotBytes;
      for (size_t rhsIdx = lhsIdx + 1;
           rhsIdx < static_cast<size_t>(addrs.size()); ++rhsIdx) {
        uint64_t rhsBegin = static_cast<uint64_t>(addrs[rhsIdx]);
        uint64_t rhsEnd = rhsBegin + slotBytes;
        if (std::max(lhsBegin, rhsBegin) < std::min(lhsEnd, rhsEnd)) {
          return emitOpError() << "planned slots " << lhsIdx << " and "
                               << rhsIdx << " overlap";
        }
      }
    }
  }

  return success();
}

LogicalResult MultiTileGetOp::verify() {
  auto srcTy = getSource().getType();
  auto resultTy = getResult().getType();
  if (!srcTy || !resultTy) {
    return emitOpError("source and result types must be non-null");
  }

  if (srcTy.getSlotType() != resultTy) {
    return emitOpError()
           << "result tile_buf must match the multi_tile_buf slot type: "
           << "expected " << srcTy.getSlotType() << ", got " << resultTy;
  }

  // If slot is an `arith.constant`, check it is in range.
  if (auto slotDef = getSlot().getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<IntegerAttr>(slotDef.getValue())) {
      int64_t slotVal = attr.getValue().getSExtValue();
      int64_t count = static_cast<int64_t>(srcTy.getCount());
      if (slotVal < 0 || slotVal >= count) {
        return emitOpError()
               << "constant slot " << slotVal
               << " is out of range for multi_tile_buf count=" << count;
      }
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }

  auto tileTy = dyn_cast<TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError("expects tile operand and result to be !pto.tile_buf");
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        tileTy.getMemorySpace()))) {
    return failure();
  }

  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::PartitionTensorViewType, pto::TileBufType>> {
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
    auto dstTile = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!srcPart || !dstTile) {
      emitOpError("expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision))) {
      return failure();
    }

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        emitOpError() << "expects src shape[" << i << "] to be positive";
        return failure();
      }
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        emitOpError() << "expects dst valid_shape[" << i << "] to be non-negative";
        return failure();
      }
    }
    return std::make_pair(srcPart, dstTile);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common)) {
      return failure();
    }
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tload low-precision element types to be unsupported");
    }
    if (!(dstElem.isInteger(8) || dstElem.isInteger(16) || dstElem.isInteger(32) ||
          dstElem.isInteger(64) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())) {
      return emitOpError("expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
    }

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
      return emitOpError("expects src and dst element types to have the same bitwidth");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common)) {
      return failure();
    }
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    unsigned srcBytes = getElemByteSize(srcElem);
    unsigned dstBytes = getElemByteSize(dstElem);
    if (srcBytes != dstBytes) {
      return emitOpError("expects src and dst element types to have the same element size");
    }
    if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8)) {
      return emitOpError("expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
    }
    if (!isA5TLoadStoreTransferElemType(srcElem)) {
      return emitOpError("expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    if (!isA5TLoadStoreTransferElemType(dstElem)) {
      return emitOpError("expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }

    if (dstElem.isInteger(64)) {
      auto pad = dstTile.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero)) {
        return emitOpError("expects A5 i64/u64 tload dst pad to be null or zero");
      }
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (dstSpace && *dstSpace == pto::AddressSpace::VEC) {
      int32_t bl = dstTile.getBLayoutValueI32();
      int32_t sl = dstTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      if (!isND && !isDN && !isNZ) {
        return emitOpError("expects A5 tload vec dst layout to be ND, DN, or NZ");
      }
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyImpl = [&](bool allowLowPrecision) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();

    Type srcElem;
    Type dstElem;

    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy);
    if (!srcPart) {
      return emitOpError("expects src to be !pto.partition_tensor_view");
    }
    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        return emitOpError() << "expects src shape[" << i << "] to be positive";
      }
    }
    srcElem = srcPart.getElementType();

    auto dstTile = dyn_cast<pto::TileBufType>(dstTy);
    if (!dstTile) {
      return emitOpError("expects dst to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision))) {
      return failure();
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        return emitOpError()
               << "expects dst valid_shape[" << i << "] to be non-negative";
      }
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to use loc=vec or loc=mat");
    }
    dstElem = dstTile.getElementType();

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
      return emitOpError("expects src and dst element types to have the same element size");
    }
    if (!allowLowPrecision &&
        (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))) {
      return emitOpError("expects A2/A3 tprefetch low-precision element types to be unsupported");
    }
    if (allowLowPrecision &&
        (!isA5TLoadStoreTransferElemType(srcElem) ||
         !isA5TLoadStoreTransferElemType(dstElem))) {
      return emitOpError("expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/true);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  auto ptrTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!ptrTy) {
    return emitOpError("expects workspace to be !pto.ptr<i8>");
  }
  Type elemTy = ptrTy.getElementType();
  if (!isByteIntegerType(elemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src"))) {
    return failure();
  }
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto ptrTy = llvm::dyn_cast<mlir::pto::PtrType>(getFfts().getType());
  if (!ptrTy) {
    return emitOpError("expects a !pto.ptr operand");
  }

  if (!ptrTy.getElementType().isInteger(64) &&
      !ptrTy.getElementType().isInteger(8)) {
    return emitOpError("expects element type i64 (or i8)");
  }

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic) {
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2) {
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
    }
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    if (IntegerAttr eventIdAttr = getEventIdAttr()) {
      int64_t eventId = eventIdAttr.getInt();
      if (eventId < 0 || eventId > 15) {
        return emitOpError()
               << "A5 sync.set expects static FFTS event_id in [0, 15], but got "
               << eventId;
      }
    }
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.set expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<Type, 2> operandTypes;
  Attribute modeAttr;
  Attribute coreTypeAttr;

  if (parser.parseLParen()) {
    return failure();
  }

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) || parser.parseColonTypeList(operandTypes) ||
        parser.parseRParen()) {
      return failure();
    }
    if (operands.size() != operandTypes.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
    }
  }

  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr)) {
    return failure();
  }

  auto mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";
  }

  auto coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";
  }

  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  auto addSegmentSizes = [&](int32_t gm, int32_t used) {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({gm, used}));
  };

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    }
    addSegmentSizes(0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  if (operands.size() != 1 && operands.size() != 2) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects soft syncall to have gm_workspace and optional "
              "used_cores";
  }
  if (parser.resolveOperand(operands[0], operandTypes[0], result.operands)) {
    return failure();
  }
  if (operands.size() == 2 &&
      parser.resolveOperand(operands[1], operandTypes[1], result.operands)) {
    return failure();
  }
  addSegmentSizes(1, operands.size() == 2 ? 1 : 0);
  return success();
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 2> operands;
  if (getGmWorkspace()) {
    operands.push_back(getGmWorkspace());
  }
  if (getUsedCores()) {
    operands.push_back(getUsedCores());
  }

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}
