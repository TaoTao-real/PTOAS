// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy))) {
    return failure();
  }
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyA5MxMatTileOperands(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true))) {
    return failure();
  }

  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  if (lhsShape.size() == 2 && rhsShape.size() == 2) {
    int64_t lhsK = lhsShape[1];
    int64_t rhsK = rhsShape[0];
    auto checkPhysicalK = [&](int64_t value, StringRef name) -> LogicalResult {
      if (value != ShapedType::kDynamic && (value < 1 || (value % 64) != 0)) {
        return op->emitOpError() << "expects " << name
                                 << " physical K shape to be a positive multiple of 64 on A5";
      }
      return success();
    };
    if (failed(checkPhysicalK(lhsK, "lhs"))) {
      return failure();
    }
    if (failed(checkPhysicalK(rhsK, "rhs"))) {
      return failure();
    }
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }
  return success();
}

static int64_t ceilDivKnown(int64_t value, int64_t divisor) {
  if (value == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return (value + divisor - 1) / divisor;
}

static LogicalResult verifyA5MxMatScaleTile(Operation *op, Type scaleTy,
                                            Type lhsTy, Type rhsTy,
                                            StringRef scaleName,
                                            bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING) {
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";
  }

  auto checkDims = [&](ArrayRef<int64_t> scaleDims, ArrayRef<int64_t> lhsDims,
                       ArrayRef<int64_t> rhsDims, StringRef dimsName) -> LogicalResult {
    if (scaleDims.size() != 2 || lhsDims.size() != 2 || rhsDims.size() != 2) {
      return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 "
                               << dimsName;
    }

    int64_t m = lhsDims[0];
    int64_t k = lhsDims[1];
    int64_t n = rhsDims[1];
    int64_t scaleK = ceilDivKnown(k, 32);
    int64_t expectedRows = isLeftScale ? m : scaleK;
    int64_t expectedCols = isLeftScale ? scaleK : n;
    if (!hasCompatibleKnownExtent(scaleDims[0], expectedRows) ||
        !hasCompatibleKnownExtent(scaleDims[1], expectedCols)) {
      return op->emitOpError()
             << "expects " << scaleName << " " << dimsName << " to be "
             << (isLeftScale ? "[M, ceil(K/32)]" : "[ceil(K/32), N]");
    }
    return success();
  };

  if (failed(checkDims(getShapeVec(scaleTy), getShapeVec(lhsTy), getShapeVec(rhsTy),
                       "shape"))) {
    return failure();
  }
  if (failed(checkDims(getValidShapeVec(scaleTy), getValidShapeVec(lhsTy),
                       getValidShapeVec(rhsTy), "valid_shape"))) {
    return failure();
  }

  auto scaleTb = dyn_cast<pto::TileBufType>(scaleTy);
  if (!scaleTb) {
    return success();
  }
  if (scaleTb.getBLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::BLayout::RowMajor
                                       : pto::BLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " blayout on A5";
  }
  if (scaleTb.getSLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::SLayout::RowMajor
                                       : pto::SLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " slayout on A5";
  }
  if (scaleTb.getSFractalSizeI32() != 32) {
    return op->emitOpError() << "expects " << scaleName
                             << " to use fractal=32 on A5";
  }
  return success();
}

static LogicalResult verifyA5MxMatScaleTiles(Operation *op, Type lhsScaleTy,
                                             Type rhsScaleTy, Type lhsTy,
                                             Type rhsTy) {
  if (failed(verifyA5MxMatScaleTile(op, lhsScaleTy, lhsTy, rhsTy, "a_scale",
                                    /*isLeftScale=*/true))) {
    return failure();
  }
  return verifyA5MxMatScaleTile(op, rhsScaleTy, lhsTy, rhsTy, "b_scale",
                                /*isLeftScale=*/false);
}

static LogicalResult verifyA5MxGemvTileOperands(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", /*allowLowPrecision=*/true)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }

  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] ||
       lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }

  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1) {
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  }
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] != 1) {
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  }
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0]) {
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  }
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  }

  auto lhsTb = dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyA5MxGemvScaleTile(Operation *op, Type scaleTy,
                                             Type lhsTy, Type rhsTy,
                                             StringRef scaleName,
                                             bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING) {
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";
  }

  auto scaleShape = getShapeVec(scaleTy);
  auto scaleValid = getValidShapeVec(scaleTy);
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (scaleShape.size() != 2 || scaleValid.size() != 2 ||
      lhsShape.size() != 2 || rhsShape.size() != 2 || lhsValid.size() != 2 ||
      rhsValid.size() != 2) {
    return op->emitOpError() << "expects " << scaleName
                             << ", lhs, and rhs to have rank-2 shape/valid_shape";
  }

  int64_t logicalM = lhsValid[0];
  int64_t logicalK = lhsValid[1];
  int64_t logicalN = rhsValid[1];
  int64_t scaleK = ceilDivKnown(logicalK, 32);

  int64_t expectedShapeRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedShapeCols = isLeftScale ? scaleK : rhsShape[1];
  int64_t expectedValidRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedValidCols = isLeftScale ? scaleK : logicalN;

  if (!hasCompatibleKnownExtent(scaleShape[0], expectedShapeRows) ||
      !hasCompatibleKnownExtent(scaleShape[1], expectedShapeCols) ||
      !hasCompatibleKnownExtent(scaleValid[0], expectedValidRows) ||
      !hasCompatibleKnownExtent(scaleValid[1], expectedValidCols)) {
    if (isLeftScale) {
      return op->emitOpError()
             << "expects " << scaleName
             << " shape/valid_shape to be [M, ceil(K/32)]";
    }
    return op->emitOpError()
           << "expects " << scaleName
           << " shape/valid_shape to be [ceil(K/32), aligned_N]/[ceil(K/32), N]";
  }
  return success();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias"))) {
    return failure();
  }
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS) {
    return op->emitOpError("expects bias to be in the bias address space");
  }
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1) {
    return op->emitOpError("expects bias to have 1 row");
  }
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32()) {
      return op->emitOpError("expects bias to have element type f32");
    }
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias))) {
    return failure();
  }
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError("expects bias to use the row_major blayout on A5");
    }
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(32) && isInt8(lhsElemTy) && isInt8(rhsElemTy)) {
    return success();
  }

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy)) {
    return success();
  }

  auto isA5TMatmulFp8Type = [](Type ty) {
    return isPTOFloat8Type(ty);
  };
  if (isA5 && dstElemTy.isF32()) {
    if (isA5TMatmulFp8Type(lhsElemTy) && isA5TMatmulFp8Type(rhsElemTy)) {
      return success();
    }
    if (isPTOHiFloat8Type(lhsElemTy) && lhsElemTy == rhsElemTy) {
      return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", (f32, fp8, fp8), or (f32, hif8, hif8)" : "");
}

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    Type elemTy = *elemOr;
    if (elemTy.isInteger(16) || elemTy.isF16() || elemTy.isF32()) {
      return success();
    }
    return emitOpError("expects element type to be i16/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return emitOpError("taddrelu is only supported on A2/A3 targets");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddCOp::verify() {
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();

  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/src2/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd) {
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  }
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

LogicalResult pto::TAxpyOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type scalarTy = getScalar().getType();
    Type srcElem = getElemTy(srcTy);
    if (scalarTy != srcElem) {
      return emitOpError("expects scalar type to match src element type");
    }
    if (getShapeVec(srcTy) != getShapeVec(dstTy)) {
      return emitOpError("expects src and dst to have the same shape");
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32())) {
      return emitOpError("expects A2/A3 taxpy dst element type to be f16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32())) {
      return emitOpError("expects A2/A3 taxpy src element type to be f16/f32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32() || dstElem.isBF16())) {
      return emitOpError("expects A5 taxpy dst element type to be f16/bf16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isBF16())) {
      return emitOpError("expects A5 taxpy src element type to be f16/bf16/f32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    return emitOpError("expects src0/src1/dst to have the same shape");
  }
  return success();
}

LogicalResult pto::TAndOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tand src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tand src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }

  auto v0 = getValidShapeVec(op.getSrc0());
  auto v1 = getValidShapeVec(op.getSrc1());
  auto vd = getValidShapeVec(op.getDst());
  if (v0.size() != 2 || v1.size() != 2 || vd.size() != 2) {
    return op.emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  // validRow must match dst (when known).
  if (v0[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v0[0] != vd[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (v1[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v1[0] != vd[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }

  // Total valid columns must fit within dst static cols (when known).
  auto sd = getShapeVec(td);
  if (sd.size() == 2 && sd[1] != ShapedType::kDynamic &&
      v0[1] != ShapedType::kDynamic && v1[1] != ShapedType::kDynamic) {
    if (v0[1] + v1[1] > sd[1]) {
      return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
    }
  }

  return e0;
}

static LogicalResult verifyTConcatElemType(TConcatOp op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16()) {
    return success();
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it ||
      (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
    return op.emitOpError("expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatLocVec(TConcatOp op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op.emitOpError() << "expects " << name << " to use loc=vec";
  }
  return success();
}

static LogicalResult verifyTConcatA2A3(TConcatOp op) {
  FailureOr<Type> elemOr = verifyTConcatCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst"))) {
    return failure();
  }
  return verifyTConcatElemType(op, *elemOr);
}

static LogicalResult verifyTConcatA5(TConcatOp op) {
  FailureOr<Type> elemOr = verifyTConcatCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(op.getSrc0().getType()) || !isRowMajorTileBuf(op.getSrc1().getType()) ||
      !isRowMajorTileBuf(op.getDst().getType())) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return verifyTConcatElemType(op, *elemOr);
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTConcatA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTConcatA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type ti0 = getSrc0Idx().getType();
    Type ti1 = getSrc1Idx().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, ti0, "src0Idx")) ||
        failed(verifyTileBufCommon(*this, ti1, "src1Idx")) ||
        failed(verifyTileBufCommon(*this, td, "dst"))) {
      return failure();
    }

    // Check data element type consistency.
    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for data operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    // Check index element type consistency.
    Type ei0 = getElemTy(ti0);
    Type ei1 = getElemTy(ti1);
    if (!ei0 || !ei1) {
      emitOpError("failed to get element type for index operands");
      return failure();
    }
    if (ei0 != ei1) {
      emitOpError("expects src0Idx and src1Idx to have the same element type");
      return failure();
    }

    // All five tiles must be rank-2.
    auto v0  = getValidShapeVec(getSrc0());
    auto v1  = getValidShapeVec(getSrc1());
    auto vi0 = getValidShapeVec(getSrc0Idx());
    auto vi1 = getValidShapeVec(getSrc1Idx());
    auto vd  = getValidShapeVec(getDst());
    if (v0.size() != 2 || v1.size() != 2 || vi0.size() != 2 ||
        vi1.size() != 2 || vd.size() != 2) {
      return emitOpError("expects all operands to have rank-2 valid_shape");
    }

    // validRow must match dst (when known).
    auto checkValidRow = [&](const auto &v, StringRef name) -> LogicalResult {
      if (v[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
          v[0] != vd[0]) {
        return emitOpError("expects ") << name << " valid row to match dst valid row";
      }
      return success();
    };
    if (failed(checkValidRow(v0, "src0")) ||
        failed(checkValidRow(v1, "src1")) ||
        failed(checkValidRow(vi0, "src0Idx")) ||
        failed(checkValidRow(vi1, "src1Idx"))) {
      return failure();
    }

    // Index tile must have cols >= 1 (when known).
    if (vi0[1] != ShapedType::kDynamic && vi0[1] < 1) {
      return emitOpError("expects src0Idx valid_col >= 1");
    }
    if (vi1[1] != ShapedType::kDynamic && vi1[1] < 1) {
      return emitOpError("expects src1Idx valid_col >= 1");
    }

    return std::make_pair(e0, ei0);
  };

  auto verifyElementTypes = [&](Type dataElem, Type idxElem) -> LogicalResult {
    // Data element type: f16, f32, bf16, i8, i16, i32 (signless).
    if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
      auto it = mlir::dyn_cast<IntegerType>(dataElem);
      if (!it || !it.isSignless() ||
          (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
        return emitOpError()
               << "expects data element type to be i8, i16, i32, f16, f32, or bf16";
      }
    }

    // Index element type: i8, i16, i32 (signless).
    auto it = mlir::dyn_cast<IntegerType>(idxElem);
    if (!it || !it.isSignless() ||
        (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError()
             << "expects index element type to be i8, i16, or i32";
    }
    return success();
  };

  auto verifyLocVec = [&](Type ty, StringRef name) -> LogicalResult {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC) {
      return emitOpError() << "expects " << name << " to use loc=vec";
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getSrc0Idx().getType()) ||
        !isRowMajorTileBuf(getSrc1Idx().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError(
          "expects all operands to use row-major layout");
    }
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16)) {
      return emitOpError(
          "expects A2/A3 tands src, scalar, and dst element type to be i8/i16");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tands src, scalar, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static ParseResult parseTCILikeOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand s, tmp, dst;
  Type sTy, tmpTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(s)) {
    return failure();
  }

  bool hasTmp = succeeded(parser.parseOptionalComma());
  if (hasTmp && parser.parseOperand(tmp)) {
    return failure();
  }

  if (parser.parseColonType(sTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(s, sTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printTCILikeOp(OpAsmPrinter &p, Operation *op, Value s, Value tmp,
                           Value dst) {
  p << " ins(" << s;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << s.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TCIOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTCILikeOp(parser, result);
}

void mlir::pto::TCIOp::print(OpAsmPrinter &p) {
  printTCILikeOp(p, getOperation(), getOperand(0), getTmp(), getDst());
}

LogicalResult pto::TCIOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }

  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy) {
    return emitOpError("expects dst element type to be integer");
  }

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32) {
    return emitOpError("expects dst element type to be i16/i32");
  }

  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto tmpTy = mlir::dyn_cast<TileBufType>(getTmp().getType());
    if (!tmpTy) {
      return emitOpError("expects tmp to be a tile buffer");
    }
    auto tmpSpace =
        mlir::dyn_cast_or_null<AddressSpaceAttr>(tmpTy.getMemorySpace());
    if (!tmpSpace || tmpSpace.getAddressSpace() != AddressSpace::VEC) {
      return emitOpError("expects tmp to be in vec address space");
    }
    Type tmpElemTy = tmpTy.getElementType();
    if (!(tmpElemTy.isF32() || tmpElemTy.isInteger(32))) {
      return emitOpError("expects A2/A3 tmp element type to be a 4-byte type");
    }
    if (tmpTy.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor)) {
      return emitOpError("expects tmp blayout to be row_major");
    }
    if (tmpTy.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox)) {
      return emitOpError("expects tmp slayout to be none_box");
    }
    if (tmpTy.getSFractalSizeI32() != 512) {
      return emitOpError("expects tmp fractal size to be 512");
    }
    auto tmpBytes = getStaticByteSize(tmpTy);
    if (!tmpBytes) {
      return emitOpError("expects tmp to have static byte size");
    }
    uint64_t minTmpBytes = bw == 32 ? 768 : 1792;
    if (*tmpBytes < minTmpBytes) {
      return emitOpError("expects A2/A3 tmp capacity to be at least ")
             << minTmpBytes << " bytes for " << bw
             << "-bit dst element type";
    }
  }

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy) {
    return emitOpError("expects S to be integer");
  }

  if (sTy != elemTy) {
    return emitOpError("expects S and dst element type to be exactly the same type");
  }
  auto shape = getShapeVec(dstTy);
  if (shape.size() != 2) {
    return emitOpError("expects dst to be rank-2");
  }
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1) {
    return emitOpError("expects dst cols to be different from 1");
  }

  return success();
}

LogicalResult pto::TTriOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy) {
    return emitOpError("expects diagonal to be an integer operand");
  }

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1) {
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");
  }

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false)) {
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        }
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true)) {
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        }
        return success();
      });
}

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, t0, "src0")) ||
        failed(verifyVecTileStorage(*this, t1, "src1")) ||
        failed(verifyVecTileStorage(*this, td, "dst"))) {
      return failure();
    }

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      return emitOpError("failed to get element type for src0/src1/dst");
    }
    if (e0 != e1) {
      return emitOpError("expects src0 and src1 to have the same element type");
    }
    if (!(e0.isInteger(32) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tcmp input element type to be i32/f16/f32");
    }
    if (!ed.isInteger(8)) {
      return emitOpError("expects dst element type to be i8");
    }

    auto valid0 = getValidShapeVec(t0);
    auto valid1 = getValidShapeVec(t1);
    auto validd = getValidShapeVec(td);
    if (valid0.size() != 2 || valid1.size() != 2 || validd.size() != 2) {
      return emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    }
    if (!hasCompatibleKnownExtent(valid0[0], valid1[0])) {
      return emitOpError("expects src0 and src1 to have the same valid row");
    }
    if (!hasCompatibleKnownExtent(valid0[1], valid1[1])) {
      return emitOpError("expects src0 and src1 to have the same valid column");
    }
    if (!hasCompatibleKnownExtent(valid0[0], validd[0])) {
      return emitOpError("expects src0 valid row to equal dst valid row");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst"))) {
      return failure();
    }

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      return emitOpError("failed to get element type for src0/src1/dst");
    }
    if (e0 != e1) {
      return emitOpError("expects src0 and src1 to have the same element type");
    }
    bool inputOk = e0.isF16() || e0.isF32() || e0.isBF16() ||
                   e0.isInteger(8) || e0.isInteger(16) || e0.isInteger(32);
    if (!inputOk) {
      return emitOpError("expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
    }
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8) {
        return emitOpError("expects dst element type to be i8");
      }
    } else {
      return emitOpError("expects dst element type to be i8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td)) {
      return emitOpError("expects src0, src1, and dst to have the same shape");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32())) {
      return emitOpError("expects A2/A3 tcmps input element type to be i16/i32/f16/f32");
    }

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat())) {
      return emitOpError("expects scalar to be integer, index, or float");
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0]) {
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32())) {
      return emitOpError("expects A5 tcmps input element type to be i8/i16/i32/f16/f32");
    }

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat())) {
      return emitOpError("expects scalar to be integer, index, or float");
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0]) {
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return emitOpError("expects tcolexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td)) {
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for src0/src1/dst");
  }

  auto isSupportedElem = [allowIntegerTypes, targetArch](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32()) {
      return true;
    }
    if (!allowIntegerTypes) {
      return false;
    }
    if (elemTy.isInteger(16) || elemTy.isInteger(32)) {
      return true;
    }
    return targetArch == PTOArch::A5 && elemTy.isInteger(8);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes) {
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    }
    if (targetArch == PTOArch::A5) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    }
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (getShapeVec(t0) != getShapeVec(td)) {
    return op->emitOpError("expects src0/dst to have same shape");
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst"))) {
    return failure();
  }

  if (auto src0TileTy = dyn_cast<TileBufType>(t0)) {
    if (src0TileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects src0 to use row-major layout");
    }
  }

  if (auto src1TileTy = dyn_cast<TileBufType>(t1)) {
    if (src1TileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects src1 to use row-major layout");
    }
  }
  if (auto dstTileTy = dyn_cast<TileBufType>(td)) {
    if (dstTileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects dst to use row-major layout");
    }
  }

  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == 2 && dstValid.size() == 2 &&
      src1Valid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1]) {
    return op->emitOpError("expects src1 valid_shape[1] to equal dst valid_shape[1]");
  }

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMaxOp::verify() {
  if (!getTmp()) {
    return verifyTColArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  if (!getTmp()) {
    return verifyTColArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}



ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  // Parse: ins(%src : type) or ins(%src, %tmp {isBinary = ...}: type, type)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }

  // Check for optional tmp operand (format 2)
  if (succeeded(parser.parseOptionalComma())) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type)
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;

    // Parse attributes (isBinary)
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }

    // Parse types: : type, type
    if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  } else {
    // Format 1: ins(%src : type)
    if (parser.parseColonType(srcTy)) {
      return failure();
    }
  }

  if (parser.parseRParen()) {
    return failure();
  }

  // Parse: outs(%dst : type)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }

  // Parse any remaining attributes (for format 1)
  if (!hasTmp) {
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
  }

  // Resolve operands
  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }

  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }

  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 2> elidedAttrs = {"operandSegmentSizes"};
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
    SmallVector<StringRef, 2> elidedAttrs = {"isBinary", "operandSegmentSizes"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
      return failure();
    }
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp && !hasIsBinary) {
      return emitOpError("tmp operand requires isBinary attribute");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy)) {
        return emitOpError("expects src/tmp/dst element types to match");
      }
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary()))) {
        return failure();
      }
      if (getIsBinary()) {
        auto srcValid = getValidShapeVec(srcTy);
        auto elemBytes = getElemByteSize(getElemTy(srcTy));
        if (srcValid.size() != 2 || srcValid[0] == ShapedType::kDynamic ||
            srcValid[1] == ShapedType::kDynamic || elemBytes == 0) {
          return emitOpError(
              "expects static src valid_shape and element size to verify tcolsum tmp");
        }
        uint64_t requiredBytes =
            static_cast<uint64_t>(ceilDivInt64(srcValid[0], 2)) *
            static_cast<uint64_t>(srcValid[1]) * elemBytes;
        if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, requiredBytes))) {
          return failure();
        }
      }
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return emitOpError("expects src/dst element types to match");
    }
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError("expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
      return failure();
    }
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp && !hasIsBinary) {
      return emitOpError("tmp operand requires isBinary attribute");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy)) {
        return emitOpError("expects src/tmp/dst element types to match");
      }
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary()))) {
        return failure();
      }
      if (getIsBinary()) {
        auto srcValid = getValidShapeVec(srcTy);
        auto elemBytes = getElemByteSize(getElemTy(srcTy));
        if (srcValid.size() != 2 || srcValid[0] == ShapedType::kDynamic ||
            srcValid[1] == ShapedType::kDynamic || elemBytes == 0) {
          return emitOpError(
              "expects static src valid_shape and element size to verify tcolsum tmp");
        }
        uint64_t requiredBytes =
            static_cast<uint64_t>(ceilDivInt64(srcValid[0], 2)) *
            static_cast<uint64_t>(srcValid[1]) * elemBytes;
        if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, requiredBytes))) {
          return failure();
        }
      }
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return emitOpError("expects src/dst element types to match");
    }
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() || elem.isInteger(8) ||
          elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError("expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

static bool tcvtIsResolvedSubview(Value value) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  auto semantics =
      alloc ? alloc->getAttrOfType<StringAttr>("pto.view_semantics")
            : StringAttr();
  return semantics && semantics.getValue() == "subview";
}

static bool tcvtNeedsTmp(TCvtOp op, Type srcElem, Type dstElem) {
  if (op.getSatMode() != pto::SaturationMode::OFF) {
    return false;
  }
  return (srcElem.isF32() && dstElem.isInteger(16)) ||
         (srcElem.isF16() &&
          (dstElem.isInteger(16) || dstElem.isInteger(8)));
}

static int64_t computeTCvtTmpRequiredBytes(Type srcElem, Type dstElem,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> dstValid) {
  int64_t rows = dstValid[0], cols = dstValid[1];
  int64_t requiredBytes = 0;
  if (rows > 0 && cols > 0 && srcElem.isF32()) {
    int64_t head = 4 * 64 * std::min<int64_t>(cols / 64, 255);
    int64_t remainder = cols % 64;
    int64_t tail = remainder == 0
                       ? 0
                       : 32 * ((std::min<int64_t>(rows, 255) - 1) *
                                   (srcShape[1] / 8) +
                               llvm::divideCeil(remainder, int64_t{8}));
    requiredBytes = std::max(head, tail);
  } else if (cols > 0 && srcElem.isF16()) {
    int64_t width = std::min<int64_t>(cols, 64);
    int64_t halfToI16 = 32 * llvm::divideCeil(width, int64_t{8});
    int64_t halfToI8 = std::max<int64_t>(
        halfToI16,
        128 + 32 * static_cast<int64_t>(
                       llvm::divideCeil(width, int64_t{16})));
    requiredBytes = dstElem.isInteger(8) ? halfToI8 : halfToI16;
  }
  return requiredBytes;
}

static LogicalResult verifyTCvtTmp(TCvtOp op, Type srcTy, Type dstTy,
                                   Type srcElem, Type dstElem) {
  if (!op.getTmp()) {
    return success();
  }
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!tcvtNeedsTmp(op, srcElem, dstElem)) {
    return success();
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != 2 || dstValid.size() != 2 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) ||
      llvm::is_contained(dstValid, ShapedType::kDynamic)) {
    return op.emitOpError(
        "expects static src shape and dst valid_shape to verify tcvt tmp");
  }
  int64_t requiredBytes =
      computeTCvtTmpRequiredBytes(srcElem, dstElem, srcShape, dstValid);
  auto tmpBytes = getStaticByteSize(tmpTy);
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(requiredBytes)) {
    return op.emitOpError()
           << "expects tcvt tmp capacity to be at least " << requiredBytes
           << " bytes";
  }
  return success();
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
    return failure();
  }
  // A resolved subview keeps its parent's physical shape so the generated
  // Tile retains the parent stride. Its valid shape is the logical tcvt
  // extent, so comparing physical shapes would reject a valid sliced tile.
  if (!tcvtIsResolvedSubview(getSrc()) && !tcvtIsResolvedSubview(getDst()) &&
      failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    }
    return verifyTCvtTmp(*this, srcTy, dstTy, srcElem, dstElem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem)) {
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    }
    if (getTmp() && failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
