// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects dst to use row-major layout");
    }

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32)) {
      return emitOpError("expects dst element type to be i32 or ui32");
    }

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32) {
        return emitOpError() << "expects " << name << " to be i32/ui32";
      }
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3"))) {
      return failure();
    }

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10) {
      return emitOpError("expects rounds to be 7 or 10");
    }

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32())) {
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32))) {
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);

    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile)) {
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");
    }

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true))) {
      return failure();
    }
    if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
      return emitOpError("scalar must be a scalar type (integer/float)");
    }
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    }
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32()) {
      return emitOpError("expects element type to be f16 or f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem) {
      return emitOpError("expects scalar type == dst element type");
    }
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 16 || w == 32) {
        return mlir::success();
      }
    }
    return emitOpError("expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem) {
      return emitOpError("expects scalar type == dst element type");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 8 || w == 16 || w == 32) {
        return mlir::success();
      }
    }
    return emitOpError("expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3AccCastExtractTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  }
  return false;
}

static bool isA5AccCastExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(32);
  }
  return false;
}

static bool isA5AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
           dstElem.isF32() ||
           (llvm::isa<FloatType>(dstElem) &&
            llvm::cast<FloatType>(dstElem).getWidth() == 8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  if (dstSpace == pto::AddressSpace::LEFT) {
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
           bl == static_cast<int32_t>(pto::BLayout::RowMajor);
  }
  return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool isA2A3ExtractElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isRowMajorNoneBoxND(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

struct TExtractCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TExtractCommon> verifyTExtractCommon(TExtractOp op,
                                                      bool allowLowPrecision) {
  const bool hasFp = static_cast<bool>(op.getFp());
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb) {
    return op.emitOpError("expects src and dst to be !pto.tile_buf");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(),
          /*includeIndexAndIntOpsInConstFold=*/hasFp)) ||
      failed(verifyExtractStaticBoundsCommon(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(), srcTy, dstTy,
          /*includeIndexAndIntOpsInConstFold=*/hasFp))) {
    return failure();
  }
  if (hasFp) {
    Type fpTy = op.getFp().getType();
    if (failed(verifyTileBufCommon(op, fpTy, "fp", allowLowPrecision))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to use loc=scaling");
    }
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("expects src and dst to have element types");
  }
  if ((!srcSpace || *srcSpace != pto::AddressSpace::ACC) && srcElem != dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  return TExtractCommon{srcTy, dstTy, srcTb, dstTb,
                        srcElem, dstElem, srcSpace, dstSpace};
}

static LogicalResult
verifyTExtractFpFormLoc(TExtractOp op,
                        std::optional<pto::AddressSpace> srcSpace) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && !srcIsAcc) {
    return op.emitOpError("expects fp form to use loc=acc src");
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && !srcIsAcc) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Acc(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
  }
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Mat(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.srcSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
  }
  if (*c.dstSpace != pto::AddressSpace::LEFT &&
      *c.dstSpace != pto::AddressSpace::RIGHT) {
    return op.emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(c.srcTb)) {
    return op.emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    }
  } else {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA2A3(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA2A3ExtractElemType(c.dstElem) && !(hasFp && c.dstElem.isInteger(16))) {
    return op.emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr()) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA2A3Acc(op, c);
  }
  return verifyTExtractA2A3Mat(op, c);
}

static bool isTExtractA5SupportedPair(pto::AddressSpace srcSpace,
                                      pto::AddressSpace dstSpace) {
  return (srcSpace == pto::AddressSpace::MAT &&
          (dstSpace == pto::AddressSpace::LEFT ||
           dstSpace == pto::AddressSpace::RIGHT ||
           dstSpace == pto::AddressSpace::SCALING)) ||
         (srcSpace == pto::AddressSpace::VEC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC)) ||
         (srcSpace == pto::AddressSpace::ACC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC));
}

static LogicalResult verifyTExtractA5Mat(TExtractOp op,
                                         const TExtractCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
  }
  if (!hasMatExtractSourceLayoutA5(c.srcTb, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
    }
  } else if (*c.dstSpace == pto::AddressSpace::RIGHT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA5Acc(TExtractOp op,
                                         const TExtractCommon &c) {
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    if (!isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError("expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
    }
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA5AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A5 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
    }
  } else if (!isA5AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A5 acc textract element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA5ExtractElemType(c.dstElem)) {
    return op.emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr() &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (!isTExtractA5SupportedPair(*c.srcSpace, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract to use a supported src/dst loc pair");
  }
  if (*c.srcSpace == pto::AddressSpace::MAT) {
    return verifyTExtractA5Mat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
    }
    if (!isRowMajorNoneBoxND(c.srcTb) || !isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError(
          "expects A5 vec->vec textract src/dst to use ND layout "
          "(blayout=row_major, slayout=none_box)");
    }
    return success();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA5Acc(op, c);
  }
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExtractA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExtractA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
static bool isA2A3AccCastInsertTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantInsertTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  }
  return false;
}

static bool isColMajorRowMajorNZ(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5SupportedVecElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 32;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

struct TInsertCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TInsertCommon> verifyTInsertCommon(TInsertOp op,
                                                    bool allowLowPrecision) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb) {
    return op.emitOpError("expects src and dst to be !pto.tile_buf");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(),
          /*includeIndexAndIntOpsInConstFold=*/true)) ||
      failed(verifyInsertStaticBoundsCommon(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(), srcTy, dstTy,
          /*includeIndexAndIntOpsInConstFold=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  return TInsertCommon{srcTy, dstTy, srcTb, dstTb,
                       srcElem, dstElem, srcSpace, dstSpace};
}

static LogicalResult
verifyTInsertOptionalFp(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                        bool isA5) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("fp and preQuantScalar are mutually exclusive");
  }
  if (hasFp) {
    if (!srcIsAcc) {
      return op.emitOpError("fp is only valid with src loc=acc");
    }
    auto fpTy = op.getFp().getType();
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    if (!fpTb) {
      return op.emitOpError("expects fp to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(op, fpTy, "fp", /*allowLowPrecision=*/isA5))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to be loc=scaling");
    }
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("preQuantScalar is only valid with src loc=acc");
  }
  if (reluNonDefault && !srcIsAcc) {
    return op.emitOpError("reluPreMode is only valid with src loc=acc");
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalAttrs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                           std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasAccToVecMode) {
    if (!isA5) {
      return op.emitOpError("accToVecMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::VEC)
      return op.emitOpError("accToVecMode is only valid with src=acc, dst=vec");
  }
  if (hasInsertMode) {
    if (!isA5) {
      return op.emitOpError("tinsertMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::MAT) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
    }
    auto srcTb = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!srcTb || !isColMajorRowMajorNZ(srcTb)) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
          "(blayout=col_major, slayout=row_major)");
    }
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalArgs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                          std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  if (failed(verifyTInsertOptionalFp(op, srcSpace, isA5))) {
    return failure();
  }
  return verifyTInsertOptionalAttrs(op, srcSpace, dstSpace, isA5);
}

static LogicalResult verifyTInsertA2A3AccMat(TInsertOp op,
                                             const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantInsertTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc fp/preQuantScalar tinsert element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastInsertTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTInsertA2A3(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/false))) {
    return failure();
  }
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError(
          "expects vec->vec tinsert to use the base form without "
          "preQuantScalar or reluPreMode");
    }
    if (c.srcElem != c.dstElem || !isA2A3VecInsertElemType(c.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
      *c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  return verifyTInsertA2A3AccMat(op, c);
}

static LogicalResult verifyTInsertA5Acc(TInsertOp op, const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (!isColMajorRowMajorNZ(c.dstTb)) {
      return op.emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
    bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
    if (!dstIsND && !dstIsNZ) {
      return op.emitOpError(
          "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
    }
  }
  const bool hasQuant =
      static_cast<bool>(op.getFp()) || static_cast<bool>(op.getPreQuantScalar());
  bool okTypes;
  if (hasQuant) {
    okTypes = isA5VectorPreQuantTypePair(c.srcElem, c.dstElem);
  } else {
    okTypes = (c.srcElem.isF32() &&
               (c.dstElem.isF16() || c.dstElem.isBF16() || c.dstElem.isF32())) ||
              (c.srcElem.isInteger(32) && c.dstElem.isInteger(32));
  }
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc-source tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)" +
        (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
  }
  return success();
}

static LogicalResult verifyTInsertA5VecMat(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->mat tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (hasTInsertMode && !srcIsNZ) {
    return op.emitOpError("expects tinsertMode vec->mat tinsert src to use NZ(col_major/row_major) layout");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecVec(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->vec tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
  if (srcIsND && dstIsND) {
    // ND->ND path
  } else if (srcIsNZ && dstIsNZ) {
    // NZ->NZ path
  } else {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst layouts to match: "
        "both ND(row_major/none_box) or both NZ(col_major/row_major)");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (hasAccToVecMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec tinsert forms");
  }
  if (hasTInsertMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::VEC ||
       *c.dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects tinsertMode only on A5 vec->mat tinsert forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");
  }
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/true))) {
    return failure();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC &&
      (*c.dstSpace == pto::AddressSpace::MAT || *c.dstSpace == pto::AddressSpace::VEC)) {
    return verifyTInsertA5Acc(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::MAT) {
    return verifyTInsertA5VecMat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
    return verifyTInsertA5VecVec(op, c);
  }
  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, acc->vec, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTInsertA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTInsertA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8;
  }
  return false;
}

static bool isA5MxFp8InputType(Type ty) {
  return ty && isa<Float8E4M3FNType, Float8E5M2Type>(ty);
}

static bool isA5MxInputTypePair(Type lhsTy, Type rhsTy) {
  return (isA5MxFp8InputType(lhsTy) && isA5MxFp8InputType(rhsTy)) ||
         (isPTOFloat4PackedType(lhsTy) && isPTOFloat4PackedType(rhsTy));
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputTypePair(lhsElem, rhsElem)) {
    return op->emitOpError()
           << "expects A5 mx " << lhsName << "/" << rhsName
           << " element types to be a supported fp8/fp8 or fp4/fp4 pair";
  }

  if (!dstElem.isF32()) {
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";
  }

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8) || isA5Fp8LikeType(dstElem) ||
           isPTOHiFloat8Type(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy,
                                              Type dstTy) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op->emitError("expects src/dst to be PTO shaped-like types");
  }

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op->emitError("expects rank-2 shaped types for src/dst");
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    unsigned elemBytes = getPTOStorageElemByteSize(t);
    return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0) {
    return op->emitError("unsupported element type (expects int/float element types)");
  }
  if (srcB != dstB) {
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  }
  if (!(srcB == 1 || srcB == 2 || srcB == 4)) {
    return op->emitError("expects element size to be 1, 2, or 4 bytes");
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);

  bool expanded = false;
  for (auto [srcDim, dstDim] : llvm::zip_equal(srcShape, dstShape)) {
    if (srcDim == dstDim) {
      continue;
    }
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim)) {
      return op->emitError("cannot infer TFILLPAD lowering from mismatched "
                           "dynamic physical shapes");
    }
    if (srcDim > dstDim) {
      return op->emitError(
          "expects each dst physical shape dimension to be >= src");
    }
    expanded = true;
  }
  if (expanded &&
      (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
       *dstSpace != pto::AddressSpace::VEC)) {
    return op->emitError("expects expanded TFILLPAD only for loc=vec");
  }

  // pto.tfillpad lowers to TFILLPAD(dst, src). For loc=mat, pto-isa only
  // exposes the homogeneous overload, so src/dst must use the same Tile<...>
  // specialization (including valid_shape and pad).
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  if (srcTb && dstTb && srcSpace && dstSpace &&
      *srcSpace == mlir::pto::AddressSpace::MAT &&
      *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
    auto dimToStr = [](int64_t dim) -> std::string {
      return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
    };
    SmallVector<std::string, 4> mismatchFields;
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() == 2 && dstValid.size() == 2) {
      if (srcValid[0] != dstValid[0]) {
        mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                 dimToStr(dstValid[0]) + ")");
      }
      if (srcValid[1] != dstValid[1]) {
        mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                 dimToStr(dstValid[1]) + ")");
      }
    }
    if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
      mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                               " vs " + std::to_string(dstTb.getPadValueI32()) +
                               ")");
    }

    auto diag = op->emitError()
                << "expects src/dst tile types to be lowerable to TFILLPAD "
                   "for loc=mat";
    if (!mismatchFields.empty()) {
      diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
    }
    diag << "\n  src: " << srcTy;
    diag << "\n  dst: " << dstTy;
    diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
    return failure();
  }

  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null) {
      return op->emitError("expects dst PadVal != Null for tfillpad");
    }
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  if (getOperation()->getAttr("mode")) {
    return emitOpError("does not accept 'mode'; PTOAS infers TFILLPAD lowering "
                       "from physical shape and planned addresses");
  }

  if (failed(verifyTFillPadLike(getOperation(), getSrc().getType(),
                                getDst().getType()))) {
    return failure();
  }

  if (auto padValueAttr = getPadValueAttr()) {
    auto dstSpace = getPTOMemorySpaceEnum(getDst().getType());
    if (!dstSpace || *dstSpace != pto::AddressSpace::MAT) {
      return emitOpError("expects padValue attribute only for loc=mat tfillpad");
    }
    auto dstTileTy = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!dstTileTy) {
      return emitOpError("expects dst to be tile_buf when padValue is specified");
    }
    if (dstTileTy.getPadValueI32() != static_cast<int32_t>(padValueAttr.getValue())) {
      return emitOpError("expects padValue attribute to match dst tile pad configuration");
    }
  }

  return success();
}


static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (isPTOFloat8Type(ty)) {
    return true;
  }
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32;
  }
  return false;
}

static unsigned getMaskGatherTimes(mlir::pto::MaskPatternAttr mp) {
  switch (mp.getValue()) {
  case mlir::pto::MaskPattern::P1111:
    return 1;
  case mlir::pto::MaskPattern::P0101:
  case mlir::pto::MaskPattern::P1010:
    return 2;
  default:
    return 4;
  }
}

static LogicalResult verifyTGatherMaskShapes(TGatherOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tgather to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tgather to provide maskPattern");
  }
  const unsigned times = getMaskGatherTimes(mp);
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != static_cast<int64_t>(dstValid[1] * times)) {
      return op.emitOpError("expects src valid cols to equal dst valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != static_cast<int64_t>(dstValid[0] * times)) {
      return op.emitOpError("expects src valid rows to equal dst valid rows times the mask expansion factor for col direction");
    }
  } else {
    return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return success();
}

static LogicalResult verifyTGatherMaskForm(TGatherOp op, bool allowA5MaskTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5MaskTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5MaskTypes))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src and dst to be in the vec address space");
  }
  unsigned srcElemBytes = getPTOStorageElemByteSize(srcElem);
  unsigned dstElemBytes = getPTOStorageElemByteSize(dstElem);
  if (srcElemBytes == 0 || dstElemBytes == 0) {
    return op.emitOpError("failed to get element size for src/dst");
  }
  if (srcElemBytes != dstElemBytes) {
    return op.emitOpError("expects src and dst element sizes to match");
  }
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  if (failed(verifyTGatherMaskShapes(op))) {
    return failure();
  }
  if (allowA5MaskTypes) {
    if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4)) {
      return op.emitOpError("expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
    }
    if (!isSupportedGatherElemTypeA5(srcElem) || !isSupportedGatherElemTypeA5(dstElem)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    }
  } else {
    if (!(srcElemBytes == 2 || srcElemBytes == 4)) {
      return op.emitOpError("expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
    }
  }
  return success();
}

static LogicalResult verifyTGatherIndexTypes(TGatherOp op, bool allow16BitIndices,
                                             bool allowA5ElemTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, idxTy, "indices"))) {
    return failure();
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (allowA5ElemTypes) {
    if (!isSupportedGatherElemTypeA5Index(srcElem) ||
        !isSupportedGatherElemTypeA5Index(dstElem)) {
      return op.emitOpError(
          "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    }
  } else if (!isSupportedGatherElemTypeA2A3(srcElem) ||
             !isSupportedGatherElemTypeA2A3(dstElem)) {
    return op.emitOpError("expects gather src/dst element type to be i16/i32/f16/f32");
  }
  auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
  if (!idxElem) {
    return op.emitOpError("indices element type must be integer");
  }
  unsigned width = idxElem.getWidth();
  if (!(width == 32 || (allow16BitIndices && width == 16))) {
    return op.emitOpError() << "expects indices element type to be i32"
                            << (allow16BitIndices ? " or i16" : "");
  }
  return success();
}

static LogicalResult verifyTGatherIndexShapes(TGatherOp op, bool allowA5ElemTypes) {
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  auto idxValid = getValidShapeVec(idxTy);
  auto idxShape = getShapeVec(idxTy);
  if (idxValid.size() == 2 && idxShape.size() == 2 &&
      idxValid[1] != ShapedType::kDynamic && idxShape[1] != ShapedType::kDynamic &&
      idxValid[1] != idxShape[1]) {
    return op.emitOpError("expects indices valid_shape[1] to equal indices cols");
  }
  if (!allowA5ElemTypes) {
    if (failed(verifyTileBufSameValidShape(op, dstTy, idxTy, "dst", "indices"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(idxTy) ||
        !isRowMajorTileBuf(op.getTmp().getType())) {
      return op.emitOpError(
          "expects A2/A3 index-form dst, indices, and tmp to use row-major layout");
    }
    auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
    Type tmpElem = getElemTy(op.getTmp().getType());
    if (tmpElem != idxElem) {
      return op.emitOpError("expects tmp and indices to have the same element type");
    }
    if (failed(verifyTileBufSameValidShape(op, idxTy, op.getTmp().getType(), "indices", "tmp"))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTGatherIndexForm(TGatherOp op, bool allow16BitIndices,
                                            bool allowA5ElemTypes) {
  if (failed(verifyTGatherIndexTypes(op, allow16BitIndices, allowA5ElemTypes))) {
    return failure();
  }
  return verifyTGatherIndexShapes(op, allowA5ElemTypes);
}

static LogicalResult verifyTGatherCompareSrcType(TGatherOp op, Type srcElem,
                                                 pto::CmpMode cmpMode,
                                                 bool allowA5SrcTypes) {
  if (allowA5SrcTypes) {
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isInteger(16) ||
          srcElem.isInteger(32))) {
      return op.emitOpError(
          "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
    }
  } else {
    if (!(srcElem.isF16() || srcElem.isF32() ||
          (srcElem.isInteger(32) && cmpMode == pto::CmpMode::EQ))) {
      return op.emitOpError(
          "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
    }
  }
  return success();
}

static LogicalResult verifyTGatherCompareForm(TGatherOp op, bool allowA5SrcTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type cdstTy = op.getCdst().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufCommon(op, cdstTy, "cdst")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type cdstElem = getElemTy(cdstTy);
  if (!srcElem || !dstElem || !cdstElem) {
    return op.emitOpError("failed to get element type for src/dst/cdst");
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || dstInt.getWidth() != 32) {
    return op.emitOpError("expects dst element type to be i32");
  }
  if (cdstElem != dstElem) {
    return op.emitOpError("expects cdst to have the same element type as dst");
  }
  if (op.getKValue().getType() != srcElem) {
    return op.emitOpError("expects kValue to have the same type as src element type");
  }
  auto cmpAttr = op.getCmpModeAttr();
  auto cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
  if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT) {
    return op.emitOpError("expects compare-form tgather cmpMode to be eq or gt");
  }
  if (failed(verifyTGatherCompareSrcType(op, srcElem, cmpMode, allowA5SrcTypes))) {
    return failure();
  }
  if (failed(verifyVecTileCommonA2A3(op, srcTy, "src")) ||
      failed(verifyVecTileCommonA2A3(op, dstTy, "dst")) ||
      failed(verifyVecTileCommonA2A3(op, cdstTy, "cdst")) ||
      failed(verifyVecTileCommonA2A3(op, tmpTy, "tmp"))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTGatherA2A3(TGatherOp op) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue()) {
      return op.emitOpError("mask-pattern tgather only allows src and dst operands");
    }
    return verifyTGatherMaskForm(op, /*allowA5MaskTypes=*/false);
  }
  if (op.getAxisAttr()) {
    return op.emitOpError("axis attribute must not be provided without maskPattern");
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp()) {
      return op.emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
    }
    if (op.getIndices()) {
      return op.emitOpError("compare-form tgather does not take indices");
    }
    return verifyTGatherCompareForm(op, /*allowA5SrcTypes=*/false);
  }
  if (!op.getIndices() || !op.getTmp()) {
    return op.emitOpError("index-form tgather expects both indices and tmp");
  }
  return verifyTGatherIndexForm(op, /*allow16BitIndices=*/false, /*allowA5ElemTypes=*/false);
}

static LogicalResult verifyTGatherA5(TGatherOp op) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue()) {
      return op.emitOpError("mask-pattern tgather only allows src and dst operands");
    }
    return verifyTGatherMaskForm(op, /*allowA5MaskTypes=*/true);
  }
  if (op.getAxisAttr()) {
    return op.emitOpError("axis attribute must not be provided without maskPattern");
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp()) {
      return op.emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
    }
    if (op.getIndices()) {
      return op.emitOpError("compare-form tgather does not take indices");
    }
    return verifyTGatherCompareForm(op, /*allowA5SrcTypes=*/true);
  }
  if (!op.getIndices()) {
    return op.emitOpError("index-form tgather expects indices");
  }
  return verifyTGatherIndexForm(op, /*allow16BitIndices=*/true, /*allowA5ElemTypes=*/true);
}

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static std::optional<unsigned> tgatherbElemBytes(Type ty) {
  unsigned elemBytes = getPTOStorageElemByteSize(ty);
  if (elemBytes == 0) {
    return std::nullopt;
  }
  return elemBytes;
}

static FailureOr<std::pair<Type, Type>> verifyTGatherBCommon(TGatherBOp op) {
  Type srcTy = op.getSrc().getType();
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, offTy, "offsets")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto srcElemTy = getElemTy(srcTy);
  auto dstElemTy = getElemTy(dstTy);
  if (!srcElemTy || !dstElemTy) {
    return op.emitOpError() << "failed to get element type for src/dst";
  }
  return std::make_pair(srcElemTy, dstElemTy);
}

static LogicalResult verifyTGatherBA2A3(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  Type dstElemTy = elems->second;
  if (failed(
          verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(offTy)) {
    return op.emitOpError()
           << "expects dst and offsets to use row-major layout";
  }
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError()
           << "expects A2/A3 dst element size to be 2 or 4 bytes";
  }
  Type offElemTy = getElemTy(offTy);
  if (!offElemTy.isInteger(32)) {
    return op.emitOpError() << "expects offsets element type to be i32";
  }

  auto dstValid = getValidShapeVec(dstTy);
  auto offValid = getValidShapeVec(offTy);
  if (dstValid.size() != 2 || offValid.size() != 2) {
    return op.emitOpError() << "expects rank-2 src/offsets/dst tile buffers";
  }
  if (dstValid[0] != ShapedType::kDynamic &&
      offValid[0] != ShapedType::kDynamic && dstValid[0] != offValid[0]) {
    return op.emitOpError() << "expects offsets valid rows to match dst valid rows";
  }
  if (dstValid[1] != ShapedType::kDynamic &&
      offValid[1] != ShapedType::kDynamic) {
    int64_t blockElems = 32 / *dstBytes;
    int64_t blocks = (dstValid[1] + blockElems - 1) / blockElems;
    int64_t expectedOffsetCols = ((blocks + 7) / 8) * 8;
    if (offValid[1] != expectedOffsetCols) {
      return op.emitOpError()
             << "expects offsets valid cols to be compact 32B block address "
                "count padded to 8 entries; expected "
             << expectedOffsetCols << ", got " << offValid[1];
    }
  }
  return mlir::success();
}

static LogicalResult verifyTGatherBA5(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type dstElemTy = elems->second;
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherBA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherBA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2) {
      return emitOpError("expects src to have rank-2 valid_shape");
    }
    if (valid[0] != ShapedType::kDynamic && valid[0] < 0) {
      return emitOpError("expects src valid_shape[0] to be non-negative");
    }
    if (valid[1] != ShapedType::kDynamic && valid[1] < 0) {
      return emitOpError("expects src valid_shape[1] to be non-negative");
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    }
    if (!getSlope().getType().isF32()) {
      return emitOpError() << "expects slope to have type f32";
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

static std::optional<int64_t> tmovCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> tmovCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> tmovAlign16(int64_t value) {
  auto biased = tmovCheckedAdd(value, 15);
  return biased ? tmovCheckedMul(*biased / 16, 16) : std::nullopt;
}

static std::optional<int64_t> tmovCheckedElements(ArrayRef<int64_t> shape) {
  if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0 ||
      (shape[1] != 0 &&
       shape[0] > std::numeric_limits<int64_t>::max() / shape[1])) {
    return std::nullopt;
  }
  return shape[0] * shape[1];
}

static LogicalResult verifyTMovXToZzElemLayout(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  auto srcTb = cast<pto::TileBufType>(srcTy);
  auto dstTb = cast<pto::TileBufType>(dstTy);
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type tmpElem = getElemTy(fp.getType());
  if (srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError("expects src, dst, and tmp to share one element type");
  }
  bool validElem = isPTOHiFloat8Type(srcElem) || isPTOF8E8M0Type(srcElem);
  if (auto integer = dyn_cast<IntegerType>(srcElem)) {
    validElem = integer.getWidth() == 8 &&
                integer.getSignedness() == IntegerType::Unsigned;
  }
  if (!validElem) {
    return op.emitOpError("expects element type to be one of ui8, !pto.hif8, !pto.f8E8M0 (i8 lowers to int8_t, which PTO-ISA CommonCheckZZ rejects)");
  }
  if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op.emitOpError("expects src to use blayout=row_major, slayout=none_box");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects dst to use blayout=row_major, slayout=row_major (ZZ box)");
  }
  return success();
}
