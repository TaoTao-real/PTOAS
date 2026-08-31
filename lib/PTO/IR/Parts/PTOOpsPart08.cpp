// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTQuantInt8Common(TQuantOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (op.getTmp() && failed(verifyTileBufCommon(op, op.getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (!getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects src to have element type f32";
  }
  if (op.getOffset()) {
    Type offsetTy = op.getOffset().getType();
    if (failed(verifyTileBufCommon(op, offsetTy, "offset"))) {
      return failure();
    }
    if (!getElemTy(offsetTy).isF32()) {
      return op.emitOpError() << "expects offset to have element type f32";
    }
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (!getElemTy(tmpTy).isF32()) {
      return op.emitOpError() << "expects tmp to have element type f32";
    }
  }
  return success();
}

static LogicalResult verifyTQuantA2A3Tmp(TQuantOp op, Type srcTy, Type tmpTy) {
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to use row-major layout";
  }
  if (getShapeVec(srcTy) != getShapeVec(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to have the same shape as src";
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  auto requiredBytes = getStaticByteSize(srcTy);
  if (!requiredBytes) {
    return op.emitOpError(
        "expects A2/A3 tquant src shape to be static when tmp is provided");
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes);
}

static LogicalResult verifyTQuantA2A3Param(TQuantOp op, Type paramTy, Type dstTy,
                                           StringRef paramName) {
  if (isRowMajorTileBuf(paramTy)) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " to use non-row-major layout";
  }
  auto paramValid = getValidShapeVec(paramTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (paramValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " and dst to have rank-2 valid_shape";
  }
  if (paramValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && paramValid[0] != dstValid[0]) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[0] to equal dst valid_shape[0]";
  }
  if (paramValid[1] != ShapedType::kDynamic && paramValid[1] != 1) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[1] to be 1";
  }
  return success();
}

static LogicalResult verifyTQuantA2A3(TQuantOp op) {
  if (failed(verifyTQuantInt8Common(op))) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError()
           << "expects A2/A3 src and dst to use row-major layout";
  }
  if (op.getTmp() &&
      failed(verifyTQuantA2A3Tmp(op, srcTy, op.getTmp().getType()))) {
    return failure();
  }
  if (failed(verifyTQuantA2A3Param(op, fpTy, dstTy, "fp"))) {
    return failure();
  }
  if (op.getOffset() &&
      failed(verifyTQuantA2A3Param(op, op.getOffset().getType(), dstTy, "offset"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this))) {
    return failure();
  }
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTQuantA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantInt8Common(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static std::optional<int64_t> mxCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> mxCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> mxCeilDiv(int64_t value, int64_t divisor) {
  if (value < 0 || divisor <= 0) {
    return std::nullopt;
  }
  auto plus = mxCheckedAdd(value, divisor - 1);
  return plus ? std::optional<int64_t>(*plus / divisor) : std::nullopt;
}

static std::optional<int64_t> mxAlignTo(int64_t value, int64_t alignment) {
  auto quotient = mxCeilDiv(value, alignment);
  if (!quotient || *quotient > std::numeric_limits<int64_t>::max() / alignment) {
    return std::nullopt;
  }
  return *quotient * alignment;
}

static std::optional<int64_t> mxCapacityElems(Type type) {
  auto shape = getShapeVec(type);
  return mxCheckedMul(shape[0], shape[1]);
}

static std::optional<int64_t> mxCapacityBytes(Type type) {
  auto elems = mxCapacityElems(type);
  unsigned bytes = getElemByteSize(getElemTy(type));
  if (!elems || bytes == 0 ||
      *elems >
          std::numeric_limits<int64_t>::max() / static_cast<int64_t>(bytes)) {
    return std::nullopt;
  }
  return *elems * static_cast<int64_t>(bytes);
}

static LogicalResult mxRequireCapacity(TQuantMxOp op, StringRef name, Type type,
                                       int64_t required) {
  auto actual = mxCapacityElems(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " elements";
  }
  return success();
}

static LogicalResult mxRequireCapacityBytes(TQuantMxOp op, StringRef name,
                                            Type type, int64_t required) {
  auto actual = mxCapacityBytes(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " bytes";
  }
  return success();
}

static LogicalResult mxRequireCompact(TQuantMxOp op, StringRef name, Type type) {
  auto valid = getValidShapeVec(type);
  auto physical = getShapeVec(type);
  if (valid[0] != 1 && physical[1] != valid[1]) {
    return op.emitOpError() << "expects " << name
                            << " valid elements to form a compact physical prefix";
  }
  return success();
}

static LogicalResult mxRequireStaticShape(TQuantMxOp op, StringRef name,
                                          Type type) {
  for (int64_t dim : getValidShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  for (int64_t dim : getShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxTilesAndForm(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst", /*allowLowPrecision=*/true)) ||
      failed(verifyNDStyleVecTile(op, expTy, "exp")) ||
      failed(verifyNDStyleVecTile(op, maxTy, "max")) ||
      failed(verifyNDStyleVecTile(op, scalingTy, "scaling"))) {
    return failure();
  }
  if (op.getExpZz() &&
      failed(verifyNDStyleVecTile(op, op.getExpZz().getType(), "exp_zz"))) {
    return failure();
  }
  const bool isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  if (op.getInterleave() && !isDn) {
    return op.emitOpError("expects interleave to be used only with grpAxis=axis0");
  }
  if (op.getExpZz() && isDn) {
    return op.emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; use pto.tmov with a non-scaling tmp for axis0 exponents");
  }
  auto quantType = op.getQuantType();
  if (quantType != mlir::pto::QuantType::MXFP8 &&
      quantType != mlir::pto::QuantType::MXFP4_E2M1) {
    return op.emitOpError("expects quant_type to be MXFP8 or MXFP4_E2M1");
  }
  if (op.getExpZz() && !op.getStoreMode()) {
    return op.emitOpError("expects storeMode when exp_zz is present");
  }
  if (op.getStoreMode() && !op.getExpZz()) {
    return op.emitOpError("expects exp_zz when storeMode is present");
  }
  if (op.getStoreMode() && op.getQuantScaleAlg() != mlir::pto::QuantScaleAlg::OCP) {
    return op.emitOpError("storeMode form must not override quantScaleAlg");
  }
  return success();
}

static LogicalResult verifyTQuantMxElemTypes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(op.getDst().getType());
  Type expElem = getElemTy(op.getExp().getType());
  Type maxElem = getElemTy(op.getMax().getType());
  Type scalingElem = getElemTy(op.getScaling().getType());
  if (!(srcElem.isF32() || srcElem.isF16() || srcElem.isBF16())) {
    return op.emitOpError("expects src element type to be f32/f16/bf16");
  }
  if (!expElem.isInteger(8)) {
    return op.emitOpError("expects exp element type to be i8/ui8");
  }
  if (op.getExpZz() && !getElemTy(op.getExpZz().getType()).isInteger(8)) {
    return op.emitOpError("expects exp_zz element type to be i8/ui8");
  }
  if (maxElem != srcElem) {
    return op.emitOpError("expects max element type to match src element type");
  }
  if (scalingElem != srcElem) {
    return op.emitOpError("expects scaling element type to match src element type");
  }
  if (op.getQuantType() == mlir::pto::QuantType::MXFP8) {
    if (!dstElem.isInteger(8)) {
      return op.emitOpError("expects MXFP8 dst element type to be i8/ui8");
    }
  } else {
    if (!isa<pto::F4E2M1x2Type>(dstElem)) {
      return op.emitOpError("expects MXFP4_E2M1 dst element type to be !pto.f4E2M1x2");
    }
    if (!(srcElem.isF16() || srcElem.isBF16())) {
      return op.emitOpError("expects MXFP4_E2M1 src element type to be f16/bf16");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto expValid = getValidShapeVec(expTy);
  auto maxValid = getValidShapeVec(maxTy);
  auto scalingValid = getValidShapeVec(scalingTy);
  if (srcValid.size() != 2 || dstValid.size() != 2 || expValid.size() != 2 ||
      maxValid.size() != 2 || scalingValid.size() != 2 ||
      getShapeVec(srcTy).size() != 2 || getShapeVec(dstTy).size() != 2 ||
      getShapeVec(expTy).size() != 2 || getShapeVec(maxTy).size() != 2 ||
      getShapeVec(scalingTy).size() != 2) {
    return op.emitOpError("expects rank-2 valid and physical shapes for MX quantization");
  }
  if (op.getExpZz() && (getValidShapeVec(op.getExpZz().getType()).size() != 2 ||
                        getShapeVec(op.getExpZz().getType()).size() != 2)) {
    return op.emitOpError("expects rank-2 valid and physical shapes for exp_zz");
  }
  for (auto [name, type] : {std::pair<StringRef, Type>("src", srcTy),
                            {"dst", dstTy}, {"exp", expTy}, {"max", maxTy},
                            {"scaling", scalingTy}}) {
    if (failed(mxRequireStaticShape(op, name, type))) {
      return failure();
    }
  }
  if (op.getExpZz()) {
    Type expZzTy = op.getExpZz().getType();
    if (llvm::is_contained(getValidShapeVec(expZzTy), ShapedType::kDynamic) ||
        llvm::is_contained(getShapeVec(expZzTy), ShapedType::kDynamic)) {
      return op.emitOpError("expects static valid and physical shapes for exp_zz in the deprecated fused MX quantization form");
    }
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, maxTy, "src", "max")) ||
      failed(verifyTileBufSameElemType(op, srcTy, scalingTy, "src", "scaling")) ||
      failed(verifyTileBufSameLogicalExtent(op, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  return success();
}

struct TQuantMxA5 {
  Type srcTy, dstTy, expTy, maxTy, scalingTy;
  Type srcElem;
  SmallVector<int64_t, 4> dstValid, expValid, expPhysical;
  SmallVector<int64_t, 4> dstPhysical, maxPhysical, scalingPhysical;
  bool isDn, isMxFp4;
  int64_t srcRows, srcCols, srcPhysicalRows, srcPhysicalCols;
  int64_t pack, dstValidCols, groups;
};

static LogicalResult mxRequireGroupedShape(TQuantMxOp op, const TQuantMxA5 &s,
                                           StringRef name, Type type,
                                           bool allowLegacy) {
  auto valid = getValidShapeVec(type);
  SmallVector<int64_t, 2> canonical = {s.isDn ? s.srcRows / 32 : s.srcRows,
                                       s.isDn ? s.srcCols : s.srcCols / 32};
  if (llvm::equal(valid, canonical)) {
    return success();
  }
  if (!s.isDn && allowLegacy && valid[0] == 1 && valid[1] == s.groups) {
    return success();
  }
  return op.emitOpError()
         << "expects " << name << " valid_shape to match "
         << (s.isDn ? "canonical [M/32, N] for grpAxis=axis0"
                    : "canonical [M, N/32] or legacy flat [1, M*N/32] for grpAxis=axis1");
}

static FailureOr<TQuantMxA5> buildTQuantMxA5State(TQuantMxOp op) {
  TQuantMxA5 s;
  s.srcTy = op.getSrc().getType();
  s.dstTy = op.getDst().getType();
  s.expTy = op.getExp().getType();
  s.maxTy = op.getMax().getType();
  s.scalingTy = op.getScaling().getType();
  s.srcElem = getElemTy(s.srcTy);
  auto srcValid = getValidShapeVec(s.srcTy);
  s.dstValid = getValidShapeVec(s.dstTy);
  s.expValid = getValidShapeVec(s.expTy);
  auto srcPhysical = getShapeVec(s.srcTy);
  s.dstPhysical = getShapeVec(s.dstTy);
  s.expPhysical = getShapeVec(s.expTy);
  s.maxPhysical = getShapeVec(s.maxTy);
  s.scalingPhysical = getShapeVec(s.scalingTy);
  s.isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  s.srcRows = srcValid[0];
  s.srcCols = srcValid[1];
  s.srcPhysicalRows = srcPhysical[0];
  s.srcPhysicalCols = srcPhysical[1];
  if (s.srcRows <= 0 || s.srcCols <= 0 || s.srcPhysicalRows < s.srcRows ||
      s.srcPhysicalCols < s.srcCols) {
    return op.emitOpError("expects positive source valid shape within physical shape");
  }
  if ((s.isDn ? s.srcRows : s.srcCols) % 32 != 0) {
    return op.emitOpError() << "expects src valid_shape[" << (s.isDn ? 0 : 1)
                            << "] to be a multiple of 32 when grpAxis is "
                            << (s.isDn ? "axis0" : "axis1");
  }
  auto groupsOr = mxCheckedMul(s.srcRows, s.srcCols / 32);
  if (!groupsOr) {
    return op.emitOpError("cannot compute MX quantization group count without overflow");
  }
  s.groups = *groupsOr;
  s.isMxFp4 = op.getQuantType() == mlir::pto::QuantType::MXFP4_E2M1;
  s.pack = s.isMxFp4 ? 2 : 1;
  s.dstValidCols = s.isMxFp4 ? s.srcCols / 2 : s.srcCols;
  return s;
}

static LogicalResult verifyTQuantMxGrouping(TQuantMxOp op, const TQuantMxA5 &s) {
  if (op.getExpZz()) {
    auto expZzElements =
        mxCheckedMul(getValidShapeVec(op.getExpZz().getType())[0],
                     getValidShapeVec(op.getExpZz().getType())[1]);
    if (!expZzElements || *expZzElements != s.groups) {
      return op.emitOpError("expects exp_zz valid element count to equal MX group count");
    }
  }
  if (failed(mxRequireGroupedShape(op, s, "max", s.maxTy, /*allowLegacy=*/true)) ||
      failed(mxRequireGroupedShape(op, s, "scaling", s.scalingTy,
                                   /*allowLegacy=*/true))) {
    return failure();
  }
  if (!op.getInterleave()) {
    return mxRequireGroupedShape(op, s, "exp", s.expTy, /*allowLegacy=*/true);
  }
  if (s.srcRows % 64 != 0) {
    return op.emitOpError("expects src valid rows to be a multiple of 64 when interleave is true");
  }
  if (s.srcPhysicalRows % 64 != 0) {
    return op.emitOpError("expects src physical rows to be a multiple of 64 when interleave is true");
  }
  auto doubledValidCols = mxCheckedMul(s.srcCols, 2);
  if (!doubledValidCols) {
    return op.emitOpError("cannot compute interleaved exp valid shape without overflow");
  }
  if (s.expValid[0] != s.srcRows / 64 || s.expValid[1] != *doubledValidCols) {
    return op.emitOpError("expects exp valid_shape to match [M/64, 2N] for grpAxis=axis0 with interleave=true");
  }
  return success();
}

static LogicalResult verifyTQuantMxDstShape(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4 && s.srcPhysicalCols % 2 != 0) {
    return op.emitOpError("expects MXFP4 src physical cols to be even for packed destination addressing");
  }
  if (s.dstValid[0] != s.srcRows || s.dstValid[1] != s.dstValidCols) {
    return op.emitOpError() << "expects dst valid_shape to be [" << s.srcRows
                            << ", " << s.dstValidCols << "] for MX quantization";
  }
  if (s.dstPhysical[0] < s.srcRows) {
    return op.emitOpError("expects dst physical rows to cover src valid rows");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4) {
    if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
      return op.emitOpError("expects MXFP4 axis0 dst physical cols to equal src physical cols / 2");
    }
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
    if ((s.srcPhysicalCols / s.pack) % 32 != 0 && s.srcElem.isF16()) {
      return op.emitOpError("does not support FP16 MXFP4 axis0 when packed source stride is not a multiple of 32 bytes");
    }
  } else {
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.dstPhysical[1]);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
  }
  if (s.maxPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects max physical cols to equal src physical cols for grpAxis=axis0");
  }
  if (s.scalingPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects scaling physical cols to equal src physical cols for grpAxis=axis0");
  }
  auto auxRequired = mxCheckedMul(s.srcRows / 32, s.srcPhysicalCols);
  if (!auxRequired || failed(mxRequireCapacity(op, "max", s.maxTy, *auxRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, *auxRequired))) {
    return failure();
  }
  if (!op.getInterleave()) {
    if (s.expPhysical[1] != s.srcPhysicalCols) {
      return op.emitOpError("expects exp physical cols to equal src physical cols for grpAxis=axis0");
    }
    if (failed(mxRequireCapacity(op, "exp", s.expTy, *auxRequired))) {
      return failure();
    }
  } else {
    auto doubledPhysicalCols = mxCheckedMul(s.srcPhysicalCols, 2);
    auto alignedPhysicalCols =
        doubledPhysicalCols ? mxAlignTo(*doubledPhysicalCols, 32) : std::nullopt;
    if (!alignedPhysicalCols) {
      return op.emitOpError("cannot compute interleaved exp physical cols without overflow");
    }
    if (s.expPhysical[0] != s.srcPhysicalRows / 64) {
      return op.emitOpError("expects interleaved exp physical rows to be src physical rows / 64");
    }
    if (s.expPhysical[1] != *alignedPhysicalCols) {
      return op.emitOpError("expects interleaved exp physical cols to be align32(2 * src physical cols)");
    }
  }
  if (s.srcElem.isF32() && op.getInterleave()) {
    return op.emitOpError("does not support FP32 interleave with the pinned pto-isa revision");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Flat(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.srcPhysicalCols != s.srcCols) {
    return op.emitOpError("expects axis1 flat exp to use a tight source with physical cols equal to valid cols");
  }
  if (s.expValid[0] != 1 || s.expValid[1] != s.groups) {
    return op.emitOpError("expects axis1 flat exp valid_shape to match legacy flat [1, M*N/32]");
  }
  if (failed(mxRequireCapacity(op, "exp", s.expTy, s.groups))) {
    return failure();
  }
  auto srcElems = mxCheckedMul(s.srcPhysicalRows, s.srcPhysicalCols);
  auto validElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
  bool unroll = srcElems && validElems && *srcElems > 1024 &&
                *srcElems % 256 == 0 && *validElems % 256 == 0;
  if (s.srcElem.isF32()) {
    auto scaleGroups =
        unroll ? mxCheckedMul(s.groups, 2) : std::optional<int64_t>(s.groups);
    auto aligned = scaleGroups ? mxAlignTo(*scaleGroups, 64) : std::nullopt;
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 4) : std::nullopt;
    StringRef scalingName = unroll ? "axis1 flat unrolled f32 scaling"
                                   : "axis1 flat f32 scaling";
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, scalingName, s.scalingTy,
                                      *requiredBytes))) {
      return failure();
    }
  } else if (op.getQuantScaleAlg() == pto::QuantScaleAlg::OCP) {
    auto aligned = mxAlignTo(s.groups, 128);
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 2) : std::nullopt;
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, "axis1 flat B16 OCP scaling",
                                      s.scalingTy, *requiredBytes))) {
      return failure();
    }
  } else {
    return op.emitOpError("does not support axis1 flat B16 NV quantization with the pinned pto-isa revisions");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Canonical(TQuantMxOp op,
                                                  const TQuantMxA5 &s) {
  if (s.expValid[0] == 1) {
    return op.emitOpError("expects legacy flat exp to use physical rows == 1");
  }
  if (s.srcElem.isF16() || s.srcElem.isBF16()) {
    return op.emitOpError("does not support axis1 canonical 2D B16 quantization with the pinned pto-isa revision");
  }
  SmallVector<int64_t, 2> canonicalShape = {s.srcRows, s.srcCols / 32};
  if (!llvm::equal(s.expValid, canonicalShape)) {
    return op.emitOpError("expects exp valid_shape to match canonical [M, N/32] for grpAxis=axis1");
  }
  if (s.expPhysical[0] < s.srcRows || s.expPhysical[1] < s.srcCols / 32) {
    return op.emitOpError("expects axis1 canonical exp physical shape to cover [M, N/32]");
  }
  auto expPrefix = mxCheckedMul(s.srcRows - 1, s.expPhysical[1]);
  auto expRequired =
      expPrefix ? mxCheckedAdd(*expPrefix, s.srcCols / 32) : std::nullopt;
  if (!expRequired || failed(mxRequireCapacity(op, "exp", s.expTy, *expRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, s.groups))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
    if (s.isMxFp4) {
      return op.emitOpError("expects MXFP4 axis1 dst physical cols to equal src physical cols / 2");
    }
    return op.emitOpError("expects MXFP8 axis1 dst physical cols to equal src physical cols");
  }
  auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
  auto dstRequired =
      dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
  if (!dstRequired || failed(mxRequireCapacity(op, "dst", s.dstTy, *dstRequired))) {
    return failure();
  }
  if (failed(mxRequireCompact(op, "max", s.maxTy)) ||
      failed(mxRequireCompact(op, "scaling", s.scalingTy)) ||
      failed(mxRequireCapacity(op, "max", s.maxTy, s.groups))) {
    return failure();
  }
  const bool flat = s.expPhysical[0] == 1;
  if (flat) {
    return verifyTQuantMxAxis1Flat(op, s);
  }
  return verifyTQuantMxAxis1Canonical(op, s);
}

static LogicalResult verifyTQuantMxSrcPadding(TQuantMxOp op,
                                              const TQuantMxA5 &s) {
  if ((s.srcElem.isF16() || s.srcElem.isBF16()) &&
      s.srcCols < s.srcPhysicalCols && 128 % s.srcPhysicalCols == 0) {
    auto validPhysicalElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
    if (!validPhysicalElems) {
      return op.emitOpError("cannot compute B16 source padding extent without overflow");
    }
    if (*validPhysicalElems % 128 != 0) {
      return op.emitOpError("does not support padded B16 source whose VL-aligned padding store has an incomplete final VL");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxA5(TQuantMxOp op) {
  if (failed(verifyTQuantMxTilesAndForm(op)) ||
      failed(verifyTQuantMxElemTypes(op)) || failed(verifyTQuantMxShapes(op))) {
    return failure();
  }
  auto stateOr = buildTQuantMxA5State(op);
  if (failed(stateOr)) {
    return failure();
  }
  const TQuantMxA5 &s = *stateOr;
  if (failed(verifyTQuantMxGrouping(op, s)) ||
      failed(verifyTQuantMxDstShape(op, s))) {
    return failure();
  }
  if (s.isDn) {
    if (failed(verifyTQuantMxAxis0(op, s))) {
      return failure();
    }
  } else {
    if (failed(verifyTQuantMxAxis1(op, s))) {
      return failure();
    }
  }
  return verifyTQuantMxSrcPadding(op, s);
}

mlir::LogicalResult mlir::pto::TQuantMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tquant.mx is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantMxA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16)) {
      return emitOpError()
             << "expects src element type i8 or i16";
    }
    if (!getElemTy(getDst().getType()).isF32()) {
      return emitOpError() << "expects dst element type f32";
    }
    if (!getElemTy(getScale().getType()).isF32()) {
      return emitOpError() << "expects scale element type f32";
    }
    if (!getElemTy(getOffset().getType()).isF32()) {
      return emitOpError() << "expects offset element type f32";
    }
    return success();
  };

  if (failed(verifyStructural())) {
    return failure();
  }

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst"))) {
      return failure();
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst()) {
    return emitOpError("expects A3 trecip src and dst to use different storage");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << errorMessage;
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemNoTmp(TRemOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trem element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemTmpA2A3(TRemOp op, Type tmpTy, Type elem) {
  Type dstTy = op.getDst().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 2) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 2");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(dstTy);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trem dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tmpTy, static_cast<uint64_t>(2) * static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmpA5(TRemOp op, Type tmpTy, Type elem) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmp(TRemOp op, Type elem) {
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemTmpA2A3(op, tmpTy, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemTmpA5(op, tmpTy, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src0, src1, and dst to use row-major layout");
  }

  Type elem = getElemTy(src0Ty);
  if (!getTmp()) {
    return verifyTRemNoTmp(*this, elem);
  }
  return verifyTRemTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static LogicalResult verifyTRemSNoTmp(TRemSOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trems element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemSTmpA2A3(TRemSOp op, Type tt, Type elem) {
  Type td = op.getDst().getType();
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (getElemTy(tt) != getElemTy(td)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(td);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trems dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmpA5(TRemSOp op, Type tt, Type elem) {
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmp(TRemSOp op, Type elem) {
  Type tt = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemSTmpA2A3(op, tt, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemSTmpA5(op, tt, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, ts, td, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(ts) || !isRowMajorTileBuf(td)) {
    return emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = getElemTy(ts);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }
  if (!getTmp()) {
    return verifyTRemSNoTmp(*this, elem);
  }
  return verifyTRemSTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpShape(Operation *op, Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufSameElemType(op, tmpTy, dstTy, "tmp", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op->emitOpError("expects tmp to use row-major layout");
  }
  return verifyTileBufSameValidShape(op, tmpTy, dstTy, "tmp", "dst");
}

static LogicalResult verifyTPowElemType(TPowOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpow element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpow element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpow element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmp(TPowOp op, Type dstTy, bool isIntElem) {
  if (isIntElem && op.getTmp()) {
    return op.emitOpError(
        "does not accept tmp when element type is integer (the integer pow "
        "lowering uses the 3-operand form TPOW(dst, base, exp))");
  }
  if (auto tmp = op.getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op.getOperation(), tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op.getOperation()) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op.emitOpError(
            "expects A2/A3 tpow dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPowOp::verify() {
  Type baseTy = getBase().getType();
  Type expTy = getExp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, baseTy, "base")) ||
      failed(verifyTileBufCommon(*this, expTy, "exp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameElemType(*this, baseTy, dstTy, "base", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, dstTy, "base", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(baseTy) || !isRowMajorTileBuf(expTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects base, exp, and dst to use row-major layout");
  }

  Type elem = getElemTy(baseTy);
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowTmp(*this, dstTy, isIntElem);
}

static LogicalResult verifyTPowSElemType(TPowSOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpows element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpows element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpows element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowSTmp(TPowSOp op, Type dstTy, bool isIntElem) {
  if (isIntElem && op.getTmp()) {
    return op.emitOpError(
        "does not accept tmp when element type is integer (the integer pows "
        "lowering uses the 3-operand form TPOWS(dst, src, scalar))");
  }
  if (auto tmp = op.getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op.getOperation(), tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op.getOperation()) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op.emitOpError(
            "expects A2/A3 tpows dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPowSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  // Same dtype matrix as TPowOp; see comment in TPowOp::verify.
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowSElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowSTmp(*this, dstTy, isIntElem);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic) {
      return std::nullopt;
    }
    if (d < 0) {
      return std::nullopt;
    }
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy) {
    return std::nullopt;
  }
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16()) {
      return 2;
    }
    if (ft.isF32()) {
      return 4;
    }
    if (ft.isF64()) {
      return 8;
    }
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0) {
      return std::nullopt;
    }
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value)) {
    return false;
  }

  if (isa<AllocTileOp, DeclareTileOp>(value.getDefiningOp())) {
    return true;
  }

  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    return isLocallyBoundTileSource(bitcast.getSrc());
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    return isLocallyBoundTileSource(reshape.getSrc());
  }

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return cOp.value();
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    return cInt.value();
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      return ia.getInt();
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndexLike(castOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndexLike(truncOp.getIn());
  }
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != 2) {
    return emitOpError("expects rank-2 tile_buf source");
  }

  ArrayRef<int64_t> validShape = srcTy.getValidShape();
  if (validShape.size() != 2) {
    return emitOpError("expects source validShape to be rank-2");
  }
  if (!srcTy.hasDynamicValid()) {
    return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");
  }

  shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

  if (!isLocallyBoundTileSource(getSource())) {
    return emitOpError(
        "requires a locally bound tile source; function arguments/results "
        "are unsupported");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal) {
      return success();
    }

    if (*constVal < 0) {
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    }
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic) {
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    }
    return success();
  };

  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row"))) {
    return failure();
  }
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col"))) {
    return failure();
  }

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != 2) {
    return emitOpError("expects rank-2 tile_buf source");
  }
  if (srcTy.getValidShape().size() != 2) {
    return emitOpError("expects source validShape to be rank-2");
  }
  return success();
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb) {
    return emitOpError("expects src/result to be !pto.tile_buf types");
  }

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst"))) {
    return failure();
  }

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace()) {
    return emitOpError("expects src and dst to use the same loc");
  }

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value()) {
    return emitOpError("failed to get element byte width for src/dst");
  }

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value()) {
    return emitOpError("expects static shapes for treshape");
  }

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value()) {
    return emitOpError("expects src and dst to have the same total byte size");
  }

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed) {
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");
  }

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace()) {
    return emitOpError("expects src/result to have the same memorySpace");
  }

  if (srcTy.getElementType() == dstTy.getElementType()) {
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");
  }

  if (srcTy.getShape() != dstTy.getShape()) {
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");
  }

  if (srcTy.getValidShape() != dstTy.getValidShape()) {
    return emitOpError("expects src/result to have the same validShape");
  }

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg) {
    return emitOpError("expects src/result to have the same tile config");
  }

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value()) {
    return emitOpError("expects static shapes for bitcast");
  }

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value()) {
    return emitOpError("unsupported element type for bitcast");
  }

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes) {
    return emitOpError("bitcast result requires more bytes than source storage");
  }

  return success();
}


static LogicalResult verifyTRowExpandCommon(TRowExpandOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
    return failure();
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src to be in the vec address space");
  }
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op.emitOpError("expects src to use the none_box slayout");
    }
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return op.emitOpError("expects trowexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(op.getSrc());
  auto dstValid = getValidShapeVec(op.getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
    return op.emitOpError("expects src valid_shape[0] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
    return op.emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0) {
    return op.emitOpError("expects dst valid_shape[0] to be non-zero");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0) {
    return op.emitOpError("expects dst valid_shape[1] to be non-zero");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verify = [&]() -> LogicalResult { return verifyTRowExpandCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verify, verify);
}


ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(idx)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(tmp)) {
        return failure();
      }
      hasTmp = true;
    }
  } else {
    return failure();
  }
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands)) {
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

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TSort32Op::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx();
  if (getTmp()) {
    p << ", " << getTmp();
    p << " : " << getSrc().getType() << ", " << getIdx().getType()
      << ", " << getTmp().getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getIdx().getType() << ")";
  }
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

// TPOW assembly format (mirrors TRsqrt's optional-tmp style):
//   pto.tpow ins(%base, %exp[, %tmp] : !tile, !tile[, !tile])
//            outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand base, exp, tmp, dst;
  Type baseTy, expTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(base) || parser.parseComma() ||
      parser.parseOperand(exp)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(baseTy) || parser.parseComma() || parser.parseType(expTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(base, baseTy, result.operands) ||
      parser.resolveOperand(exp, expTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}
