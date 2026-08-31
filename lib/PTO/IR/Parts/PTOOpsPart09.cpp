// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void mlir::pto::TPowOp::print(OpAsmPrinter &p) {
  p << " ins(" << getBase() << ", " << getExp();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getBase().getType() << ", " << getExp().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

// TPOWS assembly format:
//   pto.tpows ins(%src, %scalar[, %tmp] : !tile, scalar_t[, !tile])
//             outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, scalar, tmp, dst;
  Type srcTy, scalarTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(scalar)) {
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
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(scalarTy)) {
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
      parser.resolveOperand(scalar, scalarTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPowSOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getScalar();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType() << ", " << getScalar().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() || parser.parseOperand(src1)) {
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
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty)) {
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

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands)) {
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

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  p << " ins(" << src0 << ", " << src1;
  if (tmp) {
    p << ", " << tmp;
    p << " : " << src0.getType() << ", " << src1.getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << src0.getType() << ", " << src1.getType() << ")";
  }
  p << " outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRowExpandDivOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandDivOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMulOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandAddOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandAddOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

enum class TRowExpandBinaryMode {
  Unknown,
  Mode1ColMajorScalar,
  Mode2RowMajorBlock,
};

static bool validShapesCompatibleForTRowExpand(ArrayRef<int64_t> lhs,
                                               ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
      return false;
    }
  }
  return true;
}

static TRowExpandBinaryMode classifyTRowExpandBinaryMode(Type src0Ty,
                                                         Type src1Ty,
                                                         Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return TRowExpandBinaryMode::Unknown;
  }

  Type expandedTy;
  ArrayRef<int64_t> expandedValid;
  if (validShapesCompatibleForTRowExpand(src0Valid, dstValid)) {
    expandedTy = src1Ty;
    expandedValid = src1Valid;
  } else if (validShapesCompatibleForTRowExpand(src1Valid, dstValid)) {
    expandedTy = src0Ty;
    expandedValid = src0Valid;
  } else {
    return TRowExpandBinaryMode::Unknown;
  }

  int64_t expandedCols = expandedValid[1];
  if (isColMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic || expandedCols == 1)) {
    return TRowExpandBinaryMode::Mode1ColMajorScalar;
  }

  std::optional<int64_t> elemBytes = getElemBytes(getElemTy(dstTy));
  if (!elemBytes || *elemBytes == 0) {
    return TRowExpandBinaryMode::Unknown;
  }
  int64_t expectedMode2Cols = 32 / *elemBytes;
  if (isRowMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic ||
       expandedCols == expectedMode2Cols)) {
    return TRowExpandBinaryMode::Mode2RowMajorBlock;
  }

  return TRowExpandBinaryMode::Unknown;
}

static int64_t getTRowExpandTmpMinBytes(int64_t dstValidRows) {
  if (dstValidRows == ShapedType::kDynamic) {
    return 8192;
  }
  if (dstValidRows < 0) {
    return 8192;
  }
  if (dstValidRows < 256) {
    return ceilDivInt64(dstValidRows, 8) * 256;
  }
  return 30 * 256;
}

static std::optional<int64_t> getStaticTileCapacityBytes(Type ty) {
  auto numElems = getStaticNumElements(getShapeVec(ty));
  auto elemBytes = getElemBytes(getElemTy(ty));
  if (!numElems || !elemBytes) {
    return std::nullopt;
  }
  return *numElems * *elemBytes;
}

static LogicalResult verifyTRowExpandImplicitTmpContract(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch) {
  if (!hasTmp || targetArch == PTOArch::A5) {
    return success();
  }

  if (classifyTRowExpandBinaryMode(src0Ty, src1Ty, dstTy) !=
      TRowExpandBinaryMode::Mode1ColMajorScalar) {
    return op->emitOpError(
        "expects A2/A3 tmp-form trowexpand to use mode 1 "
        "(ColMajor per-row scalar expanded operand)");
  }

  if (failed(verifyVecTileStorage(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects tmp and dst to have the same element type");
  }

  auto dstValid = getValidShapeVec(dstTy);
  if (dstValid.size() != 2) {
    return op->emitOpError("expects dst to have rank-2 valid_shape");
  }
  int64_t minBytes = getTRowExpandTmpMinBytes(dstValid[0]);
  std::optional<int64_t> tmpBytes = getStaticTileCapacityBytes(tmpTy);
  if (!tmpBytes) {
    return op->emitOpError(
        "expects A2/A3 trowexpand tmp capacity to be statically known");
  }
  if (*tmpBytes < minBytes) {
    return op->emitOpError()
           << "expects A2/A3 trowexpand tmp capacity to be at least "
           << minBytes << " bytes, but got " << *tmpBytes << " bytes";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp()) {
      return emitOpError("expects tmp when precisionType is high_precision");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError(
          "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError(
          "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandAddCore(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
      op, src0Ty, src1Ty, dstTy, op.getTmp() ? op.getTmp().getType() : Type{},
      static_cast<bool>(op.getTmp()));
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty)) {
    return op.emitOpError("expects src0 to use row-major layout");
  }
  Type elem = *elemOr;
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem.isInteger(8));
  if (!supported) {
    if (targetArch == PTOArch::A5) {
      return op.emitOpError(
          "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
    }
    return op.emitOpError(
        "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
  }
  return elem;
}

static LogicalResult verifyTRowExpandAddSrc1(TRowExpandAddOp op, Type elem,
                                             PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src1 and dst to have rank-2 valid_shape");
  }
  if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
  }
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol) {
      return op.emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    }
  } else {
    if (src1Col != ShapedType::kDynamic && src1Col != 1) {
      return op.emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
  }
  if (failed(verifyTRowExpandImplicitTmpContract(
          op.getOperation(), src0Ty, src1Ty, dstTy,
          op.getTmp() ? op.getTmp().getType() : Type{},
          static_cast<bool>(op.getTmp()), targetArch))) {
    return failure();
  }
  return mlir::success();
}

static LogicalResult verifyTRowExpandAddByArch(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  FailureOr<Type> elemOr = verifyTRowExpandAddCore(op, targetArch);
  if (failed(elemOr)) {
    return failure();
  }
  return verifyTRowExpandAddSrc1(op, *elemOr, targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  bool enforceTmpContract,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasTmp) {
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != getElemTy(dstTy)) {
      return op->emitOpError() << "expects tmp and dst to have the same element type";
    }
  }

  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem) {
    return op->emitOpError("expects src0, src1, and dst to have the same element type");
  }
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (!supported) {
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

  if (!isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects dst to use row-major layout");
  }

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }

  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0) {
    return op->emitOpError("expects dst valid_shape[0] to be non-zero");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0) {
    return op->emitOpError("expects dst valid_shape[1] to be non-zero");
  }

  auto validShapeMatches = [](ArrayRef<int64_t> lhs,
                              ArrayRef<int64_t> rhs) -> bool {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (auto [l, r] : llvm::zip(lhs, rhs)) {
      if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
        return false;
      }
    }
    return true;
  };

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);

  auto checkBroadcastOperand = [&](Type operandTy, ArrayRef<int64_t> operandValid,
                                   StringRef operandName,
                                   bool requireNonRowMajor) -> LogicalResult {
    if (operandValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        operandValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << operandName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    int64_t expectedCol = elem.isInteger(8) ? 32 : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t operandCol = operandValid[1];
    bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
    if (requireNonRowMajor && operandIsRowMajor) {
      return op->emitOpError() << "expects " << operandName
                               << " to use a non-row-major layout when tmp is present";
    }
    if (operandIsRowMajor) {
      if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
        return op->emitOpError()
               << "expects row-major " << operandName
               << " valid_shape[1] to be 32/sizeof(dtype)";
      }
      return success();
    }
    if (operandCol != ShapedType::kDynamic && operandCol != 1) {
      return op->emitOpError() << "expects non-row-major " << operandName
                               << " valid_shape[1] to be 1";
    }
    return success();
  };

  auto checkFullAndBroadcast = [&](Type fullTy, ArrayRef<int64_t> fullValid,
                                   StringRef fullName, Type broadcastTy,
                                   ArrayRef<int64_t> broadcastValid,
                                   StringRef broadcastName) -> LogicalResult {
    if (!isRowMajorTileBuf(fullTy)) {
      return op->emitOpError() << "expects " << fullName
                               << " to use row-major layout when it matches dst";
    }
    if (fullValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        fullValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    if (fullValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        fullValid[1] != dstValid[1]) {
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[1] to equal dst valid_shape[1]";
    }
    return checkBroadcastOperand(broadcastTy, broadcastValid, broadcastName,
                                 /*requireNonRowMajor=*/hasTmp &&
                                     targetArch == PTOArch::A3);
  };

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  auto verifyTmpContract = [&]() -> LogicalResult {
    if (!enforceTmpContract) {
      return success();
    }
    return verifyTRowExpandImplicitTmpContract(op, src0Ty, src1Ty, dstTy,
                                               tmpTy, hasTmp, targetArch);
  };

  if (src0MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src0Ty, src0Valid, "src0", src1Ty,
                                        src1Valid, "src1")) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }
  if (src1MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src1Ty, src1Valid, "src1", src0Ty,
                                        src0Valid, "src0")) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }

  return op->emitOpError() << "expects one of src0/src1 to match dst valid_shape"
                           << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static ParseResult parseOptionalTmpRowReductionOp(OpAsmParser &parser,
                                                  OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src)) {
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
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
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

static void printOptionalTmpRowReductionOp(OpAsmPrinter &p, Operation *op,
                                           Value src, Value tmp, Value dst) {
  p << " ins(" << src;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << src.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseOptionalTmpFixedDpsOp(
    OpAsmParser &parser, OperationState &result, unsigned minInputs,
    unsigned maxInputs, ArrayRef<int32_t> noTmpSegments,
    ArrayRef<int32_t> withTmpSegments) {
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SmallVector<Type> inputTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstType;
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    inputs.emplace_back();
    if (parser.parseOperand(inputs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  if (inputs.size() < minInputs || inputs.size() > maxInputs ||
      parser.parseColon()) {
    return failure();
  }
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (i && parser.parseComma()) {
      return failure();
    }
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    inputTypes.push_back(type);
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstType) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(inputs, inputTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          inputs.size() == minInputs ? noTmpSegments : withTmpSegments));
  return success();
}

static void printOptionalTmpFixedDpsOp(OpAsmPrinter &p, Operation *op,
                                       ArrayRef<Value> inputs, Value dst) {
  p << " ins(";
  llvm::interleaveComma(inputs, p, [&](Value value) { p << value; });
  p << " : ";
  llvm::interleaveComma(inputs, p,
                        [&](Value value) { p << value.getType(); });
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TTransOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 1, 2, {1, 0, 1},
                                    {1, 1, 1});
}
void mlir::pto::TTransOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TPReluOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TPReluOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TRemOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TRemSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc(), getScalar()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 3, 4,
                                    {1, 1, 1, 0, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 3, 4,
                                    {1, 1, 0, 1, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  inputs.push_back(getScalar());
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TColArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TColArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMaxOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMinOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowSumOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowSumOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowProdOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowProdOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  if (!getTmp()) {
    return verifyTRowArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  if (!getTmp()) {
    return verifyTRowArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }

    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }

    bool mismatchedValidShapes =
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedValidShapes) {
      return failure();
    }

    auto validShape = getValidShapeVec(dst0Ty);
    bool hasInvalidRank = validShape.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src0, src1, dst0, and dst1 to have rank-2 valid_shape");
    }
    bool hasOddValidColumns =
        validShape[1] != ShapedType::kDynamic && (validShape[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects valid_shape[1] to be even");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDeInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tdeinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }
    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }
    bool hasNonRowMajorTile =
        !isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(dst0Ty) ||
        !isRowMajorTileBuf(dst1Ty);
    if (hasNonRowMajorTile) {
      return emitOpError("expects src and dst tiles to use row-major layout");
    }

    auto src0Valid = getValidShapeVec(src0Ty);
    auto dst0Valid = getValidShapeVec(dst0Ty);
    auto dst1Valid = getValidShapeVec(dst1Ty);
    bool hasInvalidRank =
        src0Valid.size() != 2 || dst0Valid.size() != 2 || dst1Valid.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src and dst tiles to have rank-2 valid_shape");
    }

    bool hasSecondSource = getSrcs().size() == 2;
    if (hasSecondSource) {
      Type src1Ty = getSrc1().getType();
      bool invalidSource =
          failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
          failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
      if (invalidSource) {
        return failure();
      }
      if (!isRowMajorTileBuf(src1Ty)) {
        return emitOpError("expects src1 to use row-major layout");
      }
      bool hasOddValidColumns =
          src0Valid[1] != ShapedType::kDynamic && (src0Valid[1] & 1) != 0;
      if (hasOddValidColumns) {
        return emitOpError("expects two-source valid_shape[1] to be even");
      }
      return success();
    }

    bool hasRowMismatchWithDst0 =
        src0Valid[0] != ShapedType::kDynamic &&
        dst0Valid[0] != ShapedType::kDynamic &&
        src0Valid[0] != dst0Valid[0];
    if (hasRowMismatchWithDst0) {
      return emitOpError("expects src0 and dst0 to have the same valid_shape[0]");
    }
    bool hasRowMismatchWithDst1 =
        src0Valid[0] != ShapedType::kDynamic &&
        dst1Valid[0] != ShapedType::kDynamic &&
        src0Valid[0] != dst1Valid[0];
    if (hasRowMismatchWithDst1) {
      return emitOpError("expects src0 and dst1 to have the same valid_shape[0]");
    }
    bool hasOddValidColumns =
        src0Valid[1] != ShapedType::kDynamic && (src0Valid[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects single-source valid_shape[1] to be even");
    }
    bool hasColumnMismatchWithDst0 =
        src0Valid[1] != ShapedType::kDynamic &&
        dst0Valid[1] != ShapedType::kDynamic &&
        dst0Valid[1] != src0Valid[1] / 2;
    if (hasColumnMismatchWithDst0) {
      return emitOpError(
          "expects dst0 valid_shape[1] to be half of src0 valid_shape[1]");
    }
    bool hasColumnMismatchWithDst1 =
        src0Valid[1] != ShapedType::kDynamic &&
        dst1Valid[1] != ShapedType::kDynamic &&
        dst1Valid[1] != src0Valid[1] / 2;
    if (hasColumnMismatchWithDst1) {
      return emitOpError(
          "expects dst1 valid_shape[1] to be half of src0 valid_shape[1]");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A5 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32())) {
    return emitOpError("expects element type to be f16 or f32");
  }
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp()) {
    return emitOpError("expects tmp when precisionType is high_precision");
  }
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp"))) {
      return failure();
    }

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value()) {
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    }
    if (tmpElemBytes.value() * tmpNumel.value() < 32) {
      return emitOpError("expects tmp to be at least 32 bytes when provided");
    }
  }
  return mlir::success();
}


static bool isTScatterAllowedDataElem(mlir::Type t) {
  if (t.isF16() || t.isF32() || t.isBF16()) {
    return true;
  }
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
    return (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
  }
  return false;
}

static bool isTScatterAllowedIndexElem(mlir::Type t) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
    return (it.getWidth() == 16 || it.getWidth() == 32);
  }
  return false;
}

static unsigned getMaskScatterTimes(mlir::pto::MaskPatternAttr mp) {
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

static LogicalResult verifyTScatterIndexedElemTypes(TScatterOp op, Type ts,
                                                    Type ti, Type td) {
  Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
  if (!srcElem || !dstElem || !idxElem) {
    return op.emitOpError("failed to get element type for operands");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src/dst to have the same element type");
  }

  if (!isTScatterAllowedDataElem(srcElem)) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  if (!isTScatterAllowedIndexElem(idxElem)) {
    return op.emitOpError("expects indexes element type to be i16/i32");
  }

  auto bwData = getPTOStorageElemBitWidth(srcElem);
  auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
  if (bwData != 8 && bwData != 16 && bwData != 32) {
    return op.emitOpError("unexpected src/dst element bitwidth");
  }

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes) {
    return op.emitOpError("expects indexes element size to match the documented scatter rule");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedShapes(TScatterOp op, Type ts, Type ti,
                                                 Type td) {
  auto srcValid = getValidShapeVec(ts);
  auto idxValid = getValidShapeVec(ti);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src, indexes and dst to have rank-2 valid_shape");
  }

  for (unsigned d = 0; d < 2; ++d) {
    if (srcValid[d] != ShapedType::kDynamic && idxValid[d] != ShapedType::kDynamic &&
        srcValid[d] != idxValid[d]) {
      return op.emitOpError("expects src and indexes to have the same valid_shape");
    }
    if (srcValid[d] != ShapedType::kDynamic && dstValid[d] != ShapedType::kDynamic &&
        dstValid[d] < srcValid[d]) {
      return op.emitOpError("expects dst valid_shape to be >= src valid_shape");
    }
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type ti = op.getIndexes().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, ti, "indexes")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTScatterIndexedElemTypes(op, ts, ti, td))) {
    return failure();
  }
  return verifyTScatterIndexedShapes(op, ts, ti, td);
}

static LogicalResult verifyTScatterMaskAxisShapes(TScatterOp op, StringRef axisVal,
                                                  ArrayRef<int64_t> srcValid,
                                                  ArrayRef<int64_t> dstValid,
                                                  unsigned times) {
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != static_cast<int64_t>(srcValid[1] * times)) {
      return op.emitOpError("expects dst valid cols to equal src valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != static_cast<int64_t>(srcValid[0] * times)) {
      return op.emitOpError("expects dst valid rows to equal src valid rows times the mask expansion factor for col direction");
    }
  } else {
      return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterMaskForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }

  auto srcTB = dyn_cast<pto::TileBufType>(ts);
  auto dstTB = dyn_cast<pto::TileBufType>(td);
  if (!srcTB || !dstTB) {
    return op.emitOpError("expects src and dst to be tile_buf types");
  }

  if (getElemTy(ts) != getElemTy(td)) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (!isTScatterAllowedDataElem(getElemTy(ts))) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }

  auto srcValid = getValidShapeVec(ts);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }

  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tscatter to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tscatter to provide maskPattern");
  }
  const unsigned times = getMaskScatterTimes(mp);
  if (failed(verifyTScatterMaskAxisShapes(op, axisVal, srcValid, dstValid, times))) {
    return failure();
  }

  if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError("expects mask-pattern tscatter to use row_major blayout");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }
  if (hasIndexes && getAxisAttr()) {
    return emitOpError("axis attribute must not be provided with indexes operand");
  }
  auto verifyForm = [&]() -> LogicalResult {
    if (hasMaskPattern) {
      return verifyTScatterMaskForm(*this);
    }
    return verifyTScatterIndexedForm(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyForm, verifyForm);
}


static FailureOr<Type> verifyTSelCommon(TSelOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }
  if (op.getTmp() &&
      failed(verifyVecTileCommon(op, op.getTmp().getType(), "tmp"))) {
    return failure();
  }

  Type srcElem = getElemTy(t0);
  Type src1Elem = getElemTy(t1);
  Type dstElem = getElemTy(td);
  if (!srcElem || !src1Elem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != src1Elem || srcElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }

  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
      !isRowMajorTileBuf(td)) {
    op.emitOpError(
        "expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return srcElem;
}

static LogicalResult verifyTSelA2A3(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
    ok = it.getWidth() == 16 || it.getWidth() == 32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32");
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemByteSize(getElemTy(tmpTy)) != 4) {
      return op.emitOpError("expects A2/A3 tsel tmp element type to be 4 bytes wide");
    }
    unsigned elemBits = getPTOStorageElemBitWidth(elem);
    if (elemBits != 16 && elemBits != 32) {
      return op.emitOpError("expects A2/A3 tsel data element type to be 16 or 32 bits");
    }
    uint64_t minBytes = elemBits == 16 ? 16 : 8;
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, minBytes))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTSelA5(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
    ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static FailureOr<Type> verifyTSelSCommon(TSelSOp op) {
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tTmp = op.getTmp() ? op.getTmp().getType() : Type{};
  Type tDst = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, tMask, "mask")) ||
      failed(verifyTileBufCommon(op, tSrc, "src")) ||
      failed(verifyTileBufCommon(op, tDst, "dst"))) {
    return failure();
  }
  if (tTmp && failed(verifyTileBufCommon(op, tTmp, "tmp"))) {
    return failure();
  }
  Type eMask = getElemTy(tMask), eSrc = getElemTy(tSrc);
  Type eDst = getElemTy(tDst);
  if (!eMask || !eSrc || !eDst) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (eSrc != eDst) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyTileBufSameValidShape(op, tSrc, tDst, "src", "dst"))) {
    return failure();
  }
  return eDst;
}

static LogicalResult verifyTSelSA2A3(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError("expects A2/A3 tsels tmp to have the same element type as src and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects A2/A3 tsels tmp to use row-major layout");
    }
    auto srcShape = getShapeVec(tSrc);
    if (srcShape.size() != 2 || srcShape[1] == ShapedType::kDynamic) {
      return op.emitOpError(
          "expects A2/A3 tsels src shape to be static when tmp is provided");
    }
    auto elemBytes = getElemByteSize(elem);
    if (elemBytes == 0 ||
        failed(verifyTmpCapacityAtLeast(
            op, tmpTy, static_cast<uint64_t>(srcShape[1]) * elemBytes))) {
      return failure();
    }
  }
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
    ok = (it.getWidth() == 16 || it.getWidth() == 32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32");
  }
  return success();
}
