// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTSelSA5(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tMask) || !isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects mask, src, and dst to use row-major layout");
  }
  Type elem = *elemOr;
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
    ok = (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelSA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelSA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx"))) {
    return failure();
  }
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto requiredBytes = getStaticByteSize(srcTy);
    if (!requiredBytes) {
      return emitOpError(
          "expects A2/A3 tsort32 src shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(*this, getTmp().getType(),
                                        *requiredBytes))) {
      return failure();
    }
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!(srcElem.isF16() || srcElem.isF32())) {
    return emitOpError() << "expects src and dst element type to be f16 or f32";
  }

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return emitOpError() << "expects idx element type to be i32/u32";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem))) {
    return emitOpError() << "expects src and dst element type to be float or half";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size()) {
    return emitOpError() << "expects all tensors to have the same rank";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size()) {
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  }
  return mlir::success();
}
static bool ttransUsesTmp(Type srcTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  unsigned elemBytes = getPTOStorageElemByteSize(getElemTy(srcTy));
  if (srcShape.size() != 2 || dstShape.size() != 2 || elemBytes == 0 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) ||
      llvm::is_contained(dstShape, ShapedType::kDynamic)) {
    return true;
  }
  int64_t rowStride = elemBytes == 1 ? 32 : 16;
  int64_t elemPerBlock = 32 / elemBytes;
  int64_t srcStride = srcShape[1];
  int64_t dstStride = dstShape[1];
  return dstStride % rowStride == 0 && srcStride % elemPerBlock == 0 &&
         srcStride / elemPerBlock <= 255;
}

static bool ttransIsAllowedWidthType(Type ty, unsigned elemBytes) {
  if (elemBytes == 4) {
    return ty.isInteger(32) || ty.isF32();
  }
  if (elemBytes == 2) {
    return ty.isInteger(16) || ty.isF16() || ty.isBF16();
  }
  return ty.isInteger(8);
}

static LogicalResult verifyTTransElemWidth(TTransOp op, Type srcElem,
                                           unsigned &elemBytes) {
  elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    return op.emitOpError() << "failed to get transpose element size";
  }
  if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4) {
    return op.emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
  }
  if (!ttransIsAllowedWidthType(srcElem, elemBytes)) {
    return op.emitOpError() << "expects transpose element type to match the supported set for its width";
  }
  return success();
}

static LogicalResult verifyTTransAlignedMajor(TTransOp op, Type ty, StringRef name,
                                              unsigned elemBytes) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return success();
  }
  auto shape = getShapeVec(ty);
  if (shape.size() != 2) {
    return success();
  }
  bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % 32 != 0) {
    return op.emitOpError() << "expects " << name << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

static LogicalResult verifyTTransA2A3(TTransOp op) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError() << "expects src and dst to have the same element type";
  }
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op.emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  if (tmpTy) {
    uint64_t requiredBytes = 32;
    if (ttransUsesTmp(srcTy, dstTy)) {
      auto srcBytes = getStaticByteSize(srcTy);
      if (!srcBytes) {
        return op.emitOpError(
            "expects A2/A3 transpose src shape to be static when tmp is used");
      }
      requiredBytes = *srcBytes;
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, requiredBytes))) {
      return failure();
    }
  }
  return mlir::success();
}

static LogicalResult verifyTTransA5(TTransOp op) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError() << "expects src, tmp, and dst to have the same element type";
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  if (tmpTy && failed(verifyTmpCapacityAtLeast(op, tmpTy, 32))) {
    return failure();
  }
  if (failed(verifyTTransAlignedMajor(op, srcTy, "src", elemBytes)) ||
      failed(verifyTTransAlignedMajor(op, dstTy, "dst", elemBytes))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTTransA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTTransA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TXorOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() ||
      parser.parseOperand(src1)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(src0Ty) || parser.parseComma() ||
      parser.parseType(src1Ty)) {
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
  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TXorOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc0() << ", " << getSrc1();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc0().getType() << ", " << getSrc1().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TXorSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
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
  if (parser.parseColonType(srcTy) || parser.parseComma() ||
      parser.parseType(scalarTy)) {
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
  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(scalar, scalarTy, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TXorSOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getScalar();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType() << ", " << getScalar().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static LogicalResult verifyTXorA2A3(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError(
          "expects tmp to have the same element type as src0, src1, and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects tmp to use row-major layout");
    }
    if (failed(verifyTileBufSameValidShape(
            op, tmpTy, op.getDst().getType(), "tmp", "dst"))) {
      return failure();
    }
    auto requiredBytes = getStaticByteSize(op.getDst().getType());
    if (!requiredBytes) {
      return op.emitOpError(
          "expects A2/A3 txor dst shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
      return failure();
    }
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
              it.getWidth() != 32)) {
    return op.emitOpError(
        "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16/i32");
  }
  return success();
}

static LogicalResult verifyTXorA5(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  auto it = mlir::dyn_cast<IntegerType>(*elemOr);
  if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
              it.getWidth() != 32)) {
    return op.emitOpError(
        "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTXorA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTXorA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyTileBufCommon(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(tmpTy) != elem) {
        return emitOpError(
            "expects tmp to have the same element type as src and dst");
      }
      if (!isRowMajorTileBuf(tmpTy)) {
        return emitOpError("expects tmp to use row-major layout");
      }
      auto requiredBytes = getStaticByteSize(getDst().getType());
      if (!requiredBytes) {
        return emitOpError(
            "expects A2/A3 txors dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16)) {
      return emitOpError(
          "expects A2/A3 txors src and dst element type to be i8/i16");
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
          "expects A5 txors src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TPrintOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  Type srcTy, tmpTy;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

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
  if (parser.parseRParen()) {
    return failure();
  }
  if (failed(parsePTOInherentAttrs<TPrintOp>(
          parser, result, parsedAttrs, {"printFormat"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPrintOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (Value tmp = getTPrintTmpIfPresent(*this)) {
    p << ", " << tmp << " : " << getSrc().getType() << ", "
      << tmp.getType();
  } else {
    p << " : " << getSrc().getType();
  }
  p << ")";
  NamedAttrList attrs = getNonInherentAttrs(getOperation(), {"printFormat"});
  if (auto printFormatAttr =
          dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat)) {
    attrs.append("printFormat", printFormatAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  Value tmp = getTPrintTmpIfPresent(*this);
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError() << "expects printable tile element type";
}
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!tmp) {
      if (!space || *space != pto::AddressSpace::VEC) {
        return emitOpError() << "expects printable tile_buf without tmp to be in vec address space";
}
      return success();
    }

    if (!space) {
      return emitOpError() << "expects printable tile_buf with tmp to use a supported address space";
}
    if (*space == pto::AddressSpace::MAT && isTargetArchA5(getOperation())) {
      return emitOpError() << "expects mat tile printing with tmp only on A2/A3 targets";
}
    if (*space != pto::AddressSpace::VEC && *space != pto::AddressSpace::MAT &&
        *space != pto::AddressSpace::ACC) {
      return emitOpError() << "expects printable tile_buf with tmp to be in vec/mat/acc address space";
}
    if (failed(verifyMGatherMScatterMemOperand(getOperation(), tmp, elem, "tmp"))) {
      return failure();
}
    return success();
  }
  if (tmp) {
    return emitOpError() << "expects tmp only when src is a tile_buf";
}
  if (mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType)) {
    return mlir::success();
}
  return emitOpError() << "expects tile_buf or partition_tensor_view for src";
}



[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor ----
  if (auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType())) {
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!rhsTy) {
      return op->emitOpError("expects lhs and rhs to be ranked tensors");
    }

    if (lhsTy.getElementType() != rhsTy.getElementType()) {
      return op->emitOpError()
             << "expects lhs and rhs to have the same element type, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();
    }

    if (biasOpt) {
      auto biasTy = dyn_cast<RankedTensorType>(biasOpt.getType());
      if (!biasTy) {
        return op->emitOpError("expects bias to be a ranked tensor");
      }
      if (biasTy.getElementType() != lhsTy.getElementType()) {
        return op->emitOpError()
               << "expects bias to have the same element type as lhs and rhs, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
      }
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects dst to have the same element type as lhs and rhs, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();
    }

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects result to have the same element type as lhs and rhs, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();
    }

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile) {
    return op->emitOpError("expects lhs and rhs to be ranked tensors or !pto.tile");
  }

  if (lhsTile.getElementType() != rhsTile.getElementType()) {
    return op->emitOpError() << "expects lhs and rhs tiles to have the same element type, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();
  }

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2) {
    return op->emitOpError("expects lhs and rhs tiles to be 2D");
  }

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0]) {
    return op->emitOpError() << "expects lhs dim1 to equal rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];
  }

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile) {
      return op->emitOpError("expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    }
    if (biasTile.getElementType() != lhsTile.getElementType()) {
      return op->emitOpError("expects bias to have the same element type as lhs and rhs");
    }
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType()) {
    return op->emitOpError() << "expects dst to have the same element type as lhs and rhs";
  }

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType()) {
    return op->emitOpError() << "expects result to have the same element type as lhs and rhs";
  }

  return success();
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                      getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2) {
    return mlir::Type();
  }

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile) {
    return mlir::Type();
  }

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2) {
    return RankedTensorType();
  }

  auto lhsTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhsTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!lhsTy || !rhsTy) {
    return RankedTensorType();
  }

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType())) {
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty()) {
    return RankedTensorType();
  }
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType())) {
    return accRT;
  }
  return RankedTensorType();
}

namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess()) {
    return failure();
  }

  if (parser.parseDimensionList(shape, allowDynamic)) {
    return failure();
  }

  if (parser.parseType(elementType)) {
    return failure();
  }

  if (parser.parseGreater()) {
    return failure();
  }

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << d;
}
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true))) {
    return Type();
  }

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true))) {
    return Type();
  }
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false))) {
    return Type();
  }
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty()) {
    return emitError() << "'!pto.local_array' requires at least one dimension";
  }
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0) {
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
    }
  }
  if (!elementType.isIntOrFloat()) {
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  }
  return success();
}

// ---- StructType ----
// Asm form: !pto.struct<T0, T1, ..., Tn-1>
// A field type must be "scalar-storable": an exactly-nameable scalar or a
// nested !pto.struct (see the two predicates below). The allowlist deliberately
// excludes the vec/cube types (tile_buf / tensor_view / partition view) and any
// other handle type, keeping the scalar struct world disjoint from the
// fractal/layout world.

// A struct field's scalar type must map onto a C++ scalar that the backend can
// name exactly: integers of width 8/16/32/64 and f16/bf16/f32/f64. Widths the
// backend has no spelling for (i1, i24, ...) and the packed low-precision
// vec/cube formats (f8/f4 variants) would otherwise be emitted as `float`,
// silently changing the field's width and semantics, so reject them here.
static bool isStructScalar(Type t) {
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(t)) {
    return true;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    return w == 8 || w == 16 || w == 32 || w == 64;
  }
  return false;
}

// A field is either such a scalar or a nested !pto.struct. !pto.local_array is
// deliberately NOT allowed: a field is reached with `emitc.member`, whose result
// must be an `!emitc.lvalue`, and `!emitc.lvalue` cannot wrap `!emitc.array`
// (the type an array field lowers to). There is no way to spell the access, so
// the restriction is enforced here rather than failing later in the backend.
static bool isStructStorable(Type t) {
  return isStructScalar(t) || llvm::isa<StructType>(t);
}

Type StructType::parse(AsmParser &parser) {
  SmallVector<Type> fields;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            Type t;
            if (parser.parseType(t)) {
              return failure();
            }
            fields.push_back(t);
            return success();
          })) {
    return Type();
  }
  return StructType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), fields);
}

void StructType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::ArrayRef<Type> fields = getFieldTypes();
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer.printType(fields[i]);
  }
  printer << ">";
}

LogicalResult StructType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<Type> fieldTypes) {
  if (fieldTypes.empty()) {
    return emitError() << "'!pto.struct' requires at least one field";
  }
  for (auto [i, f] : llvm::enumerate(fieldTypes)) {
    if (!isStructStorable(f)) {
      return emitError()
             << "'!pto.struct' field " << i << " type " << f
             << " is not scalar-storable; only i8/i16/i32/i64 (signed, "
                "unsigned or signless), f16/bf16/f32/f64, or a nested "
                "!pto.struct are allowed (!pto.local_array cannot be a field "
                "because emitc.member cannot yield an array lvalue; tile_buf / "
                "tensor_view belong to the vec/cube world)";
    }
  }
  return success();
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);

  if (map.getNumResults() != 1) {
    return;
  }

  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term)) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(lhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(rhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }

        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(rhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(lhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = llvm::dyn_cast<AffineDimExpr>(term)) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx,
                                             ArrayRef<int64_t> strides,
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();

  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  }
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) {
    return failure();
  }
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) {
      return failure();
    }
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) {
    return failure();
  }

  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) {
    return failure();
  }

  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) {
      return failure();
    }

    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) {
          return failure();
        }
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) {
        return failure();
      }
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }

  if (parser.parseGreater()) {
    return failure();
  }

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);

  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) {
    return;
  }
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) {
    return;
  }

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer);
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) {
        printer << ", ";
      }
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subview 相关实现

// =============================================================================
// Op Interface Implementation: SubViewOp
// =============================================================================

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes")) {
    return failure();
  }

  ArrayAttr sizesAttr;
  if (parser.parseAttribute(sizesAttr, "sizes", result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("valid"))) {
    OpAsmParser::UnresolvedOperand vrow, vcol;
    if (parser.parseLSquare() || parser.parseOperand(vrow) || parser.parseComma() ||
        parser.parseOperand(vcol) || parser.parseRSquare()) {
      return failure();
    }
    valids.push_back(vrow);
    valids.push_back(vcol);
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy)) {
      return failure();
    }
    hasExplicitResultTy = true;
  }

  if (parser.resolveOperand(source, sourceTy, result.operands)) {
    return failure();
  }

  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsets, indexTy, result.operands)) {
    return failure();
  }
  if (!valids.empty() &&
      parser.resolveOperands(valids, indexTy, result.operands)) {
    return failure();
  }

  int32_t hasValid = valids.empty() ? 0 : 1;
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(offsets.size()), hasValid, hasValid}));

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  SmallVector<Type> inferredReturnTypes;
  DictionaryAttr attrs = result.attributes.getDictionary(parser.getContext());
  if (failed(SubViewOp::inferReturnTypes(
          parser.getContext(), std::nullopt, result.operands, attrs, nullptr,
          RegionRange(), inferredReturnTypes))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.subview result type");
  }
  result.addTypes(inferredReturnTypes);
  return success();
}

void mlir::pto::SubViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[";
  printer.printOperands(getOffsets());
  printer << "] sizes " << getSizes();
  if (getValidRow()) {
    printer << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "sizes"});
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

// The inferred result type derives valid_shape from `sizes` (or the explicit
// valid operands). With the operand omitted the result type is authoritative for
// the valid extent (any static value, including the v=0 no-op-replay marker or a
// partial valid), so accept a static declared valid that differs from the
// size-inferred one here; SubViewOp::verify() enforces the precise per-path rule
// (operand clamping vs the [0, size] range). Only a dynamic declared valid that
// disagrees with the inferred extent is incompatible -- it needs an explicit
// operand to supply the runtime value. Every other difference (shape, element
// type, address space, config) is still rejected as the default check would.
bool SubViewOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [inferred, declared] : llvm::zip(lhs, rhs)) {
    if (inferred == declared) {
      continue;
    }
    auto inferredTb = dyn_cast<TileBufType>(inferred);
    auto declaredTb = dyn_cast<TileBufType>(declared);
    if (!inferredTb || !declaredTb) {
      return false;
    }
    if (inferredTb.getShape() != declaredTb.getShape() ||
        inferredTb.getElementType() != declaredTb.getElementType() ||
        inferredTb.getMemorySpace() != declaredTb.getMemorySpace() ||
        inferredTb.getConfigAttr() != declaredTb.getConfigAttr()) {
      return false;
    }
    auto inferredValid = inferredTb.getValidShape();
    auto declaredValid = declaredTb.getValidShape();
    if (inferredValid.size() != declaredValid.size()) {
      return false;
    }
    for (auto [inferredDim, declaredDim] : llvm::zip(inferredValid, declaredValid)) {
      // Any static declared valid extent is accepted in place of the inferred
      // one; only a dynamic declared valid that disagrees is incompatible.
      if (inferredDim != declaredDim && declaredDim == ShapedType::kDynamic) {
        return false;
      }
    }
  }
  return true;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // 1. 获取 Source Type
  if (operands.empty()) {
    return failure();
  }
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) {
    return failure();
  }

  // 2. 获取 subview 逻辑窗口（sizes）
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop) {
      sizeAttr = prop->sizes;
    }
  }
  if (!sizeAttr && attributes) {
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  }
  if (!sizeAttr) {
    return failure();
  }

  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    subviewShape.push_back(dim);
  }

  // Design: subview 的结果 tile 类型显式表达逻辑子窗口 shape（sizes）。
  ArrayRef<int64_t> parentShape = sourceType.getShape();
  if (subviewShape.size() != parentShape.size()) {
    return failure();
  }

  // Derive valid shape from explicit valid_row/valid_col when provided.
  // Otherwise default to subview shape (no parent valid-shape inheritance).
  SmallVector<int64_t> validShape;
  constexpr int64_t kDynamicValidDim = -1;
  int64_t rank = static_cast<int64_t>(subviewShape.size());
  Value explicitVRow;
  Value explicitVCol;

  // Robustly decode optional valid operands using AttrSizedOperandSegments:
  //   [source, offsets..., valid_row?, valid_col?]
  if (attributes) {
    if (auto segAttr =
            attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes")) {
      ArrayRef<int32_t> segs = segAttr.asArrayRef();
      if (segs.size() == 4) {
        int32_t srcSeg = segs[0];
        int32_t offSeg = segs[1];
        int32_t vRowSeg = segs[2];
        int32_t vColSeg = segs[3];
        if (srcSeg == 1 && offSeg >= 0 && (vRowSeg == 0 || vRowSeg == 1) &&
            (vColSeg == 0 || vColSeg == 1)) {
          size_t idx = static_cast<size_t>(srcSeg + offSeg);
          if (vRowSeg == 1 && idx < operands.size()) {
            explicitVRow = operands[idx++];
          }
          if (vColSeg == 1 && idx < operands.size()) {
            explicitVCol = operands[idx];
          }
        }
      }
    }
  }

  // Fallback for legacy callers that may not provide operandSegmentSizes.
  if (!explicitVRow && !explicitVCol && rank == 2) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + 2) {
      explicitVRow = operands[expectedWithoutValid];
      explicitVCol = operands[expectedWithoutValid + 1];
    }
  }

  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }

  // 3. 继承 Config (若为空使用默认)
  auto cfg = sourceType.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(context);
  }

  // 4. 构建 Result Type
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndex(castOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndex(truncOp.getIn(), out);
  }
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  auto readBLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<BLayoutAttr>(attr)) {
      out = static_cast<int32_t>(a.getValue());
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(a.getInt());
      return true;
    }
    return false;
  };
  auto readSLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<SLayoutAttr>(attr)) {
      out = static_cast<int32_t>(a.getValue());
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(a.getInt());
      return true;
    }
    return false;
  };
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) {
    fr = static_cast<int32_t>(attr.getInt());
  }

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) {
    return failure();
  }

  if (fr == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fr == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fr == 512) {
    if (sl == 1) {
      innerRows = 16;
      innerCols = 32 / elemBytes;
      return success();
    }
    if (sl == 2) {
      innerRows = 32 / elemBytes;
      innerCols = 16;
      return success();
    }
  }
  return failure();
}

struct SubViewInfo {
  int64_t sizeR = 0, sizeC = 0;
  int64_t offR = 0, offC = 0;
  bool offRConst = false, offCConst = false;
};

static LogicalResult verifySubViewSizesAndOffsets(SubViewOp op,
                                                  SubViewInfo &info) {
  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != 2) {
    return op.emitOpError("subview expects 2D sizes");
  }
  info.sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  info.sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (info.sizeR <= 0 || info.sizeC <= 0) {
    return op.emitOpError("subview sizes must be positive");
  }
  if (op.getOffsets().size() != 2) {
    return op.emitOpError("subview expects 2D offsets");
  }

  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if (info.offRConst && info.offR < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  if (info.offCConst && info.offC < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  return success();
}

static LogicalResult verifySubViewValidBounds(SubViewOp op, int64_t sizeR,
                                              int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(op.getValidRow(), vRow)) {
      if (vRow < 0) {
        return op.emitOpError("valid_row must be non-negative when constant");
      }
      if (vRow > sizeR) {
        return op.emitOpError("valid_row must be <= subview row size");
      }
    }
    if (getConstIndex(op.getValidCol(), vCol)) {
      if (vCol < 0) {
        return op.emitOpError("valid_col must be non-negative when constant");
      }
      if (vCol > sizeC) {
        return op.emitOpError("valid_col must be <= subview col size");
      }
    }
  }
  return success();
}
