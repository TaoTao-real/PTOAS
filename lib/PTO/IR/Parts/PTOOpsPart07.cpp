// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

// ---- GetBufDynOp / RlsBufDynOp parse/print ----
static ParseResult parseBufDynSyncOp(OpAsmParser &parser,
                                     OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc)
             << "invalid get_buf_dyn/rls_buf_dyn token: " << token;
}

    if (parser.parseComma()) {
      return failure();
    }

    OpAsmParser::UnresolvedOperand bufOperand;
    if (parser.parseOperand(bufOperand)) {
      return failure();
    }
    if (parser.resolveOperand(bufOperand,
                              parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma()) {
      return failure();
    }

    OpAsmParser::UnresolvedOperand bufOperand;
    if (parser.parseOperand(bufOperand)) {
      return failure();
    }
    if (parser.resolveOperand(bufOperand,
                              parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
    if (parser.parseRSquare()) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufDynSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                              Value bufId, IntegerAttr modeAttr,
                              ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", " << bufId << ", "
      << modeAttr.getInt();
  } else {
    p << "[" << opTypeAttr << ", " << bufId << ", " << modeAttr.getInt()
      << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "mode"});
}

ParseResult GetBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void GetBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}

ParseResult RlsBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void RlsBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}
// ---- TOp ----
LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                         getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2) {
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    }
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1]) {
      return emitOpError("expects bias and dst to have the same column shape");
    }
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                     getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType()))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType()))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                         getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType()) {
      return emitOpError("expects val type to match dst element type");
    }
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType>(srcTy)) {
    return emitOpError("expects src to be tile_buf type");
  }

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace = cast<pto::TileBufType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT) {
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    }
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType()) {
    return emitOpError("expects dst type to match src element type");
  }
  return success();
}

LogicalResult THistogramOp::verify() {
  auto isIntegerWidth = [](Type ty, unsigned width) {
    auto it = dyn_cast<IntegerType>(ty);
    return it && it.getWidth() == width;
  };
  int64_t byte = 1;
  auto byteAttr = getByteAttr();
  if (byteAttr) {
    byte = byteAttr.getInt();
  }
  if (auto legacyIsMSB = (*this)->getAttrOfType<BoolAttr>("isMSB")) {
    int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
    if (byteAttr && byte != legacyByte) {
      return emitOpError("does not allow conflicting 'byte' and legacy 'isMSB' attributes");
    }
    byte = legacyByte;
  }
  if (byte < 0 || byte > 3) {
    return emitOpError("expects byte to be in range [0, 3]");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type idxTy = getIdx().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, idxTy, "idx")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto idxSpace = getPTOMemorySpaceEnum(idxTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects src to be in the vec address space");
    }
    if (!idxSpace || *idxSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects idx to be in the vec address space");
    }
    if (!dstSpace || *dstSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects dst to be in the vec address space");
    }

    auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
    auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
    auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTB || !idxTB || !dstTB) {
      return emitOpError("expects src, idx, and dst to be tile_buf types");
    }

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        srcTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return emitOpError("expects src to use row_major + none_box layout");
    }
    if (dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return emitOpError("expects dst to use row_major + none_box layout");
    }

    bool srcIsUi16 = isIntegerWidth(getElemTy(srcTy), 16);
    bool srcIsUi32 = isIntegerWidth(getElemTy(srcTy), 32);
    if (!srcIsUi16 && !srcIsUi32) {
      return emitOpError("expects src element type to be ui16 or ui32");
    }
    if (!isIntegerWidth(getElemTy(idxTy), 8)) {
      return emitOpError("expects idx element type to be ui8");
    }
    if (!isIntegerWidth(getElemTy(dstTy), 32)) {
      return emitOpError("expects dst element type to be ui32");
    }

    auto srcShape = getShapeVec(srcTy);
    auto idxShape = getShapeVec(idxTy);
    auto dstShape = getShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    auto idxValid = getValidShapeVec(idxTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
        srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError(
          "expects src, idx, and dst to have rank-2 shape and valid_shape");
    }

    if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
        !hasCompatibleKnownExtent(srcValid[0], dstValid[0])) {
      return emitOpError("expects dst rows and valid rows to match src");
    }

    if (srcIsUi16) {
      if (byte > 1) {
        return emitOpError("expects byte to be 0 or 1 when src element type is ui16");
      }
      if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
        return emitOpError(
            "expects idx to use DN layout (col_major + none_box) when src element type is ui16");
      }
      if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
          !hasCompatibleKnownExtent(srcValid[0], idxValid[0])) {
        return emitOpError("expects idx rows and valid rows to match src when src element type is ui16");
      }
      if (!isKnownUnitExtent(idxShape[1]) || !isKnownZeroOrUnitExtent(idxValid[1])) {
        return emitOpError("expects idx to have exactly one physical column and 0 or 1 valid column when src element type is ui16");
      }
    } else {
      if (byte != 3) {
        if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
          return emitOpError(
              "expects idx to use row_major + none_box layout when src element type is ui32 and byte is 0, 1, or 2");
        }
        if (!hasCompatibleKnownExtent(srcShape[1], idxShape[1]) ||
            !hasCompatibleKnownExtent(srcValid[1], idxValid[1])) {
          return emitOpError(
              "expects idx cols and valid cols to match src when src element type is ui32 and byte is 0, 1, or 2");
        }

        int64_t expectedIdxRows = 1;
        if (byte == 1) {
          expectedIdxRows = 2;
        } else if (byte == 0) {
          expectedIdxRows = 3;
        }
        if (!hasCompatibleKnownExtent(idxShape[0], expectedIdxRows) ||
            !hasCompatibleKnownExtentOrZero(idxValid[0], expectedIdxRows)) {
          return emitOpError(
              "expects idx rows to match the byte-selected filter depth and idx valid rows to be 0 or match it when src element type is ui32 and byte is 0, 1, or 2");
        }
      }
    }
    if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256) {
      return emitOpError("expects dst shape[1] to be at least 256");
    }
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
        dstValid[1] < 256) {
      return emitOpError("expects dst valid_shape[1] to be 0 or at least 256");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true))) {
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
      return failure();
    }
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace ||
        (*srcSpace != pto::AddressSpace::LEFT && *srcSpace != pto::AddressSpace::RIGHT)) {
      return emitOpError("expects src to be in the left or right address space");
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != pto::AddressSpace::SCALING) {
      return emitOpError("expects dst to be in the scaling address space");
    }
    auto dstShape = getShapeVec(dstTy);
    auto srcShape = getShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    if (dstShape.size() != 2 || srcShape.size() != 2 || dstValid.size() != 2 ||
        srcValid.size() != 2) {
      return emitOpError(
          "expects src/dst to have rank-2 shape and valid_shape");
    }
    if (*srcSpace == pto::AddressSpace::LEFT) {
      int64_t mShape = srcShape[0];
      int64_t vk = srcValid[1];
      int64_t expectedScaleK = ceilDivKnown(vk, 32);
      if (!hasCompatibleKnownExtent(dstShape[0], mShape) ||
          !hasCompatibleKnownExtent(dstShape[1], expectedScaleK) ||
          !hasCompatibleKnownExtent(dstValid[0], srcValid[0]) ||
          !hasCompatibleKnownExtent(dstValid[1], expectedScaleK)) {
        return emitOpError("expects dst shape/valid_shape to be [M, ceil(K/32)]");
      }
    } else {
      int64_t k = srcValid[0];
      int64_t n = srcShape[1];
      int64_t vk = srcValid[0];
      int64_t vn = srcValid[1];
      if (!hasCompatibleKnownExtent(dstShape[0], ceilDivKnown(k, 32)) ||
          !hasCompatibleKnownExtent(dstShape[1], n) ||
          !hasCompatibleKnownExtent(dstValid[0], ceilDivKnown(vk, 32)) ||
          !hasCompatibleKnownExtent(dstValid[1], vn)) {
        return emitOpError("expects dst shape/valid_shape to be [ceil(K/32), N]");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
ParseResult mlir::pto::MScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand idx;
  OpAsmParser::UnresolvedOperand mem;
  Type srcTy, idxTy, memTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(idx) || parser.parseColonType(srcTy) ||
      parser.parseComma() || parser.parseType(idxTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(mem) || parser.parseColonType(memTy) ||
      parser.parseRParen() ||
      parsePTOInherentAttrs<MScatterOp>(
          parser, result, parsedAttrs,
          {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"})) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands) ||
      parser.resolveOperand(mem, memTy, result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx() << " : "
    << getSrc().getType() << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  p << ") outs(" << getMem() << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ")";

  NamedAttrList attrs = getNonInherentAttrs(
      getOperation(),
      {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"});
  if (auto coalesceAttr = getMScatterCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto scatterAtomicAttr = getMScatterScatterAtomicOpAttrIfPresent(*this);
      scatterAtomicAttr &&
      scatterAtomicAttr.getValue() != pto::ScatterAtomicOp::None) {
    attrs.append("scatterAtomicOp", scatterAtomicAttr);
  }
  if (auto scatterOobAttr = getMScatterScatterOobAttrIfPresent(*this);
      scatterOobAttr &&
      scatterOobAttr.getValue() != pto::ScatterOOB::Undefined) {
    attrs.append("scatterOob", scatterOobAttr);
  }
  if (auto scatterConflictAttr =
          getMScatterScatterConflictAttrIfPresent(*this)) {
    attrs.append("scatterConflict", scatterConflictAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}
LogicalResult MScatterOp::verify() {
  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1) {
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");
  }

  if (failed(verifyNDStyleVecTile(
          *this, srcTy, "src",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  pto::ScatterAtomicOp scatterAtomicOp = getScatterAtomicOpOrDefault(*this);
  pto::ScatterOOB scatterOob = getScatterOobOrDefault(*this);
  if (!srcElem || !idxElem) {
    return emitOpError("failed to resolve element types for src or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem)) {
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src",
                                            coalesce))) {
    return failure();
  }

  if (!coalesce &&
      (scatterAtomicOp != pto::ScatterAtomicOp::None ||
       scatterOob != pto::ScatterOOB::Undefined ||
       getScatterConflictAttrIfPresent(*this))) {
    return emitOpError(
        "expects coalesce when scatterAtomicOp/scatterOob/scatterConflict is specified");
  }

  if (getScatterConflictAttrIfPresent(*this) && !isTargetArchA5(getOperation())) {
    return emitOpError("expects scatterConflict only on A5 targets");
  }

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, scatterAtomicOp)) {
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");
  }

  return success();
}

// ---- MGatherOp ----
// GM -> L1 (cube Mat) gather verifier. The destination is an L1 (loc=mat) tile
// in NZ layout; the index is a GM tensor (the cube core cannot read UB on A5),
// and Coalesce::Elem carries a contiguous GM scratch workspace. Mirrors the
// pto-isa MGATHER GM -> L1 overloads / MGatherCheckGm2L1.
static LogicalResult verifyMGatherGm2L1(Operation *op, Value mem, Value idx,
                                        Value dst, Value scratch,
                                        std::optional<pto::Coalesce> coalesce) {
  Type dstTy = dst.getType();
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!dstTb) {
    return op->emitOpError("expects GM->L1 mgather dst to be a tile_buf");
  }

  // dst must be an L1 / cube Mat tile in NZ layout (col_major + row_major sub +
  // fractal 512), matching the matmul A/NZ operand a TLOAD would produce.
  if (!isColMajorRowMajorNZTileBuf(dstTb)) {
    return op->emitOpError("expects GM->L1 mgather dst (loc=mat) to use "
                           "blayout=col_major and slayout=row_major (NZ)");
  }
  if (dstTb.getSFractalSizeI32() != 512) {
    return op->emitOpError("expects GM->L1 mgather dst fractal size to be 512");
  }

  Type dstElem = getElemTy(dstTy);
  if (!dstElem) {
    return op->emitOpError("failed to resolve GM->L1 mgather dst element type");
  }
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem)) {
    return op->emitOpError(
        "expects GM->L1 mgather dst element type to be "
        "i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 (and on A5 targets also "
        "float8_e4m3/float8_e5m2 family types)");
  }

  // NZ tile shape: padded Cols a multiple of C0 (= 32 / sizeof(elem)) and padded
  // Rows a multiple of FRACTAL_NZ_ROW (= 16).
  unsigned elemBytes =
      std::max<unsigned>(1u, dstElem.getIntOrFloatBitWidth() / 8u);
  int64_t kC0 = 32 / static_cast<int64_t>(elemBytes);
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2) {
    if (kC0 > 0 && dstShape[1] != ShapedType::kDynamic &&
        dstShape[1] % kC0 != 0) {
      return op->emitOpError()
             << "expects GM->L1 mgather dst padded cols to be a multiple of "
             << kC0 << " (C0 = 32 / sizeof(elem))";
    }
    if (dstShape[0] != ShapedType::kDynamic && dstShape[0] % 16 != 0) {
      return op->emitOpError("expects GM->L1 mgather dst padded rows to be a "
                             "multiple of 16 (FRACTAL_NZ_ROW)");
    }
  }

  // mem table: GM, element type matches dst.
  if (failed(verifyMGatherMScatterMemOperand(op, mem, dstElem, "dst"))) {
    return failure();
  }

  // idx: GM partition tensor view of i32 -- NOT a UB tile.
  Type idxTy = idx.getType();
  if (isa<pto::TileBufType>(idxTy)) {
    return op->emitOpError("expects GM->L1 mgather idx to be a GM tensor "
                           "partition_tensor_view, not a tile_buf");
  }
  if (!isa<pto::PartitionTensorViewType>(idxTy)) {
    return op->emitOpError(
        "expects GM->L1 mgather idx to be a partition_tensor_view");
  }
  Type idxElem = getElemTy(idxTy);
  if (!idxElem || !isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return op->emitOpError("expects GM->L1 mgather idx element type to be i32");
  }

  // Coalesce must be explicit: the GM index has no UB tile shape to infer from.
  if (!coalesce) {
    return op->emitOpError("expects GM->L1 mgather to specify an explicit "
                           "coalesce attribute (row or elem)");
  }

  if (*coalesce == pto::Coalesce::Elem) {
    // Elem mode stages discrete elements into NZ layout through a GM scratch
    // workspace before the bulk GM -> L1 copy.
    if (!scratch) {
      return op->emitOpError("expects GM->L1 mgather with coalesce=elem to "
                             "provide a GM scratch operand");
    }
    Type scTy = scratch.getType();
    if (!isa<pto::PartitionTensorViewType>(scTy)) {
      return op->emitOpError(
          "expects GM->L1 mgather scratch to be a partition_tensor_view");
    }
    Type scElem = getElemTy(scTy);
    if (!scElem || scElem != dstElem) {
      return op->emitOpError("expects GM->L1 mgather scratch element type to "
                             "match dst element type");
    }
  } else { // Row
    if (scratch) {
      return op->emitOpError("expects GM->L1 mgather with coalesce=row to omit "
                             "the scratch operand");
    }
  }

  return success();
}
ParseResult mlir::pto::MGatherOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOperands;
  SmallVector<Type, 3> insTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    insOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() < 2 || insOperands.size() > 3) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects mgather ins(mem, idx[, scratch])");
  }

  if (parser.parseColon()) {
    return failure();
  }

  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    insTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() != insTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of ins operands to match the number of ins types");
  }

  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parsePTOInherentAttrs<MGatherOp>(
          parser, result, parsedAttrs, {"coalesce", "gatherOob"})) {
    return failure();
  }

  if (parser.resolveOperand(insOperands[0], insTypes[0], result.operands) ||
      parser.resolveOperand(insOperands[1], insTypes[1], result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (insOperands.size() == 3 &&
      parser.resolveOperand(insOperands[2], insTypes[2], result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getMem() << ", " << getIdx();
  if (auto scratch = getScratch()) {
    p << ", " << scratch;
  }
  p << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  if (auto scratch = getScratch()) {
    p << ", ";
    p.printStrippedAttrOrType(scratch.getType());
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"coalesce", "gatherOob"});
  if (auto coalesceAttr = getMGatherCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto gatherOobAttr = getMGatherGatherOobAttrIfPresent(*this);
      gatherOobAttr &&
      gatherOobAttr.getValue() != pto::GatherOOB::Undefined) {
    attrs.append("gatherOob", gatherOobAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

LogicalResult MGatherOp::verify() {
  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();

  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1) {
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");
  }

  // GM -> L1 (cube Mat) gather: dst is an L1 (loc=mat) tile; idx comes from GM
  // and Coalesce::Elem carries a GM scratch operand.
  if (isa<pto::TileBufType>(dstTy)) {
    if (auto as = getPTOMemorySpaceEnum(dstTy);
        as && *as == pto::AddressSpace::MAT) {
      std::optional<pto::Coalesce> coalesce;
      if (auto coalesceAttr = getCoalesceAttr()) {
        coalesce = coalesceAttr.getValue();
      }
      return verifyMGatherGm2L1(getOperation(), getMem(), getIdx(), getDst(),
                                getScratch(), coalesce);
    }
  }

  // GM -> UB (VEC) gather: the default path. A GM scratch operand is only valid
  // for the GM -> L1 path above.
  if (getScratch()) {
    return emitOpError("expects scratch operand only on GM->L1 (loc=mat) "
                       "mgather");
  }

  if (failed(verifyNDStyleVecTile(
          *this, dstTy, "dst",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  pto::GatherOOB gatherOob = getGatherOobOrDefault(*this);
  if (!dstElem || !idxElem) {
    return emitOpError("failed to resolve element types for dst or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem)) {
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst",
                                            coalesce))) {
    return failure();
  }

  if (gatherOob != pto::GatherOOB::Undefined && !coalesce) {
    return emitOpError("expects coalesce when gatherOob is specified");
  }

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
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
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode")) {
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    }
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen()) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TDeInterleaveOp::print(OpAsmPrinter &p) {
  p << " ins(";
  llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
  p << " : ";
  llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type type) { p << type; });
  p << ") outs(" << getDst0() << ", " << getDst1() << " : "
    << getDst0().getType() << ", " << getDst1().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TDeInterleaveOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  bool invalidHeader = parser.parseKeyword("ins") || parser.parseLParen();
  if (invalidHeader) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 2> srcs;
  OpAsmParser::UnresolvedOperand src;
  bool invalidSrc = failed(parser.parseOperand(src));
  if (invalidSrc) {
    return failure();
  }
  srcs.push_back(src);
  while (succeeded(parser.parseOptionalComma())) {
    bool invalidNextSrc = failed(parser.parseOperand(src));
    if (invalidNextSrc) {
      return failure();
    }
    srcs.push_back(src);
  }

  SmallVector<Type, 2> srcTypes;
  Type srcType;
  bool invalidSrcType = failed(parser.parseColonType(srcType));
  if (invalidSrcType) {
    return failure();
  }
  srcTypes.push_back(srcType);
  while (succeeded(parser.parseOptionalComma())) {
    bool invalidNextType = failed(parser.parseType(srcType));
    if (invalidNextType) {
      return failure();
    }
    srcTypes.push_back(srcType);
  }
  bool invalidOutsHeader =
      parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen();
  if (invalidOutsHeader) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 2> dsts;
  OpAsmParser::UnresolvedOperand dst;
  bool invalidDst = parser.parseOperand(dst) || parser.parseComma();
  if (invalidDst) {
    return failure();
  }
  dsts.push_back(dst);
  bool invalidSecondDst = failed(parser.parseOperand(dst));
  if (invalidSecondDst) {
    return failure();
  }
  dsts.push_back(dst);
  Type dst0Ty;
  Type dst1Ty;
  bool invalidDstTypes = parser.parseColonType(dst0Ty) ||
                         parser.parseComma() || parser.parseType(dst1Ty) ||
                         parser.parseRParen();
  if (invalidDstTypes) {
    return failure();
  }
  bool invalidSourceCount = srcs.size() < 1 || srcs.size() > 2;
  if (invalidSourceCount) {
    return parser.emitError(parser.getCurrentLocation(),
                            "tdeinterleave expects one or two source operands");
  }

  bool unresolvedSources =
      failed(parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                                    result.operands));
  bool unresolvedDsts =
      failed(parser.resolveOperand(dsts[0], dst0Ty, result.operands)) ||
      failed(parser.resolveOperand(dsts[1], dst1Ty, result.operands));
  if (unresolvedSources || unresolvedDsts) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(srcs.size()), 2}));
  return parser.parseOptionalAttrDict(result.attributes);
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2() || isFormat2WithoutTmp()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    if (getTmp()) {
      p << ", " << getTmp();
    } else {
      p << " no_tmp";
}
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    if (getTmp()) {
      p << ", " << getTmp().getType();
    }
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second)) {
    return failure();
  }

  if (parser.parseOptionalColon().succeeded()) {
    Type srcTy, blockLenTy, dstTy;
    if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(blockLenTy) ||
        parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen()) {
      return failure();
    }
    OpAsmParser::UnresolvedOperand dstOp;
    if (parser.parseOperand(dstOp) || parser.parseColon() || parser.parseType(dstTy) ||
        parser.parseRParen()) {
      return failure();
    }
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
    if (parser.resolveOperand(first, srcTy, result.operands) ||
        parser.resolveOperand(second, blockLenTy, result.operands) ||
        parser.resolveOperand(dstOp, dstTy, result.operands)) {
      return failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    if (!result.attributes.get("exhausted")) {
      result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
    }
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next)) {
      return failure();
    }
    srcs.push_back(next);
  }
  bool noTmp = succeeded(parser.parseOptionalKeyword("no_tmp"));
  if ((noTmp && (srcs.size() < 2 || srcs.size() > 4)) ||
      (!noTmp && (srcs.size() < 3 || srcs.size() > 5))) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands and optional no_tmp marker");
  }
  OpAsmParser::UnresolvedOperand tmpOp;
  if (!noTmp) {
    tmpOp = srcs.pop_back_val();
  }
  bool exhaustedVal = false;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parser.parseKeyword("exhausted") || parser.parseEqual()) {
      return failure();
    }
    StringRef kw;
    if (parser.parseKeyword(&kw) || parser.parseRBrace()) {
      return failure();
    }
    exhaustedVal = (kw == "true");
  }
  SmallVector<Type, 4> srcTypes;
  srcTypes.reserve(srcs.size());
  if (parser.parseColon()) {
    return failure();
  }
  Type firstSrcTy;
  if (parser.parseType(firstSrcTy)) {
    return failure();
  }
  srcTypes.push_back(firstSrcTy);
  while (parser.parseOptionalComma().succeeded()) {
    Type nextTy;
    if (parser.parseType(nextTy)) {
      return failure();
    }
    srcTypes.push_back(nextTy);
  }
  if (srcTypes.size() != srcs.size() + (noTmp ? 0 : 1) ||
      parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }
  Type tmpTy;
  if (!noTmp) {
    tmpTy = srcTypes.pop_back_val();
  }
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (parser.parseOperand(dstOp) || parser.parseComma() || parser.parseOperand(excutedOp) ||
      parser.parseColon() || parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1,
                           noTmp ? 0 : 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      (!noTmp && parser.resolveOperand(tmpOp, tmpTy, result.operands)) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (!result.attributes.get("exhausted")) {
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(exhaustedVal));
  }
  return success();
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "format1 expects PTO shaped-like types for src/dst";
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src/dst to have the same element type";
  }
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects element type to be f16 or f32";
  }
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != 2 || ds.size() != 2) {
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  }
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1) {
    return op.emitOpError() << "expects src rows == 1";
  }
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1) {
    return op.emitOpError() << "expects dst rows == 1";
  }
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1]) {
    return op.emitOpError() << "expects src/dst cols to match";
  }
  if (op.getBlockLen()) {
    if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
        int64_t v = intAttr.getValue().getSExtValue();
        if (v <= 0 || (v % 64) != 0) {
          return op.emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortFormat2Basics(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType())) {
      return op.emitOpError() << "format2 expects PTO shaped-like type for each src";
    }
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u) {
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  }
  if (op.getDsts().size() != 1u || !op.getExcuted()) {
    return op.emitOpError()
           << "format2 expects 2 to 4 srcs, one dst, and excuted=vector";
  }
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  if (!isPTOShapedLike(dstTy) ||
      (tmpTy && !isPTOShapedLike(tmpTy))) {
    return op.emitOpError() << "format2 dst/tmp must be PTO shaped-like";
  }
  auto excutedTy = mlir::dyn_cast<mlir::VectorType>(op.getExcuted().getType());
  if (!excutedTy || excutedTy.getRank() != 1 || excutedTy.getNumElements() != 4 ||
      !excutedTy.getElementType().isInteger(16)) {
    return op.emitOpError() << "format2 excuted must be vector<4xi16>";
  }
  Type elemTy = getElemTy(dstTy);
  if (tmpTy && elemTy != getElemTy(tmpTy)) {
    return op.emitOpError() << "format2 expects dst/tmp element types to match";
  }
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  if (dstShape.size() != 2 || (tmpTy && tmpShape.size() != 2)) {
    return op.emitOpError() << "format2 expects dst/tmp to be rank-2 tile-shaped";
  }
  if ((dstShape[0] != mlir::ShapedType::kDynamic && dstShape[0] != 1) ||
      (tmpTy && tmpShape[0] != mlir::ShapedType::kDynamic &&
       tmpShape[0] != 1)) {
    return op.emitOpError() << "format2 expects dst/tmp rows == 1";
  }
  if (tmpTy && dstShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < dstShape[1]) {
    return op.emitOpError() << "format2 expects tmp.cols >= dst.cols";
  }
  return success();
}

static LogicalResult verifyTMrgSortFormat2Srcs(TMrgSortOp op) {
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type elemTy = getElemTy(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  int64_t requiredTmpCols = 0;
  for (Value src : op.getSrcs()) {
    Type srcTy = src.getType();
    auto srcShape = getShapeVec(srcTy);
    auto srcValidShape = getValidShapeVec(src);
    if (srcShape.size() != 2 || srcValidShape.size() != 2) {
      return op.emitOpError() << "format2 expects src to be rank-2 tile-shaped";
    }
    if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1) {
      return op.emitOpError() << "format2 expects src rows == 1";
    }
    if (getElemTy(srcTy) != elemTy) {
      return op.emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    if (srcValidShape[1] == mlir::ShapedType::kDynamic) {
      requiredTmpCols = mlir::ShapedType::kDynamic;
    } else if (requiredTmpCols != mlir::ShapedType::kDynamic) {
      requiredTmpCols += srcValidShape[1];
    }
  }
  if (tmpTy && requiredTmpCols != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < requiredTmpCols) {
    return op.emitOpError()
           << "format2 expects tmp.cols >= sum(src.cols) = "
           << requiredTmpCols;
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (isFormat1()) {
    return verifyTMrgSortFormat1(*this);
  }
  if (isFormat2() || isFormat2WithoutTmp()) {
    if (failed(verifyTMrgSortFormat2Basics(*this))) {
      return failure();
    }
    return verifyTMrgSortFormat2Srcs(*this);
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return emitOpError() << "failed to get element type for src/dst";
  }
  if (srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!mlir::isa<IntegerType>(srcElem)) {
    return emitOpError() << "expects integral element types";
  }
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0) {
    return emitOpError("expects tshls scalar to be non-negative");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
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
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
          elemTy.isF32())) {
      return emitOpError()
             << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
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
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError() << "expects src and dst to have rank-2 valid_shape";
    }
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1]) {
      return emitOpError()
             << "expects src and dst to have the same valid_shape[1]";
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32() || elemTy.isBF16())) {
      return emitOpError()
             << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      return emitOpError() << "expects src and dst to have the same element type";
    }
    if (!elemTy.isInteger(16)) {
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      return emitOpError() << "expects src and dst to have the same element type";
    }
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32))) {
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
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
          "expects A2/A3 tor src0, src1, and dst element type to be i8/i16/i32");
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
          "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
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
          "expects A2/A3 tors src and dst element type to be i8/i16");
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
          "expects A5 tors src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError(
               "expects src0/src1/dst to be tensor/tile_buf/tile_view types"),
           failure();
  }
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for operands"), failure();
  }
  if (e0 != e1 || e0 != ed) {
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  }
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd) {
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  }
  return e0;
}

static LogicalResult verifyTPartAddA2A3(TPartAddOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A2/A3 tpartadd element type to be i32/i16/f16/f32");
  }
  return mlir::success();
}
