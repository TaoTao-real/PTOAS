// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

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

static LogicalResult verifyTPartAddA5(TPartAddOp op) {
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
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32())) {
    return op.emitOpError("expects A5 tpartadd element type to be i32/i16/i8/f16/bf16/f32");
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartAddA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartAddA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyPartialValidPattern(*this, t0, t1, td))) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tpartmax element type to be i32/i16/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32())) {
      return emitOpError("expects A5 tpartmax element type to be i32/i16/i8/f16/bf16/f32");
    }
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td))) {
      return failure();
    }
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyPartialValidPattern(*this, t0, t1, td))) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tpartmin element type to be i32/i16/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32())) {
      return emitOpError("expects A5 tpartmin element type to be i32/i16/i8/f16/bf16/f32");
    }
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td))) {
      return failure();
    }
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr)) {
    return failure();
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }

  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy)) {
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  }
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  }
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");
  }

  Type elem = *dataElemOr;
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32())) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
    }
  } else {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op->emitOpError() << "expects A2/A3 " << opName
                               << " element type to be i32/i16/f16/f32";
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static LogicalResult verifyTPartMulA2A3(TPartMulOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
        elem.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tpartmul element type to be i32/i16/f16/f32");
  }
  return mlir::success();
}

static LogicalResult verifyTPartMulA5(TPartMulOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32())) {
    return op.emitOpError(
        "expects A5 tpartmul element type to be i32/i16/i8/f16/bf16/f32");
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartMulA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartMulA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<std::tuple<Type, Type, Type, Type>>
verifyTPReluCommon(TPReluOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type tt = op.getTmp() ? op.getTmp().getType() : Type{};
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }
  if (tt && failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }

  Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op.emitOpError("expects dst/src0/src1 to have the same element type");
    return failure();
  }
  if (!(e0.isF16() || e0.isF32())) {
    op.emitOpError("expects dst/src0/src1 element type to be f16 or f32");
    return failure();
  }
  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td)) {
    op.emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, t1, td, "src1", "dst"))) {
    return failure();
  }

  auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    op.emitOpError("expects src0/src1/dst to have the same shape");
    return failure();
  }
  return std::make_tuple(t0, t1, tt, td);
}

static LogicalResult verifyTPReluA2A3Tmp(TPReluOp op, Type tt, Type td) {
  Type tmpElem = getElemTy(tt);
  auto tmpIntTy = mlir::dyn_cast<IntegerType>(tmpElem);
  if (!tmpIntTy || tmpIntTy.getWidth() != 8) {
    return op.emitOpError("expects A2/A3 tmp element type to be u8");
  }
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto tmpShape = getShapeVec(tt);
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (tmpShape.size() != 2 || dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  if (dstValid[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] < dstValid[0] + 1) {
    return op.emitOpError()
           << "expects A2/A3 tmp shape[0] to be at least dst valid_shape[0] + 1 ("
           << (dstValid[0] + 1) << ")";
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic) {
    int64_t packedMaskCols = llvm::divideCeil(dstValid[1], int64_t{8});
    if (tmpValid[1] < packedMaskCols) {
      return op.emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least ceil(dst valid_shape[1] / 8) ("
             << packedMaskCols << ")";
    }
  }
  if (dstValid[0] == ShapedType::kDynamic ||
      dstValid[1] == ShapedType::kDynamic) {
    return op.emitOpError(
        "expects A2/A3 tprelu dst valid_shape to be static when tmp is provided");
  }
  int64_t packedCols = std::max<int64_t>(
      32, llvm::divideCeil(llvm::divideCeil(dstValid[1], int64_t{8}),
                           int64_t{32}) *
              32);
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstValid[0] + 1) * static_cast<uint64_t>(packedCols)))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTPReluA2A3(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  if (!tt) {
    return success();
  }
  if (failed(verifyTPReluA2A3Tmp(op, tt, td))) {
    return failure();
  }
  if (auto arch = getVerifierArchName(op.getOperation());
      arch && arch->equals_insensitive("a3")) {
    if (op.getSrc0() == op.getSrc1() || op.getSrc0() == op.getTmp() ||
        op.getSrc0() == op.getDst() || op.getSrc1() == op.getTmp() ||
        op.getSrc1() == op.getDst() || op.getTmp() == op.getDst()) {
      return op.emitOpError(
          "expects A3 src0, src1, tmp, and dst to use different storage");
    }
  }
  return success();
}

static LogicalResult verifyTPReluA5(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  (void)td;
  if (tt && failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPReluA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPReluA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TQuantOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src, fp, offset, dst, tmp;
  Type srcTy, fpTy, offsetTy, dstTy, tmpTy;
  bool hasOffset = false;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() || parser.parseOperand(fp)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(offset)) {
      return failure();
    }
    hasOffset = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(fpTy)) {
    return failure();
  }
  if (hasOffset) {
    if (parser.parseComma() || parser.parseType(offsetTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(dstTy)) {
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
  if (failed(parsePTOInherentAttrs<TQuantOp>(
          parser, result, parsedAttrs, {"quant_type", "operandSegmentSizes"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(fp, fpTy, result.operands)) {
    return failure();
  }
  if (hasOffset) {
    if (parser.resolveOperand(offset, offsetTy, result.operands)) {
      return failure();
    }
  }
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<TQuantOp::Properties>();
  llvm::copy(ArrayRef<int32_t>({1, 1, hasOffset ? 1 : 0, hasTmp ? 1 : 0, 1}),
             properties.operandSegmentSizes.begin());
  return success();
}

void mlir::pto::TQuantOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getFp();
  if (auto offset = getOffset()) {
    p << ", " << offset << " : " << getSrc().getType() << ", "
      << getFp().getType() << ", " << offset.getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getFp().getType() << ")";
  }
  p << " outs(" << getDst();
  if (auto tmp = getTmp()) {
    p << ", " << tmp << " : " << getDst().getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << getDst().getType() << ")";
  }
  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"quant_type", "operandSegmentSizes"});
  attrs.append("quant_type", getQuantTypeAttr());
  p.printOptionalAttrDict(attrs.getAttrs());
}

ParseResult mlir::pto::TQuantMxOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  Type srcTy;
  SmallVector<OpAsmParser::UnresolvedOperand, 5> outOperands;
  SmallVector<Type, 5> outTypes;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseColonType(srcTy) ||
      parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    outOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseColon()) {
    return failure();
  }

  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    outTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen()) {
    return failure();
  }

  if (outOperands.size() != outTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of outs operands to match the number of outs types");
  }
  if (outOperands.size() != 4 && outOperands.size() != 5) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects 4 or 5 operands in outs(...)");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (!llvm::isa_and_nonnull<pto::QuantTypeAttr>(
          result.attributes.get("quant_type"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects quant_type attribute");
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  for (auto [operand, type] : llvm::zip_equal(outOperands, outTypes)) {
    if (parser.resolveOperand(operand, type, result.operands)) {
      return failure();
    }
  }

  return success();
}

void mlir::pto::TQuantMxOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  p << " outs(" << getDst() << ", " << getExp() << ", " << getMax() << ", "
    << getScaling();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz;
  }
  p << " : " << getDst().getType() << ", " << getExp().getType() << ", "
    << getMax().getType() << ", " << getScaling().getType();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz.getType();
  }
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static LogicalResult verifyTQuantStructural(TQuantOp op) {
  Type dstElemTy = getElemTy(op.getDst().getType());
  auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
  if (op.getQuantType() == mlir::pto::QuantType::INT8_SYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_SYM quantization requires an fp operand";
    }
    if (op.getOffset()) {
      return op.emitOpError()
             << "INT8_SYM quantization must not have an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_SYM quantization";
    }
  } else if (op.getQuantType() == mlir::pto::QuantType::INT8_ASYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an fp operand";
    }
    if (!op.getOffset()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_ASYM quantization";
    }
  } else {
    return op.emitOpError("expects plain tquant quant_type to be INT8_SYM or INT8_ASYM; use tquant.mx for MX quantization");
  }
  return success();
}
