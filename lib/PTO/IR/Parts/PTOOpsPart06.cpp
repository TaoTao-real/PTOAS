// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyTMovXToZzForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (!isA5) {
    return op.emitOpError("X-to-ZZ tmov is only supported on A5");
  }
  if (op.getNumResults() != 0) {
    return op.emitOpError("expects X-to-ZZ tmov not to have results");
  }
  if (op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
      op.getReluPreMode() != pto::ReluPreMode::NoRelu) {
    return op.emitOpError("expects the X-to-ZZ tmov form not to use preQuantScalar, accToVecMode, or reluPreMode");
  }

  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  auto tmpTb = dyn_cast<pto::TileBufType>(fp.getType());
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  auto tmpSpace = getPTOMemorySpaceEnum(fp.getType());
  if (!srcTb || !dstTb || !tmpTb || !srcSpace || !dstSpace ||
      *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC ||
      *tmpSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects X-to-ZZ src/dst/tmp to be vec tiles");
  }
  if (op.getSrc() == op.getDst() || op.getSrc() == fp || op.getDst() == fp) {
    return op.emitOpError("expects X-to-ZZ src, dst, and tmp to be distinct tile values");
  }
  if (srcTb.getRank() != 2 || dstTb.getRank() != 2 || tmpTb.getRank() != 2) {
    return op.emitOpError("expects rank-2 valid_shape for src/dst/tmp");
  }
  auto hasDynamic = [](ArrayRef<int64_t> shape) {
    return llvm::is_contained(shape, ShapedType::kDynamic);
  };
  if (hasDynamic(getValidShapeVec(srcTy)) || hasDynamic(getShapeVec(srcTy)) ||
      hasDynamic(getValidShapeVec(dstTy)) || hasDynamic(getShapeVec(dstTy)) ||
      hasDynamic(getShapeVec(fp.getType()))) {
    return op.emitOpError("expects static valid and physical shapes for src/dst and a static tmp physical shape for X-to-ZZ");
  }
  return verifyTMovXToZzElemLayout(op);
}

static LogicalResult verifyTMovXToZzAxis1(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> dstValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (dstValid[1] % 2 != 0) {
    return op.emitOpError("expects ND-to-ZZ dst valid_shape[1] (the grouped exponent column count) to be even");
  }
  if (srcValid[0] != 1 && srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects ND-to-ZZ src valid elements to form a compact prefix (single-row legacy flat or physical row stride equal to valid cols)");
  }
  auto paddedRows = tmovAlign16(dstValid[0]);
  auto required =
      paddedRows ? tmovCheckedMul(*paddedRows, dstValid[1]) : std::nullopt;
  if (!required) {
    return op.emitOpError("cannot compute ND-to-ZZ padded capacity without overflow");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  if (!srcBytes || *srcBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ src physical capacity to cover align16(dst rows) * dst cols because source padding is zeroed in place");
  }
  if (!dstBytes || *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ dst physical capacity to cover align16(dst rows) * dst cols");
  }
  auto rowBlocksBias = tmovCheckedAdd(dstValid[0], 15);
  auto offsetBytes = rowBlocksBias
                         ? tmovCheckedMul(*rowBlocksBias / 16, dstValid[1])
                         : std::nullopt;
  auto tmpRequired =
      offsetBytes ? tmovCheckedAdd(64, *offsetBytes) : std::nullopt;
  if (!tmpRequired) {
    return op.emitOpError("cannot compute ND-to-ZZ tmp capacity without overflow");
  }
  auto tmpBytes = getStaticByteSize(fp.getType());
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(*tmpRequired)) {
    return op.emitOpError() << "expects tmp to provide at least " << *tmpRequired
                            << " bytes for ND-to-ZZ (64 + ceil(dst rows / 16) * dst cols)";
  }
  return success();
}

static LogicalResult verifyTMovXToZzAxis0(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (srcValid[0] < 2 || srcValid[0] % 2 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[0] to be an even count >= 2; a single row-group produces no output in PTO-ISA");
  }
  if (srcValid[1] % 16 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[1] to be a multiple of 16");
  }
  if (srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects DN-to-ZZ src physical row stride to equal src valid_shape[1]");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  auto required = tmovCheckedMul(srcValid[0], srcValid[1]);
  if (!required || !srcBytes || !dstBytes ||
      *srcBytes < static_cast<uint64_t>(*required) ||
      *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects DN-to-ZZ src/dst physical capacity to cover src valid rows * src valid cols");
  }
  return success();
}

static LogicalResult verifyTMovXToZzCapacity(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto srcPhysical = getShapeVec(srcTy);
  auto srcElements = tmovCheckedElements(srcValid);
  auto dstElements = tmovCheckedElements(dstValid);
  if (!srcElements || !dstElements || *srcElements != *dstElements) {
    return op.emitOpError("expects src and dst to hold the same exponent count");
  }
  const MxGroupAxis axis =
      op.getGrpAxisAttr() ? op.getGrpAxisAttr().getValue() : MxGroupAxis::Axis1;
  if (axis == MxGroupAxis::Axis1) {
    return verifyTMovXToZzAxis1(op, srcValid, dstValid, srcPhysical);
  }
  return verifyTMovXToZzAxis0(op, srcValid, srcPhysical);
}

static LogicalResult verifyTMovXToZz(TMovOp op, bool isA5) {
  if (failed(verifyTMovXToZzForm(op, isA5))) {
    return failure();
  }
  return verifyTMovXToZzCapacity(op);
}

static LogicalResult verifyTMovGenericPreconditions(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  if (op.getGrpAxisAttr()) {
    return op.emitOpError("expects grpAxis only on the X-to-ZZ form with a non-scaling third tile");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", /*allowLowPrecision=*/isA5)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp",
                                          /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && op.getPreQuantScalar()) {
    return op.emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace) {
    return op.emitOpError() << "expects src and dst to have explicit address spaces";
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
  }
  if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";
  }
  return success();
}

static LogicalResult verifyTMovGenericPairing(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isMatToTile =
      *srcSpace == pto::AddressSpace::MAT &&
      (*dstSpace == pto::AddressSpace::LEFT ||
       *dstSpace == pto::AddressSpace::RIGHT ||
       *dstSpace == pto::AddressSpace::BIAS ||
       *dstSpace == pto::AddressSpace::SCALING);
  const bool isVecToVec = *srcSpace == pto::AddressSpace::VEC &&
                          *dstSpace == pto::AddressSpace::VEC;
  const bool isVecToMat = *srcSpace == pto::AddressSpace::VEC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  bool okPair = isMatToTile || isVecToVec || isAccToMat || isAccToVec;
  if (isA5) {
    okPair = okPair || isVecToMat;
  }
  if (!okPair) {
    return op.emitOpError() << "expects a supported tmov address-space pair for this target";
  }
  if (op.getAccToVecModeAttr() && !isAccToVec) {
    return op.emitOpError() << "expects accToVecMode to be used only for acc-to-vec tmov";
  }
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects reluPreMode form to use loc=acc src";
  }
  if (op.getPreQuantScalar() && !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects preQuantScalar form to use loc=acc src";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpLayout(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  const bool hasFp = static_cast<bool>(op.getFp());
  auto reluMode = op.getReluPreMode();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
      (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError() << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
  }
  if (hasFp && !isA5 && dstTb && isAccToMat &&
      (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
       dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))) {
    return op.emitOpError() << "expects fp tmov dst to use blayout=col_major and slayout=row_major";
  }
  if (srcTb && dstTb && isAccToMat && !isA5 &&
      dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (hasFp) {
    auto fpSpace = getPTOMemorySpaceEnum(fp.getType());
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError() << "expects fp to be in the scaling address space";
    }
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
      return op.emitOpError() << "expects fp form src to have element type f32, i32";
    }
    if (!(isAccToMat || isAccToVec)) {
      return op.emitOpError() << "expects fp form to use loc=acc src";
    }
  }
  if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
    switch (accToVecModeAttr.getValue()) {
    case pto::AccToVecMode::SingleModeVec0:
    case pto::AccToVecMode::SingleModeVec1:
      break;
    case pto::AccToVecMode::DualModeSplitM:
    case pto::AccToVecMode::DualModeSplitN:
      return op.emitOpError() << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
    }
  }
  return verifyTMovGenericFpLayout(op, isA5);
}

static LogicalResult verifyTMovGeneric(TMovOp op, bool isA5) {
  if (failed(verifyTMovGenericPreconditions(op, isA5)) ||
      failed(verifyTMovGenericPairing(op, isA5)) ||
      failed(verifyTMovGenericFpForm(op, isA5))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTMovImpl(TMovOp op, bool isA5) {
  Value fp = op.getFp();
  if (fp && !getPTOMemorySpaceEnum(fp.getType())) {
    return op.emitOpError("expects the third tile to have an explicit address space");
  }
  if (classifyTMovForm(fp) == TMovForm::XToZz) {
    return verifyTMovXToZz(op, isA5);
  }
  return verifyTMovGeneric(op, isA5);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<RankedTensorType>(t)) {
    return s.getRank();
  }
  if (auto tile = dyn_cast<pto::TileBufType>(t)) {
    return tile.getRank();
  }
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) {
    return view.getRank();
  }
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (Tensor 或 Tile 类型)
  bool aValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid) {
    return op->emitOpError("expects inputs/outputs to be tensors or PTO tile types");
  }

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank) {
      return op->emitOpError("expects a and dst to have the same rank");
    }
    if (bRank != -1 && dRank != -1 && bRank != dRank) {
      return op->emitOpError("expects b and dst to have the same rank");
    }
  }

  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else {
    return emitOpError("expects ptr to be !pto.ptr type");
  }

  if (getValue().getType() != elemTy) {
    return emitOpError("expects result type to match ptr element type");
  }

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else {
    return emitOpError("expects ptr to be !pto.ptr type");
  }

  if (getValue().getType() != elemTy) {
    return emitOpError("expects value type to match ptr element type");
  }

  return success();
}

// ---- CmoCacheInvalidOp ----
static bool isGmOrDefaultAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

static bool isGmOrDefaultCmoAddressType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return isGmOrDefaultAddressSpace(ptrTy.getMemorySpace().getAddressSpace());
  }
  if (isa<mlir::pto::TensorViewType, mlir::pto::PartitionTensorViewType>(type)) {
    return true;
  }
  return false;
}

ParseResult CmoCacheInvalidOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  if (succeeded(parser.parseOptionalKeyword("all"))) {
    AddressSpaceAttr spaceAttr;
    if (parser.parseAttribute(spaceAttr, "space", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    return success();
  }

  OpAsmParser::UnresolvedOperand addr;
  Type addrTy;
  if (parser.parseOperand(addr) ||
      parser.parseKeyword("single_cache_line") ||
      parser.parseColonType(addrTy) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(addr, addrTy, result.operands)) {
    return failure();
  }

  if (!result.attributes.get("space")) {
    result.addAttribute(
        "space", AddressSpaceAttr::get(parser.getContext(), AddressSpace::GM));
  }
  return success();
}

void CmoCacheInvalidOp::print(OpAsmPrinter &p) {
  if (Value addr = getAddr()) {
    p << " " << addr << " single_cache_line";
    p << " : " << addr.getType();
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"space"});
    return;
  }

  p << " all " << getSpace();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"space"});
}

LogicalResult CmoCacheInvalidOp::verify() {
  if (!isGmOrDefaultAddressSpace(getSpace().getAddressSpace())) {
    return emitOpError("only supports GM cache maintenance");
  }

  if (Value addr = getAddr()) {
    if (!isGmOrDefaultCmoAddressType(addr.getType())) {
      return emitOpError("single_cache_line address expects a GM pointer or GM tensor view");
    }
  }

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr,
                                     IntegerAttr modeAttr) {
  if (!opTypeAttr) {
    return op->emitOpError("expects 'op_type' attribute");
  }

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe)) {
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
  }

  if (!bufIdAttr) {
    return op->emitOpError("expects 'buf_id' attribute");
  }
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31) {
    return op->emitOpError("expects 'buf_id' in range [0, 31]");
  }

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0) {
      return op->emitOpError("expects 'mode' to be non-negative");
    }
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

// ---- GetBufDynOp / RlsBufDynOp ----
static LogicalResult verifyBufDynSyncOp(Operation *op, Attribute opTypeAttr,
                                        Value bufId, IntegerAttr modeAttr) {
  if (!opTypeAttr) {
    return op->emitOpError("expects 'op_type' attribute");
  }

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe)) {
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
  }

  if (!bufId) {
    return op->emitOpError("expects 'buf_id' operand");
  }

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0) {
      return op->emitOpError("expects 'mode' to be non-negative");
    }
  }

  return success();
}

LogicalResult GetBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

LogicalResult RlsBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid membar token: " << token;
    }
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr) {
    return parser.emitError(loc, "expected membar attribute");
  }
  attr = memBarAttr;
  return success();
}

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrDsbMem(OpAsmParser &parser,
                                           DsbMemAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDsbMem(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dsb memory token: " << token;
    }
    attr = DsbMemAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dsbMemAttr = dyn_cast<DsbMemAttr>(parsed);
  if (!dsbMemAttr) {
    return parser.emitError(loc, "expected dsb_mem attribute");
  }
  attr = dsbMemAttr;
  return success();
}

static void printLegacyOrAttrDsbMem(OpAsmPrinter &printer, Operation *op,
                                    DsbMemAttr mem) {
  (void)op;
  printer << ' ' << '"' << stringifyDsbMem(mem.getKind()) << '"';
}

static ParseResult parseLegacyOrAttrDcciCacheLine(OpAsmParser &parser,
                                                  DcciCacheLineAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciCacheLine(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci cache token: " << token;
    }
    attr = DcciCacheLineAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto cacheAttr = dyn_cast<DcciCacheLineAttr>(parsed);
  if (!cacheAttr) {
    return parser.emitError(loc, "expected dcci_cache_line attribute");
  }
  attr = cacheAttr;
  return success();
}

static void printLegacyOrAttrDcciCacheLine(OpAsmPrinter &printer, Operation *op,
                                           DcciCacheLineAttr cache) {
  (void)op;
  printer << ' ' << '"' << stringifyDcciCacheLine(cache.getKind()) << '"';
}

static ParseResult parseOptionalDcciDst(OpAsmParser &parser,
                                        DcciDstAttr &attr) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciDst(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci dst token: " << token;
    }
    attr = DcciDstAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dstAttr = dyn_cast<DcciDstAttr>(parsed);
  if (!dstAttr) {
    return parser.emitError(loc, "expected dcci_dst attribute");
  }
  attr = dstAttr;
  return success();
}

static void printOptionalDcciDst(OpAsmPrinter &printer, Operation *op,
                                 DcciDstAttr dst) {
  (void)op;
  if (!dst) {
    return;
  }
  printer << ", \"" << stringifyDcciDst(dst.getKind()) << '"';
}

LogicalResult DcciOp::verify() {
  auto space = getPTOMemorySpaceEnum(getPtr().getType());
  if (!space) {
    return emitOpError("expects ptr to have a PTO memory space");
  }
  if (*space != pto::AddressSpace::GM && *space != pto::AddressSpace::VEC) {
    return emitOpError("expects ptr memory space to be gm or ub/vec");
  }

  return success();
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << token;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto pipe = symbolizePIPE(keyword);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr) {
    return parser.emitError(loc, "expected pipe attribute");
  }
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << token;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto event = symbolizeEVENT(keyword);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << keyword;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr) {
    return parser.emitError(loc, "expected event attribute");
  }
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value))) {
    return failure();
  }
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    return parser.emitError(loc, "expected 32-bit integer literal");
  }
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
  return success();
}

static void printLegacySyncTriplet(OpAsmPrinter &p, PipeAttr srcPipe,
                                   PipeAttr dstPipe, EventAttr eventId,
                                   ArrayRef<NamedAttribute> attrs) {
  p << "[<" << stringifyPIPE(srcPipe.getPipe()) << ">, <"
    << stringifyPIPE(dstPipe.getPipe()) << ">, <"
    << stringifyEVENT(eventId.getEvent()) << ">]";
  p.printOptionalAttrDict(attrs, {"src_pipe", "dst_pipe", "event_id"});
}

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("kind", kind);
  return success();
}

void MemBarOp::print(OpAsmPrinter &p) {
  printLegacyOrAttrMemBar(p, getKind(), (*this)->getAttrs());
}

static ParseResult parseBufSyncOp(OpAsmParser &parser, OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr bufIdAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;
}

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr)) {
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
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr)) {
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
  result.addAttribute("buf_id", bufIdAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                           IntegerAttr bufIdAttr, IntegerAttr modeAttr,
                           ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", "
      << bufIdAttr.getInt() << ", " << modeAttr.getInt();
  } else if (auto pipeEventType = dyn_cast<PipeEventTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else if (auto syncOpType = dyn_cast<SyncOpTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "buf_id", "mode"});
}

ParseResult GetBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void GetBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

ParseResult RlsBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

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
