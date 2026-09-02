// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyInternalOddSplitSupport(Operation *op,
                                                   Value pipeHandle,
                                                   int64_t split,
                                                   bool producerSide) {
  if (!isOddSplit(split)) {
    return success();
  }

  bool isCubeSide = isInsideCubeKernelOrSection(op);
  bool isVectorSide = isInsideVectorKernelOrSection(op);
  bool isC2VSide = producerSide ? isCubeSide : isVectorSide;
  bool isV2CSide = producerSide ? isVectorSide : isCubeSide;
  int8_t directionMask = isC2VSide ? 1 : (isV2CSide ? 2 : 0);
  if (isV2CSide && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }
  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  Value consumerBuffer;
  if (initOp && directionMask != 0) {
    consumerBuffer = directionMask == 2 && initOp.getDirMask() == 3
                         ? initOp.getPeerLocalAddr()
                         : initOp.getLocalAddr();
  }
  if (!initOp || directionMask == 0 ||
      (initOp.getDirMask() & directionMask) == 0 || !consumerBuffer) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for a "
        "pto.initialize_l2g2l_pipe whose dir_mask enables the operation "
        "direction and provides its local consumer buffer");
  }
  return success();
}

static bool getTensorLikeElementAndShape(Type ty, Type &elementType,
                                         ArrayRef<int64_t> &shape) {
  if (auto tvTy = dyn_cast<TensorViewType>(ty)) {
    elementType = tvTy.getElementType();
    shape = tvTy.getShape();
    return true;
  }
  return false;
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use a pipe produced by "
              "pto.initialize_l2g2l_pipe";
  }
  if (initOp.getLocalAddr()) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use global-only "
              "pto.initialize_l2g2l_pipe without local_addr";
  }

  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor_view slot type";
  }

  if (slotElementType != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match initialize_l2g2l_pipe "
              "gm_addr element type";
  }
  if (slotShape.size() != static_cast<size_t>(entryViewTy.getRank())) {
    return op->emitOpError()
           << "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr "
              "rank";
  }

  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension "
           << slotDim;
  }

  if (auto entryElemCount = getStaticElementCount(entryShape)) {
    uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
    uint64_t entryBytes = *entryElemCount * elemBytes;
    if (elemBytes != 0) {
      int8_t split = 0;
      if (auto alloc = dyn_cast<TAllocOp>(op)) {
        split = alloc.getSplit();
      } else if (auto push = dyn_cast<TPushOp>(op)) {
        split = push.getSplit();
      } else if (auto pop = dyn_cast<TPopOp>(op)) {
        split = pop.getSplit();
      } else if (auto free = dyn_cast<TFreeOp>(op)) {
        split = free.getSplit();
      }

      uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
      bool isSplitEntry = split != 0;
      bool byteSizeMatches =
          entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
      if (!byteSizeMatches) {
        return op->emitOpError()
               << "expects pipe entry byte size to match initialize_l2g2l_pipe "
                  "slot_size"
               << (isSplitEntry ? " or half slot_size for split entries" : "")
               << " (got entry byte size = " << entryBytes
               << ", slot_size = " << initOp.getSlotSize() << ")";
      }
    }
  }

  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  Type scratchTy = getScratch().getType();
  if (!isa<pto::TileBufType>(scratchTy)) {
    return emitOpError("expects scratch to be tile_buf type");
  }

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC) {
    return emitOpError("expects scratch to be in vec address space");
  }

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > 2) {
    return emitOpError("expects scratch to be rank-1 or rank-2");
  }
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic) {
      return emitOpError("expects scratch to have a static shape");
    }
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes) {
    return emitOpError("expects scratch byte size to be statically known");
  }
  if (*scratchBytes < sizeof(uint64_t)) {
    return emitOpError("expects scratch to provide at least 8 bytes");
  }

  auto workspaceTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!workspaceTy) {
    return emitOpError("expects workspace to be !pto.ptr type");
  }
  Type workspaceElemTy = workspaceTy.getElementType();
  if (!isByteIntegerType(workspaceElemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }

  if (auto syncIdAttr = getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < 0 || syncId > 7) {
      return emitOpError("expects sync_id in range [0, 7]");
    }
  }
  if (auto blockBytesAttr = getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0) {
      return emitOpError("expects block_bytes to be greater than 0");
    }
  }
  if (auto commBlockOffsetAttr = getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0) {
      return emitOpError("expects comm_block_offset to be non-negative");
    }
  }
  if (auto queueNumAttr = getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0) {
      return emitOpError("expects queue_num to be greater than 0");
    }
  }
  if (auto channelGroupIdxAttr = getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative()) {
      return emitOpError("expects channel_group_idx to be non-negative");
    }
    if (value.ugt(UINT32_MAX)) {
      return emitOpError("expects channel_group_idx to fit in uint32");
    }
  }

  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy) {
    return op->emitOpError("expects src and dst to have element types");
  }
  if (dstElemTy != srcElemTy) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src"))) {
    return failure();
  }
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType())) {
    return op->emitOpError("expects src and dst to have the same static shape");
  }
  return success();
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TPutOp::verify() {
  if (failed(verifyCommGlobalLike(
          *this, getDst(), "dst",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommGlobalLike(
          *this, getSrc(), "src",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong"))) {
    return failure();
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType())) {
    return emitOpError(
        "expects src and dst to have the same static/dynamic shape signature");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src/dst");
  }
  return success();
}

LogicalResult TGetOp::verify() {
  if (failed(verifyCommGlobalLike(
          *this, getDst(), "dst",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommGlobalLike(
          *this, getSrc(), "src",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong"))) {
    return failure();
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType())) {
    return emitOpError(
        "expects src and dst to have the same static/dynamic shape signature");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src/dst");
  }
  return success();
}

LogicalResult TNotifyOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != 32) {
    return emitOpError("expects value to be i32");
  }
  return success();
}

LogicalResult TWaitOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32) {
    return emitOpError("expects cmp_value to be i32");
  }
  return success();
}

LogicalResult TTestOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32) {
    return emitOpError("expects cmp_value to be i32");
  }
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  Type elemType;
  SmallVector<int64_t, 4> shape;
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    if (ptrTy.getMemorySpace().getAddressSpace() != pto::AddressSpace::GM) {
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
    }
    elemType = ptrTy.getElementType();
  } else if (isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    elemType = getElemTy(ty);
    shape = getShapeVec(ty);
  } else {
    return op->emitOpError()
           << "expects " << name
           << " to be a GM ptr/tensor_view/partition_view";
  }

  auto elemTy = dyn_cast<IntegerType>(elemType);
  if (!elemTy || elemTy.getWidth() != 32) {
    return op->emitOpError() << "expects " << name << " element type to be i32";
  }

  // A pointer does not carry capacity metadata. It is lowered as the fixed
  // 16 x i32 workspace required by PTO-ISA; allocation size remains a runtime
  // responsibility.
  if (isa<pto::PtrType>(ty)) {
    return success();
  }

  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0) {
      return op->emitOpError() << "expects " << name << " shape to be positive";
    }
  }

  constexpr int64_t kMinWorkspaceElements = 16;
  if (!llvm::is_contained(shape, ShapedType::kDynamic)) {
    int64_t staticCapacity = 1;
    for (int64_t dim : shape) {
      int64_t product = 0;
      if (llvm::MulOverflow(staticCapacity, dim, product)) {
        staticCapacity = std::numeric_limits<int64_t>::max();
        break;
      }
      staticCapacity = product;
    }
    if (staticCapacity < kMinWorkspaceElements) {
      return op->emitOpError()
             << "expects " << name << " to contain at least "
             << kMinWorkspaceElements
             << " i32 elements (64 bytes), but static capacity is "
             << staticCapacity;
    }
  }

  return success();
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  auto mode = getMode().getValue();

  if (mode == pto::SyncAllMode::Hard) {
    if (hasGm || getUsedCores()) {
      return emitOpError(
          "expects hard syncall to have no gm_workspace or used_cores");
    }
    return success();
  }

  if (!hasGm) {
    return emitOpError("expects soft syncall to provide gm_workspace");
  }
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace"))) {
    return failure();
  }

  return success();
}

LogicalResult TBroadcastOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getSrc().getType() != getGroup().front().getType()) {
    return emitOpError("expects src type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src");
  }
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects dst element type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType())) {
    return emitOpError("expects staging tile element type to match dst");
  }
  return success();
}

LogicalResult CommTScatterOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects src element type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src");
  }
  return success();
}

LogicalResult TReduceOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects dst element type to match group member type");
  }
  if (getAcc().getType() != getRecvPing().getType()) {
    return emitOpError("expects acc and recv_ping to have identical types");
  }
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType())) {
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  }
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  if (failed(verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube"))) {
    return failure();
  }

  auto accPushEpilogue = getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success();
  }

  auto peerConsumerInitOr = lookupFixpipePeerConsumerInit(*this);
  if (failed(peerConsumerInitOr)) {
    return emitOpError()
           << "expects peer consumer function to contain a matching "
              "aiv_initialize_pipe with the same consumer buffer contract";
  }

  Operation *peerConsumerInit = *peerConsumerInitOr;
  auto peerAccPushEpilogue = getAccPushEpilogueFromInitOp(peerConsumerInit);
  if (!peerAccPushEpilogue) {
    return emitOpError()
           << "expects peer consumer pipe init to also have "
              "'acc_push_epilogue' for fixpipe contract consistency";
  }

  if (peerAccPushEpilogue.getLayout() != accPushEpilogue.getLayout()) {
    return emitOpError()
           << "expects acc_push_epilogue.layout to match peer consumer "
           << "(producer has " << stringifyFixpipeLayout(accPushEpilogue.getLayout())
           << ", consumer has " << stringifyFixpipeLayout(peerAccPushEpilogue.getLayout())
           << ")";
  }
  if (peerAccPushEpilogue.getQuant() != accPushEpilogue.getQuant()) {
    return emitOpError()
           << "expects acc_push_epilogue.quant to match peer consumer "
           << "(producer has " << stringifyFixpipeQuant(accPushEpilogue.getQuant())
           << ", consumer has " << stringifyFixpipeQuant(peerAccPushEpilogue.getQuant())
           << ")";
  }
  if (peerAccPushEpilogue.getRelu() != accPushEpilogue.getRelu()) {
    return emitOpError()
           << "expects acc_push_epilogue.relu to match peer consumer "
           << "(producer has " << stringifyFixpipeRelu(accPushEpilogue.getRelu())
           << ", consumer has " << stringifyFixpipeRelu(peerAccPushEpilogue.getRelu())
           << ")";
  }

  return success();
}
LogicalResult AivInitializePipeOp::verify() {
  if (failed(verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector"))) {
    return failure();
  }

  // Rule 3 & 22: Peer fixpipe contract verification
  auto accPushEpilogue = getAccPushEpilogueAttr();
  if (accPushEpilogue) {
    // This is a consumer-side fixpipe pipe
    // Need to find the corresponding producer-side init and verify contract match

    // Trace c2v_consumer_buf to reserve_buffer or import_reserved_buffer
    if (!getC2vConsumerBuf()) {
      return emitOpError(
          "expects fixpipe consumer pipe to have 'c2v_consumer_buf'");
    }

    Value c2vBuf = getC2vConsumerBuf();
    Operation *bufDefOp = c2vBuf.getDefiningOp();

    func::FuncOp peerProducerFunc;
    auto currentConsumerFunc = getOperation()->getParentOfType<func::FuncOp>();
    if (!currentConsumerFunc) {
      return emitOpError("must be nested under a func.func");
    }
    StringRef bufferName;

    if (auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(bufDefOp)) {
      return emitOpError(
          "expects consumer-side fixpipe pipe to use reserve_buffer, not import_reserved_buffer");
    } else if (auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(bufDefOp)) {
      bufferName = reserveOp.getName();

      ModuleOp moduleOp = currentConsumerFunc->getParentOfType<ModuleOp>();
      if (!moduleOp) {
        return emitOpError("must be nested under a module for fixpipe contract verification");
      }

      ImportReservedBufferOp matchedImport;
      unsigned matchedImportCount = 0;
      moduleOp.walk([&](ImportReservedBufferOp candidateImport) {
        if (candidateImport.getName() != bufferName) {
          return WalkResult::advance();
        }
        auto peerConsumerFunc =
            lookupPeerFuncAcrossContainer(candidateImport.getOperation(),
                                          candidateImport.getPeerFuncAttr());
        if (peerConsumerFunc != currentConsumerFunc) {
          return WalkResult::advance();
        }
        matchedImport = candidateImport;
        ++matchedImportCount;
        return WalkResult::advance();
      });

      if (matchedImportCount == 0) {
        return emitOpError()
               << "cannot find peer import_reserved_buffer for consumer buffer '"
               << bufferName << "'";
      }
      if (matchedImportCount > 1) {
        return emitOpError()
               << "finds multiple peer import_reserved_buffer ops for consumer buffer '"
               << bufferName << "'";
      }

      peerProducerFunc = matchedImport->getParentOfType<func::FuncOp>();
    } else {
      return emitOpError(
          "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
          "import_reserved_buffer for peer contract verification");
    }

    if (!peerProducerFunc) {
      return emitOpError("cannot find peer producer function for fixpipe contract verification");
    }

    auto peerProducerInitOr = lookupFixpipePeerProducerInit(
        *this, peerProducerFunc, bufferName, currentConsumerFunc);
    if (failed(peerProducerInitOr)) {
      return failure();
    }

    Operation *peerProducerInitOp = *peerProducerInitOr;
    auto peerProducerFrontendInit =
        dyn_cast<AicInitializePipeOp>(peerProducerInitOp);
    std::optional<uint32_t> peerProducerId;
    if (peerProducerFrontendInit) {
      if (!peerProducerFrontendInit.getC2vConsumerBuf()) {
        return emitOpError()
               << "expects peer producer aic_initialize_pipe to have 'c2v_consumer_buf'";
      }

      Operation *peerBufDefOp =
          peerProducerFrontendInit.getC2vConsumerBuf().getDefiningOp();
      auto peerImportOp = dyn_cast_or_null<ImportReservedBufferOp>(peerBufDefOp);
      if (!peerImportOp) {
        return emitOpError()
               << "expects peer producer aic_initialize_pipe to use import_reserved_buffer "
               << "for c2v_consumer_buf";
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(peerImportOp.getOperation(),
                                        peerImportOp.getPeerFuncAttr());
      if (peerImportOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return emitOpError()
               << "cannot find matching producer aic_initialize_pipe for buffer '"
               << bufferName << "' in peer function";
      }
      peerProducerId = peerProducerFrontendInit.getId();
    } else if (isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(
                   peerProducerInitOp)) {
      auto frontendIdAttr =
          peerProducerInitOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
      if (!frontendIdAttr) {
        return emitOpError()
               << "expects lowered peer producer fixpipe pipe to retain "
               << kFrontendPipeIdAttrName;
      }
      peerProducerId = static_cast<uint32_t>(frontendIdAttr.getInt());
    }
    if (!peerProducerId) {
      return emitOpError()
             << "expects peer producer fixpipe contract to resolve to "
                "frontend or lowered aic_initialize_pipe";
    }

    // Verify peer producer also has acc_push_epilogue
    auto peerAccPushEpilogue = getAccPushEpilogueFromInitOp(peerProducerInitOp);
    if (!peerAccPushEpilogue) {
      return emitOpError()
             << "expects peer producer pipe init to also have "
             << "'acc_push_epilogue' for fixpipe contract consistency";
    }

    // Verify layout/quant/relu match
    if (peerAccPushEpilogue.getLayout() != accPushEpilogue.getLayout()) {
      return emitOpError()
             << "expects acc_push_epilogue.layout to match peer producer "
             << "(consumer has " << stringifyFixpipeLayout(accPushEpilogue.getLayout())
             << ", producer has " << stringifyFixpipeLayout(peerAccPushEpilogue.getLayout())
             << ")";
    }

    if (peerAccPushEpilogue.getQuant() != accPushEpilogue.getQuant()) {
      return emitOpError()
             << "expects acc_push_epilogue.quant to match peer producer "
             << "(consumer has " << stringifyFixpipeQuant(accPushEpilogue.getQuant())
             << ", producer has " << stringifyFixpipeQuant(peerAccPushEpilogue.getQuant())
             << ")";
    }

    if (peerAccPushEpilogue.getRelu() != accPushEpilogue.getRelu()) {
      return emitOpError()
             << "expects acc_push_epilogue.relu to match peer producer "
             << "(consumer has " << stringifyFixpipeRelu(accPushEpilogue.getRelu())
             << ", producer has " << stringifyFixpipeRelu(peerAccPushEpilogue.getRelu())
             << ")";
    }

    SmallVector<TPopFromAicOp> matchingPops;
    currentConsumerFunc.walk([&](TPopFromAicOp pop) {
      if (pop.getId() == getId()) {
        matchingPops.push_back(pop);
      }
      return WalkResult::advance();
    });

    if (matchingPops.empty()) {
      return emitOpError()
             << "expects at least one tpop_from_aic for fixpipe pipe id = "
             << getId() << " to resolve the consumer entry type";
    }

    Type resolvedConsumerTileType = matchingPops.front().getTile().getType();
    for (TPopFromAicOp pop : llvm::drop_begin(matchingPops)) {
      if (pop.getTile().getType() != resolvedConsumerTileType) {
        return emitOpError()
               << "expects all tpop_from_aic results for fixpipe pipe id = "
               << getId() << " to use the same tile type";
      }
    }

    auto resolvedConsumerTileTy =
        dyn_cast<pto::TileBufType>(resolvedConsumerTileType);
    if (!resolvedConsumerTileTy) {
      return emitOpError()
             << "expects fixpipe consumer tpop result to be !pto.tile_buf";
    }

    Type resolvedConsumerElemTy = resolvedConsumerTileTy.getElementType();
    auto quant = accPushEpilogue.getQuant();
    if (!matchesFixpipeConsumerElementType(quant, resolvedConsumerElemTy)) {
      return emitOpError()
             << "expects consumer element type to match acc_push_epilogue.quant "
             << stringifyFixpipeQuant(quant);
    }
    if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(),
                                      resolvedConsumerTileTy)) {
      return emitOpError()
             << "expects consumer tile layout to match acc_push_epilogue.layout "
             << stringifyFixpipeLayout(accPushEpilogue.getLayout());
    }

    auto getInitSlotSize = [](Operation *initOp) -> std::optional<uint32_t> {
      if (auto aicInit = dyn_cast_or_null<AicInitializePipeOp>(initOp)) {
        return aicInit.getSlotSize();
      }
      if (auto aivInit = dyn_cast_or_null<AivInitializePipeOp>(initOp)) {
        return aivInit.getSlotSize();
      }
      if (auto l2lInit = dyn_cast_or_null<InitializeL2LPipeOp>(initOp)) {
        return l2lInit.getSlotSize();
      }
      if (auto l2g2lInit = dyn_cast_or_null<InitializeL2G2LPipeOp>(initOp)) {
        return l2g2lInit.getSlotSize();
      }
      return std::nullopt;
    };

    if (auto requiredSlotBytes = getStaticTileByteSize(resolvedConsumerTileTy)) {
      if (static_cast<uint64_t>(getSlotSize()) < *requiredSlotBytes) {
        return emitOpError()
               << "expects consumer-side fixpipe slot_size to be at least "
               << *requiredSlotBytes
               << " bytes for the resolved post-fixpipe consumer entry";
      }
      auto peerProducerSlotSize = getInitSlotSize(peerProducerInitOp);
      if (peerProducerSlotSize &&
          static_cast<uint64_t>(*peerProducerSlotSize) < *requiredSlotBytes) {
        return emitOpError()
               << "expects peer producer fixpipe slot_size to be at least "
               << *requiredSlotBytes
               << " bytes for the resolved post-fixpipe consumer entry";
      }
    }

    bool producerConsumerTypeMismatch = false;
    peerProducerFunc.walk([&](TPushToAivOp push) {
      if (push.getId() != *peerProducerId) {
        return WalkResult::advance();
      }
      auto srcTileTy = dyn_cast<pto::TileBufType>(push.getTile().getType());
      if (!srcTileTy) {
        return WalkResult::advance();
      }
      if (!matchesFixpipeProducerAndConsumerTypes(
              quant, srcTileTy.getElementType(), resolvedConsumerElemTy)) {
        producerConsumerTypeMismatch = true;
        emitOpError()
            << "expects producer source element type and consumer tpop result "
               "element type to satisfy acc_push_epilogue.quant "
            << stringifyFixpipeQuant(quant);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (producerConsumerTypeMismatch) {
      return failure();
    }
  }

  return success();
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
        if (aicInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TALLOC to have split = 0");
          }
        }
      }
    }
  }

  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                  getTile().getType()))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }

  // Fixpipe validation
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aicInit = dyn_cast<AicInitializePipeOp>(initOp);
  if (!aicInit) {
    return success(); // Not an AIC init, no fixpipe check needed
  }

  auto accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // No fixpipe config, normal pipe
  }

  // Rule 9: Source tile must be acc tile
  auto tileTy = dyn_cast<pto::TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError(
        "expects fixpipe TPUSH source tile to be a tile type");
  }
  auto tileSpace = getPTOMemorySpaceEnum(tileTy);
  if (!tileSpace || *tileSpace != pto::AddressSpace::ACC) {
    return emitOpError("expects fixpipe TPUSH source tile to use loc=acc");
  }

  // Rule 2: split must be 0 for fixpipe
  if (getSplit() != 0) {
    return emitOpError(
        "expects fixpipe TPUSH to have split = 0");
  }

  // Rule 10: Check source element type matches quant mode
  Type elemTy = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();

  bool srcTypeValid = true;
  if (quant == pto::FixpipeQuant::NoConvert) {
    // no_convert requires f32 or i32, and consumer type must match
    if (!elemTy.isF32() && !elemTy.isInteger(32)) {
      srcTypeValid = false;
    }
  } else if (quant == pto::FixpipeQuant::F32F16 ||
             quant == pto::FixpipeQuant::F32BF16 ||
             quant == pto::FixpipeQuant::QF322B8PreScalar ||
             quant == pto::FixpipeQuant::QF322B8PreVec ||
             quant == pto::FixpipeQuant::QF322F16PreScalar ||
             quant == pto::FixpipeQuant::QF322BF16PreScalar ||
             quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
             quant == pto::FixpipeQuant::QF322FP8PreScalar) {
    // f32-based quant modes
    if (!elemTy.isF32()) {
      srcTypeValid = false;
    }
  } else if (quant == pto::FixpipeQuant::REQ8Scalar ||
             quant == pto::FixpipeQuant::REQ8Vec ||
             quant == pto::FixpipeQuant::DEQF16Scalar ||
             quant == pto::FixpipeQuant::DEQF16Vec ||
             quant == pto::FixpipeQuant::QS322BF16PreScalar ||
             quant == pto::FixpipeQuant::QS322BF16PreVec) {
    // i32-based quant modes
    if (!elemTy.isInteger(32)) {
      srcTypeValid = false;
    }
  }

  if (!srcTypeValid) {
    return emitOpError()
           << "expects fixpipe TPUSH source element type to match "
           << "acc_push_epilogue.quant mode requirements";
  }

  // Rule 16-17: Check for required set_quant_scalar/vector ops
  bool isScalarQuant = isScalarFixpipeQuant(quant);
  bool isVectorQuant = isVectorFixpipeQuant(quant);

  if (isScalarQuant || isVectorQuant) {
    Block *tpushBlock = getOperation()->getBlock();
    bool foundQuantConfig = false;
    for (Operation &op : tpushBlock->getOperations()) {
      if (&op == getOperation()) {
        break;
      }

      if (isScalarQuant) {
        if (auto setQuant = dyn_cast<SetQuantScalarOp>(&op)) {
          if (setQuant.getId() == getId()) {
            foundQuantConfig = true;
          }
        }
      } else if (isVectorQuant) {
        if (auto setQuant = dyn_cast<SetQuantVectorOp>(&op)) {
          if (setQuant.getId() == getId()) {
            foundQuantConfig = true;
          }
        }
      }
    }

    if (!foundQuantConfig) {
      if (isScalarQuant) {
        return emitOpError()
               << "expects a preceding pto.set_quant_scalar with id = "
               << getId() << " in the same block";
      }
      return emitOpError()
             << "expects a preceding pto.set_quant_vector with id = "
             << getId() << " in the same block";
    }
  }

  return success();
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                  getTile().getType()))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  return verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                    getSplit(), getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                                 /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  return verifyFixpipeConsumerType(getOperation(), getId(), getTile().getType());
}

LogicalResult TPopFromAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
        if (aivInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TFREE to have split = 0");
          }
        }
      }
    }
  }

  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry()) {
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  }
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
        if (aivInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TFREE to have split = 0");
          }
        }
      }
    }
  }

  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry()) {
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  }
  return success();
}

LogicalResult SetQuantScalarOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  // Look up the referenced pipe
  auto initOr =
      lookupFrontendOrLoweredInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  pto::AccPushEpilogueAttr accPushEpilogue;
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  } else if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    accPushEpilogue = l2lInit.getAccPushEpilogueAttr();
  } else if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    accPushEpilogue = l2g2lInit.getAccPushEpilogueAttr();
  } else {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference an aic_initialize_pipe or lowered producer pipe";
  }

  if (!accPushEpilogue) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a fixpipe pipe (with acc_push_epilogue)";
  }

  // Check that quant mode is scalar
  auto quant = accPushEpilogue.getQuant();
  bool isScalarQuant = isScalarFixpipeQuant(quant);

  if (!isScalarQuant) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with scalar quantization mode, but found non-scalar mode";
  }

  // Verify scale operand is f32 to match SET_QUANT_SCALAR(float)
  if (!getScale().getType().isF32()) {
    return emitOpError("expects 'scale' to be f32");
  }

  return success();
}

LogicalResult SetQuantVectorOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  // Look up the referenced pipe
  auto initOr =
      lookupFrontendOrLoweredInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  pto::AccPushEpilogueAttr accPushEpilogue;
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  } else if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    accPushEpilogue = l2lInit.getAccPushEpilogueAttr();
  } else if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    accPushEpilogue = l2g2lInit.getAccPushEpilogueAttr();
  } else {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference an aic_initialize_pipe or lowered producer pipe";
  }

  if (!accPushEpilogue) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a fixpipe pipe (with acc_push_epilogue)";
  }

  // Check that quant mode is vector
  auto quant = accPushEpilogue.getQuant();
  bool isVectorQuant = isVectorFixpipeQuant(quant);

  if (!isVectorQuant) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with vector quantization mode, but found non-vector mode";
  }

  Type scalingTy = getScalingTile().getType();
  if (!isa<pto::TileBufType>(scalingTy)) {
    return emitOpError("expects 'scaling_tile' to be a tile type");
  }
  auto scalingSpace = getPTOMemorySpaceEnum(scalingTy);
  if (!scalingSpace || *scalingSpace != pto::AddressSpace::SCALING) {
    return emitOpError("expects 'scaling_tile' to use loc=scaling");
  }
  Type scalingElemTy = getElemTy(scalingTy);
  PTOArch arch = getTargetArch(getOperation());
  if (!isFixpipeQuantPayloadElemType(scalingElemTy, arch)) {
    if (arch == PTOArch::A3) {
      return emitOpError(
          "expects 'scaling_tile' element type to be packed i64/ui64 on A3");
    }
    return emitOpError(
        "expects 'scaling_tile' element type to be f16, bf16, or f32 on A5");
  }

  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  if (!getLocalAddr()) {
    if (getPeerLocalAddr()) {
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    }
    if (getLocalSlotNumAttr()) {
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    }
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0) {
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    }
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum()) {
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
    }
  }

  if (getDirMask() == 3 && !getPeerLocalAddr()) {
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  }
  if (getDirMask() != 3 && getPeerLocalAddr()) {
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  }
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  if (getDirMask() == 3 && !getPeerLocalAddr()) {
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  }
  if (getDirMask() != 3 && getPeerLocalAddr()) {
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  }
  return success();
}

LogicalResult TPushOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/true))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (isInsideCubeKernelOrSection(getOperation()) &&
      failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType()))) {
    return failure();
  }
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED) {
    return emitOpError("tile type must map to a supported producer pipe");
  }
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/true))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

LogicalResult TPopOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/false))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (isInsideCubeKernelOrSection(getOperation()) &&
      failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType()))) {
    return failure();
  }
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED) {
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  }
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/false))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;

  if (parser.parseLParen() || parser.parseOperand(first)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalComma())) {
    hasEntry = true;
    if (parser.parseOperand(pipe) || parser.parseColonType(firstTy) ||
        parser.parseComma() || parser.parseType(pipeTy) || parser.parseRParen()) {
      return failure();
    }
  } else {
    if (parser.parseColonType(pipeTy) || parser.parseRParen()) {
      return failure();
    }
    pipe = first;
  }

  NamedAttrList attrs;
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual()) {
    return failure();
  }
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  result.addAttributes(attrs);
  if (hasEntry &&
      parser.resolveOperand(first, firstTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(pipe, pipeTy, result.operands)) {
    return failure();
  }
  return success();
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 32) {
      return 1;
    }
    if (intType.getWidth() == 64) {
      return 2;
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
  return std::nullopt;
}

static Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

static Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == 2) {
    if ((slot % 2) != 0) {
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    }
    if (slot + 1 >= kSimtKeepResumeSlotLimit) {
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - 2)
             << "] for 64-bit keep/resume values";
    }
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto resume = dyn_cast<ResumeOp>(cur);
    if (!resume) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) &&
        resume.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << resume.getSlot()
             << " in the SIMT resume prologue group";
    }
  }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto keep = dyn_cast<KeepOp>(cur);
    if (!keep) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) &&
        keep.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << keep.getSlot()
             << " in the SIMT keep epilogue group";
    }
    if (cur == last) {
      break;
    }
  }
  return success();
}

static bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() <= 64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static bool isInsideSimtExecutionScope(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  return (func && func->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static LogicalResult verifyInsideSimtExecutionScope(Operation *op) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName
           << "' or inside pto.section.simt";
  }
  return success();
}

static LogicalResult verifySimtKeepResumeCommon(Operation *op, int64_t slot) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "' or inside pto.section.simt";
  }
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op->emitOpError("requires slot in range [0, ")
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  return success();
}

LogicalResult SyncthreadsOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceBlockOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

void SyncthreadsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceBlockOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}
