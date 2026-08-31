// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

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

LogicalResult KeepOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getPayload().getType())) {
    return emitOpError()
           << "supports integer scalar payloads up to 64 bits and "
              "f16/bf16/f32 payloads";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }

  Block *block = getOperation()->getBlock();
  bool insideSection =
      getOperation()->getParentOfType<SectionSimtOp>() != nullptr;
  Operation *lastPayloadOp = nullptr;
  if (insideSection) {
    if (!block->empty()) {
      lastPayloadOp = &block->back();
    }
  } else {
    Operation *terminator = block->getTerminator();
    if (isa<func::ReturnOp>(terminator)) {
      lastPayloadOp = terminator->getPrevNode();
    }
  }
  if (!lastPayloadOp) {
    return emitOpError(
        "must be placed in the SIMT epilogue before func.return or the end "
        "of pto.section.simt");
  }

  Operation *cur = lastPayloadOp;
  while (cur && isa<SyncthreadsOp>(cur)) {
    cur = cur->getPrevNode();
  }
  Operation *lastKeep = cur;
  if (!lastKeep || !isa<KeepOp>(lastKeep)) {
    return emitOpError()
           << "must be placed in the SIMT epilogue before func.return; only "
              "'pto.syncthreads' may appear between the final 'pto.keep' group "
              "and func.return or the end of pto.section.simt";
  }

  Operation *firstKeep = lastKeep;
  while (Operation *prev = firstKeep->getPrevNode()) {
    if (!isa<KeepOp>(prev)) {
      break;
    }
    firstKeep = prev;
  }
  if (!isOpInRange(getOperation(), firstKeep, lastKeep)) {
    return emitOpError()
           << "must be in the contiguous SIMT keep epilogue group immediately "
              "before optional 'pto.syncthreads' and func.return or the end "
              "of pto.section.simt";
  }
  if (failed(verifyUniqueKeepGroupSlots(*this, firstKeep, lastKeep))) {
    return failure();
  }
  return success();
}

LogicalResult ResumeOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getResult().getType())) {
    return emitOpError()
           << "supports integer scalar results up to 64 bits and "
              "f16/bf16/f32 results";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }
  Block *block = getOperation()->getBlock();
  Operation *first = getFirstNonConstantLikeOp(block);
  if (!first || !isa<ResumeOp>(first)) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }

  bool found = false;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (!isa<ResumeOp>(cur)) {
      break;
    }
    if (cur == getOperation()) {
      found = true;
      break;
    }
  }
  if (!found) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }
  if (failed(verifyUniqueResumeGroupSlots(*this, first))) {
    return failure();
  }
  return success();
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TBroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Write::get());
  }
}

void CommTGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Read::get());
  }
}

void CommTScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Write::get());
  }
}

void TReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Write::get());
  if (getRecvPong()) {
    auto recvPongRange = getRecvPongMutable();
    if (auto it = recvPongRange.begin(); it != recvPongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Read::get());
  }
}

void WaitAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TestAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2G2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getGmAddrMutable(), MemoryEffects::Read::get());
  auto localAddr = getLocalAddrMutable();
  if (!localAddr.empty()) {
    addEffect(effects, &*localAddr.begin(), MemoryEffects::Read::get());
  }
  auto peerLocalAddr = getPeerLocalAddrMutable();
  if (!peerLocalAddr.empty()) {
    addEffect(effects, &*peerLocalAddr.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getLocalAddrMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPushOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());

  if (auto pipeId = getFrontendPipeIdFromHandle(getPipeHandle())) {
    auto accPushEpilogue =
        getAccPushEpilogueFromInitOp(getPipeHandle().getDefiningOp());
    if (accPushEpilogue &&
        (isScalarFixpipeQuant(accPushEpilogue.getQuant()) ||
         isVectorFixpipeQuant(accPushEpilogue.getQuant()))) {
      effects.emplace_back(MemoryEffects::Read::get(),
                           getFixpipeQuantStateIdAttr(getOperation(), *pipeId),
                           FixpipeQuantStateResource::get());
    }
  }
}

void TPushToAivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());

  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return;
  }

  auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return;
  }

  auto aicInit = dyn_cast<AicInitializePipeOp>(*initOr);
  if (!aicInit) {
    return;
  }

  auto accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return;
  }

  auto quant = accPushEpilogue.getQuant();
  if (isScalarFixpipeQuant(quant) || isVectorFixpipeQuant(quant)) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         getFixpipeQuantStateIdAttr(getOperation(), getId()),
                         FixpipeQuantStateResource::get());
  }
}

void TAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEntryMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TPopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getTileMutable(), MemoryEffects::Write::get());
}

void TFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto entry = getEntryMutable();
  if (!entry.empty()) {
    addEffect(effects, &*entry.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void SetQuantScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScaleMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

void SetQuantVectorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScalingTileMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

static constexpr const char kConvertRoundingKeywords[] = "r/a/f/c/z/o/h";

static ParseResult parseConvertRounding(OpAsmParser &parser,
                                        RoundingAttr &roundingAttr) {
  StringRef roundingKeyword;
  if (parser.parseKeyword("round") || parser.parseLParen() ||
      parser.parseKeyword(&roundingKeyword) || parser.parseRParen()) {
    return failure();
  }
  std::optional<Rounding> rounding = symbolizeRounding(roundingKeyword);
  if (!rounding) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert rounding to be one of "
           << kConvertRoundingKeywords;
  }
  roundingAttr = RoundingAttr::get(parser.getContext(), *rounding);
  return success();
}

static void printConvertRounding(OpAsmPrinter &printer, Operation *op,
                                 RoundingAttr rounding) {
  printer << "round(" << stringifyRounding(rounding.getValue()) << ")";
}

static ParseResult parseConvertSaturation(OpAsmParser &parser,
                                          SaturationAttr &saturationAttr) {
  StringRef saturationKeyword;
  if (parser.parseKeyword(&saturationKeyword)) {
    return failure();
  }
  std::optional<Saturation> saturation =
      symbolizeSaturation(saturationKeyword);
  if (!saturation) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert saturation to be sat or nosat";
  }
  saturationAttr = SaturationAttr::get(parser.getContext(), *saturation);
  return success();
}

static void printConvertSaturation(OpAsmPrinter &printer, Operation *op,
                                   SaturationAttr saturation) {
  printer << stringifySaturation(saturation.getValue());
}

static ParseResult parseSignedness(OpAsmParser &parser,
                                   SignednessAttr &signedness) {
  StringRef signednessKeyword;
  if (parser.parseKeyword(&signednessKeyword)) {
    return failure();
  }
  std::optional<Signedness> parsed = symbolizeSignedness(signednessKeyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected signedness to be signed or unsigned";
  }
  signedness = SignednessAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printSignedness(OpAsmPrinter &printer, Operation *op,
                            SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static OptionalParseResult parseOptionalSignedness(OpAsmParser &parser,
                                                   SignednessAttr &signedness) {
  if (succeeded(parser.parseOptionalKeyword("signed"))) {
    signedness = SignednessAttr::get(parser.getContext(), Signedness::Signed);
    return success();
  }
  if (succeeded(parser.parseOptionalKeyword("unsigned"))) {
    signedness =
        SignednessAttr::get(parser.getContext(), Signedness::Unsigned);
    return success();
  }
  return std::nullopt;
}

static void printOptionalSignedness(OpAsmPrinter &printer, Operation *op,
                                    SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static constexpr const char kLdL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmpref/nakeep/naclean/nadrop/idsfv/idslv/idsprs/"
    "idspref/exfv/exlv/exprs/expref";

static constexpr const char kStL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmred/naci/napw/napi/nared/wbhfv/wbhlv/wbhprs/"
    "wbhred/wtsfv/wtslv/wtsprs/wtsred";

static ParseResult parseL1Cache(OpAsmParser &parser, L1CacheAttr &l1cache) {
  if (failed(parser.parseOptionalKeyword("l1cache"))) {
    l1cache = L1CacheAttr::get(parser.getContext(), L1Cache::Cache);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<L1Cache> parsed = symbolizeL1Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected memory l1cache to be cache or uncache";
  }
  l1cache = L1CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printL1Cache(OpAsmPrinter &printer, Operation *op,
                         L1CacheAttr l1cache) {
  if (!l1cache) {
    return;
  }
  printer << "l1cache(" << stringifyL1Cache(l1cache.getValue()) << ")";
}

static ParseResult parseLdL2Cache(OpAsmParser &parser,
                                  LdL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = LdL2CacheAttr::get(parser.getContext(), LdL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<LdL2Cache> parsed = symbolizeLdL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected load L2 cache control to be one of "
           << kLdL2CacheKeywords;
  }
  l2cache = LdL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}
