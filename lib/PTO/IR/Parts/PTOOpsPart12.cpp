// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op.getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName))) {
    return failure();
  }

  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op.emitOpError("must be nested under a func.func");
  }

  if (op.getId() < 0) {
    return op.emitOpError("expects 'id' to be non-negative");
  }

  unsigned sameIdInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == op.getId()) {
        ++sameIdInitCount;
      }
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == op.getId()) {
        ++sameIdInitCount;
      }
    }
  });
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != 2 && dirMask != 3) {
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (op.getSlotSize() <= 0) {
    return op.emitOpError("expects 'slot_size' to be greater than 0");
  }
  int32_t slotNum = dirMask == 3 ? 4 : 8;
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    slotNum = slotNumAttr.getInt();
    if (slotNum <= 0) {
      return op.emitOpError("expects 'slot_num' to be greater than 0");
    }
  }
  PTOArch arch = getTargetArch(op.getOperation());

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasGmSlotBuffer = static_cast<bool>(op.getGmSlotBuffer());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    if (hasGmSlotBuffer) {
      return op.emitOpError(
          "'gm_slot_tensor' cannot be combined with 'gm_slot_buffer'");
    }
    if (hasC2vConsumerBuf || hasV2cConsumerBuf) {
      bool supportsC2V =
          dirMask == 1 && hasC2vConsumerBuf && !hasV2cConsumerBuf;
      bool supportsA2A3V2C =
          arch != PTOArch::A5 &&
          ((dirMask == 2 && !hasC2vConsumerBuf && hasV2cConsumerBuf) ||
           (dirMask == 3 && hasC2vConsumerBuf && hasV2cConsumerBuf));
      if (!supportsC2V && !supportsA2A3V2C) {
        return op.emitOpError(
            "GM-backed tile pipe init supports dir_mask = 1 with "
            "'c2v_consumer_buf' on all targets and dir_mask = 2/3 with "
            "matching consumer buffers only on a2/a3");
      }
    }
    if (!hasC2vConsumerBuf && !hasV2cConsumerBuf) {
      if (op.getLocalSlotNumAttr()) {
        return op.emitOpError(
            "globaltensor pipe init does not use 'local_slot_num'");
      }
      return verifyFrontendGlobalSlotTensor(
          op.getOperation(), op.getGmSlotTensor(), dirMask, op.getSlotSize());
    }
    if (failed(verifyFrontendGlobalSlotTensor(
            op.getOperation(), op.getGmSlotTensor(), dirMask,
            op.getSlotSize()))) {
      return failure();
    }
  }

  if (!hasC2vConsumerBuf && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide at least one consumer buffer "
        "operand; use 'gm_slot_tensor' for globaltensor pipe entries");
  }
  if (dirMask == 1 && !hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' when dir_mask is 1");
  }
  if (dirMask == 2 && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'v2c_consumer_buf' when dir_mask is 2");
  }
  if (dirMask == 3 && (!hasC2vConsumerBuf || !hasV2cConsumerBuf)) {
    return op.emitOpError(
        "expects both 'c2v_consumer_buf' and 'v2c_consumer_buf' when dir_mask is 3");
  }

  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    if (arch == PTOArch::A5) {
      return op.emitOpError(
          "'local_slot_num' is only supported for a2/a3 frontend pipe lowering");
    }
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0) {
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    }
    if (localSlotNum > slotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to slot_num ("
             << slotNum << ") for dir_mask = " << static_cast<int>(dirMask);
    }
  }

  // Fixpipe validation
  if (auto accPushEpilogue = op.getAccPushEpilogueAttr()) {
    // Rule 1: fixpipe must be C2V only (dir_mask == 1)
    if (dirMask != 1) {
      return op.emitOpError(
          "expects fixpipe pipe (with 'acc_push_epilogue') to have dir_mask = 1 (C2V only)");
    }

    // Rule 2: fixpipe must have nosplit = true
    if (!op.getNosplit()) {
      return op.emitOpError(
          "expects fixpipe pipe (with 'acc_push_epilogue') to have nosplit = true");
    }

    // Rule 5: C2V consumer buf must be traceable to a peerable buffer op.
    // This init-common check only enforces a local def-use precondition. The
    // full peer-contract match (buffer name + peer_func + opposite-side init)
    // is verified later in AicInitializePipeOp::verify() /
    // AivInitializePipeOp::verify().
    if (hasC2vConsumerBuf) {
      Value c2vBuf = op.getC2vConsumerBuf();
      Operation *defOp = c2vBuf.getDefiningOp();
      bool foundReserve = false;
      if (defOp) {
        if (isa<ReserveBufferOp>(defOp) || isa<ImportReservedBufferOp>(defOp)) {
          foundReserve = true;
        }
      }
      if (!foundReserve) {
        return op.emitOpError(
            "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
            "import_reserved_buffer for peer contract verification");
      }
    }

    // Rule 7: relu must be no_relu or normal_relu in v1
    auto relu = accPushEpilogue.getRelu();
    if (relu != pto::FixpipeRelu::NoRelu && relu != pto::FixpipeRelu::NormalRelu) {
      return op.emitOpError(
          "expects 'acc_push_epilogue.relu' to be 'no_relu' or 'normal_relu' in v1");
    }

    // Rule 8: quant must be in v1 allowed set
    auto quant = accPushEpilogue.getQuant();
    bool validQuant = false;
    switch (quant) {
      case pto::FixpipeQuant::NoConvert:
      case pto::FixpipeQuant::F32F16:
      case pto::FixpipeQuant::F32BF16:
      case pto::FixpipeQuant::REQ8Scalar:
      case pto::FixpipeQuant::REQ8Vec:
      case pto::FixpipeQuant::DEQF16Scalar:
      case pto::FixpipeQuant::DEQF16Vec:
      case pto::FixpipeQuant::QF322B8PreScalar:
      case pto::FixpipeQuant::QF322B8PreVec:
      case pto::FixpipeQuant::QF322F16PreScalar:
      case pto::FixpipeQuant::QF322BF16PreScalar:
      case pto::FixpipeQuant::QS322BF16PreScalar:
      case pto::FixpipeQuant::QS322BF16PreVec:
      case pto::FixpipeQuant::QF322HIF8PreScalar:
      case pto::FixpipeQuant::QF322FP8PreScalar:
        validQuant = true;
        break;
    }
    if (!validQuant) {
      return op.emitOpError(
          "expects 'acc_push_epilogue.quant' to be one of the v1 allowed quantization modes");
    }

    // Rule 10: Check A5-only modes
    if (quant == pto::FixpipeQuant::QS322BF16PreScalar ||
        quant == pto::FixpipeQuant::QS322BF16PreVec) {
      if (arch != PTOArch::A5) {
        return op.emitOpError(
            "expects 'qs322bf16_pre_*' quantization modes to be used only on A5 target");
      }
    }
    if (quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
        quant == pto::FixpipeQuant::QF322FP8PreScalar) {
      if (arch != PTOArch::A5) {
        return op.emitOpError(
            "expects 'qf322hif8_pre_scalar'/'qf322fp8_pre_scalar' to be used only on A5 target");
      }
    }

    // Rule 14-15: fixpipe slot_size is interpreted as post-fixpipe
    // consumer-visible entry bytes. This init-common check relies on the
    // later fixpipe peer verifier to resolve consumer tpop tile types and
    // enforce the concrete lower bound against that resolved entry shape.
  }

  return success();
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ReserveBufferOp mlir::pto::findReserveBufferByName(func::FuncOp funcOp,
                                                   StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name) {
      return WalkResult::advance();
    }
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  if (getSize() <= 0) {
    return emitOpError("expects 'size' to be greater than 0");
  }

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT) {
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");
  }

  if (!getAutoAlloc() && !getBaseAttr()) {
    return emitOpError("expects 'base' when 'auto' is false");
  }

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0) {
    return emitOpError("expects 'base' to be non-negative when present");
  }

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName()) {
      ++sameNameCount;
    }
  });
  if (sameNameCount > 1) {
    return emitOpError("requires 'name' to be unique within the function");
  }

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto peerFunc = lookupPeerFuncAcrossContainer(getOperation(), getPeerFuncAttr());
  if (!peerFunc) {
    return emitOpError("expects 'peer_func' to reference an existing func.func");
  }

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName())) {
    return emitOpError("expects matching peer reserve_buffer to exist");
  }

  return success();
}

constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";
constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
constexpr llvm::StringLiteral kPipePeerDirMaskAttrName = "__pto.peer_dir_mask";

struct FixpipeQuantStateResource
    : public SideEffects::Resource::Base<FixpipeQuantStateResource> {
  StringRef getName() final { return "PTOFixpipeQuantState"; }
};

static IntegerAttr getFixpipeQuantStateIdAttr(Operation *op, int32_t id) {
  return IntegerAttr::get(IntegerType::get(op->getContext(), 32), id);
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  if (!pipeHandle) {
    return std::nullopt;
  }
  Operation *defOp = pipeHandle.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto frontendIdAttr = defOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
  if (!frontendIdAttr) {
    return std::nullopt;
  }
  return static_cast<int32_t>(frontendIdAttr.getInt());
}

static pto::AccPushEpilogueAttr getAccPushEpilogueFromInitOp(Operation *initOp) {
  if (!initOp) {
    return {};
  }
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aicInit.getAccPushEpilogueAttr();
  }
  if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
    return aivInit.getAccPushEpilogueAttr();
  }
  if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    return l2lInit.getAccPushEpilogueAttr();
  }
  if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    return l2g2lInit.getAccPushEpilogueAttr();
  }
  return {};
}

static bool matchesLoweredFixpipePeerContract(Operation *initOp,
                                              func::FuncOp expectedOwnerFunc,
                                              StringRef expectedReserveName) {
  if (!initOp || !isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(initOp)) {
    return false;
  }

  auto ownerAttr =
      initOp->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveAttr =
      initOp->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      initOp->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerAttr || !reserveAttr || !dirMaskAttr) {
    return false;
  }

  if (ownerAttr.getValue() != expectedOwnerFunc.getSymName() ||
      reserveAttr.getValue() != expectedReserveName ||
      dirMaskAttr.getInt() != 1) {
    return false;
  }

  return static_cast<bool>(getAccPushEpilogueFromInitOp(initOp));
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id);

static FailureOr<Operation *>
lookupFixpipePeerConsumerInit(AicInitializePipeOp producerInit) {
  if (!producerInit.getC2vConsumerBuf()) {
    return failure();
  }
  auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
      producerInit.getC2vConsumerBuf().getDefiningOp());
  if (!importOp) {
    return failure();
  }

  auto peerConsumerFunc =
      lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                    importOp.getPeerFuncAttr());
  if (!peerConsumerFunc) {
    return failure();
  }

  StringRef bufferName = importOp.getName();
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerConsumerFunc.walk([&](Operation *candidate) {
    if (auto aivInit = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (!aivInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }
      auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(
          aivInit.getC2vConsumerBuf().getDefiningOp());
      if (!reserveOp || reserveOp.getName() != bufferName) {
        return WalkResult::advance();
      }
      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, peerConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    return failure();
  }
  return matchedInit;
}

static FailureOr<Operation *>
lookupFixpipePeerProducerInit(AivInitializePipeOp consumerInit,
                              func::FuncOp peerProducerFunc,
                              StringRef bufferName,
                              func::FuncOp currentConsumerFunc) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerProducerFunc.walk([&](Operation *candidate) {
    if (auto aicInit = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (!aicInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }

      auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
          aicInit.getC2vConsumerBuf().getDefiningOp());
      if (!importOp) {
        return WalkResult::advance();
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                        importOp.getPeerFuncAttr());
      if (importOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return WalkResult::advance();
      }

      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, currentConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    consumerInit.emitOpError()
        << "expects peer producer function to contain a matching "
           "aic_initialize_pipe with the same consumer buffer contract";
    return failure();
  }

  return matchedInit;
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id) {
  Operation *matchedFrontendInit = nullptr;
  unsigned matchedFrontendInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedFrontendInit = candidate;
        ++matchedFrontendInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedFrontendInit = candidate;
        ++matchedFrontendInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedFrontendInitCount == 1) {
    return matchedFrontendInit;
  }
  if (matchedFrontendInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }

  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate)) {
      return WalkResult::advance();
    }
    auto frontendIdAttr =
        candidate->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
    if (!frontendIdAttr || frontendIdAttr.getInt() != id) {
      return WalkResult::advance();
    }
    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split,
                                           bool expectC2V) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName))) {
    return failure();
  }
  if (id < 0) {
    return op->emitOpError("expects 'id' to be non-negative");
  }
  if (failed(verifySplitAttr(op, split))) {
    return failure();
  }
  if (!isOddSplit(split)) {
    return success();
  }

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  if (!expectC2V && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }

  auto supportsOddDirection = [op, expectC2V](auto init) {
    PTOArch arch = getTargetArch(op);
    bool hasSupportedGmBacking =
        static_cast<bool>(init.getGmSlotTensor()) ||
        (arch != PTOArch::A5 && static_cast<bool>(init.getGmSlotBuffer()));
    int8_t directionMask = expectC2V ? 1 : 2;
    Value consumerBuffer = expectC2V ? init.getC2vConsumerBuf()
                                     : init.getV2cConsumerBuf();
    return (init.getDirMask() & directionMask) != 0 &&
           hasSupportedGmBacking && static_cast<bool>(consumerBuffer);
  };

  bool supported = false;
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    supported = supportsOddDirection(aic);
  } else {
    supported = supportsOddDirection(cast<AivInitializePipeOp>(*initOr));
  }
  if (!supported) {
    return op->emitOpError()
           << "supports odd split modes (split = 3 or 4) only for a "
              "GM-backed tile pipe whose dir_mask enables "
           << (expectC2V ? "C2V" : "V2C")
           << " and provides the matching consumer buffer";
  }
  return success();
}

static LogicalResult verifyOddSplitTileEntry(Operation *op, int64_t split,
                                             Type entryTy) {
  if (isOddSplit(split) && isa<TensorViewType>(entryTy)) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for tile entries; "
        "the pinned pto-isa does not implement odd GlobalTensor offsets");
  }
  return success();
}

static LogicalResult verifyFullTileSplitParity(Operation *op, int64_t split,
                                               Type entryTy) {
  if (split == 0) {
    return success();
  }

  ArrayRef<int64_t> shape;
  if (auto tileTy = dyn_cast<TileBufType>(entryTy)) {
    shape = tileTy.getValidShape();
  } else if (auto viewTy = dyn_cast<TensorViewType>(entryTy)) {
    shape = viewTy.getShape();
  } else {
    return success();
}
  if (shape.size() != 2) {
    return success();
  }

  bool splitRows = split == 1 || split == 3;
  int64_t axisSize = shape[splitRows ? 0 : 1];
  if (axisSize == ShapedType::kDynamic) {
    return success();
  }

  bool expectOdd = isOddSplit(split);
  if ((axisSize % 2 != 0) != expectOdd) {
    return op->emitOpError()
           << "expects a statically " << (expectOdd ? "odd" : "even")
           << " valid-" << (splitRows ? "row" : "column")
           << " count for split = " << split;
  }
  return success();
}

static LogicalResult verifyAivSubblockIdOperand(Operation *op,
                                                Value aivSubblockId,
                                                int64_t split,
                                                Type pipeEntryType) {
  if (!aivSubblockId) {
    return success();
  }

  if (split == 0) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only when 'split' is 1, 2, 3, or 4");
  }

  if (isa<TensorViewType>(pipeEntryType)) {
    return op->emitOpError(
        "does not support 'aiv_subblockid' for !pto.tensor_view pipe entries");
  }

  auto addrSpace = getPTOMemorySpaceEnum(pipeEntryType);
  if (!addrSpace || *addrSpace != AddressSpace::VEC) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only on AIV-side vector tile pipe entries");
  }

  return success();
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    return aic.getDirMask();
  }
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr)) {
    return failure();
  }

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != 1 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != 2 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aic.getGmSlotTensor();
  }
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }

  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }
  if (slotTensorTy.getElementType() != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match gm_slot_tensor element type";
  }
  if (slotTensorTy.getRank() != entryViewTy.getRank()) {
    return op->emitOpError()
           << "expects pipe entry rank to match gm_slot_tensor rank";
  }

  ArrayRef<int64_t> slotShape = slotTensorTy.getShape();
  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit(), expectC2V))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(op.getOperation(), op.getSplit(),
                                     op.getTile().getType()))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType()))) {
    return failure();
  }
  if (!expectC2V &&
      failed(verifyFullTileSplitParity(op.getOperation(), op.getSplit(),
                                       op.getTile().getType()))) {
    return failure();
  }

  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  }
  if (expectC2V && isOddSplit(op.getSplit()) && !hasValidRow &&
      !isa<TensorViewType>(op.getTile().getType())) {
    return op.emitOpError(
        "expects odd C2V split tpop to provide per-sub-core valid_row and "
        "valid_col operands");
  }
  if (!hasValidRow) {
    return success();
  }

  if (isa<TensorViewType>(op.getTile().getType())) {
    return op.emitOpError(
        "does not accept valid_row/valid_col when result is !pto.tensor_view");
  }

  auto tileTy = dyn_cast<TileBufType>(op.getTile().getType());
  if (!tileTy) {
    return op.emitOpError(
        "expects tile result to be !pto.tile_buf when valid_row/valid_col operands are provided");
  }
  if (!tileTy.hasDynamicValid()) {
    return op.emitOpError(
        "expects tile result to have dynamic validShape (?, ?) when valid_row/valid_col operands are provided");
  }
  return success();
}

static bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

static bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

static bool matchesFixpipeConsumerLayout(FixpipeLayout layout,
                                         TileBufType tileTy) {
  auto memorySpace = dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!memorySpace || memorySpace.getAddressSpace() != AddressSpace::VEC) {
    return false;
  }

  int32_t bLayout = tileTy.getBLayoutValueI32();
  int32_t sLayout = tileTy.getSLayoutValueI32();
  switch (layout) {
  case FixpipeLayout::NZ2ND:
    return bLayout == static_cast<int32_t>(BLayout::RowMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2DN:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2NZ:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::RowMajor);
  }
  llvm_unreachable("unhandled FixpipeLayout");
}

static bool isSignedOrUnsignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.getWidth() == 8 && (intTy.isSigned() || intTy.isUnsigned());
  }
  return false;
}

static bool isSignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.isSignedInteger(8);
  }
  return false;
}

static bool matchesFixpipeConsumerElementType(FixpipeQuant quant,
                                              Type resultElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return resultElemType.isF32() || resultElemType.isInteger(32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QF322F16PreScalar:
    return resultElemType.isF16();
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return resultElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
    return isSignedOrUnsignedI8(resultElemType);
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
    return isSignedI8(resultElemType);
  case FixpipeQuant::QF322HIF8PreScalar:
    return isa<HiF8Type>(resultElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return isPTOFloat8E4M3LikeType(resultElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isFixpipeQuantPayloadElemType(Type elemTy, PTOArch arch) {
  if (!elemTy) {
    return false;
  }
  if (arch == PTOArch::A3) {
    return elemTy.isUnsignedInteger(64) || elemTy.isSignlessInteger(64) ||
           elemTy.isSignedInteger(64);
  }
  return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32();
}

static bool matchesFixpipeProducerAndConsumerTypes(FixpipeQuant quant,
                                                   Type srcElemType,
                                                   Type dstElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return (srcElemType.isF32() || srcElemType.isInteger(32)) &&
           srcElemType == dstElemType;
  case FixpipeQuant::F32F16:
    return srcElemType.isF32() && dstElemType.isF16();
  case FixpipeQuant::F32BF16:
    return srcElemType.isF32() && dstElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
    return srcElemType.isInteger(32) && isSignedOrUnsignedI8(dstElemType);
  case FixpipeQuant::REQ8Vec:
    return srcElemType.isInteger(32) && isSignedI8(dstElemType);
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
    return srcElemType.isInteger(32) && dstElemType.isF16();
  case FixpipeQuant::QF322B8PreScalar:
    return srcElemType.isF32() && isSignedOrUnsignedI8(dstElemType);
  case FixpipeQuant::QF322B8PreVec:
    return srcElemType.isF32() && isSignedI8(dstElemType);
  case FixpipeQuant::QF322F16PreScalar:
    return srcElemType.isF32() && dstElemType.isF16();
  case FixpipeQuant::QF322BF16PreScalar:
    return srcElemType.isF32() && dstElemType.isBF16();
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return srcElemType.isInteger(32) && dstElemType.isBF16();
  case FixpipeQuant::QF322HIF8PreScalar:
    return srcElemType.isF32() && isa<HiF8Type>(dstElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return srcElemType.isF32() && isPTOFloat8E4M3LikeType(dstElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isUnpublishedFixpipeFrontendAttrName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("stPhase", true)
      .Case("st_phase", true)
      .Case("atomicType", true)
      .Case("atomic_type", true)
      .Case("subBlockId", true)
      .Case("subBlockid", true)
      .Case("sub_blockid", true)
      .Case("clipReluMode", true)
      .Case("clip_relu_mode", true)
      .Case("isChannelSplit", true)
      .Case("is_channel_split", true)
      .Case("channelSplit", true)
      .Case("channel_split", true)
      .Default(false);
}

static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (!isUnpublishedFixpipeFrontendAttrName(name)) {
      continue;
    }
    return op->emitOpError()
           << "does not allow unpublished fixpipe attr '" << name
           << "'; STPhase / AtomicType / SubBlockId / ClipReluMode / "
              "IsChannelSplit are not part of the PTOIR frontend surface";
  }
  return success();
}

static std::optional<uint64_t> getStaticTileByteSize(TileBufType tileTy) {
  auto shape = getShapeVec(tileTy);
  auto elemCount = getStaticElementCount(shape);
  uint64_t elemBytes = getElemByteSize(tileTy.getElementType());
  if (!elemCount || elemBytes == 0) {
    return std::nullopt;
  }
  return *elemCount * elemBytes;
}

// Helper to verify fixpipe consumer tpop result type consistency
static LogicalResult verifyFixpipeConsumerType(Operation *tpopOp, int32_t id,
                                               Type resultTileType) {
  auto funcOp = tpopOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return success(); // Already checked elsewhere
  }

  // Look up the consumer-side init
  auto initOr = lookupFrontendInitOpById(tpopOp, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aivInit = dyn_cast<AivInitializePipeOp>(initOp);
  if (!aivInit) {
    return success(); // Not consumer init, skip
  }

  auto accPushEpilogue = aivInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // Not a fixpipe, skip
  }

  if (auto tpop = dyn_cast<TPopFromAicOp>(tpopOp); tpop && tpop.getSplit() != 0) {
    return tpop.emitOpError("expects fixpipe TPOP to have split = 0");
  }

  // Rule 11: At least one tpop must exist (checked by counting all tpops for this pipe)
  // Rule 12: Verify result element type matches expected type from quant mode
  auto tileTy = dyn_cast<pto::TileBufType>(resultTileType);
  if (!tileTy) {
    return tpopOp->emitOpError(
        "expects fixpipe TPOP result to be a tile type");
  }

  Type resultElemType = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();
  bool mismatchedPeerType = false;
  funcOp.walk([&](TPopFromAicOp otherPop) {
    if (otherPop.getOperation() == tpopOp ||
        otherPop.getId() != static_cast<uint32_t>(id)) {
      return WalkResult::advance();
    }
    if (otherPop.getTile().getType() != resultTileType) {
      mismatchedPeerType = true;
      tpopOp->emitOpError()
          << "expects all tpop_from_aic results for fixpipe pipe id = " << id
          << " to use the same tile type";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (mismatchedPeerType) {
    return failure();
  }

  if (!matchesFixpipeConsumerElementType(quant, resultElemType)) {
    return tpopOp->emitOpError()
           << "expects consumer element type to match acc_push_epilogue.quant "
           << stringifyFixpipeQuant(quant);
  }

  if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(), tileTy)) {
    return tpopOp->emitOpError()
           << "expects consumer tile layout to match acc_push_epilogue.layout "
           << stringifyFixpipeLayout(accPushEpilogue.getLayout());
  }

  return success();
}


static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != 2 && dirMask != 3) {
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (slotSize <= 0) {
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  }
  if (slotNum <= 0) {
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  }
  if (flagBase && *flagBase < 0) {
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  }
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType())) {
    return op->emitOpError("expects pipe operand type !pto.pipe");
  }
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

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
