// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (!getTmpMutable().empty()) {
    PTO_ADD_WRITE(getTmpMutable()[0]);
  }
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxOp ===
// Read: a, a_scale, b, b_scale, Write: dst
void TGemvMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxAccOp ===
// Read: c_in, a, a_scale, b, b_scale, Write: dst
void TGemvMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxBiasOp ===
// Read: a, a_scale, b, b_scale, bias, Write: dst
void TGemvMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

static bool isInsideSectionCube(Operation *op) {
  return op->getParentOfType<pto::SectionCubeOp>() != nullptr;
}

static bool isInsideSectionVector(Operation *op) {
  return op->getParentOfType<pto::SectionVectorOp>() != nullptr;
}

static bool isInsideTileOpHelper(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return funcOp && funcOp->hasAttr("pto.tileop.helper");
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return std::nullopt;
  }

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr) {
    return std::nullopt;
  }

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         isInsideTileOpHelper(op) || getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
  if (split < 0 || split > 4) {
    return op->emitOpError("expects 'split' to be 0, 1, 2, 3, or 4");
  }
  return success();
}

static bool isOddSplit(int64_t split) {
  return split == 3 || split == 4;
}

static bool isInsideCubeKernelOrSection(Operation *op) {
  if (isInsideSectionCube(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Cube;
}

static bool isInsideVectorKernelOrSection(Operation *op) {
  if (isInsideSectionVector(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Vector;
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (isInsideTileOpHelper(op)) {
    return success();
  }
  if (isInsideSectionCube(op)) {
    if (expected == FunctionKernelKind::Cube) {
      return success();
    }
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  if (isInsideSectionVector(op)) {
    if (expected == FunctionKernelKind::Vector) {
      return success();
    }
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }

  std::optional<FunctionKernelKind> kernelKind =
      getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawSlotNum = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;
  bool sawAccPushEpilogue = false;

  if (parser.parseLBrace()) {
    return failure();
  }

  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) {
      return failure();
    }

    if (keyword == "id") {
      if (sawId) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'id' clause");
      }
      IntegerAttr idAttr;
      if (parser.parseAttribute(idAttr, parser.getBuilder().getI32Type(), "id",
                                attrs)) {
        return failure();
      }
      sawId = true;
    } else if (keyword == "dir_mask") {
      if (sawDirMask) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'dir_mask' clause");
      }
      IntegerAttr dirMaskAttr;
      if (parser.parseAttribute(dirMaskAttr, parser.getBuilder().getI8Type(),
                                "dir_mask", attrs)) {
        return failure();
      }
      sawDirMask = true;
    } else if (keyword == "slot_size") {
      if (sawSlotSize) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_size' clause");
      }
      IntegerAttr slotSizeAttr;
      if (parser.parseAttribute(slotSizeAttr, parser.getBuilder().getI32Type(),
                                "slot_size", attrs)) {
        return failure();
      }
      sawSlotSize = true;
    } else if (keyword == "slot_num") {
      if (sawSlotNum) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_num' clause");
      }
      IntegerAttr slotNumAttr;
      if (parser.parseAttribute(slotNumAttr, parser.getBuilder().getI32Type(),
                                "slot_num", attrs)) {
        return failure();
      }
      sawSlotNum = true;
    } else if (keyword == "local_slot_num") {
      if (sawLocalSlotNum) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'local_slot_num' clause");
      }
      IntegerAttr localSlotNumAttr;
      if (parser.parseAttribute(localSlotNumAttr, parser.getBuilder().getI32Type(),
                                "local_slot_num", attrs)) {
        return failure();
      }
      sawLocalSlotNum = true;
    } else if (keyword == "nosplit") {
      if (sawNoSplit) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'nosplit' clause");
      }
      BoolAttr noSplitAttr;
      if (parser.parseAttribute(noSplitAttr, "nosplit", attrs)) {
        return failure();
      }
      sawNoSplit = true;
    } else if (keyword == "acc_push_epilogue") {
      if (sawAccPushEpilogue) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'acc_push_epilogue' clause");
      }
      AccPushEpilogueAttr accPushEpilogueAttr;
      if (parser.parseAttribute(accPushEpilogueAttr, "acc_push_epilogue",
                                attrs)) {
        return failure();
      }
      sawAccPushEpilogue = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected keyword '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRBrace())) {
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
  }

  if (!sawDirMask) {
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  }
  if (!sawSlotSize) {
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  }
  if (!sawId) {
    attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));
  }

  OpAsmParser::UnresolvedOperand gmSlotBuffer;
  OpAsmParser::UnresolvedOperand gmSlotTensor;
  OpAsmParser::UnresolvedOperand c2vConsumerBuf;
  OpAsmParser::UnresolvedOperand v2cConsumerBuf;
  Type gmSlotBufferTy;
  Type gmSlotTensorTy;
  Type c2vConsumerBufTy;
  Type v2cConsumerBufTy;
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;

  if (parser.parseLParen()) {
    return failure();
  }
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) {
      return failure();
    }

    if (keyword == "gm_slot_buffer") {
      if (hasGmSlotBuffer) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_buffer' operand");
      }
      if (parser.parseOperand(gmSlotBuffer) ||
          parser.parseColonType(gmSlotBufferTy)) {
        return failure();
      }
      hasGmSlotBuffer = true;
    } else if (keyword == "gm_slot_tensor") {
      if (hasGmSlotTensor) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_tensor' operand");
      }
      if (parser.parseOperand(gmSlotTensor) ||
          parser.parseColonType(gmSlotTensorTy)) {
        return failure();
      }
      hasGmSlotTensor = true;
    } else if (keyword == "c2v_consumer_buf") {
      if (hasC2vConsumerBuf) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'c2v_consumer_buf' operand");
      }
      if (parser.parseOperand(c2vConsumerBuf) ||
          parser.parseColonType(c2vConsumerBufTy)) {
        return failure();
      }
      hasC2vConsumerBuf = true;
    } else if (keyword == "v2c_consumer_buf") {
      if (hasV2cConsumerBuf) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'v2c_consumer_buf' operand");
      }
      if (parser.parseOperand(v2cConsumerBuf) ||
          parser.parseColonType(v2cConsumerBufTy)) {
        return failure();
      }
      hasV2cConsumerBuf = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected initialize_pipe operand '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRParen())) {
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  result.addAttributes(attrs);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {hasGmSlotBuffer ? 1 : 0, hasGmSlotTensor ? 1 : 0,
                           hasC2vConsumerBuf ? 1 : 0,
                           hasV2cConsumerBuf ? 1 : 0}));
  if (hasGmSlotBuffer &&
      parser.resolveOperand(gmSlotBuffer, gmSlotBufferTy, result.operands)) {
    return failure();
  }
  if (hasGmSlotTensor &&
      parser.resolveOperand(gmSlotTensor, gmSlotTensorTy, result.operands)) {
    return failure();
  }
  if (hasC2vConsumerBuf &&
      parser.resolveOperand(c2vConsumerBuf, c2vConsumerBufTy, result.operands)) {
    return failure();
  }
  if (hasV2cConsumerBuf &&
      parser.resolveOperand(v2cConsumerBuf, v2cConsumerBufTy, result.operands)) {
    return failure();
  }
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma) {
      p << ", ";
    }
    p << keyword << " = " << value;
    needsComma = true;
  };

  printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    printClause("slot_num", slotNumAttr.getInt());
  }
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    printClause("local_slot_num", localSlotNumAttr.getInt());
  }
  if (auto noSplitAttr = op.getNosplitAttr()) {
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  }
  if (auto accPushEpilogueAttr = op.getAccPushEpilogueAttr()) {
    printClause("acc_push_epilogue", accPushEpilogueAttr);
  }
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma) {
      p << ", ";
    }
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor()) {
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  }
  if (op.getC2vConsumerBuf()) {
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  }
  if (op.getV2cConsumerBuf()) {
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  }
  p << ")";
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "slot_num",
                       "local_slot_num", "acc_push_epilogue",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
  return tensorBytes == slotBytes || tensorBytes * 2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty()) {
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");
  }

  if (auto elemCount = getStaticElementCount(shape)) {
    uint64_t elemBytes = getElemByteSize(tvTy.getElementType());
    if (elemBytes != 0) {
      uint64_t tensorBytes = *elemCount * elemBytes;
      if (!isSameOrHalfSlotByteSize(tensorBytes,
                                    static_cast<uint64_t>(slotSize))) {
        return op->emitOpError()
               << "expects 'slot_size' to equal gm_slot_tensor byte size "
                  "or twice gm_slot_tensor byte size for split GlobalTensor "
                  "entries (got slot_size = "
               << slotSize << ", gm_slot_tensor byte size = " << tensorBytes
               << ")";
      }
    }
  }

  return success();
}
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
