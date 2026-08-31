// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic) {
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2) {
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
    }
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    if (IntegerAttr eventIdAttr = getEventIdAttr()) {
      int64_t eventId = eventIdAttr.getInt();
      if (eventId < 0 || eventId > 15) {
        return emitOpError()
               << "A5 sync.wait expects static FFTS event_id in [0, 15], but got "
               << eventId;
      }
    }
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyNamedSyncEventOp(Operation *op, PipeAttr pipe,
                                            IntegerAttr eventIdAttr,
                                            Value eventIdDyn,
                                            int64_t maxEventId,
                                            StringRef opName) {
  const bool hasStaticEventId = eventIdAttr != nullptr;
  const bool hasDynamicEventId = static_cast<bool>(eventIdDyn);
  if (hasStaticEventId == hasDynamicEventId) {
    return op->emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  const bool staticEventIdOutOfRange =
      hasStaticEventId &&
      (eventIdAttr.getInt() < 0 || eventIdAttr.getInt() > maxEventId);
  if (staticEventIdOutOfRange) {
    return op->emitOpError() << "expects static event_id in [0, " << maxEventId
                             << "], but got " << eventIdAttr.getInt();
  }
  switch (pipe.getPipe()) {
  case PIPE::PIPE_FIX:
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_V:
    return success();
  default:
    return op->emitOpError() << opName << " expects pipe to be one of "
                              << "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              << "<PIPE_MTE3>, <PIPE_V>";
  }
}

ParseResult mlir::pto::SetCrossBlockOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetCrossBlockOp::getPipeAttrName(result.name),
                                SetCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetCrossBlockOp::verify() {
  if (IntegerAttr mode = getFftsModeAttr()) {
    int64_t modeValue = mode.getInt();
    if (modeValue != 0) {
      return emitOpError() << "requires ffts_mode 0, but got "
                           << modeValue;
    }
  }
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.set_cross_block");
}

ParseResult mlir::pto::WaitCrossBlockOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitCrossBlockOp::getPipeAttrName(result.name),
                                WaitCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitCrossBlockOp::verify() {
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.wait_cross_block");
}

ParseResult mlir::pto::SetIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetIntraBlockOp::getPipeAttrName(result.name),
                                SetIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.set_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.set_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::WaitIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitIntraBlockOp::getPipeAttrName(result.name),
                                WaitIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.wait_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.wait_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  const bool hasFp = static_cast<bool>(getFp());
  const bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  if (hasFp && hasPreQuant) {
    return emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && getStPhase() != pto::STPhase::Unspecified) {
    return emitOpError("expects fp form to use the default stPhase");
  }

  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::TileBufType, pto::PartitionTensorViewType>> {
    auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
    auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
    if (!srcTile || !dstPart) {
      emitOpError("expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, srcTile, "src", allowLowPrecision))) {
      return failure();
    }
    if (hasFp) {
      Type fpTy = getFp().getType();
      if (failed(verifyTileBufCommon(*this, fpTy, "fp", allowLowPrecision))) {
        return failure();
      }
      auto fpSpace = getPTOMemorySpaceEnum(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
        emitOpError("expects fp to use loc=scaling");
        return failure();
      }
    }
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects dst shape[" << idx << "] to be positive";
        return failure();
      }
    }
    auto srcValid = srcTile.getValidShape();
    for (auto [idx, dim] : llvm::enumerate(srcValid)) {
      if (dim != ShapedType::kDynamic && dim < 0) {
        emitOpError() << "expects src valid_shape[" << idx << "] to be non-negative";
        return failure();
      }
    }

    // Keep TSTORE contract explicit while preserving existing legal layout
    // reinterpretation paths (e.g. 1x1024 <-> 32x32, 5D partition views).
    // When both sides are fully static, require equal element counts between
    // dst shape and src valid_shape.
    auto getStaticElemCount = [](ArrayRef<int64_t> shape) -> std::optional<int64_t> {
      int64_t total = 1;
      for (int64_t dim : shape) {
        if (dim == ShapedType::kDynamic) {
          return std::nullopt;
        }
        if (dim <= 0) {
          return std::nullopt;
        }
        if (total > std::numeric_limits<int64_t>::max() / dim) {
          return std::nullopt;
        }
        total *= dim;
      }
      return total;
    };

    auto dstElemCount = getStaticElemCount(dstPart.getShape());
    auto srcValidElemCount = getStaticElemCount(srcValid);
    if (!hasFp && dstElemCount && srcValidElemCount &&
        *dstElemCount != *srcValidElemCount) {
      emitOpError() << "expects dst static element count (" << *dstElemCount
                    << ") to match src valid_shape static element count ("
                    << *srcValidElemCount << ")";
      return failure();
    }
    return std::make_pair(srcTile, dstPart);
  };

  auto isLoadStoreElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
           ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isI8Like = [&](Type ty) -> bool { return ty.isInteger(8); };
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common)) {
      return failure();
    }
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::MAT &&
                      *srcSpace != pto::AddressSpace::ACC)) {
      return emitOpError("expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
    }
    if ((hasFp || hasPreQuant) && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
    }
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects reluPreMode form to use loc=acc src");
    }

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
      if (hasFp || hasPreQuant) {
        return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
      }
      if (isPTOLowPrecisionType(dstElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
      }
      if (!isLoadStoreElemType(srcElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
      }
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
      }
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32())) {
      return emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
    }
    if (hasPreQuant) {
      if (srcElem.isInteger(32)) {
        if (!(isI8Like(dstElem) || dstElem.isF16())) {
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
        }
      } else if (srcElem.isF32()) {
        if (!isI8Like(dstElem)) {
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
        }
      }
    } else if (!hasFp) {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16())) {
        return emitOpError("expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
      }
    }

    auto srcShape = srcTile.getShape();
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095)) {
      return emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
    }
    auto srcValid = srcTile.getValidShape();
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095)) {
      return emitOpError("expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common)) {
      return failure();
    }
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::ACC)) {
      return emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
    }
    if ((hasFp || hasPreQuant) && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
    }
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects reluPreMode form to use loc=acc src");
    }

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC) {
      if (hasFp || hasPreQuant) {
        return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
      }
      if (!isA5TLoadStoreTransferElemType(srcElem)) {
        return emitOpError("expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
      }
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
        return emitOpError("expects A5 vec tstore src and dst element types to have the same bitwidth");
      }

      int32_t bl = srcTile.getBLayoutValueI32();
      int32_t sl = srcTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      auto srcShape = srcTile.getShape();
      bool isSpecialCase = (srcShape.size() == 2 && (srcShape[0] == 1 || srcShape[1] == 1));
      if (!isSpecialCase && !isND && !isDN && !isNZ) {
        return emitOpError("expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
      }
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32())) {
      return emitOpError("expects A5 acc tstore src element type to be i32 or f32");
    }
    if (hasPreQuant) {
      if (!isA5AccStorePreQuantDstType(srcElem, dstElem)) {
        return emitOpError("expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
      }
    } else if (!hasFp) {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16())) {
        return emitOpError("expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
      }
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
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

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy)) {
    elemTy = tb.getElementType();
  }
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<RankedTensorType, pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return tt.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return tv.getElementType();
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return tb.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return tv.getElementType();
  }
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return SmallVector<int64_t, 4>(tt.getShape().begin(), tt.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getShape().begin(), tb.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  }
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy)) {
    return rawDim;
  }
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  }
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2) {
    return dims;
  }

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i) {
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  }
  return dims;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value) {
    return {};
  }
  auto valid = getValidShapeVec(value.getType());
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size()) {
    return shape;
  }

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic) {
      shape[i] = valid[i];
    }
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    return op->emitOpError()
           << "expects " << name << " to be a tensor_view or partition_view";
  }

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
    }
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    logical1D &= shape[i] == 1;
  }
  if (!logical1D) {
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";
  }

  return success();
}

static bool isCommGlobalLikeType(Type ty) {
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

enum class CommGlobalShapePolicy {
  StaticOnly,
  AllowDynamicPartitionView,
};

static LogicalResult verifyCommGlobalLike(
    Operation *op, Value value, StringRef name,
    CommGlobalShapePolicy policy = CommGlobalShapePolicy::StaticOnly) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty)) {
    return op->emitOpError()
           << "expects " << name << " to be a tensor_view or partition_view";
  }

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }

  bool opAllowsDynamic =
      policy == CommGlobalShapePolicy::AllowDynamicPartitionView;
  bool isAllowedDynamicType = isa<pto::PartitionTensorViewType>(ty);
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      if (!opAllowsDynamic) {
        return op->emitOpError()
               << "does not support dynamic dimensions on " << name;
      }
      if (!isAllowedDynamicType) {
        return op->emitOpError() << "allows dynamic dimensions on " << name
                                 << " only for partition_tensor_view";
      }
      continue;
    }
    if (dim <= 0) {
      return op->emitOpError() << "expects every static dimension of " << name
                               << " to be positive";
    }
  }
  return success();
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name))) {
    return failure();
  }
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32)) {
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  }
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType>(ty)) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf";
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  }
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0) {
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
    }
  }
  return success();
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty()) {
    return op->emitOpError() << "expects at least one " << name << " operand";
  }
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str()))) {
      return failure();
    }
    if (it.value().getType() != groupTy) {
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
    }
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong) {
    return success();
  }
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName))) {
    return failure();
  }
  if (ping.getType() != pong.getType()) {
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  }
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return std::nullopt;
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0) {
    return std::nullopt;
  }

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static LogicalResult verifyTmpCapacityAtLeast(Operation *op, Type tmpTy,
                                              uint64_t requiredBytes,
                                              StringRef tmpName) {
  auto actualBytes = getStaticByteSize(tmpTy);
  if (!actualBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " to have statically known byte capacity";
  }
  if (*actualBytes < requiredBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " capacity to be at least "
           << requiredBytes << " bytes, but got " << *actualBytes << " bytes";
  }
  return success();
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto ptr = dyn_cast<pto::PtrType>(ty)) {
    return ptr.getMemorySpace().getAddressSpace();
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace())) {
      return as.getAddressSpace();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

TMovForm mlir::pto::classifyTMovForm(Value fp) {
  if (!fp)
    return TMovForm::NoTileAux;
  auto space = getPTOMemorySpaceEnum(fp.getType());
  if (!space)
    return TMovForm::XToZz;
  return *space == pto::AddressSpace::SCALING ? TMovForm::Fp
                                              : TMovForm::XToZz;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (allowBf16 && ty.isBF16()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32) {
    return false;
  }
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true)) {
    return true;
  }
  if (!isTargetArchA5(op)) {
    return false;
  }
  return isPTOHiFloat8Type(ty) || isPTOFloat8Type(ty);
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy) {
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";
  }

  if (!isa<pto::PartitionTensorViewType>(memTy)) {
    return op->emitOpError("expects mem to be !pto.partition_tensor_view");
  }
  if (auto layout = getLogicalViewLayout(memValue)) {
    if (*layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects mem partition view to use ND logical layout when layout "
          "can be inferred");
    }
  }
  return success();
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs);

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName,
                                                    std::optional<pto::Coalesce> coalesce) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2) {
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";
  }

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile) {
    return op->emitOpError("expects idx to be a tile_buf type");
  }

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownZeroOrUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool baseRowCoalesce =
      idxRowMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  if (!coalesce) {
    if (baseRowCoalesce || elemCoalesce) {
      return success();
    }
    return op->emitOpError()
           << "expects idx valid_shape to be [" << dataName
           << ".valid_row, 0|1] or match " << dataName
           << " valid_shape when coalesce is omitted";
  }

  if (*coalesce == pto::Coalesce::Row && (rowCoalesce1xR || rowCoalesceRx1)) {
    return success();
  }

  if (*coalesce == pto::Coalesce::Elem && elemCoalesce) {
    return success();
  }

  if (*coalesce == pto::Coalesce::Row) {
    return op->emitOpError()
           << "expects row-coalesce idx valid_shape to be [0|1, " << dataName
           << ".valid_row] or [" << dataName << ".valid_row, 0|1]";
  }

  return op->emitOpError()
         << "expects elem-coalesce idx valid_shape to match " << dataName
         << " valid_shape";
}

template <typename AttrT>
static AttrT getPTOOpAttr(Operation *op, StringRef name) {
  if (Attribute propsAttr = op->getPropertiesAsAttribute()) {
    if (auto props = dyn_cast<DictionaryAttr>(propsAttr)) {
      if (auto attr = dyn_cast_or_null<AttrT>(props.get(name))) {
        return attr;
      }
    }
  }
  return dyn_cast_or_null<AttrT>(op->getRawDictionaryAttrs().get(name));
}

template <typename OpTy>
static ParseResult parsePTOInherentAttrs(OpAsmParser &parser,
                                         OperationState &result,
                                         NamedAttrList &parsedAttrs,
                                         ArrayRef<StringRef> inherentAttrNames) {
  auto attrLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(parsedAttrs)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<typename OpTy::Properties>();
  OpTy::populateDefaultProperties(result.name, properties);
  if (failed(OpTy::setPropertiesFromAttr(
          properties, parsedAttrs.getDictionary(parser.getContext()), [&] {
            return parser.emitError(attrLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          }))) {
    return failure();
  }

  for (StringRef attrName : inherentAttrNames) {
    parsedAttrs.erase(attrName);
  }
  result.attributes = parsedAttrs;
  return success();
}

static NamedAttrList getNonInherentAttrs(Operation *op,
                                         ArrayRef<StringRef> inherentAttrNames) {
  NamedAttrList attrs;
  for (NamedAttribute attr : op->getRawDictionaryAttrs()) {
    if (llvm::is_contained(inherentAttrNames, attr.getName().getValue())) {
      continue;
    }
    attrs.append(attr);
  }
  return attrs;
}

static pto::CoalesceAttr getMGatherCoalesceAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::GatherOOBAttr getMGatherGatherOobAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
}

static pto::GatherOOB getGatherOobOrDefault(pto::MGatherOp op) {
  if (auto attr = getMGatherGatherOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::GatherOOB::Undefined;
}

static pto::CoalesceAttr getMScatterCoalesceAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::ScatterAtomicOpAttr
getMScatterScatterAtomicOpAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
      op.getProperties().scatterAtomicOp);
}

static pto::ScatterOOBAttr getMScatterScatterOobAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterOOBAttr>(op.getProperties().scatterOob);
}

static pto::ScatterConflictAttr getMScatterScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterConflictAttr>(
      op.getProperties().scatterConflict);
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MGatherOp op) {
  if (auto attr = getMGatherCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MScatterOp op) {
  if (auto attr = getMScatterCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static pto::ScatterAtomicOp getScatterAtomicOpOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterAtomicOpAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterAtomicOp::None;
}

static pto::ScatterOOB getScatterOobOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterOOB::Undefined;
}

static pto::ScatterConflictAttr getScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return getMScatterScatterConflictAttrIfPresent(op);
}

static Value getTPrintTmpIfPresent(pto::TPrintOp op) {
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name
                             << " to be in the vec address space";
  }
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  }
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  }
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  }
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  if (!srcElem.isF32()) {
    return false;
  }
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         isPTOFloat8E4M3LikeType(dstElem);
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isF16()) {
    return isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isBF16()) {
    return isPTOFloat4PackedType(dstElem);
  }
  if (isPTOFloat4PackedType(srcElem)) {
    return dstElem.isBF16();
  }
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem)) {
    return dstElem.isF32();
  }
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  }
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2) {
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    }
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy)) {
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
    }
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != 2) {
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  }
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i]) {
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
    }
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf";
  }
  if (getElemTy(lhs) != getElemTy(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  }
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i]) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
  }
  if (lhsValid.size() != rhsValid.size()) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  }
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [&]() -> LogicalResult {
    if (compareValidShape) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size()) {
    return emitMismatch();
  }

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i]) {
      return emitMismatch();
    }
  }
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b) {
        return false;
      }
    }
    return true;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i])) {
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
    }
  }
  if (!equalsKnown(src0Valid, dstValid) && !equalsKnown(src1Valid, dstValid)) {
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  }
  return success();
}

static LogicalResult verifyPartialValidPatternLoose(Operation *op, Type src0Ty,
                                                    Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i])) {
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
    }
  }
  return success();
}

[[maybe_unused]] static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2) {
    return false;
  }
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  }
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have rank-2 valid_shape";
  }
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  }
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  }
  return success();
}

static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                         Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual) {
  if (failed(verifyScalarTileOp(op, srcTy, dstTy, "src", "dst",
                                requireValidRowsEqual,
                                /*requireValidColsEqual=*/true))) {
    return failure();
  }
  if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
    op->emitOpError("scalar must be a scalar type (integer/float)");
    return failure();
  }
  return getElemTy(srcTy);
}

static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type e0 = getElemTy(src0Ty);
  Type e1 = getElemTy(src1Ty);
  Type ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    op->emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op->emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst"))) {
    return failure();
  }
  return e0;
}

static FailureOr<Type> verifyDistinctRowMajorUnaryTileOpCommon(
    Operation *op, Value src, Value dst, StringRef srcName = "src",
    StringRef dstName = "dst") {
  if (src == dst) {
    op->emitOpError("expects src and dst to use different storage");
    return failure();
  }
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    op->emitOpError("failed to get element type for src/dst");
    return failure();
  }
  if (srcElem != dstElem) {
    op->emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  return srcElem;
}

static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool supported = elemTy.isInteger(32) || elemTy.isInteger(16) ||
                   elemTy.isF16() || elemTy.isF32();
  if (targetArch == PTOArch::A5) {
    supported = supported || (allowInt8OnA5 && elemTy.isInteger(8)) ||
                (allowBf16OnA5 && elemTy.isBF16());
  }
  if (supported) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyArithmeticBinaryTileOpWithArchDispatch(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyArithmeticScalarTileOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error,
    bool requireValidRowsEqualOnA2A3 = true,
    bool requireValidRowsEqualOnA5 = false) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireValidRowsEqual) -> LogicalResult {
    FailureOr<Type> elemOr = verifyNumericScalarTileOpCommon(
        op, srcTy, dstTy, scalarTy, requireValidRowsEqual);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireValidRowsEqualOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireValidRowsEqualOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColReductionElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isInteger(16) ||
            elemTy.isInteger(32);
  if (targetArch == PTOArch::A5) {
    ok = ok || (allowInt8OnA5 && elemTy.isInteger(8)) ||
         (allowBf16OnA5 && elemTy.isBF16());
  }
  if (ok) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyTColReductionOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, bool requireNonZeroSrcOnA2A3,
    bool requireNonZeroSrcOnA5, bool allowInt8OnA5, bool allowBf16OnA5,
    StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireNonZeroSrc) -> LogicalResult {
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
      return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return op->emitOpError("expects src and dst to have the same element type");
    }
    if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    return verifyTColReductionElemTypeForArch(op, elem, targetArch, allowInt8OnA5,
                                              allowBf16OnA5, a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireNonZeroSrcOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireNonZeroSrcOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static bool isKnownZeroOrUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 0 || value == 1;
}

static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == 0 || lhs == rhs;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  return success();
}
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8)) {
    return op->emitOpError() << "expects vec tile element types to be supported";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC) {
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", allowLowPrecision)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 0 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 0 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 0 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [0, 4095]");
    }
  }
  return success();
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                       allowLowPrecision))) {
    return failure();
  }

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                     allowLowPrecision);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                   allowLowPrecision);
  }
  return failure();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace) {
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT) {
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1) {
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  }
  if (isa<pto::TileBufType>(dstTy) && dstValid[0] != ShapedType::kDynamic &&
      dstValid[0] != 1) {
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  }
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0]) {
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  }
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  }
  return success();
}
