// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

ParseResult mlir::pto::PartitionViewOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> sizes;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseKeyword("offsets") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(offsets) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseKeyword("sizes") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(sizes) ||
      parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
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
  if (parser.resolveOperands(offsets, indexTy, result.operands) ||
      parser.resolveOperands(sizes, indexTy, result.operands)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<PartitionViewOp::Properties>();
  llvm::copy(ArrayRef<int32_t>(
                 {1, static_cast<int32_t>(offsets.size()),
                  static_cast<int32_t>(sizes.size())}),
             properties.operandSegmentSizes.begin());

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  ValueRange allOperands(result.operands);
  ValueRange sizeOperands =
      allOperands.slice(1 + offsets.size(), sizes.size());
  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(sourceTy), sizeOperands);
  if (failed(inferredResultType)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.partition_view result type");
  }

  result.addTypes(*inferredResultType);
  return success();
}

void mlir::pto::PartitionViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", offsets = [";
  printer.printOperands(getOffsets());
  printer << "], sizes = [";
  printer.printOperands(getSizes());
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getSource().getType();

  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(getSource().getType()), getSizes());
  if (succeeded(inferredResultType) && *inferredResultType == getResult().getType()) {
    return;
  }

  printer << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(
    Value v, bool includeIndexAndIntOpsInConstFold) {
  if (includeIndexAndIntOpsInConstFold) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
      return c.value();
    }
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
      return c.value();
    }
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation &op, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold) {
  if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex()) {
    return op.emitOpError("expects indexRow and indexCol to be index type");
  }
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  if (row && *row < 0) {
    return op.emitOpError("expects indexRow to be non-negative");
  }
  if (col && *col < 0) {
    return op.emitOpError("expects indexCol to be non-negative");
  }
  return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  }
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + dstShape[0] > srcShape[0]) {
    return op.emitOpError("expects indexRow + dst.rows <= src.rows");
  }
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + dstShape[1] > srcShape[1]) {
    return op.emitOpError("expects indexCol + dst.cols <= src.cols");
  }
  return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getValidShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  }
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + srcShape[0] > dstShape[0]) {
    return op.emitOpError("expects indexRow + src.rows <= dst.rows");
  }
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + srcShape[1] > dstShape[1]) {
    return op.emitOpError("expects indexCol + src.cols <= dst.cols");
  }
  return success();
}

static unsigned getElemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static LogicalResult verifyTileBufLayoutConstraints(Operation *op,
                                                    pto::TileBufType tb,
                                                    StringRef name) {
  auto shape = tb.getShape();
  if (shape.size() != 2) {
    return op->emitOpError() << "expects " << name << " to be rank-2";
  }

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  if (rows != ShapedType::kDynamic && rows <= 0) {
    return op->emitOpError() << "expects " << name << " rows to be positive";
  }
  if (cols != ShapedType::kDynamic && cols <= 0) {
    return op->emitOpError() << "expects " << name << " cols to be positive";
  }

  unsigned elemBytes = getElemByteSize(tb.getElementType());
  if (elemBytes == 0) {
    return op->emitOpError() << "expects " << name
                             << " element type to have a byte size";
  }

  auto cfg = tb.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(tb.getContext());
  }
  auto readBLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<BLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  auto readSLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<SLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  int32_t blayout = 0;
  int32_t slayout = 0;
  if (!readBLayout(cfg.getBLayout(), blayout) ||
      !readSLayout(cfg.getSLayout(), slayout)) {
    return op->emitOpError() << "expects " << name
                             << " to have concrete tile layout attributes";
  }
  constexpr int64_t kAlignedBytes = 32;
  constexpr int64_t kPackedFp4AlignedBytes = 16;
  const int64_t requiredAlignment = isPTOFloat4PackedType(tb.getElementType())
                                        ? kPackedFp4AlignedBytes
                                        : kAlignedBytes;

  auto checkByteAlignment = [&](int64_t dim, StringRef layoutName,
                                StringRef byteExpr) -> LogicalResult {
    if (dim == ShapedType::kDynamic) {
      return success();
    }
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    if (bytes % requiredAlignment == 0)
      return success();
    return op->emitOpError()
           << "expects " << name << " " << layoutName
           << " none_box tile " << byteExpr
           << " to be " << requiredAlignment << "-byte aligned, but got "
           << bytes << " bytes";
  };

  if (slayout == static_cast<int32_t>(SLayout::NoneBox)) {
    if (blayout == static_cast<int32_t>(BLayout::RowMajor)) {
      return checkByteAlignment(cols, "row-major",
                                "row byte size (cols * sizeof(dtype))");
    }
    return checkByteAlignment(rows, "col-major",
                              "column byte size (rows * sizeof(dtype))");
  }

  int64_t innerRows = 0;
  int64_t innerCols = 0;
  int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
  switch (fractal) {
  case 1024:
    innerRows = 16;
    innerCols = 16;
    break;
  case 32:
    innerRows = 16;
    innerCols = 2;
    break;
  case 512:
    if (kAlignedBytes % elemBytes != 0) {
      return op->emitOpError() << "expects " << name
                               << " element byte size to divide 32 for boxed "
                                  "fractal-512 tile layout";
    }
    if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
      innerRows = 16;
      innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
    } else if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
      innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
      innerCols = 16;
    }
    break;
  default:
    break;
  }
  if (innerRows <= 0 || innerCols <= 0) {
    return op->emitOpError() << "expects " << name
                             << " to use a supported boxed tile layout";
  }

  auto loc = getPTOMemorySpaceEnum(tb);
  bool allowUnalignedRows =
      (loc && *loc == pto::AddressSpace::VEC) || fractal == 32 || rows == 1;
  if (!allowUnalignedRows && rows != ShapedType::kDynamic &&
      rows % innerRows != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile rows to be a multiple of innerRows (" << innerRows
           << "), but got " << rows;
  }
  if (cols != ShapedType::kDynamic && cols % innerCols != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile cols to be a multiple of innerCols (" << innerCols
           << "), but got " << cols;
  }

  return success();
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 16 || width == 32;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16()) {
    return true;
  }
  if (isPTOHiFloat8Type(ty)) {
    return true;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  }
  return false;
}

static bool isStaticLayoutInt(int64_t value) {
  return value != ShapedType::kDynamic && value >= 0;
}

static std::optional<int64_t> multiplyLayoutInts(int64_t lhs, int64_t rhs) {
  int64_t product = 0;
  if (llvm::MulOverflow(lhs, rhs, product)) {
    return std::nullopt;
  }
  return product;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0) {
    return std::nullopt;
  }
  if (llvm::any_of(shape, [](int64_t dim) { return !isStaticLayoutInt(dim); }) ||
      llvm::any_of(strides,
                   [](int64_t stride) { return !isStaticLayoutInt(stride); })) {
    return std::nullopt;
  }

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    auto sh3TimesSh4 = multiplyLayoutInts(sh3, sh4);
    auto fractalBytes =
        sh3TimesSh4
            ? multiplyLayoutInts(*sh3TimesSh4, static_cast<int64_t>(elemBytes))
            : std::nullopt;
    bool alignMatch = (sh3 == 16) && fractalBytes && (*fractalBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch) {
      return mlir::pto::Layout::NZ;
    }
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    auto expectedStride = multiplyLayoutInts(strides[i + 1], shape[i + 1]);
    if (!expectedStride || strides[i] != *expectedStride) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1) {
    return mlir::pto::Layout::ND;
  }

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    auto expectedStride = multiplyLayoutInts(strides[i], shape[i]);
    if (!expectedStride || strides[i + 1] != *expectedStride) {
      isColMajor = false;
      break;
    }
  }
  if (isColMajor && strides.front() == 1) {
    return mlir::pto::Layout::DN;
  }

  return mlir::pto::Layout::ND; // fallback
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>()) {
    return getLogicalViewLayout(part.getSource());
  }
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    // Prefer the explicit layout attribute when available.  After rank-2 →
    // rank-5 canonicalization, the padded leading strides satisfy the ND
    // (row-major) recurrence even for DN (col-major) data, so inferLayout
    // alone would misclassify DN as ND (the col-major recurrence breaks at
    // the boundary between padded unit-extent dims and real dims).  The
    // layout attribute carries the *intended* memory layout and is the
    // authoritative source — inferLayout is only a fallback for views that
    // lack an explicit layout.
    if (auto layoutAttr = make.getLayoutAttr()) {
      return layoutAttr.getLayout();
    }
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy) {
      return std::nullopt;
    }
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst) {
        return std::nullopt;
      }
      strides.push_back(*cst);
    }
    return inferLayout(shape, strides, getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type) {
    return std::nullopt;
  }
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return pto::Layout::NZ;
  }
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return pto::Layout::ND;
  }
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return pto::Layout::DN;
  }
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static bool isColMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() ==
                   static_cast<int32_t>(pto::BLayout::ColMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND) {
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
    }
  }
  return success();
}

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
        tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
      return op->emitOpError() << "expects " << name
                               << " to use the row_major or col_major blayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout == pto::Layout::DN) {
      auto shape = getShapeVec(ty);
      if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1) {
        return op->emitOpError() << "expects DN-style " << name
                                 << " to have shape[1] == 1";
      }
      return success();
    }
    if (layout && *layout == pto::Layout::ND) {
      return success();
    }
    if (layout) {
      return op->emitOpError() << "expects " << name
                               << " to use a DN-style column vector tile or legacy ND-style tile";
    }
  }
  // The dst valid_shape[1] == 1 constraint for row reductions is enforced in
  // verifyRowReductionValidRegion (it must be conditional on the no-op-marker
  // path), so it is intentionally not duplicated here. A previous unreachable
  // copy of that check lived after this return and has been removed.
  return success();
}

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool allowEmptyMarker) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // A fully-empty dst valid region (0x0) is PyPTO's dual-AIV no-op replay
  // marker: the op writes no elements, so accept it and skip the non-empty
  // structural constraints. Only plain reductions opt in (allowEmptyMarker);
  // arg reductions (trowargmax/trowargmin) still produce a real per-row index,
  // so they stay strict. One-sided empties (only one dim 0) still fall through
  // and are rejected below. Hardware Rv=0 no-op is tracked in pto-isa#143;
  // PTOAS only guarantees the IR is legal here.
  if (allowEmptyMarker && dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1) {
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  }
  return success();
}

static bool isSupportedRowReductionElemType(Type elem) {
  return elem.isInteger(16) || elem.isInteger(32) || elem.isF16() ||
         elem.isF32();
}

[[maybe_unused]] static LogicalResult
verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy, Type dstTy,
                               StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true))) {
    return failure();
  }
  if (!isSupportedRowReductionElemType(getElemTy(srcTy))) {
    return op->emitOpError(elemTypeError);
  }
  return success();
}

static LogicalResult verifyTRowReductionWithTmpCommon(Operation *op, Type srcTy,
                                                      Type tmpTy, Type dstTy,
                                                      StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (getTargetArch(op) != PTOArch::A5 &&
      getElemTy(srcTy) != getElemTy(tmpTy)) {
    return op->emitOpError("expects A2/A3 tmp to have the same element type as src and dst");
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true))) {
    return failure();
  }
  if (!isSupportedRowReductionElemType(getElemTy(srcTy))) {
    return op->emitOpError(elemTypeError);
  }
  if (getTargetArch(op) != PTOArch::A5 &&
      failed(verifyTmpCapacityAtLeast(op, tmpTy, 32))) {
    return failure();
  }
  return success();
}

static std::optional<int64_t> getVectorRepeatElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 2048 % elemBits != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(2048 / elemBits);
}

static std::optional<int64_t> getVectorBlockElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 256 % elemBits != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(256 / elemBits);
}

static int64_t ceilDivInt64(int64_t numerator, int64_t denominator) {
  if (denominator <= 0 || numerator < 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}
static std::optional<int64_t> getArgReductionTmpMinStride(Type elemTy,
                                                          int64_t srcValidCols) {
  if (srcValidCols == ShapedType::kDynamic || srcValidCols < 0) {
    return std::nullopt;
  }
  auto repeatElems = getVectorRepeatElements(elemTy);
  auto blockElems = getVectorBlockElements(elemTy);
  if (!repeatElems || !blockElems) {
    return std::nullopt;
  }
  int64_t repeats = ceilDivInt64(srcValidCols, *repeatElems);
  return (ceilDivInt64(repeats * 2, *blockElems) +
          ceilDivInt64(repeats, *blockElems)) *
         *blockElems;
}

static bool hasExactKnownValidShape(Type lhsTy, Type rhsTy) {
  return getValidShapeVec(lhsTy) == getValidShapeVec(rhsTy);
}

static LogicalResult verifyTColArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to have rank-2 valid_shape");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTColArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyTColArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColSumTmpStride(Operation *op, Type srcTy,
                                            Type tmpTy, bool isBinary) {
  if (!isBinary) {
    return success();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpShape.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  int64_t srcValidCols = srcValid[1];
  int64_t tmpStride = tmpShape[1];
  if (srcValidCols != ShapedType::kDynamic && tmpStride != ShapedType::kDynamic &&
      tmpStride < srcValidCols) {
    return op->emitOpError()
           << "expects tmp shape[1] to be at least src valid_shape[1] when "
              "isBinary is true; got "
           << tmpStride << " vs " << srcValidCols;
  }
  return success();
}

static LogicalResult verifyTRowArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcShape = getShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcShape.size() != 2 || tmpShape.size() != 2 || srcValid.size() != 2 ||
      tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  auto repeatElems = getVectorRepeatElements(getElemTy(srcTy));
  if (!repeatElems) {
    return op->emitOpError("failed to infer A2/A3 tmp contract from src element type");
  }

  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] <= *repeatElems) {
    auto tmpTile = dyn_cast<pto::TileBufType>(tmpTy);
    auto layout = tmpTile ? getTileBufLogicalLayout(tmpTile) : std::nullopt;
    if (layout && *layout == pto::Layout::DN) {
      if (tmpShape[1] != ShapedType::kDynamic && tmpShape[1] != 1) {
        return op->emitOpError("expects A2/A3 tmp DN layout to have shape[1] == 1");
      }
      if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] != 1) {
        return op->emitOpError(
            "expects A2/A3 tmp DN layout to have valid_shape[1] == 1");
      }
      if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
          tmpValid[0] < srcValid[0] * 2) {
        return op->emitOpError()
               << "expects A2/A3 tmp DN layout to have valid_shape[0] >= "
               << (srcValid[0] * 2);
      }
      return verifyTmpCapacityAtLeast(op, tmpTy, 32);
    }

    if (!layout || *layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects A2/A3 tmp to use DN 1-col or ND 2-col layout when src valid_shape[1] fits in one repeat");
    }
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
        tmpValid[0] < srcValid[0]) {
      return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < 2) {
      return op->emitOpError(
          "expects A2/A3 tmp valid_shape[1] to be at least 2 in the small-col ND path");
    }
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (srcShape[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] != srcShape[0]) {
    return op->emitOpError("expects A2/A3 tmp shape[0] to match src shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
      tmpValid[0] < srcValid[0]) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTRowArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyTRowArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, ty, name, allowLowPrecision))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    }
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
    }
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
  // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
  // first, so they stay strict without needing a flag here (unlike the row
  // path, whose dst-layout check does not constrain valid).
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    }
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
    }
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name))) {
    return failure();
  }
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2) {
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  }
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1) {
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  }
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

LogicalResult mlir::pto::SectionSimtOp::verify() {
  func::FuncOp func = getOperation()->getParentOfType<func::FuncOp>();
  if (!func) {
    return emitOpError("must be nested under a func.func");
  }

  if (getDimXAttr().getInt() < 0 || getDimYAttr().getInt() < 0 ||
      getDimZAttr().getInt() < 0) {
    return emitOpError("requires non-negative i32 launch dimensions");
  }

  if (func->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return emitOpError("must not appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "'";
  }

  WalkResult nested = getBody().walk([&](SectionSimtOp nestedOp) {
    nestedOp.emitOpError("nested pto.section.simt is not allowed");
    return WalkResult::interrupt();
  });
  if (nested.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult mlir::pto::FusionRegionOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>()) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  if (yield.getValues().size() != getOutputs().size()) {
    return emitOpError() << "expects pto.yield to return "
                         << getOutputs().size() << " values, got "
                         << yield.getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

LogicalResult mlir::pto::YieldOp::verify() {
  auto parent = dyn_cast_or_null<FusionRegionOp>(getOperation()->getParentOp());
  if (!parent) {
    return emitOpError("expects parent op to be pto.fusion_region");
  }

  if (getValues().size() != parent.getOutputs().size()) {
    return emitOpError() << "expects " << parent.getOutputs().size()
                         << " yielded values to match parent results, got "
                         << getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getValues(), parent.getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy) {
    return emitOpError("result must be pto.tensor_view<...>");
  }

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  Type ptrElemTy = ptrTy.getElementType();

  if (ptrElemTy != tvTy.getElementType()) {
    return emitOpError() << "ptr element type must match tensor_view element "
                            "type, but got ptr="
                         << ptrElemTy << " view=" << tvTy.getElementType();
  }

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank) {
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;
  }

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy) {
    return emitOpError("expects tensor_view source and partition_tensor_view result");
  }

  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();
  }

  int64_t srcRank = srcTy.getRank();
  if ((int64_t)getOffsets().size() != srcRank) {
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";
  }

  if ((int64_t)getSizes().size() != srcRank) {
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";
  }

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;

  for (int64_t i = 0; i < srcRank; ++i) {
    auto offVal = getConstIndexValue(getOffsets()[i]);
    auto sizeVal = getConstIndexValue(getSizes()[i]);

    if (offVal && *offVal < 0) {
      return emitOpError() << "offset at dim " << i
                           << " must be non-negative, got " << *offVal;
    }

    if (sizeVal && *sizeVal <= 0) {
      return emitOpError() << "size at dim " << i
                           << " must be positive, got " << *sizeVal;
    }

    if (sameRank && sizeVal) {
      int64_t resDim = resShape[i];
      if (resDim != ShapedType::kDynamic && *sizeVal != resDim) {
        return emitOpError() << "size/result mismatch at dim " << i
                             << ": size operand=" << *sizeVal
                             << " result type dim=" << resDim;
      }
    }

    int64_t srcDim = srcShape[i];
    if (srcDim == ShapedType::kDynamic) {
      continue;
    }

    if (sizeVal && *sizeVal > srcDim) {
      return emitOpError() << "size at dim " << i << " (" << *sizeVal
                           << ") exceeds static source dim (" << srcDim << ")";
    }

    if (offVal && sizeVal && (*offVal + *sizeVal > srcDim)) {
      return emitOpError() << "offset+size at dim " << i << " ("
                           << (*offVal + *sizeVal)
                           << ") exceeds static source dim (" << srcDim << ")";
    }
  }

  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  if (ptrTy != resTy) {
    return emitOpError("result type must match ptr operand type");
  }

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrTy.getElementType();
  }
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type) {
    return false;
  }
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64()) {
    return true;
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  }
  if (mlir::pto::isPTOFloat8Type(type)) {
    return true;
  }
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type)) {
    return true;
  }
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64) {
    return emitOpError("result must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getPtr().getType())) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  return success();
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64) {
    return emitOpError("address operand must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getResult().getType())) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem)) {
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;
  }

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getResult().getType() != arrayTy.getElementType()) {
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  }
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getValue().getType() != arrayTy.getElementType()) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  }
  return success();
}

// Resolve the field type reached by following a constant `path` of field
// indices from `root`, descending through nested structs. Emits an actionable
// op error and returns failure on an empty path, an out-of-range index, or a
// descent into a non-struct field. On success writes the terminal field type to
// `fieldTyOut`.
static LogicalResult walkStructPath(Operation *op, mlir::pto::StructType root,
                                    llvm::ArrayRef<int64_t> path,
                                    Type &fieldTyOut) {
  if (path.empty()) {
    return op->emitOpError() << "struct path must have at least one index";
  }
  Type cur = root;
  for (auto [depth, idx] : llvm::enumerate(path)) {
    auto st = dyn_cast<mlir::pto::StructType>(cur);
    if (!st) {
      return op->emitOpError()
             << "struct path index " << depth
             << " descends into non-struct field of type " << cur;
    }
    if (idx < 0 || idx >= static_cast<int64_t>(st.getNumFields())) {
      return op->emitOpError()
             << "struct path index " << depth << " (" << idx
             << ") is out of range for " << st << " with " << st.getNumFields()
             << " field(s)";
    }
    cur = st.getFieldType(static_cast<unsigned>(idx));
  }
  fieldTyOut = cur;
  return success();
}

// The declared struct is stack storage owned by the enclosing scope, and the
// value lowers to a pointer to that storage. Letting it reach a terminator
// would publish that address outside the owning scope: `return %s` hands the
// caller a pointer into a dead frame, and `scf.yield %s` carries it out of the
// region that owns it. Both are rejected here rather than emitted as C++ that
// looks fine and is undefined at run time.
LogicalResult mlir::pto::DeclareStructOp::verify() {
  for (Operation *user : getResult().getUsers()) {
    if (!user->hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    return emitOpError()
           << "stack-local struct must not escape the scope that declares it, "
              "but its value is passed to '"
           << user->getName()
           << "', which would expose the address of storage that is about to "
              "die; declare the struct in the outer scope and mutate it from "
              "the nested region instead (pto.struct_set mutates in place, "
              "so a struct never needs to be returned or yielded)";
  }
  return success();
}

// Both accessors bottom out at a scalar. A path ending on a nested !pto.struct
// is rejected: the member chain lowers to `emitc.member`, which yields an
// lvalue, and handing a whole aggregate back as an SSA value would mean copying
// it out of the struct — so reaching into a nested struct is spelled as a longer
// path instead.
static LogicalResult verifyStructLeafIsScalar(Operation *op, Type fieldTy) {
  if (!fieldTy.isIntOrFloat()) {
    return op->emitOpError()
           << "struct path must end at a scalar field, but ends at " << fieldTy
           << "; extend the path to reach a scalar inside it";
  }
  return success();
}

LogicalResult mlir::pto::StructGetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(
          getOperation(),
          cast<mlir::pto::StructType>(getOperation()->getOperand(0).getType()),
          getPath(), fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "result type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::StructSetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(
          getOperation(),
          cast<mlir::pto::StructType>(getOperation()->getOperand(0).getType()),
          getPath(), fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::CastPtrOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
  auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
  auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
  bool inputIsInteger = isa<IntegerType>(inputType);
  bool resultIsInteger = isa<IntegerType>(resultType);

  if (!inputPtrType && !inputMemRefType && !inputIsInteger) {
    return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
  }
  if (!resultPtrType && !resultIsInteger) {
    return emitOpError("result must be an integer or !pto.ptr<...>");
  }

  if (inputIsInteger && resultIsInteger) {
    return emitOpError("integer-to-integer cast is not a ptr cast");
  }

  if (inputMemRefType && resultIsInteger) {
    return emitOpError("memref-to-integer cast is unsupported");
  }

  if (inputMemRefType && resultPtrType) {
    auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(
        inputMemRefType.getMemorySpace());
    auto resultSpace = resultPtrType.getMemorySpace();
    if (memrefSpace && memrefSpace != resultSpace) {
      return emitOpError(
          "memref-to-ptr cast must stay within the same PTO memory space");
    }
  }

  if (inputPtrType && resultPtrType &&
      inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
    return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
  }

  return success();
}




void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
      >();

  addInterfaces<PTOInlinerInterface>();
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace();
  }
  return {};
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  if (isPTOEntryFunction(func)) {
    return true;
  }
  auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
  if (!attr) {
    return false;
  }
  return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func,
                                              bool isExternal) {
  if (!func) {
    return;
  }
  if (isExternal) {
    func->setAttr(kPTOVisibilityAttrName,
                  StringAttr::get(func.getContext(),
                                  kPTOVisibilityExternalValue));
    return;
  }
  func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module) {
    return success();
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func)) {
      continue;
    }
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func)) {
      continue;
    }
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

// A !pto.struct is represented as a pointer to stack storage. Its provenance
// must therefore remain explicit: the value comes directly from
// pto.declare_struct in the owning function. Function arguments/results and
// operations such as arith.select and scf.if must not manufacture or relay a
// struct-typed SSA value, because that alias hides the declaration from
// DeclareStructOp's direct-use escape check. CFG block arguments cannot make a
// declaration safe to forward either: the branch is a terminator and is
// rejected by DeclareStructOp::verify.
LogicalResult mlir::pto::validateStructProvenance(ModuleOp module) {
  if (!module) {
    return success();
  }

  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      for (auto [i, inputTy] :
           llvm::enumerate(func.getFunctionType().getInputs())) {
        if (!isa<StructType>(inputTy)) {
          continue;
        }
        func.emitOpError()
            << "argument " << i << " has type " << inputTy
            << ", but a stack-local struct must not be a function argument; "
               "structs must originate from 'pto.declare_struct' in the same "
               "function";
        return WalkResult::interrupt();
      }
      for (auto [i, resultTy] :
           llvm::enumerate(func.getFunctionType().getResults())) {
        if (!isa<StructType>(resultTy)) {
          continue;
        }
        func.emitOpError()
            << "result " << i << " has type " << resultTy
            << ", but a stack-local struct must not be returned: the value is "
               "a pointer into the callee's frame, and returning it (even "
               "when it merely passes an argument back through) launders its "
               "provenance; keep the struct in its declaring function "
               "(pto.struct_set mutates in place, so a result is never needed)";
        return WalkResult::interrupt();
      }
    }

    if (!isa<DeclareStructOp>(op)) {
      for (auto [i, opResult] : llvm::enumerate(op->getResults())) {
        if (!isa<StructType>(opResult.getType())) {
          continue;
        }
        op->emitOpError()
            << "result " << i << " has type " << opResult.getType()
            << ", but only 'pto.declare_struct' may produce a !pto.struct "
               "result; derived results hide the stack-storage lifetime and "
               "can escape their declaring scope";
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  (void)module;
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//===----------------------------------------------------------------------===//
