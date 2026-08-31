// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifySubViewShapeAndConfig(SubViewOp op, TileBufType srcTy,
                                                 TileBufType dstTy, int64_t sizeR,
                                                 int64_t sizeC) {
  auto dstShape = dstTy.getShape();
  if (dstShape.size() != 2) {
    return op.emitOpError("expects result to be rank-2");
  }
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2) {
    return op.emitOpError("expects source to be rank-2");
  }
  if (dstShape[0] != sizeR || dstShape[1] != sizeC) {
    return op.emitOpError("expects result shape to match subview sizes");
  }

  if (dstTy.getElementType() != srcTy.getElementType()) {
    return op.emitOpError("expects result element type to match source");
  }
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace()) {
    return op.emitOpError("expects result address space to match source");
  }
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) {
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) {
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  if (dstCfg != srcCfg) {
    return op.emitOpError("expects result tile config to match source");
  }
  return success();
}

static LogicalResult verifySubViewValidShape(SubViewOp op, TileBufType dstTy,
                                             int64_t sizeR, int64_t sizeC) {
  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid) {
      return defaultSize;
    }
    int64_t c = 0;
    if (getConstIndex(explicitValid, c)) {
      return std::min<int64_t>(c, defaultSize);
    }
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(op.getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(op.getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != 2) {
    return op.emitOpError("expects result to have rank-2 valid_shape");
  }
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !op.getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !op.getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred) {
    return op.emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  }
  if (dstValid[1] != expectedVCol && !colInferred) {
    return op.emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");
  }
  return success();
}

static LogicalResult verifySubViewBoxed(SubViewOp op, TileBufType srcTy,
                                        const SubViewInfo &info) {
  auto cfg = srcTy.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(op.getContext());
  }

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }

  if (!boxed) {
    return success();
  }

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError("boxed layout subview sizes must be multiples of inner shape");
  }

  if (info.offRConst) {
    if (info.offR % innerRows != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }
  if (info.offCConst) {
    if (info.offC % innerCols != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }

  (void)bl;
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2 ||
      srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
    return op.emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2) {
    return emitOpError("expects rank-2 tilebuf for src/dst");
  }

  SubViewInfo info;
  if (failed(verifySubViewSizesAndOffsets(*this, info))) {
    return failure();
  }
  if (failed(verifySubViewValidBounds(*this, info.sizeR, info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewShapeAndConfig(*this, srcTy, dstTy, info.sizeR,
                                         info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewValidShape(*this, dstTy, info.sizeR, info.sizeC))) {
    return failure();
  }
  return verifySubViewBoxed(*this, srcTy, info);
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

// =============================================================================
// Helper Functions
// =============================================================================

// =============================================================================
// Side Effects Implementation
// =============================================================================

// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value

// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand) {
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
  }
}

// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result) {
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
  }
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty()) {
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (classifyTMovForm(getFp()) == TMovForm::XToZz) {
    const MxGroupAxis axis = getGrpAxisAttr()
                                 ? getGrpAxisAttr().getValue()
                                 : MxGroupAxis::Axis1;
    if (axis == MxGroupAxis::Axis1) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
    } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    }
    addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
    auto fp = getFpMutable();
    if (axis == MxGroupAxis::Axis1 && !fp.empty()) {
      addEffect(effects, &fp[0], MemoryEffects::Read::get());
      addEffect(effects, &fp[0], MemoryEffects::Write::get());
    }
    return;
  }
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty()) {
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  // GM -> L1 Elem mode stages the gathered elements into the GM scratch buffer
  // before the bulk copy: the op clobbers scratch, so model it as a write.
  auto scratchRange = getScratchMutable();
  if (!scratchRange.empty()) {
    addEffect(effects, &*scratchRange.begin(), MemoryEffects::Write::get());
  }
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddReluOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty()) {
    PTO_ADD_WRITE(cdst[0]);
  }
  if (auto indices = getIndicesMutable(); !indices.empty()) {
    PTO_ADD_READ(indices[0]);
  }
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getDst0Mutable());
  PTO_ADD_WRITE(getDst1Mutable());
}
void TDeInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &operand : getSrcsMutable()) {
    PTO_ADD_READ(operand);
  }
  for (auto &operand : getDstsMutable()) {
    PTO_ADD_WRITE(operand);
  }
}
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty()) {
    PTO_ADD_READ(offsetRange[0]);
  }
  auto tmpRange = getTmpMutable();
  if (!tmpRange.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmpRange[0]);
    PTO_ADD_WRITE(tmpRange[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  Type srcTy = getSrc().getType();
  auto valid = getValidShapeVec(srcTy);
  auto physical = getShapeVec(srcTy);
  Type elem = getElemTy(srcTy);
  if ((elem.isF16() || elem.isBF16()) && valid.size() == 2 && physical.size() == 2 &&
      valid[1] < physical[1]) {
    addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
  } else {
    addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  }
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getExpMutable());
  PTO_ADD_WRITE(getMaxMutable());
  PTO_ADD_WRITE(getScalingMutable());
  auto expZzRange = getExpZzMutable();
  if (!expZzRange.empty()) {
    PTO_ADD_WRITE(expZzRange[0]);
  }
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPowOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getBaseMutable());
  PTO_ADD_READ(getExpMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPowSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty()) {
      PTO_ADD_READ(idx[0]);
    }
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp on A2/A3, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TSEL; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(mask, src) -> Write(tmp on A2/A3, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TSELS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp on A2/A3, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TXORS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp on A2/A3, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TXOR; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && ttransUsesTmp(getSrc().getType(), getDst().getType())) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

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
