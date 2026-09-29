// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOOPTIMIZEVCVT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kB16StorageBits = 16;
constexpr unsigned kB32StorageBits = 32;
constexpr unsigned kFourLanePackingFactor = 4;
constexpr int64_t kU16LaneMask = 0xFFFF;

static bool isOddPart(StringRef part) {
  return part == "ODD" || part == "PART_ODD";
}

static bool isAllTrueMask(Value mask) {
  if (auto op = mask.getDefiningOp<PsetB8Op>()) {
    return op.getPattern() == "PAT_ALL";
  }
  if (auto op = mask.getDefiningOp<PsetB16Op>()) {
    return op.getPattern() == "PAT_ALL";
  }
  if (auto op = mask.getDefiningOp<PsetB32Op>()) {
    return op.getPattern() == "PAT_ALL";
  }
  return false;
}

static bool isMatchingPrefixMaskPair(Value cvtMask, Value storeMask) {
  auto cvtPset = cvtMask.getDefiningOp<PsetB32Op>();
  auto storePset = storeMask.getDefiningOp<PsetB16Op>();
  if (!cvtPset || !storePset) {
    return false;
  }

  StringRef pattern = cvtPset.getPattern();
  // The same PAT_VL prefix selects the same logical lanes at b32 and b16.
  return pattern.starts_with("PAT_VL") && pattern == storePset.getPattern();
}

static std::optional<int64_t> getConstantInt(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    size_t numInputs = cast.getInputs().size();
    if (numInputs != 1) {
      return std::nullopt;
    }
    value = cast.getInputs().front();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
    return constant.value();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
      return integer.getInt();
    }
  }
  return std::nullopt;
}

static bool isPairEquivalentLoadDist(StringRef dist) {
  return dist == "BRC_B8" || dist == "BRC_B16" || dist == "BRC_B32" ||
         dist == "US_B8" || dist == "US_B16" || dist == "E2B_B16" ||
         dist == "E2B_B32";
}

static bool hasEvenOddEquivalentLanes(Value value) {
  if (value.getDefiningOp<VbrOp>()) {
    return true;
  }

  auto load = value.getDefiningOp<VldsOp>();
  if (!load || value != load.getResult()) {
    return false;
  }

  std::optional<StringRef> dist = load.getDist();
  return dist && isPairEquivalentLoadDist(*dist);
}

static bool isNarrowToWideVcvt(VcvtOp op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType) {
    return false;
  }

  unsigned inputBits = getPTOStorageElemBitWidth(inputType.getElementType());
  unsigned resultBits = getPTOStorageElemBitWidth(resultType.getElementType());
  return inputBits != 0 && resultBits != 0 && inputBits < resultBits;
}

static Value stripVbitcasts(Value value) {
  while (auto bitcast = value.getDefiningOp<VbitcastOp>()) {
    value = bitcast.getInput();
  }
  return value;
}

struct AlignedUnsignedWidening {
  unsigned payloadBits;
  unsigned carrierBits;
};

static bool isZeroGapLoad(Value value, AlignedUnsignedWidening widening) {
  auto load = stripVbitcasts(value).getDefiningOp<VldsOp>();
  if (!load) {
    return false;
  }

  auto loadType = dyn_cast<VRegType>(load.getResult().getType());
  std::optional<StringRef> dist = load.getDist();
  if (!loadType || !dist ||
      getPTOStorageElemBitWidth(loadType.getElementType()) !=
          widening.payloadBits) {
    return false;
  }

  if (widening.payloadBits == mlir::pto::kValue8 &&
      widening.carrierBits == kB16StorageBits) {
    return *dist == "UNPK_B8";
  }
  if (widening.payloadBits == mlir::pto::kValue16 &&
      widening.carrierBits == kB32StorageBits) {
    return *dist == "UNPK_B16";
  }
  if (widening.payloadBits == mlir::pto::kValue8 &&
      widening.carrierBits == kB32StorageBits) {
    return *dist == "UNPK4";
  }
  return false;
}

static bool isZeroGapNarrowingVcvt(Value value,
                                   AlignedUnsignedWidening widening) {
  auto cvt = value.getDefiningOp<VcvtOp>();
  if (!cvt || !isAllTrueMask(cvt.getMask())) {
    return false;
  }

  auto inputType = dyn_cast<VRegType>(cvt.getInput().getType());
  auto resultType = dyn_cast<VRegType>(cvt.getResult().getType());
  if (!inputType || !resultType) {
    return false;
  }

  unsigned inputBits = getPTOStorageElemBitWidth(inputType.getElementType());
  unsigned resultBits = getPTOStorageElemBitWidth(resultType.getElementType());
  if (inputBits != widening.carrierBits ||
      resultBits != widening.payloadBits ||
      inputBits <= resultBits ||
      inputType.getElementCount() * inputBits !=
          resultType.getElementCount() * resultBits) {
    return false;
  }

  std::optional<StringRef> part = cvt.getPart();
  if (!part) {
    return false;
  }
  return (inputBits == resultBits * mlir::pto::kValue2 && *part == "EVEN") ||
         (inputBits == resultBits * mlir::pto::kValue4 && *part == "P0");
}

// Deliberately admit only instructions that construct a carrier by zero
// filling its gaps. Ordinary computation is not treated as provenance even
// when a particular operand combination could preserve zero gaps.
static bool isCanonicalZeroGapCarrier(Value value,
                                      AlignedUnsignedWidening widening) {
  value = stripVbitcasts(value);
  if (isZeroGapLoad(value, widening)) {
    return true;
  }
  if (isZeroGapNarrowingVcvt(value, widening)) {
    return true;
  }

  auto unpack = value.getDefiningOp<VzunpackOp>();
  if (!unpack) {
    return false;
  }
  auto sourceType = dyn_cast<VRegType>(unpack.getSrc().getType());
  auto resultType = dyn_cast<VRegType>(unpack.getResult().getType());
  if (!sourceType || !resultType) {
    return false;
  }

  unsigned sourceBits =
      getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits != widening.carrierBits ||
      resultBits != sourceBits * mlir::pto::kValue2 || widening.payloadBits > sourceBits) {
    return false;
  }

  if (widening.payloadBits == sourceBits) {
    return true;
  }
  return isCanonicalZeroGapCarrier(
      unpack.getSrc(), {widening.payloadBits, sourceBits});
}

static std::optional<AlignedUnsignedWidening>
matchAlignedUnsignedWidening(VcvtOp op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType || op.getRndAttr() || op.getSatAttr() ||
      !isAllTrueMask(op.getMask())) {
    return std::nullopt;
  }

  auto inputElementType = dyn_cast<IntegerType>(inputType.getElementType());
  auto resultElementType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!inputElementType || !resultElementType ||
      !inputElementType.isUnsigned() || !resultElementType.isUnsigned()) {
    return std::nullopt;
  }

  unsigned inputBits = inputElementType.getWidth();
  unsigned resultBits = resultElementType.getWidth();
  if (inputBits >= resultBits ||
      inputType.getElementCount() * inputBits !=
          resultType.getElementCount() * resultBits) {
    return std::nullopt;
  }

  std::optional<StringRef> part = op.getPart();
  if (!part) {
    return std::nullopt;
  }
  if ((resultBits == inputBits * mlir::pto::kValue2 && *part != "EVEN") ||
      (resultBits == inputBits * mlir::pto::kValue4 && *part != "P0") ||
      (resultBits != inputBits * mlir::pto::kValue2 &&
       resultBits != inputBits * kFourLanePackingFactor)) {
    return std::nullopt;
  }

  return AlignedUnsignedWidening{inputBits, resultBits};
}

struct CanonicalizeEquivalentPartPattern : public OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> part = op.getPart();
    if (!part || !isOddPart(*part) || !isNarrowToWideVcvt(op) ||
        !isAllTrueMask(op.getMask()) ||
        !hasEvenOddEquivalentLanes(op.getInput())) {
      return failure();
    }

    rewriter.modifyOpInPlace(
        op, [&] { op.setPartAttr(rewriter.getStringAttr("EVEN")); });
    return success();
  }
};

struct FoldZeroGapExtensionPattern : public OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<AlignedUnsignedWidening> widening =
        matchAlignedUnsignedWidening(op);
    if (!widening || !isCanonicalZeroGapCarrier(op.getInput(), *widening)) {
      return failure();
    }

    Value carrier = stripVbitcasts(op.getInput());
    Value result = carrier.getType() == op.getResult().getType()
                       ? carrier
                       : rewriter
                             .create<VbitcastOp>(op.getLoc(),
                                                 op.getResult().getType(),
                                                 op.getInput())
                             .getResult();
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Fold u32 -> u16 narrowing when the source has already selected one half of
// every u32 lane:
//
//   high = vshrs(x, 16)
//   low  = vand(x, 0xffff)
//
// Truncating either result to u16 is exactly the corresponding high/low
// sublane stream of x.  The 2x integer-narrowing lowering uses the even-part
// form, so both source kinds are matched with part = "EVEN"; the source op,
// not the part, decides whether vdintlv's high or low result is selected.
// Reinterpret x once as u16 and use vdintlv(x16, x16) to expose both streams
// without materializing the vshr/vand source and without requiring a
// pre-existing vbitcast in the input pattern.
struct FoldU32ToU16ExtractionPattern : public OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isConstantVector(Value value, int64_t expected) {
    if (auto duplicate = value.getDefiningOp<VdupOp>()) {
      if (!isAllTrueMask(duplicate.getMask())) {
        return false;
      }
      std::optional<int64_t> constant = getConstantInt(duplicate.getInput());
      return constant && *constant == expected;
    }
    if (auto broadcast = value.getDefiningOp<VbrOp>()) {
      std::optional<int64_t> constant = getConstantInt(broadcast.getValue());
      return constant && *constant == expected;
    }
    return false;
  }

  static std::optional<Value> getLowHalfSource(Value value) {
    auto andOp = value.getDefiningOp<VandOp>();
    if (!andOp || !isAllTrueMask(andOp.getMask())) {
      return std::nullopt;
    }
    if (isConstantVector(andOp.getRhs(), kU16LaneMask)) {
      return andOp.getLhs();
    }
    if (isConstantVector(andOp.getLhs(), kU16LaneMask)) {
      return andOp.getRhs();
    }
    return std::nullopt;
  }

  static std::optional<Value> getHighHalfSource(Value value) {
    auto shift = value.getDefiningOp<VshrsOp>();
    if (!shift || !isAllTrueMask(shift.getMask())) {
      return std::nullopt;
    }
    std::optional<int64_t> amount = getConstantInt(shift.getScalar());
    if (!amount || *amount != static_cast<int64_t>(kB16StorageBits)) {
      return std::nullopt;
    }
    return shift.getInput();
  }

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<VRegType>(op.getInput().getType());
    auto resultType = dyn_cast<VRegType>(op.getResult().getType());
    if (!inputType || !resultType || !isAllTrueMask(op.getMask())) {
      return failure();
    }

    auto inputIntType = dyn_cast<IntegerType>(inputType.getElementType());
    auto resultIntType = dyn_cast<IntegerType>(resultType.getElementType());
    if (!inputIntType || !resultIntType) {
      return failure();
    }
    bool unsignedTypes =
        inputIntType.isUnsigned() && resultIntType.isUnsigned();
    unsigned inputBits = inputIntType.getWidth();
    unsigned resultBits = resultIntType.getWidth();
    if (!unsignedTypes || inputBits != kB32StorageBits ||
        resultBits != kB16StorageBits) {
      return failure();
    }
    int64_t inputTotalBits = inputType.getElementCount() * inputBits;
    int64_t resultTotalBits = resultType.getElementCount() * resultBits;
    if (inputTotalBits != resultTotalBits) {
      return failure();
    }
    StringAttr sat = op.getSatAttr();
    bool noSaturation = sat && sat.getValue() == "NOSAT";
    if (!noSaturation) {
      return failure();
    }

    std::optional<StringRef> part = op.getPart();
    if (!part || *part != "EVEN") {
      return failure();
    }

    std::optional<Value> highSource = getHighHalfSource(op.getInput());
    std::optional<Value> lowSource = getLowHalfSource(op.getInput());
    if (highSource && lowSource) {
      return failure();
    }

    bool useHigh = static_cast<bool>(highSource);
    std::optional<Value> source = useHigh ? highSource : lowSource;
    if (!source) {
      return failure();
    }

    Value packed = rewriter
                       .create<VbitcastOp>(op.getLoc(), resultType, *source)
                       .getResult();
    auto extracted = rewriter.create<VdintlvOp>(
        op.getLoc(), resultType, resultType, packed, packed);
    rewriter.replaceOp(op, useHigh ? extracted.getHigh() : extracted.getLow());
    return success();
  }
};

// A NORM b16 store consumes the logical bf16 lane stream, but an EVEN vcvt
// leaves each payload in the lower 16-bit sublane. Promote such a store to
// PK_B32, which extracts those sublanes directly, and widen its mask to b32.
struct PromoteEvenBF16StoreToPkB32Pattern : OpRewritePattern<VcvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VcvtOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }
    auto inputType = dyn_cast<VRegType>(op.getInput().getType());
    auto resultType = dyn_cast<VRegType>(op.getResult().getType());
    if (!inputType || !resultType || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isBF16()) {
      return failure();
    }
    std::optional<StringRef> part = op.getPart();
    if (!part || *part != "EVEN") {
      return failure();
    }

    Operation *user = *op->user_begin();
    auto store = dyn_cast<VstsOp>(user);
    bool hasExpectedStore = store && store.getValue() == op.getResult();
    if (!hasExpectedStore) {
      return failure();
    }
    StringAttr storeDist = store.getDistAttr();
    bool hasUnsupportedStoreDist =
        storeDist && storeDist.getValue() != "NORM_B16";
    if (hasUnsupportedStoreDist) {
      return failure();
    }
    bool hasCompatibleMasks =
        isAllTrueMask(op.getMask()) ||
        isMatchingPrefixMaskPair(op.getMask(), store.getMask());
    if (!hasCompatibleMasks) {
      return failure();
    }
    auto maskType = dyn_cast<MaskType>(store.getMask().getType());
    if (!maskType || maskType.getGranularity() != "b16") {
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(store);
    auto b32MaskType = MaskType::get(rewriter.getContext(), "b32");
    Value mask32 =
        rewriter
            .create<PunpackOp>(store.getLoc(), b32MaskType, store.getMask(),
                               rewriter.getStringAttr("LOWER"))
            .getResult();
    store.getMaskMutable().set(mask32);
    store.setDistAttr(rewriter.getStringAttr("PK_B32"));
    return success();
  }
};

struct VPTOOptimizeVcvtPass
    : public pto::impl::VPTOOptimizeVcvtBase<VPTOOptimizeVcvtPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeEquivalentPartPattern,
                 FoldZeroGapExtensionPattern,
                 FoldU32ToU16ExtractionPattern,
                 PromoteEvenBF16StoreToPkB32Pattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createVPTOOptimizeVcvtPass() {
  return std::make_unique<VPTOOptimizeVcvtPass>();
}

} // namespace mlir::pto
