// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <optional>
#include <string>

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOMASKSIMPLIFY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kDecimalRadix = 10;

template <typename OpTy> static bool isAllTrueMaskFrom(Value mask) {
  auto op = mask.getDefiningOp<OpTy>();
  return op && op.getPattern() == "PAT_ALL";
}

static bool isAllTrueMask(Value mask) {
  return isAllTrueMaskFrom<PsetB8Op>(mask) ||
         isAllTrueMaskFrom<PsetB16Op>(mask) ||
         isAllTrueMaskFrom<PsetB32Op>(mask) ||
         isAllTrueMaskFrom<PgeB8Op>(mask) ||
         isAllTrueMaskFrom<PgeB16Op>(mask) || isAllTrueMaskFrom<PgeB32Op>(mask);
}

// Number of predicate lanes carried by one physical mask part.
static int64_t getMaskLaneCount(Type type) {
  auto maskType = dyn_cast<MaskType>(type);
  if (!maskType) {
    return 0;
  }
  if (maskType.isB8()) {
    return 256;
  }
  if (maskType.isB16()) {
    return 128;
  }
  if (maskType.isB32()) {
    return 64;
  }
  return 0;
}

// Active lane count of a static prefix predicate (PAT_ALL / PAT_ALLF / PAT_VL),
// or nullopt when the value is not a static prefix pattern.
static std::optional<int64_t> getStaticPrefixLanes(Value mask) {
  Operation *patternOp = mask.getDefiningOp();
  bool isStaticPattern =
      patternOp &&
      isa<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(
          patternOp);
  if (!isStaticPattern) {
    return std::nullopt;
  }
  auto pattern = patternOp->getAttrOfType<StringAttr>("pattern");
  if (!pattern) {
    return std::nullopt;
  }
  StringRef name = pattern.getValue();
  if (name == "PAT_ALL") {
    return getMaskLaneCount(mask.getType());
  }
  if (name == "PAT_ALLF") {
    return 0;
  }
  StringRef digits = name;
  if (!digits.consume_front("PAT_VL")) {
    return std::nullopt;
  }
  if (digits.empty()) {
    return std::nullopt;
  }
  int64_t lanes = 0;
  if (digits.getAsInteger(kDecimalRadix, lanes)) {
    return std::nullopt;
  }
  return lanes;
}

static bool isSupportedPrefixLanes(int64_t lanes) {
  switch (lanes) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return true;
  default:
    return false;
  }
}

static Value createPatternMask(Location loc, Type type, StringRef pattern,
                               PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(type);
  if (!maskType) {
    return {};
  }
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  if (maskType.isB8()) {
    return rewriter.create<PsetB8Op>(loc, type, patternAttr).getResult();
  }
  if (maskType.isB16()) {
    return rewriter.create<PsetB16Op>(loc, type, patternAttr).getResult();
  }
  if (maskType.isB32()) {
    return rewriter.create<PsetB32Op>(loc, type, patternAttr).getResult();
  }
  return {};
}

static Value createPrefixMaskForLanes(Location loc, Type type, int64_t lanes,
                                      PatternRewriter &rewriter) {
  int64_t laneCount = getMaskLaneCount(type);
  if (laneCount <= 0) {
    return {};
  }
  if (lanes <= 0) {
    return createPatternMask(loc, type, "PAT_ALLF", rewriter);
  }
  if (lanes >= laneCount) {
    return createPatternMask(loc, type, "PAT_ALL", rewriter);
  }
  if (!isSupportedPrefixLanes(lanes)) {
    return {};
  }
  std::string pattern = "PAT_VL" + std::to_string(lanes);
  return createPatternMask(loc, type, pattern, rewriter);
}

template <typename OpTy>
struct SimplifyAllTruePredicateReorder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isAllTrueMask(op.getLhs()) || !isAllTrueMask(op.getRhs())) {
      return failure();
    }

    rewriter.replaceOp(op, {op.getLhs(), op.getRhs()});
    return success();
  }
};

// Folds a deinterleave of static prefix predicates back into the prefix masks it
// produces.  pdintlv(a, b) yields low = [even(a); even(b)] and
// high = [odd(a); odd(b)], each image occupying one half of the result; the two
// halves form one prefix lane run only when the first half is full or the second
// half is empty.  The sub-chunk form pdintlv(m, PAT_ALLF) is the useful case: it
// produces even(m) / odd(m), which are plain prefixes.
template <typename OpTy>
struct SimplifyStaticPredicateDeinterleave : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // All-true pairs keep the existing identity rewrite (and their uses).
    if (isAllTrueMask(op.getLhs()) && isAllTrueMask(op.getRhs())) {
      return failure();
    }
    std::optional<int64_t> lhsLanes = getStaticPrefixLanes(op.getLhs());
    std::optional<int64_t> rhsLanes = getStaticPrefixLanes(op.getRhs());
    if (!lhsLanes || !rhsLanes) {
      return failure();
    }
    int64_t laneCount = getMaskLaneCount(op.getLow().getType());
    if (laneCount <= 0) {
      return failure();
    }
    int64_t half = laneCount / 2;
    int64_t lowEven = (*lhsLanes + 1) / 2;
    int64_t lowOdd = *lhsLanes / 2;
    int64_t highEven = (*rhsLanes + 1) / 2;
    int64_t highOdd = *rhsLanes / 2;
    bool lowIsPrefix = lowEven == half || highEven == 0;
    bool highIsPrefix = lowOdd == half || highOdd == 0;
    if (!lowIsPrefix || !highIsPrefix) {
      return failure();
    }
    Value low = createPrefixMaskForLanes(op.getLoc(), op.getLow().getType(),
                                         lowEven + highEven, rewriter);
    Value high = createPrefixMaskForLanes(op.getLoc(), op.getHigh().getType(),
                                          lowOdd + highOdd, rewriter);
    if (!low || !high) {
      return failure();
    }
    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

struct SimplifyVselAllTrueMask : OpRewritePattern<VselOp> {
  using OpRewritePattern<VselOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VselOp op,
                                PatternRewriter &rewriter) const override {
    // vsel(src0, src1, mask) selects src0 for all-true masks.
    if (!isAllTrueMask(op.getMask())) {
      return failure();
    }

    rewriter.replaceOp(op, op.getSrc0());
    return success();
  }
};

struct VPTOMaskSimplifyPass
    : public pto::impl::VPTOMaskSimplifyBase<VPTOMaskSimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyVselAllTrueMask,
                 SimplifyAllTruePredicateReorder<PintlvB8Op>,
                 SimplifyAllTruePredicateReorder<PintlvB16Op>,
                 SimplifyAllTruePredicateReorder<PintlvB32Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB8Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB16Op>,
                 SimplifyAllTruePredicateReorder<PdintlvB32Op>,
                 SimplifyStaticPredicateDeinterleave<PdintlvB8Op>,
                 SimplifyStaticPredicateDeinterleave<PdintlvB16Op>,
                 SimplifyStaticPredicateDeinterleave<PdintlvB32Op>>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createVPTOMaskSimplifyPass() {
  return std::make_unique<VPTOMaskSimplifyPass>();
}

} // namespace mlir::pto
