// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <memory>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSOFTPOSTUPDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// A hardware block is 32 bytes; block-strided ops count in these units.
static constexpr int64_t kBlockSizeBytes = 32;

// What one unit of an op's strideOperand means, in address terms.  This is a
// property of the op's lowering, not of the pass: `Element` ops run their
// offset through convertElementOffsetToBytes, `Block` ops pass a packed
// control word straight to the intrinsic, and `Byte` ops pass a raw byte
// offset.  See `strideUnitBytes` for the conversion.
enum class StrideUnit {
  Element, // vlds/vsts/vldsx2/vstas: offset in pointer elements
  Block,   // vsstb/vsldb: repeat_stride in 32-byte blocks
  Byte,    // sprsts/sprsti: raw byte offset
};

// Per-op-type descriptor: how to extract address operands and check post-update.
// base/strideOperand indices are operand positions.
struct PostUpdateOpInfo {
  int baseOperandIdx;
  int strideOperandIdx;
  StrideUnit strideUnit;
  unsigned minResultsForPost;  // numResults > this means already post-update
};

using PostUpdateTable = llvm::StringMap<PostUpdateOpInfo>;

static const PostUpdateTable &getPostUpdateTable() {
  static const PostUpdateTable table = [] {
    PostUpdateTable t;
    //                       base  strideOp  strideUnit             minResults
    t["pto.vlds"]         = { 0,    1,        StrideUnit::Element,   1 };
    t["pto.vsts"]         = { 1,    2,        StrideUnit::Element,   0 };
    t["pto.vsstb"]        = { 1,    3,        StrideUnit::Block,     0 };
    return t;
  }();
  return table;
}

// Bytes covered by one unit of `pto.addptr`'s offset on `base`.  This is the
// size of the GEP element type the pointer lowers to, which for ordinary
// int/float element types is just their byte width.  Packed low-precision
// types are normalized to something else during lowering
// (normalizeGEPElementTypeForLLVMLowering), so bail on anything that is not a
// plain byte-sized int/float rather than guess.
static std::optional<int64_t> addPtrUnitBytes(Value base) {
  Type elemTy;
  if (auto ptrTy = dyn_cast<pto::PtrType>(base.getType()))
    elemTy = ptrTy.getElementType();
  else if (auto memrefTy = dyn_cast<MemRefType>(base.getType()))
    elemTy = memrefTy.getElementType();
  else
    return std::nullopt;

  if (!elemTy || !elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return static_cast<int64_t>(bits / 8);
}

// Bytes covered by one unit of the op's strideOperand.
static int64_t strideUnitBytes(StrideUnit unit, int64_t elemBytes) {
  switch (unit) {
  case StrideUnit::Element:
    return elemBytes;
  case StrideUnit::Block:
    return kBlockSizeBytes;
  case StrideUnit::Byte:
    return 1;
  }
  llvm_unreachable("unhandled StrideUnit");
}

static const PostUpdateOpInfo *getPostUpdateInfo(Operation *op) {
  auto it = getPostUpdateTable().find(op->getName().getStringRef());
  if (it == getPostUpdateTable().end())
    return nullptr;
  return &it->second;
}

// Extract base and stride operand from a candidate op using table info.
static void extractBaseAndStrideOperand(Operation *op,
                                        const PostUpdateOpInfo &info,
                                        Value &base, Value &strideOperand) {
  base = op->getOperand(info.baseOperandIdx);
  strideOperand = op->getOperand(info.strideOperandIdx);
}

// Check if op already has an updated_base result.
static bool isAlreadyPostUpdate(Operation *op, const PostUpdateOpInfo &info) {
  return op->getNumResults() > info.minResultsForPost;
}

// Check if op is directly inside the scf.for body (not nested in scf.if etc).
static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
  return op->getParentOp() == forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Accumulator Analysis: Linear Decomposition
//===----------------------------------------------------------------------===//

// Stride analysis is purely symbolic: it never creates IR.  A candidate is
// analyzed, checked for viability, and only then materialized at a single
// insertion point.  Two properties follow from that split:
//   - Every Value the analysis inspects is a pre-existing loop-body value, so
//     "defined before insertPt" transitively implies its operands are too.
//     Availability can be decided by looking at the expression's leaves alone.
//   - The recursion is side-effect free, so it can be memoized; decomposition
//     stays linear in the size of the def-chain DAG instead of exponential.
struct StrideExpr;
using StrideExprRef = std::shared_ptr<const StrideExpr>;

struct StrideExpr {
  enum class Kind { Const, Leaf, Add, Sub, Mul, Cast };
  Kind kind;
  int64_t constant = 0;        // Kind::Const
  Value leaf;                  // Kind::Leaf
  Operation *castOp = nullptr; // Kind::Cast — template op to clone
  StrideExprRef lhs, rhs;
};

static StrideExprRef makeConst(int64_t c) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Const;
  e->constant = c;
  return e;
}

static StrideExprRef makeLeaf(Value v) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Leaf;
  e->leaf = v;
  return e;
}

// Compile-time value of `e`, if it has one.
static std::optional<int64_t> foldConst(const StrideExprRef &e) {
  if (!e)
    return std::nullopt;
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return e->constant;
  case StrideExpr::Kind::Leaf:
    return getConstantIntValue(e->leaf);
  case StrideExpr::Kind::Cast:
    // index_cast/index_castui preserve the numeric value; only the type
    // changes, and the type is chosen at materialization time.
    return foldConst(e->lhs);
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    auto a = foldConst(e->lhs);
    auto b = foldConst(e->rhs);
    if (!a || !b)
      return std::nullopt;
    if (e->kind == StrideExpr::Kind::Add)
      return *a + *b;
    if (e->kind == StrideExpr::Kind::Sub)
      return *a - *b;
    return *a * *b;
  }
  }
  return std::nullopt;
}

static StrideExprRef makeBinary(StrideExpr::Kind kind, StrideExprRef a,
                                StrideExprRef b) {
  auto e = std::make_shared<StrideExpr>();
  e->kind = kind;
  e->lhs = std::move(a);
  e->rhs = std::move(b);
  return e;
}

static StrideExprRef makeAdd(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb)
    return makeConst(*ca + *cb);
  if (ca && *ca == 0)
    return b;
  if (cb && *cb == 0)
    return a;
  return makeBinary(StrideExpr::Kind::Add, a, b);
}

static StrideExprRef makeSub(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb)
    return makeConst(*ca - *cb);
  if (cb && *cb == 0)
    return a;
  return makeBinary(StrideExpr::Kind::Sub, a, b);
}

static StrideExprRef makeMul(StrideExprRef a, StrideExprRef b) {
  auto ca = foldConst(a), cb = foldConst(b);
  if (ca && cb)
    return makeConst(*ca * *cb);
  if ((ca && *ca == 0) || (cb && *cb == 0))
    return makeConst(0);
  if (ca && *ca == 1)
    return b;
  if (cb && *cb == 1)
    return a;
  return makeBinary(StrideExpr::Kind::Mul, a, b);
}

static StrideExprRef makeCast(Operation *castOp, StrideExprRef a) {
  if (auto c = foldConst(a))
    return makeConst(*c);
  auto e = std::make_shared<StrideExpr>();
  e->kind = StrideExpr::Kind::Cast;
  e->castOp = castOp;
  e->lhs = std::move(a);
  return e;
}

static void collectLeaves(const StrideExprRef &e,
                          SmallVectorImpl<Value> &out) {
  if (!e)
    return;
  if (e->kind == StrideExpr::Kind::Leaf) {
    out.push_back(e->leaf);
    return;
  }
  collectLeaves(e->lhs, out);
  collectLeaves(e->rhs, out);
}

// Determine the concrete type `e` will materialize to.  Returns false when two
// subexpressions demand different types (an expression we must not build).
// `out` is left null when `e` is fully constant and can adapt to any type.
static bool exprType(const StrideExprRef &e, Type &out) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return true; // adapts to context
  case StrideExpr::Kind::Leaf:
    out = e->leaf.getType();
    return true;
  case StrideExpr::Kind::Cast:
    out = e->castOp->getResult(0).getType();
    return true;
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    Type ta, tb;
    if (!exprType(e->lhs, ta) || !exprType(e->rhs, tb))
      return false;
    if (ta && tb && ta != tb)
      return false;
    out = ta ? ta : tb;
    return true;
  }
  }
  return false;
}

// Result of decomposing a value into blockArg * coeff + increment.
struct LinearDecomp {
  int64_t coeff;
  StrideExprRef increment; // never null; zero is makeConst(0)
};

using DecompCache = DenseMap<Value, std::optional<LinearDecomp>>;

// Decompose `v` into blockArg * coeff + increment by recursing through
// addi/subi/muli/index_cast/addptr chains.  Pure: builds only StrideExprs.
// `cache` is scoped to one decomposition (a single `blockArg`).
static std::optional<LinearDecomp>
decomposeLinear(Value v, BlockArgument blockArg, scf::ForOp forOp,
                DecompCache &cache) {
  // v == blockArg → {1, 0}
  if (v == blockArg)
    return LinearDecomp{1, makeConst(0)};

  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;
  auto record = [&](std::optional<LinearDecomp> r) {
    cache[v] = r;
    return r;
  };

  // v is other block arg (IV, different iter_arg, func arg) → {0, v}
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return record(LinearDecomp{0, makeLeaf(v)});

  // v is loop-invariant or constant → {0, v}
  if (forOp.isDefinedOutsideOfLoop(v) ||
      defOp->hasTrait<OpTrait::ConstantLike>())
    return record(LinearDecomp{0, makeLeaf(v)});

  // v = addi(a, b) → {ca + cb, ia + ib}
  // v = subi(a, b) → {ca - cb, ia - ib}
  if (isa<arith::AddIOp, arith::SubIOp>(defOp)) {
    auto da = decomposeLinear(defOp->getOperand(0), blockArg, forOp, cache);
    auto db = decomposeLinear(defOp->getOperand(1), blockArg, forOp, cache);
    if (!da || !db)
      return record(std::nullopt);
    if (da->coeff == 0 && db->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    bool isSub = isa<arith::SubIOp>(defOp);
    return record(LinearDecomp{
        isSub ? da->coeff - db->coeff : da->coeff + db->coeff,
        isSub ? makeSub(da->increment, db->increment)
              : makeAdd(da->increment, db->increment)});
  }

  // v = muli(a, b), one side blockArg-free with constant k → {c * k, i * k}
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto da = decomposeLinear(mulOp.getLhs(), blockArg, forOp, cache);
    auto db = decomposeLinear(mulOp.getRhs(), blockArg, forOp, cache);
    if (!da || !db)
      return record(std::nullopt);
    if (da->coeff == 0 && db->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    if (da->coeff != 0 && db->coeff != 0)
      return record(std::nullopt);
    const LinearDecomp &withBA = (da->coeff != 0) ? *da : *db;
    Value multiplier = (da->coeff != 0) ? mulOp.getRhs() : mulOp.getLhs();
    auto constMul = getConstantIntValue(multiplier);
    if (!constMul)
      return record(std::nullopt);
    return record(LinearDecomp{withBA.coeff * *constMul,
                               makeMul(withBA.increment, makeConst(*constMul))});
  }

  // v = index_cast(a) → {ca, cast(ia)}
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(defOp)) {
    auto d = decomposeLinear(defOp->getOperand(0), blockArg, forOp, cache);
    if (!d)
      return record(std::nullopt);
    if (d->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    return record(LinearDecomp{d->coeff, makeCast(defOp, d->increment)});
  }

  // v = addptr(ptr, offset) → {c_ptr, i_ptr + offset}
  if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
    auto dp = decomposeLinear(addPtrOp.getPtr(), blockArg, forOp, cache);
    if (!dp)
      return record(std::nullopt);
    if (dp->coeff == 0)
      return record(LinearDecomp{0, makeLeaf(v)});
    return record(LinearDecomp{
        dp->coeff, makeAdd(dp->increment, makeLeaf(addPtrOp.getOffset()))});
  }

  // Unrecognized op → unknown
  return record(std::nullopt);
}

// Outcome of accumulator analysis.  Distinguishing "not an iter_arg" from
// "is an iter_arg but could not be decomposed" matters: the former falls back
// to delta analysis, the latter must give up (treating an unknown increment as
// zero would silently miscompile).
enum class StrideStatus { NotIterArg, Failed, Ok };

struct StrideResult {
  StrideStatus status;
  StrideExprRef expr; // valid only when status == Ok
};

// Trace `v` back to an iter_arg BlockArgument of `forOp`, then decompose
// the yield expression to extract the per-iteration increment. Walks through
// index_cast (type-changing), addi/subi with loop-invariant offset, and
// addptr with loop-invariant offset. Type casts along the path are applied
// to the increment so its type matches v's context.  Pure: creates no IR.
static StrideResult getIterArgIncrement(Value v, scf::ForOp forOp) {
  SmallVector<Operation *> casts;
  Value current = v;

  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (blockArg.getOwner() != forOp.getBody() ||
          blockArg.getArgNumber() == 0)
        return {StrideStatus::NotIterArg, nullptr};

      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      DecompCache cache;
      auto decomp =
          decomposeLinear(yieldOp.getOperand(idx), blockArg, forOp, cache);
      if (!decomp || decomp->coeff != 1)
        return {StrideStatus::Failed, nullptr};

      StrideExprRef inc = decomp->increment;
      for (Operation *op : llvm::reverse(casts))
        inc = makeCast(op, inc);
      return {StrideStatus::Ok, inc};
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp || forOp.isDefinedOutsideOfLoop(current) ||
        defOp->hasTrait<OpTrait::ConstantLike>())
      return {StrideStatus::NotIterArg, nullptr};

    if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(defOp)) {
      casts.push_back(defOp);
      current = defOp->getOperand(0);
      continue;
    }

    if (isa<arith::AddIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(0))) {
        current = defOp->getOperand(1);
        continue;
      }
    }

    if (isa<arith::SubIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
    }

    if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(addPtrOp.getOffset())) {
        current = addPtrOp.getPtr();
        continue;
      }
    }

    return {StrideStatus::NotIterArg, nullptr};
  }
}

//===----------------------------------------------------------------------===//
// Delta Analysis
//===----------------------------------------------------------------------===//

// Cached delta results.  A cached null expr means "delta is unknown"; absence
// from the map means "not computed yet".
using DeltaCache = DenseMap<Value, StrideExprRef>;

// Compute the per-iteration delta of value `v` within `forOp`.
// Returns a loop-invariant symbolic delta, or null if unknown.  Pure.
static StrideExprRef computeDelta(Value v, scf::ForOp forOp,
                                  DeltaCache &cache) {
  // IV: delta = step
  if (v == forOp.getInductionVar())
    return makeLeaf(forOp.getStep());

  // Constant or loop-invariant: delta = 0
  if (forOp.isDefinedOutsideOfLoop(v))
    return makeConst(0);

  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;
  auto record = [&](StrideExprRef r) {
    cache[v] = r;
    return r;
  };

  // Block argument from iter_args: check yield = arg + c
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value yieldVal = yieldOp.getOperand(idx);
      if (auto addOp = yieldVal.getDefiningOp<arith::AddIOp>()) {
        Value other;
        if (addOp.getLhs() == blockArg)
          other = addOp.getRhs();
        else if (addOp.getRhs() == blockArg)
          other = addOp.getLhs();
        if (other && forOp.isDefinedOutsideOfLoop(other))
          return record(makeLeaf(other));
      }
      return record(nullptr);
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return record(nullptr);

  // arith.addi(a, b): delta = delta(a) + delta(b)
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto da = computeDelta(addOp.getLhs(), forOp, cache);
    auto db = computeDelta(addOp.getRhs(), forOp, cache);
    if (!da || !db)
      return record(nullptr);
    return record(makeAdd(da, db));
  }

  // arith.subi(a, b): delta = delta(a) - delta(b)
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto da = computeDelta(subOp.getLhs(), forOp, cache);
    auto db = computeDelta(subOp.getRhs(), forOp, cache);
    if (!da || !db)
      return record(nullptr);
    return record(makeSub(da, db));
  }

  // arith.muli(a, b) where one is loop-invariant:
  //   delta = invariant * delta(other)
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
    for (auto [invariant, variant] :
         {std::pair{rhs, lhs}, std::pair{lhs, rhs}}) {
      if (forOp.isDefinedOutsideOfLoop(invariant)) {
        auto dv = computeDelta(variant, forOp, cache);
        if (!dv)
          continue;
        return record(makeMul(makeLeaf(invariant), dv));
      }
    }
    return record(nullptr);
  }

  // arith.index_castui / arith.index_cast: delta = delta(input)
  if (auto castOp = dyn_cast<arith::IndexCastUIOp>(defOp))
    return record(computeDelta(castOp.getIn(), forOp, cache));
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return record(computeDelta(castOp.getIn(), forOp, cache));

  return record(nullptr);
}

// Get the per-iteration stride of `v`: tries accumulator analysis first
// (for iter_arg-derived values with possibly loop-varying increment),
// falls back to delta analysis (for IV-derived values, loop-invariant result).
// Returns null if the stride cannot be determined.
static StrideExprRef getStride(Value v, scf::ForOp forOp, DeltaCache &cache) {
  StrideResult r = getIterArgIncrement(v, forOp);
  if (r.status == StrideStatus::Ok)
    return r.expr;
  if (r.status == StrideStatus::Failed)
    return nullptr;
  return computeDelta(v, forOp, cache);
}

//===----------------------------------------------------------------------===//
// Rewrite: create new ForOp with additional iter_arg
//===----------------------------------------------------------------------===//

// Compute the value of `v` at the first iteration (IV = lower bound) by
// cloning the def-chain with IV replaced by the lower bound.  Returns nullptr
// if `v` cannot be materialized outside the loop.
static Value materializeAtLoopEntry(Value v, scf::ForOp forOp,
                                    OpBuilder &builder) {
  // IV → lower bound
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();

  // Already defined outside the loop — use directly.
  if (forOp.isDefinedOutsideOfLoop(v))
    return v;

  // iter_arg → its init value
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      return forOp.getInitArgs()[idx];
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return nullptr;

  // Cloning duplicates the op, so it must be safe to execute an extra time and
  // its result must not depend on anything but its operands.
  if (!isPure(defOp))
    return nullptr;

  // Clone the defining op with operands materialized at loop entry.
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    Value materialized = materializeAtLoopEntry(operand, forOp, builder);
    if (!materialized)
      return nullptr;
    newOperands.push_back(materialized);
  }
  builder.setInsertionPoint(forOp);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
  // Preserve which result was asked for; `v` need not be result 0.
  return cloned->getResult(cast<OpResult>(v).getResultNumber());
}

// Compute the initial pointer, i.e. the address the op reaches on the first
// iteration: base_at_iter0 + strideOperand_at_iter0 restated in addptr units.
//
// This is the mirror of the rescaling in `scaleBaseDelta`.  The initial offset
// is W * strideOperand_0 bytes, and `pto.addptr` takes elements, so the offset
// it needs is (W/E) * strideOperand_0.  For Element-class ops W == E and the
// operand is used as-is; for Block-class ops the factor W/E is exact (E always
// divides 32); for Byte-class ops the factor is a division that can fail.
static Value computeInitialPtr(Value base, Value strideOperand,
                               int64_t elemBytes, int64_t unitBytes,
                               scf::ForOp forOp, OpBuilder &builder) {
  Value baseAtEntry = materializeAtLoopEntry(base, forOp, builder);
  if (!baseAtEntry)
    return nullptr;

  if (!strideOperand)
    return baseAtEntry;

  Value soAtEntry = materializeAtLoopEntry(strideOperand, forOp, builder);
  if (!soAtEntry)
    return nullptr;

  auto constSo = getConstantIntValue(soAtEntry);
  if (constSo && *constSo == 0)
    return baseAtEntry;

  builder.setInsertionPoint(forOp);
  Value scaledOffset = soAtEntry;
  if (unitBytes != elemBytes) {
    if (unitBytes % elemBytes == 0) {
      Value soIndex = soAtEntry;
      if (soAtEntry.getType() != builder.getIndexType())
        soIndex = builder.create<arith::IndexCastUIOp>(
            forOp.getLoc(), builder.getIndexType(), soAtEntry);
      Value factor = builder.create<arith::ConstantIndexOp>(
          forOp.getLoc(), unitBytes / elemBytes);
      scaledOffset =
          builder.create<arith::MulIOp>(forOp.getLoc(), soIndex, factor);
    } else if (elemBytes % unitBytes == 0) {
      // Finer stride unit than the pointer element (a raw byte offset on a
      // wider element type): only representable when it lands on an element
      // boundary, which we can only prove for a compile-time constant.
      int64_t divisor = elemBytes / unitBytes;
      if (!constSo || *constSo % divisor != 0)
        return nullptr;
      scaledOffset = builder.create<arith::ConstantIndexOp>(
          forOp.getLoc(), *constSo / divisor);
    } else {
      return nullptr;
    }
  }
  return builder.create<pto::AddPtrOp>(forOp.getLoc(), baseAtEntry,
                                       scaledOffset);
}

// Rescale a per-iteration base delta from `pto.addptr` units (elements) into
// the op's strideOperand units.  Returns null when the conversion is not exact.
//
// Expanding the byte-denominated form
//     stride_new = (E*delta(base) + W*delta(strideOperand)) / W
//                = (E/W)*delta(base) + delta(strideOperand)
// shows that only delta(base) is ever rescaled; delta(strideOperand) passes
// through untouched.  That matters: it keeps the stride symbolic, so a
// loop-varying increment stays supported for every op class.  When E == W the
// factor is 1 and nothing is emitted at all, so Element-class ops (vlds/vsts)
// behave exactly as before this scaling existed.
static StrideExprRef scaleBaseDelta(StrideExprRef deltaBase, int64_t elemBytes,
                                    int64_t unitBytes) {
  if (unitBytes == elemBytes)
    return deltaBase;

  if (unitBytes % elemBytes == 0) {
    // Coarser stride unit (e.g. 32-byte blocks over 4-byte elements): the base
    // delta must be a compile-time constant that lands on a whole unit.
    int64_t divisor = unitBytes / elemBytes;
    auto constBase = foldConst(deltaBase);
    if (!constBase || *constBase % divisor != 0)
      return nullptr;
    return makeConst(*constBase / divisor);
  }

  if (elemBytes % unitBytes == 0)
    return makeMul(deltaBase, makeConst(elemBytes / unitBytes));

  return nullptr;
}

// Combine per-operand strides into the final stride_new for the post-update op.
// stride_new = (E/W) * deltaBase + deltaOffset, where E is the byte size of one
// addptr unit and W the byte size of one strideOperand unit.  Returns null if
// the stride is zero or the rescaling is inexact.  Purely symbolic: a rejected
// candidate leaves no IR behind, and the scaling is folded rather than emitted,
// so no `index` factor is ever multiplied against a narrower stride operand.
static StrideExprRef combineStride(StrideExprRef deltaBase,
                                   StrideExprRef deltaOffset, int64_t elemBytes,
                                   int64_t unitBytes) {
  StrideExprRef scaledBase =
      scaleBaseDelta(deltaBase, elemBytes, unitBytes);
  if (!scaledBase)
    return nullptr;

  StrideExprRef total = makeAdd(scaledBase, deltaOffset);
  if (auto constTotal = foldConst(total); constTotal && *constTotal == 0)
    return nullptr;
  return total;
}

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

// Can `v` be made available immediately before `insertPt`, cloning a pure
// def-chain if needed?  Pure: inspects only, never mutates the IR.
static bool canHoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                           DenseMap<Value, bool> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v))
    return true;
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return true;
  if (defOp->getBlock() != insertPt->getBlock())
    return false;
  // Already earlier in the block, so usable as-is: SSA guarantees its operands
  // are defined even earlier.  This reasoning is only sound because analysis
  // creates no IR — every value examined here predates the transform.
  if (defOp->isBeforeInBlock(insertPt))
    return true;
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;
  memo[v] = false;
  if (!isPure(defOp))
    return false;
  for (Value operand : defOp->getOperands())
    if (!canHoistBefore(operand, insertPt, forOp, memo))
      return false;
  memo[v] = true;
  return true;
}

// Clone `v`'s def-chain before `insertPt` as needed.  Only valid after
// canHoistBefore has approved `v`.
static Value hoistBefore(Value v, Operation *insertPt, scf::ForOp forOp,
                         OpBuilder &builder, DenseMap<Value, Value> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v) || isa<BlockArgument>(v))
    return v;
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getBlock() != insertPt->getBlock() ||
      defOp->isBeforeInBlock(insertPt))
    return v;
  auto it = memo.find(v);
  if (it != memo.end())
    return it->second;

  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands())
    newOperands.push_back(hoistBefore(operand, insertPt, forOp, builder, memo));

  builder.setInsertionPoint(insertPt);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
  // Preserve which result was asked for; `v` need not be result 0.
  Value res = cloned->getResult(cast<OpResult>(v).getResultNumber());
  memo[v] = res;
  return res;
}

// Rewrite `e` so that all of its leaves are available at `insertPt`.
static StrideExprRef makeAvailableAt(const StrideExprRef &e,
                                     Operation *insertPt, scf::ForOp forOp,
                                     OpBuilder &builder,
                                     DenseMap<Value, Value> &memo) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return e;
  case StrideExpr::Kind::Leaf: {
    Value hv = hoistBefore(e->leaf, insertPt, forOp, builder, memo);
    return hv == e->leaf ? e : makeLeaf(hv);
  }
  case StrideExpr::Kind::Cast:
    return makeCast(e->castOp,
                    makeAvailableAt(e->lhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Add:
    return makeAdd(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Sub:
    return makeSub(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  case StrideExpr::Kind::Mul:
    return makeMul(makeAvailableAt(e->lhs, insertPt, forOp, builder, memo),
                   makeAvailableAt(e->rhs, insertPt, forOp, builder, memo));
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

// Constants are loop-invariant, so they are always emitted before the loop and
// shared across every candidate in it.  Sharing matters beyond tidiness: the
// rewrite groups ops by (base, stride) Value identity, so two candidates with
// the same numeric stride must end up with the *same* Value to share an
// iter_arg.
using ConstCache = DenseMap<std::pair<int64_t, Type>, Value>;

static Value materializeConst(int64_t c, Type ty, Location loc,
                              scf::ForOp forOp, ConstCache &cache,
                              OpBuilder &builder) {
  if (!ty)
    ty = builder.getIndexType();
  auto key = std::make_pair(c, ty);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Value v;
  if (ty.isIndex())
    v = builder.create<arith::ConstantIndexOp>(loc, c);
  else
    v = builder.create<arith::ConstantIntOp>(loc, c,
                                             ty.getIntOrFloatBitWidth());
  cache[key] = v;
  return v;
}

// Check that every constant in `e` is representable in the type it would be
// materialized as.  `stride_new` for block-stride ops is a narrow integer
// (i16), and a stride that does not fit would otherwise become an out-of-range
// arith.constant.  Pure, so an out-of-range candidate is rejected before any
// IR is created.
static bool constantsFitType(const StrideExprRef &e, Type wantType) {
  switch (e->kind) {
  case StrideExpr::Kind::Const: {
    if (!wantType || wantType.isIndex())
      return true; // index is 64-bit, always holds an int64_t
    unsigned bitWidth = wantType.getIntOrFloatBitWidth();
    return bitWidth >= 64 || llvm::isIntN(bitWidth, e->constant);
  }
  case StrideExpr::Kind::Leaf:
    return true;
  case StrideExpr::Kind::Cast:
    return constantsFitType(e->lhs, e->castOp->getOperand(0).getType());
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul:
    return constantsFitType(e->lhs, wantType) &&
           constantsFitType(e->rhs, wantType);
  }
  return false;
}

// Emit `e` at the builder's current insertion point.  Sub-expressions are
// emitted bottom-up, so every operand is created before its user and the
// result dominates the insertion point by construction.
static Value materialize(const StrideExprRef &e, Type wantType, Location loc,
                         scf::ForOp forOp, ConstCache &cache,
                         OpBuilder &builder) {
  switch (e->kind) {
  case StrideExpr::Kind::Const:
    return materializeConst(e->constant, wantType, loc, forOp, cache, builder);
  case StrideExpr::Kind::Leaf:
    return e->leaf;
  case StrideExpr::Kind::Cast: {
    Value in = materialize(e->lhs, e->castOp->getOperand(0).getType(), loc,
                           forOp, cache, builder);
    Operation *cloned = builder.clone(*e->castOp);
    cloned->setOperand(0, in);
    return cloned->getResult(0);
  }
  case StrideExpr::Kind::Add:
  case StrideExpr::Kind::Sub:
  case StrideExpr::Kind::Mul: {
    Value a = materialize(e->lhs, wantType, loc, forOp, cache, builder);
    Value b = materialize(e->rhs, a.getType(), loc, forOp, cache, builder);
    if (e->kind == StrideExpr::Kind::Add)
      return builder.create<arith::AddIOp>(loc, a, b);
    if (e->kind == StrideExpr::Kind::Sub)
      return builder.create<arith::SubIOp>(loc, a, b);
    return builder.create<arith::MulIOp>(loc, a, b);
  }
  }
  llvm_unreachable("unhandled StrideExpr kind");
}

// Information about a post-update transformation to apply.
struct PostUpdateRewrite {
  Operation *op;
  Value base;
  Value strideOperand; // original offset / repeat_stride operand
  Value stride;        // stride value (stride_new for block-stride ops)
  Value initPtr;       // base + strideOperand_at_iter0, in addptr units
};

// A unique key for grouping rewrites that can share an iter_arg.
//
// Two ops may share an iter_arg only if they walk the same address sequence,
// i.e. they start at the same address and advance by the same stride.  The
// start address is `initPtr`, which is derived from base *and* strideOperand,
// so strideOperand has to be part of the key: same base and same stride but
// different offsets (e.g. %ub[%iv] and %ub[%iv + 64]) are distinct sequences,
// and merging them would make the second op start at the first one's address.
//
// Keying on the original operands rather than on `initPtr` itself keeps the
// comparison by Value identity meaningful: computeInitialPtr may materialize a
// fresh pto.addptr per candidate, so equal start addresses do not necessarily
// share a Value.  This is conservative — it can split groups that could have
// been merged — but never merges groups that must stay apart.
using IterArgGroupKey = std::tuple<Value, Value, Value>;

static IterArgGroupKey getGroupKey(const PostUpdateRewrite &rw) {
  return {rw.base, rw.strideOperand, rw.stride};
}

// Apply post-update rewrites to a single scf.for.
// Returns the new ForOp if any rewrites were applied, null otherwise.
static scf::ForOp applyPostUpdateRewrites(
    scf::ForOp forOp, ArrayRef<PostUpdateRewrite> rewrites,
    OpBuilder &builder) {
  if (rewrites.empty())
    return nullptr;

  // Group rewrites by (base, stride). Ops in the same group share one iter_arg
  // and all use the pre-update pointer. Only one updated_base per group is
  // yielded. This avoids redundant iter_args for same-address ops (e.g. vlds
  // + vsts both accessing %base[%iv]).
  DenseMap<IterArgGroupKey, unsigned> groupToIdx; // group key -> iter_arg index
  SmallVector<unsigned> rwGroupIdx(rewrites.size()); // rewrite -> group index
  SmallVector<Value> groupInitPtrs; // initial pointer per group (base + offset_at_iter0)

  for (auto [i, rw] : llvm::enumerate(rewrites)) {
    auto key = getGroupKey(rw);
    auto [it, inserted] = groupToIdx.try_emplace(key, groupInitPtrs.size());
    if (inserted)
      groupInitPtrs.push_back(rw.initPtr);
    rwGroupIdx[i] = it->second;
  }

  unsigned numGroups = groupInitPtrs.size();

  // Build new init args: original + one new pointer per group.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
  for (Value ptr : groupInitPtrs)
    newInitArgs.push_back(ptr);

  unsigned origIterArgCount = forOp.getInitArgs().size();

  // Create new ForOp.
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());

  // Map old block args to new: IV + original iter_args.
  IRMapping mapping;
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < origIterArgCount; ++i)
    mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

  // Clone the body, tracking old->new op correspondence.
  DenseMap<Operation *, Operation *> opMapping;
  builder.setInsertionPointToStart(newBody);
  for (auto &op : oldBody->without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    opMapping[&op] = cloned;
  }

  // Apply rewrites. All ops in a group use the same pre-update pointer (block
  // arg). Track the last updated_base per group for yielding.
  SmallVector<Value> groupYieldPtrs(numGroups);
  for (unsigned g = 0; g < numGroups; ++g)
    groupYieldPtrs[g] = newBody->getArgument(origIterArgCount + 1 + g);

  for (auto [rwIdx, rw] : llvm::enumerate(rewrites)) {
    auto it = opMapping.find(rw.op);
    if (it == opMapping.end())
      continue;
    Operation *clonedOp = it->second;
    unsigned gIdx = rwGroupIdx[rwIdx];
    Value ptr = newBody->getArgument(origIterArgCount + 1 + gIdx);
    Value strideNew = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    const PostUpdateOpInfo *info = getPostUpdateInfo(clonedOp);
    if (!info)
      continue;

    // Build the post-update op generically: replace base and strideOperand,
    // keep all other operands, append updated_base to result types.
    OperationState state(clonedOp->getLoc(), clonedOp->getName());
    for (auto [i, operand] : llvm::enumerate(clonedOp->getOperands())) {
      if (static_cast<int>(i) == info->baseOperandIdx)
        state.addOperands(ptr);
      else if (static_cast<int>(i) == info->strideOperandIdx)
        state.addOperands(strideNew);
      else
        state.addOperands(operand);
    }
    for (Type t : clonedOp->getResultTypes())
      state.addTypes(t);
    state.addTypes(ptr.getType()); // updated_base (appended last)
    state.addAttributes(clonedOp->getAttrs());

    Operation *newOp = builder.create(state);

    // Replace old results with new and update the mapping so that later
    // yield construction via mapping.lookupOrDefault sees the new results
    // instead of dangling pointers to the erased clonedOp.
    for (unsigned r = 0; r < clonedOp->getNumResults(); ++r) {
      clonedOp->getResult(r).replaceAllUsesWith(newOp->getResult(r));
      mapping.map(rw.op->getResult(r), newOp->getResult(r));
    }

    // updated_base is the last result.
    groupYieldPtrs[gIdx] = newOp->getResult(newOp->getNumResults() - 1);
    clonedOp->erase();
  }

  // Build yield: original yields + one pointer per group.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYields;
  for (Value v : oldYield.getOperands())
    newYields.push_back(mapping.lookupOrDefault(v));
  for (Value ptr : groupYieldPtrs)
    newYields.push_back(ptr);

  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYields);

  // Replace original ForOp results (only the original ones).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  forOp.erase();
  return newForOp;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct VPTOSoftPostUpdatePass
    : public pto::impl::VPTOSoftPostUpdateBase<VPTOSoftPostUpdatePass> {
  using pto::impl::VPTOSoftPostUpdateBase<
      VPTOSoftPostUpdatePass>::VPTOSoftPostUpdateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    module.walk([&](pto::VecScopeOp vecscope) {
      processVecScope(vecscope, builder);
    });
  }

private:
  void processVecScope(pto::VecScopeOp vecscope, OpBuilder &builder) {
    // Collect scf.for ops inside this vecscope.  Operation::walk defaults to
    // post-order, so nested loops already come before the loops enclosing
    // them.
    SmallVector<scf::ForOp> forOps;
    vecscope.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    // Process inner-to-outer, i.e. in collection order.  The order is load
    // bearing: rewriting a loop erases it, which also destroys every loop
    // nested inside it.  Visiting an enclosing loop first would leave the
    // already-collected inner ForOp handles dangling.
    for (scf::ForOp forOp : forOps)
      processForOp(forOp, builder);
  }

  void processForOp(scf::ForOp forOp, OpBuilder &builder) {
    SmallVector<PostUpdateRewrite> rewrites;
    // Shared across all candidates in this loop so equal strides map to one
    // Value, which is what lets same-address ops share an iter_arg.
    ConstCache constCache;

    for (Operation &op : *forOp.getBody()) {
      const PostUpdateOpInfo *info = getPostUpdateInfo(&op);
      if (!info)
        continue;
      if (isAlreadyPostUpdate(&op, *info))
        continue;
      if (!isDirectlyInForBody(&op, forOp))
        continue;

      Value base, strideOperand;
      extractBaseAndStrideOperand(&op, *info, base, strideOperand);

      // Both address terms have to be expressed in the same currency before
      // they can be combined: delta(base) is measured in pto.addptr units
      // (elements), while the strideOperand is counted in whatever unit the
      // op's lowering expects.  Bail on pointers whose addptr unit we cannot
      // pin down rather than guess at the scale.
      std::optional<int64_t> elemBytes = addPtrUnitBytes(base);
      if (!elemBytes)
        continue;
      int64_t unitBytes = strideUnitBytes(info->strideUnit, *elemBytes);

      // Analyze each operand independently: accumulator (iter_arg) first,
      // delta (IV/affine) fallback. Both return a symbolic per-iteration
      // stride; no IR is created until the candidate is known to be viable.
      DeltaCache deltaCache;
      StrideExprRef deltaBase = getStride(base, forOp, deltaCache);
      StrideExprRef deltaOffset = getStride(strideOperand, forOp, deltaCache);

      if (!deltaBase || !deltaOffset)
        continue;

      StrideExprRef total =
          combineStride(deltaBase, deltaOffset, *elemBytes, unitBytes);
      if (!total)
        continue;

      // Reject expressions whose subterms demand conflicting types.
      Type strideType;
      if (!exprType(total, strideType))
        continue;
      if (!strideType)
        strideType = strideOperand.getType();

      // Reject strides whose constants do not fit the target operand type.
      if (!constantsFitType(total, strideType))
        continue;

      // A stride built only from loop-invariant leaves is materialized before
      // the loop; otherwise it goes immediately before the candidate op.
      SmallVector<Value> leaves;
      collectLeaves(total, leaves);
      bool allInvariant = llvm::all_of(
          leaves, [&](Value l) { return forOp.isDefinedOutsideOfLoop(l); });

      StrideExprRef finalExpr = total;
      if (!allInvariant) {
        // Every leaf must be usable at the candidate op.  Checked before any
        // IR is created, so a rejected candidate leaves nothing behind.
        DenseMap<Value, bool> canCache;
        if (!llvm::all_of(leaves, [&](Value l) {
              return canHoistBefore(l, &op, forOp, canCache);
            }))
          continue;
        DenseMap<Value, Value> hoistMemo;
        finalExpr = makeAvailableAt(total, &op, forOp, builder, hoistMemo);
      }

      builder.setInsertionPoint(allInvariant ? forOp.getOperation() : &op);
      Value strideNew = materialize(finalExpr, strideType, op.getLoc(), forOp,
                                    constCache, builder);

      Value initPtr = computeInitialPtr(base, strideOperand, *elemBytes,
                                        unitBytes, forOp, builder);
      if (!initPtr)
        continue;

      rewrites.push_back({&op, base, strideOperand, strideNew, initPtr});
    }

    if (!rewrites.empty())
      applyPostUpdateRewrites(forOp, rewrites, builder);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
