// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>
#include <limits>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOPLANVMISCALARPHASES
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {

constexpr StringLiteral kAccumulatorPhaseAttr = "pto.vmi.accumulator.phase";
constexpr StringLiteral kAccumulatorStatusAttr =
    "pto.vmi.accumulator.phase_status";
constexpr StringLiteral kGroupAttr = "pto.vmi.scalar.phase_group";
constexpr StringLiteral kPhaseAttr = "pto.vmi.scalar.phase";
constexpr StringLiteral kStatusAttr = "pto.vmi.scalar.phase_status";
constexpr StringLiteral kRejectAttr = "pto.vmi.scalar.phase_reject_reason";
constexpr StringLiteral kRootAttr = "pto.vmi.scalar.storage_root";
constexpr StringLiteral kIterationsAttr = "pto.vmi.scalar.apply_iterations";

struct ScalarPhaseAttempt {
  pto::TRowSumOp reduction;
  pto::TMulSOp scale;
  pto::TAddSOp shift;
  pto::TSqrtOp rootOp;
  pto::AllocTileOp divisor;
  scf::ForOp applyLoop;
  pto::TRowExpandDivOp divide;
  int64_t root = 0;
  StringRef rejection;
};

static bool isSelectedVMI(Operation *op, StringRef candidate) {
  auto impl = op->getAttrOfType<StringAttr>("pto.tilelib.impl");
  auto selected = op->getAttrOfType<StringAttr>("pto.tilelib.candidate");
  return impl && impl.getValue() == "vmi" && selected &&
         selected.getValue() == candidate;
}

static bool hasAcceptedAccumulatorReduction(pto::TRowSumOp reduction) {
  auto phase = reduction->getAttrOfType<StringAttr>(kAccumulatorPhaseAttr);
  auto status = reduction->getAttrOfType<StringAttr>(kAccumulatorStatusAttr);
  return phase && phase.getValue() == "reduction" && status &&
         status.getValue() == "accepted";
}

static std::optional<int64_t> getConstantInt(Value value) {
  if (!value)
    return std::nullopt;
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

struct IntInterval {
  int64_t lower;
  int64_t upper;
};

static std::optional<IntInterval>
evaluateNonNegativeInterval(Value value, unsigned depth = 0) {
  if (!value || depth > 12)
    return std::nullopt;
  if (auto constant = getConstantInt(value))
    return IntInterval{*constant, *constant};
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return evaluateNonNegativeInterval(cast.getIn(), depth + 1);
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = evaluateNonNegativeInterval(add.getLhs(), depth + 1);
    auto rhs = evaluateNonNegativeInterval(add.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    return IntInterval{lhs->lower + rhs->lower, lhs->upper + rhs->upper};
  }
  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = evaluateNonNegativeInterval(multiply.getLhs(), depth + 1);
    auto rhs = evaluateNonNegativeInterval(multiply.getRhs(), depth + 1);
    if (!lhs || !rhs || lhs->lower < 0 || rhs->lower < 0)
      return std::nullopt;
    return IntInterval{lhs->lower * rhs->lower, lhs->upper * rhs->upper};
  }
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  auto loop = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!loop || blockArg != loop.getInductionVar())
    return std::nullopt;
  auto lower = getConstantInt(loop.getLowerBound());
  auto upper = getConstantInt(loop.getUpperBound());
  auto step = getConstantInt(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  return IntInterval{*lower, *upper - *step};
}

static std::optional<int64_t> getStaticTileBytes(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  if (!type || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim) || dim <= 0 ||
        elements > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    elements *= dim;
  }
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return elements * static_cast<int64_t>(bitWidth / 8);
}

static std::optional<int64_t> getStaticStorageRoot(Value tile) {
  auto alloc = tile.getDefiningOp<pto::AllocTileOp>();
  if (!alloc)
    return std::nullopt;
  return getConstantInt(alloc.getAddr());
}

static bool hasLayout(pto::TileBufType type, pto::BLayout expected) {
  auto layout = dyn_cast<pto::BLayoutAttr>(type.getConfigAttr().getBLayout());
  return layout && layout.getValue() == expected;
}

static bool isCompactScalar(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  return type && type.getElementType().isF32() &&
         type.getShape() == ArrayRef<int64_t>({1, 8}) &&
         type.getValidShape() == ArrayRef<int64_t>({1, 1}) &&
         hasLayout(type, pto::BLayout::RowMajor);
}

static bool isCompactDivisor(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  return type && type.getElementType().isF32() &&
         type.getShape() == ArrayRef<int64_t>({8, 1}) &&
         type.getValidShape() == ArrayRef<int64_t>({1, 1}) &&
         hasLayout(type, pto::BLayout::ColMajor);
}

static bool isFullF32VL(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  return type && type.getElementType().isF32() &&
         type.getShape() == ArrayRef<int64_t>({1, 64}) &&
         type.getValidShape() == ArrayRef<int64_t>({1, 64}) &&
         hasLayout(type, pto::BLayout::RowMajor);
}

static bool isBF16VL(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  return type && type.getElementType().isBF16() &&
         type.getShape() == ArrayRef<int64_t>({1, 64}) &&
         type.getValidShape() == ArrayRef<int64_t>({1, 64}) &&
         hasLayout(type, pto::BLayout::RowMajor);
}

static bool isSyncOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.barrier" || name == "pto.barrier_all";
}

static void reject(ScalarPhaseAttempt &attempt, StringRef reason) {
  if (attempt.rejection.empty())
    attempt.rejection = reason;
}

static bool hasSameRoot(Value tile, int64_t root) {
  auto tileRoot = getStaticStorageRoot(tile);
  return tileRoot && *tileRoot == root;
}

static ScalarPhaseAttempt analyzeReduction(pto::TRowSumOp reduction) {
  ScalarPhaseAttempt attempt;
  attempt.reduction = reduction;

  if (!isSelectedVMI(reduction, "vmi_trowsum") ||
      !hasAcceptedAccumulatorReduction(reduction)) {
    reject(attempt, "reduction_not_accepted_selected_vmi_trowsum");
    return attempt;
  }
  if (!isCompactScalar(reduction.getDst())) {
    reject(attempt, "reduction_result_not_compact_1x1_f32");
    return attempt;
  }
  auto root = getStaticStorageRoot(reduction.getDst());
  if (!root) {
    reject(attempt, "scalar_storage_root_not_static");
    return attempt;
  }
  attempt.root = *root;

  Block *block = reduction->getBlock();
  bool afterReduction = false;
  unsigned scalarStage = 0;
  for (Operation &op : *block) {
    if (&op == reduction.getOperation()) {
      afterReduction = true;
      continue;
    }
    if (!afterReduction)
      continue;
    if (isSyncOperation(&op) || isa<CallOpInterface>(&op)) {
      reject(attempt, "scalar_phase_contains_sync_or_call");
      return attempt;
    }
    if (auto loop = dyn_cast<scf::ForOp>(&op)) {
      if (scalarStage != 3) {
        reject(attempt, "scalar_chain_not_exact_tmuls_tadds_tsqrt");
        return attempt;
      }
      attempt.applyLoop = loop;
      break;
    }
    if (!isa<pto::OpPipeInterface>(&op))
      continue;
    if (scalarStage == 0) {
      attempt.scale = dyn_cast<pto::TMulSOp>(&op);
      if (!attempt.scale || !isSelectedVMI(&op, "vmi_tmuls")) {
        reject(attempt, "scale_not_selected_vmi_tmuls");
        return attempt;
      }
    } else if (scalarStage == 1) {
      attempt.shift = dyn_cast<pto::TAddSOp>(&op);
      if (!attempt.shift || !isSelectedVMI(&op, "vmi_tadds")) {
        reject(attempt, "shift_not_selected_vmi_tadds");
        return attempt;
      }
    } else if (scalarStage == 2) {
      attempt.rootOp = dyn_cast<pto::TSqrtOp>(&op);
      if (!attempt.rootOp || !isSelectedVMI(&op, "vmi_tsqrt") ||
          attempt.rootOp.getPrecisionType() != pto::SqrtPrecision::Default) {
        reject(attempt, "sqrt_not_selected_default_vmi_tsqrt");
        return attempt;
      }
    } else {
      reject(attempt, "scalar_chain_has_extra_tileop");
      return attempt;
    }
    ++scalarStage;
  }

  if (scalarStage != 3 || !attempt.applyLoop) {
    reject(attempt, "scalar_chain_or_apply_loop_not_found");
    return attempt;
  }
  if (!isCompactScalar(attempt.scale.getSrc0()) ||
      !isCompactScalar(attempt.scale.getDst()) ||
      !isCompactScalar(attempt.shift.getSrc()) ||
      !isCompactScalar(attempt.shift.getDst()) ||
      !isCompactScalar(attempt.rootOp.getSrc()) ||
      !isCompactScalar(attempt.rootOp.getDst()) ||
      !hasSameRoot(attempt.scale.getSrc0(), attempt.root) ||
      !hasSameRoot(attempt.scale.getDst(), attempt.root) ||
      !hasSameRoot(attempt.shift.getSrc(), attempt.root) ||
      !hasSameRoot(attempt.shift.getDst(), attempt.root) ||
      !hasSameRoot(attempt.rootOp.getSrc(), attempt.root) ||
      !hasSameRoot(attempt.rootOp.getDst(), attempt.root)) {
    reject(attempt, "scalar_shape_dtype_or_storage_chain_mismatch");
    return attempt;
  }

  auto lower = getConstantInt(attempt.applyLoop.getLowerBound());
  auto upper = getConstantInt(attempt.applyLoop.getUpperBound());
  auto step = getConstantInt(attempt.applyLoop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *upper != 4096 ||
      *step != 64 || !attempt.applyLoop.getInitArgs().empty() ||
      !attempt.applyLoop.getResults().empty()) {
    reject(attempt, "apply_iteration_not_static_0_4096_64");
    return attempt;
  }

  for (Operation *cursor = attempt.rootOp->getNextNode();
       cursor && cursor != attempt.applyLoop; cursor = cursor->getNextNode()) {
    auto alloc = dyn_cast<pto::AllocTileOp>(cursor);
    if (!alloc || !isCompactDivisor(alloc.getResult()) ||
        !hasSameRoot(alloc.getResult(), attempt.root))
      continue;
    if (attempt.divisor) {
      reject(attempt, "multiple_compact_divisor_aliases");
      return attempt;
    }
    attempt.divisor = alloc;
  }
  if (!attempt.divisor) {
    reject(attempt, "compact_divisor_alias_not_found");
    return attempt;
  }

  SmallVector<Operation *, 8> tileOps;
  bool unsafeApplyOperation = false;
  attempt.applyLoop.walk([&](Operation *op) {
    if (op == attempt.applyLoop.getOperation())
      return;
    if (isSyncOperation(op) || isa<CallOpInterface>(op)) {
      unsafeApplyOperation = true;
      return;
    }
    if (isa<pto::OpPipeInterface>(op))
      tileOps.push_back(op);
  });
  if (unsafeApplyOperation) {
    reject(attempt, "apply_loop_contains_sync_or_call");
    return attempt;
  }
  constexpr StringLiteral expectedCandidates[] = {
      "vmi_tcvt", "vmi_tcvt", "vmi_trowexpanddiv", "vmi_tmul", "vmi_tcvt"};
  if (tileOps.size() != std::size(expectedCandidates)) {
    reject(attempt, "apply_tileop_sequence_not_exact");
    return attempt;
  }
  for (auto [op, candidate] : llvm::zip(tileOps, expectedCandidates)) {
    if (!isSelectedVMI(op, candidate)) {
      reject(attempt, "apply_tileop_not_selected_vmi");
      return attempt;
    }
  }

  attempt.divide = dyn_cast<pto::TRowExpandDivOp>(tileOps[2]);
  auto narrowing = dyn_cast<pto::TCvtOp>(tileOps[4]);
  if (!attempt.divide ||
      attempt.divide.getPrecisionType() != pto::DivPrecision::Default ||
      attempt.divide.getSrc1() != attempt.divisor.getResult() ||
      !isFullF32VL(attempt.divide.getSrc0()) ||
      !isFullF32VL(attempt.divide.getDst()) || !narrowing ||
      narrowing.getRmode() != pto::RoundMode::RINT ||
      !isFullF32VL(narrowing.getSrc()) || !isBF16VL(narrowing.getDst())) {
    reject(attempt, "divide_dtype_precision_or_rounding_contract_changed");
    return attempt;
  }

  constexpr int64_t scalarBytes = 8 * 4;
  llvm::DenseSet<Value> allowedAliases;
  SmallVector<Value, 5> modeledAliases{
      reduction.getDst(), attempt.scale.getDst(), attempt.shift.getDst(),
      attempt.rootOp.getDst(), attempt.divisor.getResult()};
  for (Value value : modeledAliases)
    allowedAliases.insert(value);
  SmallVector<pto::AllocTileOp, 16> allocations;
  block->getParentOp()->walk(
      [&](pto::AllocTileOp alloc) { allocations.push_back(alloc); });
  for (pto::AllocTileOp alias : allocations) {
    auto range = evaluateNonNegativeInterval(alias.getAddr());
    auto bytes = getStaticTileBytes(alias.getResult());
    if (!range || !bytes) {
      reject(attempt, "scalar_alias_range_not_proven");
      return attempt;
    }
    bool overlaps = range->lower < attempt.root + scalarBytes &&
                    range->upper + *bytes > attempt.root;
    if (!overlaps)
      continue;
    if (range->lower != attempt.root || range->upper != attempt.root) {
      reject(attempt, "scalar_partial_alias");
      return attempt;
    }
    if (!allowedAliases.contains(alias.getResult())) {
      reject(attempt, "scalar_unmodeled_alias");
      return attempt;
    }
  }

  auto usesOnly = [&](Value value, ArrayRef<Operation *> allowed) {
    return llvm::all_of(value.getUsers(), [&](Operation *user) {
      return llvm::is_contained(allowed, user);
    });
  };
  if (!usesOnly(reduction.getDst(),
                {reduction.getOperation(), attempt.scale.getOperation()}) ||
      !usesOnly(attempt.scale.getDst(),
                {attempt.scale.getOperation(), attempt.shift.getOperation()}) ||
      !usesOnly(attempt.shift.getDst(), {attempt.shift.getOperation(),
                                         attempt.rootOp.getOperation()}) ||
      !usesOnly(attempt.rootOp.getDst(), {attempt.rootOp.getOperation()}) ||
      !usesOnly(attempt.divisor.getResult(), {attempt.divide.getOperation()})) {
    reject(attempt, "scalar_or_divisor_external_use");
    return attempt;
  }

  return attempt;
}

static SmallVector<Operation *, 8> getMembers(ScalarPhaseAttempt &attempt) {
  return {attempt.reduction.getOperation(), attempt.scale.getOperation(),
          attempt.shift.getOperation(),     attempt.rootOp.getOperation(),
          attempt.divisor.getOperation(),   attempt.applyLoop.getOperation(),
          attempt.divide.getOperation()};
}

static void setRejectedAttrs(ScalarPhaseAttempt &attempt, MLIRContext *ctx) {
  if (attempt.rejection.empty())
    return;
  auto rejected = StringAttr::get(ctx, "rejected");
  auto reason = StringAttr::get(ctx, attempt.rejection);
  for (Operation *op : getMembers(attempt)) {
    if (!op)
      continue;
    op->setAttr(kStatusAttr, rejected);
    op->setAttr(kRejectAttr, reason);
  }
}

static void setAcceptedAttrs(ScalarPhaseAttempt &attempt, int64_t groupId,
                             MLIRContext *ctx) {
  auto group = IntegerAttr::get(IntegerType::get(ctx, 64), groupId);
  auto root = IntegerAttr::get(IntegerType::get(ctx, 64), attempt.root);
  auto accepted = StringAttr::get(ctx, "accepted");
  auto annotate = [&](Operation *op, StringRef phase) {
    op->setAttr(kGroupAttr, group);
    op->setAttr(kPhaseAttr, StringAttr::get(ctx, phase));
    op->setAttr(kStatusAttr, accepted);
    op->setAttr(kRootAttr, root);
  };
  annotate(attempt.reduction, "reduction");
  annotate(attempt.scale, "scale");
  annotate(attempt.shift, "shift");
  annotate(attempt.rootOp, "sqrt");
  annotate(attempt.divisor, "divisor");
  annotate(attempt.applyLoop, "apply_loop");
  annotate(attempt.divide, "divide");
  attempt.applyLoop->setAttr(kIterationsAttr,
                             IntegerAttr::get(IntegerType::get(ctx, 64), 64));
}

struct PTOPlanVMIScalarPhasesPass
    : public pto::impl::PTOPlanVMIScalarPhasesBase<PTOPlanVMIScalarPhasesPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    func.walk([](Operation *op) {
      for (StringRef attr : {kGroupAttr, kPhaseAttr, kStatusAttr, kRejectAttr,
                             kRootAttr, kIterationsAttr})
        op->removeAttr(attr);
    });

    SmallVector<pto::TRowSumOp, 4> reductions;
    func.walk([&](pto::TRowSumOp reduction) {
      if (isSelectedVMI(reduction, "vmi_trowsum") &&
          hasAcceptedAccumulatorReduction(reduction))
        reductions.push_back(reduction);
    });

    int64_t nextGroupId = 0;
    for (pto::TRowSumOp reduction : reductions) {
      ScalarPhaseAttempt attempt = analyzeReduction(reduction);
      if (!attempt.rejection.empty()) {
        setRejectedAttrs(attempt, ctx);
        continue;
      }
      setAcceptedAttrs(attempt, nextGroupId++, ctx);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPlanVMIScalarPhasesPass() {
  return std::make_unique<PTOPlanVMIScalarPhasesPass>();
}
