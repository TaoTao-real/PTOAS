// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOPLANVMIACCUMULATORPHASES
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {

constexpr StringLiteral kGroupAttr = "pto.vmi.accumulator.phase_group";
constexpr StringLiteral kPhaseAttr = "pto.vmi.accumulator.phase";
constexpr StringLiteral kStatusAttr = "pto.vmi.accumulator.phase_status";
constexpr StringLiteral kRejectAttr = "pto.vmi.accumulator.phase_reject_reason";
constexpr StringLiteral kRootAttr = "pto.vmi.accumulator.storage_root";
constexpr StringLiteral kIterationsAttr = "pto.vmi.accumulator.iterations";

struct PhaseAttempt {
  pto::TExpandsOp init;
  scf::ForOp loop;
  pto::TAddOp update;
  pto::TRowSumOp reduction;
  int64_t root = 0;
  StringRef rejection;
};

static bool isSelectedVMI(Operation *op, StringRef candidate) {
  auto impl = op->getAttrOfType<StringAttr>("pto.tilelib.impl");
  auto selected = op->getAttrOfType<StringAttr>("pto.tilelib.candidate");
  return impl && impl.getValue() == "vmi" && selected &&
         selected.getValue() == candidate;
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

static bool isConstantFloatZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto floatValue = dyn_cast<FloatAttr>(constant.getValue());
  return floatValue && floatValue.getValue().isZero();
}

static bool isOneVLAccumulator(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  if (!type || !type.getElementType().isF32())
    return false;
  return type.getShape() == ArrayRef<int64_t>({1, 64}) &&
         type.getValidShape() == ArrayRef<int64_t>({1, 64});
}

static bool isSyncOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.barrier" || name == "pto.barrier_all";
}

static void rememberReject(PhaseAttempt &attempt, StringRef reason) {
  if (attempt.rejection.empty())
    attempt.rejection = reason;
}

static PhaseAttempt analyzeInit(pto::TExpandsOp init) {
  PhaseAttempt attempt;
  attempt.init = init;

  if (!isConstantFloatZero(init.getScalar())) {
    rememberReject(attempt, "init_not_zero");
    return attempt;
  }
  if (!isOneVLAccumulator(init.getDst())) {
    rememberReject(attempt, "accumulator_not_full_1x64xf32");
    return attempt;
  }
  std::optional<int64_t> initRoot = getStaticStorageRoot(init.getDst());
  if (!initRoot) {
    rememberReject(attempt, "accumulator_storage_root_not_static");
    return attempt;
  }
  attempt.root = *initRoot;

  Block *phaseBlock = init->getBlock();
  bool afterInit = false;
  for (Operation &op : *phaseBlock) {
    if (&op == init.getOperation()) {
      afterInit = true;
      continue;
    }
    if (!afterInit)
      continue;
    auto loop = dyn_cast<scf::ForOp>(&op);
    if (!loop)
      continue;
    SmallVector<pto::TAddOp, 2> updates;
    loop.walk([&](pto::TAddOp update) {
      if (isSelectedVMI(update, "vmi_tadd_block64"))
        updates.push_back(update);
    });
    if (!updates.empty()) {
      attempt.loop = loop;
      if (updates.size() == 1)
        attempt.update = updates.front();
      else
        rememberReject(attempt, "multiple_accumulator_updates");
      break;
    }
  }
  if (!attempt.loop) {
    rememberReject(attempt, "chunk_loop_not_found");
    return attempt;
  }
  if (!attempt.update)
    return attempt;

  auto lower = getConstantInt(attempt.loop.getLowerBound());
  auto upper = getConstantInt(attempt.loop.getUpperBound());
  auto step = getConstantInt(attempt.loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *upper != 4096 ||
      *step != 64) {
    rememberReject(attempt, "chunk_iteration_not_static_0_4096_64");
    return attempt;
  }

  bool unsafeLoopOperation = false;
  attempt.loop.walk([&](Operation *op) {
    if (op == attempt.loop.getOperation())
      return;
    if (isSyncOperation(op) || isa<CallOpInterface>(op)) {
      unsafeLoopOperation = true;
      return;
    }
    if (isa<pto::OpPipeInterface>(op)) {
      auto impl = op->getAttrOfType<StringAttr>("pto.tilelib.impl");
      if (!impl || impl.getValue() != "vmi")
        unsafeLoopOperation = true;
    }
  });
  if (unsafeLoopOperation) {
    rememberReject(attempt, "chunk_loop_contains_sync_dma_call_or_fallback");
    return attempt;
  }

  if (!isOneVLAccumulator(attempt.update.getSrc0()) ||
      !isOneVLAccumulator(attempt.update.getDst())) {
    rememberReject(attempt, "accumulator_update_not_full_1x64xf32");
    return attempt;
  }
  auto readRoot = getStaticStorageRoot(attempt.update.getSrc0());
  auto writeRoot = getStaticStorageRoot(attempt.update.getDst());
  if (!readRoot || !writeRoot || *readRoot != attempt.root ||
      *writeRoot != attempt.root) {
    rememberReject(attempt, "accumulator_update_storage_root_mismatch");
    return attempt;
  }

  bool squareChainFound = false;
  auto squareRoot = getStaticStorageRoot(attempt.update.getSrc1());
  if (squareRoot) {
    attempt.loop.walk([&](pto::TMulOp multiply) {
      if (squareChainFound || !isSelectedVMI(multiply, "vmi_tmul") ||
          !multiply->isBeforeInBlock(attempt.update))
        return;
      auto multiplyRoot = getStaticStorageRoot(multiply.getDst());
      if (multiplyRoot && *multiplyRoot == *squareRoot)
        squareChainFound = true;
    });
  }
  if (!squareChainFound) {
    rememberReject(attempt, "selected_square_update_chain_not_proven");
    return attempt;
  }

  bool afterLoop = false;
  for (Operation &op : *phaseBlock) {
    if (&op == attempt.loop.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (!afterLoop)
      continue;
    if (auto reduction = dyn_cast<pto::TRowSumOp>(&op)) {
      attempt.reduction = reduction;
      break;
    }
  }
  if (!attempt.reduction || !isSelectedVMI(attempt.reduction, "vmi_trowsum")) {
    rememberReject(attempt, "selected_final_trowsum_not_found");
    return attempt;
  }
  if (!isOneVLAccumulator(attempt.reduction.getSrc())) {
    rememberReject(attempt, "final_reduction_source_not_full_1x64xf32");
    return attempt;
  }
  auto reductionRoot = getStaticStorageRoot(attempt.reduction.getSrc());
  if (!reductionRoot || *reductionRoot != attempt.root) {
    rememberReject(attempt, "final_reduction_storage_root_mismatch");
    return attempt;
  }

  constexpr int64_t accumulatorBytes = 64 * 4;
  SmallVector<pto::AllocTileOp, 16> allocations;
  for (Operation &top : *phaseBlock) {
    top.walk([&](pto::AllocTileOp alloc) { allocations.push_back(alloc); });
  }
  for (pto::AllocTileOp alias : allocations) {
    auto addressRange = evaluateNonNegativeInterval(alias.getAddr());
    auto byteSize = getStaticTileBytes(alias.getResult());
    if (!addressRange || !byteSize) {
      rememberReject(attempt, "accumulator_alias_range_not_proven");
      return attempt;
    }
    bool mayOverlap = addressRange->lower < attempt.root + accumulatorBytes &&
                      addressRange->upper + *byteSize > attempt.root;
    if (!mayOverlap)
      continue;
    if (addressRange->lower != attempt.root ||
        addressRange->upper != attempt.root) {
      rememberReject(attempt, "accumulator_partial_alias");
      return attempt;
    }
    if (!isOneVLAccumulator(alias.getResult())) {
      rememberReject(attempt, "accumulator_alias_byte_range_or_mask_mismatch");
      return attempt;
    }
    for (Operation *user : alias.getResult().getUsers()) {
      if (user == attempt.init.getOperation() ||
          user == attempt.update.getOperation() ||
          user == attempt.reduction.getOperation())
        continue;
      rememberReject(attempt, "accumulator_external_or_unmodeled_use");
      return attempt;
    }
  }

  return attempt;
}

static void setRejectAttrs(PhaseAttempt &attempt, MLIRContext *ctx) {
  if (attempt.rejection.empty())
    return;
  auto reason = StringAttr::get(ctx, attempt.rejection);
  for (Operation *op :
       {attempt.init.getOperation(), attempt.loop.getOperation(),
        attempt.update.getOperation(), attempt.reduction.getOperation()}) {
    if (!op)
      continue;
    op->setAttr(kStatusAttr, StringAttr::get(ctx, "rejected"));
    op->setAttr(kRejectAttr, reason);
  }
}

static void setAcceptedAttrs(PhaseAttempt &attempt, int64_t groupId,
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
  annotate(attempt.init, "init");
  annotate(attempt.loop, "chunk_loop");
  annotate(attempt.update, "update");
  annotate(attempt.reduction, "reduction");
  attempt.loop->setAttr(kIterationsAttr,
                        IntegerAttr::get(IntegerType::get(ctx, 64), 64));
}

struct PTOPlanVMIAccumulatorPhasesPass
    : public pto::impl::PTOPlanVMIAccumulatorPhasesBase<
          PTOPlanVMIAccumulatorPhasesPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    func.walk([](Operation *op) {
      for (StringRef attr : {kGroupAttr, kPhaseAttr, kStatusAttr, kRejectAttr,
                             kRootAttr, kIterationsAttr})
        op->removeAttr(attr);
    });

    SmallVector<pto::TExpandsOp, 4> candidates;
    func.walk([&](pto::TExpandsOp init) {
      if (isSelectedVMI(init, "vmi_texpands"))
        candidates.push_back(init);
    });

    int64_t nextGroupId = 0;
    for (pto::TExpandsOp init : candidates) {
      PhaseAttempt attempt = analyzeInit(init);
      if (!attempt.rejection.empty()) {
        setRejectAttrs(attempt, ctx);
        continue;
      }
      setAcceptedAttrs(attempt, nextGroupId++, ctx);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPlanVMIAccumulatorPhasesPass() {
  return std::make_unique<PTOPlanVMIAccumulatorPhasesPass>();
}
