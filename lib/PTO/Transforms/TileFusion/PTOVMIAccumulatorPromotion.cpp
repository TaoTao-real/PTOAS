// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOVMIACCUMULATORPROMOTION
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {

constexpr StringLiteral kPhaseAttr = "pto.vmi.accumulator.phase";
constexpr StringLiteral kPhaseStatusAttr = "pto.vmi.accumulator.phase_status";
constexpr StringLiteral kRootAttr = "pto.vmi.accumulator.storage_root";
constexpr StringLiteral kPromotionRejectAttr =
    "pto.vmi.accumulator.promotion_reject_reason";
constexpr StringLiteral kPromotedAttr = "pto.vmi.accumulator.promoted";

struct PromotionCandidate {
  scf::ForOp phaseLoop;
  int64_t root = 0;
  pto::FusionRegionOp initRegion;
  pto::VMIVbrcOp initBroadcast;
  pto::VMIvStoreOp initStore;
  pto::FusionRegionOp updateRegion;
  scf::ForOp updateInnerLoop;
  pto::VMIvLoadOp accumulatorLoad;
  pto::VMIVaddOp updateAdd;
  pto::VMIvStoreOp accumulatorStore;
  pto::FusionRegionOp reductionRegion;
  scf::ForOp reductionInnerLoop;
  pto::VMIvLoadOp reductionLoad;
  StringRef rejection;
};

static bool hasCandidate(Operation *op, StringRef candidate) {
  auto selected = op->getAttrOfType<StringAttr>("pto.tilelib.candidate");
  auto impl = op->getAttrOfType<StringAttr>("pto.tilelib.impl");
  return selected && selected.getValue() == candidate && impl &&
         impl.getValue() == "vmi";
}

static bool regionHasCandidate(pto::FusionRegionOp region,
                               StringRef candidate) {
  bool found = false;
  region.walk([&](Operation *op) {
    if (hasCandidate(op, candidate))
      found = true;
  });
  return found;
}

static std::optional<int64_t> getConstantInt(Value value) {
  if (!value)
    return std::nullopt;
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

static bool isZeroIndex(Value value) {
  auto integer = getConstantInt(value);
  return integer && *integer == 0;
}

static std::optional<int64_t> getTileRoot(Value tile) {
  auto alloc = tile.getDefiningOp<pto::AllocTileOp>();
  if (!alloc)
    return std::nullopt;
  return getConstantInt(alloc.getAddr());
}

static std::optional<int64_t> getPointerRoot(Value pointer) {
  auto address = pointer.getDefiningOp<pto::TileBufAddrOp>();
  if (!address)
    return std::nullopt;
  return getTileRoot(address.getSrc());
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
  auto ownerLoop =
      dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!ownerLoop || blockArg != ownerLoop.getInductionVar())
    return std::nullopt;
  auto lower = getConstantInt(ownerLoop.getLowerBound());
  auto upper = getConstantInt(ownerLoop.getUpperBound());
  auto step = getConstantInt(ownerLoop.getStep());
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

static bool isOneVLF32(Type type) {
  auto vreg = dyn_cast<pto::VMIVRegType>(type);
  return vreg && vreg.getElementCount() == 64 &&
         vreg.getElementType().isF32() && !vreg.hasLayout();
}

static bool isFullMask(ValueRange values) {
  if (values.size() != 1)
    return false;
  Value value = values.front();
  auto mask = dyn_cast<pto::VMIMaskType>(value.getType());
  if (!mask || mask.getElementCount() != 64 || !mask.isPred() ||
      mask.hasLayout())
    return false;
  auto create = value.getDefiningOp<pto::VMICreateMaskOp>();
  auto activeLanes =
      create ? getConstantInt(create.getActiveLanes()) : std::nullopt;
  return activeLanes && *activeLanes == 64;
}

static bool isSingleIterationLoop(scf::ForOp loop) {
  auto lower = getConstantInt(loop.getLowerBound());
  auto upper = getConstantInt(loop.getUpperBound());
  auto step = getConstantInt(loop.getStep());
  return lower && upper && step && *lower == 0 && *upper == 1 && *step == 1 &&
         loop.getInitArgs().empty() && loop.getResults().empty();
}

static bool isSyncOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.barrier" || name == "pto.barrier_all";
}

static void reject(PromotionCandidate &candidate, StringRef reason) {
  if (candidate.rejection.empty())
    candidate.rejection = reason;
}

static PromotionCandidate analyzePhaseLoop(scf::ForOp loop) {
  PromotionCandidate candidate;
  candidate.phaseLoop = loop;
  auto root = loop->getAttrOfType<IntegerAttr>(kRootAttr);
  auto status = loop->getAttrOfType<StringAttr>(kPhaseStatusAttr);
  if (!root || !status || status.getValue() != "accepted") {
    reject(candidate, "missing_accepted_phase_provenance");
    return candidate;
  }
  candidate.root = root.getInt();
  if (!loop.getInitArgs().empty()) {
    reject(candidate, "chunk_loop_already_has_iter_args");
    return candidate;
  }
  auto lower = getConstantInt(loop.getLowerBound());
  auto upper = getConstantInt(loop.getUpperBound());
  auto step = getConstantInt(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *upper != 4096 ||
      *step != 64) {
    reject(candidate, "chunk_iteration_contract_changed");
    return candidate;
  }

  Block *phaseBlock = loop->getBlock();
  for (Operation *cursor = loop->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    auto region = dyn_cast<pto::FusionRegionOp>(cursor);
    if (!region)
      continue;
    SmallVector<pto::VMIVbrcOp, 2> broadcasts;
    SmallVector<pto::VMIvStoreOp, 2> stores;
    region.walk([&](pto::VMIVbrcOp op) {
      if (hasCandidate(op, "vmi_texpands"))
        broadcasts.push_back(op);
    });
    region.walk([&](pto::VMIvStoreOp op) {
      if (hasCandidate(op, "vmi_texpands"))
        stores.push_back(op);
    });
    if (!broadcasts.empty() || !stores.empty()) {
      candidate.initRegion = region;
      if (broadcasts.size() == 1)
        candidate.initBroadcast = broadcasts.front();
      if (stores.size() == 1)
        candidate.initStore = stores.front();
      break;
    }
  }
  if (!candidate.initRegion || !candidate.initBroadcast ||
      !candidate.initStore ||
      !isOneVLF32(candidate.initBroadcast.getResult().getType()) ||
      candidate.initStore.getValues().size() != 1 ||
      candidate.initStore.getValues().front() !=
          candidate.initBroadcast.getResult() ||
      getPointerRoot(candidate.initStore.getDestination()) != candidate.root ||
      !isZeroIndex(candidate.initStore.getOffset()) ||
      !isFullMask(candidate.initStore.getMask())) {
    reject(candidate, "init_vbrc_store_contract_changed");
    return candidate;
  }

  SmallVector<pto::VMIvLoadOp, 2> accumulatorLoads;
  SmallVector<pto::VMIvStoreOp, 2> accumulatorStores;
  loop.walk([&](pto::VMIvLoadOp op) {
    if (getPointerRoot(op.getSource()) == candidate.root)
      accumulatorLoads.push_back(op);
  });
  loop.walk([&](pto::VMIvStoreOp op) {
    if (getPointerRoot(op.getDestination()) == candidate.root)
      accumulatorStores.push_back(op);
  });
  if (accumulatorLoads.empty()) {
    reject(candidate, "update_accumulator_load_missing");
    return candidate;
  }
  if (accumulatorLoads.size() != 1) {
    reject(candidate, "update_accumulator_load_not_unique");
    return candidate;
  }
  if (accumulatorStores.empty()) {
    reject(candidate, "update_accumulator_store_missing");
    return candidate;
  }
  if (accumulatorStores.size() != 1) {
    reject(candidate, "update_accumulator_store_not_unique");
    return candidate;
  }
  candidate.accumulatorLoad = accumulatorLoads.front();
  candidate.accumulatorStore = accumulatorStores.front();
  if (candidate.accumulatorStore.getValues().size() == 1)
    candidate.updateAdd = candidate.accumulatorStore.getValues()
                              .front()
                              .getDefiningOp<pto::VMIVaddOp>();
  if (!candidate.accumulatorLoad || !candidate.updateAdd ||
      getPointerRoot(candidate.accumulatorStore.getDestination()) !=
          candidate.root ||
      !isZeroIndex(candidate.accumulatorLoad.getOffset()) ||
      !isZeroIndex(candidate.accumulatorStore.getOffset()) ||
      !isOneVLF32(candidate.accumulatorLoad.getResult(0).getType()) ||
      !isOneVLF32(candidate.updateAdd.getResult().getType()) ||
      !isFullMask(candidate.accumulatorStore.getMask()) ||
      !isFullMask(candidate.updateAdd.getMask()) ||
      candidate.accumulatorStore.getValues().size() != 1 ||
      candidate.accumulatorStore.getValues().front() !=
          candidate.updateAdd.getResult() ||
      (candidate.updateAdd.getLhs() != candidate.accumulatorLoad.getResult(0) &&
       candidate.updateAdd.getRhs() !=
           candidate.accumulatorLoad.getResult(0))) {
    reject(candidate, "update_address_mask_or_value_contract_changed");
    return candidate;
  }
  candidate.updateInnerLoop =
      candidate.updateAdd->getParentOfType<scf::ForOp>();
  candidate.updateRegion =
      candidate.updateAdd->getParentOfType<pto::FusionRegionOp>();
  if (!candidate.updateRegion ||
      !regionHasCandidate(candidate.updateRegion, "vmi_tadd_block64")) {
    reject(candidate, "update_single_iteration_scope_changed");
    return candidate;
  }
  if (candidate.updateInnerLoop == loop)
    candidate.updateInnerLoop = nullptr;
  if (candidate.updateInnerLoop &&
      (!isSingleIterationLoop(candidate.updateInnerLoop) ||
       candidate.accumulatorLoad->getParentOfType<scf::ForOp>() !=
           candidate.updateInnerLoop ||
       candidate.accumulatorStore->getParentOfType<scf::ForOp>() !=
           candidate.updateInnerLoop)) {
    reject(candidate, "update_single_iteration_scope_changed");
    return candidate;
  }

  bool afterLoop = false;
  for (Operation &op : *phaseBlock) {
    if (&op == loop.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (!afterLoop)
      continue;
    auto region = dyn_cast<pto::FusionRegionOp>(&op);
    if (!region)
      continue;
    if (!regionHasCandidate(region, "vmi_trowsum"))
      continue;
    SmallVector<pto::VMIvLoadOp, 2> loads;
    region.walk([&](pto::VMIvLoadOp load) {
      if (getPointerRoot(load.getSource()) == candidate.root)
        loads.push_back(load);
    });
    if (!loads.empty()) {
      candidate.reductionRegion = region;
      if (loads.size() == 1)
        candidate.reductionLoad = loads.front();
      break;
    }
  }
  if (!candidate.reductionRegion || !candidate.reductionLoad ||
      !isZeroIndex(candidate.reductionLoad.getOffset()) ||
      !isOneVLF32(candidate.reductionLoad.getResult(0).getType())) {
    reject(candidate, "final_reduction_load_contract_changed");
    return candidate;
  }
  candidate.reductionInnerLoop =
      candidate.reductionLoad->getParentOfType<scf::ForOp>();
  if (candidate.reductionInnerLoop &&
      candidate.reductionInnerLoop->getParentOfType<pto::FusionRegionOp>() !=
          candidate.reductionRegion)
    candidate.reductionInnerLoop = nullptr;
  if (candidate.reductionInnerLoop &&
      !isSingleIterationLoop(candidate.reductionInnerLoop)) {
    reject(candidate, "reduction_single_iteration_scope_changed");
    return candidate;
  }

  unsigned rootLoads = 0;
  unsigned rootStores = 0;
  bool unsafeControl = false;
  bool unknownAlias = false;
  bool partialAlias = false;
  constexpr int64_t accumulatorBytes = 64 * sizeof(float);
  for (Operation &top : *phaseBlock) {
    top.walk([&](Operation *op) {
      if (isSyncOperation(op) && loop->isProperAncestor(op))
        unsafeControl = true;
      if (isa<CallOpInterface>(op) && loop->isProperAncestor(op))
        unsafeControl = true;
      if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
        if (getPointerRoot(load.getSource()) == candidate.root)
          ++rootLoads;
      } else if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
        if (getPointerRoot(store.getDestination()) == candidate.root)
          ++rootStores;
      } else if (auto alloc = dyn_cast<pto::AllocTileOp>(op)) {
        auto addressRange = evaluateNonNegativeInterval(alloc.getAddr());
        auto byteSize = getStaticTileBytes(alloc.getResult());
        if (!addressRange || !byteSize) {
          unknownAlias = true;
          return;
        }
        if (addressRange->lower == candidate.root &&
            addressRange->upper == candidate.root) {
          if (*byteSize != accumulatorBytes)
            partialAlias = true;
          return;
        }
        bool overlaps =
            addressRange->lower < candidate.root + accumulatorBytes &&
            addressRange->upper + *byteSize > candidate.root;
        if (overlaps)
          partialAlias = true;
      }
    });
  }
  if (unsafeControl) {
    reject(candidate, "chunk_loop_contains_sync_or_call");
    return candidate;
  }
  if (unknownAlias) {
    reject(candidate, "accumulator_alias_not_proven_disjoint");
    return candidate;
  }
  if (partialAlias) {
    reject(candidate, "accumulator_partial_alias");
    return candidate;
  }
  if (rootLoads != 2 || rootStores != 2) {
    reject(candidate, "accumulator_memory_access_count_changed");
    return candidate;
  }

  return candidate;
}

static LogicalResult inlineSingleIterationLoop(scf::ForOp loop) {
  if (!isSingleIterationLoop(loop))
    return failure();
  Block *body = loop.getBody();
  loop.getInductionVar().replaceAllUsesWith(loop.getLowerBound());
  SmallVector<Operation *, 16> operations;
  for (Operation &op : body->without_terminator())
    operations.push_back(&op);
  for (Operation *op : operations)
    op->moveBefore(loop);
  body->getTerminator()->erase();
  loop.erase();
  return success();
}

static FailureOr<Value> appendFusionRegionOutput(pto::FusionRegionOp region,
                                                 Value output,
                                                 IRRewriter &rewriter) {
  auto oldYield =
      dyn_cast<pto::YieldOp>(region.getBody().front().getTerminator());
  if (!oldYield)
    return failure();

  SmallVector<Type, 4> resultTypes(region.getResultTypes().begin(),
                                   region.getResultTypes().end());
  resultTypes.push_back(output.getType());
  rewriter.setInsertionPoint(region);
  auto replacement = rewriter.create<pto::FusionRegionOp>(
      region.getLoc(), TypeRange(resultTypes));
  replacement->setAttrs(region->getAttrs());
  replacement.getBody().takeBody(region.getBody());

  SmallVector<Value, 4> yieldValues(oldYield.getValues().begin(),
                                    oldYield.getValues().end());
  yieldValues.push_back(output);
  rewriter.setInsertionPoint(oldYield);
  rewriter.create<pto::YieldOp>(oldYield.getLoc(), ValueRange(yieldValues));
  rewriter.eraseOp(oldYield);

  for (auto [oldResult, newResult] :
       llvm::zip(region.getOutputs(), replacement.getOutputs()))
    oldResult.replaceAllUsesWith(newResult);
  Value appended = replacement.getOutputs().back();
  rewriter.eraseOp(region);
  return appended;
}

static LogicalResult promote(PromotionCandidate &candidate,
                             IRRewriter &rewriter) {
  rewriter.setInsertionPoint(candidate.phaseLoop);
  Operation *clonedInit = rewriter.clone(*candidate.initBroadcast);
  Value initValue = clonedInit->getResult(0);

  if (candidate.updateInnerLoop &&
      failed(inlineSingleIterationLoop(candidate.updateInnerLoop)))
    return failure();
  if (candidate.reductionInnerLoop &&
      failed(inlineSingleIterationLoop(candidate.reductionInnerLoop)))
    return failure();

  FailureOr<Value> updateValue = appendFusionRegionOutput(
      candidate.updateRegion, candidate.updateAdd.getResult(), rewriter);
  if (failed(updateValue))
    return failure();
  FailureOr<LoopLikeOpInterface> replaced =
      candidate.phaseLoop.replaceWithAdditionalYields(
          rewriter, ValueRange{initValue},
          /*replaceInitOperandUsesInLoop=*/false,
          [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
            return SmallVector<Value>{*updateValue};
          });
  if (failed(replaced))
    return failure();
  auto newLoop = cast<scf::ForOp>((*replaced).getOperation());
  BlockArgument accumulator = newLoop.getRegionIterArgs().back();
  Value loopResult = newLoop.getResults().back();

  candidate.accumulatorLoad.getResult(0).replaceAllUsesWith(accumulator);
  candidate.reductionLoad.getResult(0).replaceAllUsesWith(loopResult);
  candidate.accumulatorLoad.erase();
  candidate.accumulatorStore.erase();
  candidate.reductionLoad.erase();
  candidate.initStore.erase();
  newLoop->setAttr(kPromotedAttr, UnitAttr::get(rewriter.getContext()));
  newLoop->setAttr(kPhaseStatusAttr,
                   StringAttr::get(rewriter.getContext(), "promoted"));
  return success();
}

struct PTOVMIAccumulatorPromotionPass
    : public pto::impl::PTOVMIAccumulatorPromotionBase<
          PTOVMIAccumulatorPromotionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<scf::ForOp, 4> phaseLoops;
    func.walk([&](scf::ForOp loop) {
      auto phase = loop->getAttrOfType<StringAttr>(kPhaseAttr);
      if (phase && phase.getValue() == "chunk_loop" &&
          !loop->hasAttr(kPromotedAttr))
        phaseLoops.push_back(loop);
      loop->removeAttr(kPromotionRejectAttr);
    });

    SmallVector<PromotionCandidate, 4> candidates;
    candidates.reserve(phaseLoops.size());
    for (scf::ForOp loop : phaseLoops)
      candidates.push_back(analyzePhaseLoop(loop));

    IRRewriter rewriter(&getContext());
    for (PromotionCandidate &candidate : candidates) {
      if (!candidate.rejection.empty()) {
        candidate.phaseLoop->setAttr(
            kPromotionRejectAttr,
            StringAttr::get(&getContext(), candidate.rejection));
        continue;
      }
      if (failed(promote(candidate, rewriter))) {
        candidate.phaseLoop.emitError(
            "failed after VMI accumulator promotion proof was accepted");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVMIAccumulatorPromotionPass() {
  return std::make_unique<PTOVMIAccumulatorPromotionPass>();
}
