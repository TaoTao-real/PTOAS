// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOVMISCALARPROMOTION
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {

constexpr StringLiteral kPhaseAttr = "pto.vmi.scalar.phase";
constexpr StringLiteral kStatusAttr = "pto.vmi.scalar.phase_status";
constexpr StringLiteral kRootAttr = "pto.vmi.scalar.storage_root";
constexpr StringLiteral kRejectAttr = "pto.vmi.scalar.promotion_reject_reason";
constexpr StringLiteral kPromotedAttr = "pto.vmi.scalar.promoted";

struct ScalarPromotionCandidate {
  scf::ForOp applyLoop;
  int64_t root = 0;
  pto::FusionRegionOp reductionRegion;
  pto::VMIvcaddOp reduction;
  pto::VMIvStoreOp reductionStore;
  pto::FusionRegionOp scalarRegion;
  pto::VMIvLoadOp scaleLoad;
  pto::VMIMulSOp scale;
  pto::VMIvStoreOp scaleStore;
  pto::VMIvLoadOp shiftLoad;
  pto::VMIAddSOp shift;
  pto::VMIvStoreOp shiftStore;
  pto::VMIvLoadOp sqrtLoad;
  pto::VMIVsqrtOp rootOp;
  pto::VMIvStoreOp rootStore;
  pto::VMIvLoadOp divisorLoad;
  pto::VMIVdivOp divide;
  StringRef rejection;
};

static void reject(ScalarPromotionCandidate &candidate, StringRef reason) {
  if (candidate.rejection.empty())
    candidate.rejection = reason;
}

static bool hasCandidate(Operation *op, StringRef candidate) {
  auto impl = op->getAttrOfType<StringAttr>("pto.tilelib.impl");
  auto selected = op->getAttrOfType<StringAttr>("pto.tilelib.candidate");
  return impl && impl.getValue() == "vmi" && selected &&
         selected.getValue() == candidate;
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

static std::optional<int64_t> getTileRoot(Value tile, unsigned depth = 0) {
  if (!tile || depth > 8)
    return std::nullopt;
  auto alloc = tile.getDefiningOp<pto::AllocTileOp>();
  if (alloc)
    return getConstantInt(alloc.getAddr());
  auto result = dyn_cast<OpResult>(tile);
  auto region = result ? dyn_cast<pto::FusionRegionOp>(result.getOwner())
                       : pto::FusionRegionOp{};
  if (!region)
    return std::nullopt;
  auto yield = dyn_cast<pto::YieldOp>(region.getBody().front().getTerminator());
  if (!yield || result.getResultNumber() >= yield.getValues().size())
    return std::nullopt;
  return getTileRoot(yield.getValues()[result.getResultNumber()], depth + 1);
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

static bool isVRegF32(Type type, int64_t lanes) {
  auto vreg = dyn_cast<pto::VMIVRegType>(type);
  return vreg && vreg.getElementCount() == lanes &&
         vreg.getElementType().isF32() && !vreg.hasLayout();
}

static bool isExactMask(ValueRange values, int64_t lanes) {
  if (values.size() != 1)
    return false;
  Value value = values.front();
  auto type = dyn_cast<pto::VMIMaskType>(value.getType());
  if (!type || type.getElementCount() != lanes || !type.isPred() ||
      type.hasLayout())
    return false;
  auto create = value.getDefiningOp<pto::VMICreateMaskOp>();
  auto active = create ? getConstantInt(create.getActiveLanes()) : std::nullopt;
  return active && *active == lanes;
}

static bool hasOnlyUsers(Value value, ArrayRef<Operation *> allowed) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return llvm::is_contained(allowed, user);
  });
}

static bool isSyncOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.set_flag" || name == "pto.wait_flag" ||
         name == "pto.barrier" || name == "pto.barrier_all";
}

static pto::VMIvLoadOp findLoadFor(Value value,
                                   ArrayRef<pto::VMIvLoadOp> loads) {
  for (pto::VMIvLoadOp load : loads) {
    if (load->getNumResults() == 1 && load.getResult(0) == value)
      return load;
  }
  return {};
}

static pto::VMIvStoreOp findStoreFor(Value value,
                                     ArrayRef<pto::VMIvStoreOp> stores) {
  for (pto::VMIvStoreOp store : stores) {
    if (store.getValues().size() == 1 && store.getValues().front() == value)
      return store;
  }
  return {};
}

static ScalarPromotionCandidate analyzeApplyLoop(scf::ForOp loop) {
  ScalarPromotionCandidate candidate;
  candidate.applyLoop = loop;
  auto root = loop->getAttrOfType<IntegerAttr>(kRootAttr);
  auto status = loop->getAttrOfType<StringAttr>(kStatusAttr);
  if (!root || !status || status.getValue() != "accepted") {
    reject(candidate, "missing_accepted_scalar_phase_provenance");
    return candidate;
  }
  candidate.root = root.getInt();
  if (!loop.getInitArgs().empty() || !loop.getResults().empty()) {
    reject(candidate, "apply_loop_already_has_iter_args");
    return candidate;
  }

  Block *block = loop->getBlock();
  for (Operation *cursor = loop->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    auto region = dyn_cast<pto::FusionRegionOp>(cursor);
    if (!region)
      continue;
    if (!candidate.scalarRegion && regionHasCandidate(region, "vmi_tmuls") &&
        regionHasCandidate(region, "vmi_tadds") &&
        regionHasCandidate(region, "vmi_tsqrt")) {
      candidate.scalarRegion = region;
    }
    if (!candidate.reductionRegion &&
        regionHasCandidate(region, "vmi_trowsum")) {
      candidate.reductionRegion = region;
      if (candidate.scalarRegion)
        break;
    }
  }
  if (!candidate.reductionRegion || !candidate.scalarRegion ||
      (candidate.reductionRegion != candidate.scalarRegion &&
       !candidate.reductionRegion->isBeforeInBlock(candidate.scalarRegion)) ||
      !candidate.scalarRegion->isBeforeInBlock(loop)) {
    reject(candidate, "reduction_or_scalar_region_not_found_in_order");
    return candidate;
  }

  SmallVector<pto::VMIvStoreOp, 2> reductionStores;
  SmallVector<pto::VMIvcaddOp, 2> reductions;
  candidate.reductionRegion.walk([&](pto::VMIvStoreOp store) {
    if (getPointerRoot(store.getDestination()) == candidate.root)
      reductionStores.push_back(store);
  });
  candidate.reductionRegion.walk(
      [&](pto::VMIvcaddOp op) { reductions.push_back(op); });
  if (reductions.size() != 1) {
    reject(candidate, "reduction_value_or_store_not_unique");
    return candidate;
  }
  candidate.reduction = reductions.front();
  SmallVector<pto::VMIvStoreOp, 2> matchingReductionStores;
  llvm::copy_if(reductionStores, std::back_inserter(matchingReductionStores),
                [&](pto::VMIvStoreOp store) {
                  return store.getValues().size() == 1 &&
                         store.getValues().front() ==
                             candidate.reduction.getResult();
                });
  if (matchingReductionStores.size() != 1) {
    reject(candidate, "reduction_value_or_store_not_unique");
    return candidate;
  }
  candidate.reductionStore = matchingReductionStores.front();
  if (!isVRegF32(candidate.reduction.getResult().getType(), 1) ||
      candidate.reductionStore.getValues().size() != 1 ||
      candidate.reductionStore.getValues().front() !=
          candidate.reduction.getResult() ||
      !isZeroIndex(candidate.reductionStore.getOffset()) ||
      !isExactMask(candidate.reductionStore.getMask(), 1) ||
      !hasOnlyUsers(candidate.reduction.getResult(),
                    {candidate.reductionStore.getOperation()})) {
    reject(candidate, "reduction_scalar_store_contract_changed");
    return candidate;
  }

  SmallVector<pto::VMIvLoadOp, 4> scalarLoads;
  SmallVector<pto::VMIvStoreOp, 4> scalarStores;
  SmallVector<pto::VMIMulSOp, 2> scales;
  SmallVector<pto::VMIAddSOp, 2> shifts;
  SmallVector<pto::VMIVsqrtOp, 2> roots;
  candidate.scalarRegion.walk([&](pto::VMIvLoadOp load) {
    if (getPointerRoot(load.getSource()) == candidate.root)
      scalarLoads.push_back(load);
  });
  candidate.scalarRegion.walk([&](pto::VMIvStoreOp store) {
    if (store != candidate.reductionStore &&
        getPointerRoot(store.getDestination()) == candidate.root)
      scalarStores.push_back(store);
  });
  candidate.scalarRegion.walk([&](pto::VMIMulSOp op) { scales.push_back(op); });
  candidate.scalarRegion.walk([&](pto::VMIAddSOp op) { shifts.push_back(op); });
  candidate.scalarRegion.walk([&](pto::VMIVsqrtOp op) { roots.push_back(op); });
  if (scalarLoads.size() != 3) {
    reject(candidate, "scalar_root_load_count_changed");
    return candidate;
  }
  if (scalarStores.size() != 3) {
    reject(candidate, "scalar_root_store_count_changed");
    return candidate;
  }
  if (scales.size() != 1 || shifts.size() != 1 || roots.size() != 1) {
    reject(candidate, "scalar_compute_count_changed");
    return candidate;
  }
  candidate.scale = scales.front();
  candidate.shift = shifts.front();
  candidate.rootOp = roots.front();
  candidate.scaleLoad = findLoadFor(candidate.scale.getSrc(), scalarLoads);
  candidate.scaleStore =
      findStoreFor(candidate.scale.getResult(), scalarStores);
  candidate.shiftLoad = findLoadFor(candidate.shift.getSrc(), scalarLoads);
  candidate.shiftStore =
      findStoreFor(candidate.shift.getResult(), scalarStores);
  candidate.sqrtLoad = findLoadFor(candidate.rootOp.getSource(), scalarLoads);
  candidate.rootStore =
      findStoreFor(candidate.rootOp.getResult(), scalarStores);
  if (!candidate.scaleLoad || !candidate.scaleStore || !candidate.shiftLoad ||
      !candidate.shiftStore || !candidate.sqrtLoad || !candidate.rootStore ||
      !candidate.scaleStore->isBeforeInBlock(candidate.shiftLoad) ||
      !candidate.shiftStore->isBeforeInBlock(candidate.sqrtLoad)) {
    reject(candidate, "scalar_compute_memory_chain_changed");
    return candidate;
  }
  for (pto::VMIvLoadOp load : scalarLoads) {
    if (!isZeroIndex(load.getOffset()) || load->getNumResults() != 1 ||
        !isVRegF32(load.getResult(0).getType(), 1)) {
      reject(candidate, "scalar_load_address_or_type_changed");
      return candidate;
    }
  }
  for (pto::VMIvStoreOp store : scalarStores) {
    if (!isZeroIndex(store.getOffset()) || store.getValues().size() != 1 ||
        !isVRegF32(store.getValues().front().getType(), 1) ||
        !isExactMask(store.getMask(), 1)) {
      reject(candidate, "scalar_store_address_mask_or_type_changed");
      return candidate;
    }
  }
  if (!isVRegF32(candidate.scale.getResult().getType(), 1) ||
      !isVRegF32(candidate.shift.getResult().getType(), 1) ||
      !isVRegF32(candidate.rootOp.getResult().getType(), 1) ||
      !isExactMask(candidate.scale.getMask(), 1) ||
      !isExactMask(candidate.shift.getMask(), 1) ||
      !isExactMask(candidate.rootOp.getMask(), 1) ||
      !hasOnlyUsers(candidate.scaleLoad.getResult(0),
                    {candidate.scale.getOperation()}) ||
      !hasOnlyUsers(candidate.scale.getResult(),
                    {candidate.scaleStore.getOperation()}) ||
      !hasOnlyUsers(candidate.shiftLoad.getResult(0),
                    {candidate.shift.getOperation()}) ||
      !hasOnlyUsers(candidate.shift.getResult(),
                    {candidate.shiftStore.getOperation()}) ||
      !hasOnlyUsers(candidate.sqrtLoad.getResult(0),
                    {candidate.rootOp.getOperation()}) ||
      !hasOnlyUsers(candidate.rootOp.getResult(),
                    {candidate.rootStore.getOperation()})) {
    reject(candidate, "scalar_mask_or_escape_contract_changed");
    return candidate;
  }

  SmallVector<pto::VMIvLoadOp, 2> divisorLoads;
  SmallVector<pto::VMIVdivOp, 4> divides;
  loop.walk([&](pto::VMIvLoadOp load) {
    if (getPointerRoot(load.getSource()) == candidate.root)
      divisorLoads.push_back(load);
  });
  loop.walk([&](pto::VMIVdivOp op) { divides.push_back(op); });
  if (divisorLoads.size() != 1) {
    reject(candidate, "divisor_load_or_divide_not_unique");
    return candidate;
  }
  candidate.divisorLoad = divisorLoads.front();
  SmallVector<pto::VMIVdivOp, 2> matchingDivides;
  llvm::copy_if(divides, std::back_inserter(matchingDivides),
                [&](pto::VMIVdivOp divide) {
                  return divide.getRhs() == candidate.divisorLoad.getResult(0);
                });
  if (matchingDivides.size() != 1) {
    reject(candidate, "divisor_load_or_divide_not_unique");
    return candidate;
  }
  candidate.divide = matchingDivides.front();
  if (!candidate.divisorLoad.getDistMode() ||
      candidate.divisorLoad.getDistMode() != "brc" ||
      !isZeroIndex(candidate.divisorLoad.getOffset()) ||
      candidate.divisorLoad->getNumResults() != 1 ||
      !isVRegF32(candidate.divisorLoad.getResult(0).getType(), 64) ||
      candidate.divide.getRhs() != candidate.divisorLoad.getResult(0) ||
      !hasOnlyUsers(candidate.divisorLoad.getResult(0),
                    {candidate.divide.getOperation()})) {
    reject(candidate, "divisor_broadcast_load_contract_changed");
    return candidate;
  }

  unsigned rootLoads = 0;
  unsigned rootStores = 0;
  bool unsafeControl = false;
  bool unknownAlias = false;
  bool partialAlias = false;
  constexpr int64_t scalarBytes = 8 * sizeof(float);
  block->getParentOp()->walk([&](Operation *op) {
    if ((isSyncOperation(op) || isa<CallOpInterface>(op)) &&
        (candidate.reductionRegion->isProperAncestor(op) ||
         candidate.scalarRegion->isProperAncestor(op) ||
         loop->isProperAncestor(op)))
      unsafeControl = true;
    if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      if (getPointerRoot(load.getSource()) == candidate.root)
        ++rootLoads;
    } else if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
      if (getPointerRoot(store.getDestination()) == candidate.root)
        ++rootStores;
    } else if (auto alloc = dyn_cast<pto::AllocTileOp>(op)) {
      auto range = evaluateNonNegativeInterval(alloc.getAddr());
      auto bytes = getStaticTileBytes(alloc.getResult());
      if (!range || !bytes) {
        unknownAlias = true;
        return;
      }
      bool overlaps = range->lower < candidate.root + scalarBytes &&
                      range->upper + *bytes > candidate.root;
      if (!overlaps)
        return;
      if (range->lower != candidate.root || range->upper != candidate.root ||
          *bytes != scalarBytes)
        partialAlias = true;
    }
  });
  if (unsafeControl) {
    reject(candidate, "scalar_phase_contains_sync_or_call_after_expansion");
    return candidate;
  }
  if (unknownAlias) {
    reject(candidate, "scalar_alias_not_proven_disjoint_after_expansion");
    return candidate;
  }
  if (partialAlias) {
    reject(candidate, "scalar_partial_alias_after_expansion");
    return candidate;
  }
  if (rootLoads != 4 || rootStores != 4) {
    reject(candidate, "scalar_memory_access_count_changed");
    return candidate;
  }

  return candidate;
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

static LogicalResult promote(ScalarPromotionCandidate &candidate,
                             IRRewriter &rewriter) {
  Value reachingReduction = candidate.reduction.getResult();
  if (candidate.reductionRegion != candidate.scalarRegion) {
    FailureOr<Value> reductionValue = appendFusionRegionOutput(
        candidate.reductionRegion, reachingReduction, rewriter);
    if (failed(reductionValue))
      return failure();
    reachingReduction = *reductionValue;
  }

  candidate.scaleLoad.getResult(0).replaceAllUsesWith(reachingReduction);
  candidate.shiftLoad.getResult(0).replaceAllUsesWith(
      candidate.scale.getResult());
  candidate.sqrtLoad.getResult(0).replaceAllUsesWith(
      candidate.shift.getResult());

  FailureOr<Value> rootValue = appendFusionRegionOutput(
      candidate.scalarRegion, candidate.rootOp.getResult(), rewriter);
  if (failed(rootValue))
    return failure();

  rewriter.setInsertionPoint(candidate.applyLoop);
  auto divisor = rewriter.create<pto::VMIVbrcOp>(
      candidate.applyLoop.getLoc(),
      candidate.divisorLoad.getResult(0).getType(), *rootValue, IntegerAttr{});
  candidate.divisorLoad.getResult(0).replaceAllUsesWith(divisor.getResult());

  candidate.reductionStore.erase();
  candidate.scaleStore.erase();
  candidate.shiftStore.erase();
  candidate.rootStore.erase();
  candidate.scaleLoad.erase();
  candidate.shiftLoad.erase();
  candidate.sqrtLoad.erase();
  candidate.divisorLoad.erase();

  candidate.applyLoop->setAttr(kPromotedAttr,
                               UnitAttr::get(rewriter.getContext()));
  candidate.applyLoop->setAttr(
      kStatusAttr, StringAttr::get(rewriter.getContext(), "promoted"));
  return success();
}

struct PTOVMIScalarPromotionPass
    : public pto::impl::PTOVMIScalarPromotionBase<PTOVMIScalarPromotionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<scf::ForOp, 4> applyLoops;
    func.walk([&](scf::ForOp loop) {
      auto phase = loop->getAttrOfType<StringAttr>(kPhaseAttr);
      if (phase && phase.getValue() == "apply_loop" &&
          !loop->hasAttr(kPromotedAttr))
        applyLoops.push_back(loop);
      loop->removeAttr(kRejectAttr);
    });

    SmallVector<ScalarPromotionCandidate, 4> candidates;
    candidates.reserve(applyLoops.size());
    for (scf::ForOp loop : applyLoops)
      candidates.push_back(analyzeApplyLoop(loop));

    IRRewriter rewriter(&getContext());
    for (ScalarPromotionCandidate &candidate : candidates) {
      if (!candidate.rejection.empty()) {
        candidate.applyLoop->setAttr(
            kRejectAttr, StringAttr::get(&getContext(), candidate.rejection));
        continue;
      }
      if (failed(promote(candidate, rewriter))) {
        candidate.applyLoop.emitError(
            "failed after VMI scalar promotion proof was accepted");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVMIScalarPromotionPass() {
  return std::make_unique<PTOVMIScalarPromotionPass>();
}
