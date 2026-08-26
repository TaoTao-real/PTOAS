// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIVectorPressure.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <limits>
#include <optional>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOVMISTATEPROMOTION
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir::pto

using namespace mlir;

namespace {

constexpr StringLiteral kStatusAttr = "pto.vmi.state_promotion.status";
constexpr StringLiteral kFlowAttr = "pto.vmi.state_promotion.flow";
constexpr StringLiteral kReasonAttr = "pto.vmi.state_promotion.reason";
constexpr StringLiteral kDetailAttr = "pto.vmi.state_promotion.detail";
constexpr StringLiteral kBeforePressureAttr =
    "pto.vmi.state_promotion.pressure_before_chunks";
constexpr StringLiteral kAfterPressureAttr =
    "pto.vmi.state_promotion.pressure_after_chunks";
constexpr StringLiteral kLegacyAccumulatorStatus =
    "pto.vmi.accumulator.phase_status";
constexpr StringLiteral kLegacyScalarStatus = "pto.vmi.scalar.phase_status";
constexpr StringLiteral kShadowAttr = "pto.vmi.state_promotion.shadow";

enum class RejectReason {
  Alias,
  Mask,
  Escape,
  ControlFlow,
  ResourcePressure,
  UnknownEffect,
  UnknownLocation,
  TypeLayout,
  MultipleDefinitions,
};

static StringRef stringify(RejectReason reason) {
  switch (reason) {
  case RejectReason::Alias:
    return "alias";
  case RejectReason::Mask:
    return "mask";
  case RejectReason::Escape:
    return "escape";
  case RejectReason::ControlFlow:
    return "control-flow";
  case RejectReason::ResourcePressure:
    return "resource-pressure";
  case RejectReason::UnknownEffect:
    return "unknown-effect";
  case RejectReason::UnknownLocation:
    return "unknown-location";
  case RejectReason::TypeLayout:
    return "type-layout";
  case RejectReason::MultipleDefinitions:
    return "multiple-reaching-definitions";
  }
  llvm_unreachable("unhandled VMI state-promotion rejection");
}

enum class StateFlow { StraightLine, LoopInvariant, LoopCarried };

static StringRef stringify(StateFlow flow) {
  switch (flow) {
  case StateFlow::StraightLine:
    return "straight-line";
  case StateFlow::LoopInvariant:
    return "loop-invariant";
  case StateFlow::LoopCarried:
    return "loop-carried";
  }
  llvm_unreachable("unhandled VMI state flow");
}

static void stripLegacyPhaseMetadata(func::FuncOp func) {
  func.walk([&](Operation *op) {
    SmallVector<StringAttr, 8> obsolete;
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name.starts_with("pto.vmi.accumulator.") ||
          name.starts_with("pto.vmi.scalar."))
        obsolete.push_back(attr.getName());
    }
    for (StringAttr name : obsolete)
      op->removeAttr(name);
  });
}

struct StateLocation {
  int64_t storageRoot = 0;
  int64_t byteOffset = 0;
  int64_t byteLength = 0;
  Type elementType;
  StringRef distribution = "continuous";

  bool sameBytes(const StateLocation &other) const {
    return storageRoot + byteOffset == other.storageRoot + other.byteOffset &&
           byteLength == other.byteLength && elementType == other.elementType;
  }

  bool overlaps(const StateLocation &other) const {
    int64_t begin = storageRoot + byteOffset;
    int64_t otherBegin = other.storageRoot + other.byteOffset;
    int64_t end = begin + byteLength;
    int64_t otherEnd = otherBegin + other.byteLength;
    return begin < otherEnd && otherBegin < end;
  }
};

struct AccessProof {
  std::optional<StateLocation> location;
  std::optional<RejectReason> rejection;
  StringRef detail;
  int64_t activeLanes = -1;
};

struct LoopCandidate {
  scf::ForOp loop;
  pto::VMIvLoadOp load;
  pto::VMIvStoreOp updateStore;
  pto::VMIvStoreOp initStore;
  SmallVector<pto::VMIvLoadOp, 4> finalLoads;
  StateLocation location;
  int64_t activeLanes = -1;
  int64_t pressureBefore = 0;
  int64_t pressureAfter = 0;
  std::optional<RejectReason> rejection;
  StringRef detail;
};

struct ForwardCandidate {
  pto::VMIvStoreOp store;
  pto::VMIvLoadOp load;
  StateLocation location;
  int64_t activeLanes = -1;
  bool broadcast = false;
  bool canDeleteStore = false;
};

static std::optional<int64_t> getConstantInt(Value value) {
  if (!value)
    return std::nullopt;
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

static std::optional<int64_t> getElementBytes(Type type) {
  if (!type || !type.isIntOrFloat())
    return std::nullopt;
  unsigned bits = type.getIntOrFloatBitWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return static_cast<int64_t>(bits / 8);
}

static std::optional<int64_t> getStaticTileBytes(Value tile) {
  auto type = dyn_cast<pto::TileBufType>(tile.getType());
  if (!type)
    return std::nullopt;
  std::optional<int64_t> elementBytes = getElementBytes(type.getElementType());
  if (!elementBytes)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim) || dim <= 0 ||
        elements > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    elements *= dim;
  }
  return elements * *elementBytes;
}

struct IntInterval {
  int64_t lower = 0;
  int64_t upper = 0;
};

static std::optional<IntInterval>
evaluateNonNegativeInterval(Value value, unsigned depth = 0) {
  if (!value || depth > 16)
    return std::nullopt;
  if (std::optional<int64_t> constant = getConstantInt(value))
    return IntInterval{*constant, *constant};
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return evaluateNonNegativeInterval(cast.getIn(), depth + 1);
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return evaluateNonNegativeInterval(ext.getIn(), depth + 1);
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return evaluateNonNegativeInterval(ext.getIn(), depth + 1);
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
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
  if (!loop || argument != loop.getInductionVar())
    return std::nullopt;
  auto lower = getConstantInt(loop.getLowerBound());
  auto upper = getConstantInt(loop.getUpperBound());
  auto step = getConstantInt(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  int64_t last = *lower + ((*upper - 1 - *lower) / *step) * *step;
  return IntInterval{*lower, last};
}

static std::optional<int64_t> getTileRoot(Value tile, unsigned depth = 0) {
  if (!tile || depth > 12)
    return std::nullopt;
  if (auto alloc = tile.getDefiningOp<pto::AllocTileOp>())
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

static std::optional<Value> getPointerTile(Value pointer) {
  auto address = pointer.getDefiningOp<pto::TileBufAddrOp>();
  if (!address)
    return std::nullopt;
  return address.getSrc();
}

static std::optional<int64_t> getPointerRoot(Value pointer) {
  std::optional<Value> tile = getPointerTile(pointer);
  return tile ? getTileRoot(*tile) : std::nullopt;
}

static std::optional<int64_t> getVRegBytes(Type type) {
  auto vreg = dyn_cast<pto::VMIVRegType>(type);
  if (!vreg || vreg.getElementCount() <= 0)
    return std::nullopt;
  std::optional<int64_t> elementBytes = getElementBytes(vreg.getElementType());
  if (!elementBytes)
    return std::nullopt;
  return vreg.getElementCount() * *elementBytes;
}

static int64_t getStaticActiveLanes(ValueRange masks, Type valueType) {
  auto vreg = dyn_cast<pto::VMIVRegType>(valueType);
  if (!vreg)
    return -1;
  if (masks.empty())
    return vreg.getElementCount();
  if (masks.size() != 1)
    return -1;
  auto maskType = dyn_cast<pto::VMIMaskType>(masks.front().getType());
  auto create = masks.front().getDefiningOp<pto::VMICreateMaskOp>();
  std::optional<int64_t> active =
      create ? getConstantInt(create.getActiveLanes()) : std::nullopt;
  if (!maskType || !active || *active <= 0 || *active > vreg.getElementCount())
    return -1;
  return *active;
}

static AccessProof analyzeLoad(pto::VMIvLoadOp load) {
  AccessProof proof;
  if (load.getResults().size() != 1 || load.getStride() ||
      load.getBlockStride() || load.getGroup()) {
    proof.rejection = RejectReason::UnknownLocation;
    proof.detail = "non-contiguous-or-multi-result-load";
    return proof;
  }
  StringRef dist = load.getDistMode().value_or("continuous");
  if (dist != "continuous" && dist != "brc") {
    proof.rejection = RejectReason::UnknownLocation;
    proof.detail = "unsupported-load-distribution";
    return proof;
  }
  std::optional<int64_t> root = getPointerRoot(load.getSource());
  std::optional<int64_t> offset = getConstantInt(load.getOffset());
  auto vreg = dyn_cast<pto::VMIVRegType>(load.getResult(0).getType());
  std::optional<int64_t> elementBytes =
      vreg ? getElementBytes(vreg.getElementType()) : std::nullopt;
  if (!root || !offset || !vreg || !elementBytes) {
    proof.rejection = RejectReason::UnknownLocation;
    proof.detail = "dynamic-root-offset-or-type";
    return proof;
  }
  std::optional<int64_t> bytes = getVRegBytes(vreg);
  if (!bytes) {
    proof.rejection = RejectReason::TypeLayout;
    proof.detail = "unsupported-vreg-type";
    return proof;
  }
  proof.location = StateLocation{*root, *offset * *elementBytes,
                                 dist == "brc" ? *elementBytes : *bytes,
                                 vreg.getElementType(), dist};
  proof.activeLanes = vreg.getElementCount();
  return proof;
}

static AccessProof analyzeStore(pto::VMIvStoreOp store) {
  AccessProof proof;
  if (store.getValues().size() != 1 || store.getStride() ||
      store.getBlockStride() || store.getGroup() ||
      store.getDistMode().value_or("continuous") != "continuous") {
    proof.rejection = RejectReason::UnknownLocation;
    proof.detail = "non-contiguous-or-multi-value-store";
    return proof;
  }
  Value value = store.getValues().front();
  auto vreg = dyn_cast<pto::VMIVRegType>(value.getType());
  std::optional<int64_t> root = getPointerRoot(store.getDestination());
  std::optional<int64_t> offset = getConstantInt(store.getOffset());
  std::optional<int64_t> elementBytes =
      vreg ? getElementBytes(vreg.getElementType()) : std::nullopt;
  std::optional<int64_t> bytes = vreg ? getVRegBytes(vreg) : std::nullopt;
  if (!root || !offset || !vreg || !elementBytes || !bytes) {
    proof.rejection = RejectReason::UnknownLocation;
    proof.detail = "dynamic-root-offset-or-type";
    return proof;
  }
  proof.activeLanes = getStaticActiveLanes(store.getMask(), value.getType());
  if (proof.activeLanes < 0) {
    proof.rejection = RejectReason::Mask;
    proof.detail = "dynamic-or-invalid-store-mask";
    return proof;
  }
  proof.location = StateLocation{*root, *offset * *elementBytes, *bytes,
                                 vreg.getElementType(), "continuous"};
  return proof;
}

static bool masksLimitObservation(Operation *op, int64_t activeLanes) {
  bool sawMask = false;
  for (Value operand : op->getOperands()) {
    auto maskType = dyn_cast<pto::VMIMaskType>(operand.getType());
    if (!maskType)
      continue;
    sawMask = true;
    auto create = operand.getDefiningOp<pto::VMICreateMaskOp>();
    std::optional<int64_t> active =
        create ? getConstantInt(create.getActiveLanes()) : std::nullopt;
    if (!active || *active > activeLanes)
      return false;
  }
  return sawMask;
}

static bool allConsumersRespectPrefix(Value value, int64_t activeLanes,
                                      unsigned depth = 0) {
  auto vreg = dyn_cast<pto::VMIVRegType>(value.getType());
  if (!vreg)
    return false;
  if (activeLanes == vreg.getElementCount())
    return true;
  if (depth > 8)
    return false;
  for (Operation *user : value.getUsers()) {
    if (isa<pto::VMIvStoreOp>(user))
      continue;
    if (!masksLimitObservation(user, activeLanes))
      return false;
    for (Value result : user->getResults()) {
      if (isa<pto::VMIVRegType>(result.getType()) &&
          !allConsumersRespectPrefix(result, activeLanes, depth + 1))
        return false;
    }
  }
  return true;
}

static bool dependsOn(Value value, Value source, unsigned depth = 0) {
  if (value == source)
    return true;
  if (depth > 32)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOn(operand, source, depth + 1);
  });
}

static bool isSyncOrCall(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return isa<CallOpInterface>(op) || name == "pto.set_flag" ||
         name == "pto.wait_flag" || name == "pto.barrier" ||
         name == "pto.barrier_all" || name == "pto.tload" ||
         name == "pto.tstore" || name == "pto.tcopy";
}

static bool hasUnsafeEffect(Operation *scope) {
  bool unsafe = false;
  scope->walk([&](Operation *op) {
    if (op == scope || isa<pto::VMIvLoadOp, pto::VMIvStoreOp,
                           pto::TileBufAddrOp, pto::AllocTileOp>(op))
      return;
    if (isSyncOrCall(op)) {
      unsafe = true;
      return;
    }
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 2> instances;
      effects.getEffects(instances);
      if (!instances.empty())
        unsafe = true;
    }
  });
  return unsafe;
}

static bool hasOverlappingAllocation(func::FuncOp func,
                                     const StateLocation &location) {
  bool overlap = false;
  func.walk([&](pto::AllocTileOp alloc) {
    std::optional<IntInterval> roots =
        evaluateNonNegativeInterval(alloc.getAddr());
    std::optional<int64_t> bytes = getStaticTileBytes(alloc.getResult());
    if (!roots || !bytes) {
      overlap = true;
      return;
    }
    int64_t candidateEnd =
        location.storageRoot + location.byteOffset + location.byteLength;
    int64_t allocEnd = roots->upper + *bytes;
    bool intersects = roots->lower < candidateEnd &&
                      allocEnd > location.storageRoot + location.byteOffset;
    if (!intersects)
      return;
    bool exactContainingRoot =
        roots->lower == location.storageRoot &&
        roots->upper == location.storageRoot && location.byteOffset >= 0 &&
        location.byteOffset + location.byteLength <= *bytes;
    if (!exactContainingRoot)
      overlap = true;
  });
  return overlap;
}

static bool rootEscapes(func::FuncOp func, int64_t root) {
  bool escaped = false;
  func.walk([&](pto::AllocTileOp alloc) {
    if (getConstantInt(alloc.getAddr()) != root)
      return;
    for (Operation *user : alloc.getResult().getUsers()) {
      if (!isa<pto::TileBufAddrOp, pto::YieldOp>(user))
        escaped = true;
    }
  });
  func.walk([&](pto::TileBufAddrOp address) {
    if (getTileRoot(address.getSrc()) != root)
      return;
    for (Operation *user : address.getResult().getUsers()) {
      if (!isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(user))
        escaped = true;
    }
  });
  return escaped;
}

// Return true when the allocation represented by `tile` is an internal VMI
// state slot rather than an observable tile result.  This is deliberately
// based on SSA allocation identity, not only on its assigned UB byte address:
// the memory planner is allowed to reuse the same address for a later output
// tile once this allocation's lifetime ends.
static bool isPrivateVMIStateTile(Value tile) {
  if (!tile || !tile.getDefiningOp<pto::AllocTileOp>())
    return false;
  for (Operation *tileUser : tile.getUsers()) {
    auto address = dyn_cast<pto::TileBufAddrOp>(tileUser);
    if (!address)
      return false;
    for (Operation *addressUser : address.getResult().getUsers())
      if (!isa<pto::VMIvLoadOp, pto::VMIvStoreOp>(addressUser))
        return false;
  }
  return true;
}

// A store to a private state tile is dead only after every load through that
// exact allocation has been forwarded.  Keeping this check until the rewrite
// transaction has selected all candidates preserves failure atomicity and
// avoids deleting a store when a mask/type rejection left a reload behind.
static bool allPrivateTileLoadsPromoted(pto::VMIvStoreOp store,
                                        const DenseSet<Operation *> &loads) {
  std::optional<Value> tile = getPointerTile(store.getDestination());
  if (!tile || !isPrivateVMIStateTile(*tile))
    return false;
  for (Operation *tileUser : tile->getUsers()) {
    auto address = cast<pto::TileBufAddrOp>(tileUser);
    for (Operation *addressUser : address.getResult().getUsers())
      if (isa<pto::VMIvLoadOp>(addressUser) && !loads.contains(addressUser))
        return false;
  }
  return true;
}

static pto::FusionRegionOp enclosingFusionRegion(Operation *op) {
  return op ? op->getParentOfType<pto::FusionRegionOp>()
            : pto::FusionRegionOp{};
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
  SmallVector<Value, 4> values(oldYield.getValues().begin(),
                               oldYield.getValues().end());
  values.push_back(output);
  rewriter.setInsertionPoint(oldYield);
  rewriter.create<pto::YieldOp>(oldYield.getLoc(), ValueRange(values));
  rewriter.eraseOp(oldYield);
  for (auto [oldResult, newResult] :
       llvm::zip(region.getOutputs(), replacement.getOutputs()))
    oldResult.replaceAllUsesWith(newResult);
  Value appended = replacement.getOutputs().back();
  rewriter.eraseOp(region);
  return appended;
}

static void setDecision(Operation *op, bool accepted, StateFlow flow,
                        std::optional<RejectReason> reason, StringRef detail,
                        bool emitRemarks) {
  if (!op)
    return;
  MLIRContext *context = op->getContext();
  op->setAttr(kStatusAttr,
              StringAttr::get(context, accepted ? "accepted" : "rejected"));
  op->setAttr(kFlowAttr, StringAttr::get(context, stringify(flow)));
  if (reason)
    op->setAttr(kReasonAttr, StringAttr::get(context, stringify(*reason)));
  else
    op->removeAttr(kReasonAttr);
  if (!detail.empty())
    op->setAttr(kDetailAttr, StringAttr::get(context, detail));
  else
    op->removeAttr(kDetailAttr);
  if (!emitRemarks)
    return;
  if (accepted)
    op->emitRemark() << "vmi-state-promotion accepted flow=" << stringify(flow);
  else
    op->emitRemark() << "vmi-state-promotion rejected reason="
                     << stringify(*reason) << " detail=" << detail;
}

static SmallVector<pto::VMIvStoreOp, 4>
storesInTopLevelOperation(Operation *top) {
  SmallVector<pto::VMIvStoreOp, 4> stores;
  if (auto store = dyn_cast<pto::VMIvStoreOp>(top))
    stores.push_back(store);
  else
    top->walk([&](pto::VMIvStoreOp store) { stores.push_back(store); });
  return stores;
}

static SmallVector<pto::VMIvLoadOp, 4>
loadsInTopLevelOperation(Operation *top) {
  SmallVector<pto::VMIvLoadOp, 4> loads;
  if (auto load = dyn_cast<pto::VMIvLoadOp>(top))
    loads.push_back(load);
  else
    top->walk([&](pto::VMIvLoadOp load) { loads.push_back(load); });
  return loads;
}

static pto::VMIvStoreOp findReachingStore(scf::ForOp loop,
                                          const StateLocation &location) {
  for (Operation *cursor = loop->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    SmallVector<pto::VMIvStoreOp, 4> stores = storesInTopLevelOperation(cursor);
    for (pto::VMIvStoreOp store : llvm::reverse(stores)) {
      AccessProof proof = analyzeStore(store);
      if (proof.location && proof.location->sameBytes(location))
        return store;
    }
    if (isa<scf::ForOp>(cursor) || isSyncOrCall(cursor))
      break;
  }
  return {};
}

static SmallVector<pto::VMIvLoadOp, 4>
findFinalLoads(scf::ForOp loop, const StateLocation &location) {
  SmallVector<pto::VMIvLoadOp, 4> loads;
  for (Operation *cursor = loop->getNextNode(); cursor;
       cursor = cursor->getNextNode()) {
    for (pto::VMIvStoreOp store : storesInTopLevelOperation(cursor)) {
      AccessProof proof = analyzeStore(store);
      if (proof.location && proof.location->sameBytes(location))
        return loads;
    }
    for (pto::VMIvLoadOp load : loadsInTopLevelOperation(cursor)) {
      AccessProof proof = analyzeLoad(load);
      if (proof.location && proof.location->sameBytes(location))
        loads.push_back(load);
    }
    if (isa<scf::ForOp>(cursor) || isSyncOrCall(cursor))
      break;
  }
  return loads;
}

static LoopCandidate analyzeLoopCandidate(func::FuncOp func, scf::ForOp loop,
                                          int64_t maxVectorChunks) {
  LoopCandidate candidate;
  candidate.loop = loop;
  SmallVector<pto::VMIvLoadOp, 4> loads;
  SmallVector<pto::VMIvStoreOp, 4> stores;
  bool nestedControl = false;
  loop.walk([&](Operation *op) {
    if (auto nested = dyn_cast<scf::ForOp>(op)) {
      if (nested != loop)
        nestedControl = true;
    } else if (auto load = dyn_cast<pto::VMIvLoadOp>(op)) {
      loads.push_back(load);
    } else if (auto store = dyn_cast<pto::VMIvStoreOp>(op)) {
      stores.push_back(store);
    }
  });
  if (nestedControl) {
    candidate.rejection = RejectReason::ControlFlow;
    candidate.detail = "nested-control-flow";
    return candidate;
  }
  if (loads.empty() && stores.empty()) {
    candidate.rejection = RejectReason::UnknownLocation;
    candidate.detail = "no-state-access";
    return candidate;
  }
  unsigned viablePairs = 0;
  for (pto::VMIvLoadOp load : loads) {
    AccessProof loadProof = analyzeLoad(load);
    if (!loadProof.location)
      continue;
    for (pto::VMIvStoreOp store : stores) {
      AccessProof storeProof = analyzeStore(store);
      if (!storeProof.location ||
          !storeProof.location->sameBytes(*loadProof.location))
        continue;
      if (load.getResult(0).getType() != store.getValues().front().getType() ||
          !dependsOn(store.getValues().front(), load.getResult(0)))
        continue;
      pto::VMIvStoreOp init = findReachingStore(loop, *loadProof.location);
      if (!init)
        continue;
      ++viablePairs;
      candidate.load = load;
      candidate.updateStore = store;
      candidate.initStore = init;
      candidate.location = *loadProof.location;
      candidate.activeLanes = storeProof.activeLanes;
    }
  }
  if (viablePairs > 1) {
    candidate.rejection = RejectReason::MultipleDefinitions;
    candidate.detail = "multiple-loop-carried-state-pairs";
    return candidate;
  }
  if (!candidate.load || !candidate.updateStore) {
    candidate.rejection = RejectReason::UnknownLocation;
    candidate.detail = "no-exact-loop-state-pair";
    return candidate;
  }
  Value update = candidate.updateStore.getValues().front();
  if (candidate.load.getResult(0).getType() != update.getType()) {
    candidate.rejection = RejectReason::TypeLayout;
    candidate.detail = "load-store-vreg-type-mismatch";
    return candidate;
  }
  if (!dependsOn(update, candidate.load.getResult(0))) {
    candidate.rejection = RejectReason::MultipleDefinitions;
    candidate.detail = "store-does-not-depend-on-load";
    return candidate;
  }
  if (!allConsumersRespectPrefix(candidate.load.getResult(0),
                                 candidate.activeLanes) ||
      !allConsumersRespectPrefix(update, candidate.activeLanes)) {
    candidate.rejection = RejectReason::Mask;
    candidate.detail = "inactive-lanes-may-be-observed";
    return candidate;
  }
  if (hasUnsafeEffect(loop)) {
    candidate.rejection = RejectReason::UnknownEffect;
    candidate.detail = "loop-has-unknown-effect";
    return candidate;
  }
  if (hasOverlappingAllocation(func, candidate.location)) {
    candidate.rejection = RejectReason::Alias;
    candidate.detail = "overlapping-static-or-unknown-allocation";
    return candidate;
  }
  if (rootEscapes(func, candidate.location.storageRoot)) {
    candidate.rejection = RejectReason::Escape;
    candidate.detail = "storage-or-pointer-escapes";
    return candidate;
  }
  AccessProof initProof = analyzeStore(candidate.initStore);
  if (!initProof.location || initProof.activeLanes != candidate.activeLanes ||
      candidate.initStore.getValues().front().getType() !=
          candidate.load.getResult(0).getType()) {
    candidate.rejection = RejectReason::Mask;
    candidate.detail = "preheader-mask-or-type-mismatch";
    return candidate;
  }
  candidate.finalLoads = findFinalLoads(loop, candidate.location);

  pto::VMIVectorPressureEstimate pressure = pto::estimateVMILoopPressure(loop);
  FailureOr<int64_t> carriedChunks =
      pto::getVMIVectorChunks(candidate.load.getResult(0).getType());
  if (!pressure.isExact || failed(carriedChunks)) {
    candidate.rejection = RejectReason::ResourcePressure;
    candidate.detail = "inexact-vector-pressure";
    return candidate;
  }
  candidate.pressureBefore = pressure.peakVectorChunks;
  candidate.pressureAfter =
      std::max(pressure.peakVectorChunks, pressure.persistentVectorChunks +
                                              *carriedChunks +
                                              pressure.temporaryVectorChunks);
  if (maxVectorChunks > 0 && candidate.pressureAfter > maxVectorChunks) {
    candidate.rejection = RejectReason::ResourcePressure;
    candidate.detail = "physical-chunk-budget-exceeded";
    return candidate;
  }
  return candidate;
}

static FailureOr<Value> makeValueVisible(Value value, Operation *consumer,
                                         IRRewriter &rewriter,
                                         DenseMap<Value, Value> &cache) {
  if (value.getParentBlock() == consumer->getBlock() ||
      !enclosingFusionRegion(value.getDefiningOp()))
    return value;
  pto::FusionRegionOp region = enclosingFusionRegion(value.getDefiningOp());
  if (region == enclosingFusionRegion(consumer))
    return value;
  auto found = cache.find(value);
  if (found != cache.end())
    return found->second;
  FailureOr<Value> visible = appendFusionRegionOutput(region, value, rewriter);
  if (failed(visible))
    return failure();
  cache.try_emplace(value, *visible);
  return *visible;
}

static LogicalResult promoteLoop(LoopCandidate &candidate,
                                 IRRewriter &rewriter) {
  DenseMap<Value, Value> visibleValues;
  Value init = candidate.initStore.getValues().front();
  FailureOr<Value> visibleInit =
      makeValueVisible(init, candidate.loop, rewriter, visibleValues);
  if (failed(visibleInit))
    return failure();
  Value update = candidate.updateStore.getValues().front();
  FailureOr<Value> visibleUpdate =
      makeValueVisible(update, candidate.loop.getBody()->getTerminator(),
                       rewriter, visibleValues);
  if (failed(visibleUpdate))
    return failure();

  FailureOr<LoopLikeOpInterface> replaced =
      candidate.loop.replaceWithAdditionalYields(
          rewriter, ValueRange{*visibleInit},
          /*replaceInitOperandUsesInLoop=*/false,
          [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
            return SmallVector<Value>{*visibleUpdate};
          });
  if (failed(replaced))
    return failure();
  auto newLoop = cast<scf::ForOp>((*replaced).getOperation());
  BlockArgument carried = newLoop.getRegionIterArgs().back();
  Value result = newLoop.getResults().back();
  candidate.load.getResult(0).replaceAllUsesWith(carried);
  for (pto::VMIvLoadOp load : candidate.finalLoads)
    load.getResult(0).replaceAllUsesWith(result);
  rewriter.eraseOp(candidate.load);
  rewriter.eraseOp(candidate.updateStore);
  rewriter.eraseOp(candidate.initStore);
  for (pto::VMIvLoadOp load : candidate.finalLoads)
    rewriter.eraseOp(load);
  newLoop->setAttr(kStatusAttr,
                   StringAttr::get(rewriter.getContext(), "accepted"));
  newLoop->setAttr(kFlowAttr,
                   StringAttr::get(rewriter.getContext(), "loop-carried"));
  newLoop->setAttr(kBeforePressureAttr,
                   IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64),
                                    candidate.pressureBefore));
  newLoop->setAttr(kAfterPressureAttr,
                   IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64),
                                    candidate.pressureAfter));
  return success();
}

static SmallVector<ForwardCandidate, 8> analyzeStraightLine(func::FuncOp func,
                                                            bool emitRemarks) {
  SmallVector<ForwardCandidate, 8> candidates;
  struct ReachingState {
    StateLocation location;
    pto::VMIvStoreOp store;
    int64_t activeLanes = -1;
  };
  SmallVector<ReachingState, 8> reaching;

  auto processStore = [&](pto::VMIvStoreOp store) {
    AccessProof proof = analyzeStore(store);
    if (!proof.location) {
      setDecision(store, false, StateFlow::StraightLine, proof.rejection,
                  proof.detail, emitRemarks);
      return;
    }
    llvm::erase_if(reaching, [&](const ReachingState &state) {
      return state.location.overlaps(*proof.location);
    });
    reaching.push_back(
        ReachingState{*proof.location, store, proof.activeLanes});
  };
  auto processLoad = [&](pto::VMIvLoadOp load) {
    AccessProof proof = analyzeLoad(load);
    if (!proof.location) {
      setDecision(load, false, StateFlow::StraightLine, proof.rejection,
                  proof.detail, emitRemarks);
      return;
    }
    ReachingState *found = nullptr;
    bool sawPartialAlias = false;
    for (ReachingState &state : llvm::reverse(reaching)) {
      if (state.location.sameBytes(*proof.location)) {
        found = &state;
        break;
      }
      sawPartialAlias |= state.location.overlaps(*proof.location);
    }
    if (!found) {
      if (sawPartialAlias)
        setDecision(load, false, StateFlow::StraightLine, RejectReason::Alias,
                    "partial-or-different-byte-range", emitRemarks);
      return;
    }
    Value stored = found->store.getValues().front();
    bool broadcast = proof.location->distribution == "brc";
    auto storedType = dyn_cast<pto::VMIVRegType>(stored.getType());
    auto loadType = dyn_cast<pto::VMIVRegType>(load.getResult(0).getType());
    bool typeCompatible =
        (!broadcast && stored.getType() == load.getResult(0).getType()) ||
        (broadcast && storedType && loadType &&
         storedType.getElementCount() == 1 &&
         storedType.getElementType() == loadType.getElementType());
    if (!typeCompatible) {
      setDecision(load, false, StateFlow::StraightLine,
                  RejectReason::TypeLayout, "reaching-value-type-mismatch",
                  emitRemarks);
      return;
    }
    int64_t observedLanes = broadcast ? proof.activeLanes : found->activeLanes;
    if (!allConsumersRespectPrefix(load.getResult(0), observedLanes)) {
      setDecision(load, false, StateFlow::StraightLine, RejectReason::Mask,
                  "inactive-lanes-may-be-observed", emitRemarks);
      return;
    }
    bool canDeleteStore = !hasOverlappingAllocation(func, *proof.location) &&
                          !rootEscapes(func, proof.location->storageRoot);
    candidates.push_back(ForwardCandidate{found->store, load, *proof.location,
                                          observedLanes, broadcast,
                                          canDeleteStore});
  };

  std::function<void(Region &)> scanRegion = [&](Region &scope) {
    for (Block &block : scope) {
      reaching.clear();
      for (Operation &top : block.without_terminator()) {
        if (auto loop = dyn_cast<scf::ForOp>(top)) {
          scanRegion(loop.getRegion());
          reaching.clear();
          continue;
        }
        if (isSyncOrCall(&top)) {
          reaching.clear();
          continue;
        }
        if (auto region = dyn_cast<pto::FusionRegionOp>(top)) {
          for (Operation &nested :
               region.getBody().front().without_terminator()) {
            if (auto store = dyn_cast<pto::VMIvStoreOp>(nested))
              processStore(store);
            else if (auto load = dyn_cast<pto::VMIvLoadOp>(nested))
              processLoad(load);
            else if (nested.getNumRegions() != 0 || isSyncOrCall(&nested) ||
                     (isa<MemoryEffectOpInterface>(nested) &&
                      !isMemoryEffectFree(&nested)))
              reaching.clear();
          }
          continue;
        }
        if (auto store = dyn_cast<pto::VMIvStoreOp>(top))
          processStore(store);
        else if (auto load = dyn_cast<pto::VMIvLoadOp>(top))
          processLoad(load);
        else if (top.getNumRegions() != 0) {
          for (Region &nested : top.getRegions())
            scanRegion(nested);
          reaching.clear();
        } else if (isa<MemoryEffectOpInterface>(top) &&
                   !isMemoryEffectFree(&top)) {
          reaching.clear();
        }
      }
    }
  };
  scanRegion(func.getBody());
  return candidates;
}

static LogicalResult promoteStraightLine(ArrayRef<ForwardCandidate> candidates,
                                         IRRewriter &rewriter,
                                         bool emitRemarks) {
  DenseMap<Value, Value> visibleValues;
  DenseSet<Operation *> deadLoads;
  DenseSet<Operation *> deadStores;
  DenseSet<Operation *> forwardedStores;
  for (ForwardCandidate candidate : candidates) {
    Value value = candidate.store.getValues().front();
    FailureOr<Value> visible =
        makeValueVisible(value, candidate.load, rewriter, visibleValues);
    if (failed(visible))
      return failure();
    Value replacement = *visible;
    if (candidate.broadcast) {
      rewriter.setInsertionPoint(candidate.load);
      replacement =
          rewriter
              .create<pto::VMIVbrcOp>(candidate.load.getLoc(),
                                      candidate.load.getResult(0).getType(),
                                      replacement, IntegerAttr{})
              .getResult();
    }
    candidate.load.getResult(0).replaceAllUsesWith(replacement);
    setDecision(candidate.store, true, StateFlow::StraightLine, std::nullopt,
                "", emitRemarks);
    deadLoads.insert(candidate.load);
    forwardedStores.insert(candidate.store);
    if (candidate.canDeleteStore)
      deadStores.insert(candidate.store);
  }
  for (Operation *store : forwardedStores) {
    auto vmiStore = cast<pto::VMIvStoreOp>(store);
    if (allPrivateTileLoadsPromoted(vmiStore, deadLoads))
      deadStores.insert(store);
  }
  for (Operation *load : deadLoads)
    if (llvm::all_of(load->getResults(),
                     [](Value result) { return result.use_empty(); }))
      rewriter.eraseOp(load);
  for (Operation *store : deadStores)
    rewriter.eraseOp(store);
  return success();
}

static bool hoistInvariantLoads(func::FuncOp func, IRRewriter &rewriter,
                                bool shadow, bool emitRemarks) {
  bool changed = false;
  SmallVector<pto::VMIvLoadOp, 8> loads;
  func.walk([&](pto::VMIvLoadOp load) {
    if (load->getParentOfType<scf::ForOp>())
      loads.push_back(load);
  });
  for (pto::VMIvLoadOp load : loads) {
    scf::ForOp loop = load->getParentOfType<scf::ForOp>();
    if (!loop || load.getResults().size() != 1)
      continue;
    AccessProof proof = analyzeLoad(load);
    if (!proof.location)
      continue;
    bool written = false;
    loop.walk([&](pto::VMIvStoreOp store) {
      AccessProof storeProof = analyzeStore(store);
      if (storeProof.location &&
          storeProof.location->sameBytes(*proof.location))
        written = true;
    });
    if (written)
      continue;
    if (hasUnsafeEffect(loop)) {
      setDecision(load, false, StateFlow::LoopInvariant,
                  RejectReason::UnknownEffect, "loop-has-unknown-effect",
                  emitRemarks);
      continue;
    }
    setDecision(load, true, StateFlow::LoopInvariant, std::nullopt, "",
                emitRemarks);
    if (shadow)
      continue;
    auto address = load.getSource().getDefiningOp<pto::TileBufAddrOp>();
    if (!address)
      continue;
    rewriter.setInsertionPoint(loop);
    auto clonedAddress = cast<pto::TileBufAddrOp>(rewriter.clone(*address));
    Operation *clonedLoad = rewriter.clone(*load);
    clonedLoad->setOperand(0, clonedAddress.getResult());
    load.getResult(0).replaceAllUsesWith(clonedLoad->getResult(0));
    rewriter.eraseOp(load);
    changed = true;
  }
  return changed;
}

struct PTOVMIStatePromotionPass
    : public pto::impl::PTOVMIStatePromotionBase<PTOVMIStatePromotionPass> {
  using pto::impl::PTOVMIStatePromotionBase<
      PTOVMIStatePromotionPass>::PTOVMIStatePromotionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    if (mode != "generic" && mode != "shadow") {
      func.emitError("pto-vmi-state-promotion mode must be generic or shadow");
      return signalPassFailure();
    }
    if (maxVectorChunks < 0) {
      func.emitError("max-vector-chunks must be non-negative");
      return signalPassFailure();
    }
    const bool shadow = mode == "shadow";
    IRRewriter rewriter(&getContext());

    SmallVector<scf::ForOp, 8> loops;
    func.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      LoopCandidate candidate =
          analyzeLoopCandidate(func, loop, maxVectorChunks);
      bool hasLegacyAccepted = false;
      if (auto status =
              loop->getAttrOfType<StringAttr>(kLegacyAccumulatorStatus))
        hasLegacyAccepted |= status.getValue() == "accepted";
      if (auto status = loop->getAttrOfType<StringAttr>(kLegacyScalarStatus))
        hasLegacyAccepted |= status.getValue() == "accepted";
      bool genericAccepted = !candidate.rejection;
      if (shadow) {
        StringRef parity =
            genericAccepted
                ? (hasLegacyAccepted ? "equivalent" : "generic-only")
                : (hasLegacyAccepted ? "legacy-only" : "both-rejected");
        loop->setAttr(kShadowAttr, StringAttr::get(&getContext(), parity));
      }
      if (candidate.rejection) {
        if (candidate.detail != "no-state-access")
          setDecision(loop, false, StateFlow::LoopCarried, candidate.rejection,
                      candidate.detail, emitRemarks);
        continue;
      }
      setDecision(loop, true, StateFlow::LoopCarried, std::nullopt, "",
                  emitRemarks);
      loop->setAttr(kBeforePressureAttr,
                    rewriter.getI64IntegerAttr(candidate.pressureBefore));
      loop->setAttr(kAfterPressureAttr,
                    rewriter.getI64IntegerAttr(candidate.pressureAfter));
      if (!shadow && failed(promoteLoop(candidate, rewriter))) {
        loop.emitError("failed after generic VMI state proof was accepted");
        return signalPassFailure();
      }
    }

    hoistInvariantLoads(func, rewriter, shadow, emitRemarks);
    SmallVector<ForwardCandidate, 8> forwarding =
        analyzeStraightLine(func, emitRemarks);
    if (!shadow &&
        failed(promoteStraightLine(forwarding, rewriter, emitRemarks))) {
      func.emitError("failed after straight-line VMI state proof was accepted");
      return signalPassFailure();
    }
    if (!shadow)
      stripLegacyPhaseMetadata(func);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVMIStatePromotionPass() {
  return std::make_unique<PTOVMIStatePromotionPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOVMIStatePromotionPass(
    llvm::StringRef mode, int64_t maxVectorChunks, bool emitRemarks) {
  mlir::pto::PTOVMIStatePromotionOptions options;
  options.mode = mode.str();
  options.maxVectorChunks = maxVectorChunks;
  options.emitRemarks = emitRemarks;
  return std::make_unique<PTOVMIStatePromotionPass>(options);
}
