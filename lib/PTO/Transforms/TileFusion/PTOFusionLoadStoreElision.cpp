// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONLOADSTOREELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

constexpr StringLiteral kStatePromotionVecScopeBridgeAttr =
    "pto.vmi.state_promotion.vecscope_bridge";

struct TrackedStore {
  Operation *op = nullptr;
  Value base;
  SmallVector<Value, 2> indices;
  Value mask;
  Value value;
};

struct FusionRegionStoreContext {
  Block *body = nullptr;
  Block *parentBlock = nullptr;
  Operation *regionOp = nullptr;
  llvm::DenseSet<Value> yieldedValues;
  SmallVector<Value, 8> externallyObservedAddresses;
};

static bool areEquivalentValues(Value lhs, Value rhs);
static bool isAddressScaffoldOp(Operation *op);
static Value getCanonicalBufferAddress(Value value);
static bool areEquivalentValueRanges(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
           return areEquivalentValues(std::get<0>(pair), std::get<1>(pair));
         });
}

static bool areEquivalentOperations(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs)
    return false;
  if (lhs->getName() != rhs->getName())
    return false;
  if (lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0)
    return false;
  if (lhs->getNumResults() != rhs->getNumResults())
    return false;
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary())
    return false;
  if (!llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()))
    return false;

  if (auto lhsDim = dyn_cast<memref::DimOp>(lhs)) {
    auto rhsDim = cast<memref::DimOp>(rhs);
    return lhsDim.getSource().getType() == rhsDim.getSource().getType() &&
           areEquivalentValues(lhsDim.getIndex(), rhsDim.getIndex());
  }

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (!areEquivalentValues(lhsOperand, rhsOperand))
      return false;
  }
  return true;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs.getType() != rhs.getType())
    return false;

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    return lhsArg && rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
           lhsArg.getArgNumber() == rhsArg.getArgNumber();
  }

  return areEquivalentOperations(lhs.getDefiningOp(), rhs.getDefiningOp());
}

static bool isAllTruePatternMask(Value value) {
  Operation *op = value ? value.getDefiningOp() : nullptr;
  if (!op)
    return false;
  if (isa<pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op, pto::PgeB8Op,
          pto::PgeB16Op, pto::PgeB32Op>(op)) {
    auto pattern = op->getAttrOfType<StringAttr>("pattern");
    return pattern && pattern.getValue() == "PAT_ALL";
  }

  int64_t physicalLanes = 0;
  if (isa<pto::PltB8Op>(op))
    physicalLanes = 256;
  else if (isa<pto::PltB16Op>(op))
    physicalLanes = 128;
  else if (isa<pto::PltB32Op>(op))
    physicalLanes = 64;
  else
    return false;

  APInt activeLanes;
  return matchPattern(op->getOperand(0), m_ConstantInt(&activeLanes)) &&
         activeLanes.getSExtValue() >= physicalLanes;
}

static int64_t getPhysicalMaskLanes(Type type) {
  auto maskType = dyn_cast<pto::MaskType>(type);
  if (!maskType)
    return 0;
  if (maskType.isB8())
    return 256;
  if (maskType.isB16())
    return 128;
  if (maskType.isB32())
    return 64;
  return 0;
}

/// Return the number of active lanes when a VPTO mask is provably a prefix.
/// This recognizes both pattern masks and the conservative
/// `pand(PAT_VLx, plt(x), PAT_ALL)` form emitted for compact masked stores.
static std::optional<int64_t> getStaticPrefixActiveLanes(Value value) {
  Operation *op = value ? value.getDefiningOp() : nullptr;
  int64_t physicalLanes = value ? getPhysicalMaskLanes(value.getType()) : 0;
  if (!op || physicalLanes == 0)
    return std::nullopt;

  if (isa<pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op, pto::PgeB8Op,
          pto::PgeB16Op, pto::PgeB32Op>(op)) {
    auto pattern = op->getAttrOfType<StringAttr>("pattern");
    if (!pattern)
      return std::nullopt;
    StringRef value = pattern.getValue();
    if (value == "PAT_ALL")
      return physicalLanes;
    if (value == "PAT_ALLF")
      return 0;
    if (!value.consume_front("PAT_VL"))
      return std::nullopt;
    int64_t activeLanes = 0;
    if (value.getAsInteger(10, activeLanes))
      return std::nullopt;
    return std::clamp<int64_t>(activeLanes, 0, physicalLanes);
  }

  if (isa<pto::PltB8Op, pto::PltB16Op, pto::PltB32Op>(op)) {
    APInt activeLanes;
    if (!matchPattern(op->getOperand(0), m_ConstantInt(&activeLanes)))
      return std::nullopt;
    return std::clamp<int64_t>(activeLanes.getSExtValue(), 0, physicalLanes);
  }

  if (isa<pto::PandOp>(op)) {
    int64_t activeLanes = physicalLanes;
    for (Value operand : op->getOperands()) {
      std::optional<int64_t> operandLanes = getStaticPrefixActiveLanes(operand);
      if (!operandLanes)
        return std::nullopt;
      activeLanes = std::min(activeLanes, *operandLanes);
    }
    return activeLanes;
  }

  return std::nullopt;
}

static bool areEquivalentMaskValues(Value lhs, Value rhs) {
  if (areEquivalentValues(lhs, rhs))
    return true;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return false;
  if (isAllTruePatternMask(lhs) && isAllTruePatternMask(rhs))
    return true;
  std::optional<int64_t> lhsLanes = getStaticPrefixActiveLanes(lhs);
  std::optional<int64_t> rhsLanes = getStaticPrefixActiveLanes(rhs);
  return lhsLanes && rhsLanes && *lhsLanes == *rhsLanes;
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isStatePromotionPipeSyncBridge(Operation *op) {
  return op->hasAttr(kStatePromotionVecScopeBridgeAttr) &&
         isa<pto::SetFlagOp, pto::WaitFlagOp>(op);
}

/// Elide a repeated physical full-vector load across only those static pipe
/// events that the generic VMI state proof marked as UB-transparent.  This is
/// deliberately delayed until after VMI layout assignment: rearrangement and
/// arithmetic users may impose different logical layouts even though they
/// lower to the same physical VLD.  Any store, DMA, untagged synchronization,
/// call, nested control flow, or unknown effect clears the read frontier.
static bool elideRedundantLoadsAcrossPromotedPipeSync(func::FuncOp func) {
  struct TrackedLoad {
    pto::VldsOp load;
    Value base;
    Value offset;
  };

  SmallVector<Operation *, 8> eraseOrder;
  std::function<void(Region &)> scanRegion = [&](Region &region) {
    for (Block &block : region) {
      SmallVector<TrackedLoad, 8> trackedLoads;
      for (Operation &op : block.without_terminator()) {
        if (auto load = dyn_cast<pto::VldsOp>(op)) {
          if (load->getNumResults() != 1) {
            trackedLoads.clear();
            continue;
          }
          auto found = llvm::find_if(trackedLoads, [&](TrackedLoad &old) {
            return old.load.getResult().getType() ==
                       load.getResult().getType() &&
                   old.load.getDistAttr() == load.getDistAttr() &&
                   areEquivalentValues(
                       getCanonicalBufferAddress(old.base),
                       getCanonicalBufferAddress(load.getSource())) &&
                   areEquivalentValues(old.offset, load.getOffset());
          });
          if (found != trackedLoads.end()) {
            load.getResult().replaceAllUsesWith(found->load.getResult());
            eraseOrder.push_back(load.getOperation());
          } else {
            trackedLoads.push_back(
                TrackedLoad{load, load.getSource(), load.getOffset()});
          }
          continue;
        }

        if (isStatePromotionPipeSyncBridge(&op))
          continue;
        if (op.getNumRegions() != 0) {
          for (Region &nested : op.getRegions())
            scanRegion(nested);
          trackedLoads.clear();
          continue;
        }
        if (!isPureNoRegionOp(&op) && !isAddressScaffoldOp(&op) &&
            !isa<pto::AllocTileOp>(op))
          trackedLoads.clear();
      }
    }
  };

  scanRegion(func.getBody());
  for (Operation *op : eraseOrder)
    if (op && op->getBlock())
      op->erase();
  return !eraseOrder.empty();
}

static bool isSupportedLoopPreludeOp(Operation *op) {
  if (isa<pto::UvldOp>(op))
    return true;
  return isPureNoRegionOp(op);
}

static bool isAddressScaffoldOp(Operation *op) {
  return isa<pto::PointerCastOp, pto::CastPtrOp, pto::AddPtrOp, pto::BindTileOp,
             pto::TileBufAddrOp, pto::SubViewOp, memref::SubViewOp,
             memref::CastOp, memref::ReshapeOp, memref::ReinterpretCastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp,
             memref::MemorySpaceCastOp, memref::TransposeOp>(op);
}

/// Remove address-only chains made dead by the previous forwarding round.
///
/// A dead pointer chain in a later fusion phase must not make an earlier UB
/// store appear externally observable.  Erase only the explicitly supported
/// address scaffold operations, and iterate in reverse lexical order so a
/// dead cast user exposes its dead pointer producer in the same cleanup.
static bool eraseDeadAddressScaffoldOps(func::FuncOp func) {
  bool changed = false;
  while (true) {
    SmallVector<Operation *, 16> dead;
    func.walk([&](Operation *op) {
      if (!isAddressScaffoldOp(op) || op->getNumResults() == 0)
        return;
      if (llvm::all_of(op->getResults(),
                       [](Value result) { return result.use_empty(); }))
        dead.push_back(op);
    });
    if (dead.empty())
      break;

    bool erasedThisRound = false;
    for (Operation *op : llvm::reverse(dead)) {
      if (!llvm::all_of(op->getResults(),
                        [](Value result) { return result.use_empty(); }))
        continue;
      op->erase();
      erasedThisRound = true;
      changed = true;
    }
    if (!erasedThisRound)
      break;
  }
  return changed;
}

static bool isSupportedLeafOp(Operation *op) {
  if (isa<pto::VldsOp, pto::VstsOp, pto::AllocTileOp>(op))
    return true;
  if (isAddressScaffoldOp(op))
    return true;
  return isPureNoRegionOp(op);
}

static Value getCanonicalTrackedValue(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;

    if (auto bind = dyn_cast<pto::BindTileOp>(def)) {
      value = bind.getSource();
      continue;
    }
    if (auto tileBufAddr = dyn_cast<pto::TileBufAddrOp>(def)) {
      value = tileBufAddr.getSrc();
      continue;
    }
    if (auto subview = dyn_cast<pto::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto bitcast = dyn_cast<pto::BitcastOp>(def)) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<pto::TReshapeOp>(def)) {
      value = reshape.getSrc();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
      value = reshape.getSource();
      continue;
    }
    if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpretCast.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    if (auto memorySpaceCast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = memorySpaceCast.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      value = transpose.getIn();
      continue;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().empty())
        break;
      if (auto result = dyn_cast<OpResult>(value)) {
        unsigned resultNumber = result.getResultNumber();
        if (resultNumber < cast.getInputs().size()) {
          value = cast.getInputs()[resultNumber];
          continue;
        }
      }
      if (cast.getInputs().size() == 1) {
        value = cast.getInputs().front();
        continue;
      }
    }
    break;
  }
  return value;
}

static Value getCanonicalBufferAddress(Value value) {
  value = getCanonicalTrackedValue(value);
  if (!value)
    return {};

  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>())
    return alloc.getAddr();
  if (auto cast = value.getDefiningOp<pto::CastPtrOp>())
    return getCanonicalBufferAddress(cast.getInput());
  if (auto pointerCast = value.getDefiningOp<pto::PointerCastOp>()) {
    if (pointerCast.getAddrs().size() == 1)
      return getCanonicalTrackedValue(pointerCast.getAddrs().front());
  }
  return value;
}

static bool aliasesYieldedBuffer(Value buffer,
                                 const llvm::DenseSet<Value> &yieldedValues) {
  Value address = getCanonicalBufferAddress(buffer);
  if (!address)
    return false;
  return llvm::any_of(yieldedValues, [&](Value yielded) {
    Value yieldedAddress = getCanonicalBufferAddress(yielded);
    return yieldedAddress && areEquivalentValues(address, yieldedAddress);
  });
}

static bool aliasesAnyAddress(Value buffer, ArrayRef<Value> addresses) {
  Value address = getCanonicalBufferAddress(buffer);
  if (!address)
    return false;
  return llvm::any_of(addresses, [&](Value candidate) {
    return candidate && areEquivalentValues(address, candidate);
  });
}

static bool
normalizeFusionRegionYieldFrontier(pto::FusionRegionOp fusionRegion) {
  Block &body = fusionRegion.getBody().front();
  auto yieldOp = dyn_cast<pto::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return false;

  bool changed = false;
  for (auto [index, yielded] : llvm::enumerate(yieldOp.getValues())) {
    auto bind = yielded.getDefiningOp<pto::BindTileOp>();
    if (!bind)
      continue;

    Value normalized = bind.getSource();
    if (!normalized || normalized == yielded)
      continue;

    Value regionResult = fusionRegion.getResult(index);
    Type originalResultType = regionResult.getType();

    yieldOp->setOperand(index, normalized);
    if (regionResult.getType() != normalized.getType())
      regionResult.setType(normalized.getType());

    if (originalResultType != normalized.getType() &&
        !regionResult.use_empty()) {
      OpBuilder builder(fusionRegion);
      builder.setInsertionPointAfter(fusionRegion);
      auto rebound = builder.create<pto::BindTileOp>(
          bind.getLoc(), originalResultType, regionResult, bind.getValidRow(),
          bind.getValidCol(), bind.getConfig());
      rebound->setAttrs(bind->getAttrDictionary());
      regionResult.replaceAllUsesExcept(rebound.getResult(), rebound);
    }
    changed = true;
  }
  return changed;
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur->getBlock() == block)
      return cur;
  return nullptr;
}

static Region *getDirectRegionUnderAncestor(Operation *op,
                                            Operation *ancestor) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    Operation *parent = cur->getParentOp();
    if (parent == ancestor)
      return cur->getBlock() ? cur->getBlock()->getParent() : nullptr;
  }
  return nullptr;
}

static bool areMutuallyExclusiveByIfRegion(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs)
    return false;

  for (Operation *ancestor = lhs; ancestor;
       ancestor = ancestor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(ancestor);
    if (!ifOp)
      continue;

    Region *lhsRegion = getDirectRegionUnderAncestor(lhs, ifOp);
    Region *rhsRegion = getDirectRegionUnderAncestor(rhs, ifOp);
    if (!lhsRegion || !rhsRegion)
      continue;
    if (lhsRegion != rhsRegion)
      return true;
  }

  return false;
}

static bool isLexicallyAfter(Operation *anchor, Operation *candidate) {
  if (!anchor || !candidate || anchor == candidate)
    return false;
  for (Operation *anchorAncestor = anchor; anchorAncestor;
       anchorAncestor = anchorAncestor->getParentOp()) {
    for (Operation *candidateAncestor = candidate; candidateAncestor;
         candidateAncestor = candidateAncestor->getParentOp()) {
      if (anchorAncestor == candidateAncestor)
        continue;
      if (anchorAncestor->getBlock() != candidateAncestor->getBlock())
        continue;
      return anchorAncestor->isBeforeInBlock(candidateAncestor);
    }
  }
  return false;
}

static std::optional<FusionRegionStoreContext>
buildFusionRegionStoreContext(pto::FusionRegionOp fusionRegion) {
  Block &body = fusionRegion.getBody().front();
  auto yieldOp = dyn_cast<pto::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return std::nullopt;

  FusionRegionStoreContext context;
  context.body = &body;
  context.parentBlock = fusionRegion->getBlock();
  context.regionOp = fusionRegion.getOperation();

  for (Value yielded : yieldOp.getValues()) {
    Value canonical = getCanonicalTrackedValue(yielded);
    if (canonical)
      context.yieldedValues.insert(canonical);
  }

  if (auto func = fusionRegion->getParentOfType<func::FuncOp>()) {
    func.walk([&](pto::PointerCastOp pointerCast) {
      if (fusionRegion->isProperAncestor(pointerCast) ||
          !isLexicallyAfter(fusionRegion, pointerCast))
        return;
      for (Value address : pointerCast.getAddrs())
        context.externallyObservedAddresses.push_back(
            getCanonicalTrackedValue(address));
    });
    func.walk([&](pto::AllocTileOp alloc) {
      if (fusionRegion->isProperAncestor(alloc) || !alloc.getAddr() ||
          !isLexicallyAfter(fusionRegion, alloc))
        return;
      context.externallyObservedAddresses.push_back(
          getCanonicalTrackedValue(alloc.getAddr()));
    });
  }

  return context;
}

static bool isSupportedLoopRoot(scf::ForOp loop) {
  if (!loop)
    return false;
  return isa<pto::FusionRegionOp, pto::VecScopeOp, pto::StrictVecScopeOp,
             func::FuncOp, scf::ForOp>(loop->getParentOp());
}

static Block *getLeafLoopBody(scf::ForOp carrierLoop) {
  if (!carrierLoop)
    return nullptr;

  scf::ForOp currentLoop = carrierLoop;
  while (currentLoop) {
    SmallVector<Operation *, 8> bodyOps;
    scf::ForOp innerLoop;
    for (Operation &op : currentLoop.getBody()->without_terminator()) {
      bodyOps.push_back(&op);
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (innerLoop)
          return nullptr;
        innerLoop = loop;
      }
    }

    if (!innerLoop) {
      Block *leafBody = currentLoop.getBody();
      if (!leafBody)
        return nullptr;
      for (Operation &op : leafBody->without_terminator())
        if (!isSupportedLeafOp(&op))
          return nullptr;
      return leafBody;
    }

    bool seenInnerLoop = false;
    for (Operation *op : bodyOps) {
      if (op == innerLoop.getOperation()) {
        seenInnerLoop = true;
        continue;
      }
      if (seenInnerLoop || !isSupportedLoopPreludeOp(op))
        return nullptr;
    }

    currentLoop = innerLoop;
  }

  return nullptr;
}

static bool isSupportedStraightLineBlock(Block &body) {
  for (Operation &op : body.without_terminator())
    if (!isSupportedLeafOp(&op))
      return false;
  return true;
}

static Value inferVPTOLoadUserMask(pto::VldsOp load) {
  Value inferredMask;
  for (OpOperand &use : load.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (!owner || owner->getNumRegions() != 0)
      return Value();

    Value ownerMask;
    for (Value operand : owner->getOperands()) {
      if (!isa<pto::MaskType>(operand.getType()))
        continue;
      if (!ownerMask)
        ownerMask = operand;
      else if (!areEquivalentMaskValues(ownerMask, operand))
        return Value();
    }

    if (!ownerMask)
      return Value();

    if (!inferredMask)
      inferredMask = ownerMask;
    else if (!areEquivalentMaskValues(inferredMask, ownerMask))
      return Value();
  }
  return inferredMask;
}

static int findTrackedStoreIndex(ArrayRef<TrackedStore> stores, Value base,
                                 ArrayRef<Value> indices, Value mask) {
  Value canonicalBaseAddress = getCanonicalBufferAddress(base);
  for (int index = static_cast<int>(stores.size()) - 1; index >= 0; --index) {
    const TrackedStore &store = stores[index];
    if (areEquivalentValues(getCanonicalBufferAddress(store.base),
                            canonicalBaseAddress) &&
        areEquivalentValueRanges(store.indices, indices) &&
        areEquivalentMaskValues(store.mask, mask)) {
      return index;
    }
  }
  return -1;
}

static bool isFullVectorBroadcastLoad(pto::VldsOp load, Value userMask) {
  std::optional<StringRef> dist = load.getDist();
  if (!dist || *dist != "BRC_B32")
    return false;
  std::optional<int64_t> activeLanes = getStaticPrefixActiveLanes(userMask);
  return activeLanes &&
         *activeLanes == getPhysicalMaskLanes(userMask.getType());
}

/// Match the exact compact-scalar materialization used by a one-VL row
/// pipeline.  The producer writes lane zero and BRC_B32 reloads that scalar
/// into every lane.  This is deliberately separate from ordinary forwarding:
/// replacing the load with the stored vector directly would leave lanes 1..63
/// undefined and would be a miscompile.
static int findTrackedScalarBroadcastStoreIndex(ArrayRef<TrackedStore> stores,
                                                pto::VldsOp load,
                                                Value userMask) {
  if (!isFullVectorBroadcastLoad(load, userMask) ||
      !load.getResult().hasOneUse())
    return -1;

  Value canonicalBaseAddress = getCanonicalBufferAddress(load.getSource());
  for (int index = static_cast<int>(stores.size()) - 1; index >= 0; --index) {
    const TrackedStore &store = stores[index];
    if (!areEquivalentValues(getCanonicalBufferAddress(store.base),
                             canonicalBaseAddress) ||
        store.indices.size() != 1 ||
        !areEquivalentValues(store.indices.front(), load.getOffset()) ||
        store.value.getType() != load.getResult().getType())
      continue;
    std::optional<int64_t> storedLanes = getStaticPrefixActiveLanes(store.mask);
    if (storedLanes && *storedLanes == 1)
      return index;
  }
  return -1;
}

static void pruneTrackedStoresForLoadBase(SmallVectorImpl<TrackedStore> &stores,
                                          Value base) {
  if (!base) {
    stores.clear();
    return;
  }
  Value canonicalBaseAddress = getCanonicalBufferAddress(base);
  llvm::erase_if(stores, [&](const TrackedStore &store) {
    return areEquivalentValues(getCanonicalBufferAddress(store.base),
                               canonicalBaseAddress);
  });
}

static bool shouldElideTailStore(
    const TrackedStore &store, const FusionRegionStoreContext &context,
    Operation *scopeOp,
    const llvm::SmallPtrSetImpl<Operation *> &scheduledForErase) {
  Value canonicalBase = getCanonicalTrackedValue(store.base);
  if (!canonicalBase)
    return false;
  Operation *localScopeOp = scopeOp ? scopeOp : store.op;
  if (!localScopeOp)
    return false;
  // Yielded frontier is still region-observable in v1, so its final
  // materializing store must be preserved even if there is no reload.
  if (context.yieldedValues.contains(canonicalBase) ||
      aliasesYieldedBuffer(canonicalBase, context.yieldedValues) ||
      aliasesAnyAddress(canonicalBase, context.externallyObservedAddresses))
    return false;

  // Different TileLib phases commonly rebuild pointer/address scaffolding for
  // the same static UB range.  Those equivalent pointers need not share an SSA
  // use chain with this store's base.  Preserve the store when a later
  // top-level phase reads or writes an equivalent address; otherwise an
  // in-place producer such as Softmax expdif can be mistaken for dead even
  // though the final divide loop reloads the exponent matrix.
  bool hasLaterEquivalentAccess = false;
  context.regionOp->walk([&](Operation *op) {
    if (hasLaterEquivalentAccess || op == store.op)
      return;
    Value accessedBase;
    if (auto load = dyn_cast<pto::VldsOp>(op))
      accessedBase = load.getSource();
    else if (auto otherStore = dyn_cast<pto::VstsOp>(op))
      accessedBase = otherStore.getDestination();
    else
      return;
    if (!areEquivalentValues(getCanonicalBufferAddress(accessedBase),
                             getCanonicalBufferAddress(canonicalBase)))
      return;
    Operation *topLevelUser = getTopLevelAncestorInBlock(op, context.body);
    if (!topLevelUser || topLevelUser == localScopeOp)
      return;
    if (localScopeOp->getBlock() == topLevelUser->getBlock() &&
        localScopeOp->isBeforeInBlock(topLevelUser))
      hasLaterEquivalentAccess = true;
  });
  if (hasLaterEquivalentAccess)
    return false;

  APInt staticAddress;
  if (!matchPattern(getCanonicalBufferAddress(canonicalBase),
                    m_ConstantInt(&staticAddress)))
    return false;

  for (OpOperand &use : canonicalBase.getUses()) {
    Operation *owner = use.getOwner();
    if (!owner || scheduledForErase.contains(owner))
      continue;
    if (context.regionOp->isProperAncestor(owner)) {
      // Uses nested under the current carrier loop are fine: erasing the tail
      // store only affects memory materialization, while SSA users still
      // observe the forwarded vector value. A later top-level op in the same
      // fusion region may still require the buffer to stay materialized, so
      // keep the store.
      Operation *topLevelUser = getTopLevelAncestorInBlock(owner, context.body);
      if (!topLevelUser)
        return false;
      if (scheduledForErase.contains(topLevelUser))
        continue;
      if (topLevelUser == localScopeOp)
        continue;
      if (localScopeOp->getBlock() == topLevelUser->getBlock() &&
          localScopeOp->isBeforeInBlock(topLevelUser))
        return false;
      continue;
    }

    // Any observable use after the fusion_region means the buffer escapes the
    // region boundary, so the final store must remain.
    Operation *topLevelUser =
        getTopLevelAncestorInBlock(owner, context.parentBlock);
    if (!topLevelUser) {
      if (areMutuallyExclusiveByIfRegion(localScopeOp, owner))
        continue;
      return false;
    }
    if (scheduledForErase.contains(topLevelUser))
      continue;
    if (topLevelUser == context.regionOp)
      continue;
    if (context.regionOp->isBeforeInBlock(topLevelUser))
      return false;
  }
  return true;
}

static bool elideLoadStoreRoundTripsInLeafBody(
    Block &body, const FusionRegionStoreContext *context, Operation *scopeOp) {
  SmallVector<Operation *, 8> eraseOrder;
  llvm::SmallPtrSet<Operation *, 8> scheduledForErase;
  SmallVector<TrackedStore, 8> trackedStores;
  bool changed = false;

  auto scheduleErase = [&](Operation *op) {
    if (scheduledForErase.insert(op).second)
      eraseOrder.push_back(op);
  };

  for (Operation &op : body.without_terminator()) {
    if (auto load = dyn_cast<pto::VldsOp>(op)) {
      Value inferredMask = inferVPTOLoadUserMask(load);
      if (!inferredMask) {
        // VPTO vlds does not carry an explicit predicate operand. If use-side
        // mask information is not uniquely recoverable, keep behavior
        // conservative by dropping only potentially aliasing tracked stores.
        pruneTrackedStoresForLoadBase(trackedStores, load.getSource());
        continue;
      }

      Value base = load.getSource();
      Value offset = load.getOffset();
      SmallVector<Value, 4> loadIndices{offset};
      int matchIndex =
          findTrackedStoreIndex(trackedStores, base, loadIndices, inferredMask);
      bool scalarBroadcast = false;
      if (matchIndex < 0) {
        matchIndex = findTrackedScalarBroadcastStoreIndex(trackedStores, load,
                                                          inferredMask);
        scalarBroadcast = matchIndex >= 0;
      }
      if (matchIndex >= 0) {
        Value forwarded = trackedStores[matchIndex].value;
        if (scalarBroadcast) {
          OpBuilder builder(load);
          // The physical user mask is commonly materialized after BRC_B32.
          // Insert the replacement immediately before the sole user so both
          // the producer vector and its mask dominate the vdup.
          builder.setInsertionPoint(*load.getResult().user_begin());
          forwarded = builder
                          .create<pto::VdupOp>(load.getLoc(),
                                               load.getResult().getType(),
                                               forwarded, inferredMask,
                                               builder.getStringAttr("LOWEST"))
                          .getResult();
        }
        load.getResult().replaceAllUsesWith(forwarded);
        scheduleErase(load);
        changed = true;
      } else {
        pruneTrackedStoresForLoadBase(trackedStores, base);
      }
      continue;
    }

    if (auto store = dyn_cast<pto::VstsOp>(op)) {
      Value base = store.getDestination();
      Value offset = store.getOffset();
      Value mask = store.getMask();
      SmallVector<Value, 4> storeIndices{offset};
      int matchIndex =
          findTrackedStoreIndex(trackedStores, base, storeIndices, mask);
      if (matchIndex >= 0) {
        scheduleErase(trackedStores[matchIndex].op);
        trackedStores.erase(trackedStores.begin() + matchIndex);
        changed = true;
      }

      trackedStores.push_back(TrackedStore{
          store.getOperation(),
          base,
          SmallVector<Value, 2>{offset},
          mask,
          store.getValue(),
      });
      continue;
    }

    if (!isPureNoRegionOp(&op) && !isAddressScaffoldOp(&op) &&
        !isa<pto::AllocTileOp>(op))
      trackedStores.clear();
  }

  if (context) {
    for (const TrackedStore &store : trackedStores) {
      if (!shouldElideTailStore(store, *context, scopeOp, scheduledForErase))
        continue;
      scheduleErase(store.op);
      changed = true;
    }
  }

  for (Operation *op : eraseOrder)
    op->erase();
  return changed;
}

static bool isLoopPipeline(scf::ForOp loop) {
  auto kind = loop->getAttrOfType<StringAttr>("pto.vmi.fusion.kind");
  return kind && kind.getValue() == "loop_pipeline";
}

static bool isUniqueVPTOMemoryPair(func::FuncOp func, pto::VstsOp store,
                                   pto::VldsOp load) {
  Value address = getCanonicalBufferAddress(store.getDestination());
  APInt staticAddress;
  if (!address || !matchPattern(address, m_ConstantInt(&staticAddress)))
    return false;

  bool unique = true;
  func.walk([&](Operation *op) {
    if (!unique || op == store.getOperation() || op == load.getOperation())
      return;
    StringRef opName = op->getName().getStringRef();
    if (opName == "pto.mem_bar" || opName.contains("sync") ||
        opName == "func.call") {
      unique = false;
      return;
    }
    if (auto otherStore = dyn_cast<pto::VstsOp>(op)) {
      if (areEquivalentValues(
              getCanonicalBufferAddress(otherStore.getDestination()), address))
        unique = false;
      return;
    }
    if (auto otherLoad = dyn_cast<pto::VldsOp>(op)) {
      if (areEquivalentValues(getCanonicalBufferAddress(otherLoad.getSource()),
                              address))
        unique = false;
      return;
    }
    // alloc_tile reserves a statically named handle but does not read or
    // write UB.  Different PTODSL Python launchers may preserve this dead
    // handle until the post-flatten cleanup, so treating it as an observable
    // access would make the proof environment-dependent.
    if (isa<pto::AllocTileOp>(op) || isAddressScaffoldOp(op) ||
        isMemoryEffectFree(op))
      return;
    for (Value operand : op->getOperands())
      if (areEquivalentValues(getCanonicalBufferAddress(operand), address)) {
        unique = false;
        return;
      }
  });
  return unique;
}

/// Forward a proven read-only preheader value after fusion regions have been
/// flattened.  The exact contract is one full-vector store, one full-vector
/// reload in the same block, and reload uses confined to a row-pipeline loop.
/// Requiring a unique static UB address prevents aliasing, escape, sync, or an
/// observable temporary from being partially rewritten.
static bool elideLoopInvariantPreheaderRoundTrip(func::FuncOp func) {
  SmallVector<scf::ForOp, 2> pipelines;
  func.walk([&](scf::ForOp loop) {
    if (isLoopPipeline(loop))
      pipelines.push_back(loop);
  });
  if (pipelines.size() != 1)
    return false;

  scf::ForOp loop = pipelines.front();
  Block *block = loop->getBlock();
  if (!block)
    return false;

  for (Operation &candidate : *block) {
    auto load = dyn_cast<pto::VldsOp>(candidate);
    if (!load || !load->isBeforeInBlock(loop))
      continue;
    if (load.getDist() && *load.getDist() != "NORM")
      continue;
    if (load.getResult().use_empty() ||
        !llvm::all_of(load.getResult().getUsers(), [&](Operation *user) {
          return loop->isProperAncestor(user);
        }))
      continue;

    Value inferredMask = inferVPTOLoadUserMask(load);
    if (!inferredMask || !isAllTruePatternMask(inferredMask))
      continue;

    for (Operation *previous = load->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      auto store = dyn_cast<pto::VstsOp>(previous);
      if (!store)
        continue;
      if (!areEquivalentValues(
              getCanonicalBufferAddress(store.getDestination()),
              getCanonicalBufferAddress(load.getSource())) ||
          !areEquivalentValues(store.getOffset(), load.getOffset()) ||
          !areEquivalentMaskValues(store.getMask(), inferredMask) ||
          store.getValue().getType() != load.getResult().getType() ||
          !isUniqueVPTOMemoryPair(func, store, load))
        continue;

      load.getResult().replaceAllUsesWith(store.getValue());
      load.erase();
      store.erase();
      return true;
    }
  }
  return false;
}

static bool isSyncOrUnknownCall(Operation *op) {
  if (!op)
    return true;
  StringRef name = op->getName().getStringRef();
  return name == "pto.mem_bar" || name.contains("sync") ||
         isa<func::CallOp>(op);
}

/// Forward a compact column state from one completed vector phase into the
/// next phase of the same fusion region.  Softmax max/sum are the motivating
/// cases: a loop returns the final column vreg, the ordinary standalone
/// candidate materializes it in UB, and the immediately following candidate
/// reloads exactly the same bytes before entering its own loop.
///
/// This deliberately does not look through a region-bearing operation between
/// the pair.  Consequently it cannot remove the exp matrix materialization:
/// that store is dynamic inside the exp/sum loop and its reload belongs to the
/// later divide loop.  Only a unique, static-address, top-level VST/VLD pair
/// separated by pure address/scalar scaffolding is eligible.
static bool
elideAdjacentPhaseStateRoundTrips(func::FuncOp func,
                                  pto::FusionRegionOp fusionRegion,
                                  const FusionRegionStoreContext &context) {
  Block &body = fusionRegion.getBody().front();
  SmallVector<Operation *, 8> eraseOrder;
  bool changed = false;

  auto hasUniqueMemoryPair = [&](pto::VstsOp store, pto::VldsOp load) {
    Value address = getCanonicalBufferAddress(store.getDestination());
    APInt staticAddress;
    if (!address || !matchPattern(address, m_ConstantInt(&staticAddress)))
      return false;
    if (aliasesYieldedBuffer(store.getDestination(), context.yieldedValues) ||
        aliasesAnyAddress(store.getDestination(),
                          context.externallyObservedAddresses))
      return false;

    bool unique = true;
    fusionRegion.walk([&](Operation *op) {
      if (!unique || op == store.getOperation() || op == load.getOperation())
        return;
      if (isSyncOrUnknownCall(op)) {
        unique = false;
        return;
      }
      if (auto otherStore = dyn_cast<pto::VstsOp>(op)) {
        if (areEquivalentValues(
                getCanonicalBufferAddress(otherStore.getDestination()),
                address))
          unique = false;
        return;
      }
      if (auto otherLoad = dyn_cast<pto::VldsOp>(op)) {
        if (areEquivalentValues(
                getCanonicalBufferAddress(otherLoad.getSource()), address))
          unique = false;
        return;
      }
      if (isa<pto::AllocTileOp>(op) || isAddressScaffoldOp(op) ||
          isMemoryEffectFree(op))
        return;
      for (Value operand : op->getOperands()) {
        if (areEquivalentValues(getCanonicalBufferAddress(operand), address)) {
          unique = false;
          return;
        }
      }
    });
    return unique;
  };

  SmallVector<Operation *, 32> topLevelOps;
  for (Operation &op : body.without_terminator())
    topLevelOps.push_back(&op);

  for (auto [index, candidate] : llvm::enumerate(topLevelOps)) {
    auto store = dyn_cast<pto::VstsOp>(candidate);
    if (!store || llvm::is_contained(eraseOrder, candidate))
      continue;

    for (Operation *next :
         ArrayRef<Operation *>(topLevelOps).drop_front(index + 1)) {
      if (auto load = dyn_cast<pto::VldsOp>(next)) {
        Value inferredMask = inferVPTOLoadUserMask(load);
        if (!inferredMask ||
            !areEquivalentValues(
                getCanonicalBufferAddress(store.getDestination()),
                getCanonicalBufferAddress(load.getSource())) ||
            !areEquivalentValues(store.getOffset(), load.getOffset()) ||
            !areEquivalentMaskValues(store.getMask(), inferredMask) ||
            store.getValue().getType() != load.getResult().getType() ||
            !hasUniqueMemoryPair(store, load))
          break;

        load.getResult().replaceAllUsesWith(store.getValue());
        eraseOrder.push_back(load.getOperation());
        eraseOrder.push_back(store.getOperation());
        changed = true;
        break;
      }

      // Do not cross a vector computation, control-flow phase, memory effect,
      // sync, or unknown call.  The intended state roundtrip has only pointer
      // casts, masks, and scalar address arithmetic between VST and VLD.
      if (next->getNumRegions() != 0 || isSyncOrUnknownCall(next) ||
          (!isAddressScaffoldOp(next) && !isPureNoRegionOp(next)))
        break;
    }
  }

  for (Operation *op : eraseOrder)
    if (op && op->getBlock())
      op->erase();
  return changed;
}

static Value getConservativeStaticAddressRoot(Value buffer) {
  Value address = getCanonicalBufferAddress(buffer);
  while (auto add =
             address ? address.getDefiningOp<pto::AddPtrOp>() : pto::AddPtrOp{})
    address = getCanonicalTrackedValue(add.getPtr());
  APInt staticAddress;
  if (!address || !matchPattern(address, m_ConstantInt(&staticAddress)))
    return {};
  return address;
}

/// Remove only post-flatten static UB stores whose value is already consumed
/// through SSA and whose address has no remaining read or non-vector observer.
/// This is intentionally narrower than general dead-store elimination: it is
/// the final cleanup for stores made redundant by proven state forwarding.
static bool elideUnobservedStaticVectorStores(func::FuncOp func) {
  bool hasFusionRegion = false;
  func.walk([&](pto::FusionRegionOp) { hasFusionRegion = true; });
  if (hasFusionRegion)
    return false;

  struct StoreGroup {
    Value root;
    SmallVector<pto::VstsOp, 4> stores;
    bool observed = false;
  };
  SmallVector<StoreGroup, 4> groups;

  func.walk([&](pto::VstsOp store) {
    bool hasSSAConsumer =
        llvm::any_of(store.getValue().getUsers(), [&](Operation *user) {
          return user != store.getOperation() && !isa<pto::VstsOp>(user);
        });
    if (!hasSSAConsumer)
      return;
    Value root = getConservativeStaticAddressRoot(store.getDestination());
    if (!root)
      return;
    auto it = llvm::find_if(groups, [&](const StoreGroup &group) {
      return areEquivalentValues(group.root, root);
    });
    if (it == groups.end()) {
      groups.push_back(StoreGroup{root, {}, false});
      it = std::prev(groups.end());
    }
    it->stores.push_back(store);
  });
  if (groups.empty())
    return false;

  auto operationUsesRoot = [&](Operation *op, Value root) {
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      Value operandRoot = getConservativeStaticAddressRoot(operand);
      return operandRoot && areEquivalentValues(operandRoot, root);
    });
  };

  func.walk([&](Operation *op) {
    if (isa<pto::VstsOp, pto::AllocTileOp>(op) || isAddressScaffoldOp(op) ||
        isMemoryEffectFree(op))
      return;
    for (StoreGroup &group : groups) {
      if (group.observed)
        continue;
      if (auto load = dyn_cast<pto::VldsOp>(op)) {
        Value root = getConservativeStaticAddressRoot(load.getSource());
        group.observed = root && areEquivalentValues(root, group.root);
        continue;
      }
      if (operationUsesRoot(op, group.root))
        group.observed = true;
    }
  });

  bool changed = false;
  for (StoreGroup &group : groups) {
    if (group.observed)
      continue;
    for (pto::VstsOp store : group.stores) {
      if (store && store->getBlock()) {
        store.erase();
        changed = true;
      }
    }
  }
  return changed;
}

struct PTOFusionLoadStoreElisionPass
    : public pto::impl::PTOFusionLoadStoreElisionBase<
          PTOFusionLoadStoreElisionPass> {
  using pto::impl::PTOFusionLoadStoreElisionBase<
      PTOFusionLoadStoreElisionPass>::PTOFusionLoadStoreElisionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    bool changed = false;
    changed |= elideRedundantLoadsAcrossPromotedPipeSync(func);
    changed |= elideLoopInvariantPreheaderRoundTrip(func);
    func.walk([&](pto::FusionRegionOp fusionRegion) {
      changed |= normalizeFusionRegionYieldFrontier(fusionRegion);
    });

    // Run before the leaf-body fixed point: phase-state forwarding exposes
    // dead pointer scaffolding and removes the only UB hazards between the
    // column reduction phases.
    func.walk([&](pto::FusionRegionOp fusionRegion) {
      std::optional<FusionRegionStoreContext> context =
          buildFusionRegionStoreContext(fusionRegion);
      if (!context)
        return;
      changed |=
          elideAdjacentPhaseStateRoundTrips(func, fusionRegion, *context);
    });

    auto runElisionRound = [&]() {
      bool roundChanged = false;
      llvm::DenseMap<Operation *, FusionRegionStoreContext> regionContexts;
      func.walk([&](pto::FusionRegionOp fusionRegion) {
        std::optional<FusionRegionStoreContext> context =
            buildFusionRegionStoreContext(fusionRegion);
        if (!context)
          return;
        regionContexts.try_emplace(fusionRegion.getOperation(),
                                   std::move(*context));
      });

      func.walk([&](pto::FusionRegionOp fusionRegion) {
        auto it = regionContexts.find(fusionRegion.getOperation());
        if (it == regionContexts.end())
          return;

        Block &body = fusionRegion.getBody().front();
        if (!isSupportedStraightLineBlock(body))
          return;

        roundChanged |=
            elideLoadStoreRoundTripsInLeafBody(body, &it->second, nullptr);
      });

      auto runElisionForLeafBody = [&](Block *leafBody, Operation *scopeOp,
                                       pto::FusionRegionOp fusionRegion) {
        if (!leafBody)
          return;

        const FusionRegionStoreContext *context = nullptr;
        if (fusionRegion) {
          auto it = regionContexts.find(fusionRegion.getOperation());
          if (it == regionContexts.end())
            return;
          context = &it->second;
        }

        roundChanged |=
            elideLoadStoreRoundTripsInLeafBody(*leafBody, context, scopeOp);
      };

      func.walk([&](pto::VecScopeOp vecscope) {
        if (auto fusionRegion =
                vecscope->getParentOfType<pto::FusionRegionOp>()) {
          if (isSupportedStraightLineBlock(vecscope.getBody().front()))
            runElisionForLeafBody(&vecscope.getBody().front(), vecscope,
                                  fusionRegion);
        }
      });
      func.walk([&](pto::StrictVecScopeOp vecscope) {
        if (auto fusionRegion =
                vecscope->getParentOfType<pto::FusionRegionOp>()) {
          if (isSupportedStraightLineBlock(vecscope.getBody().front()))
            runElisionForLeafBody(&vecscope.getBody().front(), vecscope,
                                  fusionRegion);
        }
      });

      func.walk([&](scf::ForOp loop) {
        if (!isSupportedLoopRoot(loop))
          return;
        runElisionForLeafBody(getLeafLoopBody(loop), loop.getOperation(),
                              loop->getParentOfType<pto::FusionRegionOp>());
      });
      return roundChanged;
    };

    // Forwarding can kill a later phase's pointer-only chain. Rebuild the
    // escape frontier after cleaning that chain, then reconsider tail stores.
    // Each successful round erases at least one operation, so this reaches a
    // finite fixed point without weakening alias, mask, or yielded-frontier
    // proofs.
    while (true) {
      bool roundChanged = runElisionRound();
      bool scaffoldChanged = eraseDeadAddressScaffoldOps(func);
      changed |= roundChanged || scaffoldChanged;
      if (!roundChanged && !scaffoldChanged)
        break;
    }

    changed |= elideUnobservedStaticVectorStores(func);

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionLoadStoreElisionPass() {
  return std::make_unique<PTOFusionLoadStoreElisionPass>();
}
