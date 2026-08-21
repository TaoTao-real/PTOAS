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

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONLOADSTOREELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

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
      std::optional<int64_t> operandLanes =
          getStaticPrefixActiveLanes(operand);
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

static bool isSupportedLeafOp(Operation *op) {
  if (isa<pto::VldsOp, pto::VstsOp>(op))
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
  return isa<pto::FusionRegionOp, pto::VecScopeOp, pto::StrictVecScopeOp>(
      loop->getParentOp());
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
      if (matchIndex >= 0) {
        load.getResult().replaceAllUsesWith(trackedStores[matchIndex].value);
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

    if (!isPureNoRegionOp(&op) && !isAddressScaffoldOp(&op))
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
    func.walk([&](pto::FusionRegionOp fusionRegion) {
      changed |= normalizeFusionRegionYieldFrontier(fusionRegion);
    });

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

      changed |= elideLoadStoreRoundTripsInLeafBody(body, &it->second, nullptr);
    });

    auto runElisionForLeafBody = [&](Block *leafBody, Operation *scopeOp,
                                     pto::FusionRegionOp fusionRegion) {
      if (!leafBody || !fusionRegion)
        return;

      auto it = regionContexts.find(fusionRegion.getOperation());
      if (it == regionContexts.end())
        return;

      changed |=
          elideLoadStoreRoundTripsInLeafBody(*leafBody, &it->second, scopeOp);
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

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionLoadStoreElisionPass() {
  return std::make_unique<PTOFusionLoadStoreElisionPass>();
}
