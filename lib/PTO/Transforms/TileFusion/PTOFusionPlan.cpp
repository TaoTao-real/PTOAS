// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/TileFusion/FusionAnalysis.h"
#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VmiMemoryLocation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>

#ifdef PTO_ENABLE_VFSIM_IR_PLANNER
#include "native/IRPlanner.h"
#endif

namespace mlir {
namespace pto {
// Passes.h (included above) pulls in the global GEN_PASS_DECL block, which
// defines GEN_PASS_DECL_FUSIONPLAN and leaves it set.  Undef it before
// re-including the .inc for GEN_PASS_DEF so the options struct is not defined
// twice.
#undef GEN_PASS_DECL_FUSIONPLAN
#define GEN_PASS_DEF_FUSIONPLAN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kFusionRowUnrollAttr =
    "pto.fusion.row_unroll_factor";
static constexpr llvm::StringLiteral kFusionColUnrollAttr =
    "pto.fusion.col_unroll_factor";

struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};

struct PlanningContext {
  const pto::FusionBlockAnalysis &blockAnalysis;
};

struct PlanningCost {
  int64_t dependencyBenefit = 0;
  int64_t loopMergeBenefit = 0;
  int64_t liveTilePenalty = 0;
  int64_t vfParameterPenalty = 0;
  bool rejectedForDynamicShape = false;

  int64_t total() const {
    return dependencyBenefit + loopMergeBenefit - liveTilePenalty -
           vfParameterPenalty;
  }
};

struct PlanningDecision {
  bool accept = false;
  PlanningCost cost;
};

static bool isCurrentlyPlannableOp(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tmul", "tdiv", "tadd", "tsub", "tmax", "tmin", true)
      .Cases("tmuls", "tdivs", "tadds", "tsubs", "tmaxs", "tmins", true)
      .Case("texp", true)
      .Case("texpands", true)
      .Cases("trowexpandsub", "trowexpandmul", "trowexpanddiv", true)
      .Cases("trowsum", "trowmax", "trowmin", true)
      .Cases("tcolsum", "tcolmax", "tcolmin", true)
      .Default(false);
}

static bool isProvenIterationDomain(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &node) {
  if (node.iterationDomainClass >= blockAnalysis.iterationDomainClasses.size()) {
    return false;
  }
  return blockAnalysis.iterationDomainClasses[node.iterationDomainClass]
             .info.proof == pto::IterationDomainProof::Proven;
}

static bool hasHardBoundaryBetween(const pto::FusionComputeNode &a,
                                   const pto::FusionComputeNode &b) {
  const pto::FusionComputeNode &earlier =
      a.blockOrder < b.blockOrder ? a : b;
  const pto::FusionComputeNode &later =
      a.blockOrder < b.blockOrder ? b : a;

  Operation *cursor = earlier.op->getNextNode();
  while (cursor && cursor != later.op) {
    if (cursor->hasTrait<OpTrait::IsTerminator>() ||
        !cursor->getRegions().empty() || isa<CallOpInterface>(cursor)) {
      return true;
    }
    cursor = cursor->getNextNode();
  }
  return false;
}

static bool hasHardBoundaryToGroup(
    ArrayRef<const pto::FusionComputeNode *> group,
    const pto::FusionComputeNode &candidate) {
  for (const pto::FusionComputeNode *member : group) {
    if (hasHardBoundaryBetween(*member, candidate)) {
      return true;
    }
  }
  return false;
}

static bool dependsOnPreviousNode(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &previous,
    const pto::FusionComputeNode &current) {
  for (unsigned edgeId : current.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size()) {
      continue;
    }
    if (blockAnalysis.edges[edgeId].producerNode == previous.id) {
      return true;
    }
  }

  for (Value output : previous.semantics.tileOutputs) {
    if (llvm::is_contained(current.semantics.tileInputs, output)) {
      return true;
    }
  }

  return false;
}

static SmallVector<const pto::FusionComputeNode *, 8>
buildStableInGroupOrder(ArrayRef<const pto::FusionComputeNode *> members) {
  SmallVector<const pto::FusionComputeNode *, 8> ordered(members.begin(),
                                                         members.end());
  llvm::stable_sort(ordered, [](const pto::FusionComputeNode *lhs,
                                const pto::FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder) {
      return lhs->blockOrder < rhs->blockOrder;
    }
    return lhs->id < rhs->id;
  });
  return ordered;
}

static void assignStableGroupMetadata(ArrayRef<PlannedFusionGroup> groups,
                                      MLIRContext *ctx,
                                      int64_t &nextGroupId) {
  SmallVector<const PlannedFusionGroup *, 8> orderedGroups;
  orderedGroups.reserve(groups.size());
  for (const PlannedFusionGroup &group : groups) {
    orderedGroups.push_back(&group);
  }

  llvm::stable_sort(orderedGroups, [](const PlannedFusionGroup *lhs,
                                      const PlannedFusionGroup *rhs) {
    const pto::FusionComputeNode *lhsFirst = lhs->members.front();
    const pto::FusionComputeNode *rhsFirst = rhs->members.front();
    if (lhsFirst->blockOrder != rhsFirst->blockOrder) {
      return lhsFirst->blockOrder < rhsFirst->blockOrder;
    }
    return lhsFirst->id < rhsFirst->id;
  });

  for (const PlannedFusionGroup *group : orderedGroups) {
    const int64_t groupId = nextGroupId++;
    SmallVector<const pto::FusionComputeNode *, 8> stableOrder =
        buildStableInGroupOrder(group->members);
    for (auto [order, node] : llvm::enumerate(stableOrder)) {
      node->op->setAttr(kFusionGroupIdAttr,
                        IntegerAttr::get(IntegerType::get(ctx, mlir::pto::kValue64), groupId));
      node->op->setAttr(
          kFusionOrderAttr,
          IntegerAttr::get(IntegerType::get(ctx, mlir::pto::kValue64),
                           static_cast<int64_t>(order)));
    }
  }
}

static bool isSupportedPlanningNode(const pto::FusionComputeNode &node) {
  return node.semantics.kind == pto::FusionOpKind::Compute &&
         isCurrentlyPlannableOp(node.semantics.opName);
}

static unsigned
countEdgesFromGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                    ArrayRef<const pto::FusionComputeNode *> group,
                    const pto::FusionComputeNode &candidate) {
  DenseSet<unsigned> producerIds;
  for (const pto::FusionComputeNode *member : group) {
    producerIds.insert(member->id);
  }

  unsigned count = 0;
  for (unsigned edgeId : candidate.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size()) {
      continue;
    }
    if (producerIds.contains(blockAnalysis.edges[edgeId].producerNode)) {
      ++count;
    }
  }
  return count;
}

struct GroupFootprint {
  unsigned liveTileCount = 0;
  unsigned vfParameterCount = 0;
};

static bool nodesHaveDirectDataFlowConnection(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &lhs, const pto::FusionComputeNode &rhs) {
  for (unsigned edgeId : lhs.outgoingEdges) {
    if (edgeId >= blockAnalysis.edges.size()) {
      continue;
    }
    if (blockAnalysis.edges[edgeId].consumerNode == rhs.id) {
      return true;
    }
  }

  for (unsigned edgeId : lhs.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size()) {
      continue;
    }
    if (blockAnalysis.edges[edgeId].producerNode == rhs.id) {
      return true;
    }
  }

  for (Value output : lhs.semantics.tileOutputs) {
    if (llvm::is_contained(rhs.semantics.tileInputs, output)) {
      return true;
    }
  }

  for (Value output : rhs.semantics.tileOutputs) {
    if (llvm::is_contained(lhs.semantics.tileInputs, output)) {
      return true;
    }
  }

  return false;
}

static unsigned
countConnectionsToGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                        ArrayRef<const pto::FusionComputeNode *> group,
                        const pto::FusionComputeNode &candidate) {
  unsigned connections = 0;
  for (const pto::FusionComputeNode *member : group) {
    if (nodesHaveDirectDataFlowConnection(blockAnalysis, *member, candidate)) {
      ++connections;
    }
  }
  return connections;
}

static GroupFootprint
computeGroupFootprint(ArrayRef<const pto::FusionComputeNode *> members) {
  DenseSet<Value> producedTiles;
  DenseSet<Value> touchedTiles;
  DenseSet<Value> externalInputs;

  for (const pto::FusionComputeNode *member : members) {
    for (Value output : member->semantics.tileOutputs) {
      producedTiles.insert(output);
      touchedTiles.insert(output);
    }
  }

  for (const pto::FusionComputeNode *member : members) {
    for (Value input : member->semantics.tileInputs) {
      touchedTiles.insert(input);
      if (!producedTiles.contains(input)) {
        externalInputs.insert(input);
      }
    }
  }

  GroupFootprint footprint;
  footprint.liveTileCount = touchedTiles.size();
  footprint.vfParameterCount = externalInputs.size() + producedTiles.size();
  return footprint;
}

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                        const pto::FusionComputeNode &candidate)
      const = 0;

  virtual PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const = 0;
};

class ConservativeGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate)) {
      return decision;
    }

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept) {
      return seedDecision;
    }

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    const pto::FusionComputeNode &previous = *currentGroup.back();
    const bool sameDomainClass =
        previous.iterationDomainClass == candidate.iterationDomainClass;
    const bool contiguousInBlock =
        candidate.blockOrder == previous.blockOrder + 1;
    const bool directlyDependent =
        dependsOnPreviousNode(ctx.blockAnalysis, previous, candidate);
    if (!sameDomainClass || !contiguousInBlock || !directlyDependent) {
      return decision;
    }

    SmallVector<const pto::FusionComputeNode *, 8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    GroupFootprint footprint = computeGroupFootprint(proposedGroup);

    decision.cost.dependencyBenefit =
        mlir::pto::kValue4 * static_cast<int64_t>(
                countEdgesFromGroup(ctx.blockAnalysis, currentGroup, candidate));
    decision.cost.loopMergeBenefit = mlir::pto::kValue2;
    decision.cost.liveTilePenalty =
        std::max<int64_t>(0, static_cast<int64_t>(footprint.liveTileCount) - mlir::pto::kValue4);
    decision.cost.vfParameterPenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.vfParameterCount) - mlir::pto::kValue6);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class ConservativeDAGGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate)) {
      return decision;
    }

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept) {
      return seedDecision;
    }

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    if (currentGroup.front()->iterationDomainClass !=
        candidate.iterationDomainClass) {
      return decision;
    }

    if (hasHardBoundaryToGroup(currentGroup, candidate)) {
      return decision;
    }

    const unsigned connectionCount =
        countConnectionsToGroup(ctx.blockAnalysis, currentGroup, candidate);
    if (connectionCount == 0) {
      return decision;
    }

    SmallVector<const pto::FusionComputeNode *, mlir::pto::kValue8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    GroupFootprint footprint = computeGroupFootprint(proposedGroup);

    decision.cost.dependencyBenefit = mlir::pto::kValue4 * static_cast<int64_t>(connectionCount);
    decision.cost.loopMergeBenefit = mlir::pto::kValue4;
    decision.cost.liveTilePenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.liveTileCount) - mlir::pto::kValue10);
    decision.cost.vfParameterPenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.vfParameterCount) - mlir::pto::kValue12);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class StrategyEngine {
public:
  virtual ~StrategyEngine() = default;

  virtual SmallVector<PlannedFusionGroup, mlir::pto::kValue8>
  planBlock(const PlanningContext &ctx, const CostModel &costModel) const = 0;
};

class ConservativeGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, mlir::pto::kValue8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, mlir::pto::kValue8> groups;
    SmallVector<const pto::FusionComputeNode *, 8> chain;

    auto flushChain = [&chain, &groups]() {
      if (chain.size() < 2) {
        chain.clear();
        return;
      }

      PlannedFusionGroup group;
      group.members = chain;
      groups.push_back(std::move(group));
      chain.clear();
    };

    for (const pto::FusionComputeNode &node : ctx.blockAnalysis.computeNodes) {
      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, node);
      if (!seedDecision.accept) {
        flushChain();
        continue;
      }

      if (chain.empty()) {
        chain.push_back(&node);
        continue;
      }

      PlanningDecision appendDecision =
          costModel.evaluateAppend(ctx, chain, node);
      if (!appendDecision.accept) {
        flushChain();
        chain.push_back(&node);
        continue;
      }

      chain.push_back(&node);
    }

    flushChain();
    return groups;
  }
};

class ConservativeDAGGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, mlir::pto::kValue8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, mlir::pto::kValue8> groups;
    DenseSet<unsigned> assignedNodes;

    for (const pto::FusionComputeNode &seed : ctx.blockAnalysis.computeNodes) {
      if (assignedNodes.contains(seed.id)) {
        continue;
      }

      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, seed);
      if (!seedDecision.accept) {
        continue;
      }

      SmallVector<const pto::FusionComputeNode *, 8> groupMembers;
      DenseSet<unsigned> groupNodeIds;
      groupMembers.push_back(&seed);
      groupNodeIds.insert(seed.id);

      bool changed = true;
      while (changed) {
        changed = false;
        for (const pto::FusionComputeNode &candidate :
             ctx.blockAnalysis.computeNodes) {
          if (assignedNodes.contains(candidate.id) ||
              groupNodeIds.contains(candidate.id)) {
            continue;
          }

          PlanningDecision appendDecision =
              costModel.evaluateAppend(ctx, groupMembers, candidate);
          if (!appendDecision.accept) {
            continue;
          }

          groupMembers.push_back(&candidate);
          groupNodeIds.insert(candidate.id);
          changed = true;
        }
      }

      if (groupMembers.size() < mlir::pto::kValue2) {
        continue;
      }

      PlannedFusionGroup group;
      group.members = buildStableInGroupOrder(groupMembers);
      groups.push_back(group);
      for (const pto::FusionComputeNode *member : group.members) {
        assignedNodes.insert(member->id);
      }
    }

    return groups;
  }
};

// VMI planning is deliberately driven by the selected provider, not by an
// op-name whitelist.  A PTODSL fallback therefore closes the VMI region even
// when it has the same tile op name as a supported VMI implementation.
//
// On top of that provenance gate, the VMI strategy partitions by memory
// access: a fusion region reorders and interleaves its members, so two nodes
// may only share one when their tile accesses are either proven must-alias
// (identical byte interval -- the dataflow graph already orders them) or
// proven no-alias (disjoint intervals, or distinct allocations).  Anything
// merely *possibly* overlapping -- partial overlap, an unknown extent, an
// unknown offset, an unattributable storage root -- must land in a different
// region.  The non-VMI strategies are untouched by this rule.
static SmallVector<pto::VmiTileAccess, 8>
collectVmiTileAccesses(const pto::FusionComputeNode &node) {
  SmallVector<pto::VmiTileAccess, 8> accesses;
  auto collect = [&](ArrayRef<Value> values) {
    for (Value value : values) {
      if (auto access = pto::resolveVmiTileAccess(value)) {
        accesses.push_back(*access);
      }
    }
  };
  collect(node.semantics.tileInputs);
  collect(node.semantics.tileOutputs);
  return accesses;
}

static bool areVmiAccessesRegionCompatible(const pto::FusionComputeNode &a,
                                           const pto::FusionComputeNode &b) {
  SmallVector<pto::VmiTileAccess, 8> lhs = collectVmiTileAccesses(a);
  SmallVector<pto::VmiTileAccess, 8> rhs = collectVmiTileAccesses(b);
  for (const pto::VmiTileAccess &l : lhs) {
    for (const pto::VmiTileAccess &r : rhs) {
      if (pto::mayAliasVmiTileAccess(l, r)) {
        return false;
      }
    }
  }
  return true;
}

static bool hasIncompatibleAccessWithRun(
    ArrayRef<const pto::FusionComputeNode *> run,
    const pto::FusionComputeNode &candidate) {
  for (const pto::FusionComputeNode *member : run) {
    if (!areVmiAccessesRegionCompatible(*member, candidate)) {
      return true;
    }
  }
  return false;
}

class VMISelectedStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &) const override {
    const auto &block = ctx.blockAnalysis;
    SmallVector<PlannedFusionGroup, 8> groups;
    SmallVector<const pto::FusionComputeNode *, 8> current;
    auto flush = [&]() {
      if (!current.empty()) {
        groups.push_back({buildStableInGroupOrder(current)});
        current.clear();
      }
    };
    // VMI F3-adjacency run building: every compute node joins the current loose
    // group; only a non-plannable op sitting between two compute nodes closes
    // it. The resulting fusion_region is a container, not a fusion mandate —
    // PTOVmiLoopFusion still decides which same-header loops may actually
    // merge, and rejects any run whose UB exchange is not a proven
    // same-iteration injective-affine transfer.
    DenseMap<unsigned, bool> precededByNonPlannable;
    DenseMap<Operation *, unsigned> nodeIdByOp;
    for (const pto::FusionComputeNode &n : block.computeNodes) {
      nodeIdByOp[n.op] = n.id;
    }
    bool sawCompute = false;
    bool hardBoundarySinceCompute = false;
    for (Operation &op : *block.block) {
      auto it = nodeIdByOp.find(&op);
      if (it != nodeIdByOp.end()) {
        precededByNonPlannable[it->second] =
            sawCompute && hardBoundarySinceCompute;
        sawCompute = true;
        hardBoundarySinceCompute = false;
        continue;
      }
      if (pto::isFusionTransparentScaffold(&op)) {
        continue;
      }
      FailureOr<pto::FusionOpSemantics> semanticsOr =
          pto::getFusionOpSemantics(&op);
      if (failed(semanticsOr) ||
          semanticsOr->kind == pto::FusionOpKind::HardBoundary ||
          (semanticsOr->kind == pto::FusionOpKind::LocalBoundary &&
           !op.hasAttr("pto.vmi.fusion.boundary"))) {
        hardBoundarySinceCompute = true;
      }
    }

    for (const pto::FusionComputeNode &node : block.computeNodes) {
      auto precIt = precededByNonPlannable.find(node.id);
      if (precIt != precededByNonPlannable.end() && precIt->second) {
        flush();
      }
      // Second guard: never co-locate a node whose tile accesses may overlap a
      // member's. The historical container semantics leave this to the loop
      // fusion pass; keeping it here narrows the container to accesses that are
      // must-alias or provably disjoint.
      if (hasIncompatibleAccessWithRun(current, node)) {
        flush();
      }
      current.push_back(&node);
    }
    flush();
    return groups;
  }
};

static void clearPlanningAttrs(func::FuncOp func) {
  func.walk([](Operation *op) {
    op->removeAttr(kFusionGroupIdAttr);
    op->removeAttr(kFusionOrderAttr);
    op->removeAttr(kFusionRowUnrollAttr);
    op->removeAttr(kFusionColUnrollAttr);
  });
}

static LogicalResult runVfSimFusionPlanner(func::FuncOp func,
                                           bool dumpCandidates) {
#ifdef PTO_ENABLE_VFSIM_IR_PLANNER
  vfsim::PlannerOptions options;
  options.dumpCandidates = dumpCandidates;
  return vfsim::planTileFusionIR(func.getOperation(), options);
#else
  return func.emitError()
         << "--enable-vfsim-costmodel-optimization requires configuring PTOAS with "
            "-DPTO_ENABLE_VFSIM_COSTMODEL=ON";
#endif
}

struct FusionPlanPass : public pto::impl::FusionPlanBase<FusionPlanPass> {
  using pto::impl::FusionPlanBase<FusionPlanPass>::FusionPlanBase;

  FusionPlanPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    clearPlanningAttrs(func);

    // Reuse the shared (analysis-manager-cached) pre-fusion dataflow graph
    // rather than rebuilding it from scratch.  The DFG — compute nodes, edges,
    // liveness, write instances — is identical regardless of whether shape
    // inference is enabled, so it is built once by PreFusionAnalysis and shared
    // with FusionRegionGen via the analysis manager.  Only iteration-domain
    // inference depends on the --enable-shape-inference option, so we run that
    // separable step ourselves on a local copy of the cached graph.
    const pto::PreFusionAnalysis &sharedAnalysis =
        getAnalysis<pto::PreFusionAnalysis>();
    if (!sharedAnalysis.isValid()) {
      signalPassFailure();
      return;
    }
    pto::PreFusionAnalysisResult analysis = sharedAnalysis.getResult();
    if (failed(pto::inferIterationDomainClasses(analysis, enableShapeInference))) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    int64_t nextGroupId = 0;
    ConservativeDAGGreedyCostModel costModel;
    std::unique_ptr<StrategyEngine> strategyEngine;
    if (strategy.getValue() == "vmi-ub-disjoint") {
      strategyEngine = std::make_unique<VMISelectedStrategyEngine>();
    } else if (strategy.getValue() == "conservative-dag-greedy") {
      strategyEngine = std::make_unique<ConservativeDAGGreedyStrategyEngine>();
    } else {
      emitError(getOperation()->getLoc())
          << "unknown fusion strategy " << strategy.getValue();
      signalPassFailure();
      return;
    }

    for (const pto::FusionBlockAnalysis &blockAnalysis : analysis.blocks) {
      PlanningContext planningCtx{blockAnalysis};
      SmallVector<PlannedFusionGroup, mlir::pto::kValue8> groups =
          strategyEngine->planBlock(planningCtx, costModel);
      assignStableGroupMetadata(groups, ctx, nextGroupId);
    }

    if (enableVfSimCostmodelOptimization &&
        failed(runVfSimFusionPlanner(func, dumpVfSimUnrollTest))) {
      signalPassFailure();
      return;
    }

    // The fusion metadata we annotate (group_id/order) is a planning *output*;
    // it does not alter tile semantics, operand types, aliasing or liveness,
    // so it cannot invalidate the shared PreFusionAnalysis DFG.  Preserve it so
    // the downstream FusionRegionGen pass reuses the cached graph instead of
    // rebuilding it.
    markAnalysesPreserved<pto::PreFusionAnalysis>();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createFusionPlanPass() {
  return std::make_unique<FusionPlanPass>();
}

std::unique_ptr<Pass>
mlir::pto::createFusionPlanPass(const pto::FusionPlanOptions &options) {
  return std::make_unique<FusionPlanPass>(options);
}
