// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "pto-insert-sync-analysis"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {

static size_t countCompoundSlotsForRoot(const CompoundInstanceElement *compound,
                                        Value sharedRoot, int sharedGroup,
                                        int factor) {
  if (!compound || !sharedRoot || factor <= 1)
    return 0;

  llvm::SmallDenseSet<int, MAX_MULTI_BUFFER_NUM> slots;
  auto collect = [&](const SmallVector<const BaseMemInfo *> &vec) {
    for (const BaseMemInfo *info : vec) {
      if (!info || !info->isMultibufferSlotValid)
        continue;
      if (info->multibufferRoot != sharedRoot)
        continue;
      if (info->multibufferGroup != sharedGroup)
        continue;
      if (info->multibufferFactor != factor)
        continue;
      if (info->multibufferSlot < 0 || info->multibufferSlot >= factor)
        continue;
      slots.insert(info->multibufferSlot);
    }
  };
  collect(compound->defVec);
  collect(compound->useVec);
  return slots.size();
}

struct BranchMultibufferIdentity {
  Value root{nullptr};
  int group{0};
  int factor{1};
};

static bool matchesMultibufferIdentity(
    const BaseMemInfo *info, const BranchMultibufferIdentity &identity) {
  if (!info || !info->isMultibufferSlotValid)
    return false;
  if (info->multibufferRoot != identity.root)
    return false;
  if (info->multibufferGroup != identity.group)
    return false;
  if (info->multibufferFactor != identity.factor)
    return false;
  return info->multibufferSlot >= 0 && info->multibufferSlot < identity.factor;
}

static SmallVector<std::pair<const void *, int>>
canonicalizeCompoundRootGroups(
    const SmallVector<std::pair<const void *, int>> &roots) {
  SmallVector<std::pair<const void *, int>> result = roots;
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    return lhs.second < rhs.second;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

struct BranchCompoundSignature {
  std::string opName;
  PipelineType pipe{PipelineType::PIPE_UNASSIGNED};
  int targetDefCount{0};
  int targetUseCount{0};
  SmallVector<std::pair<const void *, int>> otherDefRoots;
  SmallVector<std::pair<const void *, int>> otherUseRoots;

  bool operator==(const BranchCompoundSignature &other) const {
    return opName == other.opName && pipe == other.pipe &&
           targetDefCount == other.targetDefCount &&
           targetUseCount == other.targetUseCount &&
           otherDefRoots == other.otherDefRoots &&
           otherUseRoots == other.otherUseRoots;
  }
};

struct BranchSelectorFamilyCandidate {
  Operation *ifOp{nullptr};
  int branchBeginId{-1};
  Operation *representativeOp{nullptr};
};

static std::optional<BranchCompoundSignature> buildBranchCompoundSignature(
    const CompoundInstanceElement *compound,
    const BranchMultibufferIdentity &identity) {
  if (!compound)
    return std::nullopt;

  BranchCompoundSignature signature;
  signature.opName = compound->opName.getStringRef().str();
  signature.pipe = compound->kPipeValue;

  auto collect = [&](const SmallVector<const BaseMemInfo *> &vec, bool isDef) {
    auto &targetCount =
        isDef ? signature.targetDefCount : signature.targetUseCount;
    auto &otherRoots =
        isDef ? signature.otherDefRoots : signature.otherUseRoots;
    for (const BaseMemInfo *info : vec) {
      if (!info)
        continue;
      if (matchesMultibufferIdentity(info, identity)) {
        ++targetCount;
        continue;
      }
      if (info->isMultibufferSlotValid && info->multibufferRoot) {
        otherRoots.emplace_back(info->multibufferRoot.getAsOpaquePointer(),
                                info->multibufferGroup);
      } else if (info->rootBuffer) {
        otherRoots.emplace_back(info->rootBuffer.getAsOpaquePointer(), -1);
      }
    }
  };
  collect(compound->defVec, /*isDef=*/true);
  collect(compound->useVec, /*isDef=*/false);
  signature.otherDefRoots =
      canonicalizeCompoundRootGroups(signature.otherDefRoots);
  signature.otherUseRoots =
      canonicalizeCompoundRootGroups(signature.otherUseRoots);

  if (signature.targetDefCount + signature.targetUseCount == 0)
    return std::nullopt;
  return signature;
}

static std::optional<int> getCompoundSingleTargetSlot(
    const CompoundInstanceElement *compound,
    const BranchMultibufferIdentity &identity) {
  if (!compound)
    return std::nullopt;
  llvm::SmallDenseSet<int, MAX_MULTI_BUFFER_NUM> slots;
  auto collect = [&](const SmallVector<const BaseMemInfo *> &vec) {
    for (const BaseMemInfo *info : vec) {
      if (matchesMultibufferIdentity(info, identity))
        slots.insert(info->multibufferSlot);
    }
  };
  collect(compound->defVec);
  collect(compound->useVec);
  if (slots.size() != 1)
    return std::nullopt;
  return *slots.begin();
}

static const BranchInstanceElement *
findIfBeginElementForOp(const SyncIRs &syncIR, Operation *ifOp) {
  if (!ifOp)
    return nullptr;
  for (const auto &element : syncIR) {
    auto *branch = dyn_cast<BranchInstanceElement>(element.get());
    if (!branch || branch->getBranchKind() != KindOfBranch::IF_BEGIN)
      continue;
    if (branch->elementOp == ifOp)
      return branch;
  }
  return nullptr;
}

static SmallVector<BranchSelectorFamilyCandidate> collectBranchSelectorFamilies(
    const SyncIRs &syncIR, const CompoundInstanceElement *reference,
    const BranchMultibufferIdentity &identity, scf::ForOp ownerLoop) {
  SmallVector<BranchSelectorFamilyCandidate> families;
  if (!reference || !reference->elementOp || !ownerLoop)
    return families;

  auto referenceSignature = buildBranchCompoundSignature(reference, identity);
  auto referenceSlot = getCompoundSingleTargetSlot(reference, identity);
  if (!referenceSignature || !referenceSlot)
    return families;

  for (Operation *parent = reference->elementOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == ownerLoop.getOperation())
      break;
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp)
      continue;

    const BranchInstanceElement *ifBegin =
        findIfBeginElementForOp(syncIR, ifOp.getOperation());
    if (!ifBegin)
      continue;

    llvm::SmallDenseMap<int, const CompoundInstanceElement *, MAX_MULTI_BUFFER_NUM>
        slotToCompound;
    bool invalidFamily = false;
    for (unsigned idx = ifBegin->GetIndex() + 1; idx < ifBegin->endId; ++idx) {
      auto *candidate = dyn_cast<CompoundInstanceElement>(syncIR[idx].get());
      if (!candidate)
        continue;
      auto candidateSignature = buildBranchCompoundSignature(candidate, identity);
      if (!candidateSignature ||
          !(*candidateSignature == *referenceSignature))
        continue;
      auto candidateSlot = getCompoundSingleTargetSlot(candidate, identity);
      if (!candidateSlot) {
        invalidFamily = true;
        break;
      }
      if (slotToCompound.count(*candidateSlot)) {
        invalidFamily = true;
        break;
      }
      slotToCompound[*candidateSlot] = candidate;
    }

    if (!invalidFamily &&
        slotToCompound.size() == static_cast<size_t>(identity.factor) &&
        slotToCompound.count(*referenceSlot)) {
      Operation *representativeOp = nullptr;
      for (int slot = 0; slot < identity.factor; ++slot) {
        auto it = slotToCompound.find(slot);
        if (it == slotToCompound.end() || !it->second)
          continue;
        representativeOp = it->second->elementOp;
        break;
      }
      families.push_back(
          {ifOp.getOperation(), static_cast<int>(ifBegin->GetIndex()),
           representativeOp});
    }
  }

  return families;
}

static std::optional<BranchSelectorFamilyCandidate> findCommonBranchSelectorFamily(
    const SyncIRs &syncIR, const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    const BranchMultibufferIdentity &identity, scf::ForOp ownerLoop) {
  SmallVector<BranchSelectorFamilyCandidate> nowFamilies =
      collectBranchSelectorFamilies(syncIR, nowCompound, identity, ownerLoop);
  if (nowFamilies.empty())
    return std::nullopt;

  SmallVector<BranchSelectorFamilyCandidate> frontFamilies =
      collectBranchSelectorFamilies(syncIR, frontCompound, identity, ownerLoop);
  if (frontFamilies.empty())
    return std::nullopt;

  llvm::SmallDenseMap<const void *, BranchSelectorFamilyCandidate, 4> frontFamilyMap;
  for (const auto &family : frontFamilies)
    frontFamilyMap[family.ifOp] = family;
  for (const auto &family : nowFamilies) {
    if (frontFamilyMap.count(family.ifOp))
      return family;
  }
  return std::nullopt;
}

static constexpr unsigned kPipeStateSize =
    static_cast<unsigned>(PipelineType::PIPE_LAST) + 1U;

static bool isValidPipeIndex(PipelineType pipe) {
  return static_cast<unsigned>(pipe) < kPipeStateSize;
}

// ==============================================================================
// 1. Entry Point
// ==============================================================================

void InsertSyncAnalysis::Run(bool insertBarAllAtLast) {
  syncIndex_ = syncOperations_.size();

  for (auto &nowElement : syncIR_) {
    if (auto *nowCompound =
            dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      DealWithCompoundSync(nowCompound);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(nowElement.get())) {
      DealWithLoopSync(loopElement);
    } else if (isa<BranchInstanceElement>(nowElement.get())) {
      continue;
    } else if (isa<PlaceHolderInstanceElement>(nowElement.get())) {
      continue;
    }
  }

  if (insertBarAllAtLast) {
    InsertLastPipeAll();
  }
}

// ==============================================================================
// 2. High-Level Traversal
// ==============================================================================

void InsertSyncAnalysis::DealWithCompoundSync(
    CompoundInstanceElement *nowCompound) {
  SyncRecordList syncRecordList;
  InsertSeqSync(nowCompound, syncIR_, 0, nowCompound->GetIndex(), syncRecordList,
                std::nullopt);
}

void InsertSyncAnalysis::DealWithLoopSync(LoopInstanceElement *nowElement) {
  // Insert backward sync by copying the loop body slice and running the same
  // sequential insertion on the copied structure.
  if (nowElement->getLoopKind() != KindOfLoop::LOOP_END) {
    return;
  }

  SyncIRs backSyncIr;
  assert(syncIR_.size() >= nowElement->endId);
  for (unsigned i = nowElement->beginId; i < nowElement->endId; i++) {
    if (auto *compound = dyn_cast<CompoundInstanceElement>(syncIR_[i].get())) {
      InsertBackForSync(compound, backSyncIr, nowElement);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(syncIR_[i].get())) {
      auto loopKind = loopElement->getLoopKind();
      backSyncIr.emplace_back(loopElement->CloneFor(loopKind));
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(
          branchElement->CloneBranch(branchElement->getBranchKind()));
    } else if (auto *placeHolderElement =
                   dyn_cast<PlaceHolderInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(placeHolderElement->Clone());
    }
  }
}

void InsertSyncAnalysis::InsertBackForSync(
    CompoundInstanceElement *nowCompound, SyncIRs &backSyncIr,
    const LoopInstanceElement *loopElement) {
  SyncRecordList syncRecordList;

  auto backCompound = std::make_unique<CompoundInstanceElement>(
      nowCompound->GetIndex(), nowCompound->defVec, nowCompound->useVec,
      nowCompound->kPipeValue, nowCompound->opName);
  backCompound->compoundCoreType = nowCompound->compoundCoreType;
  backCompound->elementOp = nowCompound->elementOp;

  auto *backCompoundPtr = backCompound.get();
  backSyncIr.emplace_back(std::move(backCompound));

  // Insert sync between the copied commands (j+1 slice).
  InsertSeqSync(backCompoundPtr, backSyncIr, 0,
                static_cast<int>(backSyncIr.size()) - 1, syncRecordList,
                loopElement->endId);

  // Insert sync between original and copied commands to model loop-carried deps.
  InsertSeqSync(nowCompound, syncIR_, nowCompound->GetIndex(), loopElement->endId,
                syncRecordList, loopElement->endId);
}

// ==============================================================================
// 3. Sequential Sync Insertion (Core Logic)
// ==============================================================================

bool InsertSyncAnalysis::IsNoNeedToInsertSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound, bool isBackwardDep) const {
  const PipelineType frontPipe = frontCompound->kPipeValue;
  const PipelineType nowPipe = nowCompound->kPipeValue;

  if (frontPipe == nowPipe && frontPipe == PipelineType::PIPE_S) {
    return true;
  }

  if (nowCompound->elementOp == frontCompound->elementOp && !isBackwardDep) {
    return true;
  }

  // Do not short-circuit same-pipe pairs here. If a real memory dependency is
  // present, MemAnalyze will insert a PIPE_BARRIER to serialize that pipe,
  // matching the "bar_v/bar_m" style intra-pipe synchronization expected by
  // higher-level frontends.

  return false;
}

void InsertSyncAnalysis::InsertSeqSync(
    CompoundInstanceElement *nowCompound, SyncIRs &syncElement, int begin,
    int end, SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  const PipelineType nowPipeValue = nowCompound->kPipeValue;

  checkSyncIRIndex(syncElement, begin);
  checkSyncIRIndex(syncElement, end);

  unsigned syncIRIndex = syncElement[end]->GetIndex();
  UpdateAlreadySync(syncIR_[syncIRIndex]->pipeBefore, syncRecordList, nowPipeValue);

  for (int i = end - 1; i >= begin; i--) {
    auto &frontPtr = syncElement[i];
    unsigned frontIndex = frontPtr->GetIndex();
    assert(frontIndex < syncIR_.size());
    assert(syncIR_[frontIndex] != nullptr);

    if (auto *frontCompound =
            dyn_cast<CompoundInstanceElement>(frontPtr.get())) {
      UpdateAlreadySync(syncIR_[frontIndex]->pipeAfter, syncRecordList,
                        nowPipeValue);
      InsertSync(nowCompound, frontCompound, syncRecordList, forEndIndex);
      UpdateAlreadySync(syncIR_[frontIndex]->pipeBefore, syncRecordList,
                        nowPipeValue);
    } else if (auto *loopInstance =
                   dyn_cast<LoopInstanceElement>(frontPtr.get())) {
      int skipLoop = static_cast<int>(InsertLoopSync(
          i, nowCompound, begin, loopInstance, syncElement, syncRecordList,
          forEndIndex));
      i -= skipLoop;
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(frontPtr.get())) {
      int skipBranch = static_cast<int>(InsertBranchSync(
          i, nowCompound, begin, branchElement, syncElement, syncRecordList,
          forEndIndex));
      i -= skipBranch;
    }
  }
}

unsigned InsertSyncAnalysis::InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (loopElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncRecordList syncRecordForList = syncRecordList;
    unsigned newBegin =
        std::max(begin, index - (loopElement->endId - loopElement->beginId));
    unsigned newEnd = index;
    InsertSeqSync(nowCompound, syncElement, static_cast<int>(newBegin),
                  static_cast<int>(newEnd), syncRecordForList, forEndIndex);
    // A loop may execute zero iterations at runtime. Keep correctness for both
    // paths by not promoting alreadySync from the loop-body traversal into the
    // outer state. We only carry syncFinder updates, matching no-else branch
    // behavior in InsertBranchSync.
    for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++)
      syncRecordList[bufferIdx].syncFinder =
          syncRecordForList[bufferIdx].syncFinder;
    return (loopElement->endId - loopElement->beginId);
  }
  return 0;
}

unsigned InsertSyncAnalysis::InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (branchElement->getBranchKind() == KindOfBranch::IF_END) {
    SyncRecordList syncRecordIfList = syncRecordList;

    // The indices here are positions in `syncElement` (which may be a slice
    // like backSyncIr), so compute ranges relative to `index`.
    unsigned branchIf =
        index - (branchElement->endId - branchElement->beginId);
    unsigned branchElse =
        index - (branchElement->endId - branchElement->branchId);
    unsigned branchEnd = index;

    InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchIf),
                  static_cast<int>(branchElse), syncRecordIfList, forEndIndex);

    if (branchElement->branchId != branchElement->endId) {
      SyncRecordList syncRecordElseList = syncRecordList;
      InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchElse),
                    static_cast<int>(branchEnd), syncRecordElseList, forEndIndex);
      MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordElseList);
    } else {
      // No else-branch: do not promote `alreadySync`, but keep syncFinder
      // updates from the then-branch.
      for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++)
        syncRecordList[bufferIdx].syncFinder = syncRecordIfList[bufferIdx].syncFinder;
    }
    return (branchElement->endId - branchElement->beginId);
  } else if (branchElement->getBranchKind() == KindOfBranch::ELSE_BEGIN &&
             index != begin) {
    assert(nowCompound->GetIndex() > branchElement->branchId);
    return (branchElement->branchId - branchElement->beginId);
  }
  return 0;
}

void InsertSyncAnalysis::MergeAlreadySync(
    SyncRecordList &syncRecordList, const SyncRecordList &syncRecordIfList,
    const SyncRecordList &syncRecordElseList) {
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    for (size_t pipeIdx = 0; pipeIdx < kPipeStateSize; pipeIdx++) {
      if (syncRecordIfList[bufferIdx].alreadySync[pipeIdx] &&
          syncRecordElseList[bufferIdx].alreadySync[pipeIdx]) {
        syncRecordList[bufferIdx].alreadySync[pipeIdx] = true;
      }
    }
  }
}

// ==============================================================================
// 4. Dependency Analysis & Operation Insertion
// ==============================================================================

void InsertSyncAnalysis::InsertSync(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (IsNoNeedToInsertSync(nowCompound, frontCompound, forEndIndex.has_value())) {
    return;
  }
  MemAnalyze(nowCompound, frontCompound, syncRecordList, forEndIndex);
}

void InsertSyncAnalysis::MemAnalyze(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (isAlreadySync(nowCompound, frontCompound, syncRecordList, 0)) {
    return;
  }

  DepBaseMemInfoPairVec depVec;
  if (!IsMemInfoHasDependency(nowCompound, frontCompound, depVec)) {
    return;
  }

  if (forEndIndex.has_value()) {
    int eventIdNum =
        AnalyzeMultibufferSync(nowCompound, frontCompound, depVec, forEndIndex)
            .eventIdNum;
    for (int i = 1; i < eventIdNum; i++) {
      if (isAlreadySync(nowCompound, frontCompound, syncRecordList,
                        static_cast<unsigned>(i))) {
        return;
      }
    }
  }

  InsertSyncOperation(nowCompound, frontCompound, depVec, forEndIndex);
  UpdateSyncRecordInfo(frontCompound, syncRecordList);
}

bool InsertSyncAnalysis::IsMemInfoHasDependency(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  bool hasDependency = false;
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->defVec,
                                          depBaseMemInfosVec);
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->useVec,
                                          depBaseMemInfosVec);
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->defVec,
                                          depBaseMemInfosVec);

  // Special hazard: ACC (L0C) read/read cross-pipe ordering.
  //
  // Some PTO-ISA sequences have semantically "read/read" patterns on ACC, but
  // executing them concurrently across pipelines can trigger device-side issues.
  if (nowCompound->kPipeValue != frontCompound->kPipeValue) {
    DepBaseMemInfoPairVec rrDepVec;
    if (memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->useVec,
                               rrDepVec)) {
      for (auto &pair : rrDepVec) {
        if (!pair.first) continue;
        if (pair.first->scope != pto::AddressSpace::ACC) continue;
        depBaseMemInfosVec.push_back(pair);
        hasDependency = true;
      }
    }
  }

  return hasDependency;
}

void InsertSyncAnalysis::InsertSyncOperation(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  PipelineType nowPipe = nowCompound->kPipeValue;
  PipelineType frontPipe = frontCompound->kPipeValue;

  if (nowPipe == frontPipe) {
    unsigned insertBarrierId = nowCompound->GetIndex();
    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe, syncIndex_,
        insertBarrierId, forEndIndex);
    barrierOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    syncIR_[insertBarrierId]->pipeBefore.push_back(barrierOp.get());
    barrierOp->SetSyncIRIndex(insertBarrierId);

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(newSync));
  } else {
    unsigned insertWaitId = nowCompound->GetIndex();
    unsigned insertSetId = frontCompound->GetIndex();
    auto setOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe, syncIndex_,
        insertSetId, forEndIndex);
    auto waitOp = setOp->GetMatchSync(insertWaitId);
    SmallVector<std::pair<Value, int>> depRootGroups =
        GetMemInfoRootGroups(depBaseMemInfosVec);
    for (const auto &[depRoot, depGroup] : depRootGroups) {
      setOp->depRootBuffers.push_back(depRoot);
      setOp->depRootGroups.push_back(depGroup);
      waitOp->depRootBuffers.push_back(depRoot);
      waitOp->depRootGroups.push_back(depGroup);
    }
    setOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    waitOp->SetDepSyncIRIndex(frontCompound->GetIndex());

    // Back-edge dependencies may require multi-buffer event IDs.
    if (forEndIndex.has_value()) {
      auto decision = AnalyzeMultibufferSync(nowCompound, frontCompound,
                                             depBaseMemInfosVec, forEndIndex);
      setOp->eventIdNum = decision.eventIdNum;
      waitOp->eventIdNum = decision.eventIdNum;
      ConfigureMultibufferSyncMetadata(setOp.get(), waitOp.get(), decision);
    }

    syncIR_[insertSetId]->pipeAfter.push_back(setOp.get());
    syncIR_[insertWaitId]->pipeBefore.push_back(waitOp.get());

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(setOp));
    newSync.emplace_back(std::move(waitOp));
    syncOperations_.emplace_back(std::move(newSync));
  }

  syncIndex_++;
  assert(syncOperations_.size() == syncIndex_);
}

// ==============================================================================
// 5. Sync Record Maintenance
// ==============================================================================

bool InsertSyncAnalysis::isAlreadySync(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    SyncRecordList &syncRecordList, unsigned recordListIndex) {
  (void)nowCompound;
  const PipelineType frontPipe = frontCompound->kPipeValue;
  if (recordListIndex >= syncRecordList.size()) return false;
  if (!isValidPipeIndex(frontPipe)) return false;
  return syncRecordList[recordListIndex]
      .alreadySync[static_cast<unsigned>(frontPipe)];
}

void InsertSyncAnalysis::UpdateAlreadySync(const SyncOps &syncVector,
                                           SyncRecordList &syncRecordList,
                                           const PipelineType nowPipeValue) {
  for (auto *sync : syncVector) {
    for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
      if (bufferIdx == 0 && sync->eventIdNum > 1 &&
          sync->GetForEndIndex().has_value()) {
        continue;
      }
      UpdateSyncRecord(sync, syncRecordList[bufferIdx], nowPipeValue);
    }
  }
}

void InsertSyncAnalysis::UpdateSyncRecord(const SyncOperation *sync,
                                          SyncRecord &syncRecord,
                                          PipelineType nowPipeValue) {
  PipelineType setPipeValue = sync->GetSrcPipe();
  PipelineType waitPipeValue = sync->GetDstPipe();

  // Block-sync mode behaves like a global blocking pipe-s wait.
  if (syncAnalysisMode_ == SyncAnalysisMode::BLOCKSYNC) {
    nowPipeValue = PipelineType::PIPE_S;
    waitPipeValue = PipelineType::PIPE_S;
  }

  if (!isValidPipeIndex(nowPipeValue) || !isValidPipeIndex(waitPipeValue) ||
      !isValidPipeIndex(setPipeValue)) {
    return;
  }

  auto &recordAlready = syncRecord.alreadySync;
  auto &recordFinder = syncRecord.syncFinder;

  bool barrierFinder =
      (nowPipeValue == waitPipeValue) &&
      (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER);
  if (barrierFinder) {
    recordAlready[static_cast<unsigned>(nowPipeValue)] = true;
    return;
  }

  bool canTransitivelyEliminate =
      recordAlready[static_cast<unsigned>(waitPipeValue)] ||
      (nowPipeValue == waitPipeValue);
  if (!canTransitivelyEliminate) return;

  if (recordFinder[sync->GetSyncIndex()] &&
      (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
       sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET)) {
    recordAlready[static_cast<unsigned>(setPipeValue)] = true;
  }

  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    recordFinder[sync->GetSyncIndex()] = true;
  }
}

void InsertSyncAnalysis::UpdateSyncRecordInfo(
    CompoundInstanceElement *frontCompound, SyncRecordList &syncRecordList) {
  (void)frontCompound;
  assert(!syncOperations_.empty());
  auto &syncPair = syncOperations_.back();
  assert(!syncPair.empty());

  auto *newSync = syncPair[0].get();
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    if (bufferIdx == 0 && newSync->eventIdNum > 1) {
      continue;
    }
    if (!isValidPipeIndex(newSync->GetSrcPipe())) continue;
    syncRecordList[bufferIdx]
        .alreadySync[static_cast<unsigned>(newSync->GetSrcPipe())] = true;
  }
}

// ==============================================================================
// 6. Final Barrier
// ==============================================================================

void InsertSyncAnalysis::InsertLastPipeAll() {
  for (auto it = syncIR_.rbegin(); it != syncIR_.rend(); ++it) {
    auto *element = it->get();
    if (isa<PlaceHolderInstanceElement>(element)) continue;

    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, PipelineType::PIPE_ALL,
        PipelineType::PIPE_ALL, syncIndex_, element->GetIndex(), std::nullopt);
    barrierOp->MarkAutoSyncTailBarrier();

    SyncOperation *barrierRawPtr = barrierOp.get();
    SmallVector<std::unique_ptr<SyncOperation>> syncGroup;
    syncGroup.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(syncGroup));
    syncIndex_++;

    element->pipeAfter.push_back(barrierRawPtr);
    return;
  }
}

// ==============================================================================
// 7. Helpers
// ==============================================================================

bool InsertSyncAnalysis::IsMemAllocOp(Operation *op) const {
  return isa<memref::AllocOp>(op) || isa<pto::PointerCastOp>(op);
}

SmallVector<std::pair<Value, int>> InsertSyncAnalysis::GetMemInfoRootGroups(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) const {
  SmallVector<std::pair<Value, int>> result;
  auto append = [&](const BaseMemInfo *info) {
    if (!info || !info->rootBuffer)
      return;
    result.emplace_back(info->rootBuffer, info->multibufferGroup);
  };
  for (auto &pair : depBaseMemInfosVec) {
    append(pair.first);
    append(pair.second);
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first.getAsOpaquePointer() < rhs.first.getAsOpaquePointer();
    return lhs.second < rhs.second;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

InsertSyncAnalysis::MultibufferSyncDecision
InsertSyncAnalysis::AnalyzeMultibufferSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound,
    const DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) const {
  MultibufferSyncDecision decision;
  if (depBaseMemInfosVec.empty())
    return decision;

  if (auto identity = GetSharedMultibufferIdentity(depBaseMemInfosVec)) {
    if (!AreSlotwiseNonOverlapping(depBaseMemInfosVec, identity->factor))
      return decision;

    llvm::SmallDenseSet<int, MAX_MULTI_BUFFER_NUM> slotsUsed;
    auto recordInfo = [&](const BaseMemInfo *info) {
      if (!info || !info->isMultibufferSlotValid)
        return;
      if (info->multibufferRoot != identity->root)
        return;
      if (info->multibufferGroup != identity->group)
        return;
      if (info->multibufferFactor != identity->factor)
        return;
      slotsUsed.insert(info->multibufferSlot);
    };
    for (const auto &pair : depBaseMemInfosVec) {
      recordInfo(pair.first);
      recordInfo(pair.second);
    }

    decision.sharedRoot = identity->root;
    if (!forEndIndex.has_value() ||
        static_cast<size_t>(forEndIndex.value()) >= syncIR_.size()) {
      decision.slotMode =
          identity->root ? MultibufferSlotMode::SINGLE
                         : MultibufferSlotMode::NONE;
      return decision;
    }

    auto *loopEndElem =
        dyn_cast<LoopInstanceElement>(syncIR_[forEndIndex.value()].get());
    auto loopOp =
        loopEndElem ? dyn_cast_or_null<scf::ForOp>(loopEndElem->elementOp)
                    : scf::ForOp();
    if (!loopEndElem || !loopOp) {
      decision.slotMode = MultibufferSlotMode::SINGLE;
      return decision;
    }

    BranchMultibufferIdentity branchIdentity{identity->root, identity->group,
                                             identity->factor};
    if (auto family = findCommonBranchSelectorFamily(syncIR_, nowCompound,
                                                     frontCompound,
                                                     branchIdentity, loopOp)) {
      decision.eventIdNum = identity->factor;
      decision.slotMode = MultibufferSlotMode::SELECTOR;
      decision.slotCount = identity->factor;
      decision.ownerLoopBeginId = static_cast<int>(loopEndElem->beginId);
      decision.ownerLoopEndId = static_cast<int>(loopEndElem->endId);
      decision.branchSelectorFamilyBeginId = family->branchBeginId;
      decision.branchSelectorRepresentativeOp = family->representativeOp;
      return decision;
    }

    if (slotsUsed.size() <= 1) {
      decision.slotMode =
          identity->root ? MultibufferSlotMode::SINGLE
                         : MultibufferSlotMode::NONE;
      return decision;
    }

    if (slotsUsed.size() != static_cast<size_t>(identity->factor))
      return decision;
    if (countCompoundSlotsForRoot(nowCompound, identity->root, identity->group,
                                  identity->factor) <= 1 &&
        countCompoundSlotsForRoot(frontCompound, identity->root,
                                  identity->group, identity->factor) <= 1) {
      decision.slotMode = MultibufferSlotMode::BRANCH;
      decision.slotCount = identity->factor;
      return decision;
    }

    decision.eventIdNum = identity->factor;
    decision.slotMode = MultibufferSlotMode::SELECTOR;
    decision.slotCount = identity->factor;
    decision.ownerLoopBeginId = static_cast<int>(loopEndElem->beginId);
    decision.ownerLoopEndId = static_cast<int>(loopEndElem->endId);
    return decision;
  }

  return decision;
}

void InsertSyncAnalysis::ConfigureMultibufferSyncMetadata(
    SyncOperation *setOp, SyncOperation *waitOp,
    const MultibufferSyncDecision &decision) {
  auto reset = [](SyncOperation *sync) {
    if (!sync)
      return;
    sync->slotMode = MultibufferSlotMode::NONE;
    sync->slotCount = 1;
    sync->ownerLoopBeginId = -1;
    sync->ownerLoopEndId = -1;
    sync->branchSelectorFamilyBeginId = -1;
    sync->branchSelectorRepresentativeOp = nullptr;
  };

  reset(setOp);
  reset(waitOp);

  if (!setOp || !waitOp)
    return;

  auto configure = [&](SyncOperation *sync) {
    sync->slotMode = decision.slotMode;
    sync->slotCount = decision.slotCount;
    sync->ownerLoopBeginId = decision.ownerLoopBeginId;
    sync->ownerLoopEndId = decision.ownerLoopEndId;
    sync->branchSelectorFamilyBeginId = decision.branchSelectorFamilyBeginId;
    sync->branchSelectorRepresentativeOp =
        decision.branchSelectorRepresentativeOp;
    if (decision.sharedRoot)
      sync->lowestCommonAncestorBuffer = decision.sharedRoot;
  };

  configure(setOp);
  configure(waitOp);
}

std::optional<InsertSyncAnalysis::MultibufferIdentity>
InsertSyncAnalysis::GetSharedMultibufferIdentity(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) const {
  std::optional<MultibufferIdentity> identity;
  for (const auto &pair : depBaseMemInfosVec) {
    const BaseMemInfo *a = pair.first;
    const BaseMemInfo *b = pair.second;
    if (!a || !b) return std::nullopt;
    if (!a->isMultibufferSlotValid || !b->isMultibufferSlotValid)
      return std::nullopt;
    if (a->multibufferRoot != b->multibufferRoot)
      return std::nullopt;
    if (a->multibufferGroup != b->multibufferGroup)
      return std::nullopt;
    if (a->multibufferFactor != b->multibufferFactor)
      return std::nullopt;
    if (a->multibufferFactor <= 1) return std::nullopt;
    if (!identity) {
      identity = MultibufferIdentity{
          a->multibufferRoot, a->multibufferGroup, a->multibufferFactor};
      continue;
    }
    if (identity->root != a->multibufferRoot ||
        identity->group != a->multibufferGroup ||
        identity->factor != a->multibufferFactor) {
      return std::nullopt;
    }
  }
  return identity;
}

bool InsertSyncAnalysis::AreSlotwiseNonOverlapping(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec, int factor) const {
  for (const auto &pair : depBaseMemInfosVec) {
    if (!IsSlotAwareMultibufferPair(pair.first, pair.second, factor)) {
      return false;
    }
  }
  return true;
}

bool InsertSyncAnalysis::IsSlotAwareMultibufferPair(const BaseMemInfo *a,
                                                    const BaseMemInfo *b,
                                                    int factor) const {
  if (!a || !b) return false;
  if (a->scope == pto::AddressSpace::GM || b->scope == pto::AddressSpace::GM)
    return false;
  if (!a->isMultibufferSlotValid || !b->isMultibufferSlotValid)
    return false;
  if (a->multibufferRoot != b->multibufferRoot)
    return false;
  if (a->multibufferGroup != b->multibufferGroup)
    return false;
  if (a->multibufferFactor != factor || b->multibufferFactor != factor)
    return false;
  if (a->multibufferSlot < 0 || b->multibufferSlot < 0)
    return false;
  if (a->multibufferSlot >= factor || b->multibufferSlot >= factor)
    return false;
  //
  // Translator-side slot proof is based on explicit factor/slot annotations and
  // a statically-proven equal partition on the same logical root buffer. Once
  // that proof exists, different slots are disjoint by construction, even when
  // the lowered memref subviews inherit non-unit parent strides and their
  // flattened byte ranges look overlapping. Same-slot pairs are conservatively
  // kept on one lane.
  return true;
}

bool InsertSyncAnalysis::IsGMHazard(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound) const {
  auto hasGM = [](const SmallVector<const BaseMemInfo *> &vec) {
    for (const auto *info : vec) {
      if (info->scope == pto::AddressSpace::GM) return true;
    }
    return false;
  };

  bool frontWritesGM = hasGM(frontCompound->defVec);
  bool frontReadsGM = hasGM(frontCompound->useVec);

  bool nowWritesGM = hasGM(nowCompound->defVec);
  bool nowReadsGM = hasGM(nowCompound->useVec);

  if (frontWritesGM && nowReadsGM) return true;  // RAW
  if (frontReadsGM && nowWritesGM) return true;  // WAR
  if (frontWritesGM && nowWritesGM) return true; // WAW

  // RAR is considered safe for GM in this simplified model.
  return false;
}

} // namespace mlir::pto
