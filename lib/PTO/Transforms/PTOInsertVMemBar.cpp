// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInsertVMemBar.cpp - VPTO V-pipeline mem_bar insertion -----------===//
//
// Structured tile ops can later be expanded and inlined into adjacent VPTO
// vlds/vsts sequences in the same vecscope. A5 V-pipeline memory operations are
// not a sufficient ordering guarantee for overlapping UB RAW/WAR/WAW hazards,
// so this pass inserts pto.mem_bar before TileLang expansion while tile operand
// read/write roles and alloc_tile addresses are still explicit.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSERTVMEMBAR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

struct AddrRange {
  uint64_t base = 0;
  uint64_t size = 0;
  bool valid = false;
};

struct TileAccess {
  AddrRange range;
  bool isWrite = false;
};

struct HazardSet {
  bool raw = false; // previous write -> current read
  bool war = false; // previous read  -> current write
  bool waw = false; // previous write -> current write

  bool empty() const { return !raw && !war && !waw; }
  unsigned count() const {
    return static_cast<unsigned>(raw) + static_cast<unsigned>(war) +
           static_cast<unsigned>(waw);
  }

  void merge(const HazardSet &rhs) {
    raw |= rhs.raw;
    war |= rhs.war;
    waw |= rhs.waw;
  }
};

static bool overlaps(const AddrRange &lhs, const AddrRange &rhs) {
  if (!lhs.valid || !rhs.valid)
    return true;
  return lhs.base < rhs.base + rhs.size && rhs.base < lhs.base + lhs.size;
}

static std::optional<uint64_t> getConstantUInt(Value value) {
  APInt intValue;
  if (!value || !matchPattern(value, m_ConstantInt(&intValue)) ||
      intValue.isNegative())
    return std::nullopt;
  return intValue.getZExtValue();
}

static uint64_t getTileByteSize(TileBufType tileTy) {
  uint64_t elemBytes = getPTOStorageElemByteSize(tileTy.getElementType());
  if (elemBytes == 0)
    return 0;

  uint64_t elems = 1;
  for (int64_t dim : tileTy.getShape()) {
    if (dim <= 0)
      return 0;
    elems *= static_cast<uint64_t>(dim);
  }
  return elems * elemBytes;
}

static std::optional<uint64_t> resolveTileBase(Value tileVal,
                                               unsigned depth = 0) {
  if (!tileVal || depth > 8)
    return std::nullopt;

  if (auto alloc = tileVal.getDefiningOp<pto::AllocTileOp>())
    return getConstantUInt(alloc.getAddr());

  if (auto subview = tileVal.getDefiningOp<pto::SubViewOp>()) {
    auto parentBase = resolveTileBase(subview.getSource(), depth + 1);
    if (!parentBase)
      return std::nullopt;

    auto sourceTy = dyn_cast<TileBufType>(subview.getSource().getType());
    if (!sourceTy || sourceTy.getShape().size() < 2)
      return std::nullopt;
    int64_t sourceCols = sourceTy.getShape()[1];
    if (sourceCols <= 0)
      return std::nullopt;

    auto offsets = subview.getOffsets();
    if (offsets.size() < 2)
      return std::nullopt;
    auto rowOffset = getConstantUInt(offsets[0]);
    auto colOffset = getConstantUInt(offsets[1]);
    if (!rowOffset || !colOffset)
      return std::nullopt;

    uint64_t elemBytes = getPTOStorageElemByteSize(sourceTy.getElementType());
    if (elemBytes == 0)
      return std::nullopt;

    uint64_t linearElemOffset =
        (*rowOffset) * static_cast<uint64_t>(sourceCols) + (*colOffset);
    return *parentBase + linearElemOffset * elemBytes;
  }

  return std::nullopt;
}

static AddrRange resolveTileAddrRange(Value tileVal) {
  AddrRange range;
  auto tileTy = dyn_cast<TileBufType>(tileVal.getType());
  if (!tileTy)
    return range;

  range.size = getTileByteSize(tileTy);
  auto base = resolveTileBase(tileVal);
  if (!base || range.size == 0)
    return range;

  range.base = *base;
  range.valid = true;
  return range;
}

static SmallVector<TileAccess> collectTileAccesses(Operation *op) {
  SmallVector<TileAccess> accesses;
  auto dps = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
  if (!dps)
    return accesses;

  SmallPtrSet<OpOperand *, 4> writeOperands;
  for (OpOperand &init : dps.getDpsInitsMutable()) {
    if (isa<TileBufType>(init.get().getType()))
      writeOperands.insert(&init);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<TileBufType>(operand.get().getType()))
      continue;
    accesses.push_back(
        TileAccess{resolveTileAddrRange(operand.get()),
                   writeOperands.contains(&operand)});
  }
  return accesses;
}

static HazardSet classifyHazards(const SmallVector<TileAccess> &previous,
                                 const SmallVector<TileAccess> &current) {
  HazardSet hazards;
  for (const TileAccess &prev : previous) {
    for (const TileAccess &cur : current) {
      if (!overlaps(prev.range, cur.range))
        continue;
      if (prev.isWrite && !cur.isWrite)
        hazards.raw = true;
      if (!prev.isWrite && cur.isWrite)
        hazards.war = true;
      if (prev.isWrite && cur.isWrite)
        hazards.waw = true;
    }
  }
  return hazards;
}

static bool memBarCovers(MemBarKind kind, const HazardSet &hazards) {
  if (hazards.empty())
    return true;
  if (kind == MemBarKind::VV_ALL)
    return true;
  if (hazards.raw && kind != MemBarKind::VST_VLD)
    return false;
  if (hazards.war && kind != MemBarKind::VLD_VST)
    return false;
  if (hazards.waw && kind != MemBarKind::VST_VST)
    return false;
  return hazards.count() == 1;
}

static MemBarKind getMemBarKindFor(const HazardSet &hazards) {
  if (hazards.count() != 1)
    return MemBarKind::VV_ALL;
  if (hazards.raw)
    return MemBarKind::VST_VLD;
  if (hazards.war)
    return MemBarKind::VLD_VST;
  return MemBarKind::VST_VST;
}

static bool hasCoveringMemBarBetween(Operation *previous, Operation *current,
                                     const HazardSet &hazards) {
  for (Operation *op = previous->getNextNode(); op && op != current;
       op = op->getNextNode()) {
    if (auto memBar = dyn_cast<pto::MemBarOp>(op)) {
      if (memBarCovers(memBar.getKind().getKind(), hazards))
        return true;
    }
  }
  return false;
}

static bool isStructuredVTileOp(Operation *op) {
  auto pipe = dyn_cast<pto::OpPipeInterface>(op);
  if (!pipe || pipe.getPipe() != pto::PIPE::PIPE_V)
    return false;
  return isa<pto::PTO_DpsInitOpInterface>(op);
}

static bool hasUnknownRange(ArrayRef<TileAccess> accesses) {
  return llvm::any_of(accesses, [](const TileAccess &access) {
    return !access.range.valid;
  });
}

struct VTileOpInfo {
  Operation *op = nullptr;
  SmallVector<TileAccess> accesses;
};

struct PTOInsertVMemBarPass
    : public mlir::pto::impl::PTOInsertVMemBarBase<PTOInsertVMemBarPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!pto::isTargetArchA5(func.getOperation()))
      return;

    SmallVector<std::pair<Operation *, MemBarKind>, 16> insertions;
    bool sawUnknownRange = false;

    for (Block &block : func.getBody()) {
      SmallVector<VTileOpInfo, 16> vOps;
      for (Operation &op : block) {
        if (!isStructuredVTileOp(&op))
          continue;
        auto accesses = collectTileAccesses(&op);
        if (accesses.empty())
          continue;
        sawUnknownRange |= hasUnknownRange(accesses);

        HazardSet uncovered;
        for (const VTileOpInfo &previous : vOps) {
          HazardSet hazards = classifyHazards(previous.accesses, accesses);
          if (hazards.empty())
            continue;
          if (!hasCoveringMemBarBetween(previous.op, &op, hazards))
            uncovered.merge(hazards);
        }

        if (!uncovered.empty())
          insertions.push_back({&op, getMemBarKindFor(uncovered)});
        vOps.push_back(VTileOpInfo{&op, std::move(accesses)});
      }
    }

    OpBuilder builder(func.getContext());
    for (auto [op, kind] : insertions) {
      builder.setInsertionPoint(op);
      builder.create<pto::MemBarOp>(op->getLoc(),
                                    pto::MemBarAttr::get(func.getContext(),
                                                         kind));
    }

    if (sawUnknownRange) {
      func.emitRemark()
          << "pto-insert-v-membar conservatively treats unresolved VPTO tile "
             "UB ranges as may-alias";
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertVMemBarPass() {
  return std::make_unique<PTOInsertVMemBarPass>();
}
