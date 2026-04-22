// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "Utils.h"

#define DEBUG_TYPE "pto-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace pto {

func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  if (auto subViewOp = dyn_cast<pto::SubViewOp>(op))
    return std::make_pair(subViewOp.getResult(), subViewOp.getSource());
  if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op))
    return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
  if (auto reshapeOp = dyn_cast<pto::TReshapeOp>(op))
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getSrc());
  if (auto bindTileOp = dyn_cast<pto::BindTileOp>(op))
    return std::make_pair(bindTileOp.getResult(), bindTileOp.getSource());
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op))
    return std::make_pair(sliceOp.getResult(), sliceOp.getSource());
  return std::nullopt;
}

std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand) {
  if (auto tileBufType = dyn_cast<pto::TileBufType>(operand.getType())) {
    auto memorySpace = tileBufType.getMemorySpace();
    if (!memorySpace)
      return std::nullopt;
    return dyn_cast<AddressSpaceAttr>(memorySpace);
  }
  if (isa<pto::TensorViewType, pto::PartitionTensorViewType>(operand.getType())) {
    return AddressSpaceAttr::get(operand.getContext(), pto::AddressSpace::GM);
  }
  return std::nullopt;
}

static Value tracebackOneStep(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(arg.getParentRegion()->getParentOp())) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() > arg.getArgNumber() - 1) {
        return forOp.getInitArgs()[arg.getArgNumber() - 1];
      }
    }
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return Value{};

  if (auto aliasPair = getOperationAliasInfo(def)) {
    auto [aliasValue, sourceValue] = *aliasPair;
    if (aliasValue == value)
      return sourceValue;
  }

  if (auto op = dyn_cast<UnrealizedConversionCastOp>(def))
    return op.getOperand(cast<OpResult>(value).getResultNumber());
  if (auto op = dyn_cast<scf::ForOp>(def))
    return op.getInitArgs()[cast<OpResult>(value).getResultNumber()];

  return Value{};
}

static Value tracebackToRoot(Value value) {
  int loopBound = 256;
  while (value) {
    Value upward = tracebackOneStep(value);
    if (!upward)
      break;
    value = upward;
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs() << "tracebackToRoot exceeds loopBound(" << loopBound
                              << ")!");
      break;
    }
  }
  return value;
}

std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes) {
  int64_t totalSize = 1;
  for (int64_t shape : shapes) {
    if (ShapedType::isDynamic(shape))
      return std::nullopt;
    totalSize *= shape;
  }
  return totalSize;
}

uint64_t AlignUp(uint64_t lhs, uint64_t rhs) {
  if (rhs == 0)
    return lhs;
  if (lhs % rhs != 0)
    lhs += rhs - (lhs % rhs);
  return lhs;
}

bool isFromFunctionArg(Value v) {
  return tracebackToRoot(v).getDefiningOp() == nullptr;
}

bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr) {
  if (!memorySpaceAttr.has_value())
    return false;
  if (memorySpaceAttr->getAddressSpace() == pto::AddressSpace::GM)
    return false;
  if (LocalBufferSpace.count(memorySpaceAttr->getAddressSpace()))
    return true;
  llvm_unreachable("unsupported non-local address space");
}

static SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands())
    touchBuffer.push_back(operand.get());
  return touchBuffer;
}

bool isOpTouchLocalBuffer(Operation *op) {
  for (Value buffer : getOpTouchBuffer(op)) {
    if (isLocalBuffer(GetBufferSpaceAttr(buffer)))
      return true;
  }
  return false;
}

ModuleOp getTopLevelModuleOp(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  while (moduleOp && moduleOp->getParentOp())
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  return moduleOp;
}

static std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end())
    return it - yieldedValues.begin();
  return std::nullopt;
}

LoopLikeOpInterface getParentLoop(Value val) {
  if (!val.getDefiningOp())
    return nullptr;

  LoopLikeOpInterface parentLoop =
      val.getDefiningOp()->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop)
    return nullptr;

  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return parentLoop;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
    return getParentLoop(res);
  }

  auto parentIf = val.getDefiningOp()->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return parentLoop;

  auto idxThenYielded = getYieldValueIdx(val, parentIf.thenYield().getOperands());
  if (idxThenYielded.has_value()) {
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  auto idxElseYielded = getYieldValueIdx(val, parentIf.elseYield().getOperands());
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
  }

  return parentLoop;
}

} // namespace pto
} // namespace mlir
