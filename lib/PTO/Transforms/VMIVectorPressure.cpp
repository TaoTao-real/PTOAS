// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0

#include "PTO/Transforms/VMIVectorPressure.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/VMIUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

struct LiveInterval {
  int64_t chunks = 0;
  int64_t begin = 0;
  int64_t end = 0;
  bool persistent = false;
};

} // namespace

FailureOr<int64_t> mlir::pto::getVMIVectorChunks(Type type) {
  auto vreg = dyn_cast<VMIVRegType>(type);
  if (!vreg)
    return failure();
  if (vreg.hasLayout())
    return getVMIPhysicalArity(vreg);
  FailureOr<int64_t> lanesPerChunk = getDataLanesPerPart(vreg.getElementType());
  if (failed(lanesPerChunk) || *lanesPerChunk <= 0 ||
      vreg.getElementCount() <= 0)
    return failure();
  return (vreg.getElementCount() + *lanesPerChunk - 1) / *lanesPerChunk;
}

VMIVectorPressureEstimate mlir::pto::estimateVMILoopPressure(scf::ForOp loop) {
  VMIVectorPressureEstimate estimate;
  DenseMap<Value, LiveInterval> intervals;
  DenseSet<Value> persistentValues;
  int64_t operationCount =
      std::distance(loop.getBody()->without_terminator().begin(),
                    loop.getBody()->without_terminator().end());

  auto chunksFor = [&](Value value) -> std::optional<int64_t> {
    if (!isa<VMIVRegType>(value.getType()))
      return std::nullopt;
    FailureOr<int64_t> chunks = getVMIVectorChunks(value.getType());
    if (failed(chunks)) {
      estimate.isExact = false;
      return std::nullopt;
    }
    return *chunks;
  };

  auto define = [&](Value value, int64_t position, bool persistent) {
    std::optional<int64_t> chunks = chunksFor(value);
    if (!chunks)
      return;
    intervals.try_emplace(value,
                          LiveInterval{*chunks, position,
                                       persistent ? operationCount : position,
                                       persistent});
    if (persistent)
      persistentValues.insert(value);
  };
  auto use = [&](Value value, int64_t position) {
    std::optional<int64_t> chunks = chunksFor(value);
    if (!chunks)
      return;
    auto [it, inserted] = intervals.try_emplace(
        value,
        LiveInterval{*chunks, 0, position, persistentValues.contains(value)});
    if (!inserted && !it->second.persistent)
      it->second.end = std::max(it->second.end, position);
  };

  for (BlockArgument arg : loop.getRegionIterArgs()) {
    define(arg, 0, true);
    if (std::optional<int64_t> chunks = chunksFor(arg))
      estimate.loopCarriedVectorChunks += *chunks;
  }

  int64_t position = 0;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions() != 0 && !isa<FusionRegionOp>(op))
      estimate.isExact = false;
    op.walk([&](Operation *nested) {
      if (nested != &op && nested->getNumRegions() != 0)
        estimate.isExact = false;
      for (Value operand : nested->getOperands()) {
        if (auto result = dyn_cast<OpResult>(operand)) {
          if (!loop->isProperAncestor(result.getOwner())) {
            define(operand, 0, true);
            continue;
          }
        } else if (auto argument = dyn_cast<BlockArgument>(operand)) {
          if (argument.getOwner() != loop.getBody()) {
            define(operand, 0, true);
            continue;
          }
        }
        use(operand, position);
      }
      for (Value result : nested->getResults())
        define(result, position, false);
    });
    ++position;
  }
  for (Value yielded : loop.getBody()->getTerminator()->getOperands())
    use(yielded, operationCount);

  SmallVector<int64_t, 32> chunkDelta(operationCount + 2, 0);
  SmallVector<int64_t, 32> valueDelta(operationCount + 2, 0);
  for (const auto &entry : intervals) {
    const LiveInterval &interval = entry.second;
    chunkDelta[interval.begin] += interval.chunks;
    chunkDelta[interval.end + 1] -= interval.chunks;
    valueDelta[interval.begin] += 1;
    valueDelta[interval.end + 1] -= 1;
    if (interval.persistent)
      estimate.persistentVectorChunks += interval.chunks;
  }

  int64_t liveChunks = 0;
  int64_t liveValues = 0;
  for (int64_t i = 0; i <= operationCount; ++i) {
    liveChunks += chunkDelta[i];
    liveValues += valueDelta[i];
    estimate.peakVectorChunks = std::max(estimate.peakVectorChunks, liveChunks);
    estimate.peakVectorValues = std::max(estimate.peakVectorValues, liveValues);
  }
  estimate.temporaryVectorChunks = std::max<int64_t>(
      0, estimate.peakVectorChunks - estimate.persistentVectorChunks);
  return estimate;
}
