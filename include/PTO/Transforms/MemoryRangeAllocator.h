// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef PTO_TRANSFORMS_MEMORY_RANGE_ALLOCATOR_H
#define PTO_TRANSFORMS_MEMORY_RANGE_ALLOCATOR_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::pto::detail {
/// Reserve a first-fit aligned range. Every successful placement, including
/// placement in a hole, is inserted before returning to the caller.
inline std::optional<uint64_t> reserveMemoryRange(
    llvm::SmallVectorImpl<std::pair<uint64_t, uint64_t>> &occupied,
    uint64_t bytes, uint64_t alignment, uint64_t capacity) {
  if (!bytes || !alignment || bytes > capacity) return std::nullopt;
  llvm::sort(occupied);
  uint64_t cursor = 0;
  for (auto [start, end] : occupied) {
    if (start > end || end > capacity) return std::nullopt;
    uint64_t padding = (alignment - cursor % alignment) % alignment;
    if (padding > capacity - cursor) return std::nullopt;
    cursor += padding;
    if (bytes <= capacity - cursor && cursor + bytes <= start) {
      occupied.push_back({cursor, cursor + bytes});
      return cursor;
    }
    cursor = std::max(cursor, end);
  }
  uint64_t padding = (alignment - cursor % alignment) % alignment;
  if (padding > capacity - cursor) return std::nullopt;
  cursor += padding;
  if (bytes > capacity - cursor) return std::nullopt;
  occupied.push_back({cursor, cursor + bytes});
  return cursor;
}
} // namespace mlir::pto::detail
#endif
