// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "PTO/Transforms/MemoryRangeAllocator.h"
#include <cstdio>
#include <limits>

int main() {
  using mlir::pto::detail::reserveMemoryRange;
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> occupied{{1024, 2048}};
  auto a = reserveMemoryRange(occupied, 512, 256, 4096);
  auto b = reserveMemoryRange(occupied, 512, 256, 4096);
  auto c = reserveMemoryRange(occupied, 512, 256, 4096);
  if (a != 0 || b != 512 || c != 2048) return 1;
  if (reserveMemoryRange(occupied, 2048, 256, 4096)) return 2;
  occupied = {{256, 768}, {0, 512}};
  if (reserveMemoryRange(occupied, 256, 256, 1024) != 768) return 3;
  occupied.clear();
  uint64_t max = std::numeric_limits<uint64_t>::max();
  occupied.push_back({0, max - 1});
  if (reserveMemoryRange(occupied, 1, 256, max)) return 4;
  std::puts("MEMORY_RANGE_PASS: hole reservations, overlap, capacity, alignment overflow");
  return 0;
}
