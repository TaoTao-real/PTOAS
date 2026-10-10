// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License).
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Shared interval-arithmetic helpers for the VMI storage alias analyses.
//
// VmiMemoryLocation.cpp (storage-root and tile-interval aliasing) and
// PTOVmiLoopFusion.cpp (cross-iteration UB dependency checks) both reason
// about statically-known accessed byte ranges. The primitives they shared as
// duplicated blocks now live here once:
//   - getBareConstantInteger: the integer value of a bare arith constant
//     (ConstantOp holding an IntegerAttr, ConstantIndexOp, ConstantIntOp);
//   - mayOverlapStaticByteRanges: the overflow-checked byte-interval overlap
//     test for two element-indexed accesses into statically-known base
//     addresses.
// Unknown inputs and any arithmetic overflow conservatively report overlap.

#ifndef PTO_TRANSFORMS_VMIINTERVALARITHMETIC_H
#define PTO_TRANSFORMS_VMIINTERVALARITHMETIC_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

// Integer value of a bare arith constant: a ConstantOp holding an IntegerAttr,
// a ConstantIndexOp, or a ConstantIntOp. Returns nullopt for a null value and
// for anything that is not one of these constant forms; compound expressions
// (addi/subi/muli folding) stay with the callers, which layer their own
// policies on top.
inline std::optional<int64_t> getBareConstantInteger(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(c.getValue())) {
      return attr.getInt();
    }
  }
  if (auto c = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  }
  if (auto c = value.getDefiningOp<arith::ConstantIntOp>()) {
    return c.value();
  }
  return std::nullopt;
}

// Whether two statically-known accessed byte ranges may overlap:
// [baseAddress + offsetElements * elementBytes, +accessBytes) on each side,
// with both accesses sharing one element width. An unknown offset or element
// width, or any arithmetic overflow, conservatively reports overlap (true);
// only provably disjoint intervals return false.
inline bool mayOverlapStaticByteRanges(int64_t lhsBaseAddress,
                                       std::optional<int64_t> lhsOffsetElements,
                                       int64_t lhsAccessBytes,
                                       int64_t rhsBaseAddress,
                                       std::optional<int64_t> rhsOffsetElements,
                                       int64_t rhsAccessBytes,
                                       std::optional<int64_t> elementBytes) {
  if (!lhsOffsetElements || !rhsOffsetElements || !elementBytes) {
    return true;
  }
  int64_t lhsDelta = 0;
  int64_t rhsDelta = 0;
  if (__builtin_mul_overflow(*lhsOffsetElements, *elementBytes, &lhsDelta) ||
      __builtin_mul_overflow(*rhsOffsetElements, *elementBytes, &rhsDelta)) {
    return true;
  }
  int64_t lhsBegin = 0;
  int64_t rhsBegin = 0;
  if (__builtin_add_overflow(lhsBaseAddress, lhsDelta, &lhsBegin) ||
      __builtin_add_overflow(rhsBaseAddress, rhsDelta, &rhsBegin)) {
    return true;
  }
  int64_t lhsEnd = 0;
  int64_t rhsEnd = 0;
  if (__builtin_add_overflow(lhsBegin, lhsAccessBytes, &lhsEnd) ||
      __builtin_add_overflow(rhsBegin, rhsAccessBytes, &rhsEnd)) {
    return true;
  }
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VMIINTERVALARITHMETIC_H
