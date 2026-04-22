// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TILE_BUFFER_SEMANTICS_H
#define PTO_TILE_BUFFER_SEMANTICS_H

#include "PTO/IR/PTO.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace pto {

/// Unified view-like category used by tile-buffer semantic normalization.
enum class TileViewKind {
  Unknown = 0,
  BindTile,
  Subset,
  Bitcast,
  TReshape
};

/// Normalized semantic payload consumed by PlanMemory.
struct TileBufferSemantics {
  /// queried value.
  Value value;
  /// traced root value (alloc/pointer-cast/function arg/...).
  Value root;
  /// storage scope resolved from value/root type.
  pto::AddressSpace scope{pto::AddressSpace::Zero};
  /// element type used for byte/bits calculation.
  Type elementType;
  /// logical shape.
  SmallVector<int64_t, 4> shape;
  /// logical valid shape.
  SmallVector<int64_t, 4> validShape;
  /// optional tile config carried by tile/bind semantics.
  TileBufConfigAttr config;
  /// view-kind of the queried value's defining op.
  TileViewKind viewKind{TileViewKind::Unknown};
  /// static size in bits, if computable.
  int64_t constBits{0};
};

/// Reads PTO address-space from tile/view values for planning.
std::optional<AddressSpaceAttr> getPlanningBufferSpaceAttr(Value operand);

/// Returns (result, source) when `op` is an alias/view-like op for planning.
std::optional<std::pair<Value, Value>> getBufferAliasInfo(Operation *op);

/// Classifies the view semantics of an op for tracing/debug/planning.
TileViewKind getTileViewKind(Operation *op);

/// Traces through alias/view-like chains to the storage root value.
Value tracebackBufferRoot(Value value);

/// Returns true when an operation touches any local plannable buffer.
bool isOpTouchPlannableLocalBuffer(Operation *op);

/// Infers normalized tile semantics (scope/shape/valid/config/root/bytes).
/// Returns failure when static bits cannot be proven. If `failureReason` is
/// provided, it will be filled with a concise cause.
LogicalResult inferTileBufferSemantics(Value value, TileBufferSemantics &out,
                                       std::string *failureReason);

/// Convenience wrapper for callers that do not need failure diagnostics.
inline LogicalResult inferTileBufferSemantics(Value value,
                                              TileBufferSemantics &out) {
  return inferTileBufferSemantics(value, out, nullptr);
}

} // namespace pto
} // namespace mlir

#endif // PTO_TILE_BUFFER_SEMANTICS_H
