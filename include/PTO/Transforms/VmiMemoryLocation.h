// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VMIMEMORYLOCATION_H
#define PTO_TRANSFORMS_VMIMEMORYLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

/// A compile-time storage root for a pointer_cast-created UB view.  The
/// address is the numeric address from the IR, not the SSA identity of the
/// arith.constant operation.  `storageBytes` is the statically-known extent of
/// the view when available; it is used only to prove two roots disjoint.
struct VmiStorageRoot {
  int64_t address = 0;
  std::optional<int64_t> storageBytes;
  Type viewType;
  // Present only for a compiler-verified, distinct prepared-VF buffer argument.
  // Its address is a relative offset; it is never an invented physical address.
  Value logicalRoot = {};

  bool operator==(const VmiStorageRoot &other) const {
    return logicalRoot == other.logicalRoot && address == other.address &&
           viewType == other.viewType;
  }
};

std::optional<VmiStorageRoot> resolveVmiStorageRoot(Value base);

/// Returns true when the two statically-known views may overlap.  Unknown
/// extents are treated conservatively.  Equal numeric addresses always alias,
/// including views with different element types.
bool mayAliasVmiStorageRoot(const VmiStorageRoot &lhs,
                            const VmiStorageRoot &rhs);

struct VmiAccessLocation {
  VmiStorageRoot root;
  Value elementOffset;
  int64_t accessBytes = 0;
};

/// Compare two accesses in the same element-index domain.  Unknown offsets,
/// element widths, or access sizes conservatively return true when their
/// storage roots may alias.
bool mayAliasVmiAccess(const VmiAccessLocation &lhs,
                       const VmiAccessLocation &rhs);

/// A tile-typed access described by its owning UB allocation and the static
/// byte interval it denotes inside that allocation.
///
/// This is the fusion-planning view of a memory access.  FusionPlan runs before
/// PlanMemory, so a `pto.alloc_tile` result has no physical address yet; the
/// identity of the base allocation plus the static byte interval is then the
/// only alias information in the IR, and it is exactly what region
/// partitioning needs.  Two accesses to the same root whose intervals coincide
/// are already ordered by the dataflow graph (the fusion region preserves
/// order); two whose intervals are disjoint are provably independent.  Anything
/// in between -- partial overlap, or an interval that cannot be computed -- has
/// no such guarantee and must not share a region.
struct VmiTileAccess {
  /// The tile-typed SSA value this access was derived from.  Two accesses
  /// through the same value denote the same region by construction, even when
  /// the byte interval cannot be computed.
  Value accessValue;
  /// Base allocation handle: the `pto.alloc_tile` / `pto.alloc_multi_tile`
  /// result, or the block argument / other defining value a view chain bottoms
  /// out at.  Distinct roots denote distinct logical buffers.
  Value root;
  /// True when the value's storage could not be attributed to a known
  /// allocation.  Such an access may alias anything and never shares a region.
  /// `pto.tassign` sets this: it rebinds a tile handle to a *new* runtime
  /// address, so following it to its source would be unsound.
  bool unknownRoot = false;
  /// Static byte offset of the view inside `root`; nullopt when the view chain
  /// carries a non-constant offset (dynamic subview offset, dynamic multi-tile
  /// slot) or a layout whose strides are not statically known.
  std::optional<int64_t> byteOffset;
  /// Static byte span of the region the view touches inside `root`, measured
  /// from `byteOffset`; nullopt for a dynamic shape or an element width this
  /// analysis does not model.  For a subview this is the strided span of the
  /// window in the parent, not the product of the window's own dimensions.
  std::optional<int64_t> byteSize;
  Type viewType;
};

/// How two tile accesses relate.
enum class VmiAccessRelation {
  /// Both denote exactly the same byte interval.
  MustAlias,
  /// Provably disjoint byte intervals.
  NoAlias,
  /// Overlapping but not identical, or not provable either way.
  MayAlias,
};

/// Resolve a tile-typed value to its storage root and static byte interval.
/// Every tile value yields a root; `byteOffset` / `byteSize` stay nullopt when
/// they cannot be computed statically.  Returns nullopt only when \p tile is
/// not a tile-typed value at all.
std::optional<VmiTileAccess> resolveVmiTileAccess(Value tile);

/// Relate two tile accesses.  Anything that is neither must-alias nor provably
/// no-alias is reported as MayAlias, including unknown offsets and extents.
VmiAccessRelation relateVmiTileAccess(const VmiTileAccess &lhs,
                                      const VmiTileAccess &rhs);

/// Region-partitioning predicate: true when the two accesses must not share a
/// fusion region.
inline bool mayAliasVmiTileAccess(const VmiTileAccess &lhs,
                                  const VmiTileAccess &rhs) {
  return relateVmiTileAccess(lhs, rhs) == VmiAccessRelation::MayAlias;
}

} // namespace pto
} // namespace mlir

#endif
