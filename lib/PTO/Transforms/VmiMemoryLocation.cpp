// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VmiMemoryLocation.h"

#include "PTO/IR/PTO.h"
#include "VmiIntervalArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace {

static std::optional<int64_t> getConstantInteger(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto constant = pto::getBareConstantInteger(value)) {
    return constant;
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getConstantInteger(add.getLhs());
    auto rhs = getConstantInteger(add.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_add_overflow(*lhs, *rhs, &result)) {
      return result;
    }
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = getConstantInteger(sub.getLhs());
    auto rhs = getConstantInteger(sub.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_sub_overflow(*lhs, *rhs, &result)) {
      return result;
    }
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = getConstantInteger(mul.getLhs());
    auto rhs = getConstantInteger(mul.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_mul_overflow(*lhs, *rhs, &result)) {
      return result;
    }
  }
  return std::nullopt;
}

/// Byte width of a scalar element type, or nullopt when this analysis does not
/// model it.
static std::optional<int64_t> getScalarElementBytes(Type element) {
  if (!element) {
    return std::nullopt;
  }
  if (element.isF32() || element.isInteger(32)) {
    return 4;
  }
  if (element.isF16() || element.isBF16() || element.isInteger(16)) {
    return 2;
  }
  if (element.isInteger(8) || element.isInteger(1)) {
    return 1;
  }
  if (element.isInteger(64)) {
    return 8;
  }
  return std::nullopt;
}

static std::optional<int64_t> getElementBytes(Type type) {
  Type element;
  if (auto tile = dyn_cast<pto::TileBufType>(type)) {
    element = tile.getElementType();
  } else if (auto memref = dyn_cast<MemRefType>(type)) {
    element = memref.getElementType();
  } else if (auto unranked = dyn_cast<UnrankedMemRefType>(type)) {
    element = unranked.getElementType();
  } else if (auto ptr = dyn_cast<pto::PtrType>(type)) {
    element = ptr.getElementType();
  }
  return getScalarElementBytes(element);
}

/// Static physical byte footprint of a tile-shaped type: product of the
/// physical shape times the element width.  A dynamic dimension, a dynamic
/// element width, or an overflow yields nullopt.
static std::optional<int64_t> getTileFootprintBytes(Type type) {
  ArrayRef<int64_t> shape;
  Type element;
  if (auto tile = dyn_cast<pto::TileBufType>(type)) {
    shape = tile.getShape();
    element = tile.getElementType();
  } else {
    return std::nullopt;
  }
  auto elementBytes = getScalarElementBytes(element);
  if (!elementBytes) {
    return std::nullopt;
  }
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim) || dim < 0) {
      return std::nullopt;
    }
    if (__builtin_mul_overflow(elements, dim, &elements)) {
      return std::nullopt;
    }
  }
  int64_t bytes = 0;
  if (__builtin_mul_overflow(elements, *elementBytes, &bytes)) {
    return std::nullopt;
  }
  return bytes;
}

/// Row-major byte strides of a statically-shaped tile, or nullopt when the
/// shape is dynamic.  The physical footprint is row-major for the default tile
/// configuration, which is what FusionPlan sees; a tile whose element width is
/// unmodelled also yields nullopt.
static std::optional<SmallVector<int64_t, 4>>
getRowMajorElementStrides(pto::TileBufType tile) {
  SmallVector<int64_t, 4> strides(tile.getShape().size(), 1);
  int64_t running = 1;
  for (auto [index, dim] : llvm::enumerate(llvm::reverse(tile.getShape()))) {
    if (ShapedType::isDynamic(dim) || dim < 0) {
      return std::nullopt;
    }
    strides[tile.getShape().size() - 1 - index] = running;
    if (__builtin_mul_overflow(running, dim, &running)) {
      return std::nullopt;
    }
  }
  return strides;
}

static std::optional<int64_t> getStaticOffset(Value value) {
  return getConstantInteger(value);
}

static std::optional<int64_t>
getTileStorageBytesFromAddr(Operation *context, int64_t address) {
  std::optional<int64_t> result;
  context->getParentOfType<func::FuncOp>().walk([&](pto::AllocTileOp alloc) {
    auto allocAddr = alloc.getAddr();
    if (!allocAddr) {
      return WalkResult::advance();
    }
    auto addrConst = getConstantInteger(allocAddr);
    if (!addrConst || *addrConst != address) {
      return WalkResult::advance();
    }
    auto tileType = dyn_cast<pto::TileBufType>(alloc.getResult().getType());
    if (!tileType) {
      return WalkResult::advance();
    }
    auto elemBytes = getElementBytes(tileType.getElementType());
    if (!elemBytes) {
      return WalkResult::advance();
    }
    int64_t elements = 1;
    for (int64_t dim : tileType.getShape()) {
      if (dim <= 0) {
        return WalkResult::interrupt();
      }
      if (__builtin_mul_overflow(elements, dim, &elements)) {
        return WalkResult::interrupt();
      }
    }
    result = elements * *elemBytes;
    return WalkResult::interrupt();
  });
  return result;
}


static std::optional<int64_t>
getStaticSubviewByteOffset(memref::SubViewOp subview, MemRefType sourceType) {
  SmallVector<int64_t, 4> strides;
  int64_t sourceOffset = 0;
  if (failed(getStridesAndOffset(sourceType, strides, sourceOffset)) ||
      ShapedType::isDynamic(sourceOffset) ||
      llvm::any_of(strides, ShapedType::isDynamic)) {
    return std::nullopt;
  }
  auto elementBytes = getElementBytes(sourceType);
  if (!elementBytes) {
    return std::nullopt;
  }

  // resolveVmiStorageRoot(source) already accounts for sourceType's static
  // layout offset. Add only the subview's mixed offsets here.
  int64_t offset = 0;
  int64_t byteOffset = 0;
  auto mixedOffsets = subview.getMixedOffsets();
  if (mixedOffsets.size() != strides.size()) {
    return std::nullopt;
  }
  for (auto [mixed, stride] : llvm::zip(mixedOffsets, strides)) {
    auto value = dyn_cast<Value>(mixed);
    std::optional<int64_t> constant;
    if (value) {
      constant = getStaticOffset(value);
    } else if (auto attr = dyn_cast<Attribute>(mixed)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        constant = integer.getInt();
      }
    }
    if (!constant || stride < 0) {
      return std::nullopt;
    }
    int64_t term = 0;
    if (__builtin_mul_overflow(*constant, stride, &term) ||
        __builtin_add_overflow(offset, term, &offset)) {
      return std::nullopt;
    }
  }
  if (__builtin_mul_overflow(offset, *elementBytes, &byteOffset)) {
    return std::nullopt;
  }
  return byteOffset;
}

static std::optional<int64_t>
getStaticSubviewSpanBytes(memref::SubViewOp subview) {
  auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
  if (!resultType) {
    return std::nullopt;
  }
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  auto elementBytes = getElementBytes(resultType);
  if (failed(getStridesAndOffset(resultType, strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic) || !elementBytes) {
    return std::nullopt;
  }
  int64_t spanElements = 1;
  for (auto [dim, stride] : llvm::zip(resultType.getShape(), strides)) {
    if (ShapedType::isDynamic(dim) || stride < 0) {
      return std::nullopt;
    }
    if (dim == 0) {
      spanElements = 0;
      break;
    }
    int64_t term = 0;
    if (dim > 0 && __builtin_mul_overflow(dim - 1, stride, &term)) {
      return std::nullopt;
    }
    if (__builtin_add_overflow(spanElements, term, &spanElements)) {
      return std::nullopt;
    }
  }
  int64_t spanBytes = 0;
  if (__builtin_mul_overflow(spanElements, *elementBytes, &spanBytes)) {
    return std::nullopt;
  }
  return spanBytes;
}

// Storage bytes from a `pto.tile_shape` attribute on a castptr, used when the
// owning allocation cannot be found (e.g. the castptr was hoisted out of its
// allocation function).  Returns nullopt when the attribute is absent, the
// element width is unmodelled, or the shape has a non-positive dimension or
// overflows.
static std::optional<int64_t>
getTileShapeAttrStorageBytes(pto::CastPtrOp cast) {
  auto shapeAttr = cast->getAttrOfType<DenseI64ArrayAttr>("pto.tile_shape");
  if (!shapeAttr) {
    return std::nullopt;
  }
  auto elemBytes = getElementBytes(cast.getResult().getType());
  if (!elemBytes) {
    return std::nullopt;
  }
  int64_t elements = 1;
  for (int64_t dim : shapeAttr.asArrayRef()) {
    if (dim <= 0) {
      return std::nullopt;
    }
    if (__builtin_mul_overflow(elements, dim, &elements)) {
      return std::nullopt;
    }
  }
  return elements * *elemBytes;
}

// A pto.castptr of a constant integer/index address is a UB view root.  The
// storage extent comes from the owning allocation, or from the castptr's
// `pto.tile_shape` attribute when the allocation cannot be found; it may stay
// unknown (nullopt).  Returns nullopt when the input is not a constant
// integer/index, telling the caller to keep walking the chain through the
// castptr's input.
static std::optional<pto::VmiStorageRoot>
resolveConstAddrCastPtrRoot(pto::CastPtrOp cast) {
  Value input = cast.getInput();
  if (!isa<IntegerType>(input.getType()) && !input.getType().isIndex()) {
    return std::nullopt;
  }
  auto address = getConstantInteger(input);
  if (!address) {
    return std::nullopt;
  }
  auto storage = getTileStorageBytesFromAddr(cast, *address);
  if (!storage) {
    storage = getTileShapeAttrStorageBytes(cast);
  }
  return pto::VmiStorageRoot{*address, storage,
                             cast.getResult().getType()};
}

// The storage root of a memref.subview: the source's root shifted by the
// subview's static byte offset, with the subview's static byte span as the
// extent.  Any unknown piece makes the whole subview unresolvable.
static std::optional<pto::VmiStorageRoot>
resolveMemRefSubviewRoot(memref::SubViewOp subview) {
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceType) {
    return std::nullopt;
  }
  auto sourceRoot = mlir::pto::resolveVmiStorageRoot(subview.getSource());
  auto byteOffset = getStaticSubviewByteOffset(subview, sourceType);
  auto spanBytes = getStaticSubviewSpanBytes(subview);
  if (!sourceRoot || !byteOffset || !spanBytes) {
    return std::nullopt;
  }
  int64_t address = 0;
  if (__builtin_add_overflow(sourceRoot->address, *byteOffset, &address)) {
    return std::nullopt;
  }
  return pto::VmiStorageRoot{address, *spanBytes,
                             subview.getResult().getType()};
}

} // namespace

std::optional<mlir::pto::VmiStorageRoot>
mlir::pto::resolveVmiStorageRoot(Value base) {
  while (base) {
    if (auto address = base.getDefiningOp<pto::TileBufAddrOp>()) {
      auto alloc = address.getSrc().getDefiningOp<pto::AllocTileOp>();
      if (alloc && alloc->hasAttr("pto.cv.borrowed")) {
        auto cast = alloc.getAddr().getDefiningOp<pto::CastPtrOp>();
        auto arg = cast ? dyn_cast<BlockArgument>(cast.getInput()) : BlockArgument{};
        auto function = arg ? dyn_cast_or_null<func::FuncOp>(arg.getOwner()->getParentOp()) : func::FuncOp{};
        if (function && function->hasAttr("pto.cv.distinct_buffers"))
          return pto::VmiStorageRoot{0, getTileFootprintBytes(alloc.getResult().getType()),
                                    alloc.getResult().getType(), arg};
      }
    }
    if (auto cast = base.getDefiningOp<pto::CastPtrOp>()) {
      if (auto root = resolveConstAddrCastPtrRoot(cast)) {
        return root;
      }
      base = cast.getInput();
      continue;
    }
    if (auto cast = base.getDefiningOp<memref::CastOp>()) {
      base = cast.getSource();
      continue;
    }
    if (auto subview = base.getDefiningOp<memref::SubViewOp>()) {
      return resolveMemRefSubviewRoot(subview);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool mlir::pto::mayAliasVmiStorageRoot(const VmiStorageRoot &lhs,
                                       const VmiStorageRoot &rhs) {
  if (lhs.logicalRoot || rhs.logicalRoot) {
    if (!lhs.logicalRoot || !rhs.logicalRoot) return true;
    if (lhs.logicalRoot != rhs.logicalRoot) return false;
  }
  if (lhs.address == rhs.address) {
    return true;
  }
  // With no extent on either root there is no proof of disjointness.  Keep
  // the alias result conservative instead of treating one arbitrary root as
  // the "known" side and dereferencing an empty optional below.
  if (!lhs.storageBytes && !rhs.storageBytes) {
    return true;
  }
  if (!lhs.storageBytes || !rhs.storageBytes) {
    const auto &known = lhs.storageBytes ? lhs : rhs;
    int64_t knownEnd = known.address;
    if (__builtin_add_overflow(known.address, *known.storageBytes, &knownEnd)) {
      return true;
    }
    const auto &unknown = lhs.storageBytes ? rhs : lhs;
    return unknown.address >= known.address && unknown.address < knownEnd;
  }
  if (*lhs.storageBytes < 0 || *rhs.storageBytes < 0) {
    return true;
  }
  const int64_t lhsEnd = lhs.address > INT64_MAX - *lhs.storageBytes
                             ? INT64_MAX
                             : lhs.address + *lhs.storageBytes;
  const int64_t rhsEnd = rhs.address > INT64_MAX - *rhs.storageBytes
                             ? INT64_MAX
                             : rhs.address + *rhs.storageBytes;
  return lhs.address < rhsEnd && rhs.address < lhsEnd;
}

bool mlir::pto::mayAliasVmiAccess(const VmiAccessLocation &lhs,
                                  const VmiAccessLocation &rhs) {
  if (!mayAliasVmiStorageRoot(lhs.root, rhs.root)) {
    return false;
  }
  if (lhs.root.viewType != rhs.root.viewType || lhs.accessBytes <= 0 ||
      rhs.accessBytes <= 0) {
    return true;
  }
  // Overflow-checked byte-interval overlap of the two element-indexed
  // accesses (shared helper; unknowns conservatively report overlap).
  return pto::mayOverlapStaticByteRanges(
      lhs.root.address, getConstantInteger(lhs.elementOffset), lhs.accessBytes,
      rhs.root.address, getConstantInteger(rhs.elementOffset), rhs.accessBytes,
      getElementBytes(lhs.root.viewType));
}

namespace {

static bool isTileLikeType(Type type) {
  return isa<pto::TileBufType, pto::MultiTileBufType>(type);
}

/// Whether the row-major element math may place this tile's bytes.  The config
/// fields are graded one by one, because only some of them can move a byte.
/// The layout fields do move bytes: getNoneBoxPointerStrides() builds the
/// strides from bLayout (see PTOResolveBufferSelect.cpp), a boxed sLayout
/// pulls s_fractal_size into the pointer arithmetic (getTilePointerStrides
/// through getBoxedPointerStrides), and a RowPlusOne compactMode adds one
/// element to the outer stride, which the footprint product below does not
/// count.  The remaining fields are inert here: getFractalInnerDims() is only
/// reachable from the boxed path and the tile verifier reads the fractal only
/// after sLayout != NoneBox (see PTOTileLayoutVerification.cpp), while pad is
/// a fill value for copies (TFillPad), never part of a footprint.
///
/// So a row-major, non-boxed, non-compact tile is laid out exactly like the
/// default config whatever its fractal size or pad says.  Every other config
/// keeps reporting unknown rather than being guessed: the NZ destination of
/// RowPlusOne, for instance, is both col-major and boxed.
static bool hasRowMajorPlacement(pto::TileBufType type) {
  auto config = type.getConfig();
  if (!config) {
    return false;
  }
  auto def = pto::TileBufConfigAttr::getDefault(type.getContext());
  return config.getBLayout() == def.getBLayout() &&
         config.getSLayout() == def.getSLayout() &&
         config.getCompactMode() == def.getCompactMode();
}

/// The storage identity a view chain bottoms out at, plus the static byte
/// region the view touches inside it.
struct RootRegion {
  Value root;
  std::optional<int64_t> byteOffset;
  std::optional<int64_t> byteSpan;
};

/// Byte span touched by a `sizes`-shaped window walked with \p strides.  This
/// is deliberately NOT `product(sizes) * elementBytes`: a subview of a
/// 16x512 tile with sizes [16,128] touches 15*512+128 elements, not 16*128, so
/// using the product would under-approximate the region and let two genuinely
/// overlapping subviews be reported as disjoint.
static std::optional<int64_t> getStridedSpanBytes(ArrayRef<int64_t> sizes,
                                                  ArrayRef<int64_t> strides,
                                                  int64_t elementBytes) {
  if (sizes.size() != strides.size()) {
    return std::nullopt;
  }
  int64_t elements = 1;
  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    if (ShapedType::isDynamic(size) || size < 0 || stride < 0) {
      return std::nullopt;
    }
    if (size == 0) {
      return int64_t{0};
    }
    int64_t term = 0;
    if (__builtin_mul_overflow(size - 1, stride, &term) ||
        __builtin_add_overflow(elements, term, &elements)) {
      return std::nullopt;
    }
  }
  int64_t bytes = 0;
  if (__builtin_mul_overflow(elements, elementBytes, &bytes)) {
    return std::nullopt;
  }
  return bytes;
}

/// Combine a base byte offset with a view-local byte offset.  Either being
/// unknown makes the result unknown.
static std::optional<int64_t> combineByteOffset(std::optional<int64_t> base,
                                                std::optional<int64_t> local) {
  if (!base || !local) {
    return std::nullopt;
  }
  int64_t result = 0;
  if (__builtin_add_overflow(*base, *local, &result)) {
    return std::nullopt;
  }
  return result;
}

/// Byte region a `pto.subview` window covers inside its source.  The source
/// must be a tile the row-major strides place (hasRowMajorPlacement): a boxed or
/// column-major parent needs its real layout to place the window, and guessing
/// would risk an under-approximation, so those report unknown instead.
struct SubviewRegion {
  int64_t byteOffset = 0;
  int64_t byteSpan = 0;
};

static std::optional<SubviewRegion> getSubviewRegion(pto::SubViewOp subview) {
  auto sourceType = dyn_cast<pto::TileBufType>(subview.getSource().getType());
  if (!sourceType) {
    return std::nullopt;
  }
  if (!hasRowMajorPlacement(sourceType)) {
    return std::nullopt;
  }
  auto elementBytes = getScalarElementBytes(sourceType.getElementType());
  auto strides = getRowMajorElementStrides(sourceType);
  if (!elementBytes || !strides) {
    return std::nullopt;
  }

  auto offsets = subview.getOffsets();
  if (offsets.size() != strides->size()) {
    return std::nullopt;
  }
  int64_t offsetElements = 0;
  for (auto [offset, stride] : llvm::zip(offsets, *strides)) {
    auto constant = getConstantInteger(offset);
    if (!constant || *constant < 0) {
      return std::nullopt;
    }
    int64_t term = 0;
    if (__builtin_mul_overflow(*constant, stride, &term) ||
        __builtin_add_overflow(offsetElements, term, &offsetElements)) {
      return std::nullopt;
    }
  }

  SmallVector<int64_t, 4> sizes;
  for (Attribute size : subview.getSizes()) {
    auto integer = dyn_cast<IntegerAttr>(size);
    if (!integer) {
      return std::nullopt;
    }
    sizes.push_back(integer.getInt());
  }
  auto spanElements = getStridedSpanBytes(sizes, *strides, *elementBytes);
  if (!spanElements) {
    return std::nullopt;
  }

  int64_t offsetBytes = 0;
  if (__builtin_mul_overflow(offsetElements, *elementBytes, &offsetBytes)) {
    return std::nullopt;
  }
  return SubviewRegion{offsetBytes, *spanElements};
}

static std::optional<RootRegion> resolveRootRegion(Value value);

/// Root region of a pto.alloc_multi_tile result: the sum of the slot
/// footprints when the slot layout is row-major placeable, otherwise an
/// unknown extent.
static std::optional<RootRegion>
resolveAllocMultiTileRegion(pto::AllocMultiTileOp alloc) {
  auto multiType = dyn_cast<pto::MultiTileBufType>(alloc.getResult().getType());
  if (!multiType) {
    return std::nullopt;
  }
  auto slotType = dyn_cast<pto::TileBufType>(multiType.getSlotType());
  if (!slotType || !hasRowMajorPlacement(slotType)) {
    return RootRegion{alloc.getResult(), int64_t{0}, std::nullopt};
  }
  auto slotBytes = getTileFootprintBytes(slotType);
  if (!slotBytes) {
    return RootRegion{alloc.getResult(), int64_t{0}, std::nullopt};
  }
  int64_t total = 0;
  if (__builtin_mul_overflow(*slotBytes,
                             static_cast<int64_t>(multiType.getCount()),
                             &total)) {
    return RootRegion{alloc.getResult(), int64_t{0}, std::nullopt};
  }
  return RootRegion{alloc.getResult(), int64_t{0}, total};
}

/// Root region of a pto.multi_tile_get result: the source's root, offset by
/// the slot's byte offset when both the slot offset and the slot footprint
/// are static.  The slot's own config decides where its bytes live, so a slot
/// the row-major stride cannot place falls back to the unknown extent.
static std::optional<RootRegion>
resolveMultiTileGetRegion(pto::MultiTileGetOp get) {
  auto base = resolveRootRegion(get.getSource());
  if (!base) {
    return std::nullopt;
  }
  auto slotType = dyn_cast<pto::TileBufType>(get.getResult().getType());
  if (!slotType || !hasRowMajorPlacement(slotType)) {
    return RootRegion{base->root, std::nullopt, std::nullopt};
  }
  auto slotBytes = getTileFootprintBytes(slotType);
  auto slot = getConstantInteger(get.getSlot());
  if (!slotBytes || !slot || *slot < 0) {
    return RootRegion{base->root, std::nullopt, slotBytes};
  }
  int64_t local = 0;
  if (__builtin_mul_overflow(*slot, *slotBytes, &local)) {
    return RootRegion{base->root, std::nullopt, slotBytes};
  }
  return RootRegion{base->root, combineByteOffset(base->byteOffset, local),
                    slotBytes};
}

/// Root region of a pto.subview window over a tile: the source's root, the
/// window's byte offset combined with the base offset, and the window's byte
/// span as the extent.  An unplaceable window reports unknown offset and
/// span instead of guessing.
static std::optional<RootRegion>
resolveTileSubviewRegion(pto::SubViewOp subview) {
  auto base = resolveRootRegion(subview.getSource());
  if (!base) {
    return std::nullopt;
  }
  auto window = getSubviewRegion(subview);
  if (!window) {
    return RootRegion{base->root, std::nullopt, std::nullopt};
  }
  return RootRegion{base->root,
                    combineByteOffset(base->byteOffset, window->byteOffset),
                    window->byteSpan};
}

/// Walk a tile-typed value back to the allocation that owns its storage and
/// the static byte region it touches.  Returns nullopt when the storage cannot
/// be attributed to a known allocation; the caller then records an
/// unknown-root access.
static std::optional<RootRegion> resolveRootRegion(Value value) {
  if (!value || !isTileLikeType(value.getType())) {
    return std::nullopt;
  }

  Operation *def = value.getDefiningOp();
  // A block argument (function parameter, region iter_arg) is its own root:
  // distinct arguments denote distinct buffers supplied by the caller.
  if (!def) {
    return RootRegion{value, int64_t{0},
                      getTileFootprintBytes(value.getType())};
  }

  if (auto alloc = dyn_cast<pto::AllocTileOp>(def)) {
    return RootRegion{alloc.getResult(), int64_t{0},
                      getTileFootprintBytes(alloc.getResult().getType())};
  }

  if (auto alloc = dyn_cast<pto::AllocMultiTileOp>(def)) {
    return resolveAllocMultiTileRegion(alloc);
  }

  if (auto get = dyn_cast<pto::MultiTileGetOp>(def)) {
    return resolveMultiTileGetRegion(get);
  }

  if (auto subview = dyn_cast<pto::SubViewOp>(def)) {
    return resolveTileSubviewRegion(subview);
  }

  // bitcast / treshape reinterpret the same storage without moving or resizing
  // it, so root, offset and span all carry over unchanged.
  if (auto bitcast = dyn_cast<pto::BitcastOp>(def)) {
    if (!isTileLikeType(bitcast.getSrc().getType())) {
      return std::nullopt;
    }
    return resolveRootRegion(bitcast.getSrc());
  }
  if (auto reshape = dyn_cast<pto::TReshapeOp>(def)) {
    if (!isTileLikeType(reshape.getSrc().getType())) {
      return std::nullopt;
    }
    return resolveRootRegion(reshape.getSrc());
  }

  // `pto.tassign` deliberately does NOT reach here as a followed view: it
  // rebinds the handle to a new runtime address, so its source tells us
  // nothing about where the result lives.  It falls through to the
  // unknown-root case below, as does any other unrecognized producer.
  return std::nullopt;
}

} // namespace

std::optional<mlir::pto::VmiTileAccess>
mlir::pto::resolveVmiTileAccess(Value tile) {
  if (!tile || !isa<pto::TileBufType>(tile.getType())) {
    return std::nullopt;
  }

  VmiTileAccess access;
  access.accessValue = tile;
  access.viewType = tile.getType();

  if (auto region = resolveRootRegion(tile)) {
    access.root = region->root;
    access.byteOffset = region->byteOffset;
    access.byteSize = region->byteSpan;
    return access;
  }

  // Unknown storage: keep the value as the root for identification, but mark it
  // so it never shares a region with anything.
  access.root = tile;
  access.unknownRoot = true;
  return access;
}

mlir::pto::VmiAccessRelation
mlir::pto::relateVmiTileAccess(const VmiTileAccess &lhs,
                               const VmiTileAccess &rhs) {
  // The same SSA value always denotes the same region, even when the byte
  // interval could not be computed: a tile value is a fixed view, so two
  // accesses through it cannot be placed differently.
  if (lhs.accessValue && lhs.accessValue == rhs.accessValue) {
    return VmiAccessRelation::MustAlias;
  }
  // An access whose storage could not be attributed may alias anything.
  if (lhs.unknownRoot || rhs.unknownRoot) {
    return VmiAccessRelation::MayAlias;
  }
  // Distinct logical buffers never alias.
  if (lhs.root != rhs.root) {
    return VmiAccessRelation::NoAlias;
  }
  // Same root: only a fully static interval can be ordered.
  if (!lhs.byteOffset || !lhs.byteSize || !rhs.byteOffset || !rhs.byteSize) {
    return VmiAccessRelation::MayAlias;
  }
  // An empty access cannot overlap anything.  A negative span never comes out
  // of the footprint helpers (they reject negative dimensions), so if one ever
  // appears, treat it as unattributable rather than silently disjoint.
  if (*lhs.byteSize == 0 || *rhs.byteSize == 0) {
    return VmiAccessRelation::NoAlias;
  }
  if (*lhs.byteSize < 0 || *rhs.byteSize < 0) {
    return VmiAccessRelation::MayAlias;
  }
  const int64_t lhsBegin = *lhs.byteOffset;
  const int64_t rhsBegin = *rhs.byteOffset;
  const int64_t lhsEnd = lhsBegin > INT64_MAX - *lhs.byteSize
                             ? INT64_MAX
                             : lhsBegin + *lhs.byteSize;
  const int64_t rhsEnd = rhsBegin > INT64_MAX - *rhs.byteSize
                             ? INT64_MAX
                             : rhsBegin + *rhs.byteSize;
  if (lhsBegin == rhsBegin && lhsEnd == rhsEnd) {
    return VmiAccessRelation::MustAlias;
  }
  if (lhsBegin >= rhsEnd || rhsBegin >= lhsEnd) {
    return VmiAccessRelation::NoAlias;
  }
  return VmiAccessRelation::MayAlias;
}
