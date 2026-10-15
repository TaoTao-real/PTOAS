// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool isA5NdToNzTMov(TMovOp op, bool isA5) {
  auto src = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto dst = dyn_cast<pto::TileBufType>(op.getDst().getType());
  return isA5 && src && dst && isVecTile(src) && isVecTile(dst) &&
         src.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         src.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox) &&
         dst.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::ColMajor) &&
         dst.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static int64_t getNdToNzValidBound(pto::TileBufType type, unsigned axis) {
  int64_t valid = type.getValidShape()[axis];
  return ShapedType::isDynamic(valid) ? type.getShape()[axis] : valid;
}

static bool usesNdToNzRowPlusOne(pto::TileBufType type) {
  return type.getCompactModeI32() ==
         static_cast<int32_t>(pto::CompactMode::RowPlusOne);
}

static std::optional<int64_t> getNdToNzElementsPerBlock(int64_t elemBytes) {
  switch (elemBytes) {
  case 1:
  case 2:
  case 4:
  case 8:
    return mlir::pto::kValue32 / elemBytes;
  default:
    return std::nullopt;
  }
}

static std::optional<int64_t>
getNdToNzCapacityBytes(pto::TileBufType type, int64_t elemBytes) {
  ArrayRef<int64_t> shape = type.getShape();
  auto elements = tmovCheckedElements(shape);
  if (usesNdToNzRowPlusOne(type)) {
    // Keep this capacity consistent with PlanMemory's col-major allocation.
    auto rowStride = tmovCheckedAdd(shape[0], 1);
    auto prefix = rowStride ? tmovCheckedMul(shape[1] - 1, *rowStride)
                            : std::nullopt;
    elements = prefix ? tmovCheckedAdd(*prefix, shape[0]) : std::nullopt;
  }
  return elements ? tmovCheckedMul(*elements, elemBytes) : std::nullopt;
}

static std::optional<int64_t>
getNdToNzWriteBytes(pto::TileBufType src, pto::TileBufType dst,
                    int64_t elementsPerBlock) {
  constexpr int64_t blockBytes = mlir::pto::kValue32;
  int64_t columns = getNdToNzValidBound(dst, 1);
  if (columns == 0) {
    return 0;
  }
  auto rows = tmovAlign16(getNdToNzValidBound(dst, 0));
  if (rows && usesNdToNzRowPlusOne(dst)) {
    rows = tmovCheckedAdd(*rows, 1);
  }
  auto columnBias = tmovCheckedAdd(columns, elementsPerBlock - 1);
  if (!rows || !columnBias) {
    return std::nullopt;
  }
  // TMovToVecNd2Nz scatters C0 blocks at align16(dst valid rows), but
  // TMovNd2NzLoop writes src valid rows. Include a full final C0 block.
  int64_t columnBlocks = *columnBias / elementsPerBlock;
  auto prefixRows = tmovCheckedMul(columnBlocks - 1, *rows);
  auto writtenRows = prefixRows
                         ? tmovCheckedAdd(*prefixRows,
                                          getNdToNzValidBound(src, 0))
                         : std::nullopt;
  return writtenRows ? tmovCheckedMul(*writtenRows, blockBytes) : std::nullopt;
}

static LogicalResult verifyA5NdToNzStorage(TMovOp op, bool isA5) {
  if (!isA5NdToNzTMov(op, isA5)) {
    return success();
  }
  auto src = cast<pto::TileBufType>(op.getSrc().getType());
  auto dst = cast<pto::TileBufType>(op.getDst().getType());
  Type element = dst.getElementType();
  if (pto::isPTOFloat4PackedType(element)) {
    return success();
  }
  int64_t elemBytes = pto::getPTOStorageElemByteSize(element);
  auto elementsPerBlock = getNdToNzElementsPerBlock(elemBytes);
  if (!elementsPerBlock) {
    return success();
  }
  const bool hasColumns = getNdToNzValidBound(dst, 1) > 0;
  const bool hasEmptyRows =
      src.getValidShape()[0] == 0 || dst.getValidShape()[0] == 0;
  if (hasColumns && hasEmptyRows) {
    return op.emitOpError(
        "expects positive source and destination valid rows for A5 ND-to-NZ "
        "with nonzero valid columns");
  }
  auto required = getNdToNzWriteBytes(src, dst, *elementsPerBlock);
  auto capacity = getNdToNzCapacityBytes(dst, elemBytes);
  if (!required || !capacity) {
    return op.emitOpError(
        "cannot compute A5 ND-to-NZ destination storage bounds without overflow");
  }
  if (*required > *capacity) {
    return op.emitOpError()
           << "A5 ND-to-NZ destination write span requires " << *required
           << " bytes, but the physical tile provides " << *capacity
           << " bytes; use a padded NZ tile and keep pipe split geometry "
              "consistent with its physical rows";
  }
  return success();
}
