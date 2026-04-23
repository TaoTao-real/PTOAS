// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTOTypeUtils.h"

#include "PTO/IR/PTO.h"

using namespace mlir;
using namespace mlir::pto;

bool mlir::pto::isPTOFloat8Type(Type t) {
  return t.isFloat8E4M3() || t.isFloat8E4M3FN() || t.isFloat8E4M3FNUZ() ||
         t.isFloat8E4M3B11FNUZ() || t.isFloat8E5M2() || t.isFloat8E5M2FNUZ();
}

bool mlir::pto::isPTOHiFloat8Type(Type t) { return isa<HiF8Type>(t); }

bool mlir::pto::isPTOFloat4PackedType(Type t) {
  return isa<F4E1M2x2Type, F4E2M1x2Type>(t);
}

bool mlir::pto::isPTOLowPrecisionType(Type t) {
  return isPTOFloat8Type(t) || isPTOHiFloat8Type(t) || isPTOFloat4PackedType(t);
}

unsigned mlir::pto::getPTOStorageElemByteSize(Type t) {
  if (isPTOLowPrecisionType(t))
    return 1;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth() / 8;
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth() / 8;
  return 0;
}
