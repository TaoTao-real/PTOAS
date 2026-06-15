// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
// LICENSE in the root of the software repository for the full text of the
// License.

//===- MemoryConsistencyAttrs.h --------------------------------*- C++ -*-===//
// Attribute names produced by the memory consistency pass and consumed by
// lowering.
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYCONSISTENCYATTRS_H
#define PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYCONSISTENCYATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::pto {

static constexpr llvm::StringLiteral kTNotifyDrainMte2AttrName =
    "__pto.emitc.tnotify_drain_mte2";
static constexpr llvm::StringLiteral kTNotifyDrainMte3AttrName =
    "__pto.emitc.tnotify_drain_mte3";
static constexpr llvm::StringLiteral kTNotifyDrainFixAttrName =
    "__pto.emitc.tnotify_drain_fix";
static constexpr llvm::StringLiteral kTNotifyDsbDdrAttrName =
    "__pto.emitc.tnotify_dsb_ddr";
static constexpr llvm::StringLiteral kTNotifyCleanGmCacheAttrName =
    "__pto.emitc.tnotify_clean_gm_cache";
static constexpr llvm::StringLiteral kAcquireInvalidateGmCacheAttrName =
    "__pto.emitc.acquire_invalidate_gm_cache";

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_MEMORYCONSISTENCY_MEMORYCONSISTENCYATTRS_H
