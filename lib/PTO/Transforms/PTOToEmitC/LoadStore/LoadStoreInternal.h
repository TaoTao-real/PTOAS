// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Internal helper declarations shared within the LoadStore lowering domain.
#pragma once

#include "../PTOToEmitCEmitters.h"

namespace mlir {
namespace pto {

using namespace mlir;

void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter, StringRef callee, ArrayAttr templateArgs, ValueRange operands, Value dst);

// Rebuild the GM descriptor handed to the ISA for packed-FP4 UB transfers.
// The IR keeps descriptors in address units (carrier strides drive typed
// pointers, pto.get_tensor_view_stride, and nested partition views), but the A5
// vec paths - TLoadVecND2ND/TLoadVecDN2DN and the TStore mirrors - run every
// stride through GetByteSize<float4_e2m1x2_t>(n) == (n + 1) >> 1 and therefore
// read them as nibble counts. Scaling the transfer-axis strides here converts
// the descriptor exactly once, at the consumer that needs it. Unpacked
// transfers and the L1/L0 paths - which scale with sizeof(L1Type) and so already
// count carriers - return the original descriptor.
FailureOr<Value> buildPackedFp4TransferGlobalTensor(
    ConversionPatternRewriter &rewriter, Operation *anchor, Value gmTensor,
    Value gmSource, pto::PartitionTensorViewType partitionType,
    pto::TileBufType tileType);

void populateLoadStoreTLoadPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateLoadStoreTMatmulPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);
void populateLoadStoreTStorePatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx);

} // namespace pto
} // namespace mlir
