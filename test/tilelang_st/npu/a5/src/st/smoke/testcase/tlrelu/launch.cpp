// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 32x64 -> dst 32x128 (valid 32x64)

extern "C" __global__ AICORE void TLRELU_f32_32x64_dst128(__gm__ float *src, __gm__ float *dst, float slope);
extern "C" __global__ AICORE void TLRELU_f32_7x448_dst512(__gm__ float *src, __gm__ float *dst, float slope);

void LaunchTLRELU_f32_7x448_dst512(float *src, float *dst, float slope, void *stream) {
    TLRELU_f32_7x448_dst512<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, slope);
}



void LaunchTLRELU_f32_32x64_dst128(float *src, float *dst, float slope, void *stream) {
    TLRELU_f32_32x64_dst128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, slope);
}
