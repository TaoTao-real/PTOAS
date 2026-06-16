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

// Scalar value for remainder (must match gen_data.py SCALAR)
static constexpr float TREMS_SCALAR_F32 = 3.0f;

// Case 0: f32 32x64

extern "C" __global__ AICORE void TREMS_f32_32x64(__gm__ float *src, __gm__ float *dst, float scalar);
extern "C" __global__ AICORE void TREMS_f16_63x64(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);

void LaunchTREMS_f32_32x64(float *src, float *dst, void *stream) {
    TREMS_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TREMS_SCALAR_F32);
}



void LaunchTREMS_f16_63x64(unsigned short *src, unsigned short *dst, void *stream) {
    TREMS_f16_63x64<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}
