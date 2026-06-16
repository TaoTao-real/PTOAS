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

// Scalar value used for element-wise maximum (must match gen_data.py SCALAR)
static constexpr float TMAXS_SCALAR_F32 = 5.0f;

// Case 0: f32 32x64

extern "C" __global__ AICORE void TMAXS_f32_32x64(__gm__ float *src, __gm__ float *dst, float scalar);
extern "C" __global__ AICORE void TMAXS_i16_15x192(__gm__ int16_t *src, __gm__ int16_t *dst, int16_t scalar);

void LaunchTMAXS_f32_32x64(float *src, float *dst, void *stream) {
    TMAXS_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TMAXS_SCALAR_F32);
}



void LaunchTMAXS_i16_15x192(int16_t *src, int16_t *dst, void *stream) {
    TMAXS_i16_15x192<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst, (int16_t)5);
}
