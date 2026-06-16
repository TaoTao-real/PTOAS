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

// Scalar value for comparison (must match gen_data.py SCALAR)
static constexpr float TCMP_SCALAR_F32 = 5.0f;
static constexpr int32_t TCMP_SCALAR_I32 = 5;

// Case 0: f32 1x64

extern "C" __global__ AICORE void TCMP_f32_1x64(__gm__ float *src, __gm__ uint8_t *dst, float scalar);
extern "C" __global__ AICORE void TCMP_i32_16x32(__gm__ int32_t *src, __gm__ uint8_t *dst, int32_t scalar);

void LaunchTCMP_f32_1x64(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_1x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}



void LaunchTCMP_i32_16x32(int32_t *src, uint8_t *dst, void *stream) {
    TCMP_i32_16x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I32);
}
