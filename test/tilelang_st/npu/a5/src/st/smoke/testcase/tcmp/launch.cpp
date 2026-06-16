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

// Case 1: f16 32x32 eq (half_32x32_32x32)

extern "C" __global__ AICORE void TCMP_f32_8x64_gt(__gm__ float *a, __gm__ float *b, __gm__ int8_t *c);
extern "C" __global__ AICORE void TCMP_i32_4x64_ne(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);
extern "C" __global__ AICORE void TCMP_i16_32x32_eq(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int8_t *c);

void LaunchTCMP_f32_8x64_gt(float *a, float *b, int8_t *c, void *stream) {
    TCMP_f32_8x64_gt<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ int8_t *)c);
}



void LaunchTCMP_i32_4x64_ne(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_4x64_ne<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}



void LaunchTCMP_i16_32x32_eq(int16_t *a, int16_t *b, int8_t *c, void *stream) {
    TCMP_i16_32x32_eq<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int8_t *)c);
}
