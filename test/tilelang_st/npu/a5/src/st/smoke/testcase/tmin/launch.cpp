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

// Smoke case: f32 32x32

extern "C" __global__ AICORE void TMIN_f32_32x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);
extern "C" __global__ AICORE void TMIN_i32_64x64_v60x60(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTMIN_i32_64x64_v60x60(void *a, void *b, void *c, void *stream) {
    TMIN_i32_64x64_v60x60<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}



void LaunchTMIN_f32_32x32(void *a, void *b, void *c, void *stream) {
    TMIN_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
