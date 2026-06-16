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

extern "C" __global__ AICORE void TROWPROD_f32_15x192(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWPROD_i32_7x448_valid7x447(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWPROD_i32_7x448_valid7x447(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_7x448_valid7x447<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}



void LaunchTROWPROD_f32_15x192(float *src, float *dst, void *stream) {
    TROWPROD_f32_15x192<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
