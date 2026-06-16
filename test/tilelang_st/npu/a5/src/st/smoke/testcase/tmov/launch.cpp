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

// Case 0: f32 64x64

extern "C" __global__ AICORE void TMOV_f32_32x32(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TMOV_f16_32x32(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTMOV_f32_32x32(float *src, float *dst, void *stream) {
    TMOV_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}



void LaunchTMOV_f16_32x32(uint16_t *src, uint16_t *dst, void *stream) {
    TMOV_f16_32x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}
