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

// Case: f32 1x32

extern "C" __global__ AICORE void TSORT32_f16_1x32(__gm__ uint16_t *src, __gm__ uint32_t *idx, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TSORT32_f32_2x13(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_2x13(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_2x13<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}



void LaunchTSORT32_f16_1x32(uint16_t *src, uint32_t *idx, uint16_t *dst, void *stream) {
    TSORT32_f16_1x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)idx, (__gm__ uint16_t *)dst);
}
