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

// Case 0: f32 1x256 (input: 1x256, output: 1x256)

extern "C" __global__ AICORE void TCOLPROD_f32_1x256(__gm__ float *dst, __gm__ float *src);
extern "C" __global__ AICORE void TCOLPROD_i16_1x256(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLPROD_f32_1x256(float *dst, float *src, void *stream) {
    TCOLPROD_f32_1x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}



void LaunchTCOLPROD_i16_1x256(void *dst, void *src, void *stream) {
    TCOLPROD_i16_1x256<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}
