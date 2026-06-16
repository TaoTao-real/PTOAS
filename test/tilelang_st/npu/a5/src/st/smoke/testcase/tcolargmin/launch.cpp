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

// Case 0: f32 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)

extern "C" __global__ AICORE void TCOLARGMIN_f32_1x256(__gm__ int32_t *dst, __gm__ float *tmp, __gm__ float *src);
extern "C" __global__ AICORE void TCOLARGMIN_f16_1x256(__gm__ int32_t *dst, __gm__ half *tmp, __gm__ half *src);

void LaunchTCOLARGMIN_f32_1x256(int32_t *dst, float *tmp, float *src, void *stream) {
    TCOLARGMIN_f32_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ float *)tmp, (__gm__ float *)src);
}



void LaunchTCOLARGMIN_f16_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMIN_f16_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ half *)tmp, (__gm__ half *)src);
}
