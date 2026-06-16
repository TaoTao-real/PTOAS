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

extern "C" __global__ AICORE void TROWARGMIN_uint32_float_8x1_8x8_8x8(__gm__ float *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TROWARGMIN_int32_half_16x1_13x16_13x13(__gm__ uint16_t *src, __gm__ int32_t *dst);

void LaunchTROWARGMIN_uint32_float_8x1_8x8_8x8(float *src, uint32_t *dst, void *stream) {
    TROWARGMIN_uint32_float_8x1_8x8_8x8<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}



void LaunchTROWARGMIN_int32_half_16x1_13x16_13x13(uint16_t *src, int32_t *dst, void *stream) {
    TROWARGMIN_int32_half_16x1_13x16_13x13<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}
