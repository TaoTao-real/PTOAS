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

// Case 1: fp32_32_64_1_64

extern "C" __global__ AICORE void TCOLEXPANDDIV_fp32_8_32_1_32(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);
extern "C" __global__ AICORE void TCOLEXPANDDIV_int32_16_32_1_32(__gm__ int32_t *src0, __gm__ int32_t *src1, __gm__ int32_t *dst);

void LaunchTCOLEXPANDDIV_fp32_8_32_1_32(float *src0, float *src1, float *dst, void *stream) {
    TCOLEXPANDDIV_fp32_8_32_1_32<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}



void LaunchTCOLEXPANDDIV_int32_16_32_1_32(int32_t *src0, int32_t *src1, int32_t *dst, void *stream) {
    TCOLEXPANDDIV_int32_16_32_1_32<<<1, nullptr, stream>>>((__gm__ int32_t *)src0, (__gm__ int32_t *)src1, (__gm__ int32_t *)dst);
}
