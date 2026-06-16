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

// f32 kernels

extern "C" __global__ AICORE void TROWEXPANDDIV_f32_40x32_hp(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);
extern "C" __global__ AICORE void TROWEXPANDDIV_f16_16x32(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTROWEXPANDDIV_f32_40x32_hp(float *src0, float *src1, float *dst, void *stream) {
    TROWEXPANDDIV_f32_40x32_hp<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}



void LaunchTROWEXPANDDIV_f16_16x32(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDDIV_f16_16x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}
