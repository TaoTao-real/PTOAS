// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>
#include <cstring>

#ifndef AICORE
#define AICORE [aicore]
#endif

static constexpr float TDIVS_SCALAR_F32 = 3.0f;

// Helper to convert IEEE 754 hex bits to float (runtime initialization)
inline float bits_to_float(uint32_t bits) {
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ========== src / scalar direction ==========

// Case 0: f32 32x64

extern "C" __global__ AICORE void TDIVS_f32_32x64(__gm__ float *src, __gm__ float *dst, float scalar);
extern "C" __global__ AICORE void TDIVS_f32_16x64_hp_overflow(__gm__ float *src, __gm__ float *dst, float scalar);
extern "C" __global__ AICORE void TDIVS_f16_16x64_hp_subnormal_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);

void LaunchTDIVS_f32_16x64_hp_overflow(float *src, float *dst, void *stream) {
    TDIVS_f32_16x64_hp_overflow<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, bits_to_float(0x0DA24260U));
}



void LaunchTDIVS_f32_32x64(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}



void LaunchTDIVS_f16_16x64_hp_subnormal_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_16x64_hp_subnormal_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x00A8);
}
