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

// f16_16x16x16: working baseline copied from PTOAS_matmul0_copy
extern "C" __global__ AICORE void TMATMUL_BIAS_f16_16x16x16(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ float *bias, __gm__ float *c);
void LaunchTMATMUL_BIAS_f16_16x16x16(void *a, void *b, void *bias, void *c, void *stream) {
    TMATMUL_BIAS_f16_16x16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ float *)bias, (__gm__ float *)c);
}

// ---- case_bias_1: i8 x i8 -> i32, bias i32, 8x7x6 ----
extern "C" __global__ AICORE void TMATMUL_BIAS_i8_bias_i32_8x7x6(__gm__ int8_t *a, __gm__ int8_t *b, __gm__ int32_t *bias, __gm__ int32_t *c);
void LaunchTMATMUL_BIAS_i8_bias_i32_8x7x6(void *a, void *b, void *bias, void *c, void *stream) {
    TMATMUL_BIAS_i8_bias_i32_8x7x6<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b, (__gm__ int32_t *)bias, (__gm__ int32_t *)c);
}

// ---- case_bias_2: f16 x f16 -> f32, bias f32, 16x15x16 (DEBUG: f32 bias test) ----
extern "C" __global__ AICORE void TMATMUL_BIAS_f16_bias_f16_16x15x16(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ float *bias, __gm__ float *c);
void LaunchTMATMUL_BIAS_f16_bias_f16_16x15x16(void *a, void *b, void *bias, void *c, void *stream) {
    TMATMUL_BIAS_f16_bias_f16_16x15x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ float *)bias, (__gm__ float *)c);
}

// ---- case_bias_3: f16 x f16 -> f32, bias bf16, 112x127x80 ----
extern "C" __global__ AICORE void TMATMUL_BIAS_f16_bias_bf16_112x127x80(__gm__ uint16_t *a1, __gm__ uint16_t *b1, __gm__ uint16_t *a2, __gm__ uint16_t *b2, __gm__ float *bias, __gm__ float *c);
void LaunchTMATMUL_BIAS_f16_bias_bf16_112x127x80(void *a, void *b, void *bias, void *c, void *stream) {
    uint16_t *a_ = (uint16_t *)a;
    uint16_t *b_ = (uint16_t *)b;
    TMATMUL_BIAS_f16_bias_bf16_112x127x80<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)(a_),          // A[:,0:64]  (BASEK=64)
        (__gm__ uint16_t *)(b_),          // B[0:64,:]
        (__gm__ uint16_t *)(a_ + 64),     // A[:,64:128]
        (__gm__ uint16_t *)(b_ + 64 * 80),// B[64:128,:]
        (__gm__ float *)bias,
        (__gm__ float *)c
    );
}

// ---- case_bias_4: bf16 x bf16 -> f32, bias bf16, 80x112x63 ----
extern "C" __global__ AICORE void TMATMUL_BIAS_bf16_bias_bf16_80x112x63(__gm__ uint16_t *a1, __gm__ uint16_t *b1, __gm__ uint16_t *a2, __gm__ uint16_t *b2, __gm__ float *bias, __gm__ float *c);
void LaunchTMATMUL_BIAS_bf16_bias_bf16_80x112x63(void *a, void *b, void *bias, void *c, void *stream) {
    uint16_t *a_ = (uint16_t *)a;
    uint16_t *b_ = (uint16_t *)b;
    TMATMUL_BIAS_bf16_bias_bf16_80x112x63<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)(a_),          // A[:,0:64]  (BASEK=64)
        (__gm__ uint16_t *)(b_),          // B[0:64,:]
        (__gm__ uint16_t *)(a_ + 64),     // A[:,64:128]
        (__gm__ uint16_t *)(b_ + 64 * 64),// B[64:128,:]
        (__gm__ float *)bias,
        (__gm__ float *)c
    );
}

// ---- case_bias_5: f32 x f32 -> f32, bias f32, 127x128x63 (Split-K) ----
extern "C" __global__ AICORE void TMATMUL_BIAS_f32_bias_f32_127x128x63(__gm__ float *a, __gm__ float *b, __gm__ float *bias, __gm__ float *c);
void LaunchTMATMUL_BIAS_f32_bias_f32_127x128x63(void *a, void *b, void *bias, void *c, void *stream) {
    TMATMUL_BIAS_f32_bias_f32_127x128x63<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)bias, (__gm__ float *)c);
}



