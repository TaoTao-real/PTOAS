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

extern "C" __global__ AICORE void TMATMUL_ACC_f16_16x32x16(__gm__ uint16_t *a1, __gm__ uint16_t *b1, __gm__ uint16_t *a2, __gm__ uint16_t *b2, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_ACC_f16_128x128x64(__gm__ uint16_t *a1, __gm__ uint16_t *b1, __gm__ uint16_t *a2, __gm__ uint16_t *b2, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_ACC_f16_127x128x61(__gm__ uint16_t *a1, __gm__ uint16_t *b1, __gm__ uint16_t *a2, __gm__ uint16_t *b2, __gm__ float *c);

void LaunchTMATMUL_ACC_f16_16x32x16(void *a, void *b, void *c, void *stream) {
    uint16_t *a_ = (uint16_t *)a;
    uint16_t *b_ = (uint16_t *)b;
    TMATMUL_ACC_f16_16x32x16<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)(a_),          // A[:,0:16]  (BASEK=16)
        (__gm__ uint16_t *)(b_),          // B[0:16,:]
        (__gm__ uint16_t *)(a_ + 16),     // A[:,16:32]
        (__gm__ uint16_t *)(b_ + 16 * 16),// B[16:32,:]
        (__gm__ float *)c
    );
}

void LaunchTMATMUL_ACC_f16_128x128x64(void *a, void *b, void *c, void *stream) {
    uint16_t *a_ = (uint16_t *)a;
    uint16_t *b_ = (uint16_t *)b;
    TMATMUL_ACC_f16_128x128x64<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)(a_),          // A[:,0:64]  (BASEK=64)
        (__gm__ uint16_t *)(b_),          // B[0:64,:]
        (__gm__ uint16_t *)(a_ + 64),     // A[:,64:128]
        (__gm__ uint16_t *)(b_ + 64 * 64),// B[64:128,:]
        (__gm__ float *)c
    );
}

void LaunchTMATMUL_ACC_f16_127x128x61(void *a, void *b, void *c, void *stream) {
    uint16_t *a_ = (uint16_t *)a;
    uint16_t *b_ = (uint16_t *)b;
    TMATMUL_ACC_f16_127x128x61<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)(a_),          // A[:,0:64]  (BASEK=64)
        (__gm__ uint16_t *)(b_),          // B[0:64,:]
        (__gm__ uint16_t *)(a_ + 64),     // A[:,64:128]
        (__gm__ uint16_t *)(b_ + 64 * 64),// B[64:128,:]
        (__gm__ float *)c
    );
}
