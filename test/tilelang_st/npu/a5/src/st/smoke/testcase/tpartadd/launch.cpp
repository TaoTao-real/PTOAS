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

// Case 0: f32 64x64 full

extern "C" __global__ AICORE void TPARTADD_f16_8x48_src0_col_less(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);
extern "C" __global__ AICORE void TPARTADD_i16_8x48_src1_col_less(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTPARTADD_f16_8x48_src0_col_less(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TPARTADD_f16_8x48_src0_col_less<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}



void LaunchTPARTADD_i16_8x48_src1_col_less(int16_t *a, int16_t *b, int16_t *c, void *stream) {
    TPARTADD_i16_8x48_src1_col_less<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}
