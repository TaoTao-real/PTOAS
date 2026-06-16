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


// Case: f32_16x64

extern "C" __global__ AICORE void TDIV_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);
extern "C" __global__ AICORE void TDIV_f32_16x64_hp_subnormal(__gm__ float *a, __gm__ float *b, __gm__ float *c);
extern "C" __global__ AICORE void TDIV_f16_16x64_hp_partial(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f32_16x64_hp_subnormal(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64_hp_subnormal<<<1, nullptr, stream>>>(a, b, c);
}



void LaunchTDIV_f16_16x64_hp_partial(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x64_hp_partial<<<1, nullptr, stream>>>(a, b, c);
}



void LaunchTDIV_f32_16x64(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64<<<1, nullptr, stream>>>(a, b, c);
}
