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

// Case: f32_single_1x256_b64 (transplanted from pto-isa case_single1)

extern "C" __global__ AICORE void TMRGSORT_f32_single_1x256_b64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TMRGSORT_f16_topk_1280_512(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f32_single_1x256_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x256_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}



void LaunchTMRGSORT_f16_topk_1280_512(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_topk_1280_512<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}
