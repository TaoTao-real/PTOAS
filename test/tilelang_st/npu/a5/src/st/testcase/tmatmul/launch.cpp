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

// ---- case1: f16 x f16 -> f32, 40x50x60 ----
extern "C" __global__ AICORE void TMATMUL_f16_40x50x60(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ float *c);
void LaunchTMATMUL_f16_40x50x60(void *a, void *b, void *c, void *stream) {
    TMATMUL_f16_40x50x60<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ float *)c);
}

// ---- case2: i8 x i8 -> i32, 6x7x8 ----
extern "C" __global__ AICORE void TMATMUL_i8_6x7x8(__gm__ int8_t *a, __gm__ int8_t *b, __gm__ int32_t *c);
void LaunchTMATMUL_i8_6x7x8(void *a, void *b, void *c, void *stream) {
    TMATMUL_i8_6x7x8<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b, (__gm__ int32_t *)c);
}

// ---- case3: f16 x f16 -> f32, 127x128x61 ----
extern "C" __global__ AICORE void TMATMUL_f16_127x128x61(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ float *c);
void LaunchTMATMUL_f16_127x128x61(void *a, void *b, void *c, void *stream) {
    TMATMUL_f16_127x128x61<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ float *)c);
}

// ---- case4: f32 x f32 -> f32, 120x110x50 ----
extern "C" __global__ AICORE void TMATMUL_f32_120x110x50(__gm__ float *a, __gm__ float *b, __gm__ float *c);
void LaunchTMATMUL_f32_120x110x50(void *a, void *b, void *c, void *stream) {
    TMATMUL_f32_120x110x50<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// ---- case5: bf16 x bf16 -> f32, 144x80x48 ----
extern "C" __global__ AICORE void TMATMUL_bf16_144x80x48(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ float *c);
void LaunchTMATMUL_bf16_144x80x48(void *a, void *b, void *c, void *stream) {
    TMATMUL_bf16_144x80x48<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ float *)c);
}

// // ---- case6: f8e4m3 x f8e4m3 -> f32, 32x64x96 ----
// extern "C" __global__ AICORE void TMATMUL_f8e4m3_32x64x96(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ float *c);
// void LaunchTMATMUL_f8e4m3_32x64x96(void *a, void *b, void *c, void *stream) {
//     TMATMUL_f8e4m3_32x64x96<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ float *)c);
// }
//
// // ---- case7: f8e4m3 x f8e5m2 -> f32, 128x96x64 ----
// extern "C" __global__ AICORE void TMATMUL_f8e4m3_f8e5m2_128x96x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ float *c);
// void LaunchTMATMUL_f8e4m3_f8e5m2_128x96x64(void *a, void *b, void *c, void *stream) {
//     TMATMUL_f8e4m3_f8e5m2_128x96x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ float *)c);
// }
//
// // ---- case8: f8e5m2 x f8e4m3 -> f32, 145x115x85 ----
// extern "C" __global__ AICORE void TMATMUL_f8e5m2_f8e4m3_145x115x85(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ float *c);
// void LaunchTMATMUL_f8e5m2_f8e4m3_145x115x85(void *a, void *b, void *c, void *stream) {
//     TMATMUL_f8e5m2_f8e4m3_145x115x85<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ float *)c);
// }
//
// // ---- case9: f8e5m2 x f8e5m2 -> f32, 120x90x160 ----
// extern "C" __global__ AICORE void TMATMUL_f8e5m2_120x90x160(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ float *c);
// void LaunchTMATMUL_f8e5m2_120x90x160(void *a, void *b, void *c, void *stream) {
//     TMATMUL_f8e5m2_120x90x160<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ float *)c);
// }
//
// // ---- case10: hif8 x hif8 -> f32, 30x90x60 ----
// extern "C" __global__ AICORE void TMATMUL_hif8_30x90x60(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ float *c);
// void LaunchTMATMUL_hif8_30x90x60(void *a, void *b, void *c, void *stream) {
//     TMATMUL_hif8_30x90x60<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ float *)c);
// }

// ---- case12: f32 x f32 -> f32, 16x32x64 ----
extern "C" __global__ AICORE void TMATMUL_f32_16x32x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);
void LaunchTMATMUL_f32_16x32x64(void *a, void *b, void *c, void *stream) {
    TMATMUL_f32_16x32x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// ---- case13: f32 x f32 -> f32, 128x96x64 ----
extern "C" __global__ AICORE void TMATMUL_f32_128x96x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);
void LaunchTMATMUL_f32_128x96x64(void *a, void *b, void *c, void *stream) {
    TMATMUL_f32_128x96x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
