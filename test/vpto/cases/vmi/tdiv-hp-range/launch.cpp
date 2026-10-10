// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use it in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void
tdiv_hp_f32_16x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

extern "C" __global__ [aicore] void
tdiv_default_f32_16x32(__gm__ float *a, __gm__ float *b, __gm__ float *d);

void LaunchTdiv_hp_f32_16x32(float *a, float *b, float *c, void *stream) {
  tdiv_hp_f32_16x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b,
                                            (__gm__ float *)c);
}

void LaunchTdiv_default_f32_16x32(float *a, float *b, float *d, void *stream) {
  tdiv_default_f32_16x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b,
                                                 (__gm__ float *)d);
}
