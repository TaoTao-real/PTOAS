// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void VMI_TILELIB_B1_f32_4x64(
    __gm__ float *a, __gm__ float *b, __gm__ float *out);

void LaunchVMITileLibB1(float *a, float *b, float *out, void *stream) {
  VMI_TILELIB_B1_f32_4x64<<<1, nullptr, stream>>>(
      (__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)out);
}
