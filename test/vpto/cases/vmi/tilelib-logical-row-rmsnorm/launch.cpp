// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include <cstdint>

extern "C" __global__ [aicore] void
vmi_tilelib_logical_row_rmsnorm_kernel(__gm__ half *src, __gm__ half *dst);

void LaunchVmiTilelibLogicalRowRmsnorm(uint16_t *src, uint16_t *dst,
                                       void *stream) {
  vmi_tilelib_logical_row_rmsnorm_kernel<<<1, nullptr, stream>>>(
      (__gm__ half *)src, (__gm__ half *)dst);
}
