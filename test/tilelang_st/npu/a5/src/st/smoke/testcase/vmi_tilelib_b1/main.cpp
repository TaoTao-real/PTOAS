// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void LaunchVMITileLibB1(float *a, float *b, float *out, void *stream);

int main() {
  constexpr size_t kElements = 4 * 64;
  constexpr size_t kBytes = kElements * sizeof(float);
  int deviceId = 0;
  if (const char *value = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(value);

  std::vector<float> a(kElements), b(kElements), golden(kElements), out(kElements);
  for (size_t i = 0; i < kElements; ++i) {
    a[i] = 2.0f + static_cast<float>(i % 17) * 0.125f;
    b[i] = 1.0f + static_cast<float>(i % 11) * 0.0625f;
    const float divided = a[i] / b[i];
    const float minimum = std::min(divided, a[i]);
    const float negated = -std::abs(minimum - 0.5f);
    golden[i] = 3.0f / negated + 3.0f;
  }

  if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS)
    return 1;

  aclrtStream stream = nullptr;
  float *aDevice = nullptr;
  float *bDevice = nullptr;
  float *outDevice = nullptr;
  aclrtCreateStream(&stream);
  aclrtMalloc((void **)&aDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&bDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&outDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(aDevice, kBytes, a.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(bDevice, kBytes, b.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE);

  LaunchVMITileLibB1(aDevice, bDevice, outDevice, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(out.data(), kBytes, outDevice, kBytes, ACL_MEMCPY_DEVICE_TO_HOST);

  float maxError = 0.0f;
  size_t mismatch = kElements;
  for (size_t i = 0; i < kElements; ++i) {
    const float error = std::abs(out[i] - golden[i]);
    maxError = std::max(maxError, error);
    if (error > 2.0e-5f * (1.0f + std::abs(golden[i]))) {
      mismatch = i;
      break;
    }
  }

  aclrtFree(aDevice);
  aclrtFree(bDevice);
  aclrtFree(outDevice);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  if (mismatch != kElements) {
    std::fprintf(stderr,
                 "[ERROR] mismatch at %zu: actual=%g expected=%g max_error=%g\n",
                 mismatch, out[mismatch], golden[mismatch], maxError);
    return 2;
  }
  std::printf("[PASS] VMI TileLib B1 A5 correctness, max_error=%g\n", maxError);
  return 0;
}
