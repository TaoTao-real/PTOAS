// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use it in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "acl/acl.h"
#include "test_common.h"
#include <cstdio>
#include <cstdlib>
using namespace PtoTestCommon;
#define ACL_CHECK(expr)                                                                                                \
  do {                                                                                                                 \
    const aclError _ret = (expr);                                                                                      \
    if (_ret != ACL_SUCCESS) {                                                                                         \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__);                   \
      rc = 1;                                                                                                          \
      goto cleanup;                                                                                                    \
    }                                                                                                                  \
  } while (0)
void LaunchTdiv_hp_f32_16x32(float *a, float *b, float *c, void *stream);
void LaunchTdiv_default_f32_16x32(float *a, float *b, float *d, void *stream);
int main() {
  const size_t kElems = 16 * 32;
  size_t kBytes = kElems * sizeof(float);
  float *aHost = nullptr, *aDevice = nullptr;
  float *bHost = nullptr, *bDevice = nullptr;
  float *cHost = nullptr, *cDevice = nullptr;
  float *dHost = nullptr, *dDevice = nullptr;
  int rc = 0; bool aclInited = false, deviceSet = false; int deviceId = 0; aclrtStream stream = nullptr;
  ACL_CHECK(aclInit(nullptr)); aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) { deviceId = std::atoi(envDevice); }
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet = true; ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&aHost), kBytes)); ACL_CHECK(aclrtMallocHost((void **)(&bHost), kBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&cHost), kBytes)); ACL_CHECK(aclrtMallocHost((void **)(&dHost), kBytes));
  ACL_CHECK(aclrtMalloc((void **)&aDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&bDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&cDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", kBytes, aHost, kBytes);
  ReadFile("./v2.bin", kBytes, bHost, kBytes);
  ACL_CHECK(aclrtMemcpy(aDevice, kBytes, aHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, kBytes, bHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(cDevice, kBytes, 0, kBytes));
  ACL_CHECK(aclrtMemset(dDevice, kBytes, 0, kBytes));
  LaunchTdiv_hp_f32_16x32(aDevice, bDevice, cDevice, stream);
  LaunchTdiv_default_f32_16x32(aDevice, bDevice, dDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(cHost, kBytes, cDevice, kBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", cHost, kBytes);
  ACL_CHECK(aclrtMemcpy(dHost, kBytes, dDevice, kBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v4.bin", dHost, kBytes);
cleanup:
  aclrtFree(aDevice); aclrtFree(bDevice); aclrtFree(cDevice); aclrtFree(dDevice);
  aclrtFreeHost(aHost); aclrtFreeHost(bHost); aclrtFreeHost(cHost); aclrtFreeHost(dHost);
  if (stream) { aclrtDestroyStream(stream); }
  if (deviceSet) { aclrtResetDevice(deviceId); }
  if (aclInited) { aclFinalize(); }
  return rc;
}
