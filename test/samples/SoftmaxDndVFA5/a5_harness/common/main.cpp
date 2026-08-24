#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef SOFTMAX_LAUNCH_NAME
#error "SOFTMAX_LAUNCH_NAME must be defined"
#endif

#ifndef SOFTMAX_INNER
#error "SOFTMAX_INNER must be defined"
#endif

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d at %s:%d\n", #expr,       \
                   static_cast<int>(ret), __FILE__, __LINE__);                 \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

extern "C" void SOFTMAX_LAUNCH_NAME(
    float *x, float *y, int64_t tileCount, int32_t spmdBlockIdx,
    int32_t spmdBlockCount, void *stream);

int main() {
  constexpr size_t kBatch = 4;
  constexpr size_t kReduce = 16;
  constexpr size_t kInner = SOFTMAX_INNER;
  constexpr size_t kBytes = kBatch * kReduce * kInner * sizeof(float);

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  float *xHost = nullptr;
  float *yHost = nullptr;
  float *xDevice = nullptr;
  float *yDevice = nullptr;
  size_t xFileBytes = kBytes;
  size_t yFileBytes = kBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *value = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(value);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&xHost), kBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&yHost), kBytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&xDevice), kBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&yDevice), kBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./x.bin", xFileBytes, xHost, kBytes);
  ReadFile("./y.bin", yFileBytes, yHost, kBytes);
  ACL_CHECK(aclrtMemcpy(xDevice, kBytes, xHost, kBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(yDevice, kBytes, yHost, kBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));

  SOFTMAX_LAUNCH_NAME(xDevice, yDevice, kBatch, 0, 1, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(yHost, kBytes, yDevice, kBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./y.bin", yHost, kBytes);

cleanup:
  if (xDevice != nullptr)
    aclrtFree(xDevice);
  if (yDevice != nullptr)
    aclrtFree(yDevice);
  if (xHost != nullptr)
    aclrtFreeHost(xHost);
  if (yHost != nullptr)
    aclrtFreeHost(yHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
