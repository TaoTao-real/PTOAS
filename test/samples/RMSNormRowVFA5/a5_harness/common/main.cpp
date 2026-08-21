#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef RMSNORM_LAUNCH_NAME
#error "RMSNORM_LAUNCH_NAME must be defined"
#endif

#ifndef RMSNORM_ROWS
#error "RMSNORM_ROWS must be defined"
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

extern "C" void RMSNORM_LAUNCH_NAME(
    uint16_t *x, uint16_t *y, uint16_t *gamma, int64_t tokenCount,
    int32_t spmdBlockIdx, int32_t spmdBlockCount, void *stream);

int main() {
  constexpr size_t kRows = RMSNORM_ROWS;
  constexpr size_t kCols = 64;
  constexpr size_t kXBytes = kRows * kCols * sizeof(uint16_t);
  constexpr size_t kYBytes = kXBytes;
  constexpr size_t kGammaBytes = kCols * sizeof(uint16_t);

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  uint16_t *xHost = nullptr;
  uint16_t *yHost = nullptr;
  uint16_t *gammaHost = nullptr;
  uint16_t *xDevice = nullptr;
  uint16_t *yDevice = nullptr;
  uint16_t *gammaDevice = nullptr;
  size_t xFileBytes = kXBytes;
  size_t yFileBytes = kYBytes;
  size_t gammaFileBytes = kGammaBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *value = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(value);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&xHost), kXBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&yHost), kYBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&gammaHost), kGammaBytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&xDevice), kXBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&yDevice), kYBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gammaDevice), kGammaBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./x.bin", xFileBytes, xHost, kXBytes);
  ReadFile("./y.bin", yFileBytes, yHost, kYBytes);
  ReadFile("./gamma.bin", gammaFileBytes, gammaHost, kGammaBytes);
  ACL_CHECK(aclrtMemcpy(xDevice, kXBytes, xHost, kXBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(yDevice, kYBytes, yHost, kYBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(gammaDevice, kGammaBytes, gammaHost, kGammaBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));

  RMSNORM_LAUNCH_NAME(xDevice, yDevice, gammaDevice, kRows, 0, 1, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(yHost, kYBytes, yDevice, kYBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./y.bin", yHost, kYBytes);

cleanup:
  if (xDevice != nullptr)
    aclrtFree(xDevice);
  if (yDevice != nullptr)
    aclrtFree(yDevice);
  if (gammaDevice != nullptr)
    aclrtFree(gammaDevice);
  if (xHost != nullptr)
    aclrtFreeHost(xHost);
  if (yHost != nullptr)
    aclrtFreeHost(yHost);
  if (gammaHost != nullptr)
    aclrtFreeHost(gammaHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
