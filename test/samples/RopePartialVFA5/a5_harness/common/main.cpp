#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef ROPE_LAUNCH_NAME
#error "ROPE_LAUNCH_NAME must be defined"
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

extern "C" void ROPE_LAUNCH_NAME(float *input, float *sin, float *cos,
                                  uint16_t *output, void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kActualCol = 512;
  constexpr size_t kRotaryCol = 64;
  constexpr size_t kInputBytes =
      kRows * kActualCol * sizeof(float);
  constexpr size_t kTrigBytes = kRows * kRotaryCol * sizeof(float);
  constexpr size_t kOutputBytes =
      kRows * kActualCol * sizeof(uint16_t);

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  float *inputHost = nullptr;
  float *sinHost = nullptr;
  float *cosHost = nullptr;
  uint16_t *outputHost = nullptr;
  float *inputDevice = nullptr;
  float *sinDevice = nullptr;
  float *cosDevice = nullptr;
  uint16_t *outputDevice = nullptr;
  size_t inputFileBytes = kInputBytes;
  size_t sinFileBytes = kTrigBytes;
  size_t cosFileBytes = kTrigBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *value = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(value);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&inputHost),
                           kInputBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&sinHost), kTrigBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&cosHost), kTrigBytes));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                           kOutputBytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kInputBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&sinDevice), kTrigBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&cosDevice), kTrigBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kOutputBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input_f32.bin", inputFileBytes, inputHost, kInputBytes);
  ReadFile("./sin_f32.bin", sinFileBytes, sinHost, kTrigBytes);
  ReadFile("./cos_f32.bin", cosFileBytes, cosHost, kTrigBytes);
  ACL_CHECK(aclrtMemcpy(inputDevice, kInputBytes, inputHost, kInputBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(sinDevice, kTrigBytes, sinHost, kTrigBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(cosDevice, kTrigBytes, cosHost, kTrigBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(outputDevice, kOutputBytes, 0, kOutputBytes));

  ROPE_LAUNCH_NAME(inputDevice, sinDevice, cosDevice, outputDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outputHost, kOutputBytes, outputDevice, kOutputBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output_bf16.bin", outputHost, kOutputBytes);

cleanup:
  if (inputDevice != nullptr)
    aclrtFree(inputDevice);
  if (sinDevice != nullptr)
    aclrtFree(sinDevice);
  if (cosDevice != nullptr)
    aclrtFree(cosDevice);
  if (outputDevice != nullptr)
    aclrtFree(outputDevice);
  if (inputHost != nullptr)
    aclrtFreeHost(inputHost);
  if (sinHost != nullptr)
    aclrtFreeHost(sinHost);
  if (cosHost != nullptr)
    aclrtFreeHost(cosHost);
  if (outputHost != nullptr)
    aclrtFreeHost(outputHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
