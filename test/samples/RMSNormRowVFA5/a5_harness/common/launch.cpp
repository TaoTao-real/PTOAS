#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#ifndef RMSNORM_KERNEL_NAME
#error "RMSNORM_KERNEL_NAME must be defined"
#endif

#ifndef RMSNORM_LAUNCH_NAME
#error "RMSNORM_LAUNCH_NAME must be defined"
#endif

extern "C" __global__ AICORE void RMSNORM_KERNEL_NAME(
    __gm__ bfloat16_t *x, __gm__ bfloat16_t *y, __gm__ bfloat16_t *gamma,
    int64_t tokenCount, int32_t spmdBlockIdx, int32_t spmdBlockCount);

extern "C" void RMSNORM_LAUNCH_NAME(
    uint16_t *x, uint16_t *y, uint16_t *gamma, int64_t tokenCount,
    int32_t spmdBlockIdx, int32_t spmdBlockCount, void *stream) {
  RMSNORM_KERNEL_NAME<<<1, nullptr, stream>>>(
      (__gm__ bfloat16_t *)x, (__gm__ bfloat16_t *)y,
      (__gm__ bfloat16_t *)gamma, tokenCount, spmdBlockIdx, spmdBlockCount);
}
