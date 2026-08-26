#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#ifndef ROPE_KERNEL_NAME
#error "ROPE_KERNEL_NAME must be defined"
#endif

#ifndef ROPE_LAUNCH_NAME
#error "ROPE_LAUNCH_NAME must be defined"
#endif

extern "C" __global__ AICORE void ROPE_KERNEL_NAME(
    __gm__ float *input, __gm__ float *sin, __gm__ float *cos,
    __gm__ bfloat16_t *output);

extern "C" void ROPE_LAUNCH_NAME(float *input, float *sin, float *cos,
                                  uint16_t *output, void *stream) {
  ROPE_KERNEL_NAME<<<1, nullptr, stream>>>(
      (__gm__ float *)input, (__gm__ float *)sin, (__gm__ float *)cos,
      (__gm__ bfloat16_t *)output);
}
