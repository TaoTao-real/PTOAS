#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#ifndef SOFTMAX_KERNEL_NAME
#error "SOFTMAX_KERNEL_NAME must be defined"
#endif

#ifndef SOFTMAX_LAUNCH_NAME
#error "SOFTMAX_LAUNCH_NAME must be defined"
#endif

extern "C" __global__ AICORE void SOFTMAX_KERNEL_NAME(
    __gm__ float *x, __gm__ float *y, int64_t tileCount,
    int32_t spmdBlockIdx, int32_t spmdBlockCount);

extern "C" void SOFTMAX_LAUNCH_NAME(
    float *x, float *y, int64_t tileCount, int32_t spmdBlockIdx,
    int32_t spmdBlockCount, void *stream) {
  SOFTMAX_KERNEL_NAME<<<1, nullptr, stream>>>(
      (__gm__ float *)x, (__gm__ float *)y, tileCount, spmdBlockIdx,
      spmdBlockCount);
}
