#include "kernel_operator.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"

// The CANN 9.2 A5 headers expose this API as AscendC::Reg unless
// __NPU_ARCH__ is injected by the product build.  The attached source uses
// the compatibility name AscendC::MicroAPI.
namespace AscendC {
namespace MicroAPI = Reg;
}

#include "vf_rope_attached.h"

#include <cstdint>

using namespace AscendC;

#ifndef AICORE
#define AICORE __aicore__
#endif

#ifndef ROPE_KERNEL_NAME
#define ROPE_KERNEL_NAME rope_partial_manual_vf
#endif

namespace {

constexpr uint32_t kRows = 8;
constexpr uint32_t kActualCol = 512;
constexpr uint32_t kRotaryCol = 64;
constexpr uint64_t kBaseAddr = 448;

} // namespace

extern "C" __global__ AICORE void ROPE_KERNEL_NAME(
    __gm__ float *input, __gm__ float *sin, __gm__ float *cos,
    __gm__ bfloat16_t *output) {
  GlobalTensor<float> inputGm;
  GlobalTensor<float> sinGm;
  GlobalTensor<float> cosGm;
  GlobalTensor<bfloat16_t> outputGm;
  inputGm.SetGlobalBuffer(input, kRows * kActualCol);
  sinGm.SetGlobalBuffer(sin, kRows * kRotaryCol);
  cosGm.SetGlobalBuffer(cos, kRows * kRotaryCol);
  outputGm.SetGlobalBuffer(output, kRows * kActualCol);

  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBuffer;
  TBuf<TPosition::VECCALC> sinBuffer;
  TBuf<TPosition::VECCALC> cosBuffer;
  TBuf<TPosition::VECCALC> outputBuffer;
  pipe.InitBuffer(inputBuffer, kRows * kActualCol * sizeof(float));
  pipe.InitBuffer(sinBuffer, kRows * kRotaryCol * sizeof(float));
  pipe.InitBuffer(cosBuffer, kRows * kRotaryCol * sizeof(float));
  pipe.InitBuffer(outputBuffer, kRows * kActualCol * sizeof(bfloat16_t));

  LocalTensor<float> inputLocal = inputBuffer.Get<float>();
  LocalTensor<float> sinLocal = sinBuffer.Get<float>();
  LocalTensor<float> cosLocal = cosBuffer.Get<float>();
  LocalTensor<bfloat16_t> outputLocal = outputBuffer.Get<bfloat16_t>();

  DataCopy(inputLocal, inputGm, kRows * kActualCol);
  DataCopy(sinLocal, sinGm, kRows * kRotaryCol);
  DataCopy(cosLocal, cosGm, kRows * kRotaryCol);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

#if defined(ROPE_MODE_INTERLEAVE)
  RopeVF<Compressor::ROTARY_MODE::INTERLEAVE, float, bfloat16_t>(
      sinLocal, cosLocal, inputLocal, outputLocal, kRows, kRotaryCol,
      kActualCol, kBaseAddr);
#else
  RopeVF<Compressor::ROTARY_MODE::HALF, float, bfloat16_t>(
      sinLocal, cosLocal, inputLocal, outputLocal, kRows, kRotaryCol,
      kActualCol, kBaseAddr);
#endif

  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(outputGm, outputLocal, kRows * kActualCol);
}
