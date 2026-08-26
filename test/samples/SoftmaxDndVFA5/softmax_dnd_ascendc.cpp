#include "kernel_operator.h"
#if defined(SOFTMAX_USE_VF)
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
// The installed compressor header still spells the register API as the
// historical global MicroAPI namespace, while CANN 9.2 exposes it as
// AscendC::Reg for dav-c310-vec compilation.
namespace MicroAPI = AscendC::Reg;
#include "vf_softmax.h"
#endif

#include <cstdint>

using namespace AscendC;

#ifndef AICORE
#define AICORE __aicore__
#endif

#ifndef SOFTMAX_INNER
#define SOFTMAX_INNER 64
#endif

#ifndef SOFTMAX_KERNEL_NAME
#define SOFTMAX_KERNEL_NAME softmax_dnd_ascendc
#endif

namespace {

constexpr uint32_t kBatch = 4;
constexpr uint32_t kReduce = 16;
constexpr uint32_t kInner = SOFTMAX_INNER;
constexpr uint32_t kLanes = 64;
constexpr uint32_t kElements = kBatch * kReduce * kInner;
constexpr float kMinValue = -3.402823466e+38F;
static_assert(kInner == 32 || kInner == 64 || kInner == 128,
              "Softmax Dn fixture supports Base32/Base64/Base128");
static_assert(kReduce % 2 == 0, "paired reduction requires even M");

__aicore__ inline void SoftmaxDnOrdinary(
    const LocalTensor<float> &output, const LocalTensor<float> &data,
    const LocalTensor<float> &maxEven, const LocalTensor<float> &maxOdd,
    const LocalTensor<float> &sumEven, const LocalTensor<float> &sumOdd,
    const LocalTensor<float> &rowTemp) {
  constexpr uint32_t kChunkCount = (kInner + kLanes - 1) / kLanes;
  for (uint32_t tile = 0; tile < kBatch; ++tile) {
    const uint32_t tileBase = tile * kReduce * kInner;
    for (uint32_t chunk = 0; chunk < kChunkCount; ++chunk) {
      const uint32_t col = chunk * kLanes;
      const uint32_t active =
          (kInner - col < kLanes) ? (kInner - col) : kLanes;

      Duplicate(maxEven, kMinValue, active);
      PipeBarrier<PIPE_V>();
      Duplicate(maxOdd, kMinValue, active);
      PipeBarrier<PIPE_V>();
      for (uint32_t pair = 0; pair < kReduce / 2; ++pair) {
        const uint32_t evenOffset =
            tileBase + (pair * 2) * kInner + col;
        const uint32_t oddOffset = evenOffset + kInner;
        Max(maxEven, maxEven, data[evenOffset], active);
        PipeBarrier<PIPE_V>();
        Max(maxOdd, maxOdd, data[oddOffset], active);
        PipeBarrier<PIPE_V>();
      }
      Max(maxEven, maxEven, maxOdd, active);
      PipeBarrier<PIPE_V>();

      Duplicate(sumEven, 0.0F, active);
      PipeBarrier<PIPE_V>();
      Duplicate(sumOdd, 0.0F, active);
      PipeBarrier<PIPE_V>();
      for (uint32_t pair = 0; pair < kReduce / 2; ++pair) {
        const uint32_t evenOffset =
            tileBase + (pair * 2) * kInner + col;
        const uint32_t oddOffset = evenOffset + kInner;

        Sub(rowTemp, data[evenOffset], maxEven, active);
        PipeBarrier<PIPE_V>();
        Exp(rowTemp, rowTemp, active);
        PipeBarrier<PIPE_V>();
        Add(sumEven, sumEven, rowTemp, active);
        PipeBarrier<PIPE_V>();
        Adds(data[evenOffset], rowTemp, 0.0F, active);
        PipeBarrier<PIPE_V>();

        Sub(rowTemp, data[oddOffset], maxEven, active);
        PipeBarrier<PIPE_V>();
        Exp(rowTemp, rowTemp, active);
        PipeBarrier<PIPE_V>();
        Add(sumOdd, sumOdd, rowTemp, active);
        PipeBarrier<PIPE_V>();
        Adds(data[oddOffset], rowTemp, 0.0F, active);
        PipeBarrier<PIPE_V>();
      }
      Add(sumEven, sumEven, sumOdd, active);

      // The exp values above overwrite the source UB tile, exactly like the
      // installed manual VF implementation.  The next phase reloads them.
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < kReduce; ++row) {
        const uint32_t offset = tileBase + row * kInner + col;
        Div(output[offset], data[offset], sumEven, active);
        PipeBarrier<PIPE_V>();
      }
    }
  }
}

}  // namespace

extern "C" __global__ AICORE void SOFTMAX_KERNEL_NAME(
    __gm__ float *x, __gm__ float *y, int64_t tileCount,
    int32_t spmdBlockIdx, int32_t spmdBlockCount) {
  if (spmdBlockIdx != 0 || spmdBlockCount != 1 ||
      tileCount != static_cast<int64_t>(kBatch)) {
    return;
  }

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  xGm.SetGlobalBuffer(x, kElements);
  yGm.SetGlobalBuffer(y, kElements);

  TPipe pipe;
  TBuf<TPosition::VECCALC> dataBuffer;
  TBuf<TPosition::VECCALC> outputBuffer;
  TBuf<TPosition::VECCALC> maxEvenBuffer;
  TBuf<TPosition::VECCALC> maxOddBuffer;
  TBuf<TPosition::VECCALC> sumEvenBuffer;
  TBuf<TPosition::VECCALC> sumOddBuffer;
  TBuf<TPosition::VECCALC> rowTempBuffer;
  pipe.InitBuffer(dataBuffer, kElements * sizeof(float));
  pipe.InitBuffer(outputBuffer, kElements * sizeof(float));
  pipe.InitBuffer(maxEvenBuffer, kLanes * sizeof(float));
  pipe.InitBuffer(maxOddBuffer, kLanes * sizeof(float));
  pipe.InitBuffer(sumEvenBuffer, kLanes * sizeof(float));
  pipe.InitBuffer(sumOddBuffer, kLanes * sizeof(float));
  pipe.InitBuffer(rowTempBuffer, kLanes * sizeof(float));

  LocalTensor<float> data = dataBuffer.Get<float>();
  LocalTensor<float> output = outputBuffer.Get<float>();
  LocalTensor<float> maxEven = maxEvenBuffer.Get<float>();
  LocalTensor<float> maxOdd = maxOddBuffer.Get<float>();
  LocalTensor<float> sumEven = sumEvenBuffer.Get<float>();
  LocalTensor<float> sumOdd = sumOddBuffer.Get<float>();
  LocalTensor<float> rowTemp = rowTempBuffer.Get<float>();

  DataCopy(data, xGm, kElements);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

#if defined(SOFTMAX_USE_VF)
#if defined(SOFTMAX_ATTACHED_VF_CALL)
  // The frozen 2026 attachment maps dstTensor to the Base write pointer and
  // srcTensor to the Base exp scratch/read pointer. Honor that implementation
  // without editing the attached header.
  FaVectorApi::SoftmaxDnVF<float>(output, data, kInner, kReduce, kBatch,
                                  kMinValue, kInner);
#else
  // CANN 9.2's installed wrapper names these parameters dst/src, but passes
  // dstTensor as the Base implementation's exp scratch/source and srcTensor
  // as its normalized destination.  Preserve that authoritative call
  // convention explicitly: data is read and overwritten with exp, output
  // receives the normalized result.
  FaVectorApi::SoftmaxDnVF<float>(data, output, kInner, kReduce, kBatch,
                                  kMinValue, kInner);
#endif
#else
  SoftmaxDnOrdinary(output, data, maxEven, maxOdd, sumEven, sumOdd,
                    rowTemp);
#endif

  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(yGm, output, kElements);
}
