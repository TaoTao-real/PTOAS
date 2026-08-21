#include "kernel_operator.h"
#if defined(RMSNORM_USE_VF)
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#endif

#include <cstdint>

using namespace AscendC;

#ifndef AICORE
#define AICORE __aicore__
#endif

#ifndef RMSNORM_ROWS
#define RMSNORM_ROWS 64
#endif

#ifndef RMSNORM_KERNEL_NAME
#define RMSNORM_KERNEL_NAME rmsnorm_row_vf_ascendc
#endif

namespace {

constexpr uint32_t kRows = RMSNORM_ROWS;
constexpr uint32_t kCols = 64;
constexpr float kReciprocal = 1.0f / static_cast<float>(kCols);
constexpr float kEpsilon = 1.0e-6f;

__aicore__ inline void RmsNormOrdinaryRows(
    const LocalTensor<bfloat16_t> &output,
    const LocalTensor<bfloat16_t> &input,
    const LocalTensor<bfloat16_t> &gammaBf16,
    const LocalTensor<float> &gammaFp32,
    const LocalTensor<float> &xFp32,
    const LocalTensor<float> &square,
    const LocalTensor<float> &rowState,
    const LocalTensor<float> &divisor) {
  Cast(gammaFp32, gammaBf16, RoundMode::CAST_NONE, kCols);
  PipeBarrier<PIPE_V>();
  for (uint32_t row = 0; row < kRows; ++row) {
    const uint32_t offset = row * kCols;
    Cast(xFp32, input[offset], RoundMode::CAST_NONE, kCols);
    PipeBarrier<PIPE_V>();
    Mul(square, xFp32, xFp32, kCols);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(rowState, square, kCols, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    Muls(rowState, rowState, kReciprocal, 1);
    PipeBarrier<PIPE_V>();
    Adds(rowState, rowState, kEpsilon, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(rowState, rowState, 1);
    PipeBarrier<PIPE_V>();
    BrcbRepeatParams brcbParams = {1, 1};
    Brcb(divisor, rowState, 1, brcbParams);
    PipeBarrier<PIPE_V>();
    uint64_t mask[2] = {UINT64_MAX, 0};
    BinaryRepeatParams divParams = {1, 1, 0, 8, 8, 0};
    Div(xFp32, xFp32, divisor, mask, 1, divParams);
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, gammaFp32, kCols);
    PipeBarrier<PIPE_V>();
    Cast(output[offset], xFp32, RoundMode::CAST_RINT, kCols);
    PipeBarrier<PIPE_V>();
  }
}

#if defined(RMSNORM_USE_VF)
constexpr Reg::CastTrait kBf16ToFp32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait kFp32ToBf16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_vf__ void RmsNormFusedRowsImpl(
    __ubuf__ bfloat16_t *input,
    __ubuf__ bfloat16_t *gamma,
    __ubuf__ bfloat16_t *output) {
  Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
  Reg::MaskReg first = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
  Reg::RegTensor<bfloat16_t> gammaBf16;
  Reg::RegTensor<float> gammaFp32;
  Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
      gammaBf16, gamma);
  Reg::Cast<float, bfloat16_t, kBf16ToFp32>(gammaFp32, gammaBf16, all);

  for (uint16_t row = 0; row < uint16_t(kRows); ++row) {
    Reg::RegTensor<bfloat16_t> xBf16;
    Reg::RegTensor<float> xFp32;
    Reg::RegTensor<float> square;
    Reg::RegTensor<float> rowState;
    Reg::RegTensor<float> divisor;
    Reg::RegTensor<bfloat16_t> outputBf16;
    const uint32_t offset = row * kCols;
    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
        xBf16, input + offset);
    Reg::Cast<float, bfloat16_t, kBf16ToFp32>(xFp32, xBf16, all);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(
        square, xFp32, xFp32, all);
    Reg::Reduce<Reg::ReduceType::SUM, float, float,
                Reg::MaskMergeMode::ZEROING>(rowState, square, all);
    Reg::Muls<float, float, Reg::MaskMergeMode::ZEROING>(
        rowState, rowState, kReciprocal, first);
    Reg::Adds<float, float, Reg::MaskMergeMode::ZEROING>(
        rowState, rowState, kEpsilon, first);
    Reg::Sqrt<float, Reg::MaskMergeMode::ZEROING>(
        rowState, rowState, first);
    Reg::Duplicate<float, Reg::HighLowPart::LOWEST,
                   Reg::MaskMergeMode::ZEROING>(divisor, rowState, all);
    Reg::Div<float, Reg::MaskMergeMode::ZEROING>(
        xFp32, xFp32, divisor, all);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(
        xFp32, xFp32, gammaFp32, all);
    Reg::Cast<bfloat16_t, float, kFp32ToBf16>(
        outputBf16, xFp32, all);
    Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
        output + offset, outputBf16, all);
  }
}
#endif

}  // namespace

extern "C" __global__ AICORE void RMSNORM_KERNEL_NAME(
    __gm__ bfloat16_t *x,
    __gm__ bfloat16_t *y,
    __gm__ bfloat16_t *gamma,
    int64_t tokenCount,
    int32_t spmdBlockIdx,
    int32_t spmdBlockCount) {
  if (spmdBlockIdx != 0 || spmdBlockCount != 1 || tokenCount != kRows) {
    return;
  }

  GlobalTensor<bfloat16_t> xGm;
  GlobalTensor<bfloat16_t> yGm;
  GlobalTensor<bfloat16_t> gammaGm;
  xGm.SetGlobalBuffer(x, kRows * kCols);
  yGm.SetGlobalBuffer(y, kRows * kCols);
  gammaGm.SetGlobalBuffer(gamma, kCols);

  TPipe pipe;
  TBuf<TPosition::VECCALC> xBuffer;
  TBuf<TPosition::VECCALC> gammaBuffer;
  TBuf<TPosition::VECCALC> outputBuffer;
  TBuf<TPosition::VECCALC> gammaFp32Buffer;
  TBuf<TPosition::VECCALC> xFp32Buffer;
  TBuf<TPosition::VECCALC> squareBuffer;
  TBuf<TPosition::VECCALC> rowStateBuffer;
  TBuf<TPosition::VECCALC> divisorBuffer;
  pipe.InitBuffer(xBuffer, kRows * kCols * sizeof(bfloat16_t));
  pipe.InitBuffer(gammaBuffer, kCols * sizeof(bfloat16_t));
  pipe.InitBuffer(outputBuffer, kRows * kCols * sizeof(bfloat16_t));
  pipe.InitBuffer(gammaFp32Buffer, kCols * sizeof(float));
  pipe.InitBuffer(xFp32Buffer, kCols * sizeof(float));
  pipe.InitBuffer(squareBuffer, kCols * sizeof(float));
  pipe.InitBuffer(rowStateBuffer, 32);
  pipe.InitBuffer(divisorBuffer, kCols * sizeof(float));

  LocalTensor<bfloat16_t> xLocal = xBuffer.Get<bfloat16_t>();
  LocalTensor<bfloat16_t> gammaLocal = gammaBuffer.Get<bfloat16_t>();
  LocalTensor<bfloat16_t> outputLocal = outputBuffer.Get<bfloat16_t>();
  LocalTensor<float> gammaFp32 = gammaFp32Buffer.Get<float>();
  LocalTensor<float> xFp32 = xFp32Buffer.Get<float>();
  LocalTensor<float> square = squareBuffer.Get<float>();
  LocalTensor<float> rowState = rowStateBuffer.Get<float>();
  LocalTensor<float> divisor = divisorBuffer.Get<float>();

  DataCopy(xLocal, xGm, kRows * kCols);
  DataCopy(gammaLocal, gammaGm, kCols);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

#if defined(RMSNORM_USE_VF)
  RmsNormFusedRowsImpl(
      reinterpret_cast<__ubuf__ bfloat16_t *>(xLocal.GetPhyAddr()),
      reinterpret_cast<__ubuf__ bfloat16_t *>(gammaLocal.GetPhyAddr()),
      reinterpret_cast<__ubuf__ bfloat16_t *>(outputLocal.GetPhyAddr()));
#else
  RmsNormOrdinaryRows(outputLocal, xLocal, gammaLocal, gammaFp32, xFp32,
                      square, rowState, divisor);
#endif

  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(yGm, outputLocal, kRows * kCols);
}

