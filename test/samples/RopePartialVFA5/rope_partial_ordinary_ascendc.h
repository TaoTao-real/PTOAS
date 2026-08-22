// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0.

#ifndef PTOAS_TEST_SAMPLES_ROPE_PARTIAL_VF_A5_ORDINARY_H_
#define PTOAS_TEST_SAMPLES_ROPE_PARTIAL_VF_A5_ORDINARY_H_

#include "kernel_operator.h"

#include <cstdint>

namespace RopePartialA5 {

using namespace AscendC;

// Ordinary AscendC equivalent of HalfModeRopeVF.  T is FP32 and ROPET is BF16
// in the acceptance workload.  The caller owns all scratch tensors so their
// UB lifetime and aliasing are explicit.
template <typename T, typename ROPET>
__aicore__ inline void HalfModeRopeOrdinary(
    const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
    const LocalTensor<T> &inTensor, const LocalTensor<ROPET> &outTensor,
    const LocalTensor<T> &product0, const LocalTensor<T> &product1,
    const LocalTensor<T> &outFp32, uint32_t row, uint32_t col,
    uint32_t actualCol, uint64_t baseAddr) {
  const uint32_t halfCol = col / 2;

  for (uint32_t rowIdx = 0; rowIdx < row; ++rowIdx) {
    const uint32_t trigBase = rowIdx * col;
    const uint32_t inBase = rowIdx * actualCol + baseAddr;
    const uint32_t outBase = rowIdx * actualCol + baseAddr;

    // y[0:h] = cos[0:h] * x[0:h] - sin[0:h] * x[h:2h].
    Mul(product0, sinTensor[trigBase], inTensor[inBase + halfCol],
        halfCol);
    PipeBarrier<PIPE_V>();
    Mul(product1, cosTensor[trigBase], inTensor[inBase], halfCol);
    PipeBarrier<PIPE_V>();
    Sub(outFp32, product1, product0, halfCol);
    PipeBarrier<PIPE_V>();
    Cast(outTensor[outBase], outFp32, RoundMode::CAST_RINT, halfCol);
    PipeBarrier<PIPE_V>();

    // y[h:2h] = sin[h:2h] * x[0:h] + cos[h:2h] * x[h:2h].
    Mul(product0, sinTensor[trigBase + halfCol], inTensor[inBase],
        halfCol);
    PipeBarrier<PIPE_V>();
    Mul(product1, cosTensor[trigBase + halfCol],
        inTensor[inBase + halfCol], halfCol);
    PipeBarrier<PIPE_V>();
    Add(outFp32, product0, product1, halfCol);
    PipeBarrier<PIPE_V>();
    Cast(outTensor[outBase + halfCol], outFp32, RoundMode::CAST_RINT,
         halfCol);
    PipeBarrier<PIPE_V>();

    // Match the manual VF order: rotary part first, unchanged prefix second.
    for (uint64_t offset = 0; offset < baseAddr; offset += 64) {
      const uint32_t count = static_cast<uint32_t>(
          (baseAddr - offset) < 64 ? (baseAddr - offset) : 64);
      Cast(outTensor[rowIdx * actualCol + offset],
           inTensor[rowIdx * actualCol + offset], RoundMode::CAST_RINT,
           count);
      PipeBarrier<PIPE_V>();
    }
  }
}

// Ordinary AscendC equivalent of InterleaveModeRopeVF.  For each adjacent
// pair [x0, x1], rotated holds [-x1, x0].  The one-input DeInterleave and the
// two-output Interleave operate entirely on caller-provided UB scratch.
template <typename T, typename ROPET>
__aicore__ inline void InterleaveModeRopeOrdinary(
    const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
    const LocalTensor<T> &inTensor, const LocalTensor<ROPET> &outTensor,
    const LocalTensor<T> &even, const LocalTensor<T> &odd,
    const LocalTensor<T> &rotated, const LocalTensor<T> &cosProduct,
    const LocalTensor<T> &sinProduct, const LocalTensor<T> &outFp32,
    uint32_t row, uint32_t col, uint32_t actualCol, uint64_t baseAddr) {
  const uint32_t halfCol = col / 2;

  for (uint32_t rowIdx = 0; rowIdx < row; ++rowIdx) {
    const uint32_t trigBase = rowIdx * col;
    const uint32_t inBase = rowIdx * actualCol + baseAddr;
    const uint32_t outBase = rowIdx * actualCol + baseAddr;

    Mul(cosProduct, cosTensor[trigBase], inTensor[inBase], col);
    PipeBarrier<PIPE_V>();
    DeInterleave(even, odd, inTensor[inBase], col);
    PipeBarrier<PIPE_V>();
    Muls(odd, odd, static_cast<T>(-1.0), halfCol);
    PipeBarrier<PIPE_V>();
    Interleave(rotated, rotated[halfCol], odd, even, halfCol);
    PipeBarrier<PIPE_V>();
    Mul(sinProduct, sinTensor[trigBase], rotated, col);
    PipeBarrier<PIPE_V>();
    Add(outFp32, cosProduct, sinProduct, col);
    PipeBarrier<PIPE_V>();
    Cast(outTensor[outBase], outFp32, RoundMode::CAST_RINT, col);
    PipeBarrier<PIPE_V>();

    for (uint64_t offset = 0; offset < baseAddr; offset += 64) {
      const uint32_t count = static_cast<uint32_t>(
          (baseAddr - offset) < 64 ? (baseAddr - offset) : 64);
      Cast(outTensor[rowIdx * actualCol + offset],
           inTensor[rowIdx * actualCol + offset], RoundMode::CAST_RINT,
           count);
      PipeBarrier<PIPE_V>();
    }
  }
}

}  // namespace RopePartialA5

#endif  // PTOAS_TEST_SAMPLES_ROPE_PARTIAL_VF_A5_ORDINARY_H_
