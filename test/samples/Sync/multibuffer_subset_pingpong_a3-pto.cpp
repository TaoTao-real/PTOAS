// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void multibuffer_subset_pingpong_a3(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 16;
  unsigned v4 = 1;
  unsigned v5 = 32;
  unsigned v6 = 0;
  const int32_t v7 = 16;
  const int32_t v8 = 2;
  const int32_t v9 = 1;
  const int32_t v10 = 0;
  const int64_t v11 = 0;
  using T = float;
  Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v12;
  TASSIGN(v12, v11);
  __ubuf__ half* v13 = v12.data();
  __ubuf__ half* v14 = v13 + (v6 + v6 * v5 + v6 * v4);
  __ubuf__ half* v15 = (__ubuf__ half*) v14;
  Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16;
  uint64_t v17 = reinterpret_cast<uint64_t>(v14);
  TASSIGN(v16, v17);
  __ubuf__ half* v18 = v12.data();
  __ubuf__ half* v19 = v18 + (v6 + v6 * v5 + v3 * v4);
  __ubuf__ half* v20 = (__ubuf__ half*) v19;
  Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21;
  uint64_t v22 = reinterpret_cast<uint64_t>(v19);
  TASSIGN(v21, v22);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v23 = (size_t) v10; v23 < ((size_t) v8); v23 += (size_t) v9) {
    int32_t v24 = (int32_t) v23;
    int32_t v25 = (int32_t) ((uint32_t) v24 % (uint32_t) v8) == v9 ? v9 : v10;
    if ((int32_t) ((uint32_t) v24 % (uint32_t) v8) == v10) {
      pto::Shape<1, 1, 1, 16, 16> v26 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v27 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v26, v27);
      pto::Shape<1, 1, 1, 16, 16> v29 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v30 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v29, v30);
      event_t v32 = (event_t) v25;
      wait_flag(PIPE_MTE3, PIPE_MTE2, v32);
      TLOAD(v16, v28);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v31, v16);
      event_t v33 = (event_t) v25;
      set_flag(PIPE_MTE3, PIPE_MTE2, v33);
    } else {
      pto::Shape<1, 1, 1, 16, 16> v34 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v35 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v34, v35);
      pto::Shape<1, 1, 1, 16, 16> v37 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v38 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v37, v38);
      event_t v40 = (event_t) v25;
      wait_flag(PIPE_MTE3, PIPE_MTE2, v40);
      TLOAD(v21, v36);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v39, v21);
      event_t v41 = (event_t) v25;
      set_flag(PIPE_MTE3, PIPE_MTE2, v41);
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
