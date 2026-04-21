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
  const int64_t v12 = 1024;
  using T = float;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  for (size_t v13 = (size_t) v10; v13 < ((size_t) v8); v13 += (size_t) v9) {
    int32_t v14 = (int32_t) v13;
    int64_t v15 = (int32_t) ((uint32_t) v14 % (uint32_t) v8) != v10 ? v12 : v11;
    Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16;
    TASSIGN(v16, v15);
    __ubuf__ half* v17 = v16.data();
    Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v18;
    uint64_t v19 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v17 + (v6 + v6 * v5) + v6 * v4));
    TASSIGN(v18, v19);
    __ubuf__ half* v20 = v16.data();
    Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21;
    uint64_t v22 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v20 + (v6 + v6 * v5) + v3 * v4));
    TASSIGN(v21, v22);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    if ((int32_t) ((uint32_t) v14 % (uint32_t) v8) == v10) {
      pto::Shape<1, 1, 1, 16, 16> v23 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v24 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v25 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v23, v24);
      pto::Shape<1, 1, 1, 16, 16> v26 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v27 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v26, v27);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v18, v25);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v28, v18);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    } else {
      pto::Shape<1, 1, 1, 16, 16> v29 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v30 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v29, v30);
      pto::Shape<1, 1, 1, 16, 16> v32 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v33 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v9), v32, v33);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v21, v31);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v34, v21);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
