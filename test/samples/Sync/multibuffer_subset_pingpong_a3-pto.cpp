#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void multibuffer_subset_pingpong_a3(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 16;
  unsigned v4 = 32;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 16;
  int32_t v8 = 8;
  int32_t v9 = 2;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int32_t v12 = 32;
  int64_t v13 = 0;
  int64_t v14 = 1024;
  using T = float;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v15 = (size_t) v11; v15 < ((size_t) v8); v15 += (size_t) v10) {
    int32_t v16 = (int32_t) v15;
    bool v17 = (int32_t) ((uint32_t) v16 % (uint32_t) v9) != v11;
    int64_t v18 = v17 ? v14 : v13;
    Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 32> v19;
    TASSIGN(v19, v18);
    Tile<TileType::Vec, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
    TRESHAPE(v20, v19);
    __ubuf__ half* v21 = v20.data();
    Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null> v22;
    uint64_t v23 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v21 + (v6 + v6 * v4) + v6 * v5));
    TASSIGN(v22, v23);
    __ubuf__ half* v24 = v20.data();
    Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null> v25;
    uint64_t v26 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v24 + (v6 + v6 * v4) + v3 * v5));
    TASSIGN(v25, v26);
    int32_t v27 = v17 ? v10 : v11;
    pto::Shape<1, 1, 1, 1, 16> v28 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v29 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v16 * (unsigned) v7 + v6 * (unsigned) v10), v28, v29);
    pto::Shape<1, 1, 1, 1, 16> v31 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v32 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v16 * (unsigned) v7 + v6 * (unsigned) v10), v31, v32);
    event_t v34 = static_cast<event_t>(v27);
    wait_flag(PIPE_MTE3, PIPE_MTE2, v34);
    if ((int32_t) ((uint32_t) v16 % (uint32_t) v9) == v11) {
      TLOAD(v22, v30);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v33, v22);
    } else {
      TLOAD(v25, v30);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v33, v25);
    };
    event_t v35 = static_cast<event_t>(v27);
    set_flag(PIPE_MTE3, PIPE_MTE2, v35);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  return;
}
