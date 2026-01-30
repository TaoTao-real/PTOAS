#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void rowmin_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int64_t v7 = 0;
  int64_t v8 = 4096;
  using T = float;
  unsigned v9 = (unsigned) v5;
  unsigned v10 = v4 * v9;
  unsigned v11 = v4 + v10;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v4 * v12;
  unsigned v14 = v11 + v13;
  __gm__ float* v15 = v1 + v14;
  using GTShape_5241179056 = pto::Shape<32, 32>;
  using GTStride_5241179056 = pto::Stride<32, 1>;
  GTShape_5241179056 v16 = GTShape_5241179056();
  GTStride_5241179056 v17 = GTStride_5241179056();
  using GT_5241179056 = GlobalTensor<float, GTShape_5241179056, GTStride_5241179056>;
  GT_5241179056 v18 = GT_5241179056(v15, v16, v17);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v7);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v8);
  TLOAD(v19, v18);
  TROWMIN(v21, v19, v20);
  unsigned v22 = (unsigned) v5;
  unsigned v23 = v4 * v22;
  unsigned v24 = v4 + v23;
  unsigned v25 = (unsigned) v6;
  unsigned v26 = v4 * v25;
  unsigned v27 = v24 + v26;
  __gm__ float* v28 = v2 + v27;
  using GTShape_5241159232 = pto::Shape<32, 32>;
  using GTStride_5241159232 = pto::Stride<32, 1>;
  GTShape_5241159232 v29 = GTShape_5241159232();
  GTStride_5241159232 v30 = GTStride_5241159232();
  using GT_5241159232 = GlobalTensor<float, GTShape_5241159232, GTStride_5241159232>;
  GT_5241159232 v31 = GT_5241159232(v28, v29, v30);
  TSTORE(v31, v21);
  return;
}


