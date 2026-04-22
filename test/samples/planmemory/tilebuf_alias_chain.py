# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PTO_IR = r"""
module {
  func.func @tilebuf_alias_chain(%arg0: !pto.ptr<f16>,
                                 %arg1: !pto.ptr<i16>) {
    %c0_arg0 = arith.constant 0 : index
    %c32_arg0_0 = arith.constant 32 : index
    %c32_arg0_1 = arith.constant 32 : index
    %c32_arg0_s0 = arith.constant 32 : index
    %c1_arg0_s1 = arith.constant 1 : index
    %arg0_tv = pto.make_tensor_view %arg0, shape = [%c32_arg0_0, %c32_arg0_1], strides = [%c32_arg0_s0, %c1_arg0_s1] : !pto.tensor_view<32x32xf16>
    %arg0_pt = pto.partition_view %arg0_tv, offsets = [%c0_arg0, %c0_arg0], sizes = [%c32_arg0_0, %c32_arg0_1] : !pto.tensor_view<32x32xf16> -> !pto.partition_tensor_view<32x32xf16>
    %c0_arg1 = arith.constant 0 : index
    %c32_arg1_0 = arith.constant 32 : index
    %c16_arg1_1 = arith.constant 16 : index
    %c16_arg1_s0 = arith.constant 16 : index
    %c1_arg1_s1 = arith.constant 1 : index
    %arg1_tv = pto.make_tensor_view %arg1, shape = [%c32_arg1_0, %c16_arg1_1], strides = [%c16_arg1_s0, %c1_arg1_s1] : !pto.tensor_view<32x16xi16>
    %arg1_pt = pto.partition_view %arg1_tv, offsets = [%c0_arg1, %c0_arg1], sizes = [%c32_arg1_0, %c16_arg1_1] : !pto.tensor_view<32x16xi16> -> !pto.partition_tensor_view<32x16xi16>
    %c0 = arith.constant 0 : index

    %base = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sub = pto.subview %base[%c0, %c0] sizes [16, 32]
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
        -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %reshape = pto.treshape %sub
      : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
        -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=16, v_row=32, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %cast = pto.bitcast %reshape
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=16, v_row=32, v_col=16,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
        -> !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=16, v_row=32, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<32x32xf16>)
             outs(%base : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                                        blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%cast : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=16, v_row=32, v_col=16,
                                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<32x16xi16>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
