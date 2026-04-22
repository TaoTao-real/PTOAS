# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PTO_IR = r"""

module {
  func.func @peak_8_overlapping(%arg0: !pto.ptr<f16>,
                                %arg1: !pto.ptr<f16>) {
    %c0_arg0 = arith.constant 0 : index
    %c16_arg0_0 = arith.constant 16 : index
    %c256_arg0_1 = arith.constant 256 : index
    %c256_arg0_s0 = arith.constant 256 : index
    %c1_arg0_s1 = arith.constant 1 : index
    %arg0_tv = pto.make_tensor_view %arg0, shape = [%c16_arg0_0, %c256_arg0_1], strides = [%c256_arg0_s0, %c1_arg0_s1] : !pto.tensor_view<16x256xf16>
    %arg0_pt = pto.partition_view %arg0_tv, offsets = [%c0_arg0, %c0_arg0], sizes = [%c16_arg0_0, %c256_arg0_1] : !pto.tensor_view<16x256xf16> -> !pto.partition_tensor_view<16x256xf16>
    %c0_arg1 = arith.constant 0 : index
    %c16_arg1_0 = arith.constant 16 : index
    %c256_arg1_1 = arith.constant 256 : index
    %c256_arg1_s0 = arith.constant 256 : index
    %c1_arg1_s1 = arith.constant 1 : index
    %arg1_tv = pto.make_tensor_view %arg1, shape = [%c16_arg1_0, %c256_arg1_1], strides = [%c256_arg1_s0, %c1_arg1_s1] : !pto.tensor_view<16x256xf16>
    %arg1_pt = pto.partition_view %arg1_tv, offsets = [%c0_arg1, %c0_arg1], sizes = [%c16_arg1_0, %c256_arg1_1] : !pto.tensor_view<16x256xf16> -> !pto.partition_tensor_view<16x256xf16>
    // Peak liveness: 8 buffers live at once.
    %u0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u4 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u5 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u6 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %u7 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u2 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u3 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u4 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u5 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u6 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%u7 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tstore ins(%u0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u2 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u3 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u4 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u5 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u6 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%u7 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    return
  }
}

// 8 live buffers implies a max offset of 7*8192 = 57344 bytes.
"""

if __name__ == "__main__":
    print(PTO_IR)
