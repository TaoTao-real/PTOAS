# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PTO_IR = r"""

module {
  func.func @no_reuse_overlap(%arg0: !pto.ptr<f16>,
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
    %ub0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %ub1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    // Make lifetimes overlap by using both buffers after both are created.
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0_pt : !pto.partition_tensor_view<16x256xf16>)
             outs(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)
    pto.tstore ins(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_pt : !pto.partition_tensor_view<16x256xf16>)

    return
  }
}

// With overlapping lifetimes, offsets must differ.
"""

if __name__ == "__main__":
    print(PTO_IR)
