# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmatmul.bias."""

import tilelang_dsl as pto


@pto.ckernel(
    target="a5",
    op="pto.tmatmul.bias",
    dtypes=[
        (pto.f16, pto.f16, pto.f32, pto.f32),
        (pto.bf16, pto.bf16, pto.f32, pto.f32),
        (pto.f32, pto.f32, pto.f32, pto.f32),
        (pto.i8, pto.i8, pto.i32, pto.i32),
    ],
)
def template_tmatmul_bias(lhs: pto.Tile, rhs: pto.Tile, bias: pto.Tile, dst: pto.Tile):
    m, k = lhs.valid_shape           # (validM, validK)
    _, n = rhs.valid_shape            # (validK, validN) → n = validN
    pto.mad_bias(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), bias.as_ptr(), m, n, k, disable_gemv=True)
    return None