# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib fallback template for ``pto.tgather``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _index_form(src_dtype, dst_dtype, indices_dtype, tmp_dtype, **_):
    return (
        src_dtype == dst_dtype
        and indices_dtype in {"i16", "i32"}
        and tmp_dtype == indices_dtype
    )


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_index",
    dtypes=[
        ("f32", "f32", "i32", "i32"),
        ("f16", "f16", "i16", "i16"),
        ("bf16", "bf16", "i16", "i16"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        tilelib.require_same_valid_shape("src", "dst", "indices"),
        tilelib.require_same_valid_shape("indices", "tmp"),
        _index_form,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "index", "hard_boundary"),
)
def template_tgather_index(
    src: pto.Tile,
    dst: pto.Tile,
    indices: pto.Tile,
    tmp: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    src_ptr = src.as_ptr()

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            offsets = pto.vlds(indices[row, col:])
            _ = pto.vlds(tmp[row, col:])
            data = pto.vgather2(src_ptr, offsets, mask)
            pto.vsts(data, dst[row, col:], mask)
