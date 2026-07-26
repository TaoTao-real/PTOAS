# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the scalar-pipe ``pto.tci`` form."""

from ptodsl import pto
from ptodsl import scalar
import ptodsl.tilelib as tilelib


def _single_row_ub(operand_kinds, dst_shape, dst_valid_shape,
                   dst_memory_space, dst_config, **_):
    return (
        operand_kinds == ("scalar", "tile")
        and len(dst_shape) == 2
        and dst_shape[0] == 1
        and len(dst_valid_shape) == 2
        and dst_valid_shape[0] == 1
        and dst_memory_space in {"ub", "vec"}
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


@tilelib.tile_template(
    op="pto.tci",
    target="a5",
    name="template_tci",
    dtypes=[("i32", "i32"), ("i16", "i16")],
    iteration_axis="none",
    op_engine="other",
    op_class="other",
    constraints=[_single_row_ub],
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("scalar", "sequence", "boundary"),
)
def template_tci(start, dst: pto.Tile):
    _, valid_cols = dst.valid_shape
    descending = bool(int(pto.get_op_attr("descending", "0")))
    with pto.for_(0, valid_cols, step=1) as col:
        offset = scalar.index_cast(dst.dtype, col)
        value = start - offset if descending else start + offset
        scalar.store(value, dst[0, col])
