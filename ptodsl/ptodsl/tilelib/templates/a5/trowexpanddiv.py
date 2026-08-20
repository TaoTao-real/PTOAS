# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trowexpanddiv — default precision only."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._expand_binary import (
    FLOAT_SIGNATURES,
    _emit_row_expand_body,
    _row_expand_layout,
    _valid_row_expand_binary,
    register_row_expand_binary,
)
from .div_hp import _div_ieee754_f16_impl, _div_ieee754_f32_impl


def _is_high_precision_with_tmp(precisionType="default", operand_kinds=(), **_):
    return (
        precisionType == "high_precision"
        and tuple(operand_kinds) == ("tile", "tile", "tile", "tile")
    )


template_trowexpanddiv = register_row_expand_binary(
    op="pto.trowexpanddiv",
    name="template_trowexpanddiv",
    vector_op=pto.vdiv,
    dtypes=FLOAT_SIGNATURES,
)


@tilelib.tile_template(
    op="pto.trowexpanddiv",
    target="a5",
    name="template_trowexpanddiv_high_precision",
    dtypes=[
        ("f16", "f16", "f16", "f16"),
        ("f32", "f32", "f32", "f32"),
    ],
    iteration_axis="row",
    op_engine="vector",
    op_class="broadcast",
    constraints=[
        _row_expand_layout,
        _valid_row_expand_binary,
        _is_high_precision_with_tmp,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("row_expand", "binary", "high_precision"),
)
def template_trowexpanddiv_high_precision(
    src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile
):
    _ = tmp
    divide = (
        _div_ieee754_f32_impl
        if str(dst.dtype) == "f32"
        else _div_ieee754_f16_impl
    )
    _emit_row_expand_body(src0, src1, dst, divide)


from ._vmi_common import (  # noqa: E402
    _single_vl_row_expand_div_vmi_legal,
    canonical_vmi_template,
    emit_row_expand_div_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="trowexpanddiv",
    name="vmi_trowexpanddiv",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default",)},
    constraints=(_single_vl_row_expand_div_vmi_legal,),
)
def vmi_trowexpanddiv(src: pto.Tile, row_values: pto.Tile, dst: pto.Tile):
    emit_row_expand_div_vmi(src, row_values, dst)
