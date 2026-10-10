# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tmax."""

from ptodsl import pto

from .._common import same_dtype_signatures
from .._elementwise import register_binary
from .._vmi_common import (
    NUMERIC_DTYPES,
    _max as _vmi_max,
    canonical_vmi_template,
    emit_elementwise_vmi,
)


def _vmax(lhs, rhs, mask):
    return pto.vmax(lhs, rhs, mask)


_DTYPES = same_dtype_signatures(3)

# VMI candidate dtype set: A5 tmax ODS accepts i8/i16/i32/ui8/ui16/ui32/f16/f32
# and rejects bf16 (tsub shares this exact set); the order mirrors the
# historical literal triplets.
_VMI_DTYPES = tuple(
    same_dtype_signatures(3, ("f32", "f16", "i8", "i16", "i32", "ui8", "ui16", "ui32"))
)


template_tmax = register_binary(
    op="pto.tmax",
    name="template_tmax",
    vector_op=_vmax,
    dtypes=_DTYPES,
)


template_tmax_1d = register_binary(
    op="pto.tmax",
    name="template_tmax_1d",
    vector_op=_vmax,
    dtypes=_DTYPES,
    traversal="1d",
)


@canonical_vmi_template(
    target="a5",
    op="tmax",
    name="vmi_tmax",
    dtypes=_VMI_DTYPES,
)
def vmi_tmax(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    # A5 tmax ODS rejects bf16 (only i8/i16/i32/ui8/ui16/ui32/f16/f32); bf16 tmax
    # conservatively falls back to the ordinary PTODSL path.
    emit_elementwise_vmi(dst, (src0, src1), _vmi_max, allowed_dtypes=NUMERIC_DTYPES)
