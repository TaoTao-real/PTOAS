# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for basic pto.tmov UB-to-UB."""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ptodsl._tile_template_tracing import (
    _require_vmi_trace,
    for_,
    index_mul,
    vmi_create_mask_lanes,
    vmi_prepare_tile_access,
)
from ptodsl._vmi_namespace import vmi as _vmi

from .._vmi_common import (
    NUMERIC_DTYPES,
    _move as _vmi_move,
    bf16,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
    full_physical_row_vmi_constraint,
)


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _vmi_tmov_shape_supported(**metadata):
    """Stage-1 VMI move gate; ordinary TileLib remains the fallback."""
    # RowPlusOne deliberately has a larger physical destination (the padding
    # band never leaves UB), so legality is defined by logical valid shape.
    src_shape = metadata.get("src_valid_shape", metadata.get("src_shape", ()))
    dst_shape = metadata.get("dst_valid_shape", metadata.get("dst_shape", ()))
    return src_shape == dst_shape and len(src_shape) == 2


def _vmi_tmov_physicalization_supported(**metadata):
    """Allow only layouts representable by the current VMI move emitter."""
    dst_config = metadata.get("dst_config")
    if dst_config is None:
        return True
    layout = getattr(dst_config, "b_layout", "row_major")
    if layout == "col_major":
        shape = metadata.get("src_valid_shape", metadata.get("src_shape", ()))
        # The VMI implementation constructs a logical vreg and mask of this
        # exact width. Reject unsupported formal PTODSL widths before template
        # selection so the existing MI ND->NZ template can handle them.
        if len(shape) != 2 or shape[1] not in {1, 2, 4, 8, 64, 128, 256}:
            return False
    return layout in {"row_major", "col_major"}


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_basic",
    dtypes=[
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("i32", "i32"),
        ("i16", "i16"),
        ("i8", "i8"),
        ("ui8", "ui8"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape("src", "dst"),
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("move", "ub", "ub"),
)
def template_tmov(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            if str(src.dtype) != str(dst.dtype):
                data = pto.vbitcast(data, dst.dtype)
            pto.vsts(data, dst[row, col:], mask)


@canonical_vmi_template(
    target="a5",
    op="tmov",
    name="vmi_tmov",
    requires_full_physical_row=False,
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
        ("ui8", "ui8"),
    ),
    constraints=(
        _vmi_tmov_physicalization_supported,
        _vmi_tmov_shape_supported,
        tilelib.require_same_valid_shape("src", "dst"),
    ),
    tags=("supports_partial_valid_shape",),
)
def vmi_tmov(src: pto.Tile, dst: pto.Tile):
    if dst.spec.b_layout != "col_major":
        # Keep the implementation contract aligned with the candidate
        # metadata above.  In particular, DSv4 uses bf16 ND-to-ND moves;
        # emit_elementwise_vmi defaults to f32 when no dtype set is passed.
        emit_elementwise_vmi(
            dst,
            (src,),
            _vmi_move,
            allowed_dtypes=NUMERIC_DTYPES,
        )
        return

    if src.element_type != dst.element_type:
        raise ValueError("tmov ND->NZ requires matching source and destination dtypes")
    if src.spec.b_layout != "row_major":
        raise ValueError("tmov ND->NZ requires a row-major source")
    valid_rows, cols = src.spec.effective_valid_shape
    if (valid_rows, cols) != dst.spec.effective_valid_shape:
        raise ValueError("tmov ND->NZ requires matching valid shapes")
    lanes = src.element_type.lanes
    if cols > lanes:
        raise ValueError("tmov ND->NZ currently supports at most one vector of columns")

    trace = _require_vmi_trace("tmov_nd2nz")
    vmi_prepare_tile_access(src, dst)
    src_ptr = trace.ensure_tile_ptr(src)
    dst_ptr = trace.ensure_tile_ptr(dst)
    mask = vmi_create_mask_lanes(cols, cols, src.element_type)
    # NZ stores advance by one 32-byte fractal row, not by one logical ND row.
    # The historical emitter expressed this as a byte/element offset; retaining
    # that explicit affine offset is required for RowPlusOne and avoids relying
    # on the removed post-update vstore API.
    elements_per_block = lanes // 8
    with for_(0, valid_rows, step=1, state={"dst": dst_ptr.value}) as loop:
        src_offset = index_mul(loop.iv, cols)
        dst_offset = index_mul(loop.iv, elements_per_block)
        value = _vmi.vload(src_ptr.value, src_offset.value, size=cols)
        _vmi.vstore(
            value,
            loop.state.dst.value,
            dst_offset.value,
            mask.value,
            block_stride=dst.spec.shape[0],
        )
        # VMI vstore has no result; the destination pointer is loop-invariant.
        loop.yield_state(dst=loop.state.dst.value)
