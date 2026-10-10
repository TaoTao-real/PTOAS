# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use it in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for ``pto.tmov`` UB ND -> UB NZ, ordinary route.

The historical tree kept this candidate in ``lib/TileOps/a5/tmov_nd2nz.py``.
The provider reorganisation into category subdirectories did not carry it over,
so ND->NZ survived only as the VMI candidate ``vmi_tmov`` (``enable_vmi``
required, priority 100) and a kernel that asked for that layout with the VMI
route off had no legal candidate at all -- which is what the ported
``test/vpto/cases/vmi/fa-softmax-dn-init`` case hit.

This restores the ordinary candidate: an ordinary template at priority 0 with
no route constraint, so it is legal exactly when the VMI candidate is not, and
the two can neither compete nor tie.

It is not the VMI candidate's body: ``vmi_tmov`` renders through the VMI trace
(``_require_vmi_trace`` insists on ``ir_level='vmi'``), so this one emits the
VPTO-level sequence the historical template used -- ``pto.vlds`` from an
explicit per-row offset, then ``pto.vsstb`` with the NZ block stride, whose
POST_UPDATE advances the destination one fractal row per beat.  The load is
deliberately the pointer-and-offset form rather than a tile slice: pto-isa's
``vlds`` source-advance semantics are not what the reference relies on, and the
explicit offset is arithmetically equivalent (see the historical module's note,
which this body follows).

Coverage: the single-column-repeat shape (``cols <= lanes``), which is what
every kernel in the tree uses.  The historical candidate also covered
``cols > lanes`` (nested loops plus a trailing ``cfgVsstbLast`` beat); that
shape is refused here with the same message ``vmi_tmov`` uses, so
``tmov_nd2nz_multirepeat.pto`` stays unported.

``is_post_update`` is True as in history: the destination is advanced by the
store's POST_UPDATE, which is what the metadata flag describes.
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _nd_src(src_kind, src_memory_space, src_config, **_):
    # src: UB/vec RowMajor+NoneBox (ND)
    return (
        src_kind == "tile"
        and src_memory_space in {"ub", "vec"}
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
    )


def _nz_dst(dst_kind, dst_memory_space, dst_config, **_):
    # dst: UB/vec ColMajor+RowMajor (NZ)
    return (
        dst_kind == "tile"
        and dst_memory_space in {"ub", "vec"}
        and dst_config.b_layout != "row_major"
        and dst_config.s_layout == "row_major"
    )


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_nd2nz",
    dtypes=[("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"),
            ("i32", "i32"), ("i16", "i16"), ("i8", "i8"), ("ui8", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_nd_src,
                 _nz_dst,
                 tilelib.require_same_valid_shape("src", "dst")],
    id=8,
    loop_depth=1,
    is_post_update=True,
    tags=("move", "ub", "ub", "nd2nz", "nz"),
)
def template_tmov_nd2nz(src: pto.Tile, dst: pto.Tile):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    # ND is row-major with contiguous columns (RowStride = physical cols) and
    # never pads columns, so one row is one slice of `cols` elements.
    _, cols = src.spec.shape
    # NZ destination: physical (aligned) row count is the block stride, and one
    # beat advances the destination by repeat_stride fractal rows.
    block_stride = dst.spec.shape[0]
    repeat_stride = 1
    lanes = pto.elements_per_vreg(dtype)

    if not pto.const_expr(cols <= lanes):
        raise ValueError(
            "tmov ND->NZ currently supports at most one vector of columns"
        )

    # One row loop, destination carried and advanced by the store: the
    # historical single-column-repeat body (a plain NZ store, no last-block
    # reposition because there is no following column-block group).  The load
    # is the full vector at the row's start address and the mask carries the
    # row width, which is the shape the VMI candidate lowers to as well
    # (`vlds ptr[cols]` into a full vreg + `vsstb ..., block_stride=rows,
    # repeat_stride=1, mask`), so both routes store the same bytes.
    preg, _ = pto.make_mask(dtype, valid_cols)
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    loop = pto.for_(0, valid_rows, step=1).carry(dst_ptr=dst_ptr)
    with loop:
        src_off = loop.iv * cols
        vec = pto.vlds(src_ptr, src_off, dist="NORM")
        next_dst = pto.vsstb(vec, loop.dst_ptr, block_stride, repeat_stride,
                             preg, post_update="ON")
        loop.update(dst_ptr=next_dst)
