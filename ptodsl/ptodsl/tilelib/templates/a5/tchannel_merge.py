# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""A5 selected-VMI candidates for ``pto.tchannel_merge``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._vmi_common import canonical_vmi_template, emit_channel_merge_vmi


def _merge_legal(channels):
    def predicate(**context):
        dst_shape = context.get("dst_shape")
        dst_valid = context.get("dst_valid_shape")
        dst_config = context.get("dst_config")
        if (
            not isinstance(dst_shape, tuple)
            or len(dst_shape) != 2
            or dst_valid != dst_shape
            or dst_config is None
            or dst_config.b_layout != "row_major"
            or dst_config.s_layout != "none_box"
        ):
            return False
        rows, wide_cols = dst_shape
        if wide_cols <= 0 or wide_cols % channels:
            return False
        for index in range(channels):
            if context.get(f"src{index}_shape") != (
                rows,
                wide_cols // channels,
            ):
                return False
            if context.get(f"src{index}_valid_shape") != context.get(
                f"src{index}_shape"
            ):
                return False
            config = context.get(f"src{index}_config")
            if (
                config is None
                or config.b_layout != "row_major"
                or config.s_layout != "none_box"
            ):
                return False
        return True

    return predicate


_MERGE2_DTYPES = tuple((dtype,) * 3 for dtype in ("f32", "bf16", "f16"))
_MERGE4_DTYPES = tuple((dtype,) * 5 for dtype in ("f32", "bf16", "f16"))


def _emit_merge(srcs, dst):
    rows, channel_cols = srcs[0].valid_shape
    channels = len(srcs)
    wide_cols = channels * channel_cols
    with pto.for_(0, rows, step=1) as row:
        values = [pto.vlds(src[row, 0:]) for src in srcs]
        if channels == 2:
            merged, _ = pto.vintlv(values[0], values[1])
        else:
            pair02, _ = pto.vintlv(values[0], values[2])
            pair13, _ = pto.vintlv(values[1], values[3])
            merged, _ = pto.vintlv(pair02, pair13)
        mask, _ = pto.make_mask(dst.dtype, wide_cols)
        pto.vsts(merged, dst[row, 0:], mask)


@tilelib.tile_template(
    op="pto.tchannel_merge",
    target="a5",
    name="template_tchannel_merge_k2",
    dtypes=_MERGE2_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_merge_legal(2)],
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("movement", "rearrange"),
)
def template_tchannel_merge_k2(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    _emit_merge((src0, src1), dst)


@tilelib.tile_template(
    op="pto.tchannel_merge",
    target="a5",
    name="template_tchannel_merge_k4",
    dtypes=_MERGE4_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_merge_legal(4)],
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("movement", "rearrange"),
)
def template_tchannel_merge_k4(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    src3: pto.Tile,
    dst: pto.Tile,
):
    _emit_merge((src0, src1, src2, src3), dst)


@canonical_vmi_template(
    op="tchannel_merge",
    name="vmi_tchannel_merge_k2",
    dtypes=_MERGE2_DTYPES,
    constraints=(_merge_legal(2),),
)
def vmi_tchannel_merge_k2(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_channel_merge_vmi((src0, src1), dst)


@canonical_vmi_template(
    op="tchannel_merge",
    name="vmi_tchannel_merge_k4",
    dtypes=_MERGE4_DTYPES,
    constraints=(_merge_legal(4),),
)
def vmi_tchannel_merge_k4(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    src3: pto.Tile,
    dst: pto.Tile,
):
    emit_channel_merge_vmi((src0, src1, src2, src3), dst)
