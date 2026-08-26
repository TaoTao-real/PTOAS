# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""A5 selected-VMI candidates for ``pto.tchannel_split``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._vmi_common import canonical_vmi_template, emit_channel_split_vmi


def _split_legal(channels):
    def predicate(**context):
        src_shape = context.get("src_shape")
        src_valid = context.get("src_valid_shape")
        src_config = context.get("src_config")
        if (
            not isinstance(src_shape, tuple)
            or len(src_shape) != 2
            or src_valid != src_shape
            or src_config is None
            or src_config.b_layout != "row_major"
            or src_config.s_layout != "none_box"
        ):
            return False
        rows, wide_cols = src_shape
        if wide_cols <= 0 or wide_cols % channels:
            return False
        for index in range(channels):
            if context.get(f"dst{index}_shape") != (
                rows,
                wide_cols // channels,
            ):
                return False
            if context.get(f"dst{index}_valid_shape") != context.get(
                f"dst{index}_shape"
            ):
                return False
            config = context.get(f"dst{index}_config")
            if (
                config is None
                or config.b_layout != "row_major"
                or config.s_layout != "none_box"
            ):
                return False
        return True

    return predicate


_SPLIT2_DTYPES = tuple((dtype,) * 3 for dtype in ("f32", "bf16", "f16"))
_SPLIT4_DTYPES = tuple((dtype,) * 5 for dtype in ("f32", "bf16", "f16"))


def _emit_split(src, dsts):
    rows, wide_cols = src.valid_shape
    channels = len(dsts)
    channel_cols = wide_cols // channels
    with pto.for_(0, rows, step=1) as row:
        source = pto.vlds(src[row, 0:])
        low, high = pto.vdintlv(source, source)
        if channels == 2:
            values = (low, high)
        else:
            ch0, ch2 = pto.vdintlv(low, low)
            ch1, ch3 = pto.vdintlv(high, high)
            values = (ch0, ch1, ch2, ch3)
        mask, _ = pto.make_mask(src.dtype, channel_cols)
        for value, dst in zip(values, dsts):
            pto.vsts(value, dst[row, 0:], mask)


@tilelib.tile_template(
    op="pto.tchannel_split",
    target="a5",
    name="template_tchannel_split_k2",
    dtypes=_SPLIT2_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_split_legal(2)],
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("movement", "rearrange"),
)
def template_tchannel_split_k2(src: pto.Tile, dst0: pto.Tile, dst1: pto.Tile):
    _emit_split(src, (dst0, dst1))


@tilelib.tile_template(
    op="pto.tchannel_split",
    target="a5",
    name="template_tchannel_split_k4",
    dtypes=_SPLIT4_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_split_legal(4)],
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("movement", "rearrange"),
)
def template_tchannel_split_k4(
    src: pto.Tile,
    dst0: pto.Tile,
    dst1: pto.Tile,
    dst2: pto.Tile,
    dst3: pto.Tile,
):
    _emit_split(src, (dst0, dst1, dst2, dst3))


@canonical_vmi_template(
    op="tchannel_split",
    name="vmi_tchannel_split_k2",
    dtypes=_SPLIT2_DTYPES,
    constraints=(_split_legal(2),),
)
def vmi_tchannel_split_k2(src: pto.Tile, dst0: pto.Tile, dst1: pto.Tile):
    emit_channel_split_vmi(src, (dst0, dst1))


@canonical_vmi_template(
    op="tchannel_split",
    name="vmi_tchannel_split_k4",
    dtypes=_SPLIT4_DTYPES,
    constraints=(_split_legal(4),),
)
def vmi_tchannel_split_k4(
    src: pto.Tile,
    dst0: pto.Tile,
    dst1: pto.Tile,
    dst2: pto.Tile,
    dst3: pto.Tile,
):
    emit_channel_split_vmi(src, (dst0, dst1, dst2, dst3))
