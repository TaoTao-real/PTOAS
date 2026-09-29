#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the fast f32↔bf16 cast path on a group-slot reduction result."""

import numpy as np

from common import auto_main
from ptodsl import pto

LANES = 256
GROUPS = 8
GROUP_SIZE = LANES // GROUPS
SOURCE_BYTES = LANES * np.dtype(np.float32).itemsize
OUTPUT_ELEMENTS = 256
OUTPUT_BYTES = OUTPUT_ELEMENTS * np.dtype(np.uint16).itemsize


@pto.jit(
    name="vmi_group8_f32_bf16_fastmath_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def _kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.bf16, "gm")):
    src_ub = pto.castptr(pto.i64(0), pto.ptr(pto.f32, "ub"))
    dst_ub = pto.castptr(pto.i64(4096), pto.ptr(pto.bf16, "ub"))
    pto.mte_gm_ub(
        src,
        src_ub,
        0,
        SOURCE_BYTES,
        nburst=(1, SOURCE_BYTES, SOURCE_BYTES),
    )
    pto.mte_gm_ub(
        dst,
        dst_ub,
        0,
        OUTPUT_BYTES,
        nburst=(1, OUTPUT_BYTES, OUTPUT_BYTES),
    )
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    source = pto.vmi.vload(src_ub, 0, size=LANES)
    mask = pto.vmi.create_mask(LANES, size=LANES)
    magnitudes = pto.vmi.vabs(source)
    maxima = pto.vmi.vcmax(magnitudes, mask, group=GROUPS)
    maxima_bf16 = pto.vmi.vcvt(
        maxima,
        to_dtype=pto.bf16,
        rounding=pto.VcvtRoundMode.Z,
        saturate=pto.VcvtSatMode.NOSAT,
    )
    pto.vmi.vstore(maxima_bf16, dst_ub, 0, stride=1, group=GROUPS)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(
        dst_ub,
        dst,
        OUTPUT_BYTES,
        nburst=(1, OUTPUT_BYTES, OUTPUT_BYTES),
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_source():
    # Keep the test finite because --vmi-fastmath permits changed NaN results.
    source = np.empty(LANES, dtype=np.float32)
    for group in range(GROUPS):
        maximum = np.float32(1.0078125 + 0.5 * group)
        group_values = np.full(GROUP_SIZE, maximum / np.float32(2.0))
        group_values[0] = -maximum
        source[group * GROUP_SIZE : (group + 1) * GROUP_SIZE] = group_values
    return source


def _make_expected(source):
    maxima = np.max(np.abs(source.reshape(GROUPS, GROUP_SIZE)), axis=1)
    return (maxima.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _make_case():
    source = _make_source()
    output = np.full(OUTPUT_ELEMENTS, 0xA5A5, dtype=np.uint16)
    expected = output.copy()
    expected[:GROUPS] = _make_expected(source)
    return [source, output], expected


def _check_case(device_inputs, expected):
    actual = device_inputs[1].cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


CASES = [
    {
        "name": "vmi_group8_f32_bf16_fastmath",
        "kernel": _kernel,
        "make_case": _make_case,
        "check": _check_case,
    }
]


auto_main(globals())
