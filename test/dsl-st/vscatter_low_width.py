#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

REQUESTS = 128
SEED = 29

def _copy_gm_to_ub(src, dst):
    pto.mte_gm_ub(src, dst, 0, 256, nburst=(8, 256, 256))

def _copy_ub_to_gm(src, dst):
    pto.mte_ub_gm(src, dst, 256, nburst=(8, 256, 256))

@pto.jit(
    name="vscatter_b8_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vscatter_b8_kernel(
    src: pto.ptr(pto.ui8, "gm"),
    indices: pto.ptr(pto.ui16, "gm"),
    output: pto.ptr(pto.ui8, "gm"),
):
    ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    ub_indices = pto.castptr(pto.const(256, dtype=pto.i64), pto.ptr(pto.ui16, "ub"))
    ub_output = pto.castptr(pto.const(512, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))

    _copy_gm_to_ub(src, ub_src)
    _copy_gm_to_ub(indices, ub_indices)
    _copy_gm_to_ub(output, ub_output)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    offset = pto.const(0, dtype=pto.index)
    mask = pto.pset_b16(pto.MaskPattern.ALL)
    values = pto.vlds(ub_src, offset)
    offsets = pto.vlds(ub_indices, offset)
    pto.vscatter(values, ub_output, offsets, mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    _copy_ub_to_gm(ub_output, output)
    pto.pipe_barrier(pto.Pipe.ALL)

@pto.jit(
    name="vscatter_b16_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vscatter_b16_kernel(
    src: pto.ptr(pto.i16, "gm"),
    indices: pto.ptr(pto.ui16, "gm"),
    output: pto.ptr(pto.i16, "gm"),
):
    ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i16, "ub"))
    ub_indices = pto.castptr(pto.const(256, dtype=pto.i64), pto.ptr(pto.ui16, "ub"))
    ub_output = pto.castptr(pto.const(512, dtype=pto.i64), pto.ptr(pto.i16, "ub"))

    _copy_gm_to_ub(src, ub_src)
    _copy_gm_to_ub(indices, ub_indices)
    _copy_gm_to_ub(output, ub_output)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    offset = pto.const(0, dtype=pto.index)
    mask = pto.pset_b16(pto.MaskPattern.ALL)
    values = pto.vlds(ub_src, offset)
    offsets = pto.vlds(ub_indices, offset)
    pto.vscatter(values, ub_output, offsets, mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    _copy_ub_to_gm(ub_output, output)
    pto.pipe_barrier(pto.Pipe.ALL)

def _indices() -> np.ndarray:
    return np.random.default_rng(SEED).permutation(REQUESTS).astype(np.uint16)

def _b8_inputs():
    src = np.empty(REQUESTS * 2, dtype=np.uint8)
    src[0::2] = (np.arange(REQUESTS, dtype=np.uint16) * 7 + 3).astype(np.uint8)
    src[1::2] = 255 - src[0::2]
    return [src, _indices()]

def _b8_expected(src, indices):
    result = np.zeros(REQUESTS * 2, dtype=np.uint8)
    result[indices] = src[0::2]
    return result

def _b16_inputs():
    src = (np.arange(REQUESTS, dtype=np.int32) * 257 - 16000).astype(np.int16)
    return [src, _indices()]

def _b16_expected(src, indices):
    result = np.zeros(REQUESTS, dtype=np.int16)
    result[indices] = src
    return result

CASES = [
    golden_output_case(
        "vscatter_b8_even_bytes",
        vscatter_b8_kernel,
        inputs=_b8_inputs,
        expected=_b8_expected,
        output_shape=(REQUESTS * 2,),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vscatter_b16",
        vscatter_b16_kernel,
        inputs=_b16_inputs,
        expected=_b16_expected,
        output_shape=(REQUESTS,),
        output_dtype=np.int16,
        rtol=0.0,
        atol=0.0,
    ),
]

auto_main(globals())
