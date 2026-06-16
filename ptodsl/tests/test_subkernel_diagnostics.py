#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def define_bad_subkernel_signature_probe():
    @pto.simd
    def bad_tensor_formal(A: pto.tensor_spec(rank=2, dtype=pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_tensor_formal


def define_removed_ukernel_surface_probe():
    return pto.ukernel


def define_invalid_jit_mode_probe():
    @pto.jit(target="a5", mode="hybrid")
    def bad_mode_probe():
        pass

    return bad_mode_probe


@pto.simd
def host_tensor_operand_probe(tensor):
    pto.pipe_barrier(pto.Pipe.ALL)


def define_host_tensor_into_subkernel_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.tensor_spec(rank=2, dtype=pto.f32)):
        host_tensor_operand_probe(A)

    return bad_probe


@pto.simt
def nested_simt_probe():
    pto.get_tid_x()


@pto.simd
def illegal_simt_placement_probe():
    nested_simt_probe()


@pto.jit(target="a5")
def nested_simt_from_simd_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    illegal_simt_placement_probe()


@pto.simd
def illegal_inline_simt_placement_probe():
    with pto.simt():
        pto.get_tid_x()


@pto.jit(target="a5")
def nested_inline_simt_from_simd_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    illegal_inline_simt_placement_probe()


@pto.simd
def simd_value_escape_probe():
    return pto.pset_b32("PAT_ALL")


@pto.jit(target="a5")
def simd_value_escape_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    simd_value_escape_probe()


def main() -> None:
    expect_raises(
        define_removed_ukernel_surface_probe,
        AttributeError,
        "pto.ukernel is not a supported PTODSL public interface",
        '@pto.jit(mode="explicit")',
        "@pto.simd/@pto.simt/@pto.cube",
    )
    expect_raises(
        define_invalid_jit_mode_probe,
        ValueError,
        "unsupported PTODSL jit mode 'hybrid'",
        "bad_mode_probe",
        __file__,
        "expected 'auto' or 'explicit'",
    )
    expect_raises(
        define_bad_subkernel_signature_probe,
        TypeError,
        "@pto.simd parameter 'A' cannot be annotated with pto.tensor_spec(...)",
        "@pto.jit positional parameters",
    )
    expect_raises(
        define_host_tensor_into_subkernel_probe,
        TypeError,
        "@pto.jit positional parameter 'A' still uses legacy host-tensor entry annotation",
        "no longer accepts pto.tensor_spec(...)",
        "pto.make_tensor_view(...)",
    )
    expect_raises(
        nested_simt_from_simd_entry.compile,
        RuntimeError,
        "@pto.simt helper materialization is only supported from the top-level @pto.jit body",
        "inside @pto.simd",
    )
    expect_raises(
        nested_inline_simt_from_simd_entry.compile,
        RuntimeError,
        "inline pto.simt() may only be used from the top-level @pto.jit body",
        "inside @pto.simd",
    )
    expect_raises(
        simd_value_escape_entry.compile,
        RuntimeError,
        "@pto.simd cannot return transient SIMD values",
        "!pto.mask<b32>",
        "Write the value back to a Tile/UB buffer instead",
    )
    print("ptodsl_subkernel_diagnostics: PASS")


if __name__ == "__main__":
    main()
