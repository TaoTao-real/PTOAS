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

from ptodsl import pto, scalar
from ptodsl._ast_rewrite import PTODSLAstRewriteError
from ptodsl._host_tensors import inspect_host_tensor_metadata


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


@pto.jit(target="a5", ast_rewrite=False)
def native_python_if_runtime_const_probe():
    if pto.const(1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", ast_rewrite=False)
def native_python_range_runtime_metadata_probe(rows: pto.i32):
    for _ in range(rows):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def float_loop_bound_probe():
    with pto.for_(0, pto.const(1.5, dtype=pto.f32), step=1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def float_addptr_offset_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    _ = pto.addptr(tile.as_ptr(), pto.const(1.5, dtype=pto.f32))


@pto.jit(target="a5")
def float_bitwise_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 1])
    value = scalar.load(tile[0, 0])
    _ = value & 1


@pto.jit(target="a5")
def carry_update_mismatch_probe(*, BLOCK: pto.const_expr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(other=acc)


@pto.jit(target="a5")
def carry_final_mismatch_probe(*, BLOCK: pto.const_expr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(acc=acc)
    loop.final("missing")


@pto.jit(target="a5")
def misaligned_row_major_tile_probe():
    pto.alloc_tile(shape=[128, 1], dtype=pto.f32, valid_shape=[128, 1])


class MissingDTypeTensor:
    shape = (4, 8)
    strides = (8, 1)

    def data_ptr(self):
        return 1024


class BadDataHandleTensor:
    shape = (4, 8)
    strides = (8, 1)
    dtype = "float32"

    def data_ptr(self):
        return "not-an-int"


def define_missing_constexpr_default_probe():
    @pto.jit(target="a5")
    def bad_probe(*, BLOCK: pto.const_expr):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_illegal_keyword_only_probe():
    @pto.jit(target="a5")
    def bad_probe(*, BLOCK: pto.i32 = 8):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_missing_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_gm_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def good_probe(A: pto.ptr(pto.f32, "gm"), rows: pto.i32):
        pto.pipe_barrier(pto.Pipe.ALL)

    return good_probe


def define_default_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.ptr(pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ub_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.ptr(pto.f32, "ub")):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_legacy_tensor_spec_entry_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.tensor_spec(rank=2, dtype=pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_conflict_probe():
    @pto.jit(target="a5", ast_rewrite=False, frontend_options={"ast_rewrite": True})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_scalar_rewrite_probe():
    @pto.jit(target="a5", frontend_options={"rewrite_part": {"control_flow", "scalar"}})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_dump_source_probe():
    @pto.jit(target="a5", frontend_options={"dump_rewritten_source": True})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ast_if_undefined_old_value_probe():
    @pto.jit(target="a5")
    def bad_probe():
        cond = pto.const(1, dtype=pto.i1)
        lhs = pto.const(4, dtype=pto.i32)
        if cond:
            value = lhs
        _ = value

    return bad_probe


def define_ast_for_else_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            pto.pipe_barrier(pto.Pipe.ALL)
        else:
            pto.mem_bar(pto.BarrierType.VST_VLD)

    return bad_probe


def define_ast_for_non_range_probe():
    @pto.jit(target="a5")
    def bad_probe():
        for _ in [0, 1]:
            pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ast_for_tuple_target_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _, other in range(rows):
            _ = other

    return bad_probe


def define_ast_for_break_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            break

    return bad_probe


def define_ast_runtime_for_constexpr_break_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            if pto.const_expr(True):
                break

    return bad_probe


def define_ast_for_last_value_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for i in range(rows):
            last = i
        _ = last

    return bad_probe


def define_ast_for_loop_target_live_after_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for i in range(rows):
            pto.pipe_barrier(pto.Pipe.ALL)
        _ = i

    return bad_probe


@pto.jit(target="a5")
def make_tensor_view_missing_metadata_probe(
    x_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    _ = rows
    _ = cols
    pto.make_tensor_view(x_ptr)


@pto.jit(target="a5")
def missing_if_branch_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        _ = br


@pto.jit(target="a5")
def stray_if_body_op_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        pto.pipe_barrier(pto.Pipe.ALL)
        with br.then_:
            pto.mem_bar(pto.BarrierType.VST_VLD)


@pto.jit(target="a5")
def assign_outside_branch_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        br.assign(val=pto.const(1, dtype=pto.i32))


@pto.jit(target="a5")
def missing_else_assign_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def assign_name_mismatch_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(other=rhs)


@pto.jit(target="a5")
def assign_type_mismatch_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2.0, dtype=pto.f32)
    cond = lhs > pto.const(0, dtype=pto.i32)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=rhs)


@pto.jit(target="a5")
def duplicate_assign_probe():
    lhs = pto.const(4, dtype=pto.i32)
    cond = lhs > pto.const(0, dtype=pto.i32)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(val=lhs)
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=lhs)


@pto.jit(target="a5")
def unknown_branch_result_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=rhs)
    _ = br.other


def main() -> None:
    expect_raises(
        native_python_if_runtime_const_probe.compile,
        TypeError,
        "native Python if/while condition",
        "pto.if_(...)",
        "pto.const_expr",
    )
    expect_raises(
        native_python_range_runtime_metadata_probe.compile,
        TypeError,
        "native Python range()/loop bound",
        "pto.for_(...)",
        "runtime value",
    )
    expect_raises(
        float_loop_bound_probe.compile,
        TypeError,
        "pto.for_(...) loop bound",
        "expects an index or integer runtime scalar",
        "f32",
    )
    expect_raises(
        float_addptr_offset_probe.compile,
        TypeError,
        "addptr(ptr, offset)",
        "expects an index-like scalar",
        "f32",
    )
    expect_raises(
        float_bitwise_probe.compile,
        TypeError,
        "runtime scalar bitwise operator",
        "expects integer-like operands",
    )
    expect_raises(
        carry_update_mismatch_probe.compile,
        RuntimeError,
        "loop.update(...) must match carry names exactly",
        "missing: acc",
        "unexpected: other",
    )
    expect_raises(
        carry_final_mismatch_probe.compile,
        RuntimeError,
        "loop.final(...) requested unknown carry state 'missing'",
        "expected one of: acc",
    )
    expect_raises(
        misaligned_row_major_tile_probe.compile,
        TypeError,
        "alloc_tile(shape=...) physical row layout is invalid",
        "shape=[128, 1]",
        "row byte size of 4",
        "32-byte aligned",
        "prefer blayout='ColMajor'",
    )
    expect_raises(
        define_missing_constexpr_default_probe,
        TypeError,
        "@pto.jit constexpr parameter 'BLOCK' must declare a default value",
        ".compile(...)",
    )
    expect_raises(
        define_illegal_keyword_only_probe,
        TypeError,
        "@pto.jit keyword-only parameter 'BLOCK' uses unsupported compile-time annotation",
        "pto.const_expr",
        "move runtime data to positional pointer/scalar parameters",
    )
    expect_raises(
        define_missing_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' does not declare an entry ABI annotation",
        'pto.ptr(pto.f32, "gm")',
        "pto.i32/pto.f32/pto.i1",
    )
    gm_ptr_entry_probe = define_gm_ptr_entry_annotation_probe()
    expect(hasattr(gm_ptr_entry_probe, "compile"), "expected explicit GM pointer entry to be accepted")
    expect_raises(
        define_default_ptr_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' uses non-GM pointer entry annotation",
        'pto.ptr(pto.f32, "gm")',
        'spell out "gm" explicitly',
    )
    expect_raises(
        define_ub_ptr_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' uses non-GM pointer entry annotation",
        'pto.ptr(pto.f32, "gm")',
        "only accepts explicit GM pointers",
    )
    expect_raises(
        define_legacy_tensor_spec_entry_probe,
        TypeError,
        "@pto.jit positional parameter 'A' still uses legacy host-tensor entry annotation",
        "no longer accepts pto.tensor_spec(...)",
        "pto.make_tensor_view(...)",
    )
    expect_raises(
        define_frontend_options_conflict_probe,
        ValueError,
        "ast_rewrite conflicts with frontend_options",
    )
    expect_raises(
        define_frontend_options_scalar_rewrite_probe,
        ValueError,
        "rewrite_part",
        "currently only supports",
        "scalar",
    )
    expect_raises(
        define_frontend_options_dump_source_probe,
        ValueError,
        "dump_rewritten_source",
        "reserved but not implemented",
    )
    expect_raises(
        lambda: define_ast_if_undefined_old_value_probe().compile(),
        NameError,
        "value",
    )
    expect_raises(
        lambda: define_ast_for_else_probe().compile(),
        PTODSLAstRewriteError,
        "does not support for-else",
    )
    expect_raises(
        lambda: define_ast_for_non_range_probe().compile(),
        PTODSLAstRewriteError,
        "only rewrites for-loops over range(...)",
    )
    expect_raises(
        lambda: define_ast_for_tuple_target_probe().compile(),
        PTODSLAstRewriteError,
        "runtime for-loops require a simple name target",
    )
    expect_raises(
        lambda: define_ast_for_break_probe().compile(),
        PTODSLAstRewriteError,
        "does not support break/continue",
    )
    expect_raises(
        lambda: define_ast_runtime_for_constexpr_break_probe().compile(),
        PTODSLAstRewriteError,
        "does not support break/continue",
    )
    expect_raises(
        lambda: define_ast_for_last_value_probe().compile(),
        PTODSLAstRewriteError,
        "cannot expose last-iteration-only values yet",
        "last",
    )
    expect_raises(
        lambda: define_ast_for_loop_target_live_after_probe().compile(),
        PTODSLAstRewriteError,
        "cannot expose the loop induction variable outside the loop yet",
        "i",
    )
    expect_raises(
        make_tensor_view_missing_metadata_probe.compile,
        TypeError,
        "make_tensor_view(",
        "requires explicit shape= and strides=",
        "Do not rely on host tensor proxy metadata",
    )
    expect_raises(
        missing_if_branch_probe.compile,
        RuntimeError,
        "requires at least one explicit branch block",
        "with br.then_:",
    )
    expect_raises(
        stray_if_body_op_probe.compile,
        RuntimeError,
        "body may only contain explicit 'with br.then_:' / 'with br.else_:' blocks",
        "outer if body",
    )
    expect_raises(
        assign_outside_branch_probe.compile,
        RuntimeError,
        "br.assign(...) may only be used inside br.then_ or br.else_",
    )
    expect_raises(
        missing_else_assign_probe.compile,
        RuntimeError,
        "automatic branch merge requires both br.then_ and br.else_ to call br.assign(...)",
    )
    expect_raises(
        assign_name_mismatch_probe.compile,
        RuntimeError,
        "br.assign(...) names must match across branches",
        "missing in else: val",
        "missing in then: other",
    )
    expect_raises(
        assign_type_mismatch_probe.compile,
        RuntimeError,
        "br.assign(...) type mismatch for 'val'",
    )
    expect_raises(
        duplicate_assign_probe.compile,
        RuntimeError,
        "br.then_ may call br.assign(...) at most once",
    )
    expect_raises(
        unknown_branch_result_probe.compile,
        AttributeError,
        "br.other was not assigned by this conditional",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(MissingDTypeTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "missing .dtype",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(BadDataHandleTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "data_ptr must return an integer-like data handle",
    )
    print("ptodsl_jit_diagnostics: PASS")


if __name__ == "__main__":
    main()
