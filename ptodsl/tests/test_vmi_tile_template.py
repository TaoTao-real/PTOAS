#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    Tile,
    TileSpec,
    bf16,
    f16,
    f32,
    for_,
    i32,
    make_mask,
    scalar_const,
    tile_template,
    vadd,
    vecscope,
    vlds,
    vsts,
)
from ptodsl.tilelib.registry import TileTemplateRegistry
from ptodsl.vmi_tilelib import (
    VMI_TILELIB_REGISTRY,
    vmi_tadd_block64,
    vmi_tcolmax,
    vmi_tcolsum,
    vmi_tcolexpandsub,
    vmi_tcvt,
    vmi_texpands,
    vmi_texp_block64,
    vmi_trowexpanddiv,
)
from ptodsl.vmi_tilelib_helper import instantiate_candidate


TILE_SHAPE = (32, 64)
WIDE_TILE_SHAPE = (32, 128)
NARROW_TILE_SHAPE = (1, 32)


@tile_template(op="tadd", name="legacy_vpto_tadd")
def legacy_vpto_tadd(src0: Tile, src1: Tile, dst: Tile):
    with vecscope():
        rows, cols = dst.valid_shape
        with for_(0, rows, step=1) as row:
            remained = scalar_const(256, i32)
            with for_(0, cols, step=64) as col:
                mask, _ = make_mask(dst.element_type, remained)
                lhs = vlds(src0[row, col:])
                rhs = vlds(src1[row, col:])
                vsts(vadd(lhs, rhs, mask), dst[row, col:], mask)


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


def specialize_tadd(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_tadd_block64.specialize(src0=spec, src1=spec, dst=spec)


def specialize_texp(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_texp_block64.specialize(src=spec, dst=spec)


def check_canonical_block_map() -> None:
    block_map = CanonicalBlockMap(TILE_SHAPE, logical_lanes=64)
    expect(block_map.blocks_per_row == 1, "[32,64]xf32 should contain one block per row")
    expect(block_map.logical_block_count == 32, "[32,64]xf32 should contain 32 blocks")

    coordinate = block_map.coordinate(17)
    expect(coordinate.row == 17, "logical block 17 should map directly to row 17")
    expect(coordinate.block_in_row == 0, "each row should contain only block 0")
    expect(coordinate.col_start == 0, "the row-local block should start at column 0")
    expect(coordinate.linear_offset == 1088, "logical block 17 should start at offset 1088")
    expect(coordinate.active_lanes == 64, "the f32 contract should activate 64 lanes")

    wide_block_map = CanonicalBlockMap(WIDE_TILE_SHAPE, logical_lanes=128)
    expect(wide_block_map.blocks_per_row == 1, "[32,128]xf32 should contain one block per row")
    expect(wide_block_map.logical_block_count == 32, "[32,128]xf32 should contain 32 blocks")
    wide_coordinate = wide_block_map.coordinate(17)
    expect(wide_coordinate.row == 17, "wide logical block 17 should map directly to row 17")
    expect(wide_coordinate.col_start == 0, "the wide row-local block should start at column 0")
    expect(wide_coordinate.linear_offset == 2176, "wide logical block 17 should start at offset 2176")
    expect(wide_coordinate.active_lanes == 128, "wide rows should activate their full inner width")

    narrow_block_map = CanonicalBlockMap(NARROW_TILE_SHAPE, logical_lanes=32)
    expect(narrow_block_map.blocks_per_row == 1, "[1,32]xf32 should contain one block per row")
    expect(narrow_block_map.logical_block_count == 1, "[1,32]xf32 should contain one block")

    expect_raises(
        lambda: CanonicalBlockMap((32, 128), logical_lanes=64),
        ValueError,
        "exactly one logical VL block per row",
    )
    expect_raises(
        lambda: CanonicalBlockMap((32, 32), logical_lanes=64),
        ValueError,
        "exactly one logical VL block per row",
    )


def check_candidate_ir() -> tuple[str, str, str]:
    tadd = specialize_tadd()
    tadd.verify()
    tadd_text = tadd.mlir_text()
    expect("pto.vecscope" not in tadd_text, "VMI templates must remain scope-free")
    expect(tadd_text.count("scf.for") == 1, "tadd candidate should contain one flat loop")
    expect("arith.constant 32 : index" in tadd_text, "tadd should iterate once per row")
    expect(tadd_text.count("pto.vmi.vload") == 2, "tadd should issue two VMI loads")
    expect(tadd_text.count("pto.vmi.vadd") == 1, "tadd should issue one VMI add")
    expect(tadd_text.count("pto.vmi.vstore") == 1, "tadd should issue one VMI store")
    expect(tadd_text.count("pto.tile_buf_addr") == 3, "tadd should materialize three tile pointers")
    expect(
        tadd_text.rfind("pto.tile_buf_addr") < tadd_text.index("scf.for"),
        "tadd tile pointers should be materialized before the logical-block loop",
    )
    expect("!pto.vmi.vreg<64xf32>" in tadd_text, "tadd should use 64 logical f32 lanes")
    expect("pto.vlds" not in tadd_text, "VMI candidate should not emit physical vlds")
    expect("pto.vsts" not in tadd_text, "VMI candidate should not emit physical vsts")

    wide_tadd = specialize_tadd(shape=WIDE_TILE_SHAPE)
    wide_tadd.verify()
    wide_tadd_text = wide_tadd.mlir_text()
    expect(wide_tadd_text.count("scf.for") == 1, "wide tadd should still contain one row loop")
    expect("arith.constant 32 : index" in wide_tadd_text, "wide tadd should still iterate by row")
    expect("!pto.vmi.vreg<128xf32>" in wide_tadd_text, "wide tadd should use a 128-lane logical vreg")
    expect("!pto.vmi.mask<128xpred>" in wide_tadd_text, "wide tadd should use a 128-lane logical mask")
    expect(wide_tadd_text.count("pto.vmi.vadd") == 1, "wide tadd should still issue one logical add")

    texp = specialize_texp()
    texp.verify()
    texp_text = texp.mlir_text()
    expect("pto.vecscope" not in texp_text, "VMI templates must remain scope-free")
    expect(texp_text.count("scf.for") == 1, "texp candidate should contain one flat loop")
    expect(texp_text.count("pto.vmi.vload") == 1, "texp should issue one VMI load")
    expect(texp_text.count("pto.vmi.vexp") == 1, "texp should issue one VMI exp")
    expect(texp_text.count("pto.vmi.vstore") == 1, "texp should issue one VMI store")

    expect_raises(
        lambda: specialize_tadd(dtype=f16).mlir_text(),
        ValueError,
        "dtype is not supported",
    )
    wide_texp = specialize_texp(shape=WIDE_TILE_SHAPE)
    wide_texp.verify()
    wide_texp_text = wide_texp.mlir_text()
    expect(wide_texp_text.count("scf.for") == 1, "wide texp should still contain one row loop")
    expect("!pto.vmi.vreg<128xf32>" in wide_texp_text, "wide texp should use a 128-lane logical vreg")
    return tadd_text, wide_tadd_text, texp_text


def check_provider_helper() -> None:
    registered_tadd = VMI_TILELIB_REGISTRY.lookup("tadd", "a5")
    expect(
        len(registered_tadd) == 1,
        "tadd must have one registered canonical VMI template",
    )
    expect(
        registered_tadd[0] is vmi_tadd_block64,
        "the registered tadd template must be the exported canonical implementation",
    )
    expect(
        dict(vmi_texp_block64.context_constraints)
        == {"precisionType": ("default",)},
        "texp must declare its supported context attrs on the candidate",
    )

    raw_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    text = artifact.mlir_text()
    expect("pto.vmi.vadd" in text, "provider helper should instantiate the tadd VMI candidate")
    expect(text.count("scf.for") == 1, "provider helper should preserve one logical-block loop")

    f16_tile_spec = {
        **raw_tile_spec,
        "dtype": "f16",
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tadd",
            operand_specs=[f16_tile_spec, f16_tile_spec, f16_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )
    exp_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.texp",
        operand_specs=[raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    )
    expect(
        "pto.vmi.vexp" in exp_artifact.mlir_text(),
        "provider helper should accept the default texp precision contract",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tadd",
            operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "default"},
        ),
        ValueError,
        "does not support context attrs",
    )

    tmul_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tmul",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    expect("pto.vmi.vmul" in tmul_artifact.mlir_text(), "tmul should lower to VMI")

    narrow_tile_spec = {
        **raw_tile_spec,
        "shape": list(NARROW_TILE_SHAPE),
        "valid_shape": list(NARROW_TILE_SHAPE),
    }
    narrow = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[narrow_tile_spec, narrow_tile_spec, narrow_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(narrow.count("scf.for") == 1, "narrow tadd should emit one logical row loop")
    expect("!pto.vmi.vreg<32xf32>" in narrow, "narrow tadd should use a 32-lane logical vreg")

    scalar_spec = {"kind": "scalar", "dtype": "f32"}
    tmuls = instantiate_candidate(
        target="a5",
        op_name="pto.tmuls",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("%arg1: f32" in tmuls, "tmuls should preserve its runtime scalar parameter")
    expect("pto.vmi.vmuls" in tmuls, "tmuls should lower to VMI scalar multiply")

    scalar_expectations = {
        "tadds": "pto.vmi.vadds",
        "tmaxs": "pto.vmi.vmaxs",
        "tmins": "pto.vmi.vmins",
    }
    for op_name, expected_op in scalar_expectations.items():
        text = instantiate_candidate(
            target="a5",
            op_name=f"pto.{op_name}",
            operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")

    compact_elementwise = {
        **raw_tile_spec,
        "shape": [32, 8],
        "valid_shape": [32, 1],
    }
    compact_tadd = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[compact_elementwise] * 3,
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        compact_tadd.count("scf.for") == 1,
        "compact tadd should preserve one logical row loop",
    )
    expect(
        "!pto.vmi.vreg<1xf32>" in compact_tadd,
        "compact tadd should use its one valid column as the logical width",
    )
    expect(
        "arith.muli" in compact_tadd,
        "compact tadd must retain the physical row stride",
    )
    for op_name, expected_op in {
        "tadds": "pto.vmi.vadds",
        "tmuls": "pto.vmi.vmuls",
    }.items():
        text = instantiate_candidate(
            target="a5",
            op_name=f"pto.{op_name}",
            operand_specs=[compact_elementwise, scalar_spec, compact_elementwise],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        expect(expected_op in text, f"compact {op_name} should lower to {expected_op}")
    compact_sqrt = instantiate_candidate(
        target="a5",
        op_name="pto.tsqrt",
        operand_specs=[compact_elementwise, compact_elementwise],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vsqrt" in compact_sqrt, "compact tsqrt should lower to VMI")
    partial_rows = {**compact_elementwise, "valid_shape": [31, 1]}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tadd",
            operand_specs=[partial_rows] * 3,
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )

    tdivs = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vbrc" in tdivs, "tdivs should broadcast its scalar operand")
    expect("pto.vmi.vdiv" in tdivs, "tdivs should lower to VMI vector divide")
    tdivs_hp = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "high_precision"},
    ).mlir_text()
    expect(
        tdivs_hp.count("scf.for") == 1,
        "high-precision tdivs should still emit one logical row loop",
    )
    expect(
        "pto.vmi.vmula" in tdivs_hp,
        "high-precision tdivs should lower to the VMI refinement sequence",
    )

    reduced_tile_spec = {
        **raw_tile_spec,
        "shape": [32, 1],
        "valid_shape": [32, 1],
        "config": {**raw_tile_spec["config"], "b_layout": "col_major"},
    }
    rowmax = instantiate_candidate(
        target="a5",
        op_name="pto.trowmax",
        operand_specs=[raw_tile_spec, raw_tile_spec, reduced_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(rowmax.count("scf.for") == 1, "rowmax should emit only one runtime loop")
    expect(rowmax.count("pto.vmi.vcmax") == 1, "rowmax should reduce one VL per row")
    expect("!pto.vmi.vreg<1xf32>" in rowmax, "rowmax should produce 1-lane reductions")

    compact_row_state = {
        **raw_tile_spec,
        "shape": [32, 8],
        "valid_shape": [32, 1],
        "config": {**raw_tile_spec["config"], "b_layout": "row_major"},
    }
    compact_rowmax = instantiate_candidate(
        target="a5",
        op_name="pto.trowmax",
        operand_specs=[raw_tile_spec, raw_tile_spec, compact_row_state],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        compact_rowmax.count("scf.for") == 1,
        "compact rowmax should preserve one logical row loop",
    )
    expect(
        "pto.vmi.vcmax" in compact_rowmax
        and "pto.vmi.vstore" in compact_rowmax,
        "compact rowmax should lower to VMI reduction and store",
    )
    expect(
        "arith.muli" in compact_rowmax,
        "row-major compact output must use its physical row stride",
    )
    compact_rowsum = instantiate_candidate(
        target="a5",
        op_name="pto.trowsum",
        operand_specs=[raw_tile_spec, raw_tile_spec, compact_row_state],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vcadd" in compact_rowsum, "compact rowsum should lower to VMI")
    padded_col_row_state = {
        **raw_tile_spec,
        "shape": [256, 1],
        "valid_shape": [32, 1],
        "config": {**raw_tile_spec["config"], "b_layout": "col_major"},
    }
    padded_rowsum = instantiate_candidate(
        target="a5",
        op_name="pto.trowsum",
        operand_specs=[raw_tile_spec, raw_tile_spec, padded_col_row_state],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        padded_rowsum.count("scf.for") == 1
        and "pto.vmi.vcadd" in padded_rowsum,
        "padded col-major rowsum should preserve one logical row loop",
    )
    expect(
        padded_rowsum.count("pto.vmi.vstore") == 1,
        "padded col-major rowsum must retain its standalone compact store",
    )
    for invalid_padded in (
        {
            **padded_col_row_state,
            "shape": [32, 2],
            "valid_shape": [32, 1],
        },
        {**padded_col_row_state, "valid_shape": [31, 1]},
    ):
        expect_raises(
            lambda spec=invalid_padded: instantiate_candidate(
                target="a5",
                op_name="pto.trowsum",
                operand_specs=[raw_tile_spec, raw_tile_spec, spec],
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={},
            ),
            LookupError,
            "no legal PTODSL VMI candidate",
        )
    invalid_compact = {**compact_row_state, "valid_shape": [31, 1]}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.trowmax",
            operand_specs=[raw_tile_spec, raw_tile_spec, invalid_compact],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )

    row_expand = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpandsub",
        operand_specs=[raw_tile_spec, reduced_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        'pto.vmi.vload' in row_expand and 'dist_mode = "brc"' in row_expand,
        "row expand should use a direct broadcast load for compact row state",
    )
    expect(
        "pto.vmi.vbrc" not in row_expand,
        "row expand should not issue an unaligned wide load followed by vbrc",
    )

    f16_tile_spec = {**raw_tile_spec, "dtype": "f16"}
    tcvt = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "saturation_mode": "OFF"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in tcvt, "tcvt should lower to VMI conversion")

    tdiv = instantiate_candidate(
        target="a5",
        op_name="pto.tdiv",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vdiv" in tdiv, "default tdiv should lower to VMI vector divide")
    tdiv_hp = instantiate_candidate(
        target="a5",
        op_name="pto.tdiv",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "high_precision"},
    ).mlir_text()
    expect(
        "pto.vmi.vmula" in tdiv_hp,
        "high-precision tdiv should lower to the VMI refinement sequence",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.texp",
            operand_specs=[raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high"},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )

    duplicate_module = ModuleType("ptodsl_test_duplicate_vmi_candidates")
    duplicate_module.VMI_TILELIB_REGISTRY = TileTemplateRegistry()

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_a", ir_level="vmi")
    def duplicate_tadd_a(src0: Tile, src1: Tile, dst: Tile):
        pass

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_b", ir_level="vmi")
    def duplicate_tadd_b(src0: Tile, src1: Tile, dst: Tile):
        pass

    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_a)
    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_b)
    sys.modules[duplicate_module.__name__] = duplicate_module
    try:
        expect_raises(
            lambda: instantiate_candidate(
                target="a5",
                op_name="pto.tadd",
                operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
                provider_module=duplicate_module.__name__,
                context_attrs={},
            ),
            LookupError,
            "requires exactly one canonical candidate",
            "found 2",
        )
    finally:
        del sys.modules[duplicate_module.__name__]


def check_col_reduce_candidate() -> tuple[str, str, str]:
    """ColReduce (tcolmax / tcolsum) candidates must lower to one runtime
    ``scf.for`` carrying a VL-wide accumulator as a ``vreg`` iter_arg — mirroring
    the pto-isa ``TColReduceInstr_NoPostUpdate`` repeat loop — and must NOT
    statically unroll one merge per row.

    Each candidate runs over a single-VL-block column tile: src is
    [rows, VL] row-major, dst is [1, VL] row-major (the surviving column axis).
    """
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }

    colmax = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vecscope" not in colmax, "colmax template must remain scope-free")
    expect(colmax.count("scf.for") == 1, "colmax should emit one runtime reduce loop")
    expect(colmax.count("scf.yield") == 1, "colmax should yield the merged accumulator")
    expect(
        "iter_args" in colmax and "!pto.vmi.vreg<64xf32>" in colmax,
        "colmax should carry a VL-wide vreg accumulator through the loop",
    )
    expect(colmax.count("pto.vmi.vmax") == 1, "colmax should issue one VMI max inside the loop")
    expect(
        colmax.count("pto.vmi.vload") == 1,
        "colmax should load exactly one row per iteration inside the loop "
        "(the accumulator seed is a vbr of the identity, not a dummy vload — a "
        "vload carries a Read memory effect and cannot be DCE'd, so a dummy "
        "load would duplicate the row-0 read)",
    )
    expect(colmax.count("pto.vmi.vstore") == 1, "colmax should store the reduced result once")
    expect("pto.vmi.vcmax" not in colmax, "colmax must not collapse to a 1-lane vcmax")
    expect("pto.vmi.vreduce_max" not in colmax, "colmax must not collapse to a 1-lane vreduce")

    colsum = instantiate_candidate(
        target="a5",
        op_name="pto.tcolsum",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(colsum.count("scf.for") == 1, "colsum should emit one runtime reduce loop")
    expect(
        "iter_args" in colsum and "!pto.vmi.vreg<64xf32>" in colsum,
        "colsum should carry a VL-wide vreg accumulator through the loop",
    )
    expect(colsum.count("pto.vmi.vadd") == 1, "colsum should issue one VMI add inside the loop")

    # A non-binary colsum must not accept the binary 3-operand form (it has no
    # fallback path); the two-operand form is the only supported lowering.
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcolsum",
            operand_specs=[col_tile_spec, reduced_col_spec, reduced_col_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "expects 2 operands, got 3",
    )
    return colmax, colsum, reduced_col_spec


def check_col_expand_candidate() -> None:
    """ColExpandBinary (tcolexpandsub/add/mul/div) broadcasts a [1, VL] column
    result across every row of a [rows, VL] tile, mirroring pto-isa
    ``TColExpandBinOp`` (reload the same VL block per row, not a 1-lane vbrc).
    """
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }
    binops = {
        "pto.tcolexpandsub": "pto.vmi.vsub",
        "pto.tcolexpandadd": "pto.vmi.vadd",
        "pto.tcolexpandmul": "pto.vmi.vmul",
        "pto.tcolexpanddiv": "pto.vmi.vdiv",
    }
    for op_name, expected_op in binops.items():
        # tcolexpanddiv is the only ColExpandBinary op that ExpandTileOp
        # decorates with a `precisionType` context attr (even at default). Real
        # TileOp -> PTODSL VMI provider selection passes that attr; a candidate
        # that didn't declare it under context_constraints would be rejected by
        # validate_context_attrs. Instantiate with the real attr here so the
        # candidate is exercised through the same path.
        ctx_attrs = (
            {"precisionType": "default"} if op_name == "pto.tcolexpanddiv" else {}
        )
        text = instantiate_candidate(
            target="a5",
            op_name=op_name,
            operand_specs=[col_tile_spec, reduced_col_spec, col_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs=ctx_attrs,
        ).mlir_text()
        expect(text.count("scf.for") == 1, f"{op_name} should emit one runtime row loop")
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")
        expect("pto.vmi.vbrc" not in text, f"{op_name} must reload the VL block, not 1-lane vbrc")
        expect(
            text.count("pto.vmi.vload") == 2,
            f"{op_name} should load one source row plus the broadcast VL block",
        )
        # The broadcast VL block is loop-invariant (col_values is [1, VL]); it
        # must be hoisted out of the row loop so a later mem2reg can forward the
        # ColMax result straight to the consumer without a per-row reload. So
        # exactly one vload precedes scf.for (the broadcast) and one sits inside
        # (the source row).
        for_pos = text.find("scf.for")
        expect(
            for_pos > 0 and text[:for_pos].count("pto.vmi.vload") == 1,
            f"{op_name} should hoist the broadcast vload out of the row loop",
        )
        expect(
            text[for_pos:].count("pto.vmi.vload") == 1,
            f"{op_name} should keep only the source-row vload inside the loop",
        )


def check_tcvt_bf16_candidate() -> None:
    """tcvt covers one-VL bf16 widening and f32 narrowing."""
    raw_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    f16_dst_spec = {**raw_tile_spec, "dtype": "f16"}
    bf16_dst_spec = {**raw_tile_spec, "dtype": "bf16"}
    bf16_src_spec = {**raw_tile_spec, "dtype": "bf16"}
    context_attrs = {"round_mode": "RINT", "saturation_mode": "OFF"}
    f16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, f16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs=context_attrs,
    ).mlir_text()
    expect("pto.vmi.vcvt" in f16_text, "tcvt f32->f16 should lower to VMI conversion")
    expect("vreg<64xf16>" in f16_text, "tcvt f32->f16 should target the f16 vreg type")

    bf16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, bf16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs=context_attrs,
    ).mlir_text()
    expect("pto.vmi.vcvt" in bf16_text, "tcvt f32->bf16 should lower to VMI conversion")
    expect("vreg<64xbf16>" in bf16_text, "tcvt f32->bf16 should target the bf16 vreg type")

    widening_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[bf16_src_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs=context_attrs,
    ).mlir_text()
    widening_line = next(
        line for line in widening_text.splitlines() if "pto.vmi.vcvt" in line
    )
    expect("vreg<64xbf16>" in widening_line, "bf16 widening should read 64 logical lanes")
    expect("vreg<64xf32>" in widening_line, "bf16 widening should produce one f32 VL")
    expect("rounding" not in widening_line, "bf16 widening must not carry rounding")
    expect("saturate" not in widening_line, "bf16 widening must not carry saturation")

    wide_bf16 = {
        **bf16_src_spec,
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    wide_f32 = {
        **raw_tile_spec,
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[wide_bf16, wide_f32],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs=context_attrs,
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )
    tail_bf16 = {**bf16_src_spec, "valid_shape": [32, 32]}
    tail_f32 = {**raw_tile_spec, "valid_shape": [32, 32]}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[tail_bf16, tail_f32],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs=context_attrs,
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[bf16_src_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"round_mode": "RINT", "saturation_mode": "ON"},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )


def check_texpands_candidate() -> None:
    scalar_spec = {"kind": "scalar", "dtype": "f32", "value": 0.0}
    dst_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [1, 64],
        "valid_shape": [1, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    text = instantiate_candidate(
        target="a5",
        op_name="pto.texpands",
        operand_specs=[scalar_spec, dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vbrc" in text, "texpands should broadcast its scalar in VMI")
    expect("!pto.vmi.vreg<64xf32>" in text, "texpands should create one f32 VL")
    expect(text.count("pto.vmi.vstore") == 1, "standalone texpands must retain its store")
    expect("scf.for" not in text, "one-VL texpands should not require a loop")

    for invalid_spec in (
        {**dst_spec, "shape": [2, 64], "valid_shape": [2, 64]},
        {**dst_spec, "valid_shape": [1, 32]},
        {**dst_spec, "dtype": "f16"},
        {
            **dst_spec,
            "config": {**dst_spec["config"], "b_layout": "col_major"},
        },
    ):
        invalid_scalar = (
            {**scalar_spec, "dtype": invalid_spec["dtype"]}
            if invalid_spec["dtype"] != "f32"
            else scalar_spec
        )
        expect_raises(
            lambda spec=invalid_spec, scalar=invalid_scalar: instantiate_candidate(
                target="a5",
                op_name="pto.texpands",
                operand_specs=[scalar, spec],
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={},
            ),
            LookupError,
            "no legal PTODSL VMI candidate",
        )


def check_trowexpanddiv_candidate() -> None:
    """RMSNorm's compact one-scalar-per-row broadcast is the VMI form."""
    tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [1, 64],
        "valid_shape": [1, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    denominator_spec = {
        **tile_spec,
        "shape": [8, 1],
        "valid_shape": [1, 1],
        "config": {**tile_spec["config"], "b_layout": "col_major"},
    }
    text = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpanddiv",
        operand_specs=[tile_spec, denominator_spec, tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect(vmi_trowexpanddiv.name == "vmi_trowexpanddiv", "candidate must be exported")
    expect(text.count("pto.vmi.vload") == 2, "candidate must load numerator and scalar")
    expect('dist_mode = "brc"' in text, "denominator must use one-scalar BRC load")
    expect(text.count("pto.vmi.vdiv") == 1, "candidate must issue one default divide")
    expect(text.count("pto.vmi.vstore") == 1, "standalone candidate must retain store")
    expect(text.count("scf.for") == 1, "candidate must preserve one logical row loop")

    for rows in (8, 64):
        row_tile_spec = {
            **tile_spec,
            "shape": [rows, 64],
            "valid_shape": [rows, 64],
        }
        row_denominator_spec = {
            **denominator_spec,
            "shape": [max(rows, 8), 1],
            "valid_shape": [rows, 1],
        }
        row_text = instantiate_candidate(
            target="a5",
            op_name="pto.trowexpanddiv",
            operand_specs=[row_tile_spec, row_denominator_spec, row_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "default"},
        ).mlir_text()
        expect(row_text.count("scf.for") == 1, f"N={rows} must use one row loop")
        expect(
            row_text.count("pto.vmi.vload") == 2
            and row_text.count("pto.vmi.vdiv") == 1
            and row_text.count("pto.vmi.vstore") == 1,
            f"N={rows} row loop must contain one load/div/store instruction body",
        )

    invalid_specs = [
        {**tile_spec, "shape": [1, 128], "valid_shape": [1, 128]},
        {**tile_spec, "valid_shape": [1, 63]},
        {
            **denominator_spec,
            "config": {**denominator_spec["config"], "b_layout": "row_major"},
        },
        {**denominator_spec, "valid_shape": [2, 1]},
        {
            **denominator_spec,
            "shape": [8, 1],
            "valid_shape": [8, 1],
        },
    ]
    for invalid_spec in invalid_specs[:2]:
        expect_raises(
            lambda spec=invalid_spec: instantiate_candidate(
                target="a5",
                op_name="pto.trowexpanddiv",
                operand_specs=[spec, denominator_spec, spec],
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={"precisionType": "default"},
            ),
            LookupError,
            "no legal PTODSL VMI candidate",
        )
    for invalid_denominator in invalid_specs[2:4]:
        expect_raises(
            lambda spec=invalid_denominator: instantiate_candidate(
                target="a5",
                op_name="pto.trowexpanddiv",
                operand_specs=[tile_spec, spec, tile_spec],
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={"precisionType": "default"},
            ),
            LookupError,
            "no legal PTODSL VMI candidate",
        )
    multirow_tile = {**tile_spec, "shape": [9, 64], "valid_shape": [9, 64]}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.trowexpanddiv",
            operand_specs=[multirow_tile, invalid_specs[4], multirow_tile],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "default"},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.trowexpanddiv",
            operand_specs=[tile_spec, denominator_spec, tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high_precision"},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )


def check_col_reduce_vmi_to_vpto_lowering() -> None:
    """The vreg-carrying ColReduce loop must survive VMI->VPTO lowering as a
    real physical ``scf.for iter_args(%acc = ...) -> !pto.vreg<...>`` (the seed
    loaded once before the loop, one vlds+vmax per iteration), proving the
    pto-isa reduce loop shape reaches the physical layer."""
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }
    colmax = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    colsum = instantiate_candidate(
        target="a5",
        op_name="pto.tcolsum",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    check_vmi_to_vpto_lowering("vmi_tcolmax", colmax, "pto.vmax")
    check_vmi_to_vpto_lowering("vmi_tcolsum", colsum, "pto.vadd")


def check_legacy_vpto_compatibility() -> None:
    spec = TileSpec(TILE_SHAPE, f32)
    artifact = legacy_vpto_tadd.specialize(src0=spec, src1=spec, dst=spec)
    artifact.verify()
    text = artifact.mlir_text()
    expect(text.count("scf.for") == 2, "legacy VPTO template should retain its two loops")
    expect("pto.vlds" in text, "legacy VPTO template should still emit vlds")
    expect("pto.vadd" in text, "legacy VPTO template should still emit vadd")
    expect("pto.vsts" in text, "legacy VPTO template should still emit vsts")


def check_vmi_to_vpto_lowering(name: str, mlir_text: str, expected_op: str) -> None:
    ptoas = shutil.which("ptoas")
    expect(ptoas is not None, "ptoas must be available for VMI-to-VPTO regression coverage")
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"{name}.pto"
        input_path.write_text(mlir_text, encoding="utf-8")
        completed = subprocess.run(
            [
                ptoas,
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--enable-vmi",
                "--vmi-fusion-mode=off",
                "--emit-vpto",
                str(input_path),
                "-o",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    expect(
        completed.returncode == 0,
        f"VMI-to-VPTO lowering failed for {name}:\n{completed.stderr}",
    )
    expect("pto.vmi." not in completed.stdout, f"{name} should contain no VMI ops after lowering")
    expect(expected_op in completed.stdout, f"{name} should lower to {expected_op}")
    expect(completed.stdout.count("scf.for") == 1, f"{name} should preserve one flat loop")


def main() -> None:
    check_canonical_block_map()
    check_legacy_vpto_compatibility()
    check_provider_helper()
    tadd_text, wide_tadd_text, texp_text = check_candidate_ir()
    check_vmi_to_vpto_lowering("vmi_tadd_block64", tadd_text, "pto.vadd")
    check_vmi_to_vpto_lowering("vmi_tadd_block128", wide_tadd_text, "pto.vadd")
    check_vmi_to_vpto_lowering("vmi_texp_block64", texp_text, "pto.vexp")
    check_col_reduce_candidate()
    check_col_expand_candidate()
    check_tcvt_bf16_candidate()
    check_texpands_candidate()
    check_trowexpanddiv_candidate()
    check_col_reduce_vmi_to_vpto_lowering()
    print("ptodsl_vmi_tile_template: PASS")


if __name__ == "__main__":
    main()
