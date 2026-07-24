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
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    LogicalRowMap,
    Tile,
    TileSpec,
    bf16,
    f16,
    f32,
    tile_template,
)
from ptodsl.tilelib.registry import TileTemplateRegistry
from ptodsl.vmi_tilelib import (
    VMI_TILELIB_REGISTRY,
    vmi_tadd_block64,
    vmi_tcolmax,
    vmi_tcolsum,
    vmi_tcolexpandsub,
    vmi_tcvt,
    vmi_texp_block64,
)
from ptodsl.vmi_tilelib_helper import instantiate_candidate


TILE_SHAPE = (32, 64)


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


def check_logical_row_map() -> None:
    block_map = LogicalRowMap(TILE_SHAPE, logical_lanes=64)
    expect(block_map.blocks_per_row == 1, "[32,64]xf32 should contain one block per row")
    expect(block_map.logical_block_count == 32, "[32,64]xf32 should contain 32 blocks")

    coordinate = block_map.coordinate(17)
    expect(coordinate.row == 17, "logical block 17 should map directly to row 17")
    expect(coordinate.block_in_row == 0, "each row should contain only block 0")
    expect(coordinate.col_start == 0, "the row-local block should start at column 0")
    expect(coordinate.linear_offset == 1088, "logical block 17 should start at offset 1088")
    expect(coordinate.active_lanes == 64, "the f32 contract should activate 64 lanes")

    wide_map = LogicalRowMap((32, 128), logical_lanes=128)
    expect(wide_map.blocks_per_row == 1, "a wide row must remain one logical block")
    expect(wide_map.logical_block_count == 32, "wide rows must iterate only over rows")
    expect(wide_map.coordinate(17).linear_offset == 2176, "wide row offsets must use 128 lanes")
    very_wide_map = LogicalRowMap((1, 1024), logical_lanes=1024)
    expect(
        very_wide_map.logical_block_count == 1
        and very_wide_map.blocks_per_row == 1,
        "a 1024-lane DSv4 row must remain one logical block",
    )
    expect(
        CanonicalBlockMap is LogicalRowMap,
        "CanonicalBlockMap must remain a compatibility alias",
    )
    expect_raises(
        lambda: CanonicalBlockMap((32, 32), logical_lanes=64),
        ValueError,
        "logical_lanes to equal the tile inner extent",
    )


def check_candidate_ir() -> tuple[str, str]:
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

    texp = specialize_texp()
    texp.verify()
    texp_text = texp.mlir_text()
    expect("pto.vecscope" not in texp_text, "VMI templates must remain scope-free")
    expect(texp_text.count("scf.for") == 1, "texp candidate should contain one flat loop")
    expect(texp_text.count("pto.vmi.vload") == 1, "texp should issue one VMI load")
    expect(texp_text.count("pto.vmi.vexp") == 1, "texp should issue one VMI exp")
    expect(texp_text.count("pto.vmi.vstore") == 1, "texp should issue one VMI store")

    f16_tadd = specialize_tadd(dtype=f16, shape=(32, 128)).mlir_text()
    expect(
        "!pto.vmi.vreg<128xf16>" in f16_tadd,
        "f16 tadd should use its 128-lane logical row",
    )
    wide_texp = specialize_texp(shape=(32, 128)).mlir_text()
    expect(
        wide_texp.count("scf.for") == 1
        and "!pto.vmi.vreg<128xf32>" in wide_texp,
        "wide f32 texp should remain one row loop over a 128-lane logical vreg",
    )
    return tadd_text, texp_text


def check_provider_helper() -> dict[str, str]:
    b1_artifacts = {}
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

    compact_tile_spec = {
        **raw_tile_spec,
        "shape": [1, 32],
        "valid_shape": [1, 32],
    }
    compact_tadd = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[compact_tile_spec, compact_tile_spec, compact_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        compact_tadd.count("scf.for") == 1
        and "!pto.vmi.vreg<32xf32>" in compact_tadd,
        "compact tadd should use one 32-lane logical row",
    )

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

    tdivs = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vbrc" in tdivs, "tdivs should broadcast its scalar operand")
    expect("pto.vmi.vdiv" in tdivs, "tdivs should lower to VMI vector divide")
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tdivs",
            operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high_precision"},
        ),
        LookupError,
        "no legal PTODSL VMI semantic form",
        "precisionType",
    )

    scalar_lhs_tdivs = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[scalar_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect(
        "pto.vmi.vdiv" in scalar_lhs_tdivs,
        "scalar/tile tdivs should select its scalar_tile semantic form",
    )
    b1_artifacts["vmi_tdivs_scalar_tile"] = scalar_lhs_tdivs

    base_op_expectations = {
        "tdiv": "pto.vmi.vdiv",
        "tmin": "pto.vmi.vmin",
        "tabs": "pto.vmi.vabs",
        "tneg": "pto.vmi.vneg",
    }
    for op_name, expected_op in base_op_expectations.items():
        text = instantiate_candidate(
            target="a5",
            op_name=f"pto.{op_name}",
            operand_specs=(
                [raw_tile_spec, raw_tile_spec, raw_tile_spec]
                if op_name in {"tdiv", "tmin"}
                else [raw_tile_spec, raw_tile_spec]
            ),
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "default"} if op_name == "tdiv" else {},
        ).mlir_text()
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")
        b1_artifacts[f"vmi_{op_name}"] = text

    tsubs = instantiate_candidate(
        target="a5",
        op_name="pto.tsubs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vbrc" in tsubs, "tsubs should broadcast its scalar operand")
    expect("pto.vmi.vsub" in tsubs, "tsubs should lower to VMI subtract")
    b1_artifacts["vmi_tsubs"] = tsubs

    texpands = instantiate_candidate(
        target="a5",
        op_name="pto.texpands",
        operand_specs=[scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vbrc" in texpands, "texpands should broadcast its scalar")
    expect("pto.vmi.vstore" in texpands, "texpands should store one logical vector")
    b1_artifacts["vmi_texpands"] = texpands

    f16_tile_spec = {
        **raw_tile_spec,
        "dtype": "f16",
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    f16_tadd = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[f16_tile_spec, f16_tile_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "!pto.vmi.vreg<128xf16>" in f16_tadd,
        "provider should select the f16 native-VL form",
    )
    b1_artifacts["vmi_tadd_f16"] = f16_tadd

    dtype_lanes = {
        "f32": 64,
        "f16": 128,
        "bf16": 128,
        "i32": 64,
        "i16": 128,
        "i8": 256,
    }
    generic_forms = {
        "tadd": ("binary", set(dtype_lanes)),
        "tsub": ("binary", {"f32", "f16", "i32", "i16", "i8"}),
        "tmul": ("binary", {"f32", "f16", "i32", "i16"}),
        "tmov": ("unary", set(dtype_lanes)),
        "tadds": ("scalar", set(dtype_lanes)),
        "tmuls": ("scalar", set(dtype_lanes)),
        "tsubs": ("scalar", set(dtype_lanes)),
        "texpands": ("fill", set(dtype_lanes)),
    }
    for dtype, lanes in dtype_lanes.items():
        tile_spec = {
            **raw_tile_spec,
            "dtype": dtype,
            "shape": [2, lanes],
            "valid_shape": [2, lanes],
        }
        typed_scalar_spec = {"kind": "scalar", "dtype": dtype}
        for op_name, (form, supported_dtypes) in generic_forms.items():
            if dtype not in supported_dtypes:
                continue
            form_specs = {
                "binary": [tile_spec, tile_spec, tile_spec],
                "unary": [tile_spec, tile_spec],
                "scalar": [tile_spec, typed_scalar_spec, tile_spec],
                "fill": [typed_scalar_spec, tile_spec],
            }[form]
            text = instantiate_candidate(
                target="a5",
                op_name=f"pto.{op_name}",
                operand_specs=form_specs,
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={},
            ).mlir_text()
            expect(
                f"!pto.vmi.vreg<{lanes}x{dtype}>" in text,
                f"{op_name}/{dtype} should use its dtype-native VL",
            )
            if op_name == "tadd" and dtype not in {"f32", "f16"}:
                b1_artifacts[f"vmi_tadd_{dtype}"] = text

    for op_name, dtype in (("tsub", "bf16"), ("tmul", "bf16"), ("tmul", "i8")):
        lanes = dtype_lanes[dtype]
        tile_spec = {
            **raw_tile_spec,
            "dtype": dtype,
            "shape": [2, lanes],
            "valid_shape": [2, lanes],
        }
        expect_raises(
            lambda op_name=op_name, tile_spec=tile_spec: instantiate_candidate(
                target="a5",
                op_name=f"pto.{op_name}",
                operand_specs=[tile_spec, tile_spec, tile_spec],
                provider_module="ptodsl.vmi_tilelib",
                context_attrs={},
            ).mlir_text(),
            ValueError,
            f"does not support dtype {dtype}",
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

    row_expand = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpandsub",
        operand_specs=[raw_tile_spec, reduced_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vbrc" in row_expand, "row expand should broadcast one value per row")

    f16_tile_spec = {**raw_tile_spec, "dtype": "f16"}
    tcvt = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in tcvt, "tcvt should lower to VMI conversion")

    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.texp",
            operand_specs=[raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high"},
        ),
        ValueError,
        "does not support context attrs",
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
            "requires one canonical candidate per (target, op, semantic_form)",
            "default",
        )
    finally:
        del sys.modules[duplicate_module.__name__]

    ambiguous_module = ModuleType("ptodsl_test_ambiguous_vmi_forms")
    ambiguous_module.VMI_TILELIB_REGISTRY = TileTemplateRegistry()

    @tile_template(
        target="a5",
        op="tadd",
        name="ambiguous_tadd_a",
        ir_level="vmi",
        semantic_form="form_a",
    )
    def ambiguous_tadd_a(src0: Tile, src1: Tile, dst: Tile):
        pass

    @tile_template(
        target="a5",
        op="tadd",
        name="ambiguous_tadd_b",
        ir_level="vmi",
        semantic_form="form_b",
    )
    def ambiguous_tadd_b(src0: Tile, src1: Tile, dst: Tile):
        pass

    ambiguous_module.VMI_TILELIB_REGISTRY.register(ambiguous_tadd_a)
    ambiguous_module.VMI_TILELIB_REGISTRY.register(ambiguous_tadd_b)
    sys.modules[ambiguous_module.__name__] = ambiguous_module
    try:
        expect_raises(
            lambda: instantiate_candidate(
                target="a5",
                op_name="pto.tadd",
                operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
                provider_module=ambiguous_module.__name__,
                context_attrs={},
            ),
            LookupError,
            "ambiguous PTODSL VMI semantic forms",
            "form_a/ambiguous_tadd_a",
            "form_b/ambiguous_tadd_b",
        )
    finally:
        del sys.modules[ambiguous_module.__name__]
    return b1_artifacts


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
        ValueError,
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


def check_tcvt_vmi_contract() -> None:
    """Tile conversion semantics remain explicit in VMI, before VPTO lowering."""
    base = {
        "kind": "tile",
        "shape": [2, 128],
        "valid_shape": [2, 128],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    cases = (
        ("bf16_to_f32", "bf16", "f32", "ROUND", "OFF", None, None),
        ("f32_to_bf16", "f32", "bf16", "RINT", "OFF", "R", "OFF"),
        ("f32_to_f16", "f32", "f16", "ROUND", "ON", "A", "SAT"),
        ("f32_to_i32", "f32", "i32", "TRUNC", "OFF", "Z", "OFF"),
        ("i32_to_f16", "i32", "f16", "ROUND", "OFF", "A", "OFF"),
        ("f16_to_i8", "f16", "i8", "TRUNC", "OFF", "Z", "OFF"),
    )
    for name, src_dtype, dst_dtype, round_mode, sat_mode, rounding, saturate in cases:
        text = instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[{**base, "dtype": src_dtype}, {**base, "dtype": dst_dtype}],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"round_mode": round_mode, "sat_mode": sat_mode},
        ).mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should emit one logical row loop")
        expect(
            f"!pto.vmi.vreg<128x{src_dtype}>" in text
            and f"!pto.vmi.vreg<128x{dst_dtype}>" in text,
            f"{name} should preserve 128 logical lanes",
        )
        expect("pto.vmi.vcvt" in text, f"{name} should emit VMI conversion")
        if rounding is None:
            expect("rounding =" not in text, f"{name} should not invent rounding control")
            expect("saturate =" not in text, f"{name} should not invent saturation control")
        else:
            expect(f'rounding = "{rounding}"' in text, f"{name} should preserve rounding")
            expect(f'saturate = "{saturate}"' in text, f"{name} should preserve saturation")


def check_compact_logical_vregs() -> None:
    """Compact states remain legal VMI values without assuming VPTO lowering."""
    row = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [8, 64],
        "valid_shape": [8, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    compact_row = {**row, "shape": [1, 8], "valid_shape": [1, 8]}
    compact_column = {
        **row,
        "shape": [8, 1],
        "valid_shape": [8, 1],
        "config": {**row["config"], "b_layout": "col_major"},
    }

    sqrt = instantiate_candidate(
        target="a5",
        op_name="pto.tsqrt",
        operand_specs=[compact_row, compact_row],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect(sqrt.count("scf.for") == 1, "tsqrt should emit one row loop")
    expect("!pto.vmi.vreg<8xf32>" in sqrt, "tsqrt should preserve eight logical lanes")
    expect("pto.vmi.vsqrt" in sqrt, "tsqrt should emit VMI sqrt")

    recip = instantiate_candidate(
        target="a5",
        op_name="pto.trecip",
        operand_specs=[compact_row, compact_row],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect(recip.count("scf.for") == 1, "trecip should emit one row loop")
    expect("pto.vmi.vbrc" in recip and "pto.vmi.vdiv" in recip,
           "trecip should express one over the logical row")

    expanded = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpandmul",
        operand_specs=[row, compact_column, row],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(expanded.count("scf.for") == 1, "row expand should emit one row loop")
    expect("!pto.vmi.vreg<1xf32>" in expanded, "row expand should retain scalar state")
    expect("pto.vmi.vbrc" in expanded, "row expand should broadcast scalar state")

    tmp = {**compact_row, "shape": [1, 8], "valid_shape": [1, 8]}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.trsqrt",
            operand_specs=[row, row, tmp],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high_precision"},
        ).mlir_text(),
        LookupError,
        "no legal PTODSL VMI semantic form",
        "does not support context attrs",
    )


def main() -> None:
    check_logical_row_map()
    check_provider_helper()
    check_candidate_ir()
    check_col_reduce_candidate()
    check_col_expand_candidate()
    check_tcvt_vmi_contract()
    check_compact_logical_vregs()
    print("ptodsl_vmi_tile_template: PASS")


if __name__ == "__main__":
    main()
