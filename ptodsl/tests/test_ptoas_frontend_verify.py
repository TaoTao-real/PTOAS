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
import tempfile
from importlib.util import module_from_spec, spec_from_file_location


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from ptodsl import pto
from ptodsl import scalar


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def resolve_ptoas_binary() -> Path:
    candidates = [
        REPO_ROOT / "build" / "tools" / "ptoas" / "ptoas",
        REPO_ROOT / "install" / "bin" / "ptoas",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    from_path = shutil.which("ptoas")
    if from_path:
        return Path(from_path)

    raise FileNotFoundError("unable to locate a ptoas binary under build/, install/, or PATH")


def run_ptoas_frontend_verify(ptoas_bin: Path, mlir_text: str, label: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
        handle.write(mlir_text)
        input_path = Path(handle.name)

    try:
        result = subprocess.run(
            [str(ptoas_bin), str(input_path), "--emit-pto-ir", "-o", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)

    expect(
        result.returncode == 0,
        f"{label} should pass PTOAS frontend verification.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    expect(result.stdout.strip(), f"{label} should emit non-empty PTO IR after PTOAS frontend passes")
    return result.stdout

@pto.jit(target="a5")
def host_vec_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.simt
def simt_gm_memory_core_body(gm: pto.ptr(pto.i32, "gm")):
    tx = pto.get_tid_x()
    src_idx = scalar.index_cast(tx)
    loaded = scalar.load(gm, src_idx)
    with_bias = loaded + tx + 1000
    scalar.store(with_bias, gm, scalar.index_cast(tx + 32))
    scalar.store(tx, gm, scalar.index_cast(tx + 64))


@pto.jit(target="a5", mode="explicit")
def simt_gm_memory_core_kernel(gm: pto.ptr(pto.i32, "gm")):
    simt_gm_memory_core_body[32, 1, 1](gm)
    pto.pipe_barrier(pto.Pipe.ALL)


def main() -> None:
    ptoas_bin = resolve_ptoas_binary()

    simple_text = host_vec_copy.compile().mlir_text()
    simple_frontend_text = run_ptoas_frontend_verify(
        ptoas_bin,
        simple_text,
        "host_vec_copy PTODSL artifact",
    )
    expect(
        "func.func @host_vec_copy" in simple_frontend_text,
        "host_vec_copy frontend verification output should preserve the kernel symbol",
    )
    expect(
        "pto.tload" in simple_frontend_text and "pto.tstore" in simple_frontend_text,
        "host_vec_copy frontend verification output should keep the tile IO contract visible",
    )

    simt_gm_memory_text = simt_gm_memory_core_kernel.compile().mlir_text()
    simt_frontend_text = run_ptoas_frontend_verify(
        ptoas_bin,
        simt_gm_memory_text,
        "simt_gm_memory_core PTODSL artifact",
    )
    expect(
        "func.func @simt_gm_memory_core_kernel" in simt_frontend_text,
        "simt_gm_memory_core frontend output should preserve the kernel symbol",
    )
    expect(
        "pto.simt_launch @simt_gm_memory_core_body__simt_" in simt_frontend_text,
        "simt_gm_memory_core frontend output should preserve the SIMT launch",
    )
    expect(
        "pto.get_tid_x" in simt_frontend_text,
        "simt_gm_memory_core frontend output should preserve SIMT thread queries",
    )
    expect(
        "pto.load" in simt_frontend_text and simt_frontend_text.count("pto.store") >= 2,
        "simt_gm_memory_core frontend output should preserve GM load/store operations",
    )

    print("ptodsl_ptoas_frontend_verify: PASS")


if __name__ == "__main__":
    main()
