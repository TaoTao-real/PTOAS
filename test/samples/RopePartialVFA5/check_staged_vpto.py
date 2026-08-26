#!/usr/bin/env python3
"""Validate the manual-VF-aligned GM/UB data flow in final RoPE VPTO."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXPECTED = {
    "half": {"pto.vlds": 9, "pto.vsts": 3},
    "interleave": {
        "pto.vlds": 4,
        "pto.vsts": 2,
        "pto.vdintlv": 1,
        "pto.vintlv": 1,
    },
}


def count_op(text: str, op: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z0-9_.]){re.escape(op)}\b", text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(EXPECTED), required=True)
    parser.add_argument("vpto", type=Path)
    args = parser.parse_args()

    text = args.vpto.read_text(encoding="utf-8")
    expected = {
        "pto.copy_gm_to_ubuf": 3,
        "pto.copy_ubuf_to_gm": 1,
        "pto.mem_bar": 0,
        **EXPECTED[args.mode],
    }
    errors: list[str] = []
    for op, want in expected.items():
        got = count_op(text, op)
        if got != want:
            errors.append(f"{op}: expected {want}, got {got}")

    first_loop = text.find("scf.for")
    gm_loads = [m.start() for m in re.finditer(r"\bpto\.copy_gm_to_ubuf\b", text)]
    gm_stores = [m.start() for m in re.finditer(r"\bpto\.copy_ubuf_to_gm\b", text)]
    last_vstore = text.rfind("pto.vsts")
    if first_loop < 0:
        errors.append("missing row/prefix scf.for loops")
    elif any(pos > first_loop for pos in gm_loads):
        errors.append("GM->UB copy remains inside/after vector row computation")
    if gm_stores and last_vstore >= 0 and any(pos < last_vstore for pos in gm_stores):
        errors.append("UB->GM copy occurs before the final vector write")

    residual_vmi = re.findall(
        r"^\s*(?:%[A-Za-z0-9_.$-]+\s*=\s*)?vmi\.[A-Za-z0-9_.$-]+\b",
        text,
        flags=re.MULTILINE,
    )
    if residual_vmi:
        errors.append(f"residual VMI operations: {len(residual_vmi)}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    counts = ", ".join(f"{op}={want}" for op, want in expected.items())
    print(f"PASS mode={args.mode}: {counts}, residual_vmi=0, bulk_dma_order=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
