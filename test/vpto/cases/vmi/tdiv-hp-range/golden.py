#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Inputs and golden for tdiv with precisionType = high_precision (VMI route).

The band values come from the repository's own pto.tdiv ST data
(test/tilelang_st/npu/a5/src/st/testcase/tdiv/gen_data.py, the `overflow` and
`precision_sensitive` patterns); the golden is numpy f32 division, i.e. plain
IEEE 754, which is what compare.py holds the VMI soft-div body to.

The deliberately excluded pattern is that generator's `subnormal` one: the
ieee754 body detects a subnormal dividend with an equality against
0x007FFFFF and a subnormal divisor with a less-than, an inherited asymmetry
the ST generator itself documents.  Feeding a subnormal dividend therefore
measures that known asymmetry, not the exponent arithmetic this case exists to
check, so every value below is a normal f32.

Rows (16x32 f32, see kernel.pto):
   0- 4 ordinary range, mixed signs                -> ULP compare
   5- 7 quotient underflow, both operands normal   -> must be +-0
   8-10 quotient overflow                          -> must be +-inf
  11-13 quotient underflow, sign combinations      -> must be +-0
  14-15 ratios that are not representable in f32   -> ULP compare
"""

import argparse
from pathlib import Path

import numpy as np

ROWS = 16
COLS = 32

TINY = np.float32(1e-30)        # normal f32
LARGE = np.float32(1e30)        # normal f32
OVERFLOW_TRIGGER = np.float32(1e38)
MAX_NORMAL = np.float32(3.4e38)
SMALL_NORMAL = np.float32(1e-25)


def generate(output_dir: Path) -> None:
    rng = np.random.RandomState(20260928)
    a = np.zeros((ROWS, COLS), dtype=np.float32)
    b = np.ones((ROWS, COLS), dtype=np.float32)

    # rows 0-4: ordinary range, mixed signs
    a[0:5] = rng.uniform(0.1, 100.0, size=(5, COLS)).astype(np.float32)
    b[0:5] = rng.uniform(0.1, 100.0, size=(5, COLS)).astype(np.float32)
    a[0:5] *= rng.choice([-1.0, 1.0], size=(5, COLS)).astype(np.float32)
    b[0:5] *= rng.choice([-1.0, 1.0], size=(5, COLS)).astype(np.float32)

    # rows 5-7: 1e-30 / (1e30 .. f32 max) = 1e-60 .. 3e-69, below the smallest
    # subnormal, so the quotient underflows to zero.  Both operands are normal,
    # so nothing but the quotient exponent decides this band, and a body that
    # misreads scale = exp0 - exp1 + 127 reports the largest quotient the field
    # can spell instead of zero.
    b[5:8] = rng.uniform(LARGE, MAX_NORMAL, size=(3, COLS)).astype(np.float32)
    a[5] = TINY
    a[6] = -TINY
    a[7] = TINY
    b[7] = -b[7]

    # rows 8-10: 1e30 / 1e-25 = 1e55 -> overflow to +-inf
    a[8:11] = LARGE
    b[8:11] = np.float32(1e-25)
    a[9] = -a[9]          # -1e30 / 1e-25  -> -inf
    b[10] = -b[10]        #  1e30 / -1e-25 -> -inf

    # rows 11-13: 1e-25 / (1e32 .. f32 max) -> underflow, all sign pairs
    b[11:14] = rng.uniform(np.float32(1e32), MAX_NORMAL, size=(3, COLS)).astype(np.float32)
    a[11:14] = SMALL_NORMAL
    a[12] = -SMALL_NORMAL
    b[13] = -b[13]

    # rows 14-15: ratios with no exact f32 representation (1/3, 1/7, 7/3, ...)
    ratios = [(1.0, 3.0), (1.0, 7.0), (7.0, 3.0), (1.0, 11.0), (5.0, 3.0), (10.0, 3.0)]
    for i, (num, den) in enumerate(ratios):
        a[14, i] = np.float32(num)
        b[14, i] = np.float32(den)
        a[15, i] = np.float32(-num)
        b[15, i] = np.float32(den)
    for i in range(len(ratios), COLS):
        a[14, i] = np.float32(1.0)
        b[14, i] = np.float32(3.0)
        a[15, i] = np.float32(-7.0)
        b[15, i] = np.float32(3.0)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        golden = (a / b).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.tofile(output_dir / "v1.bin")
    b.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v3.bin")
    # Same reference for the default-precision control: both entry points
    # divide the same two tiles, so both are held to IEEE 754.
    golden.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
