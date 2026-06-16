#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np
import struct

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import result_cmp, style_fail, style_pass

from cases import CASES


def read_value_index_pairs(filepath, dtype, count):
    """Read interleaved (value, index) pairs from file.

    Format: value followed by index (uint32).
    For f16: value (2 bytes) + padding (2 bytes) + index (4 bytes) = 8 bytes per pair.
    For f32: value (4 bytes) + index (4 bytes) = 8 bytes per pair.
    """
    values = []
    indices = []

    struct_fmt = 'fI' if dtype == np.float32 else 'e2xI'
    struct_size = struct.calcsize(struct_fmt)

    with open(filepath, 'rb') as f:
        for _ in range(count):
            data = f.read(struct_size)
            if not data:
                break
            unpacked = struct.unpack(struct_fmt, data)
            values.append(unpacked[0])
            indices.append(unpacked[1])

    return np.array(values, dtype=dtype), np.array(indices, dtype=np.uint32)


def handle_output_data(golden_vals, golden_idx, output_vals, output_idx):
    """Handle exhausted case: zero output values and indices where golden values are 0.

    Following pto-isa HandleOutputData logic:
    - Scan from end, find first non-zero golden value
    - Zero output values where golden values are 0

    Also zero output indices where golden indices are 0 (matching gen_data.py behavior).
    """
    size = len(golden_vals)
    i = size - 1
    while i > 0:
        if golden_vals[i] == 0.0:
            output_vals[i] = 0.0
            if golden_idx[i] == 0:
                output_idx[i] = 0
            i -= 1
        else:
            return


def compare_multilist(case):
    """Compare multi-list merge sort output.

    For multi-list format:
    - Read input0.bin, input1.bin, etc.
    - Read output.bin
    - Compare top-k elements with golden.bin
    """
    dtype = case["dtype"]
    list_num = case["list_num"]
    src_cols = case["src_cols"]
    topk = case["topk"]
    exhausted = case.get("exhausted", False)

    # Calculate element divisor
    if dtype == np.float16:
        elem_divisor = 4
    else:
        elem_divisor = 2

    # Total structures to compare
    total_structures = sum(src_cols)

    # Read golden output
    golden_vals, golden_indices = read_value_index_pairs(
        os.path.join(case["name"], "golden.bin"), dtype, total_structures
    )

    # Read actual output
    output_vals, output_indices = read_value_index_pairs(
        os.path.join(case["name"], "output.bin"), dtype, total_structures
    )

    if exhausted:
        handle_output_data(golden_vals, golden_indices, output_vals, output_indices)

    # Compare top-k elements (only compare the valid output)
    vals_ok = result_cmp(golden_vals[:topk], output_vals[:topk], case["eps"])
    indices_ok = np.allclose(golden_indices[:topk], output_indices[:topk], atol=0, rtol=0)

    return vals_ok and indices_ok


def compare_topk(case):
    """Compare TopK output.

    For TopK format:
    - Read input0.bin (unsorted raw data)
    - Read output.bin (top-k sorted data)
    - Compare with golden.bin
    """
    dtype = case["dtype"]
    valid_shape = case["valid_shape"]
    valid_rows, valid_cols = valid_shape
    topk = case["topk"]

    # Get element divisor based on dtype
    if dtype == np.float16:
        elem_divisor = 4
    else:
        elem_divisor = 2

    # Total structures in input
    total_structures = valid_cols // elem_divisor

    # Read golden output
    golden_vals, golden_indices = read_value_index_pairs(
        os.path.join(case["name"], "golden.bin"), dtype, total_structures
    )

    # Read actual output
    output_vals, output_indices = read_value_index_pairs(
        os.path.join(case["name"], "output.bin"), dtype, topk
    )

    # Compare top-k elements
    vals_ok = result_cmp(golden_vals[:topk], output_vals[:topk], case["eps"])
    indices_ok = np.allclose(golden_indices[:topk], output_indices[:topk], atol=0, rtol=0)

    return vals_ok and indices_ok


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        format_type = case.get("format", "single")

        if format_type == "single":
            dtype = case["dtype"]
            valid_shape = case["valid_shape"]
            valid_rows, valid_cols = valid_shape
            block_len = case["block_len"]

            # Get element divisor based on dtype
            if dtype == np.float16:
                elem_divisor = 4
            else:
                elem_divisor = 2

            cols = valid_cols // elem_divisor

            golden_vals, golden_indices = read_value_index_pairs(
                os.path.join(case["name"], "golden.bin"), dtype, cols
            )
            output_vals, output_indices = read_value_index_pairs(
                os.path.join(case["name"], "output.bin"), dtype, cols
            )

            vals_ok = result_cmp(golden_vals, output_vals, case["eps"])
            indices_ok = np.allclose(golden_indices, output_indices, atol=0, rtol=0)

            if vals_ok and indices_ok:
                print(style_pass(f"[INFO] {case['name']}: compare passed"))
            else:
                if not vals_ok:
                    print(style_fail(f"[ERROR] {case['name']}: values mismatch"))
                if not indices_ok:
                    print(style_fail(f"[ERROR] {case['name']}: indices mismatch"))
                all_passed = False

        elif format_type == "multi":
            ok = compare_multilist(case)
            if ok:
                print(style_pass(f"[INFO] {case['name']}: compare passed"))
            else:
                print(style_fail(f"[ERROR] {case['name']}: values or indices mismatch"))
                all_passed = False

        elif format_type == "topk":
            ok = compare_topk(case)
            if ok:
                print(style_pass(f"[INFO] {case['name']}: compare passed"))
            else:
                print(style_fail(f"[ERROR] {case['name']}: values or indices mismatch"))
                all_passed = False

        else:
            print(style_fail(f"[ERROR] {case['name']}: unsupported format {format_type}"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
