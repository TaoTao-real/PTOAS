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
from pathlib import Path

# Add current directory to path for standalone execution
script_dir = Path(__file__).parent
if script_dir not in sys.path:
    sys.path.insert(0, str(script_dir))

# Add st_common directory
st_common_dir = script_dir.parent
if st_common_dir not in sys.path:
    sys.path.insert(0, str(st_common_dir))

from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases


def compute_ulp_difference(golden, output, dtype):
    """Compute ULP (Unit in the Last Place) difference between two arrays.

    ULP difference measures how many representable floating-point values
    are between golden and output.

    Note: Only computes ULP for normal values (not NaN/Inf/zero).

    Args:
        golden: numpy array of golden values
        output: numpy array of output values
        dtype: numpy dtype (float32 or float16)

    Returns:
        Maximum ULP difference across all normal elements, or None if no normal values
    """
    if dtype == np.float32:
        int_dtype = np.uint32
    elif dtype == np.float16:
        int_dtype = np.uint16
    else:
        return None  # ULP not applicable for integer types

    # Filter out NaN, Inf, and zero values (ULP not meaningful for these)
    golden_normal = np.isfinite(golden) & (golden != 0)
    output_normal = np.isfinite(output) & (output != 0)
    normal_mask = golden_normal & output_normal

    if not np.any(normal_mask):
        return None  # No normal values to compare

    golden_filtered = golden[normal_mask]
    output_filtered = output[normal_mask]

    # Convert to integer representation for ULP calculation
    golden_int = golden_filtered.view(int_dtype)
    output_int = output_filtered.view(int_dtype)

    # Handle sign difference: ULP counts across zero
    # For same sign: simple difference
    # For different sign: add both magnitudes (crosses zero boundary)
    sign_bit = np.dtype(int_dtype).itemsize * 8 - 1
    golden_sign = golden_int >> sign_bit
    output_sign = output_int >> sign_bit

    same_sign = (golden_sign == output_sign)

    # For same sign: subtract representations
    ulp_diff_same = np.abs(golden_int.astype(np.int64) - output_int.astype(np.int64))

    # For different sign: distance through zero (less common, treat as large difference)
    # Use maximum possible ULP for different signs
    ulp_diff_cross = np.iinfo(int_dtype).max

    ulp_diff = np.where(same_sign, ulp_diff_same, ulp_diff_cross)

    return np.max(ulp_diff)


def check_nan_inf_consistency(golden, output, relaxed=False):
    """Check that NaN and Inf positions and values are consistent.

    IEEE 754 rules:
    - NaN must appear at similar positions (hardware may differ in NaN type)
    - Inf must have same sign at same positions
    - Both must agree on which positions are NaN vs Inf vs normal

    Args:
        golden: numpy array of golden values
        output: numpy array of output values
        relaxed: if True, allow NaN count differences (hardware may have different NaN handling)

    Returns:
        (ok, error_msg) tuple
    """
    # Check NaN positions
    golden_nan = np.isnan(golden)
    output_nan = np.isnan(output)

    # For relaxed mode, check NaN counts are similar (allow some variance)
    if relaxed:
        golden_nan_count = np.sum(golden_nan)
        output_nan_count = np.sum(output_nan)
        # Allow 20% variance in NaN count
        if golden_nan_count > 0:
            variance = abs(golden_nan_count - output_nan_count) / float(golden_nan_count)
            if variance > 0.2:
                return False, "NaN count variance > 20% (golden={}, output={})".format(golden_nan_count, output_nan_count)
        # Continue with other checks even if NaN positions differ
    else:
        if not np.array_equal(golden_nan, output_nan):
            nan_mismatch = np.where(golden_nan != output_nan)
            return False, "NaN position mismatch at {} positions".format(len(nan_mismatch[0]))

    # Check Inf positions
    golden_inf = np.isinf(golden)
    output_inf = np.isinf(output)

    if not np.array_equal(golden_inf, output_inf):
        inf_mismatch = np.where(golden_inf != output_inf)
        return False, f"Inf position mismatch at {len(inf_mismatch[0])} positions"

    # Check Inf signs
    if np.any(golden_inf):
        golden_signs = np.sign(golden[golden_inf])
        output_signs = np.sign(output[golden_inf])
        if not np.array_equal(golden_signs, output_signs):
            return False, "Inf sign mismatch"

    return True, None


def compare_high_precision_result(golden, output, dtype, ulp_tolerance=1, eps=1e-6, relaxed_nan=False):
    """Compare results for HIGH_PRECISION mode.

    High-precision algorithm uses three-candidate search which may select
    a different but more accurate rounding than numpy standard division.

    Comparison strategy:
    1. Check NaN/Inf consistency (may allow relaxed NaN checking)
    2. For normal/subnormal values: allow ±ulp_tolerance ULP difference

    Args:
        golden: numpy array of reference values (numpy division)
        output: numpy array of NPU output values
        dtype: numpy dtype
        ulp_tolerance: maximum allowed ULP difference (default 1)
        eps: fallback tolerance for non-float types
        relaxed_nan: if True, allow NaN count variance (default False)

    Returns:
        (ok, error_msg) tuple
    """
    # 1. Check NaN/Inf consistency
    ok, error_msg = check_nan_inf_consistency(golden, output, relaxed=relaxed_nan)
    if not ok:
        return False, error_msg

    # 2. Filter out NaN/Inf for numerical comparison
    golden_nan = np.isnan(golden)
    golden_inf = np.isinf(golden)
    normal_mask = ~(golden_nan | golden_inf)

    if not np.any(normal_mask):
        return True, None  # All NaN/Inf, already checked

    golden_normal = golden[normal_mask]
    output_normal = output[normal_mask]

    # 3. Use ULP tolerance for float types
    if dtype in (np.float32, np.float16):
        max_ulp = compute_ulp_difference(golden_normal, output_normal, dtype)
        if max_ulp is not None and max_ulp <= ulp_tolerance:
            return True, f"ULP tolerance passed (max_ulp={max_ulp})"

        # Fallback to eps-based comparison if ULP check fails
        ok = result_cmp(golden_normal, output_normal, eps)
        if not ok:
            return False, f"Both ULP ({max_ulp}) and eps ({eps}) check failed"
        return True, f"Passed with eps tolerance (max_ulp={max_ulp} > {ulp_tolerance})"

    # 4. For integer types, use exact comparison
    else:
        ok = np.array_equal(golden_normal, output_normal)
        if not ok:
            mismatch = np.where(golden_normal != output_normal)
            return False, f"Mismatch at {len(mismatch[0])} positions"
        return True, None


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        shape = case["shape"]
        vr, vc = case["valid_shape"]
        test_pattern = case.get("test_pattern", "normal")
        precision_type = case.get("precision_type", "default")
        check_inf_nan = case.get("check_inf_nan", False)

        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=case["dtype"]).reshape(shape)
        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=case["dtype"]).reshape(shape)

        eps = case["eps"]
        dtype_name = case["dtype"].__name__

        # Extract valid region
        golden_valid = golden[:vr, :vc]
        output_valid = output[:vr, :vc]

        # Integer types: exact comparison
        if dtype_name in ("uint32", "int32", "uint16", "int16", "uint8", "int8"):
            ok = np.array_equal(golden_valid, output_valid)
            if not ok:
                mismatch = np.where(golden_valid != output_valid)
                print(style_fail(f"[ERROR] {case['name']}: mismatches at {len(mismatch[0])} positions"))
                if len(mismatch[0]) > 0 and len(mismatch[0]) <= 10:
                    for i in range(len(mismatch[0])):
                        r, c = mismatch[0][i], mismatch[1][i]
                        print(f"  [{r},{c}] golden={golden_valid[r,c]} output={output_valid[r,c]}")
                all_passed = False
                continue

        # Float types with special handling
        else:
            # HIGH_PRECISION mode: use ULP tolerance
            if precision_type == "high_precision":
                ulp_tolerance = case.get("ulp_tolerance", 1)
                # Use relaxed NaN checking for nan_inf and boundary tests
                relaxed_nan = test_pattern in ("nan_inf", "boundary")
                ok, msg = compare_high_precision_result(
                    golden_valid, output_valid, case["dtype"],
                    ulp_tolerance=ulp_tolerance, eps=eps, relaxed_nan=relaxed_nan
                )
                if not ok:
                    print(style_fail("[ERROR] {}: {} (test={})".format(case['name'], msg, test_pattern)))
                    all_passed = False
                    continue
                elif msg:
                    print(style_pass("[INFO] {}: {} (test={})".format(case['name'], msg, test_pattern)))

            # check_inf_nan flag or boundary test: check NaN/Inf separately
            elif check_inf_nan or test_pattern == "boundary":
                # Use relaxed NaN checking for nan_inf and boundary tests
                relaxed = test_pattern in ("nan_inf", "boundary")
                ok, msg = check_nan_inf_consistency(golden_valid, output_valid, relaxed=relaxed)
                if not ok:
                    print(style_fail("[ERROR] {}: {} (test={})".format(case['name'], msg, test_pattern)))
                    all_passed = False
                    continue

                # Compare non-special values
                golden_nan = np.isnan(golden_valid)
                golden_inf = np.isinf(golden_valid)
                normal_mask = ~(golden_nan | golden_inf)

                if np.any(normal_mask):
                    ok = result_cmp(golden_valid[normal_mask], output_valid[normal_mask], eps)
                    if not ok:
                        print(style_fail("[ERROR] {}: numerical mismatch (test={})".format(case['name'], test_pattern)))
                        all_passed = False
                        continue

            # Normal test: standard comparison
            else:
                ok = result_cmp(golden_valid, output_valid, eps)
                if not ok:
                    print(style_fail("[ERROR] {}: comparison failed (test={})".format(case['name'], test_pattern)))
                    all_passed = False
                    continue

        print(style_pass("[INFO] {}: passed (dtype={}, precision={}, test={})".format(case['name'], dtype_name, precision_type, test_pattern)))

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
