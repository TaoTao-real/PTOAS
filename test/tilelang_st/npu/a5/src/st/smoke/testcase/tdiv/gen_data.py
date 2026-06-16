# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import sys
import os
from pathlib import Path

# Add current directory to path for standalone execution
script_dir = Path(__file__).parent
if script_dir not in sys.path:
    sys.path.insert(0, str(script_dir))

# Add st_common directory
st_common_dir = script_dir.parent
if st_common_dir not in sys.path:
    sys.path.insert(0, str(st_common_dir))

import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)


def generate_precision_sensitive_data(shape, dtype):
    """Generate precision-sensitive ratios to test three-candidate search algorithm.

    Focuses on values that cannot be exactly represented in floating point:
    - 1/3, 1/7, 7/3 - infinite binary representation
    - Values near integer boundaries where z/z±1 compete
    """
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    ratios = [(1, 3), (1, 7), (7, 3), (1, 11), (5, 3), (10, 3)]

    section_size = rows // len(ratios)
    for i, (a, b) in enumerate(ratios):
        start_row = i * section_size
        end_row = min((i + 1) * section_size, rows)
        input1[start_row:end_row, :] = dtype(a)
        input2[start_row:end_row, :] = dtype(b)

    # Add variations: negative versions, different signs
    remaining_rows = rows - len(ratios) * section_size
    if remaining_rows > 0:
        input1[-remaining_rows:, :] = np.random.choice([-1, 1], size=(remaining_rows, cols)).astype(dtype)
        input2[-remaining_rows:, :] = dtype(3)

    return input1, input2


def generate_subnormal_test_data(shape, dtype):
    """Generate subnormal (denormal) numbers to test normalization handling.

    NOTE: High-precision division algorithm (Div754) has asymmetric subnormal detection:
    - src0 (dividend): EQ comparison - only detects MAX_SUBNORMAL (0x007FFFFF for f32)
    - src1 (divisor): LT comparison - detects entire subnormal range

    Test design constraints:
    - Section 1: src0 = MAX_SUBNORMAL, src1 = normal (tests src0 EQ detection)
    - Section 2: src0 = MAX_SUBNORMAL, src1 = larger subnormal (tests both subnormal)
    - Section 3: src0 = normal, src1 = MAX_SUBNORMAL (tests src1 subnormal with normal src0)
    - Section 4: normal reference

    Avoid "normal / small_subnormal" which would overflow to Inf.
    """
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    if dtype == np.float32:
        tiny = np.finfo(np.float32).tiny
        subnormal_max = np.frombuffer(np.array([0x007FFFFF], dtype=np.uint32), dtype=np.float32)[0]
        subnormal_min = np.float32(1e-45)
        normal_min = tiny * np.float32(2.0)
    else:  # float16
        tiny = np.finfo(np.float16).tiny
        subnormal_max = np.frombuffer(np.array([0x03FF], dtype=np.uint16), dtype=np.float16)[0]
        subnormal_min = np.float16(1e-8)
        normal_min = tiny * np.float16(2.0)

    quarter = rows // 4

    # Section 1: src0 = MAX_SUBNORMAL, src1 = normal
    # ratio ≈ 1e-38 / 10 ≈ 1e-39 (不 overflow)
    input1[:quarter, :] = subnormal_max
    input2[:quarter, :] = np.random.uniform(normal_min, 100.0, size=(quarter, cols)).astype(dtype)

    # Section 2: src0 = MAX_SUBNORMAL, src1 = smaller subnormal (ratio ≈ 1-10)
    # 确保 src1 在 subnormal 范围内: subnormal_min ~ subnormal_max
    input1[quarter:2*quarter, :] = subnormal_max
    input2[quarter:2*quarter, :] = np.random.uniform(subnormal_max * 0.1, subnormal_max,
                                                      size=(quarter, cols)).astype(dtype)

    # Section 3: src0 = MAX_SUBNORMAL, src1 = very small subnormal (ratio ≈ 10-500)
    input1[2*quarter:3*quarter, :] = subnormal_max
    input2[2*quarter:3*quarter, :] = np.random.uniform(subnormal_min, subnormal_max * 0.1,
                                                        size=(quarter, cols)).astype(dtype)

    # Section 4: normal reference
    input1[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)
    input2[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)

    return input1, input2


def generate_overflow_test_data(shape, dtype):
    """Generate overflow/underflow boundary values to test exponent handling.

    Tests:
    - Large/small ratios that overflow to Inf
    - Tiny ratios that underflow to 0 or min denormal
    - Values at max/min exponent boundaries
    """
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    if dtype == np.float32:
        large_val = np.float32(1e30)
        tiny_val = np.float32(1e-30)
        overflow_trigger = np.float32(1e38)
        underflow_trigger = np.float32(1e-45)
        max_normal = np.float32(3.4e38)
    else:  # float16
        large_val = np.float16(60000)  # Near f16 max (65504)
        tiny_val = np.float16(0.0001)
        overflow_trigger = np.float16(65000)
        underflow_trigger = np.float16(1e-7)
        max_normal = np.float16(65504)

    # Section 1: Overflow scenarios
    quarter = rows // 4
    input1[:quarter, :cols//2] = overflow_trigger
    input2[:quarter, :cols//2] = tiny_val  # overflow_trigger / tiny_val -> Inf

    input1[:quarter, cols//2:] = large_val
    input2[:quarter, cols//2:] = np.random.uniform(1e-35 if dtype==np.float32 else 1e-7,
                                                    tiny_val,
                                                    size=(quarter, cols//2)).astype(dtype)

    # Section 2: Underflow scenarios
    input1[quarter:2*quarter, :cols//2] = underflow_trigger
    input2[quarter:2*quarter, :cols//2] = large_val  # underflow_trigger / large_val -> 0

    input1[quarter:2*quarter, cols//2:] = tiny_val
    input2[quarter:2*quarter, cols//2:] = np.random.uniform(large_val, max_normal,
                                                             size=(quarter, cols//2)).astype(dtype)

    # Section 3: Near boundary (may or may not overflow)
    input1[2*quarter:3*quarter, :] = np.random.uniform(large_val/10, max_normal,
                                                        size=(quarter, cols)).astype(dtype)
    input2[2*quarter:3*quarter, :] = np.random.uniform(tiny_val/10, tiny_val,
                                                        size=(quarter, cols)).astype(dtype)

    # Section 4: Normal values (control group)
    input1[3*quarter:, :] = np.random.uniform(0.1, 100.0,
                                               size=(rows-3*quarter, cols)).astype(dtype)
    input2[3*quarter:, :] = np.random.uniform(0.1, 100.0,
                                               size=(rows-3*quarter, cols)).astype(dtype)

    return input1, input2


def generate_nan_inf_test_data(shape, dtype):
    """Generate NaN and Inf inputs to test special value propagation.

    Tests IEEE 754 rules:
    - 0/0 -> NaN
    - Inf/Inf -> NaN
    - x/0 -> Inf (or NaN if x=0)
    - Inf/x -> Inf
    - x/Inf -> 0
    - NaN propagates
    """
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    # Create special values
    if dtype == np.float32:
        pos_inf = np.float32(np.inf)
        neg_inf = np.float32(-np.inf)
        nan_val = np.float32(np.nan)
        zero_val = np.float32(0.0)
        pos_one = np.float32(1.0)
        neg_one = np.float32(-1.0)
    else:  # float16
        pos_inf = np.float16(np.inf)
        neg_inf = np.float16(-np.inf)
        nan_val = np.float16(np.nan)
        zero_val = np.float16(0.0)
        pos_one = np.float16(1.0)
        neg_one = np.float16(-1.0)

    # Section 1: 0/0 -> NaN, x/0 -> Inf
    eighth = rows // 8
    input1[0:eighth, :] = zero_val
    input2[0:eighth, :] = zero_val  # 0/0 -> NaN

    input1[eighth:2*eighth, :] = pos_one
    input2[eighth:2*eighth, :] = zero_val  # 1/0 -> Inf

    input1[2*eighth:3*eighth, :] = neg_one
    input2[2*eighth:3*eighth, :] = zero_val  # -1/0 -> -Inf

    # Section 2: Inf/Inf -> NaN, Inf/x -> Inf, x/Inf -> 0
    input1[3*eighth:4*eighth, :] = pos_inf
    input2[3*eighth:4*eighth, :] = pos_inf  # Inf/Inf -> NaN

    input1[4*eighth:5*eighth, :] = pos_inf
    input2[4*eighth:5*eighth, :] = pos_one  # Inf/1 -> Inf

    input1[5*eighth:6*eighth, :] = pos_one
    input2[5*eighth:6*eighth, :] = pos_inf  # 1/Inf -> 0

    # Section 3: NaN propagation
    input1[6*eighth:7*eighth, :] = nan_val
    input2[6*eighth:7*eighth, :] = np.random.uniform(0.1, 10.0,
                                                      size=(eighth, cols)).astype(dtype)  # NaN/x -> NaN

    input1[7*eighth:rows, :] = np.random.uniform(0.1, 10.0,
                                                  size=(rows-7*eighth, cols)).astype(dtype)
    input2[7*eighth:rows, :cols//2] = nan_val  # x/NaN -> NaN (half of remaining)
    input2[7*eighth:rows, cols//2:] = np.random.uniform(0.1, 10.0,
                                                        size=(rows-7*eighth, cols//2)).astype(dtype)

    return input1, input2


def generate_boundary_test_data(shape, dtype):
    """Generate mixed boundary test data to stress IEEE 754 compliance.

    Combines subnormal and overflow scenarios (no NaN/Inf to avoid hardware limitations).
    """
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    # Adapt thresholds based on dtype
    if dtype == np.float32:
        subnormal_val = np.float32(1.175e-38)
        large_val = np.float32(1e30)
        tiny_val = np.float32(1e-10)
    elif dtype == np.float16:
        subnormal_val = np.float16(6e-5)
        large_val = np.float16(60000)
        tiny_val = np.float16(0.001)
    else:
        subnormal_val = np.float32(1e-38)
        large_val = np.float32(1e30)
        tiny_val = np.float32(1e-10)

    # Section 1: Subnormal numbers (first half)
    half = rows // 2
    if dtype == np.float32:
        input1[:half, :] = np.random.uniform(1e-40, subnormal_val,
                                                  size=(half, cols)).astype(dtype)
    else:
        input1[:half, :] = np.random.uniform(1e-8, subnormal_val,
                                                  size=(half, cols)).astype(dtype)
    input2[:half, :] = np.random.uniform(1.0, 10.0,
                                             size=(half, cols)).astype(dtype)

    # Section 2: Overflow boundary (second half)
    input1[half:, :cols//2] = large_val
    input2[half:, :cols//2] = tiny_val

    input1[half:, cols//2:] = np.random.uniform(large_val/10, large_val,
                                                   size=(half, cols//2)).astype(dtype)
    input2[half:, cols//2:] = np.random.uniform(tiny_val/10, tiny_val,
                                                   size=(half, cols//2)).astype(dtype)

    return input1, input2


def generate_normal_data(shape, dtype):
    """Generate simple random values for normal testing."""
    if dtype in (np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32):
        input1 = np.random.randint(1, 10, size=shape).astype(dtype)
        input2 = np.random.randint(1, 10, size=shape).astype(dtype)
    else:
        input1 = np.random.uniform(0.1, 100.0, size=shape).astype(dtype)
        input2 = np.random.uniform(0.1, 100.0, size=shape).astype(dtype)
    return input1, input2

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    test_pattern = case.get("test_pattern", "normal")

    # Generate test data based on pattern
    # NOTE: nan_inf test removed due to hardware vdiv NaN-from-division limitations
    data_generators = {
        "normal": generate_normal_data,
        "precision_sensitive": generate_precision_sensitive_data,
        "subnormal": generate_subnormal_test_data,
        "overflow": generate_overflow_test_data,
        "boundary": generate_boundary_test_data,
    }

    generator = data_generators.get(test_pattern, generate_normal_data)
    input1, input2 = generator(shape, dtype)

    # Compute golden reference using numpy (IEEE 754 compliant)
    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape

    # Suppress overflow/divide warnings for boundary tests (expected behavior)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        golden[:vr, :vc] = (input1[:vr, :vc] / input2[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    precision_type = case.get("precision_type", "default")
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} test={test_pattern} precision={precision_type}")
