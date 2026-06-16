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

# Default scalar value for division (matches the scalar passed in launch.cpp)
DEFAULT_SCALAR = 3.0


def generate_precision_sensitive_scalar(shape, dtype, direction):
    """Generate precision-sensitive test data for scalar division.

    Uses scalar values that create precision-sensitive ratios when divided
    with tile data (e.g., 1/3, 1/7 patterns).
    """
    rows, cols = shape

    # For src / scalar: tile contains precision-sensitive values
    # For scalar / src: scalar is precision-sensitive, src contains small integers
    if direction == "src_scalar":
        # Tile contains values like 1, 7, 5, 10 etc divided by scalar 3
        # Results: 1/3, 7/3, 5/3, 10/3 - precision-sensitive
        input1 = np.zeros(shape, dtype=dtype)
        values = [1, 7, 5, 10, 1, 3, 2, 11]
        section_size = rows // len(values)
        for i, v in enumerate(values):
            start_row = i * section_size
            end_row = min((i + 1) * section_size, rows)
            input1[start_row:end_row, :] = dtype(v)
        scalar = dtype(DEFAULT_SCALAR)
    else:  # scalar_src
        # Scalar is 1, tile contains 3, 7, etc -> 1/3, 1/7 precision-sensitive
        input1 = np.full(shape, dtype(3), dtype=dtype)  # Avoid zeros
        # Fill with divisor values that create precision-sensitive ratios
        values = [3, 7, 11, 3, 5, 7, 11, 3]
        section_size = rows // len(values)
        for i, v in enumerate(values):
            start_row = i * section_size
            end_row = min((i + 1) * section_size, rows)
            input1[start_row:end_row, :] = dtype(v)
        scalar = dtype(1.0)

    return input1, scalar


def generate_subnormal_test_data(shape, dtype, direction):
    """Generate subnormal (denormal) numbers for scalar division tests.

    For src / scalar:
    - src contains subnormal values, scalar is normal
    - Tests subnormal dividend handling

    For scalar / src:
    - scalar is normal, src contains subnormal values
    - Tests subnormal divisor handling (can produce large results)
    """
    rows, cols = shape

    if dtype == np.float32:
        subnormal_max = np.frombuffer(np.array([0x007FFFFF], dtype=np.uint32), dtype=np.float32)[0]
        subnormal_min = np.float32(1e-45)
        normal_min = np.float32(1e-38) * np.float32(2.0)  # smallest normal
    else:  # float16
        subnormal_max = np.frombuffer(np.array([0x03FF], dtype=np.uint16), dtype=np.float16)[0]
        subnormal_min = np.float16(1e-8)
        normal_min = np.float16(6e-5) * np.float16(2.0)

    if direction == "src_scalar":
        # src contains subnormal values, scalar is normal (e.g., 10)
        input1 = np.zeros(shape, dtype=dtype)
        quarter = rows // 4

        # Section 1: MAX_SUBNORMAL / normal -> tiny normal result
        input1[:quarter, :] = subnormal_max

        # Section 2: Mid-range subnormal / normal
        input1[quarter:2*quarter, :] = np.random.uniform(
            subnormal_min, subnormal_max, size=(quarter, cols)).astype(dtype)

        # Section 3: Smallest subnormal / normal
        input1[2*quarter:3*quarter, :] = subnormal_min

        # Section 4: Normal reference
        input1[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)

        scalar = dtype(10.0)
    else:  # scalar_src
        # scalar is normal (e.g., 1e-20 for f32), src contains subnormal
        # This tests: normal / subnormal -> large result (potential overflow)
        input1 = np.zeros(shape, dtype=dtype)
        quarter = rows // 4

        # Section 1: normal / MAX_SUBNORMAL -> large but not overflow
        input1[:quarter, :] = subnormal_max

        # Section 2: normal / mid subnormal -> larger
        input1[quarter:2*quarter, :] = np.random.uniform(
            subnormal_max * 0.1, subnormal_max, size=(quarter, cols)).astype(dtype)

        # Section 3: normal / tiny subnormal -> very large (near overflow)
        input1[2*quarter:3*quarter, :] = np.random.uniform(
            subnormal_min, subnormal_max * 0.1, size=(quarter, cols)).astype(dtype)

        # Section 4: Normal reference
        input1[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)

        # Use a small normal scalar that won't overflow when divided by smallest subnormal
        if dtype == np.float32:
            scalar = np.float32(1e-20)  # Safe: 1e-20 / 1e-45 = 1e25, within f32 range
        else:
            scalar = np.float16(1e-5)  # Safe: 1e-5 / 1e-8 = 1000, within f16 range

    return input1, scalar


def generate_overflow_test_data(shape, dtype, direction):
    """Generate overflow/underflow boundary values for scalar division tests.

    For src / scalar:
    - Large src / tiny scalar -> overflow
    - Tiny src / large scalar -> underflow

    For scalar / src:
    - Large scalar / tiny src -> overflow
    - Tiny scalar / large src -> underflow
    """
    rows, cols = shape

    if dtype == np.float32:
        large_val = np.float32(1e30)
        tiny_val = np.float32(1e-30)
        overflow_trigger = np.float32(1e38)
        underflow_trigger = np.float32(1e-45)
    else:  # float16
        large_val = np.float16(60000)
        tiny_val = np.float16(0.0001)
        overflow_trigger = np.float16(65000)
        underflow_trigger = np.float16(1e-7)

    if direction == "src_scalar":
        input1 = np.zeros(shape, dtype=dtype)
        quarter = rows // 4

        # Section 1: Overflow - large / tiny
        input1[:quarter, :] = overflow_trigger

        # Section 2: Near overflow boundary
        input1[quarter:2*quarter, :] = np.random.uniform(large_val, overflow_trigger,
                                                          size=(quarter, cols)).astype(dtype)

        # Section 3: Underflow - tiny / large
        input1[2*quarter:3*quarter, :] = underflow_trigger

        # Section 4: Normal reference
        input1[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)

        scalar = dtype(tiny_val)  # Tiny scalar triggers overflow

    else:  # scalar_src
        input1 = np.zeros(shape, dtype=dtype)
        quarter = rows // 4

        # Section 1: Overflow - scalar / tiny src
        input1[:quarter, :] = tiny_val  # Tiny divisor

        # Section 2: Near overflow boundary
        input1[quarter:2*quarter, :] = np.random.uniform(
            tiny_val/10, tiny_val, size=(quarter, cols)).astype(dtype)

        # Section 3: Underflow - scalar / large src
        input1[2*quarter:3*quarter, :] = large_val

        # Section 4: Normal reference
        input1[3*quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows-3*quarter, cols)).astype(dtype)

        # Large scalar triggers overflow when divided by tiny src
        scalar = dtype(overflow_trigger)

    return input1, scalar


def generate_normal_data(shape, dtype, direction):
    """Generate simple random values for normal testing."""
    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    scalar = dtype(DEFAULT_SCALAR)
    return input1, scalar


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    direction = case.get("direction", "src_scalar")
    test_pattern = case.get("test_pattern", "normal")

    # Generate test data based on pattern and direction
    data_generators = {
        "normal": generate_normal_data,
        "precision_sensitive": generate_precision_sensitive_scalar,
        "subnormal": generate_subnormal_test_data,
        "overflow": generate_overflow_test_data,
    }

    generator = data_generators.get(test_pattern, generate_normal_data)
    input1, scalar_val = generator(shape, dtype, direction)

    # Compute golden reference using numpy (IEEE 754 compliant)
    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape

    # Suppress overflow/divide warnings for boundary tests (expected behavior)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        if direction == "src_scalar":
            golden[:vr, :vc] = (input1[:vr, :vc] / scalar_val).astype(dtype, copy=False)
        else:  # scalar_src
            golden[:vr, :vc] = (scalar_val / input1[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "golden": golden})
    precision_type = case.get("precision_type", "default")
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} direction={direction} test={test_pattern} precision={precision_type} scalar={scalar_val}")
