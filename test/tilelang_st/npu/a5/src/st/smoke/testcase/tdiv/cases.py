# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tdiv ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - eps:         tolerance for numpy.allclose (atol and rtol).
  - precision_type: optional, "default" or "high_precision".
  - test_pattern: optional, "normal", "boundary", "subnormal", "overflow", "nan_inf"

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # ============================================================
    # Normal cases - basic functionality (DEFAULT precision mode)
    # ============================================================
    {
        "name": "f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "test_pattern": "normal",
    },
    {
        "name": "f32_32x32",
        "dtype": np.float32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-6,
        "test_pattern": "normal",
    },
    {
        "name": "f32_64x64",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-6,
        "test_pattern": "normal",
    },
    {
        "name": "f16_16x256",
        "dtype": np.float16,
        "shape": (16, 256),
        "valid_shape": (16, 256),
        "eps": 1e-3,
        "test_pattern": "normal",
    },

    # ============================================================
    # HIGH_PRECISION mode - comprehensive boundary tests
    # ============================================================
    # Precision-sensitive ratios (1/3, 1/7, 7/3) - tests three-candidate search
    {
        "name": "f32_16x64_hp_precision",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,  # Allow ±1 ULP for high-precision algorithm
    },
    {
        "name": "f16_16x64_hp_precision",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },

    # Subnormal numbers - tests denormal normalization and compensation
    {
        "name": "f32_16x64_hp_subnormal",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,  # Subnormal handling may have ±2 ULP variance
    },
    {
        "name": "f16_16x64_hp_subnormal",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,
    },

# Overflow/Underflow boundaries - tests exponent handling
    {
        "name": "f32_16x64_hp_overflow",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "test_pattern": "overflow",
    },
    {
        "name": "f16_16x64_hp_overflow",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "overflow",
    },

    # Different shapes - test tile size variations
    {
        "name": "f32_32x32_hp",
        "dtype": np.float32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-5,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 2,
    },
    {
        "name": "f32_64x64_hp",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-5,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 2,
    },
    {
        "name": "f16_16x256_hp",
        "dtype": np.float16,
        "shape": (16, 256),
        "valid_shape": (16, 256),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 2,
    },

    # Partial valid shape - test masked computation
    {
        "name": "f32_16x64_hp_partial",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 31),
        "eps": 1e-5,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 2,
    },
    {
        "name": "f16_16x64_hp_partial",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 63),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 2,
    },

    # Small shape HP tests - aligned with pto-isa (case_float_hp_2x16, case_half_hp_2x32)
    {
        "name": "f32_2x16_hp",
        "dtype": np.float32,
        "shape": (2, 16),
        "valid_shape": (2, 16),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },
    {
        "name": "f16_2x32_hp",
        "dtype": np.float16,
        "shape": (2, 32),
        "valid_shape": (2, 32),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },
]

_SMOKE_CASE_NAMES = ['f32_16x64', 'f32_16x64_hp_subnormal', 'f16_16x64_hp_partial']
_SMOKE_CASE_NAME_SET = set(_SMOKE_CASE_NAMES)
_missing = [name for name in _SMOKE_CASE_NAMES if name not in {case["name"] for case in CASES}]
if _missing:
    raise RuntimeError("unknown smoke case(s): " + ", ".join(_missing))
CASES = [case for case in CASES if case["name"] in _SMOKE_CASE_NAME_SET]
