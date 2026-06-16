# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tdivs ST test cases.

vdiv only supports f16/f32 in TileLang DSL v1.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - eps:         tolerance for numpy.allclose (atol and rtol).
  - direction:   "src_scalar" (src / scalar) or "scalar_src" (scalar / src)
  - precision_type: optional, "default" or "high_precision".
  - test_pattern: optional, "normal", "precision_sensitive", "subnormal", "overflow"

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # ============================================================
    # Normal cases - basic functionality (DEFAULT precision mode)
    # ============================================================
    # src / scalar direction
    {
        "name": "f32_32x64",
        "dtype": np.float32,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 1e-6,
        "direction": "src_scalar",
    },
    {
        "name": "f16_63x64",
        "dtype": np.float16,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
        "direction": "src_scalar",
    },
    {
        "name": "f32_7x448",
        "dtype": np.float32,
        "shape": (7, 448),
        "valid_shape": (7, 448),
        "eps": 1e-6,
        "direction": "src_scalar",
    },
    {
        "name": "f32_256x16",
        "dtype": np.float32,
        "shape": (256, 16),
        "valid_shape": (256, 16),
        "eps": 1e-6,
        "direction": "src_scalar",
    },
    # scalar / src direction
    {
        "name": "f32_32x64_scalar_src",
        "dtype": np.float32,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 1e-6,
        "direction": "scalar_src",
    },
    {
        "name": "f16_63x64_scalar_src",
        "dtype": np.float16,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
        "direction": "scalar_src",
    },
    {
        "name": "f32_7x448_scalar_src",
        "dtype": np.float32,
        "shape": (7, 448),
        "valid_shape": (7, 448),
        "eps": 1e-6,
        "direction": "scalar_src",
    },
    {
        "name": "f32_256x16_scalar_src",
        "dtype": np.float32,
        "shape": (256, 16),
        "valid_shape": (256, 16),
        "eps": 1e-6,
        "direction": "scalar_src",
    },

    # ============================================================
    # HIGH_PRECISION mode - src / scalar direction
    # ============================================================
    # Precision-sensitive ratios
    {
        "name": "f32_32x64_hp",
        "dtype": np.float32,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },
    {
        "name": "f16_63x64_hp",
        "dtype": np.float16,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },

    # Subnormal numbers
    {
        "name": "f32_16x64_hp_subnormal",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,
    },
    {
        "name": "f16_16x64_hp_subnormal",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,
    },

    # Overflow/Underflow boundaries
    {
        "name": "f32_16x64_hp_overflow",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "overflow",
    },
    {
        "name": "f16_16x64_hp_overflow",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "src_scalar",
        "test_pattern": "overflow",
    },

    # ============================================================
    # HIGH_PRECISION mode - scalar / src direction
    # ============================================================
    {
        "name": "f32_32x64_hp_scalar_src",
        "dtype": np.float32,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },
    {
        "name": "f16_63x64_hp_scalar_src",
        "dtype": np.float16,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "precision_sensitive",
        "ulp_tolerance": 1,
    },

    # Subnormal - scalar / src (scalar is normal, src contains subnormals)
    {
        "name": "f32_16x64_hp_subnormal_scalar_src",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,
    },
    {
        "name": "f16_16x64_hp_subnormal_scalar_src",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "subnormal",
        "ulp_tolerance": 2,
    },

    # Overflow - scalar / src (division by small src values)
    {
        "name": "f32_16x64_hp_overflow_scalar_src",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "overflow",
    },
    {
        "name": "f16_16x64_hp_overflow_scalar_src",
        "dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
        "precision_type": "high_precision",
        "direction": "scalar_src",
        "test_pattern": "overflow",
    },
]

_SMOKE_CASE_NAMES = ['f32_32x64', 'f32_16x64_hp_overflow', 'f16_16x64_hp_subnormal_scalar_src']
_SMOKE_CASE_NAME_SET = set(_SMOKE_CASE_NAMES)
_missing = [name for name in _SMOKE_CASE_NAMES if name not in {case["name"] for case in CASES}]
if _missing:
    raise RuntimeError("unknown smoke case(s): " + ", ".join(_missing))
CASES = [case for case in CASES if case["name"] in _SMOKE_CASE_NAME_SET]
