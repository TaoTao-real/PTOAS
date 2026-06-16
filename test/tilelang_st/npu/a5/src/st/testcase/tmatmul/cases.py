# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmatmul ST test cases.

Each case maps to a pto-isa tmatmul test (TMATMULTest.case1..case13).
Excludes bias and acc variants (those live in tmatmul_bias / tmatmul_acc).
"""

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2


def ceil_align(num, align):
    return (num + align - 1) // align * align


CASES = [
    # ---- case1: f16 x f16 -> f32, 40x50x60 (M pad→48, K pad→64 for block-align) ----
    {
        "name": "f16_40x50x60",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "c_dtype": np.float32,
        "M": 40, "K": 50, "N": 60,
        "M_aligned": 48, "K_use": 64, "N_aligned": 64,
        "eps": 1e-2,
    },
    # ---- case2: i8 x i8 -> i32, 6x7x8 (M pad→16, N pad→32) ----
    {
        "name": "i8_6x7x8",
        "a_dtype": np.int8,
        "b_dtype": np.int8,
        "c_dtype": np.int32,
        "M": 6, "K": 7, "N": 8,
        "M_aligned": 16, "K_use": 32, "N_aligned": 32,
        "eps": 1e-6,
    },
    # ---- case3: f16 x f16 -> f32, 127x128x61 (M pad→128, N pad→64, K aligned) ----
    {
        "name": "f16_127x128x61",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "c_dtype": np.float32,
        "M": 127, "K": 128, "N": 61,
        "M_aligned": 128, "K_use": 128, "N_aligned": 64,
        "eps": 1e-2,
    },
    # ---- case4: f32 x f32 -> f32, 120x110x50 (M pad→128, K pad→112, N pad→64) ----
    {
        "name": "f32_120x110x50",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 120, "K": 110, "N": 50,
        "M_aligned": 128, "K_use": 112, "N_aligned": 64,
        "eps": 1e-5,
    },
    # ---- case5: bf16 x bf16 -> f32, 144x80x48 (fully aligned) ----
    {
        "name": "bf16_144x80x48",
        "a_dtype": bfloat16,
        "b_dtype": bfloat16,
        "c_dtype": np.float32,
        "M": 144, "K": 80, "N": 48,
        "M_aligned": 144, "K_use": 80, "N_aligned": 48,
        "eps": 1e-2,
    },
    # # ---- case6: f8e4m3 x f8e4m3 -> f32, 32x64x96 ----
    # ...
    # # ---- case7..10 commented out ----
    # ---- case12: f32 x f32 -> f32, 16x32x64 (fully aligned) ----
    {
        "name": "f32_16x32x64",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 32, "N": 64,
        "M_aligned": 16, "K_use": 32, "N_aligned": 64,
        "eps": 1e-5,
    },
    # ---- case13: f32 x f32 -> f32, 128x96x64 (fully aligned) ----
    {
        "name": "f32_128x96x64",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 128, "K": 96, "N": 64,
        "M_aligned": 128, "K_use": 96, "N_aligned": 64,
        "eps": 1e-5,
    },
]
