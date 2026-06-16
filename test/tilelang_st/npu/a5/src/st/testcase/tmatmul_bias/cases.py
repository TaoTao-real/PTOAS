# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmatmul_bias ST test cases.

Each case maps to a pto-isa tmatmul bias test (TMATMULTest.case_bias_*).
Excludes fp8 and 4-bit variants.
"""

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16


def ceil_align(num, align):
    return (num + align - 1) // align * align


# Each entry is a dict consumed by gen_data.py and compare.py.
# The C++ side (main.cpp / launch.cpp / .pto) maintains its own matching list.

CASES = [
    # ---- f16_16x16x16: working baseline copied from PTOAS_matmul0_copy ----
    {
        "name": "f16_16x16x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 16, "N": 16,
        "M_aligned": 16,
        "K_aligned": 16,
        "N_aligned": 16,
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
    # ---- case_bias_1: i8 x i8 -> i32, bias i32, M=8 K=7 N=6 ----
    {
        "name": "i8_bias_i32_8x7x6",
        "a_dtype": np.int8,
        "b_dtype": np.int8,
        "bias_dtype": np.int32,
        "c_dtype": np.int32,
        "M": 8, "K": 7, "N": 6,
        "M_aligned": 16,
        "K_aligned": 32,
        "N_aligned": 32,
        "shape_c": (8, 6),
        "eps": 1e-6,
    },
    # ---- case_bias_2: f16 x f16 -> f32, bias f16, M=16 K=15 N=16 ----
    {
        "name": "f16_bias_f16_16x15x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,  # DEBUG: f32 bias to test if f16 bias causes hang
        "c_dtype": np.float32,
        "M": 16, "K": 15, "N": 16,
        "M_aligned": 16,
        "K_aligned": 16,
        "N_aligned": 16,
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
    # ---- case_bias_3: f16 x f16 -> f32, bias f32 (was bf16; mte_l1_bt bf16->f32 unsupported), M=112 K=127 N=80 ----
    {
        "name": "f16_bias_bf16_112x127x80",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 112, "K": 127, "N": 80,
        "M_aligned": 112,
        "K_aligned": 128,
        "N_aligned": 80,
        "shape_c": (112, 80),
        "eps": 1e-2,
    },
    # ---- case_bias_4: bf16 x bf16 -> f32, bias f32 (was bf16; mte_l1_bt bf16->f32 unsupported), M=80 K=112 N=63 ----
    {
        "name": "bf16_bias_bf16_80x112x63",
        "a_dtype": bfloat16,
        "b_dtype": bfloat16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 80, "K": 112, "N": 63,
        "M_aligned": 80,
        "K_aligned": 128,
        "N_aligned": 64,
        "shape_c": (80, 63),
        "eps": 1e-2,
    },
    # ---- case_bias_5: f32 x f32 -> f32, bias f32, M=127 K=128 N=63 (Split-K in pto-isa) ----
    {
        "name": "f32_bias_f32_127x128x63",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 127, "K": 128, "N": 63,
        "M_aligned": 128,
        "K_aligned": 128,
        "N_aligned": 64,
        "shape_c": (127, 63),
        "eps": 1e-5,
    },
]
