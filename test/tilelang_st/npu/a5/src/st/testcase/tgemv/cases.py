# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tgemv ST test cases.

Ports GEMV test cases from pto-isa:
  1. TGEMV:              f16xf16->f32, M=1 K=300 N=60  (basic gemv, no bias)
  2. TGEMV_BIAS+TGEMV_ACC: f16xf16->f32, M=1 K=512 N=85  (gemv with bias + split-K)
"""

import numpy as np


CASES = [
    {
        "name": "gemv_f16_1x300x60",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "c_dtype": np.float32,
        "M": 1, "K": 300, "N": 60,
        "K_use": 320, "N_aligned": 64,
        "eps": 1e-2,
    },
    {
        "name": "gemv_bias_f16_1x512x85",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 1, "K": 512, "N": 85,
        "K_use": 512, "N_aligned": 96,
        "eps": 1e-2,
        "is_bias": True,
        "is_split_k": True,
        "BASEK": 256,
    },
]
