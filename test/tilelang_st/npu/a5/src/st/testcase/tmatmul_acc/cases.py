# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmatmul_acc ST test cases.

Test Split-K pattern: C[M, N] = A[M, K] x B[K, N]
K is split into chunks of BASEK; each chunk computed by mad (first) / mad_acc (subsequent).
"""

import numpy as np


def _ceil_align(num, align):
    return (num + align - 1) // align * align


CASES = [
    {
        "name": "f16_16x32x16",
        "dtype": np.float16,
        "M": 16,
        "K": 32,
        "N": 16,
        "BASEK": 16,
        "M_aligned": 16,
        "N_aligned": 16,
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
    {
        "name": "f16_128x128x64",
        "dtype": np.float16,
        "M": 128,
        "K": 128,
        "N": 64,
        "BASEK": 64,
        "M_aligned": 128,
        "N_aligned": 64,
        "shape_c": (128, 64),
        "eps": 1e-2,
    },
    {
        "name": "f16_127x128x61",
        "dtype": np.float16,
        "M": 127,
        "K": 128,
        "N": 61,
        "BASEK": 64,
        "M_aligned": 128,
        "N_aligned": 64,
        "shape_c": (127, 61),
        "eps": 1e-2,
    },
]