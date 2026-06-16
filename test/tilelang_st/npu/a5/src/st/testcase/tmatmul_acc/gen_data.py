# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


for case in CASES:
    setup_case_rng(case)

    M = case["M"]
    K = case["K"]
    N = case["N"]
    dtype = case["dtype"]
    M_aligned = case.get("M_aligned", M)
    N_aligned = case.get("N_aligned", N)

    a = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(dtype)
    b = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(dtype)

    golden = np.matmul(a.astype(np.float32), b.astype(np.float32))

    # Pad A and B to aligned dimensions so the kernel can load aligned tiles
    # without reading out-of-bounds memory.
    # Golden also padded to match the full L0C storeback size.
    a_padded = np.zeros((M_aligned, K), dtype=dtype)
    a_padded[:M, :] = a
    b_padded = np.zeros((K, N_aligned), dtype=dtype)
    b_padded[:, :N] = b
    golden_padded = np.zeros((M_aligned, N_aligned), dtype=np.float32)
    golden_padded[:M, :N] = golden

    save_case_data(case["name"], {"input1": a_padded, "input2": b_padded, "golden": golden_padded})
    print(
        f"[INFO] gen_data: {case['name']} "
        f"A={M}x{K} B={K}x{N} C={M}x{N} dtype={dtype.__name__} BASEK={case['BASEK']} iter={K // case['BASEK']}"
    )