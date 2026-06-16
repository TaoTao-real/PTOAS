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

np.random.seed(19)


for case in CASES:
    setup_case_rng(case)

    a_dtype = case["a_dtype"]
    b_dtype = case["b_dtype"]
    bias_dtype = case["bias_dtype"]
    c_dtype = case["c_dtype"]
    M, K, N = case["M"], case["K"], case["N"]
    M_aligned = case.get("M_aligned", M)
    K_aligned = case.get("K_aligned", K)
    N_aligned = case.get("N_aligned", N)

    x1 = np.random.randint(-10, 10, size=(M, K)).astype(a_dtype)
    x2 = np.random.randint(-10, 10, size=(K, N)).astype(b_dtype)
    bias = np.random.randint(1, 10, size=(N,)).astype(bias_dtype)

    golden = np.matmul(x1.astype(c_dtype), x2.astype(c_dtype)).astype(c_dtype) + bias.astype(c_dtype)

    # Pad A, B, bias and golden to aligned dimensions so the kernel can load aligned
    # tiles without reading out-of-bounds memory.
    a_padded = np.zeros((M_aligned, K_aligned), dtype=a_dtype)
    a_padded[:M, :K] = x1
    b_padded = np.zeros((K_aligned, N_aligned), dtype=b_dtype)
    b_padded[:K, :N] = x2
    bias_padded = np.zeros((N_aligned,), dtype=bias_dtype)
    bias_padded[:N] = bias
    golden_padded = np.zeros((M_aligned, N_aligned), dtype=c_dtype)
    golden_padded[:M, :N] = golden

    save_case_data(case["name"], {
        "input1": a_padded,
        "input2": b_padded,
        "input3": bias_padded,
        "golden": golden_padded,
    })
    print(
        f"[INFO] gen_data: {case['name']} "
        f"M={M} K={K} N={N} M_aligned={M_aligned} K_aligned={K_aligned} N_aligned={N_aligned} "
        f"a={a_dtype.__name__} b={b_dtype.__name__} bias={bias_dtype.__name__} c={c_dtype.__name__}"
    )
