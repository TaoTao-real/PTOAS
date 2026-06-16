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
    c_dtype = case["c_dtype"]
    M, K, N = case["M"], case["K"], case["N"]
    N_aligned = case.get("N_aligned", N)
    K_use = case.get("K_use", K)

    a = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(a_dtype)
    b = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(b_dtype)

    if case.get("is_bias", False):
        bias_dtype = case["bias_dtype"]
        bias = np.random.uniform(-1.0, 1.0, size=(N,)).astype(bias_dtype)
        golden = (np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(c_dtype)
                  + bias.astype(c_dtype))
    else:
        golden = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(c_dtype)

    a_save = np.zeros((M, K_use), dtype=a_dtype)
    a_save[:M, :K] = a
    b_save = np.zeros((K_use, N_aligned), dtype=b_dtype)
    b_save[:K, :N] = b
    golden_save = np.zeros((M, N_aligned), dtype=c_dtype)
    golden_save[:M, :N] = golden

    data = {"input1": a_save, "input2": b_save}
    if case.get("is_bias", False):
        bias_save = np.zeros((N_aligned,), dtype=bias_dtype)
        bias_save[:N] = bias
        data["input3"] = bias_save
    data["golden"] = golden_save

    save_case_data(case["name"], data)
    print(f"[INFO] gen_data: {case['name']} M={M} K={K} N={N} "
          f"padded_A=({M}x{K_use}) padded_B=({K_use}x{N_aligned}) "
          f"a={a_dtype.__name__} b={b_dtype.__name__} c={c_dtype.__name__}")
