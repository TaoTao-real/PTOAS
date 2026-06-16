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


def check(x, n):
    if len(x) < n:
        x = '0' * (n - len(x)) + x
    elif len(x) > n:
        x = x[1:]
    return x


def hf8_to_float(input_str):
    if len(input_str) != 8:
        raise ValueError("input must be 8 bits")
    s = input_str[0]
    m = input_str[5:]
    m1 = int(input_str[5])
    m2 = int(input_str[6])
    m3 = int(input_str[7])
    if input_str[1] == '1' or input_str[2] == '1':
        d = input_str[1:3]; e = input_str[3:5]
    elif input_str[3] == '1':
        d = input_str[1:4]; e = input_str[4]
    else:
        d = input_str[1:5]; e = ''
    f1 = 1; f2 = 1
    if d == '0000':
        if s == '1': f1 = -1
        if m == '000': return np.nan if s == '1' else 0.0
        return 2 ** (m1 * 4 + m2 * 2 + m3 - 23) * (f1 if s == '1' else 1)
    elif d == '0001':
        if s == '1': f1 = -1
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8.0) * 2 ** 0 * f1
    elif d == '001':
        if s == '1': f1 = -1
        f2 = -1 if e == '1' else 1
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8.0) * 2 ** f2 * f1
    elif d == '01':
        if s == '1': f1 = -1
        e1_val, e2_val = int(input_str[3]), int(input_str[4])
        f2 = -1 if e1_val == 1 else 1
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8.0) * 2 ** (f2 * (2 + e2_val)) * f1
    elif d == '10':
        if s == '1': f1 = -1
        e1_val, e2_val, e3_val = int(input_str[3]), int(input_str[4]), int(input_str[5])
        f2 = -1 if e1_val == 1 else 1
        return (1 + (m2 * 2 + m3) / 4.0) * 2 ** (f2 * (4 + e2_val * 2 + e3_val)) * f1
    elif d == '11':
        if s == '1': f1 = -1
        e1_val, e2_val, e3_val, e4_val = int(input_str[3]), int(input_str[4]), int(input_str[5]), int(input_str[6])
        f2 = -1 if e1_val == 1 else 1
        if e == '01' and m == '111': return f1 * np.inf
        return (1 + m3 / 2.0) * 2 ** (f2 * (8 + e2_val * 4 + e3_val * 2 + e4_val)) * f1
    return 0.0


def convert_hif8_array(arr):
    flat = arr.reshape(-1)
    result = np.zeros(len(flat), dtype=np.float32)
    for i, val in enumerate(flat):
        temp = bin(val); temp = temp.split('b')[1]; temp = check(temp, 8)
        result[i] = hf8_to_float(temp)
    return result.reshape(arr.shape)


for case in CASES:
    setup_case_rng(case)
    a_dtype = case["a_dtype"]
    b_dtype = case["b_dtype"]
    c_dtype = case["c_dtype"]
    M, K, N = case["M"], case["K"], case["N"]
    M_aligned = case.get("M_aligned", M)
    N_aligned = case.get("N_aligned", N)
    K_use = case.get("K_use", K)

    if a_dtype in (np.float16, np.float32):
        a = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(a_dtype)
        b = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(b_dtype)
    elif np.issubdtype(a_dtype, np.integer):
        a = np.random.randint(-10, 10, size=(M, K)).astype(a_dtype)
        b = np.random.randint(-10, 10, size=(K, N)).astype(b_dtype)
    else:
        a = np.random.randint(-10, 10, size=(M, K)).astype(a_dtype)
        b = np.random.randint(-10, 10, size=(K, N)).astype(b_dtype)

    is_hifloat = case.get("is_hifloat", False)
    if is_hifloat:
        a_float = convert_hif8_array(a); b_float = convert_hif8_array(b)
    else:
        a_float = a.astype(np.float64); b_float = b.astype(np.float64)
    golden = np.matmul(a_float, b_float).astype(c_dtype)

    # Pad to aligned/block-sized K (K_use) if needed for cube block alignment.
    need_pad = (M != M_aligned or K != K_use or N != N_aligned)
    if need_pad:
        a_save = np.zeros((M_aligned, K_use), dtype=a_dtype); a_save[:M, :K] = a
        b_save = np.zeros((K_use, N_aligned), dtype=b_dtype); b_save[:K, :N] = b
        golden_save = np.zeros((M_aligned, N_aligned), dtype=c_dtype); golden_save[:M, :N] = golden
    else:
        a_save = a; b_save = b; golden_save = golden

    save_case_data(case["name"], {"input1": a_save, "input2": b_save, "golden": golden_save})
    print(f"[INFO] gen_data: {case['name']} M={M} K={K} N={N} "
          f"padded_A=({M_aligned}x{K}) padded_B=({K}x{N_aligned}) "
          f"a={a_dtype.__name__} b={b_dtype.__name__} c={c_dtype.__name__}")
