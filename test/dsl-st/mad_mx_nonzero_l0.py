#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

M = 16
N = 16
K = 64
SCALE_BYTES = 64

MAD_MX_NONZERO_L0_SOURCE = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<cube>} {
  func.func @mad_mx_nonzero_l0_kernel(
      %a_gm: !pto.ptr<f8E4M3FN, gm>,
      %b_gm: !pto.ptr<f8E4M3FN, gm>,
      %a_scale_gm: !pto.ptr<f8E4M3FN, gm>,
      %b_scale_gm: !pto.ptr<f8E4M3FN, gm>,
      %c_gm: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c64_burst_i64 = arith.constant 64 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c32768_i64 = arith.constant 32768 : i64

    %l1_a_data = pto.castptr %c0_i64 : i64 -> !pto.ptr<f8E4M3FN, l1>
    %l1_a_scale = pto.castptr %c1024_i64 : i64 -> !pto.ptr<f8E4M3FN, l1>
    %l1_b_data = pto.castptr %c1088_i64 : i64 -> !pto.ptr<f8E4M3FN, l1>
    %l1_b_scale = pto.castptr %c2112_i64 : i64 -> !pto.ptr<f8E4M3FN, l1>
    %l0a = pto.castptr %c32768_i64 : i64 -> !pto.ptr<f8E4M3FN, l0a>
    %l0b = pto.castptr %c32768_i64 : i64 -> !pto.ptr<f8E4M3FN, l0b>
    %l0c = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, l0c>

    pto.mte_gm_l1 %a_gm, %l1_a_data, %c1024_i64
      nburst(%c1_i64, %c0_i64, %c0_i64)
      : !pto.ptr<f8E4M3FN, gm>, !pto.ptr<f8E4M3FN, l1>, i64, i64, i64, i64
    pto.mte_gm_l1 %a_scale_gm, %l1_a_scale, %c64_burst_i64
      nburst(%c1_i64, %c0_i64, %c0_i64)
      : !pto.ptr<f8E4M3FN, gm>, !pto.ptr<f8E4M3FN, l1>, i64, i64, i64, i64
    pto.mte_gm_l1 %b_gm, %l1_b_data, %c1024_i64
      nburst(%c1_i64, %c0_i64, %c0_i64)
      : !pto.ptr<f8E4M3FN, gm>, !pto.ptr<f8E4M3FN, l1>, i64, i64, i64, i64
    pto.mte_gm_l1 %b_scale_gm, %l1_b_scale, %c64_burst_i64
      nburst(%c1_i64, %c0_i64, %c0_i64)
      : !pto.ptr<f8E4M3FN, gm>, !pto.ptr<f8E4M3FN, l1>, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]

    pto.mte_l1_l0a %l1_a_data, %l0a, %c16_i64, %c64_i64, %c0_i64, %c0_i64
      : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0a>, i64, i64, i64, i64
    pto.mte_l1_l0b %l1_b_data, %l0b, %c64_i64, %c16_i64, %c0_i64, %c0_i64 {transpose = true}
      : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64, i64, i64
    pto.mte_l1_l0a_mx %l1_a_scale, %l0a, %c16_i64, %c64_i64, %c0_i64, %c0_i64
      : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0a>, i64, i64, i64, i64
    pto.mte_l1_l0b_mx %l1_b_scale, %l0b, %c64_i64, %c16_i64, %c0_i64, %c0_i64
      : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64, i64, i64
    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]
    pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64 unit_flag(check_only) disable_gemv sat
      : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64

    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]
    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]

    pto.mte_l0c_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64,
      nz2nd
      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""

@pto.jit(
    name="mad_mx_nonzero_l0_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=MAD_MX_NONZERO_L0_SOURCE,
)
def mad_mx_nonzero_l0_kernel(
    a_gm: pto.ptr(pto.f8e4m3, "gm"),
    b_gm: pto.ptr(pto.f8e4m3, "gm"),
    a_scale_gm: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_gm: pto.ptr(pto.f8e4m3, "gm"),
    c_gm: pto.ptr(pto.f32, "gm"),
):
    pass

def fp8_e4m3_to_f32(bits: np.ndarray) -> np.ndarray:
    raw = bits.astype(np.uint8)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.float32)
    normal = exponent != 0
    value = np.where(
        normal,
        (1.0 + mantissa / 8.0) * np.exp2(exponent - 7),
        (mantissa / 8.0) * np.exp2(-6),
    ).astype(np.float32)
    return sign * value

def e8m0_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.exp2(bits.astype(np.int32) - 127).astype(np.float32)

def pack_a_scale(a_scale: np.ndarray) -> np.ndarray:
    packed = np.zeros(SCALE_BYTES, dtype=np.uint8)
    packed[0:32] = a_scale.reshape(-1)
    return packed

def pack_b_scale(b_scale: np.ndarray) -> np.ndarray:
    packed = np.zeros(SCALE_BYTES, dtype=np.uint8)
    packed[0:32] = b_scale.T.reshape(-1)
    return packed

def make_inputs():
    a_codes = np.array([0x30, 0x38, 0x40, 0xB8], dtype=np.uint8)
    m_idx = np.arange(M).reshape(M, 1)
    k_idx = np.arange(K).reshape(1, K)
    a_matrix = a_codes[(m_idx * 3 + k_idx * 5) % a_codes.size]
    b_matrix = np.full((K, N), 0x38, dtype=np.uint8)
    a_scale_matrix = np.where(
        (np.arange(M).reshape(M, 1) + np.arange(2)) % 2 == 0, 127, 128
    ).astype(np.uint8)
    b_scale_matrix = np.array([[126], [127]], dtype=np.uint8).repeat(N, axis=1)
    return [
        a_matrix.reshape(-1),
        b_matrix.reshape(-1),
        pack_a_scale(a_scale_matrix),
        pack_b_scale(b_scale_matrix),
    ]

def make_expected(a, b, a_scale, b_scale):
    a_matrix = a.reshape(M, K)
    b_matrix = b.reshape(K, N)
    a_scale_matrix = a_scale[:32].reshape(M, K // 32)
    b_scale_matrix = b_scale[:32].reshape(N, K // 32).T
    a_f32 = fp8_e4m3_to_f32(a_matrix)
    b_f32 = fp8_e4m3_to_f32(b_matrix)
    a_scale_f32 = e8m0_to_f32(a_scale_matrix)
    b_scale_f32 = e8m0_to_f32(b_scale_matrix)
    golden = np.zeros((M, N), dtype=np.float32)
    for group in range(K // 32):
        k_slice = slice(group * 32, (group + 1) * 32)
        scaled_a = a_f32[:, k_slice] * a_scale_f32[:, group : group + 1]
        scaled_b = b_f32[k_slice, :] * b_scale_f32[group : group + 1, :]
        golden += scaled_a @ scaled_b
    return golden

CASES = [
    golden_output_case(
        "mad_mx_nonzero_l0",
        mad_mx_nonzero_l0_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1e-2,
        atol=1e-2,
    ),
]

auto_main(globals())
