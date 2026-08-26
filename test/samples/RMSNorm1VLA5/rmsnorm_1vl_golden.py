#!/usr/bin/env python3
"""Independent BF16 RMSNorm golden with the accepted 64-lane association."""

import argparse
import hashlib
from pathlib import Path
import numpy as np

ROWS = 8
COLS = 4096
VL = 64
EPSILON = np.float32(1.0e-6)


def f32_to_bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(values):
    words = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def fixed_inputs():
    linear = np.arange(ROWS * COLS, dtype=np.int32).reshape(ROWS, COLS)
    x = ((linear % 251) - 125).astype(np.float32) / np.float32(32.0)
    x[x == 0.0] = np.float32(0.03125)
    gamma_index = np.arange(COLS, dtype=np.int32)
    gamma = np.float32(0.75) + (gamma_index % 41).astype(np.float32) / np.float32(64.0)
    return f32_to_bf16_bits(x), f32_to_bf16_bits(gamma)


def rmsnorm_lane_accum(x_bits, gamma_bits):
    x = bf16_bits_to_f32(x_bits)
    gamma = bf16_bits_to_f32(gamma_bits)
    output = np.empty((ROWS, COLS), dtype=np.uint16)
    for row in range(ROWS):
        lanes = np.zeros(VL, dtype=np.float32)
        for col in range(0, COLS, VL):
            chunk = x[row, col : col + VL]
            lanes = np.add(lanes, np.multiply(chunk, chunk, dtype=np.float32), dtype=np.float32)
        total = np.sum(lanes, dtype=np.float32)
        divisor = np.sqrt(np.add(np.multiply(total, np.float32(1.0 / COLS), dtype=np.float32), EPSILON, dtype=np.float32), dtype=np.float32)
        for col in range(0, COLS, VL):
            normalized = np.divide(x[row, col : col + VL], divisor, dtype=np.float32)
            scaled = np.multiply(normalized, gamma[col : col + VL], dtype=np.float32)
            output[row, col : col + VL] = f32_to_bf16_bits(scaled)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-dir")
    args = parser.parse_args()
    x_bits, gamma_bits = fixed_inputs()
    output = rmsnorm_lane_accum(x_bits, gamma_bits)
    if args.write_dir:
        output_dir = Path(args.write_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        x_bits.tofile(output_dir / "x.bin")
        gamma_bits.tofile(output_dir / "gamma.bin")
        np.zeros_like(output).tofile(output_dir / "y.init.bin")
        output.tofile(output_dir / "golden.bin")
    print("x_sha256=" + hashlib.sha256(x_bits.tobytes()).hexdigest())
    print("gamma_sha256=" + hashlib.sha256(gamma_bits.tobytes()).hexdigest())
    print("output_sha256=" + hashlib.sha256(output.tobytes()).hexdigest())


if __name__ == "__main__":
    main()
