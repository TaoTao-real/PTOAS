#!/usr/bin/env python3
"""Independent BF16 golden for the static multi-row one-VL RMSNorm fixture."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


COLS = 64
EPSILON = np.float32(1.0e-6)
SUPPORTED_ROWS = (1, 8, 32, 64)
DATASETS = ("exact-association", "layout-sensitive")


def f32(values):
    return np.asarray(values, dtype=np.float32)


def f32_to_bf16_bits(values):
    words = f32(values).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(values):
    words = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def fixed_inputs(rows, dataset):
    row = np.arange(rows, dtype=np.int32)[:, None]
    col = np.arange(COLS, dtype=np.int32)[None, :]
    if dataset == "exact-association":
        magnitude = np.float32(1.0) + ((row + col) % 4).astype(np.float32)
        sign = np.where(((row * 3 + col) & 1) == 0, np.float32(1.0), np.float32(-1.0))
        x = np.multiply(magnitude, sign, dtype=np.float32)
        gamma = np.float32(0.5) + (np.arange(COLS) % 8).astype(np.float32) / np.float32(8.0)
    elif dataset == "layout-sensitive":
        encoded = ((row * 17 + col * 5) % 17) - 8
        x = encoded.astype(np.float32)
        x[x == 0.0] = np.float32(0.5)
        gamma = np.float32(0.25) + (np.arange(COLS) % 13).astype(np.float32) / np.float32(8.0)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return f32_to_bf16_bits(x), f32_to_bf16_bits(gamma.reshape(1, COLS))


def reduce_one_vl_exact(values):
    """Fixed FP32 pair tree; selected datasets make every add exact."""
    work = f32(values).copy()
    distance = 1
    while distance < COLS:
        for base in range(0, COLS, distance * 2):
            work[base] = np.add(work[base], work[base + distance], dtype=np.float32)
        distance *= 2
    return np.float32(work[0])


def rmsnorm_one_vl(x_bits, gamma_bits):
    x = bf16_bits_to_f32(x_bits)
    gamma = bf16_bits_to_f32(gamma_bits)[0]
    output = np.empty(x.shape, dtype=np.uint16)
    for row in range(x.shape[0]):
        square = np.multiply(x[row], x[row], dtype=np.float32)
        total = reduce_one_vl_exact(square)
        mean = np.multiply(total, np.float32(1.0 / COLS), dtype=np.float32)
        mean_eps = np.add(mean, EPSILON, dtype=np.float32)
        divisor = np.sqrt(mean_eps, dtype=np.float32)
        normalized = np.divide(x[row], divisor, dtype=np.float32)
        scaled = np.multiply(normalized, gamma, dtype=np.float32)
        output[row] = f32_to_bf16_bits(scaled)
    return output


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_case(rows, dataset, write_dir=None):
    x_bits, gamma_bits = fixed_inputs(rows, dataset)
    output = rmsnorm_one_vl(x_bits, gamma_bits)
    if write_dir is not None:
        case_dir = Path(write_dir) / f"n{rows}" / dataset
        case_dir.mkdir(parents=True, exist_ok=True)
        x_bits.tofile(case_dir / "x.bin")
        gamma_bits.tofile(case_dir / "gamma.bin")
        output.tofile(case_dir / "golden.bin")
    return sha256(x_bits), sha256(gamma_bits), sha256(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=SUPPORTED_ROWS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--write-dir")
    args = parser.parse_args()
    rows_set = (args.rows,) if args.rows else SUPPORTED_ROWS
    datasets = (args.dataset,) if args.dataset else DATASETS
    for rows in rows_set:
        for dataset in datasets:
            x_hash, gamma_hash, output_hash = run_case(rows, dataset, args.write_dir)
            print(
                f"rows={rows} dataset={dataset} x_sha256={x_hash} "
                f"gamma_sha256={gamma_hash} output_sha256={output_hash}"
            )


if __name__ == "__main__":
    main()

