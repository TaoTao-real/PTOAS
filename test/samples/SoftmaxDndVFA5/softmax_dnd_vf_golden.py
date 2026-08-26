#!/usr/bin/env python3
"""Independent FP32 golden for the three-width Dn Softmax VF fixture."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


BATCH = 4
REDUCE = 16
WIDTHS = (32, 64, 128)
DATASETS = ("exact-onehot", "finite-sensitive")


def f32(values):
    return np.asarray(values, dtype=np.float32)


def fixed_input(width, dataset):
    tile = np.arange(BATCH, dtype=np.int32)[:, None, None]
    row = np.arange(REDUCE, dtype=np.int32)[None, :, None]
    col = np.arange(width, dtype=np.int32)[None, None, :]
    if dataset == "exact-onehot":
        winner = (tile * 7 + col * 3) % REDUCE
        values = np.full((BATCH, REDUCE, width), -np.inf, dtype=np.float32)
        values[row == winner] = np.float32(0.0)
        return values
    if dataset == "finite-sensitive":
        encoded = ((tile * 31 + row * 11 + col * 5) % 29) - 14
        return np.multiply(encoded.astype(np.float32), np.float32(0.125), dtype=np.float32)
    raise ValueError("unsupported dataset: {}".format(dataset))


def paired_max(values):
    even = f32(values[0]).copy()
    odd = f32(values[1]).copy()
    for row in range(2, REDUCE, 2):
        even = np.maximum(even, f32(values[row])).astype(np.float32)
        odd = np.maximum(odd, f32(values[row + 1])).astype(np.float32)
    return np.maximum(even, odd).astype(np.float32)


def paired_sum(values):
    even = f32(values[0]).copy()
    odd = f32(values[1]).copy()
    for row in range(2, REDUCE, 2):
        even = np.add(even, f32(values[row]), dtype=np.float32)
        odd = np.add(odd, f32(values[row + 1]), dtype=np.float32)
    return np.add(even, odd, dtype=np.float32)


def softmax_dnd(values):
    values = f32(values)
    result = np.empty_like(values)
    for tile in range(BATCH):
        maximum = paired_max(values[tile])
        exponent = np.exp(
            np.subtract(values[tile], maximum[None, :], dtype=np.float32),
            dtype=np.float32,
        )
        denominator = paired_sum(exponent)
        result[tile] = np.divide(
            exponent, denominator[None, :], dtype=np.float32
        )
    return result


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_case(width, dataset, write_dir=None):
    source = fixed_input(width, dataset)
    output = softmax_dnd(source)
    if write_dir is not None:
        case_dir = Path(write_dir) / "n{}".format(width) / dataset
        case_dir.mkdir(parents=True, exist_ok=True)
        source.tofile(case_dir / "x.bin")
        np.zeros_like(output).tofile(case_dir / "y.init.bin")
        output.tofile(case_dir / "golden.bin")
    return sha256(source), sha256(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--write-dir")
    args = parser.parse_args()
    widths = (args.width,) if args.width else WIDTHS
    datasets = (args.dataset,) if args.dataset else DATASETS
    for width in widths:
        for dataset in datasets:
            input_hash, output_hash = run_case(width, dataset, args.write_dir)
            print(
                "width={} dataset={} input_sha256={} output_sha256={}".format(
                    width, dataset, input_hash, output_hash
                )
            )


if __name__ == "__main__":
    main()
