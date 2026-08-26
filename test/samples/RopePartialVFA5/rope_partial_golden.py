#!/usr/bin/env python3
"""Independent FP32/BF16 golden for Partial RoPE HALF and INTERLEAVE."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


ROWS = 8
ACTUAL_COL = 512
ROTARY_COL = 64
BASE_ADDR = 448
MODES = ("half", "interleave")
DATASETS = ("exact", "layout-sensitive")


def f32(values):
    return np.asarray(values, dtype=np.float32)


def f32_to_bf16_bits(values):
    words = f32(values).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + bias) >> np.uint32(16)).astype(np.uint16)


def fixed_inputs(dataset):
    row = np.arange(ROWS, dtype=np.int32)[:, None]
    col = np.arange(ACTUAL_COL, dtype=np.int32)[None, :]
    trig_col = np.arange(ROTARY_COL, dtype=np.int32)[None, :]
    if dataset == "exact":
        x = (((row * 3 + col) % 9) - 4).astype(np.float32) / np.float32(4.0)
        angle = (row + trig_col) % 4
        cos = np.where((angle & 1) == 0, np.float32(1.0), np.float32(0.5))
        sin = np.where((angle & 2) == 0, np.float32(0.25), np.float32(-0.5))
    elif dataset == "layout-sensitive":
        x = (((row * 131 + col * 17) % 257) - 128).astype(np.float32)
        x *= np.where((col % 3) == 0, np.float32(0.03125), np.float32(0.5))
        cos = (((row * 7 + trig_col * 5) % 23) - 11).astype(np.float32) / np.float32(8.0)
        sin = (((row * 11 + trig_col * 3) % 19) - 9).astype(np.float32) / np.float32(16.0)
    else:
        raise ValueError(dataset)
    return f32(x), f32(sin), f32(cos)


def rope_reference(x, sin, cos, mode):
    out_f32 = np.empty_like(x, dtype=np.float32)
    out_f32[:, :BASE_ADDR] = x[:, :BASE_ADDR]
    rotary = x[:, BASE_ADDR : BASE_ADDR + ROTARY_COL]
    if mode == "half":
        half = ROTARY_COL // 2
        x0 = rotary[:, :half]
        x1 = rotary[:, half:]
        # Keep each multiply and add/sub as a separate FP32 rounding point.
        sin0_x1 = np.multiply(sin[:, :half], x1, dtype=np.float32)
        cos0_x0 = np.multiply(cos[:, :half], x0, dtype=np.float32)
        y0 = np.subtract(cos0_x0, sin0_x1, dtype=np.float32)
        sin1_x0 = np.multiply(sin[:, half:], x0, dtype=np.float32)
        cos1_x1 = np.multiply(cos[:, half:], x1, dtype=np.float32)
        y1 = np.add(sin1_x0, cos1_x1, dtype=np.float32)
        out_f32[:, BASE_ADDR:] = np.concatenate((y0, y1), axis=1)
    elif mode == "interleave":
        rotated = np.empty_like(rotary)
        rotated[:, 0::2] = np.negative(rotary[:, 1::2], dtype=np.float32)
        rotated[:, 1::2] = rotary[:, 0::2]
        cos_x = np.multiply(cos, rotary, dtype=np.float32)
        sin_rot = np.multiply(sin, rotated, dtype=np.float32)
        out_f32[:, BASE_ADDR:] = np.add(cos_x, sin_rot, dtype=np.float32)
    else:
        raise ValueError(mode)
    return f32_to_bf16_bits(out_f32)


def rope_scalar_reference(x, sin, cos, mode):
    """Index-by-index oracle, independent of the vectorized formulation."""
    output = np.empty((ROWS, ACTUAL_COL), dtype=np.uint16)
    for row in range(ROWS):
        output[row, :BASE_ADDR] = f32_to_bf16_bits(x[row, :BASE_ADDR])
        if mode == "half":
            half = ROTARY_COL // 2
            for lane in range(half):
                x0 = np.float32(x[row, BASE_ADDR + lane])
                x1 = np.float32(x[row, BASE_ADDR + half + lane])
                y0 = np.subtract(
                    np.multiply(cos[row, lane], x0, dtype=np.float32),
                    np.multiply(sin[row, lane], x1, dtype=np.float32),
                    dtype=np.float32,
                )
                y1 = np.add(
                    np.multiply(sin[row, half + lane], x0, dtype=np.float32),
                    np.multiply(cos[row, half + lane], x1, dtype=np.float32),
                    dtype=np.float32,
                )
                output[row, BASE_ADDR + lane] = f32_to_bf16_bits(y0)
                output[row, BASE_ADDR + half + lane] = f32_to_bf16_bits(y1)
        elif mode == "interleave":
            for lane in range(0, ROTARY_COL, 2):
                x0 = np.float32(x[row, BASE_ADDR + lane])
                x1 = np.float32(x[row, BASE_ADDR + lane + 1])
                for inner, rotated in ((0, np.negative(x1)), (1, x0)):
                    index = lane + inner
                    value = np.add(
                        np.multiply(cos[row, index], x[row, BASE_ADDR + index], dtype=np.float32),
                        np.multiply(sin[row, index], rotated, dtype=np.float32),
                        dtype=np.float32,
                    )
                    output[row, BASE_ADDR + index] = f32_to_bf16_bits(value)
        else:
            raise ValueError(mode)
    return output


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(mode, dataset, write_dir=None):
    x, sin, cos = fixed_inputs(dataset)
    output = rope_reference(x, sin, cos, mode)
    scalar_output = rope_scalar_reference(x, sin, cos, mode)
    if not np.array_equal(output, scalar_output):
        mismatch = np.argwhere(output != scalar_output)[0]
        raise AssertionError(
            f"vector/scalar golden mismatch at row={mismatch[0]} col={mismatch[1]}"
        )
    if write_dir:
        case_dir = Path(write_dir) / mode / dataset
        case_dir.mkdir(parents=True, exist_ok=True)
        x.tofile(case_dir / "input_f32.bin")
        sin.tofile(case_dir / "sin_f32.bin")
        cos.tofile(case_dir / "cos_f32.bin")
        output.tofile(case_dir / "golden_bf16.bin")
    return tuple(map(sha256, (x, sin, cos, output)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--write-dir")
    args = parser.parse_args()
    for mode in (args.mode,) if args.mode else MODES:
        for dataset in (args.dataset,) if args.dataset else DATASETS:
            x_hash, sin_hash, cos_hash, out_hash = run(
                mode, dataset, args.write_dir
            )
            print(
                f"mode={mode} dataset={dataset} input_sha256={x_hash} "
                f"sin_sha256={sin_hash} cos_sha256={cos_hash} "
                f"output_sha256={out_hash}"
            )


if __name__ == "__main__":
    main()
