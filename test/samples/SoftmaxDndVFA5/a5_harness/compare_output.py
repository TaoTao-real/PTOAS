#!/usr/bin/env python3
"""Validate one Softmax output and report deterministic error metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=("exact-onehot", "finite-sensitive"))
    parser.add_argument("--width", required=True, type=int,
                        choices=(32, 64, 128))
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    args = parser.parse_args()

    shape = (4, 16, args.width)
    actual = np.fromfile(args.actual, dtype=np.float32).reshape(shape)
    golden = np.fromfile(args.golden, dtype=np.float32).reshape(shape)
    difference = np.abs(actual - golden)
    exact = Path(args.actual).read_bytes() == Path(args.golden).read_bytes()
    finite = bool(np.all(np.isfinite(actual)))
    normalized_error = float(
        np.max(np.abs(actual.sum(axis=1, dtype=np.float32) - np.float32(1.0)))
    )
    close = bool(np.allclose(
        actual, golden, atol=args.atol, rtol=args.rtol, equal_nan=False
    ))
    passed = exact if args.dataset == "exact-onehot" else finite and close
    result = {
        "actual_sha256": digest(args.actual),
        "golden_sha256": digest(args.golden),
        "byte_exact": exact,
        "finite": finite,
        "max_abs_error": float(np.max(difference)),
        "max_normalization_error": normalized_error,
        "pass": passed,
    }
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
