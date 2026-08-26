#!/usr/bin/env python3
"""Evaluate paired D/manual-VF task and vector performance gates."""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_gate(rows, metric, threshold, samples=20000):
    by_variant = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]][int(row["repeat"])] = float(row[metric])
    repeats = sorted(set(by_variant["D"]) & set(by_variant["ACF"]))
    ratios = [by_variant["D"][repeat] / by_variant["ACF"][repeat]
              for repeat in repeats]
    if not ratios:
        raise ValueError("no paired D/ACF samples")
    rng = random.Random(0xA5_256)
    bootstrap = [
        statistics.median(rng.choice(ratios) for _ in ratios)
        for _ in range(samples)
    ]
    median = statistics.median(ratios)
    upper = percentile(bootstrap, 0.95)
    return {
        "samples": len(ratios),
        "paired_ratios": ratios,
        "median_ratio": median,
        "bootstrap95_upper": upper,
        "threshold": threshold,
        "pass": median <= threshold and upper <= threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=1.03)
    args = parser.parse_args()

    grouped = defaultdict(list)
    with Path(args.samples).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            grouped[row["mode"]].append(row)

    result = {}
    for mode, rows in sorted(grouped.items()):
        result[mode] = {
            "task": paired_gate(rows, "task_duration_us", args.threshold),
            "vector": paired_gate(rows, "aiv_vec_time_us", args.threshold),
        }
        result[mode]["pass"] = (
            result[mode]["task"]["pass"]
            and result[mode]["vector"]["pass"]
        )
    result["pass"] = bool(result) and all(
        value["pass"] for key, value in result.items() if key != "pass"
    )
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
