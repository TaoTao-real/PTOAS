#!/usr/bin/env python3
"""Evaluate paired generic-D/legacy-D RMSNorm performance gates."""

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


def paired_gate(rows, metric, candidate, baseline, threshold, samples=20000):
    by_variant = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]][int(row["repeat"])] = float(row[metric])
    repeats = sorted(set(by_variant[candidate]) & set(by_variant[baseline]))
    ratios = [
        by_variant[candidate][repeat] / by_variant[baseline][repeat]
        for repeat in repeats
    ]
    if not ratios:
        raise ValueError(f"no paired {candidate}/{baseline} samples")
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
    parser.add_argument("--candidate", default="D")
    parser.add_argument("--baseline", default="DL")
    parser.add_argument("--threshold", type=float, default=1.03)
    args = parser.parse_args()

    grouped = defaultdict(list)
    with Path(args.samples).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            grouped[int(row["rows"])].append(row)

    result = {}
    for rows, records in sorted(grouped.items()):
        result[str(rows)] = {
            "task": paired_gate(
                records,
                "task_duration_us",
                args.candidate,
                args.baseline,
                args.threshold,
            ),
            "vector": paired_gate(
                records,
                "aiv_vec_time_us",
                args.candidate,
                args.baseline,
                args.threshold,
            ),
        }
        result[str(rows)]["pass"] = (
            result[str(rows)]["task"]["pass"]
            and result[str(rows)]["vector"]["pass"]
        )
    result["candidate"] = args.candidate
    result["baseline"] = args.baseline
    result["pass"] = bool(grouped) and all(
        result[str(rows)]["pass"] for rows in grouped
    )
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
