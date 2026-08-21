#!/usr/bin/env python3
"""Summarize RMSNorm row-VF profiler samples and the D/B bootstrap interval."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ratio(numerator, denominator, samples=20000):
    rng = random.Random(0xA5_256)
    ratios = []
    for _ in range(samples):
        lhs = [rng.choice(numerator) for _ in numerator]
        rhs = [rng.choice(denominator) for _ in denominator]
        ratios.append(statistics.median(lhs) / statistics.median(rhs))
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def stats(values):
    median = statistics.median(values)
    stddev = statistics.pstdev(values)
    return {
        "median": median,
        "min": min(values),
        "max": max(values),
        "stddev": stddev,
        "cv": stddev / median if median else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    with Path(args.samples).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            grouped[(int(row["rows"]), row["variant"])].append(row)

    summaries = {}
    for key, rows in grouped.items():
        vector = [float(row["aiv_vec_time_us"]) for row in rows]
        task = [float(row["task_duration_us"]) for row in rows]
        aiv = [float(row["aiv_time_us"]) for row in rows]
        cycles = [float(row["aiv_total_cycles"]) for row in rows]
        summaries[key] = {
            "vector": stats(vector),
            "task": stats(task),
            "aiv": stats(aiv),
            "cycles": stats(cycles),
            "samples": len(rows),
        }

    header = [
        "rows", "variant", "samples", "vector_median_us", "vector_min_us",
        "vector_max_us", "vector_stddev_us", "vector_cv", "task_median_us",
        "aiv_median_us", "aiv_total_cycles_median", "vector_us_per_row",
        "speedup_vs_ACU", "speedup_vs_B",
    ]
    output = ["\t".join(header)]
    serializable = {}
    for key in sorted(summaries):
        rows, variant = key
        summary = summaries[key]
        vector = summary["vector"]
        acu = summaries.get((rows, "ACU"), summary)["vector"]["median"]
        candidate = summaries.get((rows, "B"), summary)["vector"]["median"]
        values = [
            rows, variant, summary["samples"], vector["median"], vector["min"],
            vector["max"], vector["stddev"], vector["cv"],
            summary["task"]["median"], summary["aiv"]["median"],
            summary["cycles"]["median"], vector["median"] / rows,
            acu / vector["median"], candidate / vector["median"],
        ]
        output.append("\t".join(str(value) for value in values))
        serializable["{}:{}".format(rows, variant)] = summary

    if (64, "D") in grouped and (64, "B") in grouped:
        d_values = [float(row["aiv_vec_time_us"]) for row in grouped[(64, "D")]]
        b_values = [float(row["aiv_vec_time_us"]) for row in grouped[(64, "B")]]
        low, high = bootstrap_ratio(d_values, b_values)
        serializable["n64_d_over_b_bootstrap95"] = {"low": low, "high": high}

    Path(args.output_tsv).write_text("\n".join(output) + "\n")
    Path(args.output_json).write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
