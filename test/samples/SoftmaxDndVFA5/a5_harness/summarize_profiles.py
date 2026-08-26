#!/usr/bin/env python3
"""Summarize Base32/Base64/Base128 A5 Softmax profile samples."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


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
    parser.add_argument("samples", nargs="+")
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for sample in args.samples:
        with Path(sample).open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                grouped[(int(row["width"]), row["variant"])].append(row)

    summaries = {}
    for key, rows in grouped.items():
        summaries[key] = {
            "vector": stats([float(row["aiv_vec_time_us"]) for row in rows]),
            "task": stats([float(row["task_duration_us"]) for row in rows]),
            "aiv": stats([float(row["aiv_time_us"]) for row in rows]),
            "cycles": stats([float(row["aiv_total_cycles"]) for row in rows]),
            "samples": len(rows),
        }

    header = (
        "width variant samples vector_median_us vector_min_us vector_max_us "
        "vector_stddev_us vector_cv task_median_us aiv_median_us "
        "aiv_total_cycles_median speedup_vs_ACU speedup_vs_B"
    ).split()
    output = ["\t".join(header)]
    serializable = {}
    for (width, variant), summary in sorted(summaries.items()):
        vector = summary["vector"]
        acu = summaries.get((width, "ACU"))
        baseline = summaries.get((width, "B"))
        values = (
            width,
            variant,
            summary["samples"],
            vector["median"],
            vector["min"],
            vector["max"],
            vector["stddev"],
            vector["cv"],
            summary["task"]["median"],
            summary["aiv"]["median"],
            summary["cycles"]["median"],
            (acu["vector"]["median"] / vector["median"]
             if acu is not None else "NA"),
            (baseline["vector"]["median"] / vector["median"]
             if baseline is not None else "NA"),
        )
        output.append("\t".join(str(value) for value in values))
        serializable["{}:{}".format(width, variant)] = summary

    Path(args.output_tsv).write_text("\n".join(output) + "\n")
    Path(args.output_json).write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
