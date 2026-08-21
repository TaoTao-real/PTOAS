#!/usr/bin/env python3
"""Extract the single RMSNorm kernel sample from an msprof directory."""

import argparse
import csv
from pathlib import Path


def non_placeholder(rows, key):
    return [row for row in rows if row.get(key) not in (None, "", "N/A")]


def read_latest(root, pattern, key):
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise ValueError("missing {} under {}".format(pattern, root))
    with paths[-1].open(newline="") as stream:
        rows = non_placeholder(csv.DictReader(stream), key)
    if len(rows) != 1:
        raise ValueError(
            "expected one non-placeholder row in {}, found {}".format(
                paths[-1], len(rows)
            )
        )
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir")
    args = parser.parse_args()

    root = Path(args.profile_dir)
    task = read_latest(root, "task_time_*.csv", "kernel_name")
    summary = read_latest(root, "op_summary_*.csv", "Op Name")
    fields = (
        task["task_time(us)"],
        summary["Task Duration(us)"],
        summary["aiv_time(us)"],
        summary["aiv_total_cycles"],
        summary["aiv_vec_time(us)"],
        summary.get("aiv_scalar_time(us)", "NA"),
        summary.get("aiv_mte2_time(us)", "NA"),
        summary.get("aiv_mte3_time(us)", "NA"),
    )
    print("\t".join(fields))


if __name__ == "__main__":
    main()
