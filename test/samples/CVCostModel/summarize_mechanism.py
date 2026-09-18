# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Validate four-way paired evidence and decide mechanism benefit."""
import argparse
import json
from pathlib import Path
import statistics

from run_mechanism import ORDERS
from summarize_ab import POLICY, paired_statistics, sha, task_duration


def collect(report):
    rows = []
    for row in report["rows"]:
        runtime = Path(row["runtime"])
        saved = json.loads((runtime / "result.json").read_text())
        if saved != row or row["status"] != "pass" or sha(runtime / "output.bin") != row["output_sha256"]:
            raise ValueError("runtime evidence mismatch")
        for name, digest in row["input_sha256"].items():
            if sha(runtime / name) != digest:
                raise ValueError("input/golden evidence mismatch")
        if row["stage"] == "paired_profile":
            duration, profiler = task_duration(runtime, row["kernel_symbol"], report["device"])
            rows.append(dict(**row, duration_us=duration, profiler=profiler))
    if len(rows) != 80:
        raise ValueError("exactly 80 paired profiler samples are required")
    return rows


def analyze(matrix, case, rows, report, experiment):
    values = {name: [] for name in ("A", "B", "P0", "M")}
    pairs = []
    for block in range(20):
        block_rows = [row for row in rows if row["pair_block"] == block]
        order = ORDERS[block % len(ORDERS)]
        order_id = ",".join(order)
        if (len(block_rows) != 4 or [row["pair_role"] for row in block_rows] != list(order)
                or any(row["pair_order"] != order_id or row["seed"] != block % 3 for row in block_rows)):
            raise ValueError("four-way order, seed, or membership mismatch")
        by_role = {row["pair_role"]: row for row in block_rows}
        sample = dict(block=block, seed=block % 3, order=list(order))
        for name in values:
            value = by_role[name]["duration_us"]
            values[name].append(value)
            sample[name + "_us"] = value
        pairs.append(sample)
    ba = paired_statistics(values["A"], values["B"])
    bp0 = paired_statistics(values["P0"], values["B"])
    mp0 = paired_statistics(values["P0"], values["M"])
    benefit = (ba["mean_gain"] >= POLICY["minimum_speedup"]
               and bp0["mean_gain"] >= POLICY["minimum_speedup"]
               and ba["confidence_interval"][0] > 0 and bp0["confidence_interval"][0] > 0
               and ba["worst_regression"] <= POLICY["maximum_regression"]
               and bp0["worst_regression"] <= POLICY["maximum_regression"]
               and mp0["confidence_interval"][0] >= -0.02
               and mp0["confidence_interval"][1] <= 0.02
               and case["invalid"]["status"] == "rejected"
               and case["invalid"]["reason"] == "INSUFFICIENT_SLOTS")
    selection = case["selection"]
    prediction = next(row["predicted_latency_us"] for row in selection["ranking"]
                      if row["candidate_id"] == selection["recommended_candidate_id"])
    prediction_error = abs(prediction - statistics.mean(values["B"])) / statistics.mean(values["B"])
    correctness = json.loads((experiment / "results" / "correctness.json").read_text())
    if correctness["status"] != "pass" or len(correctness["rows"]) != 24:
        raise ValueError("complete G3 evidence is required")
    return dict(case=case["case"], status="MECHANISM_BENEFIT_PASS" if benefit else "MECHANISM_BENEFIT_FAIL",
                G4="not_claimed", repetitions=case["repetitions"], p_buffer_id=case["p_buffer_id"],
                selected_candidate_id=selection["recommended_candidate_id"],
                selected_preload=next(row["preload_count"] for row in selection["ranking"]
                                      if row["candidate_id"] == selection["recommended_candidate_id"]),
                predicted_B_us=prediction, measured_B_mean_us=statistics.mean(values["B"]),
                prediction_error=prediction_error, statistics={"B_over_A": ba, "B_over_P0": bp0,
                "M_over_P0": mp0}, means_us={key: statistics.mean(value) for key, value in values.items()},
                pairs=pairs, invalid=case["invalid"], all_G3_and_performance_correct=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.experiment / "results" / "performance.json").read_text())
    manifest = json.loads((args.matrix / "manifest.json").read_text())
    if report["status"] != "pass" or report["matrix_sha256"] != sha(args.matrix / "manifest.json"):
        raise ValueError("performance evidence is incomplete or stale")
    rows = collect(report)
    workload = analyze(args.matrix, manifest["cases"][0], rows, report, args.experiment)
    result = dict(schema_version="ptoas.tilesim.mechanism.report.v1", policy=dict(
        minimum_samples=20, minimum_speedup=0.02, maximum_regression=0.02,
        confidence_level=0.95, bootstrap_resamples=10000, bootstrap_seed=20260916,
        multibuffer_only_equivalence_interval=[-0.02, 0.02]), workload=workload,
        source_commit=manifest["provenance"]["commit"],
        tilesim_commit=manifest["provenance"]["tilesim"]["commit"],
        matrix_sha256=report["matrix_sha256"],
        measurement_report_sha256=sha(args.experiment / "results" / "performance.json"),
        measurement_boundary="exactly one MIX_AIC task, Block Num=1, Mix Block Num=2")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stats = workload["statistics"]
    lines = ["# preload + multi-buffer mechanism acceptance", "",
             f"Result: **{workload['status']}** (G4 is not claimed).", "",
             "| Comparison | Mean gain | 95% CI | Worst regression |",
             "|---|---:|---:|---:|"]
    for label, key in (("B/A", "B_over_A"), ("B/P0", "B_over_P0"), ("M/P0", "M_over_P0")):
        row = stats[key]
        lines.append(f"| {label} | {row['mean_gain']:.2%} | "
                     f"[{row['confidence_interval'][0]:.2%}, {row['confidence_interval'][1]:.2%}] | "
                     f"{row['worst_regression']:.2%} |")
    lines.extend(["", f"TileSim predicted B: {workload['predicted_B_us']:.4f} us; "
                  f"measured mean: {workload['measured_B_mean_us']:.4f} us; "
                  f"absolute prediction error: {workload['prediction_error']:.2%}.",
                  "", "P-invalid was rejected by G2 with `INSUFFICIENT_SLOTS`."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
