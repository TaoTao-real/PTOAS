# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

"""Check and classify RMSNorm vector-function UB traffic in final VPTO IR.

The manual AscendC vector function has three vector loads (reduction x, apply
x, and apply gamma), one vector store (apply y), and no vector-scope memory
barrier in its static body.  The GM/UB DataCopy synchronization around the
vector function is intentionally outside this gate.
"""

import argparse
import collections
import re
import sys


EXPECTED_TOTALS = {
    "baseline": (7, 6, 6),
    "compact-scalar": (5, 4, 4),
    "scalar-vreg": (3, 2, 2),
    "dead-store": (3, 1, 1),
    "vf-boundary": (3, 1, 0),
    "row-vf-boundary": (2, 1, 0),
}

MANUAL_VF_BOUNDARY = (3, 1, 0)


def fail(message):
    print("RMSNorm VF traffic gate: " + message, file=sys.stderr)
    return 1


def operation_lines(ir, opname):
    pattern = re.compile(r"^\s*(?:%[^=]+\s*=\s*)?" + re.escape(opname) + r"\b")
    return [line for line in ir.splitlines() if pattern.search(line)]


def distribution(line):
    match = re.search(r'\bdist\s*=\s*"([^"]+)"', line)
    return match.group(1) if match else "NORM"


def barrier_kind(line):
    match = re.search(r'pto\.mem_bar\s+"([^"]+)"', line)
    return match.group(1) if match else "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(EXPECTED_TOTALS), required=True)
    parser.add_argument("input", nargs="?", default="-")
    args = parser.parse_args()

    if args.input == "-":
        ir = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as stream:
            ir = stream.read()

    loads = operation_lines(ir, "pto.vlds")
    stores = operation_lines(ir, "pto.vsts")
    barriers = operation_lines(ir, "pto.mem_bar")
    totals = (len(loads), len(stores), len(barriers))
    expected = EXPECTED_TOTALS[args.stage]
    if totals != expected:
        return fail(
            "stage {} expected VLD/VST/mem_bar = {}/{}/{}, got {}/{}/{}".format(
                args.stage, *expected, *totals
            )
        )

    # A constant for the promoted accumulator UB root would imply that the
    # storage was materialized again.  Its provenance attribute may remain.
    if re.search(r"arith\.constant\s+41728\s*:\s*i64", ir):
        return fail("promoted accumulator UB root 41728 was materialized")
    scalar_promoted = args.stage in {"scalar-vreg", "dead-store", "vf-boundary"}
    if scalar_promoted and re.search(r"arith\.constant\s+42240\s*:\s*i64", ir):
        return fail("promoted scalar UB root 42240 was materialized")

    load_dist = collections.Counter(distribution(line) for line in loads)
    store_dist = collections.Counter(distribution(line) for line in stores)
    barrier_kinds = collections.Counter(barrier_kind(line) for line in barriers)

    # Commit-12 records the exact pre-fix classification.  Later stages retain
    # the total-count gate while their dedicated tests prove each rewrite.
    if args.stage == "baseline":
        expected_load_dist = {"UNPK_B16": 3, "BRC_B32": 1, "NORM": 3}
        expected_store_dist = {"PK_B32": 1, "NORM": 5}
        expected_barriers = {"VST_VST": 2, "VST_VLD": 4}
        if dict(load_dist) != expected_load_dist:
            return fail("unexpected VLD classification: {}".format(dict(load_dist)))
        if dict(store_dist) != expected_store_dist:
            return fail("unexpected VST classification: {}".format(dict(store_dist)))
        if dict(barrier_kinds) != expected_barriers:
            return fail(
                "unexpected mem_bar classification: {}".format(dict(barrier_kinds))
            )

    essential_loads = load_dist["UNPK_B16"]
    essential_stores = store_dist["PK_B32"]
    excess = (
        len(loads) - essential_loads,
        len(stores) - essential_stores,
        len(barriers),
    )

    print("stage={}".format(args.stage))
    print("static_total_vld_vst_membar={}/{}/{}".format(*totals))
    manual_boundary = (
        (2, 1, 0) if args.stage == "row-vf-boundary" else MANUAL_VF_BOUNDARY
    )
    print(
        "manual_vf_boundary_vld_vst_membar={}/{}/{}".format(
            *manual_boundary
        )
    )
    print("essential_boundary_vld_vst={}/{}".format(essential_loads, essential_stores))
    print("excess_vld_vst_membar={}/{}/{}".format(*excess))
    print("accumulator_ub_root_materialized=no")
    if scalar_promoted:
        print("scalar_ub_root_materialized=no")

    if args.stage == "baseline":
        # Each chunk loop has 64 iterations.  Scalar-chain operations execute
        # once per row.  These numbers expose the runtime weight hidden by the
        # compact static operation count.
        print("dynamic_per_row_total_vld_vst_membar=259/132/132")
        print("dynamic_per_row_manual_vf_boundary=192/64/0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
