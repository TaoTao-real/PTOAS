# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

"""Validate final Softmax Dn VPTO structure for one static width."""

import argparse
import re
import sys


def count_ops(ir, name):
    pattern = re.compile(
        r"^\s*(?:%[^=]+\s*=\s*)?" + re.escape(name) + r"\b", re.MULTILINE
    )
    return len(pattern.findall(ir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--loops", type=int, required=True)
    parser.add_argument("--vld", type=int, required=True)
    parser.add_argument("--vst", type=int, required=True)
    parser.add_argument("--membar", type=int, required=True)
    args = parser.parse_args()
    ir = sys.stdin.read()
    actual = (
        count_ops(ir, "scf.for"),
        count_ops(ir, "pto.vlds"),
        count_ops(ir, "pto.vsts"),
        count_ops(ir, "pto.mem_bar"),
    )
    expected = (args.loops, args.vld, args.vst, args.membar)
    if actual != expected:
        print(
            "Softmax N={} expected loop/VLD/VST/membar={}/{}/{}/{}, "
            "got {}/{}/{}/{}".format(args.width, *expected, *actual),
            file=sys.stderr,
        )
        return 1
    expected_expdif = 16 if args.width == 128 else 8
    if count_ops(ir, "pto.vexpdif") != expected_expdif:
        print("unexpected vexpdif count", file=sys.stderr)
        return 1
    if count_ops(ir, "pto.copy_ubuf_to_gm") != 4:
        print("observable output stores were not preserved", file=sys.stderr)
        return 1
    if re.search(r"^\s*(?:%[^=]+\s*=\s*)?pto\.vmi\.v", ir, re.MULTILINE):
        print("residual VMI vector operation", file=sys.stderr)
        return 1
    print(
        "width={} loop_vld_vst_membar={}/{}/{}/{} expdif={} outputs=4".format(
            args.width, *actual, expected_expdif
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
