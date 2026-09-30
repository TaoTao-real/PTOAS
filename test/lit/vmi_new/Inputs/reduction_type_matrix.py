#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate real verifier/lowering inputs for the reduction type contract."""

import argparse

OPS = ("vcadd", "vcmax", "vcmin")
LOW_PRECISION = (
    "bf16", "f8E4M3", "f8E4M3FN", "f8E4M3FNUZ", "f8E4M3B11FNUZ",
    "f8E5M2", "f8E5M2FNUZ", "!pto.hif8", "!pto.f8E8M0",
)
INTEGERS = tuple(f"{prefix}{width}" for width in (8, 16, 32)
                 for prefix in ("i", "si", "ui"))


def emit(level, invalid):
    print("// Reduction element-type matrix")
    types = LOW_PRECISION + INTEGERS[:3] if invalid else INTEGERS[3:] + ("f16", "f32")
    for element in types:
        width = 16 if element in ("bf16", "f16") else 8
        if element in INTEGERS or element == "f32":
            width = int(element.lstrip("siuf"))
        for op in OPS:
            modes = ("full", "group") if level == "vmi" else ("physical",)
            if not invalid and level == "vmi" and (width == 32 or element == "f16"):
                modes += ("layout",)
            for mode in modes:
                name = f"{op}_{element.replace('!pto.', '')}_{mode}"
                attrs = []
                if level == "vmi":
                    layout = ", #pto.vmi.layout<contiguous>" if mode == "layout" else ""
                    source = f"!pto.vmi.vreg<64x{element}{layout}>"
                    count = 4 if mode == "group" else 1
                    result = f"!pto.vmi.vreg<{count}x{element}{layout}>"
                    granularity = f"b{width}" if mode == "layout" else "pred"
                    mask = f"!pto.vmi.mask<64x{granularity}{layout}>"
                    if mode == "group":
                        attrs.append("group = 4")
                    if op == "vcadd":
                        attrs.append("reassoc")
                    opname = f"pto.vmi.{op}"
                    diagnostic = "requires 16-bit or 32-bit integer, f16, or f32 VMI source element type"
                    if element in INTEGERS[:3]:
                        diagnostic = "VMI-UNSUPPORTED: 8-bit integer reductions are not supported"
                    kind = "f" if element in ("f16", "f32") else "i"
                    grouped = "" if mode == "layout" else "group_"
                    expected = f"pto.vmi.{grouped}reduce_{op[2:]}{kind}"
                else:
                    source = f"!pto.vreg<{2048 // width}x{element}>"
                    mask = f"!pto.mask<b{width}>"
                    result = source
                    if op == "vcadd" and element in INTEGERS and width == 16:
                        result = f"!pto.vreg<64x{element.replace('16', '32')}>"
                    opname = f"pto.{op}"
                    diagnostic = "requires 16-bit or 32-bit integer, f16, or f32 vector element type"
                    expected = opname
                attributes = " {" + ", ".join(attrs) + "}" if attrs else ""
                print("// -----")
                if not invalid:
                    print(f"// CHECK-LABEL: func.func @{name}(")
                    print(f"// CHECK: {expected} ")
                    print(f"// CHECK-SAME: -> {result}")
                print(f"func.func @{name}(%source: {source}, %mask: {mask}) -> {result} {{")
                if invalid:
                    print("  // expected-error@+1 {{'" + opname + "' op " + diagnostic + "}}")
                print(f"  %r = {opname} %source, %mask{attributes} : {source}, {mask} -> {result}")
                print(f"  return %r : {result}\n}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("vmi", "vpto"))
    parser.add_argument("--invalid", action="store_true")
    args = parser.parse_args()
    emit(args.level, args.invalid)
