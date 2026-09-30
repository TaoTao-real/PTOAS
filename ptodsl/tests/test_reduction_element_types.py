#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise both public DSL reduction entry points with real MLIR values."""

from itertools import product
import logging

from ptoas.mlir import ir
from ptoas.mlir.dialects import func, pto as dialect
from ptodsl import pto

OPS = ("vcadd", "vcmax", "vcmin")
LOW_PRECISION = (
    "bf16", "f8E4M3", "f8E4M3FN", "f8E4M3FNUZ", "f8E4M3B11FNUZ",
    "f8E5M2", "f8E5M2FNUZ", "!pto.hif8", "!pto.f8E8M0",
)
INTEGERS = tuple(f"{prefix}{width}" for width in (8, 16, 32)
                 for prefix in ("i", "si", "ui"))


def reduction_signature(level, op, element, group):
    width = 16 if element in ("bf16", "f16") else 8
    if element in INTEGERS or element == "f32":
        width = int(element.lstrip("siuf"))
    if level == "vmi":
        return (
            f"!pto.vmi.vreg<64x{element}>",
            "!pto.vmi.mask<64xpred>",
            f"!pto.vmi.vreg<{group or 1}x{element}>",
        )
    source = f"!pto.vreg<{2048 // width}x{element}>"
    expected = source
    if op == "vcadd" and element in INTEGERS and width == 16:
        expected = f"!pto.vreg<64x{element.replace('16', '32')}>"
    return source, f"!pto.mask<b{width}>", expected


def check_rejection(block, opname, element, exc):
    message = str(exc)
    assert f"{opname}(...)" in message, message
    if opname.startswith("pto.vmi.") and element in INTEGERS[:3]:
        assert isinstance(exc, ValueError), message
        assert "8-bit integer reductions are not supported" in message, message
    else:
        assert isinstance(exc, TypeError), message
        assert element in message and "f16, or f32 source vector" in message, message
    assert len(list(block.operations)) == 0, "invalid reduction emitted IR"


def probe(level, op, element, group=None, *, invalid=False):
    source, mask, expected = reduction_signature(level, op, element, group)
    opname = f"pto.vmi.{op}" if level == "vmi" else f"pto.{op}"
    namespace = pto.vmi if level == "vmi" else pto
    kwargs = {"group": group} if level == "vmi" else {}
    if level == "vmi" and op == "vcadd":
        kwargs["reassoc"] = True
    with ir.Context() as ctx, ir.Location.unknown():
        dialect.register_dialect(ctx)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("probe", ([ir.Type.parse(source), ir.Type.parse(mask)], []))
        block = fn.add_entry_block()
        with ir.InsertionPoint(block):
            try:
                result = getattr(namespace, op)(*block.arguments, **kwargs)
            except (TypeError, ValueError) as exc:
                assert invalid, (level, op, element, str(exc))
                check_rejection(block, opname, element, exc)
            else:
                assert not invalid, f"{level}.{op} accepted {element}"
                assert str(result.type) == expected, (result.type, expected)
            func.ReturnOp([])
        module.operation.verify()
        if not invalid:
            assert opname + " " in str(module), str(module)


def main():
    rejected = accepted = 0
    levels = (("vmi", None), ("vmi", 4), ("vpto", None))
    invalid_types = LOW_PRECISION + INTEGERS[:3]
    valid_types = INTEGERS[3:] + ("f16", "f32")
    for (level, group), op in product(levels, OPS):
        for element in invalid_types:
            probe(level, op, element, group, invalid=True)
            rejected += 1
        for element in valid_types:
            probe(level, op, element, group)
            accepted += 1
    logging.info("reduction type contracts: %d rejected, %d accepted", rejected, accepted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
