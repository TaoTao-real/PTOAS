# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Keep micro and VMI vcadd constructors aligned with the IR type contract."""

import logging

from ptoas.mlir.dialects import func
from ptoas.mlir.dialects import pto as pto_ir
from ptoas.mlir.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    Location,
    Module,
    Type,
)

from ptodsl import pto

LOGGER = logging.getLogger(__name__)


def build_vcadd(element, width, expected):
    source = Type.parse(f"!pto.vreg<{2048 // width}x{element}>")
    mask = Type.parse(f"!pto.mask<b{width}>")
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([source, mask], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        result = pto.vcadd(*entry.arguments)
        assert str(result.type) == expected, (element, result.type, expected)
        func.ReturnOp([])
    assert module.operation.verify()


def build_vmi_vcadd(element, *, group=None, reassoc=None):
    source = Type.parse(f"!pto.vmi.vreg<8x{element}>")
    mask = Type.parse("!pto.vmi.mask<8xpred>")
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([source, mask], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        kwargs = {"group": group}
        if reassoc is not None:
            kwargs["reassoc"] = reassoc
        result = pto.vmi.vcadd(*entry.arguments, **kwargs)
        assert str(result.type) == f"!pto.vmi.vreg<{group or 1}x{element}>"
        func.ReturnOp([])
    assert module.operation.verify()


def check_unsupported_floats():
    element, width = "bf16", 16
    try:
        build_vcadd(element, width, f"!pto.vreg<{2048 // width}x{element}>")
    except TypeError as error:
        assert "f16, or f32 source vector" in str(error)
    else:
        raise AssertionError(f"pto.vcadd accepted unsupported {element}")
    for group in (None, 2, 8):
        for reassoc in (None, True):
            try:
                build_vmi_vcadd(element, group=group, reassoc=reassoc)
            except TypeError as error:
                assert "f16, or f32 source vector" in str(error)
            else:
                raise AssertionError(f"pto.vmi.vcadd accepted unsupported {element}")


def main():
    with Context() as context, Location.unknown():
        pto_ir.register_dialect(context)
        for element in ("i8", "si8", "ui8"):
            try:
                build_vcadd(element, 8, f"!pto.vreg<128x{element.replace('8', '16')}>")
            except TypeError as error:
                assert "requires a 16-bit or 32-bit integer, f16, or f32 source vector" in str(error)
            else:
                raise AssertionError(f"pto.vcadd accepted unsupported {element}")
        for element, width, expected in (
            ("i16", 16, "!pto.vreg<64xi32>"),
            ("si16", 16, "!pto.vreg<64xsi32>"),
            ("ui16", 16, "!pto.vreg<64xui32>"),
            ("i32", 32, "!pto.vreg<64xi32>"),
            ("si32", 32, "!pto.vreg<64xsi32>"),
            ("ui32", 32, "!pto.vreg<64xui32>"),
            ("f16", 16, "!pto.vreg<128xf16>"),
            ("f32", 32, "!pto.vreg<64xf32>"),
        ):
            build_vcadd(element, width, expected)
            for group in (None, 2, 8):
                build_vmi_vcadd(
                    element,
                    group=group,
                    reassoc=True if element in ("f16", "f32") else None,
                )
        check_unsupported_floats()
    LOGGER.info(
        "vcadd types: micro 4 rejected/8 supported; VMI 6 rejected/24 supported"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
