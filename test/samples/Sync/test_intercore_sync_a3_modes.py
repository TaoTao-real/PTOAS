#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, IndexType, InsertionPoint, IntegerType, Location, MemRefType, Module, UnitAttr
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()

            idx = IndexType.get(ctx)
            i64 = IntegerType.get_signless(64, ctx)
            ffts_ty = MemRefType.get([256], i64)
            fn_ty = func.FunctionType.get([ffts_ty, idx], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a3_modes", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pto.set_ffts(entry.arguments[0])
                pto.sync_set(pipe_mte3, 3, ffts_mode=0)
                pto.sync_set(pipe_mte3, entry.arguments[1], ffts_mode=1)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
