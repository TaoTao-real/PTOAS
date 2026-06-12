# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Attribute, Context, IndexType, InsertionPoint, IntegerType, Location, Module, UnitAttr
from mlir.dialects import arith, func, pto


def _mode(name):
    return Attribute.parse(f"#pto.sync_all_mode<{name}>")


def _core_type(name):
    return Attribute.parse(f"#pto.sync_core_type<{name}>")


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            i64 = IntegerType.get_signless(64, ctx)
            idx = IndexType.get(ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            workspace_elems = 48 * 8
            tv_i32 = pto.TensorViewType.get([workspace_elems], i32, ctx)
            pv_i32 = pto.PartitionTensorViewType.get([workspace_elems], i32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            ub_i32 = pto.TileBufType.get([1, 64], i32, vec, [1, 64], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32, i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("syncall_binding_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                gm_workspace_ptr, used_cores = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c384 = arith.ConstantOp(idx, workspace_elems).result
                c0x3000 = arith.ConstantOp(i64, 0x3000).result

                gm_view = pto.MakeTensorViewOp(tv_i32, gm_workspace_ptr, [c384], [c1]).result
                gm_workspace = pto.PartitionViewOp(
                    pv_i32, gm_view, offsets=[c0], sizes=[c384]
                ).result
                ub_workspace = pto.AllocTileOp(ub_i32, addr=c0x3000).result
                pto.syncall(
                    _mode("soft"),
                    _core_type("aiv_only"),
                    gm_workspace=gm_workspace,
                    ub_workspace=ub_workspace,
                    used_cores=used_cores,
                )
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
