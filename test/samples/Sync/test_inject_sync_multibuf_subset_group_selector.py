# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import arith, func, scf, pto
from mlir.ir import F16Type, IndexType, IntegerAttr, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tile_view_16 = pto.PartitionTensorViewType.get([16, 16], f16, ctx)

            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            slot_ty = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)
            workspace_ty = pto.TileBufType.get([16, 64], f16, vec, [16, 64], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp(
                    "test_inject_sync_multibuf_subset_group_selector_py", fn_ty
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result
                c4 = arith.ConstantOp(idx, 4).result
                c16 = arith.ConstantOp(idx, 16).result
                c32 = arith.ConstantOp(idx, 32).result
                c48 = arith.ConstantOp(idx, 48).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c16, c16], [c16, c1]).result
                sv_in = pto.PartitionViewOp(
                    tile_view_16, tv_in, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                sv_out = pto.PartitionViewOp(
                    tile_view_16, tv_out, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                workspace = pto.AllocTileOp(workspace_ty).result

                group0_slot0_op = pto.SubViewOp(workspace, [c0, c0], sizes=[16, 16])
                group0_slot1_op = pto.SubViewOp(workspace, [c0, c16], sizes=[16, 16])
                group1_slot0_op = pto.SubViewOp(workspace, [c0, c32], sizes=[16, 16])
                group1_slot1_op = pto.SubViewOp(workspace, [c0, c48], sizes=[16, 16])

                group0_slot0 = group0_slot0_op.result
                group0_slot1 = group0_slot1_op.result
                group1_slot0 = group1_slot0_op.result
                group1_slot1 = group1_slot1_op.result

                for group, slot0_op, slot1_op in [
                    (0, group0_slot0_op, group0_slot1_op),
                    (1, group1_slot0_op, group1_slot1_op),
                ]:
                    slot0_op.operation.attributes["pto.multi_buffer_factor"] = IntegerAttr.get(
                        i32, 2
                    )
                    slot0_op.operation.attributes["pto.multi_buffer_group"] = IntegerAttr.get(
                        i32, group
                    )
                    slot0_op.operation.attributes["pto.multi_buffer_slot"] = IntegerAttr.get(
                        i32, 0
                    )
                    slot1_op.operation.attributes["pto.multi_buffer_factor"] = IntegerAttr.get(
                        i32, 2
                    )
                    slot1_op.operation.attributes["pto.multi_buffer_group"] = IntegerAttr.get(
                        i32, group
                    )
                    slot1_op.operation.attributes["pto.multi_buffer_slot"] = IntegerAttr.get(
                        i32, 1
                    )

                loop = scf.ForOp(c0, c4, c1, [])
                with InsertionPoint(loop.body):
                    parity = arith.RemUIOp(loop.induction_variable, c2).result
                    is_slot0 = arith.CmpIOp(arith.CmpIPredicate.eq, parity, c0).result

                    group0_if = scf.IfOp(is_slot0, [slot_ty], hasElse=True)
                    with InsertionPoint(group0_if.then_block):
                        scf.YieldOp([group0_slot0])
                    with InsertionPoint(group0_if.else_block):
                        scf.YieldOp([group0_slot1])
                    selected_group0 = group0_if.results[0]

                    group1_if = scf.IfOp(is_slot0, [slot_ty], hasElse=True)
                    with InsertionPoint(group1_if.then_block):
                        scf.YieldOp([group1_slot0])
                    with InsertionPoint(group1_if.else_block):
                        scf.YieldOp([group1_slot1])
                    selected_group1 = group1_if.results[0]

                    pto.TLoadOp(None, sv_in, selected_group0)
                    pto.TStoreOp(None, selected_group0, sv_out)
                    pto.TLoadOp(None, sv_in, selected_group1)
                    pto.TStoreOp(None, selected_group1, sv_out)
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
