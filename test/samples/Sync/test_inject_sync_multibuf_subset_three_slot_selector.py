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
            workspace_ty = pto.TileBufType.get([16, 48], f16, vec, [16, 48], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp(
                    "test_inject_sync_multibuf_subset_three_slot_selector_py", fn_ty
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result
                c3 = arith.ConstantOp(idx, 3).result
                c6 = arith.ConstantOp(idx, 6).result
                c16 = arith.ConstantOp(idx, 16).result
                c32 = arith.ConstantOp(idx, 32).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c16, c16], [c16, c1]).result
                sv_in = pto.PartitionViewOp(
                    tile_view_16, tv_in, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                sv_out = pto.PartitionViewOp(
                    tile_view_16, tv_out, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                alloc = pto.AllocTileOp(workspace_ty)
                workspace = alloc.result

                slot0_op = pto.SubViewOp(workspace, [c0, c0], sizes=[16, 16])
                slot1_op = pto.SubViewOp(workspace, [c0, c16], sizes=[16, 16])
                slot2_op = pto.SubViewOp(workspace, [c0, c32], sizes=[16, 16])
                slot0 = slot0_op.result
                slot1 = slot1_op.result
                slot2 = slot2_op.result
                for slot_index, slot_op in enumerate([slot0_op, slot1_op, slot2_op]):
                    slot_op.operation.attributes["pto.multi_buffer_factor"] = IntegerAttr.get(
                        i32, 3
                    )
                    slot_op.operation.attributes["pto.multi_buffer_slot"] = IntegerAttr.get(
                        i32, slot_index
                    )

                loop = scf.ForOp(c0, c6, c1, [])
                with InsertionPoint(loop.body):
                    slot_sel = arith.RemUIOp(loop.induction_variable, c3).result
                    is_slot0 = arith.CmpIOp(arith.CmpIPredicate.eq, slot_sel, c0).result
                    is_slot1 = arith.CmpIOp(arith.CmpIPredicate.eq, slot_sel, c1).result

                    slot_if = scf.IfOp(is_slot0, [slot_ty], hasElse=True)
                    with InsertionPoint(slot_if.then_block):
                        scf.YieldOp([slot0])
                    with InsertionPoint(slot_if.else_block):
                        slot1_if = scf.IfOp(is_slot1, [slot_ty], hasElse=True)
                        with InsertionPoint(slot1_if.then_block):
                            scf.YieldOp([slot1])
                        with InsertionPoint(slot1_if.else_block):
                            scf.YieldOp([slot2])
                        scf.YieldOp([slot1_if.results[0]])

                    selected_slot = slot_if.results[0]
                    # The same loop-carried hazard now rotates over three slots.
                    # Autosync should assign three event lanes and lower them
                    # through a dynamic selector instead of hard-wiring parity.
                    pto.TLoadOp(None, sv_in, selected_slot)
                    pto.TStoreOp(None, selected_slot, sv_out)
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
