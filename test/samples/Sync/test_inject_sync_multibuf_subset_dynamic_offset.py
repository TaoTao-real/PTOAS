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
            workspace_ty = pto.TileBufType.get([16, 32], f16, vec, [16, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_inject_sync_multibuf_subset_dynamic_offset_py", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result
                c16 = arith.ConstantOp(idx, 16).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c16, c16], [c16, c1]).result
                sv_in = pto.PartitionViewOp(
                    tile_view_16, tv_in, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                sv_out = pto.PartitionViewOp(
                    tile_view_16, tv_out, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                alloc = pto.AllocTileOp(workspace_ty)
                alloc.operation.attributes["pto.multi_buffer"] = IntegerAttr.get(i32, 2)
                workspace = alloc.result

                # Dynamic offset means V1 cannot statically prove a stable
                # ping/pong slot geometry, so autosync must fall back to the
                # normal single-event path.
                loop = scf.ForOp(c0, c2, c1, [])
                with InsertionPoint(loop.body):
                    slot_off = arith.MulIOp(loop.induction_variable, c16).result
                    slot = pto.SubsetOp(workspace, [c0, slot_off], sizes=[16, 16]).result

                    pto.TLoadOp(None, sv_in, slot)
                    pto.TStoreOp(None, slot, sv_out)

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
