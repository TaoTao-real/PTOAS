# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module, UnitAttr
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i8 = pto.PtrType.get(i8, ctx)
            tv_f32 = pto.TensorViewType.get([128], f32, ctx)
            pv_f32 = pto.PartitionTensorViewType.get([128], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_f32 = pto.TileBufType.get([1, 128], f32, vec, [1, 128], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_i8], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("tprefetch_async_binding_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src_ptr, dst_ptr, workspace_ptr = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c128 = arith.ConstantOp(idx, 128).result

                src_view = pto.MakeTensorViewOp(tv_f32, src_ptr, [c128], [c1]).result
                dst_view = pto.MakeTensorViewOp(tv_f32, dst_ptr, [c128], [c1]).result
                src = pto.PartitionViewOp(pv_f32, src_view, offsets=[c0], sizes=[c128]).result
                dst = pto.PartitionViewOp(pv_f32, dst_view, offsets=[c0], sizes=[c128]).result

                prefetch_ctx = pto.MakePrefetchAsyncContextOp(workspace_ptr).result
                event = pto.TPrefetchAsyncOp(src, prefetch_ctx).result
                session = pto.GetPrefetchAsyncSessionOp(prefetch_ctx).result
                pto.WaitAsyncEventOp(event, session)
                tile = pto.AllocTileOp(tile_f32).result
                pto.TLoadOp(None, src, tile)
                pto.TStoreOp(None, tile, dst)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
