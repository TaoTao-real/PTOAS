# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import arith, func, pto
from mlir.ir import F16Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            u64 = IntegerType.get_unsigned(64, ctx)
            idx = IndexType.get(ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)
            ptr_i8 = pto.PtrType.get(i8, ctx)
            dst_tv_ty = pto.TensorViewType.get([16, 32], i8, ctx)
            dst_pt_ty = pto.PartitionTensorViewType.get([16, 32], i8, ctx)

            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg_vec = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )
            cfg_fp = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )

            src_tile_ty = pto.TileBufType.get([16, 32], f16, vec, [16, 32], cfg_vec, ctx)
            fp_tile_ty = pto.TileBufType.get([1, 16], u64, scaling, [1, 16], cfg_fp, ctx)

            fn_ty = func.FunctionType.get([ptr_i8], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tstore_fp_invalid_vec_f16_to_i8", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr = entry.arguments[0]
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c16 = arith.ConstantOp(idx, 16).result
                c32 = arith.ConstantOp(idx, 32).result
                dst_tv = pto.MakeTensorViewOp(dst_tv_ty, dst_ptr, [c16, c32], [c32, c1]).result
                dst = pto.PartitionViewOp(dst_pt_ty, dst_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                src_tile = pto.AllocTileOp(src_tile_ty).result
                fp_tile = pto.AllocTileOp(fp_tile_ty).result
                pto.TStoreFPOp(src_tile, fp_tile, dst)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            raise SystemExit(1)


if __name__ == "__main__":
    print(build())
