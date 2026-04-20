# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Elementwise type conversion: f32 -> i16 using pto.tcvt with tmp tile and
explicit saturation mode.

Kernel signature:
    tcvt_kernel_2d(src: ptr<f32>, dst: ptr<i16>)

Pipeline:
    make_tensor_view -> partition_view -> alloc_tile -> tload -> tcvt -> tstore
"""

from mlir.ir import (
    Attribute,
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    UnitAttr,
)
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i16 = IntegerType.get_signless(16, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i16 = pto.PtrType.get(i16, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_i16 = pto.TensorViewType.get(2, i16, ctx)

            part_view_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            part_view_i16 = pto.PartitionTensorViewType.get([32, 32], i16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            rmode_attr = pto.RoundModeAttr.get(pto.RoundMode.TRUNC, ctx)
            sat_attr = pto.SaturationModeAttr.get(pto.SaturationMode.ON, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tile_buf_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            tile_buf_i16 = pto.TileBufType.get([32, 32], i16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_i16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tcvt_kernel_2d", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                fn.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                    "#pto.kernel_kind<vector>", ctx
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg_src, arg_dst = entry.arguments

                tv_src = pto.MakeTensorViewOp(tv2_f32, arg_src, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_i16, arg_dst, [c32, c32], [c32, c1]).result

                sv_src = pto.PartitionViewOp(
                    part_view_f32,
                    tv_src,
                    offsets=[c0, c0],
                    sizes=[c32, c32],
                ).result
                sv_dst = pto.PartitionViewOp(
                    part_view_i16,
                    tv_dst,
                    offsets=[c0, c0],
                    sizes=[c32, c32],
                ).result

                tb_src = pto.AllocTileOp(tile_buf_f32).result
                tb_dst = pto.AllocTileOp(tile_buf_i16).result
                tb_tmp = pto.AllocTileOp(tile_buf_f32).result

                pto.TLoadOp(None, sv_src, tb_src)
                pto.TCvtOp(tb_src, tb_dst, tmp=tb_tmp, rmode=rmode_attr, sat_mode=sat_attr)
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
