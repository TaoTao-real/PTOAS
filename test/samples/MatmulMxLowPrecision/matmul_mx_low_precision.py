#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.ir import (
    Attribute,
    Context,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    InsertionPoint,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from ptoas.mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            module.operation.attributes["pto.target_arch"] = StringAttr.get("a5")

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)
            f8e4m3fn = Float8E4M3FNType.get()
            f8e5m2 = Float8E5M2Type.get()

            m = 16
            k = 512
            n = 16
            scale_k = k // 32

            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            bias = pto.AddressSpaceAttr.get(pto.AddressSpace.BIAS, ctx)

            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg_left = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )
            cfg_right = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )
            cfg_scale_left = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalMxSize,
                pad,
                ctx,
            )
            cfg_scale_right = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalMxSize,
                pad,
                ctx,
            )
            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pad,
                ctx,
            )
            cfg_bias = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )

            a_ty = pto.TileBufType.get([m, k], f8e4m3fn, left, [m, k], cfg_left, ctx)
            b_ty = pto.TileBufType.get([k, n], f8e5m2, right, [k, n], cfg_right, ctx)
            a_scale_ty = pto.TileBufType.get([m, scale_k], f16, scaling, [m, scale_k], cfg_scale_left, ctx)
            b_scale_ty = pto.TileBufType.get([scale_k, n], f16, scaling, [scale_k, n], cfg_scale_right, ctx)
            bias_ty = pto.TileBufType.get([1, n], f32, bias, [1, n], cfg_bias, ctx)
            c_ty = pto.TileBufType.get([m, n], f32, acc, [m, n], cfg_acc, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("matmul_mx_low_precision", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                fn.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                    "#pto.kernel_kind<cube>", ctx
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                a = pto.AllocTileOp(a_ty).result
                b = pto.AllocTileOp(b_ty).result
                a_scale = pto.AllocTileOp(a_scale_ty).result
                b_scale = pto.AllocTileOp(b_scale_ty).result
                c_in = pto.AllocTileOp(c_ty).result
                bias_tile = pto.AllocTileOp(bias_ty).result
                dst0 = pto.AllocTileOp(c_ty).result
                dst1 = pto.AllocTileOp(c_ty).result
                dst2 = pto.AllocTileOp(c_ty).result

                pto.TMatmulMxOp(None, a, a_scale, b, b_scale, dst0)
                pto.TMatmulMxAccOp(None, c_in, a, a_scale, b, b_scale, dst1)
                pto.TMatmulMxBiasOp(None, a, a_scale, b, b_scale, bias_tile, dst2)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
