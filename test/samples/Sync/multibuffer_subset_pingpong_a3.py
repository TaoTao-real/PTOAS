from mlir.ir import Context, InsertionPoint, Location, Module
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
                fn = func.FuncOp("multibuffer_subset_pingpong_a3", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result
                c8 = arith.ConstantOp(idx, 8).result
                c16 = arith.ConstantOp(idx, 16).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c8, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c8, c16], [c16, c1]).result

                alloc = pto.AllocTileOp(workspace_ty)
                alloc.operation.attributes["pto.multi_buffer"] = IntegerAttr.get(i32, 2)
                workspace = alloc.result
                ping = pto.SubsetOp(workspace, [c0, c0], sizes=[16, 16]).result
                pong = pto.SubsetOp(workspace, [c0, c16], sizes=[16, 16]).result

                loop = scf.ForOp(c0, c8, c1, [])
                with InsertionPoint(loop.body):
                    row = loop.induction_variable
                    in_row = pto.PartitionViewOp(
                        tile_view_16, tv_in, offsets=[row, c0], sizes=[c1, c16]
                    ).result
                    out_row = pto.PartitionViewOp(
                        tile_view_16, tv_out, offsets=[row, c0], sizes=[c1, c16]
                    ).result

                    parity = arith.RemUIOp(row, c2).result
                    is_ping = arith.CmpIOp(arith.CmpIPredicate.eq, parity, c0).result

                    slot_if = scf.IfOp(is_ping, [], hasElse=True)
                    with InsertionPoint(slot_if.then_block):
                        pto.TLoadOp(None, in_row, ping)
                        pto.TStoreOp(None, ping, out_row)
                        scf.YieldOp([])
                    with InsertionPoint(slot_if.else_block):
                        pto.TLoadOp(None, in_row, pong)
                        pto.TStoreOp(None, pong, out_row)
                        scf.YieldOp([])

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
