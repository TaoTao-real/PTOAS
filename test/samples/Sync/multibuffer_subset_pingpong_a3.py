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
                c16 = arith.ConstantOp(idx, 16).result
                c32 = arith.ConstantOp(idx, 32).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c16, c16], [c16, c1]).result

                alloc = pto.AllocTileOp(workspace_ty)
                alloc.operation.attributes["pto.multi_buffer"] = IntegerAttr.get(i32, 2)
                workspace = alloc.result
                # This A3-friendly sample uses explicit ping/pong branches, so
                # the expected lowering is slot-bound static event ids instead
                # of dynamic MTE3->MTE2 event selection.
                ping = pto.SubViewOp(workspace, [c0, c0], sizes=[16, 16]).result
                pong = pto.SubViewOp(workspace, [c0, c16], sizes=[16, 16]).result

                loop = scf.ForOp(c0, c2, c1, [])
                with InsertionPoint(loop.body):
                    parity = arith.RemUIOp(loop.induction_variable, c2).result
                    is_ping = arith.CmpIOp(arith.CmpIPredicate.eq, parity, c0).result

                    slot_if = scf.IfOp(is_ping, [], hasElse=True)
                    with InsertionPoint(slot_if.then_block):
                        in_tile = pto.PartitionViewOp(
                            tile_view_16,
                            tv_in,
                            offsets=[c0, c0],
                            sizes=[c16, c16],
                        ).result
                        out_tile = pto.PartitionViewOp(
                            tile_view_16,
                            tv_out,
                            offsets=[c0, c0],
                            sizes=[c16, c16],
                        ).result
                        pto.TLoadOp(None, in_tile, ping)
                        pto.TStoreOp(None, ping, out_tile)
                        scf.YieldOp([])
                    with InsertionPoint(slot_if.else_block):
                        in_tile = pto.PartitionViewOp(
                            tile_view_16,
                            tv_in,
                            offsets=[c0, c0],
                            sizes=[c16, c16],
                        ).result
                        out_tile = pto.PartitionViewOp(
                            tile_view_16,
                            tv_out,
                            offsets=[c0, c0],
                            sizes=[c16, c16],
                        ).result
                        pto.TLoadOp(None, in_tile, pong)
                        pto.TStoreOp(None, pong, out_tile)
                        scf.YieldOp([])

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
