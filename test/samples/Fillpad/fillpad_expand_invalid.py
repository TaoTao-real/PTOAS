from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import func, pto
from mlir.ir import F32Type


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Zero, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            src_ty = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            dst_ty = pto.TileBufType.get([32, 16], f32, vec, [32, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("fillpad_expand_invalid", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src = pto.AllocTileOp(src_ty).result
                dst = pto.AllocTileOp(dst_ty).result
                pto.TFillPadExpandOp(src, dst)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            raise SystemExit(1)


if __name__ == "__main__":
    print(build())
