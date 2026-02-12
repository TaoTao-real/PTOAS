from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build_invalid(ctx):
    with Location.unknown(ctx):
        m = Module.create()

        f32 = F32Type.get(ctx)
        idx = IndexType.get(ctx)

        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
        try:
            ptr_f32_vec = pto.PtrType.get(f32, vec, ctx)
        except Exception:
            return True

        fn_ty = func.FunctionType.get([ptr_f32_vec], [])
        with InsertionPoint(m.body):
            fn = func.FuncOp("ptr_scalar_rw_vec", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c0 = arith.ConstantOp(idx, 0).result
            src = entry.arguments[0]
            val = pto.LoadScalar(f32, src, c0).value
            pto.StoreScalar(src, c0, val)
            func.ReturnOp([])

        try:
            ok = m.operation.verify()
        except Exception:
            return True
        if isinstance(ok, bool) and not ok:
            return True
        return False


def build_valid(ctx):
    with Location.unknown(ctx):
        m = Module.create()

        f32 = F32Type.get(ctx)
        idx = IndexType.get(ctx)

        gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
        ptr_f32_gm = pto.PtrType.get(f32, gm, ctx)

        fn_ty = func.FunctionType.get([ptr_f32_gm, ptr_f32_gm], [])
        with InsertionPoint(m.body):
            fn = func.FuncOp("ptr_scalar_rw_gm", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c4 = arith.ConstantOp(idx, 4).result
            c8 = arith.ConstantOp(idx, 8).result

            src, dst = entry.arguments
            src_off = pto.AddPtr(src, c8).result

            val = pto.LoadScalar(f32, src_off, c4).value
            pto.StoreScalar(dst, c4, val)
            func.ReturnOp([])

        m.operation.verify()
        return m


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        if not build_invalid(ctx):
            raise RuntimeError("expected non-GM scalar ptr to be rejected")

        return build_valid(ctx)


if __name__ == "__main__":
    print(build())
