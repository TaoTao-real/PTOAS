#!/usr/bin/env python3
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    MemRefType,
    F16Type,
    IndexType,
)
from mlir.dialects import func, arith, scf, memref, pto
from mlir.dialects.arith import CmpIPredicate


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)

            # A minimal loop that reuses the same UB buffer across iterations.
            # This should trigger multi-buffer synchronization insertion on the
            # loop back-edge (MTE3 -> MTE2) when `--enable-insert-sync` is on.
            gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            mem_gm = MemRefType.get([16, 16, 16], f16, memory_space=gm)
            mem_ub = MemRefType.get([16, 16, 16], f16, memory_space=vec)

            fn_ty = func.FunctionType.get([mem_gm, mem_gm], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_inject_sync_multibuf_loop", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result

                ub = memref.AllocOp(mem_ub, [], []).result

                loop = scf.ForOp(c0, c2, c1, [])
                with InsertionPoint(loop.body):
                    iv = loop.induction_variable

                    pto.TLoadOp(None, entry.arguments[0], ub)

                    cond = arith.CmpIOp(CmpIPredicate.eq, iv, c0).result
                    ifop = scf.IfOp(cond, [], hasElse=False)
                    with InsertionPoint(ifop.then_block):
                        pto.TAddOp(ub, ub, ub)
                        scf.YieldOp([])

                    pto.TStoreOp(None, ub, entry.arguments[1])
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
