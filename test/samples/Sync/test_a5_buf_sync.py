from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                fn = func.FuncOp("test_a5_buf_sync", func.FunctionType.get([], []))
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # Acquire/release buffer-id token on MTE2.
                pto.get_buf(pto.TLOAD, 0)
                pto.rls_buf(pto.TLOAD, 0)
                # Acquire/release buffer-id token on VEC.
                pto.get_buf(pto.TVEC, 0)
                pto.rls_buf(pto.TVEC, 0)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
