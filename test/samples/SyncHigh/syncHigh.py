#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, pto, arith

def cidx(v):
    return arith.ConstantOp(IndexType.get(), v).result

def main():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("run_sync_high", func.FunctionType.get([], []))
        entry = f.add_entry_block()
        with InsertionPoint(entry):
            # Cover all SyncOpType values with matching record/wait.
            all_types = [
                pto.SyncOpType.TLOAD,
                pto.SyncOpType.TSTORE_ACC,
                pto.SyncOpType.TSTORE_VEC,
                pto.SyncOpType.TMOV_M2L,
                pto.SyncOpType.TMOV_M2S,
                pto.SyncOpType.TMOV_M2B,
                pto.SyncOpType.TMOV_M2V,
                pto.SyncOpType.TMOV_V2M,
                pto.SyncOpType.TMATMUL,
                pto.SyncOpType.TVEC,
                pto.SyncOpType.TVECWAIT_EVENT,
            ]
            events = [
                pto.EVENT.EVENT_ID0,
                pto.EVENT.EVENT_ID1,
                pto.EVENT.EVENT_ID2,
                pto.EVENT.EVENT_ID3,
                pto.EVENT.EVENT_ID4,
                pto.EVENT.EVENT_ID5,
                pto.EVENT.EVENT_ID6,
                pto.EVENT.EVENT_ID7,
            ]
            for i, ty in enumerate(all_types):
                ev = events[i % len(events)]
                # record type -> type (python helper accepts enums directly)
                pto.record_event(ty, ty, ev)
                # wait type -> type
                pto.wait_event(ty, ty, ev)

            # Add barrier coverage for TMATMUL and TVEC
            pto.barrier(pto.SyncOpType.TMATMUL)
            pto.barrier(pto.SyncOpType.TVEC)
            func.ReturnOp([])
        print(module)

if __name__ == "__main__":
    main()
