#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from . import _pto_ops_gen as _pto_ops_gen
from ._pto_ops_gen import *
from mlir import ir as _ods_ir

from .._mlir_libs._pto import (
    register_dialect,
    PtrType,
    TensorViewType,
    PartitionTensorViewType,
    TileType,
    TileBufType,
    AddressSpace,
    AddressSpaceAttr,
    get_gm_type,
    TileBufConfigAttr,
    BLayout,
    BLayoutAttr,
    SLayout,
    SLayoutAttr,
    PadValue,
    PadValueAttr,
    RoundMode,
    RoundModeAttr,
    CmpMode,
    CmpModeAttr,
    SyncOpType,
    SyncOpTypeAttr,
    EVENT,
    EventAttr
)

__all__ = [
    # Dialect utilities
    "register_dialect",

    # Types
    "PtrType",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "AddressSpace", "AddressSpaceAttr",
    "BLayout","BLayoutAttr",
    "SLayout","SLayoutAttr",
    "PadValue","PadValueAttr",
    "RoundMode", "RoundModeAttr",
    "CmpMode", "CmpModeAttr",
    "SyncOpType", "SyncOpTypeAttr",
    "EVENT", "EventAttr",
    "get_gm_type", "TileBufConfigAttr",
    # High-level sync helpers
    "record_event", "wait_event", "barrier"
]

# -----------------------------------------------------------------------------
# Convenience wrappers for high-level sync to allow passing enums directly
# -----------------------------------------------------------------------------

def _ensure_sync_attr(val, ctx):
    if isinstance(val, SyncOpType):
        return SyncOpTypeAttr.get(val, ctx)
    return val

def _ensure_event_attr(val, ctx):
    if isinstance(val, EVENT):
        return EventAttr.get(val, ctx)
    return val

def record_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.record_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def wait_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.wait_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def barrier(op, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    # If user passes SyncOpType/Attr, route to barrier_sync (maps to PIPE)
    if isinstance(op, (SyncOpType, SyncOpTypeAttr)):
        op_attr = _ensure_sync_attr(op, ctx)
        return _pto_ops_gen.barrier_sync(op_attr, loc=loc, ip=ip)
    # Otherwise fall back to low-level barrier expecting PipeAttr
    return _pto_ops_gen.barrier(op, loc=loc, ip=ip)
