# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Trace-session objects shared by PTODSL tracing runtimes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from .control_flow import (
    build_carry_loop_frame,
    finish_carry_loop_frame,
    yield_carry_loop_state,
)
from .._surface_values import unwrap_surface_value, wrap_like_surface_value
from .._types import _strip_integer_signedness

from mlir.dialects import arith, func
from mlir.dialects import pto as _pto
from mlir.ir import FlatSymbolRefAttr, IndexType, InsertionPoint, IntegerAttr, IntegerType, Operation, UnitAttr


@dataclass(frozen=True)
class HelperFunctionSpec:
    """Declarative description of a helper function emitted during tracing."""

    symbol_name: str
    arg_types: tuple
    result_types: tuple = ()
    attributes: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class SimtHelperSpecializationKey:
    """Cache key for one specialized ``@pto.simt`` helper body."""

    symbol_name: str
    arg_types: tuple
    static_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class SubkernelTraceFrame:
    """Active inline-lowering frame for one PTODSL subkernel call."""

    role: str
    symbol_name: str
    target: str


class TraceSession:
    """Shared per-build state for a traced PTODSL module."""

    def __init__(self, module_spec, module, entry_function):
        self.module_spec = module_spec
        self.module = module
        self.entry_function = entry_function
        self.entry_block = None
        self._function_stack = [entry_function]
        self._function_symbol_table = entry_function.operation.parent.regions[0].blocks[0]
        self._helpers: dict[str, object] = {}
        self._simt_helper_specializations: dict[SimtHelperSpecializationKey, object] = {}
        self._simt_helper_symbol_counters: dict[str, int] = {}
        self._subkernel_stack: list[SubkernelTraceFrame] = []
        self._carry_loop_stack = []

    @property
    def current_function(self):
        return self._function_stack[-1]

    @property
    def current_subkernel(self):
        if not self._subkernel_stack:
            return None
        return self._subkernel_stack[-1]

    @property
    def subkernel_stack_depth(self):
        return len(self._subkernel_stack)

    @property
    def current_carry_loop(self):
        if not self._carry_loop_stack:
            return None
        return self._carry_loop_stack[-1]

    def bind_entry_block(self, entry_block) -> None:
        """Record the root entry block for the active trace."""
        self.entry_block = entry_block

    @contextmanager
    def enter_function(self, ir_fn):
        """Push *ir_fn* as the current active function in this session."""
        self._function_stack.append(ir_fn)
        try:
            yield ir_fn
        finally:
            popped = self._function_stack.pop()
            if popped is not ir_fn:
                raise RuntimeError("PTODSL trace-session function stack corruption detected")

    @contextmanager
    def enter_inline_subkernel(self, role: str, symbol_name: str, target: str):
        """Push one inline subkernel frame onto the active tracing stack."""
        frame = SubkernelTraceFrame(
            role=role,
            symbol_name=symbol_name,
            target=target,
        )
        self._subkernel_stack.append(frame)
        try:
            yield frame
        finally:
            popped = self._subkernel_stack.pop()
            if popped is not frame:
                raise RuntimeError("PTODSL trace-session subkernel stack corruption detected")

    @contextmanager
    def enter_subkernel(self, subkernel):
        """Push *subkernel* as the current active inline-lowering frame."""
        with self.enter_inline_subkernel(
            subkernel.spec.role.value,
            subkernel.spec.symbol_name,
            subkernel.spec.target,
        ) as frame:
            yield frame

    def lower_inline_subkernel(self, subkernel, *args, **kwargs):
        """Lower one inline PTODSL subkernel call through the shared session."""
        with self.enter_subkernel(subkernel):
            role = subkernel.spec.role.value
            if role in {"cube", "simd"}:
                section = (
                    _pto.SectionCubeOp()
                    if role == "cube"
                    else _pto.SectionVectorOp()
                )
                section_block = section.body.blocks.append()
                with InsertionPoint(section_block):
                    return subkernel.emit_body(*args, **kwargs)
            return subkernel.emit_body(*args, **kwargs)

    def begin_carry_loop(self, start, stop, step, state_items):
        """Materialize one authored ``pto.for_(...).carry(...)`` loop body."""
        frame = build_carry_loop_frame(start, stop, step, state_items)
        self._carry_loop_stack.append(frame)
        return frame

    def update_carry_loop(self, frame, **kwargs):
        """Emit the one legal ``loop.update(...)`` for the active carry loop."""
        active = self.current_carry_loop
        if active is None or active is not frame:
            raise RuntimeError("loop.update(...) may only be called inside the active carry loop body")
        yield_carry_loop_state(frame, **kwargs)

    def finish_carry_loop(self, frame, exc_type, exc, tb):
        """Finalize one active authored carry loop and close its body insertion point."""
        if not self._carry_loop_stack:
            raise RuntimeError("carry-loop exit without a matching active PTODSL trace-session frame")
        popped = self._carry_loop_stack.pop()
        if popped is not frame:
            raise RuntimeError("PTODSL trace-session carry-loop stack corruption detected")
        finish_carry_loop_frame(frame, exc_type, exc, tb)

    def lower_simt_helper_subkernel(self, subkernel, *args, **kwargs):
        """Lower one ``@pto.simt`` call through a dedicated helper function."""
        helper_fn, arg_templates = self._get_or_create_simt_helper_function(subkernel, *args, **kwargs)

        i32 = IntegerType.get_signless(32)
        dim_z = arith.ConstantOp(i32, 1).result
        dim_y = arith.ConstantOp(i32, 1).result
        dim_x = arith.ConstantOp(i32, 1).result
        _pto.StoreVfSimtInfoOp(dim_z, dim_y, dim_x)
        func.CallOp(helper_fn, [unwrap_surface_value(arg) for arg in arg_templates])

    def lower_simt_launch_subkernel(self, subkernel, *args, dims, **kwargs):
        """Lower one explicit ``pto.simt_launch`` call through a SIMT helper."""
        helper_fn, arg_templates = self._get_or_create_simt_helper_function(subkernel, *args, **kwargs)
        dim_x, dim_y, dim_z = _coerce_simt_launch_dims(dims)
        Operation.create(
            "pto.simt_launch",
            attributes={"callee": FlatSymbolRefAttr.get(_symbol_name(helper_fn))},
            operands=[dim_x, dim_y, dim_z, *[unwrap_surface_value(arg) for arg in arg_templates]],
        )

    def _get_or_create_simt_helper_function(self, subkernel, *args, **kwargs):
        """Return the reusable ``pto.simt_entry`` helper for *subkernel*."""
        outer_frame = self.current_subkernel
        if outer_frame is not None and outer_frame.role == "simt":
            raise RuntimeError("@pto.simt helper lowering does not support nested SIMT helper calls")

        arg_templates = tuple(args)
        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        static_kwargs = _simt_static_kwargs_signature(kwargs)
        specialization_key = SimtHelperSpecializationKey(
            symbol_name=subkernel.spec.symbol_name,
            arg_types=arg_types,
            static_kwargs=static_kwargs,
        )
        helper_fn = self._simt_helper_specializations.get(specialization_key)
        if helper_fn is not None:
            return helper_fn, arg_templates

        helper_symbol = self._next_simt_helper_symbol(subkernel.spec.symbol_name)
        helper_attributes = [("pto.simt_entry", UnitAttr.get())]
        i32_attr_type = IntegerType.get_signless(32)
        if subkernel.spec.simt_max_threads is not None:
            helper_attributes.append(
                (
                    "pto.simt_max_threads",
                    IntegerAttr.get(i32_attr_type, subkernel.spec.simt_max_threads),
                )
            )
        if subkernel.spec.simt_max_regs is not None:
            helper_attributes.append(
                (
                    "pto.simt_max_regs",
                    IntegerAttr.get(i32_attr_type, subkernel.spec.simt_max_regs),
                )
            )
        helper_spec = HelperFunctionSpec(
            symbol_name=helper_symbol,
            arg_types=arg_types,
            attributes=tuple(helper_attributes),
        )
        helper_fn, created = self.get_or_create_helper_function(helper_spec)
        self._simt_helper_specializations[specialization_key] = helper_fn

        if created:
            entry_block = helper_fn.add_entry_block()
            wrapped_args = tuple(
                wrap_like_surface_value(template, value)
                for template, value in zip(arg_templates, entry_block.arguments)
            )
            with self.enter_function(helper_fn), self.enter_subkernel(subkernel), InsertionPoint(entry_block):
                subkernel.emit_body(*wrapped_args, **kwargs)
                func.ReturnOp([])

        return helper_fn, arg_templates

    def _next_simt_helper_symbol(self, base_symbol: str) -> str:
        index = self._simt_helper_symbol_counters.get(base_symbol, 0)
        while True:
            symbol = f"{base_symbol}__simt_{index}"
            index += 1
            if symbol not in self._helpers:
                self._simt_helper_symbol_counters[base_symbol] = index
                return symbol

    def resolve_simt_peer_symbol(self, subkernel) -> str:
        """Return the unique materialized helper symbol for a ``@pto.simt`` peer."""
        symbol_name = subkernel.spec.symbol_name
        matches = [
            helper_fn
            for key, helper_fn in self._simt_helper_specializations.items()
            if key.symbol_name == symbol_name
        ]
        if not matches:
            raise RuntimeError(
                f"pto.import_reserved_buffer(..., peer_func={symbol_name}) cannot resolve "
                "the @pto.simt helper symbol before the helper is called or launched"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"pto.import_reserved_buffer(..., peer_func={symbol_name}) is ambiguous "
                "because the @pto.simt helper has multiple specializations; pass the "
                "materialized peer function symbol explicitly"
            )
        return _symbol_name(matches[0])

    def lookup_helper(self, symbol_name: str):
        """Return a previously declared helper function, or ``None``."""
        return self._helpers.get(symbol_name)

    def get_or_create_helper_function(self, spec: HelperFunctionSpec):
        """
        Look up or create a helper ``func.func`` in the current symbol table.

        Returns ``(helper_fn, created)`` where *created* reports whether a new
        symbol was emitted in this trace session.
        """
        helper = self._helpers.get(spec.symbol_name)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        with InsertionPoint(self._function_symbol_table):
            helper = func.FuncOp(spec.symbol_name, fn_ty)
            for attr_name, attr_value in spec.attributes:
                helper.attributes[attr_name] = attr_value
        self._helpers[spec.symbol_name] = helper
        return helper, True

    def validate_final_state(self) -> None:
        """Check that tracing-time session stacks were fully unwound."""
        if self._subkernel_stack:
            raise RuntimeError("PTODSL trace-session exited with an open subkernel lowering frame")
        if self._carry_loop_stack:
            raise RuntimeError("PTODSL trace-session exited with an open loop-carry lowering frame")


def _coerce_simt_launch_dims(dims):
    if not isinstance(dims, (tuple, list)) or len(dims) != 3:
        raise TypeError("pto.simt_launch(..., dims=...) expects a 3-item (dim_x, dim_y, dim_z) tuple")
    return tuple(
        _coerce_i32_dim(dim, context=f"pto.simt_launch(..., dims[{index}])")
        for index, dim in enumerate(dims)
    )


def _coerce_i32_dim(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i32 = IntegerType.get_signless(32)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        if raw_value < 0:
            raise ValueError(f"{context} expects a non-negative i32 launch dimension, got {raw_value}")
        return arith.ConstantOp(i32, raw_value).result
    if IndexType.isinstance(raw_value.type):
        return arith.IndexCastOp(i32, raw_value).result
    if IntegerType.isinstance(raw_value.type):
        width = IntegerType(raw_value.type).width
        if width != 32:
            raise TypeError(f"{context} expects i32 launch dimension, got {raw_value.type}")
        return _strip_integer_signedness(raw_value)
    raise TypeError(f"{context} expects i32 launch dimension, got {raw_value.type}")


def _symbol_name(ir_fn) -> str:
    try:
        name_attr = ir_fn.attributes["sym_name"]
    except KeyError as exc:
        raise RuntimeError("PTODSL helper function is missing sym_name")
    if name_attr is None:
        raise RuntimeError("PTODSL helper function has empty sym_name")
    return str(name_attr.value)


def _simt_static_kwargs_signature(kwargs):
    return tuple(
        (name, _simt_static_signature_atom(value))
        for name, value in sorted(kwargs.items())
    )


def _simt_static_signature_atom(value):
    raw_value = unwrap_surface_value(value)
    if hasattr(raw_value, "type"):
        raise TypeError(
            "pto.simt_launch keyword arguments must be static hashable values; "
            "pass runtime SSA arguments positionally"
        )
    try:
        hash(value)
    except TypeError:
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        tuple(
                            (
                                _simt_static_signature_atom(key),
                                _simt_static_signature_atom(item),
                            )
                            for key, item in value.items()
                        ),
                        key=repr,
                    )
                ),
            )
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(_simt_static_signature_atom(item) for item in value),
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(sorted((_simt_static_signature_atom(item) for item in value), key=repr)),
            )
        return (type(value).__name__, repr(value))
    return value


__all__ = [
    "HelperFunctionSpec",
    "SubkernelTraceFrame",
    "TraceSession",
]
