# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Layered PTODSL subkernel decorators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import update_wrapper
import inspect

from ._diagnostics import (
    illegal_inline_subkernel_placement_error,
    illegal_subkernel_placement_error,
    simd_value_escape_error,
    subkernel_host_tensor_boundary_error,
    subkernel_signature_boundary_error,
)
from ._ast_rewrite import rewrite_jit_function
from ._host_tensors import TensorSpec, looks_like_host_tensor
from ._surface_values import unwrap_surface_value
from ._tracing import current_runtime, current_session


class KernelRole(str, Enum):
    CUBE = "cube"
    SIMD = "simd"
    SIMT = "simt"


@dataclass(frozen=True)
class SubkernelSpec:
    """Declarative metadata for a PTODSL subkernel surface."""

    role: KernelRole
    symbol_name: str
    target: str = "a5"
    simt_max_threads: int | None = None
    simt_max_regs: int | None = None


class SubkernelTemplate:
    """Callable decorated PTODSL subkernel surface."""

    def __init__(self, spec: SubkernelSpec, py_fn, *, ast_rewrite: bool = True):
        self.spec = spec
        self.py_fn = py_fn
        self._ast_rewrite = ast_rewrite
        self.signature = inspect.signature(py_fn)
        self._validate_definition()
        update_wrapper(self, py_fn)

    def emit_body(self, *args, **kwargs):
        """Emit this subkernel body into the currently active trace."""
        py_fn = rewrite_jit_function(self.py_fn) if self._ast_rewrite else self.py_fn
        result = py_fn(*args, **kwargs)
        self._validate_result(result)
        return result

    def trace_body(self, *args, **kwargs):
        """Backward-compatible alias for body emission."""
        return self.emit_body(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"@pto.{self.spec.role.value} kernels may only be called while tracing "
                "a compatible PTODSL kernel"
            )
        self._validate_invocation(*args, **kwargs)
        return runtime.dispatch_subkernel_call(self, *args, **kwargs)

    def __getitem__(self, dims):
        if self.spec.role != KernelRole.SIMT:
            raise TypeError(
                f"@pto.{self.spec.role.value} kernels do not support launch dimensions; "
                "only @pto.simt helpers support helper[dim_x, dim_y, dim_z](...)"
            )
        if not isinstance(dims, tuple):
            dims = (dims,)
        if len(dims) != 3:
            raise TypeError("@pto.simt launch syntax expects helper[dim_x, dim_y, dim_z](...)")
        return _SimtLaunchTemplate(self, dims)

    def _validate_definition(self) -> None:
        for param in self.signature.parameters.values():
            if isinstance(param.annotation, TensorSpec):
                raise subkernel_signature_boundary_error(self.spec.role.value, param.name)

    def _validate_invocation(self, *args, **kwargs) -> None:
        session = current_session()
        outer = session.current_subkernel if session is not None else None
        _validate_subkernel_placement(self.spec.role, outer)

        bound = self.signature.bind_partial(*args, **kwargs)
        for name, value in bound.arguments.items():
            if looks_like_host_tensor(value):
                raise subkernel_host_tensor_boundary_error(self.spec.role.value, name)

    def _validate_result(self, result) -> None:
        if self.spec.role != KernelRole.SIMD:
            return
        escaped_type = _find_transient_simd_escape(result)
        if escaped_type is not None:
            raise simd_value_escape_error(escaped_type)


class _SimtLaunchTemplate:
    """Callable ``helper[x, y, z]`` launch descriptor for a decorated SIMT helper."""

    def __init__(self, body: SubkernelTemplate, dims):
        self._body = body
        self._dims = dims

    def __call__(self, *args, **kwargs):
        from ._ops import simt_launch

        return simt_launch(self._body, *args, dims=self._dims, **kwargs)


def _find_transient_simd_escape(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            escaped = _find_transient_simd_escape(item)
            if escaped is not None:
                return escaped
        return None
    if isinstance(value, dict):
        for item in value.values():
            escaped = _find_transient_simd_escape(item)
            if escaped is not None:
                return escaped
        return None
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return None
    type_text = str(type_obj)
    if type_text.startswith("!pto.vreg<") or type_text.startswith("!pto.mask<"):
        return type_text
    return None


def _validate_subkernel_placement(role: KernelRole, outer_frame, *, inline: bool = False) -> None:
    if outer_frame is None:
        return
    if inline:
        raise illegal_inline_subkernel_placement_error(role.value, outer_frame.role)
    raise illegal_subkernel_placement_error(role.value, outer_frame.role)


class _SubkernelSurface:
    """Dual-use surface that supports both decorators and inline context-manager scopes."""

    def __init__(
        self,
        role: KernelRole,
        *,
        name: str | None = None,
        target: str = "a5",
        ast_rewrite: bool = True,
        simt_max_threads: int | None = None,
        simt_max_regs: int | None = None,
    ):
        self._role = role
        self._name = name
        self._target = target
        self._ast_rewrite = ast_rewrite
        self._simt_max_threads = simt_max_threads
        self._simt_max_regs = simt_max_regs
        self._session_cm = None

    def __call__(self, fn):
        return SubkernelTemplate(
            SubkernelSpec(
                role=self._role,
                symbol_name=self._name or fn.__name__,
                target=self._target,
                simt_max_threads=self._simt_max_threads,
                simt_max_regs=self._simt_max_regs,
            ),
            fn,
            ast_rewrite=self._ast_rewrite,
        )

    def __enter__(self):
        if self._role == KernelRole.SIMT and (
            self._simt_max_threads is not None or self._simt_max_regs is not None
        ):
            raise TypeError("@pto.simt(max_threads=..., max_regs=...) is only supported as a function decorator")
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"inline pto.{self._role.value}() may only be used while tracing "
                "a compatible PTODSL kernel"
            )
        session = current_session()
        outer = session.current_subkernel if session is not None else None
        _validate_subkernel_placement(self._role, outer, inline=True)
        symbol_name = self._name or f"inline_{self._role.value}"
        self._session_cm = session.enter_inline_subkernel(
            self._role.value,
            symbol_name,
            self._target,
        )
        self._session_cm.__enter__()
        return None

    def __exit__(self, *exc):
        try:
            return self._session_cm.__exit__(*exc)
        finally:
            self._session_cm = None


def _subkernel_decorator(
    role: KernelRole,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    simt_max_threads: int | None = None,
    simt_max_regs: int | None = None,
):
    return _SubkernelSurface(
        role,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=simt_max_threads,
        simt_max_regs=simt_max_regs,
    )


def _decorate_subkernel(
    role: KernelRole,
    fn=None,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    simt_max_threads: int | None = None,
    simt_max_regs: int | None = None,
):
    if fn is not None:
        return _subkernel_decorator(
            role,
            name=name,
            target=target,
            ast_rewrite=ast_rewrite,
            simt_max_threads=simt_max_threads,
            simt_max_regs=simt_max_regs,
        )(fn)
    return _subkernel_decorator(
        role,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=simt_max_threads,
        simt_max_regs=simt_max_regs,
    )


def cube(fn=None, *, name: str | None = None, target: str = "a5", ast_rewrite: bool = True):
    return _decorate_subkernel(KernelRole.CUBE, fn, name=name, target=target, ast_rewrite=ast_rewrite)


def simd(fn=None, *, name: str | None = None, target: str = "a5", ast_rewrite: bool = True):
    return _decorate_subkernel(KernelRole.SIMD, fn, name=name, target=target, ast_rewrite=ast_rewrite)


def _validate_simt_resource_attr(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"@pto.simt(..., {name}=...) expects a positive Python int")
    if value <= 0:
        raise ValueError(f"@pto.simt(..., {name}=...) expects a positive Python int")
    if value > 2**31 - 1:
        raise ValueError(f"@pto.simt(..., {name}=...) must fit in signless i32")
    return value


def simt(
    fn=None,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    max_threads: int | None = None,
    max_regs: int | None = None,
):
    max_threads = _validate_simt_resource_attr("max_threads", max_threads)
    max_regs = _validate_simt_resource_attr("max_regs", max_regs)
    return _decorate_subkernel(
        KernelRole.SIMT,
        fn,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=max_threads,
        simt_max_regs=max_regs,
    )


__all__ = [
    "KernelRole",
    "SubkernelSpec",
    "SubkernelTemplate",
    "cube",
    "simd",
    "simt",
]
