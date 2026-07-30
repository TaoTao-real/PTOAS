# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lowering backend abstraction for TileLang DSL.

This module provides the abstract backend infrastructure for lowering
SemanticKernel to authoring-form VPTO MLIR. The design supports:
- TextBackend: existing string emitter (reference implementation)
- PybindBackend: future pybinding-based IR builder

Issue #237: TileLang DSL lowering backend 渐进迁移到 pybinding builder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Any

if TYPE_CHECKING:
    from .semantic import SemanticKernel
    from ptoas.mlir import ir as _ods_ir


@dataclass(frozen=True)
class LoweringResult:
    """Container for lowering output supporting both text and module forms.

    The result can hold either:
    - text: raw MLIR text string (from TextBackend)
    - module: structured mlir.ir.Module object (from PybindBackend)

    Both forms can be converted to each other via as_text() / as_module().
    """

    text: str | None = None
    module: "_ods_ir.Module | None" = None

    def as_text(self) -> str:
        """Get the result as MLIR text string."""
        if self.text is not None:
            return self.text
        if self.module is not None:
            return str(self.module)
        raise ValueError("LoweringResult has no output (neither text nor module)")

    def as_module(self, context: "_ods_ir.Context | None" = None) -> "_ods_ir.Module":
        """Get the result as structured MLIR Module.

        If the result already has a module, return it directly.
        If only text is available, parse it into a module.
        """
        if self.module is not None:
            return self.module
        if self.text is not None:
            return self._parse_text_to_module(self.text, context)
        raise ValueError("LoweringResult has no output (neither text nor module)")

    @staticmethod
    def _parse_text_to_module(
        text: str,
        context: "_ods_ir.Context | None" = None,
    ) -> "_ods_ir.Module":
        """Parse MLIR text into a structured Module.

        The PTOAS MLIR site initializer registers the bundled upstream dialects.
        Register the project-owned PTO dialect explicitly before parsing.
        """
        from ptoas.mlir import ir as _ods_ir
        from ptoas.mlir.dialects import pto as _pto_dialect

        ctx = context if context is not None else _ods_ir.Context()
        _pto_dialect.register_dialect(ctx, load=True)

        return _ods_ir.Module.parse(text, ctx)

    def __str__(self) -> str:
        return self.as_text()

    def __bool__(self) -> bool:
        return self.text is not None or self.module is not None


class LoweringBackend(ABC):
    """Abstract base class for lowering backends.

    Each backend must implement:
    - lower(): convert SemanticKernel to LoweringResult
    - name(): return backend identifier for diagnostics
    """

    @abstractmethod
    def lower(self, kernel: "SemanticKernel") -> LoweringResult:
        """Lower SemanticKernel to authoring-form VPTO.

        Args:
            kernel: The semantic kernel to lower.

        Returns:
            LoweringResult containing the output (text or module form).
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return backend identifier for diagnostics.

        Returns:
            Backend name string (e.g., "text", "pybind").
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name()!r})"


class TextBackend(LoweringBackend):
    """Wrapper for existing text emitter (reference implementation).

    This backend wraps the existing AuthoringModule from lowering.py,
    preserving the stable reference implementation during the migration.
    """

    def __init__(self):
        # Lazy import to avoid circular dependency
        self._authoring_module_cls: Any = None

    def _get_authoring_module_cls(self) -> Any:
        """Lazy load AuthoringModule class."""
        if self._authoring_module_cls is None:
            from .lowering import AuthoringModule
            self._authoring_module_cls = AuthoringModule
        return self._authoring_module_cls

    def lower(self, kernel: "SemanticKernel") -> LoweringResult:
        """Lower kernel using text emitter.

        Args:
            kernel: The semantic kernel to lower.

        Returns:
            LoweringResult with text output.
        """
        AuthoringModule = self._get_authoring_module_cls()
        module = AuthoringModule(kernel=kernel)
        text = module.render()
        return LoweringResult(text=text)

    def name(self) -> str:
        return "text"


class PybindBackend(LoweringBackend):
    """Pybinding-based lowering backend.

    This backend constructs MLIR IR directly using Python bindings
    instead of emitting text strings.

    Requires MLIR Python bindings from LLVM build. See env_config.py
    for environment setup instructions.
    """

    def __init__(self):
        self._renderer_cls: Any = None
        self._mlir_available: bool | None = None

    def _check_mlir_available(self) -> bool:
        """Check if MLIR Python bindings are actually available."""
        if self._mlir_available is not None:
            return self._mlir_available

        try:
            from ptoas.mlir import ir
            self._mlir_available = True
        except ImportError:
            self._mlir_available = False

        return self._mlir_available

    def _get_renderer_cls(self) -> Any:
        """Lazy load PybindRenderer class."""
        if self._renderer_cls is None:
            if not self._check_mlir_available():
                raise NotImplementedError(
                    "PybindBackend requires MLIR Python bindings. "
                    "Please set up the environment. "
                    "Run: python3 -c 'from tilelang_dsl import print_environment_help; print_environment_help()'"
                )
            from .pybind_renderer import PybindRenderer
            self._renderer_cls = PybindRenderer
        return self._renderer_cls

    def lower(self, kernel: "SemanticKernel") -> LoweringResult:
        """Lower kernel using pybinding IR builder.

        Args:
            kernel: The semantic kernel to lower.

        Returns:
            LoweringResult with both module and text output.

        Raises:
            NotImplementedError: If MLIR Python bindings are not available.
        """
        PybindRenderer = self._get_renderer_cls()
        renderer = PybindRenderer(kernel)
        module = renderer.render()
        return LoweringResult(text=str(module), module=module)

    def name(self) -> str:
        return "pybind"

    def is_available(self) -> bool:
        """Check if MLIR Python bindings are available for actual lowering."""
        return self._check_mlir_available()


# Backend factory

_DEFAULT_BACKEND_NAME = "text"
_SUPPORTED_BACKENDS = frozenset({"text", "pybind"})


def get_backend(name: str = _DEFAULT_BACKEND_NAME) -> LoweringBackend:
    """Factory function for backend selection.

    Args:
        name: Backend identifier ("text" or "pybind").

    Returns:
        LoweringBackend instance.

    Raises:
        ValueError: If backend name is not recognized.
    """
    if name not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend name {name!r}. "
            f"Supported backends: {sorted(_SUPPORTED_BACKENDS)}"
        )
    if name == "text":
        return TextBackend()
    if name == "pybind":
        return PybindBackend()
    raise ValueError(f"Unhandled backend name: {name}")


def lower_with_backend(
    kernel: "SemanticKernel",
    backend: str = _DEFAULT_BACKEND_NAME,
) -> LoweringResult:
    """Convenience function to lower kernel with specified backend.

    Args:
        kernel: The semantic kernel to lower.
        backend: Backend identifier ("text" or "pybind").

    Returns:
        LoweringResult from the specified backend.
    """
    return get_backend(backend).lower(kernel)


__all__ = [
    "LoweringResult",
    "LoweringBackend",
    "TextBackend",
    "PybindBackend",
    "get_backend",
    "lower_with_backend",
]
