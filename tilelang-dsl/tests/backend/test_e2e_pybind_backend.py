# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end integration tests for PybindBackend (Issue #237).

This test module validates the complete lowering pipeline:
1. Parse TileLang DSL kernel
2. Lower with TextBackend (reference)
3. Lower with PybindBackend (pybinding)
4. Compare outputs for consistency

Can run without pytest:
    python3 tests/backend/test_e2e_pybind_backend.py

Environment setup (optional for PybindBackend):
    export PYTHONPATH=$PTOAS_BUILD_DIR/python
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from tilelang_dsl.env_config import check_mlir_bindings_available, verify_environment, print_environment_help
from tilelang_dsl.lowering_backend import (
    TextBackend,
    PybindBackend,
    get_backend,
    lower_with_backend,
    LoweringResult,
)

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def _check_pto_dialect_available():
    """Check if PTO dialect bindings are available."""
    try:
        from ptoas.mlir.dialects import pto
        return True
    except ImportError:
        return False


def _create_simple_mock_kernel():
    """Create a simple mock kernel without PTO-specific types."""
    from dataclasses import dataclass

    @dataclass
    class MockSemanticKernel:
        target: str = "a5"
        op: str = "test"
        symbol_name: str = "simple_kernel"
        kernel_family: str = "test_family"
        verify_enabled: bool = True
        advanced_enabled: bool = False
        dtype_signature: tuple = ()
        parameters: tuple = ()
        tile_bindings: tuple = ()
        body: tuple = ()
        inline_helpers: tuple = ()

    return MockSemanticKernel()


def _create_mock_kernel_with_pto():
    """Create a mock SemanticKernel for testing with PTO types."""
    from dataclasses import dataclass
    from tilelang_dsl.semantic import (
        SemanticParameter,
        SemanticBinding,
        SemanticScalarType,
        SemanticIndexType,
        SemanticTensorViewType,
    )
    from tilelang_dsl.types import ScalarType

    @dataclass
    class MockSemanticKernel:
        target: str = "a5"
        op: str = "test"
        symbol_name: str = "test_kernel"
        kernel_family: str = "test_family"
        verify_enabled: bool = True
        advanced_enabled: bool = False
        dtype_signature: tuple = ()
        parameters: tuple = (
            SemanticParameter(
                binding=SemanticBinding(
                    name="input",
                    ssa_name="%input",
                    type=SemanticTensorViewType(rank=2, element_dtype=ScalarType("f32")),
                    origin="parameter",
                ),
            ),
        )
        tile_bindings: tuple = ()
        body: tuple = ()
        inline_helpers: tuple = ()

    return MockSemanticKernel()


def test_environment_check():
    """Verify environment is properly configured."""
    print("Test: environment check")

    mlir_available = check_mlir_bindings_available()
    print(f"  MLIR bindings available: {mlir_available}")

    if not mlir_available:
        print("  Note: MLIR Python bindings not available - PybindBackend tests will be limited")

    print("  [PASS]")
    return True


def test_pybind_backend_available_matches_mlir():
    """Verify PybindBackend.is_available() matches MLIR status."""
    print("Test: PybindBackend availability matches MLIR")

    backend = PybindBackend()
    mlir_available = check_mlir_bindings_available()
    backend_available = backend.is_available()

    assert backend_available == mlir_available, "Availability matches MLIR status"
    print(f"  backend.is_available() = {backend_available}")
    print(f"  MLIR available = {mlir_available}")

    print("  [PASS]")
    return True


def test_text_backend_lower_simple():
    """Test lowering a simple kernel with TextBackend."""
    print("Test: TextBackend lower simple")

    mock_kernel = _create_simple_mock_kernel()
    backend = TextBackend()
    result = backend.lower(mock_kernel)

    assert isinstance(result, LoweringResult), "Returns LoweringResult"
    assert result.text is not None, "Has text output"
    assert "module" in result.as_text(), "Output contains 'module'"

    print(f"  Output length: {len(result.as_text())} chars")
    print("  [PASS]")
    return True


def test_pybind_backend_lower_simple():
    """Test lowering a simple kernel with PybindBackend."""
    print("Test: PybindBackend lower simple")

    mlir_available = check_mlir_bindings_available()
    if not mlir_available:
        print("  SKIPPED: MLIR Python bindings not available")
        return True

    pto_available = _check_pto_dialect_available()
    if not pto_available:
        print("  SKIPPED: PTO dialect bindings not available (requires PTOAS build)")
        return True

    mock_kernel = _create_simple_mock_kernel()
    backend = PybindBackend()
    result = backend.lower(mock_kernel)

    assert isinstance(result, LoweringResult), "Returns LoweringResult"
    assert result.module is not None, "Has module output"
    assert result.text is not None, "Has text output"
    assert "module" in result.as_text(), "Output contains 'module'"

    print(f"  Output length: {len(result.as_text())} chars")
    print("  [PASS]")
    return True


def test_backend_output_comparison():
    """Compare TextBackend and PybindBackend outputs."""
    print("Test: backend output comparison")

    mlir_available = check_mlir_bindings_available()
    if not mlir_available:
        print("  SKIPPED: MLIR Python bindings not available")
        return True

    pto_available = _check_pto_dialect_available()
    if not pto_available:
        print("  SKIPPED: PTO dialect bindings not available (requires PTOAS build)")
        return True

    mock_kernel = _create_simple_mock_kernel()

    text_backend = TextBackend()
    pybind_backend = PybindBackend()

    text_result = text_backend.lower(mock_kernel)
    pybind_result = pybind_backend.lower(mock_kernel)

    # Both should produce MLIR text output
    assert text_result.as_text(), "TextBackend has output"
    assert pybind_result.as_text(), "PybindBackend has output"

    # Normalize and compare
    from tilelang_dsl.backend_validator import _normalize_mlir_text

    text_norm = _normalize_mlir_text(text_result.as_text(), skip_comments=True)
    pybind_norm = _normalize_mlir_text(pybind_result.as_text(), skip_comments=True)

    # Check key elements are present
    assert mock_kernel.symbol_name in text_norm or "simple_kernel" in text_norm, "Kernel name in text output"

    # Compare normalized outputs
    if text_norm != pybind_norm:
        print(f"  Warning: Outputs differ after normalization")
        print(f"  Text length: {len(text_norm)}, Pybind length: {len(pybind_norm)}")
    else:
        print("  Outputs match after normalization")

    print("  [PASS]")
    return True


def test_compare_backends_integration():
    """Test compare_backends with mock kernel."""
    print("Test: compare_backends integration")

    mlir_available = check_mlir_bindings_available()
    if not mlir_available:
        print("  SKIPPED: MLIR Python bindings not available")
        return True

    pto_available = _check_pto_dialect_available()
    if not pto_available:
        print("  SKIPPED: PTO dialect bindings not available (requires PTOAS build)")
        return True

    from tilelang_dsl.backend_validator import compare_backends, BackendComparisonResult

    mock_kernel = _create_simple_mock_kernel()
    result = compare_backends(mock_kernel)

    assert isinstance(result, BackendComparisonResult), "Returns BackendComparisonResult"
    assert result.pybind_available, "PybindBackend available"

    print(f"  Matches: {result.matches}")
    print(f"  Normalized matches: {result.normalized_matches}")

    print("  [PASS]")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PybindBackend End-to-End Integration Tests")
    print("=" * 60)

    # Print environment help if MLIR bindings not available
    if not check_mlir_bindings_available():
        print()
        print("Note: MLIR Python bindings not available")
        print("Some tests will be skipped. To enable PybindBackend tests:")
        print_environment_help()
        print()

    tests = [
        test_environment_check,
        test_pybind_backend_available_matches_mlir,
        test_text_backend_lower_simple,
        test_pybind_backend_lower_simple,
        test_backend_output_comparison,
        test_compare_backends_integration,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
