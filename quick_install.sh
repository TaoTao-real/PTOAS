#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Build and install PTOAS in editable mode against an existing LLVM/MLIR
# build. The persistent build directory can subsequently be used with Ninja,
# for example: ninja -C build check-pto.
#
# Optional env:
#   LLVM_BUILD_DIR   - default: ${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}/build-shared
#   PTO_BUILD_DIR    - default: <repo>/build
#   PYTHON_BIN       - default: python

set -euo pipefail

PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-/llvm-workspace/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"
PTO_BUILD_DIR="${PTO_BUILD_DIR:-${PTO_SOURCE_DIR}/build}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v ccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-ccache}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
fi

"${PYTHON_BIN}" -m pip install \
  'scikit-build-core>=0.12.2,<2' \
  'pybind11<3' \
  'nanobind>=2.4'

LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  "${PYTHON_BIN}" -m pip install --editable "${PTO_SOURCE_DIR}" \
    --no-build-isolation \
    --config-settings="build-dir=${PTO_BUILD_DIR}"

echo "PTOAS editable install complete."
echo "Build directory: ${PTO_BUILD_DIR}"
