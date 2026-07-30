#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PTO_FILE="${1:-}"
OUT_DIR_ARG="${2:-}"

PTOAS_BIN="${PTOAS_BIN:-}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5}"
VPTO_FLAGS="${VPTO_FLAGS:---pto-backend=vpto --vpto-emit-hivm-llvm}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${HOME}/cann}"
BISHENG_BIN="${BISHENG_BIN:-}"
BISHENG_FLAGS="${BISHENG_FLAGS:-}"
LLVM_IR=""
DEVICE_OBJ=""

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

on_error() {
  local exit_code="$1"
  if [[ -n "${LLVM_IR}" && -f "${LLVM_IR}" ]]; then
    echo "Retained LLVM IR: ${LLVM_IR}" >&2
  fi
  if [[ -n "${DEVICE_OBJ}" ]]; then
    echo "Expected device object: ${DEVICE_OBJ}" >&2
  fi
  exit "${exit_code}"
}

trap 'on_error $?' ERR

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <input.pto> [output_dir]

Environment overrides:
  PTOAS_BIN     path to ptoas
  PTOAS_FLAGS   default: --pto-arch a5
  VPTO_FLAGS    default: --pto-backend=vpto --vpto-emit-hivm-llvm
  ASCEND_HOME_PATH default: \$HOME/cann
  BISHENG_BIN
  BISHENG_FLAGS extra flags passed to bisheng when compiling .ll to .o
  AICORE_ARCH   default: dav-c310-vec

Example:
  $(basename "$0") test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto
EOF
}

[[ -n "${PTO_FILE}" ]] || {
  usage
  exit 1
}

[[ "${PTO_FILE}" == *.pto ]] || die "input must be a .pto file: ${PTO_FILE}"
[[ -f "${PTO_FILE}" ]] || die "missing input file: ${PTO_FILE}"

if [[ -n "${ASCEND_HOME_PATH}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  set +u
  source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1
  set -u
fi

if [[ -z "${PTOAS_BIN}" ]]; then
  PTOAS_BIN="$(command -v ptoas || true)"
elif [[ "${PTOAS_BIN}" != */* ]]; then
  PTOAS_BIN="$(command -v "${PTOAS_BIN}" || true)"
fi
BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"

[[ -n "${PTOAS_BIN}" && -x "${PTOAS_BIN}" ]] || \
  die "ptoas not found; run ./quick_install.sh or set PTOAS_BIN"
command -v "${BISHENG_BIN}" >/dev/null 2>&1 || die "bisheng not found: ${BISHENG_BIN}"

pto_abs="$(cd "$(dirname "${PTO_FILE}")" && pwd)/$(basename "${PTO_FILE}")"
pto_base="$(basename "${PTO_FILE}" .pto)"

if [[ -n "${OUT_DIR_ARG}" ]]; then
  OUT_DIR="${OUT_DIR_ARG}"
else
  OUT_DIR="${ROOT_DIR}/build/vpto_quick/${pto_base}"
fi

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

LLVM_IR="${OUT_DIR}/${pto_base}.ll"
DEVICE_OBJ="${OUT_DIR}/${pto_base}.o"

log "step 1/2: lower PTO to VPTO LLVM IR"
"${PTOAS_BIN}" ${PTOAS_FLAGS} ${VPTO_FLAGS} \
  "${pto_abs}" \
  -o "${LLVM_IR}"

log "step 2/2: compile LLVM IR to device object"
"${BISHENG_BIN}" \
  --target=hiipu64-hisilicon-cce \
  -march="${AICORE_ARCH}" \
  --cce-aicore-arch="${AICORE_ARCH}" \
  --cce-aicore-only \
  ${BISHENG_FLAGS} \
  -c -x ir "${LLVM_IR}" \
  -o "${DEVICE_OBJ}"

log "done"
echo "LLVM IR: ${LLVM_IR}"
echo "Device object: ${DEVICE_OBJ}"
