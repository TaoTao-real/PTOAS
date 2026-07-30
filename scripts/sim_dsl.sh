#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Run a PTODSL JIT example under `msprof op simulator`.

Usage:
  scripts/sim_dsl.sh [options] <example.py|dsl-st-dir> [-- <example args...>]

Options:
  --output <dir>        Final directory to sync results into after the run.
  --soc-version <soc>   Override simulator soc version. Default: Ascend950PR_9599
  --verbose-msprof      Show the full `msprof` simulator log stream.
  --quiet-msprof        Suppress noisy `msprof` INFO/WARN/ERROR lines. Default.
  -h, --help            Show this help.

Environment:
  PTOAS_MSPROF_PRIVATE_ROOT
                        Private root used for the actual `msprof --output`.
                        Defaults to `$XDG_RUNTIME_DIR/ptoas-msprof` when available,
                        otherwise `$HOME/.cache/ptoas/msprof`.
  PTOAS_KEEP_MSPROF_STAGING=1
                        Keep the private staging directory after a successful sync.
  PTOAS_MSPROF_LOG_MODE=quiet|verbose
                        Override the default simulator log rendering mode.
  PYTHON_BIN             Python executable used for the PTODSL example.
                        Defaults to python3.

Examples:
  scripts/sim_dsl.sh ptodsl/examples/jit/tadd_launch.py
  scripts/sim_dsl.sh test/dsl-st
  scripts/sim_dsl.sh \
    --output "$PWD/build/msprof_res/flash_softmax" \
    ptodsl/examples/jit/flash_attention_softmax_launch.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[sim_dsl] $*"
}

resolve_executable() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    command -v -- "${candidate}"
    return 0
  fi
  command -v -- "${candidate}"
}

prepend_path() {
  local dir="$1"
  if [[ -z "${dir}" || ! -d "${dir}" ]]; then
    return 0
  fi
  if [[ ":${PATH}:" == *":${dir}:"* ]]; then
    return 0
  fi
  PATH="${dir}:${PATH}"
  export PATH
}

ensure_private_dir() {
  local dir="$1"
  umask 077
  mkdir -p "${dir}"
  chmod 700 "${dir}"
}

sync_msprof_output() {
  local src_dir="$1"
  local dst_dir="$2"

  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"
}

print_msprof_log() {
  local log_file="$1"
  local mode="$2"
  local status="$3"

  if [[ "${mode}" == "verbose" ]]; then
    cat "${log_file}"
    return
  fi

  grep -Ev '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
    "${log_file}" || true

  if [[ ${status} -ne 0 ]]; then
    local suppressed_count
    suppressed_count=$(wc -l < "${log_file}")
    local filtered_count
    filtered_count=$(
      grep -Evc '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
        "${log_file}" || true
    )
    suppressed_count=$((suppressed_count - filtered_count))
    if [[ ${suppressed_count} -gt 0 ]]; then
      log "msprof failure tail:"
      tail -n 20 "${log_file}"
    fi
    log "full msprof log saved at ${log_file}"
  fi
}

python_can_import_ptodsl() {
  local python_bin="$1"
  "${python_bin}" - <<'PY' >/dev/null 2>&1
import ptoas.mlir.ir  # noqa: F401
from ptodsl import pto  # noqa: F401
PY
}

SOC_VERSION="Ascend950PR_9599"
OUTPUT_DIR=""
EXAMPLE_PATH=""
EXAMPLE_ARGS=()
MSPROF_LOG_MODE="${PTOAS_MSPROF_LOG_MODE:-quiet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --soc-version)
      [[ $# -ge 2 ]] || die "--soc-version requires a value"
      SOC_VERSION="$2"
      shift 2
      ;;
    --verbose-msprof)
      MSPROF_LOG_MODE="verbose"
      shift
      ;;
    --quiet-msprof)
      MSPROF_LOG_MODE="quiet"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXAMPLE_ARGS=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${EXAMPLE_PATH}" ]]; then
        EXAMPLE_PATH="$1"
      else
        EXAMPLE_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

[[ -n "${EXAMPLE_PATH}" ]] || die "missing <example.py|dsl-st-dir>"

if [[ "${EXAMPLE_PATH}" != /* ]]; then
  EXAMPLE_PATH="${REPO_ROOT}/${EXAMPLE_PATH}"
fi
if [[ -d "${EXAMPLE_PATH}" ]]; then
  EXAMPLE_PATH="${EXAMPLE_PATH}/__main__.py"
fi
[[ -f "${EXAMPLE_PATH}" ]] || die "example script not found: ${EXAMPLE_PATH}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  die "ASCEND_HOME_PATH is not set; source CANN setenv or export it first"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  EXAMPLE_STEM="$(basename -- "${EXAMPLE_PATH}" .py)"
  OUTPUT_DIR="${REPO_ROOT}/build/msprof_res/${EXAMPLE_STEM}"
else
  EXAMPLE_STEM="$(basename -- "${EXAMPLE_PATH}" .py)"
fi

SIM_LIB_DIR="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
[[ -d "${SIM_LIB_DIR}" ]] || die "simulator library directory not found: ${SIM_LIB_DIR}"

PRIVATE_ROOT="${PTOAS_MSPROF_PRIVATE_ROOT:-${XDG_RUNTIME_DIR:-${HOME}/.cache}/ptoas/msprof}"
ensure_private_dir "${PRIVATE_ROOT}"
RUNTIME_OUTPUT_DIR="$(mktemp -d "${PRIVATE_ROOT}/${EXAMPLE_STEM}.XXXXXX")"
chmod 700 "${RUNTIME_OUTPUT_DIR}"
MSPROF_STDIO_LOG="${RUNTIME_OUTPUT_DIR}/msprof.stdout.log"
EXAMPLE_EXIT_CODE_FILE="${RUNTIME_OUTPUT_DIR}/example.exitcode"
EXAMPLE_LAUNCHER="${RUNTIME_OUTPUT_DIR}/run_example.sh"
REQUESTED_PYTHON_BIN="${PTO_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
REQUESTED_PTOAS_BIN="${PTOAS_BIN:-}"

if ! RESOLVED_PYTHON_BIN="$(resolve_executable "${REQUESTED_PYTHON_BIN}")"; then
  die "PYTHON_BIN is not executable or not found on PATH: ${REQUESTED_PYTHON_BIN}"
fi
if [[ -n "${REQUESTED_PTOAS_BIN}" ]]; then
  if ! RESOLVED_PTOAS_BIN="$(resolve_executable "${REQUESTED_PTOAS_BIN}")"; then
    die "PTOAS_BIN is not executable or not found on PATH: ${REQUESTED_PTOAS_BIN}"
  fi
fi

source "${ASCEND_HOME_PATH}/bin/setenv.bash"
if [[ -n "${RESOLVED_PTOAS_BIN:-}" ]]; then
  prepend_path "$(dirname -- "${RESOLVED_PTOAS_BIN}")"
  export PTOAS_BIN="${RESOLVED_PTOAS_BIN}"
fi
if ! python_can_import_ptodsl "${RESOLVED_PYTHON_BIN}"; then
  die "active Python environment cannot import ptodsl/ptoas.mlir.ir; install the PTOAS wheel or use a preconfigured environment"
fi
if ! command -v ptoas >/dev/null 2>&1; then
  die "ptoas is not available on PATH; install the PTOAS wheel or export PTOAS_BIN"
fi
log "using installed Python environment from ${RESOLVED_PYTHON_BIN}"
log "using ptoas from $(command -v ptoas)"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

cat > "${EXAMPLE_LAUNCHER}" <<'EOF'
#!/usr/bin/env bash
set +e
"${PTOAS_SIM_DSL_PYTHON_BIN}" "${PTOAS_SIM_DSL_EXAMPLE_PATH}" "$@"
status=$?
printf '%s\n' "${status}" > "${PTOAS_SIM_DSL_EXIT_CODE_FILE}"
exit "${status}"
EOF
chmod 700 "${EXAMPLE_LAUNCHER}"
export PTOAS_SIM_DSL_PYTHON_BIN="${RESOLVED_PYTHON_BIN}"
export PTOAS_SIM_DSL_EXAMPLE_PATH="${EXAMPLE_PATH}"
export PTOAS_SIM_DSL_EXIT_CODE_FILE="${EXAMPLE_EXIT_CODE_FILE}"

# msprof rejects group/other-writable working directories, so always launch
# from a private directory and use an absolute path for the example script.
cd "${HOME}"

log "staging msprof output in ${RUNTIME_OUTPUT_DIR}"
if [[ "${OUTPUT_DIR}" != "${RUNTIME_OUTPUT_DIR}" ]]; then
  log "final results will be synced to ${OUTPUT_DIR}"
fi

set +e
msprof op simulator \
  --soc-version="${SOC_VERSION}" \
  --output="${RUNTIME_OUTPUT_DIR}" \
  "${EXAMPLE_LAUNCHER}" "${EXAMPLE_ARGS[@]}" \
  > "${MSPROF_STDIO_LOG}" 2>&1
MSPROF_STATUS=$?
set -e

EXAMPLE_STATUS=0
if [[ -f "${EXAMPLE_EXIT_CODE_FILE}" ]]; then
  EXAMPLE_STATUS="$(< "${EXAMPLE_EXIT_CODE_FILE}")"
  if [[ ! "${EXAMPLE_STATUS}" =~ ^[0-9]+$ ]]; then
    log "invalid example exit code recorded in ${EXAMPLE_EXIT_CODE_FILE}: ${EXAMPLE_STATUS}"
    EXAMPLE_STATUS=1
  fi
else
  log "example exit code file was not produced: ${EXAMPLE_EXIT_CODE_FILE}"
  EXAMPLE_STATUS=1
fi

STATUS=0
if [[ ${MSPROF_STATUS} -ne 0 ]]; then
  STATUS=${MSPROF_STATUS}
elif [[ ${EXAMPLE_STATUS} -ne 0 ]]; then
  STATUS=${EXAMPLE_STATUS}
fi

print_msprof_log "${MSPROF_STDIO_LOG}" "${MSPROF_LOG_MODE}" "${STATUS}"

SYNC_STATUS=0
if [[ -d "${RUNTIME_OUTPUT_DIR}" ]]; then
  if sync_msprof_output "${RUNTIME_OUTPUT_DIR}" "${OUTPUT_DIR}"; then
    log "synced msprof results to ${OUTPUT_DIR}"
  else
    SYNC_STATUS=$?
    log "failed to sync msprof results to ${OUTPUT_DIR}"
  fi
fi

if [[ "${PTOAS_KEEP_MSPROF_STAGING:-0}" != "1" && ${SYNC_STATUS} -eq 0 ]]; then
  rm -rf "${RUNTIME_OUTPUT_DIR}"
else
  log "kept staging directory at ${RUNTIME_OUTPUT_DIR}"
fi

if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi

exit ${SYNC_STATUS}
