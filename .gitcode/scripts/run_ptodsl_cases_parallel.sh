#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Parallel driver for the PTODSL/DSL ST suite (test/dsl-st).
#
# The suite is one "msprof op simulator" invocation of the whole directory and it
# prints no case level progress while it runs, so a single slow or hung case
# holds the entire CI step until the job deadline. Two facts make that expensive:
#
#   * "test/dsl-st/__main__.py --list" reports more than 1500 cases that live in
#     19 .py files, three of which own over 90 percent of them, and every case
#     costs a few seconds of simulator launch; and
#   * the product runner offers no case filter, so the smallest unit it can run
#     on its own is one case module, which also means it only reports results
#     once that whole module is done.
#
# This driver runs the suite as one simulator process per slice. A module with
# few cases is a slice of its own; a module with many cases is cut into fixed
# size case slices and each slice is handed to the product's own case loop by a
# generated driver (see write_slice_driver). Every slice is bounded by its own
# timeout, so a stuck case costs one timeout instead of the whole step, and
# slices are started longest first so the expensive ones are not left for last.
#
# Coverage is verified before anything runs: the case names of every slice come
# from that module's own "--list" output, and their union has to equal the names
# the suite reports. When anything about that is not trustworthy -- discovery
# fails, a module cannot list its cases, the lists disagree, or the requested
# concurrency is 1 -- the driver declines to parallelise and runs the suite in a
# single process exactly as before.
#
# Environment:
#   WORKSPACE                repository root (default: two levels above this script)
#   BUILD_ROOT               per run workspace (default: ${WORKSPACE}/.work/gitcode-vpto-sim)
#   PTODSL_CASE_JOBS         slices to run at once; 1 falls back to the serial suite
#   PTODSL_CASE_SHARD_SIZE   cases per slice (default: 100, 0 disables slicing)
#   PTODSL_CASE_TIMEOUT      per slice timeout in seconds
#   PTODSL_CASE_KILL_AFTER   grace period before a timed out slice is killed (default: 60)
#   PTODSL_DISCOVERY_TIMEOUT bound on one case listing call (default: 300)
#   PTODSL_FAILED_LOG_LINES  log tail printed for a failing slice (default: 200)
#   PTODSL_DSL_ROOT          suite directory (default: ${WORKSPACE}/test/dsl-st)
#   PYTHON_BIN               interpreter used to list cases (default: python3)
#   SIM_DSL_SCRIPT           serial runner (default: ${WORKSPACE}/scripts/sim_dsl.sh)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

WORKSPACE="${WORKSPACE:-${REPO_ROOT}}"
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-vpto-sim}"
DSL_ROOT="${PTODSL_DSL_ROOT:-${WORKSPACE}/test/dsl-st}"
SIM_DSL_SCRIPT="${SIM_DSL_SCRIPT:-${WORKSPACE}/scripts/sim_dsl.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${PTODSL_CASE_JOBS:-1}"
SHARD_SIZE="${PTODSL_CASE_SHARD_SIZE:-50}"
CASE_TIMEOUT="${PTODSL_CASE_TIMEOUT:-1200}"
KILL_AFTER="${PTODSL_CASE_KILL_AFTER:-60}"
DISCOVERY_TIMEOUT="${PTODSL_DISCOVERY_TIMEOUT:-300}"
FAILED_LOG_LINES="${PTODSL_FAILED_LOG_LINES:-200}"

WORK_DIR="${BUILD_ROOT}/ptodsl-cases"
RESULT_DIR="${WORK_DIR}/results"
MODULE_DIR="${RESULT_DIR}/modules"
SUMMARY_FILE="${WORK_DIR}/parallel-summary.tsv"
SLICE_DRIVER="${WORK_DIR}/run_case_slice.py"

log() {
  echo "[ptodsl-parallel] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_integer() {
  local name="$1" value="$2" minimum="$3"
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt "${minimum}" ]]; then
    die "${name} must be an integer >= ${minimum}, got: ${value}"
  fi
}

[[ -d "${DSL_ROOT}" ]] || die "missing DSL ST suite directory: ${DSL_ROOT}"
[[ -f "${SIM_DSL_SCRIPT}" ]] || die "missing serial PTODSL runner: ${SIM_DSL_SCRIPT}"
require_integer PTODSL_CASE_JOBS "${JOBS}" 1
require_integer PTODSL_CASE_SHARD_SIZE "${SHARD_SIZE}" 0
require_integer PTODSL_CASE_TIMEOUT "${CASE_TIMEOUT}" 1
require_integer PTODSL_CASE_KILL_AFTER "${KILL_AFTER}" 1
require_integer PTODSL_DISCOVERY_TIMEOUT "${DISCOVERY_TIMEOUT}" 1
require_integer PTODSL_FAILED_LOG_LINES "${FAILED_LOG_LINES}" 1

# A previous run of this job left its own case lists and results behind, and a
# stale list would make the coverage check below reject a perfectly good run.
rm -rf "${RESULT_DIR}"
mkdir -p "${WORK_DIR}" "${MODULE_DIR}"

# The serial path stays byte for byte the command the CI step used before this
# driver existed; it is both the fallback and what a single job degree means.
run_serial_suite() {
  log "serial suite: bash ${SIM_DSL_SCRIPT} ${DSL_ROOT}"
  bash "${SIM_DSL_SCRIPT}" "${DSL_ROOT}"
}

fallback_to_serial() {
  local reason="$1" status=0
  log "not parallelising: ${reason}"
  log "running the whole suite in one simulator process, as before."
  set +e
  run_serial_suite
  status=$?
  set -e
  exit "${status}"
}

if [[ "${JOBS}" -lt 2 ]]; then
  fallback_to_serial "PTODSL_CASE_JOBS=${JOBS}"
fi
if ! command -v timeout >/dev/null 2>&1; then
  fallback_to_serial "the timeout command is unavailable, so a slice cannot be bounded"
fi

list_suite_cases() {
  timeout "${DISCOVERY_TIMEOUT}" "${PYTHON_BIN}" "${DSL_ROOT}/__main__.py" --list
}

list_module_cases() {
  timeout "${DISCOVERY_TIMEOUT}" "${PYTHON_BIN}" "$1" --list
}

# The slice driver hands a case range of one module to the product's own case
# loop, common.run_cases, so the cases run exactly as the suite runs them; only
# which process they run in changes. Importing the module cannot run it: the
# modules guard their entry point with auto_main, which does nothing unless the
# module is __main__.
write_slice_driver() {
  cat > "${SLICE_DRIVER}" <<'PY'
#!/usr/bin/env python3
"""Run one case slice of a PTODSL/DSL ST module.

Written by .gitcode/scripts/run_ptodsl_cases_parallel.sh. Arguments:
    run_case_slice.py <module.py> <begin> <end>
"""

import importlib.util
import sys
from pathlib import Path

module_path = Path(sys.argv[1]).resolve()
begin = int(sys.argv[2])
end = int(sys.argv[3])

sys.path.insert(0, str(module_path.parent))
import common  # noqa: E402  pylint: disable=wrong-import-position

spec = importlib.util.spec_from_file_location("_dsl_st_slice", module_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load case module: {module_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

cases = module.CASES[begin:end]
if not cases:
    raise SystemExit(f"empty case slice {begin}:{end} of {module_path}")

raise SystemExit(common.run_cases(cases, argv=[]))
PY
}

# Case discovery is the product's own: the suite list is the authoritative set of
# case names, and every module reports its cases through the same entry point.
DISCOVERY_LOG="${RESULT_DIR}/discovery.log"
: > "${DISCOVERY_LOG}"

suite_cases=""
if ! suite_cases="$(list_suite_cases 2>"${DISCOVERY_LOG}")"; then
  fallback_to_serial "case discovery failed; see ${DISCOVERY_LOG}"
fi
if [[ -z "${suite_cases}" ]]; then
  fallback_to_serial "case discovery listed no case"
fi

if ! write_slice_driver; then
  log "cannot write the slice driver; running one process per case module"
  SHARD_SIZE=0
fi

module_names=()
declare -A module_case_count=()
declare -A module_path_of=()

for path in "${DSL_ROOT}"/*.py; do
  if [[ ! -f "${path}" ]]; then
    continue
  fi
  module="$(basename -- "${path}" .py)"
  case "${module}" in
    common | __main__ | _*) continue ;;
  esac
  module_cases=""
  if ! module_cases="$(list_module_cases "${path}" 2>>"${DISCOVERY_LOG}")"; then
    fallback_to_serial "cannot list the cases of ${module}.py; see ${DISCOVERY_LOG}"
  fi
  if [[ -z "${module_cases}" ]]; then
    log "skipping ${module}.py: it declares no case"
    continue
  fi
  # The order of this list is the order of the module's own case list, which is
  # what a slice indexes into, so it is stored unsorted.
  printf '%s\n' "${module_cases}" > "${MODULE_DIR}/${module}.cases"
  module_names+=("${module}")
  module_path_of["${module}"]="${path}"
  module_case_count["${module}"]="$(wc -l < "${MODULE_DIR}/${module}.cases")"
done

if (( ${#module_names[@]} == 0 )); then
  fallback_to_serial "no case module found under ${DSL_ROOT}"
fi

unit_names=()
declare -A unit_case_count=()
declare -A unit_target=()
declare -A unit_module_arg=()
declare -A unit_begin=()
declare -A unit_end=()

shard_count=0
for module in "${module_names[@]}"; do
  total="${module_case_count[${module}]}"
  if (( SHARD_SIZE < 1 || total <= SHARD_SIZE )); then
    cp "${MODULE_DIR}/${module}.cases" "${RESULT_DIR}/${module}.cases"
    unit_names+=("${module}")
    unit_case_count["${module}"]="${total}"
    unit_target["${module}"]="${module_path_of[${module}]}"
    continue
  fi
  part=0
  begin=0
  while (( begin < total )); do
    end=$(( begin + SHARD_SIZE ))
    (( end > total )) && end="${total}"
    unit="${module}.part${part}"
    sed -n "$(( begin + 1 )),${end}p" "${MODULE_DIR}/${module}.cases" > "${RESULT_DIR}/${unit}.cases"
    unit_names+=("${unit}")
    unit_case_count["${unit}"]=$(( end - begin ))
    unit_target["${unit}"]="${SLICE_DRIVER}"
    unit_module_arg["${unit}"]="${module_path_of[${module}]}"
    unit_begin["${unit}"]="${begin}"
    unit_end["${unit}"]="${end}"
    begin="${end}"
    part=$(( part + 1 ))
    shard_count=$(( shard_count + 1 ))
  done
done

# Every name the suite reports has to be owned by exactly one slice, and no slice
# may claim a name the suite does not report: otherwise the parallel run would
# silently cover a different set of cases than the serial one.
printf '%s\n' "${suite_cases}" | sort > "${RESULT_DIR}/cases-suite.txt"
cat "${RESULT_DIR}"/*.cases | sort > "${RESULT_DIR}/cases-modules.txt"
if ! diff -q "${RESULT_DIR}/cases-suite.txt" "${RESULT_DIR}/cases-modules.txt" >/dev/null; then
  uncovered="$(comm -23 "${RESULT_DIR}/cases-suite.txt" "${RESULT_DIR}/cases-modules.txt" |
    head -3 | paste -sd, - || true)"
  unexpected="$(comm -13 "${RESULT_DIR}/cases-suite.txt" "${RESULT_DIR}/cases-modules.txt" |
    head -3 | paste -sd, - || true)"
  fallback_to_serial "the slice case lists differ from the suite list" \
    "(missing: ${uncovered:-none}; unexpected: ${unexpected:-none})"
fi

# A slice's case count is a good cost proxy, because every case pays the same
# simulator launch. Start the expensive slices first and let the short ones keep
# the tail of the run busy.
readarray -t units < <(
  for unit in "${unit_names[@]}"; do
    printf '%s\t%s\n' "${unit_case_count[${unit}]}" "${unit}"
  done | sort -k1,1nr -k2,2 | cut -f2
)

total_cases="$(wc -l < "${RESULT_DIR}/cases-suite.txt")"
log "=== PTODSL/DSL ST parallel ==="
log "suite: ${DSL_ROOT}"
log "modules=${#module_names[@]} slices=${#units[@]} cases=${total_cases}"
log "jobs=${JOBS} slice size=${SHARD_SIZE} timeout=${CASE_TIMEOUT}s (kill after ${KILL_AFTER}s)"
log "per slice log: ${WORK_DIR}/<slice>.log; msprof output: ${WORK_DIR}/<slice>/"

: > "${SUMMARY_FILE}"

declare -A pid_to_unit=()

launch_unit() {
  local unit="$1"
  local log_file="${WORK_DIR}/${unit}.log"
  local out_dir="${WORK_DIR}/${unit}"
  local -a command=(bash "${SIM_DSL_SCRIPT}" --output "${out_dir}" "${unit_target[${unit}]}")
  if [[ -n "${unit_module_arg[${unit}]:-}" ]]; then
    command+=(-- "${unit_module_arg[${unit}]}" "${unit_begin[${unit}]}" "${unit_end[${unit}]}")
  fi
  local started
  started="$(date +%s)"
  (
    local rc=0
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    set +e
    # timeout signals the whole process group, so a slice that hangs inside
    # msprof cannot leave simulator threads running past the grace period.
    timeout --signal=TERM --kill-after="${KILL_AFTER}" "${CASE_TIMEOUT}" "${command[@]}" \
      > "${log_file}" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\n' "${rc}" "$(( $(date +%s) - started ))" > "${RESULT_DIR}/${unit}.result"
  ) &
  pid_to_unit[$!]="${unit}"
}

# comm only reports the intersection of sorted inputs, and a slice case list keeps
# the module's own order (it is what a slice indexes into), so both sides go
# through sort and nothing here may end the run early.
count_passed_cases() {
  local unit="$1" count=0
  awk '/^PASS /{print $2}' "${WORK_DIR}/${unit}.log" 2>/dev/null | sort -u \
    > "${RESULT_DIR}/${unit}.passed" || true
  sort -u "${RESULT_DIR}/${unit}.cases" > "${RESULT_DIR}/${unit}.cases.sorted" || true
  count="$(comm -12 "${RESULT_DIR}/${unit}.passed" "${RESULT_DIR}/${unit}.cases.sorted" \
    2>/dev/null | wc -l || true)"
  printf '%s\n' "${count}"
}

reap_unit() {
  local pid="$1"
  local unit="${pid_to_unit[${pid}]}"
  unset "pid_to_unit[${pid}]"

  local rc=-1 elapsed=0
  if [[ -f "${RESULT_DIR}/${unit}.result" ]]; then
    read -r rc elapsed < "${RESULT_DIR}/${unit}.result"
  fi

  local status="FAIL" reason="exit code ${rc}"
  case "${rc}" in
    0) status="PASS" ;;
    124) status="TIMEOUT" reason="no result within ${CASE_TIMEOUT}s" ;;
    137) status="TIMEOUT" reason="still running ${KILL_AFTER}s after the timeout signal" ;;
  esac

  local cases_expected="${unit_case_count[${unit}]}" cases_passed
  cases_passed="$(count_passed_cases "${unit}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${unit}" "${status}" "${rc}" "${elapsed}" "${cases_expected}" "${cases_passed}" \
    >> "${SUMMARY_FILE}"

  if [[ "${status}" == "PASS" ]]; then
    # Same progress the serial run printed: one result line per case, so this
    # step keeps reporting every case it validated.
    log "[${unit}] PASS cases=${cases_passed}/${cases_expected} ${elapsed}s"
    cat "${WORK_DIR}/${unit}.log"
  else
    log "[${unit}] ${status} cases=${cases_passed}/${cases_expected} ${elapsed}s" \
      "(${reason}; full log ${WORK_DIR}/${unit}.log)"
    local lines
    lines="$(wc -l < "${WORK_DIR}/${unit}.log")"
    tail -n "${FAILED_LOG_LINES}" "${WORK_DIR}/${unit}.log"
    if (( lines > FAILED_LOG_LINES )); then
      log "[${unit}] ${lines} log lines in total; the rest is in ${WORK_DIR}/${unit}.log"
    fi
  fi
}

# A driver that exits while slices are still running must not leave simulator
# processes behind: they would hold the CI step's output pipe open and keep
# burning the runner's CPU after the step already reported its result.
terminate_slices() {
  local pid
  for pid in "${!pid_to_unit[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
}
trap terminate_slices EXIT

started_at="$(date +%s)"
next_index=0
while (( next_index < ${#units[@]} )) || (( ${#pid_to_unit[@]} > 0 )); do
  while (( next_index < ${#units[@]} )) && (( ${#pid_to_unit[@]} < JOBS )); do
    launch_unit "${units[${next_index}]}"
    next_index=$(( next_index + 1 ))
  done
  if (( ${#pid_to_unit[@]} == 0 )); then
    continue
  fi
  # Wait for any slice to finish and then collect the ones that are gone, so a
  # slice that timed out stops only itself and every other result is still
  # reported before this step decides whether the suite passed.
  wait -n 2>/dev/null || true
  for pid in "${!pid_to_unit[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      reap_unit "${pid}"
      break
    fi
  done
done
elapsed_total=$(( $(date +%s) - started_at ))

pass_units="$(awk -F '\t' '$2 == "PASS" {count++} END {print count + 0}' "${SUMMARY_FILE}")"
fail_units="$(awk -F '\t' '$2 == "FAIL" {count++} END {print count + 0}' "${SUMMARY_FILE}")"
timeout_units="$(awk -F '\t' '$2 == "TIMEOUT" {count++} END {print count + 0}' "${SUMMARY_FILE}")"
passed_cases="$(awk -F '\t' '{count += $6} END {print count + 0}' "${SUMMARY_FILE}")"

log "=== PTODSL/DSL ST summary ==="
log "slices: total=${#units[@]} (sharded modules=${shard_count}) PASS=${pass_units}" \
  "FAIL=${fail_units} TIMEOUT=${timeout_units}"
log "cases:  passed=${passed_cases}/${total_cases}"
log "elapsed: ${elapsed_total}s (jobs=${JOBS}, slice timeout=${CASE_TIMEOUT}s)"
log "per slice results: ${SUMMARY_FILE}"

if (( fail_units + timeout_units > 0 )); then
  log "one or more slices did not pass; see the slice logs above"
  exit 1
fi

log "all ${passed_cases} case(s) passed"
