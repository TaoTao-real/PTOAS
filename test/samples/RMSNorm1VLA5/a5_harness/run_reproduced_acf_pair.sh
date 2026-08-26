#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <experiment-dir> <source-root> <pto-isa-root>" >&2
  exit 2
fi

experiment_dir=$1
source_root=$2
pto_isa_root=$3
profile_repeats=${PROFILE_REPEATS:-10}
profile_attempts=${PROFILE_ATTEMPTS:-3}
profile_cooldown_seconds=${PROFILE_COOLDOWN_SECONDS:-15}
device=${ACL_DEVICE_ID:-1}
cann_root=${ASCEND_HOME_PATH:-/data/s00454010/Ascend/cann-9.2.0}
msprof=${MSPROF:-$cann_root/tools/profiler/bin/msprof}
shared_harness="$source_root/test/samples/RMSNormRowVFA5/a5_harness"
fixture="$experiment_dir/inputs/fixture"

test -f "$experiment_dir/manifest.env"
test -x "$experiment_dir/inputs/prebuilt/D"
test -f "$experiment_dir/inputs/prebuilt/libD_kernel.so"
test -f "$experiment_dir/inputs/ascendc/CMakeLists.txt"
test -x "$msprof"

set +u
source "$cann_root/set_env.sh" >/dev/null 2>&1
set -u
export ACL_DEVICE_ID="$device"
export LD_LIBRARY_PATH="$experiment_dir/inputs/prebuilt:${LD_LIBRARY_PATH:-}"

mkdir -p "$experiment_dir/work/build-acf" "$experiment_dir/work/runs" \
  "$experiment_dir/artifacts/profiles" "$experiment_dir/logs" \
  "$experiment_dir/results"
cmake -S "$experiment_dir/inputs/ascendc" \
  -B "$experiment_dir/work/build-acf" -DPTO_ISA_ROOT="$pto_isa_root" \
  >"$experiment_dir/logs/configure-acf.log" 2>&1
cmake --build "$experiment_dir/work/build-acf" --target AC-F-1VL -j2 \
  >"$experiment_dir/logs/build-acf.log" 2>&1

binary_for() {
  case "$1" in
    D) printf '%s\n' "$experiment_dir/inputs/prebuilt/D" ;;
    ACF) printf '%s\n' "$experiment_dir/work/build-acf/AC-F-1VL" ;;
    *) return 2 ;;
  esac
}

next_unique_path() {
  local base=$1 candidate=$1 attempt=0
  while [[ -e "$candidate" ]]; do
    attempt=$((attempt + 1))
    candidate="${base}-retry-${attempt}"
  done
  printf '%s\n' "$candidate"
}

printf 'variant\trun_kind\trun\trc\toutput_sha256\tgolden_sha256\tresult\n' \
  >"$experiment_dir/results/correctness.tsv"
run_plain() {
  local variant=$1 run_kind=$2 run=$3 binary run_dir rc output_hash golden_hash
  binary=$(binary_for "$variant")
  run_dir=$(next_unique_path \
    "$experiment_dir/work/runs/${variant}-${run_kind}-${run}")
  mkdir -p "$run_dir"
  cp "$fixture/x.bin" "$run_dir/x.bin"
  cp "$fixture/gamma.bin" "$run_dir/gamma.bin"
  cp "$fixture/y.init.bin" "$run_dir/y.bin"
  set +e
  (cd "$run_dir" && timeout 120 "$binary") \
    >"$experiment_dir/logs/${variant}-${run_kind}-${run}.stdout" \
    2>"$experiment_dir/logs/${variant}-${run_kind}-${run}.stderr"
  rc=$?
  set -e
  output_hash=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  golden_hash=$(sha256sum "$fixture/golden.bin" | awk '{print $1}')
  result=FAIL
  if ((rc == 0)) && cmp -s "$run_dir/y.bin" "$fixture/golden.bin"; then
    result=PASS
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$run_kind" "$run" "$rc" "$output_hash" \
    "$golden_hash" "$result" >>"$experiment_dir/results/correctness.tsv"
  [[ "$result" == PASS ]]
}

for variant in D ACF; do
  run_plain "$variant" cold 0
  run_plain "$variant" nonprofile 1
  run_plain "$variant" nonprofile 2
done

printf 'rows\tvariant\trepeat\ttask_time_us\ttask_duration_us\taiv_time_us\taiv_total_cycles\taiv_vec_time_us\taiv_scalar_time_us\taiv_mte2_time_us\taiv_mte3_time_us\toutput_sha256\tprofile\n' \
  >"$experiment_dir/results/samples.tsv"

run_profile() {
  local variant=$1 repeat=$2 binary run_dir profile log_base rc attempt
  local digest metrics
  binary=$(binary_for "$variant")
  for ((attempt = 1; attempt <= profile_attempts; ++attempt)); do
    run_dir=$(next_unique_path \
      "$experiment_dir/work/runs/${variant}-profile-${repeat}")
    profile=$(next_unique_path \
      "$experiment_dir/artifacts/profiles/${variant}/profile-${repeat}")
    log_base="$experiment_dir/logs/${variant}-profile-${repeat}"
    if ((attempt > 1)); then
      log_base="${log_base}-attempt-${attempt}"
    fi
    mkdir -p "$run_dir" "$profile"
    cp "$fixture/x.bin" "$run_dir/x.bin"
    cp "$fixture/gamma.bin" "$run_dir/gamma.bin"
    cp "$fixture/y.init.bin" "$run_dir/y.bin"
    set +e
    (cd "$run_dir" && timeout 300 "$msprof" \
      --output="$profile" --application="$binary") \
      >"${log_base}.stdout" 2>"${log_base}.stderr"
    rc=$?
    set -e
    if ((rc == 0)); then
      break
    fi
    printf 'profile failed: variant=%s repeat=%s attempt=%s rc=%s\n' \
      "$variant" "$repeat" "$attempt" "$rc" >&2
    if ((attempt == profile_attempts)); then
      return "$rc"
    fi
    sleep "$profile_cooldown_seconds"
  done
  cmp -s "$run_dir/y.bin" "$fixture/golden.bin"
  digest=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  metrics=$(/usr/bin/python3 "$shared_harness/extract_profile.py" "$profile")
  printf '8\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$repeat" "$metrics" "$digest" "$profile" \
    >>"$experiment_dir/results/samples.tsv"
  sleep "$profile_cooldown_seconds"
}

run_plain D warmup 0
run_plain ACF warmup 0
for ((repeat = 1; repeat <= profile_repeats; ++repeat)); do
  if ((repeat % 2)); then
    order=(ACF D)
  else
    order=(D ACF)
  fi
  for variant in "${order[@]}"; do
    run_profile "$variant" "$repeat"
  done
done

/usr/bin/python3 "$shared_harness/summarize_profiles.py" \
  "$experiment_dir/results/samples.tsv" \
  --output-tsv "$experiment_dir/results/performance-summary.tsv" \
  --output-json "$experiment_dir/results/performance-summary.json"
/usr/bin/python3 "$shared_harness/paired_gate.py" \
  "$experiment_dir/results/samples.tsv" --candidate D --baseline ACF \
  --output "$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.json" \
  >"$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.stdout"

