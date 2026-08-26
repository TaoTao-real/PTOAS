#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <experiment-dir> <source-root>" >&2
  exit 2
fi

experiment_dir=$1
source_root=$2
widths_list=${WIDTHS_LIST:-32 64 128}
profile_repeats=${PROFILE_REPEATS:-10}
device=${ACL_DEVICE_ID:-1}
cann_root=${ASCEND_HOME_PATH:-/data/s00454010/Ascend/cann-9.2.0}
msprof=${MSPROF:-$cann_root/tools/profiler/bin/msprof}
dataset=${PROFILE_DATASET:-finite-sensitive}
profile_attempts=${PROFILE_ATTEMPTS:-3}
profile_cooldown_seconds=${PROFILE_COOLDOWN_SECONDS:-15}
harness="$source_root/test/samples/SoftmaxDndVFA5/a5_harness"
samples="$experiment_dir/results/samples.tsv"

test -x "$msprof"
test -f "$experiment_dir/manifest.env"
test -f "$samples"

set +u
source "$cann_root/set_env.sh" >/dev/null 2>&1
set -u
export ACL_DEVICE_ID="$device"

sample_exists() {
  local width=$1 variant=$2 repeat=$3
  awk -F '\t' -v w="$width" -v v="$variant" -v r="$repeat" \
    'NR > 1 && $1 == w && $2 == v && $3 == r { found = 1 } END { exit !found }' \
    "$samples"
}

next_unique_path() {
  local base=$1 candidate=$1 attempt=0
  while [[ -e "$candidate" ]]; do
    attempt=$((attempt + 1))
    candidate="${base}-retry-${attempt}"
  done
  printf '%s\n' "$candidate"
}

run_warmup() {
  local width=$1 variant=$2
  local fixture="$experiment_dir/inputs/fixture/n${width}/${dataset}"
  local binary="$experiment_dir/work/build/n${width}/${variant}"
  local run_dir
  run_dir=$(next_unique_path \
    "$experiment_dir/work/runs/n${width}-${variant}-pair-resume-warmup")
  mkdir -p "$run_dir"
  cp "$fixture/x.bin" "$run_dir/x.bin"
  cp "$fixture/y.init.bin" "$run_dir/y.bin"
  (cd "$run_dir" && timeout 120 "$binary") >/dev/null 2>&1
  /usr/bin/python3 "$harness/compare_output.py" \
    --actual "$run_dir/y.bin" --golden "$fixture/golden.bin" \
    --dataset "$dataset" --width "$width" >/dev/null
}

run_profile() {
  local width=$1 variant=$2 repeat=$3
  local fixture="$experiment_dir/inputs/fixture/n${width}/${dataset}"
  local binary="$experiment_dir/work/build/n${width}/${variant}"
  local run_dir profile log_base digest metrics rc attempt
  for ((attempt = 1; attempt <= profile_attempts; ++attempt)); do
    run_dir=$(next_unique_path \
      "$experiment_dir/work/runs/n${width}-${variant}-pair-resume-${repeat}")
    profile=$(next_unique_path \
      "$experiment_dir/artifacts/profiles-pair/n${width}/${variant}/profile-${repeat}")
    log_base="$experiment_dir/logs/n${width}-${variant}-pair-resume-${repeat}"
    if ((attempt > 1)); then
      log_base="${log_base}-attempt-${attempt}"
    fi
    mkdir -p "$run_dir" "$profile"
    cp "$fixture/x.bin" "$run_dir/x.bin"
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
    printf 'profile failed: width=%s variant=%s repeat=%s attempt=%s rc=%s\n' \
      "$width" "$variant" "$repeat" "$attempt" "$rc" >&2
    if ((attempt == profile_attempts)); then
      return "$rc"
    fi
    sleep "$profile_cooldown_seconds"
  done
  /usr/bin/python3 "$harness/compare_output.py" \
    --actual "$run_dir/y.bin" --golden "$fixture/golden.bin" \
    --dataset "$dataset" --width "$width" >/dev/null
  digest=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  metrics=$(/usr/bin/python3 "$harness/extract_profile.py" "$profile")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$width" "$variant" "$repeat" "$metrics" "$digest" "$profile" \
    >>"$samples"
  sleep "$profile_cooldown_seconds"
}

for width in $widths_list; do
  missing=0
  for variant in D ACF; do
    for ((repeat = 1; repeat <= profile_repeats; ++repeat)); do
      if ! sample_exists "$width" "$variant" "$repeat"; then
        missing=1
      fi
    done
  done
  if ((missing)); then
    run_warmup "$width" D
    run_warmup "$width" ACF
  fi
  for ((repeat = 1; repeat <= profile_repeats; ++repeat)); do
    if ((repeat % 2)); then
      order=(ACF D)
    else
      order=(D ACF)
    fi
    for variant in "${order[@]}"; do
      if ! sample_exists "$width" "$variant" "$repeat"; then
        run_profile "$width" "$variant" "$repeat"
      fi
    done
  done
done

/usr/bin/python3 "$harness/summarize_profiles.py" "$samples" \
  --output-tsv "$experiment_dir/results/performance-summary.tsv" \
  --output-json "$experiment_dir/results/performance-summary.json"
/usr/bin/python3 "$harness/paired_gate.py" "$samples" \
  --candidate D --baseline ACF \
  --output "$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.json" \
  >"$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.stdout"
