#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <experiment-dir> <source-root>" >&2
  exit 2
fi

experiment_dir=$1
source_root=$2
widths_list=${WIDTHS_LIST:?WIDTHS_LIST is required}
profile_start=${PROFILE_START:-11}
profile_end=${PROFILE_END:-20}
device=${ACL_DEVICE_ID:-1}
cann_root=${ASCEND_HOME_PATH:-/data/s00454010/Ascend/cann-9.2.0}
msprof=${MSPROF:-$cann_root/tools/profiler/bin/msprof}
dataset=${PROFILE_DATASET:-finite-sensitive}
harness="$source_root/test/samples/SoftmaxDndVFA5/a5_harness"

test -x "$msprof"
test -f "$experiment_dir/results/samples.tsv"
set +u
source "$cann_root/set_env.sh" >/dev/null 2>&1
set -u
export ACL_DEVICE_ID="$device"

run_profile() {
  local width=$1 variant=$2 repeat=$3
  local fixture="$experiment_dir/inputs/fixture/n${width}/${dataset}"
  local binary="$experiment_dir/work/build/n${width}/${variant}"
  local run_dir="$experiment_dir/work/runs/n${width}-${variant}-profile-${repeat}"
  local profile="$experiment_dir/artifacts/profiles/n${width}/${variant}/profile-${repeat}"
  mkdir -p "$run_dir" "$profile"
  cp "$fixture/x.bin" "$run_dir/x.bin"
  cp "$fixture/y.init.bin" "$run_dir/y.bin"
  (cd "$run_dir" && timeout 300 "$msprof" \
    --output="$profile" --application="$binary") \
    >"$experiment_dir/logs/n${width}-${variant}-profile-${repeat}.stdout" \
    2>"$experiment_dir/logs/n${width}-${variant}-profile-${repeat}.stderr"
  /usr/bin/python3 "$harness/compare_output.py" \
    --actual "$run_dir/y.bin" --golden "$fixture/golden.bin" \
    --dataset "$dataset" --width "$width" >/dev/null
  local digest metrics
  digest=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  metrics=$(/usr/bin/python3 "$harness/extract_profile.py" "$profile")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$width" "$variant" "$repeat" "$metrics" "$digest" "$profile" \
    >>"$experiment_dir/results/samples.tsv"
}

for width in $widths_list; do
  for variant in D DL; do
    fixture="$experiment_dir/inputs/fixture/n${width}/${dataset}"
    warmup="$experiment_dir/work/runs/n${width}-${variant}-extension-warmup"
    mkdir -p "$warmup"
    cp "$fixture/x.bin" "$warmup/x.bin"
    cp "$fixture/y.init.bin" "$warmup/y.bin"
    (cd "$warmup" && timeout 120 "$experiment_dir/work/build/n${width}/${variant}")
    /usr/bin/python3 "$harness/compare_output.py" \
      --actual "$warmup/y.bin" --golden "$fixture/golden.bin" \
      --dataset "$dataset" --width "$width" >/dev/null
  done
  for ((repeat = profile_start; repeat <= profile_end; ++repeat)); do
    if ((repeat % 2)); then
      order=(DL D)
    else
      order=(D DL)
    fi
    for variant in "${order[@]}"; do
      run_profile "$width" "$variant" "$repeat"
    done
  done
done

/usr/bin/python3 "$harness/summarize_profiles.py" \
  "$experiment_dir/results/samples.tsv" \
  --output-tsv "$experiment_dir/results/performance-summary.tsv" \
  --output-json "$experiment_dir/results/performance-summary.json"
/usr/bin/python3 "$harness/paired_gate.py" \
  "$experiment_dir/results/samples.tsv" \
  --output "$experiment_dir/results/paired-gate-${profile_end}.json" \
  >"$experiment_dir/results/paired-gate-${profile_end}.stdout"
