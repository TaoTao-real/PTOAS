#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <experiment-dir> <source-root>" >&2
  exit 2
fi

experiment_dir=$1
source_root=$2
profile_start=${PROFILE_START:-11}
profile_end=${PROFILE_END:-20}
device=${ACL_DEVICE_ID:-1}
cann_root=${ASCEND_HOME_PATH:-/data/s00454010/Ascend/cann-9.2.0}
msprof=${MSPROF:-$cann_root/tools/profiler/bin/msprof}
shared_harness="$source_root/test/samples/RMSNormRowVFA5/a5_harness"

test -x "$msprof"
test -f "$experiment_dir/results/samples.tsv"
set +u
source "$cann_root/set_env.sh" >/dev/null 2>&1
set -u
export ACL_DEVICE_ID="$device"

run_profile() {
  local variant=$1 repeat=$2
  local fixture="$experiment_dir/inputs/fixture"
  local binary="$experiment_dir/work/build/${variant}"
  local run_dir="$experiment_dir/work/runs/${variant}-profile-${repeat}"
  local profile="$experiment_dir/artifacts/profiles/${variant}/profile-${repeat}"
  mkdir -p "$run_dir" "$profile"
  cp "$fixture/x.bin" "$run_dir/x.bin"
  cp "$fixture/gamma.bin" "$run_dir/gamma.bin"
  cp "$fixture/y.init.bin" "$run_dir/y.bin"
  (cd "$run_dir" && timeout 300 "$msprof" \
    --output="$profile" --application="$binary") \
    >"$experiment_dir/logs/${variant}-profile-${repeat}.stdout" \
    2>"$experiment_dir/logs/${variant}-profile-${repeat}.stderr"
  cmp -s "$run_dir/y.bin" "$fixture/golden.bin"
  local digest metrics
  digest=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  metrics=$(/usr/bin/python3 "$shared_harness/extract_profile.py" "$profile")
  printf '8\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$repeat" "$metrics" "$digest" "$profile" \
    >>"$experiment_dir/results/samples.tsv"
}

for variant in D DL; do
  fixture="$experiment_dir/inputs/fixture"
  warmup="$experiment_dir/work/runs/${variant}-extension-warmup"
  mkdir -p "$warmup"
  cp "$fixture/x.bin" "$warmup/x.bin"
  cp "$fixture/gamma.bin" "$warmup/gamma.bin"
  cp "$fixture/y.init.bin" "$warmup/y.bin"
  (cd "$warmup" && timeout 120 "$experiment_dir/work/build/${variant}")
  cmp -s "$warmup/y.bin" "$fixture/golden.bin"
done
for ((repeat = profile_start; repeat <= profile_end; ++repeat)); do
  if ((repeat % 2)); then
    order=(DL D)
  else
    order=(D DL)
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
  "$experiment_dir/results/samples.tsv" \
  --output "$experiment_dir/results/paired-gate-${profile_end}.json" \
  >"$experiment_dir/results/paired-gate-${profile_end}.stdout"
