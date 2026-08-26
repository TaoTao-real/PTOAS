#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <experiment-dir> <source-root> <ptoas> <mlir-python-root> <pto-isa-root>" >&2
  exit 2
fi

experiment_dir=$1
source_root=$2
ptoas=$3
mlir_python_root=$4
pto_isa_root=$5
widths_list=${WIDTHS_LIST:-32 64 128}
profile_repeats=${PROFILE_REPEATS:-10}
device=${ACL_DEVICE_ID:-1}
cann_root=${ASCEND_HOME_PATH:-/data/s00454010/Ascend/cann-9.2.0}
msprof=${MSPROF:-$cann_root/tools/profiler/bin/msprof}
dataset=${PROFILE_DATASET:-finite-sensitive}
tag=${EXPERIMENT_TAG:-$(basename "$experiment_dir")}
harness="$source_root/test/samples/SoftmaxDndVFA5/a5_harness"
golden="$source_root/test/samples/SoftmaxDndVFA5/softmax_dnd_vf_golden.py"
canonical_pto="$source_root/test/samples/SoftmaxDndVFA5/softmax_dnd_vf.pto"
ascendc_source="$source_root/test/samples/SoftmaxDndVFA5/softmax_dnd_ascendc.cpp"
vf_softmax_include=${VF_SOFTMAX_INCLUDE:-}
attached_vf_call=${SOFTMAX_ATTACHED_VF_CALL:-0}
prebuilt_pto_experiment=${PREBUILT_PTO_EXPERIMENT:-}

test -x "$ptoas"
test -x "$msprof"
test -d "$mlir_python_root"
test -d "$pto_isa_root"
test -f "$experiment_dir/manifest.env"

set +u
source "$cann_root/set_env.sh" >/dev/null 2>&1
set -u
export ACL_DEVICE_ID="$device"
export PTO_ISA_ROOT="$pto_isa_root"
export MLIR_PYTHON_ROOT="$mlir_python_root"
export PYTHONPATH="$mlir_python_root"

mkdir -p "$experiment_dir/inputs/generated" "$experiment_dir/inputs/fixture" \
  "$experiment_dir/work/build" "$experiment_dir/work/runs" \
  "$experiment_dir/artifacts/vpto" "$experiment_dir/artifacts/profiles" \
  "$experiment_dir/logs" "$experiment_dir/results"

capture_npu_health() {
  local output=$1
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"$output" 2>&1
    return
  fi
  {
    echo "npu-smi unavailable"
    echo "driver_version:"
    cat /usr/local/Ascend/driver/version.info 2>/dev/null || true
    echo "device_nodes:"
    ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm 2>/dev/null || true
  } >"$output"
}

/usr/bin/python3 --version >"$experiment_dir/logs/python-version.log" 2>&1
/usr/bin/python3 -c 'import mlir, numpy, sys; from mlir._mlir_libs import _pto; print(sys.executable); print(sys.version); print(mlir.__path__); print(_pto.__file__); print(numpy.__file__)' \
  >"$experiment_dir/logs/python-provenance.log" 2>&1
/usr/bin/python3 "$ptoas" --version >"$experiment_dir/logs/ptoas-version.log" 2>&1
capture_npu_health "$experiment_dir/results/npu-health-before.txt"

width_args=()
for width in $widths_list; do
  width_args+=(--width "$width")
  /usr/bin/python3 "$golden" --width "$width" \
    --write-dir "$experiment_dir/inputs/fixture" \
    >"$experiment_dir/logs/golden-n${width}.log" 2>&1
done

if [[ -n "$prebuilt_pto_experiment" ]]; then
  test -f "$prebuilt_pto_experiment/FINALIZED"
  cp -a "$prebuilt_pto_experiment/inputs/generated/." \
    "$experiment_dir/inputs/generated/"
  cp -a "$prebuilt_pto_experiment/artifacts/vpto/." \
    "$experiment_dir/artifacts/vpto/"
  cp "$prebuilt_pto_experiment/results/lowering-stats.tsv" \
    "$experiment_dir/results/lowering-stats.tsv"
  {
    echo "PREBUILT_PTO_EXPERIMENT=$prebuilt_pto_experiment"
    find "$experiment_dir/inputs/generated" "$experiment_dir/artifacts/vpto" \
      -type f -print0 | sort -z | xargs -0 sha256sum
  } >"$experiment_dir/results/prebuilt-pto.sha256"
else
  /usr/bin/python3 "$harness/generate_pto_variants.py" \
    --source "$canonical_pto" \
    --output-dir "$experiment_dir/inputs/generated" \
    --experiment-tag "$tag" "${width_args[@]}"

  common_ptoas_flags=(
    --pto-arch=a5
    --pto-backend=vpto
    --pto-level=level3
    --tile-lib-backend=ptodsl
    --enable-insert-sync
    --enable-vecscope-mem-bar
    --bisheng-vf-auto-sync=off
  )

  printf 'width\tvariant\tscf_for\tvld\tvst\tmem_bar\tvexpdif\tresidual_vmi_ops\tfat_object_bytes\n' \
    >"$experiment_dir/results/lowering-stats.tsv"
  while IFS=$'\t' read -r width variant symbol policy mode state_mode pto_file; do
    [[ "$width" == width ]] && continue
    generated="$experiment_dir/inputs/generated"
    object_root="$generated/n${width}-objects"
    mkdir -p "$object_root"
    flags=("${common_ptoas_flags[@]}" \
      --tilelib-candidate-policy="$policy" --vmi-fusion-mode="$mode" \
      --vmi-state-promotion-mode="$state_mode")
    /usr/bin/python3 "$ptoas" "${flags[@]}" --emit-vpto \
      "$generated/$pto_file" -o "$experiment_dir/artifacts/vpto/n${width}-${variant}.mlir" \
      >"$experiment_dir/logs/n${width}-${variant}-vpto.stdout" \
      2>"$experiment_dir/logs/n${width}-${variant}-vpto.stderr"
    /usr/bin/python3 "$ptoas" "${flags[@]}" \
      "$generated/$pto_file" -o "$object_root/${variant}.fat.o" \
      >"$experiment_dir/logs/n${width}-${variant}-object.stdout" \
      2>"$experiment_dir/logs/n${width}-${variant}-object.stderr"
    ir="$experiment_dir/artifacts/vpto/n${width}-${variant}.mlir"
    residual=$(grep -E -c '^[[:space:]]*(%[^=]+=[[:space:]]*)?pto\.vmi\.' "$ir" || true)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$width" "$variant" \
      "$(grep -c 'scf.for' "$ir" || true)" \
      "$(grep -c 'pto.vld' "$ir" || true)" \
      "$(grep -c 'pto.vst' "$ir" || true)" \
      "$(grep -c 'pto.mem_bar' "$ir" || true)" \
      "$(grep -c 'pto.vexpdif' "$ir" || true)" "$residual" \
      "$(stat -c %s "$object_root/${variant}.fat.o")" \
      >>"$experiment_dir/results/lowering-stats.tsv"
  done <"$experiment_dir/inputs/generated/variants.tsv"
fi

sanitize_identifier() {
  /usr/bin/python3 - "$1" <<'PY'
import re
import sys
value = re.sub(r"[^a-zA-Z0-9_]", "_", sys.argv[1])
if not value or value[0].isdigit():
    value = "e_" + value
print(value.lower())
PY
}
safe_tag=$(sanitize_identifier "$tag")

for width in $widths_list; do
  build_dir="$experiment_dir/work/build/n${width}"
  object_root="$experiment_dir/inputs/generated/n${width}-objects"
  cmake_args=(
    -S "$harness" -B "$build_dir"
    -DPTO_ISA_ROOT="$pto_isa_root"
    -DGENERATED_ROOT="$object_root"
    -DASCENDC_SOURCE="$ascendc_source"
    -DSOFTMAX_INNER="$width"
    -DSOFTMAX_ATTACHED_VF_CALL="$attached_vf_call"
  )
  if [[ -n "$vf_softmax_include" ]]; then
    cmake_args+=("-DVF_SOFTMAX_INCLUDE=$vf_softmax_include")
  fi
  for variant in A B C D DL ACU ACF; do
    lower=$(printf '%s' "$variant" | tr '[:upper:]' '[:lower:]')
    symbol="softmax_dnd_b4_m16_n${width}_${lower}_${safe_tag}"
    cmake_args+=("-D${variant}_KERNEL_NAME=$symbol")
    cmake_args+=("-D${variant}_LAUNCH_NAME=launch_${symbol}")
  done
  cmake "${cmake_args[@]}" >"$experiment_dir/logs/n${width}-configure.log" 2>&1
  cmake --build "$build_dir" -j2 \
    >"$experiment_dir/logs/n${width}-build.log" 2>&1
done

printf 'width\tvariant\tdataset\trun_kind\trun\trc\toutput_sha256\tresult\tcomparison_json\n' \
  >"$experiment_dir/results/correctness.tsv"

run_plain() {
  local width=$1 variant=$2 input_set=$3 run_kind=$4 run=$5
  local fixture="$experiment_dir/inputs/fixture/n${width}/${input_set}"
  local run_dir="$experiment_dir/work/runs/n${width}-${variant}-${input_set}-${run_kind}-${run}"
  local binary="$experiment_dir/work/build/n${width}/${variant}"
  mkdir -p "$run_dir"
  cp "$fixture/x.bin" "$run_dir/x.bin"
  cp "$fixture/y.init.bin" "$run_dir/y.bin"
  set +e
  (cd "$run_dir" && timeout 120 "$binary") \
    >"$experiment_dir/logs/n${width}-${variant}-${input_set}-${run_kind}-${run}.stdout" \
    2>"$experiment_dir/logs/n${width}-${variant}-${input_set}-${run_kind}-${run}.stderr"
  local rc=$?
  set -e
  local output_hash=missing result=FAIL comparison='{}'
  if [[ -f "$run_dir/y.bin" ]]; then
    output_hash=$(sha256sum "$run_dir/y.bin" | awk '{print $1}')
  fi
  if [[ $rc -eq 0 ]]; then
    set +e
    comparison=$(/usr/bin/python3 "$harness/compare_output.py" \
      --actual "$run_dir/y.bin" --golden "$fixture/golden.bin" \
      --dataset "$input_set" --width "$width")
    local compare_rc=$?
    set -e
    if [[ $compare_rc -eq 0 ]]; then
      result=PASS
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$width" "$variant" "$input_set" "$run_kind" "$run" "$rc" \
    "$output_hash" "$result" "$comparison" \
    >>"$experiment_dir/results/correctness.tsv"
  [[ "$result" == PASS ]]
}

for width in $widths_list; do
  for variant in D DL ACF B ACU A C; do
    for input_set in exact-onehot finite-sensitive; do
      run_plain "$width" "$variant" "$input_set" cold 0
      run_plain "$width" "$variant" "$input_set" nonprofile 1
      run_plain "$width" "$variant" "$input_set" nonprofile 2
    done
  done
done

printf 'width\tvariant\trepeat\ttask_time_us\ttask_duration_us\taiv_time_us\taiv_total_cycles\taiv_vec_time_us\taiv_scalar_time_us\taiv_mte2_time_us\taiv_mte3_time_us\toutput_sha256\tprofile\n' \
  >"$experiment_dir/results/samples.tsv"

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
  for variant in D DL ACF B ACU A C; do
    run_plain "$width" "$variant" "$dataset" warmup 0
  done
  for ((repeat = 1; repeat <= profile_repeats; ++repeat)); do
    if ((repeat % 2)); then
      order=(DL ACF D C B ACU A)
    else
      order=(D ACF DL A ACU B C)
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
  --output "$experiment_dir/results/paired-gate-${profile_repeats}.json" \
  >"$experiment_dir/results/paired-gate-${profile_repeats}.stdout"
/usr/bin/python3 "$harness/paired_gate.py" \
  "$experiment_dir/results/samples.tsv" --candidate D --baseline ACF \
  --output "$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.json" \
  >"$experiment_dir/results/d-vs-acf-paired-${profile_repeats}.stdout"
capture_npu_health "$experiment_dir/results/npu-health-after.txt"
