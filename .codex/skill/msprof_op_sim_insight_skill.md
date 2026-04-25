---
name: msprof-op-simulator-insight
description: Use when you need to compile a PTOAS-generated kernel source in this repo with a host runner that launches the kernel, run `msprof op simulator` on A3 `dav_2201`, and export MindStudio Insight visualization files such as `trace.json` and `visualize_data.bin`. Supports arbitrary generated kernel source files, resolves mangled kernel symbols for arbitrary kernel entry names, and uses direct WSL commands.
---

# Msprof Op Simulator Insight

Use this skill from WSL and run commands from the repo root. Execute the build, collect, and export commands directly.

## When to use

- You need the full `build -> collect -> export` workflow for `msprof op simulator`.
- You need MindStudio Insight files like `trace.json` and `visualize_data.bin`, not just raw `*.dump`.
- You want the exact exported kernel symbol instead of guessing the mangled name by hand.

## Workflow

1. Run from WSL after `source /usr/local/Ascend/cann/set_env.sh`.
2. Run the commands from the repo root when possible so source paths can stay relative.
3. Prefer a Linux-local run directory under `~/...`; do not run the produced binary from `/mnt/...`.
4. Build a host runner executable that launches your kernel and points at the generated source.
5. Locate the resulting application binary and shared library that contain the kernel symbol.
6. Resolve the real mangled kernel symbol from that shared library.
7. Collect raw data with `msprof op simulator`.
8. Export from `.../device0/tmp_dump` to generate Insight files.

## Important details

- `msprof op simulator` collection does not create `trace.json` by itself. You must run the export step.
- `--export` must target `.../device0/tmp_dump`, not the top-level `out/` directory.
- `msprof op simulator --kernel-name` must match the exact exported mangled symbol. Passing only the bare kernel base name is often not enough and may be filtered.
- If multiple exported symbols demangle to the same base name, choose the one whose full demangled signature matches the kernel entry you intend to profile, or set `KERNEL_SYMBOL` manually.
- `RUN_ROOT` is just the working directory for one profiling session. It is where build artifacts, raw collect output, logs, and exported Insight files are stored.
- `APPLICATION` is the host runner executable you actually built. `msprof` does not generate this path for you.
- `APPLICATION` does not have to be named `msprof_native_a3`. Any executable is fine as long as it initializes runtime, launches the target kernel, and can run under the simulator.
- Large generated kernels can take a long time on `dav_2201`; use a generous `--timeout-minutes`.
- Exported files usually include:
  - `simulator/trace.json`
  - `simulator/visualize_data.bin`
  - `simulator/core*.*/trace.json`
  - `simulator/core*.*/core*_instr_exe_*.csv`
- If export warns that `tmp_dump` lacks `pc_start_addr.txt`, copy it from `.../device0/<kernel>/0/dump/pc_start_addr.txt` into `tmp_dump` first.

## Full Run Template

Run this from the repo root and adjust `SOURCE_CPP`, `KERNEL_BASE_NAME`, `RUN_ROOT`, and the runner build variables as needed:

```bash
cd "$(git rev-parse --show-toplevel)"
source /usr/local/Ascend/cann/set_env.sh

SOURCE_CPP=./your-kernel.cpp
KERNEL_BASE_NAME=your_kernel_base_name
SOC_VERSION=dav_2201
TIMEOUT_MINUTES=120
RUN_TAG="${KERNEL_BASE_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT=~/msprof-op-simulator-runs/"$RUN_TAG"
RUNNER_CMAKE_DIR=./path-to-runner-project
RUNNER_TARGET=your_runner_target
APPLICATION_RELATIVE_PATH="bin/$RUNNER_TARGET"
KERNEL_LIB_RELATIVE_PATH="lib/your_kernel_library.so"

BUILD_DIR="$RUN_ROOT/build"
COLLECT_DIR="$RUN_ROOT/msprof_run_$(date +%Y%m%d_%H%M%S)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0-beta.1}"
SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/$SOC_VERSION/lib"
BASE_LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib:$SIM_LIB_DIR:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_ROOT"

cmake -G Ninja \
  -S "$RUNNER_CMAKE_DIR" \
  -B "$BUILD_DIR" \
  -DNATIVE_CPP="$(realpath "$SOURCE_CPP")"

cmake --build "$BUILD_DIR" --target "$RUNNER_TARGET" -v

APPLICATION="$BUILD_DIR/$APPLICATION_RELATIVE_PATH"
KERNEL_LIB="$BUILD_DIR/$KERNEL_LIB_RELATIVE_PATH"

if [[ ! -x "$APPLICATION" ]]; then
  echo "Missing application binary: $APPLICATION" >&2
  exit 1
fi

if [[ ! -f "$KERNEL_LIB" ]]; then
  echo "Missing kernel library: $KERNEL_LIB" >&2
  exit 1
fi

KERNEL_SYMBOL="$(
  nm -D "$KERNEL_LIB" | awk '/ [TW] / {print $3}' | while read -r candidate; do
    demangled="$(c++filt "$candidate" 2>/dev/null || true)"
    if [[ "$demangled" == "$KERNEL_BASE_NAME("* ]]; then
      printf '%s\n' "$candidate"
    fi
  done | head -n 1
)"

if [[ -z "$KERNEL_SYMBOL" ]]; then
  echo "Failed to resolve mangled kernel symbol for $KERNEL_BASE_NAME" >&2
  exit 1
fi

echo "Resolved kernel symbol: $KERNEL_SYMBOL"

export LD_LIBRARY_PATH="$(dirname "$KERNEL_LIB"):$BASE_LD_LIBRARY_PATH"

msprof op simulator \
  --application="$APPLICATION" \
  --kernel-name="$KERNEL_SYMBOL" \
  --launch-count=1 \
  --soc-version="$SOC_VERSION" \
  --timeout="$TIMEOUT_MINUTES" \
  --output="$COLLECT_DIR/out" \
  2>&1 | tee "$COLLECT_DIR/msprof_collect.log"
```

This creates a directory layout like:

```text
~/msprof-op-simulator-runs/<run-tag>/
  build/
    <application-relative-path>
    <kernel-lib-relative-path>
  msprof_run_<timestamp>/
    msprof_collect.log
    out/
```

The important alignment is:

- `RUNNER_TARGET` is the build target you ask CMake to compile.
- `APPLICATION_RELATIVE_PATH` is where that target's executable lands under `BUILD_DIR`.
- `KERNEL_LIB_RELATIVE_PATH` is the shared library under `BUILD_DIR` that exports the kernel symbol.

If your runner is not built by CMake, replace the `cmake ...` lines with your actual build command, then set:

```bash
APPLICATION=/absolute/or/build-relative/path/to/your_runner
KERNEL_LIB=/absolute/or/build-relative/path/to/your_kernel_library.so
```

before running `nm` and `msprof`.

If the first-match resolution is too loose for your kernel, inspect all candidates first:

```bash
nm -D "$KERNEL_LIB" | awk '/ [TW] / {print $3}' | while read -r candidate; do
  demangled="$(c++filt "$candidate" 2>/dev/null || true)"
  if [[ "$demangled" == "$KERNEL_BASE_NAME("* ]]; then
    printf '%s  # %s\n' "$candidate" "$demangled"
  fi
done
```

Then set:

```bash
KERNEL_SYMBOL=<exact_mangled_symbol>
```

and use that symbol directly in the collect command.

## Export Existing Collect Data

Use this when you already have raw `msprof op simulator` output and only need Insight files:

```bash
cd "$(git rev-parse --show-toplevel)"
source /usr/local/Ascend/cann/set_env.sh

RUN_ROOT=~/msprof-op-simulator-runs/<run-tag>
COLLECT_ROOT="$RUN_ROOT/msprof_run_<timestamp>/out"
EXPORT_ROOT="$RUN_ROOT/insight_export_$(date +%Y%m%d_%H%M%S)"

if [[ -d "$COLLECT_ROOT/device0/tmp_dump" ]]; then
  TMP_DUMP_DIR="$COLLECT_ROOT/device0/tmp_dump"
else
  OPPROF_DIR="$(find "$COLLECT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'OPPROF_*' | sort | tail -n 1)"
  TMP_DUMP_DIR="$OPPROF_DIR/device0/tmp_dump"
fi

DEVICE0_DIR="$(dirname "$TMP_DUMP_DIR")"
PC_START_FILE="$(find "$DEVICE0_DIR" -path '*/dump/pc_start_addr.txt' | sort | head -n 1 || true)"
if [[ -n "$PC_START_FILE" && -f "$PC_START_FILE" ]]; then
  cp -f "$PC_START_FILE" "$TMP_DUMP_DIR/pc_start_addr.txt"
fi

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0-beta.1}"
SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_2201/lib"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib:$SIM_LIB_DIR:${LD_LIBRARY_PATH:-}"

mkdir -p "$EXPORT_ROOT"

msprof op simulator \
  --export="$TMP_DUMP_DIR" \
  --output="$EXPORT_ROOT" \
  2>&1 | tee "$EXPORT_ROOT/msprof_export.log"

find "$EXPORT_ROOT" \
  \( -path '*/simulator/trace.json' -o -path '*/simulator/visualize_data.bin' \) \
  | sort
```

## Quick Checks

- Confirm collect produced an `OPPROF_*` directory under `out/`.
- Confirm export produced:
  - `simulator/trace.json`
  - `simulator/visualize_data.bin`
- If export fails with `Failed to get any available dump file to parse`, the `--export` path is wrong. Point it to `tmp_dump`.
- If export warns about missing `debug_line`, that only means there is no source-level call stack. The trace can still be valid.
