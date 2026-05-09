# 使用 msprof op simulator 采集并导出 Insight trace

本文说明如何把一个 PTOAS 生成的 kernel C++ 文件构建成可运行的 host runner，使用 CANN camodel 通过 `msprof op simulator` 采集，再导出 MindStudio Insight 可读的 `trace.json`、`visualize_data.bin` 和指令 CSV。

示例输入使用仓库根目录下的 `4-stage.cpp`，但命令都按变量组织；换 kernel 时只需要改 `SOURCE_CPP`、`KERNEL_BASE_NAME` 和必要的编译 arch。

## 0. 生成器从哪里来

本文使用的 NPU validation 生成器不是外部工具，而是 PTOAS 仓库自带脚本：

```text
test/npu_validation/scripts/generate_testcase.py
```

相关模板在：

```text
test/npu_validation/templates/main_template.cpp
test/npu_validation/templates/run_sh_template.sh
test/npu_validation/templates/golden_template.py
test/npu_validation/templates/compare_template.py
```

这个生成器会解析 PTOAS 生成的 kernel C++ 源码，生成一个临时 testcase 目录，包括：

```text
CMakeLists.txt
main.cpp
launch.cpp
<testcase>_kernel.cpp
golden.py
run.sh
```

其中 `main.cpp` 是 ACL host runner，负责 `aclInit`、分配 host/device 内存、读取 `.bin` 输入、调用 `Launch...` wrapper、同步 stream，并把输出写回 `.bin`。`CMakeLists.txt` 会构建两个关键产物：

```text
<testcase>_sim
lib<testcase>_kernel.so
```

`<testcase>_sim` 链接 `runtime_camodel`，用于 simulator/camodel；`lib<testcase>_kernel.so` 包含真实 kernel symbol，后面要用 `nm -D` 从这里解析 `--kernel-name`。

## 1. 设置变量

在 WSL 中执行，命令从 PTOAS 仓库根目录启动：

```bash
cd /mnt/c/Users/rdp/Documents/ptoas
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
```

先设置通用变量。换 kernel 时优先改这里：

```bash
PTOAS_ROOT=/mnt/c/Users/rdp/Documents/ptoas
SOURCE_CPP="$PTOAS_ROOT/4-stage.cpp"
KERNEL_BASE_NAME=qwen3_decode_incore_1

PTO_ISA_ROOT="/mnt/c/Users/rdp/Documents/pto isa/pto-isa"
SOC_VERSION=dav_2201
AICORE_ARCH=dav-c220-cube

TESTCASE="${KERNEL_BASE_NAME}_msprof"
RUN_TAG="${KERNEL_BASE_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$HOME/msprof-op-simulator-runs/$RUN_TAG"
CASE_ROOT="$RUN_ROOT/cases"
CASE_DIR="$CASE_ROOT/ptoas/$TESTCASE"
BUILD_DIR="$RUN_ROOT/build"
```

`KERNEL_BASE_NAME` 是源码里的 kernel 函数裸名，例如：

```cpp
__global__ AICORE void qwen3_decode_incore_1(...)
```

如果不知道 kernel 名，可以先查：

```bash
grep -nE '__global__[[:space:]]+AICORE[[:space:]]+void|AICORE[[:space:]]+void' "$SOURCE_CPP"
```

`AICORE_ARCH` 要和源码分支及目标 SoC 对齐。本次 `4-stage.cpp` 只有 `__DAV_CUBE__` 分支，所以使用 `dav-c220-cube`。如果换成 vector kernel，需要按源码里的宏和目标 SoC 改成对应 arch。

## 2. 生成 host runner

运行仓库自带生成器：

```bash
python3 test/npu_validation/scripts/generate_testcase.py \
  --input "$SOURCE_CPP" \
  --testcase "$TESTCASE" \
  --output-root "$CASE_ROOT" \
  --run-mode sim \
  --soc-version "$SOC_VERSION" \
  --aicore-arch "$AICORE_ARCH"
```

检查生成结果：

```bash
find "$CASE_DIR" -maxdepth 1 -type f | sort
```

正常应看到：

```text
CMakeLists.txt
main.cpp
launch.cpp
<testcase>_kernel.cpp
golden.py
run.sh
```

建议把 `RUN_ROOT` 放在 Linux 本地目录，比如 `$HOME/msprof-op-simulator-runs/...`，不要放在 `/mnt/c/...`。后续执行二进制和 camodel 时，本地目录更稳定也更快。

## 3. 编译 runner 和 kernel so

```bash
cmake -G Ninja -S "$CASE_DIR" -B "$BUILD_DIR" \
  -DSOC_VERSION="$SOC_VERSION" \
  -DPTO_ISA_ROOT="$PTO_ISA_ROOT"

cmake --build "$BUILD_DIR" --target "${TESTCASE}_sim" -v
```

编译完成后设置两个路径：

```bash
APPLICATION="$BUILD_DIR/${TESTCASE}_sim"
KERNEL_LIB="$BUILD_DIR/lib${TESTCASE}_kernel.so"
```

检查它们是否存在：

```bash
test -x "$APPLICATION" || { echo "missing APPLICATION: $APPLICATION"; exit 1; }
test -f "$KERNEL_LIB" || { echo "missing KERNEL_LIB: $KERNEL_LIB"; exit 1; }
```

## 4. 解析真实 mangled kernel symbol

`msprof op simulator --kernel-name` 要用共享库导出的真实 mangled symbol，不能只写 kernel 裸名。

先列出候选：

```bash
nm -D "$KERNEL_LIB" | awk '/ [TW] / {print $3}' | while read -r candidate; do
  demangled="$(c++filt "$candidate" 2>/dev/null || true)"
  if [[ "$demangled" == "$KERNEL_BASE_NAME("* ]]; then
    printf '%s  # %s\n' "$candidate" "$demangled"
  fi
done
```

如果只有一个候选，可以自动取第一个：

```bash
KERNEL_SYMBOL="$(
  nm -D "$KERNEL_LIB" | awk '/ [TW] / {print $3}' | while read -r candidate; do
    demangled="$(c++filt "$candidate" 2>/dev/null || true)"
    if [[ "$demangled" == "$KERNEL_BASE_NAME("* ]]; then
      printf '%s\n' "$candidate"
    fi
  done | head -n 1
)"

test -n "$KERNEL_SYMBOL" || { echo "failed to resolve kernel symbol"; exit 1; }
echo "KERNEL_SYMBOL=$KERNEL_SYMBOL"
```

如果多个候选 demangle 后都匹配同一个裸名，手动选择签名符合目标 kernel 的那一个：

```bash
KERNEL_SYMBOL=<exact_mangled_symbol>
```

## 5. 准备输入文件

生成的 runner 会在当前工作目录读写 `.bin` 文件，所以采集前要进入 `CASE_DIR`：

```bash
cd "$CASE_DIR"
```

推荐先运行生成器产出的 `golden.py`：

```bash
python3 golden.py
```

如果 WSL Python 没有 `numpy`，而输入可以是全零，可以按 `main.cpp` 里的大小手动创建零文件。先查看 runner 需要哪些输入和大小：

```bash
grep -nE 'elemCount_|fileSize_|ReadFile|WriteFile' "$CASE_DIR/main.cpp"
```

例如本次 `4-stage.cpp` 生成的 runner 需要：

```text
v1: 131584 * sizeof(float)    = 526336 bytes
v2: 131072 * sizeof(uint16_t) = 262144 bytes
v3: 67101184 * sizeof(uint16_t) = 134202368 bytes
```

因此可以创建：

```bash
truncate -s 526336 v1.bin
truncate -s 262144 v2.bin
truncate -s 134202368 v3.bin
```

换 kernel 时不要照抄这三个大小，要以当前 `main.cpp` 为准。

## 6. 运行 msprof op simulator 采集

仍然从 `CASE_DIR` 运行，因为 runner 使用相对路径读写 `.bin` 文件：

```bash
COLLECT_DIR="$RUN_ROOT/msprof_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COLLECT_DIR"

SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/$SOC_VERSION/lib"
export LD_LIBRARY_PATH="$BUILD_DIR:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib:$SIM_LIB_DIR:${LD_LIBRARY_PATH:-}"

cd "$CASE_DIR"

msprof op simulator \
  --application="$APPLICATION" \
  --kernel-name="$KERNEL_SYMBOL" \
  --launch-count=1 \
  --soc-version="$SOC_VERSION" \
  --timeout=120 \
  --output="$COLLECT_DIR/out" \
  2>&1 | tee "$COLLECT_DIR/msprof_collect.log"
```

成功时日志里会出现类似内容：

```text
Profiling running finished. All task success.
Core operator results run in simulator as follow:
core0.cubecore0     ...
Profiling results saved in .../out/OPPROF_...
```

## 7. 导出 Insight trace

不同 CANN 版本的 collect 输出布局可能不同。先定位 `OPPROF_*`：

```bash
OPPROF_DIR="$(find "$COLLECT_DIR/out" -maxdepth 1 -mindepth 1 -type d -name 'OPPROF_*' | sort | tail -n 1)"
test -n "$OPPROF_DIR" || { echo "missing OPPROF dir"; exit 1; }
```

再选择 export 源目录：

```bash
if [[ -d "$OPPROF_DIR/device0/tmp_dump" ]]; then
  EXPORT_SRC="$OPPROF_DIR/device0/tmp_dump"
elif [[ -d "$OPPROF_DIR/dump" ]]; then
  EXPORT_SRC="$OPPROF_DIR/dump"
else
  echo "cannot find tmp_dump or dump under $OPPROF_DIR" >&2
  find "$OPPROF_DIR" -maxdepth 4 -type d | sort >&2
  exit 1
fi

echo "EXPORT_SRC=$EXPORT_SRC"
```

如果使用 `device0/tmp_dump`，并且 export 提示缺少 `pc_start_addr.txt`，可以从同一个 `device0` 下的 kernel dump 目录复制：

```bash
DEVICE0_DIR="$(dirname "$EXPORT_SRC")"
PC_START_FILE="$(find "$DEVICE0_DIR" -path '*/dump/pc_start_addr.txt' | sort | head -n 1 || true)"
if [[ -n "$PC_START_FILE" && -f "$PC_START_FILE" ]]; then
  cp -f "$PC_START_FILE" "$EXPORT_SRC/pc_start_addr.txt"
fi
```

然后执行 export：

```bash
EXPORT_ROOT="$RUN_ROOT/insight_export_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPORT_ROOT"

SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/$SOC_VERSION/lib"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$ASCEND_HOME_PATH/x86_64-linux/devlib:$SIM_LIB_DIR:${LD_LIBRARY_PATH:-}"

msprof op simulator \
  --export="$EXPORT_SRC" \
  --output="$EXPORT_ROOT" \
  2>&1 | tee "$EXPORT_ROOT/msprof_export.log"
```

注意保留目标 SoC 的 simulator lib 目录在 `LD_LIBRARY_PATH` 中。否则 export 仍可能成功，但会警告找不到 `libruntime_camodel.so`，并回退到默认 SoC。

## 8. 检查结果

```bash
find "$EXPORT_ROOT" \
  \( -path '*/simulator/trace.json' \
     -o -path '*/simulator/visualize_data.bin' \
     -o -path '*/simulator/core*.*/trace.json' \
     -o -name '*instr_exe*.csv' \) \
  -printf '%p %s bytes\n' \
  | sort
```

通常会得到：

```text
.../simulator/trace.json
.../simulator/visualize_data.bin
.../simulator/core0.cubecore0/trace.json
.../simulator/core0.cubecore0/*_instr_exe_*.csv
```

本次 `4-stage.cpp` 示例导出了：

```text
trace.json: 1409888 bytes
visualize_data.bin: 1528044 bytes
core0.cubecore0_instr_exe_20260427121022665.csv: 36458 bytes
```

## 常见问题

`ModuleNotFoundError: No module named 'numpy'`

`golden.py` 依赖 `numpy`。如果当前 kernel 的输入可以是全零，可以用 `truncate` 按 `main.cpp` 里的大小创建 `.bin` 文件；如果需要真实随机输入，请先安装或切换到带 `numpy` 的 Python。

`--kernel-name` 过滤不到 kernel

用 `nm -D "$KERNEL_LIB"` 加 `c++filt` 查 mangled symbol。裸名通常不可靠。

`--export` 找不到可解析 dump

先看 collect 输出布局：

```bash
find "$COLLECT_DIR/out" -maxdepth 4 -type d | sort
```

如果有 `device0/tmp_dump`，导出它；如果是 `OPPROF_.../dump` 布局，导出 `dump`。

缺少 `debug_line` 的 warning

日志可能出现：

```text
Kernel missed debug_line information. If you need code call stack, please recompile kernel with -g option
```

这不影响生成 `trace.json`、`visualize_data.bin` 或指令 CSV，只是 Insight 里没有源码级 call stack。如需源码行信息，给 kernel 编译选项补 `-g` 后重新采集。

## 使用 MindStudio Insight 可视化

`msprof op simulator --export` 生成的 `trace.json`、`visualize_data.bin` 和 core 级别 `trace.json` 是给 MindStudio Insight 使用的可视化数据，不建议只靠文本编辑器查看。需要先安装 MindStudio Insight 应用，然后在 Insight 中导入导出目录下的 simulator 数据进行时间线、指令执行、pipeline 等视图分析。

通常选择包含这些文件的目录：

```text
<EXPORT_ROOT>/OPPROF_.../simulator/
```

其中顶层 `simulator/trace.json` 和 `simulator/visualize_data.bin` 用于整体可视化，`simulator/core*.*/trace.json` 与 `*_instr_exe_*.csv` 可用于查看单 core 的执行细节。
