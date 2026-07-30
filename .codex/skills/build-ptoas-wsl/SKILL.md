---
name: build-ptoas-wsl
description: Build PTOAS from source inside WSL using the repository README workflow. Use when Codex is asked to build, configure, install, test, or troubleshoot ptoas/PTOAS in WSL or Ubuntu, including the VPTO LLVM 21 setup, editable scikit-build-core installs, CMake/Ninja incremental builds, namespaced MLIR Python bindings, CLI smoke tests, or Python dialect import validation.
---

# Build PTOAS in WSL

## Overview

Use this skill to build PTOAS in a Linux environment under WSL. Keep the build aligned with the README: use the VPTO-enabled LLVM branch `feature-vpto-llvm21`, build shared LLVM/MLIR libraries with Python bindings, and install PTOAS through the repository's standard editable PEP 517 workflow.

Prefer running Linux build commands through WSL, not Windows PowerShell/CMake. If the user is already in a Windows checkout, either convert the path to `/mnt/c/...` for a quick build or clone/copy into the WSL ext4 filesystem for better performance.

## Preflight

1. Identify the WSL distro and Linux environment:

```powershell
wsl.exe -l -v
wsl.exe -- bash -lc 'uname -a; cat /etc/os-release | head'
```

2. Decide the source location:

- If the user wants to build the current Windows checkout, convert `C:\Users\...\ptoas` to `/mnt/c/Users/.../ptoas` and set `PTO_SOURCE_DIR` to that path.
- For faster and cleaner builds, use a WSL-native workspace such as `$HOME/llvm-workspace` and clone PTOAS there.
- Do not switch branches or overwrite user changes unless the user explicitly asks.

3. Install prerequisites in WSL when missing:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential python3 python3-pip python3-dev
python3 -m pip install --user \
  'scikit-build-core>=0.12.2,<2' 'pybind11<3' 'nanobind>=2.4' numpy
```

Keep `pybind11<3`; LLVM/MLIR Python bindings are not compatible with pybind11 3.x in this workflow.

## Environment Variables

Set the workspace variables before configuring LLVM or PTOAS. Adjust `PTO_SOURCE_DIR` if building an existing checkout.

```bash
export WORKSPACE_DIR=$HOME/llvm-workspace
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS

mkdir -p "$WORKSPACE_DIR"
```

When invoking from PowerShell, wrap the Linux commands with:

```powershell
wsl.exe -- bash -lc '<linux commands here>'
```

For multi-step builds, prefer creating or using a shell script inside WSL rather than stuffing a very long command into one PowerShell string.

## Build LLVM/MLIR

Build LLVM only once per WSL workspace unless the user requests a clean rebuild.

```bash
cd "$WORKSPACE_DIR"
if [ ! -d "$LLVM_SOURCE_DIR/.git" ]; then
  git clone https://github.com/vpto-dev/llvm-project.git "$LLVM_SOURCE_DIR"
fi

cd "$LLVM_SOURCE_DIR"
git fetch --tags
git checkout feature-vpto-llvm21

cmake -G Ninja -S llvm -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_EXECUTABLE="$(which python3)" \
  -DPython_EXECUTABLE="$(which python3)" \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host"

ninja -C "$LLVM_BUILD_DIR"
```

Expected CMake package outputs:

```bash
test -d "$LLVM_BUILD_DIR/lib/cmake/llvm"
test -d "$LLVM_BUILD_DIR/lib/cmake/mlir"
test -d "$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core"
```

## Build PTOAS

If the source is not already present, clone the canonical repository from the README. If the user has a local checkout, reuse it.

```bash
cd "$WORKSPACE_DIR"
if [ ! -d "$PTO_SOURCE_DIR/.git" ]; then
  git clone https://gitcode.com/cann/pto-as.git "$PTO_SOURCE_DIR"
fi

cd "$PTO_SOURCE_DIR"
PYTHON_BIN=python3 \
LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTO_SOURCE_DIR/build" \
  ./quick_install.sh

# Reuse the persistent CMake tree for incremental builds and checks.
ninja -C "$PTO_SOURCE_DIR/build" check-pto
```

Expected outputs:

```bash
test -x "$PTO_SOURCE_DIR/build/tools/ptoas/ptoas"
test -x "$PTO_SOURCE_DIR/build/tools/ptobc/ptobc"
test -n "$(find "$PTO_SOURCE_DIR/build/python/ptoas" \
  -name '_core.cpython-*.so' -print -quit)"
test -f "$PTO_SOURCE_DIR/build/python/ptoas/mlir/dialects/pto.py"
```

## Runtime Environment

`quick_install.sh` performs an editable install into the selected Python environment.
Use the installed `ptoas` console script and import `ptoas.mlir` directly. Do not add
LLVM's top-level `mlir_core` package or the deleted top-level `pto` package to
`PYTHONPATH`.

Only when invoking artifacts directly from the CMake build tree, expose that single
tree explicitly:

```bash
export PTOAS_BUILD_PYTHON_ROOT=$PTO_SOURCE_DIR/build/python
export PYTHONPATH=$PTOAS_BUILD_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_SOURCE_DIR/build/lib:$LD_LIBRARY_PATH
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PTO_SOURCE_DIR/build/tools/ptobc:$PATH
```

For persistent use, append those exports to a WSL-side shell startup file or a project-specific `env.sh`.

## Smoke Tests

Run both CLI and Python checks after the build.

```bash
cd "$PTO_SOURCE_DIR"
ptoas --version

cd "$PTO_SOURCE_DIR/test/samples/MatMul"
python3 ./tmatmulk.py > ./tmatmulk.pto
"$PTO_SOURCE_DIR/build/tools/ptoas/ptoas" ./tmatmulk.pto -o ./tmatmulk.cpp
```

Validate Python dialect loading:

```bash
python3 - <<'PY'
from ptoas.mlir.ir import Context, Module, Location
from ptoas.mlir.dialects import pto

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
    module = Module.create()
    print("PTO Dialect registered successfully!")
PY
```

## Troubleshooting

- `def_property family does not currently support keep_alive`: reinstall a supported
  `pybind11<3`, clear the affected CMake cache if needed, and reconfigure.
- CMake cannot find LLVM or MLIR: confirm
  `LLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm` and
  `MLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir` exist in the
  `feature-vpto-llvm21` build.
- Python cannot import `ptoas.mlir.dialects.pto`: confirm the editable install is
  active, then check `build/python/ptoas/_core.cpython-*.so` and
  `build/python/ptoas/mlir/_mlir_libs/_mlir*.so` in the same build tree. For
  direct build-tree imports, prepend only `build/python` to `PYTHONPATH`.
- Runtime linker errors for LLVM or PTO libraries: re-export
  `LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_SOURCE_DIR/build/lib:$LD_LIBRARY_PATH`.
- Builds are very slow under `/mnt/c`: move the workspace into the WSL Linux filesystem, for example `$HOME/llvm-workspace`.
