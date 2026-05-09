#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Collect only *.so actually needed by ptoas (transitive closure under /llvm-workspace).
# Expects: LLVM_BUILD_DIR, PTO_INSTALL_DIR, PTOAS_DEPS_DIR, PTO_SOURCE_DIR

set -euo pipefail

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"

remove_rpath() {
  local path="$1"
  if ! has_rpath "$path"; then
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path"
  fi
  if has_rpath "$path" && command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path"
  fi
  if has_rpath "$path"; then
    echo "Error: failed to scrub RPATH/RUNPATH from ${path}" >&2
    exit 1
  fi
}

strip_symbols() {
  local path="$1"
  strip --strip-unneeded "$path"
}

has_rpath() {
  local path="$1"
  if command -v patchelf >/dev/null 2>&1; then
    local rpath_value
    rpath_value="$(patchelf --print-rpath "$path" 2>/dev/null || true)"
    [[ -n "$rpath_value" ]]
    return
  fi
  readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'
}

assert_relro() {
  local path="$1"
  if ! readelf -l "$path" 2>/dev/null | grep -q 'GNU_RELRO'; then
    echo "WARN: RELRO segment missing in ${path}" >&2
    return
  fi
  if ! readelf -d "$path" 2>/dev/null | grep -Eq '(BIND_NOW|FLAGS.*NOW|FLAGS_1.*NOW)'; then
    echo "WARN: NOW binding missing in ${path}" >&2
  fi
}

assert_no_symtab() {
  local path="$1"
  if readelf -S "$path" 2>/dev/null | grep -Eq '[[:space:]]\\.symtab[[:space:]]'; then
    echo "Error: symbol table still present in ${path}" >&2
    exit 1
  fi
}

assert_no_rpath() {
  local path="$1"
  if has_rpath "$path"; then
    echo "Error: runtime search path still present in ${path}" >&2
    exit 1
  fi
}

harden_elf() {
  local path="$1"
  remove_rpath "$path"
  strip_symbols "$path"
  assert_relro "$path"
  assert_no_symtab "$path"
  assert_no_rpath "$path"
}

copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${PTOAS_DEPS_DIR}/${name}" ]] && return 0
  cp -L -n "$f" "${PTOAS_DEPS_DIR}/" 2>/dev/null || true
  harden_elf "${PTOAS_DEPS_DIR}/${name}"
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$f" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
}

mkdir -p "$PTOAS_DEPS_DIR"
while read -r res; do
  copy_so "$res"
done < <(ldd "$PTOAS_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')

while read -r packaged; do
  harden_elf "$packaged"
done < <(find "$PTOAS_DEPS_DIR" -type f | sort)
