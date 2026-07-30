# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standard Python console entry point for PTOAS."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    raise SystemExit(f"unable to locate the active ptoas entry point: {candidate}")


def _load_native_module():
    from ptoas import _core

    return _core


def _resolve_runtime_paths(native_module) -> tuple[Path, Path]:
    module_file = getattr(native_module, "__file__", None)
    if not module_file:
        raise SystemExit("ptoas._core does not expose a module file")

    package_root = Path(module_file).resolve().parent
    python_root = package_root.parent
    runtime_root = package_root / "_runtime"
    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    if not tileops_dir.is_dir():
        raise SystemExit(
            "unable to locate packaged PTOAS TileOps resources: expected "
            f"{tileops_dir}"
        )
    return python_root, tileops_dir.resolve()


def _has_cli_option(arguments: Sequence[str], option: str) -> bool:
    option_with_value = f"{option}="
    return any(
        argument == option or argument.startswith(option_with_value)
        for argument in arguments
    )


def launch(user_args: Sequence[str], *, wrapper: Path | None = None) -> int:
    native_module = _load_native_module()
    python_root, tileops_dir = _resolve_runtime_paths(native_module)
    wrapper = wrapper.resolve() if wrapper is not None else _resolve_wrapper_path()

    os.environ["PTOAS_BIN"] = str(wrapper)
    os.environ["PTOAS_PYTHON_EXE"] = sys.executable
    argv = [str(wrapper)]
    if not _has_cli_option(user_args, "--tilelang-path"):
        argv.extend(["--tilelang-path", str(tileops_dir)])
    if not _has_cli_option(user_args, "--tilelang-pkg-path"):
        argv.extend(["--tilelang-pkg-path", str(python_root)])
    if not _has_cli_option(user_args, "--ptodsl-pkg-path"):
        argv.extend(["--ptodsl-pkg-path", str(python_root)])
    argv.extend(user_args)

    return int(native_module.main(argv))


def main() -> int:
    return launch(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
