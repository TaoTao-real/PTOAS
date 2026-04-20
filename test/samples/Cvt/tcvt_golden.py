#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Golden generator for the tcvt sample (f32 -> i16 with tmp tile and sat_mode=ON).

The input range intentionally stays inside int16 bounds so saturation mode is
still exercised in lowering without making the numeric golden depend on
overflow behavior. The expected result is therefore just truncation toward zero.
"""

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs

    generator = rng()
    src = generator.uniform(-2048.0, 2048.0, size=meta.elem_counts[src_name]).astype(np.float32)

    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    golden_i16 = np.trunc(src).astype(np.int16)
    write_golden(meta, {single_output(meta): golden_i16})


if __name__ == "__main__":
    main()
