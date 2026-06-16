#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


CASES = [
    {
        "name": "f32_rows8_seq32",
        "dtype": np.float32,
        "shape": (8, 128),
        "valid_shape": (8, 32),
        "eps": 1e-4,
        "rows": 8,
        "cols": 128,
        "seq": 32,
        "seed": 7,
    },
    {
        "name": "f32_rows24_seq73",
        "dtype": np.float32,
        "shape": (24, 128),
        "valid_shape": (24, 73),
        "eps": 1e-4,
        "rows": 24,
        "cols": 128,
        "seq": 73,
        "seed": 19,
    },
]

_SMOKE_CASE_NAMES = ['f32_rows8_seq32', 'f32_rows24_seq73']
_SMOKE_CASE_NAME_SET = set(_SMOKE_CASE_NAMES)
_missing = [name for name in _SMOKE_CASE_NAMES if name not in {case["name"] for case in CASES}]
if _missing:
    raise RuntimeError("unknown smoke case(s): " + ", ".join(_missing))
CASES = [case for case in CASES if case["name"] in _SMOKE_CASE_NAME_SET]
