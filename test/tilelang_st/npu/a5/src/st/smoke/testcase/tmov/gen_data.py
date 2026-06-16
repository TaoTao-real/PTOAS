# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for tmov ST test cases.

For tmov (Vec-to-Vec data movement):
  - input: source tile data
  - golden: exact copy of source tile (valid_shape region)
"""

import os
import sys
import numpy as np

from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        setup_case_rng(case)

        dtype = case["dtype"]
        shape = case["shape"]
        valid_shape = case["valid_shape"]

        # Generate random input data
        if dtype == np.uint8:
            input_data = np.random.randint(0, 256, size=shape).astype(dtype)
        else:
            input_data = np.random.rand(*shape).astype(dtype)

        # Golden is exact copy of input (valid_shape region)
        golden = np.zeros(shape, dtype=dtype)
        vr, vc = valid_shape
        golden[:vr, :vc] = input_data[:vr, :vc].copy()

        save_case_data(case["name"], {"input": input_data, "golden": golden})
        print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__}")


if __name__ == "__main__":
    main()
