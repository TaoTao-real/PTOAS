#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main() -> None:
    rows, cols = 8, 128
    values = np.linspace(-1.25, 1.75, rows * cols, dtype=np.float32)
    src = values.reshape(rows, cols).astype(np.float16)
    src_f32 = src.astype(np.float32)
    row_sum = np.sum(src_f32 * src_f32, axis=1, keepdims=True, dtype=np.float32)
    golden = (src_f32 / np.sqrt(row_sum)).astype(np.float16)
    src.tofile("v1.bin")
    np.zeros_like(src).tofile("v2.bin")
    golden.tofile("golden_v2.bin")


if __name__ == "__main__":
    main()
