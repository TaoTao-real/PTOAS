#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=np.float16).astype(np.float32)
    output = np.fromfile("v2.bin", dtype=np.float16).astype(np.float32)
    close = np.isclose(golden, output, atol=2.0e-3, rtol=2.0e-3)
    if golden.shape != output.shape or not np.all(close):
        bad = np.nonzero(~close)[0]
        index = int(bad[0]) if bad.size else -1
        print(
            "[ERROR] tilelib logical-row RMSNorm mismatch "
            f"index={index} expected={golden[index] if index >= 0 else 'n/a'} "
            f"actual={output[index] if index >= 0 else 'n/a'}"
        )
        sys.exit(2)
    max_error = float(np.max(np.abs(golden - output)))
    print(f"[PASS] TileLib logical-row RMSNorm, max_error={max_error}")


if __name__ == "__main__":
    main()
