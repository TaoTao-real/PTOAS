# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest

from ptoas.mlir import ir
from tilelang_dsl.lowering_backend import LoweringResult


class LoweringBackendConversionTests(unittest.TestCase):
    def test_text_result_parses_with_namespaced_pto_dialect(self):
        module = LoweringResult(text="module {}").as_module()

        self.assertIsInstance(module, ir.Module)


if __name__ == "__main__":
    unittest.main()
