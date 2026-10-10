# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A5 VMI TileLib provider for the stage-1 fusion acceptance pipeline.

The first migration slice intentionally registers only the candidates required
by ``fa-softmax-rowplusone``. Ordinary PTODSL candidates remain registered in
the canonical TileOps registry and are the fallback when these VMI constraints
reject an operand shape, dtype, or layout.
"""

__all__ = [
    "VMI_TILELIB_REGISTRY",
    "canonical_vmi_template",
    "emit_elementwise_vmi",
    "vmi_tcolmax",
    "vmi_tcolexpandsub",
    "vmi_tmuls",
    "vmi_texp_block64",
    "vmi_tcolsum",
    "vmi_tcvt",
    "vmi_tmov",
]

from TileOps.a5._vmi_common import (
    VMI_TILELIB_REGISTRY,
    canonical_vmi_template,
    emit_elementwise_vmi,
)
from TileOps.a5.elementwise.tmuls import vmi_tmuls
from TileOps.a5.expand.tcolexpandsub import vmi_tcolexpandsub
from TileOps.a5.memory.tmov import vmi_tmov
from TileOps.a5.reduce.tcolsum import vmi_tcolsum
from TileOps.a5.tcvt import vmi_tcvt
from TileOps.a5.tcolmax import vmi_tcolmax
from TileOps.a5.unary.texp import vmi_texp_block64
