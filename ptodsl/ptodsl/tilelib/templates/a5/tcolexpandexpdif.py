# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pto.tcolexpandexpdif."""

from ptodsl import pto

from ._expand_binary import register_column_expand_expdif


template_tcolexpandexpdif_f32, template_tcolexpandexpdif_f16 = register_column_expand_expdif()


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    emit_col_expand_binary_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tcolexpandexpdif",
    name="vmi_tcolexpandexpdif",
    dtypes=(("f32", "f32", "f32"),),
)
def vmi_tcolexpandexpdif(
    src: pto.Tile, col_values: pto.Tile, dst: pto.Tile
):
    # The installed Softmax Dn VF visits even and odd M rows together and
    # carries two reduction accumulators.  Preserve that phase header so the
    # exp-difference producer can later fuse with tcolsum without changing the
    # FP32 association.
    emit_col_expand_binary_vmi(
        src, col_values, dst, binop="expdif", paired_rows=True
    )
