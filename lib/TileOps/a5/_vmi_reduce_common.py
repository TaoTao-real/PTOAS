# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared canonical VMI decorator arguments for the A5 row-reduce candidates.

``trowsum`` and ``trowmax`` register the same three row-reduce candidate
shapes (tile-grouped, row-streaming, and the Sinkhorn row-streaming variant);
only the op name and the reduce ``kind`` differ.  The keyword groups below are
the single source of those shared ``canonical_vmi_template`` arguments.
"""

from ._vmi_common import (
    row_reduce_streaming_vmi_constraint,
    row_reduce_vmi_constraint,
    sinkhorn_row_reduce_streaming_vmi_constraint,
)

# Row-reduce is ODS-validated only for {f32, i32} on A5 (trowmax/trowsum
# reject i8/i16/f16/bf16/ui*); other dtypes fall back to the ordinary PTODSL
# path (see _vmi_common._ROW_REDUCE_DTYPES).
ROW_REDUCE_DTYPE_SIGNATURES = (
    ("f32", "f32", "f32"),
    ("i32", "i32", "i32"),
)

# Tile-grouped candidate: one grouped vector reduce over the whole tile.
ROW_REDUCE_TILE_KWARGS = dict(
    requires_full_physical_row=False,
    dtypes=ROW_REDUCE_DTYPE_SIGNATURES,
    constraints=(row_reduce_vmi_constraint,),
    tags=("grouped_rows", "supports_partial_valid_shape"),
    priority=101,
    single_logical_row_loop=False,
    resource_scope="tile",
    resource_vector_values=1,
)

# Row-streaming candidate: one logical row per iteration.
ROW_REDUCE_ROW_KWARGS = dict(
    requires_full_physical_row=False,
    dtypes=ROW_REDUCE_DTYPE_SIGNATURES,
    constraints=(row_reduce_streaming_vmi_constraint,),
    tags=("row_streaming",),
    candidate_id=1001,
    resource_scope="row",
    resource_vector_values=1,
)

# Sinkhorn row-streaming candidate: the statically bounded 8x8 forms.
ROW_REDUCE_SINKHORN_ROW_KWARGS = dict(
    requires_full_physical_row=False,
    dtypes=ROW_REDUCE_DTYPE_SIGNATURES,
    constraints=(sinkhorn_row_reduce_streaming_vmi_constraint,),
    tags=("row_streaming", "supports_partial_valid_shape"),
    priority=102,
    candidate_id=1002,
    resource_scope="row",
    resource_vector_values=1,
)
