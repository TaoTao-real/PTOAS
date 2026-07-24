# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Initial canonical VMI TileLib candidates for static Softmax-related coverage."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ._surface_types import Tile
from ._ops import get_op_attr
from ._tile_template_tracing import (
    LogicalRowMap,
    Scalar,
    ScalarType,
    _MaskValue,
    _TileProxy,
    _Value,
    _VectorValue,
    bf16,
    f16,
    f32,
    for_,
    i16,
    i32,
    i8,
    tile_template as _trace_tile_template,
    vmi_create_mask,
    vmi_create_mask_lanes,
    vmi_prepare_tile_access,
    vmi_vadd,
    vmi_vadds,
    vmi_vabs,
    vmi_vbroadcast,
    vmi_vbroadcast_scalar,
    vmi_scalar_constant,
    vmi_vcvt,
    vmi_vdiv,
    vmi_vexp,
    vmi_vgather,
    vmi_vload,
    vmi_vload_linear,
    vmi_vmax,
    vmi_vmaxs,
    vmi_vmin,
    vmi_vmins,
    vmi_vmuls,
    vmi_vmul,
    vmi_vneg,
    vmi_vreduce_add,
    vmi_vreduce_max,
    vmi_vsub,
    vmi_vstore,
    vmi_vstore_linear,
    vmi_vsqrt,
)
from .tilelib.registry import TileTemplateRegistry


ElementwiseCompute = Callable[[Sequence[_VectorValue], _MaskValue], _VectorValue]


NUMERIC_DTYPES = (f32, f16, bf16, i32, i16, i8)
FLOAT_DTYPES = (f32, f16)
SUB_DTYPES = (f32, f16, i32, i16, i8)
MUL_DTYPES = (f32, f16, i32, i16)
SUPPORTED_LOGICAL_WIDTHS = frozenset(
    (1, 8, 16, 32, 64, 128, 256, 448, 512, 1024)
)
ROUND_MODE_TO_VMI = {
    "RINT": "R",
    "ROUND": "A",
    "FLOOR": "F",
    "CEIL": "C",
    "TRUNC": "Z",
    "ODD": "O",
}


VMI_TILELIB_REGISTRY = TileTemplateRegistry()


def _validate_logical_width(lanes: int) -> None:
    if lanes not in SUPPORTED_LOGICAL_WIDTHS:
        supported = ", ".join(
            str(width) for width in sorted(SUPPORTED_LOGICAL_WIDTHS)
        )
        raise ValueError(
            f"VMI TileLib logical width {lanes} is not verified; "
            f"supported widths are {supported}"
        )


# Reduce kind -> (merge op, identity element). The identity mirrors pto-isa
# `TColReduceOps.hpp` `InstrOp::InitVal` / a5 `Padding<T>::Min/Max`:
#   max -> vmax, init -inf (Padding<T>::Min)
#   min -> vmin, init +inf (Padding<T>::Max)
#   add -> vadd, init 0
#   prod-> vmul, init 1
# `emit_col_reduce_vmi` only exercises max/add today; min/prod map to vmi
# merge ops that the VMI tilelib does not yet expose as elementwise-vector
# forms (only the -s scalar variants), so they raise if used.
_REDUCE_MERGE_OP = {
    "max": vmi_vmax,
    "add": vmi_vadd,
}


def canonical_vmi_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
    semantic_form: str = "default",
    context_constraints: dict[str, tuple[object, ...]] | None = None,
):
    """Register one canonical VMI implementation in this provider module."""

    def decorator(fn):
        normalized_op = op[4:] if op.startswith("pto.") else op
        descriptor = _trace_tile_template(
            target=target,
            op=normalized_op,
            name=name,
            ir_level="vmi",
            semantic_form=semantic_form,
            context_constraints=context_constraints,
        )(fn)
        VMI_TILELIB_REGISTRY.register(descriptor)
        return descriptor

    return decorator


def emit_elementwise_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
    *,
    logical_lanes: int | None = None,
    allowed_dtypes: Sequence[ScalarType] = NUMERIC_DTYPES,
) -> None:
    """Emit one flat logical-block loop for a standalone elementwise candidate."""

    if not sources:
        raise ValueError("emit_elementwise_vmi requires at least one source tile")
    if logical_lanes is None:
        logical_lanes = dst._spec.shape[1]
    _validate_elementwise_tiles(
        dst,
        sources,
        logical_lanes=logical_lanes,
        allowed_dtypes=allowed_dtypes,
    )
    block_map = LogicalRowMap.from_tile(dst, logical_lanes=logical_lanes)

    vmi_prepare_tile_access(*sources, dst)
    mask = vmi_create_mask(block_map, dst.element_type)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        values = tuple(vmi_vload(source, coordinate) for source in sources)
        result = compute(values, mask)
        vmi_vstore(result, dst, coordinate, mask)


def _validate_elementwise_tiles(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    *,
    logical_lanes: int,
    allowed_dtypes: Sequence[ScalarType],
) -> None:
    if not isinstance(dst, _TileProxy):
        raise TypeError("elementwise VMI candidate destination must be a traced Tile")
    if dst.element_type not in allowed_dtypes:
        raise ValueError(
            "VMI elementwise candidate does not support dtype "
            f"{dst.element_type}; supported dtypes are "
            f"{', '.join(dtype.name for dtype in allowed_dtypes)}"
        )
    if logical_lanes != dst._spec.shape[1]:
        raise ValueError(
            "VMI elementwise candidates require one logical vector per row; "
            f"got logical_lanes={logical_lanes}, inner={dst._spec.shape[1]}"
        )
    _validate_logical_width(logical_lanes)
    if dst._spec.b_layout != "row_major":
        raise ValueError("VMI elementwise candidates require row-major tiles")
    for source in sources:
        if not isinstance(source, _TileProxy):
            raise TypeError("elementwise VMI candidate sources must be traced Tiles")
        if source._spec.shape != dst._spec.shape:
            raise ValueError(
                "elementwise VMI candidate source and destination shapes must match; "
                f"got {source._spec.shape} and {dst._spec.shape}"
            )
        if source.element_type != dst.element_type:
            raise ValueError(
                "elementwise VMI candidate source and destination dtypes must match; "
                f"got {source.element_type} and {dst.element_type}"
            )
        if source._spec.b_layout != dst._spec.b_layout:
            raise ValueError("elementwise VMI candidate layouts must match")


def _add(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tadd VMI candidate expects two source vectors")
    return vmi_vadd(values[0], values[1], mask)


def _exp(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("texp VMI candidate expects one source vector")
    return vmi_vexp(values[0], mask)


def _sub(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tsub VMI candidate expects two source vectors")
    return vmi_vsub(values[0], values[1], mask)


def _mul(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmul VMI candidate expects two source vectors")
    return vmi_vmul(values[0], values[1], mask)


def _max(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmax VMI candidate expects two source vectors")
    return vmi_vmax(values[0], values[1], mask)


def _min(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmin VMI candidate expects two source vectors")
    return vmi_vmin(values[0], values[1], mask)


def _move(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tmov VMI candidate expects one source vector")
    return values[0]


def _abs(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tabs VMI candidate expects one source vector")
    return vmi_vabs(values[0], mask)


def _neg(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tneg VMI candidate expects one source vector")
    return vmi_vneg(values[0], mask)


def _divide_by_scalar(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = vmi_vbroadcast_scalar(scalar, like=value)
    return vmi_vdiv(value, scalar_vector, mask)


def _divide_scalar_by_vector(
    scalar: _Value, value: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = vmi_vbroadcast_scalar(scalar, like=value)
    return vmi_vdiv(scalar_vector, value, mask)


def _subtract_scalar(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = vmi_vbroadcast_scalar(scalar, like=value)
    return vmi_vsub(value, scalar_vector, mask)


def emit_scalar_fill_vmi(scalar: _Value, dst: _TileProxy) -> None:
    logical_lanes = dst._spec.shape[1]
    _validate_elementwise_tiles(
        dst,
        (),
        logical_lanes=logical_lanes,
        allowed_dtypes=NUMERIC_DTYPES,
    )
    block_map = LogicalRowMap.from_tile(dst, logical_lanes=logical_lanes)
    vmi_prepare_tile_access(dst)
    mask = vmi_create_mask(block_map, dst.element_type)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        value = vmi_vbroadcast_scalar(
            scalar, dtype=dst.element_type, lanes=logical_lanes
        )
        vmi_vstore(value, dst, coordinate, mask)


def _validate_row_reduce_tiles(
    src: _TileProxy, workspace: _TileProxy, dst: _TileProxy
) -> LogicalRowMap:
    if (
        src.element_type != f32
        or workspace.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("row-reduce VMI candidates currently support only f32")
    if (
        src._spec.b_layout != "row_major"
        or workspace._spec.b_layout != "row_major"
    ):
        raise ValueError("row-reduce source and workspace must be row-major")
    if len(workspace._spec.shape) != 2:
        raise ValueError("row-reduce workspace must be a rank-2 tile")
    workspace_capacity = workspace._spec.shape[0] * workspace._spec.shape[1]
    source_capacity = src._spec.shape[0] * src._spec.shape[1]
    if workspace_capacity < source_capacity:
        raise ValueError(
            "row-reduce workspace capacity must be at least the source capacity"
        )
    rows, cols = src._spec.shape
    if dst._spec.shape != (rows, 1) or dst._spec.b_layout != "col_major":
        raise ValueError("row-reduce destination must be a col-major [rows, 1] tile")
    _validate_logical_width(cols)
    return LogicalRowMap.from_tile(src, logical_lanes=cols)


def emit_row_reduce_vmi(
    src: _TileProxy,
    workspace: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    block_map = _validate_row_reduce_tiles(src, workspace, dst)
    reduce_op = vmi_vreduce_max if kind == "max" else vmi_vreduce_add
    vmi_prepare_tile_access(src, dst)
    full_mask = vmi_create_mask(block_map, f32)
    scalar_mask = vmi_create_mask_lanes(1, 1, f32)
    with for_(0, block_map.rows, step=1) as row:
        coordinate = block_map.coordinate(row)
        accumulator = reduce_op(vmi_vload(src, coordinate), full_mask)
        vmi_vstore_linear(accumulator, dst, row, scalar_mask)


def emit_row_expand_binary_vmi(
    src: _TileProxy,
    row_values: _TileProxy,
    dst: _TileProxy,
    *,
    binop: str,
) -> None:
    if (
        src.element_type not in FLOAT_DTYPES
        or row_values.element_type != src.element_type
        or dst.element_type != src.element_type
    ):
        raise ValueError("row-expand-binary VMI candidates support matching f16/f32 tiles")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("row-expand-binary source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("row-expand-binary source and destination must be row-major")
    rows, cols = src._spec.shape
    if (
        row_values._spec.shape != (rows, 1)
        or row_values._spec.b_layout != "col_major"
    ):
        raise ValueError(
            "row-expand-binary row values must be a col-major [rows, 1] tile"
        )
    operations = {
        "sub": vmi_vsub,
        "mul": vmi_vmul,
        "div": vmi_vdiv,
    }
    op_fn = operations.get(binop)
    if op_fn is None:
        raise ValueError(
            f"unsupported row-expand-binary operation {binop!r}; "
            f"expected one of {sorted(operations)}"
        )
    _validate_logical_width(cols)
    block_map = LogicalRowMap.from_tile(src, logical_lanes=cols)

    vmi_prepare_tile_access(src, row_values, dst)
    full_mask = vmi_create_mask(block_map, src.element_type)
    with for_(0, rows, step=1) as row:
        row_scalar = vmi_vload_linear(row_values, row, lanes=1)
        broadcast = vmi_vbroadcast(row_scalar, lanes=cols)
        coordinate = block_map.coordinate(row)
        value = vmi_vload(src, coordinate)
        result = op_fn(value, broadcast, full_mask)
        vmi_vstore(result, dst, coordinate, full_mask)


def emit_row_expand_sub_vmi(
    src: _TileProxy, row_values: _TileProxy, dst: _TileProxy
) -> None:
    emit_row_expand_binary_vmi(src, row_values, dst, binop="sub")


def _validate_optional_tmp(
    src: _TileProxy,
    tmp: _TileProxy | None,
    *,
    operation: str,
) -> None:
    if tmp is None:
        return
    if tmp.element_type != src.element_type:
        raise ValueError(f"{operation} tmp must match the source dtype")
    element_bytes = src.element_type.bytewidth
    if tmp._spec.shape[0] * tmp._spec.shape[1] * element_bytes < 32:
        raise ValueError(f"{operation} tmp must provide at least 32 bytes")


def emit_recip_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    def reciprocal(values, mask):
        one = vmi_vbroadcast_scalar(
            vmi_scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        return vmi_vdiv(one, values[0], mask)

    emit_elementwise_vmi(
        dst, (src,), reciprocal, allowed_dtypes=FLOAT_DTYPES
    )


def emit_sqrt_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vsqrt(values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


def emit_rsqrt_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    tmp: _TileProxy | None = None,
) -> None:
    _validate_optional_tmp(src, tmp, operation="trsqrt")

    def reciprocal_sqrt(values, mask):
        root = vmi_vsqrt(values[0], mask)
        one = vmi_vbroadcast_scalar(
            vmi_scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        return vmi_vdiv(one, root, mask)

    emit_elementwise_vmi(
        dst, (src,), reciprocal_sqrt, allowed_dtypes=FLOAT_DTYPES
    )


def emit_gather_index_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    indices: _TileProxy,
    tmp: _TileProxy,
) -> None:
    if src.element_type != f32 or dst.element_type != f32:
        raise ValueError("tgather:index currently requires f32 source and destination")
    if indices.element_type != i32:
        raise ValueError("tgather:index currently requires i32 indices")
    if dst._spec.shape != indices._spec.shape:
        raise ValueError("tgather:index destination and indices shapes must match")
    rows, cols = dst._spec.shape
    if cols != 64:
        raise ValueError("tgather:index currently requires exactly 64 logical lanes")
    if any(
        tile._spec.b_layout != "row_major"
        for tile in (src, dst, indices, tmp)
    ):
        raise ValueError("tgather:index requires row-major tiles")

    block_map = LogicalRowMap.from_tile(dst, logical_lanes=cols)
    vmi_prepare_tile_access(src, dst, indices)
    full_mask = vmi_create_mask(block_map, f32)
    with for_(0, rows, step=1) as row:
        coordinate = block_map.coordinate(row)
        index_values = vmi_vload(indices, coordinate)
        gathered = vmi_vgather(src, index_values, full_mask)
        vmi_vstore(gathered, dst, coordinate, full_mask)


def _validate_col_reduce_tiles(
    src: _TileProxy, dst: _TileProxy
) -> LogicalRowMap:
    """Validate tiles for a ColReduce (tcolmax / tcolsum) VMI candidate.

    Mirror of `_validate_row_reduce_tiles` but the surviving axis is the column
    dimension: src is [rows, cols] row-major, dst is [1, cols] row-major, and the
    reduction runs across all rows. First slice only supports a single VL block
    wide tile (cols == VL), matching the pto-isa `TColReduceInstr_NoPostUpdate`
    one-repeat layout.
    """
    if src.element_type != f32 or dst.element_type != f32:
        raise ValueError("col-reduce VMI candidates currently support only f32")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-reduce source and destination must be row-major")
    rows, cols = src._spec.shape
    if dst._spec.shape != (1, cols):
        raise ValueError("col-reduce destination must be a row-major [1, cols] tile")
    _validate_logical_width(cols)
    return LogicalRowMap.from_tile(src, logical_lanes=cols)


def emit_col_reduce_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    """Emit a ColReduce (tcolmax / tcolsum / tcolmin / ...) VMI candidate.

    Mirrors pto-isa `TColReduceInstr_NoPostUpdate` with a single VL block:
      acc = vbr(InitVal)                        # VL-wide, reduce-neutral init
      for row in 0..rows: acc = op(acc, load(row))   # runtime scf.for
      store(acc, dst)

    The accumulator stays VL-wide for the whole reduction (the column axis is
    the surviving axis). This intentionally avoids `vmi_vreduce_max`/`vmi_vcmax`,
    which collapse to a 1-lane scalar — wrong for a column-preserving ColMax.

    The init is the reduce's identity element (max->-inf, min->+inf, add->0,
    prod->1), broadcast VL-wide via `vbr` — exactly pto-isa's
    `vbr(dstVReg, InstrOp::InitVal)` (see a5/common.hpp `Padding<T>::Min/Max`).
    The reduce runs from row 0 (not 1): iteration 0 does op(InitVal, load(0))
    which absorbs row 0 through the op (e.g. max(-inf, x) = x, 0 + x = x), so a
    c0..rows header matches the element-wise VMI candidates' c0..N header and
    the downstream loop-fusion pass can merge this reduce with its same-index
    neighbors into one scf.for.

    The cross-row reduction is a runtime ``scf.for`` carrying the VL-wide
    accumulator as loop state (one ``vmi.vmax``/``vmi.vadd`` per iteration),
    matching the pto-isa repeat loop. It must NOT be a Python ``range`` here:
    a trace-time ``range`` would statically unroll one merge per row (e.g. 127
    for ``rows=128``), producing a flat vmax chain with no surrounding loop.
    """
    # Reduce identity element per kind (pto-isa InstrOp::InitVal /
    # a5 Padding<T>::Min/Max): max -> -inf (0xff800000), min -> +inf
    # (0x7f800000), add -> 0.0, prod -> 1.0. Broadcast VL-wide with vbr.
    reduce_identity = {
        "max": float("-inf"),
        "min": float("inf"),
        "add": 0.0,
        "prod": 1.0,
    }[kind]
    block_map = _validate_col_reduce_tiles(src, dst)
    merge_op = _REDUCE_MERGE_OP[kind]

    vmi_prepare_tile_access(src, dst)
    full_mask = vmi_create_mask(block_map, f32)
    # Seed the VL-wide accumulator with the reduce-neutral identity (vbr InitVal,
    # matching pto-isa `TColReduceInstr_NoPostUpdate`), so the loop runs c0..rows
    # and absorbs row 0 via op(InitVal, load(0)) instead of preloading row 0.
    # The broadcast takes the element type/lanes from `f32` directly — no dummy
    # load needed (a vload would carry a Read memory effect and survive DCE,
    # duplicating the row-0 read the loop itself does).
    accumulator = vmi_vbroadcast_scalar(
        vmi_scalar_constant(reduce_identity, f32),
        dtype=f32,
        lanes=block_map.logical_lanes,
    )
    # The whole reduction is a runtime scf.for from row 0 carrying the
    # VL-wide accumulator; each iteration does one element-wise merge (VL
    # stays full). Row r maps to logical block r*blocks_per_row.
    with for_(0, block_map.rows, step=1, state={"acc": accumulator}) as loop:
        loaded = vmi_vload(src, block_map.coordinate(loop.iv))
        merged = merge_op(loop.state.acc, loaded, full_mask)
        loop.yield_state(acc=merged)
    accumulator = loop.results[0]
    # dst [1, cols] is a single VL block; store via linear offset to avoid the
    # src/dst shape mismatch in CanonicalBlockCoordinate validation (src is
    # [rows, cols], dst is [1, cols]).
    vmi_vstore_linear(accumulator, dst, 0, full_mask)


def _validate_col_expand_binary_tiles(
    src: _TileProxy, col_values: _TileProxy, dst: _TileProxy
) -> LogicalRowMap:
    """Validate tiles for a ColExpandBinary (tcolexpandsub/...) VMI candidate.

    src is [rows, cols] row-major, col_values is [1, cols] row-major (one VL
    block of surviving reduce result), dst is [rows, cols] row-major. cols must
    equal VL(f32) so the single broadcast loads exactly one VL block.
    """
    if (
        src.element_type != f32
        or col_values.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("col-expand-binary VMI candidates currently support only f32")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("col-expand-binary source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-expand-binary source and destination must be row-major")
    rows, cols = src._spec.shape
    if (
        col_values._spec.shape != (1, cols)
        or col_values._spec.b_layout != "row_major"
    ):
        raise ValueError(
            "col-expand-binary col_values must be a row-major [1, cols] tile"
        )
    _validate_logical_width(cols)
    return LogicalRowMap.from_tile(src, logical_lanes=cols)


def emit_col_expand_binary_vmi(
    src: _TileProxy,
    col_values: _TileProxy,
    dst: _TileProxy,
    *,
    binop: str,
) -> None:
    """Emit a ColExpandBinary (tcolexpandsub/add/mul/div) VMI candidate.

    Mirrors pto-isa `TColExpandBinOp`: the single VL block of col_values is
    broadcast to every row, then a binary op is applied per row block.
    """
    binop_dispatch = {
        "sub": vmi_vsub,
        "add": vmi_vadd,
        "mul": vmi_vmul,
        "div": vmi_vdiv,
    }
    if binop not in binop_dispatch:
        raise ValueError(
            f"col-expand-binary VMI candidate does not support op {binop!r}; "
            f"expected one of {sorted(binop_dispatch)}"
        )
    op_fn = binop_dispatch[binop]
    block_map = _validate_col_expand_binary_tiles(src, col_values, dst)

    vmi_prepare_tile_access(src, col_values, dst)
    full_mask = vmi_create_mask(block_map, f32)
    # pto-isa TColExpandBinOp broadcasts by reloading the same col_values VL
    # block per row (vlds with fixed offset), NOT a 1-lane vbrc. col_values is
    # [1, cols] (one VL block), so the broadcast load is loop-invariant: hoist
    # it out of the row loop so a later mem2reg (Stage C) can forward the
    # ColMax result directly to the consumer without a per-row reload.
    broadcast = vmi_vload_linear(col_values, 0, lanes=block_map.logical_lanes)
    with for_(0, block_map.rows, step=1) as row:
        coordinate = block_map.coordinate(row)
        value = vmi_vload(src, coordinate)
        result = op_fn(value, broadcast, full_mask)
        vmi_vstore(result, dst, coordinate, full_mask)


def emit_convert_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    round_mode: str,
    sat_mode: str,
) -> None:
    widening_round_modes = set(ROUND_MODE_TO_VMI)
    supported_modes = {
        # The Tile round mode is semantically inert for 16-bit float widening,
        # but real PyPTO emits both RINT and ROUND. Accept every named mode and
        # omit the physical rounding control below.
        (bf16, f32): widening_round_modes,
        (f16, f32): widening_round_modes,
        (f32, bf16): {"RINT"},
        (f32, f16): {"RINT", "ROUND"},
        (f32, i32): {"RINT", "ROUND", "TRUNC"},
        (i32, f32): {"RINT", "ROUND"},
        (i32, f16): {"ROUND"},
        (f16, i8): {"TRUNC"},
    }
    modes = supported_modes.get((src.element_type, dst.element_type))
    if modes is None:
        raise ValueError(
            "tcvt VMI candidate does not support conversion "
            f"{src.element_type}->{dst.element_type}"
        )
    if round_mode not in modes:
        raise ValueError(
            f"tcvt {src.element_type}->{dst.element_type} does not support "
            f"round_mode={round_mode}; expected one of {sorted(modes)}"
        )
    if sat_mode not in {"ON", "OFF"}:
        raise ValueError("tcvt sat_mode must be ON or OFF")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("tcvt source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcvt VMI candidate requires row-major tiles")
    logical_lanes = src._spec.shape[1]
    _validate_logical_width(logical_lanes)
    block_map = LogicalRowMap.from_tile(src, logical_lanes=logical_lanes)

    vmi_prepare_tile_access(src, dst)
    dst_mask = vmi_create_mask_lanes(logical_lanes, logical_lanes, dst.element_type)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        no_sat_control = dst.element_type == f32 and src.element_type in (
            f16,
            bf16,
            i32,
        )
        if no_sat_control and sat_mode != "OFF":
            raise ValueError(
                f"tcvt {src.element_type}->{dst.element_type} requires sat_mode=OFF"
            )
        converted = vmi_vcvt(
            vmi_vload(src, coordinate),
            dst.element_type,
            rounding=(
                None
                if src.element_type in (f16, bf16) and dst.element_type == f32
                else ROUND_MODE_TO_VMI[round_mode]
            ),
            saturate=(
                None
                if no_sat_control
                else ("SAT" if sat_mode == "ON" else "OFF")
            ),
        )
        vmi_vstore(converted, dst, coordinate, dst_mask)


@canonical_vmi_template(
    target="a5",
    op="tadd",
    name="vmi_tadd_block64",
)
def vmi_tadd_block64(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _add)


@canonical_vmi_template(
    target="a5",
    op="texp",
    name="vmi_texp_block64",
    context_constraints={"precisionType": ("default",)},
)
def vmi_texp_block64(src: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src,), _exp, allowed_dtypes=FLOAT_DTYPES)


@canonical_vmi_template(target="a5", op="tsub", name="vmi_tsub")
def vmi_tsub(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _sub, allowed_dtypes=SUB_DTYPES)


@canonical_vmi_template(target="a5", op="tmul", name="vmi_tmul")
def vmi_tmul(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _mul, allowed_dtypes=MUL_DTYPES)


@canonical_vmi_template(target="a5", op="tmax", name="vmi_tmax")
def vmi_tmax(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst, (src0, src1), _max, allowed_dtypes=FLOAT_DTYPES
    )


@canonical_vmi_template(target="a5", op="tmov", name="vmi_tmov")
def vmi_tmov(src: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src,), _move)


@canonical_vmi_template(target="a5", op="tmuls", name="vmi_tmuls")
def vmi_tmuls(src: Tile, scale: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmuls(values[0], scale, mask),
    )


@canonical_vmi_template(target="a5", op="tadds", name="vmi_tadds")
def vmi_tadds(src: Tile, scalar: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vadds(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="tmaxs", name="vmi_tmaxs")
def vmi_tmaxs(src: Tile, scalar: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmaxs(values[0], scalar, mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(target="a5", op="tmins", name="vmi_tmins")
def vmi_tmins(src: Tile, scalar: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmins(values[0], scalar, mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs",
    semantic_form="tile_scalar",
    context_constraints={"precisionType": ("default",)},
)
def vmi_tdivs(src: Tile, scalar: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_by_scalar(values[0], scalar, mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs_scalar_tile",
    semantic_form="scalar_tile",
    context_constraints={"precisionType": ("default",)},
)
def vmi_tdivs_scalar_tile(scalar: Scalar, src: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_scalar_by_vector(scalar, values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdiv",
    name="vmi_tdiv",
    context_constraints={"precisionType": ("default",)},
)
def vmi_tdiv(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst, (src0, src1),
        lambda values, mask: vmi_vdiv(values[0], values[1], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(target="a5", op="tmin", name="vmi_tmin")
def vmi_tmin(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst, (src0, src1), _min, allowed_dtypes=FLOAT_DTYPES
    )


@canonical_vmi_template(target="a5", op="tsubs", name="vmi_tsubs")
def vmi_tsubs(src: Tile, scalar: Scalar, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _subtract_scalar(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="tabs", name="vmi_tabs")
def vmi_tabs(src: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst, (src,), _abs, allowed_dtypes=FLOAT_DTYPES
    )


@canonical_vmi_template(target="a5", op="tneg", name="vmi_tneg")
def vmi_tneg(src: Tile, dst: Tile):
    emit_elementwise_vmi(
        dst, (src,), _neg, allowed_dtypes=FLOAT_DTYPES
    )


@canonical_vmi_template(
    target="a5",
    op="texpands",
    name="vmi_texpands",
    semantic_form="scalar_fill",
)
def vmi_texpands(scalar: Scalar, dst: Tile):
    emit_scalar_fill_vmi(scalar, dst)


@canonical_vmi_template(target="a5", op="trowmax", name="vmi_trowmax")
def vmi_trowmax(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_vmi(src, workspace, dst, kind="max")


@canonical_vmi_template(target="a5", op="trowsum", name="vmi_trowsum")
def vmi_trowsum(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_vmi(src, workspace, dst, kind="sum")


@canonical_vmi_template(
    target="a5",
    op="trowexpandsub",
    name="vmi_trowexpandsub",
)
def vmi_trowexpandsub(src: Tile, row_values: Tile, dst: Tile):
    emit_row_expand_sub_vmi(src, row_values, dst)


@canonical_vmi_template(target="a5", op="trowexpandmul", name="vmi_trowexpandmul")
def vmi_trowexpandmul(src: Tile, row_values: Tile, dst: Tile):
    emit_row_expand_binary_vmi(src, row_values, dst, binop="mul")


@canonical_vmi_template(
    target="a5",
    op="trowexpanddiv",
    name="vmi_trowexpanddiv",
    context_constraints={"precisionType": ("default",)},
)
def vmi_trowexpanddiv(src: Tile, row_values: Tile, dst: Tile):
    emit_row_expand_binary_vmi(src, row_values, dst, binop="div")


@canonical_vmi_template(
    target="a5",
    op="tsqrt",
    name="vmi_tsqrt",
    context_constraints={"precisionType": ("default",)},
)
def vmi_tsqrt(src: Tile, dst: Tile):
    emit_sqrt_vmi(src, dst)


@canonical_vmi_template(
    target="a5",
    op="trecip",
    name="vmi_trecip",
    context_constraints={"precisionType": ("default",)},
)
def vmi_trecip(src: Tile, dst: Tile):
    emit_recip_vmi(src, dst)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt",
    semantic_form="default",
    context_constraints={"precisionType": ("default",)},
)
def vmi_trsqrt(src: Tile, dst: Tile):
    emit_rsqrt_vmi(src, dst)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt_with_tmp",
    semantic_form="with_tmp",
    context_constraints={"precisionType": ("default",)},
)
def vmi_trsqrt_with_tmp(src: Tile, dst: Tile, tmp: Tile):
    emit_rsqrt_vmi(src, dst, tmp=tmp)


@canonical_vmi_template(
    target="a5",
    op="tgather",
    name="vmi_tgather_index",
    semantic_form="index",
)
def vmi_tgather_index(src: Tile, dst: Tile, indices: Tile, tmp: Tile):
    emit_gather_index_vmi(src, dst, indices, tmp)


@canonical_vmi_template(target="a5", op="tcolmax", name="vmi_tcolmax")
def vmi_tcolmax(src: Tile, dst: Tile):
    emit_col_reduce_vmi(src, dst, kind="max")


@canonical_vmi_template(target="a5", op="tcolsum", name="vmi_tcolsum")
def vmi_tcolsum(src: Tile, dst: Tile):
    emit_col_reduce_vmi(src, dst, kind="add")


@canonical_vmi_template(target="a5", op="tcolexpandsub", name="vmi_tcolexpandsub")
def vmi_tcolexpandsub(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="sub")


@canonical_vmi_template(target="a5", op="tcolexpandadd", name="vmi_tcolexpandadd")
def vmi_tcolexpandadd(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="add")


@canonical_vmi_template(target="a5", op="tcolexpandmul", name="vmi_tcolexpandmul")
def vmi_tcolexpandmul(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="mul")


@canonical_vmi_template(
    target="a5",
    op="tcolexpanddiv",
    name="vmi_tcolexpanddiv",
    # ExpandTileOp::appendOpContextAttrs unconditionally adds a `precisionType`
    # context attr to TColExpandDivOp (even when default), and validate_context_attrs
    # rejects attrs the candidate didn't declare — so the candidate must declare it.
    context_constraints={"precisionType": ("default",)},
)
def vmi_tcolexpanddiv(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="div")


@canonical_vmi_template(
    target="a5",
    op="tcvt",
    name="vmi_tcvt",
    context_constraints={
        "round_mode": ("RINT", "ROUND", "FLOOR", "CEIL", "TRUNC", "ODD"),
        "sat_mode": ("ON", "OFF"),
    },
)
def vmi_tcvt(src: Tile, dst: Tile):
    emit_convert_vmi(
        src,
        dst,
        round_mode=get_op_attr("round_mode", "RINT"),
        sat_mode=get_op_attr("sat_mode", "OFF"),
    )


__all__ = [
    "VMI_TILELIB_REGISTRY",
    "canonical_vmi_template",
    "emit_elementwise_vmi",
    "emit_scalar_fill_vmi",
    "emit_sqrt_vmi",
    "vmi_tadd_block64",
    "vmi_texp_block64",
    "vmi_tsub",
    "vmi_tmul",
    "vmi_tmax",
    "vmi_tmov",
    "vmi_tmuls",
    "vmi_tadds",
    "vmi_tmaxs",
    "vmi_tmins",
    "vmi_tdivs",
    "vmi_tdivs_scalar_tile",
    "vmi_tdiv",
    "vmi_tmin",
    "vmi_tsubs",
    "vmi_tabs",
    "vmi_tneg",
    "vmi_texpands",
    "vmi_trowmax",
    "vmi_trowsum",
    "vmi_trowexpandsub",
    "vmi_trowexpandmul",
    "vmi_trowexpanddiv",
    "vmi_tsqrt",
    "vmi_trecip",
    "vmi_trsqrt",
    "vmi_trsqrt_with_tmp",
    "vmi_tgather_index",
    "vmi_tcvt",
    "vmi_tcolmax",
    "vmi_tcolsum",
    "vmi_tcolexpandsub",
    "vmi_tcolexpandadd",
    "vmi_tcolexpandmul",
    "vmi_tcolexpanddiv",
]
