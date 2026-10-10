# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Low-level VMI vector primitives and IEEE754 high-precision division emitters.

The thin ``_v*`` wrappers around the VMI builder namespace and the two
IEEE754 division emitters live here so ``_vmi_common`` stays under the
platform source-size limit; ``_vmi_common`` imports them back unchanged, so
per-op candidates keep importing the same names from
``TileOps.a5._vmi_common``.

The division emitters are straight-line VMI op sequences whose order is
semantically load-bearing, so they are split into fixed-order stages that
share a per-dtype state object. Each stage runs the original statements in
the original order, and the public ``_div_ieee754_{f32,f16}_vmi`` drivers
call the stages sequentially, leaving the emitted op sequence unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptoas.mlir.dialects import pto as _pto_dialect

from ptodsl import pto
from ptodsl._surface_values import unwrap_surface_value
from ptodsl._tile_template_tracing import (
    _MaskValue,
    _Value,
    _VectorValue,
    ScalarType,
    f16,
    f32,
    si16,
    si32,
)
from ptodsl._vmi_namespace import vmi as _vmi_builder

# ScalarType aliases without a ptodsl exported form. Mirrors the lane/mask
# geometry of the signed variants from ptodsl._tile_template_tracing.
ui32 = ScalarType("ui32", lanes=64, mask_bits=32, bytewidth=4)
ui16 = ScalarType("ui16", lanes=128, mask_bits=16, bytewidth=2)


def _pto_dtype(dtype: ScalarType):
    descriptors = {
        "f32": pto.f32,
        "f16": pto.f16,
        "bf16": pto.bf16,
        "i8": pto.i8,
        "i16": pto.i16,
        "i32": pto.i32,
        "si8": pto.si8,
        "si16": pto.si16,
        "si32": pto.si32,
        "ui8": pto.ui8,
        "ui16": pto.ui16,
        "ui32": pto.ui32,
    }
    try:
        return descriptors[dtype.name]
    except KeyError as exc:
        raise ValueError(f"unsupported VMI TileLib dtype {dtype}") from exc


def _wrap_vreg(value, dtype: ScalarType) -> _VectorValue:
    return _VectorValue(unwrap_surface_value(value), dtype)


def _wrap_mask(value, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(unwrap_surface_value(value), dtype)


def _vreg_lanes(value: _VectorValue) -> int:
    return _pto_dialect.VMIVRegType(value.value.type).element_count


def _validate_same_dtype(operation: str, *values: _VectorValue) -> ScalarType:
    if not values:
        raise ValueError(f"{operation} expects at least one vector")
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values):
        raise TypeError(f"{operation} operands must use the same dtype")
    return dtype


def _validate_mask(operation: str, mask: _MaskValue, dtype: ScalarType) -> None:
    if not isinstance(mask, _MaskValue):
        raise TypeError(f"{operation} expects a VMI mask")
    if mask.dtype.mask_bits != dtype.mask_bits:
        raise TypeError(
            f"{operation} mask granularity b{mask.dtype.mask_bits} is incompatible "
            f"with {dtype} lanes using b{dtype.mask_bits}"
        )


def _vbinary(name: str, lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    dtype = _validate_same_dtype(f"pto.vmi.{name}", lhs, rhs)
    _validate_mask(f"pto.vmi.{name}", mask, dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(lhs.value, rhs.value, mask.value), dtype)


def _vadd(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vadd", lhs, rhs, mask)


def _vsub(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vsub", lhs, rhs, mask)


def _vmul(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmul", lhs, rhs, mask)


def _vdiv(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vdiv", lhs, rhs, mask)


def _vand(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vand", lhs, rhs, mask)


def _vor(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vor", lhs, rhs, mask)


def _vshl(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshl", lhs, rhs, mask)


def _vshr(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshr", lhs, rhs, mask)


def _vunary(name: str, source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, mask.value), source.dtype)


def _vabs(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vabs", source, mask)


def _vvec_scalar(
    name: str,
    source: _VectorValue,
    scalar: _Value,
    mask: _MaskValue,
) -> _VectorValue:
    """Emit one VMI vector-scalar op through the modern entry point.

    ``name`` is the modern family name (``vadd``, not ``vadds``): the modern
    entry dispatches a (vector, scalar) pair back to the same vec-scalar op, so
    the emitted IR is unchanged while the deprecated ``_VMINamespace.<op>s``
    access -- and the warning it prints on every compile -- is gone.
    """
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, scalar.value, mask.value), source.dtype)


def _vadds(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vadd", source, scalar, mask)


def _vcmp(
    lhs: _VectorValue,
    rhs: _VectorValue,
    seed: _MaskValue,
    cmp: str,
) -> _MaskValue:
    dtype = _validate_same_dtype("pto.vmi.vcmp", lhs, rhs)
    _validate_mask("pto.vmi.vcmp", seed, dtype)
    return _wrap_mask(_vmi_builder.vcmp(lhs.value, rhs.value, seed.value, cmp), dtype)


def _vsel(
    true_value: _VectorValue,
    false_value: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vsel", true_value, false_value)
    _validate_mask("pto.vmi.vsel", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vsel(mask.value, true_value.value, false_value.value),
        dtype,
    )


def _vmula(
    acc: _VectorValue,
    lhs: _VectorValue,
    rhs: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vmula", acc, lhs, rhs)
    _validate_mask("pto.vmi.vmula", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vmula(acc.value, lhs.value, rhs.value, mask.value),
        dtype,
    )


def _pand(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vand mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vand(lhs.value, rhs.value), lhs.dtype)


def _por(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vor mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vor(lhs.value, rhs.value), lhs.dtype)


def _pnot(mask: _MaskValue) -> _MaskValue:
    return _wrap_mask(_vmi_builder.vnot(mask.value), mask.dtype)


def _scalar_constant(value: float | int, dtype: ScalarType) -> _Value:
    return _Value(unwrap_surface_value(pto.const(value, dtype=_pto_dtype(dtype))))


def _vbrc_scalar(
    scalar: _Value,
    *,
    like: _VectorValue | None = None,
    dtype: ScalarType | None = None,
) -> _VectorValue:
    if like is None and dtype is None:
        raise TypeError("_vbrc_scalar requires like= or dtype=")
    ref_dtype = like.dtype if like is not None else dtype
    size = _vreg_lanes(like) if like is not None else dtype.lanes
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=size), ref_dtype)


def _vconstant(
    value: float | int,
    dtype: ScalarType,
    *,
    like: _VectorValue | None = None,
    lanes: int | None = None,
) -> _VectorValue:
    if like is None and lanes is None:
        raise TypeError("_vconstant requires like= or lanes=")
    if like is not None and lanes is not None:
        raise TypeError("_vconstant lanes cannot be combined with like=")
    scalar = _scalar_constant(value, dtype)
    if like is not None:
        return _vbrc_scalar(scalar, like=like)
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=lanes), dtype)


def _vinterpret_cast(source: _VectorValue, dst_dtype: ScalarType) -> _VectorValue:
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("_vinterpret_cast expects a destination ScalarType")
    source_type = _pto_dialect.VMIVRegType(source.value.type)
    source_bits = source_type.element_count * source.dtype.bytewidth * 8
    dst_bits = dst_dtype.bytewidth * 8
    if source_bits % dst_bits != 0:
        raise ValueError("_vinterpret_cast requires matching total bit width")
    return _wrap_vreg(
        _vmi_builder.vinterpret_cast(source.value, to_dtype=_pto_dtype(dst_dtype)),
        dst_dtype,
    )


def _mask_as(mask: _MaskValue, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(mask.value, dtype)


def _vbrc_constant(
    value: float | int, dtype: ScalarType, like: _VectorValue
) -> _VectorValue:
    if dtype.name.startswith("ui"):
        return _vconstant(value, dtype, like=like)
    return _vbrc_scalar(_scalar_constant(value, dtype), like=like)


def _div_three_candidate_search_f32(
    lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    lhs_u32 = _vinterpret_cast(lhs, ui32)
    inf_bound = _vbrc_constant(0x7F800000, ui32, like=lhs_u32)
    sign_bit = _vbrc_constant(0x80000000, ui32, like=lhs_u32)
    zero = _vbrc_constant(0.0, f32, like=lhs)
    one = _vbrc_constant(1.0, f32, like=lhs)
    neg_one = _vbrc_constant(-1.0, f32, like=lhs)

    z = _vdiv(lhs, rhs, mask)
    z_u32 = _vinterpret_cast(z, ui32)
    z_or_sign = _vor(z_u32, sign_bit, _mask_as(mask, ui32))
    is_inf_nan = _vcmp(z_or_sign, inf_bound, _mask_as(mask, ui32), "ge")
    is_zero = _vcmp(z, zero, mask, "eq")
    special_mask = _por(is_inf_nan, is_zero)

    y = _vmul(rhs, neg_one, mask)
    residual = _vmula(lhs, z, y, mask)
    z_pre = _vadd(z, neg_one, mask)
    z_next = _vadd(z, one, mask)
    residual_pre = _vmula(lhs, z_pre, y, mask)
    residual_next = _vmula(lhs, z_next, y, mask)

    residual_abs = _vabs(residual, mask)
    residual_pre_abs = _vabs(residual_pre, mask)
    residual_next_abs = _vabs(residual_next, mask)
    better_pre = _vcmp(residual_pre_abs, residual_abs, mask, "lt")
    z_best = _vsel(z_pre, z, better_pre)
    residual_best_abs = _vsel(residual_pre_abs, residual_abs, better_pre)
    better_next = _vcmp(residual_next_abs, residual_best_abs, mask, "lt")
    z_best = _vsel(z_next, z_best, better_next)
    return _vsel(z, z_best, special_mask)


@dataclass
class _DivF32State:
    """Live vectors shared between the f32 IEEE754 division emit stages."""

    int_mask: _MaskValue = None
    f32_inf: _VectorValue = None
    sign_extractor: _VectorValue = None
    exponent_extractor: _VectorValue = None
    exponent_normalizer: _VectorValue = None
    subnormal_threshold: _VectorValue = None
    nan_value: _VectorValue = None
    min_denormal: _VectorValue = None
    zero_u32: _VectorValue = None
    normalize_scale_enlarge: _VectorValue = None
    normalize_scale_reduce: _VectorValue = None
    src0_abs: _VectorValue = None
    src1_abs: _VectorValue = None
    src0_abs_u32: _VectorValue = None
    src1_abs_u32: _VectorValue = None
    mask_valid: _MaskValue = None
    src0_all_u32: _VectorValue = None
    src1_all_u32: _VectorValue = None
    mask_src0_subnormal: _MaskValue = None
    mask_src0_normal: _MaskValue = None
    mask_src1_subnormal: _MaskValue = None
    mask_src1_normal: _MaskValue = None
    src0_norm: _VectorValue = None
    src1_norm: _VectorValue = None
    divided: _VectorValue = None
    divided_sign: _VectorValue = None
    scale: _VectorValue = None
    scale_mask: _MaskValue = None
    shift23: _VectorValue = None
    mask_valid_temp: _MaskValue = None
    mask_underflow2: _MaskValue = None
    divided_u32_temp: _VectorValue = None
    mask_overflow2: _MaskValue = None


def _div_f32_prologue(
    state: _DivF32State,
    src0: _VectorValue,
    src1: _VectorValue,
    mask: _MaskValue,
) -> None:
    """Materialize the f32 bit-level constants and operand-validity masks."""
    state.int_mask = _mask_as(mask, ui32)
    src0_u32 = _vinterpret_cast(src0, ui32)
    state.f32_inf = _vbrc_constant(0x7F800000, ui32, like=src0_u32)
    state.sign_extractor = _vbrc_constant(0x80000000, ui32, like=src0_u32)
    state.exponent_extractor = _vbrc_constant(0x807FFFFF, ui32, like=src0_u32)
    state.exponent_normalizer = _vbrc_constant(0x3F800000, ui32, like=src0_u32)
    state.subnormal_threshold = _vbrc_constant(0x007FFFFF, ui32, like=src0_u32)
    state.nan_value = _vbrc_constant(0x7FC00000, ui32, like=src0_u32)
    state.min_denormal = _vbrc_constant(0x1, ui32, like=src0_u32)
    state.zero_u32 = _vbrc_constant(0, ui32, like=src0_u32)
    state.normalize_scale_enlarge = _vbrc_constant(8388608.0, f32, like=src0)
    state.normalize_scale_reduce = _vbrc_constant(1.1920928955078125e-07, f32, like=src0)

    state.src0_abs = _vabs(src0, mask)
    state.src1_abs = _vabs(src1, mask)
    state.src0_abs_u32 = _vinterpret_cast(state.src0_abs, ui32)
    state.src1_abs_u32 = _vinterpret_cast(state.src1_abs, ui32)

    mask_inf_src0 = _vcmp(state.src0_abs_u32, state.f32_inf, state.int_mask, "eq")
    mask_inf_src1 = _vcmp(state.src1_abs_u32, state.f32_inf, state.int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(state.src0_abs_u32, state.zero_u32, state.int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(state.src1_abs_u32, state.zero_u32, state.int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    state.mask_valid = _pnot(mask_invalid)


def _div_f32_normalize(
    state: _DivF32State,
    src0: _VectorValue,
    src1: _VectorValue,
) -> None:
    """Scale subnormal operands up and re-bias their exponents."""
    state.mask_src0_subnormal = _vcmp(
        state.src0_abs_u32, state.subnormal_threshold, state.int_mask, "eq"
    )
    state.mask_src0_normal = _pnot(state.mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, state.normalize_scale_enlarge, _mask_as(state.mask_src0_subnormal, f32)
    )
    state.mask_src1_subnormal = _vcmp(
        state.src1_abs_u32, state.subnormal_threshold, state.int_mask, "lt"
    )
    state.mask_src1_normal = _pnot(state.mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, state.normalize_scale_enlarge, _mask_as(state.mask_src1_subnormal, f32)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(state.mask_src0_normal, f32))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(state.mask_src1_normal, f32))
    state.src0_all_u32 = _vinterpret_cast(src0_all, ui32)
    state.src1_all_u32 = _vinterpret_cast(src1_all, ui32)

    src0_norm_u32 = _vand(
        state.src0_all_u32, state.exponent_extractor, state.mask_valid
    )
    src1_norm_u32 = _vand(
        state.src1_all_u32, state.exponent_extractor, state.mask_valid
    )
    src0_norm_u32 = _vadd(src0_norm_u32, state.exponent_normalizer, state.mask_valid)
    src1_norm_u32 = _vadd(src1_norm_u32, state.exponent_normalizer, state.mask_valid)
    state.src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u32, f32), src0_all, _mask_as(state.mask_valid, f32)
    )
    state.src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u32, f32), src1_all, _mask_as(state.mask_valid, f32)
    )


def _div_f32_divide_and_rescale(state: _DivF32State) -> None:
    """Run the three-candidate division and undo the subnormal scaling."""
    state.divided = _div_three_candidate_search_f32(
        state.src0_norm, state.src1_norm, _mask_as(state.mask_valid, f32)
    )
    mask0 = _pand(state.mask_src0_subnormal, state.mask_src1_normal)
    state.divided = _vsel(
        _vmul(state.divided, state.normalize_scale_reduce, _mask_as(mask0, f32)),
        state.divided,
        _mask_as(mask0, f32),
    )
    mask0 = _pand(state.mask_src0_normal, state.mask_src1_subnormal)
    state.divided = _vsel(
        _vmul(state.divided, state.normalize_scale_enlarge, _mask_as(mask0, f32)),
        state.divided,
        _mask_as(mask0, f32),
    )


def _div_f32_scale_and_underflow(
    state: _DivF32State,
    mask: _MaskValue,
) -> None:
    """Extract the exponent scale and apply the f32 underflow clamps."""
    divided_u32 = _vinterpret_cast(state.divided, ui32)
    state.divided_sign = _vand(divided_u32, state.sign_extractor, state.int_mask)
    src0_exponent = _vand(state.src0_all_u32, state.f32_inf, state.int_mask)
    src1_exponent = _vand(state.src1_all_u32, state.f32_inf, state.int_mask)
    state.shift23 = _vbrc_constant(23, ui32, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, state.shift23, state.int_mask)
    src1_exp_shifted = _vshr(src1_exponent, state.shift23, state.int_mask)

    state.scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, state.int_mask), si32
    )
    state.scale_mask = _mask_as(mask, si32)
    state.scale = _vadds(state.scale, _scalar_constant(127, si32), state.scale_mask)

    neg23 = _vbrc_constant(-23, si32, like=state.scale)
    mask_underflow1 = _vcmp(state.scale, neg23, state.scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, state.mask_valid)
    z1_u32 = _vadd(state.divided_sign, state.min_denormal, mask_underflow1)
    z2_u32 = _vadd(state.divided_sign, state.zero_u32, mask_underflow1)

    src0_norm_abs = _vabs(state.src0_norm, _mask_as(state.mask_valid, f32))
    src1_norm_abs = _vabs(state.src1_norm, _mask_as(state.mask_valid, f32))
    mask_norm = _vcmp(
        src0_norm_abs, src1_norm_abs, _mask_as(state.mask_valid, f32), "le"
    )
    state.divided_u32_temp = _vsel(
        _vsel(z2_u32, z1_u32, mask_norm), divided_u32, mask_underflow1
    )

    state.mask_valid_temp = _pand(_pnot(mask_underflow1), state.mask_valid)
    state.mask_underflow2 = _vcmp(state.scale, neg23, state.scale_mask, "lt")
    state.mask_underflow2 = _pand(state.mask_underflow2, state.mask_valid_temp)
    state.divided_u32_temp = _vsel(
        _vadd(state.divided_sign, state.zero_u32, state.mask_underflow2),
        state.divided_u32_temp,
        state.mask_underflow2,
    )


def _div_f32_overflow(
    state: _DivF32State,
    src0: _VectorValue,
) -> None:
    """Apply the f32 overflow clamps and exponent pre-scaling."""
    state.mask_valid_temp = _pand(
        _pnot(state.mask_underflow2), state.mask_valid_temp
    )
    max_exp = _vbrc_constant(255, si32, like=state.scale)
    mask_overflow1 = _vcmp(state.scale, max_exp, state.scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, state.mask_valid_temp)
    state.scale = _vsel(
        _vadds(state.scale, _scalar_constant(-1, si32), mask_overflow1),
        state.scale,
        mask_overflow1,
    )

    divided_f32_temp = _vinterpret_cast(state.divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vbrc_constant(2.0, f32, like=src0),
            _mask_as(mask_overflow1, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_overflow1, f32),
    )

    state.mask_overflow2 = _vcmp(state.scale, max_exp, state.scale_mask, "gt")
    state.mask_overflow2 = _pand(state.mask_overflow2, state.mask_valid_temp)
    state.divided_u32_temp = _vsel(
        _vadd(state.divided_sign, state.f32_inf, state.mask_overflow2),
        _vinterpret_cast(divided_f32_temp, ui32),
        state.mask_overflow2,
    )


def _div_f32_finalize(
    state: _DivF32State,
    mask: _MaskValue,
) -> _VectorValue:
    """Scale by the final exponent difference and fix up NaN operands."""
    mask_valid_final = _pand(_pnot(state.mask_overflow2), state.mask_valid_temp)
    zero_exp = _vbrc_constant(0, si32, like=state.scale)
    mask_pos_exp = _vcmp(
        state.scale, zero_exp, _mask_as(mask_valid_final, si32), "gt"
    )
    scale_u32 = _vinterpret_cast(state.scale, ui32)
    exp_shifted = _vshl(scale_u32, state.shift23, _mask_as(mask_pos_exp, ui32))
    exp_factor_f32 = _vinterpret_cast(exp_shifted, f32)
    divided_f32_temp = _vinterpret_cast(state.divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(divided_f32_temp, exp_factor_f32, _mask_as(mask_pos_exp, f32)),
        divided_f32_temp,
        _mask_as(mask_pos_exp, f32),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(state.scale, mask_pos_exp_not)
    shr_factor_u32 = _vshr(
        _vbrc_constant(4194304, ui32, like=scale_u32),
        _vinterpret_cast(scale_abs, ui32),
        _mask_as(mask_pos_exp_not, ui32),
    )
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vinterpret_cast(shr_factor_u32, f32),
            _mask_as(mask_pos_exp_not, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_pos_exp_not, f32),
    )

    mask_nan = _por(
        _vcmp(state.src0_abs, state.src0_abs, mask, "ne"),
        _vcmp(state.src1_abs, state.src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(state.nan_value, f32), divided_f32_temp, mask_nan
    )


def _div_ieee754_f32_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    """Emit the bit-exact IEEE754 f32 division sequence in fixed stage order."""
    state = _DivF32State()
    _div_f32_prologue(state, src0, src1, mask)
    _div_f32_normalize(state, src0, src1)
    _div_f32_divide_and_rescale(state)
    _div_f32_scale_and_underflow(state, mask)
    _div_f32_overflow(state, src0)
    return _div_f32_finalize(state, mask)


@dataclass
class _DivF16State:
    """Live vectors shared between the f16 IEEE754 division emit stages."""

    int_mask: _MaskValue = None
    f16_inf: _VectorValue = None
    exponent_extractor: _VectorValue = None
    exponent_normalizer: _VectorValue = None
    sign_extractor: _VectorValue = None
    subnormal_threshold: _VectorValue = None
    nan_value: _VectorValue = None
    min_denormal: _VectorValue = None
    zero_u16: _VectorValue = None
    normalize_scale_enlarge: _VectorValue = None
    normalize_scale_reduce: _VectorValue = None
    src0_abs: _VectorValue = None
    src1_abs: _VectorValue = None
    src0_abs_u16: _VectorValue = None
    src1_abs_u16: _VectorValue = None
    mask_valid: _MaskValue = None
    src0_all_u16: _VectorValue = None
    src1_all_u16: _VectorValue = None
    mask_src0_subnormal: _MaskValue = None
    mask_src0_normal: _MaskValue = None
    mask_src1_subnormal: _MaskValue = None
    mask_src1_normal: _MaskValue = None
    src0_norm: _VectorValue = None
    src1_norm: _VectorValue = None
    mask_norm: _MaskValue = None
    divided: _VectorValue = None
    divided_sign: _VectorValue = None
    scale: _VectorValue = None
    scale_mask: _MaskValue = None
    shift10: _VectorValue = None
    mask_valid_temp: _MaskValue = None
    mask_underflow2: _MaskValue = None
    divided_u16_temp: _VectorValue = None
    mask_overflow2: _MaskValue = None


def _div_f16_prologue(
    state: _DivF16State,
    src0: _VectorValue,
    src1: _VectorValue,
    mask: _MaskValue,
) -> None:
    """Materialize the f16 bit-level constants and operand-validity masks."""
    state.int_mask = _mask_as(mask, ui16)
    src0_u16 = _vinterpret_cast(src0, ui16)
    state.f16_inf = _vbrc_constant(0x7C00, ui16, like=src0_u16)
    state.exponent_extractor = _vbrc_constant(0x83FF, ui16, like=src0_u16)
    state.exponent_normalizer = _vbrc_constant(0x3C00, ui16, like=src0_u16)
    state.sign_extractor = _vbrc_constant(0x8000, ui16, like=src0_u16)
    state.subnormal_threshold = _vbrc_constant(0x03FF, ui16, like=src0_u16)
    state.nan_value = _vbrc_constant(0x7E00, ui16, like=src0_u16)
    state.min_denormal = _vbrc_constant(0x1, ui16, like=src0_u16)
    state.zero_u16 = _vbrc_constant(0, ui16, like=src0_u16)
    state.normalize_scale_enlarge = _vbrc_constant(1024.0, f16, like=src0)
    state.normalize_scale_reduce = _vbrc_constant(0.0009765625, f16, like=src0)

    state.src0_abs = _vabs(src0, mask)
    state.src1_abs = _vabs(src1, mask)
    state.src0_abs_u16 = _vinterpret_cast(state.src0_abs, ui16)
    state.src1_abs_u16 = _vinterpret_cast(state.src1_abs, ui16)

    mask_inf_src0 = _vcmp(state.src0_abs_u16, state.f16_inf, state.int_mask, "eq")
    mask_inf_src1 = _vcmp(state.src1_abs_u16, state.f16_inf, state.int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(state.src0_abs_u16, state.zero_u16, state.int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(state.src1_abs_u16, state.zero_u16, state.int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    state.mask_valid = _pnot(mask_invalid)


def _div_f16_normalize(
    state: _DivF16State,
    src0: _VectorValue,
    src1: _VectorValue,
) -> None:
    """Scale subnormal operands up and re-bias their exponents."""
    state.mask_src0_subnormal = _vcmp(
        state.src0_abs_u16, state.subnormal_threshold, state.int_mask, "lt"
    )
    state.mask_src0_normal = _pnot(state.mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, state.normalize_scale_enlarge, _mask_as(state.mask_src0_subnormal, f16)
    )
    state.mask_src1_subnormal = _vcmp(
        state.src1_abs_u16, state.subnormal_threshold, state.int_mask, "lt"
    )
    state.mask_src1_normal = _pnot(state.mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, state.normalize_scale_enlarge, _mask_as(state.mask_src1_subnormal, f16)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(state.mask_src0_normal, f16))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(state.mask_src1_normal, f16))
    state.src0_all_u16 = _vinterpret_cast(src0_all, ui16)
    state.src1_all_u16 = _vinterpret_cast(src1_all, ui16)

    src0_norm_u16 = _vand(
        state.src0_all_u16, state.exponent_extractor, state.mask_valid
    )
    src1_norm_u16 = _vand(
        state.src1_all_u16, state.exponent_extractor, state.mask_valid
    )
    src0_norm_u16 = _vadd(src0_norm_u16, state.exponent_normalizer, state.mask_valid)
    src1_norm_u16 = _vadd(src1_norm_u16, state.exponent_normalizer, state.mask_valid)
    state.src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u16, f16), src0_all, _mask_as(state.mask_valid, f16)
    )
    state.src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u16, f16), src1_all, _mask_as(state.mask_valid, f16)
    )


def _div_f16_divide_and_rescale(state: _DivF16State) -> None:
    """Run the plain f16 division and undo the subnormal scaling."""
    src0_norm_abs = _vabs(state.src0_norm, _mask_as(state.mask_valid, f16))
    src1_norm_abs = _vabs(state.src1_norm, _mask_as(state.mask_valid, f16))
    state.mask_norm = _vcmp(
        src0_norm_abs, src1_norm_abs, _mask_as(state.mask_valid, f16), "le"
    )
    state.divided = _vdiv(
        state.src0_norm, state.src1_norm, _mask_as(state.mask_valid, f16)
    )

    mask0 = _pand(state.mask_src0_subnormal, state.mask_src1_normal)
    state.divided = _vsel(
        _vmul(state.divided, state.normalize_scale_reduce, _mask_as(mask0, f16)),
        state.divided,
        _mask_as(mask0, f16),
    )
    mask0 = _pand(state.mask_src0_normal, state.mask_src1_subnormal)
    state.divided = _vsel(
        _vmul(state.divided, state.normalize_scale_enlarge, _mask_as(mask0, f16)),
        state.divided,
        _mask_as(mask0, f16),
    )


def _div_f16_scale_and_underflow(
    state: _DivF16State,
    mask: _MaskValue,
) -> None:
    """Extract the exponent scale and apply the f16 underflow clamps."""
    divided_u16 = _vinterpret_cast(state.divided, ui16)
    state.divided_sign = _vand(divided_u16, state.sign_extractor, state.int_mask)
    src0_exponent = _vand(state.src0_all_u16, state.f16_inf, state.int_mask)
    src1_exponent = _vand(state.src1_all_u16, state.f16_inf, state.int_mask)
    state.shift10 = _vbrc_constant(10, ui16, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, state.shift10, state.int_mask)
    src1_exp_shifted = _vshr(src1_exponent, state.shift10, state.int_mask)

    state.scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, state.int_mask), si16
    )
    state.scale_mask = _mask_as(mask, si16)
    state.scale = _vadds(state.scale, _scalar_constant(15, si16), state.scale_mask)

    neg9 = _vbrc_constant(-9, si16, like=state.scale)
    mask_underflow1 = _vcmp(state.scale, neg9, state.scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, state.mask_valid)
    z1_u16 = _vadd(state.divided_sign, state.min_denormal, mask_underflow1)
    z2_u16 = _vadd(state.divided_sign, state.zero_u16, mask_underflow1)
    state.divided_u16_temp = _vsel(
        _vsel(z2_u16, z1_u16, state.mask_norm), divided_u16, mask_underflow1
    )

    state.mask_valid_temp = _pand(_pnot(mask_underflow1), state.mask_valid)
    state.mask_underflow2 = _vcmp(state.scale, neg9, state.scale_mask, "lt")
    state.mask_underflow2 = _pand(state.mask_underflow2, state.mask_valid_temp)
    state.divided_u16_temp = _vsel(
        _vadd(state.divided_sign, state.zero_u16, state.mask_underflow2),
        state.divided_u16_temp,
        state.mask_underflow2,
    )


def _div_f16_overflow(
    state: _DivF16State,
    src0: _VectorValue,
) -> None:
    """Apply the f16 overflow clamps and exponent pre-scaling."""
    state.mask_valid_temp = _pand(
        _pnot(state.mask_underflow2), state.mask_valid_temp
    )
    max_exp = _vbrc_constant(31, si16, like=state.scale)
    mask_overflow1 = _vcmp(state.scale, max_exp, state.scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, state.mask_valid_temp)
    state.scale = _vsel(
        _vadds(state.scale, _scalar_constant(-1, si16), mask_overflow1),
        state.scale,
        mask_overflow1,
    )

    divided_f16_temp = _vinterpret_cast(state.divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vbrc_constant(2.0, f16, like=src0),
            _mask_as(mask_overflow1, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_overflow1, f16),
    )

    state.mask_overflow2 = _vcmp(state.scale, max_exp, state.scale_mask, "gt")
    state.mask_overflow2 = _pand(state.mask_overflow2, state.mask_valid_temp)
    state.divided_u16_temp = _vsel(
        _vadd(state.divided_sign, state.f16_inf, state.mask_overflow2),
        _vinterpret_cast(divided_f16_temp, ui16),
        state.mask_overflow2,
    )


def _div_f16_finalize(
    state: _DivF16State,
    mask: _MaskValue,
) -> _VectorValue:
    """Scale by the final exponent difference and fix up NaN operands."""
    mask_valid_final = _pand(_pnot(state.mask_overflow2), state.mask_valid_temp)
    zero_exp = _vbrc_constant(0, si16, like=state.scale)
    mask_pos_exp = _vcmp(
        state.scale, zero_exp, _mask_as(mask_valid_final, si16), "gt"
    )
    scale_u16 = _vinterpret_cast(state.scale, ui16)
    exp_factor_f16 = _vinterpret_cast(
        _vshl(scale_u16, state.shift10, _mask_as(mask_pos_exp, ui16)), f16
    )
    divided_f16_temp = _vinterpret_cast(state.divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(divided_f16_temp, exp_factor_f16, _mask_as(mask_pos_exp, f16)),
        divided_f16_temp,
        _mask_as(mask_pos_exp, f16),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(state.scale, mask_pos_exp_not)
    shr_factor_u16 = _vshr(
        _vbrc_constant(512, ui16, like=scale_u16),
        _vinterpret_cast(scale_abs, ui16),
        _mask_as(mask_pos_exp_not, ui16),
    )
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vinterpret_cast(shr_factor_u16, f16),
            _mask_as(mask_pos_exp_not, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_pos_exp_not, f16),
    )

    mask_nan = _por(
        _vcmp(state.src0_abs, state.src0_abs, mask, "ne"),
        _vcmp(state.src1_abs, state.src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(state.nan_value, f16), divided_f16_temp, mask_nan
    )


def _div_ieee754_f16_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    """Emit the bit-exact IEEE754 f16 division sequence in fixed stage order."""
    state = _DivF16State()
    _div_f16_prologue(state, src0, src1, mask)
    _div_f16_normalize(state, src0, src1)
    _div_f16_divide_and_rescale(state)
    _div_f16_scale_and_underflow(state, mask)
    _div_f16_overflow(state, src0)
    return _div_f16_finalize(state, mask)
