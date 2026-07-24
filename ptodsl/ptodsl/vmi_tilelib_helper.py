# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Instantiate a PTODSL VMI TileLib candidate for ``ExpandTileOp``."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys

from ._tile_template_tracing import (
    TileSpec,
    bf16,
    f16,
    f32,
    i8,
    i16,
    i32,
)
from .tilelib.registry import TileTemplateRegistry


_DTYPE_MAP = {
    "f32": f32,
    "f16": f16,
    "bf16": bf16,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}


def _normalize_op_name(op_name: str) -> str:
    return op_name[4:] if op_name.startswith("pto.") else op_name


def _parse_operand_specs(spec_text: str) -> list[dict]:
    try:
        raw_specs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operand-specs JSON: {exc}") from exc
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("operand-specs must be a non-empty JSON array")
    return raw_specs


def _parse_context_attrs(spec_text: str | None) -> dict[str, object]:
    if not spec_text:
        return {}
    try:
        attrs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid context-attrs JSON: {exc}") from exc
    if not isinstance(attrs, dict):
        raise ValueError("context-attrs must be a JSON object")
    return attrs


def _parse_dtype(raw: dict, index: int):
    dtype_name = raw.get("dtype")
    dtype = _DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(f"operand-specs[{index}] has unsupported dtype {dtype_name!r}")
    return dtype


def _parse_parameter_spec(raw: dict, index: int):
    if not isinstance(raw, dict):
        raise ValueError(f"operand-specs[{index}] must be an object")
    kind = raw.get("kind")
    if kind == "scalar":
        return _parse_dtype(raw, index)
    if kind != "tile":
        raise ValueError(
            f"operand-specs[{index}] must be a tile or scalar for the PTODSL VMI provider"
        )

    dtype = _parse_dtype(raw, index)
    shape = raw.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"operand-specs[{index}] requires a static rank-2 shape")
    try:
        parsed_shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"operand-specs[{index}] shape must contain integers") from exc

    valid_shape = raw.get("valid_shape")
    if valid_shape is not None:
        if not isinstance(valid_shape, list) or len(valid_shape) != 2:
            raise ValueError(f"operand-specs[{index}] valid_shape must be rank-2")
        if any(dim is None for dim in valid_shape):
            raise ValueError(
                "initial PTODSL VMI provider does not support dynamic valid_shape"
            )
        parsed_valid_shape = tuple(int(dim) for dim in valid_shape)
        if parsed_valid_shape != parsed_shape:
            raise ValueError(
                "initial PTODSL VMI provider requires valid_shape to equal physical "
                f"shape; operand {index} has valid_shape={parsed_valid_shape}, "
                f"shape={parsed_shape}"
            )

    memory_space = raw.get("memory_space", "ub")
    if memory_space != "ub":
        raise ValueError(
            f"initial PTODSL VMI provider supports only UB tiles, got {memory_space!r}"
        )
    b_layout = _parse_tile_config(raw.get("config"), index)
    return TileSpec(parsed_shape, dtype, memory_space="ub", b_layout=b_layout)


def _parse_tile_config(config: object, index: int) -> str:
    if config is None:
        return "row_major"
    if not isinstance(config, dict):
        raise ValueError(f"operand-specs[{index}] config must be an object")
    expected = {
        "s_layout": "none_box",
        "s_fractal_size": 512,
    }
    for key, expected_value in expected.items():
        value = config.get(key, expected_value)
        if key == "pad_value" and isinstance(value, str):
            value = value.lower()
        if value != expected_value:
            raise ValueError(
                "initial PTODSL VMI provider supports only the default secondary layout; "
                f"operand-specs[{index}] has {key}={config.get(key)!r}"
            )
    # `pad_value` is a semantic fill value, not a physical layout selector.
    # A VMI candidate still requires full static valid_shape above, so a
    # non-default value is safe only after a boundary op (for example
    # tfillpad) has materialized the padded lanes into the tile.
    b_layout = config.get("b_layout", "row_major")
    if b_layout not in {"row_major", "col_major"}:
        raise ValueError(
            "initial PTODSL VMI provider supports row-major or col-major tiles; "
            f"operand-specs[{index}] has b_layout={b_layout!r}"
        )
    return b_layout


def _find_candidates(module, *, target: str, op_name: str) -> list:
    registry = getattr(module, "VMI_TILELIB_REGISTRY", None)
    if not isinstance(registry, TileTemplateRegistry):
        raise TypeError(
            f"PTODSL VMI provider module {module.__name__!r} must expose "
            "VMI_TILELIB_REGISTRY as a TileTemplateRegistry"
        )
    return registry.lookup(op_name, target)


def has_registered_candidate(*, target: str, op_name: str, provider_module: str) -> bool:
    module = importlib.import_module(provider_module)
    return bool(
        _find_candidates(
            module,
            target=target,
            op_name=_normalize_op_name(op_name),
        )
    )


def _annotation_kind(annotation) -> str | None:
    token = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)
    if not isinstance(token, str):
        return None
    token = token.rsplit(".", 1)[-1]
    if token == "Tile":
        return "tile"
    if token == "Scalar" or token in _DTYPE_MAP:
        return "scalar"
    return None


def _specialize_candidate(candidate, operand_specs, context_attrs):
    signature = inspect.signature(candidate.py_fn)
    parameters = tuple(signature.parameters.items())
    if len(parameters) != len(operand_specs):
        raise ValueError(
            f"candidate {candidate.name!r} expects {len(parameters)} operands, "
            f"got {len(operand_specs)}"
        )

    parameter_specs = {}
    for index, ((name, parameter), raw_spec) in enumerate(
        zip(parameters, operand_specs)
    ):
        raw_kind = raw_spec.get("kind") if isinstance(raw_spec, dict) else None
        expected_kind = _annotation_kind(parameter.annotation)
        if expected_kind is not None and raw_kind != expected_kind:
            raise TypeError(
                f"candidate {candidate.name!r} parameter {name!r} expects "
                f"{expected_kind}, got {raw_kind!r}"
            )
        parameter_specs[name] = _parse_parameter_spec(raw_spec, index)

    return candidate.specialize(
        context_attrs=context_attrs or {},
        **parameter_specs,
    )


def _select_candidate(
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
    provider_module: str,
    context_attrs: dict[str, object] | None = None,
):
    module = importlib.import_module(provider_module)
    normalized_op = _normalize_op_name(op_name)
    candidates = _find_candidates(module, target=target, op_name=normalized_op)
    if not candidates:
        raise LookupError(
            f"no PTODSL VMI candidate for target={target!r}, op={normalized_op!r} "
            f"in module {provider_module!r}"
        )
    if len(candidates) == 1:
        candidate = candidates[0]
        return candidate, _specialize_candidate(
            candidate, operand_specs, context_attrs
        )

    forms: dict[str, list[str]] = {}
    for candidate in candidates:
        forms.setdefault(candidate.semantic_form, []).append(candidate.name)
    duplicate_forms = {
        form: names for form, names in forms.items() if len(names) > 1
    }
    if duplicate_forms:
        details = "; ".join(
            f"{form}: {', '.join(names)}"
            for form, names in sorted(duplicate_forms.items())
        )
        raise LookupError(
            "PTODSL VMI provider requires one canonical candidate per "
            f"(target, op, semantic_form); duplicate forms for "
            f"target={target!r}, op={normalized_op!r}: {details}"
        )

    legal = []
    failures = []
    for candidate in candidates:
        try:
            artifact = _specialize_candidate(
                candidate, operand_specs, context_attrs
            )
            artifact.mlir_text()
            legal.append((candidate, artifact))
        except Exception as exc:
            failures.append(
                f"{candidate.semantic_form}/{candidate.name}: {exc}"
            )

    if not legal:
        raise LookupError(
            f"no legal PTODSL VMI semantic form for target={target!r}, "
            f"op={normalized_op!r}; " + "; ".join(failures)
        )
    if len(legal) != 1:
        names = ", ".join(
            f"{candidate.semantic_form}/{candidate.name}"
            for candidate, _ in legal
        )
        raise LookupError(
            f"ambiguous PTODSL VMI semantic forms for target={target!r}, "
            f"op={normalized_op!r}: {names}"
        )
    return legal[0]


def instantiate_candidate(
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
    provider_module: str,
    context_attrs: dict[str, object] | None = None,
):
    _, artifact = _select_candidate(
        target=target,
        op_name=op_name,
        operand_specs=operand_specs,
        provider_module=provider_module,
        context_attrs=context_attrs,
    )
    return artifact


def get_candidate_metadata(
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
    provider_module: str,
    context_attrs: dict[str, object] | None = None,
) -> dict:
    candidate, artifact = _select_candidate(
        target=target,
        op_name=op_name,
        operand_specs=operand_specs,
        provider_module=provider_module,
        context_attrs=context_attrs,
    )
    mlir_text = artifact.mlir_text()
    return {
        "candidates": {
            "0": {
                "id": 0,
                "name": candidate.name,
                "semantic_form": candidate.semantic_form,
                "provider": "ptodsl-vmi",
                "loop_depth": 1 if "scf.for" in mlir_text else 0,
                "is_post_update": False,
                "has_tail": False,
            }
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTODSL VMI TileLib expand helper")
    parser.add_argument("--target", default="a5")
    parser.add_argument("--op", required=True)
    parser.add_argument("--operand-specs", required=True)
    parser.add_argument("--context-attrs")
    parser.add_argument("--provider-module", default="ptodsl.vmi_tilelib")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.metadata_only and not has_registered_candidate(
            target=args.target,
            op_name=args.op,
            provider_module=args.provider_module,
        ):
            sys.stdout.write(json.dumps({"provider_supported": False}))
            return 0
        operand_specs = _parse_operand_specs(args.operand_specs)
        context_attrs = _parse_context_attrs(args.context_attrs)
        if args.metadata_only:
            output = json.dumps(
                get_candidate_metadata(
                    target=args.target,
                    op_name=args.op,
                    operand_specs=operand_specs,
                    provider_module=args.provider_module,
                    context_attrs=context_attrs,
                )
            )
        else:
            artifact = instantiate_candidate(
                target=args.target,
                op_name=args.op,
                operand_specs=operand_specs,
                provider_module=args.provider_module,
                context_attrs=context_attrs,
            )
            output = artifact.mlir_text()
    except Exception as exc:
        print(f"vmi_tilelib_helper: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
