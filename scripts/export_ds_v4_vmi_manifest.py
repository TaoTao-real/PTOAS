#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Export a machine-readable TileOp manifest from generated DSv4 PTO files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from mlir.dialects import pto
from mlir.ir import IntegerAttr, Module

from ptodsl._bootstrap import make_context
from ptodsl.vmi_tilelib import VMI_TILELIB_REGISTRY


DYNAMIC_DIM = -(2**63)

NON_VECTOR_PIPES = {
    "tci": "PIPE_S",
    "tload": "PIPE_MTE2",
    "treshape": "PIPE_S",
    "tstore": "PIPE_MTE3",
}

VECTOR_BOUNDARY_OPS = {
    "textract",
    "tfillpad",
}

SPECIAL_FORMS = {
    "tcvt": "convert",
    "texpands": "scalar_fill",
    "tgather": "index",
    "tload": "movement",
    "trowmax": "reduce_with_tmp",
    "trowsum": "reduce_with_tmp",
    "tstore": "movement",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="ENTRYPOINT=FILE",
        help="Generated PTO file and the DSv4 entrypoint/path label it belongs to.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--source-revision", default="", help="pypto-lib revision.")
    parser.add_argument(
        "--partial",
        action="append",
        default=[],
        metavar="ENTRYPOINT=REASON",
        help="Record an entrypoint whose PyPTO export stopped at a known boundary.",
    )
    return parser.parse_args()


def parse_assignment(raw: str, option: str):
    if "=" not in raw:
        raise ValueError(f"{option} expects NAME=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    if not name or not value:
        raise ValueError(f"{option} expects non-empty NAME=VALUE")
    return name, value


def walk_operations(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk_operations(child.operation)


def enclosing_function(operation):
    current = operation
    while current is not None:
        if current.name == "func.func":
            return str(current.attributes["sym_name"]).strip('"')
        current = current.parent
    return ""


def constant_int(value):
    owner = getattr(value, "owner", None)
    operation = getattr(owner, "operation", owner)
    if operation is None or operation.name != "arith.constant":
        return None
    attribute = operation.attributes["value"]
    try:
        return int(IntegerAttr(attribute).value)
    except (TypeError, ValueError):
        return None


def normalize_dims(dims):
    return [None if int(dim) == DYNAMIC_DIM else int(dim) for dim in dims]


def static_valid_shape(value, tile_type):
    typed = normalize_dims(tile_type.valid_shape)
    if typed and all(dim is not None for dim in typed):
        return typed

    owner = getattr(value, "owner", None)
    operation = getattr(owner, "operation", owner)
    if operation is None:
        return typed
    view = operation.opview
    if operation.name in {
        "pto.alloc_tile",
        "pto.bind_tile",
        "pto.materialize_tile",
        "pto.subview",
    }:
        row = constant_int(getattr(view, "valid_row", None))
        col = constant_int(getattr(view, "valid_col", None))
        if row is not None and col is not None:
            return [row, col]
    if operation.name == "pto.treshape":
        source = view.src
        source_type = pto.TileBufType(source.type)
        if static_valid_shape(source, source_type) == list(source_type.shape):
            return list(tile_type.shape)
    return typed


def layout_name(value):
    return {0: "row_major", 1: "col_major"}.get(int(value), str(value))


def secondary_layout_name(value):
    return {0: "none_box", 1: "row_major", 2: "col_major"}.get(
        int(value), str(value)
    )


def tile_operand(value):
    tile_type = pto.TileBufType(value.type)
    return {
        "kind": "tile",
        "dtype": str(tile_type.element_type),
        "shape": list(tile_type.shape),
        "valid_shape": static_valid_shape(value, tile_type),
        "memory_space": str(tile_type.memory_space),
        "layout": {
            "b_layout": layout_name(tile_type.blayout_value),
            "s_layout": secondary_layout_name(tile_type.slayout_value),
            "s_fractal_size": int(tile_type.s_fractal_size),
        },
    }


def operand_spec(value):
    if pto.TileBufType.isinstance(value.type):
        return tile_operand(value)
    return {"kind": "scalar_or_view", "type": str(value.type)}


def operand_form(op_name, operands):
    if op_name in SPECIAL_FORMS:
        if op_name == "tgather":
            return SPECIAL_FORMS[op_name]
        if op_name in {"trowmax", "trowsum"}:
            return SPECIAL_FORMS[op_name]
        return SPECIAL_FORMS[op_name]

    kinds = [operand["kind"] for operand in operands]
    tile_positions = [index for index, kind in enumerate(kinds) if kind == "tile"]
    scalar_positions = [
        index for index, kind in enumerate(kinds) if kind == "scalar_or_view"
    ]
    if len(tile_positions) == 2 and not scalar_positions:
        return "unary"
    if len(tile_positions) == 3 and not scalar_positions:
        return "tile_tile"
    if len(tile_positions) == 2 and len(scalar_positions) == 1:
        return "scalar_tile" if scalar_positions[0] == 0 else "tile_scalar"
    return "mixed"


def context_attributes(operation):
    result = {}
    for raw_name in ("rmode", "sat_mode", "satmode", "precisionType"):
        if raw_name not in operation.attributes:
            continue
        value = str(operation.attributes[raw_name])
        match = re.search(r"\b([A-Z][A-Z0-9_]*|default|high_precision)\b", value)
        result[raw_name] = match.group(1) if match else value
    return result


def candidate_catalog():
    result = {}
    for descriptor in VMI_TILELIB_REGISTRY.all():
        if descriptor.target != "a5":
            continue
        op_name = descriptor.op.removeprefix("pto.")
        result.setdefault(op_name, []).append(
            {
                "name": descriptor.name,
                "semantic_form": descriptor.semantic_form,
            }
        )
    for candidates in result.values():
        candidates.sort(key=lambda item: (item["semantic_form"], item["name"]))
    return result


def pipe_for(op_name):
    if op_name in NON_VECTOR_PIPES:
        return NON_VECTOR_PIPES[op_name]
    if op_name in VECTOR_BOUNDARY_OPS:
        return "PIPE_V"
    return "UNRESOLVED"


def pipe_source_for(op_name):
    if op_name in NON_VECTOR_PIPES:
        return "static_non_vector_table"
    if op_name in VECTOR_BOUNDARY_OPS:
        return "static_vector_boundary_table"
    return "unresolved"


def signature_key(record):
    stable = {key: value for key, value in record.items() if key != "occurrences"}
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def collect_file(entrypoint: str, source: Path, registered_candidates):
    context = make_context()
    module = Module.parse(source.read_text(encoding="utf-8"), context)
    records = []
    for operation in walk_operations(module.operation):
        if not operation.name.startswith("pto.t"):
            continue
        op_name = operation.name.removeprefix("pto.")
        if not operation.opview.__class__.__name__.endswith("Op"):
            continue
        operands = [operand_spec(value) for value in operation.operands]
        candidates = registered_candidates.get(op_name, [])
        has_candidate = bool(candidates)
        pipe = pipe_for(op_name)
        records.append(
            {
                "entrypoint": entrypoint,
                "pto_file": source.name,
                "function": enclosing_function(operation),
                "op": op_name,
                "pipe": pipe,
                "pipe_source": pipe_source_for(op_name),
                "operand_form": operand_form(op_name, operands),
                "operands": operands,
                "context_attrs": context_attributes(operation),
                "vmi_candidate_registered": has_candidate,
                "registry_status": "registered" if has_candidate else "missing",
                "vmi_candidates": candidates,
                "implementation_status": (
                    "registry_only"
                    if has_candidate
                    else "boundary_only"
                    if op_name in VECTOR_BOUNDARY_OPS
                    else "not_applicable"
                    if op_name in NON_VECTOR_PIPES
                    else "unverified"
                ),
            }
        )
    return records


def main():
    args = parse_args()
    registered_candidates = candidate_catalog()
    records = []
    sources = []
    for raw in args.input:
        entrypoint, source_text = parse_assignment(raw, "--input")
        source = Path(source_text).resolve()
        records.extend(collect_file(entrypoint, source, registered_candidates))
        sources.append({"entrypoint": entrypoint, "pto_file": source.name})

    counts = Counter(signature_key(record) for record in records)
    unique = {}
    for record in records:
        key = signature_key(record)
        unique[key] = record
    signatures = []
    for key in sorted(unique):
        record = dict(unique[key])
        record["occurrences"] = counts[key]
        signatures.append(record)

    partial = []
    for raw in args.partial:
        entrypoint, reason = parse_assignment(raw, "--partial")
        partial.append({"entrypoint": entrypoint, "reason": reason})

    payload = {
        "schema_version": 2,
        "target": "a5",
        "source_revision": args.source_revision or None,
        "sources": sorted(sources, key=lambda item: (item["entrypoint"], item["pto_file"])),
        "partial_exports": sorted(partial, key=lambda item: item["entrypoint"]),
        "signatures": signatures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
