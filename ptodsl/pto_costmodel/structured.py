# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Explicitly negotiated assembly input; no static schedule is inferred."""
from pto_costmodel.contract import FEATURES
from pto_costmodel.wire import fields, integer, require

SEMANTICS = "pto.structured_cv.1"
FEATURES = FEATURES + ["structured_regions", "module_ownership", "symbolic_loop_domains"]
REQUIRED = ["typed_operations", "buffer_ownership", "structured_regions", "module_ownership", "symbolic_loop_domains"]


def validate_bindings(bindings, program):
    fields(bindings, ("scalars", "launch", "alias_contract"))
    require(isinstance(bindings["scalars"], dict), "BINDINGS", "scalars must map value IDs to integers")
    eligible = {a for f in program["functions"] if f["role"] == "entry"
                for a in f["arguments"] if program["values"][a]["kind"] in ("integer", "index")}
    require(set(bindings["scalars"]) <= eligible, "BINDINGS", "unknown entry scalar ID")
    for value in bindings["scalars"].values():
        integer(value, "scalar binding")
    fields(bindings["launch"], (), ("block_count",))
    if "block_count" in bindings["launch"]:
        integer(bindings["launch"]["block_count"], "block_count", 1)
    require(bindings["alias_contract"] in ("unknown", "disjoint"), "BINDINGS", "unsupported alias contract")


def validate_annotation_plan(plan, package):
    from pto_costmodel.contract import envelope, validate_model
    envelope(plan, "annotation_plan", FEATURES)
    fields(plan, ("protocol_version", "kind", "required_features", "identity", "buffers", "model"), ("extensions",))
    require(plan["identity"] == package["identity"], "STALE_PLAN", "plan input identity mismatch")
    validate_model(plan["model"])
    require(isinstance(plan["buffers"], list), "SCHEMA", "buffers must be an array")
    expected = {b["id"] for b in package["program"]["buffers"] if b["multi_buffer_eligible"]}
    seen = set()
    for row in plan["buffers"]:
        fields(row, ("buffer_id", "count"))
        require(isinstance(row["buffer_id"], str) and row["buffer_id"] in expected,
                "OWNERSHIP", "unknown or non-local buffer")
        require(row["buffer_id"] not in seen, "DUPLICATE_ID", row["buffer_id"])
        seen.add(row["buffer_id"])
        require(integer(row["count"], "count", 1) <= 256, "RANGE", "buffer count exceeds 256")
    require(seen == expected, "MISSING_BUFFER", "provide all local buffer counts")
