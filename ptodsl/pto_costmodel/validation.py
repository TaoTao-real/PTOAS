# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Candidate integrity, FIFO progress, and conservative aggregate budgets."""
from collections import deque

from pto_costmodel.contract import candidate_id, envelope, metric, validate_model
from pto_costmodel.schedule import build_schedule
from pto_costmodel.wire import fields, require


def prepare_candidate(package, config):
    schedule = build_schedule(package["program"], config)
    if package["program"]["loops"][0]["trip_count"]:
        pipe = next(p for p in package["program"]["pipes"] if p["direction"] == "v2c")
        require(pipe["effective_slot_num"] >= max(schedule["effective_preload"], 1), "CAPACITY",
                "P pipe does not satisfy ADR preload capacity requirement")
    check_progress(schedule)
    resources = resource_requirements(package, config)
    return dict(candidate_id=candidate_id(package["identity"], config), identity=package["identity"],
                configuration=config, schedule=schedule, resources=resources)


def check_progress(schedule):
    nodes = {e["id"] for e in schedule["events"]}
    incoming = {n: 0 for n in nodes}
    outgoing = {n: set() for n in nodes}
    pairs = {(e["source"], e["target"]) for e in schedule["dependencies"]}
    for order in schedule["core_order"].values():
        pairs.update(zip(order, order[1:]))
    for src, dst in pairs:
        require(src in nodes and dst in nodes, "SCHEDULE", "dangling schedule edge")
        outgoing[src].add(dst)
        incoming[dst] += 1
    ready = deque(n for n, degree in incoming.items() if degree == 0)
    visited = 0
    while ready:
        src = ready.popleft()
        visited += 1
        for dst in outgoing[src]:
            incoming[dst] -= 1
            if incoming[dst] == 0:
                ready.append(dst)
    require(visited == len(nodes), "FIFO_DEADLOCK", "core order and dependencies form a cycle")


def resource_requirements(package, config):
    profile = package["target"]["compiler_budget"]
    counts = {r["buffer_id"]: r["count"] for r in config["buffers"]}
    result = {core: {} for core in ("AIC0", "AIV0", "AIV1")}
    for buffer in package["program"]["buffers"]:
        if buffer["owner"] == "borrowed_entry":
            continue
        space = buffer["memory_space"]
        alignment = profile["alignment_bytes"][space]
        stride = (buffer["allocation_bytes"] + alignment - 1) // alignment * alignment
        size = stride * counts.get(buffer["id"], 1)
        for core in buffer["physical_instances"]:
            result[core][space] = result[core].get(space, 0) + size
    for spaces in result.values():
        for space, used in spaces.items():
            require(used <= profile["capacity_bytes"][space], "CAPACITY", f"aggregate {space} exceeds budget")
    return dict(unit="bytes", policy="conservative_no_reuse", per_core=result,
                model_temporaries=None, physical_feasibility="requires_PlanMemory")


def validate_result(result, package, candidates):
    envelope(result, "result")
    fields(result, ("protocol_version", "kind", "required_features", "identity", "model", "candidates"),
           ("extensions",))
    require(result["identity"] == package["identity"], "STALE_PLAN", "result identity mismatch")
    validate_model(result["model"])
    require(isinstance(result["candidates"], list), "SCHEMA", "candidate results must be an array")
    expected = {c["candidate_id"]: c for c in candidates}
    seen = set()
    for row in result["candidates"]:
        _validate_row(row, expected, seen)
    require(seen == set(expected), "MISSING_CANDIDATE", "model omitted candidate results")


def validate_request(request, package):
    envelope(request, "request")
    fields(request, ("protocol_version", "kind", "required_features", "identity", "candidates", "objective",
                     "memory_constraint", "execution_mode", "search_budget", "baseline_candidate_id"), ("extensions",))
    require(request["identity"] == package["identity"], "STALE_PLAN", "request identity mismatch")
    require(request["objective"] == "latency" and request["memory_constraint"] == "hard"
            and request["execution_mode"] == "recommendation", "UNSUPPORTED_FEATURE", "unsupported request mode")
    candidates = request["candidates"]
    require(isinstance(candidates, list) and 1 <= len(candidates) <= 32, "RANGE", "candidate budget exceeded")
    require(request["search_budget"] == dict(max_candidates=len(candidates)),
            "SCHEMA", "inconsistent search budget")
    require(candidates[0]["configuration"]["preload_count"] == 0
            and request["baseline_candidate_id"] == candidates[0]["candidate_id"],
            "BASELINE", "serial baseline missing")
    seen = set()
    for supplied in candidates:
        candidate = prepare_candidate(package, supplied["configuration"])
        require(candidate == supplied and candidate["candidate_id"] not in seen,
                "CANDIDATE", "request contains changed/duplicate candidate")
        seen.add(candidate["candidate_id"])


def _validate_row(row, expected, seen):
    fields(row, ("candidate_id", "schedule_fingerprint", "configuration", "latency", "coverage",
                 "resources", "evidence"))
    key = row["candidate_id"]
    require(isinstance(key, str) and key in expected and key not in seen, "CANDIDATE", "unknown/duplicate candidate")
    seen.add(key)
    candidate = expected[key]
    require(row["configuration"] == candidate["configuration"], "CANDIDATE", "model changed evaluated configuration")
    require(row["schedule_fingerprint"] == candidate["schedule"]["fingerprint"], "SCHEDULE", "wrong schedule evaluated")
    metric(row["latency"])
    fields(row["coverage"], ("status", "operations", "communication", "approximations", "unsupported"))
    require(row["coverage"]["status"] in ("complete", "partial", "unsupported", "reference"),
            "COVERAGE", "unknown coverage status")
    require(isinstance(row["coverage"]["approximations"], list)
            and isinstance(row["coverage"]["unsupported"], list), "COVERAGE", "coverage lists required")
    if row["coverage"]["status"] != "complete":
        require(row["latency"]["value"] is None, "COVERAGE", "partial coverage cannot claim full latency")
    else:
        require(row["latency"]["value"] is not None and not row["coverage"]["unsupported"],
                "COVERAGE", "complete result cannot contain unknown latency/unsupported components")
        require(row["coverage"]["operations"] == "complete" and row["coverage"]["communication"] == "complete",
                "COVERAGE", "complete coverage requires operation and transport models")
    require(isinstance(row["resources"], dict) and row["resources"].get("unit") == "bytes",
            "SCHEMA", "resource values must use bytes")
    require(isinstance(row["evidence"], dict), "SCHEMA", "evidence must be structured")
