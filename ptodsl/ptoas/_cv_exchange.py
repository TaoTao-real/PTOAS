# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compiler-side 2.0 package generation and independent integrity verification."""
from pathlib import Path

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas.costmodel import canonicalize, input_at_checkpoint, parse_graph
from ptoas._cv_common import validate_profile
from ptoas._cv_ir import attr
from ptoas._cv_plan import annotate_ids
from ptoas._cv_program import inferred_bindings, program_from_graph, validate_bindings
from pto_costmodel.contract import SEMANTICS, header
from pto_costmodel.package import digest_text, read_package
from pto_costmodel.wire import encode, fingerprint, publish, read_text, require


def export_v2(source, profile, output, bindings=None):
    validate_profile(profile)
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        canonical = canonicalize(source)
        module = ir.Module.parse(canonical)
        if any(op.operation.name == "builtin.module" for op in module.body.operations):
            from ptoas._cv_structured import export_structured
            return export_structured(module, profile, output, bindings)
        graph = parse_graph(canonical, profile, {})
        program = program_from_graph(graph)
        selected = inferred_bindings(program) if bindings is None else bindings
        validate_bindings(selected, program)
        annotate_ids(graph)
        target = dict(profile_version="pto.a5.budget.1", compiler_budget=profile,
                      hardware_profile=None, budget_source="PTOAS.PlanMemory", topology=dict(aic=1, aiv=2))
        files = {"program.json": encode(program), "runtime_bindings.json": encode(selected),
                 "target_profile.json": encode(target), "canonical.pto": graph.index.asm()}
        manifest = dict(**header("package"), semantics_version=SEMANTICS, checkpoint_version=1,
                        producer=dict(name="ptoas", version=attr(graph.index.module.operation,
                                                                 "pto.costmodel.compiler_version")),
                        program_fingerprint=fingerprint(program), bindings_fingerprint=fingerprint(selected),
                        target_fingerprint=fingerprint(target), files={k: digest_text(v) for k, v in files.items()})
        files["manifest.json"] = encode(manifest)
        publish(output, files)
    return manifest


def verified_graph(path, package, current_input=None):
    require(package["manifest"]["semantics_version"] == SEMANTICS, "UNSUPPORTED_FEATURE",
            "static candidate construction cannot map structured FA phases; use annotation-only Buffer plans")
    profile = package["target"]["compiler_budget"]
    validate_profile(profile)
    text = read_text(Path(path) / "canonical.pto") if current_input is None else input_at_checkpoint(current_input)
    graph = parse_graph(text, profile, {})
    require(program_from_graph(graph) == package["program"], "PACKAGE_INTEGRITY",
            "structured program differs from compiler-parsed PTO")
    validate_bindings(package["bindings"], package["program"])
    return graph
