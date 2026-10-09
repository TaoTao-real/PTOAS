# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Preserve assembly structure at the exchange seam; never infer an FA schedule."""
from pathlib import Path

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_common import OUTPUT_ATTRS, validate_profile
from ptoas._cv_ir import ALLOWED, ALIASES, IRIndex, attr, owner_operation, tile_info
from ptoas._cv_types import attribute_record, type_record
from pto_costmodel.structured import SEMANTICS, REQUIRED, validate_bindings, validate_annotation_plan
from pto_costmodel.wire import encode, fingerprint, publish, read_text, require

SUPPORTED = ALLOWED | {
    "func.call", "arith.floordivsi", "arith.remsi", "pto.get_block_idx", "pto.get_block_num",
    "pto.tcvt", "pto.tmax", "pto.trowmax", "pto.trowsum", "pto.trowexpanddiv",
    "pto.trowexpandmul", "pto.trowexpandsub",
}
# Storage effects of these operations follow their registered src/dst/tmp accessors.
TILE_COMPUTE = {
    "pto.tload", "pto.tstore", "pto.tmov", "pto.tmatmul", "pto.tmatmul.acc",
    "pto.tadd", "pto.tsub", "pto.tmul", "pto.tdiv", "pto.tmuls", "pto.tadds",
    "pto.tneg", "pto.texp", "pto.tcvt", "pto.tmax", "pto.trowmax", "pto.trowsum",
    "pto.trowexpanddiv", "pto.trowexpandmul", "pto.trowexpandsub",
}


def enclosing(op, name):
    parent = op.parent
    while parent is not None:
        if parent.name == name:
            return parent
        parent = parent.parent
    return None


class StructuredAssembly:
    def __init__(self, module, profile):
        require(module.operation.verify(), "IR", "invalid assembly")
        require(attr(module.operation, "pto.target_arch") == "a5", "TARGET", "A5 required")
        require(attr(module.operation, "pto.costmodel.checkpoint_version") == 1,
                "VERSION", "native checkpoint required")
        self.index, self.profile = IRIndex(module), profile
        self.funcs = [o for o in self.index.operations if o.name == "func.func"]
        self.roots = {}
        self.buffer_ops = {}

    def resolve(self, name, user, logical=False):
        def matches(f):
            return attr(f, "sym_name") == name or (logical and attr(f, "pto.ptodsl.logical_name") == name)
        matches_ = [f for f in self.funcs if matches(f)]
        local = [f for f in matches_ if enclosing(f, "builtin.module") == enclosing(user, "builtin.module")]
        if local and len(local[0].regions[0].blocks):
            require(len(local) == 1, "SYMBOL", "ambiguous local function")
            return local[0]
        definitions = [f for f in matches_ if len(f.regions[0].blocks)
                       and attr(f, "sym_visibility", "public") != "private"]
        require(len(definitions) == 1, "SYMBOL", "unresolved or ambiguous peer/callee: " + name)
        target = definitions[0]
        if local:
            require(local[0].attributes["function_type"] == target.attributes["function_type"],
                    "SYMBOL", "declaration/definition type mismatch")
        return target

    def program(self):
        idx, profile = self.index, self.profile
        program = dict(semantics_version=SEMANTICS, modules=[], functions=[], operations=[],
                       values={}, loops=[], calls=[], buffers=[], aliases=[], pipes=[], transactions=[],
                       memory_accesses=[], dependencies=[])
        for op in idx.operations:
            require(op.name in SUPPORTED, "UNSUPPORTED_OPERATION", op.name)
            require("pto.multi_buffer_addrs" not in op.attributes, "PHYSICAL_ADDRESS", "internal addresses forbidden")
            require(attr(op, "pto.cv_preload_count", 0) == 0, "UNSUPPORTED_FEATURE",
                    "structured preload semantics require a phase-aware schedule mapping")
            key = idx.ids[op]
            attributes = {a.name: attribute_record(a.attr, profile) for a in op.attributes
                          if a.name not in OUTPUT_ATTRS and not a.name.startswith("pto.costmodel.")}
            regions = []
            for region in op.regions:
                blocks = []
                for block in region.blocks:
                    blocks.append(dict(arguments=[idx.values[a] for a in block.arguments],
                                       operations=[idx.ids[o.operation] for o in block.operations]))
                    for a in block.arguments:
                        program["values"][idx.values[a]] = type_record(a.type, profile)
                regions.append(blocks)
            for result in op.results:
                program["values"][idx.values[result]] = type_record(result.type, profile)
            program["operations"].append(dict(id=key, name=op.name, attributes=attributes, regions=regions,
                parent_id=None if op == idx.module.operation else idx.ids[op.parent],
                operands=[idx.values[v] for v in op.operands], results=[idx.values[v] for v in op.results]))
            for value in op.operands:
                owner = owner_operation(value)
                if isinstance(owner, ir.Operation):
                    program["dependencies"].append(dict(source=idx.ids[owner], target=key,
                                                        kind="ssa", value_id=idx.values[value]))
            if op.name == "builtin.module":
                parent = enclosing(op, "builtin.module")
                program["modules"].append(dict(id=key, parent_id=idx.ids.get(parent), attributes=attributes))
            if op.name == "func.func":
                declaration = not len(op.regions[0].blocks)
                role = "declaration" if declaration else "entry" if "pto.entry" in op.attributes else attributes.get("pto.kernel_kind", "helper")
                program["functions"].append(dict(id=key, module_id=idx.ids[enclosing(op, "builtin.module")],
                    symbol=attr(op, "sym_name"), logical_name=attr(op, "pto.ptodsl.logical_name"), role=role,
                    arguments=[] if declaration else [idx.values[a] for a in op.regions[0].blocks[0].arguments]))
            if op.name == "func.call":
                target = self.resolve(ir.FlatSymbolRefAttr(op.attributes["callee"]).value, op)
                args = list(target.regions[0].blocks[0].arguments)
                require(len(args) == len(op.operands), "SYMBOL", "call arity mismatch")
                require(all(a.type == v.type for a, v in zip(args, op.operands)), "SYMBOL", "call type mismatch")
                program["calls"].append(dict(operation_id=key, callee_id=idx.ids[target],
                    argument_map=[dict(source=idx.values[v], target=idx.values[a]) for v, a in zip(op.operands, args)]))
            if op.name == "scf.for":
                require(len(op.operands) == 3 and not len(op.results), "UNSUPPORTED_LOOP", "loop iter_args not yet described")
                constants = [owner_operation(v) for v in op.operands]
                trip = None
                if all(isinstance(o, ir.Operation) and o.name == "arith.constant" for o in constants):
                    lo, hi, step = [attr(o, "value") for o in constants]
                    require(step > 0, "UNSUPPORTED_LOOP", "positive step required")
                    trip = max(0, (hi - lo + step - 1) // step)
                program["loops"].append(dict(id=key, function_id=idx.ids[enclosing(op, "func.func")],
                    parent_loop_id=idx.ids.get(enclosing(op, "scf.for")),
                    lower=idx.values[op.operands[0]], upper=idx.values[op.operands[1]], step=idx.values[op.operands[2]],
                    induction=idx.values[op.regions[0].blocks[0].arguments[0]], trip_count=trip))
            if op.name in ("pto.alloc_tile", "pto.declare_tile", "pto.reserve_buffer"):
                func = enclosing(op, "func.func")
                core = attr(func, "pto.kernel_kind")
                instances = ["AIC0"] if core == "#pto.kernel_kind<cube>" else ["AIV0", "AIV1"] if core == "#pto.kernel_kind<vector>" else []
                require(bool(instances), "OWNERSHIP", "buffer must belong to a physical core function")
                ownership = {"pto.alloc_tile": "local", "pto.declare_tile": "borrowed_entry", "pto.reserve_buffer": "pipe_backing"}[op.name]
                info = (dict(allocation_bytes=attr(op, "size"), memory_space=attributes["location"])
                        if ownership == "pipe_backing" else tile_info(op.results[0].type, profile))
                if ownership == "local":
                    require(not len(op.operands), "PHYSICAL_ADDRESS", "explicit/dynamic allocation operands not supported")
                if ownership == "pipe_backing":
                    require(attributes.get("autoAlloc") is True, "PHYSICAL_ADDRESS", "explicit backing address unsupported")
                buffer_id = key + ".buffer"
                self.roots[op.results[0]] = buffer_id
                self.buffer_ops[buffer_id] = op
                program["buffers"].append(dict(id=buffer_id, operation_id=key, value_id=idx.values[op.results[0]],
                    function_id=idx.ids[func], ownership=ownership, multi_buffer_eligible=ownership == "local",
                    storage_root_id=None if ownership == "borrowed_entry" else buffer_id,
                    physical_core_templates=instances, alignment_bytes=profile["alignment_bytes"][info["memory_space"]],
                    storage_bytes=0 if ownership == "borrowed_entry" else info["allocation_bytes"], **info))
            if op.name in TILE_COMPUTE:
                require(not len(op.results), "UNSUPPORTED_OPERATION",
                        "structured storage effects currently require destination-style tile operations")
            if op.name in ALIASES:
                require(op.operands[0] in self.roots, "ALIAS", "unresolved tile alias")
                for result in op.results:
                    self.roots[result] = self.roots[op.operands[0]]
                    program["aliases"].append(dict(value_id=idx.values[result], buffer_id=self.roots[result],
                                                   operation_id=key))
        self.communication(program)
        self.effects(program)
        program["coverage"] = dict(structure="complete_for_declared_operations", latency=None,
            dependency_coverage="ssa_only; memory_loop_carried_and_async_completion_unknown",
            transaction_pairing="endpoints_only; dynamic_instance_mapping_unknown",
            gm_aliasing="requires_bindings", resource_accounting="logical_sizes_per_core_template; final_layout_unknown",
            schedule_mapping="requires_phase_aware_adapter", optimization_applied=False,
            apply_modes=["annotation_only_buffers"], certification="not_evaluated")
        return program

    def communication(self, program):
        idx = self.index
        backing = {(enclosing(op, "func.func"), attr(op, "name")): op for op in idx.operations if op.name == "pto.reserve_buffer"}
        endpoints, grouped = {}, {}
        for op in idx.operations:
            if op.name != "pto.initialize_l2l_pipe":
                continue
            source = owner_operation(op.operands[0])
            if source.name == "pto.import_reserved_buffer":
                target = self.resolve(ir.FlatSymbolRefAttr(source.attributes["peer_func"]).value, source, logical=True)
                source = backing.get((target, attr(source, "name")))
            require(source is not None and source.name == "pto.reserve_buffer", "PIPE", "unresolved backing")
            bid = self.roots[source.results[0]]
            row = grouped.setdefault(bid, dict(id=bid + ".pipe", backing_buffer_id=bid, endpoints=[]))
            end = dict(id=idx.ids[op], function_id=idx.ids[enclosing(op, "func.func")],
                       direction=attr(op, "dir_mask"), slot_count=attr(op, "slot_num"),
                       slot_size_bytes=attr(op, "slot_size"), nosplit=ir.BoolAttr(op.attributes["nosplit"]).value)
            if row["endpoints"]:
                require(all(end[k] == row["endpoints"][0][k] for k in ("direction", "slot_count", "slot_size_bytes", "nosplit")),
                        "PIPE", "endpoint configuration mismatch")
            row["endpoints"].append(end)
            endpoints[op.results[0]] = row
        for row in grouped.values():
            require(len(row["endpoints"]) == 2, "PIPE", "paired endpoints required")
        buffers = {b["id"]: b for b in program["buffers"]}
        for op in idx.operations:
            if op.name not in ("pto.tpush", "pto.tpop", "pto.tfree"):
                continue
            handle = op.operands[-1]
            require(handle in endpoints, "PIPE", "unknown handle")
            pipe = endpoints[handle]
            buffer_id = self.roots.get(op.operands[0]) if op.name != "pto.tfree" else None
            if op.name == "pto.tpop":
                require(buffer_id in buffers and buffers[buffer_id]["ownership"] == "borrowed_entry", "OWNERSHIP", "pop must bind borrowed entry")
                old = buffers[buffer_id]["storage_root_id"]
                require(old is None or old == pipe["backing_buffer_id"], "ALIAS", "borrowed entry has multiple roots")
                buffers[buffer_id]["storage_root_id"] = pipe["backing_buffer_id"]
            program["transactions"].append(dict(operation_id=idx.ids[op], kind=op.name.split(".")[-1],
                pipe_id=pipe["id"], endpoint_id=idx.ids[owner_operation(handle)], buffer_id=buffer_id,
                split=attr(op, "split"), loop_id=idx.ids.get(enclosing(op, "scf.for")), completion="unknown"))
        require(all(b["storage_root_id"] is not None for b in buffers.values()), "OWNERSHIP", "unbound borrowed entry")
        program["pipes"] = list(grouped.values())

    def effects(self, program):
        idx = self.index
        for op in idx.operations:
            if op.name not in TILE_COMPUTE:
                continue
            view = op.opview
            dst, tmp = getattr(view, "dst", None), getattr(view, "tmp", None)
            require(dst is not None, "UNSUPPORTED_OPERATION", "destination semantics missing: " + op.name)
            # Operand position is retained even when an in-place source equals dst.
            for position, value in enumerate(op.operands):
                if value not in self.roots and not (pto.TensorViewType.isinstance(value.type) or pto.PartitionTensorViewType.isinstance(value.type)):
                    continue
                effect = "read_write" if value == tmp or op.name == "pto.tmatmul.acc" and value == dst else "write" if position == len(op.operands) - 1 and value == dst else "read"
                program["memory_accesses"].append(dict(operation_id=idx.ids[op], operand=position,
                    value_id=idx.values[value], buffer_id=self.roots.get(value), effect=effect,
                    loop_id=idx.ids.get(enclosing(op, "scf.for")), completion="unknown"))


def export_structured(module, profile, output, bindings=None):
    from pto_costmodel.package import digest_text
    assembly = StructuredAssembly(module, profile)
    program = assembly.program()
    selected = dict(scalars={}, launch={}, alias_contract="unknown") if bindings is None else bindings
    validate_bindings(selected, program)
    target = dict(profile_version="pto.a5.budget.1", compiler_budget=profile, hardware_profile=None,
                  budget_source="PTOAS.PlanMemory", topology=dict(aic=1, aiv=2, scope="per_block"))
    files = {"program.json": encode(program), "runtime_bindings.json": encode(selected),
             "target_profile.json": encode(target), "canonical.pto": assembly.index.asm()}
    manifest = dict(protocol_version="2.0", kind="package", required_features=REQUIRED,
        semantics_version=SEMANTICS, checkpoint_version=1,
        producer=dict(name="ptoas", version=attr(module.operation, "pto.costmodel.compiler_version")),
        program_fingerprint=fingerprint(program), bindings_fingerprint=fingerprint(selected),
        target_fingerprint=fingerprint(target), files={k: digest_text(v) for k, v in files.items()})
    files["manifest.json"] = encode(manifest)
    publish(output, files)
    return manifest


def apply_structured(path, package, plan, output=None, current_input=None):
    from ptoas.costmodel import input_at_checkpoint
    validate_annotation_plan(plan, package)
    profile = package["target"]["compiler_budget"]
    validate_profile(profile)
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        text = read_text(Path(path) / "canonical.pto") if current_input is None else input_at_checkpoint(current_input)
        assembly = StructuredAssembly(ir.Module.parse(text), profile)
        require(assembly.program() == package["program"], "PACKAGE_INTEGRITY", "program and canonical PTO differ")
        validate_bindings(package["bindings"], package["program"])
        report = dict(status="validated" if output is None else "annotation_only", validation_scope="G1",
                      input_identity=package["identity"], plan_fingerprint=fingerprint(plan),
                      optimization_applied=False, g2="unknown", mappings=plan["buffers"])
        if output is not None:
            assembly.index.clean_annotations()
            for row in plan["buffers"]:
                op = assembly.buffer_ops[row["buffer_id"]]
                assembly.index.set_string(op, "pto.costmodel.buffer_id", row["buffer_id"])
                assembly.index.set_integer(op, "pto.pipeline.multi_buffer_count", row["count"])
            require(assembly.index.module.operation.verify(), "IR", "annotation produced invalid IR")
            publish(output, {"annotated.pto": assembly.index.asm(), "validation_report.json": encode(report),
                             "plan.json": encode(plan)})
    return report
