# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Offline prepared-VF exchange and explicit native candidate compilation.

This versioned extension does not change the v1/2.0 exchange protocols. Cost
models consume structured operations, not SSA names or physical addresses.
Unknown model costs cannot be promoted to an automatic recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

SCHEMA = "pto.cv.prepared_exchange.v1"
PLAN = "pto.cv.prepared_plan.v1"
SCHEDULE = "pto.cv.prefix_suffix.iteration_distance.v1"
FEATURES = ["prepared_vf.effects.v1", "native_memory_feedback.v1", SCHEDULE]


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_compiler_identity():
    from ptoas._loader import _find_dsos, _package_dir, ensure_core
    libraries = _find_dsos(_package_dir())
    if sum("libPTOASCompiler" in p.name for p in libraries) != 1:
        raise ValueError("native compiler identity unavailable")
    core = Path(ensure_core().__file__)
    return dict(native_components={p.name: sha(p) for p in [core, *libraries]},
                exchange_implementation_sha256=sha(__file__))


def compiler_identity(compiler):
    # Ask the selected wrapper. Hashing this process's loaded package would
    # silently bind a different compiler when the caller selects another tree.
    process = subprocess.run([sys.executable, str(Path(compiler).resolve()),
                              "costmodel", "vf", "identity"], check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    identity = json.loads(process.stdout, object_pairs_hook=_pairs)
    return dict(wrapper_sha256=sha(compiler), components=identity)


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _parse(text):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    context = ir.Context()
    pto.register_dialect(context, load=True)
    with context:
        module = ir.Module.parse(text)
        module.operation.verify()
    return context, module


def _text(op, name):
    from ptoas.mlir import ir
    return ir.StringAttr(op.attributes[name]).value


def _attr(attr):
    """Encode model-visible attributes without requiring MLIR text parsing."""
    from ptoas.mlir import ir
    if ir.StringAttr.isinstance(attr): return ir.StringAttr(attr).value
    if ir.BoolAttr.isinstance(attr): return ir.BoolAttr(attr).value
    if ir.IntegerAttr.isinstance(attr): return ir.IntegerAttr(attr).value
    if ir.FloatAttr.isinstance(attr):
        value = ir.FloatAttr(attr).value
        return value if math.isfinite(value) else {"unknown": "non-finite constant"}
    if ir.ArrayAttr.isinstance(attr): return [_attr(a) for a in ir.ArrayAttr(attr)]
    if ir.DenseI64ArrayAttr.isinstance(attr): return list(ir.DenseI64ArrayAttr(attr))
    if ir.DenseI32ArrayAttr.isinstance(attr): return list(ir.DenseI32ArrayAttr(attr))
    if ir.UnitAttr.isinstance(attr): return True
    if ir.FlatSymbolRefAttr.isinstance(attr): return {"symbol": ir.FlatSymbolRefAttr(attr).value}
    if ir.DictAttr.isinstance(attr): return {a.name: _attr(a.attr) for a in ir.DictAttr(attr)}
    if ir.TypeAttr.isinstance(attr): return _type(ir.TypeAttr(attr).value)
    known_enums = {"#pto.kernel_kind<cube>": "cube", "#pto.kernel_kind<vector>": "vector"}
    if str(attr) in known_enums: return {"enum": "kernel_kind", "value": known_enums[str(attr)]}
    from ptoas.mlir.dialects import pto
    for name in ("AddressSpaceAttr", "PipeAttr", "RoundModeAttr", "BLayoutAttr", "SLayoutAttr", "LayoutAttr"):
        cls = getattr(pto, name, None)
        if cls and cls.isinstance(attr): return {"enum": name, "value": cls(attr).value}
    # Unsupported semantics must remain visible. A model has to reject this
    # attribute unless it declares support; this is never a zero cost default.
    return {"unknown": "attribute semantics", "debug": str(attr)}


def _type(type_):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    if ir.IntegerType.isinstance(type_):
        integer = ir.IntegerType(type_)
        return dict(kind="integer", bits=integer.width,
                    signedness="unsigned" if integer.is_unsigned else "signed" if integer.is_signed else "signless")
    if ir.IndexType.isinstance(type_): return dict(kind="index")
    if ir.F16Type.isinstance(type_): return dict(kind="float", bits=16, format="ieee")
    if ir.BF16Type.isinstance(type_): return dict(kind="float", bits=16, format="bfloat")
    if ir.F32Type.isinstance(type_): return dict(kind="float", bits=32, format="ieee")
    for cls, kind in ((pto.TensorViewType, "tensor_view"),
                      (pto.PartitionTensorViewType, "partition_view")):
        if cls.isinstance(type_):
            view = cls(type_)
            return dict(kind=kind, element=_type(view.element_type), shape=list(view.shape))
    if str(type_) == "!pto.pipe": return dict(kind="pipe")
    if pto.PtrType.isinstance(type_):
        ptr = pto.PtrType(type_)
        return dict(kind="pointer", element=_type(ptr.element_type), memory_space=_attr(ptr.memory_space))
    if pto.TileBufType.isinstance(type_):
        tile = pto.TileBufType(type_)
        return dict(kind="tile", element=_type(tile.element_type), shape=list(tile.shape),
                    valid_shape=list(tile.valid_shape), memory_space=_attr(tile.memory_space),
                    blayout=tile.blayout_value, slayout=tile.slayout_value,
                    compact_mode=tile.compact_mode, fractal_bytes=tile.s_fractal_size)
    if pto.VRegType.isinstance(type_):
        reg = pto.VRegType(type_)
        return dict(kind="vector_register", lanes=reg.element_count, element=_type(reg.element_type))
    if pto.MaskType.isinstance(type_): return dict(kind="mask", granularity=pto.MaskType(type_).granularity)
    return dict(unknown="type semantics", debug=str(type_))


def describe(module):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    buffers, loops, functions, borrowed_entries = [], [], [], []
    seen = set()
    for function in _walk(module.operation):
        if function.name != "func.func" or not len(function.regions[0].blocks): continue
        name = _text(function, "sym_name")
        prepared = "pto.cv.prepared_vf" in function.attributes
        kind = _attr(function.attributes["pto.kernel_kind"]) if "pto.kernel_kind" in function.attributes else None
        physical_instances = (["AIV0", "AIV1"] if kind == {"enum": "kernel_kind", "value": "vector"}
                              else ["AIC0"] if kind == {"enum": "kernel_kind", "value": "cube"} else None)
        values = {}
        operations = []
        ordered_ops = list(_walk(function))
        operation_ids = {op: f"{name}:op{i}" for i, op in enumerate(ordered_ops)}
        def value_id(value):
            if value not in values: values[value] = f"{name}:v{len(values)}"
            return values[value]
        for block in function.regions[0].blocks:
            for argument in block.arguments: value_id(argument)
        for index, op in enumerate(ordered_ops):
            if op == function: continue
            if op.name == "func.func": raise ValueError("nested function definition")
            row = dict(id=f"{name}:op{index}", name=op.name,
                       operands=[value_id(v) for v in op.operands],
                       results=[dict(id=value_id(v), type=_type(v.type)) for v in op.results],
                       attributes={a.name: _attr(a.attr) for a in op.attributes})
            # The operation order plus explicit region children retain loops
            # and conditional structure; adapters must not flatten them.
            row["regions"] = [[dict(arguments=[dict(id=value_id(v), type=_type(v.type)) for v in block.arguments],
                                     operations=[operation_ids[child.operation] for child in block.operations])
                                for block in region.blocks] for region in op.regions]
            operations.append(row)
            if op.name == "pto.alloc_tile" and not prepared:
                identity = _text(op, "pto.costmodel.buffer_id")
                if identity in seen: raise ValueError("duplicate buffer identity")
                seen.add(identity)
                tile = pto.TileBufType(op.results[0].type)
                dtype = str(tile.element_type)
                widths = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4, "si32": 4,
                          "ui32": 4, "i8": 1, "si8": 1, "ui8": 1}
                shape = list(tile.shape)
                known = dtype in widths and all(d > 0 for d in shape) and tile.compact_mode == 0
                buffers.append(dict(id=identity, function=name, value=value_id(op.results[0]),
                    ownership="local", physical_instances=physical_instances,
                    vf_scratch="pto.cv.vf_scratch" in op.attributes,
                    shape=shape, valid_shape=list(tile.valid_shape), dtype=dtype,
                    memory_space=_attr(tile.memory_space),
                    bytes=math.prod(shape) * widths[dtype] if known else None,
                    multi_buffer_eligible=known))
            if op.name == "pto.declare_tile":
                borrowed_entries.append(dict(id=row["id"], function=name,
                    value=value_id(op.results[0]), ownership="borrowed_entry",
                    type=_type(op.results[0].type), multi_buffer_eligible=False))
            if op.name == "scf.for":
                identity = _text(op, "pto.costmodel.loop_id") if "pto.costmodel.loop_id" in op.attributes else row["id"]
                bounds = []
                for value in list(op.operands)[:3]:
                    owner = getattr(value.owner, "operation", value.owner)
                    bounds.append(ir.IntegerAttr(owner.attributes["value"]).value
                                  if isinstance(owner, ir.Operation) and owner.name == "arith.constant" else None)
                lo, hi, step = bounds
                count = max(0, (hi-lo+step-1)//step) if None not in bounds and step > 0 else None
                loops.append(dict(id=identity, function=name, lower=lo, upper=hi, step=step,
                                  trip_count=count, prepared_vf_internal=prepared))
        functions.append(dict(id=name, prepared_vf=prepared, kernel_kind=kind,
                              entry="pto.entry" in function.attributes,
                              physical_instances=physical_instances,
                              arguments=[dict(id=value_id(v), type=_type(v.type))
                                         for v in function.regions[0].blocks[0].arguments],
                              effects=_attr(function.attributes["pto.tileop.effects"])
                                      if "pto.tileop.effects" in function.attributes else None,
                              operations=operations))
    return dict(schema=SCHEMA, functions=functions, buffers=buffers, loops=loops,
                borrowed_entries=borrowed_entries,
                required_features=FEATURES, schedule_semantics=SCHEDULE,
                coverage={"costs": "unknown", "physical_layout": "not_assigned"})


def _invoke(compiler, args, log):
    command = [sys.executable, str(Path(compiler).resolve()), *map(str, args)]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(log).write_text(json.dumps(command) + "\n" + process.stdout)
    if process.returncode:
        raise RuntimeError(f"PTOAS failed with code {process.returncode}; see {log}\n" + process.stdout[-8192:])


def _flags(target):
    if target.get("arch") != "a5" or not isinstance(target.get("cann_version"), str):
        raise ValueError("prepared VF target requires A5 and an explicit CANN version")
    return ["--pto-arch=a5", "--pto-level=level2", "--pto-backend=vpto", "--enable-vmi",
            "--enable-op-fusion", "--vpto-scheduler=off", "--enable-vfsim-costmodel-optimization=false",
            "--cann-output-version=" + target["cann_version"]]


def export_bundle(source, bindings, target, directory, *, compiler):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source, canonical = Path(source), directory / "canonical.pto"
    _invoke(compiler, [source.resolve(), *_flags(target), "--cv-prepare-vf", "--emit-pto-ir",
                       "-o", canonical], directory / "prepare.log")
    context, module = _parse(canonical.read_text())
    with context:
        program = describe(module)
        preparation = _text(module.operation, "pto.cv.prepared_digest")
    _write(directory / "program.json", program)
    _write(directory / "bindings.json", bindings)
    _write(directory / "target.json", target)
    manifest = dict(schema=SCHEMA, required_features=FEATURES, source_sha256=sha(source),
                    producer=compiler_identity(compiler),
                    preparation_fingerprint=preparation,
                    files={name: sha(directory / name) for name in
                           ("canonical.pto", "program.json", "bindings.json", "target.json")})
    manifest["input_fingerprint"] = fingerprint(manifest)
    _write(directory / "manifest.json", manifest)
    return manifest


def _load_wire(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA: raise ValueError("unsupported exchange version")
    claimed = manifest.get("input_fingerprint")
    if fingerprint({k: v for k, v in manifest.items() if k != "input_fingerprint"}) != claimed:
        raise ValueError("invalid input fingerprint")
    if set(manifest.get("files", {})) != {"canonical.pto", "program.json", "bindings.json", "target.json"}:
        raise ValueError("invalid package file set")
    for name, digest in manifest["files"].items():
        if sha(directory / name) != digest: raise ValueError("package digest mismatch: " + name)
    if set(manifest.get("required_features", [])) - set(FEATURES):
        raise ValueError("unsupported required feature")
    return manifest, read_json(directory / "program.json")


def load_bundle(directory):
    manifest, program = _load_wire(directory)
    context, module = _parse((Path(directory) / "canonical.pto").read_text())
    with context:
        if describe(module) != program:
            raise ValueError("structured program and canonical IR differ")
        if _text(module.operation, "pto.cv.prepared_digest") != manifest.get("preparation_fingerprint"):
            raise ValueError("preparation fingerprint mismatch")
    return manifest, program


def validate_plan(plan, manifest, program):
    fields = {"schema", "input_fingerprint", "candidate_id", "schedule_semantics", "loops", "buffers"}
    if (not isinstance(plan, dict) or not fields <= set(plan)
            or set(plan) - fields - {"schedule_fingerprint"} or plan.get("schema") != PLAN):
        raise ValueError("unknown, missing or unsupported Plan fields")
    if plan["input_fingerprint"] != manifest["input_fingerprint"]:
        raise ValueError("stale Plan input identity")
    if plan["schedule_semantics"] != SCHEDULE: raise ValueError("unknown schedule semantics")
    if "schedule_fingerprint" in plan:
        digest = plan["schedule_fingerprint"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid schedule fingerprint")
    if not isinstance(plan["candidate_id"], str) or not plan["candidate_id"]:
        raise ValueError("invalid candidate identity")
    def check(rows, expected, id_key, value_key, limit):
        if not isinstance(rows, list): raise ValueError("configuration must be a list")
        found = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {id_key, value_key}:
                raise ValueError("unknown configuration fields")
            identity, value = row[id_key], row[value_key]
            if not isinstance(identity, str): raise ValueError("identity must be a string")
            if identity not in expected or identity in found: raise ValueError("unknown or duplicate ID")
            if type(value) is not int or not 0 <= value <= limit: raise ValueError("invalid count")
            if value_key == "count" and value == 0: raise ValueError("buffer count must be positive")
            found.add(identity)
        if found != expected: raise ValueError("configuration must cover every requested object")
    check(plan["loops"], {v["id"] for v in program["loops"] if not v["prepared_vf_internal"]},
          "loop_id", "preload_count", 2**31-1)
    check(plan["buffers"], {v["id"] for v in program["buffers"]}, "buffer_id", "count", 16)
    eligible = {v["id"]: v["multi_buffer_eligible"] for v in program["buffers"]}
    if any(v["count"] > 1 and not eligible[v["buffer_id"]] for v in plan["buffers"]):
        raise ValueError("ineligible buffer slot expansion")


def annotate_plan(directory, plan, destination):
    from ptoas.mlir import ir
    manifest, program = load_bundle(directory)
    validate_plan(plan, manifest, program)
    context, module = _parse((Path(directory) / "canonical.pto").read_text())
    loops = {v["loop_id"]: v["preload_count"] for v in plan["loops"]}
    buffers = {v["buffer_id"]: v["count"] for v in plan["buffers"]}
    with context:
        for op in _walk(module.operation):
            if op.name == "scf.for" and "pto.costmodel.loop_id" in op.attributes:
                identity = _text(op, "pto.costmodel.loop_id")
                if identity in loops: op.attributes["pto.cv_preload_count"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), loops[identity])
            if op.name == "pto.alloc_tile" and "pto.costmodel.buffer_id" in op.attributes:
                identity = _text(op, "pto.costmodel.buffer_id")
                if identity in buffers: op.attributes["pto.pipeline.multi_buffer_count"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), buffers[identity])
        module.operation.attributes["pto.costmodel.plan_id"] = ir.StringAttr.get(plan["candidate_id"])
        module.operation.verify()
        Path(destination).write_text(module.operation.get_asm(print_generic_op_form=True) + "\n")
    return dict(status="annotation_only", plan_fingerprint=fingerprint(plan), output_sha256=sha(destination))


def analyze_schedule(directory, preload, output_directory, *, compiler):
    """Ask the compiler for execution instances and minimum local slots.

    This does not assign addresses or certify G2. Models use the returned
    schedule instead of guessing a mapping from another policy's depth.
    """
    root, output = Path(directory), Path(output_directory)
    manifest, program = load_bundle(root)
    if compiler_identity(compiler) != manifest["producer"]:
        raise ValueError("compiler changed since export; export a new package")
    loops = [v for v in program["loops"] if not v["prepared_vf_internal"]]
    expected = {v["id"] for v in loops}
    if set(preload) != expected: raise ValueError("preload must cover every outer loop")
    plan = dict(schema=PLAN, input_fingerprint=manifest["input_fingerprint"],
                candidate_id="schedule-analysis", schedule_semantics=SCHEDULE,
                loops=[dict(loop_id=v["id"], preload_count=preload[v["id"]]) for v in loops],
                buffers=[dict(buffer_id=v["id"], count=1) for v in program["buffers"]])
    validate_plan(plan, manifest, program)
    output.mkdir(parents=True, exist_ok=False)
    annotate_plan(root, plan, output / "annotated.pto")
    target = read_json(root / "target.json")
    product = output / "scheduled.pto"
    _invoke(compiler, [output / "annotated.pto", *_flags(target), "--cv-apply-vf",
        "--cv-analyze-schedule", "--emit-pto-ir", "--enable-insert-sync",
        "--enable-vecscope-mem-bar", "-o", product], output / "analyze.log")
    context, module = _parse(product.read_text())
    with context:
        schedules = _attr(module.operation.attributes["pto.cv.schedules"]) if "pto.cv.schedules" in module.operation.attributes else []
        counts = {}
        for op in _walk(module.operation):
            if "pto.cv.required_slots" in op.attributes and "pto.costmodel.buffer_id" in op.attributes:
                identity = _text(op, "pto.costmodel.buffer_id")
                counts[identity] = _attr(op.attributes["pto.cv.required_slots"])
        for row in plan["buffers"]: row["count"] = counts.get(row["buffer_id"], 1)
    result = dict(schema=SCHEMA, status="schedule_only", G2="not_checked",
                  input_fingerprint=manifest["input_fingerprint"], schedules=schedules,
                  minimum_slot_plan=plan, product_sha256=sha(product))
    result["schedule_fingerprint"] = fingerprint(schedules)
    plan["schedule_fingerprint"] = result["schedule_fingerprint"]
    _write(output / "schedule.json", result)
    return result


def compile_plan(directory, plan, output_directory, *, compiler, emit="vpto"):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest, program = load_bundle(directory)
    if emit == "object" and not any(v["entry"] for v in program["functions"]):
        raise ValueError("device object requires an explicit PTO kernel entry")
    if manifest.get("producer") != compiler_identity(compiler):
        raise ValueError("compiler changed since export; export a new package")
    if "schedule_fingerprint" not in plan:
        raise ValueError("compile requires a compiler-analyzed schedule fingerprint")
    annotated = output_directory / "annotated.pto"
    annotate_plan(directory, plan, annotated)
    target = read_json(Path(directory) / "target.json")
    if emit not in ("vpto", "llvm", "object"): raise ValueError("unknown product kind")
    product = output_directory / {"vpto": "candidate.pto", "llvm": "candidate.ll", "object": "candidate.o"}[emit]
    feedback = output_directory / "compiler_feedback.json"
    args = [annotated, *_flags(target), "--cv-apply-vf", "--enable-insert-sync",
            "--enable-vecscope-mem-bar", "--cv-feedback=" + str(feedback), "-o", product]
    if emit != "object": args.append("--emit-vpto" if emit == "vpto" else "--emit-vpto-llvm-ir")
    _invoke(compiler, args, output_directory / "compile.log")
    report = read_json(feedback)
    if report.get("product_sha256") != sha(product): raise ValueError("feedback/product mismatch")
    if fingerprint(report.get("schedules")) != plan["schedule_fingerprint"]:
        raise ValueError("compiled schedule differs from the evaluated candidate")
    record = dict(schema=SCHEMA, input_fingerprint=manifest["input_fingerprint"],
                  candidate_id=plan["candidate_id"], plan_fingerprint=fingerprint(plan),
                  schedule_fingerprint=plan["schedule_fingerprint"], compiler=compiler_identity(compiler), annotated_sha256=sha(annotated),
                  product_sha256=sha(product), feedback_sha256=sha(feedback),
                  g2_status=report["g2_status"])
    _write(output_directory / "compilation.json", record)
    _write(output_directory / "plan.json", plan)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("identity")
    sub.add_parser("capabilities")
    export = sub.add_parser("export")
    for name in ("source", "bindings", "target", "directory"): export.add_argument(name)
    export.add_argument("--compiler", required=True)
    analyze = sub.add_parser("analyze")
    for name in ("directory", "preload", "output"): analyze.add_argument(name)
    analyze.add_argument("--compiler", required=True)
    for action in ("annotate", "compile"):
        command = sub.add_parser(action)
        for name in ("directory", "plan", "output"): command.add_argument(name)
        if action == "compile":
            command.add_argument("--compiler", required=True)
            command.add_argument("--emit", choices=("vpto", "llvm", "object"), default="vpto")
    args = parser.parse_args(argv)
    if args.action == "capabilities":
        print(json.dumps(dict(schema=SCHEMA, required_features=FEATURES,
            operations=["export", "analyze", "annotate", "compile"],
            target="a5", storage_mode="level2", static_single_axis=True,
            recurrence=False, annotation_default=True, native_plan_memory=True,
            G2="evidence_incomplete", automatic_enablement=False), indent=2))
        return 0
    if args.action == "identity":
        print(json.dumps(_local_compiler_identity(), sort_keys=True))
        return 0
    if args.action == "export": result = export_bundle(args.source, read_json(args.bindings), read_json(args.target), args.directory, compiler=args.compiler)
    elif args.action == "analyze": result = analyze_schedule(args.directory, read_json(args.preload), args.output, compiler=args.compiler)
    elif args.action == "annotate": result = annotate_plan(args.directory, read_json(args.plan), args.output)
    else: result = compile_plan(args.directory, read_json(args.plan), args.output, compiler=args.compiler, emit=args.emit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": main()
