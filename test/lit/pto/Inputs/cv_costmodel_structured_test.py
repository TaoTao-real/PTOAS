# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Attachment regression: assembly export and source-preserving G1 round trip."""
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_cli import parser, dispatch
from ptoas._cv_exchange import export_v2
from ptoas._cv_session import search_candidates
from ptoas._cv_structured import apply_structured, StructuredAssembly
from ptoas.costmodel import canonicalize
from pto_costmodel.package import read_package
from pto_costmodel.structured import REQUIRED, SEMANTICS, validate_annotation_plan
from pto_costmodel.reference import model_info
from pto_costmodel.wire import ContractError, encode, fingerprint, read_json

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "CVCostModel"


class StructuredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pto-structured-")
        cls.root = Path(cls.temp.name)
        cls.profile = read_json(SAMPLE / "a5_profile.json")
        cls.source = SAMPLE / "flash_attention_serial.pto"
        cls.path = cls.root / "package"
        export_v2(cls.source, cls.profile, cls.path)
        cls.package = read_package(cls.path)
        cls.plan = dict(protocol_version="2.0", kind="annotation_plan", required_features=REQUIRED,
                        identity=cls.package["identity"], model=model_info(),
                        buffers=[dict(buffer_id=b["id"], count=1 + i % 3)
                                 for i, b in enumerate(cls.package["program"]["buffers"]) if b["multi_buffer_eligible"]])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def output(self, suffix=""):
        return self.root / (self.id().split(".")[-1] + suffix)

    def reject(self, code, fn, *args):
        with self.assertRaises(ContractError) as error:
            fn(*args)
        self.assertEqual(error.exception.code, code, str(error.exception))

    def test_attachment_structure_and_ownership(self):
        p = self.package["program"]
        self.assertEqual(self.package["manifest"]["semantics_version"], SEMANTICS)
        self.assertEqual({k: len(p[k]) for k in ("modules", "functions", "calls", "loops", "pipes", "transactions")},
                         dict(modules=4, functions=5, calls=2, loops=4, pipes=3, transactions=14))
        self.assertEqual([f["role"] for f in p["functions"]], ["entry", "declaration", "declaration", "cube", "vector"])
        self.assertEqual(len([b for b in p["buffers"] if b["ownership"] == "local"]), 22)
        self.assertEqual(len([b for b in p["buffers"] if b["ownership"] == "pipe_backing"]), 3)
        self.assertEqual(len([b for b in p["buffers"] if b["ownership"] == "borrowed_entry"]), 5)
        for b in p["buffers"]:
            self.assertIsNotNone(b["storage_root_id"])
            if b["ownership"] == "borrowed_entry":
                self.assertEqual(b["storage_bytes"], 0)
        self.assertTrue(all(l["trip_count"] is None for l in p["loops"]))
        self.assertIsNone(p["coverage"]["latency"])
        text = (self.path / "canonical.pto").read_text()
        self.assertNotIn('"pto.pipe.', text)
        self.assertNotIn("pto.backend", text.splitlines()[-1])
        for call in p["calls"]:
            self.assertIn(call["callee_id"], {f["id"] for f in p["functions"] if f["role"] in ("cube", "vector")})
            self.assertTrue(call["argument_map"])

    def test_recursive_tfree_validation(self):
        source = self.output(".pto")
        text = (self.path / "canonical.pto").read_text()
        # Remove one release inside a child function, retaining otherwise valid MLIR.
        text = re.sub(r'^.*"pto.tfree".*\n', '', text, count=1, flags=re.M)
        source.write_text(text)
        self.reject("COMPILER", canonicalize, source)

    def test_scoped_roundtrip_and_idempotence(self):
        first, second = self.output("-first"), self.output("-second")
        report = apply_structured(self.path, self.package, self.plan, first)
        self.assertEqual(report["status"], "annotation_only")
        self.assertEqual(report["g2"], "unknown")
        self.assertFalse(report["optimization_applied"])
        apply_structured(self.path, self.package, self.plan, second, first / "annotated.pto")
        self.assertEqual((first / "annotated.pto").read_text(), (second / "annotated.pto").read_text())
        with ir.Context() as ctx:
            pto.register_dialect(ctx, load=True)
            assembly = StructuredAssembly(ir.Module.parse((first / "annotated.pto").read_text()), self.profile)
            self.assertEqual(assembly.program(), self.package["program"])
            for row in self.plan["buffers"]:
                op = assembly.buffer_ops[row["buffer_id"]]
                self.assertEqual(ir.IntegerAttr(op.attributes["pto.pipeline.multi_buffer_count"]).value, row["count"])
                self.assertEqual(ir.StringAttr(op.attributes["pto.costmodel.buffer_id"]).value, row["buffer_id"])

    def test_ssa_rename_and_symbolic_bindings(self):
        source = self.output(".pto")
        text = (self.path / "canonical.pto").read_text()
        source.write_text(re.sub(r'%([A-Za-z0-9_]+)', r'%renamed_\1', text))
        export_v2(source, self.profile, self.output("-ssa"))
        self.assertEqual(read_package(self.output("-ssa"))["identity"], self.package["identity"])
        entry = next(f for f in self.package["program"]["functions"] if f["role"] == "entry")
        bindings = dict(scalars={entry["arguments"][-2]:128, entry["arguments"][-1]:1024},
                        launch=dict(block_count=8), alias_contract="disjoint")
        export_v2(source, self.profile, self.output("-bound"), bindings)
        bound = read_package(self.output("-bound"))
        self.reject("STALE_PLAN", validate_annotation_plan, self.plan, bound)
        self.assertEqual(bound["identity"]["program_fingerprint"], self.package["identity"]["program_fingerprint"])

    def test_ambiguous_peer_definition_rejected(self):
        with ir.Context() as ctx:
            pto.register_dialect(ctx, load=True)
            module = ir.Module.parse((self.path / "canonical.pto").read_text())
            assembly = StructuredAssembly(module, self.profile)
            definitions = [f for f in assembly.funcs if len(f.regions[0].blocks)]
            # A second definition in another symbol scope is valid MLIR, but
            # cannot be guessed as the target of the wrapper's declaration.
            entry, cube, vector = definitions
            vector.attributes["sym_name"] = cube.attributes["sym_name"]
            self.reject("SYMBOL", assembly.program)

    def test_source_change_rejected(self):
        source = self.output(".pto")
        text = (self.path / "canonical.pto").read_text()
        source.write_text(text.replace('"pto.tmul"', '"pto.tadd"', 1))
        self.reject("PACKAGE_INTEGRITY", apply_structured, self.path, self.package, self.plan, self.output(), source)

    def test_plan_validation(self):
        variants = []
        for count in (0, -1, True, 257):
            plan = deepcopy(self.plan); plan["buffers"][0]["count"] = count
            variants.append(("RANGE", plan))
        plan = deepcopy(self.plan); plan["buffers"].append(plan["buffers"][0]); variants.append(("DUPLICATE_ID", plan))
        plan = deepcopy(self.plan); plan["buffers"].pop(); variants.append(("MISSING_BUFFER", plan))
        for key in ("invalid", next(b["id"] for b in self.package["program"]["buffers"] if b["ownership"] == "borrowed_entry")):
            plan = deepcopy(self.plan); plan["buffers"][0]["buffer_id"] = key; variants.append(("OWNERSHIP", plan))
        for field in ("address", "preload_count"):
            plan = deepcopy(self.plan); plan[field] = 2; variants.append(("SCHEMA", plan))
        plan = deepcopy(self.plan); plan["buffers"][0]["address"] = 0; variants.append(("SCHEMA", plan))
        plan = deepcopy(self.plan); plan["required_features"] += ["unknown_feature"]; variants.append(("UNSUPPORTED_FEATURE", plan))
        for code, plan in variants:
            with self.subTest(code=code, plan=plan):
                self.reject(code, validate_annotation_plan, plan, self.package)

    def test_integrity_even_after_manifest_rehash(self):
        p = deepcopy(self.package)
        p["program"]["loops"][0]["trip_count"] = 8
        p["identity"]["program_fingerprint"] = fingerprint(p["program"])
        plan = deepcopy(self.plan); plan["identity"] = p["identity"]
        self.reject("PACKAGE_INTEGRITY", apply_structured, self.path, p, plan)

    def test_static_adapter_and_compile_fail_explicitly(self):
        self.reject("UNSUPPORTED_FEATURE", search_candidates, self.package, [0, 1])
        plan_path = self.output(".json"); plan_path.write_text(encode(self.plan))
        for args in (["propose", str(self.path), "--adapter", "missing.json", "--output", str(self.output())],
                     ["apply", str(self.path), "--plan", str(plan_path), "--mode", "compile", "--output", str(self.output())]):
            self.reject("UNSUPPORTED_FEATURE", dispatch, parser().parse_args(args))
        result = dispatch(parser().parse_args(["validate", str(self.path), "--plan", str(plan_path)]))
        self.assertEqual(result["validation_scope"], "G1")

    def test_semantics_cannot_be_silently_downgraded(self):
        import shutil
        target = self.output(); shutil.copytree(self.path, target)
        manifest = read_json(target / "manifest.json")
        manifest["required_features"] = ["typed_operations", "buffer_ownership"]
        (target / "manifest.json").write_text(encode(manifest))
        self.reject("UNSUPPORTED_FEATURE", read_package, target)


if __name__ == "__main__":
    unittest.main()
