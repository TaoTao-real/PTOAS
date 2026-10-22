# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Pure contract tests; the native transformation is exercised by lit."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cv_vf_contract", Path(__file__).parents[1] / "ptoas/cv_vf.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


class PreparedContractTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {"input_fingerprint": "input"}
        self.program = dict(loops=[dict(id="L", prepared_vf_internal=False)],
                            buffers=[dict(id="P", multi_buffer_eligible=True)])
        self.plan = dict(schema=cv.PLAN, input_fingerprint="input", candidate_id="candidate",
                         schedule_semantics=cv.SCHEDULE,
                         loops=[dict(loop_id="L", preload_count=2)],
                         buffers=[dict(buffer_id="P", count=3)])

    def validate(self, plan):
        cv.validate_plan(plan, self.manifest, self.program)

    def test_valid_plan(self): self.validate(self.plan)

    def test_optional_schedule_binding(self):
        self.plan["schedule_fingerprint"] = "0" * 64
        self.validate(self.plan)
        for invalid in (None, 1, "stale", "g" * 64):
            self.plan["schedule_fingerprint"] = invalid
            with self.assertRaises(ValueError): self.validate(self.plan)

    def test_configuration_types(self):
        for invalid in (None, True, [], 1, "plan"):
            with self.assertRaises(ValueError): self.validate(invalid)
        for rows in ({}, None, "buffers", [None], [True]):
            plan = copy.deepcopy(self.plan); plan["buffers"] = rows
            with self.assertRaises(ValueError): self.validate(plan)

    def test_bad_slot_counts(self):
        for count in (0, -1, 17, True, 1.5, "3"):
            with self.subTest(count=count):
                plan = copy.deepcopy(self.plan); plan["buffers"][0]["count"] = count
                with self.assertRaises(ValueError): self.validate(plan)

    def test_bad_preload_counts(self):
        for count in (-1, True, 1.5, "2", 2**31):
            with self.subTest(count=count):
                plan = copy.deepcopy(self.plan); plan["loops"][0]["preload_count"] = count
                with self.assertRaises(ValueError): self.validate(plan)

    def test_unknown_fields_and_address_injection(self):
        for target in ("plan", "loop", "buffer"):
            plan = copy.deepcopy(self.plan)
            row = plan if target == "plan" else plan["loops" if target == "loop" else "buffers"][0]
            row["physical_address"] = 0
            with self.assertRaises(ValueError): self.validate(plan)

    def test_unknown_duplicate_missing_id(self):
        for key, field in (("loops", "loop_id"), ("buffers", "buffer_id")):
            for mode in ("unknown", "duplicate", "missing"):
                plan = copy.deepcopy(self.plan)
                if mode == "unknown": plan[key][0][field] = "unknown"
                if mode == "duplicate": plan[key].append(copy.deepcopy(plan[key][0]))
                if mode == "missing": plan[key] = []
                with self.assertRaises(ValueError): self.validate(plan)

    def test_stale_or_unknown_semantics(self):
        for key in ("schema", "input_fingerprint", "schedule_semantics"):
            plan = copy.deepcopy(self.plan); plan[key] = "stale"
            with self.assertRaises(ValueError): self.validate(plan)

    def test_ineligible_buffer(self):
        self.program["buffers"][0]["multi_buffer_eligible"] = False
        with self.assertRaises(ValueError): self.validate(self.plan)

    def test_internal_vf_loops_are_not_tuned(self):
        self.program["loops"].append(dict(id="VF", prepared_vf_internal=True))
        self.validate(self.plan)

    def test_duplicate_json_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"count":1,"count":2}', '{"latency":NaN}', '{"latency":Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError): cv.read_json(path)

    def test_package_corruption_and_binding_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("canonical.pto", "program.json", "bindings.json", "target.json"):
                (root/name).write_text("{}")
            manifest = dict(schema=cv.SCHEMA, required_features=cv.FEATURES,
                            files={name:cv.sha(root/name) for name in
                                   ("canonical.pto", "program.json", "bindings.json", "target.json")})
            manifest["input_fingerprint"] = cv.fingerprint(manifest)
            cv._write(root/"manifest.json", manifest)
            cv._load_wire(root)
            (root/"bindings.json").write_text('{"N":8}')
            with self.assertRaises(ValueError): cv._load_wire(root)


if __name__ == "__main__": unittest.main()
