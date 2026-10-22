# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Exercise the compiler's prepared-VF contract on an iteration-distinct micro."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from ptoas import cv_vf as cv
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'samples/CVCostModel'))
from runtime_fixture import vf_source_text

compiler = shutil.which(sys.argv.pop(1))
assert compiler, 'PTOAS compiler is required'


class PreparedVFIntegrationTest(unittest.TestCase):
    def test_crossing_export_analyze_apply(self):
        with tempfile.TemporaryDirectory(prefix='cv-vf-native-') as directory:
            root = Path(directory)
            source = root / 'source.pto'
            source.write_text(vf_source_text(4, crossing=True))
            package = root / 'package'
            manifest = cv.export_bundle(source, {}, dict(arch='a5', cann_version='9.2.0'), package, compiler=compiler)
            loaded, program = cv.load_bundle(package)
            self.assertEqual(manifest, loaded)
            loops = [v for v in program['loops'] if not v['prepared_vf_internal']]
            self.assertEqual(len(loops), 2)
            self.assertEqual([v['trip_count'] for v in loops], [4, 4])
            self.assertEqual(sum(v['entry'] for v in program['functions']), 2)
            self.assertTrue(program['borrowed_entries'])
            self.assertTrue(all(v['ownership'] == 'borrowed_entry' for v in program['borrowed_entries']))
            analysis = cv.analyze_schedule(package, {v['id']: 2 for v in loops}, root/'analysis', compiler=compiler)
            self.assertEqual(analysis['status'], 'schedule_only')
            self.assertEqual(analysis['G2'], 'not_checked')
            plan = analysis['minimum_slot_plan']
            self.assertTrue(any(v['count'] == 3 for v in plan['buffers']))
            self.assertEqual(len(analysis['schedules']), 2)
            self.assertTrue(all(v['effective_preload'] == 2 for v in analysis['schedules']))
            self.assertTrue(all(e['source_operation'] != 'unidentified'
                                for v in analysis['schedules'] for e in v['events']))
            first, second = root/'first.pto', root/'second.pto'
            cv.annotate_plan(package, plan, first)
            cv.annotate_plan(package, plan, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            record = cv.compile_plan(package, plan, root/'compile', compiler=compiler, emit='llvm')
            self.assertEqual(record['g2_status'], 'evidence_incomplete')
            feedback = cv.read_json(root/'compile/compiler_feedback.json')
            self.assertTrue(feedback['native_plan_memory_invoked'])
            self.assertEqual(cv.fingerprint(feedback['schedules']), analysis['schedule_fingerprint'])
            self.assertEqual(feedback['product_sha256'], cv.sha(root/'compile/candidate.ll'))
            llvm_ir = (root/'compile/candidate.ll').read_text()
            self.assertIn('native_cv_vf_mix_aic', llvm_ir)
            self.assertIn('native_cv_vf_mix_aiv', llvm_ir)
            self.assertIn('hivm.annotations', llvm_ir)


if __name__ == '__main__': unittest.main()
