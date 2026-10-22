# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compare native stage order with an independent iteration-distance oracle."""
import re
import subprocess
import sys
import unittest

compiler = sys.argv.pop(1)


def source(n, preload, slots, lower=0, step=1):
    return '''module attributes {pto.target_arch = "a5"} {
  func.func @crossing() {
    %lb = arith.constant LOWER : index
    %ub = arith.constant UPPER : index
    %step = arith.constant STEP : index
    %input = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %p = pto.alloc_tile {pto.costmodel.buffer_id = "P", pto.pipeline.multi_buffer_count = SLOTS : i64} : !pto.tile_buf<vec, 32x32xf32>
    %output = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    scf.for %i = %lb to %ub step %step {
      pto.tneg ins(%input : !pto.tile_buf<vec, 32x32xf32>) outs(%p : !pto.tile_buf<vec, 32x32xf32>)
      pto.tadd ins(%p, %p : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%output : !pto.tile_buf<vec, 32x32xf32>) {pto.cv.stage = "suffix"}
    } {pto.cv_preload_count = PRELOAD : i64}
    return
  }
}'''.replace('LOWER', str(lower)).replace('UPPER', str(lower+n*step)).replace('STEP', str(step)).replace('SLOTS', str(slots)).replace('PRELOAD', str(preload))


def run(text):
    return subprocess.run([compiler, '-pto-cv-apply-schedule'], input=text, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class NativeScheduleTest(unittest.TestCase):
    def test_boundary_order_and_minimum_slots(self):
        for n in (0, 1, 2, 4, 8):
            for preload in sorted({0, 1, 2, n, n+1}):
                with self.subTest(N=n, P=preload):
                    pe = min(preload, n)
                    slots = max(1, min(pe+1, n))
                    result = run(source(n, preload, slots))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    events = []
                    for line in result.stdout.splitlines():
                        if 'pto.tneg ' in line or 'pto.tadd ' in line:
                            iteration = int(re.search(r'pto.cv.iteration = (\d+)', line)[1])
                            events.append(('prefix' if 'pto.tneg ' in line else 'suffix', iteration))
                    expected = []
                    for t in range(n+pe):
                        if t < n: expected.append(('prefix', t))
                        if t >= pe: expected.append(('suffix', t-pe))
                    self.assertEqual(events, expected)
                    if n:
                        self.assertIn('pto.cv.required_slots = '+str(slots), result.stdout)
                    self.assertNotIn('scf.for ', result.stdout)

    def test_nonzero_lower_and_nonunit_step(self):
        result = run(source(4, 2, 3, lower=7, step=3))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('trip_count = 4', result.stdout)

    def test_insufficient_slots(self):
        result = run(source(4, 2, 2))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('INSUFFICIENT_SLOTS', result.stderr)

    def test_recurrence_is_not_independent(self):
        text = source(4, 2, 3).replace('pto.tneg ins(%input', 'pto.tneg ins(%p')
        result = run(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('UNSUPPORTED_RECURRENCE', result.stderr)


if __name__ == '__main__': unittest.main()
