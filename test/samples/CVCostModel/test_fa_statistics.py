# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import unittest
from summarize_fa_ab import paired,runtime_path
from pathlib import Path


class PairedFAStatisticsTest(unittest.TestCase):
    def test_gain_threshold_and_regression(self):
        self.assertTrue(paired([100]*20,[98]*20)['benefit_pass'])
        self.assertFalse(paired([100]*20,[98.01]*20)['benefit_pass'])
        self.assertFalse(paired([100]*20,[80]*19+[103])['benefit_pass'])
    def test_overlapping_ranges_can_have_paired_benefit(self):
        baseline=[100+10*(i%3) for i in range(20)]
        candidate=[.95*x for x in baseline]
        self.assertLess(min(baseline),max(candidate))
        self.assertTrue(paired(baseline,candidate)['benefit_pass'])
        self.assertEqual(paired(baseline,candidate),paired(baseline,candidate))
    def test_reject_missing_pair(self):
        with self.assertRaises(ValueError):paired([100]*19,[80]*20)
    def test_portable_runtime_preserves_original_json(self):
        row=dict(candidate_id='p1',runtime='/remote/experiment/work/fa/p1/run01')
        self.assertEqual(runtime_path(row,Path('/local/evidence')),Path('/local/evidence/work/fa/p1/run01'))
        self.assertEqual(row['runtime'],'/remote/experiment/work/fa/p1/run01')
        row['runtime']='/remote/experiment/work/fa/../p1/run01'
        with self.assertRaises(ValueError):runtime_path(row,Path('/local/evidence'))


if __name__=='__main__':unittest.main()
