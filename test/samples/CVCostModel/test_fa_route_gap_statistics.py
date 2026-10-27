# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import copy
import unittest
from summarize_fa_route_gap import summarize


def fixture(variants=None):
    variants=variants or ['R','S','L','SL','P3','VB','VC','K','SK','T'];rows=[]
    for b in range(20):
        at=b%len(variants);order=variants[at:]+variants[:at]
        if (b//len(variants))%2:order=list(reversed(order))
        for v in order: rows.append(dict(candidate_id=v,block=b,case='pattern',seed=(12345,20260916,20261022)[b%3],status='pass',stage='profile',inputs=dict(Q='q',K='k',V='v'),duration_us=90 if v in ('S','SL') else 100))
    return dict(status='pass',mode='performance',device=1,variants=variants,rows=rows)

class GapStatisticsTest(unittest.TestCase):
    def test_pairs_and_repeatability(self):
        a=summarize(fixture());b=summarize(fixture())
        self.assertEqual(a,b);self.assertTrue(a['comparisons']['S/R']['benefit_pass'])
        self.assertAlmostEqual(a['comparisons']['S/R']['mean_gain'],.1)
    def test_missing_and_duplicate(self):
        a=fixture();a['rows'].pop()
        with self.assertRaises(ValueError):summarize(a)
        a=fixture();a['rows'][1]=copy.deepcopy(a['rows'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):summarize(a)
    def test_seed_and_order(self):
        a=fixture();a['rows'][0]['seed']=1
        with self.assertRaisesRegex(ValueError,'seed'):summarize(a)
        a=fixture();a['rows'][0],a['rows'][1]=a['rows'][1],a['rows'][0]
        with self.assertRaisesRegex(ValueError,'order'):summarize(a)
    def test_only_declared_correctness_exclusion_is_supported(self):
        variants=['R','S','L','SL','P3','VB','K','SK','T']
        result=summarize(fixture(variants))
        self.assertNotIn('R/VC',result['comparisons'])
        self.assertIn('R/VB',result['comparisons'])
        with self.assertRaisesRegex(ValueError,'active matrix'):summarize(fixture(variants[:-1]))

    def test_worst_pair_overrides_mean(self):
        a=fixture();next(r for r in a['rows'] if r['candidate_id']=='S')['duration_us']=103
        self.assertFalse(summarize(a)['comparisons']['S/R']['benefit_pass'])

if __name__=='__main__':unittest.main()
