"""Python exchange tests; native materialization is tested by lit."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('single_core_contract',Path(__file__).parents[1]/'ptoas/single_core.py')
sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)

class SingleCoreContractTest(unittest.TestCase):
    def setUp(self):
        self.request=dict(schema=sc.REQUEST,ir='IR',input_fingerprint=sc.fingerprint('IR'),
            independent_iterations=True,loop_id='L0',trip_count=4,buffer_ids={'vector::%x':'B0'})
        self.plan=dict(schema=sc.PLAN,input_fingerprint=self.request['input_fingerprint'],
            loop_id='L0',trip_count=4,schedule_semantics=sc.SCHEDULE,
            buffers=[dict(buffer_id='B0',count=3,slots=[0,1,0,2])],predicted_latency_us=1.0)

    def test_valid(self): sc.validate_plan(self.request,self.plan)

    def test_bad_slot_contracts(self):
        for slots in ([0], [0,1,2,3], [True,0,0,0], [0,-1,0,0]):
            p=copy.deepcopy(self.plan);p['buffers'][0]['slots']=slots
            with self.assertRaises(ValueError): sc.validate_plan(self.request,p)

    def test_bad_counts(self):
        for count in (True,0,17,1.5):
            p=copy.deepcopy(self.plan);p['buffers'][0]['count']=count
            with self.assertRaises(ValueError):sc.validate_plan(self.request,p)

    def test_stale_or_unknown_plan(self):
        for field,value in [('input_fingerprint','bad'),('loop_id','bad'),('schedule_semantics','preload'),('trip_count',3)]:
            p=copy.deepcopy(self.plan);p[field]=value
            with self.assertRaises(ValueError):sc.validate_plan(self.request,p)

    def test_missing_duplicate_and_unknown_buffers(self):
        for rows in ([], self.plan['buffers']*2, [dict(buffer_id='bad',count=1,slots=[0]*4)]):
            p=copy.deepcopy(self.plan);p['buffers']=rows
            with self.assertRaises(ValueError):sc.validate_plan(self.request,p)

    def test_reject_missing_independence(self):
        r=dict(self.request,independent_iterations=False)
        with self.assertRaises(ValueError):sc.validate_plan(r,self.plan)
        with self.assertRaises(ValueError):sc.prepare('IR')

    def test_reject_unknown_fields_and_nonfinite_latency(self):
        for field,value in [('addresses',{}),('predicted_latency_us',float('nan'))]:
            p=dict(self.plan);p[field]=value
            with self.assertRaises(ValueError):sc.validate_plan(self.request,p)
