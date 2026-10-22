# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import copy
import unittest
from native_ablation import progress, validate_memory


def streams(n, p):
    order=[]
    for t in range(n+p):
        if t<n:order.append(('prefix',t))
        if t>=p:order.append(('suffix',t-p))
    cube=[];vector=[]
    for phase,i in order:
        if phase=='prefix':
            cube.append(('push',0,i))
            vector.extend([('pop',0,i),('free',0,i),('push',1,i)])
        else:
            cube.extend([('pop',1,i),('free',1,i),('push',2,i)])
            vector.extend([('pop',2,i),('free',2,i)])
    return dict(AIC0=cube,AIV0=vector,AIV1=vector)

class NativeAblationTest(unittest.TestCase):
    def test_fifo(self):
        for n in (1,2,4,8):
            for p in range(n+1):self.assertEqual(progress(streams(n,p),dict.fromkeys(range(3),max(4,n)))['status'],'pass')
    def test_capacity_cycle(self):
        with self.assertRaisesRegex(ValueError,'FIFO_DEADLOCK'):progress(streams(8,4),dict.fromkeys(range(3),1))
    def test_missing_lane_release(self):
        data=streams(4,2);data['AIV1']=[x for x in data['AIV1'] if x!=('free',0,2)]
        with self.assertRaises(ValueError):progress(data,dict.fromkeys(range(3),4))
    def test_memory_domains_and_overlap(self):
        alloc=dict(buffer_id='P',slot_bytes=2048,space='vec',alignment=256,addresses=[0,2048,4096])
        vector=dict(function='v',physical_replicas=2,allocations=[alloc],reservations=[],compiler_budget_bytes=dict(vec=10000))
        cube=dict(function='c',physical_replicas=1,allocations=[],reservations=[],compiler_budget_bytes={})
        feedback=dict(functions=[cube,vector])
        self.assertEqual(len(validate_memory(feedback)['domains']),3)
        vector['allocations'].append(dict(alloc,buffer_id='other',addresses=[2048]))
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_REUSE'):validate_memory(feedback)
    def test_budget(self):
        vector=dict(function='v',physical_replicas=2,allocations=[],reservations=[dict(name='pipe',space='vec',address=0,allocated_bytes=1024)],compiler_budget_bytes=dict(vec=512))
        with self.assertRaisesRegex(ValueError,'overflow'):validate_memory(dict(functions=[dict(function='c',physical_replicas=1,allocations=[],reservations=[],compiler_budget_bytes={}),vector]))
if __name__=='__main__':unittest.main()
