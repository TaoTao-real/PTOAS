# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import copy
import unittest
from fa_semantic_audit import check_execution_witness


def op(name,attributes=None):return dict(name=name,attributes=attributes or {},regions=[])


class ExecutionWitnessTest(unittest.TestCase):
    def setUp(self):
        self.program=dict(functions=[dict(id='c',entry=True,kernel_kind=dict(value='cube'),operations=[op('pto.set_intra_block',dict(event_id=0)),op('func.return')]),
            dict(id='v',entry=True,kernel_kind=dict(value='vector'),operations=[op('arith.addi'),op('func.return')])])
        self.timeline=[dict(id=0,op='pto.set_intra_block',core='aic',flag=0,predecessors=[]),
            dict(id=1,op='arith.addi',core='aiv0',flag=None,predecessors=[]),dict(id=2,op='arith.addi',core='aiv1',flag=None,predecessors=[])]
        self.program['functions'][0]['operations'][0]=op('arith.addi');self.timeline[0].update(op='arith.addi',flag=None)
    def test_no_times_are_needed(self):
        self.assertFalse(check_execution_witness(self.program,self.timeline,[])['model_times_used_for_safety'])
    def test_model_cannot_invent_order(self):
        bad=copy.deepcopy(self.timeline);bad[2]['predecessors']=[0]
        with self.assertRaisesRegex(ValueError,'unproved'):check_execution_witness(self.program,bad,[])
    def test_model_cannot_drop_event(self):
        with self.assertRaisesRegex(ValueError,'missing'):check_execution_witness(self.program,self.timeline[:-1],[])
    def test_model_cannot_omit_memory(self):
        self.program['functions'][0]['operations'][0]=op('pto.mad_raw');self.timeline[0]['op']='pto.mad_raw'
        with self.assertRaisesRegex(ValueError,'incomplete'):check_execution_witness(self.program,self.timeline,[])


if __name__=='__main__':unittest.main()
