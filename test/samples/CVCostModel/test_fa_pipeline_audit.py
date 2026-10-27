# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import unittest
from types import SimpleNamespace
from fa_pipeline_audit import closure, event_id, verify_hazards, prove

class CompletionProofTest(unittest.TestCase):
    def test_async_read_is_not_completed_by_issue_order(self):
        accesses=[dict(event=0,object='aiv0:UB',access='read',begin=0,end=32),dict(event=1,object='aiv0:UB',access='write',begin=0,end=32)]
        with self.assertRaisesRegex(ValueError,'UNORDERED_ASYNC_ACCESS'): verify_hazards([set(),set()],accesses)
        self.assertEqual(verify_hazards([set(),{0}],accesses)[1],1)
    def test_disjoint_domains(self):
        self.assertEqual(verify_hazards([set(),set()],[dict(event=i,object='aiv'+str(i)+':UB',access='write',begin=0,end=32) for i in range(2)])[1],0)
    def test_readers_can_overlap(self):
        self.assertEqual(verify_hazards([set(),set()],[dict(event=i,object='aiv0:UB',access='read',begin=0,end=32) for i in range(2)])[1],0)
    def test_reuse_wait_cycle(self):
        with self.assertRaisesRegex(ValueError,'ASYNC_DEPENDENCY_CYCLE'): closure([{1},{0}])
    def test_cross_pipe_needs_real_completion_edge(self):
        pipe=lambda v: dict(enum='PipeAttr',value=v)
        def check(names,attributes):
            ops=[dict(id=str(i),name=n,attributes=a,regions=[]) for i,(n,a) in enumerate(zip(names,attributes))]
            events=[SimpleNamespace(id=i,subtask_id='aic',attrs=dict(source_op=n)) for i,n in enumerate(names)]
            accesses=[dict(event=0,core='aic',object='aic:L0C',access='write',begin=0,end=32),dict(event=names.index('pto.copy_matrix_cc_to_ub'),core='aic',object='aic:L0C',access='read',begin=0,end=32)]
            return prove(dict(functions=[dict(entry=True,kernel_kind=dict(value='cube'),operations=ops)]),events,accesses)
        with self.assertRaisesRegex(ValueError,'UNORDERED_ASYNC_ACCESS'):
            check(['pto.mad_raw','pto.copy_matrix_cc_to_ub','pto.barrier'],[{}, {},dict(pipe=pipe(6))])
        attrs=dict(src_pipe=pipe(2),dst_pipe=pipe(10),event_id=dict(debug='#pto.event<EVENT_ID0>'))
        result=check(['pto.mad_raw','pto.set_flag','pto.wait_flag','pto.copy_matrix_cc_to_ub','pto.barrier'],[{},attrs,attrs,{},dict(pipe=pipe(6))])
        self.assertEqual(result['status'],'pass')
        with self.assertRaisesRegex(ValueError,'UNPROVED_OUTPUT_COMPLETION'):
            check(['pto.mad_raw','pto.set_flag','pto.wait_flag','pto.copy_matrix_cc_to_ub'],[{},attrs,attrs,{}])

    def test_unsupported_event(self):
        self.assertEqual(event_id(dict(debug='#pto.event<EVENT_ID6>')),6)
        with self.assertRaisesRegex(ValueError,'unknown local event'):event_id(dict(debug='EVENT_ID6'))

if __name__=='__main__': unittest.main()
