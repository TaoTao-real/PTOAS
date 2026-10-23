# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Rebuild the bounded final-IR event witness without cost-model time values.

The model must not add fictional ordering to make an unsafe candidate appear
safe, or omit data operations from its access report. This check uses only
compiler-produced operation order, helper bodies and physical flag IDs.
"""
from collections import Counter,defaultdict


def require(ok,message):
    if not ok:raise ValueError(message)


def check_execution_witness(program,timeline,accesses):
    functions={f['id']:f for f in program['functions']};expected=[];flags=defaultdict(lambda:defaultdict(list))
    for f in program['functions']:
        if not f['entry']:continue
        require(f['kernel_kind']['value'] in ('cube','vector'),'unknown physical core')
        for lane in ((None,) if f['kernel_kind']['value']=='cube' else (0,1)):
            core='aic' if lane is None else 'aiv'+str(lane);previous=None
            for op in f['operations']:
                name=op['name'];require(not op['regions'],'outer control is outside this static witness subset')
                if name in ('arith.constant','func.return'):continue
                reads=writes=0;flag=None;pred=set() if previous is None else {previous}
                if name=='func.call':
                    callee=functions[op['attributes']['callee']['symbol']]
                    require(callee['prepared_vf'],'unknown non-VF call')
                    reads=sum(o['name'] in ('pto.vlds','pto.vsldb') for o in callee['operations'])
                    writes=sum(o['name']=='pto.vsts' for o in callee['operations'])
                    name='prepared_vf'
                elif name=='pto.mad_raw':reads,writes=2,1
                elif name in ('pto.copy_gm_to_cbuf_multi_nd2nz','pto.copy_gm_to_cbuf_multi_dn2nz',
                    'pto.load_cbuf_to_ca','pto.load_cbuf_to_cb','pto.copy_ubuf_to_cbuf','pto.copy_ubuf_to_gm'):
                    reads,writes=1,1
                elif name=='pto.copy_matrix_cc_to_ub':reads,writes=1,2
                elif name in ('pto.set_intra_block','pto.wait_intra_block'):
                    flag=op['attributes']['event_id']+(16*lane if lane is not None else 0)
                    flags[flag]['set' if name=='pto.set_intra_block' else 'wait'].append(len(expected))
                expected.append(dict(id=len(expected),op=name,core=core,flag=flag,predecessors=pred,reads=reads,writes=writes))
                previous=len(expected)-1
    require(len(timeline)==len(expected) and expected,'missing or extra execution events')
    for pairs in flags.values():
        require(len(pairs['set'])==len(pairs['wait']),'unbalanced compiler flag generations')
        for source,destination in zip(pairs['set'],pairs['wait']):expected[destination]['predecessors'].add(source)
    coverage=defaultdict(Counter)
    for access in accesses:
        require(type(access['event']) is int and 0<=access['event']<len(expected),'unknown memory event')
        require(access['access'] in ('read','write'),'unknown memory effect')
        require(access['core']==expected[access['event']]['core'],'physical access core mismatch')
        coverage[access['event']][access['access']]+=1
    for actual,wanted in zip(timeline,expected):
        require(all(actual[k]==wanted[k] for k in ('id','op','core','flag')),'event stream differs from compiler IR')
        require(set(actual['predecessors'])==wanted['predecessors'],'unproved model-added ordering or missing compiler dependency')
        require(coverage[wanted['id']]['read']==wanted['reads'] and coverage[wanted['id']]['write']==wanted['writes'],'incomplete physical access witness')
    return dict(status='pass',authority='compiler final IR and PTOAS-owned VF memory semantics',
        model_times_used_for_safety=False,events=len(expected),accesses=len(accesses),physical_streams=3,
        compiler_edges=sum(len(e['predecessors']) for e in expected),flag_ids=len(flags))
