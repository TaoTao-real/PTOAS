# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Structural evidence for the frozen factor matrix; no latency attribution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from fa_model_ab import read,write,require


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def analyze(root):
    rows=[];details={}
    for name in ('R','S','L','SL','P3','VB','VC','K','SK'):
        f=root/name
        if not f.exists():continue
        p=read(f/'program.json');fb=read(f/'compiler_feedback.json')
        prefix=next(fun['id'].split('_mix_ai')[0] for fun in p['functions'] if fun['entry'])
        def clean(x):
            if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.startswith(('pto.costmodel.','pto.cv.'))}
            if isinstance(x,list):return [clean(v) for v in x]
            return x.replace(prefix,'KERNEL') if isinstance(x,str) else x
        helpers=sorted((clean(fun) for fun in p['functions'] if fun['prepared_vf']),key=lambda x:x['id'])
        layout=[dict(core='aic' if fun['physical_replicas']==1 else 'aiv',allocations=clean(fun['allocations']),reservations=fun['reservations']) for fun in fb['functions']]
        counts=Counter(o['name'] for fun in p['functions'] if fun['entry'] for o in fun['operations'])
        body=Counter(o['name'] for fun in p['functions'] if fun['prepared_vf'] for o in fun['operations'])
        accesses=[{k:v for k,v in a.items() if k!='event'} for a in read(f/'async-accesses.json')]
        assembly={}
        for path in (f/'codegen').glob('*.preserved.s'):
            text=path.read_text();core='vector' if 'official.vector' in text else 'cube'
            opcodes=Counter()
            for line in text.splitlines():
                m=re.match(r'^\s+([A-Z][A-Z0-9_.]*)(?:\s|$)',line)
                if m:opcodes[m[1]]+=1
            assembly[core]=dict(opcodes=dict(opcodes),instructions=sum(opcodes.values()),bytes=path.stat().st_size,
                stack_annotations=len(re.findall('Reg -> Stack|Stack -> Reg',text)))
        row=dict(candidate_id=name,helper_count=len(helpers),vf_body_sha256=digest(helpers),layout_sha256=digest(layout),
            physical_access_sequence_sha256=digest(accesses),outer_op_counts=dict(counts),vf_op_counts=dict(body),assembly=assembly)
        rows.append(row);details[name]=row
    comparisons={}
    for a,b in [('R','S'),('R','L'),('S','SL'),('R','P3'),('VB','VC'),('VC','R'),('R','K'),('S','SK')]:
        if a not in details or b not in details:continue
        comparisons[b+'/'+a]={key:details[a][key]==details[b][key] for key in ('vf_body_sha256','layout_sha256','physical_access_sequence_sha256','vf_op_counts','outer_op_counts')}
    for a,b in [('R','S'),('R','K'),('S','SK')]:
        if a in details and b in details:
            require(all(comparisons[b+'/'+a][key] for key in ('vf_body_sha256','layout_sha256','physical_access_sequence_sha256')),'factor isolation failed: '+b+'/'+a)
    return dict(status='pass',variants=rows,comparisons=comparisons,assembly_count_semantics='static instructions; not dynamic traffic or a runtime gain')

if __name__=='__main__':
    root=Path(sys.argv[1]);write(Path(sys.argv[2]),analyze(root))
