# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Finite A5 FA completion proof. No predicted time establishes an edge.

The supported contract is per-pipe command completion order, explicit local
and cross-core set/wait generations, and barriers. Scalar issue order is NOT
used to order memory commands on different pipes. Unknown operations fail.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from fa_model_ab import read, write, sha, require

PIPES = {'func.call': 1, 'pto.mad_raw': 2,
    'pto.load_cbuf_to_ca': 3, 'pto.load_cbuf_to_cb': 3,
    'pto.copy_gm_to_cbuf_multi_nd2nz': 4, 'pto.copy_gm_to_cbuf_multi_dn2nz': 4,
    'pto.copy_ubuf_to_cbuf': 5, 'pto.copy_ubuf_to_gm': 5,
    'pto.copy_matrix_cc_to_ub': 10}
SCALAR = {'pto.get_subblock_idx','pto.get_block_idx','pto.get_block_num',
    'pto.castptr','pto.addptr','pto.alloc_tile','pto.tile_buf_addr',
    'pto.get_ctrl','pto.set_ctrl','pto.sbitset0','pto.sbitset1',
    'pto.set_mte2_nz_para','pto.set_loop3_para','pto.set_channel_para'}


def event_id(value):
    # This bounded reader accepts only the compiler's exact EventAttr spelling.
    if type(value) is int: return value
    match = re.fullmatch(r'#pto.event<EVENT_ID([0-9]+)>', value.get('debug', ''))
    require(match is not None, 'unknown local event identity')
    return int(match[1])


def closure(predecessors):
    successors = defaultdict(list); degrees = [len(p) for p in predecessors]
    for i, parents in enumerate(predecessors):
        for p in parents:
            require(type(p) is int and 0 <= p < len(degrees) and p != i, 'invalid completion edge')
            successors[p].append(i)
    ready = [i for i, n in enumerate(degrees) if n == 0]; ancestors = [0] * len(degrees); visited = 0
    while ready:
        i = ready.pop(); visited += 1
        for p in predecessors[i]: ancestors[i] |= ancestors[p] | (1 << p)
        for j in successors[i]:
            degrees[j] -= 1
            if not degrees[j]: ready.append(j)
    require(visited == len(degrees), 'ASYNC_DEPENDENCY_CYCLE')
    return ancestors


def verify_hazards(predecessors, accesses):
    ancestors = closure(predecessors)
    def before(a,b): return bool(ancestors[b] & (1 << a))
    grouped = defaultdict(list)
    for a in accesses: grouped[a['object']].append(a)
    hazards = 0
    for entries in grouped.values():
        for i,a in enumerate(entries):
            for b in entries[i+1:]:
                if a['event'] == b['event'] or a['access'] == b['access'] == 'read' or max(a['begin'],b['begin']) >= min(a['end'],b['end']): continue
                ordered = before(a['event'],b['event']) or before(b['event'],a['event'])
                if a.get('core') is not None and a['core'] == b.get('core'):
                    first,last = sorted((a['event'],b['event']))
                    ordered = before(first,last)
                require(ordered, 'UNORDERED_ASYNC_ACCESS: ' + str((a,b)))
                hazards += 1
    return ancestors, hazards


def prove(program, events, accesses):
    by_core = defaultdict(list)
    for event in events: by_core[event.subtask_id].append(event)
    predecessors = [set() for _ in events]; local = defaultdict(lambda: defaultdict(list)); cross = defaultdict(lambda: defaultdict(list))
    last_barriers = {}; counts = Counter()
    for f in program['functions']:
        if not f['entry']: continue
        nested = {i for o in f['operations'] for r in o['regions'] for b in r for i in b['operations']}
        ops = [o for o in f['operations'] if o['id'] not in nested and o['name'] not in ('arith.constant','func.return')]
        require(all(not o['regions'] for o in ops), 'unknown outer control')
        for lane in ((None,) if f['kernel_kind']['value'] == 'cube' else (0,1)):
            core = 'aic' if lane is None else 'aiv' + str(lane); stream = by_core[core]
            require(len(stream) == len(ops), 'compiler/event stream differs')
            tails = {}; gate = None
            for op,event in zip(ops,stream):
                i = event.id; name = op['name']; a = op['attributes']; counts[name] += 1
                require(event.attrs['source_op'] == ('prepared_vf' if name == 'func.call' else name), 'event opcode differs')
                def pipe(key):
                    value = a.get(key)
                    require(isinstance(value,dict) and value.get('enum') == 'PipeAttr' and type(value.get('value')) is int, 'unknown pipe')
                    return value['value']
                if name == 'pto.barrier' and pipe('pipe') == 6:
                    predecessors[i].update(tails.values()); tails = {p:i for p in range(11)}; gate = i
                    last_barriers[core] = i; continue
                if name in ('pto.set_flag','pto.wait_flag'):
                    key = (core,pipe('src_pipe'),pipe('dst_pipe'),event_id(a['event_id']))
                    setting = name == 'pto.set_flag'; p = key[1 if setting else 2]
                    local[key]['set' if setting else 'wait'].append(i)
                elif name in ('pto.set_intra_block','pto.wait_intra_block'):
                    p = pipe('pipe'); setting = name == 'pto.set_intra_block'
                    require(type(a['event_id']) is int, 'unknown cross-core flag')
                    key = a['event_id'] + (16*lane if lane is not None else 0)
                    cross[key]['set' if setting else 'wait'].append(i)
                elif name == 'pto.barrier': p = pipe('pipe')
                elif name in PIPES: p = PIPES[name]
                elif name.startswith('arith.') or name in SCALAR: p = 0
                else: raise ValueError('unknown completion operation: ' + name)
                require(0 <= p <= 10 and p != 6, 'unknown hardware pipe')
                if p in tails: predecessors[i].add(tails[p])
                if gate is not None: predecessors[i].add(gate)
                tails[p] = i
    pairs = []
    for kind,table in (('local',local),('crosscore',cross)):
        for key,generations in table.items():
            sets, waits = generations['set'], generations['wait']
            require(len(sets) == len(waits), 'unbalanced '+kind+' flag: '+str(key))
            for s,w in zip(sets,waits): predecessors[w].add(s)
            pairs.extend((kind,key,w,s) for w,s in zip(waits,sets[1:]))
    ancestors,hazards = verify_hazards(predecessors,accesses)
    for kind,key,w,s in pairs:
        require(ancestors[s] & (1 << w), 'UNPROVED_FLAG_REUSE: '+str((kind,key,w,s)))
    for a in accesses:
        # L0C-to-UB accesses are owned by AIC FIX, even when the target is AIV UB.
        core = events[a['event']].subtask_id
        require(core in last_barriers and ancestors[last_barriers[core]] & (1 << a['event']), 'UNPROVED_OUTPUT_COMPLETION')
    return dict(status='pass', authority='A5 per-pipe completion FIFO and final compiler synchronization',
        model_times_used_for_safety=False, scalar_issue_order_is_completion=False,
        events=len(events), physical_accesses=len(accesses), ordered_hazard_pairs=hazards,
        completion_edges=sum(map(len,predecessors)), local_flag_ids=len(local), crosscore_flag_ids=len(cross),
        reused_flag_generations=len(pairs), physical_streams=3, operation_counts=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--vfsim-root',required=True);p.add_argument('--worker-python',required=True);a=p.parse_args()
    folder=a.folder.resolve(); program=read(folder/'program.json')
    from core.frontend.adaptor.ptoas_prepared import NativeEvents
    native=NativeEvents(program); events=native.build()
    request=folder/'static-vf-request.json';write(request,dict(vfs=list(native.vfs.values())))
    command=[a.worker_python,str(Path(__file__).with_name('fa_vf_model.py')),str(request),str(folder/'static-vf-results'),'--vfsim-root',a.vfsim_root,'--footprints-only']
    with (folder/'static-vf-worker.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True, env=dict(os.environ,PYTHONPATH=a.vfsim_root),cwd=a.vfsim_root)
    vfs={v['key']:v for v in read(folder/'static-vf-results/results.json')['vfs']}; accesses=[]
    for event in events:
        for footprint in vfs.get(event.attrs.get('vf_key'),{}).get('memory_footprints',[]):
            region=native.Region('aiv'+str(event.attrs['aiv_lane'])+':UB',(footprint['begin'],),(footprint['end']-footprint['begin'],),native.Mem.UB,'uint8')
            (event.reads if footprint['access']=='read' else event.writes).append(region)
        for kind,regions in (('read',event.reads),('write',event.writes)):
            for r in regions: accesses.append(dict(event=event.id,core=event.subtask_id,access=kind,object=r.base,begin=r.offsets[0],end=r.offsets[0]+r.sizes[0]))
    result=prove(program,events,accesses)
    from native_ablation import validate_memory
    fb=read(folder/'compiler_feedback.json'); result['memory']=validate_memory(fb)
    require(fb['native_plan_memory_invoked'] and fb['product_sha256'] == sha(folder/'candidate.o'), 'unbound memory/object report')
    domains=defaultdict(list)
    # Detailed bounds are already enforced by the canonical VF bridge and the
    # independent range check below, including physical replica separation.
    for f in fb['functions']:
        for lane in range(f['physical_replicas']):
            core='aic' if f['physical_replicas']==1 else 'aiv'+str(lane)
            for b in f['allocations']:
                for address in b['addresses']: domains[(core,b['space'])].append((address,address+b['slot_bytes']))
            for b in f['reservations']: domains[(core,b['space'])].append((b['address'],b['address']+b['allocated_bytes']))
    spaces={'UB':'vec','L1':'mat','L0A':'left','L0B':'right','L0C':'acc'}
    for access in accesses:
        obj=access['object'];lo,hi=access['begin'],access['end'];require(0 <= lo < hi,'invalid physical range')
        if obj in ('Q','K','V','O'):
            require(hi <= dict(Q=32768,K=262144,V=262144,O=65536)[obj] and (access['access']=='read' or obj=='O'), 'invalid GM access')
        else:
            core,space=obj.split(':');require(any(begin<=lo and hi<=end for begin,end in domains[(core,spaces[space])]), 'unbound physical access')
    result['binding']=dict(program=sha(folder/'program.json'),feedback=sha(folder/'compiler_feedback.json'),object=sha(folder/'candidate.o'),accesses=__import__('hashlib').sha256(json.dumps(accesses,sort_keys=True).encode()).hexdigest())
    write(folder/'async-accesses.json',accesses);write(folder/'async_validation.json',result);print('ASYNC_G2_PASS',result['events'],result['ordered_hazard_pairs'])

if __name__=='__main__':main()
