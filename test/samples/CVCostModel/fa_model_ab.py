# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Exact-input native FA, frozen offline recommendation, and same-board replay.

This is a bounded validation harness, not a general FA frontend. P0 is the
serial stage graph with the SAME native VF, storage and completion policy.
All paths in the exchange and final binding are relative to the matrix root.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def require(ok,message):
    if not ok: raise ValueError(message)


def invoke(command,log):
    command=list(map(str,command))
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    Path(log).write_text(json.dumps(command)+'\n'+result.stdout)
    require(result.returncode==0,'tool failed: '+str(log)+'\n'+result.stdout[-4096:])


def prepare(a):
    from ptoas import cv_vf as cv
    from fa_native_ab import import_fa,SOURCE_SHA
    from native_ablation import validate_memory
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume or a.preload is not None)
    require(not (root/'artifacts.json').exists(),'completed build is immutable')
    if not (root/'original.pto').exists():
        require(a.preload is None,'single-candidate workers require preinitialized original.pto')
        shutil.copyfile(a.source,root/'original.pto')
    require(sha(root/'original.pto')==SOURCE_SHA,'unexpected FA source')
    rows=[]
    for p in (range(4) if a.preload is None else (a.preload,)):
        folder=root/('p'+str(p));folder.mkdir(exist_ok=a.resume);symbol='fa_'+hashlib.sha256((str(root)+str(p)).encode()).hexdigest()[:16]
        if not (folder/'source.pto').exists():import_fa(root/'original.pto',folder/'source.pto',symbol)
        imported=read(folder/'source.import.json')
        require(imported['source_sha256']==SOURCE_SHA and imported['output_sha256']==sha(folder/'source.pto'),'partial import changed')
        binding=dict(shape=[128,1024,128],tile=[32,128,128],launch=dict(aic=1,aiv=2),
            alias_constraints=[dict(kind='pairwise_disjoint',arguments=['Q','K','V','O'])],
            parameter_bindings={symbol+core:['Q','K','V','O',128,1024] for core in ('_mix_aic','_mix_aiv')})
        target=dict(arch='a5',cann_version='9.2.0',soc='Ascend950DT_9592')
        if not (folder/'package').exists():cv.export_bundle(folder/'source.pto',binding,target,folder/'package',compiler=a.compiler)
        manifest,program=cv.load_bundle(folder/'package')
        require(manifest['source_sha256']==sha(folder/'source.pto') and manifest['producer']==cv.compiler_identity(a.compiler),'partial build/compiler changed')
        require(read(folder/'package/bindings.json')==binding and read(folder/'package/target.json')==target,'partial configuration changed')
        loops={l['id']:p for l in program['loops'] if not l['prepared_vf_internal']}
        analysis=read(folder/'analysis/schedule.json') if (folder/'analysis/schedule.json').exists() else cv.analyze_schedule(folder/'package',loops,folder/'analysis',compiler=a.compiler)
        require(analysis['input_fingerprint']==manifest['input_fingerprint'],'partial schedule input changed')
        plan=analysis['minimum_slot_plan'];plan['candidate_id']='p'+str(p)
        cv.annotate_plan(folder/'package',plan,folder/'annotated.pto')
        product=folder/'candidate.o';feedback=folder/'compiler_feedback.json'
        flags=cv._flags(target)+['--cv-apply-vf','--enable-insert-sync','--enable-vecscope-mem-bar',
            '--check-sync-ids','--check-sync-interference','--check-sync-self-coverage']
        command=[folder/'annotated.pto',*flags,'--cv-feedback='+str(feedback),'-o',product]
        if feedback.exists() and product.exists():
            import sys
            expected=[sys.executable,str(Path(a.compiler).resolve()),*map(str,command)]
            require(json.loads((folder/'compile.log').read_text().splitlines()[0])==expected,'partial compile command changed')
        else:cv._invoke(a.compiler,command,folder/'compile.log')
        fb=read(feedback)
        require(fb['product_sha256']==sha(product),'object/feedback mismatch')
        require(cv.fingerprint(fb['schedules'])==plan['schedule_fingerprint'],'schedule changed')
        require(fb['native_plan_memory_invoked'],'native PlanMemory missing')
        memory=validate_memory(fb)
        log=(folder/'compile.log').read_text()
        for f in fb['functions']:
            require(f['disjoint_allocation_policy'] and f['stage_completion_policy']=='drain_vf_and_before_crosscore_signal.v1','unknown memory/completion policy')
            coverage=[l for l in log.splitlines() if '[sync-gate:g1-self]' in l and ('func='+f['function']+' ') in l]
            require(len(coverage)==1 and 'uncovered=0 ' in coverage[0] and 'unmappable=0' in coverage[0], 'missing absolute sync coverage: '+f['function'])
        require(hashlib.sha256(fb['final_vpto_ir'].encode()).hexdigest()==fb['final_vpto_ir_sha256'],'final IR fingerprint mismatch')
        context,module=cv._parse(fb['final_vpto_ir'])
        with context:
            final=cv.describe(module)
        write(folder/'program.json',final);write(folder/'plan.json',plan)
        # Completion drains are checked on the SAME physical IR consumed by the
        # model. Pure address ops may be placed between a call and its drain.
        drains=0
        for f in final['functions']:
            if not f['entry']:continue
            nested={i for o in f['operations'] for reg in o['regions'] for b in reg for i in b['operations']}
            ops=[o for o in f['operations'] if o['id'] not in nested]
            meaningful=[o for o in ops if not o['name'].startswith('arith.') and o['name'] not in ('pto.castptr','pto.addptr','pto.alloc_tile','pto.tile_buf_addr')]
            # Structured PipeAttr uses the compiler's numeric enum. In this
            # pinned PTO semantic version PIPE_ALL is 6 (not a string token).
            def barrier(o):return o['name']=='pto.barrier' and o['attributes'].get('pipe',{})==dict(enum='PipeAttr',value=6)
            for i,o in enumerate(meaningful):
                if o['name']=='func.call':
                    require(i+1<len(meaningful) and barrier(meaningful[i+1]),'VF completion not proved');drains+=1
                if o['name']=='pto.set_intra_block' and 'pto.cv.token' in o['attributes']:
                    require(i>0 and barrier(meaningful[i-1]),'crosscore signal before completion');drains+=1
                if o['name']=='func.return':require(i>0 and barrier(meaningful[i-1]),'missing final drain')
        write(folder/'compiler_validation.json',dict(scope='exact M32 FA, disjoint storage, fixed FIFO',
            status='compiler_checks_passed; model event/bounds audit pending',memory=memory,drains=drains))
        # Link the same object that produced the physical IR and feedback.
        invoke([shutil.which('bisheng'),'--cce-fatobj-link','--cce-aicore-arch=dav-c310','-r',product,'-o',folder/'kernel.linked.o'],folder/'link.log')
        cann=Path(os.environ['ASCEND_HOME_PATH'])
        invoke([shutil.which('g++'),'-O2','-std=c++17','-DKERNEL_SYMBOL='+symbol,
            Path(__file__).with_name('fa_runtime_main.cpp'),folder/'kernel.linked.o','-I'+str(cann/'include'),
            '-L'+str(cann/'lib64'),'-Wl,-rpath,'+str(cann/'lib64'),'-lascendcl','-lruntime','-lprofapi','-ldl','-pthread','-o',folder/'runner'],folder/'launcher.log')
        object_symbols=subprocess.check_output(['strings',str(product)],text=True)
        require(symbol+'_mix_aic' in object_symbols,'device object symbol missing')
        rows.append(dict(candidate_id='p'+str(p),preload_count=p,symbol=symbol,
            program='p'+str(p)+'/program.json',program_sha256=sha(folder/'program.json'),
            feedback='p'+str(p)+'/compiler_feedback.json',feedback_sha256=sha(feedback),
            final_vpto_ir_sha256=fb['final_vpto_ir_sha256'],plan=plan))
        print('COMPILED',p,symbol,flush=True)
    if a.preload is not None:
        write(root/('p'+str(a.preload))/'candidate.json',rows[0]);return
    write(root/'candidates.json',rows)
    files={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}
    write(root/'artifacts.json',dict(schema='pto.cv.fa_native_artifacts.v1',files=files,compiler=cv.compiler_identity(a.compiler),
        original_sha256=SOURCE_SHA,rows=rows,synchronization='PTOAS only; explicit completion drains; no Bisheng auto-sync',
        storage='native automatic PlanMemory; no address reuse; fixed qk=1 p=3 pv=1, alpha=4, Q-state=4'))


def verify_artifacts(root):
    artifact=read(root/'artifacts.json')
    for name,digest in artifact['files'].items():require(sha(root/name)==digest,'artifact changed: '+name)
    return artifact


def freeze(a):
    root=a.root.resolve();artifact=verify_artifacts(root);result=read(a.result)
    require(result['request_sha256']==sha(a.request),'recommendation request changed')
    predictions={r['candidate_id']:r for r in result['predictions']}
    selection=result['extensions']['tilesim.selection.v1']
    common_layout=None
    for row in artifact['rows']:
        pred=predictions[row['candidate_id']]
        require(pred['plan']==row['plan'] and pred['program_sha256']==row['program_sha256'] and pred['feedback_sha256']==row['feedback_sha256'],'recommendation identity changed')
        folder=root/row['candidate_id'];fb=read(folder/'compiler_feedback.json');compiler=read(folder/'compiler_validation.json')
        layout=[]
        for f in fb['functions']:
            layout.append(dict(core='aic' if f['physical_replicas']==1 else 'aiv',
                allocations=sorted((b['buffer_id'].split(':source:',1)[-1],b['space'],b['addresses'],b['slot_bytes'],b['alignment']) for b in f['allocations']),
                reservations=f['reservations']))
        if common_layout is None:common_layout=layout
        require(layout==common_layout,'physical layouts differ; cannot isolate CV reordering')
        timeline=read(a.result.parent/(row['candidate_id']+'-timeline.json'));access=read(a.result.parent/(row['candidate_id']+'-accesses.json'))
        from fa_semantic_audit import check_execution_witness
        witness=check_execution_witness(read(folder/'program.json'),timeline,access)
        ancestors={}
        for event in sorted(timeline,key=lambda e:e['end_us']):
            # Zero-duration control events can have equal end times: use a
            # separate topological fixed-point instead of relying on time.
            ancestors[event['id']]=None
        pending={e['id']:e for e in timeline}
        while pending:
            ready=[e for e in pending.values() if all(ancestors[p] is not None for p in e['predecessors'])]
            require(ready,'event DAG has a cycle')
            for e in ready:
                value=0
                for pred in e['predecessors']:value|=(1<<pred)|ancestors[pred]
                ancestors[e['id']]=value;del pending[e['id']]
        def before(x,y):return bool(ancestors[y]&(1<<x))
        flags=defaultdict(lambda:defaultdict(list))
        for event in timeline:
            if event['flag'] is not None:
                flags[event['flag']]['set' if event['op']=='pto.set_intra_block' else 'wait'].append(event['id'])
        for flag,pairs in flags.items():
            require(len(pairs['set'])==len(pairs['wait']),'unbalanced flag generations')
            for old,next_set in zip(pairs['wait'],pairs['set'][1:]):require(before(old,next_set),'one-bit event overwritten: '+str(flag))
        domains={}
        for f in fb['functions']:
            core='aic' if f['physical_replicas']==1 else 'aiv'
            for lane in range(f['physical_replicas']):
                for b in f['allocations']:
                    for addr in b['addresses']:domains.setdefault((core if core=='aic' else core+str(lane),b['space']),[]).append((addr,addr+b['slot_bytes']))
                for b in f['reservations']:domains.setdefault((core if core=='aic' else core+str(lane),b['space']),[]).append((b['address'],b['address']+b['allocated_bytes']))
        spaces={'UB':'vec','L1':'mat','L0A':'left','L0B':'right','L0C':'acc'}
        grouped=defaultdict(list)
        for entry in access:
            obj=entry['object'];lo,hi=entry['begin'],entry['end']
            require(0<=lo<hi,'invalid memory footprint')
            if obj in ('Q','K','V','O'):
                extent={'Q':128*128*2,'K':1024*128*2,'V':1024*128*2,'O':128*128*4}[obj]
                require(hi<=extent,'GM footprint exceeds bound: '+obj)
                require(entry['access']=='read' or obj=='O','input written')
            else:
                core,space=obj.split(':')
                require(any(a<=lo and hi<=b for a,b in domains[(core,spaces[space])]),'unbound physical range '+str(entry))
            grouped[obj].append(entry)
        hazard_pairs=0
        for entries in grouped.values():
            for i,lhs in enumerate(entries):
                for rhs in entries[i+1:]:
                    if lhs['event']==rhs['event'] or (lhs['access']==rhs['access']=='read') or max(lhs['begin'],rhs['begin'])>=min(lhs['end'],rhs['end']):continue
                    require(before(lhs['event'],rhs['event']) or before(rhs['event'],lhs['event']),'unordered physical RAW/WAR/WAW: '+str((lhs,rhs)))
                    hazard_pairs+=1
        certificate=dict(schema='pto.cv.exact_fa_g2.v1',G2='pass',scope='SHA-bound fixed M32 FA; no address reuse',
            compiler=compiler,physical_accesses=len(access),ordered_hazard_pairs=hazard_pairs,events=len(timeline),
            execution_witness=witness,
            completion='compiler drains and absolute sync coverage; time predictions are NOT a safety proof',
            binding=dict(program=row['program_sha256'],feedback=row['feedback_sha256'],object=sha(folder/'candidate.o'),plan=row['plan']))
        write(folder/'validation_report.json',certificate)
    # Freeze external evidence in a portable copy. This directory did not exist
    # during model evaluation and is never used to tune the recommendation.
    destination=root/'model';shutil.copytree(a.result.parent,destination)
    shutil.copyfile(a.request,destination/'frozen-request.json')
    paths=['artifacts.json']+[str(f.relative_to(root)) for f in destination.rglob('*') if f.is_file()]
    paths += [r['candidate_id']+'/validation_report.json' for r in artifact['rows']]
    shutil.copyfile(__file__,root/'fa_model_ab.py');paths.append('fa_model_ab.py')
    paths += [str(f.relative_to(root)) for f in (root/'model-inputs').rglob('*') if f.is_file()]
    write(root/'ab_manifest.json',dict(schema='pto.cv.fa_ab_manifest.v1',model_selection=selection,
        files={p:sha(root/p) for p in paths},hardware_soc='Ascend950DT_9592',timing_profile='950PR_9599 proxy, user authorized',
        full_target_accuracy_certification=False,device=0,rows=artifact['rows']))
    print('FROZEN',selection['candidate_id'],selection['predicted_gain'],flush=True)


def check_frozen(root):
    manifest=read(root/'ab_manifest.json');verify_artifacts(root)
    for name,digest in manifest['files'].items():require(sha(root/name)==digest,'frozen evidence changed: '+name)
    return manifest


def numpy_golden(folder,mode):
    import numpy as np
    q=np.fromfile(folder/'Q.f16',dtype=np.float16).astype(np.float64).reshape(128,128)
    k=np.fromfile(folder/'K.f16',dtype=np.float16).astype(np.float64).reshape(1024,128)
    v=np.fromfile(folder/'V.f16',dtype=np.float16).astype(np.float64).reshape(1024,128)
    scores=np.einsum('id,jd->ij',q,k,optimize=False)/np.sqrt(128)
    scores-=scores.max(axis=1,keepdims=True);scores=np.exp(scores);scores/=scores.sum(axis=1,keepdims=True)
    expected=np.einsum('ij,jd->id',scores,v,optimize=False)
    actual=np.fromfile(folder/'O.f32',dtype=np.float32).reshape(128,128)
    tolerance=1e-4+1e-3*np.abs(expected) if mode=='pattern' else 1e-6
    require(bool(np.all(np.isfinite(actual))) and bool(np.all(np.abs(actual-expected)<=tolerance)),'independent NumPy golden failed')
    return float(np.abs(actual-expected).max())


def run(a):
    from run_ab import execute
    from summarize_runtime import duration
    root=a.root.resolve();manifest=check_frozen(root);experiment=a.experiment.resolve()
    selected=manifest['model_selection']['candidate_id'];variants=[r['candidate_id'] for r in manifest['rows']]
    require(len(set(r['symbol'] for r in manifest['rows']))==4,'kernel symbols not unique')
    rows_by_id={r['candidate_id']:r for r in manifest['rows']};cases=[('zero',12345),('one_v',12345),('one_qkv',12345)]+[('pattern',s) for s in (12345,20260916,20261022)]
    sequence=[]
    if a.mode=='g3':
        forward=[selected]+[p for p in variants if p!=selected]
        sequence.append((selected,'pattern',12345,-1,'cold-first'))
        for repeat,order in enumerate((forward,list(reversed(forward)))):
            for mode,seed in cases:
                for p in order:sequence.append((p,mode,seed,repeat,'g3'))
    else:
        accepted=read(a.g3_report)
        require(accepted['status']=='pass' and accepted['matrix_sha256']==sha(root/'ab_manifest.json') and accepted['device']==0,'matching G3 required')
        reference={(r['candidate_id'],r['case'],r['seed']):r for r in accepted['rows']}
        sequence=[(p,'pattern',12345,-1,'warmup') for p in variants]
        orders=((0,1,2,3),(1,0,3,2),(2,3,0,1),(3,2,1,0))
        for block in range(20):
            for ix in orders[block%4]:sequence.append((variants[ix],'pattern',(12345,20260916,20261022)[block%3],block,'profile'))
    report=dict(status='running',matrix_sha256=sha(root/'ab_manifest.json'),device=0,soc='Ascend950DT_9592',mode=a.mode,
        execution_harness_sha256=sha(__file__),profiler_symbol_policy='exact host launch symbol; one MIX_AIC task, 1 AIC + 2 AIV',
        environment='ACL_RUNTIME_AVAILABLE; npu-smi telemetry unavailable; task-submit exclusive queue',rows=[])
    output=experiment/'results'/('fa-'+a.mode+'.json');require(not output.exists(),'run report already exists')
    try:
        for p,mode,seed,repeat,stage in sequence:
            check_frozen(root)
            helper=experiment.parents[1]/'bin/new_run_dir.sh'
            allocated=subprocess.check_output([str(helper),str(experiment),'fa',p,f'{stage}-{mode}-{seed}-{repeat}'],text=True).strip()
            runtime=Path(allocated);row=rows_by_id[p];command=[str(root/p/'runner'),'0',mode,str(seed),str(runtime/'output')]
            if stage=='profile':command=['msprof','--application='+shlex.join(command),'--output='+str(runtime/'profile'),'--aic-mode=task-based','--aic-metrics=PipeUtilization']
            write(runtime/'command.json',command);started=time.monotonic();status=execute(command,runtime)
            log=(runtime/'runtime.log').read_text()
            require(status==0 and 'NUMERICAL_PASS' in log and 'SOC=Ascend950DT_9592' in log and 'input_unchanged=1 guard_unchanged=1' in log,'runtime/golden/guard gate failed: '+str(runtime))
            error=numpy_golden(runtime/'output',mode)
            saved=dict(candidate_id=p,case=mode,seed=seed,block=repeat,stage=stage,status='pass',runtime=str(runtime),
                output_sha256=sha(runtime/'output/O.f32'),inputs={n:sha(runtime/'output'/n) for n in ('Q.f16','K.f16','V.f16')},
                numpy_max_abs_error=error,wall_seconds=time.monotonic()-started,symbol=row['symbol'])
            if a.mode=='performance':
                previous=reference[(p,mode,seed)]
                require(saved['output_sha256']==previous['output_sha256'] and saved['inputs']==previous['inputs'],'performance output/input differs from G3')
            if stage=='profile':saved.update(duration(runtime,row['symbol'],0))
            write(runtime/'result.json',saved);report['rows'].append(saved);write(output,report)
            print(stage,p,mode,seed,repeat,saved.get('duration_us','PASS'),flush=True)
        # Same input and same output must be stable in both traversal orders.
        groups=defaultdict(set)
        inputs=defaultdict(set)
        for r in report['rows']:
            groups[(r['candidate_id'],r['case'],r['seed'])].add(r['output_sha256'])
            inputs[(r['case'],r['seed'])].add(tuple(sorted(r['inputs'].items())))
        require(all(len(v)==1 for v in groups.values()) and all(len(v)==1 for v in inputs.values()),'unstable output or mismatched input')
        report['status']='pass';write(output,report)
    except Exception as error:
        report.update(status='fail',error=str(error));write(output,report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prepare_parser=sub.add_parser('prepare');prepare_parser.add_argument('source',type=Path);prepare_parser.add_argument('root',type=Path);prepare_parser.add_argument('--compiler',required=True);prepare_parser.add_argument('--resume',action='store_true');prepare_parser.add_argument('--preload',type=int,choices=range(4))
    f=sub.add_parser('freeze');f.add_argument('root',type=Path);f.add_argument('request',type=Path);f.add_argument('result',type=Path)
    r=sub.add_parser('run');r.add_argument('root',type=Path);r.add_argument('experiment',type=Path);r.add_argument('--mode',choices=['g3','performance'],required=True);r.add_argument('--g3-report',type=Path)
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
