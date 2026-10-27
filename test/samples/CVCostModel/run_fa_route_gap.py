# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Single-use same-device runs for the frozen route-gap ablation."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import shlex
import subprocess
import time
from fa_model_ab import read,write,sha,require,numpy_golden
from run_ab import execute
from summarize_runtime import duration

CASES=[('zero',12345),('one_v',12345),('one_qkv',12345)]+[('pattern',s) for s in (12345,20260916,20261022)]


def check(root):
    manifest=read(root/'acceptance_manifest.json')
    for name,digest in manifest['files'].items():require(sha(root/name)==digest,'frozen artifact changed: '+name)
    return manifest


def run(a):
    root=a.root.resolve();experiment=a.experiment.resolve();manifest=check(root)
    rows={r['candidate_id']:r for r in manifest['rows']};variants=list(rows);device=manifest['device']
    require(variants==['R','S','L','SL','P3','VB','VC','K','SK','T'] and device==1,'unexpected matrix/device')
    sequence=[];reference=None
    if a.mode=='g3':
        sequence=[(p,'pattern',12345,-1,'cold-first') for p in ('S','SL','L','P3','VB','VC','K','SK','R','T')]
        for repeat,order in enumerate((list(reversed(variants)),variants)):
            for mode,seed in CASES:
                sequence.extend((p,mode,seed,repeat,'g3') for p in order)
    else:
        g3=read(a.g3_report)
        require(g3['status']=='pass' and len(g3['rows'])==130 and g3['matrix_sha256']==sha(root/'acceptance_manifest.json') and g3['device']==device,'matching complete G3 required')
        reference={(r['candidate_id'],r['case'],r['seed']):r for r in g3['rows']}
        sequence=[(p,'pattern',12345,-1,'warmup') for p in variants]
        for block in range(20):
            at=block%len(variants);order=variants[at:]+variants[:at]
            if (block//len(variants))%2:order=list(reversed(order))
            sequence.extend((p,'pattern',(12345,20260916,20261022)[block%3],block,'profile') for p in order)
    output=experiment/'results'/('gap-'+a.mode+'.json');require(not output.exists(),'single-use experiment already has a report')
    report=dict(status='running',schema='pto.cv.fa_gap_runtime.v1',matrix_sha256=sha(root/'acceptance_manifest.json'),device=device,
        mode=a.mode,rows=[],harness_sha256=sha(__file__),environment='ACL_RUNTIME_AVAILABLE; no npu-smi health/ECC/temperature or non-cooperating occupancy telemetry; task-submit device lock')
    try:
        for variant,case,seed,block,stage in sequence:
            row=rows[variant];folder=root/row['folder']
            for name in ('runner','kernel.linked.o'):
                rel=str((folder/name).relative_to(root));require(sha(folder/name)==manifest['files'][rel],'bound runtime artifact changed')
            runtime=Path(subprocess.check_output([str(experiment.parents[1]/'bin/new_run_dir.sh'),str(experiment),'fa',variant,f'{stage}-{case}-{seed}-{block}'],text=True).strip())
            command=[str(folder/'runner'),str(device),case,str(seed),str(runtime/'output')]
            if stage=='profile':command=['msprof','--application='+shlex.join(command),'--output='+str(runtime/'profile'),'--aic-mode=task-based','--aic-metrics=PipeUtilization']
            write(runtime/'command.json',command);started=time.monotonic();status=execute(command,runtime)
            log=(runtime/'runtime.log').read_text()
            require(status==0 and 'NUMERICAL_PASS' in log and 'SOC=Ascend950DT_9592' in log and 'input_unchanged=1 guard_unchanged=1' in log,'RUNTIME_CORRECTNESS_FAILURE: '+str(runtime))
            saved=dict(candidate_id=variant,case=case,seed=seed,block=block,stage=stage,status='pass',runtime=str(runtime),symbol=row['symbol'],
                inputs={n:sha(runtime/'output'/n) for n in ('Q.f16','K.f16','V.f16')},output_sha256=sha(runtime/'output/O.f32'),
                numpy_max_abs_error=numpy_golden(runtime/'output',case),wall_seconds=time.monotonic()-started)
            if reference is not None:
                old=reference[(variant,case,seed)]
                require(saved['inputs']==old['inputs'] and saved['output_sha256']==old['output_sha256'],'input/output changed from G3')
            if stage=='profile':saved.update(duration(runtime,row['symbol'],device))
            write(runtime/'result.json',saved);report['rows'].append(saved);write(output,report)
            print(stage,variant,case,seed,block,saved.get('duration_us','PASS'),flush=True)
        outputs=defaultdict(set);inputs=defaultdict(set)
        for row in report['rows']:
            outputs[(row['candidate_id'],row['case'],row['seed'])].add(row['output_sha256'])
            inputs[(row['case'],row['seed'])].add(tuple(sorted(row['inputs'].items())))
        require(all(len(x)==1 for x in [*outputs.values(),*inputs.values()]),'unstable outputs or mismatched inputs')
        report['status']='pass';write(output,report)
    except Exception as error:
        report['status']='fail';report['error']=str(error);write(output,report);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['g3','performance']);p.add_argument('root',type=Path);p.add_argument('experiment',type=Path);p.add_argument('--g3-report',type=Path);run(p.parse_args())
