# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate exact pairs and retain distinct synchronization/layout/VF deltas."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from fa_model_ab import read,write,require,sha,numpy_golden
from summarize_fa_ab import runtime_path
from summarize_runtime import duration


def summarize(report):
    require(report['status']=='pass' and report['mode']=='performance' and report['device']==1,'complete same-device performance required')
    rows=[r for r in report['rows'] if r['stage']=='profile']
    blocks=defaultdict(dict)
    full=['R','S','L','SL','P3','VB','VC','K','SK','T']
    variants=report.get('variants',full)
    require(variants in (full,[v for v in full if v!='VC']),'unexpected active matrix')
    require(len(rows)==20*len(variants),'expected complete paired profiles')
    for row in rows:
        b=row['block'];v=row['candidate_id'];require(type(b)is int and b in range(20) and v in variants and v not in blocks[b],'unknown/duplicate block member')
        require(row['case']=='pattern' and row['seed']==(12345,20260916,20261022)[b%3] and row['status']=='pass','wrong seed or status')
        blocks[b][v]=row
    for b in range(20):
        require(set(blocks[b])==set(variants),'missing block member')
        inputs=[row['inputs'] for row in blocks[b].values()];require(all(x==inputs[0] for x in inputs),'mismatched paired inputs')
        at=b%len(variants);order=variants[at:]+variants[:at]
        if (b//len(variants))%2:order=list(reversed(order))
        require([r['candidate_id'] for r in rows if r['block']==b]==order,'wrong rotation/order')
    values={v:[blocks[b][v]['duration_us'] for b in range(20)] for v in variants}
    summaries={v:dict(median_us=statistics.median(x),min_us=min(x),max_us=max(x),samples=len(x)) for v,x in values.items()}
    comparisons={}
    for base,new in [('R','S'),('R','L'),('S','SL'),('L','SL'),('R','P3'),('R','K'),('S','SK'),('K','SK'),('VB','VC'),('VC','R'),('VB','R'),('R','T'),('T','S'),('T','SL'),('T','K'),('T','SK')]:
        if base not in values or new not in values: continue
        # paired expects a list of records, not two independent distributions.
        pairs=[dict(A=values[base][b],B=values[new][b]) for b in range(20)]
        gains=[1-p['B']/p['A'] for p in pairs]
        import random
        rng=random.Random(20260916);boot=sorted(sum(rng.choices(gains,k=20))/20 for _ in range(10000))
        def percentile(q):
            at=q*(len(boot)-1);i=int(at);return boot[i]+(boot[min(i+1,len(boot)-1)]-boot[i])*(at-i)
        ci=[percentile(.025),percentile(.975)];mean=statistics.mean(gains);worst=max(-g for g in gains)
        comparisons[new+'/'+base]=dict(mean_gain=mean,ci95=ci,worst_regression=worst,pairs=pairs,
            benefit_pass=mean>=.02 and ci[0]>0 and worst<=.02)
    gap=statistics.mean(a-b for a,b in zip(values['R'],values['T']))
    saved=statistics.mean(a-b for a,b in zip(values['R'],values['S']))
    residual=statistics.mean(a-b for a,b in zip(values['S'],values['T']))
    attribution=dict(unit='us',native_minus_delivery_mean=gap,native_minus_sync_variant_mean=saved,
        sync_variant_minus_delivery_mean=residual,recovered_share_of_observed_gap=saved/gap if gap>0 else None,
        interpretation='paired absolute-time arithmetic identity; not additive gain percentages or a decomposition of untested pass interactions')
    return dict(status='pass',scope='fixed M32 FA same-device factor ablation; no G4 certification',variants=summaries,comparisons=comparisons,
        time_gap_attribution=attribution,bootstrap=dict(method='Python Random.choices with replacement; linear percentile',resamples=10000,seed=20260916,confidence_level=.95))

def replay(root,g3_path,perf_path,*,g3_runtime_root=None,runtime_root=None):
    root=Path(root);manifest=read(root/'acceptance_manifest.json');g3=read(g3_path);perf=read(perf_path)
    for name,digest in manifest['files'].items(): require(sha(root/name)==digest,'frozen artifact changed: '+name)
    for report in (g3,perf):
        require(report['status']=='pass' and report['device']==manifest['device'] and report['matrix_sha256']==sha(root/'acceptance_manifest.json'),'mismatched run/manifest')
    full=['R','S','L','SL','P3','VB','VC','K','SK','T'];variants=[r['candidate_id'] for r in manifest['rows']]
    exclusions=manifest.get('exclusions',{})
    require(set(exclusions) in (set(),{'VC'}) and variants==[v for v in full if v not in exclusions],'unexpected active matrix')
    cases=[('zero',12345),('one_v',12345),('one_qkv',12345)]+[('pattern',s) for s in (12345,20260916,20261022)]
    expected=[(p,'pattern',12345,-1,'cold-first') for p in ('S','SL','L','P3','VB','VC','K','SK','R','T') if p in variants]
    for repeat,order in enumerate((list(reversed(variants)),variants)):
        for case,seed in cases: expected.extend((p,case,seed,repeat,'g3') for p in order)
    require([(r['candidate_id'],r['case'],r['seed'],r['block'],r['stage']) for r in g3['rows']]==expected,'G3 order/seed/completeness mismatch')
    require(len(perf['rows'])==21*len(variants) and [r['candidate_id'] for r in perf['rows'][:len(variants)]]==variants and all(r['stage']=='warmup' and r['case']=='pattern' and r['seed']==12345 for r in perf['rows'][:len(variants)]),'warmup/completeness mismatch')
    symbols={r['candidate_id']:r['symbol'] for r in manifest['rows']};outputs={};checked=set();max_error=0
    for report,rebase in ((g3,g3_runtime_root),(perf,runtime_root)):
        for row in report['rows']:
            folder=runtime_path(row,rebase);require(read(folder/'result.json')==row and row['symbol']==symbols[row['candidate_id']],'runtime row/symbol changed')
            require(sha(folder/'output/O.f32')==row['output_sha256'] and all(sha(folder/'output'/name)==digest for name,digest in row['inputs'].items()),'runtime data changed')
            log=(folder/'runtime.log').read_text();require('NUMERICAL_PASS' in log and 'input_unchanged=1 guard_unchanged=1' in log and 'SOC=Ascend950DT_9592' in log,'runtime gate missing')
            key=(row['candidate_id'],row['case'],row['seed']);signature=(row['inputs'],row['output_sha256'])
            if key in outputs:require(outputs[key]==signature,'unstable or G3-inconsistent run')
            outputs[key]=signature
            if key not in checked:max_error=max(max_error,numpy_golden(folder/'output',row['case']));checked.add(key)
            if row['stage']=='profile':require(all(row[key]==value for key,value in duration(folder,row['symbol'],manifest['device']).items()),'raw profile changed')
    result=summarize(perf);result.update(G3='pass',G3_runs=len(g3['rows']),blocked_variants=exclusions,raw_profile_replay='pass',independent_numpy_cases=len(checked),max_numpy_abs_error=max_error,matrix_sha256=sha(root/'acceptance_manifest.json'),automatic_enablement=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','g3','performance','output'):p.add_argument(name,type=Path)
    p.add_argument('--runtime-root',type=Path);p.add_argument('--g3-runtime-root',type=Path);a=p.parse_args()
    result=replay(a.root,a.g3,a.performance,runtime_root=a.runtime_root,g3_runtime_root=a.g3_runtime_root);write(a.output,result)
    print({v:r['median_us'] for v,r in result['variants'].items()})
