# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay exact-FA paired statistics; never certifies target-matched G4."""
import argparse
import math
from pathlib import Path
import random
import statistics
from fa_model_ab import check_frozen,read,require,sha,write


def percentile(values,p):
    x=sorted(values);index=(len(x)-1)*p;lo=math.floor(index);hi=math.ceil(index)
    return x[lo]+(index-lo)*(x[hi]-x[lo])


def paired(a,b):
    require(len(a)==len(b)==20 and all(math.isfinite(x) and x>0 for x in a+b),'20 valid paired samples required')
    gains=[1-y/x for x,y in zip(a,b)];rng=random.Random(20260916)
    means=[statistics.mean(gains[rng.randrange(20)] for _ in gains) for _ in range(10000)]
    lo,hi=percentile(means,.025),percentile(means,.975);mean=statistics.mean(gains);worst=max(y/x-1 for x,y in zip(a,b))
    return dict(mean_gain=mean,ci95=[lo,hi],worst_pair_regression=worst,pair_gains=gains,
        benefit_pass=mean>=.02-1e-12 and lo>0 and worst<=.02+1e-12)


def runtime_path(row,rebase):
    original=Path(row['runtime'])
    if rebase is None:return original
    parts=original.parts
    require('work' in parts,'unknown runtime path')
    suffix=parts[parts.index('work'):]
    require(len(suffix)==4 and suffix[:2]==('work','fa') and suffix[2]==row['candidate_id']
            and all(p not in ('.','..') for p in suffix),'invalid portable runtime path')
    return Path(rebase).joinpath(*suffix)


def summarize(root,g3_path,perf_path,*,runtime_root=None,g3_runtime_root=None):
    frozen=check_frozen(root);g3=read(g3_path);perf=read(perf_path)
    for report in (g3,perf):require(report['status']=='pass' and report['matrix_sha256']==sha(root/'ab_manifest.json') and report['device']==0,'invalid run report')
    require(len(g3['rows'])==49 and len(perf['rows'])==84,'incomplete G3/performance matrix')
    variants=[r['candidate_id'] for r in frozen['rows']]
    require([r['candidate_id'] for r in perf['rows'][:4]]==variants and
            all(r['stage']=='warmup' for r in perf['rows'][:4]),'warmup matrix mismatch')
    orders=((0,1,2,3),(1,0,3,2),(2,3,0,1),(3,2,1,0))
    for block in range(20):
        group=perf['rows'][4+4*block:8+4*block]
        require([r['candidate_id'] for r in group]==[variants[i] for i in orders[block%4]]
                and all(r['stage']=='profile' and r['block']==block and r['case']=='pattern'
                    and r['seed']==(12345,20260916,20261022)[block%3] for r in group),'paired order/seed mismatch')
    symbols={r['candidate_id']:r['symbol'] for r in frozen['rows']}
    for report,rebase in ((g3,g3_runtime_root),(perf,runtime_root)):
        for row in report['rows']:
            runtime=runtime_path(row,rebase)
            require(read(runtime/'result.json')==row and row['status']=='pass','runtime evidence changed')
            require(row['symbol']==symbols[row['candidate_id']],'runtime symbol binding changed')
            require(sha(runtime/'output/O.f32')==row['output_sha256'],'output changed')
            for n,digest in row['inputs'].items():require(sha(runtime/'output'/n)==digest,'input changed')
    by_candidate={r['candidate_id']:{} for r in frozen['rows']}
    for row in perf['rows']:
        runtime=runtime_path(row,runtime_root)
        if row['stage']!='profile':continue
        from summarize_runtime import duration
        for key,value in duration(runtime,row['symbol'],0).items():require(row[key]==value,'profiler evidence changed')
        p=row['candidate_id'];require(row['block'] not in by_candidate[p],'duplicate pair')
        by_candidate[p][row['block']]=row
    ordered={p:[samples[i]['duration_us'] for i in range(20)] for p,samples in by_candidate.items()}
    for i in range(20):
        require(len({tuple(sorted(rows[i]['inputs'].items())) for rows in by_candidate.values()})==1,'pair input mismatch')
        require(len({rows[i]['seed'] for rows in by_candidate.values()})==1,'pair seed mismatch')
    model=read(root/'model/result.json');predictions={r['candidate_id']:r for r in model['predictions']}
    rows=[]
    for p,values in ordered.items():
        measured=statistics.median(values);predicted=predictions[p]['latency_us']
        rows.append(dict(candidate_id=p,iteration_distance=predictions[p]['preload_count'],legacy_depth=predictions[p]['legacy_depth'],
            predicted_us=predicted,measured_median_us=measured,measured_min_us=min(values),measured_max_us=max(values),
            measured_mean_us=statistics.mean(values),prediction_error=(predicted-measured)/measured,
            absolute_prediction_error=abs(predicted-measured)/measured,paired_to_p0=paired(ordered['p0'],values)))
    selection=frozen['model_selection'];chosen=selection['candidate_id'];chosen_row=next(r for r in rows if r['candidate_id']==chosen)
    comparisons={p:paired(ordered[p],ordered[chosen]) for p in ordered if p!=chosen}
    result=dict(schema='pto.cv.fa_proxy_model_ab_result.v1',matrix_sha256=sha(root/'ab_manifest.json'),
        g3_sha256=sha(g3_path),performance_sha256=sha(perf_path),G3='PASS',G2='exact-input pass for all 4 candidates',
        samples=80,paired_blocks=20,bootstrap=dict(seed=20260916,resamples=10000,confidence=.95),
        selected=chosen,benefit=('BASELINE_RETAINED' if selection['action']=='baseline' else
            'AB_BENEFIT_PASS' if chosen_row['paired_to_p0']['benefit_pass'] else 'AB_BENEFIT_FAIL'),
        predicted_gain=selection['predicted_gain'],measured_mean_gain=chosen_row['paired_to_p0']['mean_gain'],
        gain_prediction_difference=selection['predicted_gain']-chosen_row['paired_to_p0']['mean_gain'],
        rows=rows,point_estimate_ranking=[r['candidate_id'] for r in sorted(rows,key=lambda r:r['measured_median_us'])],
        predicted_ranking=selection['ranked_candidates'],selected_vs_other= comparisons,
        prediction_error_within_10_percent=chosen_row['absolute_prediction_error']<=.1,
        G4='NOT_CERTIFIED: PR timing proxy on DT; no DT-matched certification policy/profile',
        limitations=['exact M32 FA only','same common VF and disjoint automatic PlanMemory; fixed storage; no multibuffer search',
            'P0 is serial under common static expansion, not original loop timing','1800 MHz is a proxy assumption; actual clock unavailable',
            'VSLDB timing and scalar/control approximation are explicit','npu-smi health/occupancy telemetry unavailable; exclusive task queue used'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','g3','performance','output'):p.add_argument(name,type=Path)
    p.add_argument('--runtime-root',type=Path,help='Optional portable performance experiment root')
    p.add_argument('--g3-runtime-root',type=Path,help='Optional portable correctness experiment root')
    a=p.parse_args();write(a.output,summarize(a.matrix,a.g3,a.performance,runtime_root=a.runtime_root,g3_runtime_root=a.g3_runtime_root))


if __name__=='__main__':main()
