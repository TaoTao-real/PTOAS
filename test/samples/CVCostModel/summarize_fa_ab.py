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


def summarize(root,g3_path,perf_path):
    frozen=check_frozen(root);g3=read(g3_path);perf=read(perf_path)
    for report in (g3,perf):require(report['status']=='pass' and report['matrix_sha256']==sha(root/'ab_manifest.json') and report['device']==0,'invalid run report')
    require(len(g3['rows'])==49 and len(perf['rows'])==84,'incomplete G3/performance matrix')
    by_candidate={r['candidate_id']:{} for r in frozen['rows']}
    for row in perf['rows']:
        runtime=Path(row['runtime']);require(read(runtime/'result.json')==row,'runtime evidence changed')
        require(sha(runtime/'output/O.f32')==row['output_sha256'],'output changed')
        for n,digest in row['inputs'].items():require(sha(runtime/'output'/n)==digest,'input changed')
        if row['stage']!='profile':continue
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
    a=p.parse_args();write(a.output,summarize(a.matrix,a.g3,a.performance))


if __name__=='__main__':main()
