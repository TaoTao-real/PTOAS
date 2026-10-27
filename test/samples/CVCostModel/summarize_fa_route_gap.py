# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate exact pairs and retain distinct synchronization/layout/VF deltas."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from fa_model_ab import read,write,require


def summarize(report):
    require(report['status']=='pass' and report['mode']=='performance' and report['device']==1,'complete same-device performance required')
    rows=[r for r in report['rows'] if r['stage']=='profile'];require(len(rows)==200,'expected 200 profiles')
    blocks=defaultdict(dict)
    variants=['R','S','L','SL','P3','VB','VC','K','SK','T']
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
    for base,new in [('R','S'),('R','L'),('S','SL'),('L','SL'),('R','P3'),('R','K'),('S','SK'),('K','SK'),('VB','VC'),('VC','R'),('R','T'),('T','S'),('T','SL'),('T','K'),('T','SK')]:
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
    return dict(status='pass',scope='fixed M32 FA same-device factor ablation; no G4 certification',variants=summaries,comparisons=comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('output',type=Path);a=p.parse_args();result=summarize(read(a.report));write(a.output,result);print({v:r['median_us'] for v,r in result['variants'].items()})
