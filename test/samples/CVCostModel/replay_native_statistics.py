# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay a downloaded native ablation experiment with Python standard library.
No compiler, MLIR, TileSim, NumPy, device, or absolute recorded path is needed.
"""
import argparse,csv,hashlib,json,math,random,statistics,struct
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def require(test,text):
    if not test:raise ValueError(text)
def stats(a,b):
    require(len(a)==len(b)==20,'twenty pairs required')
    gains=[1-y/x for x,y in zip(a,b)];rng=random.Random(20260916)
    means=sorted(statistics.mean(gains[rng.randrange(20)] for _ in range(20)) for _ in range(10000))
    def percentile(p):
        z=(len(means)-1)*p;lo,hi=math.floor(z),math.ceil(z)
        return means[lo]*(hi-z)+means[hi]*(z-lo) if hi!=lo else means[lo]
    alpha=(1-.95)/2
    return dict(pair_gains=gains,mean_gain=statistics.mean(gains),confidence_interval=[percentile(alpha),percentile(1-alpha)],worst_regression=max(y/x-1 for x,y in zip(a,b)))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.experiment;matrix=root/'inputs/matrix';m=json.loads((matrix/'manifest.json').read_text())
    for name,digest in m['files'].items():require(sha(matrix/name)==digest,'changed matrix '+name)
    report=json.loads((root/'results/performance.json').read_text());g3=json.loads((root/'results/correctness.json').read_text())
    require(report['status']==g3['status']=='pass' and len(g3['rows'])==24 and len(report['rows'])==84,'incomplete evidence')
    require(report['matrix_sha256']==g3['matrix_sha256']==sha(matrix/'manifest.json'),'stale matrix')
    require(report['device']==g3['device'] and report['soc']==g3['soc'],'mixed devices')
    require([r['variant'] for r in report['rows'][:4]]==['A','B','P0','M'] and all(not r['profile'] for r in report['rows'][:4]), 'wrong warmups')
    for row in g3['rows']+report['rows'][:4]:
        recorded=Path(row['runtime']);part=recorded.parts.index('work');runtime=root.joinpath(*recorded.parts[part:])
        require(json.loads((runtime/'result.json').read_text())==row,'changed correctness/warmup result')
        require(sha(runtime/'output.bin')==row['output_sha256'],'changed correctness/warmup output')
        require(list(struct.iter_unpack('<f',(runtime/'output.bin').read_bytes()))==list(struct.iter_unpack('<f',(runtime/'golden.bin').read_bytes())), 'output differs from golden')
        require('GOLDEN_PASS guards=pass input_unchanged=pass' in (runtime/'runtime.log').read_text(), 'missing runtime checks')
        for name,digest in row['input_sha256'].items():require(sha(runtime/name)==digest,'changed correctness/warmup input')
    for name in ('A','P0','M','B'):
        binary=root/'artifacts'/name;build=json.loads((binary/'build.json').read_text())
        require(sha(binary/'build.json')==g3['builds'][name] and sha(binary/'runner')==build['runner_sha256'], 'changed launcher')
        require(sha(matrix/name/'candidate/candidate.o')==build['object_sha256'], 'changed object')
    values={n:[] for n in ('A','P0','M','B')};orders=(('A','B','P0','M'),('B','A','M','P0'),('P0','M','A','B'),('M','P0','B','A'))
    for i,row in enumerate(report['rows'][4:]):
        block,place=divmod(i,4);require(row['block']==block and row['seed']==block%3 and row['variant']==orders[block%4][place],'wrong block order')
        recorded=Path(row['runtime']);part=recorded.parts.index('work');runtime=root.joinpath(*recorded.parts[part:])
        require(json.loads((runtime/'result.json').read_text())==row,'changed result')
        require(sha(runtime/'output.bin')==row['output_sha256'],'changed output')
        require(list(struct.iter_unpack('<f',(runtime/'output.bin').read_bytes()))==list(struct.iter_unpack('<f',(runtime/'golden.bin').read_bytes())), 'output differs from golden')
        require('GOLDEN_PASS guards=pass input_unchanged=pass' in (runtime/'runtime.log').read_text(), 'missing runtime checks')
        for name,digest in row['input_sha256'].items():require(sha(runtime/name)==digest,'changed input')
        paths=list((runtime/'profile').rglob('op_summary*.csv'));require(len(paths)==1,'missing/duplicate summary')
        rows=list(csv.DictReader(paths[0].open()));require(len(rows)==1,'unexpected task count');r=rows[0]
        require(r['Op Name']==row['symbol'] and r['Task Type']=='MIX_AIC' and r['Block Num']=='1' and r['Mix Block Num']=='2' and int(r['Device_id'])==report['device'],'wrong mixed task')
        number=float(r['Task Duration(us)']);require(math.isfinite(number) and number>0 and number==row['duration_us'] and sha(paths[0])==row['profiler_sha256'],'changed timing')
        values[row['variant']].append(number)
    results={key:stats(values[base],values[candidate]) for key,base,candidate in (('B/A','A','B'),('B/P0','P0','B'),('M/P0','P0','M'),('P0/A','A','P0'),('B/M','M','B'))}
    expected=json.loads((root/'results/summary.json').read_text())
    require(all(results[key]==data for key,data in expected['statistics'].items()),'statistics do not reproduce')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'statistics.json').write_text(json.dumps(dict(reproduced=True,statistics=results,source_report_sha256=sha(root/'results/summary.json')),indent=2)+'\n')
    print('REPLAY_PASS: 80 task samples, four-way order, inputs, outputs, profiler hashes and bootstrap statistics')
if __name__=='__main__':main()
