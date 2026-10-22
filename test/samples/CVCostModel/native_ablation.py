# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Bounded native-VF A/P0/M/B validation. No model selection or G4 claim.

The G2 certificate accepts only the generated crossing FP32 fixture. It rejects
physical overlap instead of trying to prove general address reuse. Completion
uses compiler sync gates plus explicit drains before borrowed-entry release.
"""
import argparse
from collections import defaultdict, deque
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from ptoas import cv_vf as cv
from runtime_fixture import input_arrays, vf_source_text
from run_ab import execute
from summarize_runtime import duration

ORDER = (("A", "B", "P0", "M"), ("B", "A", "M", "P0"),
         ("P0", "M", "A", "B"), ("M", "P0", "B", "A"))
DRAIN = 'drain_before_free_and_return.v1'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def require(ok, message):
    if not ok: raise ValueError(message)


def progress(streams, capacity):
    """Prove DAG progress including per-lane token ownership and slot release."""
    edges = set()
    pushes, pops, frees = {}, {}, {}
    nodes = set()
    for lane, events in streams.items():
        previous = None
        for pos, (action, channel, token) in enumerate(events):
            node = (lane, pos)
            nodes.add(node)
            if previous is not None: edges.add((previous, node))
            previous = node
            key = (channel, token)
            table = {'push': pushes, 'pop': pops, 'free': frees}[action]
            table.setdefault(key, []).append(node)
    require(set(pushes) == set(pops) == set(frees), 'unpaired FIFO transaction')
    for key, producers in pushes.items():
        channel, token = key
        expected = 1 if channel == 1 else 2
        require(len(producers) == (2 if channel == 1 else 1)
                and len(pops[key]) == expected and len(frees[key]) == expected,
                'wrong AIC/AIV transaction multiplicity')
        for producer in producers:
            for consumer in pops[key]: edges.add((producer, consumer))
            if token >= capacity[channel]:
                for release in frees[(channel, token-capacity[channel])]: edges.add((release, producer))
        for pop in pops[key]:
            matching = [free for free in frees[key] if free[0] == pop[0]]
            require(len(matching) == 1, 'missing lane release')
            edges.add((pop, matching[0]))
    incoming = {node: 0 for node in nodes}; outgoing = defaultdict(list)
    for src, dst in edges:
        incoming[dst] += 1; outgoing[src].append(dst)
    ready = deque(n for n in nodes if not incoming[n]); visited = 0
    while ready:
        node = ready.popleft(); visited += 1
        for dst in outgoing[node]:
            incoming[dst] -= 1
            if not incoming[dst]: ready.append(dst)
    require(visited == len(nodes), 'FIFO_DEADLOCK: communication and storage waits form a cycle')
    return dict(status='pass', nodes=len(nodes), edges=len(edges), physical_streams=3)


def validate_memory(feedback):
    domains = {}
    require(len(feedback['functions']) == 2, 'expected paired physical functions')
    for function in feedback['functions']:
        used = defaultdict(list)
        for row in function['allocations']:
            require(row['slot_bytes'] > 0 and row['alignment'] > 0, 'unknown allocation size')
            for slot, address in enumerate(row['addresses']):
                require(address >= 0 and address % row['alignment'] == 0, 'unaligned allocation')
                used[row['space']].append((address, address+row['slot_bytes'], row['buffer_id'], slot))
        for row in function['reservations']:
            require(row['allocated_bytes'] > 0, 'unknown backing size')
            used[row['space']].append((row['address'], row['address']+row['allocated_bytes'], row['name'], 0))
        for space, intervals in used.items():
            ordered = sorted(intervals)
            require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])),
                    'UNSUPPORTED_REUSE: bounded proof requires disjoint physical allocations')
            require(max(v[1] for v in ordered) <= function['compiler_budget_bytes'][space], 'memory overflow')
        for replica in range(function['physical_replicas']):
            domains[function['function']+':'+str(replica)] = dict(used)
    require(len(domains) == 3, 'physical AIV domains were merged')
    return dict(status='pass', domains=domains, reuse='not_assumed')


def audit(package, candidate, analysis, count, serial):
    from ptoas.mlir import ir
    manifest, program = cv.load_bundle(package)
    symbol = next(f['id'][:-8] for f in program['functions'] if f['id'].endswith('_mix_aic'))
    expected = hashlib.sha256(fixture(count, symbol).encode()).hexdigest()
    require(manifest['source_sha256'] == expected, 'unsupported source; fixed micro only')
    bindings = cv.read_json(package/'bindings.json')
    require(bindings.get('launch') == dict(aic=1,aiv=2) and bindings.get('iterations') == count
            and bindings.get('alias_constraints') == [dict(kind='pairwise_disjoint', arguments=[symbol+'_mix_aic:arg'+str(i) for i in range(4)])], 'unknown argument bounds or alias contract')
    record = cv.read_json(candidate/'compilation.json')
    feedback = cv.read_json(candidate/'compiler_feedback.json')
    require(feedback['product_sha256'] == cv.sha(candidate/'candidate.o') == record['product_sha256'], 'product mismatch')
    require(feedback['completion_policy'] == DRAIN, 'unknown completion policy')
    require(hashlib.sha256(feedback['planned_ir'].encode()).hexdigest() == feedback['planned_ir_sha256'], 'IR mismatch')
    require(feedback['native_plan_memory_invoked'], 'memory pass not invoked')
    memory = validate_memory(feedback)
    # The compiler gates fail compilation on an uncovered dependency, invalid id,
    # or event interference. Require nonempty outer-function reports, not just a
    # zero-dependency helper success message.
    log = (candidate/'compile.log').read_text()
    for function in feedback['functions']:
        name = function['function']
        rows = [line for line in log.splitlines() if ('func='+name+' ') in line and '[sync-gate:g1-self]' in line]
        require(len(rows) == 1 and 'uncovered=0 ' in rows[0] and 'unmappable=0' in rows[0]
                and re.search(r'deps=[1-9][0-9]*', rows[0]), 'missing nonempty sync coverage: '+name)
    context, module = cv._parse(feedback['planned_ir'])
    drains = 0
    with context:
        for func in cv._walk(module.operation):
            if func.name != 'func.func' or 'pto.cv.prepared_vf' in func.attributes: continue
            for parent in cv._walk(func):
                for region in parent.regions:
                    for block in region.blocks:
                        siblings = [v.operation for v in block.operations]
                        for position, op in enumerate(siblings):
                            if op.name in ('pto.tfree', 'func.return') or (op.name == 'pto.bridge.call' and 'pipe_free' in str(op.attributes['entry'])):
                                previous = siblings[position-1] if position else None
                                require(previous is not None and previous.name == 'pto.barrier'
                                    and 'PIPE_ALL' in str(previous.attributes['pipe']), 'missing completion drain')
                                drains += 1
    require(drains == (5 if serial else 3*count+2), 'incomplete release/drain coverage')
    schedules = {s['loop_id']: s for s in feedback['schedules']}
    functions = {f['id']: f for f in program['functions'] if not f['prepared_vf']}
    streams, capacity = {}, {}
    for name, function in functions.items():
        ops = {o['id']: o for o in function['operations']}
        handles = {}
        for op in ops.values():
            if op['name'] == 'pto.initialize_l2l_pipe':
                channel = op['attributes']['__pto.frontend_id']
                handles[op['results'][0]['id']] = channel
                slots = op['attributes']['slot_num']
                require(slots == max(4, count) and op['attributes']['slot_size'] == 4096,
                        'unsupported pipe geometry')
                capacity[channel] = slots
        outer = next(v for v in program['loops'] if v['function'] == name and not v['prepared_vf_internal'])
        if serial:
            # Canonical fixture has one independent, region-free outer body.
            loop = ops[next(o['id'] for o in ops.values() if o['name'] == 'scf.for')]
            ids = loop['regions'][0][0]['operations']
            sequence = [(i, ops[x]) for i in range(count) for x in ids]
        else:
            schedule = schedules[outer['id']]
            require(schedule == next(s for s in analysis['schedules'] if s['loop_id'] == outer['id']), 'schedule changed')
            sequence = [(e['iteration'], ops[e['source_operation']]) for e in schedule['events']]
        comm = []
        for iteration, op in sequence:
            action = {'pto.tpush':'push', 'pto.tpop':'pop', 'pto.tfree':'free'}.get(op['name'])
            if action:
                channel = handles[op['operands'][-1]]
                comm.append((action, channel, iteration))
        lanes = ('AIV0', 'AIV1') if name.endswith('_mix_aiv') else ('AIC0',)
        for lane in lanes: streams[lane] = comm
    fifo = progress(streams, capacity)
    binding = dict(input_fingerprint=manifest['input_fingerprint'], compiler=manifest['producer'],
                   compilation_sha256=cv.sha(candidate/'compilation.json'), product_sha256=record['product_sha256'],
                   feedback_sha256=cv.sha(candidate/'compiler_feedback.json'))
    result = dict(schema='pto.cv.restricted_micro_g2.v1', G2='pass', binding=binding,
        scope='generated crossing FP32 32x32, N<=8, disjoint arguments and allocations, fixed supported operations',
        completion=dict(status='pass', policy=DRAIN, drains=drains, local_hazards='compiler absolute sync coverage'),
        fifo=fifo, memory=memory, full_program_G2='not_claimed')
    write(candidate/'validation_report.json', result)
    return result


def fixture(n, symbol):
    return vf_source_text(n, crossing=True, symbol=symbol)


def prepare(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant in ('A', 'P0', 'M', 'B'):
        symbol = 'ncv_'+hashlib.sha256((str(root)+variant).encode()).hexdigest()[:16]
        path = root/variant; path.mkdir()
        source = path/'source.pto'; source.write_text(fixture(args.count, symbol))
        bindings = dict(iterations=args.count, tile_shape=[32,32], launch=dict(aic=1,aiv=2),
            alias_constraints=[dict(kind='pairwise_disjoint',arguments=[symbol+'_mix_aic:arg'+str(i) for i in range(4)])],
            parameter_bindings={symbol+core:['Q','K','V','O'] for core in ('_mix_aic','_mix_aiv')})
        target = dict(arch='a5', cann_version='9.2.0')
        cv.export_bundle(source, bindings, target, path/'package', compiler=args.compiler)
        manifest, program = cv.load_bundle(path/'package')
        p = 2 if variant == 'B' else 0
        loops = {v['id']:p for v in program['loops'] if not v['prepared_vf_internal']}
        analysis = cv.analyze_schedule(path/'package', loops, path/'analysis', compiler=args.compiler)
        plan = analysis['minimum_slot_plan']; plan['candidate_id'] = symbol+'-'+variant
        if variant == 'M':
            for b in plan['buffers']:
                if b['buffer_id'] == symbol+'_mix_aiv:buffer:0': b['count'] = 3
        # A preserves the serial loop. All other variants use the native schedule.
        annotated = path/'annotated.pto'
        if variant == 'A': shutil.copyfile(path/'package/canonical.pto', annotated)
        else: cv.annotate_plan(path/'package', plan, annotated)
        candidate = path/'candidate'; candidate.mkdir()
        feedback = candidate/'compiler_feedback.json'; product = candidate/'candidate.o'
        flags = cv._flags(target)+['--cv-apply-vf','--enable-insert-sync','--enable-vecscope-mem-bar',
              '--cv-drain-before-free','--check-sync-ids','--check-sync-interference','--check-sync-self-coverage']
        cv._invoke(args.compiler,[annotated,*flags,'--cv-feedback='+str(feedback),'-o',product],candidate/'compile.log')
        report = cv.read_json(feedback)
        record = dict(candidate_id=plan['candidate_id'], product_sha256=cv.sha(product),
                      feedback_sha256=cv.sha(feedback), input_fingerprint=manifest['input_fingerprint'])
        write(candidate/'compilation.json', record); write(path/'plan.json', plan)
        g2 = audit(path/'package',candidate,analysis,args.count,variant=='A')
        rows.append(dict(variant=variant,symbol=symbol,preload=p,plan=plan,G2=g2['G2'],object_sha256=cv.sha(product)))
    # P-invalid is compile-only and may never be launched.
    bpath=root/'B'; plan=copy.deepcopy(cv.read_json(bpath/'plan.json'))
    for row in plan['buffers']:
        if row['buffer_id'].endswith('_mix_aiv:buffer:0'): row['count']=1
    cv.annotate_plan(bpath/'package',plan,root/'invalid.pto')
    try:
        cv._invoke(args.compiler,[root/'invalid.pto',*cv._flags(target),'--cv-apply-vf','--emit-pto-ir','-o',root/'invalid-output.pto'],root/'invalid.log')
    except RuntimeError:
        require('INSUFFICIENT_SLOTS' in (root/'invalid.log').read_text(),'invalid candidate failed for wrong reason')
    else: raise ValueError('insufficient slots accepted')
    for seed in range(3):
        path=root/f'seed{seed}';path.mkdir()
        for name,data in input_arrays(args.count,seed,True).items(): data.tofile(path/(name+'.bin'))
    files={str(p.relative_to(root)):cv.sha(p) for p in root.rglob('*') if p.is_file()}
    write(root/'manifest.json',dict(schema='pto.cv.native_ablation.v1',count=args.count,preload=2,
        variants=rows, files=files, model_selection='not_run; fixed mechanism experiment',negative='INSUFFICIENT_SLOTS',
        compiler=cv.compiler_identity(args.compiler)))


def verify_matrix(root):
    manifest=cv.read_json(root/'manifest.json')
    for name,digest in manifest['files'].items(): require(cv.sha(root/name)==digest,'matrix changed: '+name)
    require({v['variant'] for v in manifest['variants']} == {'A','P0','M','B'},'incomplete variants')
    return manifest


def build(args):
    manifest=verify_matrix(args.matrix)
    toolkit=Path(os.environ['ASCEND_HOME_PATH']); compiler=shutil.which('bisheng')
    for row in manifest['variants']:
        path=args.artifacts/row['variant'];path.mkdir(parents=True,exist_ok=False)
        obj=args.matrix/row['variant']/'candidate/candidate.o'
        require(row['symbol'].encode() in obj.read_bytes(),'object symbol mismatch')
        # Verified generated CCE host-stub ABI: block count, flags, stream,
        # followed by Q/K/V/O. The linked object owns the registration and stub.
        wrapper=path/'launch.cpp'
        wrapper.write_text('extern "C" void '+row['symbol']+'(unsigned,unsigned,void*,float*,float*,float*,float*);\n'
          +'extern "C" void launch_cv(float*q,float*k,float*v,float*o,void*s){'+row['symbol']+'(1,0,s,q,k,v,o); }\n')
        flags=[compiler,'-std=c++17','-O2','-fPIC','-I'+str(toolkit/'aarch64-linux/include')]
        commands=[flags+['-c',str(wrapper),'-o',str(path/'launch.o')],
            flags+[str(Path(__file__).with_name('runtime_main.cpp')),str(obj),str(path/'launch.o'),
              '--cce-fatobj-link','-L'+str(toolkit/'aarch64-linux/lib64'),'-Wl,-rpath,'+str(toolkit/'aarch64-linux/lib64'),
              '-lruntime','-lascendcl','-lstdc++','-ldl','-lpthread','-o',str(path/'runner')]]
        write(path/'commands.json',commands)
        for i,command in enumerate(commands):
            with (path/f'build-{i}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        write(path/'build.json',dict(object_sha256=cv.sha(obj),runner_sha256=cv.sha(path/'runner'),symbol=row['symbol'],
                                   bisheng_sha256=cv.sha(Path(compiler))))


def run(args):
    manifest=verify_matrix(args.matrix)
    performance=args.mode=='performance'
    if performance:
        g3=cv.read_json(args.g3)
        require(g3['status']=='pass' and len(g3['rows'])==24 and g3['matrix_sha256']==cv.sha(args.matrix/'manifest.json'), 'G3 incomplete')
        require(g3['device']==args.device and g3['soc']==args.soc,'G3 device changed')
    output=args.experiment/'results'/(args.mode+'.json')
    require(not output.exists(),'do not overwrite experiment')
    report=dict(status='running',matrix_sha256=cv.sha(args.matrix/'manifest.json'),device=args.device,soc=args.soc,rows=[],
                health='ACL_RUNTIME_AVAILABLE',limitation='npu-smi unavailable; no full health or noncooperating occupancy telemetry')
    write(output,report)
    variants={v['variant']:v for v in manifest['variants']}
    if performance:
        sequence=[(0,0,v,False) for v in ('A','B','P0','M')]
        sequence += [(i,i%3,v,True) for i in range(20) for v in ORDER[i%4]]
    else:
        sequence=[(r,s,v,False) for r,order in enumerate((('B','M','P0','A'),('A','P0','M','B'))) for s in range(3) for v in order]
    hashes={}
    try:
        for index,(block,seed,name,profile) in enumerate(sequence):
            binary=args.artifacts/name; build=cv.read_json(binary/'build.json')
            require(cv.sha(binary/'runner')==build['runner_sha256'],'runner changed')
            if performance: require(g3['builds'][name]==cv.sha(binary/'build.json'),'G3 build changed')
            helper=args.experiment.parents[1]/'bin/new_run_dir.sh'
            runtime=Path(subprocess.check_output([str(helper),str(args.experiment),'crossing',name,f'{args.mode}-{index}'],text=True).strip())
            for p in (args.matrix/f'seed{seed}').glob('*.bin'):shutil.copyfile(p,runtime/p.name)
            initial={p.name:cv.sha(p) for p in runtime.glob('*.bin')}
            command=[str(binary/'runner'),str(args.device),str(manifest['count']),str(runtime)]
            if profile:command=['msprof','--application='+shlex.join(command),'--output='+str(runtime/'profile'),'--aic-mode=task-based','--aic-metrics=PipeUtilization']
            write(runtime/'command.json',command)
            print(args.mode,index,name,seed,flush=True)
            start=time.monotonic()
            returncode=execute(command,runtime)
            log=(runtime/'runtime.log').read_text()
            require(returncode==0 and 'GOLDEN_PASS' in log and 'soc='+args.soc in log,'runtime/golden/SOC failed: '+str(runtime))
            require(all(cv.sha(runtime/k)==v for k,v in initial.items()),'input changed')
            digest=cv.sha(runtime/'output.bin')
            require(hashes.get((name,seed),digest)==digest,'unstable output');hashes[name,seed]=digest
            row=dict(index=index,block=block,seed=seed,variant=name,profile=profile,runtime=str(runtime),
                     wall_seconds=time.monotonic()-start,output_sha256=digest,input_sha256=initial,symbol=variants[name]['symbol'])
            if profile: row.update(duration(runtime, row['symbol'], args.device))
            write(runtime/'result.json',row);report['rows'].append(row);write(output,report)
        verify_matrix(args.matrix)
        report.update(status='pass',builds={n:cv.sha(args.artifacts/n/'build.json') for n in variants})
    except Exception as error:
        report.update(status='fail',error=str(error));write(output,report);raise
    write(output,report)


def summarize(args):
    import statistics
    from summarize_ab import paired_statistics
    manifest=verify_matrix(args.matrix)
    report=cv.read_json(args.experiment/'results/performance.json')
    require(report['status']=='pass' and len(report['rows'])==84, 'incomplete performance batch')
    rows=report['rows'][4:];values={n:[] for n in ('A','P0','M','B')}
    for block in range(20):
        chunk=rows[4*block:4*block+4]
        require(tuple(r['variant'] for r in chunk)==ORDER[block%4], 'wrong paired order')
        for row in chunk:
            require(row['block']==block and row['seed']==block%3 and row['profile'], 'wrong pair membership')
            runtime=Path(row['runtime'])
            require(cv.read_json(runtime/'result.json')==row and cv.sha(runtime/'output.bin')==row['output_sha256'], 'runtime evidence mismatch')
            require(all(cv.sha(runtime/k)==v for k,v in row['input_sha256'].items()), 'input evidence mismatch')
            check=duration(runtime,row['symbol'],report['device'])
            require(check['duration_us']==row['duration_us'] and check['profiler_sha256']==row['profiler_sha256'], 'profiler evidence changed')
            values[row['variant']].append(row['duration_us'])
    stats={label:paired_statistics(values[base],values[candidate]) for label,base,candidate in
           (('B/A','A','B'),('B/P0','P0','B'),('M/P0','P0','M'),('P0/A','A','P0'))}
    passed=all(stats[k]['mean_gain']>=.02 and stats[k]['confidence_interval'][0]>0
               and stats[k]['worst_regression']<=.02 for k in ('B/A','B/P0'))
    passed &= stats['M/P0']['confidence_interval'][0]>=-.02 and stats['M/P0']['confidence_interval'][1]<=.02
    result=dict(status='MECHANISM_BENEFIT_PASS' if passed else 'MECHANISM_BENEFIT_FAIL',G4='not_claimed',
        G2='pass_restricted_micro',G3='pass',statistics=stats,
        durations={n:dict(mean_us=statistics.mean(v),median_us=statistics.median(v),min_us=min(v),max_us=max(v)) for n,v in values.items()},
        model_assessment='not_available; fixed compiler mechanism candidate, not TileSim recommendation',
        matrix_sha256=cv.sha(args.matrix/'manifest.json'),performance_sha256=cv.sha(args.experiment/'results/performance.json'))
    write(args.experiment/'results/summary.json',result)
    lines=['# Native VF × CV × PlanMemory 消融报告','',result['status'],'',
      '范围：crossing，N=8，32×32 FP32，1 AIC＋2 AIV；三个独立整数种子，精确 golden。',
      '四组均使用本分支 VF 融合、真实 PlanMemory、PTOAS 同步及释放前完成屏障。',
      'B 是上板前固定的 P=2、P Buffer 三槽候选；本轮不宣称 TileSim 推荐或 G4。','',
      '| 变体 | 调度 | P 槽数 | 平均 μs | 中位数 μs | 范围 μs |','|---|---|---:|---:|---:|---:|']
    for name in ('A','P0','M','B'):
        d=result['durations'][name]
        lines.append(f"| {name} | {'串行循环' if name=='A' else 'P=2 展开' if name=='B' else 'P=0 展开'} | {3 if name in ('M','B') else 1} | {d['mean_us']:.4f} | {d['median_us']:.4f} | {d['min_us']:.4f}–{d['max_us']:.4f} |")
    lines += ['', '| 比较 | 配对平均收益 | bootstrap 95% 区间 | 最坏回退 |', '|---|---:|---:|---:|']
    for name,d in stats.items():
        lines.append(f"| {name} | {d['mean_gain']:.2%} | [{d['confidence_interval'][0]:.2%}, {d['confidence_interval'][1]:.2%}] | {d['worst_regression']:.2%} |")
    lines += ['', '归因：P0/A 检验展开影响；M/P0 检验额外存储影响；B/P0 检验 preload 与必要多槽的组合影响。',
       '四组 VF 融合均开启，因此这些数据不能单独给出 VF 融合收益。',
       '24 次 G3、4 次预热、80 次交替 profiler 均检查输出、输入和 guard。',
       'G2 仅接受固定生成器的精确源码，拒绝物理分配重叠；检查实际内存预算、三执行流 FIFO、非空同步覆盖及释放前完成屏障。',
       '通用编译器 G2 仍不宣称通过。未知来源、通用地址复用、FA、其他 shape 不在本次证据范围。',
       '环境限制：ACL 探测通过，独占任务队列启用；npu-smi 不可用，缺少完整健康及绕过队列进程的占用遥测。',
       '模型反馈：新 prepared-VF adapter 未接通；本轮没有可比的模型时延，不能将其误差记为零。实际布局、同步及原始 profiler 可用于后续同候选重评。']
    (args.experiment/'results/report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','build','correctness','performance','summarize'),required=True)
    p.add_argument('--compiler');p.add_argument('--output',type=Path);p.add_argument('--count',type=int,choices=(4,8),default=8)
    p.add_argument('--matrix',type=Path);p.add_argument('--artifacts',type=Path);p.add_argument('--experiment',type=Path)
    p.add_argument('--g3',type=Path);p.add_argument('--device',type=int,default=0);p.add_argument('--soc')
    args=p.parse_args()
    {'prepare':prepare,'build':build,'correctness':run,'performance':run,'summarize':summarize}[args.mode](args)

if __name__=='__main__':main()
