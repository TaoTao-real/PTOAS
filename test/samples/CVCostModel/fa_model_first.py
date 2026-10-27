# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Model-first M32 FA acceptance: one prepare, select, one candidate compile.

Baseline preparation is an explicitly separate evaluation-only operation after
selection. It is not an input to search. Old FA evidence is not read for costs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from fa_model_ab import audit_candidate, numpy_golden, read, require, sha, write, invoke


def binding(symbol, original):
    return dict(shape=[128, 1024, 128], tile=[32, 128, 128], original_sha256=original,
        argument_shapes=dict(Q=[128, 128], K=[1024, 128], V=[1024, 128], O=[128, 128]),
        launch=dict(aic=1, aiv=2), alias_constraints=[dict(kind='pairwise_disjoint', arguments=['Q', 'K', 'V', 'O'])],
        parameter_bindings={symbol + core: ['Q', 'K', 'V', 'O', 128, 1024] for core in ('_mix_aic', '_mix_aiv')})


def validate_native(folder, program, plan, model):
    from native_ablation import validate_memory
    from ptoas import cv_vf as cv
    fb = read(folder / 'compiler_feedback.json')
    require(fb['native_plan_memory_invoked'] and sha(folder / 'candidate.o') == fb['product_sha256'], 'missing same-compilation native memory/object evidence')
    require(cv.fingerprint(fb['schedules']) == plan['schedule_fingerprint'], 'wrong compiled schedule')
    log = (folder / 'compile.log').read_text()
    for f in fb['functions']:
        require(f['disjoint_allocation_policy'] and f['stage_completion_policy'] == 'drain_vf_and_before_crosscore_signal.v1', 'unknown completion policy')
        coverage = [l for l in log.splitlines() if '[sync-gate:g1-self]' in l and 'func=' + f['function'] + ' ' in l]
        require(len(coverage) == 1 and 'uncovered=0 ' in coverage[0] and 'unmappable=0' in coverage[0], 'absolute sync coverage missing')
    for f in program['functions']:
        if not f['entry']: continue
        ops = [o for o in f['operations'] if not o['name'].startswith('arith.') and o['name'] not in ('pto.castptr', 'pto.addptr', 'pto.alloc_tile', 'pto.tile_buf_addr')]
        def barrier(o): return o['name'] == 'pto.barrier' and o['attributes'].get('pipe') == dict(enum='PipeAttr', value=6)
        for i, op in enumerate(ops):
            if op['name'] == 'func.call': require(i + 1 < len(ops) and barrier(ops[i + 1]), 'VF access completion missing')
            if op['name'] == 'pto.set_intra_block' and 'pto.cv.token' in op['attributes']:
                require(i > 0 and barrier(ops[i - 1]), 'cross-core signal before completion')
            if op['name'] == 'func.return': require(i > 0 and barrier(ops[i - 1]), 'final drain missing')
    write(folder / 'program.json', program)
    write(folder / 'compiler_validation.json', dict(status='compiler_checks_passed', memory=validate_memory(fb),
        synchronization='native ID/interference/absolute coverage and final completion drains'))
    row = dict(candidate_id=plan['candidate_id'], plan=plan, program_sha256=sha(folder / 'program.json'), feedback_sha256=sha(folder / 'compiler_feedback.json'))
    return audit_candidate(folder, row, model)


def launcher(folder, symbol):
    invoke([shutil.which('bisheng'), '--cce-fatobj-link', '--cce-aicore-arch=dav-c310', '-r', folder / 'candidate.o', '-o', folder / 'kernel.linked.o'], folder / 'link.log')
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    invoke([shutil.which('g++'), '-O2', '-std=c++17', '-DKERNEL_SYMBOL=' + symbol,
        Path(__file__).with_name('fa_runtime_main.cpp'), folder / 'kernel.linked.o', '-I' + str(cann / 'include'),
        '-L' + str(cann / 'lib64'), '-Wl,-rpath,' + str(cann / 'lib64'), '-lascendcl', '-lruntime', '-lprofapi', '-ldl', '-pthread', '-o', folder / 'runner'], folder / 'launcher.log')
    strings = subprocess.check_output(['strings', str(folder / 'candidate.o')], text=True)
    require(symbol + '_mix_aic' in strings and symbol + '_mix_aiv' in strings, 'kernel symbol missing in object')


def evaluate_final(folder, plan, config, command, destination):
    from ptoas import cv_vf as cv
    fb = read(folder / 'compiler_feedback.json')
    context, module = cv._parse(fb['final_vpto_ir'])
    with context: program = cv.describe(module)
    write(folder / 'program.json', program)
    row = dict(candidate_id=plan['candidate_id'], preload_count=plan['loops'][0]['preload_count'], plan=plan,
        program=str((folder / 'program.json').resolve()), program_sha256=sha(folder / 'program.json'),
        feedback=str((folder / 'compiler_feedback.json').resolve()), feedback_sha256=sha(folder / 'compiler_feedback.json'),
        final_vpto_ir_sha256=fb['final_vpto_ir_sha256'])
    request = dict(schema='pto.cv.exact_fa_proxy_request.v1', original_sha256=__import__('fa_native_ab').SOURCE_SHA,
        candidates=[row], model_target=config['hardware'], timing_model_arch=config['timing_model_arch'],
        allow_profile_proxy=config.get('allow_profile_proxy', False), target_limitations=config.get('target_limitations', []))
    path = folder / 'model-request.json'; write(path, request)
    invoke([*command, '--evaluate-only', path, destination], folder / 'model-evaluation.log')
    return program


def trace_report(path, selected):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    markers = [e for e in events if e['event'] == 'selection_frozen']
    require(len(markers) == 1, 'missing/duplicate initial selection marker')
    freeze = markers[0]['monotonic_ns']
    evaluation = next(e['monotonic_ns'] for e in events if e['event'] == 'evaluation_only_begin')
    processes = [e for e in events if e['event'] == 'process_start']
    def argv(e): return bytes.fromhex(e['argv_hex']).decode().split('\0')
    before = [e for e in processes if e['monotonic_ns'] < freeze]
    forbidden = [e for e in before if e['executable'].startswith(('bisheng', 'ld.lld', 'ld-linux')) or
        any(a in ('--emit-vpto-llvm-ir', '--cce-fatobj-link') for a in argv(e))]
    require(not forbidden, 'device compilation/linking occurred before selection')
    prepare = [e for e in processes if e['monotonic_ns'] < evaluation and '--cv-prepare-vf' in argv(e)]
    applies = [e for e in processes if e['monotonic_ns'] < evaluation and '--cv-apply-vf' in argv(e) and '--cv-analyze-schedule' not in argv(e)]
    require(len(prepare) == len(applies) == 1, 'production path did not prepare/compile exactly once')
    require(all(e['monotonic_ns'] > freeze for e in applies), 'candidate apply preceded selection')
    return dict(status='pass', evidence='Linux dynamic-process constructors plus immutable selection/compile markers',
        limitation='not an adversarial sandbox; compiler/model child environments retain the trace library',
        selected_candidate=selected, prepare_count=1, initial_device_compilations=0, selected_full_compilations=1,
        process_starts=len(processes), preselection_processes=len(before), trace_sha256=sha(path))


def prepare(a):
    from ptoas import cv_vf as cv
    from ptoas import _cv_search as search
    from fa_native_ab import import_fa, SOURCE_SHA
    root = a.root.resolve(); root.mkdir(parents=True, exist_ok=False)
    require(sha(a.source) == SOURCE_SHA, 'unexpected FA input')
    original = root / 'original.pto'; shutil.copyfile(a.source, original)
    prefix = 'fa_mf_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    selected_symbol = prefix + '_b'; import_fa(original, root / 'selected-source.pto', selected_symbol)
    profile = read(Path(__file__).with_name('a5_profile.json'))
    target = dict(arch='a5', cann_version='9.2.0', soc='Ascend950DT_9592',
        compiler_budget_bytes=profile['capacity_bytes'], alignment_bytes=profile['alignment_bytes'])
    config = read(a.model); hardware = Path(config['hardware'])
    if not hardware.is_absolute(): hardware = a.model.resolve().parent / hardware
    config['hardware'] = str(hardware.resolve()); config['hardware_sha256'] = sha(hardware)
    model_command, final_command = read(a.model_command), read(a.final_command)
    result = search.optimize(root / 'selected-source.pto', binding(selected_symbol, SOURCE_SHA), target, config,
        root / 'optimization', compiler=a.compiler, model_command=model_command, final_command=final_command)
    require(result['status'] == 'compiled_recommendation' and result['final_result'] is not None, 'BASELINE_RETAINED; no optimized B to benchmark')
    success = result['final_result']; selected = success['candidate_id']
    attempt = next(root / 'optimization' / ('attempt-' + str(i)) for i, r in enumerate(result['attempts']) if r['status'] == 'compiled')
    product = attempt / 'candidate'; plan = read(product / 'plan.json')
    require(plan['loops'][0]['preload_count'] > 0, 'selected preload is baseline')
    validate_native(product, read(attempt / 'program.json'), plan, attempt / 'final-model')
    # Baseline is created only after the model-selected candidate and its final
    # feedback have been accepted. It cannot influence the frozen selection.
    search.trace_marker('evaluation_only_begin')
    launcher(product, selected_symbol)
    baseline = root / 'baseline'; baseline.mkdir()
    baseline_symbol = prefix + '_a'; import_fa(original, baseline / 'source.pto', baseline_symbol)
    cv.export_bundle(baseline / 'source.pto', binding(baseline_symbol, SOURCE_SHA), target, baseline / 'package', compiler=a.compiler)
    _, program = cv.load_bundle(baseline / 'package')
    analysis = cv.analyze_schedule(baseline / 'package', {l['id']: 0 for l in program['loops'] if not l['prepared_vf_internal']}, baseline / 'analysis', compiler=a.compiler)
    base_plan = analysis['minimum_slot_plan']; base_plan['candidate_id'] = 'p0'
    cv.compile_plan(baseline / 'package', base_plan, baseline / 'candidate', compiler=a.compiler, emit='object', verify_sync=True)
    final = evaluate_final(baseline / 'candidate', base_plan, config, final_command, baseline / 'final-model')
    validate_native(baseline / 'candidate', final, base_plan, baseline / 'final-model')
    launcher(baseline / 'candidate', baseline_symbol)
    require(selected_symbol != baseline_symbol, 'A/B symbols overlap')
    rows = [dict(candidate_id='p0', symbol=baseline_symbol, folder='baseline/candidate', purpose='evaluation_only'),
        dict(candidate_id=selected, symbol=selected_symbol, folder=str(product.relative_to(root)), purpose='model_selected')]
    compiler_identity = cv.compiler_identity(a.compiler)
    evidence = trace_report(os.environ['PTOAS_CV_PROCESS_TRACE'], selected)
    write(root / 'process-order-validation.json', evidence)
    # Runtime outputs are allocated under separate private experiment IDs.
    files = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
    write(root / 'ab_manifest.json', dict(schema='pto.cv.fa_model_first_ab.v1', selected=selected, rows=rows,
        files=files, original_sha256=SOURCE_SHA, compiler=compiler_identity,
        optimization=result, process_order=evidence, device=0, hardware_soc='Ascend950DT_9592',
        timing_profile='950PR_9599 proxy; user authorized', full_target_accuracy_certification=False))
    print('MODEL_FIRST_READY', selected, evidence, flush=True)


def check_frozen(root):
    manifest = read(root / 'ab_manifest.json')
    for name, expected in manifest['files'].items(): require(sha(root / name) == expected, 'frozen artifact changed: ' + name)
    return manifest


def deploy(a):
    """Bind immutable compiled artifacts to one device before any new run."""
    source, output = a.root.resolve(), a.output.resolve()
    manifest = check_frozen(source)
    require(a.device >= 0, 'invalid device')
    source_digest = sha(source / 'ab_manifest.json')
    shutil.copytree(source, output)
    # Finalized evidence intentionally has no write/execute bits. Restore only
    # the new deployment's metadata and bound launchers, never the source.
    (output / 'ab_manifest.json').chmod(0o600)
    for row in manifest['rows']:
        folder = Path(row['folder'])
        require(not folder.is_absolute() and '..' not in folder.parts, 'unsafe launcher folder')
        runner = (output / folder / 'runner').resolve()
        require(runner.is_relative_to(output) and
            sha(runner) == manifest['files'][runner.relative_to(output).as_posix()], 'unbound deployment launcher')
        runner.chmod(0o500)
    manifest['device'] = a.device
    manifest['deployment'] = dict(source_manifest_sha256=source_digest,
        immutable_artifacts_reused=True, new_g3_required=True)
    write(output / 'ab_manifest.json', manifest)
    check_frozen(output)
    require(sha(source / 'ab_manifest.json') == source_digest, 'source deployment changed')


def run(a):
    from run_ab import execute
    from summarize_runtime import duration
    root, experiment = a.root.resolve(), a.experiment.resolve(); manifest = check_frozen(root)
    selected = manifest['selected']; rows = {r['candidate_id']: r for r in manifest['rows']}
    device = manifest['device']; require(type(device) is int and device >= 0, 'invalid deployment device')
    variants = ['p0', selected]; cases = [('zero', 12345), ('one_v', 12345), ('one_qkv', 12345)] + [('pattern', s) for s in (12345, 20260916, 20261022)]
    if a.mode == 'g3':
        sequence = [(selected, 'pattern', 12345, -1, 'cold-first')]
        for repeat, order in enumerate(([selected, 'p0'], ['p0', selected])):
            for mode, seed in cases:
                for p in order: sequence.append((p, mode, seed, repeat, 'g3'))
        reference = None
    else:
        g3 = read(a.g3_report)
        require(g3['status'] == 'pass' and len(g3['rows']) == 25 and g3['device'] == device and g3['matrix_sha256'] == sha(root / 'ab_manifest.json'), 'matching G3 required')
        reference = {(r['candidate_id'], r['case'], r['seed']): r for r in g3['rows']}
        sequence = [(p, 'pattern', 12345, -1, 'warmup') for p in variants]
        for block in range(20):
            for p in (variants if block % 2 == 0 else list(reversed(variants))):
                sequence.append((p, 'pattern', (12345, 20260916, 20261022)[block % 3], block, 'profile'))
    report = dict(status='running', mode=a.mode, matrix_sha256=sha(root / 'ab_manifest.json'), device=device, rows=[],
        environment='ACL_RUNTIME_AVAILABLE; no npu-smi health telemetry; exclusive task-submit queue')
    output = experiment / 'results' / ('model-first-' + a.mode + '.json'); require(not output.exists(), 'experiment output exists')
    try:
        for p, mode, seed, block, stage in sequence:
            check_frozen(root); row = rows[p]
            runtime = Path(subprocess.check_output([str(experiment.parents[1] / 'bin/new_run_dir.sh'), str(experiment), 'fa', p,
                f'{stage}-{mode}-{seed}-{block}'], text=True).strip())
            command = [str(root / row['folder'] / 'runner'), str(device), mode, str(seed), str(runtime / 'output')]
            if stage == 'profile': command = ['msprof', '--application=' + shlex.join(command), '--output=' + str(runtime / 'profile'), '--aic-mode=task-based', '--aic-metrics=PipeUtilization']
            write(runtime / 'command.json', command); started = time.monotonic(); status = execute(command, runtime)
            log = (runtime / 'runtime.log').read_text()
            require(status == 0 and 'NUMERICAL_PASS' in log and 'SOC=Ascend950DT_9592' in log and
                'input_unchanged=1 guard_unchanged=1' in log, 'runtime correctness failed: ' + str(runtime))
            saved = dict(candidate_id=p, case=mode, seed=seed, block=block, stage=stage, status='pass', runtime=str(runtime),
                symbol=row['symbol'], inputs={n: sha(runtime / 'output' / n) for n in ('Q.f16', 'K.f16', 'V.f16')},
                output_sha256=sha(runtime / 'output/O.f32'), numpy_max_abs_error=numpy_golden(runtime / 'output', mode), wall_seconds=time.monotonic() - started)
            if reference is not None:
                old = reference[(p, mode, seed)]
                require(old['inputs'] == saved['inputs'] and old['output_sha256'] == saved['output_sha256'], 'input/output differs from G3')
            if stage == 'profile': saved.update(duration(runtime, row['symbol'], device))
            write(runtime / 'result.json', saved); report['rows'].append(saved); write(output, report)
            print(stage, p, mode, seed, block, saved.get('duration_us', 'PASS'), flush=True)
        outputs, inputs = defaultdict(set), defaultdict(set)
        for row in report['rows']:
            outputs[(row['case'], row['seed'])].add(row['output_sha256'])
            inputs[(row['case'], row['seed'])].add(tuple(sorted(row['inputs'].items())))
        require(all(len(v) == 1 for v in [*outputs.values(), *inputs.values()]), 'unstable or mismatched A/B output/input')
        report['status'] = 'pass'; write(output, report)
    except Exception as error:
        report.update(status='fail', error=str(error)); write(output, report); raise


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('source', type=Path); q.add_argument('root', type=Path)
    for key in ('compiler', 'model', 'model-command', 'final-command'): q.add_argument('--' + key, type=Path, required=True)
    q = sub.add_parser('run'); q.add_argument('root', type=Path); q.add_argument('experiment', type=Path)
    q.add_argument('--mode', choices=['g3', 'performance'], required=True); q.add_argument('--g3-report', type=Path)
    q = sub.add_parser('deploy'); q.add_argument('root', type=Path); q.add_argument('output', type=Path)
    q.add_argument('--device', type=int, required=True)
    a = p.parse_args(); globals()[a.action](a)


if __name__ == '__main__': main()
