# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""File-based model-first search; model execution stays in a separate process."""
from pathlib import Path
import json
import os
import re
import subprocess
import time

from . import cv_vf as cv
from ._cv_search_events import FEATURES, LogicalEvents, require

SCHEMA = 'pto.cv.prepared_search.v1'


def trace_marker(event, **fields):
    path = os.environ.get('PTOAS_CV_PROCESS_TRACE')
    if path:
        data = (json.dumps(dict(event=event, monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **fields)) + '\n').encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            require(os.write(fd, data) == len(data), 'incomplete process trace marker')
        finally: os.close(fd)


def local(root, name):
    name = Path(name)
    require(not name.is_absolute() and '..' not in name.parts, 'nonlocal exchange file')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'exchange symlink escapes package')
    return path


def resources(program, plan, target):
    """Conservative sum of known aligned slots; not an allocation proof."""
    budgets, alignment = target['compiler_budget_bytes'], target['alignment_bytes']
    spaces = {2: 'mat', 3: 'left', 4: 'right', 5: 'acc', 6: 'vec'}
    counts = {b['buffer_id']: b['count'] for b in plan['buffers']}
    functions = {f['id']: f for f in program['functions']}
    totals, seen = {}, set()
    for b in program['buffers']:
        require(type(b['bytes']) is int and b['bytes'] > 0, 'unknown storage size')
        identity = b.get('storage_id', b['id'])
        if identity in seen: continue
        seen.add(identity)
        space = spaces[b['memory_space']['value']]
        owner = functions[b.get('storage_owner_function', b['function'])]
        cores = (['AIV0', 'AIV1'] if space == 'vec' else owner['physical_instances'])
        size = b['bytes'] * counts[b['id']]
        size = (size + alignment[space] - 1) // alignment[space] * alignment[space]
        for core in cores:
            domain = core + ':' + space
            totals[domain] = totals.get(domain, 0) + size
    for f in program['functions']:
        for op in f['operations']:
            if op['name'] != 'pto.reserve_buffer': continue
            a = op['attributes']; require(a.get('autoAlloc') is True and a['size'] > 0, 'unknown reservation')
            space = spaces[a['location']['value']]
            size = (a['size'] + alignment[space] - 1) // alignment[space] * alignment[space]
            for core in (['AIV0', 'AIV1'] if space == 'vec' else f['physical_instances']):
                domain = core + ':' + space; totals[domain] = totals.get(domain, 0) + size
    unresolved = [k for k, v in totals.items() if v > budgets[k.split(':')[1]]]
    return dict(basis='sum_known_aligned_slots_without_reuse', bytes_by_domain=totals,
        budget=budgets, status='unresolved' if unresolved else 'precheck_pass', unresolved_domains=unresolved)


def export_candidates(package, preloads, destination, *, compiler):
    package, destination = Path(package).resolve(), Path(destination).resolve()
    manifest, program = cv.load_bundle(package)
    require(type(preloads) is list and preloads and len(preloads) == len(set(preloads)) and
        preloads.count(0) == 1 and all(type(p) is int and p >= 0 for p in preloads), 'invalid preload candidates')
    destination.mkdir(parents=True, exist_ok=False)
    bindings = cv.read_json(package / 'bindings.json'); target = cv.read_json(package / 'target.json')
    all_vfs, rows = {}, []
    started = time.monotonic()
    for preload in preloads:
        identity = 'p' + str(preload); folder = destination / identity
        loops = {l['id']: preload for l in program['loops'] if not l['prepared_vf_internal']}
        analysis = cv.analyze_schedule(package, loops, folder, compiler=compiler)
        context, module = cv._parse((folder / 'scheduled.pto').read_text())
        with context: description = cv.describe(module)
        plan = analysis['minimum_slot_plan']; plan['candidate_id'] = identity
        resource = resources(description, plan, target)
        require(resource['status'] == 'precheck_pass', 'resource precheck unresolved; bounded planner trial is not implemented')
        events = LogicalEvents(description, bindings).build()
        for vf in events.pop('vfs'): all_vfs.setdefault(vf['key'], vf)
        cv._write(folder / 'events.json', events)
        cv._write(folder / 'program.json', description)
        rows.append(dict(candidate_id=identity, preload_count=preload,
            events=identity + '/events.json', events_sha256=cv.sha(folder / 'events.json'),
            program=identity + '/program.json', program_sha256=cv.sha(folder / 'program.json'),
            plan=plan, resource=resource))
    cv._write(destination / 'vfs.json', dict(vfs=list(all_vfs.values())))
    result = dict(schema=SCHEMA, required_features=FEATURES,
        input_fingerprint=manifest['input_fingerprint'], package_manifest_sha256=cv.sha(package / 'manifest.json'),
        producer=manifest['producer'], compiler_semantics_sha256=cv.sha(Path(__file__).with_name('_cv_search_events.py')),
        bindings=bindings, target=target, candidates=rows, vfs='vfs.json',
        vfs_sha256=cv.sha(destination / 'vfs.json'), full_device_compilations=0,
        candidate_analysis_seconds=time.monotonic() - started)
    result['fingerprint'] = cv.fingerprint(result)
    cv._write(destination / 'search.json', result)
    return result


def validate_search(root):
    root = Path(root); search = cv.read_json(root / 'search.json')
    require(search['schema'] == SCHEMA, 'unsupported search input')
    fields = {'schema', 'required_features', 'input_fingerprint', 'package_manifest_sha256', 'producer',
        'compiler_semantics_sha256', 'bindings', 'target', 'candidates', 'vfs', 'vfs_sha256',
        'full_device_compilations', 'candidate_analysis_seconds', 'fingerprint'}
    require(set(search) == fields and set(search['required_features']) <= set(FEATURES), 'unknown search field or required feature')
    copy = dict(search); expected = copy.pop('fingerprint')
    require(cv.fingerprint(copy) == expected, 'search fingerprint changed')
    for row in search['candidates']:
        require(set(row) == {'candidate_id', 'preload_count', 'events', 'events_sha256', 'program', 'program_sha256', 'plan', 'resource'}, 'unknown candidate field')
        for key in ('events', 'program'):
            require(cv.sha(local(root, row[key])) == row[key + '_sha256'], 'candidate changed: ' + key)
    require(cv.sha(local(root, search['vfs'])) == search['vfs_sha256'], 'VF package changed')
    return search


def validate_model_result(search, result, request_sha256):
    require(result['schema'] == 'pto.cv.logical_model_result.v1' and
        result['input_fingerprint'] == search['input_fingerprint'] and
        result['request_sha256'] == request_sha256, 'model result input/request mismatch')
    copy = dict(result); expected = copy.pop('fingerprint')
    require(cv.fingerprint(copy) == expected, 'model result fingerprint mismatch')
    predictions = result['predictions']; candidates = {r['candidate_id']: r for r in search['candidates']}
    require(len(predictions) == len(candidates) and {p['candidate_id'] for p in predictions} == set(candidates), 'missing/duplicate model candidate')
    for p in predictions:
        r = candidates[p['candidate_id']]
        require(p['plan'] == r['plan'] and p['events_sha256'] == r['events_sha256'], 'model changed Plan or schedule')
        require(type(p['latency_us']) in (int, float) and 0 < p['latency_us'] < float('inf'), 'unknown/nonfinite model cost')
    selected = result['extensions']['tilesim.selection.v1']
    require(selected['candidate_id'] in candidates and selected['plan'] == candidates[selected['candidate_id']]['plan'], 'invalid selected Plan')
    rank = selected['ranked_candidates']
    require(len(rank) == len(candidates) and set(rank) == set(candidates), 'missing/duplicate/unknown ranked candidate')
    is_baseline = candidates[selected['candidate_id']]['preload_count'] == 0
    require(selected['action'] == ('baseline' if is_baseline else 'optimize'), 'selection action does not match candidate')
    by_id = {p['candidate_id']: p for p in predictions}
    baseline = next(p for p in predictions if candidates[p['candidate_id']]['preload_count'] == 0)
    gain = selected['predicted_gain']
    require(type(gain) in (int, float) and abs(gain - (1 - by_id[selected['candidate_id']]['latency_us'] / baseline['latency_us'])) <= 1e-12,
        'selected prediction does not match candidate costs')
    return selected


def run_model(search_root, config, command, destination):
    root, destination = Path(search_root), Path(destination)
    search = validate_search(root)
    require(isinstance(command, list) and command and all(isinstance(v, str) for v in command), 'model command must be argv')
    request = dict(schema='pto.cv.prepared_search_request.v1', search=str((root / 'search.json').resolve()),
        search_sha256=cv.sha(root / 'search.json'), model=config)
    request_path = root / 'model-request.json'
    require(not request_path.exists(), 'model request already frozen')
    cv._write(request_path, request)
    with (root / 'model.log').open('w') as log:
        subprocess.run([*command, str(request_path), str(destination)], check=True,
            stdout=log, stderr=subprocess.STDOUT)
    result = cv.read_json(destination / 'result.json')
    selected = validate_model_result(search, result, cv.sha(request_path))
    freeze = dict(schema='pto.cv.model_first_selection.v1', input_fingerprint=search['input_fingerprint'],
        search_sha256=cv.sha(root / 'search.json'), request_sha256=cv.sha(request_path),
        model_result_sha256=cv.sha(destination / 'result.json'), selection=selected,
        model_config_fingerprint=cv.fingerprint(config),
        full_device_compilations_before_selection=0, initial_prediction_status='requires_final_feedback',
        created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    cv._write(root / 'selection.json', freeze)
    trace_marker('selection_frozen', selection_sha256=cv.sha(root / 'selection.json'), candidate_id=selected['candidate_id'])
    return freeze


def compile_selected(package, search_root, output, *, compiler, final_command, model_config, emit='object', selection_path=None):
    """Compile only a frozen selected Plan; final evaluation never reselects it."""
    package, root, output = Path(package), Path(search_root), Path(output)
    search = validate_search(root); selection_path = Path(selection_path or root / 'selection.json')
    selection = cv.read_json(selection_path)
    require(selection['search_sha256'] == cv.sha(root / 'search.json'), 'selection belongs to another search')
    model_path = root / 'model/result.json'
    require(cv.sha(model_path) == selection['model_result_sha256'], 'model result changed after freeze')
    model = cv.read_json(model_path)
    require(cv.sha(root / 'model-request.json') == selection['request_sha256'] and
        cv.fingerprint(model_config) == selection['model_config_fingerprint'], 'frozen request/model configuration changed')
    original = cv.read_json(root / 'selection.json')
    require(model['extensions']['tilesim.selection.v1'] == original['selection'], 'frozen selection changed')
    if selection_path != root / 'selection.json':
        require(selection.get('fallback_from') == cv.sha(root / 'selection.json') and
            any(p['candidate_id'] == selection['selection']['candidate_id'] and
                p['plan'] == selection['selection']['plan'] for p in model['predictions']), 'fallback Plan not in frozen ranking')
    require(cv.sha(package / 'manifest.json') == search['package_manifest_sha256'], 'prepared package changed')
    manifest, _ = cv.load_bundle(package)
    require(manifest['input_fingerprint'] == selection['input_fingerprint'], 'input identity mismatch')
    require(isinstance(final_command, list) and final_command, 'explicit final evaluator required')
    output.mkdir(parents=True, exist_ok=False)
    selected = selection['selection']; plan = selected['plan']
    # This durable record exists before the first call that permits object
    # generation. It binds that call to the immutable recommendation.
    invocation = dict(stage='compile_selected', selection_sha256=cv.sha(selection_path),
        candidate_id=selected['candidate_id'], plan_fingerprint=cv.fingerprint(plan),
        initial_full_device_compilations=0)
    cv._write(output / 'compile-intent.json', invocation)
    trace_marker('compile_selected_begin', candidate_id=selected['candidate_id'], selection_sha256=invocation['selection_sha256'])
    started = time.monotonic()
    product = cv.compile_plan(package, plan, output / 'candidate', compiler=compiler, emit=emit, verify_sync=True)
    require(cv.sha(selection_path) == invocation['selection_sha256'], 'selection changed during compilation')
    fb_path = output / 'candidate/compiler_feedback.json'; fb = cv.read_json(fb_path)
    context, module = cv._parse(fb['final_vpto_ir'])
    with context: final_program = cv.describe(module)
    cv._write(output / 'program.json', final_program)
    row = dict(candidate_id=selected['candidate_id'], preload_count=plan['loops'][0]['preload_count'],
        plan=plan, program=str((output / 'program.json').resolve()), program_sha256=cv.sha(output / 'program.json'),
        feedback=str(fb_path.resolve()), feedback_sha256=cv.sha(fb_path), final_vpto_ir_sha256=fb['final_vpto_ir_sha256'])
    config = dict(model_config)
    request = dict(schema='pto.cv.exact_fa_proxy_request.v1', original_sha256=search['bindings'].get('original_sha256'),
        candidates=[row], model_target=config['hardware'], timing_model_arch=config['timing_model_arch'],
        allow_profile_proxy=config.get('allow_profile_proxy', False), target_limitations=config.get('target_limitations', []))
    cv._write(output / 'final-request.json', request)
    with (output / 'final-evaluation.log').open('w') as log:
        subprocess.run([*final_command, '--evaluate-only', str(output / 'final-request.json'), str(output / 'final-model')],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    final = cv.read_json(output / 'final-model/result.json')
    require(final.get('evaluation_only') is True and final['request_sha256'] == cv.sha(output / 'final-request.json') and
        len(final['predictions']) == 1 and final['predictions'][0]['plan'] == plan, 'invalid selected-candidate re-evaluation')
    result = dict(schema='pto.cv.model_first_compilation.v1', status='compiled_recommendation',
        candidate_id=selected['candidate_id'], plan_fingerprint=cv.fingerprint(plan),
        selection_sha256=invocation['selection_sha256'], compilation=product,
        selected_device_compilations=1 if emit == 'object' else 0, initial_device_compilations=0,
        initial_prediction_reused=False, re_evaluation_reason='final synchronization and control lowering require physical feedback',
        final_latency_us=final['predictions'][0]['latency_us'], final_result_sha256=cv.sha(output / 'final-model/result.json'),
        g2_status=fb['g2_status'], auto_enable=False, compile_and_reconcile_seconds=time.monotonic() - started)
    cv._write(output / 'model-first-compilation.json', result)
    trace_marker('compile_selected_end', candidate_id=selected['candidate_id'])
    return result


def optimize(source, bindings, target, config, destination, *, compiler, model_command, final_command, preloads=(0, 1, 2, 3), emit='object'):
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cv.export_bundle(source, bindings, target, root / 'package', compiler=compiler)
    search = export_candidates(root / 'package', list(preloads), root / 'search', compiler=compiler)
    frozen = run_model(root / 'search', config, model_command, root / 'search/model')
    model = cv.read_json(root / 'search/model/result.json')
    selected = frozen['selection']; attempts = []
    ranking = model['extensions']['tilesim.selection.v1']['ranked_candidates']
    order = [selected['candidate_id']] + [p for p in ranking if p != selected['candidate_id'] and p != 'p0']
    if selected['action'] == 'baseline': order = [selected['candidate_id']]
    success = None
    for index, identity in enumerate(order[:2]):
        selection_path = root / 'search/selection.json'
        if index:
            row = next(p for p in model['predictions'] if p['candidate_id'] == identity)
            attempt = dict(frozen, fallback_from=cv.sha(selection_path))
            baseline = next(p for p in model['predictions'] if p['candidate_id'] == 'p0')
            attempt['selection'] = dict(selected, candidate_id=identity, plan=row['plan'],
                predicted_gain=1 - row['latency_us'] / baseline['latency_us'])
            selection_path = root / ('attempt-' + str(index) + '-selection.json')
            cv._write(selection_path, attempt)
        try:
            success = compile_selected(root / 'package', root / 'search', root / ('attempt-' + str(index)),
                compiler=compiler, final_command=final_command, model_config=config, emit=emit, selection_path=selection_path)
            attempts.append(dict(candidate_id=identity, status='compiled', result=success)); break
        except RuntimeError as error:
            # Native feasibility diagnostics are distinct from corrupt exchange,
            # identity failure, unknown model cost, or an arbitrary tool failure.
            diagnostic = str(error)
            known_memory_failure = re.search(r'(?:mat|left|right|acc|vec) overflow, requires [0-9]+ bits while [0-9]+ bits available', diagnostic)
            if 'INSUFFICIENT_SLOTS:' not in diagnostic and not known_memory_failure: raise
            attempts.append(dict(candidate_id=identity, status='rejected_infeasible', reason=diagnostic))
    result = dict(schema='pto.cv.model_first_run.v1', status='BASELINE_RETAINED' if selected['action'] == 'baseline'
        else 'compiled_recommendation' if success else 'BASELINE_RETAINED',
        initial_selection_sha256=cv.sha(root / 'search/selection.json'), vf_prepare_count=1,
        candidate_count=len(search['candidates']), device_compilations_before_selection=0,
        attempts=attempts, maximum_optimized_attempts=2, final_result=success,
        elapsed_seconds=time.monotonic() - started, automatic_enablement=False)
    cv._write(root / 'run.json', result)
    return result
