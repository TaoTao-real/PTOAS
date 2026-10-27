# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay model-first FA G3/A/B, with initial and postcompile errors separate."""
import argparse
import statistics
from pathlib import Path

from fa_model_first import check_frozen, read, require, sha, write
from summarize_fa_ab import paired, runtime_path
from summarize_runtime import duration


def summarize(root, g3_path, perf_path, *, g3_runtime_root=None, runtime_root=None):
    root = Path(root); manifest = check_frozen(root)
    g3, perf = read(g3_path), read(perf_path); selected = manifest['selected']; ids = ['p0', selected]
    for report in (g3, perf):
        require(report['status'] == 'pass' and report['device'] == 0 and
            report['matrix_sha256'] == sha(root / 'ab_manifest.json'), 'mismatched run evidence')
    require(len(g3['rows']) == 25 and len(perf['rows']) == 42, 'incomplete G3/performance matrix')
    require(g3['rows'][0]['stage'] == 'cold-first' and g3['rows'][0]['candidate_id'] == selected, 'selected candidate was not cold-first')
    expected_g3 = [(selected, 'pattern', 12345, -1, 'cold-first')]
    cases = [('zero', 12345), ('one_v', 12345), ('one_qkv', 12345)] + [('pattern', s) for s in (12345, 20260916, 20261022)]
    for repeat, order in enumerate(([selected, 'p0'], ['p0', selected])):
        for mode, seed in cases:
            for identity in order: expected_g3.append((identity, mode, seed, repeat, 'g3'))
    require([(r['candidate_id'], r['case'], r['seed'], r['block'], r['stage']) for r in g3['rows']] == expected_g3, 'G3 order/seed mismatch')
    require([r['candidate_id'] for r in perf['rows'][:2]] == ids and
        all(r['stage'] == 'warmup' for r in perf['rows'][:2]), 'invalid warmup matrix')
    for block in range(20):
        group = perf['rows'][2 + 2*block:4 + 2*block]
        require([r['candidate_id'] for r in group] == (ids if block % 2 == 0 else list(reversed(ids))) and
            all(r['block'] == block and r['stage'] == 'profile' and r['case'] == 'pattern' and
                r['seed'] == (12345, 20260916, 20261022)[block % 3] for r in group), 'paired order/seed mismatch')
        require(group[0]['inputs'] == group[1]['inputs'] and group[0]['output_sha256'] == group[1]['output_sha256'], 'paired input/output mismatch')
    symbols = {r['candidate_id']: r['symbol'] for r in manifest['rows']}
    for report, rebase in ((g3, g3_runtime_root), (perf, runtime_root)):
        for row in report['rows']:
            folder = runtime_path(row, rebase)
            require(read(folder / 'result.json') == row and row['status'] == 'pass' and row['symbol'] == symbols[row['candidate_id']], 'runtime row/symbol mismatch')
            require(sha(folder / 'output/O.f32') == row['output_sha256'], 'runtime output changed')
            for name, expected in row['inputs'].items(): require(sha(folder / 'output' / name) == expected, 'runtime input changed')
            if row['stage'] == 'profile':
                for key, value in duration(folder, row['symbol'], 0).items(): require(row[key] == value, 'raw profiler changed')
    initial = read(root / 'optimization/search/model/result.json')
    initial = {r['candidate_id']: r['latency_us'] for r in initial['predictions']}
    attempt = next(r for r in manifest['optimization']['attempts'] if r['status'] == 'compiled')
    actual = {identity: [r['duration_us'] for r in perf['rows'][2:] if r['candidate_id'] == identity] for identity in ids}
    baseline_final = read(root / 'baseline/final-model/result.json')['predictions'][0]['latency_us']
    final_predictions = {'p0': baseline_final, selected: attempt['result']['final_latency_us']}
    comparison = paired(actual['p0'], actual[selected]); rows = []
    for identity in ids:
        measured = statistics.median(actual[identity])
        rows.append(dict(candidate_id=identity, initial_prediction_us=initial[identity],
            final_feedback_prediction_us=final_predictions[identity], measured_median_us=measured,
            measured_min_us=min(actual[identity]), measured_max_us=max(actual[identity]),
            initial_relative_error=initial[identity]/measured - 1,
            final_feedback_relative_error=final_predictions[identity]/measured - 1))
    prediction_gain = 1 - initial[selected]/initial['p0']
    return dict(schema='pto.cv.fa_model_first_acceptance.v1',
        benefit_status='AB_BENEFIT_PASS' if comparison['benefit_pass'] else 'AB_BENEFIT_FAIL',
        G4='G4_NOT_CERTIFIED', G4_reason='PR timing proxy for DT; no target-matched certification',
        selected=selected, G2='restricted exact-FA certificate', G3_runs=25, performance_samples=40,
        comparison=comparison, rows=rows, initial_predicted_gain=prediction_gain,
        gain_prediction_minus_measured=prediction_gain - comparison['mean_gain'],
        process_order=manifest['process_order'], optimization_elapsed_seconds=manifest['optimization']['elapsed_seconds'],
        candidate_analysis_count=manifest['optimization']['candidate_count'],
        automatic_enablement=False, matrix_sha256=sha(root / 'ab_manifest.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'g3', 'performance', 'output'): p.add_argument(name, type=Path)
    p.add_argument('--runtime-root', type=Path); p.add_argument('--g3-runtime-root', type=Path)
    a = p.parse_args(); write(a.output, summarize(a.root, a.g3, a.performance,
        runtime_root=a.runtime_root, g3_runtime_root=a.g3_runtime_root))


if __name__ == '__main__': main()
