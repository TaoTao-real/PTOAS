# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay the separate fixed-artifact absolute diagnostic, never select a Plan."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from fa_model_first import check_frozen, read, require, sha, write, numpy_golden
from summarize_runtime import duration


def summarize(root, matrix, performance):
    root, matrix = Path(root), Path(matrix)
    manifest = check_frozen(matrix)
    report = read(root / 'results/controls.json')
    require(report['status'] == 'pass' and report['device'] == manifest['device'], 'incomplete/wrong-device controls')
    variants = read(root / 'inputs/control-manifest.json')
    names = ['delivery_p1', 'delivery_p4', 'native_selected']
    require(list(variants) == names and len(report['rows']) == 42, 'incomplete control matrix')
    cases = [('zero', 12345), ('one_v', 12345), ('one_qkv', 12345)] + [('pattern', s) for s in (12345, 20260916, 20261022)]
    expected = []
    for i, order in enumerate((['delivery_p4', 'delivery_p1'], ['delivery_p1', 'delivery_p4'])):
        for mode, seed in cases:
            for variant in order: expected.append((variant, mode, seed, f'g3-{mode}-{seed}-{i}', False))
    expected += [(v, 'pattern', 12345, 'warmup', False) for v in names]
    for i in range(5):
        for v in names[i % 3:] + names[:i % 3]:
            expected.append((v, 'pattern', (12345, 20260916, 20261022)[i % 3], f'profile-{i}', True))
    require([(r['variant'], r['mode'], r['seed'], r['label'], r['profiled']) for r in report['rows']] == expected,
        'control order/seed/profile matrix mismatch')
    reference = read(performance)
    require(reference['status'] == 'pass' and reference['device'] == manifest['device'] and
        reference['matrix_sha256'] == sha(matrix / 'ab_manifest.json'), 'unbound A/B reference')
    inputs = {r['seed']: r['inputs'] for r in reference['rows']}
    copied = root / 'inputs/delivery-reference'
    hashes = read(copied / 'results/local-evidence-hashes.json')
    for v in ('p1', 'p4'):
        for name in ('runner', 'kernel.o', 'kernel.linked.o'):
            require(sha(copied / 'artifacts' / v / name) == hashes[f'artifacts/{v}/{name}'], 'delivered object/launcher changed')
        require(variants['delivery_' + v]['runner_sha256'] == sha(copied / 'artifacts' / v / 'runner'), 'wrong control launcher')
    selected = next(r for r in manifest['rows'] if r['candidate_id'] == manifest['selected'])
    require(variants['native_selected']['runner_sha256'] == sha(matrix / selected['folder'] / 'runner'), 'native control differs from selected B')
    samples, outputs, unique_golden = defaultdict(list), defaultdict(set), set()
    for row in report['rows']:
        parts = Path(row['runtime']).parts
        require('work' in parts, 'nonportable control runtime')
        suffix = parts[parts.index('work'):]
        require(suffix == ('work', 'fa', row['variant'], row['label']), 'unsafe/mismatched runtime path')
        folder = root.joinpath(*suffix)
        require(read(folder / 'result.json') == row and row['symbol'] == variants[row['variant']]['symbol'], 'runtime/symbol changed')
        require(sha(folder / 'output/O.f32') == row['output_sha256'], 'control output changed')
        require(set(row['inputs']) == {'Q.f16', 'K.f16', 'V.f16'}, 'unknown control input file')
        for name, digest in row['inputs'].items(): require(sha(folder / 'output' / name) == digest, 'control input changed')
        if row['mode'] == 'pattern': require(row['inputs'] == inputs[row['seed']], 'controls and A/B use different inputs')
        key = row['variant'], row['mode'], row['seed']
        outputs[key].add(row['output_sha256'])
        if key not in unique_golden:
            numpy_golden(folder / 'output', row['mode']); unique_golden.add(key)
        if row['profiled']:
            for metric, value in duration(folder, row['symbol'], manifest['device']).items():
                require(row[metric] == value, 'raw control profiler changed')
            samples[row['variant']].append(row['duration_us'])
    require(all(len(v) == 1 for v in outputs.values()), 'unstable repeated control outputs')
    summary = {v: dict(median_us=statistics.median(samples[v]), min_us=min(samples[v]),
        max_us=max(samples[v]), samples=len(samples[v])) for v in names}
    require(summary == report['summary'], 'control statistics changed')
    return dict(schema='pto.cv.model_first_absolute_diagnostic.v1', status='pass', device=manifest['device'],
        G3_runs=24, warmups=3, profiles=15, summary=summary,
        delivered_p4_vs_p1_median_gain=1-summary['delivery_p4']['median_us']/summary['delivery_p1']['median_us'],
        native_vs_delivered_p4_median_regression=summary['native_selected']['median_us']/summary['delivery_p4']['median_us']-1,
        independent_golden_cases=len(unique_golden), certificate=False,
        limitation='five samples per route; different compiler/synchronization routes; diagnostic only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'matrix', 'performance', 'output'): p.add_argument(name, type=Path)
    a = p.parse_args(); write(a.output, summarize(a.root, a.matrix, a.performance))


if __name__ == '__main__': main()
