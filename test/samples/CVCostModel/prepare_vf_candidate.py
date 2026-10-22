# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Replay the native prepared-VF path without a model or performance claim.

Use the returned bundle and schedule as inputs to a model adapter. This tool
uses a fixed candidate to test compiler integration. It does not rank plans.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from ptoas import cv_vf as cv
from runtime_fixture import vf_source_text


def prepare(output, compiler, iterations, preload, crossing, symbol, cann_version, emit):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol):
        raise ValueError('symbol prefix must be a C identifier')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source.pto'
    text = vf_source_text(iterations, crossing=crossing, symbol=symbol)
    source.write_text(text)
    bindings = dict(iterations=iterations, tile_shape=[32, 32],
                    launch=dict(aic=1, aiv=2),
                    alias_constraints=[dict(kind='pairwise_disjoint',
                        arguments=[symbol+'_mix_aic:arg0', symbol+'_mix_aic:arg1',
                                   symbol+'_mix_aic:arg2', symbol+'_mix_aic:arg3'])],
                    parameter_bindings={symbol+suffix: ["Q", "K", "V", "O"]
                                        for suffix in ("_mix_aic", "_mix_aiv")})
    target = dict(arch='a5', cann_version=cann_version)
    cv._write(root/'bindings.json', bindings)
    cv._write(root/'target.json', target)
    cv.export_bundle(source, bindings, target, root/'package', compiler=compiler)
    _, program = cv.load_bundle(root/'package')
    loops = {v['id']: preload for v in program['loops'] if not v['prepared_vf_internal']}
    analysis = cv.analyze_schedule(root/'package', loops, root/'analysis', compiler=compiler)
    plan = analysis['minimum_slot_plan']
    plan['candidate_id'] = f'{symbol}-n{iterations}-p{preload}'
    cv._write(root/'plan.json', plan)
    result = cv.compile_plan(root/'package', plan, root/'candidate', compiler=compiler, emit=emit)
    result.update(model_selection='not_run; fixed compiler integration candidate',
                  numerical_validation='not_run', performance_validation='not_run')
    cv._write(root/'replay.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiler', default=shutil.which('ptoas'))
    parser.add_argument('--iterations', type=int, choices=(0, 1, 2, 4, 8), default=4)
    parser.add_argument('--preload', type=int, default=2)
    parser.add_argument('--crossing', action='store_true')
    parser.add_argument('--symbol', default='native_cv_vf')
    parser.add_argument('--cann-version', default='9.2.0')
    parser.add_argument('--emit', choices=('vpto', 'llvm', 'object'), default='llvm')
    args = parser.parse_args()
    if not args.compiler: parser.error('a PTOAS wrapper is required')
    result = prepare(args.output, args.compiler, args.iterations, args.preload,
                     args.crossing, args.symbol, args.cann_version, args.emit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
