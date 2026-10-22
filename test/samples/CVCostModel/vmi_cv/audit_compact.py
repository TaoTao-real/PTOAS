"""Compare compact CV execution with independently expanded fixed-case IR.

Interpret scalar control and VF loop bounds; vector values are opaque tokens.
Compare physical instruction traces and evaluated addresses. Unknown control
or local pointer addresses are errors, not zero-cost or zero-address values.
This is restricted equivalence evidence, not a general asynchronous G2 proof.
"""
import hashlib
import json
import sys
from pathlib import Path
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from fixture_native import attrs, children, opof, signature, walk
from checkpoint import validate


def traces(module):
    result = {}
    for sub in children(module.operation):
        core = 'vector' if 'vector' in str(sub.attributes['pto.kernel_kind']) else 'cube'
        func = next(o for o in children(sub) if o.name == 'func.func')
        for lane in (range(2) if core == 'vector' else range(1)):
            env = {v: ('argument', i) for i, v in enumerate(func.regions[0].blocks[0].arguments)}
            trace = []

            def value(v):
                if v not in env:
                    raise ValueError(f'unbound scalar capture: {v}')
                return env[v]

            def execute(ops):
                for o in ops:
                    name = o.name
                    if name == 'pto.vecscope':
                        trace.append(('scope.begin',))
                        execute(children(o))
                        trace.append(('scope.end',))
                        continue
                    args = [value(v) for v in o.operands]
                    if name == 'scf.for':
                        lo, hi, step = args[:3]
                        if not all(isinstance(x, int) for x in (lo, hi, step)) or step <= 0:
                            raise ValueError('unknown or invalid scalar loop domain')
                        carry = args[3:]; block = o.regions[0].blocks[0]
                        for i in range(lo, hi, step):
                            env.update(zip(block.arguments, [i, *carry]))
                            carry = execute([opof(x) for x in block.operations])
                        env.update(zip(o.results, carry)); continue
                    if name == 'scf.if':
                        if not isinstance(args[0], (bool, int)):
                            raise ValueError('unknown scalar branch condition')
                        reg = o.regions[0 if args[0] else 1]
                        vals = execute([opof(x) for x in reg.blocks[0].operations]) if len(reg.blocks) else []
                        env.update(zip(o.results, vals)); continue
                    if name in ('scf.yield', 'func.return'): return args
                    if name == 'arith.constant':
                        a = o.attributes['value']
                        try: out = ir.IntegerAttr(a).value
                        except ValueError: out = str(a)
                    elif name == 'arith.index_cast': out = args[0]
                    elif name in ('arith.addi', 'arith.muli', 'arith.subi', 'arith.floordivsi', 'arith.remsi'):
                        a, b = args
                        out = {'arith.addi': lambda: a+b, 'arith.muli': lambda: a*b,
                               'arith.subi': lambda: a-b, 'arith.floordivsi': lambda: a//b,
                               'arith.remsi': lambda: a%b}[name]()
                    elif name == 'arith.cmpi':
                        pred = ir.IntegerAttr(o.attributes['predicate']).value; a, b = args
                        out = [a == b, a != b, a < b, a <= b, a > b, a >= b,
                               a < b, a <= b, a > b, a >= b][pred]
                    elif name == 'arith.select': out = args[1] if args[0] else args[2]
                    elif name == 'pto.get_subblock_idx': out = lane
                    elif name == 'pto.get_block_idx': out = 0
                    elif name == 'pto.get_block_num': out = 1
                    elif name == 'pto.castptr': out = ('pointer', str(o.results[0].type), args[0])
                    elif name == 'pto.addptr': out = ('offset', *args)
                    else:
                        if len(o.regions): raise ValueError(f'unsupported control: {name}')
                        trace.append((name, {k: str(v) for k, v in attrs(o).items()}, args))
                        for i, v in enumerate(o.results): env[v] = ('result', name, len(trace), i)
                        continue
                    if len(o.results) > 1: raise ValueError(f'unsupported multiple pure results: {name}')
                    if len(o.results): env[o.results[0]] = out
                return []

            execute(children(func))
            result[f'{core}:{lane}'] = trace
    return result


def audit(expanded, compact):
    with ir.Context() as ctx:
        pto.register_dialect(ctx, load=True)
        modules = [ir.Module.parse(Path(p).read_text()) for p in (expanded, compact)]
        for m in modules: m.operation.verify(); validate(m)
        a, b = map(traces, modules)
        if a != b:
            for key in a:
                for i, (x, y) in enumerate(zip(a[key], b[key])):
                    if x != y: raise ValueError(f'compact trace differs: {key} event {i}: {x!r} != {y!r}')
                if len(a[key]) != len(b[key]): raise ValueError(f'trace length differs: {key}')
            raise ValueError('execution domains differ')
        report = {'schema': 'ptoas.fixed-cv.equivalence.v1', 'passed': True,
                  'expanded_sha256': hashlib.sha256(Path(expanded).read_bytes()).hexdigest(),
                  'compact_sha256': hashlib.sha256(Path(compact).read_bytes()).hexdigest(),
                  'trace_events': {k: len(v) for k, v in a.items()},
                  'scope': 'scalar execution, physical instruction/effect traces and addresses; vector values opaque; not general G2'}
        Path(compact).with_suffix('.equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))


if __name__ == '__main__': audit(*sys.argv[1:])
