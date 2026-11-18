# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Single-core multibuffer exchange, independent of CV preload search.

The caller must establish disjoint per-iteration GM writes and no aliasing
between input/output arguments. v1 supports one bounded loop and one pure
Cube/Vector function. Native parsing and annotation are lazy imports so the
plan contract can be tested without a compiler installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REQUEST = 'pto.single_core.request.v1'
PLAN = 'pto.single_core.plan.v1'
SCHEDULE = 'iteration_order.async_pipes.v1'


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _native():
    from ptoas.cv_vf import _parse, _walk, _text
    from ptoas.mlir import ir
    return _parse, _walk, _text, ir


def prepare(source, *, independent_iterations=False):
    """Export a verified canonical input with stable buffer/loop identifiers."""
    if independent_iterations is not True:
        raise ValueError('caller must establish independent iterations and non-aliasing GM arguments')
    parse, walk, text, ir = _native()
    context, module = parse(source)
    with context:
        functions = [o for o in walk(module.operation) if o.name == 'func.func']
        loops = [o for o in walk(module.operation) if o.name == 'scf.for']
        if len(functions) != 1 or len(loops) != 1:
            raise ValueError('expected one pure function and one loop')
        function, loop = functions[0], loops[0]
        kind_attr = function.attributes['pto.kernel_kind'] if 'pto.kernel_kind' in function.attributes else module.operation.attributes['pto.kernel_kind']
        kind = {'#pto.kernel_kind<vector>':'vector', '#pto.kernel_kind<cube>':'cube'}.get(str(kind_attr))
        if kind is None or len(loop.operands) != 3:
            raise ValueError('expected a pure function and no loop-carried SSA')
        module.operation.attributes['pto.kernel_kind']=kind_attr
        bounds=[]
        for v in loop.operands:
            owner=v.owner
            if owner.name != 'arith.constant':
                raise ValueError('static loop bounds required')
            bounds.append(ir.IntegerAttr(owner.attributes['value']).value)
        lower,upper,step=bounds
        if step<=0 or upper<=lower or len(range(lower,upper,step))>256:
            raise ValueError('expected 1..256 iterations')
        loop_id=text(function,'sym_name')+':loop0'
        loop.attributes['pto.costmodel.loop_id']=ir.StringAttr.get(loop_id)
        allocations=[o for o in walk(module.operation) if o.name=='pto.alloc_tile']
        if not allocations:
            raise ValueError('no local allocations')
        loop_operations=set(walk(loop))
        for i,op in enumerate(allocations):
            if op in loop_operations:
                raise ValueError('v1 requires allocations outside the candidate loop')
            # Physical addresses and borrowed/shared storage cannot be cloned.
            if len(op.operands) or any(k in op.attributes for k in ('pto.cv.borrowed','pto.cv.shared_storage')):
                raise ValueError('requires unaddressed, owned allocations')
            op.attributes['pto.costmodel.buffer_id']=ir.StringAttr.get(f'{loop_id}:buffer{i}')
        module.operation.verify()
        canonical=module.operation.get_asm(use_local_scope=True)
    buffer_ids={}
    for line in canonical.splitlines():
        if 'pto.alloc_tile' not in line: continue
        match=re.search(r'(%[\w.$-]+)\s*=\s*pto.alloc_tile.*pto.costmodel.buffer_id\s*=\s*"([^"]+)"',line)
        if not match:
            raise ValueError('unsupported allocation printer; buffer identity cannot be recovered')
        buffer_ids[f'{kind}::{match[1]}']=match[2]
    return dict(schema=REQUEST, ir=canonical, input_fingerprint=fingerprint(canonical),
        independent_iterations=True, loop_id=loop_id, trip_count=len(range(lower,upper,step)),
        buffer_ids=buffer_ids)


def validate_plan(request, plan):
    if request.get('schema')!=REQUEST or request.get('independent_iterations') is not True:
        raise ValueError('invalid single-core request')
    if fingerprint(request['ir'])!=request['input_fingerprint']:
        raise ValueError('stale request')
    expected={'schema','input_fingerprint','loop_id','trip_count','schedule_semantics','buffers','predicted_latency_us'}
    if not isinstance(plan,dict) or set(plan)!=expected or plan['schema']!=PLAN:
        raise ValueError('unsupported plan schema')
    for field in ('input_fingerprint','loop_id','trip_count'):
        if plan[field]!=request[field]: raise ValueError('stale plan '+field)
    if plan['schedule_semantics']!=SCHEDULE:
        raise ValueError('unsupported scheduling semantics')
    n=plan['trip_count']
    if type(n) is not int or not 1<=n<=256:
        raise ValueError('invalid trip count')
    latency=plan['predicted_latency_us']
    if type(latency) not in (int,float) or not math.isfinite(latency) or latency<=0:
        raise ValueError('invalid predicted latency')
    if not isinstance(plan['buffers'],list): raise ValueError('buffers must be a list')
    expected_ids=set(request['buffer_ids'].values())
    seen=set()
    for row in plan['buffers']:
        if not isinstance(row,dict) or set(row)!={'buffer_id','count','slots'}:
            raise ValueError('unsupported buffer fields')
        identity=row['buffer_id']; count=row['count']; slots=row['slots']
        if not isinstance(identity,str) or identity not in expected_ids or identity in seen:
            raise ValueError('unknown or duplicate buffer identity')
        seen.add(identity)
        if type(count) is not int or not 1<=count<=16:
            raise ValueError('native buffer count must be 1..16')
        if not isinstance(slots,list) or len(slots)!=n or any(type(s) is not int or not 0<=s<count for s in slots):
            raise ValueError('invalid per-iteration slots')
    if seen!=expected_ids: raise ValueError('missing buffer configurations')


def annotate(request, plan):
    """Bind counts AND iteration-slot choices to the exact modeled input."""
    validate_plan(request,plan)
    parse,walk,text,ir=_native()
    context,module=parse(request['ir'])
    with context:
        found=set(); loops=0
        by_id={b['buffer_id']:b for b in plan['buffers']}
        for op in walk(module.operation):
            if op.name=='scf.for' and 'pto.costmodel.loop_id' in op.attributes:
                if text(op,'pto.costmodel.loop_id')!=plan['loop_id']:
                    raise ValueError('unexpected loop identity')
                op.attributes['pto.pipeline.single_core']=ir.UnitAttr.get()
                loops+=1
            if op.name=='pto.alloc_tile' and 'pto.costmodel.buffer_id' in op.attributes:
                identity=text(op,'pto.costmodel.buffer_id'); row=by_id[identity]
                if identity in found: raise ValueError('duplicate IR buffer identity')
                found.add(identity)
                op.attributes['pto.pipeline.multi_buffer_count']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),row['count'])
                op.attributes['pto.pipeline.iteration_slots']=ir.ArrayAttr.get([
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64),slot) for slot in row['slots']])
        if found!=set(by_id) or loops!=1:
            raise ValueError('IR does not match the requested buffer/loop identities')
        module.operation.verify()
        return module.operation.get_asm(use_local_scope=True)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    export=sub.add_parser('prepare');export.add_argument('input');export.add_argument('output')
    export.add_argument('--independent-iterations',action='store_true')
    consume=sub.add_parser('annotate');consume.add_argument('request');consume.add_argument('plan');consume.add_argument('output')
    a=p.parse_args(argv)
    if a.command=='prepare':
        result=prepare(Path(a.input).read_text(),independent_iterations=a.independent_iterations)
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    else:
        Path(a.output).write_text(annotate(json.loads(Path(a.request).read_text()),json.loads(Path(a.plan).read_text())))
    return 0

if __name__=='__main__': raise SystemExit(main())
