"""Fixed-case physical VPTO materializer; no compiler pass or machine emission.

Input contract: immutable M32 S128/S1024/D128, BlockDim=1, two AIV lanes.
This prototype statically expands only the CV iteration schedule; VF loops and
instructions remain intact. Not a generic compiler transform.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from checkpoint import seal, validate
from pipe_protocol import schedule, verify

SOURCE_SHA = None  # Required immutable input digest, supplied by the experiment manifest.
PURE = {'arith.constant', 'arith.index_cast', 'arith.addi', 'arith.muli',
        'arith.subi', 'arith.floordivsi', 'arith.remsi', 'arith.cmpi',
        'arith.select', 'pto.castptr', 'pto.addptr', 'pto.get_subblock_idx',
        'pto.get_block_idx', 'pto.get_block_num'}

def opof(op): return getattr(op, 'operation', op)
def children(op): return [opof(x) for x in op.regions[0].blocks[0].operations]
def attrs(op): return {x.name: x.attr for x in op.attributes}
def walk(op):
    yield op
    for reg in op.regions:
        for block in reg.blocks:
            for ch in block.operations: yield from walk(opof(ch))
def event(op):
    if op.name in ('pto.set_intra_block', 'pto.wait_intra_block'):
        return ir.IntegerAttr(op.attributes['event_id']).value
def split_wait(ops, eid):
    index = next(i for i, o in enumerate(ops) if o.name == 'pto.wait_intra_block' and event(o) == eid)
    return ops[:index], ops[index:]
def constant(value):
    owner = opof(value.owner)
    if isinstance(owner, ir.Block) or owner.name != 'arith.constant': return None
    try: return ir.IntegerAttr(owner.attributes['value']).value
    except ValueError: return None

def interval(value):
    """Conservative address range for original pure address expressions."""
    owner = opof(value.owner)
    if isinstance(owner, ir.Block):
        parent = opof(owner.owner)
        if parent.name == 'scf.for' and value == owner.arguments[0]:
            lo, hi, step = [constant(x) for x in parent.operands[:3]]
            if None not in (lo, hi, step) and step > 0: return lo, hi-step
        return None
    n = constant(value)
    if n is not None: return n, n
    if owner.name == 'arith.index_cast': return interval(owner.operands[0])
    if owner.name in ('arith.addi', 'arith.muli'):
        a, b = [interval(x) for x in owner.operands]
        if a is None or b is None: return None
        if owner.name == 'arith.addi': return a[0]+b[0], a[1]+b[1]
        products = [x*y for x in a for y in b]
        return min(products), max(products)
    return None

def int_const(typ, n):
    return ir.Operation.create('arith.constant', results=[typ],
        attributes={'value': ir.IntegerAttr.get(typ, n)}).results[0]

def signature(scope):
    # Address arithmetic is deliberately rebound. All physical vector ops,
    # operand types, attributes, and loop topology must remain identical.
    return [(o.name, [str(v.type) for v in o.operands],
             [str(v.type) for v in o.results],
             {a.name:str(a.attr) for a in o.attributes})
            for o in walk(scope) if o.name.startswith('pto.') or o.name in ('scf.for','scf.yield')]

class Cloner:
    def __init__(self, bindings, token, preload, pcap, source_vfs, ledger):
        self.map = dict(bindings)
        self.token, self.q, self.k = token, token//8, token%8
        self.preload, self.pcap = preload, pcap
        self.source_vfs, self.ledger = source_vfs, ledger
        self.relocations = []

    def relocation_ranges(self,typ):
        if ', ub>' in typ:
            return [(36864,45056,65536+self.q*8192),
                    (46080,46592,98304+self.q*512),
                    (46592,47104,100352+self.q*512),
                    (47104,47616,102400+(self.token%self.preload)*512)]
        if ', l1>' in typ:return [(73728,81920,73728+(self.token%self.pcap)*8192)]
        return []

    def rebase(self,value,lo,target):
        delta=target-lo
        if not delta:return value
        c=int_const(value.type,delta)
        return ir.Operation.create('arith.addi',operands=[value,c],results=[value.type]).results[0]

    def value(self, value):
        if value not in self.map:
            owner = opof(value.owner)
            if isinstance(owner, ir.Block) or owner.name not in PURE:
                raise ValueError(f'unbound non-pure capture: {value}')
            self.clone(owner)
        return self.map[value]

    def clone(self, old):
        if old.name == 'pto.vecscope':
            inside = set(walk(old))
            for nested in walk(old):
                for value in nested.operands:
                    owner = opof(value.owner)
                    if isinstance(owner, ir.Block):
                        owner = opof(owner.owner)
                    if owner not in inside:
                        self.value(value)
        operands = [self.value(x) for x in old.operands]
        attributes = attrs(old)
        if event(old) is not None:
            eid = event(old)
            lane, local = divmod(eid,16)
            if local in (2,3):
                slot = self.token % self.pcap
                ready = 2 if slot == 0 else 4+2*slot
                attributes['event_id'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), lane*16+ready+(local-2))
        if old.name == 'pto.castptr':
            typ = str(old.results[0].type)
            bounds = interval(old.operands[0])
            ranges = self.relocation_ranges(typ)
            if bounds is None and ranges:
                raise ValueError(f'unclassified local pointer: {old}')
            for lo, hi, target in ranges:
                if bounds and lo <= bounds[0] and bounds[1] < hi:
                    operands[0]=self.rebase(operands[0],lo,target)
                    self.relocations.append({'old_range':list(bounds),'new_range':[x+target-lo for x in bounds] if isinstance(target,int) else 'dynamic checked slot','type':typ})
                    break
        new = ir.Operation.create(old.name, operands=operands,
            results=[v.type for v in old.results], attributes=attributes,
            regions=len(old.regions), loc=old.location)
        self.map.update(zip(old.results, new.results))
        for oldreg, newreg in zip(old.regions, new.regions):
            for oldblock in oldreg.blocks:
                block = newreg.blocks.append(*[a.type for a in oldblock.arguments])
                self.map.update(zip(oldblock.arguments, block.arguments))
                with ir.InsertionPoint(block):
                    for child in oldblock.operations: self.clone(opof(child))
        if old.name == 'pto.vecscope':
            assert signature(old) == signature(new), 'physical VF instruction/loop structure changed'
            self.ledger.append(self.source_vfs[old])
        return new

def materialize(source, destination, preload):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA: raise ValueError('input differs from validated fixed baseline')
    if preload not in (1,2,3,4): raise ValueError('currently validated candidate depths: 1,2,3,4')
    # Fixed three P slots across candidates: resource layout is held constant.
    caps = dict(qk=1,p=3,pv=1)
    protocol = verify(32,preload,caps)
    assert protocol['completed']
    with ir.Context() as ctx, ir.Location.unknown():
        pto.register_dialect(ctx,load=True)
        oldmod = ir.Module.parse(raw.decode())
        validate(oldmod)
        newmod = ir.Module.create()
        for a in oldmod.operation.attributes: newmod.operation.attributes[a.name] = a.attr
        for key,val in [('pto.cv.vf_state','physical-v2'),('pto.cv.vf_arch','a5'),('pto.cv.vf_build_level','level3')]:
            newmod.operation.attributes[key] = ir.StringAttr.get(val)
        vfs = {o:f'vf_{i:02d}' for i,o in enumerate(o for o in walk(oldmod.operation) if o.name=='pto.vecscope')}
        stages, ledger = [], []
        for oldsub in children(oldmod.operation):
            oldfunc = next(o for o in children(oldsub) if o.name=='func.func')
            vector = 'vector' in str(oldsub.attributes['pto.kernel_kind'])
            oldbody = oldfunc.regions[0].blocks[0]
            top = list(map(opof,oldbody.operations))
            outer = next(o for o in top if o.name=='scf.for')
            outerops = children(outer)
            if vector:
                steady = next(o for o in outerops if o.name=='scf.for' and
                              any(c.name=='pto.wait_intra_block' and event(c)==0 for c in children(o)))
                middle = outerops.index(steady)
                sm0, up0 = split_wait(outerops[:middle],4)
                sm1, up1 = split_wait([o for o in children(steady) if o.name!='scf.yield'],4)
                suffix = [o for o in outerops[middle+1:] if o.name!='scf.yield']
                def vf_count(ops):
                    return sum(o.name == 'pto.vecscope' for root in ops for o in walk(root))
                expected_vfs = 4 * (vf_count(sm0) + vf_count(up0) + vf_count(suffix)) + 28 * (vf_count(sm1) + vf_count(up1))
                prefix = []
            else:
                steady = next(o for o in outerops if o.name=='scf.for')
                prefix = outerops[:outerops.index(steady)]
                qk, pv = split_wait([o for o in children(steady) if o.name!='scf.yield'],2)
            with ir.InsertionPoint(newmod.body):
                sub = ir.Operation.create('builtin.module',attributes=attrs(oldsub),regions=1)
                subblock = sub.regions[0].blocks.append()
            with ir.InsertionPoint(subblock):
                func = ir.Operation.create('func.func',attributes=attrs(oldfunc),regions=1)
                body = func.regions[0].blocks.append(*[v.type for v in oldbody.arguments])
            base = dict(zip(oldbody.arguments,body.arguments))
            with ir.InsertionPoint(body):
                # Initial/final flag protocol remains separate from stage scheduling.
                def boundary(op, token):
                    Cloner(base,token,preload,3,vfs,ledger).clone(op)
                for op in top[:top.index(outer)]:
                    if op.name=='pto.set_intra_block':
                        for t in (range(3) if event(op)%16==3 else (0,)): boundary(op,t)
                for role, token in schedule(32,preload):
                    bindings = dict(base)
                    bindings[outer.regions[0].blocks[0].arguments[0]] = int_const(ir.IndexType.get(),token//8)
                    bindings[steady.regions[0].blocks[0].arguments[0]] = int_const(ir.IndexType.get(),token%8)
                    clone = Cloner(bindings,token,preload,3,vfs,ledger)
                    start = len(list(body.operations)); vstart = len(ledger)
                    if vector:
                        selected = (sm0 if token%8==0 else sm1) if role=='first' else (up0 if token%8==0 else up1)
                    else:
                        selected = ((prefix if token%8==0 else [])+qk) if role=='first' else pv
                    for op in selected: clone.clone(op)
                    if vector and role=='second' and token%8==7:
                        for op in suffix: clone.clone(op)
                    stages.append({'core':'vector' if vector else 'cube','role':role,'token':token,
                        'q':token//8,'kv':token%8,'top_op_begin':start,'top_op_end':len(list(body.operations)),
                        'vfs':ledger[vstart:],'relocations':clone.relocations})
                for op in top[top.index(outer)+1:]:
                    for t in (range(3) if event(op) is not None and event(op)%16==3 else (0,)): boundary(op,t)
        seal(newmod)
        newmod.operation.verify()
        text = str(newmod)+'\n'
        # Reparse to catch serialization/SSA errors independently of construction.
        ir.Module.parse(text).operation.verify()
        assert len(ledger)==expected_vfs, (len(ledger), expected_vfs)
        counts=Counter(o.name for o in walk(newmod.operation))
        assert counts['pto.mad_raw']==64
        assert counts['pto.copy_matrix_cc_to_ub']==64
        assert counts['pto.copy_ubuf_to_gm']==4
        dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        report={'schema':'ptoas.native_cv_candidate.v1','preload':preload,'input_sha256':SOURCE_SHA,
            'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'shape':[128,1024,128],
            'tile':[32,128,128],'block_dim':1,'mix_block_dim':2,'protocol':protocol,
            'vf_instances_per_lane':len(ledger),'source_vf_counts':dict(Counter(ledger)),
            'physical_vf_instruction_structure_preserved':True,
            'ub_high_water_bytes':102400+preload*512,'l1_high_water_bytes':73728+3*8192,
            'implementation_scope':'fixed-case physical CV schedule expansion; VF loops preserved',
            'machine_code_generated':False,'stages':stages,'op_counts':dict(counts)}
        dest.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report[k] for k in ['preload','output_sha256','vf_instances_per_lane','ub_high_water_bytes','l1_high_water_bytes']}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('destination');ap.add_argument('--preload',type=int,required=True)
    ap.add_argument('--expected-source-sha256',required=True)
    a=ap.parse_args();SOURCE_SHA=a.expected_source_sha256;materialize(a.source,a.destination,a.preload)
