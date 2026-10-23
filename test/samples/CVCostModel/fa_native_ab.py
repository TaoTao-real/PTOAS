# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Explicit compatibility import for the frozen, addressed M32 FA input.

This is a bounded frontend import, not an arbitrary FA recognizer. The original
digest is mandatory. Physical row aliases become relative views of symbolic
storage. CV ordering and PlanMemory remain native compiler passes. All four
preload candidates use the same declared storage and flag capacities.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '3c67b7a42b05b7fa1dd705e38892a335dc336bd6f14e7c025a05340e4fa00e9b'
PURE = {'arith.constant', 'arith.index_cast', 'arith.addi', 'arith.muli',
        'arith.subi', 'arith.floordivsi', 'arith.remsi', 'arith.cmpi',
        'arith.select', 'pto.castptr', 'pto.addptr', 'pto.tile_buf_addr',
        'pto.get_subblock_idx', 'pto.get_block_idx', 'pto.get_block_num',
        'pto.make_tensor_view', 'pto.partition_view'}


def opof(op): return getattr(op, 'operation', op)
def children(op): return [opof(c) for c in op.regions[0].blocks[0].operations]
def attrs(op): return {a.name: a.attr for a in op.attributes}
def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations: yield from walk(opof(child))


def import_fa(source, destination, symbol='fa_native_model_ab'):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('unsupported legacy FA source; compatibility import requires the frozen M32 input')
    destination = Path(destination)
    if destination.exists(): raise ValueError('output already exists')
    with ir.Context() as ctx, ir.Location.unknown():
        pto.register_dialect(ctx, load=True)
        oldmod = ir.Module.parse(raw.decode()); oldmod.operation.verify()
        oldfunc = next(o for o in children(oldmod.operation) if o.name == 'func.func')
        sections = {o.name.rsplit('.', 1)[1]: o for o in children(oldfunc) if o.name.startswith('pto.section.')}
        if set(sections) != {'cube', 'vector'}: raise ValueError('unexpected physical section contract')
        new = ir.Module.create()
        new.operation.attributes['pto.target_arch'] = ir.StringAttr.get('a5')
        new.operation.attributes['pto.backend'] = ir.StringAttr.get('vpto')
        new.operation.attributes['pto.cv.legacy_import'] = ir.StringAttr.get(SOURCE_SHA)
        ledger = []

        def integer(t, n):
            return ir.Operation.create('arith.constant', results=[t], attributes={'value': ir.IntegerAttr.get(t, n)}).results[0]
        def const(v):
            owner = opof(v.owner)
            if isinstance(owner, ir.Operation) and owner.name == 'arith.constant':
                try: return ir.IntegerAttr(owner.attributes['value']).value
                except ValueError: return None
            return None
        def bounds(v):
            c = const(v)
            if c is not None: return c, c
            owner = opof(v.owner)
            if isinstance(owner, ir.Block):
                parent = opof(owner.owner)
                if parent.name == 'scf.for' and v == owner.arguments[0]:
                    lo, hi, step = [const(x) for x in parent.operands[:3]]
                    if None not in (lo, hi, step) and step > 0: return lo, hi-step
                return None
            if owner.name == 'arith.index_cast': return bounds(owner.operands[0])
            if owner.name in ('arith.addi', 'arith.muli'):
                a, b = [bounds(x) for x in owner.operands]
                if a is None or b is None: return None
                if owner.name == 'arith.addi': return a[0]+b[0], a[1]+b[1]
                products = [x*y for x in a for y in b]
                return min(products), max(products)
            return None
        def flag(o):
            return ir.IntegerAttr(o.attributes['event_id']).value if o.name in ('pto.set_intra_block', 'pto.wait_intra_block') else None
        def split(ops, event):
            i = next(i for i, o in enumerate(ops) if o.name == 'pto.wait_intra_block' and flag(o) == event)
            return ops[:i], ops[i:]

        for core in ('cube', 'vector'):
            top = children(sections[core]); outer = next(o for o in top if o.name == 'scf.for')
            outerops = children(outer)
            steady = next(o for o in outerops if o.name == 'scf.for' and
                (core == 'cube' or any(c.name == 'pto.wait_intra_block' and flag(c) == 0 for c in children(o))))
            if core == 'cube':
                first0 = outerops[:outerops.index(steady)]
                first, second = split([o for o in children(steady) if o.name != 'scf.yield'], 2)
                tail = []
            else:
                first0, second0 = split(outerops[:outerops.index(steady)], 4)
                first, second = split([o for o in children(steady) if o.name != 'scf.yield'], 4)
                tail = [o for o in outerops[outerops.index(steady)+1:] if o.name != 'scf.yield']
            name = symbol + ('_mix_aic' if core == 'cube' else '_mix_aiv')
            with ir.InsertionPoint(new.body):
                f = ir.Operation.create('func.func', attributes={
                    'sym_name': ir.StringAttr.get(name),
                    'function_type': oldfunc.attributes['function_type'],
                    'pto.entry': ir.UnitAttr.get(),
                    'pto.kernel_kind': ir.Attribute.parse('#pto.kernel_kind<'+core+'>'),
                    'pto.cv.static_stage_graph': ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 32),
                    'pto.cv.no_address_reuse': ir.UnitAttr.get(),
                    'pto.cv.drain_stage_events': ir.UnitAttr.get(),
                    'pto.costmodel.loop_id': ir.StringAttr.get(name+':graph:0')}, regions=1)
                body = f.regions[0].blocks.append(*[v.type for v in oldfunc.regions[0].blocks[0].arguments])
            base = dict(zip(oldfunc.regions[0].blocks[0].arguments, body.arguments))
            roots = {}
            with ir.InsertionPoint(body):
                for old in top:
                    if old.name != 'pto.alloc_tile': continue
                    address = const(old.operands[0]); typ = pto.TileBufType(old.results[0].type)
                    if address is None: raise ValueError('nonconstant root in legacy import')
                    space = str(typ.memory_space)
                    size = 1
                    for d in typ.shape: size *= d
                    size *= {'f16': 2, 'f32': 4}[str(typ.element_type)]
                    # The fixed input's O and normalizer are Q state; alpha is
                    # a KV version consumed by its matching PV update.
                    count = 4 if core == 'vector' and address in (36864, 46592, 47104) and 'vec' in space else 1
                    root = []
                    for replica in range(count):
                        aa = {k:v for k,v in attrs(old).items() if k != 'operandSegmentSizes'}
                        aa['operandSegmentSizes'] = ir.DenseI32ArrayAttr.get([0, 0, 0])
                        aa['pto.costmodel.buffer_id'] = ir.StringAttr.get(f'{name}:source:{space}:{address}:{replica}')
                        aa['pto.cv.legacy_range'] = ir.DenseI64ArrayAttr.get([address, size])
                        shared = ('qk' if address == 0 else 'pv' if address == 8192 else None) if 'vec' in space else ('p' if address == 73728 and 'mat' in space else None)
                        if shared:
                            aa['pto.cv.shared_storage'] = ir.StringAttr.get(shared)
                            aa['pto.cv.shared_bytes'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), size*(3 if shared == 'p' else 1))
                            aa['pto.cv.shared_owner'] = ir.StringAttr.get(symbol+('_mix_aic' if shared == 'p' else '_mix_aiv'))
                        # Row broadcasts use full 256-byte VLDS, then select
                        # lane zero. At the final 32-byte row this reads 224
                        # bytes beyond the logical 512-byte row tile. Keep
                        # that physical tail private, instead of reading a
                        # neighbour allocation or an asynchronously filled FIFO.
                        row_padding = core == 'vector' and 'vec' in space and tuple(typ.shape) == (16,8)
                        allocation_type = ir.Type.parse('!pto.tile_buf<vec, 32x8xf32>') if row_padding else old.results[0].type
                        alloc = ir.Operation.create('pto.alloc_tile', results=[allocation_type], attributes=aa)
                        logical = alloc.results[0]
                        if row_padding:
                            pointer_type = pto.PtrType.get(typ.element_type,ir.Attribute.parse('#pto.address_space<vec>'))
                            pointer = ir.Operation.create('pto.tile_buf_addr',operands=[logical],results=[pointer_type]).results[0]
                            address_value = ir.Operation.create('pto.castptr',operands=[pointer],results=[ir.IntegerType.get_signless(64)]).results[0]
                            logical = ir.Operation.create('pto.alloc_tile',operands=[address_value],results=[old.results[0].type],attributes={
                                'operandSegmentSizes':ir.DenseI32ArrayAttr.get([1,0,0]),'pto.cv.borrowed':ir.UnitAttr.get()}).results[0]
                        root.append(logical)
                        ledger.append(dict(id=ir.StringAttr(aa['pto.costmodel.buffer_id']).value,
                            core=core, source_range=[address, size], replica=replica,
                            ownership='pipe_backing' if shared else 'local', shared=shared,
                            physical_bytes=1024 if row_padding else size,read_tail_padding=row_padding))
                    roots[old.results[0]] = dict(values=root, address=address, size=size, space=space)

                class Clone:
                    def __init__(self, token, phase=None):
                        self.token, self.phase = token, phase
                        self.map = dict(base)
                        self.map[outer.regions[0].blocks[0].arguments[0]] = integer(ir.IndexType.get(), token//8)
                        self.map[steady.regions[0].blocks[0].arguments[0]] = integer(ir.IndexType.get(), token%8)
                        self.inside_vf = False
                        for value, row in roots.items():
                            ix = (token%4 if row['address'] == 47104 else token//8) if len(row['values']) > 1 else 0
                            self.map[value] = row['values'][ix]
                    def tag(self, op):
                        if self.phase and not self.inside_vf:
                            op.attributes['pto.cv.token'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), self.token)
                            op.attributes['pto.cv.phase'] = ir.StringAttr.get(self.phase)
                        return op
                    def value(self, v):
                        if v not in self.map:
                            o = opof(v.owner)
                            if not isinstance(o, ir.Operation) or o.name not in PURE:
                                raise ValueError('unbound non-pure legacy capture: '+str(v))
                            self.clone(o)
                        return self.map[v]
                    def clone(self, old):
                        aa = attrs(old)
                        if old.name == 'pto.vecscope':
                            contents = children(old)
                            if len(contents) == 1 and contents[0].name in ('pto.tmuls','pto.texp','pto.tcvt','pto.tadd'):
                                tiles = [pto.TileBufType(v.type) for v in contents[0].operands if pto.TileBufType.isinstance(v.type)]
                                if tiles and all(tuple(t.shape) == (16,128) and tuple(t.valid_shape) in ((),(16,128)) for t in tiles):
                                    # Same confirmed VMI instances as the earlier
                                    # MR289 experiment. Fallback scopes stay closed.
                                    return self.clone(contents[0])
                        if old.name == 'pto.alloc_tile':
                            span = bounds(old.operands[0]); typ = pto.TileBufType(old.results[0].type)
                            matches = [(v, row) for v,row in roots.items() if str(typ.memory_space) == row['space'] and span is not None and row['address'] <= span[0] and span[1] < row['address']+row['size']]
                            if len(matches) != 1: raise ValueError('unproven legacy row alias: '+str(old))
                            original, row = matches[0]; storage = self.map[original]
                            ptrtype = pto.PtrType.get(typ.element_type, ir.Attribute.parse('#pto.address_space<vec>'))
                            ptr = self.tag(ir.Operation.create('pto.tile_buf_addr', operands=[storage], results=[ptrtype])).results[0]
                            relative = self.tag(ir.Operation.create('arith.subi', operands=[self.value(old.operands[0]), integer(ir.IntegerType.get_signless(64), row['address'])], results=[ir.IntegerType.get_signless(64)])).results[0]
                            unit = integer(ir.IntegerType.get_signless(64), {'f16':2,'f32':4}[str(typ.element_type)])
                            offset = self.tag(ir.Operation.create('arith.divsi', operands=[relative,unit], results=[ir.IntegerType.get_signless(64)])).results[0]
                            offset = self.tag(ir.Operation.create('arith.index_cast', operands=[offset], results=[ir.IndexType.get()])).results[0]
                            ptr = self.tag(ir.Operation.create('pto.addptr', operands=[ptr,offset], results=[ptrtype])).results[0]
                            address = self.tag(ir.Operation.create('pto.castptr', operands=[ptr], results=[ir.IntegerType.get_signless(64)])).results[0]
                            aa['pto.cv.borrowed'] = ir.UnitAttr.get()
                            result = self.tag(ir.Operation.create(old.name, operands=[address], results=[old.results[0].type], attributes=aa))
                            self.map[old.results[0]] = result.results[0]; return result
                        eid = flag(old)
                        if eid is not None and eid%16 in (2, 3):
                            slot = self.token%3; ready = 2 if slot == 0 else 4+2*slot
                            aa['event_id'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), eid//16*16+ready+(eid%16-2))
                        if old.name == 'scf.for' and not self.inside_vf:
                            lo,hi,step = [const(v) for v in old.operands[:3]]
                            if None in (lo,hi,step) or step < 1 or len(old.operands) != 3: raise ValueError('unsupported outer legacy copy loop')
                            for i in range(lo,hi,step):
                                self.map[old.regions[0].blocks[0].arguments[0]] = self.tag(ir.Operation.create('arith.constant', results=[ir.IndexType.get()], attributes={'value':ir.IntegerAttr.get(ir.IndexType.get(),i)})).results[0]
                                for child in children(old):
                                    if child.name != 'scf.yield': self.clone(child)
                            return
                        operands = [self.value(v) for v in old.operands]
                        # P is a shared three-slot FIFO; its relative selection
                        # is common to every candidate and guarded by its flags.
                        if old.name == 'pto.tile_buf_addr' and old.operands[0] in roots and roots[old.operands[0]]['address'] == 73728:
                            result = self.tag(ir.Operation.create(old.name, operands=operands, results=[v.type for v in old.results], attributes=aa))
                            index = self.tag(ir.Operation.create('arith.constant', results=[ir.IndexType.get()], attributes={'value':ir.IntegerAttr.get(ir.IndexType.get(),(self.token%3)*4096)})).results[0]
                            selected = self.tag(ir.Operation.create('pto.addptr', operands=[result.results[0],index], results=[old.results[0].type]))
                            self.map[old.results[0]] = selected.results[0]; return selected
                        if old.name == 'pto.tmov' and old.operands[0] in roots and roots[old.operands[0]]['address'] == 73728:
                            row = roots[old.operands[0]]; typ = pto.TileBufType(old.operands[0].type)
                            ptrtype = pto.PtrType.get(typ.element_type, ir.Attribute.parse('#pto.address_space<mat>'))
                            ptr = self.tag(ir.Operation.create('pto.tile_buf_addr', operands=[operands[0]], results=[ptrtype])).results[0]
                            idx = self.tag(ir.Operation.create('arith.constant', results=[ir.IndexType.get()], attributes={'value':ir.IntegerAttr.get(ir.IndexType.get(),(self.token%3)*4096)})).results[0]
                            ptr = self.tag(ir.Operation.create('pto.addptr', operands=[ptr,idx], results=[ptrtype])).results[0]
                            addr = self.tag(ir.Operation.create('pto.castptr', operands=[ptr], results=[ir.IntegerType.get_signless(64)])).results[0]
                            view = self.tag(ir.Operation.create('pto.alloc_tile',operands=[addr],results=[old.operands[0].type],attributes={'operandSegmentSizes':ir.DenseI32ArrayAttr.get([1,0,0]),'pto.cv.borrowed':ir.UnitAttr.get()})).results[0]
                            operands[0] = view
                        result = self.tag(ir.Operation.create(old.name, operands=operands, results=[v.type for v in old.results], attributes=aa, regions=len(old.regions)))
                        self.map.update(zip(old.results,result.results))
                        saved = self.inside_vf
                        if old.name == 'pto.vecscope': self.inside_vf = True
                        for a,b in zip(old.regions,result.regions):
                            for oldblock in a.blocks:
                                bb = b.blocks.append(*[v.type for v in oldblock.arguments]); self.map.update(zip(oldblock.arguments,bb.arguments))
                                with ir.InsertionPoint(bb):
                                    for child in oldblock.operations: self.clone(opof(child))
                        self.inside_vf = saved
                        return result

                for op in top[:top.index(outer)]:
                    if op.name == 'pto.set_intra_block':
                        for token in range(3) if flag(op)%16 == 3 else (0,): Clone(token).clone(op)
                for token in range(32):
                    # Constants used to bind outer IVs are also graph nodes.
                    for phase in ('prefix','suffix'):
                        begin = len(list(body.operations))
                        clone = Clone(token,phase)
                        if core == 'cube': selected = ((first0 if token%8 == 0 else [])+first) if phase == 'prefix' else second
                        else: selected = (first0 if token%8 == 0 else first) if phase == 'prefix' else (second0 if token%8 == 0 else second)
                        for op in selected: clone.clone(op)
                        if core == 'vector' and phase == 'suffix' and token%8 == 7:
                            for op in tail: clone.clone(op)
                        for op in list(body.operations)[begin:]: clone.tag(opof(op))
                for op in top[top.index(outer)+1:]:
                    for token in range(3) if flag(op) is not None and flag(op)%16 == 3 else (0,): Clone(token).clone(op)
                ir.Operation.create('func.return')
        new.operation.verify()
        text = new.operation.get_asm(print_generic_op_form=True)+'\n'
        ir.Module.parse(text).operation.verify(); destination.write_text(text)
        report = dict(schema='pto.cv.legacy_fa_import.v1',source_sha256=SOURCE_SHA,
            output_sha256=hashlib.sha256(text.encode()).hexdigest(),
            shape=[128,1024,128],tile=[32,128,128],launch=dict(aic=1,aiv=2),
            communication_capacity=dict(qk=1,p=3,pv=1),alpha_versions=4,q_states=4,
            address_policy='relative_alias_import_then_native_disjoint_plan_memory',
            scope='exact frozen M32 FA source only',buffers=ledger)
        destination.with_suffix('.import.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('destination');p.add_argument('--symbol',default='fa_native_model_ab')
    a=p.parse_args();print(json.dumps(import_fa(a.source,a.destination,a.symbol),indent=2))
