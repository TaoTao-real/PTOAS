# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compiler-owned, address-free execution description for prepared CV search.

This interpreter accepts only bounded structured operations. It does not assign
hardware storage. VF representative bindings are private simulator inputs and
are never returned as allocations or Plan fields.
"""
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math

SPACES = {1: 'GM', 2: 'L1', 3: 'L0A', 4: 'L0B', 5: 'L0C', 6: 'UB'}
FEATURES = ['logical_events.v1', 'vf_relative_bindings.v1', 'completion_contract.v1', 'logical_channels.v1']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Pointer:
    root: str
    space: str
    bits: int
    offset: int = 0
    shape: tuple = ()
    valid: tuple = ()
    extent: int = 0
    shared: bool = False
    slots: int = 1


@dataclass(frozen=True)
class View:
    pointer: Pointer
    shape: tuple
    strides: tuple


@dataclass(frozen=True)
class Channel:
    identity: int
    direction: int
    slot_bytes: int
    capacity: int
    backing: Pointer


class LogicalEvents:
    def __init__(self, program, bindings):
        self.program, self.bindings = program, bindings
        self.functions = {f['id']: f for f in program['functions']}
        self.buffers = {b['value']: b for b in program['buffers']}
        self.events, self.vfs = [], {}
        self.flags = defaultdict(lambda: defaultdict(list))
        self.channels, self.transactions = {}, defaultdict(lambda: defaultdict(list))
        self.channel_counts = defaultdict(int)
        self.reservations = {}
        for f in program['functions']:
            for o in f['operations']:
                if o['name'] == 'pto.reserve_buffer':
                    a = o['attributes']
                    require(a.get('autoAlloc') is True and 'base' not in a, 'fixed reservation is not logical search storage')
                    self.reservations[(f['id'], a['name'])] = Pointer('reservation:' + f['id'] + ':' + a['name'],
                        SPACES[a['location']['value']], 8, extent=a['size'], shared=True)

    def add(self, source, cost, accesses=()):
        i = len(self.events)
        self.events.append(dict(id=i, source_operation=source['id'], op=source['name'],
            core=self.core, lane=self.lane, predecessors=[] if self.last is None else [self.last],
            phase=source['attributes'].get('pto.cv.phase'), iteration=source['attributes'].get('pto.cv.token'),
            cost=cost, accesses=list(accesses)))
        self.last = i
        return i

    def domain(self, p, lane=None):
        if p.space == 'GM': return p.root
        if p.space == 'UB': return 'aiv' + str(self.lane if lane is None else lane) + ':' + p.root
        return 'aic:' + p.root

    def access(self, p, size, kind, lane=None):
        require(0 <= p.offset < p.offset + size <= p.extent, 'logical access exceeds storage: ' + p.root)
        return dict(object=self.domain(p, lane), space=p.space, begin=p.offset, end=p.offset + size, access=kind)

    def copy(self, op, src, dst, shape, source_span=None, destination_span=None, split=False):
        require(src.bits == dst.bits or src.space == 'L0C', 'unsupported transfer dtype conversion')
        n = math.prod(shape)
        reads = [self.access(src, source_span or n * src.bits // 8, 'read')]
        writes = ([self.access(dst, n * dst.bits // 16, 'write', lane) for lane in (0, 1)]
                  if split else [self.access(dst, destination_span or n * dst.bits // 8, 'write')])
        self.add(op, dict(kind='copy', source_space=src.space, destination_space=dst.space,
            shape=list(shape), source_bits=src.bits, destination_bits=dst.bits), reads + writes)

    def vf(self, op, function, arguments):
        roots, representatives, alias = {}, [], []
        cursor = 0
        for arg in arguments:
            if not isinstance(arg, Pointer):
                require(type(arg) in (int, float, bool), 'unresolved VF scalar')
                representatives.append(arg); alias.append(arg); continue
            require(arg.space == 'UB', 'unsupported prepared VF memory space')
            if arg.root not in roots:
                cursor = (cursor + 255) // 256 * 256
                roots[arg.root] = (len(roots), cursor, arg.extent)
                cursor += arg.extent + 4096
            ordinal, base, extent = roots[arg.root]
            representatives.append(dict(kind='pointer', bits=arg.bits, byte=base + arg.offset))
            alias.append(dict(root=ordinal, offset=arg.offset, bits=arg.bits,
                shape=list(arg.shape), valid=list(arg.valid), extent=extent))
        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items() if not k.startswith(('pto.cv.', 'pto.costmodel.'))}
            if isinstance(x, list): return [clean(v) for v in x]
            if isinstance(x, str): return x.replace(function['id'], 'VF')
            return x
        key = digest(dict(function=clean(function), conditions=alias,
            representative_policy='disjoint_256B_aligned.v1'))
        self.vfs.setdefault(key, dict(key=key, function=function, arguments=representatives,
            conditions=alias, representative_policy='simulator_only; not hardware addresses'))
        # Footprints are resolved by the VF worker, then translated back to these
        # logical roots. The model must retain this provenance in its report.
        mapping = [dict(root=root, begin=base, end=base + extent,
            domain='aiv' + str(self.lane) + ':' + root) for root, (_, base, extent) in roots.items()]
        self.add(op, dict(kind='vf', key=key, bindings=mapping))

    def block(self, block):
        for identity in block['operations']:
            op = self.ops[identity]
            value = self.op(op)
            if op['name'] in ('scf.yield', 'func.return'): return value
        return []

    def op(self, op):
        name, attrs = op['name'], op['attributes']
        result = op['results']; ids = [r['id'] for r in result]
        if name == 'arith.constant': self.env[ids[0]] = attrs['value']; return
        args = [self.env[v] for v in op['operands']]
        if name in ('func.return', 'scf.yield'): return args
        if name == 'func.call':
            f = self.functions[attrs['callee']['symbol']]
            require(f['prepared_vf'], 'unknown outer call')
            self.vf(op, f, args); return
        if name == 'pto.reserve_buffer':
            self.env[ids[0]] = self.reservations[(self.function, attrs['name'])]; return
        if name == 'pto.import_reserved_buffer':
            self.env[ids[0]] = self.reservations[(attrs['peer_func']['symbol'], attrs['name'])]; return
        if name == 'pto.initialize_l2l_pipe':
            require(len(args) == 1 and isinstance(args[0], Pointer) and attrs['dir_mask'] in (1, 2) and
                attrs['slot_size'] > 0 and attrs['slot_num'] > 0 and not attrs.get('nosplit'), 'unsupported channel geometry')
            ch = Channel(attrs['__pto.frontend_id'], attrs['dir_mask'], attrs['slot_size'], attrs['slot_num'], args[0])
            require(ch.slot_bytes * ch.capacity <= ch.backing.extent, 'channel exceeds backing')
            signature = (ch.direction, ch.slot_bytes, ch.capacity, ch.backing.root)
            require(ch.identity not in self.channels or self.channels[ch.identity] == signature, 'channel peers disagree')
            self.channels[ch.identity] = signature; self.env[ids[0]] = ch; return
        if name == 'pto.declare_tile':
            t = result[0]['type']
            self.env[ids[0]] = Pointer('unbound:' + ids[0], SPACES[t['memory_space']['value']],
                t['element']['bits'], shape=tuple(t['shape']), valid=tuple(t['valid_shape']))
            return
        if name in ('pto.tpush', 'pto.tpop', 'pto.tfree'):
            ch = args[-1]; require(isinstance(ch, Channel) and attrs.get('split', 1) == 1, 'unsupported channel split')
            action = name[5:]; count_key = (self.core, ch.identity, action)
            token = self.channel_counts[count_key]; self.channel_counts[count_key] += 1
            pointer = replace(ch.backing, offset=(token % ch.capacity) * ch.slot_bytes)
            if name == 'pto.tpush':
                source = args[0]; require(isinstance(source, Pointer), 'unknown channel payload')
                pointer = replace(pointer, bits=source.bits, shape=source.shape, valid=source.valid)
                if ch.direction == 1:
                    require(self.lane is None and source.space == 'L0C' and pointer.space == 'UB', 'unsupported C2V source')
                    self.copy(op, source, pointer, source.valid, split=True)
                else:
                    require(self.lane in (0, 1) and source.space == 'UB' and pointer.space == 'L1', 'unsupported V2C source')
                    payload = math.prod(source.valid) * source.bits // 8
                    require(2 * payload <= ch.slot_bytes, 'split payload exceeds slot')
                    pointer = replace(pointer, offset=pointer.offset + self.lane * payload)
                    self.copy(op, source, pointer, source.valid)
            elif name == 'pto.tpop':
                old = args[0]
                require(isinstance(old, Pointer) and (old.root.startswith('unbound:') or old.root == ch.backing.root), 'borrowed pop has an unrelated storage root')
                self.env[op['operands'][0]] = replace(pointer, bits=old.bits, shape=old.shape, valid=old.valid)
            eid = self.add(op, dict(kind='control', cycles=1, channel=ch.identity, action=action, token=token))
            self.transactions[(ch.identity, token)][action].append(eid)
            return
        if name in ('pto.alloc_tile', 'pto.alloc_multi_tile'):
            t = result[0]['type']
            slots = t['count'] if t['kind'] == 'multi_tile' else 1
            if t['kind'] == 'multi_tile': t = t['slot_type']
            if args:
                require(len(args) == 1 and isinstance(args[0], Pointer), 'physical/unbound local allocation is not search input')
                p = args[0]
            else:
                b = self.buffers.get(ids[0]); require(b is not None and b['bytes'] is not None, 'unknown logical allocation')
                p = Pointer(b.get('storage_id', b['id']), SPACES[t['memory_space']['value']],
                    t['element']['bits'], extent=b['bytes'] * slots, shared=b['ownership'] == 'pipe_backing', slots=slots)
            self.env[ids[0]] = replace(p, bits=t['element']['bits'], shape=tuple(t['shape']), valid=tuple(t['valid_shape']))
            return
        if name == 'pto.multi_tile_get':
            pointer, slot = args
            require(isinstance(pointer, Pointer) and type(slot) is int and 0 <= slot < pointer.slots, 'invalid version slot')
            self.env[ids[0]] = replace(pointer, offset=pointer.offset + slot * pointer.extent // pointer.slots)
            return
        if name == 'pto.make_tensor_view':
            segments = attrs['operandSegmentSizes']; rank = segments[1]
            require(segments == [1, rank, rank] and rank > 0, 'unknown tensor view')
            self.env[ids[0]] = View(args[0], tuple(args[1:1 + rank]), tuple(args[1 + rank:])); return
        if name == 'pto.partition_view':
            view = args[0]; rank = len(view.shape)
            require(len(args) == 1 + rank * 2, 'unknown partition view')
            delta = sum(i * s for i, s in zip(args[1:1 + rank], view.strides)) * view.pointer.bits // 8
            self.env[ids[0]] = View(replace(view.pointer, offset=view.pointer.offset + delta), tuple(args[1 + rank:]), view.strides); return
        if name in ('pto.tload', 'pto.tstore'):
            view, tile = (args[0], args[1]) if name == 'pto.tload' else (args[1], args[0])
            require(len(args) == 2 and isinstance(view, View), 'unsupported tile load/store mode')
            require(attrs.get('layout', {}).get('value') in (0, 1), 'unsupported transfer layout')
            if name == 'pto.tstore':
                require(attrs.get('atomicType') == dict(enum='atomic_type', value='none') and
                    attrs.get('reluPreMode') == dict(enum='relu_pre_mode', value='none'), 'unsupported store semantics')
            shape = tuple(view.shape[-2:])
            require(all(d == 1 for d in view.shape[:-2]), 'multidimensional transfer not supported')
            span = (1 + sum((d - 1) * s for d, s in zip(view.shape, view.strides))) * view.pointer.bits // 8
            if name == 'pto.tload': self.copy(op, view.pointer, tile, shape, source_span=span)
            else: self.copy(op, tile, view.pointer, shape, destination_span=span)
            return
        if name == 'pto.tmov':
            require(len(args) == 2, 'unsupported TMOV mode')
            require(attrs.get('reluPreMode') == dict(enum='relu_pre_mode', value='none'), 'unsupported TMOV relu')
            self.copy(op, args[0], args[1], args[0].valid); return
        if name == 'pto.tmatmul':
            require(len(args) == 3, 'unsupported matmul form')
            require(attrs.get('accPhase') == dict(enum='acc_phase', value='unspecified'), 'unsupported matmul phase')
            a, b, c = args
            m, k = a.valid; k2, n = b.valid
            require(k == k2 and c.valid == (m, n), 'incompatible matmul shape')
            self.add(op, dict(kind='matmul', m=m, n=n, k=k, bits=a.bits, output_bits=c.bits),
                [self.access(a, m * k * a.bits // 8, 'read'), self.access(b, k * n * b.bits // 8, 'read'),
                 self.access(c, m * n * c.bits // 8, 'write')]); return
        if name == 'pto.mte_l0c_ub':
            src, dst = args[:2]
            require(attrs.get('mode') == dict(enum='acc_store_mode', value='nz2nd') and
                attrs.get('dst_mode') == dict(enum='acc_store_ub_dst_mode', value='split_m'), 'unsupported Fixpipe layout/split')
            require(src.space == 'L0C' and dst.space == 'UB' and len(src.valid) == 2, 'unknown Fixpipe shape')
            require(len(args) == 6 and tuple(args[2:4]) == src.valid and
                args[4] == src.valid[0] and args[5] == src.valid[1], 'unsupported Fixpipe dimensions/strides')
            self.copy(op, src, dst, src.valid, split=True); return
        if name == 'pto.mte_ub_l1':
            require(len(args) == 6, 'unknown L2L contract')
            src, dst, count, length, ss, ds = args
            require(src.space == 'UB' and dst.space == 'L1', 'unsupported L2L direction')
            require(all(type(v) is int and v >= 0 for v in (count, length, ss, ds)) and count * length > 0, 'invalid L2L transfer')
            self.copy(op, src, dst, (count, length * 32 // (src.bits // 8)),
                source_span=((count - 1) * (length + ss) + length) * 32,
                destination_span=((count - 1) * (length + ds) + length) * 32); return
        if name in ('pto.set_intra_block', 'pto.wait_intra_block'):
            flag = attrs['event_id'] + (16 * self.lane if self.lane is not None else 0)
            eid = self.add(op, dict(kind='control', cycles=1, flag=flag))
            self.flags[flag]['set' if name.startswith('pto.set') else 'wait'].append(eid); return
        if name == 'scf.for':
            lo, hi, step = args[:3]
            require(all(type(v) is int for v in (lo, hi, step)) and step > 0 and len(range(lo, hi, step)) <= 1024, 'unbounded control')
            block = op['regions'][0][0]; carried = args[3:]
            for i in range(lo, hi, step):
                self.env.update(zip([v['id'] for v in block['arguments']], [i, *carried])); carried = self.block(block)
            self.env.update(zip(ids, carried)); return
        if name == 'scf.if': self.block(op['regions'][0 if args[0] else 1][0]); return
        if name in ('arith.index_cast', 'arith.extsi', 'arith.extui', 'arith.trunci'):
            self.env[ids[0]] = args[0]
        elif name in ('arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi', 'arith.divui', 'arith.floordivsi', 'arith.remsi', 'arith.remui'):
            a, b = args
            self.env[ids[0]] = a + b if name.endswith('addi') else a - b if name.endswith('subi') else a * b if name.endswith('muli') else a % b if 'rem' in name else a // b
        elif name == 'arith.select': self.env[ids[0]] = args[1] if args[0] else args[2]
        elif name == 'arith.cmpi':
            a, b = args
            self.env[ids[0]] = {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[attrs['predicate']]
        elif name in ('pto.get_subblock_idx', 'pto.get_block_idx', 'pto.get_block_num'):
            self.env[ids[0]] = self.lane if name == 'pto.get_subblock_idx' else 1 if name == 'pto.get_block_num' else 0
        elif name == 'pto.tile_buf_addr': self.env[ids[0]] = args[0]
        elif name == 'pto.castptr':
            require(isinstance(args[0], Pointer), 'physical pointer constant is not a logical search binding')
            t = result[0]['type']
            self.env[ids[0]] = replace(args[0], bits=t['element']['bits']) if t['kind'] == 'pointer' else args[0]
        elif name == 'pto.addptr': self.env[ids[0]] = replace(args[0], offset=args[0].offset + args[1] * args[0].bits // 8)
        elif name in ('pto.barrier', 'pto.set_flag', 'pto.wait_flag'): pass
        else: raise ValueError('unsupported logical operation: ' + name)
        self.add(op, dict(kind='control', cycles=1))

    def build(self):
        entries = [f for f in self.functions.values() if f['entry']]
        require(len(entries) == 2, 'logical search requires paired C/V entries')
        pointers = {value for f in entries for a, value in zip(f['arguments'],
            self.bindings['parameter_bindings'][f['id']]) if a['type']['kind'] == 'pointer'}
        require(any(c.get('kind') == 'pairwise_disjoint' and pointers <= set(c.get('arguments', []))
            for c in self.bindings.get('alias_constraints', [])), 'unknown GM alias constraints')
        for f in entries:
            self.function = f['id']
            self.ops = {o['id']: o for o in f['operations']}
            nested = {v for o in f['operations'] for reg in o['regions'] for block in reg for v in block['operations']}
            kind = f['kernel_kind']['value']
            require(kind in ('cube', 'vector'), 'unknown core')
            binding = self.bindings['parameter_bindings'][f['id']]
            require(len(binding) == len(f['arguments']), 'missing argument bindings')
            for lane in ((None,) if kind == 'cube' else (0, 1)):
                self.lane, self.core, self.last = lane, 'aic' if lane is None else 'aiv' + str(lane), None
                arguments = []
                for a, v in zip(f['arguments'], binding):
                    t = a['type']
                    if t['kind'] == 'pointer':
                        shape = self.bindings['argument_shapes'][v]
                        arguments.append(Pointer(v, 'GM', t['element']['bits'], extent=math.prod(shape) * t['element']['bits'] // 8))
                    else: arguments.append(v)
                self.env = dict(zip([a['id'] for a in f['arguments']], arguments))
                for op in f['operations']:
                    if op['id'] not in nested: self.op(op)
        for flag, pairs in self.flags.items():
            require(len(pairs['set']) == len(pairs['wait']), 'unbalanced channel flag: ' + str(flag))
            for source, target in zip(pairs['set'], pairs['wait']): self.events[target]['predecessors'].append(source)
        for (channel, token), transaction in self.transactions.items():
            direction, _, capacity, _ = self.channels[channel]
            expected = dict(push=1 if direction == 1 else 2, pop=2 if direction == 1 else 1, free=2 if direction == 1 else 1)
            require(all(len(transaction[k]) == n for k, n in expected.items()), 'incomplete channel lifecycle')
            for pop in transaction['pop']: self.events[pop]['predecessors'].extend(transaction['push'])
            for release in transaction['free']:
                matches = [p for p in transaction['pop'] if self.events[p]['core'] == self.events[release]['core']]
                require(len(matches) == 1, 'channel release has no matching pop')
                self.events[release]['predecessors'].extend(matches)
            if token >= capacity:
                older = self.transactions[(channel, token - capacity)]['free']
                require(older, 'missing channel slot release')
                for push in transaction['push']: self.events[push]['predecessors'].extend(older)
        return dict(schema='pto.cv.logical_events.v1', required_features=FEATURES,
            events=self.events, vfs=list(self.vfs.values()),
            assumptions=['scalar/address/control issue: one cycle per retained operation',
                'metadata views and symbolic allocations have no standalone execution cost',
                'per-core event streams serialize completion; hardware intra-core overlap is not estimated',
                'VF representative bases: disjoint, 256B aligned; bank/layout conditions require final feedback'])
