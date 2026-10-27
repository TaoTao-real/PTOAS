# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Structured physical VF -> pinned VfSimulator canonical contract.

Run this worker in a separate process from TileSim: both source repositories
use top-level ``api`` and ``core`` packages. Unknown instructions are errors.
"""
import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Aff:
    constant: int
    terms: tuple = ()
    def __add__(self, other):
        other=aff(other); terms=dict(self.terms)
        for k,v in other.terms: terms[k]=terms.get(k,0)+v
        return Aff(self.constant+other.constant,tuple(sorted((k,v) for k,v in terms.items() if v)))
    __radd__=__add__
    def __mul__(self, other):
        other=aff(other)
        if self.terms and other.terms: raise ValueError('non-affine VF address')
        if other.terms: return other*self.constant
        return Aff(self.constant*other.constant,tuple((k,v*other.constant) for k,v in self.terms))
    __rmul__=__mul__
    def __sub__(self, other): return self+aff(other)*-1
    def divide(self,n):
        if self.constant%n or any(v%n for _,v in self.terms): raise ValueError('unaligned affine VF address')
        return Aff(self.constant//n,tuple((k,v//n) for k,v in self.terms))
    def text(self): return str(self.constant)+''.join(f'+({v})*{k}' for k,v in self.terms)


def aff(x): return x if isinstance(x,Aff) else Aff(x)
def number(x):
    if isinstance(x,Aff):
        if x.terms: raise ValueError('expected a resolved scalar')
        return x.constant
    return x


@dataclass(frozen=True)
class Pointer:
    byte: object
    bits: int


@dataclass(frozen=True)
class Mask:
    count: int
    value: str


class Bridge:
    def __init__(self, function, arguments):
        from api.frontend.adapter_ir import AdapterValue,AdapterInstruction,AdapterLoop,AdapterAlias,AdapterMembar,AdapterMemoryAccess,AdapterProgram
        self.Value,self.Inst,self.Loop,self.Alias,self.Membar,self.Access,self.Program=AdapterValue,AdapterInstruction,AdapterLoop,AdapterAlias,AdapterMembar,AdapterMemoryAccess,AdapterProgram
        self.function=function;self.ops={o['id']:o for o in function['operations']};self.types={v['id']:v['type'] for o in function['operations'] for v in o['results']}
        self.types.update({a['id']:a['type'] for a in function['arguments']})
        for o in function['operations']:
            for region in o['regions']:
                for block in region: self.types.update({a['id']:a['type'] for a in block['arguments']})
        self.env={a['id']:Pointer(aff(v['byte']),v['bits']) if isinstance(v,dict) and v.get('kind')=='pointer' else v for a,v in zip(function['arguments'],arguments)}
        if len(self.env)!=len(arguments): raise ValueError('VF argument contract differs')
        self.values={};self.body=[];self.serial=0;self.ranges={};self.approximations=set();self.counts={};self.footprints=[]
    def fresh(self,prefix): self.serial+=1;return prefix+str(self.serial)
    def value(self,storage,dtype,shape=()):
        name=self.fresh('v');self.values[name]=self.Value(name,storage,dtype,shape);return name
    def dtype(self,t):
        if t['kind']=='float':return 'fp'+str(t['bits'])
        if t['kind']=='integer':return 'int'+str(t['bits'])
        raise ValueError('unsupported VF element type: '+str(t))
    def reg(self,vid):
        t=self.types[vid]
        if t['kind']=='vector_register': return self.value('Register',self.dtype(t['element']),(t['lanes'],))
        if t['kind']=='mask': return self.value('PredicateRegister','bool')
        if t['kind']=='integer': return self.value('Scalar',self.dtype(t))
        raise ValueError('unsupported VF register type')
    def scalar(self,value,dtype='int32'):
        return self.value('Scalar',dtype)
    def bounds(self,x):
        x=aff(x);lo=hi=x.constant
        for k,v in x.terms:
            a,b=self.ranges[k];lo+=min(a*v,b*v);hi+=max(a*v,b*v)
        return lo,hi
    def emit(self,name,src,dst,form=None,accesses=(),attributes=None):
        from api.frontend.instruction_catalog import DEFAULT_INSTRUCTION_CATALOG
        name=DEFAULT_INSTRUCTION_CATALOG.specialize(name,form)
        self.body.append(self.Inst(name,src,dst,form=form,memory_accesses=tuple(accesses),attributes=attributes or {}))
    def block(self,block):
        for oid in block['operations']:
            o=self.ops[oid]; result=self.op(o)
            if o['name'] in ('scf.yield','func.return'):return result
        return []
    def loop(self,o,x):
        lo,hi,step=map(number,x[:3]); n=len(range(lo,hi,step))
        if n<1 or n>1024 or step<1:raise ValueError('unsupported VF loop domain')
        block=o['regions'][0][0]; initial=x[3:]
        saved=copy.deepcopy((self.env,self.values,self.body,self.serial,self.ranges,self.approximations,self.counts,self.footprints))
        self.env.update(zip([v['id'] for v in block['arguments']],[lo,*initial]))
        yielded=self.block(block)
        delta=[]
        for a,b in zip(initial,yielded):
            if isinstance(a,Pointer):delta.append(number(aff(b.byte)-a.byte))
            elif isinstance(a,(str,Mask)):delta.append(None)
            else:delta.append(number(aff(b)-a))
        self.env,self.values,self.body,self.serial,self.ranges,self.approximations,self.counts,self.footprints=saved
        iv=self.fresh('i');self.ranges[iv]=(0,n-1);symbol=Aff(0,((iv,1),))
        self.env[block['arguments'][0]['id']]=lo+step*symbol
        for argument,a,d in zip(block['arguments'][1:],initial,delta):
            self.env[argument['id']]=Pointer(a.byte+symbol*d,a.bits) if isinstance(a,Pointer) else a if d is None else aff(a)+symbol*d
        outer=self.body;self.body=[];yielded=self.block(block);body=self.body;self.body=outer
        for result,a,d,b in zip(o['results'],initial,delta,yielded):
            if isinstance(a,Pointer):out=Pointer(a.byte+n*d,a.bits)
            elif isinstance(a,str):body.append(self.Alias(a,b));out=a
            elif isinstance(a,Mask):raise ValueError('unsupported carried predicate')
            else:out=aff(a)+n*d
            self.env[result['id']]=out
        self.body.append(self.Loop(n,1,body,loop_id=self.fresh('loop'),induction_variable=iv))
    def op(self,o):
        name=o['name'];attrs=o['attributes'];r=[v['id'] for v in o['results']]
        self.counts[name]=self.counts.get(name,0)+1
        if name=='arith.constant':self.env[r[0]]=attrs['value'];return
        x=[self.env[v] for v in o['operands']]
        if name in ('func.return','scf.yield'):return x
        if name=='pto.vecscope':return self.block(o['regions'][0][0])
        if name in ('arith.index_cast','arith.extsi','arith.extui','arith.trunci'):self.env[r[0]]=x[0]
        elif name in ('arith.addi','arith.subi','arith.muli','arith.divsi','arith.divui','arith.floordivsi'):
            a,b=x
            self.env[r[0]]=aff(a)+b if name.endswith('addi') else aff(a)-b if name.endswith('subi') else aff(a)*b if name.endswith('muli') else aff(a).divide(number(b))
        elif name=='pto.castptr':
            t=self.types[r[0]]
            self.env[r[0]]=Pointer(x[0].byte if isinstance(x[0],Pointer) else aff(x[0]),t['element']['bits']) if t['kind']=='pointer' else x[0]
        elif name=='pto.alloc_tile':
            if len(x)!=1:raise ValueError('unbound VF local allocation')
            t=self.types[r[0]];self.env[r[0]]=Pointer(x[0].byte if isinstance(x[0],Pointer) else aff(x[0]),t['element']['bits'])
        elif name=='pto.tile_buf_addr':self.env[r[0]]=x[0]
        elif name=='pto.addptr':self.env[r[0]]=Pointer(x[0].byte+aff(x[1])*(x[0].bits//8),x[0].bits)
        elif name=='scf.for':self.loop(o,x)
        elif name=='pto.plt_b32':
            lo,hi=self.bounds(x[0]);count=64 if lo>=64 else lo if lo==hi else None
            if count is None or count<0:raise ValueError('unknown predicate footprint')
            pred=self.reg(r[0]);rem=self.reg(r[1]);s=self.scalar(x[0])
            self.emit('PLT',[s],[pred,rem],form='b32')
            self.env[r[0]]=Mask(min(count,64),pred);self.env[r[1]]=aff(x[0])-64 if lo>=64 else 0
        elif name=='pto.pset_b32':
            pattern=attrs['pattern'];counts={'PAT_ALL':64,'PAT_VL1':1,'PAT_VL2':2,'PAT_VL4':4,'PAT_VL8':8,'PAT_VL16':16,'PAT_VL32':32}
            if pattern not in counts:raise ValueError('unknown PSET pattern '+str(pattern))
            dst=self.reg(r[0]);self.emit('PSET_B32',[],[dst],form='b32');self.env[r[0]]=Mask(counts[pattern],dst)
        elif name=='pto.vneg':
            mask=next((v for v in x if isinstance(v,Mask)),None)
            if mask is None:raise ValueError('missing VNEG predicate')
            dst=self.reg(r[0]);form='fp'+str(self.types[r[0]]['element']['bits'])
            # Explicit arithmetic-family timing proxy, not a kernel rewrite.
            self.emit('VMULS',[x[0],self.scalar(-1,form),mask.value],[dst],form=form)
            self.approximations.add('VNEG timing: VMULS arithmetic-family sign-change proxy; primitive calibration pending')
            self.env[r[0]]=dst
        elif name=='pto.vsstb':
            source,p,block_stride,repeat_stride,mask=x
            bs,rs=number(block_stride),number(repeat_stride)
            if not isinstance(p,Pointer) or not isinstance(mask,Mask) or p.bits not in (16,32) or bs<1 or rs<0:
                raise ValueError('unsupported VSSTB contract')
            unit=p.bits//8;per_block=32//unit;blocks=(mask.count+per_block-1)//per_block
            for i in range(blocks):
                low,high=self.bounds(aff(p.byte)+i*bs*32)
                size=min(per_block,mask.count-i*per_block)*unit
                self.footprints.append(dict(access='write',begin=low,end=high+size))
            # VSSTB uses the catalog's raw b16 form. Keep its source register
            # dtype; an FP32 value must not become a numeric BF16 conversion.
            span=((blocks-1)*bs*32+min(per_block,mask.count-(blocks-1)*per_block)*unit)//2
            mem=self.value('UB','bf16',(span,));access=self.Access(mem,'write',aff(p.byte).divide(2).text(),span,'VSSTB_CONFIG')
            self.emit('VSSTB',[source,mask.value],[mem],form='b16',accesses=[access],attributes={'config':(bs<<16)|rs})
            self.approximations.add('VSSTB: catalog b16 timing and contiguous LSU envelope; exact scattered footprints exported')
            if r:self.env[r[0]]=Pointer(p.byte+rs*32,p.bits)
        elif name in ('pto.vlds','pto.vsts','pto.vsldb'):
            load=name!='pto.vsts';pi=0 if load else 1;p=x[pi]
            if not isinstance(p,Pointer):raise ValueError('unknown VF memory base')
            mask=next((v for v in reversed(x) if isinstance(v,Mask)),None)
            unit=p.bits//8;mode=attrs.get('dist','NORM');post=len(r)==(2 if load else 1)
            if name=='pto.vsldb':
                if mask is None or mask.count!=1:raise ValueError('VSLDB mapping only supports the checked VL1 row-copy form')
                offset=aff(0);span=1
                self.approximations.add('VSLDB VL1: exact one-element footprint; VLDS LSU timing family')
            else:
                offset=aff(0) if post else aff(x[pi+1]);span=1 if mode in ('BRC_B32','BRC_B16','1PT_B32','1PT_B16') else (256//unit if load else mask.count)
            mem=self.value('UB','fp'+str(p.bits),(span,))
            low,high=self.bounds(aff(p.byte).divide(unit)+offset)
            self.footprints.append(dict(access='read' if load else 'write',begin=low*unit,end=(high+span)*unit))
            access=self.Access(mem,'read' if load else 'write',(aff(p.byte).divide(unit)+offset).text(),span,mode)
            if load:
                dst=self.reg(r[0]);self.emit('VLDS',[mem],[dst],form='fp'+str(p.bits),accesses=[access]);self.env[r[0]]=dst
            else:
                if mask is None:raise ValueError('missing VF store predicate')
                self.emit('VSTS',[x[0],mask.value],[mem],form='fp'+str(p.bits),accesses=[access])
            if post:self.env[r[-1]]=Pointer(p.byte+aff(x[pi+1])*unit,p.bits)
        elif name=='pto.barrier':
            pipe=attrs.get('pipe',{}).get('value')
            if pipe is None:raise ValueError('unknown VF barrier')
            self.body.append(self.Membar('VV_ALL'))
        elif name=='pto.membar':
            kind=attrs.get('kind',{}).get('value')
            if kind not in ('VV_ALL','VST_VLD','VLD_VST','VST_VST'):raise ValueError('unsupported VF membar '+str(kind))
            self.body.append(self.Membar(kind))
        elif name in ('pto.vbr','pto.vdup','pto.vcmax','pto.vcadd','pto.vsel','pto.vmax','pto.vsub','pto.vmuls','pto.vexp','pto.vadd','pto.vcvt','pto.vdiv','pto.vmul'):
            src=[]
            for vid,v in zip(o['operands'],x):
                if isinstance(v,Mask):src.append(v.value)
                elif isinstance(v,str):src.append(v)
                elif isinstance(v,(int,float,Aff)):src.append(self.scalar(v,self.dtype(self.types[vid])))
                else:raise ValueError('unsupported compute operand')
            dst=self.reg(r[0]);form='fp'+str(self.types[r[0]]['element']['bits'])
            if name=='pto.vcvt':
                if attrs.get('rnd')!='R' or attrs.get('part')!='EVEN' or attrs.get('sat') not in ('NOSAT','SAT'):
                    raise ValueError('unsupported VF conversion form')
                form='f32_to_f16'
            self.emit(name[4:].upper(),src,[dst],form=form);self.env[r[0]]=dst
        else:raise ValueError('unsupported physical VF operation: '+name)
    def run(self):
        from api.frontend.value_versioning import ValueVersioningPass
        from api.frontend.schema import StorageKind,CanonicalStorageObject,CanonicalLoop
        from api.frontend.validator import validate_canonical_vf_info
        root=self.function['operations'][0]
        # Function records do not contain their func.func operation itself.
        nested={v for o in self.function['operations'] for reg in o['regions'] for block in reg for v in block['operations']}
        for o in self.function['operations']:
            if o['id'] not in nested:self.op(o)
        canonical=ValueVersioningPass().run(self.Program(self.body,values=self.values),source={'adapter':'ptoas.structured_physical_vf.v1'})
        values={k:replace(v,storage_object_id='AIV_UB') if v.storage==StorageKind.UB else v for k,v in canonical.values.items()}
        def node(n):
            if isinstance(n,CanonicalLoop):return replace(n,body=tuple(node(c) for c in n.body))
            if not hasattr(n,'inputs'):return n
            def operand(v):return replace(v,memory_access=replace(v.memory_access,base_object_id='AIV_UB')) if v.memory_access else v
            return replace(n,inputs=tuple(operand(v) for v in n.inputs),outputs=tuple(operand(v) for v in n.outputs))
        canonical=replace(canonical,values=values,storage_objects={'AIV_UB':CanonicalStorageObject('AIV_UB',StorageKind.UB)},context=tuple(node(n) for n in canonical.context))
        validation=validate_canonical_vf_info(canonical)
        if not validation.ok:raise ValueError(str(validation.errors))
        return canonical


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('output');p.add_argument('--vfsim-root',required=True)
    p.add_argument('--footprints-only', action='store_true', help='Derive checked physical footprints without running the timing model')
    a=p.parse_args()
    sys.path.insert(0,str(Path(a.vfsim_root).resolve()))
    from api.simulator_costmodel import CoreVfCostModel
    from api.frontend.serialization import canonical_vf_info_to_dict
    data=json.loads(Path(a.input).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    results=[]
    for i,item in enumerate(data['vfs']):
        folder=out/str(i);folder.mkdir()
        bridge=Bridge(item['function'],item['arguments']);canonical=bridge.run()
        (folder/'canonical.json').write_text(json.dumps(canonical_vf_info_to_dict(canonical)))
        cycles = None if a.footprints_only else CoreVfCostModel(base_dir=a.vfsim_root,out_dir=folder).run_vf_info(canonical)['vf_end_cycle']
        results.append(dict(key=item['key'],cycles=cycles,approximations=sorted(bridge.approximations),operations=bridge.counts,memory_footprints=bridge.footprints))
        print(i,item['key'],cycles,flush=True)
    (out/'results.json').write_text(json.dumps(dict(vfs=results),indent=2)+'\n')


if __name__=='__main__': main()
