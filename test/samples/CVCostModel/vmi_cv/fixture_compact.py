"""Compact native CV warm-up/steady/drain schedule, reusing physical VF bodies."""
import argparse,hashlib,json
from pathlib import Path
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from checkpoint import seal, validate
from fixture_native import Cloner,children,attrs,event,opof,walk,split_wait,int_const,SOURCE_SHA

def binary(name,a,b):return ir.Operation.create('arith.'+name,operands=[a,b],results=[a.type]).results[0]
def icon(n):return int_const(ir.IndexType.get(),n)
def cmp(a,b,predicate=0):
    return ir.Operation.create('arith.cmpi',operands=[a,b],results=[ir.IntegerType.get_signless(1)],
        attributes={'predicate':ir.IntegerAttr.get(ir.IntegerType.get_signless(64),predicate)}).results[0]
def branch(cond,then,other=lambda:None):
    op=ir.Operation.create('scf.if',operands=[cond],regions=2)
    for region,fn in zip(op.regions,(then,other)):
        block=region.blocks.append()
        with ir.InsertionPoint(block):fn();ir.Operation.create('scf.yield')
def loop(lo,hi,fn):
    op=ir.Operation.create('scf.for',operands=[icon(lo),icon(hi),icon(1)],regions=1)
    block=op.regions[0].blocks.append(ir.IndexType.get())
    with ir.InsertionPoint(block):fn(block.arguments[0]);ir.Operation.create('scf.yield')

class DynamicCloner(Cloner):
    def __init__(self,bindings,token,preload,vfs,ledger):
        super().__init__(bindings,0,preload,3,vfs,ledger)
        self.runtime_token=token
        q=binary('floordivsi',token,icon(8));self.k=binary('remsi',token,icon(8))
        self.pslot=binary('remsi',token,icon(3));alpha=binary('remsi',token,icon(preload))
        def target(index,scale,base):
            val=binary('addi',binary('muli',index,icon(scale)),icon(base))
            return ir.Operation.create('arith.index_cast',operands=[val],results=[ir.IntegerType.get_signless(64)]).results[0]
        self.targets={36864:target(q,8192,65536),46080:target(q,512,98304),
                      46592:target(q,512,100352),47104:target(alpha,512,102400),
                      73728:target(self.pslot,8192,73728)}
    def relocation_ranges(self,typ):
        return [(lo,hi,self.targets[lo]) for lo,hi,_ in super().relocation_ranges(typ)]
    def rebase(self,value,lo,target):
        delta=binary('subi',target,int_const(target.type,lo))
        return binary('addi',value,delta)
    def clone(self,old):
        eid=event(old)
        if eid is not None and eid%16 in (2,3):
            lane,local=divmod(eid,16)
            for slot,ready in enumerate((2,6,8)):
                attr=attrs(old);attr['event_id']=ir.IntegerAttr.get(ir.IntegerType.get_signless(32),16*lane+ready+(local-2))
                branch(cmp(self.pslot,icon(slot)),lambda attr=attr:ir.Operation.create(old.name,attributes=attr))
            return None
        return super().clone(old)

def materialize(source,dest,preload):
    assert preload in (1,2,3,4)
    raw=Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    with ir.Context() as ctx,ir.Location.unknown():
        pto.register_dialect(ctx,load=True);original=ir.Module.parse(raw.decode());validate(original);out=ir.Module.create()
        for a in original.operation.attributes:out.operation.attributes[a.name]=a.attr
        for k,v in [('pto.cv.vf_state','physical-v2'),('pto.cv.vf_arch','a5'),('pto.cv.vf_build_level','level3')]:out.operation.attributes[k]=ir.StringAttr.get(v)
        vfs={o:f'vf_{i:02d}' for i,o in enumerate(o for o in walk(original.operation) if o.name=='pto.vecscope')}
        ledger=[]
        for sub in children(original.operation):
            vector='vector' in str(sub.attributes['pto.kernel_kind'])
            oldfunc=next(o for o in children(sub) if o.name=='func.func');oldbody=oldfunc.regions[0].blocks[0]
            top=children(oldfunc);outer=next(o for o in top if o.name=='scf.for');outerops=children(outer)
            if vector:
                steady=next(o for o in outerops if o.name=='scf.for' and any(c.name=='pto.wait_intra_block' and event(c)==0 for c in children(o)))
                mid=outerops.index(steady);sm0,up0=split_wait(outerops[:mid],4)
                sm1,up1=split_wait([o for o in children(steady) if o.name!='scf.yield'],4)
                suffix=[o for o in outerops[mid+1:] if o.name!='scf.yield']
            else:
                steady=next(o for o in outerops if o.name=='scf.for');prefix=outerops[:outerops.index(steady)]
                qk,pv=split_wait([o for o in children(steady) if o.name!='scf.yield'],2)
            with ir.InsertionPoint(out.body):
                newsub=ir.Operation.create('builtin.module',attributes=attrs(sub),regions=1);subblock=newsub.regions[0].blocks.append()
            with ir.InsertionPoint(subblock):
                newfunc=ir.Operation.create('func.func',attributes=attrs(oldfunc),regions=1)
                body=newfunc.regions[0].blocks.append(*[v.type for v in oldbody.arguments])
            base=dict(zip(oldbody.arguments,body.arguments))
            def stage(role,token):
                bindings=dict(base)
                bindings[outer.regions[0].blocks[0].arguments[0]]=binary('floordivsi',token,icon(8))
                bindings[steady.regions[0].blocks[0].arguments[0]]=binary('remsi',token,icon(8))
                def emit(ops):
                    clone=DynamicCloner(bindings,token,preload,vfs,ledger)
                    for op in ops:clone.clone(op)
                k=bindings[steady.regions[0].blocks[0].arguments[0]]
                if vector:
                    branch(cmp(k,icon(0)),lambda:emit(sm0 if role=='first' else up0),lambda:emit(sm1 if role=='first' else up1))
                    if role=='second':branch(cmp(k,icon(7)),lambda:emit(suffix))
                elif role=='first':
                    branch(cmp(k,icon(0)),lambda:emit(prefix));emit(qk)
                else:emit(pv)
            with ir.InsertionPoint(body):
                for op in top[:top.index(outer)]:
                    if op.name=='pto.set_intra_block':
                        for t in (range(3) if event(op)%16==3 else (0,)):Cloner(base,t,preload,3,vfs,ledger).clone(op)
                loop(0,preload,lambda t:stage('first',t))
                def steady_body(t):
                    stage('second',t)
                    nxt=binary('addi',t,icon(preload))
                    branch(cmp(nxt,icon(32),2),lambda:stage('first',nxt))
                loop(0,32,steady_body)
                for op in top[top.index(outer)+1:]:
                    for t in (range(3) if event(op) is not None and event(op)%16==3 else (0,)):Cloner(base,t,preload,3,vfs,ledger).clone(op)
        seal(out);out.operation.verify();text=str(out)+'\n';ir.Module.parse(text).operation.verify()
        path=Path(dest);path.parent.mkdir(exist_ok=True,parents=True);path.write_text(text)
        report={'schema':'ptoas.native_cv_compact.v1','preload':preload,'input_sha256':SOURCE_SHA,
            'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'shape':[128,1024,128],'tile':[32,128,128],
            'block_dim':1,'mix_block_dim':2,'static_vf_ids':ledger,
            'ub_high_water_bytes':102400+preload*512,'l1_high_water_bytes':98304,
            'physical_vf_instruction_structure_preserved':True,'machine_code_generated':False}
        path.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'preload':preload,'static_vfs':len(ledger),'bytes':len(text.encode()),'sha256':report['output_sha256']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('dest');p.add_argument('--preload',type=int,required=True)
    p.add_argument('--expected-source-sha256',required=True);a=p.parse_args();SOURCE_SHA=a.expected_source_sha256;materialize(a.source,a.dest,a.preload)
