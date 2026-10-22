"""Audit actual materialized communication, independently of schedule generation."""
import json, hashlib, sys
from pathlib import Path
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from fixture_native import children, event, opof, walk
from pipe_protocol import events

def flag_slot(eid):
    lane, eid=divmod(eid,16)
    if eid<2:return lane,'qk',0,'ready' if eid==0 else 'free'
    if eid<4:return lane,'p',0,'ready' if eid==2 else 'free'
    if eid<6:return lane,'pv',0,'ready' if eid==4 else 'free'
    if eid<10:return lane,'p',(eid-4)//2,'ready' if eid%2==0 else 'free'
    raise ValueError(f'unknown channel flag {eid}')

def audit(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['output_sha256']
    with ir.Context() as ctx:
        pto.register_dialect(ctx,load=True)
        module=ir.Module.parse(path.read_text());module.operation.verify()
        streamdict={};boundary={}
        for sub in children(module.operation):
            kind='vector' if 'vector' in str(sub.attributes['pto.kernel_kind']) else 'cube'
            func=next(o for o in children(sub) if o.name=='func.func')
            ops=children(func);stages=[s for s in meta['stages'] if s['core']==kind]
            covered=set();stream=[]
            for s in stages:
                covered.update(range(s['top_op_begin'],s['top_op_end']))
                for root in ops[s['top_op_begin']:s['top_op_end']]:
                    for o in walk(root):
                        eid=event(o)
                        if eid is None:continue
                        lane,ch,slot,phase=flag_slot(eid)
                        t=s['token'];cap=meta['protocol']['capacities'][ch]
                        assert slot==t%cap,(kind,t,ch,slot)
                        act=('set_' if o.name=='pto.set_intra_block' else 'wait_')+phase
                        version=t-cap if act=='wait_free' else t
                        stream.append((act,lane,ch,slot,version))
            streamdict[kind]=stream
            ends=[]
            for i,o in enumerate(ops):
                if i in covered or event(o) is None:continue
                ends.append((i<stages[0]['top_op_begin'],o.name,flag_slot(event(o))))
            boundary[kind]=ends
        actual=[streamdict['cube'],streamdict['vector'],
                [(act,1,ch,slot,t) for act,lane,ch,slot,t in streamdict['vector']]]
        expected=events(32,meta['preload'],meta['protocol']['capacities'])
        assert actual==expected,'generated flag events differ from token protocol'
        # Independently execute actual flag instructions, including init/drain.
        init=[];drain=[]
        for core,items in boundary.items():
            for before,name,key in items:
                dest=init if before else drain
                assert name==('pto.set_intra_block' if before else 'pto.wait_intra_block')
                dest.append(key)
                if core=='vector':dest.append((1,*key[1:]))
        assert len(set(init))==len(init) and set(init)==set(drain)
        flags={key:key[2]-meta['protocol']['capacities'][key[1]] for key in init}
        pos=[0,0,0]
        while any(pos[i]<len(s) for i,s in enumerate(actual)):
            moved=False
            for i,s in enumerate(actual):
                if pos[i]==len(s):continue
                act,lane,ch,slot,t=s[pos[i]];key=(lane,ch,slot,act.split('_')[1])
                if act.startswith('wait'):
                    if flags.get(key)!=t:continue
                    del flags[key]
                else:
                    assert key not in flags,('overwrite',key)
                    flags[key]=t
                pos[i]+=1;moved=True
            assert moved,('deadlock',pos)
        assert set(flags)==set(drain),'final flags cannot be drained'
        for key in drain:del flags[key]
        assert not flags
        result={'candidate_sha256':meta['output_sha256'],'preload':meta['preload'],
                'actual_ir_flag_protocol_matches':True,'actual_ir_protocol_completes':True,
                'init_and_drain_balanced':True,'flag_steps':sum(pos),
                'status':'static protocol audit; numerical and hardware tests still required'}
        path.with_suffix('.audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':audit(sys.argv[1])
