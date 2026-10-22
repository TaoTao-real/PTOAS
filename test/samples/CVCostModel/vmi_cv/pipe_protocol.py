"""Validate uniform-preload native CV communication with binary flag slots.

Model exact token identity and two AIV lanes. Every physical flag must be waited
before it is set again. Capacities are queue slots, not hidden physical memory.
"""
from itertools import product

def schedule(n,p):
    if not 1<=p<=n: raise ValueError('invalid preload')
    return [('first',i) for i in range(p)]+[(s,j) for i in range(n)
        for s,j in ([('second',i)]+([('first',i+p)] if i+p<n else []))]

def events(n,p,caps):
    cube=[];vectors=[[],[]]
    def add(dst,act,ch,t,lane):
        cap=caps[ch];dst.append((act,lane,ch,t%cap,t if act=='set_ready' or act=='set_free' or act=='wait_ready' else t-cap))
    for role,t in schedule(n,p):
        if role=='first':
            for lane in (0,1):add(cube,'wait_free','qk',t,lane)
            for lane in (0,1):add(cube,'set_ready','qk',t,lane)
        else:
            for lane in (0,1):add(cube,'wait_ready','p',t,lane)
            for lane in (0,1):add(cube,'set_free','p',t,lane)
            for lane in (0,1):add(cube,'wait_free','pv',t,lane)
            for lane in (0,1):add(cube,'set_ready','pv',t,lane)
        for lane in (0,1):
            dst=vectors[lane]
            if role=='first':
                for act,ch in [('wait_ready','qk'),('wait_free','p'),('set_ready','p'),('set_free','qk')]:add(dst,act,ch,t,lane)
            else:
                add(dst,'wait_ready','pv',t,lane);add(dst,'set_free','pv',t,lane)
    return [cube,*vectors]

def verify(n,p,caps):
    if set(caps)!={'qk','p','pv'} or any(type(v)!=int or v<1 for v in caps.values()):raise ValueError('bad capacities')
    streams=events(n,p,caps);pos=[0]*3
    flags={(lane,ch,slot,'free'):slot-cap for lane in (0,1) for ch,cap in caps.items() for slot in range(cap)}
    steps=0
    while True:
        moved=False
        for i,stream in enumerate(streams):
            if pos[i]==len(stream):continue
            act,lane,ch,slot,token=stream[pos[i]];kind=act.split('_')[1];key=(lane,ch,slot,kind)
            if act.startswith('wait'):
                if flags.get(key)!=token:continue
                del flags[key]
            else:
                if key in flags:raise ValueError(f'flag overwrite: {key} old={flags[key]} new={token}')
                flags[key]=token
            pos[i]+=1;steps+=1;moved=True
        if not moved:break
    completed=all(a==len(b) for a,b in zip(pos,streams))
    if completed:
        assert all(key[-1]=='free' for key in flags)
        assert len(flags)==2*sum(caps.values())
    return {'completed':completed,'preload':p,'capacities':caps,'flag_ids_per_lane':2*sum(caps.values()),
            'steps':steps,'blocked':[{'stream':i,'next':s[pos[i]]} for i,s in enumerate(streams) if pos[i]<len(s)]}

def choose(n,p,flag_limit=16):
    legal=[]
    for q,k,v in product(range(1,p+1),repeat=3):
        caps={'qk':q,'p':k,'pv':v}
        if 2*sum(caps.values())>flag_limit:continue
        result=verify(n,p,caps)
        if result['completed']:legal.append(result)
    # Prefer fewer slots, then less UB (QK/PV) for the fixed M32 transfer sizes.
    return min(legal,key=lambda x:(sum(x['capacities'].values()),x['capacities']['qk']+x['capacities']['pv'],x['capacities']['qk'])) if legal else None

if __name__=='__main__':
    import json
    assert verify(32,1,dict(qk=1,p=1,pv=1))['completed']
    assert not verify(32,3,dict(qk=1,p=1,pv=1))['completed']
    for p in (1,2,3,4,8):print(json.dumps(choose(32,p) or {'preload':p,'rejected':'no capacity combination fits the provisional 16-flag budget'}))
