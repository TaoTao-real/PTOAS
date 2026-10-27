# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Translate the captured CCE driver route into direct device IR assembly.

Keep all target/backend/optimization flags. -dc is a CCE-driver routing flag;
explicit target and march replace that route. Forward the AICore bitcode flag
to cc1 because the direct IR driver must not interpret it as a host-route switch.
"""
import argparse
import json
from pathlib import Path
import subprocess
from fa_model_ab import write,sha,require


def emit(folder):
    records=[]
    for p in sorted(folder.glob('*.argv0')):
        original=p.read_bytes().decode().rstrip('\0').split('\0');a=original.copy()
        require('-c' in a and '-x' in a and a[a.index('-x')+1]=='ir' and '-' in a,'not a captured object IR compile')
        a[a.index('-c')]='-S';a[a.index('-')]=str(p.with_suffix('.ll'));a[a.index('-o')+1]=str(p.with_suffix('.preserved.s'))
        if '-dc' in a:a.remove('-dc')
        if '-cce-bitcode-is-aicore' in a:a.insert(a.index('-cce-bitcode-is-aicore'),'-Xclang')
        cpu=next(x.split('=',1)[1] for x in a if x.startswith('--cce-aicore-arch='))
        a[1:1]=['--target=hiipu64-hisilicon-cce','-march='+cpu]
        result=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        p.with_suffix('.preserved.log').write_text(result.stdout);require(result.returncode==0,'assembly failed: '+str(p))
        record=dict(object_argv=original,assembly_argv=a,llvm_sha256=sha(p.with_suffix('.ll')),assembly_sha256=sha(p.with_suffix('.preserved.s')),
            driver_translation=['-dc route replaced by explicit HiIPU triple and original CPU','AICore bitcode option forwarded with -Xclang'],returncode=0)
        write(p.with_suffix('.preserved-command.json'),record);records.append(record)
    require(records,'no captured commands');return records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args();emit(a.folder)
