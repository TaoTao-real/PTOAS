# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Predeclared fixed-FA VF/sync/layout/schedule ablation; no model reselection."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from fa_model_ab import read,write,sha,require,invoke
from fa_native_ab import import_fa,SOURCE_SHA
from fa_model_first import launcher

VARIANTS = [
    dict(id='R',preload=1,flags=[],prepare_flags=[]),
    dict(id='S',preload=1,flags=['--cv-drain-vf-calls=false'],prepare_flags=[]),
    dict(id='L',preload=1,flags=['--plan-memory-order-by-size=true'],prepare_flags=[]),
    dict(id='SL',preload=1,flags=['--plan-memory-order-by-size=true','--cv-drain-vf-calls=false'],prepare_flags=[]),
    dict(id='P3',preload=3,flags=[],prepare_flags=[]),
    dict(id='VB',preload=1,flags=[],prepare_flags=['--enable-vmi-loop-fusion=false','--enable-vmi-load-store-elision=false']),
    dict(id='VC',preload=1,flags=[],prepare_flags=['--enable-vmi-load-store-elision=false']),
    dict(id='K',preload=1,flags=['--vpto-scheduler=on'],prepare_flags=[]),
    dict(id='SK',preload=1,flags=['--cv-drain-vf-calls=false','--vpto-scheduler=on'],prepare_flags=[]),
]


def prepare(a):
    from ptoas import cv_vf as cv
    from ptoas.mlir import ir
    root=a.root.resolve();root.mkdir(exist_ok=a.resume)
    require(sha(a.source)==SOURCE_SHA,'unexpected FA')
    variants=[v for v in VARIANTS if not a.variant or v['id'] in a.variant]
    design=dict(schema='pto.cv.fa_gap_design.v1',source_sha256=SOURCE_SHA,variants=variants,
        purpose='diagnose existing frozen recommendation; not a new TileSim recommendation',device=1,
        baseline_source_commit='ae2fc88c78aebd5e45bb23e5ea7f69aa8b7d8c42',compiler=cv.compiler_identity(a.compiler))
    require(not (root/'matrix.json').exists(),'completed matrix is immutable')
    if (root/'frozen-design.json').exists(): require(read(root/'frozen-design.json')==design,'resumed design/compiler changed')
    else: write(root/'frozen-design.json',design)
    target=dict(arch='a5',cann_version='9.2.0');rows=[]
    for variant in variants:
        folder=root/variant['id'];folder.mkdir(exist_ok=a.resume);symbol='fa_gap_'+hashlib.sha256(str(folder).encode()).hexdigest()[:14]
        if (folder/'source.pto').exists():
            require(a.resume and read(folder/'source.import.json')['output_sha256']==sha(folder/'source.pto'),'invalid resumed import')
            symbol=re.search(r'sym_name = "([^"]+)_mix_ai[cv]"',(folder/'source.pto').read_text()).group(1)
        else: import_fa(a.source,folder/'source.pto',symbol)
        flags=cv._flags(target)
        if not (folder/'prepared.pto').exists(): cv._invoke(a.compiler,[folder/'source.pto',*flags,*variant['prepare_flags'],'--cv-prepare-vf','--emit-pto-ir',
            '--mlir-print-ir-after=pto-fusion-plan,pto-op-scheduling,pto-fusion-region-gen,pto-vmi-loop-fusion,pto-vmi-load-store-elision',
            '--mlir-print-ir-before=pto-vmi-loop-fusion','-o',folder/'prepared.pto'],folder/'prepare.log')
        context,module=cv._parse((folder/'prepared.pto').read_text())
        with context:
            for op in cv._walk(module.operation):
                if op.name=='func.func' and 'pto.cv.static_stage_graph' in op.attributes:
                    op.attributes['pto.cv_preload_count']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),variant['preload'])
            module.operation.verify();(folder/'annotated.pto').write_text(module.operation.get_asm(print_generic_op_form=True)+'\n')
        capture=folder/'codegen';capture.mkdir(exist_ok=a.resume);os.environ['PTO_GAP_CAPTURE']=str(capture)
        apply_flags=[f for f in flags if not (f.startswith('--vpto-scheduler=') and '--vpto-scheduler=on' in variant['flags'])]
        if not (folder/'candidate.o').exists(): cv._invoke(a.compiler,[folder/'annotated.pto',*apply_flags,'--cv-apply-vf','--enable-insert-sync','--enable-vecscope-mem-bar',
            '--check-sync-ids','--check-sync-interference','--check-sync-self-coverage',*variant['flags'],
            '--cv-feedback='+str(folder/'compiler_feedback.json'),'-o',folder/'candidate.o'],folder/'compile.log')
        del os.environ['PTO_GAP_CAPTURE']
        fb=read(folder/'compiler_feedback.json');require(fb['product_sha256']==sha(folder/'candidate.o') and fb['native_plan_memory_invoked'],'missing actual compile evidence')
        context,module=cv._parse(fb['final_vpto_ir'])
        with context:program=cv.describe(module)
        write(folder/'program.json',program)
        for function in fb['functions']:
            coverage=[l for l in (folder/'compile.log').read_text().splitlines() if '[sync-gate:g1-self]' in l and 'func='+function['function']+' ' in l]
            require(len(coverage)==1 and 'uncovered=0 ' in coverage[0] and 'unmappable=0' in coverage[0],'native sync coverage missing')
        require(len(list(capture.glob('*.ll')))==2,'missing same-compilation LLVM inputs')
        from emit_gap_assembly import emit
        emit(capture)
        launcher(folder,symbol)
        rows.append(dict(candidate_id=variant['id'],symbol=symbol,folder=variant['id'],configuration=variant,
            schedule_fingerprint=cv.fingerprint(fb['schedules']),object_sha256=sha(folder/'candidate.o'),
            feedback_sha256=sha(folder/'compiler_feedback.json'),program_sha256=sha(folder/'program.json')))
        write(root/'build-progress.json',dict(status='building',rows=rows));print('BUILD_VARIANT_PASS',variant['id'],flush=True)
    # The control object is reused byte-for-byte. Its source/codegen policy is
    # separate and never described as a native VF or native sync variant.
    delivery=root/'T';delivery.mkdir()
    for name in ('runner','kernel.o','kernel.linked.o'):shutil.copyfile(a.delivery/name,delivery/name)
    (delivery/'runner').chmod(0o500)
    symbol='cv_fa_repro_p4_20261022_988f723ba'
    require(symbol.encode() in (delivery/'runner').read_bytes(),'wrong delivery reference')
    rows.append(dict(candidate_id='T',symbol=symbol,folder='T',purpose='frozen delivery depth4 diagnostic control'))
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    write(root/'matrix.json',dict(schema='pto.cv.fa_gap_matrix.v1',device=1,source_sha256=SOURCE_SHA,rows=rows,files=files))
    print('MATRIX_BUILD_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('root',type=Path);p.add_argument('--compiler',required=True);p.add_argument('--delivery',required=True,type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--variant',action='append',choices=[v['id'] for v in VARIANTS]);prepare(p.parse_args())
