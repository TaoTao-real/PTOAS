# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Post-measurement diagnostic of the pinned model's completion assumptions.

Reuse only identical, archived VF body/binding keys. This is not a prediction
frozen before board execution, a Plan recommendation, or G4 evidence. It does
not change the normal adapter or inject a synthetic Plan into the exchange.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
import textwrap
import subprocess
from fa_model_ab import read,write,sha,require


def replay(a):
    manifest=read(a.matrix/'acceptance_manifest.json')
    for name,digest in manifest['files'].items():
        require(sha(a.matrix/name)==digest,'frozen matrix artifact changed: '+name)
    known={row['candidate_id'] for row in manifest['rows']}
    require(len(a.variant)==len(set(a.variant)) and set(a.variant)<=known,'unknown or duplicate variant')
    def git(*args):
        return subprocess.check_output(['git','-C',str(a.tilesim_root),*args],text=True).strip()
    changed=git('diff','--name-only',a.reference_model_commit,'--','core').splitlines()
    untracked=git('ls-files','--others','--exclude-standard','core').splitlines()
    relevant=lambda name:name.endswith('.py') or '/config/arc_config/' in name
    require(not any(relevant(n) for n in changed+untracked),'model code/config differs from declared archive version')
    sys.path.insert(0,str(a.tilesim_root.resolve()))
    from core.frontend.adaptor import ptoas_prepared as adapter
    from core.common.backend_config import BackendConfig,resolve_backend
    from core.arch.target import CustomHardwareTarget
    from core.backend.evaluation.duration_provider import MirCostModelDurationProvider,DurationResult
    from core.common.entity import CoreUnit
    from core.arch import ArchName
    from core.backend.tile_op_costmodel import TileOpRequest,TileOpModelSelection,predict_tileop_with_selection
    adapter_path=Path(adapter.__file__);reference=read(a.reference_result)
    require(reference['adapter_sha256']==sha(adapter_path),'archived timing belongs to another adapter')
    require(reference['model_target_sha256']==sha(a.hardware),'hardware profile differs from archived evaluation')
    old_manifest=read(a.reference_root/'ab_manifest.json')
    for p in (a.vf_estimates,a.reference_result):
        rel=p.resolve().relative_to(a.reference_root.resolve()).as_posix()
        require(old_manifest['files'].get(rel)==sha(p),'archived model evidence not bound by frozen manifest')
    vf_rows=read(a.vf_estimates)['vfs'];estimates={v['key']:v for v in vf_rows}
    require(len(estimates)==len(vf_rows),'duplicate archived VF binding key')
    backend=resolve_backend(BackendConfig(target=CustomHardwareTarget(a.hardware,aic_core_num=1,aiv_core_num=2)))
    timing_arch=ArchName.from_value(reference['timing_model_arch'])
    require(backend.hardware.arch==timing_arch or a.allow_profile_proxy,'explicit profile-proxy authorization required')
    # Reuse the exact original duration-provider implementation. Do not copy a
    # model and silently change its defaults or control-cost constants.
    source=adapter_path.read_text();classes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.ClassDef) and n.name=='Costs']
    require(len(classes)==1,'unsupported adapter provider structure')
    provider_source=textwrap.dedent(ast.get_source_segment(source,classes[0]))
    namespace=dict(MirCostModelDurationProvider=MirCostModelDurationProvider,DurationResult=DurationResult,
        CoreUnit=CoreUnit,TileOpRequest=TileOpRequest,TileOpModelSelection=TileOpModelSelection,
        predict_tileop_with_selection=predict_tileop_with_selection,timing_arch=timing_arch,backend=backend,vfs=estimates)
    exec(compile(provider_source,str(adapter_path),'exec'),namespace)
    predictions=[]
    for variant in a.variant:
        folder=a.matrix/variant;program=read(folder/'program.json');feedback=read(folder/'compiler_feedback.json')
        require(feedback['product_sha256']==sha(folder/'candidate.o'),'unbound compiler product')
        native=adapter.NativeEvents(program);events=native.build()
        require(set(native.vfs)<=set(estimates),'UNKNOWN_VF_COST: do not substitute zero or another binding')
        for event in events:
            if 'vf_key' not in event.attrs:continue
            vf=estimates[event.attrs['vf_key']]
            require(type(vf['cycles']) in (int,float) and math.isfinite(vf['cycles']) and vf['cycles']>0,'unknown VF timing')
            for f in vf['memory_footprints']:
                region=native.Region('aiv'+str(event.attrs['aiv_lane'])+':UB',(f['begin'],),(f['end']-f['begin'],),native.Mem.UB,'uint8')
                (event.reads if f['access']=='read' else event.writes).append(region)
        result,evaluator=adapter.execute(events,namespace['Costs'](backend))
        require(math.isfinite(result.latency) and result.latency>0,'invalid diagnostic latency')
        predictions.append(dict(candidate_id=variant,latency_us=result.latency,events=len(events),vf_conditions=len(native.vfs),
            operation_counts=dict(native.counts),program_sha256=sha(folder/'program.json'),feedback_sha256=sha(folder/'compiler_feedback.json')))
    report=dict(schema='pto.cv.fa_gap_model_assumption_diagnostic.v1',evaluation_only=True,preselection_prediction=False,
        recommendation=None,G4_usable=False,uses_archived_vf_estimates=True,predictions=predictions,
        archive=dict(manifest_sha256=sha(a.reference_root/'ab_manifest.json'),result_sha256=sha(a.reference_result),vf_estimates_sha256=sha(a.vf_estimates)),
        matrix_sha256=sha(a.matrix/'acceptance_manifest.json'),
        model=dict(adapter_sha256=sha(adapter_path),hardware_sha256=sha(a.hardware),timing_model_arch=reference['timing_model_arch'],profile_proxy=backend.hardware.arch!=timing_arch,
            working_head=git('rev-parse','HEAD'),reference_source_commit=git('rev-parse',a.reference_model_commit),model_code_config_unchanged=True),
        limitation='post-measurement replay of the existing serial-core event assumptions; never use it to reselect the frozen board experiment')
    write(a.output,report);print([(r['candidate_id'],r['latency_us']) for r in predictions])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','reference_root','reference_result','vf_estimates','hardware','output'):p.add_argument(name,type=Path)
    p.add_argument('--tilesim-root',required=True,type=Path);p.add_argument('--reference-model-commit',required=True)
    p.add_argument('--variant',required=True,action='append');p.add_argument('--allow-profile-proxy',action='store_true');replay(p.parse_args())
