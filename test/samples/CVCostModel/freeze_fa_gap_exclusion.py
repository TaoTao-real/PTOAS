# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Create a fresh matrix for the known failing VC intermediate, before timing.

Never mutate the original frozen ten-way artifacts or accept a performance
failure as a reason to remove a contender from this ablation.
"""
import argparse
from pathlib import Path
import shutil
from fa_model_ab import read,write,sha,require,numpy_golden


def freeze(a):
    source=a.source.resolve();destination=a.destination.resolve();old=read(source/'acceptance_manifest.json');failed=read(a.g3)
    require(failed['status']=='fail' and failed['mode']=='g3' and failed['matrix_sha256']==sha(source/'acceptance_manifest.json') and '/VC/' in failed['error'],'matching VC correctness failure required')
    require(all(r['stage']!='profile' for r in failed['rows']),'performance-based exclusion is forbidden')
    log=(a.failed_runtime/'runtime.log').read_text()
    require('NUMERICAL_FAIL' in log and 'input_unchanged=1 guard_unchanged=1' in log,'numerical failure evidence missing')
    try: numpy_golden(a.failed_runtime/'output','pattern')
    except ValueError as error: require(str(error)=='independent NumPy golden failed','unexpected failure during golden replay')
    else: raise ValueError('VC output did not fail independent golden')
    for name,digest in old['files'].items():require(sha(source/name)==digest,'original artifact changed: '+name)
    destination.mkdir(exist_ok=False)
    rows=[r for r in old['rows'] if r['candidate_id']!='VC']
    for row in rows:
        folder=Path(row['folder']);require(not folder.is_absolute() and '..' not in folder.parts,'unsafe folder')
        shutil.copytree(source/folder,destination/folder);(destination/folder/'runner').chmod(0o500)
    for source_file,name in [(a.g3,'failed-g3.json'),(a.failed_runtime/'runtime.log','failed-vc-runtime.log'),(source/'acceptance_manifest.json','original-acceptance-manifest.json')]:shutil.copyfile(source_file,destination/name)
    evidence=dict(reason='G3_NUMERICAL_FAIL',candidate_id='VC',failed_g3_sha256=sha(a.g3),failed_log_sha256=sha(a.failed_runtime/'runtime.log'),independent_golden_replay='NUMERICAL_FAIL',original_vc_object_sha256=sha(source/'VC/candidate.o'),no_profile_rows_in_source_report=True)
    write(destination/'exclusion-evidence.json',evidence)
    files={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    write(destination/'acceptance_manifest.json',dict(schema=old['schema'],device=old['device'],rows=rows,files=files,
        native_compiler_commit=old['native_compiler_commit'],source_manifest_sha256=sha(source/'acceptance_manifest.json'),harness_commit=a.harness_commit,
        original_sha256=old['original_sha256'],exclusions=dict(VC=dict(reason='G3_NUMERICAL_FAIL',evidence='exclusion-evidence.json')),
        scope='fresh same-device nine-way verification; no valid timing for VC'))
    print('FRESH_MATRIX_FROZEN',sha(destination/'acceptance_manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','destination','g3','failed_runtime'):p.add_argument(name,type=Path)
    p.add_argument('--harness-commit',required=True);freeze(p.parse_args())
