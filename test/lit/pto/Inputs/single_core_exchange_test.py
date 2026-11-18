"""Native exchange smoke tests. Run with the compiler built from this branch."""
import subprocess
import sys
from pathlib import Path
from ptoas.single_core import prepare, annotate, PLAN, SCHEDULE

compiler=sys.argv[1]
for name in ('softmax', 'cube'):
    source=Path(__file__).with_name('single_core_'+name+'.pto').read_text()
    request=prepare(source,independent_iterations=True)
    plan=dict(schema=PLAN,input_fingerprint=request['input_fingerprint'],
        loop_id=request['loop_id'],trip_count=request['trip_count'],schedule_semantics=SCHEDULE,
        buffers=[dict(buffer_id=key,count=2,slots=[i%2 for i in range(request['trip_count'])])
                 for key in request['buffer_ids'].values()],predicted_latency_us=1.0)
    # A5 row reductions do not access their legacy scratch operand.
    # It has no native lifetime to duplicate; the model also returns one slot.
    if name == 'softmax':
        plan['buffers'][-1].update(count=1, slots=[0]*request['trip_count'])
    ir=annotate(request,plan)
    result=subprocess.run([compiler,'-pto-cv-apply-schedule','-pto-plan-memory','-pto-resolve-buffer-select'],
        input=ir,text=True,capture_output=True)
    assert result.returncode==0, result.stderr
    assert 'scf.for' not in result.stdout
    assert 'pto.cv.slot = 1' in result.stdout
print('single-core native exchange: softmax and cube passed')
