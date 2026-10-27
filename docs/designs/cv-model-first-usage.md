# Model-first preload selection

The opt-in `ptoas.cv_vf` Python API prepares the VF program once, exports bounded
logical schedules, freezes a model recommendation, and compiles that Plan.
The ordinary compiler and v1/2.0 annotation import defaults do not change.

```text
python -m ptoas.cv_vf optimize input.pto bindings.json target.json model.json run \
  --compiler /path/to/ptoas --model-command model-argv.json \
  --final-command final-argv.json
```

Each command file contains an argv array, not shell text. The initial adapter is
`python -m core.frontend.adaptor.ptoas_logical`; the final feedback adapter is
`python -m core.frontend.adaptor.ptoas_prepared --evaluate-only`. Both receive
explicit VF worker, VfSimulator root and Python options. Use the example setup in
`test/samples/CVCostModel/fa_model_first.py` and its acceptance record. The final
adapter currently accepts only the SHA-bound M32 FA fixture. A general initial
logical representation does not imply general final validation support.

`model.json` supplies a hardware configuration and SHA, timing architecture and
explicit profile-proxy permission. Bindings include actual scalar values,
argument shapes and proven disjoint constraints. Target budgets describe compiler
available storage separately from the hardware timing configuration.

The new `pto.cv.prepared_search.v1` package contains structured operations, bounded
logical events, candidate Plans and relative VF bindings. The schema is in
`ptodsl/pto_costmodel/prepared_search.schema.json`. An unsupported required feature,
unknown operation, unresolved resource precheck or out-of-range VF footprint
stops search. Neither unknown time nor unknown resource use becomes zero.

Costs are deduplicated within one request by VF body, shape, alias and offset
conditions. Results bind worker, model and hardware hashes; this is not a
persistent cross-model cache. Representative simulator addresses never enter a
Plan or claim to be the final physical allocation. AIV storage domains remain
separate. The event model conservatively orders completion within each core and
reports scalar/control timing approximations. Final synchronization can change
cost: the implementation always re-evaluates the selected physical program and
never silently reuses the initial prediction.

`run/search/selection.json` precedes `run/attempt-0/compile-intent.json`. Normal
success prepares once and fully compiles one selected candidate. Recognized
compiler memory/slot failures permit at most one further candidate from the
frozen ranking, from the original prepared package. Protocol/identity/tool failures
terminate. Baseline retention does not manufacture an optimized result.

The fixed FA acceptance driver separately creates P0 **after** selection and
selected-candidate G2, for independent A/B evaluation only. Its process trace
records Linux dynamic-process starts and monotonic selection markers; it is not
an adversarial sandbox. `fa_model_first.py run` performs fresh G3 and 20 paired
A/B profiles; `summarize_model_first.py` checks the retained evidence.

Limitations: static A5, one AIC/two AIV, supported bounded operations, preload
search only, restricted FA final G2. The old basic micro has a transpose VF whose
full-width load exceeds the logical Buffer extent; it is explicitly rejected by
the initial footprint check. VNEG has a declared VMULS timing-family proxy and
VSSTB uses catalog timing with an LSU envelope approximation; neither is an
accuracy certification. Target-matched DT timing, generic final validation,
physical feedback consumption by other models and broad G4 remain follow-ups.

For a different available device with the same supported SoC, bind a new copy of
the immutable acceptance matrix before running:

```text
python test/samples/CVCostModel/fa_model_first.py deploy matrix deployment --device 1
```

Deployment retains all artifact digests and the source manifest digest. Run a new
complete G3 and performance experiment against that deployment. A G3 report from
a different device/deployment is rejected. Do not combine device measurements.
