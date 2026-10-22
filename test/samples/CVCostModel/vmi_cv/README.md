# Fixed FA VF/CV integration experiment

These tools are diagnostic adapters for the delivered M32 FA workload. They
are not a general CV compiler pass, a model recommendation, or a G2 certificate.
The fixed physical schedule/protocol implementation was adapted from
`PTOAS_CV_pipeline_v1_20261021.zip`; the prepared VF bodies come from the PTOAS
VMI fusion pipeline. No model-supplied physical address is accepted.

The source workload is FP16 Q/K/V, FP32 O, Q=128, KV=1024, D=128, tile M=32,
one AIC and two AIV lanes. The fixed relocation map and three P communication
slots are specific to that workload. Source digests must be frozen explicitly.
The entry functions must retain the delivered outer/steady loop structure.
Unsupported shapes, loops or captures must fail; do not guess a replacement.

`prepare_fixture.py` opens only full 16x128 TileOp scopes supported by the
VMI candidate path. Fallback scopes remain intact. It checks that operation
order, types and attributes are unchanged. Ordinary input support is not changed.

`fixture_native.py` expands the fixed CV token schedule as an audit reference.
`fixture_compact.py` builds compact warm-up/steady/drain loops using the same
prepared physical VF bodies. Both preserve raw VF instruction signatures.
`audit_flags.py` checks expanded channel flags, initialization, drain and
protocol completion. `audit_compact.py` independently interprets scalar control
and compares compact/expanded physical instruction traces for AIC and both AIV
lanes. Vector arithmetic is opaque; independent numerical/device tests remain
mandatory. Neither audit proves general asynchronous memory safety.

`checkpoint.py` validates and reseals `physical-v2` integrity metadata after
experimental schedule construction or entry renaming. Resealing does not prove
legality. PTOAS checks CANN identity and validates VPTO again at emission.

For the four-way experiment, keep candidates, function DSE (off), unroll (off),
VPTO scheduler (off), tiling and communication slots constant. Compare:

| Variant | VMI loop fusion / load-store elision | CV depth |
| --- | --- | --- |
| A | off | 1 |
| B | off | 4 |
| C | on | 1 |
| D | on | 4 |

Here CV depth follows the delivered fixed-case token schedule. Do not silently
rename it to the ADR's iteration-distance preload annotation. A generic adapter
and versioned mapping remain future work.

Use `--expected-source-sha256` with each materializer. Keep outputs outside the
source tree. Each compile, correctness run and performance experiment must bind
source, compiler, target, CANN, flags, input, checkpoint, object and runner hashes.
