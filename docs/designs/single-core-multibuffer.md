# Single-core multibuffer exchange

The opt-in `single-core` Python entry is separate from the prepared CV candidate
search. It accepts one pure Cube/Vector function with one static, independent
loop. The existing CV protocol and preload behavior remain unchanged.

1. `ptoas single-core prepare input.pto request.json --independent-iterations`
2. In TileSim: `python -m core.frontend.adaptor.pto.single_core request.json model-output --hardware /path/to/config.yaml`
3. `ptoas single-core annotate request.json model-output/plan.json annotated.pto`
4. With a compiler built from this branch, compile `annotated.pto` with
   `--apply-single-core-pipeline --enable-insert-sync --enable-vecscope-mem-bar=false`
   plus the ordinary target/backend/output flags required by the kernel.

The caller must establish disjoint per-iteration GM writes and non-aliasing
arguments. The independence switch is a contract assertion, not a compiler
proof. Local recurrence and unsupported structures are rejected by the model
entry. v1 requires owned allocations outside the candidate loop, 1–256 static
iterations and at most 16 slots per buffer. General recurrence, nested loops,
channels, barriers, prepared VF calls and dynamic bounds are not enabled.

## Consumption

The plan is bound to the canonical input hash and compiler-owned buffer IDs.
`pto.pipeline.multi_buffer_count` carries the chosen count;
`pto.pipeline.iteration_slots` carries the exact slot for each iteration.
`pto.pipeline.single_core` selects the single-core branch in CVSchedule.cpp.
That branch preserves iteration issue order and reuses native buffer-root,
version, AllocMultiTile/MultiTileGet and operand-remapping machinery. It does
not require a prefix/suffix cut or infer slots from CV preload. The normal
memory planner and InsertSync handle physical layout and reuse hazards.

A model count is not proof of generated-code performance. Compilation, sync
verification, numerical tests and same-compilation feedback remain necessary.
A stock wheel can validate request/annotation IR but cannot execute this new
C++ branch. The native lit tests require rebuilding PTOAS from this branch.

## Tests

- `ptodsl/tests/test_single_core_contract.py`: identities, stale inputs, counts,
  per-iteration slot types/ranges and unsupported plans.
- `test/lit/pto/single_core_multibuffer.pto`: native expansion, memory lowering,
  iteration order and invalid slots.
- `test/lit/pto/single_core_exchange.pto`: native preparation and annotation for
  softmax/matmul, then materialization using the freshly built test optimizer.
