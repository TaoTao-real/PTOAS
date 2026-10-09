# Cost model exchange for backend-partitioned assemblies

Date: 2026-10-09. Status: proposed, with compiler-side implementation.

## Context

Mainline PTOAS compiles an outer module containing wrapper, Cube and Vector
modules. The static four-stage cost model adapter must not remove this ability
or route checkpoint MLIR to Bisheng as C++. The supplied serial FlashAttention
also contains runtime scalar parameters, nested loop domains and a Vector
prologue followed by recurrent steady-state work.

## Decision

Keep ordinary code generation unchanged. A cost model checkpoint is one IR job
on the original module tree, irrespective of child backend selection. Run pipe
lowering and TFree validation recursively, retaining declarations, wrappers,
symbol scopes and backend attributes. Never flatten the canonical compiler input
or set an overriding backend on its outer container.

Keep existing static v1/2.0 semantics. Add an explicitly negotiated 2.0 program
semantics `pto.structured_cv.1` with required features `structured_regions`,
`module_ownership` and `symbolic_loop_domains`. The compiler exports structured
operations, SSA values, region/block structure, call argument mappings, symbol
resolution, Buffer ownership, communication endpoints and symbolic loop bounds.
Both representations derive from the same canonical MLIR and are fingerprinted.
The existing static adapter must reject unsupported program semantics before
candidate construction; it must not invent a four-stage schedule or zero costs.

The initial structured path supports source-preserving Buffer annotation plans
only. Positive preload requires a proven cross-core phase/iteration mapping,
which the FA prologue/steady-state loop does not yet have in the static
materializer. Expose this limitation in capabilities and the coverage report.
No G2/G3/G4 claim follows from structured export or annotation round-trip.

Runtime scalars and launch dimensions are supplied as explicit scenario bindings.
Unbound values remain symbolic; their domains and dynamic execution counts are
not guessed from file names. Buffer resources are per physical core template,
not multiplied by an assumed launch grid. Unknown aliasing and asynchronous
completion remain unknown and block certified application.

## Compatibility and validation

- Flat micro v1/2.0 packages retain their existing wire semantics.
- Nested modules are accepted by native checkpoint export, including mixed
  backend containers, without CANN discovery or object compilation.
- Structured consumers negotiate the new features. Static consumers fail with a
  capability error instead of a Python field lookup or a misleading compiler error.
- Tests cover the real FA fixture, preserved wrappers and backend selection,
  recursive pipe lowering/TFree validation, scoped IDs and peer resolution,
  SSA-renaming, tampering/stale plans, and idempotent Buffer annotation import.
- Ordinary FA compilation is compared with the baseline; device verification
  requires the downstream CANN/toolchain and is reported separately.

## Alternatives

Rejecting all nested modules only masks the routing bug. Blindly flattening them
changes symbol resolution and code-generation job topology. Rewriting FA into the
static micro format invents loop alignment and loses recurrence semantics. These
are not integration strategies.

## File interface and replay

```sh
ptoas costmodel v2 export flash_attention_serial.pto \
  --profile a5_profile.json --output fa-package
ptoas costmodel v2 validate fa-package --plan buffer-plan.json
ptoas costmodel v2 apply fa-package --plan buffer-plan.json --output fa-annotated
```

The package retains `manifest.json`, `canonical.pto`, `program.json`,
`runtime_bindings.json` and `target_profile.json`. `program.json` contains:

- `modules`, `functions`, `calls`: scoped ownership, declarations/definitions,
  resolved callee IDs and operand-to-argument mappings.
- `operations`, `values`: typed attributes/results, region/block order and SSA
  expressions, including tensor-view shapes/strides/offsets as value references.
- `loops`: induction value, lower/upper/step value IDs, parent loop and optional
  constant trip count. An unknown trip count is JSON null, never zero.
- `buffers`, `aliases`: allocation sizes/alignment, storage roots, local/backing/
  borrowed ownership and per-block physical core templates. Borrowed entries
  carry their tile extent but zero additional `storage_bytes`.
- `pipes`, `transactions`: resolved backing, endpoint slot configuration,
  lexical loop scopes and push/pop/free occurrences. These are not a proof of
  dynamic transaction pairing or asynchronous completion.
- `memory_accesses`, `dependencies`, `coverage`: explicit operand effects and
  SSA edges; loop-carried memory dependencies and completion remain unknown.

Default bindings are `{ "scalars": {}, "launch": {}, "alias_contract": "unknown" }`.
For an evaluation scenario, `scalars` maps entry argument value IDs to concrete
integers, `launch.block_count` supplies the grid size, and `alias_contract` may
be `disjoint` only when the caller guarantees it. A supplied scalar binding is
recorded, not used to guess or rewrite a phase schedule. GM access bounds and
model coverage still require validation before evaluation/application.

A structured Buffer plan has protocol version `2.0`, kind `annotation_plan`,
these required features, the exact package `identity`, model provenance
(`name`, `revision`, `adapter_version`, `config_fingerprint`), and `buffers`
containing `{ "buffer_id": "opN.buffer", "count": 1 }` for every eligible local
allocation. Counts are integers in 1..256. Borrowed/backing IDs, physical addresses,
preload fields, missing/duplicate IDs and stale identities are rejected. Extensions
must be namespaced and cannot change controls. Compiler validation reconstructs
`program.json` from the canonical module instead of trusting file hashes alone.

The output contains `annotated.pto`, the plan and `validation_report.json` with
`status=annotation_only`, `optimization_applied=false`, `g2=unknown`. It validates
mapping, not the slot requirement, aggregate resource fit, or execution safety.
This deliberately separate plan kind cannot be mistaken for a schedule-bound
static `plan`. The capabilities extension `ptoas.structured_cv.v1` advertises
Buffer-only support without upgrading the static adapter's advertised semantics.

## Remaining joint work

The compiler and TileSim teams must agree on FA prologue/steady-state phase
mapping, recurrence dependencies, runtime bindings and supported operation costs.
Then extend candidate scheduling and G2 checks, followed by independent golden
and A/B device validation. This patch does not add a TileSim FA evaluator or
provide FA performance evidence. Native checkpoint tests do not replace ordinary
fat-object compilation under the downstream CANN 9.1 toolchain.
