# ADR: Pre-Allocation Logical Sync Analysis and Post-Allocation Sync Emission

Date: 2026-06-12

Status: proposed

## Context

The current PTOAS automatic synchronization pipeline runs `PTOInsertSync` after
local memory planning:

```text
PTOViewToMemref
-> PlanMemory
-> PTOResolveReservedBuffers
-> PTOInsertSync / PTOBufidSync / PTOInjectBarrierAllSync
```

This ordering is safe because the synchronizer sees the final local allocation
shape: physical tile addresses, reserved buffers, and address-space-specific
reuse decisions have already been materialized. However, it also means
`PTOInsertSync` must reason about dependencies after logical buffers may have
been folded onto the same physical local storage. In this form, a conservative
alias result can turn a logical no-dependency into an apparent physical conflict,
which may introduce extra `set_flag`/`wait_flag` pairs or `pipe_barrier`
instructions.

Moving the current `PTOInsertSync` pass wholesale before `PlanMemory` is not
sound. Before allocation, the pass lacks the final physical address and reuse
facts that can introduce real hazards. If synchronization is emitted too early,
later memory reuse may create physical dependencies that the pass never saw.

We need a design that can use the richer logical information available before
allocation without losing the physical correctness check available after
allocation.

## Goals

- Preserve correctness for all existing automatic synchronization modes.
- Allow the compiler to prove logical no-dependencies before memory planning
  erases or obscures that information.
- Give `PlanMemory` enough dependency/lifetime information to avoid allocation
  choices that create unnecessary synchronization when alternatives exist.
- Keep final sync emission after allocation, where physical reuse and address
  overlap can be validated.
- Provide a staged migration path that can be enabled and tested incrementally.

## Non-Goals

- Do not immediately replace `PTOInsertSync`, `PTOBufidSync`, or
  `PTOInjectBarrierAllSync`.
- Do not emit final `set_flag`/`wait_flag`/`pipe_barrier` before memory planning.
- Do not change PTO IR syntax, user-visible CLI defaults, or generated sync op
  semantics in this ADR.
- Do not solve event-id allocation pressure or multi-buffer event coalescing as
  part of the first implementation step.

## Decision

Introduce a two-stage automatic synchronization architecture:

```text
PreAllocLogicalSyncAnalysis
-> PlanMemory, optionally consuming sync constraints
-> PTOResolveReservedBuffers
-> PostAllocSyncRepairAndEmit
```

The key decision is to split "dependency discovery" from "sync emission".

`PreAllocLogicalSyncAnalysis` runs before `PlanMemory` and records facts about
logical buffers, slices, control predicates, and pipeline dependencies. It does
not insert hardware synchronization operations. Instead, it produces analysis
artifacts that later passes can consume.

`PostAllocSyncRepairAndEmit` runs after `PlanMemory` and remains responsible for
the final correctness boundary. It validates physical overlap and reuse,
materializes sync operations, and falls back to conservative behavior whenever
pre-allocation facts are missing or invalidated.

## Analysis Artifacts

The pre-allocation analysis should produce function-local metadata with these
concepts:

| Artifact | Purpose |
| --- | --- |
| `LogicalAccess` | Describes one op read/write in terms of logical root buffer, view chain, slice expression, memory scope, pipeline, and control predicate. |
| `LogicalDependencyEdge` | Captures RAW/WAR/WAW dependencies that are true at logical-buffer level. |
| `LogicalNoAliasFact` | Records proven non-overlap between two logical accesses before memory planning. |
| `AsyncLifetime` | Describes producer-consumer lifetime windows that must not be collapsed into unsafe physical reuse. |
| `PredicateDomain` | Identifies the control-flow predicate under which an access or dependency executes. |
| `SyncConstraintSet` | A compact summary exported to `PlanMemory` and post-allocation sync emission. |

These artifacts should be analysis-only first. They may be stored in an analysis
manager object, temporary function attributes, or a side table owned by the pass
pipeline. The initial implementation should prefer side tables or private
attributes over public IR syntax.

## PlanMemory Contract

`PlanMemory` should not be required to solve synchronization. Its contract is
limited to preserving or reporting allocation decisions that affect sync:

- It may consume `LogicalNoAliasFact` and `AsyncLifetime` to prefer allocations
  that do not create unnecessary physical conflicts.
- It must report physical reuse facts that can introduce a dependency not present
  in the logical graph.
- It must not rely on pre-allocation no-alias facts as a correctness proof after
  it chooses overlapping physical storage.
- If it cannot honor a constraint, it must leave enough information for
  post-allocation sync to insert a conservative repair synchronization.

This keeps the correctness boundary in the post-allocation phase and avoids
turning memory planning into a hidden sync emitter.

## Post-Allocation Contract

The post-allocation phase is the only phase that emits final hardware
synchronization. It must:

- Rebuild or validate `BaseMemInfo` using final addresses and root buffers.
- Reconcile logical dependencies with physical reuse dependencies.
- Preserve all existing conservative fallbacks when logical facts are absent.
- Treat pre-allocation no-alias facts as optimization hints, not as final proof,
  unless final physical allocation still preserves the same disjointness.
- Emit the same kinds of operations as today: `set_flag`, `wait_flag`,
  `pipe_barrier`, block sync, or future `get_buf`/`rls_buf` modes.

## Correctness Invariants

1. Final sync emission must be sound using post-allocation facts alone.
2. Pre-allocation facts may only remove sync if post-allocation validation
   confirms that no physical hazard was introduced.
3. Any physical reuse introduced by `PlanMemory` must be represented as a
   candidate dependency during post-allocation repair.
4. Predicate-sensitive dependencies must only be moved or merged inside regions
   where their set/wait execution predicates remain matched.
5. Loop-carried dependencies must be proven either by logical access relations
   or by physical reuse relations; unknown trip counts remain conservative.
6. Existing hand-written sync detection and early-return behavior must stay
   unchanged.

## Expected Performance Impact

The expected wins come from avoiding false conflicts caused by analyzing only
after physical address assignment:

- fewer local WAW/WAR barriers when logical buffers are independent;
- fewer cross-pipe set/wait chains caused by physical root-buffer conflation;
- better opportunity for memory planning to preserve parallel producer-consumer
  lifetimes;
- better diagnostics for whether a sync came from a logical dependency, physical
  reuse, or unknown alias information.

This ADR does not claim that moving analysis earlier always reduces sync count.
If `PlanMemory` intentionally reuses the same physical local storage across
overlapping async lifetimes, post-allocation repair must still insert sync.

## Staged Rollout

### Phase 0: Observation Only

- Add `PreAllocLogicalSyncAnalysis` behind a debug-only flag.
- Dump logical dependencies, logical no-alias facts, predicate domains, and
  async lifetimes.
- Do not feed results into `PlanMemory`.
- Do not change final sync output.

Acceptance:

- Existing sync output is byte-for-byte unchanged.
- Debug counters can separate logical dependency, physical reuse dependency, and
  unknown alias cases.

### Phase 1: Post-Allocation Validation

- Run pre-allocation analysis and post-allocation sync analysis in the same
  pipeline.
- Compare logical facts with final physical facts.
- Report cases where physical allocation created new hazards or where
  pre-allocation no-alias was preserved.
- Still do not remove sync solely because of pre-allocation facts.

Acceptance:

- No generated sync-count regression is allowed in existing tests.
- Issue-style reports can identify redundant sync candidates with a reason.

### Phase 2: PlanMemory Constraint Hints

- Allow `PlanMemory` to consume `AsyncLifetime` and no-alias hints as soft
  constraints.
- Preserve fallback behavior when constraints cannot be honored.
- Add debug output explaining when a constraint was honored or rejected.

Acceptance:

- Sync count may decrease only when post-allocation validation proves the final
  physical layout preserved no-overlap.
- Memory footprint and allocation failure behavior must be tracked.

### Phase 3: Sync Emission Pruning

- Permit post-allocation sync emission to prune a dependency when:
  - logical analysis proves no dependency,
  - final physical allocation preserves non-overlap, and
  - control predicates and loop-carried semantics are matched.
- Keep conservative behavior for unknown or invalidated facts.

Acceptance:

- Existing correctness regressions for loop backedges, nested loops, if/else,
  bind_tile/subview, and dynamic slice cases must pass.
- Issue-focused cases should show sync count reductions tied to reported proofs,
  not to unsafe deletion.

## Alternatives Considered

### Move `PTOInsertSync` Before `PlanMemory`

Rejected. This would miss dependencies introduced by physical storage reuse and
would make final correctness depend on memory planning never introducing new
hazards.

### Keep Current Ordering and Only Improve Alias Analysis

Useful but incomplete. Better post-allocation alias analysis can reduce
unknowns, but it cannot recover logical distinctions that memory allocation has
already collapsed into the same physical address.

### Make `PlanMemory` Emit Synchronization

Rejected for now. It mixes allocation policy with hardware synchronization
semantics and makes event-id allocation/control-flow motion harder to reason
about. `PlanMemory` should expose reuse facts and honor constraints, not emit
sync operations.

## Validation Plan

Correctness gates:

- single-loop backedge dependencies;
- two independent loops with the same pipe pair;
- nested loops with the same pipe pair;
- loop inside if/else with loop-carried sync;
- dynamic non-overlap loop alias cases;
- bind_tile/subview slice regressions;
- manual-sync-present early return;
- A3/A5 pipe legality checks.

Performance and sync-count gates:

- issue-style local tile whole-block reuse cases;
- vector chains such as `tload -> tgather/tadd/tsub -> tstore`;
- store-heavy MTE3 WAW cases;
- FA-like kernels with memory reuse and VF-fusion opportunities.

Debug gates:

- count logical dependencies;
- count physical reuse dependencies introduced by `PlanMemory`;
- count proven no-alias facts preserved after allocation;
- count facts invalidated by physical reuse;
- count syncs emitted for unknown alias.

## Follow-Up Work

- Define the concrete C++ data structures for `LogicalAccess`,
  `SyncConstraintSet`, and physical reuse reports.
- Add a pass-pipeline option for Phase 0 observation.
- Teach `PlanMemory` to accept soft async-lifetime constraints.
- Extend `PTOInsertSync` or introduce a new post-allocation emitter that can
  consume logical facts while preserving existing conservative fallbacks.
- Add structured debug output so issue analysis can explain why a sync was kept
  or removed.
