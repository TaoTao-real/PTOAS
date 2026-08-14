# ADR-0003: VMI grouped domain and access-aware forwarding

- Date: 2026-08-14
- Status: Proposed
- Related: ADR-0001, ADR-0002

## Context

DSv4 Sinkhorn uses compact `8x8xf32` tiles and row reductions that produce an
`8x1xf32` state.  The current preferred reduction candidate streams one row per
iteration, stores eight scalar results to UB, and a later grouped broadcast
reloads all eight values.  FusionRegion planning succeeds, but the producer and
consumer have different loop domains (`0..8` and `0..1`).  VMI loop fusion
therefore cannot combine them, and the first-version load/store elision pass
intentionally rejects grouped memory operations.

The alternative grouped reduction candidate already computes the eight compact
results as one logical grouped value.  It is independently correct, but is
currently marked non-fusible because it has no principal loop.

## Decision

1. VMI candidates may declare a `grouped_tile` principal-loop domain in
   addition to the existing `row` domain.  A grouped-tile candidate initially
   emits exactly one principal `scf.for` whose trip count is one.  The loop is
   a semantic fusion anchor, not a physical chunk loop.  Canonicalization may
   erase that one-trip loop; the enclosed operations then form one straight-
   line grouped domain inside the same FusionRegion.
2. Candidate selection prefers a legal, resource-feasible grouped-tile form
   before a row-streaming form only when the candidate explicitly declares
   `grouped_preferred`.  The existing candidate resource guard remains
   authoritative; large grouped values and ordinary reductions retain row
   streaming or ordinary PTODSL fallback.
3. Row-reduce grouped candidates keep the compact result in one logical VMI
   value and use a compact grouped store.  Compatible grouped broadcast and
   elementwise candidates use the same `0..1` domain.
4. VMI load/store elision models compact grouped access only when all of the
   following are proven equal:
   - storage root and offset;
   - element type and logical VREG type;
   - positive constant group count and group stride;
   - one scalar lane per group;
   - no mask, post-update, block stride, repeat stride, or distribution mode.
5. Forwarding normally replaces an exact grouped reload with the stored SSA
   value.  Subsequent layout operations remain explicit, and observable stores
   are retained unless ordinary dead-store rules independently prove them dead.
6. A compact grouped reload feeding `vbrc` is retained when it can lower to the
   direct `group_broadcast_load`/E2B form.  Forwarding that edge before layout
   assignment can replace one physical load-broadcast with multiple register
   packing operations.  This is a physical-form profitability barrier, not an
   alias or fusion boundary.
7. Different group counts or strides, unknown masks, dynamic access forms,
   region boundaries, synchronization, and unknown aliasing remain conservative
   barriers.  No transformation crosses a hard boundary or vecscope.

## Alternatives

### Reconstruct eight row-streaming values in mem2reg

Rejected.  It requires synthesizing a packed vector across loop iterations and
turns a memory-forwarding pass into a Sinkhorn-specific loop transformation.

### Add a Sinkhorn-only fused TileLib template

Rejected.  It would hide cross-TileOp fusion inside candidate selection and
would not generalize to other compact grouped reductions.

### Run forwarding after physical VPTO lowering

Rejected.  Physical layout and instruction selection obscure the logical group
mapping and make mask and alias proofs harder.

## Consequences

- Compact reductions and broadcasts expose a common fusion domain without
  changing the unified PTODSL backend.
- Large tiles remain protected by the resource guard.
- Grouped forwarding is an exact AccessMap extension, not a general gather or
  masked-load optimization.
- A legal forwarding opportunity is not automatically profitable.  Direct
  memory-to-broadcast forms remain materialized until layout-aware costing can
  compare them with register rematerialization.
- The one-trip semantic loop normally canonicalizes away.  Candidate-only and
  fused A5 results must still both be reported because grouped candidate
  selection and access forwarding are separate effects.

## Validation

- Positive and negative lit tests for exact grouped forwarding, mismatched
  stride/group, alias barriers, and observable stores.
- PTODSL metadata tests for grouped-domain selection and resource fallback.
- DSv4 Sinkhorn pass dumps must show grouped candidates and report the grouped
  VMI load-elision delta.  A zero delta is valid when retaining a reload enables
  a cheaper direct physical broadcast form.  Loop counts are reported at every
  fusion stage; a one-trip grouped loop may already be gone before loop fusion.
- CAmodel and A5 must pass fixed-input cold-first correctness.  Final acceptance
  requires an independent golden and reports B/C/D separately.
