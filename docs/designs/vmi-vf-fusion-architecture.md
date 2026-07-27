# VMI VF Fusion Architecture Proposal

Status: proposed

## 1. Background

Existing TileOp fusion already has a high-level planning pipeline:

```text
PreFusionAnalysis
  -> FusionPlan
  -> OpScheduling
  -> PTOFusionRegionGen
```

This pipeline works before TileOps are expanded. It can analyze TileOp-level
dataflow, iteration domains, boundaries and liveness, then marks selected TileOps
with `pto.fusion.group_id` and `pto.fusion.order`. `OpScheduling` consumes these
attributes and compacts members of the same group into a contiguous span. Finally
`PTOFusionRegionGen` wraps the span into `pto.fusion_region`.

VMI VF fusion works after TileOps have been expanded into VMI row loops. The VMI
layer has better visibility into logical vector values, masks, load/store forms,
and final lowering constraints. However, if the VMI layer starts from raw
`scf.for` operations only, it loses the TileOp semantic context that already
exists in `PreFusionAnalysis`.

The first VMI VF fusion implementation should therefore reuse the TileOp fusion
pipeline as the candidate generator, and let VMI passes provide the final
legality proof and transformation.

## 2. Design Goal

The goal is to make VMI VF fusion consume the existing TileOp fusion result
instead of creating an unrelated loose region.

The intended split is:

```text
TileOp layer:
  discover high-level candidates and build ordered fusion regions

VMI layer:
  verify expanded row-loop legality
  fuse compatible loops
  eliminate redundant UB load/store through logical vreg forwarding
```

This keeps high-level semantic analysis in the TileOp pipeline and low-level
correctness checks in the VMI pipeline.

## 3. Non-goals

This proposal does not require:

- a new VMI-specific FusionPlan strategy in the first version;
- a general VMI alias analysis for arbitrary user-written VMI;
- fusion across MTE, Cube, sync, communication or unknown calls;
- dynamic tail-mask fusion;
- register-pressure driven cost modeling;
- physical instruction scheduling.

These can be added incrementally after the basic architecture is stable.

## 4. Proposed Pipeline

For the PTODSL VMI TileLib path, the recommended pipeline is:

```text
InsertTemplateAttributes
  -> FusionPlan
  -> OpScheduling
  -> PTOFusionRegionGen
  -> ExpandTileOp(to VMI)
  -> Inline
  -> VMIFusionReadyAnalysis
  -> PTOVmiLoopFusion
  -> PTOVmiLoadStoreElision
  -> VMI layout assignment
  -> VMIToVPTO
```

`FusionPlan`, `OpScheduling`, and `PTOFusionRegionGen` remain TileOp-level
passes. They decide the candidate region before expansion.

`VMIFusionReadyAnalysis`, `PTOVmiLoopFusion`, and
`PTOVmiLoadStoreElision` are VMI-level passes. They must treat the TileOp
fusion region as a candidate, not as a correctness proof.

## 5. Reusing TileOp FusionPlan

The first version should reuse the existing conservative TileOp strategy instead
of adding a separate loose VMI strategy.

Reasons:

- Existing `PreFusionAnalysis` already builds TileOp DFG edges, write-instance
  liveness, value liveness, iteration-domain classes and boundaries.
- Existing `FusionPlan` already converts the analysis result into stable
  `group_id` and `order` metadata.
- Existing `OpScheduling` already knows how to move marked TileOps into a
  contiguous span without redefining membership.
- A loose VMI-only strategy can easily become a container generator rather than
  a real fusion plan.

The VMI path may still need to extend the existing planning capability:

- add missing VMI TileLib-supported TileOps to the plannable op set;
- classify row-wise, row-reduce, row-broadcast, layout-phase and boundary ops;
- ensure the iteration-domain rule matches the VMI logical-row execution model;
- keep unsupported or unknown TileOps outside fusion groups.

## 6. Candidate Contract

A TileOp fusion region is a candidate region. It means:

```text
The TileOp layer believes these TileOps are worth analyzing together and can be
scheduled as one ordered region.
```

It does not mean:

```text
All VMI loops inside this region are definitely fusible.
All intermediate UB load/store pairs are definitely removable.
```

The VMI layer must still prove the following conditions.

### 6.1 Region Conditions

- All fused code comes from TileLib expansion.
- The region contains only supported VMI TileLib candidates.
- MTE, Cube, sync, communication, unknown calls and unsupported layout-changing
  operations form hard boundaries.
- Dynamic `valid_shape` and dynamic tail masks are rejected in the first version.

### 6.2 Loop Conditions

- Each expanded TileOp has exactly one principal row loop.
- The loop represents one logical row per iteration.
- There is no generated physical-chunk inner loop.
- Candidate loops have compatible lower bound, upper bound and step.
- The loop body does not contain unsupported cross-iteration memory dependence.

### 6.3 Memory Conditions

- Supported load/store forms are continuous single-result `vload` and
  single-value `vstore` in the first version.
- Unknown memory effects invalidate forwarding.
- Gather, scatter, non-continuous load/store, grouped access and dynamic address
  expressions are boundaries unless explicitly modeled.
- A store-to-load forwarding candidate must be same logical row, same logical
  lane range, same dtype-compatible value and mask-safe.

### 6.4 Mask Conditions

- Full static masks are supported.
- Static lane masks may be supported only when all consumers are proven to read
  the same lane subset.
- Mixed masked and mask-free consumers force full-read semantics.
- Dynamic tail masks are rejected until the VMI layer can prove coverage and
  dominance precisely.

## 7. Preserving TileOp Provenance After Expansion

VMI passes should not need to rediscover all TileOp semantics from raw `scf.for`.
After `ExpandTileOp(to VMI)` and `Inline`, each principal row loop should retain
minimal provenance:

```text
pto.vmi.origin_group_id
pto.vmi.origin_order
pto.vmi.origin_tileop
pto.vmi.origin_candidate
pto.vmi.logical_lanes
```

These attributes are not optimization decisions. They are debug and analysis
anchors that connect VMI loops back to the TileOp fusion plan.

The attributes should be placed on the principal row loop, not on every VMI
instruction. This avoids turning every helper op into a separate fusion scope.

## 8. VMIFusionReadyAnalysis

`VMIFusionReadyAnalysis` should be added before destructive VMI transforms.

It walks each `pto.fusion_region`, groups principal row loops by TileOp
provenance, and reports one of:

```text
fusion_eligible
fusion_eligible_with_pressure_gate
layout_phase
boundary_only
unsupported
```

The analysis should check:

- all loops belong to the same TileOp fusion group;
- loop headers are compatible;
- loop bodies satisfy the supported memory and mask forms;
- TileOp order is a valid topological order;
- layout-phase ops such as gather are not treated as normal mem2reg producers;
- all memory effects are visible and conservatively modeled.

Production transforms should consume this analysis result instead of making
ad-hoc decisions from raw operation names.

## 9. Loop Fusion

`PTOVmiLoopFusion` should only fuse loops that passed `VMIFusionReadyAnalysis`.

The transformation should preserve per-row execution order:

```text
before:
  for row:
    body of TileOp A
  for row:
    body of TileOp B

after:
  for row:
    body of TileOp A for this row
    body of TileOp B for this row
```

This is legal only when the second body does not depend on a different
iteration of the first loop. Accesses such as `A[row + 1]`, indirect gather,
scatter or unknown pointer arithmetic must block fusion unless explicitly
modeled.

## 10. Load/Store Elision and Mem2Reg

`PTOVmiLoadStoreElision` should use two inputs:

- TileOp-level DFG and write-instance information to identify meaningful
  producer-consumer or shared-input opportunities;
- VMI-level load/store analysis to prove the concrete forwarding is safe.

The TileOp DFG is a strong hint, not the final proof. It tells the VMI pass
which relationships are semantically intended:

```text
TEXP writes X
TROWSUM reads the latest X
```

The VMI pass must still verify that the expanded operations match this relation:

```text
same row
same lane range
compatible mask
no intervening aliasing write
supported load/store form
```

This prevents the VMI pass from implementing a broad alias analysis in the first
version while still avoiding unsafe forwarding.

## 11. Why Not Use VMI-only Analysis First

VMI-only analysis can see `vstore -> vload` patterns, but it lacks the original
TileOp intent. Starting from raw VMI would require reconstructing:

- whether two accesses represent TileOp producer-consumer semantics;
- whether an intermediate buffer is an intended DPS write instance;
- whether an operation is elementwise, reduce, broadcast or layout phase;
- whether a store is observable outside the fusion region;
- whether a similar-looking access came from user-written VMI.

This is possible eventually, but it is too expensive and risky for the first
version. Reusing TileOp FusionPlan narrows the search space and gives VMI a
clear candidate contract.

## 12. Validation Plan

The initial validation should cover both positive and negative cases.

Positive cases:

- elementwise chain: `tadd -> tmul -> texp`;
- row-reduce chain: `tmax/texp -> trowsum`;
- row-broadcast chain: `trowmax -> trowexpandsub -> texp`;
- independent same-domain elementwise ops in one region;
- shared-input reuse in one fused loop.

Negative cases:

- different loop headers;
- dynamic `valid_shape`;
- dynamic tail mask;
- mixed masked and mask-free users of the same load;
- `vsel` consumer with ambiguous lane-read semantics;
- gather/scatter boundary;
- unknown memory-effect op between store and load;
- non-continuous or grouped `vload/vstore`;
- cross-iteration access such as `row + 1`.

Each test should verify both structure and behavior:

```text
TileOp fusion metadata exists
OpScheduling creates a contiguous region
VMI expansion preserves row-loop provenance
FusionReadyAnalysis reports the expected status
LoopFusion fuses only eligible loops
LoadStoreElision removes only proven redundant pairs
VMIToVPTO leaves no residual VMI
```

## 13. Incremental Implementation Plan

### Phase 1: Planning Reuse

- Keep the existing TileOp FusionPlan strategy as the default VMI candidate
  generator.
- Extend the plannable TileOp set only for VMI TileLib-supported operations.
- Add tests proving `FusionPlan -> OpScheduling -> PTOFusionRegionGen` works
  for VMI TileLib candidates before expansion.

### Phase 2: Provenance and Readiness

- Preserve minimal TileOp provenance on principal VMI row loops.
- Add `VMIFusionReadyAnalysis`.
- Add checker tests for eligible, boundary and unsupported regions.

### Phase 3: Conservative VMI Transform

- Gate `PTOVmiLoopFusion` on readiness results.
- Restrict load/store elision to continuous single-result/single-value forms.
- Invalidate on unknown memory effects.
- Add negative tests for mask, gather/scatter and non-continuous access.

### Phase 4: Capability Expansion

- Add support for layout-phase ops after their forwarding rules are explicit.
- Add dynamic tail-mask support.
- Add register-pressure and cost gates.
- Consider a VMI-specific FusionPlan strategy only if the reused TileOp
  strategy proves too restrictive.

## 14. Summary

The recommended architecture is:

```text
TileOp FusionPlan builds a semantically meaningful candidate region.
OpScheduling makes the selected TileOps contiguous.
VMI passes prove low-level loop and memory legality before transforming.
```

This avoids duplicating high-level fusion analysis in VMI while still preserving
the stronger checks needed by VMI loop fusion and mem2reg.
