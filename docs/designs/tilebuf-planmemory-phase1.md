# Tile Buffer -> PlanMemory (Phase-1)

## Scope
- Base: `origin/main`
- Phase-1 only: `tile_buffer -> PlanMemory`
- Explicitly out of scope:
  - Sync migration
  - New MultiBuffer capabilities

## Why
PlanMemory previously consumed mainly legacy buffer alias/shape/space signals.
Tile metadata (`bind_tile/subset/bitcast/treshape`) was available but not normalized
as a reusable semantic layer.

This phase moves PlanMemory to a tilebuf-native input path while keeping the
core planner (`MultiSpecPlan`, rollback/reuse) unchanged.

## Changes
1. Unified tile semantic extraction in `Utils`:
- alias unification: `bind_tile/subset/bitcast/treshape` + view-like ops
- root traceback: `tracebackBufferRoot(...)`
- semantic record: `TileBufferSemantics` (root/scope/shape/valid/config/view-kind/bits)

2. PlanMemory liveness/buffer info wiring:
- `MemLivenessAnalysis` uses unified alias API
- local buffer definition is `pto.alloc_tile` / `pto.declare_tile`
- `GetBufferInfo` consumes tile-native semantic extraction

3. No algorithm rewrite:
- Allocation/reuse/rollback algorithm unchanged
- No legacy bridge fallback is kept in the user path

## Capability -> Test Mapping
- Unified semantic smoke:
  - `test/basic/tilebuf_semantic_smoke.pto`
- Alias/root trace across bind + view chain:
  - `test/basic/tilebuf_root_trace.pto`
- View-like alias chain (`subset -> treshape -> bitcast`) stability:
  - `test/samples/planmemory/tilebuf_alias_chain.py`
  - `test/samples/runop.sh` check for `TRESHAPE` + `TASSIGN`
- PlanMemory auto-address reachability:
  - `test/basic/tilebuf_auto_addr_assign.pto`
  - `test/samples/planmemory/tilebuf_planmemory_auto_addr.py`
  - `test/samples/runop.sh` check for `TASSIGN` + `TPRINT`
- Address contract (manual addr preserve):
  - `test/basic/tilebuf_manual_addr_preserve.pto`

## Next (Phase-2)
- Sync analysis/input migration to consume the same tile semantic layer.
- Remove remaining internal fallback branches after boundary coverage is complete.
