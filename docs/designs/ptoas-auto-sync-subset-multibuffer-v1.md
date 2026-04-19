# PTOAS Auto Sync Subset Multibuffer V1

## Summary

`PTOInsertSync` now recognizes ping/pong buffers that are created by slicing a
single root workspace with `pto.subset` or the lowered `memref.subview` form.
When the root buffer carries `pto.multi_buffer = 2`, autosync can reuse the
existing dynamic event-id emission path and generate slot-local back-edge sync.

## User Contract

- The root workspace must carry `pto.multi_buffer = 2`.
- Ping and pong must come from the same root buffer.
- Slice geometry must be statically provable:
  - equal-size tiles
  - non-overlapping
  - exactly one partitioned dimension
  - the partitioned dimension is split evenly into two slots
- Any other shape falls back to normal alias/range-based autosync.

## Implementation Notes

- `BaseMemInfo` carries optional multibuffer slot metadata:
  - `multibufferRoot`
  - `multibufferSlot`
  - `multibufferFactor`
  - `isMultibufferSlotValid`
- `PTOIRTranslator` derives slot metadata from `pto.subset` or
  `memref.subview` when the slice is a valid ping/pong partition of the root.
- `InsertSyncAnalysis::GetEventIdNum` first checks for slot-aware multibuffer
  pairs, then falls back to the legacy `baseAddresses.size() == 2` path.

## Non-goals

- No new CLI flags or Python APIs.
- No support for `pto.multi_buffer > 2`.
- No support for dynamic subset/subview geometry.
