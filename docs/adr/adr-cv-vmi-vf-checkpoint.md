# CV scheduling over prepared VMI VF checkpoints

Date: 2026-10-22
Status: proposed

## Context

The fixed delivered FA demonstrated preload benefit. The compiler-owned VMI fusion route must first prepare its own VF. Re-running preparation after CV scheduling can change instructions, lifetimes and measured costs.

## Decision

Add opt-in A5 level3 diagnostic freeze/resume modes. Freeze follows VF fusion, layout/lowering, reduction combination, optional unroll and final preparation. It exports physical-v2 with arch, build level, CANN identity, PTOAS synchronization policy, origin and a canonical-content SHA256. Resume checks these fields and VPTO legality, preserves the VMI object-emission route, and does not re-run layout, fusion, allocation or scheduling. Normal compilation remains unchanged. physical-v1 prototype artifacts require their original compiler.

This is an experimental compiler artifact, not a model Plan or a safety certificate. A digest detects stale or altered content; it does not prove dependencies, memory lifetimes or synchronization. The restricted FA materializer must independently check its supported input and transformed protocol, seal the resulting content, and bind exact compiler/toolchain/target/bindings in its experiment manifest. No automatic optimization is enabled.

## Consequences and follow-ups

The initial matrix holds candidate selection, tiling and unroll fixed. VMI uses PTOAS InsertSync and vecscope membar, with no Bisheng VF auto-sync. Candidate-only and fused baselines use independent golden and symbols. A later production interface must add stable stage/buffer/version provenance, compiler and target capability negotiation, final allocation/synchronization feedback and general legality checks. No claim of full G2 or G4 follows from checkpoint verification.
