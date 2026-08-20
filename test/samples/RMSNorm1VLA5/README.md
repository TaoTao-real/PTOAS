# RMSNorm 1VL A5 acceptance fixture

This fixture fixes the functional and floating-point association contract used
for the AscendC AC-U / AC-F and PTO VMI-fusion comparison.

- Workload: BF16 `x/y = [8,4096]`, BF16 `gamma = [4096]`.
- Arithmetic: FP32, epsilon `1e-6`, final BF16 conversion uses RINT.
- Reduction: one FP32 `1x64` lane accumulator updated by 64 chunks, followed
  by exactly one `trowsum`.
- Apply: both x and gamma are converted in the 64-lane loop; divide, gamma
  multiply, and FP32-to-BF16 conversion remain in that loop.
- Transfer: one row tload/tstore boundary; DMA never enters a VMI region.

The golden uses deterministic nonzero BF16 inputs and reproduces the same
lane-wise accumulation order. It is independent from the compiler lowering.
The final A5 acceptance compares the byte-exact BF16 output hash across AC-U,
AC-F, and PTO A/B/C/D variants.

Alignment checklist:

| Property | AC-U | AC-F | PTO fixture |
| --- | --- | --- | --- |
| Shape | `[8,4096]` | `[8,4096]` | `[8,4096]` |
| Input/output | BF16 | BF16 | BF16 |
| Accumulator | FP32 64 lanes | FP32 64 lanes | FP32 `1x64` |
| Chunk order | increasing, 64 chunks | increasing, 64 chunks | increasing, 64 chunks |
| Final reduction | once | once | one `trowsum` |
| Narrowing | RINT | RINT | RINT |
| Launch | one vector block | one vector block | one vector block |
