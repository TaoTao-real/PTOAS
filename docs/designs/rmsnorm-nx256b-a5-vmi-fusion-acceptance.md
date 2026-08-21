# RMSNorm `N x 256B` row-VF A5 acceptance

## Scope and provenance

This acceptance covers the static `N=64` one-VL row workload defined by
[ADR-0003](adr/0003-multi-row-one-vl-rmsnorm-fusion.md):

- input/output: BF16 `[64,64]`;
- gamma: BF16 `[1,64]`;
- row computation: FP32 `[1,64]`, exactly 256 bytes and one A5 VL;
- target: Ascend950PR_9599 / `dav-c310-vec`;
- device: healthy and idle device 4 on `ptoas-a5`;
- source: `0e57d1572612841abeb33c019da7095e8a731f84`;
- experiment: `20260821-rmsnorm-nx256b-sixway-0e57d1572-a5-n64-r1`;
- finalized: `2026-08-21T09:43:04Z`.

The experiment used the compiler binary built and correctness-qualified at
`6cdb5990d`. The only source difference at `0e57d1572` is the A5 comparison
harness under `test/samples/RMSNormRowVFA5/a5_harness`; no compiler source
changed. The experiment manifest records both source and compiler hashes.

## Data and computation equivalence

All six paths use the same GM buffers, launch shape, BF16 input, BF16 gamma,
epsilon, one-VL FP32 row reduction, FP32 operation order, and final RINT BF16
conversion:

```text
BF16 x -> FP32 -> square -> one row sum -> * (1/64) -> + epsilon -> sqrt
       -> x / rms -> * FP32 gamma -> RINT BF16 y
```

Two deterministic datasets are checked against an independent CPU golden:

- `exact-association` uses exactly representable values so the fixed FP32
  reduction association is byte-stable;
- `layout-sensitive` gives every row a distinct signed pattern and gamma
  pattern to detect cross-row, mask, layout, and broadcast mistakes.

For each of AC-U, AC-F, A, B, C, and D, both datasets passed one cold and two
non-profile runs. The layout-sensitive dataset also passed the warmup and every
profile run. The result is `42/42` byte-exact executions and `30/30` valid
profiles, with no mismatch observed.

## Six-path attribution

| Path | Implementation |
| --- | --- |
| AC-U | AscendC ordinary vector API |
| AC-F | AscendC manual `__simd_vf__` |
| A | ordinary PTODSL/VPTO |
| B | selected VMI candidates with fusion disabled |
| C | B plus selected-VMI region and row-loop fusion |
| D | C plus vreg forwarding, gamma hoisting, and UB traffic elision |

Five serial samples were collected per path after warmup. All vector CV values
were below 2%, so the protocol did not require extension to nine samples.

| Path | Vector median us | Min | Max | Stddev | CV | Task median us | AIV median us | AIV cycles | Vector us/row | Speedup vs AC-U | Speedup vs B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC-U | 276.719 | 276.636 | 277.747 | 0.420 | 0.152% | 367.715 | 355.060 | 35,219 | 4.3237 | 1.00x | 0.36x |
| AC-F | 7.503 | 7.446 | 7.593 | 0.048 | 0.633% | 90.769 | 77.470 | 7,464 | 0.1172 | 36.88x | 13.36x |
| A | 138.893 | 138.870 | 138.929 | 0.020 | 0.014% | 199.019 | 192.700 | 18,986 | 2.1702 | 1.99x | 0.72x |
| B | 100.260 | 99.961 | 100.330 | 0.163 | 0.163% | 158.243 | 150.960 | 14,816 | 1.5666 | 2.76x | 1.00x |
| C | 151.130 | 151.103 | 151.236 | 0.046 | 0.030% | 210.428 | 203.100 | 20,027 | 2.3614 | 1.83x | 0.66x |
| D | 8.470 | 8.412 | 8.516 | 0.036 | 0.428% | 71.740 | 61.830 | 5,901 | 0.1323 | 32.67x | 11.84x |

The primary four-path comparison is AC-U, AC-F, B, and D. D reduces vector
time by 91.55% relative to B. The deterministic bootstrap 95% interval for
`D/B` is `[0.08390, 0.08494]`, which excludes 1.0 by a wide margin. D is also
faster than A. Its vector stage is 12.9% slower than hand-written AC-F, while
its measured task and AIV times are lower in this experiment.

## Lowering evidence and optimization attribution

| Path | Row loops | Static VLD | Static VST | Internal membar | Residual VMI ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 9 | 12 | 10 | 13 | 0 |
| B | 9 | 12 | 10 | 13 | 0 |
| C | 1 | 12 | 10 | 10 | 0 |
| D | 1 | 2 | 1 | 0 | 0 |

The C result is important: region planning and row-loop fusion alone reduce
nine loops to one, but retaining all intermediate UB round trips and membars
makes vector time worse than B. The D result isolates the effective change:

- keep x, square, reduction state, divisor, normalized value, and scaled value
  in SSA/vregs across the row pipeline;
- replace the compact scalar `VL1 VST -> BRC_B32 VLD` bridge with a register
  duplicate when the single-use address and mask proof succeeds;
- hoist the unique read-only gamma conversion to the preheader and pass the
  vreg into the flattened row loop;
- eliminate intermediate UB VLD/VST and computation-internal membars;
- retain only gamma/x input loads and the y output store in the VF computation.

For `N=64`, D therefore executes 65 input VLD operations (one gamma plus one x
load per row) and 64 output VST operations, with zero intermediate UB
load/store operations and zero VF-internal membars. Required GM-to-UB and
UB-to-GM synchronization remains outside the computation-stage count.

The transformation is enabled only after proving the selected-VMI-only region,
principal row domain, static byte range, mask, unique UB root, no alias or
escape, no sync or unknown call, and exactly one row pipeline. Failure of any
proof keeps the C path intact.

## Acceptance and deferred gates

The target `N=64` performance feature is accepted:

- all output hashes are byte-exact across independent golden and six paths;
- D is more than 3% faster than B and its 95% interval excludes 1.0;
- D is not slower than A;
- D/C is explained by the measured elimination of intermediate UB traffic and
  internal membars.

Per the performance-first execution decision, the `N=1/8/32` extension,
assembly branch/spill census, and unrelated pre-existing TileFusion lit
expectation updates remain follow-up gates. They do not change the completed
`N=64` correctness or performance result.

## Reproduction

The committed harness provides the exact generator, build, correctness,
profiling, parsing, and summary flow:

```bash
ACL_DEVICE_ID=4 ROWS_LIST=64 PROFILE_REPEATS=5 \
  test/samples/RMSNormRowVFA5/a5_harness/run_a5_matrix.sh \
  EXPERIMENT SOURCE PTOAS MLIR_PYTHON_ROOT PTO_ISA_ROOT
```

Full VPTO, binaries, compiler logs, profiler CSV files, device snapshots, and
hash manifests are retained only in the finalized private-lab experiment.

## PTOAS PR summary

The PR should retain the nine atomic commits starting at
`a9797c4687094be743026a3ae9652a225f87a47a`. It establishes selected-VMI-only
FusionRegion membership, adds the AscendC-equivalent `N x 256B` fixture and
missing one-VL candidates, plans fusion using a proven principal row domain,
fuses the row pipeline, forwards values and hoists gamma, and adds the A5
six-path acceptance harness. The central measured result is the reduction from
`12 VLD / 10 VST / 10 membar` in C to `2 VLD / 1 VST / 0 membar` in D, with
byte-exact output and an 11.84x vector speedup over unfused VMI B.
