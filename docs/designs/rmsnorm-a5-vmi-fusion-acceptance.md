# RMSNorm 1VL A5 VMI fusion acceptance

## Scope

This report records acceptance of the AscendC-aligned RMSNorm 1VL flow on A5.
The accepted compiler source is clean commit
`eb423c6218e1a9d42c846d7ddac47cf8c4251703`.

The PTO computation follows the manual AscendC vector function association:

1. Load one BF16 row into UB.
2. Accumulate 64 FP32 lanes across 64 chunks with one loop-carried vreg.
3. Perform one `trowsum` after the chunk loop.
4. Apply `tmuls -> tadds -> tsqrt` once per row.
5. Process the output in 64-element chunks: widen x and gamma, divide by the
   compact scalar, multiply gamma, narrow with RINT, and store BF16.

There is no full-row conversion and no per-chunk horizontal reduction.

## Selected candidates and regions

Candidate selection sees 16 TileOps. Thirteen select VMI candidates; the two
GM-to-UB `tload` operations and one UB-to-GM `tstore` remain PTODSL hard
boundaries. Full mode plans five selected-VMI-only regions with 13 TileOp
members:

- accumulator initialization;
- BF16 widening, lane square, and lane accumulation;
- final row reduction;
- scalar `tmuls -> tadds -> tsqrt`;
- apply-loop widen/widen/divide/multiply/narrow.

The final VPTO and emitted LLVM IR contain no residual VMI operations.

## Vector traffic

The static final VPTO traffic is:

| Variant | VLD | VST | `mem_bar` |
| --- | ---: | ---: | ---: |
| A: ordinary/off | 15 | 13 | 14 |
| B: VMI/off | 15 | 13 | 14 |
| C: VMI/loop | 15 | 13 | 14 |
| D: VMI/full | 3 | 1 | 0 |

D contains only the manual-VF boundary traffic: one reduction-x load, one
apply-x load, one apply-gamma load, and one packed output store. Accumulator
and scalar scratch roots are not materialized. In particular, there is no
per-chunk accumulator VLD/VST, no reload before `trowsum`, and no vector-scope
memory barrier.

The last false `VST_VLD` hazard came from treating an `UNPK_B16` result vreg as
a 256-byte source-memory access. On A5 it reads 64 BF16 source elements, or 128
bytes, into the 128-lane unpack representation. Using the physical 128-byte
payload proves that the final input range ends exactly where the output range
begins. The proof remains conservative for unknown affine roots and real
overlap; a dedicated negative test retains the barrier for a two-byte overlap.

## Correctness protocol

Validation used device 1 on `Ascend950PR_9599` with CANN 9.2.0. Device health
was `OK`, with error code `NA`, before and after both correctness and profiling.
Inputs, output initialization, launch parameters, synchronization, and an
independent FP32 lane-association golden were fixed across all variants.

The variants were:

- AC-U: ordinary AscendC with the same lane association;
- AC-F: hand-written `__simd_vf__`;
- A: ordinary PTODSL/VPTO;
- B: the VMI candidate set with fusion disabled;
- C: B plus region and loop fusion;
- D: C plus accumulator/scalar promotion and load/store elision.

Cold D and A passed, followed by two non-profile executions of every variant.
All 14 outputs were byte-exact to the independent golden. During profiling,
all 30 outputs were checked again and were byte-exact. The common output
SHA-256 is
`dee6899bf0ce95d91fb9faedb402f7cfbde528bac5f0a9518f1714f357d0ef61`.

## Performance protocol and results

Each variant had one warmup followed by five serial `msprof` samples. Odd
rounds used forward order and even rounds reverse order. Lower latency is
better.

| Variant | Median task (us) | Min | Max | Stddev | CV | AIV cycles | Vector (us) | Scalar (us) | MTE2 (us) | MTE3 (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC-U | 139.227 | 138.564 | 139.886 | 0.437 | 0.31% | 226954 | 108.775 | 34.742 | 15.410 | 8.181 |
| AC-F | 28.998 | 28.615 | 30.388 | 0.638 | 2.18% | 45171 | 3.494 | 5.804 | 15.742 | 8.307 |
| A | 59.661 | 59.372 | 59.932 | 0.205 | 0.34% | 96870 | 46.775 | 0.984 | 5.896 | 3.372 |
| B | 72.137 | 71.931 | 72.711 | 0.285 | 0.39% | 117461 | 59.711 | 1.003 | 5.757 | 3.329 |
| C | 72.294 | 71.990 | 73.071 | 0.466 | 0.64% | 117808 | 59.720 | 1.041 | 5.945 | 3.349 |
| D | 15.190 | 15.115 | 15.689 | 0.215 | 1.41% | 23434 | 2.325 | 1.030 | 5.796 | 3.304 |

Median latency ratios and improvements are:

| Ratio | Latency ratio | Improvement |
| --- | ---: | ---: |
| B/A | 1.209115 | -20.911% |
| C/B | 1.002176 | -0.218% |
| D/C | 0.210114 | 78.989% |
| D/A | 0.254605 | 74.539% |
| D/B | 0.210572 | 78.943% |

D/B exceeds the required 3% improvement and the 1.405% D/B noise floor, and D
is faster than A. The performance feature is accepted.

The attribution is important: candidate selection alone regresses B versus A,
and ordinary region/loop fusion is neutral within noise for C versus B. The
accepted gain comes from cross-phase vreg promotion and the resulting removal
of accumulator/scalar UB traffic and barriers. For this fixed workload D also
measures below the hand-written AC-F time; this is a measured result, not a
general claim that generated VMI fusion always outperforms hand-written VF.

## Retained evidence

Raw inputs, dumps, final VPTO, fat objects, correctness logs, individual
profiles, and device-health records are retained outside the repository in
immutable private-lab experiments:

- `20260821-rmsnorm-1vl-c17-eb423c621-vf-boundary-correctness-r1`;
- `20260821-rmsnorm-1vl-c18-eb423c621-sixway-performance-r2`.

The exact device-object Bisheng command was also retained. Bisheng 15.0.5
rejects device-side `-S`, and its `-save-temps` path crashed during the earlier
assembly attempt, so this acceptance makes no final assembly-text claim. VPTO,
LLVM IR, object generation, runtime correctness, and hardware profile evidence
are complete.
