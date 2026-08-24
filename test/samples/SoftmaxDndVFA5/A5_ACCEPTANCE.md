# Softmax Dn Base32/Base64/Base128 A5 evidence

This is the checked-in summary of the 2026-08-24 A5 experiment.  Raw VPTO,
objects, profiler CSV files, binaries, and device logs remain in the immutable
private-lab experiment `20260824-002-softmax-3width-a5-matrix`.

The fixed workload is FP32 `[B=4,M=16,N]`, reducing along `M`, for
`N=32/64/128`.  Every profiler sample uses the same finite-sensitive input and
is followed immediately by output validation.  Five serial samples were taken
on device 3, alternating forward and reverse variant order.

## Correctness

For the exact-onehot input, independent golden, AC-U, AC-F, A, B, C, and D are
byte-exact for all three widths:

| N | FP32 output SHA-256 |
| ---: | --- |
| 32 | `0da78442799714d05371fa09784d1ec0ce8be4563f75abb8ea5c4c905b0fe1ea` |
| 64 | `0539900d506093abafad8333880452b46edbd2e75593dde146ff5e02d162ef65` |
| 128 | `88ad34f07f5cc40cfca2ddfb3fb6f5f6fc2b233fc925ff7b04947aac37e0b217` |

For finite-sensitive input, AC-U, AC-F, B, C, and D are byte-identical.  A uses
the ordinary reduction association and differs by at most 1--2 FP32 ULP.  Over
all paths and widths, maximum error versus the independent host golden is
`4.47e-8`; maximum `sum(axis=M)-1` error is `1.19e-7`.

The installed CANN 9.2 `SoftmaxDnVF` wrapper passes its nominal `dstTensor` as
the Base implementation's source/exp scratch and its nominal `srcTensor` as
the normalized destination.  The fixture follows the installed header rather
than the older pasted source's reversed pointer mapping.

## Core four-path performance

| N | path | vector median us | min | max | CV | task median us | cycles median | speedup vs AC-U | speedup vs B |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | AC-U | 10.155 | 10.034 | 10.176 | 0.63% | 17.196 | 25324 | 1.00x | 0.204x |
| 32 | AC-F | 0.246 | 0.245 | 0.246 | 0.20% | 7.472 | 9514 | 41.28x | 8.439x |
| 32 | B | 2.076 | 2.074 | 2.081 | 0.12% | 6.585 | 9363 | 4.89x | 1.00x |
| 32 | D | 1.854 | 1.850 | 1.857 | 0.13% | 6.271 | 8650 | 5.48x | 1.120x |
| 64 | AC-U | 10.039 | 9.985 | 10.142 | 0.54% | 17.419 | 26034 | 1.00x | 0.059x |
| 64 | AC-F | 0.238 | 0.237 | 0.238 | 0.21% | 7.561 | 10072 | 42.18x | 2.487x |
| 64 | B | 0.592 | 0.588 | 0.594 | 0.35% | 5.838 | 8126 | 16.96x | 1.00x |
| 64 | D | 0.410 | 0.409 | 0.414 | 0.42% | 5.793 | 7942 | 24.49x | 1.444x |
| 128 | AC-U | 19.520 | 19.480 | 19.624 | 0.25% | 27.572 | 42679 | 1.00x | 0.039x |
| 128 | AC-F | 1.814 | 1.812 | 1.816 | 0.09% | 10.190 | 14179 | 10.76x | 0.418x |
| 128 | B | 0.759 | 0.757 | 0.762 | 0.22% | 6.553 | 9128 | 25.72x | 1.00x |
| 128 | D | 0.668 | 0.666 | 0.675 | 0.51% | 6.401 | 8923 | 29.22x | 1.136x |

Here AC-U is ordinary AscendC, AC-F is the installed hand-written VF, B is the
same selected VMI candidate set without region/loop/elision, and D enables the
full VMI fusion path.

## Lowering attribution

The VMI path selects `tcolmax`, `tcolexpandexpdif`, `tcolsum`, and
`tcolexpanddiv` for each of the four independent tiles.  `vexpdif` provenance
is preserved through unified-to-legacy lowering and recovered as the physical
A5 instruction.

| N | variant | loops | VLD sites | VST sites | membar sites | vexpdif sites |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | B | 16 | 36 | 20 | 36 | 8 |
| 32 | C | 12 | 36 | 20 | 36 | 8 |
| 32 | D | 12 | 20 | 12 | 24 | 8 |
| 64 | B | 16 | 36 | 20 | 20 | 8 |
| 64 | C | 12 | 36 | 20 | 20 | 8 |
| 64 | D | 12 | 20 | 12 | 8 | 8 |
| 128 | B | 16 | 72 | 40 | 20 | 16 |
| 128 | C | 12 | 72 | 40 | 20 | 16 |
| 128 | D | 12 | 56 | 40 | 16 | 16 |

D forwards the completed max and sum vregs across phases and removes their UB
roundtrips.  It intentionally preserves the exp matrix materialization: the
denominator is unavailable until the exp/sum loop completes, so the final
divide loop must reload those values.  The explicit VST-to-VLD barrier matches
the installed hand-written VF.

The remaining Base32/Base64 gap to AC-F is explained by a known structural
limit: PTO currently emits four independent tile pipelines (12 loops), while
the hand-written VF interleaves four tiles in three loops and keeps four sets
of accumulators live.  Cross-independent-tile loop coalescing and packed short
row scheduling are follow-up compiler work, not correctness exceptions.
