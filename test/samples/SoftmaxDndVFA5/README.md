# Softmax Dn three-width A5 VF fixture

This fixture is the algorithm-equivalent comparison for the installed
Compressor `vf_softmax.h`.  It intentionally exercises a two-dimensional tile
instead of a row-wise `[1,N]` VF body:

```text
x[B,M,N]
  -> max(axis=M)        [B,1,N]
  -> exp(x - max)       [B,M,N]
  -> sum(axis=M)        [B,1,N]
  -> exp / sum          [B,M,N]
```

The fixed performance workload is `B=4`, `M=16`, FP32.  Three inner widths
cover the physical mappings on either side of one A5 FP32 vector length:

| case | inner bytes | purpose |
| --- | ---: | --- |
| Base32 | 128 B | narrow, half-VL mapping |
| Base64 | 256 B | exact one-VL mapping |
| Base128 | 512 B | two-VL mapping |

`softmax_dnd_ascendc.cpp` is shared by the two AscendC controls.  AC-U is built
without `SOFTMAX_USE_VF` and uses ordinary AscendC vector APIs.  AC-F is built
with `SOFTMAX_USE_VF=1` and calls the authoritative installed
`FaVectorApi::SoftmaxDnVF<float>` implementation from CANN.  Both paths keep
two FP32 accumulators for even and odd M rows and merge them only after the
loop.  The exp result overwrites the input UB tile, followed by the required
vector-store-to-vector-load barrier and the normalization pass.

For a frozen source comparison, set `VF_SOFTMAX_INCLUDE` to a directory that
contains the exact `vf_softmax.h` under test. The experiment manifest retains
that header and its SHA-256 instead of silently substituting the CANN default.

The PTO graph uses the same phase order and observable UB lifetime:

```text
tload(x)
tcolmax(x -> max)
tcolexpandexpdif(x, max -> exp_aliasing_x)
tcolsum(exp -> sum)
tcolexpanddiv(exp, sum -> y)
tstore(y)
```

The canonical PTO source is Base64.  The A5 harness generator renders Base32,
Base64, and Base128 with independent experiment-scoped symbols.

## Correctness datasets

`exact-onehot` assigns one winning M row per `(B,N)` lane and `-Inf` to the
others.  Therefore exp is exactly zero or one, the denominator is exactly one,
and the expected FP32 output is byte-exact.  It detects cross-tile, row, lane,
chunk, and mask errors without depending on a host approximation of A5 exp.

`finite-sensitive` uses bounded finite values with a unique pattern per tile,
row, and column.  The independent golden performs explicit FP32 rounding and
the same even/odd accumulation association. Device comparison uses `atol=1e-6`
and `rtol=1e-6` because host `expf` is not an implementation of the A5
`vexpdif` approximation; every exact-onehot result remains byte-exact.

Generated input binaries, lowering dumps, objects, assembly, and profiles are
private experiment evidence and must not be committed.

`a5_harness/generate_pto_variants.py` creates experiment-scoped A/B/C/D/DL PTO
sources. `D` selects generic VMI state promotion and `DL` selects its frozen
legacy baseline. `a5_harness/run_a5_matrix.sh` performs byte-exact and tolerant
correctness checks before ten interleaved profiles of A/B/C/D/DL/AC-U/AC-F.
It evaluates both the generic-D/legacy-D no-regression gate and the paired
generic-D/manual-AC-F comparison. `a5_harness/extract_profile.py` and
`a5_harness/summarize_profiles.py` convert serial msprof output into the
four-path comparison table.  The validated 2026-08-24 results and the exact
correctness/lowering gates are recorded in `A5_ACCEPTANCE.md`.

If a width fails the task bootstrap bound at ten samples,
`a5_harness/extend_paired_profiles.sh` appends only D/DL samples 11--20 for the
affected widths without rebuilding or overwriting the initial profiles.
