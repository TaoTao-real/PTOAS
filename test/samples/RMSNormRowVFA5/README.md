# RMSNorm multi-row one-VL A5 acceptance fixture

This fixture isolates the canonical row-VF case discussed in ADR-0003.
`Nx256B` means `N` independent rows of one FP32 vector length:

```text
x/y:       BF16 [N, 64]
gamma:     BF16 [1, 64]
compute:   FP32 [N, 64]
row state: FP32 [N, 1]
```

The primary compiler fixture fixes `N=64`; the independent golden also covers
`N=1/8/32/64`.  Each row performs exactly one `trowsum`:

```text
tcvt(x) -> x*x -> trowsum -> * (1/64) -> + epsilon -> sqrt
        -> x / rms -> * tcvt(gamma) -> tcvt RINT(output)
```

`rmsnorm_row_vf_ascendc.cpp` is one source for both AscendC controls.  Compile
without `RMSNORM_USE_VF` for AC-U and with `-DRMSNORM_USE_VF=1` for AC-F.  AC-F
contains one manual `__simd_vf__` scope whose outer loop iterates over rows;
the converted gamma vector is loop invariant.  The PTO source expresses the
same graph with whole `64x64` TileOps and no source-level row loop.

## Equivalence matrix

| Stage | AC-U | AC-F | PTO |
| --- | --- | --- | --- |
| GM input/output | BF16 `[N,64]` | BF16 `[N,64]` | BF16 `[N,64]` |
| gamma | BF16 `[1,64]`, converted once | loaded/converted once before row loop | `tcvt` preheader |
| x conversion | once per row | once per row | one `tcvt [N,64]` candidate |
| square | FP32 lane-wise | FP32 lane-wise | `tmul [N,64]` |
| reduction | one 64-lane reduce per row | one `Reg::Reduce` per row | one `trowsum [N,64] -> [N,1]` |
| scalar chain | FP32, VL1 | FP32, VL1 | `tmuls -> tadds -> tsqrt` |
| broadcast | denominator scalar per row | denominator scalar per row | `trowexpanddiv` |
| gamma multiply | FP32 | FP32 | `tcolexpandmul` |
| narrowing | BF16 RINT | BF16 RINT | BF16 RINT |
| launch | one vector block | one vector block | one vector block |

The golden never calls a compiled kernel.  Both fixed datasets use BF16-exact
integer or power-of-two values, so every row's square sum is exactly
representable in FP32 and independent of an undocumented reduction-tree
permutation.  `layout-sensitive` also gives every row a distinct pattern to
detect row mixing, wrong compact-scalar indexing, mask errors, and incorrect
gamma broadcast.

Generated binaries, pass dumps, assembly, input `.bin` files, and profiles are
experiment evidence and must stay outside the source tree.

The compact row state intentionally uses a padded physical representation:
col-major `512x1 valid 64x1`, reinterpreted as row-major
`64x8 valid 64x1` for scalar elementwise ops.  This is the minimum A5-legal
row-major storage (32 bytes per physical row) and preserves one VL1 iteration
per logical row.  At the fixture baseline, `trowsum` and `trowexpanddiv` are the
two expected PTODSL fallbacks; subsequent feature commits add the exact padded
forms without weakening dynamic/tail legality.
