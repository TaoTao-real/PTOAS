# Partial RoPE A5 equivalence and VMI capability fixture

This directory reconstructs the attached Compressor manual-VF RoPE as a
standalone, reviewable contract.  The representative shape is fixed to:

- FP32 input `[8,512]`
- FP32 sin/cos `[8,64]`
- BF16 output `[8,512]`
- `baseAddr=448`, `col=64`, `actualCol=512`

The equality `actualCol == baseAddr + col` is deliberate.  The source manual
VF writes the prefix and rotary segment but does not write a suffix.  A caller
with `actualCol > baseAddr + col` must separately define and implement suffix
semantics before it can claim full-row output equivalence.

## Algorithm and data-flow contract

For every row, `[0,448)` is converted FP32 to BF16 with RINT in seven complete
64-lane chunks.  `[448,512)` is the rotary segment.

HALF mode splits the rotary segment into `x0=x[0:32]` and `x1=x[32:64]`:

```text
y0 = cos[0:32]  * x0 - sin[0:32]  * x1
y1 = sin[32:64] * x0 + cos[32:64] * x1
```

INTERLEAVE mode rotates each adjacent pair:

```text
rot([-]) = [-x1, x0, -x3, x2, ...]
y = cos * x + sin * rot(x)
```

Every multiply and add/subtract is FP32.  Only the final output conversion is
BF16 RINT.  No reduction or cross-row operation exists.

The generic source also implies these legality constraints: `col` is even,
HALF requires `col/2 <= 64` FP32 lanes, and INTERLEAVE requires `col <= 64`.
The BF16-to-FP32 cast trait declared in the source is unused because `inUb`,
`sinUb`, and `cosUb` already have type `T` (FP32 in this workload).  HALF mode
issues full-register `DataCopy` operations and masks computation to 32 lanes;
the surrounding UB allocation must therefore make those physical read windows
safe even though only the masked lanes affect output.

| Stage | Manual VF | Ordinary AscendC | PTO reference |
|---|---|---|---|
| row mapping | `rIdx * actualCol` | identical | identical fixed row loop |
| prefix | seven 64-lane load/cvt/store steps | seven 64-element `Cast` calls | seven 64-lane `tcvt` or VMI steps |
| HALF pairing | first half with second half | identical indexes | identical 32-lane TileOps |
| INTERLEAVE pairing | `DeInterleave`, negate odd, `Interleave` | same three operations | `vdintlv(x,x)`, negate odd, `vintlv` |
| arithmetic | FP32 mul + add/sub | FP32 mul + add/sub | FP32 `vmul` + `vadd/vsub` |
| output | BF16 RINT, no saturation | BF16 RINT | direct VMI uses R/NOSAT |

`rope_partial_ordinary_ascendc.h` is the non-VF AscendC implementation.  Its
scratch tensors are caller-owned to make UB lifetime and aliasing explicit.
The one-input `DeInterleave` and two-output `Interleave` are A5-supported
LocalTensor APIs; `halfCol*sizeof(float)` must preserve the API's 32-byte
alignment requirement (true for the fixed shape).

`rope_partial_half_tile.pto` is an exact TileOp HALF fixture.  It intentionally
uses 32-lane halves rather than padding or changing the algorithm.

`rope_partial_interleave_vmi.pto` is a direct VMI-control implementation of
the UB compute stage.  It is a lowering/ISA reference, not evidence that the
TileOp candidate and FusionRegion planner can discover the same graph.

`rope_partial_golden.py` contains two independent CPU formulations: a
vector-shaped implementation and an index-by-index scalar oracle.  Each case
asserts byte-exact BF16 equality between them before emitting hashes.

## Current automatic VMI fusion result

Generic VMI Vector State Promotion now covers both TileOp fixtures.

- The PTO fixtures now use the same GM/UB boundary as the attached manual VF
  wrapper: one full input load, one full sin load, one full cos load, row-local
  vector work entirely on UB subviews, and one full output store.  Final VPTO
  therefore has exactly three `copy_gm_to_ubuf` operations before the row loop
  and one `copy_ubuf_to_gm` after all vector writes; no GM DMA is nested in the
  row or prefix loops.  As in the manual wrapper, one `MTE2->V` event closes
  input staging and one `V->MTE3` event closes vector production before the
  final write-back.

- HALF selects the static 32-lane `vmi_tcvt` path.  FP32-to-BF16 narrowing is
  RINT/NOSAT and state promotion forwards all multiply/add/subtract and convert
  intermediates as SSA/vregs.  Final VPTO contains nine contractual VLDs
  (eight rotary inputs plus the prefix-loop input), three output VSTs, and no
  arithmetic temporary UB traffic or internal membar.
- INTERLEAVE is represented by reusable `pto.tchannel_split` and
  `pto.tchannel_merge` TileOps (K=2/4).  Each has ordinary PTODSL and selected
  VMI candidates, and belongs to `FusionComputeFamily::Rearrange`.  Layout
  assignment carries deinterleaved state across FusionRegion results and
  materializes a layout conversion only for conflicting SSA consumers.
- `rope_partial_interleave_tile.pto` automatically selects the rearrange,
  elementwise and convert VMI candidates.  Final VPTO contains exactly one
  `vdintlv`, one `vintlv`, four contractual VLDs (x/sin/cos plus prefix), two
  output VSTs, RINT/NOSAT narrowing, no temporary UB traffic, no residual VMI
  op, and no internal membar.
- The shared `x` producer is forwarded to both channel split and `cos*x`.
  FusionRegion result/yield layout equality is part of VMI layout assignment,
  so a deinterleaved primary layout cannot leave a stale region result type.

The independent vector-shaped and scalar element-by-element CPU formulations
remain byte-exact for both exact and layout-sensitive datasets.  Hardware
AC-U/AC-F and paired A5 performance measurements remain separate acceptance
gates; local compiler success is not reported as hardware performance data.

## Local checks

```sh
/usr/bin/python3 test/samples/RopePartialVFA5/rope_partial_golden.py

build-llvm21/tools/ptoas/ptoas \
  --pto-level=level3 --pto-arch=a5 --pto-backend=vpto \
  --tilelib-candidate-policy=prefer-vmi --vmi-fusion-mode=full \
  --vmi-state-promotion-mode=generic --emit-vpto \
  test/samples/RopePartialVFA5/rope_partial_interleave_tile.pto

/usr/bin/python3 test/samples/RopePartialVFA5/check_staged_vpto.py \
  --mode interleave /path/to/rope-interleave.vpto

PYTHONPATH=/path/to/llvm21/mlir_core:$PWD/build-llvm21/python \
build-llvm21/tools/ptoas/ptoas \
  --pto-level=level3 --pto-arch=a5 --pto-backend=vpto \
  --vmi-fusion-mode=off --emit-vpto \
  test/samples/RopePartialVFA5/rope_partial_half_tile.pto

/usr/bin/python3 test/samples/RopePartialVFA5/check_staged_vpto.py \
  --mode half /path/to/rope-half.vpto
```

Full dumps from this investigation are kept outside the source tree under
`_private_ptoas_lab/experiments/20260821-rope-partial-vmi-capability-r1`.
