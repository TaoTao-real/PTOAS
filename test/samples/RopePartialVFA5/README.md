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

The current compiler handles only a subset of this contract automatically.

For the exact HALF fixture candidate selection produces:

- four selected `vmi_tmul` candidates;
- one selected `vmi_tsub` and one selected `vmi_tadd_block64` candidate;
- two 32-lane final `tcvt` operations that fall back to PTODSL and therefore
  remain outside the selected-VMI-only regions;
- the 64-lane prefix `tcvt` selects `vmi_tcvt`.

The planner correctly forms two three-member selected-VMI-only arithmetic
regions and does not absorb the 32-lane PTODSL conversions.  This preserves the
FusionRegion invariant, but it is not yet an all-register RoPE row pipeline.

INTERLEAVE is more limited.  VMI layout assignment and VMIToVPTO already lower
`vintlv/vdintlv`, and the direct reference lowers without residual VMI or an
internal membar.  However, PTO has no TileOp-level interleave/deinterleave
compute op with a VMI TileLib candidate.  Fusion semantics consequently cannot
plan this rearrangement as part of a selected-VMI region.

Additional blockers relative to the RMSNorm D path are:

1. The 32-lane FP32-to-BF16 narrowing candidate is missing.
2. The ordinary f32-to-bf16 TileLib template currently emits `sat=SAT` even
   when the TileOp contract says saturation OFF; the attached VF explicitly
   uses RINT plus NO_SAT.  Finite acceptance data is unaffected, but Inf/NaN
   boundary equivalence must not be claimed until this is fixed and tested.
3. Principal-row inference is specialized for the RMSNorm chain and does not
   yet prove general `tadd/tsub` row domains or rearrangement domains.
4. Low-level forwarding does not prove multi-result rearrangement, the shared
   `x` use, or prefix/rotary phases as one row pipeline.
5. Region-mode lowering of this fixture currently fails a vecscope escape
   check (`pto.pand` mask used by an external `pto.vsts`) after region flatten.

Therefore the answer is: HALF arithmetic is partially supported; direct
INTERLEAVE VMI is supported; an automatic RMSNorm-quality RoPE D path is not
supported yet.

## Recommended compiler work

1. Freeze AC-U/AC-F/PTO/golden BF16 boundary tests for both modes.
2. Propagate TileOp RINT/NOSAT exactly and add masked 32-lane narrowing.
3. Add reusable TileOp-level interleave/deinterleave semantics and selected VMI
   candidates, rather than a RoPE-specific opaque fusion op.
4. Add a rearrangement compute family and prove the static principal row
   domain through its two results.
5. Extend loop fusion and physical forwarding through the shared `x`,
   deinterleave/interleave results, and prefix chunk loop.
6. Fix the vecscope mask escape and then run AC-U/AC-F/A/B/C/D correctness,
   VPTO/LLVM/assembly inspection, and serial A5 vector-time profiling.

The target D shape should retain only source sin/cos/x UB reads and BF16 output
writes required by the surrounding Compressor contract; all arithmetic and
rearrangement temporaries should stay in vregs, with no internal membar.

## Local checks

```sh
/usr/bin/python3 test/samples/RopePartialVFA5/rope_partial_golden.py

build-llvm21/tools/ptoas/ptoas \
  --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  test/samples/RopePartialVFA5/rope_partial_interleave_vmi.pto

PYTHONPATH=/path/to/llvm21/mlir_core:$PWD/build-llvm21/python \
build-llvm21/tools/ptoas/ptoas \
  --pto-level=level3 --pto-arch=a5 --pto-backend=vpto \
  --vmi-fusion-mode=off --emit-vpto \
  test/samples/RopePartialVFA5/rope_partial_half_tile.pto
```

Full dumps from this investigation are kept outside the source tree under
`_private_ptoas_lab/experiments/20260821-rope-partial-vmi-capability-r1`.
