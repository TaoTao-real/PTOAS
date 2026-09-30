# 5. Reduce

> **Category:** B (VLane-aligned), C (unaligned sub-VLane).
> **Mask:** `Pg req` (governing mask is a required operand).
>
> Reduction ops produce one logical scalar per group, governed by a mask.
> The assigned layout determines each scalar's physical slot position.
> `{group=C}` controls the number of sub-groups. Inactive lane behavior:
> `vcadd` treats inactive as 0; `vcmax`/`vcmin` treat inactive as `-∞`/`+∞`
> (fp) or type min/max (int).


All three unified reductions accept only 16/32-bit integers (signless,
signed, or unsigned), `f16`, and `f32`. BF16, all FP8 variants (including
E4M3FN, E5M2, and E8M0), and HiF8 are rejected by the operation verifier.
Convert such inputs explicitly to a supported type before reducing;
no implicit `extf -> f32 reduction -> truncf`, rounding mode, or saturation
policy is supplied by these operations.

Logical `pto.vmi.vcadd/vcmax/vcmin` and physical `pto.vcadd/vcmax/vcmin`
share this source-type range. Eight-bit integers are also rejected: explicitly
sign- or zero-extend them to 16/32 bits before reducing. Internal extension
helpers do not imply support for a direct 8-bit reduction. This applies to
full, grouped, and singleton reductions.

The A5 VPTO backend supports `group = 1, 2, 4, 8` when the group count
divides `L`, for the following one-carrier shapes:

| Element type | Supported logical `L` |
|---|---|
| 16-bit integers and `f16` | `1, 2, 4, 8, 64, 128` |
| 32-bit integers and `f32` | `1, 2, 4, 8, 64` |

The compiler prefers executable native layouts, retaining the established
broadcast and memory paths. Where these are unavailable, a dense fallback
intersects each group's logical lane range with the governing mask and packs
the scalar results for grouped stores or broadcasts. Dynamic masks may contain
holes, empty groups, or partial final groups; inactive lanes retain the
identities described above.

Eight-bit integer sources (`i8`, `si8`, `ui8`) are unsupported for
`vcadd`, `vcmax` and `vcmin`, including full reductions and singleton
groups (`group = L`, including `L = 1`). IR verification rejects them with
`VMI-UNSUPPORTED: 8-bit integer reductions are not supported`. The compiler
does not insert widening or provide an instruction-local fallback.
Direct micro `pto.vcadd` also rejects eight-bit integer inputs in its verifier;
bypassing VMI does not enable an eight-bit sum reduction.

Callers can explicitly convert an eight-bit vector to a supported 16-bit or
32-bit integer type before reducing it. The reduction then follows the wide
type's identities and result-width semantics. Truncating a wide empty min/max
result does not automatically recover the original 8-bit identity; callers
needing that behavior must select the intended identity explicitly. Explicit
casts, eight-bit loads/stores and broadcasts retain their existing shape limits.

Supported integer singleton groups retain selection between the input and
its identity. Floating singleton groups retain reduction semantics for NaNs
and signed zero. Integer add results are narrowed modulo the result element
width; native 16-bit VCG sums expose the low halfwords of 32-bit results as
`gs(8, 2)`. An explicit 16-to-8 `NOSAT` truncation retains the low bytes at
twice the source lane stride, consumed by grouped stores or compact broadcasts.
Floating-point addition still requires `reassoc`.

The dense fallback supports eligible 16/32-bit shapes with at most eight
groups, including groups spanning multiple physical input carriers. It returns
one physical result per group (`slots=1`); singleton groups retain the packet
form. Native layouts remain preferred when executable. Unsupported shapes are
rejected with `VMI-UNSUPPORTED` before layout assignment.

### Regression coverage

`vmi_integer_reductions_i8_invalid.pto` checks all three integer signedness
variants and add/max/min through public CLI, public/legacy IR and Python
constructors, including singleton groups. These rejection tests run on CPU.
`vmi_explicit_integer_cast_reduction_paths.pto` protects explicit 8-to-16
conversion, group-slot truncation, predicate views, compact broadcast and
VL256 two-carrier reduction/combine paths. The 8-to-32 cast/reduce path is
covered by `vmi_to_vpto_integer_cast_reduce.pto`.

`test/vpto/cases/vmi_new/group-reduce-boundaries.py` exposes the existing
`vmi_group_reduce_boundaries.py` and `vmi_compact_group_reduce.py` goldens
through the standard VPTO runner. They check supported 16/32-bit integer and
f16/f32 inputs, holes, empty groups, partial masks, singleton groups, modulo
sums, grouped stores, broadcasts and untouched output canaries.

From the repository root with a CMake build in `build`, a matching Python
environment and CANN configured:

```bash
export PTOAS_BIN="$PWD/build/tools/ptoas/ptoas"
export PYTHONPATH="$PWD/build/python:$PWD/ptodsl${PYTHONPATH:+:$PYTHONPATH}"
python3 test/vpto/cases/vmi_new/group-reduce-boundaries.py --list

task-submit --device auto --max-time 3600 --timeout 3600 \
  --env PTOAS_BIN --env PYTHONPATH --env ASCEND_HOME_PATH --env PATH \
  --run 'DEVICE=NPU WORK_SPACE="$PWD/build/group-reduce-runtime" \
    CASE_NAME=vmi_new/group-reduce-boundaries.py \
    bash test/vpto/scripts/run_host_vpto_validation.sh'
```

### A5 integer result widths and group slots

The logical result keeps the input element width, but the hardware's sum may
be wider. In particular, **both** 16-bit integer `vcgadd` and `vcadd` produce
32-bit sums. Their layouts reflect how many independent results each instruction produces:

| Physical path | Hardware result for 16-bit integer input | VMI result handling |
|---|---|---|
| Native `vcgadd` | Eight 32-bit sums in one register | Keep the low 16 bits at halfword lanes `0, 2, ..., 14`, yielding `gs(8, 2)` without a producer-side pack |
| Native `vcgmax` / `vcgmin` | Eight consecutive 16-bit extrema | Keep the values at their original width; no sum-narrowing pack |
| Compact `vcadd` | One 32-bit sum in lane zero per invocation | Use the widened result type, combine partial sums if needed, then expose each group's low bits in its `slots=1` result |

For example, a 32-bit sum of `131091` has low/high 16-bit halves `19` and `2`.
Two such VCG sums appear as `[19, 2, 19, 2]` in a 16-bit view. The logical
16-bit results must be `[19, 19]`, not `[19, 2]`. The high halves are part of
the hardware sums; they are not additional groups or necessarily zero.

In the compact path, `getRowResultType()` selects the widened integer sum
type. After any partial sums are combined, a register bitcast exposes the
original-width low lane. The dense `slots=1` path returns each group in its own
physical part; the packet path uses `buildCompactPacket()` to assemble slots.
The bitcast itself does not pack or clear the other physical lanes. Compact
max/min likewise consume only the
extremum value, not the index returned by the row max/min instruction.
Compact instruction lowering does not widen inputs. Integer sum narrowing
follows modulo arithmetic at the logical result width; it is not saturation.
Floating-point reductions keep
their existing result types and semantics. See the physical contracts in
[Reduction Ops](../micro-isa/10-reduction-ops.md).

### Native integer sum layout

The layout capability table selects `gs(8, 2)` for native 16-bit integer
addition, independently of consumers. Operation kind and element width are
part of the query: native floating-point addition and integer max/min retain
`gs(8)`. Scalar `vcadd` paths retain their existing row-local layout.

Two- and four-block native reductions combine their 32-bit partial sums with
`b32` predicates, then expose the low halfwords as `gs(8, 2)`. The arithmetic
still returns a logical 16-bit result modulo 2^16. The unused high halfwords
are unspecified padding; consumers must not read them as logical values or
assume they are zero.

Consumers use the regular layout support and `ensure_layout` machinery.
A 16-to-32 integer extension reads even halfword lanes with `vcvt EVEN`:
`ui16(65535 + 1)` extends to `0`, and `si16(32767 + 1)` extends to `-32768`.
It must not return the full hardware sum. A store or broadcast can consume
the strided slots directly when its layout supports them, or request a
conversion to consecutive `gs(8)` slots using `vpack LOWER`. Multiple
consumers share the same producer layout and request their own necessary
conversions.

This removes the former single-use cast-to-store peephole from
`vpto-optimize-vcvt`. It does not promise that every complete consumer chain
contains fewer instructions or runs faster. See the
[native sum layout design and validation](../../designs/a5-vcg-integer-native-layout.md).


---

## `pto.vmi.vcadd`

- **semantics:** Masked add-reduction. When `{group=C}` is absent, reduces all
  `L` active lanes to a single scalar (`V<1×T>`).

  ```c
  // Without group: full reduction to scalar
  T sum = 0;
  for (int i = 0; i < L; i++)
      if (mask[i]) sum += src[i];
  dst[0] = sum;

  // With {group=C}: per-group reduction
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++) {
      T sum = 0;
      for (int i = 0; i < gs; i++)
          if (mask[g*gs + i]) sum += src[g*gs + i];
      dst[g] = sum;
  }
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcadd %src, %mask {group = C, reassoc} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Source vector |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<C×T>` | One logical scalar per group in the assigned layout (`C = 1` if no group) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | Positive `C` dividing `L`, subject to the shape support above | `1` (full reduce) | Number of sub-groups |
  | `reassoc` | *(unit attr)* | *(absent)* | Permit reassociation (**required** for fp sources) |
  | `pmode` | `"zero"` | `"zero"` | Inactive-result behavior |

- **datatypes:** `i16`/`i32` (signless, signed, unsigned), `f16`/`f32`.
  The operation verifier rejects other floating types,
  including `bf16`, before checking `reassoc` or lowering to legacy operations.
  A public full reduction is one logical group. The layout-assigned legacy
  `reduce_addi` form retains its 32-bit integer input restriction.
- **lowering to `pto.mi`:**

  | Group / W | Category | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|---|
  | No group (`C=1`), `K=1` | B | `1 × pto.vcadd` | `1` | `1` |
  | No group, `K>1` (fold) | B | `(K-1) × vadd` + `1 × vcadd` | `K` | `K` |
  | No group, `K>1` (partial) | B | `K × vcadd` + `(K-1) × vadd` | `2K-1` | `1+⌈log₂K⌉` |
  | W=32B, VLane-aligned | B | `K × pto.vcgadd` (eight groups per full packet) | `K` | `1` |
  | W=64B/128B, `k=2/4` fragments per group | B | Per result packet: `k × vcgadd` + `(k-1) × vadd` | `2k-1` | `1+⌈log₂k⌉` |

  Layout-conversion and consumer instructions are not included in these
  native-path counts. Equivalent-mask source folding is optional and applies
  only when legal. Native 16-bit integer `vcgadd` retains the partial-result
  form with 32-bit adds and `b32` masks; it is excluded from source folding.

- **example:**
  ```mlir
  // Full sum reduction (to scalar)
  %sum = pto.vmi.vcadd %x, %mask {reassoc}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 256 f16 lanes → 8 groups of 32, two VLanes per group (W=64B)
  %sums = pto.vmi.vcadd %x, %mask {group = 8, reassoc}
      : !pto.vmi.vreg<256×f16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×f16>
  ```


---

## `pto.vmi.vcmax` / `pto.vmi.vcmin`

- **semantics:** Masked max/min reduction.

  ```c
  // vcmax: inactive lanes treated as -∞
  T best = -INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = max(best, src[i]);
  dst[0] = best;

  // vcmin: inactive lanes treated as +∞
  T best = +INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = min(best, src[i]);
  dst[0] = best;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmax %src, %mask {group = C} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:** Same as `vcadd` (without `reassoc`).
- **results:** Same as `vcadd`.
- **attributes:** `group`, `pmode` (same as `vcadd`, no `reassoc`).
- **datatypes:** same source types and shape limits as `vcadd`; BF16/FP8/HiF8
  are not supported.
- **lowering to `pto.mi`:**

  | Group / W | Physical lowering |
  |---|---|
  | No group, fold | `(K-1) × vmax` + `1 × vcmax` |
  | VLane-aligned | `K × pto.vcgmax` / `K × pto.vcgmin` |

- **example:**
  ```mlir
  // Full max reduction
  %mx = pto.vmi.vcmax %x, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 8-sub-group max (MX block-scale exponent pattern)
  %maxe = pto.vmi.vcmax %exp, %mask {group = 8}
      : !pto.vmi.vreg<256×ui16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×ui16>
  ```
