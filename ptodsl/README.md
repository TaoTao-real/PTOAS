# ptodsl — PTO Python IR Builders

A lightweight, pip-installable DSL package for building PTO MLIR IR modules
in Python. PTODSL kernels are ordinary Python functions decorated with
`@pto.jit`. Type annotations carry PTO
types as lazy descriptors, and control-flow maps 1-to-1 to MLIR operations.

---

## Directory layout

```
ptodsl/
├── ptodsl/              # pip-installable package
│   ├── __init__.py      # exports: pto, scalar
│   ├── pto.py           # main PTO DSL namespace
│   ├── scalar.py        # top-level scalar.* helper namespace
│   ├── _bootstrap.py    # MLIR path setup + context factory
│   ├── _types.py        # lazy dtype descriptors and type constructors
│   ├── _ops.py          # PTO operation wrappers
│   ├── _control_flow.py # for_, if_, yield_ context managers
│   ├── _jit.py          # @pto.jit decorator
│   ├── _tracing/        # shared tracing runtime building blocks
│   └── _tile_template_tracing.py # internal tile-template tracing implementation
├── examples/
│   ├── tadd_lowlevel.py    # TADD – raw MLIR Python binding calls
│   ├── tadd_dsl.py         # TADD – @pto.jit DSL style
│   ├── softmax_lowlevel.py # Softmax – raw MLIR Python binding calls
│   └── softmax_dsl.py      # Softmax – @pto.jit DSL style
├── pyproject.toml       # pip install -e .
└── README.md
```

---

## Prerequisites

```bash
# Install ptoas (first time only)
cd $PTOAS_REPO_ROOT          # e.g. export PTOAS_REPO_ROOT=/workdir/ptoas_a5
bash quick_install.sh

# Set up environment in every new shell
source scripts/ptoas_env.sh
```

---

## Install the package

```bash
cd $PTOAS_REPO_ROOT/ptodsl
pip install -e .
```

---

## JIT examples

`ptodsl/examples/` contains self-contained `@pto.jit` examples that cover
both compile-only and end-to-end launch flows.

### Prerequisites for launch examples

- `ptoas` + `ptodsl` installed as above
- CANN 9.0+ with `ASCEND_HOME_PATH` set
- For end-to-end launch: `torch`, `torch_npu`, `numpy`
- `bisheng` on `PATH`

Set up the environment in each new shell:

```bash
cd $PTOAS_REPO_ROOT
source scripts/ptoas_env.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
```

For CPU simulation with `msprof`, the wrapper script below will set the
simulator library path and `ulimit` for you. The normal PTOAS + CANN shell
setup above is still required.

### `tadd_launch.py`

Single script: kernel definition, compile, launch, and accuracy check.
Equivalent IR to the TileLang ST `tadd.pto` testcase.

Compile-only:

```bash
python3 ptodsl/examples/tadd_launch.py --emit-mlir
```

Expected: MLIR containing `@TADD_f32_16x64` and `@TADD_f32_32x32`.

Optional PTOAS frontend smoke:

```bash
python3 ptodsl/examples/tadd_launch.py --emit-mlir > /tmp/tadd_dsl.mlir
ptoas --emit-pto-ir /tmp/tadd_dsl.mlir -o - | head
```

End-to-end under the `msprof` CPU simulator:

```bash
scripts/sim_dsl.sh ptodsl/examples/tadd_launch.py
```

Expected output:

```text
PASS f32_16x64  compile=0.024s launch=35.193s
PASS f32_32x32  compile=0.022s launch=35.926s
All cases passed.
```

Direct run on a real NPU:

```bash
python3 ptodsl/examples/tadd_launch.py
```

### `flash_attention_softmax_launch.py`

Launchable row-wise softmax demo. The kernel surface is the ordinary
`scores -> out` contract, while the implementation preloads the score matrix to
UB and then uses a packed online-softmax recurrence so one NPU can stream
64-row packs sequentially from UB.

Compile-only:

```bash
python3 ptodsl/examples/flash_attention_softmax_launch.py --emit-mlir
```

End-to-end under the `msprof` CPU simulator:

```bash
scripts/sim_dsl.sh ptodsl/examples/flash_attention_softmax_launch.py
```

Expected output:

```text
PASS rows64_seq128
PASS rows81_seq96
All cases passed.
```

Direct run on a real NPU:

```bash
python3 ptodsl/examples/flash_attention_softmax_launch.py
```

### Launch artifacts

- `~/.cache/ptodsl/` — JIT-compiled kernel `.so` cache
- `build/msprof_res/` — `msprof` simulator trace output

---

## Running regression checks

```bash
cd $PTOAS_REPO_ROOT
python3 ptodsl/tests/test_jit_compile.py
python3 ptodsl/tests/test_jit_diagnostics.py
python3 ptodsl/tests/test_subkernel_diagnostics.py
python3 ptodsl/tests/test_flash_attention_demo_compile.py
python3 ptodsl/tests/test_ptoas_frontend_verify.py
python3 ptodsl/tests/test_docs_as_test.py
```

Expected output:

```
ptodsl_jit_compile: PASS
ptodsl_jit_diagnostics: PASS
ptodsl_subkernel_diagnostics: PASS
ptodsl_flash_attention_demo_compile: PASS
ptodsl_ptoas_frontend_verify: PASS
ptodsl_docs_as_test: PASS
```

`ptodsl/tests/` is the canonical home for PTODSL-specific regression scripts.
The launchable sources under `ptodsl/examples/` remain examples; the
regressions that protect them live here alongside compile-only and docs checks.

`test_docs_as_test.py` is the docs-as-test regression for the PTODSL user
guide under `ptodsl/docs/user_guide/`. It scans every Python fenced code block
and requires each one to be explicitly classified with either
`ptodsl-doc-test` or `ptodsl-doc-pending` metadata.

- `mode="compile"` blocks are executed as-authored and must pass the PTODSL
  compile-only path, MLIR verify, and shared PTOAS frontend validation.
- `mode="compile_fragment"` blocks are embedded into explicit test fixtures so
  representative partial snippets can be compiled under a declared outer
  kernel context instead of relying on hidden heuristic context synthesis.
- `ptodsl-doc-pending` marks snippets the manual intends to treat as contract
  later, but which are still blocked on missing implementation or missing test
  harness support.

Run it directly while editing the manual:

```bash
cd $PTOAS_REPO_ROOT
python3 ptodsl/tests/test_docs_as_test.py
```

When it fails, the diagnostic includes the Markdown path, starting line number,
and target symbol so the drift can be fixed in the manual instead of searching
through generated IR logs.

These PTODSL regressions are intentionally complementary:

- `test_jit_compile.py` protects canonical authored compile probes and
  lowering contracts for the public PTODSL surface.
- `test_flash_attention_demo_compile.py` protects the bundled
  `ptodsl/examplesflash_attention_sketch.py` authored demo as a stable end-to-end
  contract.
- `test_ptoas_frontend_verify.py` protects the handoff from PTODSL-emitted
  MLIR into standalone `ptoas` frontend verification.
- `test_docs_as_test.py` protects the user manual itself: documented
  self-contained examples must still compile, fixture-backed partial fragments
  must still compile inside their declared context, and explicitly marked
  pending snippets remain visible as docs/test debt.

`test_docs_as_test.py` is not a replacement for the authored compile/demo
regressions above. It reuses the same compile-only and frontend-validation
boundaries, but its job is to keep `ptodsl/docs/user_guide/` honest rather than
to redefine the canonical demo contracts.

The legacy `ptodsl/check_ir.py` script has been retired. PTODSL validation now
lives under `ptodsl/tests/` so every regression shares the same bootstrap,
public surface, and canonical authored targets as the tracing/JIT
implementation.

---

## DSL-style API quick reference

```python
from ptodsl import pto, scalar
s = scalar       # arith shorthand alias
```

`pto` is the main DSL namespace. `scalar` is a separate top-level helper
namespace for runtime scalar load/store, arithmetic helpers, and scalar math;
it is intentionally not exported as `pto.scalar`.

### Kernel decorator

```python
@pto.jit(name="MyKernel", kernel_kind="vector", target="a5")
def MyKernel():
    ...

@pto.jit(name="Softmax", kernel_kind="vector", target="a5")
def Softmax(
    X_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    x_view = pto.make_tensor_view(X_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    ...

print(MyKernel)               # prints MLIR text
mod = MyKernel.mlir_module()  # returns mlir.ir.Module
```

`@pto.jit` now emits a flat aicore launch-entry module by default. The traced
entry function carries the `pto.aicore` attribute and lives directly under the
top-level module, which matches the runtime-launch path and merged-MLIR example
flow.

PTODSL v1 keeps the public `@pto.jit` entry ABI intentionally narrow:

- positional device buffers are explicit GM pointers declared with
  `pto.ptr(..., "gm")`
- shape, stride, and other launch-varying metadata travel as positional runtime
  scalars such as `pto.i32`, `pto.f32`, and `pto.i1`
- kernel bodies reconstruct tensor descriptors explicitly with
  `pto.make_tensor_view(ptr, shape=..., strides=...)`
- positional runtime scalars use PTO scalar annotations such as `pto.i32`,
  `pto.f32`, and `pto.i1`, while launch-time values remain ordinary Python
  scalars
- keyword-only parameters annotated with `pto.const_expr` are compile-time
  specialization knobs

The host wrapper is responsible for extracting or deriving whatever runtime
metadata the kernel needs and passing it explicitly at launch time. PTODSL no
longer uses `pto.tensor_spec(...)` as the public `@pto.jit` entry contract.

Additional layered kernel entry modes and shared compute decorators are also
exported on the public surface: `@pto.jit(mode="auto")`,
`@pto.jit(mode="explicit")`, `@pto.cube`, `@pto.simd`, and `@pto.simt`.

### Type descriptors (lazy – safe to use in annotations)

| Expression | MLIR type |
|---|---|
| `pto.float32` | `f32` |
| `pto.int32` | `i32` |
| `pto.int64` | `i64` |
| `pto.index` | `index` |
| `pto.ptr(pto.float32, "gm")` | `!pto.ptr<f32, gm>` |
| `pto.ptr(pto.float32, "ub")` | `!pto.ptr<f32, ub>` |

### Type constructors (eager – require active context)

```python
vf32     = pto.vreg_type(64, pto.float32)                       # !pto.vreg<64xf32>
tile_col = pto.alloc_tile(shape=[8, 1], dtype=pto.float32, blayout="ColMajor")
tile_w   = pto.alloc_tile(shape=[8, 128], dtype=pto.float32)
```

### Constants

```python
c0     = pto.const(0)               # index
c1_i32 = pto.const(1, dtype=pto.int32)
c64_i64= pto.const(64, dtype=pto.int64)
```

### Control flow

```python
with pto.simd():                    # pto.simd { … }
    ...

with pto.for_(c0, c16, step=c1) as i:     # simple scf.for
    ...                                    # scf.yield inserted automatically

loop = pto.for_(c0, c128, step=c64).carry(lhs=a, rhs=b)
with loop:
    x = loop.lhs
    y = loop.rhs
    ...
    loop.update(lhs=nx, rhs=ny)
fx = loop.final("lhs")
fy = loop.final("rhs")

with pto.if_(has_rows) as br:      # simple scf.if
    with br.then_:
        ...

with pto.if_(has_chunk) as br:
    with br.then_:
        br.assign(x=merged_max, y=merged_sum)
    with br.else_:
        br.assign(x=running_max, y=running_sum)
x = br.x
y = br.y
```

### Scalar arithmetic (`s = scalar`)

```python
s.muli(a, b)                 # arith.muli
s.addi(a, b)                 # arith.addi
s.subi(a, b)                 # arith.subi
s.index_cast(val)            # arith.index_cast → index
s.index_cast(pto.int32, val) # arith.index_cast → i32
(a > b)                      # scalar compare → pto.i1
(a <= b)                     # scalar compare → pto.i1
s.select(cond, t, f)         # arith.select
```

### PTO operations

```python
pto.castptr(addr, ptr_type)              # pto.castptr
pto.addptr(ptr, offset)                  # pto.addptr
pto.vlds(ptr, offset)                    # pto.vlds, result vreg inferred from ptr element type
pto.vbr(scalar)                          # pto.vbr, scalar broadcast -> vreg
pto.vsts(v, ptr, offset, mask)           # pto.vsts
pto.plt_b32(scalar)                      # → (mask, scalar_out)
pto.pset_b32("PAT_ALL")                  # pto.pset_b32 → mask
pto.vbitcast(v, dtype)                   # pto.vbitcast
pto.pbitcast(mask, mask_type)            # pto.pbitcast
pto.vadd(a, b, mask)   # infers result type from a.type
pto.vmul / vmax / vdiv / vcmax / vcadd / vdup / vexpdif  # similarly
pto.make_tensor_view(ptr, shape=…, strides=…)    # type inferred
pto.partition_view(tv, offsets=…, sizes=…)        # type inferred
pto.alloc_tile(shape=…, dtype=…, memory_space=…, valid_shape=…, addr=…)  # authored surface
pto.tile.load(part, tile)
pto.tile.store(tile, part)
pto.tile.load(tv, tile)                           # partition + tile.load, zero offsets, sizes inferred from tile
pto.tile.store(tile, tv)                          # partition + tile.store, zero offsets, sizes inferred from tile
tile.as_ptr() / view.as_ptr()
pto.get_block_idx()           # → i64
pto.set_flag("MTE2", "V", event_id=0)
pto.wait_flag("MTE2", "V", event_id=0)
pto.pipe_barrier(pto.Pipe.ALL)
```

## How the IR check works

```
generated IR  ──┐
                ├── Module.parse() → canonical string ──── == ──── PASS/FAIL
reference .pto ──┘  (strips comments, normalises SSA names and attr order)
```

Constant declaration order is preserved after the round-trip; builders must
emit constants in the same order as the reference.  The diff output makes any
mismatch immediately visible.
