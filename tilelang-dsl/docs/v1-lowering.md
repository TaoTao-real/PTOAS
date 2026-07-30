# TileLang DSL v1 Authoring Lowering

## Scope

This document records the implemented TileLang DSL v1 lowering contract for
`add-tilelang-dsl-authoring-vpto-lowering`.

It covers:
- the current v1 lowering support matrix
- dynamic-bound and shape-profile behavior
- examples that match the implemented surface
- minimal validation commands, including the repo `ptoas` legality path

It does not define:
- matcher-driven dispatch
- raw pointer authoring surface
- advanced vector-family lowering beyond the fixed v1 matrix

For migration from that original v1 lowering boundary to the current matcher
and advanced-surface implementation, see
`tilelang-dsl/docs/matcher-and-advanced-surface-migration.md`.

## Source Of Truth

The implemented lowering surface lives under:
- `tilelang-dsl/python/tilelang_dsl/`
- `tilelang-dsl/tests/`
- `tilelang-dsl/examples/`
- `tilelang-dsl/docs/`

OpenSpec source of truth for this capability:
- `openspec/changes/add-tilelang-dsl-authoring-vpto-lowering/`

## Implemented v1 Support Matrix

The current v1 lowering contract supports:
- fixed-rank 5D `TensorView` descriptors
- 1D/2D `Tile`
- `dma_load`
- `dma_store`
- `make_mask(dtype, PAT.*)`
- `make_mask(dtype, remaining)`
- `vlds`
- `vsts`
- unary vector family: `vabs`, `vrelu`, `vexp`, `vnot`
- binary vector family: `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`
- vector-scalar family: `vadds`, `vsubs`, `vmuls`, `vdivs`, `vmaxs`, `vmins`
- `for range(lb, ub, step)`
- `if/else`
- `set_flag`, `wait_flag`, `pipe_barrier`

Current lowering shape:
- emits stable `func.func + arith/scf + pto.*` authoring-form VPTO modules
- defaults to memref-first function/tile authoring when the target VPTO family supports memref operands
- keeps `copy_*` family on typed `!pto.ptr`
- infers dedicated `pto.vecscope` for stable vector-active runs
- lowers `pto.strict_vecscope` buffer captures through ptr-form region ABI so the current emission-boundary ptr rewrite stays legal
- only accepts explicit `pto.strict_vecscope` in `advanced=True` kernels
- rejects support-matrix-external surface in the frontend

## Dynamic-Bound Profile

The implemented shape profile is:
- Tile physical shape must stay static
- TensorView parameters stay in authoring IR as `!pto.tensor_view<...>`
- TensorView shape access lowers through `pto.get_tensor_view_dim`
- TensorView stride access lowers through `pto.get_tensor_view_stride`
- TensorView slice bounds may be dynamic
- TensorView slice spelling may omit leading axes; written axes are right-aligned
  onto the trailing physical axes of the 5D descriptor
- loop bounds may be dynamic
- tail `remaining` values may be dynamic

The current DMA lowering still uses the static physical Tile shape when the
TensorView slice extent is dynamic. This keeps v1 inside the current
authoring-form contract without introducing fully dynamic Tile allocation or
tail-DMA semantics.

Although the descriptor rank is 5D, the current DMA-oriented slicing/lowering
path still only supports rank-2 TensorView slices.

## Matcher Metadata For View Operands

When TileLang templates are selected from PTO IR operand metadata, view-like
operands currently carry these selection-time fields:
- `shape`
- `strides`
- `memory_space`
- optional `config.layout`

`config.layout` is the public way for layout-sensitive templates to distinguish
different GM view organizations without re-deriving layout from the raw shape
or stride profile.

Current public layout strings are:
- `nd`
- `dn`
- `nz`
- `mx_a_zz`
- `mx_b_nn`

Rules:
- `config.layout` is present only when the source query provides explicit view
  layout metadata
- absent layout metadata remains absent; it is not normalized to `nd`
- `TensorView` and `PartitionTensorView` constraints read this value through
  `view.config.layout`
- layout-sensitive template selection may choose different descriptors even
  when element dtype is identical

## Examples

Examples aligned with the implemented surface:
- `tilelang-dsl/examples/v1_elementwise_tail_demo.py`
  - emits a guide-style elementwise authoring kernel
  - covers DMA, advanced-only explicit `strict_vecscope`, dynamic loop bound, and typed tail mask
- `tilelang-dsl/examples/v1_verify_smoke.py`
  - emits a minimal module that is expected to pass the current repo
    `ptoas --pto-backend=vpto` legality path

Typical usage from the repository root:

```bash
PYTHONPATH=$PWD/tilelang-dsl/python \
  python3 tilelang-dsl/examples/v1_elementwise_tail_demo.py

PYTHONPATH=$PWD/tilelang-dsl/python \
  python3 tilelang-dsl/examples/v1_elementwise_tail_demo.py /tmp/tilelang_v1_elementwise.mlir

PYTHONPATH=$PWD/tilelang-dsl/python \
  python3 tilelang-dsl/examples/v1_verify_smoke.py
```

## Historical Deferred Features

The following remained outside the original v1 lowering boundary and were
assigned to follow-up changes:
- implicit vecscope inference
- matcher registry and deterministic selection
- raw pointer / low-level DMA / `copy_ubuf_to_ubuf` authoring surface
- compare/select, predicate movement, carry, rearrangement, reduction families
- wildcard / type-variable dtypes
- multiple `dtypes` signatures

Primary follow-up change:
- `extend-tilelang-dsl-matcher-and-advanced-surface`

In the current package head, that follow-up has implemented matcher dispatch,
implicit vecscope inference, raw pointer / low-level DMA authoring, and
compare/select + predicate movement + carry + rearrangement families.
Reduction remains deferred because the repo still does not expose a public
authoring-form VPTO reduction op for TileLang DSL to target directly.

## Minimal Validation

The minimal validation set for the implemented v1 lowering is:

```bash
python3 -m py_compile tilelang-dsl/python/tilelang_dsl/*.py

PYTHONPATH=$PWD/tilelang-dsl/python \
  python3 -m unittest $PWD/tilelang-dsl/tests/test_tilelang_dsl_v1.py

PYTHONPATH=$PWD/tilelang-dsl/python \
  python3 tilelang-dsl/examples/v1_verify_smoke.py /tmp/tilelang_v1_verify.mlir

ptoas --pto-arch a5 --pto-backend=vpto --emit-vpto \
  /tmp/tilelang_v1_verify.mlir -o /tmp/tilelang_v1_verify.checked.mlir
```

What these commands confirm:
- the standalone source-tree package imports and compiles
- the focused unittest suite passes for lowering, diagnostics, and verify behavior
- a generated TileLang DSL v1 module can be emitted to MLIR
- the emitted verify-smoke module passes the repo VPTO authoring-stage legality path
