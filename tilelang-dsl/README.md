TileLang DSL v1 lives under this directory.

This subtree is the source of truth for the new frontend introduced by
`add-tilelang-dsl-core-foundation`.

Boundary with the existing `python/pto/dialects/pto.py` module:
- `tilelang-dsl/` owns new TileLang DSL v1 core implementation work
- `python/pto/dialects/pto.py` keeps PTO dialect bindings and the legacy
  experimental VPTO Python DSL surface
- Root-level wiring into build/install/test is allowed, but TileLang DSL core
  logic must not move back into `python/pto/dialects/pto.py`

Layout:
- `python/tilelang_dsl/`: package sources
- `tests/`: TileLang DSL focused tests
- `examples/`: self-contained examples
- `docs/`: local documentation for this frontend

## How To Generate MLIR From A `.py`

Run the examples from the repository root.

If you are developing against the in-tree Python sources, point `PYTHONPATH`
at `tilelang-dsl/python`:

```bash
cd /home/zhangzhendong/ptoas-workspace/PTOAS
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_emit_mlir_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_emit_mlir_demo.py /tmp/tilelang_demo.mlir
```

If you already built and installed the Python package into the repo build tree,
you can also point `PYTHONPATH` at `build/python`:

```bash
cd /home/zhangzhendong/ptoas-workspace/PTOAS
PYTHONPATH=$PWD/build/python python3 tilelang-dsl/examples/v1_emit_mlir_demo.py
```

Behavior:
- without an output path, the script prints MLIR to stdout
- with an output path, the script writes MLIR to that file through `emit(path)`

Useful examples:

```bash
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_elementwise_tail_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_elementwise_tail_demo.py /tmp/tilelang_v1_elementwise.mlir
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_verify_smoke.py /tmp/tilelang_v1_verify.mlir
```

## Advanced Mode

The default v1 surface still requires explicit `pto.strict_vecscope`.

If you want the follow-up advanced surface for:
- implicit `pto.vecscope` inference
- `pto.vlds(tile[row, col:])`
- `pto.vsts(vec, tile[row, col:], mask)`

set `advanced=True` on `@pto.vkernel` and follow
[`tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py`](/home/zhangzhendong/ptoas-workspace/PTOAS/tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py).

## Minimal Script Pattern

Your own `.py` only needs to:
- import `tilelang_dsl`
- define a `@pto.vkernel`
- call `specialize(...)`
- call `mlir_text()` or `emit(path)`

Minimal example:

```python
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    op="eltwise_with_tile",
    dtypes=[(pto.f32, pto.f16, pto.i32)],
    name="my_kernel",
)
def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
    return None


specialized = kernel.specialize(
    tile=pto.TileSpecialization(
        shape=(16, 32),
        memory_space=pto.MemorySpace.UB,
    )
)

print(specialized.mlir_text())
specialized.emit(Path("/tmp/my_kernel.mlir"))
```

If `python3 your_script.py` reports `ModuleNotFoundError: tilelang_dsl`, it
means the package import path is missing. Re-run with one of:

```bash
PYTHONPATH=$PWD/tilelang-dsl/python python3 your_script.py
PYTHONPATH=$PWD/build/python python3 your_script.py
```

## Optional Verifier Check

To check that the generated MLIR passes the current repo VPTO authoring-stage
legality path:

```bash
LLVM_BUILD_DIR=/path/to/llvm/build ./quick_install.sh
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_verify_smoke.py /tmp/tilelang_v1_verify.mlir
ptoas --pto-arch a5 --pto-backend=vpto --emit-vpto \
  /tmp/tilelang_v1_verify.mlir -o /tmp/tilelang_v1_verify.checked.mlir
```

For the implemented authoring-form VPTO lowering contract, support matrix,
examples, and minimal validation commands, see
`tilelang-dsl/docs/v1-lowering.md`.

Root-level wiring belongs to follow-up tasks and must stay minimal.
