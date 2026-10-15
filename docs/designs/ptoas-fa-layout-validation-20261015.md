# FlashAttention A5 layout correction and compiler diagnostic

Date: 2026-10-15. Scope: the supplied serial FP16 FA, S1=128, D=128,
S2=128/256, block_dim=8. No preload, multi-buffer, or performance claim.

## Failure and correction

The original `flash_attention_serial.pto` declares an 8x128 FP16 vector NZ tile
with 2048 bytes. The installed A5 `TMovToVecNd2Nz` uses a 16-row C0-block stride
and writes across 3840 bytes. Active writes overwrite the adjacent live FP32
accumulated numerator. For zero Q/K/V, the intermediate softmax values are 1;
two FP16 ones overwrite each FP32 word as `0x3c003c00`. After division by 256,
the erroneous value is exactly `3.057345747947693e-5`. The predicted 4096 bad
positions and every output bit matched two independently captured runs.

Padding only that allocation removes the overwrite but does not fix the NZ read
stride and V-to-C split geometry. The corrected input uses Cube M32 and 16
physical/valid rows per AIV, with corresponding query indexing and pipe slot
sizes. The mathematical operation and full GM tensor shapes remain unchanged.

The compiler now diagnoses insufficient ND-to-NZ destination capacity before
lowering. The original input is intentionally rejected; it is not automatically
rewritten. See [the storage-bound ADR](adr-a5-nd2nz-storage-bounds.md) for the
exact span calculation and its limits. This is not evidence of a Bisheng
miscompilation or a numerical regression introduced by PR #1554: the original
input also fails on unmodified upstream `master@29d24227d`.

## Cost-model integration

- `flash_attention_serial_m32.pto` is the positive structured export and device
  fixture. Its exported symbols match the original customer driver.
- The original FA remains an explicit negative export regression.
- The flat four-stage interface fixture now uses legal 32x32 / AIV16 geometry.
  Expected local bytes are five 4096-byte Cube allocations and three 2048-byte
  Vector allocations; pipe slots are 4096 bytes, capacity four.
- The existing generated basic/crossing runtime inputs for N=0/1/2/4/8 remain
  byte-for-byte unchanged. Historical measurements retain their original IDs.
- Protocol v1/2.0 fields and annotation meanings are unchanged. New source
  geometry requires new package/Plan fingerprints; old plans must be re-exported.
- Structured FA still supports G1 Buffer annotation only. This correction does
  not add recurrence-aware TileSim timing or FA preload materialization.

## Reproduce with the customer driver

Build the updated PR branch with its matching Python/native package. In a new
reproduction directory, place the supplied `runtime_check.cpp` and `run.sh`,
and copy `test/samples/CVCostModel/flash_attention_serial_m32.pto` there as
`flash_attention_serial.pto`. Keep the original faulty input separately.

Activate the target's own CANN environment, point `PTOAS_BIN` to the rebuilt CLI,
and run the supplied helper using an available device and its exclusive queue.
Use these compilation options:

```sh
ptoas flash_attention_serial.pto --pto-arch=a5 --pto-backend=emitc \
  --pto-level=level2 --vpto-scheduler=off --enable-insert-sync -o fa.o
```

For interface-only validation, no device or Bisheng is needed:

```sh
ptoas costmodel v2 export test/samples/CVCostModel/flash_attention_serial_m32.pto \
  --profile test/samples/CVCostModel/a5_profile.json --output fresh-fa-package
```

The supplied driver's `one` case means Q=K=0, V=1. A separate recorded
`allones` case explicitly initializes Q=K=V=1. Both must produce O=1. Zero
must produce O=0; pattern uses independent FP64 softmax/matmul golden.

## Retained validation evidence

The upstream-based storage-check commit `521d95826` passed 2053 lit tests
(one unsupported, zero failures) and 16/16 repeated device cases. The latter
cover two S2 values, zero/one/allones/pattern, and two repetitions. Constants
have zero error; pattern maximum absolute errors are 1.91627222e-5 (S2=128)
and 1.42407756e-5 (S2=256), within the customer's unchanged tolerance.

Platform: Ascend950DT_9592, CANN 9.2.0-beta.1, Bisheng `5c68a1cb1231`,
LLVM 19.1.7, Python 3.11.6. Independent NumPy recomputation agrees with the
driver's FP64 reference. Inputs are unchanged and repeated outputs are stable.
No GM guards were present; these checks do not prove all accesses safe.
ACL runtime probing passed, but full device health telemetry was unavailable.

The customer's Ascend950PR_9579 / CANN 9.1.0 remains a required downstream
retest. This report does not transfer results across those toolchain versions.

The PR-branch backport is independently rebuilt and validated under experiment
`20261015-fa-pr1554-integration-r7`; keep its results separate from the
upstream-based compiler and historical performance experiments.

Implementation source freeze: `cfe04b9f1c1414f28654c5bdae03d9c333bb47c1`.
All 8018 tracked source files match the native build input byte for byte.
Subsequent documentation-only evidence updates do not change that code freeze.

| PR-branch check | Result |
|---|---|
| Focused contract/storage lit entries | 7 passed, including 13 new storage scenarios |
| Full compiler lit | 1957 passed, 1 unsupported, 0 failed |
| Full sample generation/compilation smoke | 877 passed, 19 skipped, 0 failed |
| Original FA | Required 3840 bytes / available 2048 bytes; no object generated |
| Corrected M32 FA on A5 | 16/16 repeated runs passed |
| Independent NumPy golden | Passed; reference difference <=1.67e-16 |
| Comparison with upstream-based R6 | All 16 captured output files bit identical |

The four actual allones Q/K/V input sets were also inspected: every stored FP16
value is exactly one. Constant output error is zero. Pattern errors and all
toolchain limitations remain as stated above. The sample pass result here is
for the PR branch; it does not replace the separately recorded mainline sample
failure in `Img2col/timg2col_runtime.py`.
