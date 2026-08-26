# RMSNorm row-VF A5 comparison harness

This internal harness builds and compares the following static `N x 64` BF16
RMSNorm implementations on A5:

| Variant | Implementation |
| --- | --- |
| ACU | AscendC ordinary vector API |
| ACF | AscendC manual `__simd_vf__` |
| A | ordinary PTODSL/VPTO |
| B | selected VMI candidates, fusion disabled |
| C | B plus region and row-loop fusion |
| D | C plus generic VMI state promotion |
| DL | C plus the migration-period legacy state planner |

The driver requires an experiment directory created by the private-lab helper
and an exact committed source tree. It generates unique kernel symbols, builds
all six paths, checks both fixed datasets byte-for-byte against the independent
golden, and only then performs serial profiling. Every profile run restores the
input and output files and rechecks the BF16 output before recording metrics.

```bash
ACL_DEVICE_ID=1 ROWS_LIST=64 PROFILE_REPEATS=5 \
  ./run_a5_matrix.sh EXPERIMENT SOURCE PTOAS MLIR_PYTHON_ROOT PTO_ISA_ROOT
```

Set `ROWS_LIST="1 8 32 64"` for the extended four-size matrix. The committed
acceptance protocol uses the same six variants for `N=64`; reports may select
ACU, ACF, B, and D as the four primary implementation paths.

After collecting at least ten interleaved profiles, evaluate the formal D over
DL no-regression gate (paired median and 95% bootstrap upper bound <= 1.03):

```bash
python3 paired_gate.py EXPERIMENT/results/samples.tsv \
  --output EXPERIMENT/results/paired-gate-10.json
```
