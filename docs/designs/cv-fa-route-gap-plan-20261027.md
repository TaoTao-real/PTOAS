# FA native versus delivery route: controlled gap investigation

Status: planned; no new performance result yet. The previous 12.81% figure is a five-sample diagnostic comparison of complete routes, not an attribution.

## Frozen workload and controls

Use the SHA-bound M32 FA input (Q/KV/D=128/1024/128, FP16 inputs, FP32 output), the same six correctness inputs and three performance seeds, Ascend950DT_9592 on host alias ptoas-a5-39, one AIC and two AIV. Select one ACL-probed device before execution. Keep all new samples on that device. Retain delivery depth=4 and native P=1 controls; those parameter definitions are not interchangeable.

Predeclare these native variants before new device measurements:

| ID | Difference from R | Question |
|---|---|---|
| R | P=1, current fused VF, conservative completion, allocation-order disjoint PlanMemory | Reproduce native baseline |
| S | R without PIPE_ALL immediately after every VF; full drains before cross-core signals and return remain | Cost of conservative completion |
| L | R with largest-first disjoint PlanMemory | Layout/bank sensitivity |
| SL | S plus L | Synchronization/layout interaction |
| P3 | R with P=3 | Was the model's P=1/P=2/P=3 timing tie misleading? |
| VB | Same VMI candidates/regions as R; loop fusion and load/store elision disabled | Candidate versus fused body |
| VC | Same as VB, loop fusion enabled, elision disabled | Loop-fusion contribution |
| T | Frozen delivery depth=4 binary, same inputs and launch boundary | Absolute reference |

R is the D configuration in the B/C/D VF experiment. This is not a general ordinary/VMI candidate certification. Inspect pass dumps to confirm identical candidates for VB/VC/R; report no-op switches explicitly. Native variants use PTOAS InsertSync and vecscope barriers only. Do not add Bisheng auto-sync. T retains its original, separately labelled Bisheng fused policy.

## Safety and artifacts

Default compiler behavior stays unchanged. Add only an opt-in `--cv-drain-vf-calls=false` static-CV experiment. Bind its changed event schedule and completion policy in feedback. Retain actual native PlanMemory for all native variants and forbid address reuse.

Do not accept a linear IR/event sequence as proof of asynchronous completion. Build a separate finite pipeline-order DAG from final IR: per-pipe FIFO, explicit local set/wait generations, cross-core set/wait generations, and PIPE_ALL completion edges. Derive physical accesses from compiler VF bodies and checked address expressions. Require all overlapping RAW/WAR/WAW pairs, one-bit flag generations, storage bounds and output completion to be proved without predicted times. Unknown operations or cycles fail G2. Test the checker with missing completion edges and cycles before using it.

Capture same-compilation final IR, memory/validation feedback, exact object and launcher commands, final LLVM and assembly. Normalize symbols/provenance to check that S changes synchronization only and L changes layout only; report compiler-induced secondary differences.

## Correctness and performance

Run all variants twice over zero, one_v, one_qkv and three fixed pattern seeds, with reverse orders and optimized-first coverage. Use independent FP64 C++ and NumPy golden, input/guard checks and a 60-second per-run timeout. A failing variant is excluded and diagnosed; it is never used as a valid performance result.

After G3, use a new performance experiment: one warmup each, then 20 balanced rotated/reversed blocks, identical seed per block, one profile per variant, exactly one MIX_AIC task (1 AIC+2 AIV). Freeze binaries and matrix before timing. Restore and validate inputs/outputs each run. Report paired mean gains, fixed bootstrap 95% intervals (10,000 resamples, seed 20260916), medians and ranges. Do not tune variants from this timing batch. An optimized native candidate must improve R by >=2%, interval lower bound >0, and no pair regression >2% to claim benefit.

Compare R/S, R/L, S/SL and R/P3 separately. Compare VB/VC/R only with structural evidence. Compute the residual to T in this same batch; do not add independent percentages or attribute route interactions to a single pass. Model timing accuracy and target-proxy limits remain separate from measured optimization benefit.

## Delivery

Provide per-variant G2/G3 and latency tables, structural/assembly evidence, causal limits, optimization follow-up priorities and raw evidence hashes. Update the existing draft PR after validation. Broad VF key-subgraph regressions and target-matched G4 remain distinct merge/certification gates.
