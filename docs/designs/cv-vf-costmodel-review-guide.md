# Native VF / CV / PlanMemory: cost model review guide

2026-10-27 更新：固定 FA 已实现“先 TileSim 初选、后编译选中候选”。
[新验收](cv-model-first-acceptance-20261027.md) 记录新 G3 与 20 对 A/B；
本页后文的 2026-10-23 全编译后评估数据保留为历史证据，不能替代新流程。


This is an experimental integration for joint interface review. It combines
the MR289 VF implementation with native CV scheduling and late physical storage
binding. It is not a proposal to enable this pipeline by default.

2026-10-27 follow-up: the [model-first implementation plan](cv-vf-model-first-implementation-plan.md)
replaces full compilation of every candidate as the target search flow. The
results below retain their frozen scope. The new flow has not been implemented.

## Scope and dependencies

The frozen compiler baseline is upstream `37f5f25e3`, with the VF implementation
ported from GitCode MR289 (`73c820689`). Commit `f4df5091f` is the integration
checkpoint before native CV / late PlanMemory work. The implementation after
that checkpoint is the main review scope; the upstream PR also contains its
unmerged dependencies. Existing exchange work in GitHub #1554 remains a separate
PR. Do not merge this integration without resolving that overlap and rebasing
the tested dependencies.

Design: [late storage binding ADR](../adr/adr-cv-vf-late-storage-binding.md),
[prepared exchange](cv-vf-prepared-interface.md), and
[implementation plan](cv-vf-planmemory-implementation-plan.md).
The preload definition follows the iteration-distance semantics discussed in
[ADR PR #1292](https://github.com/hw-native-sys/PTOAS/pull/1292).

## Compiler and model boundary

```text
logical buffers / paired C and V functions
  -> prepare reusable VF bodies with explicit pointer interfaces
  -> export structured program and compiler-defined candidate schedules
  -> annotate a checked model Plan
  -> native CV scheduling and buffer versions / slots
  -> native PlanMemory, InsertSync and final object generation
  -> export final physical program and same-compilation feedback
  -> evaluate and freeze the model selection
  -> independently validate the compiled execution, then run G3 and paired A/B
```

The fixed FA experiment compiles its bounded candidate set before final model
evaluation. The model therefore evaluates the actual layout and synchronization,
not an unverified pre-lowering estimate. Selection is frozen before any new
device measurement. Reapplying the selected Plan must reproduce the compiled
annotation file exactly.

PTOAS owns object identity, legal schedules, storage and synchronization. TileSim
owns timing and ranking. VfSimulator runs in a separate process. The model cannot
inject physical addresses. The compiler independently reconstructs execution
order and access coverage; predicted timestamps are not a safety proof.

Existing v1 / 2.0 contracts and annotation-only defaults retain their behavior.
The prepared interface is a separately negotiated experimental extension. Native
application is explicit. Model timing, rankings and limitations stay in reports.

## Files to review

| Area | Entry points | Review question |
| --- | --- | --- |
| Reusable VF | `lib/PTO/Transforms/CVVF.cpp` | Are pointer effects, ownership and stage provenance preserved? |
| Native schedule | `lib/PTO/Transforms/CVSchedule.cpp` | Does execution follow Prefix(t), then Suffix(t-min(P,N)), with enough versions? |
| Physical storage | `lib/PTO/Transforms/Passes/PTOPlanMemoryModern.cpp`, `tools/ptoas/ptoas_pipeline.cpp` | Is actual planning performed, including shared backing and both AIV domains? |
| Python exchange | `ptodsl/ptoas/cv_vf.py` | Are input, Plan, schedule, compiler and feedback identities checked? |
| Fixed FA import | `test/samples/CVCostModel/fa_native_ab.py` | Are legacy aliases and physical row-load tails handled without changing valid shape? |
| Model bridge | `test/samples/CVCostModel/fa_vf_model.py` | Are unsupported instructions rejected and timing approximations declared? |
| Safety and measurement | `fa_semantic_audit.py`, `fa_model_ab.py`, `summarize_fa_ab.py` in the same sample directory | Can fabricated dependencies, stale artifacts or incomplete measurements pass? |

## Frozen FA results

The [acceptance report](cv-vf-fa-model-acceptance-20261023.md) contains the input
SHA, toolchain, provenance, excluded runs and full interpretation.

All four candidates use the same native VF, actual automatic PlanMemory, fixed
buffer capacities, disjoint physical layout and PTOAS synchronization. Only
preload changes. P=0 is the same statically expanded serial stage graph.

| Iteration-distance P | Predicted latency (us) | Measured median (us) | Prediction error |
| --- | ---: | ---: | ---: |
| 0 | 96.7286 | 111.5785 | -13.31% |
| 1 (selected before measurement) | 72.2706 | 74.7865 | -3.36% |
| 2 | 72.2717 | 75.3770 | -4.12% |
| 3 | 72.2728 | 75.3240 | -4.05% |

Restricted candidate G2 passed. G3 passed 49/49 runs. Formal performance contains
80/80 samples in 20 balanced blocks. Selected P=1 versus P=0 has paired mean
latency reduction **33.04%**, bootstrap 95% interval **32.52%-33.55%**:
`AB_BENEFIT_PASS`. Prediction underestimates measured gain by 7.75 percentage
points. Candidate prediction accuracy and measured optimization benefit are
separate conclusions.

Hardware is Ascend950DT_9592; timing uses the explicitly approved 950PR_9599 proxy.
The result is `G4_NOT_CERTIFIED`, with no automatic enablement. This is neither
a VF-on/off ablation nor a multi-buffer search. No claim is made about matching
the other team's absolute latency or replacing its best compiled artifact.

## Reproduction and remaining review gates

Use [the experiment procedure](cv-vf-fa-model-ab-20261023.md). Keep the compiler,
TileSim and VfSimulator revisions pinned; do not evaluate against an unrecorded
latest model. The original FA is supplied separately and checked by SHA.
The experiment requires the matching CANN/PTO headers and A5 launcher environment.

The retained evidence bundle is `PTOAS_FA_proxy_AB_20261023.tar.gz`, SHA256
`0753fdd8ccf87fcd3309e3f13c2874ffde9504ab5de2052d499087a408c57186`.
It includes frozen inputs, Plans, model results, objects, correctness outputs and
raw profiler data. It is a separately delivered artifact, not a repository file
or an automatically downloadable PR attachment.

The board compiler was `c453c1f744605fdfbf773fc586ea8b443d22e75f`.
Later commits change sample tooling, tests and documentation; `include`, `lib`,
`tools` and `ptodsl` match that compiler revision. This provenance does not replace
a current-PR-head VF key-subgraph A/B/C/D acceptance matrix. That merge gate,
current-head full compiler regression, latest-master integration and VfSim-ON
configuration validation remain pending. The PR is Draft for this reason.

Generic compiler feedback still reports `evidence_incomplete`. The fixed FA has
a separate restricted certificate; it does not prove arbitrary asynchronous
programs safe. General FA recognition, address-reuse search, multi-buffer search,
DT parameter certification, new A3 board coverage and default enablement remain
follow-up work.
