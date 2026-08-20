# ADR-0002: FusionRegion 采用已选 VMI candidate，并单独规划 accumulator phase

- 日期: 2026-08-21
- 状态: Accepted
- Supersedes: ADR-0001 中“不消费 `pto.fusion_region`”以及“不支持跨迭代 accumulator”的相关决定

## Context

当前 A5 VPTO pipeline 在 Tile-native PTO IR 上先运行 FusionPlan 和
FusionRegionGen，真正的 TileLib candidate selection 却在后续 ExpandTileOp 中完成。
因此 planner 只能依据 TileOp 名称白名单推测可融合性，可能先把普通 PTODSL fallback
放入 `pto.fusion_region`，再由 ExpandTileOp 选择非 VMI 实现。

RMSNorm 还包含另一类问题：一条 64-lane FP32 accumulator 在 chunk loop 中逐 lane
更新，循环结束后才执行一次 `trowsum`。它不是普通同迭代域 loop fusion，不能通过扩大
普通 FusionRegion 白名单解决。

## Decision

1. Candidate selection 前移到 FusionPlan 之前，并成为 ExpandTileOp 与 fusion planning
   的唯一选择结果。ExpandTileOp 只消费和验证已选 candidate，不重新决策。
2. A5 VPTO 的普通 `pto.fusion_region` 只允许包含已经选择 canonical VMI candidate 的
   TileOp。alloc、地址计算和必要控制流可以作为结构存在，但 PTODSL fallback、DMA、
   同步和未知 TileOp 不得成为 region member。
3. 非 VMI TileOp 按语义成为 local 或 hard boundary。FusionRegionGen 必须防御性验证
   region member；发现违反契约时编译失败，不静默修剪 span。
4. 每个 canonical VMI candidate 在没有 region、loop fusion 或 mem2reg 时必须独立正确。
   Fusion 只能删除已证明冗余的循环和 UB 往返。
5. 普通同域 region fusion 与 accumulator phase promotion 是两个独立优化阶段。后者仅在
   storage root、byte range、mask、迭代关系、同步和 escape 均可证明时，把 accumulator
   提升为 loop-carried VMI vreg，并将循环最终值直接交给 post-loop reduction。
6. 任一证明失败均保留原始独立实现，不改变数值语义、浮点结合顺序或可观察存储。
7. 正确性与性能验收固定为 ordinary、candidate-only、loop-only、loop+mem2reg 四档；
   AscendC 基线可用时还必须先证明 workload、dtype、layout、rounding 和计算流一致。

## Alternatives

### A. FusionRegion 继续作为可能混合 PTODSL/VMI 的宽松容器

不采用。它让 downstream pass 无法从 region membership 判断 VMI 优化是否合法，并使
candidate fallback 在规划后改变 region 的真实语义。

### B. ExpandTileOp 后再删除不合法 region member

不采用。此时 scheduling、span interface 和 escaping value 已按错误成员关系建立，静默
删除会引入新的顺序、别名和 region result 风险。

### C. 把 post-loop reduction 强行并入普通同域 loop fusion

不采用。chunk accumulator 与最终 reduction 属于不同执行 phase；把 reduction 移进
chunk loop 会改变算法和浮点结合顺序。

## Consequences

- FusionPlan、ExpandTileOp 和 lowering 共享一个可打印、可测试的 candidate 决策。
- VMI region membership 具备稳定含义，普通 fallback 自然成为融合边界。
- RMSNorm 的 lane-wise accumulator 可以在保持 AscendC 数据流的前提下单独优化。
- Pipeline 需要显式的 candidate-selection pass、阶段诊断控制和 accumulator provenance。
- 首期 accumulator promotion 仅覆盖静态 `1x64xf32` sum；动态 tail、其他 reduction 和
  任意 iter_args 泛化保留为后续工作。

## Validation

- lit 必须覆盖 VMI/PTODSL/VMI 分段、DMA hard boundary 和非法 metadata 拒绝。
- 每个 candidate 必须先通过独立 ordinary/VMI 数值对比，再进入 fusion 验收。
- lowering dump 必须分别证明 candidate、region、loop 和 load/store-elision delta。
- A5 最终验收使用固定输入、独立 golden、冷启动检查和至少五次串行采样。
