# ADR-0001: VMI VF Fusion 采用唯一 canonical 实现与融后 mem2reg

- 日期: 2026-07-15
- 修订: 2026-07-23（logical-row / wide logical vreg 协议）
- 状态: Proposed

## Context

PTOAS 已能通过 `--tile-lib-backend=ptodsl-vmi` 将 `PIPE_V` TileOp 展开为
语义完整的 VMI 模板。每个模板包含一个主 `scf.for` 和完整的 VMI
load/compute/store，但不显式生成 `pto.vecscope`，因此多个连续 TileOp 展开后仍会
产生多个循环和中间 UB 往返。

该 backend 是组合式路由：`PIPE_V` 使用唯一 canonical VMI provider，其他 Pipe
继续使用现有 PTODSL TileLib daemon。它不会将非向量 TileOp 回退到 TileLang。

旧的 `FusionPlan` / `FusionRegionGen` 分析 Tile-native PTO IR，
`PTOLowLevelLoopFusion` 分析已经物理化的 VPTO/MI IR。二者都不能直接承担新的
VMI loop fusion。此前讨论过为同一 TileOp 提供多个 VMI schedule candidate，再由
region-aware cost model 选择并锁定实现；该方案会把实现版本选择提前引入本期，扩大
设计和验证范围。

## Decision

1. RFC 首期对每个 `(target, TileOp, semantic_form)` 只允许一个 canonical VMI 实现。
   `semantic_form` 只区分影响正确性的操作数 schema 或属性语义，例如
   `tdivs:tile_scalar` / `tdivs:scalar_tile`、convert round mode 或带 tmp 的精度形式；它不能
   表示多个性能 schedule。Provider 必须从 operand kinds、dtype 和 context attrs 唯一匹配
   semantic form。每个实现必须脱离融合生成语义完整、verifier 合法的 VMI；不存在同一
   semantic form 内的 candidate 竞争、锁定或性能回退选择。后端物理化和 A5 数值闭环
   作为独立 Gate 跟踪。
2. Canonical candidate 只生成一个 row 或 semantic-form 主循环，不生成动态 inner-block
   loop。Tile inner 是 VMI logical lane domain，可以小于、等于或大于一个物理 VL；
   `!pto.vmi.vreg<NxT>` 由 LayoutAssignment/VMIToVPTO 在 Fusion 后映射为若干 256-byte
   物理 vreg。Convert 保持 logical lane count，例如 `128xbf16 -> 128xf32` 仍是一个
   128-lane SSA value；精确物理映射不属于 VMI Fusion 判据。
3. Tile-native `FusionPlan/OpScheduling/FusionRegionGen` 继续产生粗粒度
   `pto.fusion_region`。VMI Fusion 在 `ExpandTileOp`、`PTOInlineLibCall` 和 shape intrinsic
   folding 之后运行，在该 region 内分析真实 VMI `scf.for`、phase、memory effects、mask
   和 alias；Tile region 本身不代表组内所有 loop 必须融合。
4. 仅处理 Tile fusion region 内、结构合法且可恢复 TileLib provenance 的 canonical VMI
   fusion unit。任意用户手写 VMI、非 canonical 模板或 region 外循环默认不参与融合。
5. 融合采用保守、部分融合策略：只合并边界、循环域、依赖、alias、访问模式和 mask
   均可证明兼容的相邻循环；其余循环保持原样。
6. VMI mem2reg 必须在 loop fusion 之后运行。它只提升可证明同 location、同形状的
   VMI store-load，使融合后暴露的中间 UB 往返变为 SSA 直传。
7. 融合和 mem2reg 位于 VMI layout assignment 之前。物理 vreg layout、interleave、
   pack、post-update 和指令选择仍由现有 VMI semantic/layout pipeline 负责。

## Alternatives

### A. 在 Tile 层先选择并锁定多个 VMI candidate

暂不采用。它需要 schedule family、region-aware selection、代价模型和稳定的 fallback
协议，属于后续性能迭代，不是验证 VMI 融合基本闭环的前置条件。

### B. 直接用 `FusionPlan` / `FusionRegionGen` 完成全部 VMI Fusion

不采用为 VMI 实现本体，但采用其粗粒度 region 规划结果。可以复用其 block-local DFG、
活跃性和迭代域分析思路或抽取通用 utility；展开后的 VMI phase、wide value、mask 和
memory access 仍由 VMI-specific passes 判断。

### C. 复用 `PTOLowLevelLoopFusion`

不采用。该 pass 面向 VPTO/MI 物理循环，运行位置过晚，会重新引入物理 layout、
predicate 和地址模式对融合分析的干扰。

## Consequences

### Pros

- 每个 semantic form 输入唯一、结果确定，同时允许正确表达操作数 schema 不同的 TileOp。
- Elementwise、Reduce 输入和 Broadcast dense 输出共享 row iteration domain，减少
  loop mapping、alias offset 和 mask compatibility 的状态空间。
- Widening、narrowing 和静态 multi-VL inner 保持一个 logical SSA value，不需要在
  TileLib 中恢复 inner loop，也不会让 Fusion 提前感知物理寄存器拆分。
- 单个 TileOp 在融合失败时仍能独立 lower，天然具备保守 fallback。
- loop fusion 与 UB store-load elimination 顺序正确。
- VMI 层保留逻辑 lane、mask 和 SSA 数据流，避免在 MI 层恢复高层语义。

### Cons / Risks

- canonical 实现不一定是每个固定 Shape 的最优实现。
- VMI 类型可以表达任意正的静态 lane 数，但每个 TileLib form 仍需声明并测试支持的
  dtype/logical-width/layout relation；不能把“类型可表示”等同于“硬件 lowering 已验证”。
- Wide vreg 可能增加最终物理寄存器压力。VMI Fusion 不硬编码物理 chunk 公式或特定
  width 阈值；后端 capability/cost 接口具备后再接入准入判断，缺失时只给出结构结论。
- Compact state、dynamic tail 和 layout-changing op 仍需显式 semantic mapping；不能仅凭
  row loop 相同推断可融合。
- 当前 VMI load/store 使用线性 offset，alias 分析必须保守规范化 storage root 和
  index expression；无法证明时必须拒绝融合或提升。
- `PTOInlineLibCall` 需要保留 TileLib provenance，否则无法可靠区分模板代码与用户
  手写 VMI。

## Follow-ups

- 实现 VMI fusion-unit provenance、识别、规划、loop fusion 和 mem2reg passes；复用
  现有 late `PTOInferVPTOVecScope` 统一生成物理 VPTO vecscope。
- 把 TileLib `CanonicalBlockMap` 泛化为 logical-row contract，补齐 BF16/FP16 widening、
  2VL/4VL Elementwise、RowReduce/ColReduce 和 compact-state 测试。
- 完成 elementwise 链的正向和负向 lit tests。
- 在基本闭环稳定后，再独立评审多 candidate、Reduce schedule、cost model、outer-row unroll
  和算法专项深融合。
