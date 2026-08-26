# ADR-0004: 通用 VMI 向量状态提升

- 日期: 2026-08-26
- 状态: Accepted
- Extends: ADR-0002/0003 的 selected-VMI FusionRegion 与 phase/row-domain 契约

## Context

现有 RMSNorm 优化在 loop fusion 之后用专用 accumulator/scalar planner 消除
UB 中间状态。它证明了“跨 phase 保持 vreg”可以显著减少 VLD/VST 与内部同步，
但其判定依赖 RMSNorm 形状和固定算子序列，不能覆盖 Softmax、RoPE、多消费者或
多结果重排，也无法为拒绝原因和寄存器压力提供稳定接口。

这不是传统 scalar `mem2reg` 的重复实现。普通 `mem2reg` 提升的是编译器内存
对象；本优化发生在 selected-VMI region 已展开并融合之后，提升的是具有 UB
地址、lane layout、mask 与硬件向量资源约束的显式 `vmi.vstore/vmi.vload`
状态，并需要保持循环零迭代和可观察 DMA 语义。

## Decision

1. 正式能力命名为 **通用 VMI 向量状态提升**（Generic VMI Vector State
   Promotion），pass 名为 `pto-vmi-state-promotion`；其分析模型称为
   **Phase-aware Vector Mem2SSA**。
2. `VMIStateLocation` 由规范化 storage root、静态 byte range、element type、
   distribution/layout 与静态 mask coverage 构成；只有完全相同且证明兼容的
   producer/consumer 才可形成 state flow。
3. 首版支持 `StraightLine`、`LoopInvariant`、`LoopCarried` 三类 flow：
   - VST→VLD forwarding，包括跨 FusionRegion 与一对多消费者；
   - 只读循环不变 load hoist；
   - read-modify-write 状态改写为 `scf.for iter_args/yield/result`；
   - compact `vreg<1>` 通过显式 `vmi.vbrc` 扩展。
4. static full mask 与所有消费者均不观察 inactive lanes 的 static prefix mask
   可提升。dynamic/tail、未知地址、partial alias、escape、DMA/sync/call、嵌套
   控制流或未知 effect 必须保留原路径。
5. 改写必须事务化：证明、mask 和资源检查全部成功后才修改 IR。可观察 store
   不删除；内部精确 AllocTile 的全部 reload 均被替换后才删除 dead store。
   UB 地址复用不能单独作为 escape 或可观察性的依据。
6. 每个决策写入稳定的 `accepted/rejected`、flow 和 reason 属性/remark。reason
   枚举固定为 `alias`、`mask`、`escape`、`control-flow`、
   `resource-pressure`、`unknown-effect`、`unknown-location`、`type-layout` 和
   `multiple-reaching-definitions`。
7. `VMIVectorPressureEstimate` 按 A5 256-byte physical chunk 估算 peak、
   persistent、temporary 和 loop-carried live range。估算不精确时拒绝；生产
   默认采用不强制增加预算的保守 admission，显式 chunk budget 仅用于诊断和
   target profile gate。
8. pass 位于 Expand/Inline、shape folding 和 low-level loop fusion 之后，VMI
   layout assignment 之前。VMIToVPTO 后的物理 load/store elision 继续作为最终
   清理，但不负责跨控制流 promotion。
9. 迁移期提供隐藏选项
   `--vmi-state-promotion-mode=legacy|shadow|generic|off`，默认 `legacy`。
   `shadow` 只比较通用分析与 legacy planner；完成结构、正确性、spill 和性能
   门禁后切换到 `generic`，随后删除 legacy/shadow 与算子专用 metadata。
10. RoPE 重排统一归入 `FusionComputeFamily::Rearrange`。channel split/merge 的
    row mapping 由 family 与输入/输出 shape 关系证明，不以具体算子名字推断；
    arithmetic/rearrangement 临时状态保持 SSA/vreg，HALF 32-lane narrowing 使用
    RINT/NOSAT。

## Resource fallback contract

- 对 loop fusion run 与每个 promotion candidate 分别估算变换前后 physical
  chunks；不精确或超过 target budget 时仅拒绝当前候选/segment。
- 压力超限采用确定性 greedy segmentation，不影响其它已证明合法的 region。
- 离线 Bisheng probe 记录 Vector Slots、spill bytes、stack object size 与对象
  生成结果。没有跨样本安全阈值时，默认不得增加估算峰值压力。

## Consequences

- RMSNorm 专用优化成为通用分析的迁移期 oracle，而不是长期双轨实现。
- Softmax、RoPE 和后续 selected-VMI 算子可以复用同一 location/mask/effect/
  pressure 证明与相同拒绝接口。
- 首版仅覆盖 A5 VPTO、静态 full-valid/static-prefix；A2/A3、EmitC、dynamic/tail
  保持原路径。
- 不改变 reduction 次数、浮点结合顺序、rounding、saturation 或可观察存储。

## Validation

- 编译器测试覆盖 straight-line、cross-region、一对多、invariant、loop-carried、
  zero-trip、broadcast、full/prefix mask，以及 alias/mask/escape/effect/control/
  pressure 拒绝。
- RMSNorm generic D 必须达到 legacy D 的 byte-exact 结果与结构流量；
  `[64,64]` 目标为 `1 loop / 2 VLD / 1 VST / 0 internal membar`。
- RoPE HALF/INTERLEAVE 以 vector oracle 与逐元素 scalar oracle 为 correctness
  gate；临时 arithmetic/rearrangement UB traffic 与内部 membar 必须为零。
- A5 在相同对象、输入和空闲设备上做 paired profile；task/vector median ratio
  与 bootstrap 95% upper bound 均不得超过 1.03。未完成硬件门禁前不得把默认
  从 `legacy` 切换为 `generic`。

