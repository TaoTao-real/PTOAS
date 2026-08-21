# ADR-0003: 以主行迭代域规划多行单 VL RMSNorm 融合

- 日期: 2026-08-21
- 状态: Accepted
- Extends: ADR-0002 的 selected-VMI-only FusionRegion 契约

## Context

ADR-0002 要求普通 `pto.fusion_region` 只包含已经选择 VMI candidate 的
TileOp，并以完整 TileOp 迭代域隔离不同 phase。这足以覆盖 `1x4096` RMSNorm
的 chunk accumulator，但不能表达另一类典型 VF：多行数据中，每一行恰好是
一个 A5 FP32 VL，并在同一行迭代内完成 reduce、scalar 和 broadcast 阶段。

这类计算的完整 tile shape 会按阶段变化：`Nx64 -> Nx1 -> Nx64`。若只比较
完整 shape，planner 会把一条逐行流水线拆成多个 region，即使所有阶段都具有
完全相同的外层 `row = 0..N, step=1` 迭代。

## Decision

1. 本轮 `Nx256B` 固定指 FP32 计算 tile `Nx64xf32`；输入、gamma 和输出为
   BF16。`256B` 是一行一个完整 A5 FP32 VL，不是 256 个 FP32 元素。
2. Fusion analysis 为 selected VMI TileOp 推导内部的 principal row domain：
   静态 `row = 0..N, step=1`。完整 iteration-domain class 保留，继续服务现有
   conservative/EmitC 策略。
3. `vmi-ub-disjoint` 仅在下列证明均成立时，允许跨完整 shape domain 将
   `Nx64 -> Nx1 -> Nx64` 放入同一 region：
   - 相同的静态 `N`，并且每行完整有效；
   - 每行恰好一个 FP32 VL；
   - 相邻 selected VMI TileOp 之间存在 row-preserving 数据依赖；
   - 没有 tail、dynamic shape、跨行依赖、未知 alias/escape、DMA、sync、call
     或未知内存效果。
4. 首期只为 one-VL convert/elementwise、row reduction、compact scalar chain、
   row broadcast divide、gamma column-broadcast multiply 和 final convert 推导
   principal row domain。无法识别的 op 保持原 domain boundary。
5. gamma `[1,64]` 是只读 loop-invariant side input。其 BF16-to-FP32 conversion
   保持为 row loop 的 preheader producer，不以伪造的 `N` 行 domain 合入主
   region；只有后续独立的 invariant promotion 可以在证明无 alias/escape 后
   令其 vreg 跨 row loop 存活。
6. Region planning、outer-row loop fusion 和物理 UB load/store elimination 是
   三个独立优化阶段。Region membership 不代表 loop 已融合，loop fusion 也不
   代表中间 UB traffic 已消除。
7. 任一证明失败必须完整保留独立 candidate 的正确实现，且不得改变 reduction
   次数、浮点结合顺序、mask、rounding 或可观察存储。

## Alternatives

### A. 继续按完整 `vRow x vCol` domain 分组

不采用。它会把同一逐行 RMSNorm 流水线人为拆成 `Nx64`、`Nx1`、`Nx64`
三个 region，阻止典型 row VF。

### B. 将所有具有相同 row 数的 TileOp 合并

不采用。相同 row 数不能证明相同迭代映射，也不能排除 transpose、tail、跨行
依赖、side-effect 或 alias。

### C. 本轮同时支持 `Nx256` FP32 元素

不采用。每行 256 个 FP32 元素包含四个 VL，会同时引入多 VL 行内 accumulator
与跨 chunk phase fusion，应作为后续独立设计。

## Consequences

- 多行单 VL RMSNorm 可以形成一个 selected-VMI-only 主 region 和一个 row
  VLOOP，同时保持 ADR-0002 的候选选择与边界契约。
- conservative/EmitC、A2/A3、TileLang 和 ordinary PTODSL fallback 不受影响。
- `tload`/`tstore` 仍是 DMA hard boundary；必要的 GM/UB 同步不得被误报为
  VF 内部 membar。
- 首期仅支持静态、full-valid、row-major `Nx64xf32` 与 compact `Nx1xf32`。

## Validation

- 使用 BF16 `x/y=[N,64]`、BF16 `gamma=[1,64]`，覆盖 `N=1/8/32/64`。
- 主 region 只包含 selected VMI TileOp；gamma preheader、DMA 和 fallback 在外。
- lowering dump 分别证明 candidate、region、row loop 和 UB traffic delta。
- independent golden、AscendC AC-U/AC-F 和 PTO A/B/C/D 最终输出必须
  byte-exact；任何 mismatch 均停止性能验收。
- A5 性能主档为 `N=64`，另报告 `N=1/8/32/64` 的 AC-U/AC-F/B/D 扩展表。
