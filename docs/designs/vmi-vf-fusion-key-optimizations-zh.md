# VMI VF Fusion 关键优化设计与适用场景

- 状态：Implemented / A5 validated
- 日期：2026-08-25
- 适用分支：`codex/rmsnorm-softmax-vmi-vf-performance-delta`
- A5 验证源码：`149405c569b068daa738fc5e865066d1c9a7dbcd`
- 当前分支等价提交：`957f00670`
- 设计依据：ADR-0002、ADR-0003

## 1. 目标与范围

本设计解决的核心问题不是“让更多 TileOp 能展开为 VMI”，而是让一组已经选择
VMI candidate 的 TileOp 真正形成高效的 vector function：合并相同物理迭代、
在 vreg 中转发生产者结果、删除中间 UB 往返，并只保留有证明必要的同步。

当前已验证三类计算图：

1. 4096 宽 RMSNorm：每行由 64 个 FP32 1VL chunk 组成，跨 chunk 保持 lane
   accumulator，最终只做一次 reduction。
2. `N x 64` RMSNorm：每行恰好一个 FP32 VL，计算图经历
   `N x 64 -> N x 1 -> N x 64`。
3. Softmax Dn：FP32 `[B=4,M=16,N]`，沿 M 维执行 max/sum reduction，覆盖
   `N=32/64/128`。

当前非目标包括 dynamic/tail、多 VL 与多行同时泛化、任意 alias、跨行依赖、
任意 reduction，以及 Softmax 多个独立 tile 的自动交错和短行 pack。

## 2. 优化栈与归因边界

优化按下列顺序执行：

```text
TileOp graph
  -> candidate selection
  -> selected-VMI-only FusionRegion planning
  -> candidate Expand/Inline
  -> principal-loop fusion
  -> accumulator/scalar/phase-state vreg promotion
  -> physical load/store forwarding and invariant hoist
  -> alias-aware membar placement
  -> VMIToVPTO
```

各阶段的职责必须分开：

| 阶段 | 允许做什么 | 不能据此宣称什么 |
| --- | --- | --- |
| candidate selection | 选择 VMI 或 PTODSL 实现 | 不代表 loop 已融合 |
| FusionRegion planning | 标记合法优化范围 | 不代表 UB 流量已删除 |
| loop fusion | 合并被证明相同的物理循环 | 不代表值已经寄存器直传 |
| vreg promotion/elision | 删除中间 VLD/VST、转发 SSA/vreg | 不能删除不可证明的同步或可观察存储 |
| membar analysis | 删除已证明无冲突的内部 barrier | 不能删除 GM 与 UB 边界所需同步 |

A/B/C/D 性能归因采用：

- A：ordinary PTODSL/VPTO；
- B：VMI candidate，region/loop/elision 关闭；
- C：与 B 相同 candidate，开启 region 和 loop fusion；
- D：C 加 vreg forwarding、load/store elision 和受限 invariant promotion。

## 3. 优化一：candidate 前置选择与 selected-VMI-only Region

### 3.1 问题

旧流程在 FusionPlan 之后或 Expand 阶段再次选择 candidate，planner 可能把实际
会回退到 PTODSL 的 TileOp 放入 VMI region。这样生成的 region 既不是可靠的
优化契约，也会让普通 TileOp、DMA 或未知副作用污染 loop fusion。

### 3.2 设计

- 在 FusionPlan 前完成 candidate selection，并把结果写回原 TileOp；
- `FusionComputeNode` 显式记录当前实例是否 selected VMI；
- `vmi-ub-disjoint` 只连接 selected VMI 节点；
- PTODSL fallback、DMA、sync、unknown call 都是边界；
- `FusionRegionGen` 在移动成员前再次验证，发现 PTODSL member 直接失败；
- Expand/Inline 只消费并校验已选 candidate，不得独立改选。

### 3.3 效果

该优化主要提升正确性、可诊断性和后续优化命中率，不直接保证运行时变快。
实测 candidate-only B 并非所有用例都优于优化前分支，说明不能把最终收益归因
到 TileOp expand。

### 3.4 适用场景

- 同一计算图同时包含 VMI、PTODSL 和 DMA TileOp；
- candidate legality 依赖 shape、dtype、layout、mask 或 precision；
- 需要在 FusionPlan 与 Expand 后机器校验 candidate 一致性的 pipeline。

### 3.5 相关实现

- `489e8753e`：candidate selection 前置；
- `cbd2161ca`：selected-VMI-only FusionRegion；
- `e67fa1214`：将合法 `tsqrt` 纳入 scalar fusion chain。

## 4. 优化二：受限 one-VL candidate 与 compact state

### 4.1 问题

RMSNorm 的合法 VMI chain 曾被 `tcvt`、`texpands`、`trowexpanddiv` 等 PTODSL
fallback 截断。另一个问题是 reduction 后的 `1x1` 逻辑值在物理上需要满足
A5 compact/padded state 约束，不能简单按完整 tile shape 处理。

### 4.2 设计

首期 candidate 均采用严格、可独立运行的物理约束：

- BF16→FP32：静态、row-major、full-valid、每行 64 lane；
- FP32→BF16：相同范围，narrowing 使用 RINT；
- `texpands`：静态 `1x64xf32` accumulator 初始化；
- `trowexpanddiv`：FP32 default precision，分母必须是可证明的 compact scalar；
- `N x 64 / N x 1`：静态相同行数、每行完整一个 VL；
- padded row reduction：保留逻辑 valid shape 与物理对齐的区别。

dynamic、tail、高精度除法、未知 broadcast layout 和无法证明的 mask 均保守回退。

### 4.3 效果

这些 candidate 消除了主计算链中的 PTODSL 断点，使后续 region/loop/vreg 优化
成为可能。candidate 自身的收益有限；它们是必要前提，而不是主要性能来源。

### 4.4 适用场景

- 静态 A5 one-VL elementwise/convert；
- RMSNorm、LayerNorm 等由完整 64-lane FP32 row 构成的子图；
- reduction 输出为 compact scalar、随后进行 row broadcast 的子图。

### 4.5 相关实现

- `282b66f43`、`d72914f30`、`6e8a17734`；
- `3e2265d4b`、`ffa3aba0e`。

## 5. 优化三：跨 chunk accumulator promotion

### 5.1 问题

4096 宽 RMSNorm 每行包含 64 个 FP32 1VL chunk。普通 candidate lowering 会在
每次迭代中把 accumulator 写回 UB，再由下一次迭代重新加载；循环结束后还可能
再次 VLD 才进入 reduction。这与手写 VF 在 64-lane vreg 中保持 accumulator 的
数据流不一致。

### 5.2 设计

分析只识别以下精确模式：

```text
vmi_texpands(0)
  -> fixed chunk loop
       square(chunk)
       accumulator = accumulator + square
  -> one final trowsum
```

必须证明：

- accumulator 为静态 `1x64xf32`；
- chunk 步长、次数、地址和完整 mask 可证明；
- 每次更新读取上一迭代结果并覆盖同一 storage root；
- 无 alias、escape、DMA、sync、unknown call 或额外 consumer；
- final reduction 只消费循环最终 accumulator。

转换后由 `scf.for iter_args` 携带 `!pto.vreg<64xf32>`，初始化使用 vdup，循环中
直接 vadd，final reduction 直接消费 loop result。

### 5.3 效果

在当前同设备横向测试中，4096 宽 workload 为 BF16 `[8,4096]`；每一行的 VF
body 是 `1x4096`。与优化前分支 D 相比：

| 指标 | 优化前分支 D | 性能优化分支 D | 改善 |
| --- | ---: | ---: | ---: |
| Vector median | 60.471 us | 39.971 us | 33.90% / 1.51x |
| Task median | 73.296 us | 52.982 us | 27.72% / 1.38x |
| loop/VLD/VST/membar | `3/13/13/4` | `3/8/6/6` | UB 往返明显减少 |

当前结构仍有 6 个 membar，说明该用例的后续同步清理仍有改进空间；现有收益主要
来自 VLD/VST 减少。

### 5.4 适用场景

- 长行被拆成固定 one-VL chunk；
- reduction 前存在 lane-wise associative accumulator；
- accumulator 只在 chunk loop 和最终 reduction 之间流动；
- 典型场景包括 RMSNorm/LayerNorm 的 sum-of-squares、受限列 reduction。

不适用于 dynamic tail、不同 byte range 的迭代更新、可观察临时 buffer 或需要
在每轮产生外部结果的 accumulator。

### 5.5 相关实现

- `cabf6d140`：phase planning；
- `6b14104e9`：accumulator vreg promotion。

## 6. 优化四：compact scalar phase promotion

### 6.1 问题

RMSNorm 在最终 reduction 后还有：

```text
sum -> multiply reciprocal -> add epsilon -> sqrt -> broadcast -> divide
```

这些值逻辑上只有一个 FP32 scalar，但普通 TileOp lowering 会为每一步生成 compact
UB store/load，并在相邻阶段插入 membar。

### 6.2 设计

- 识别 reduction、scale、shift、sqrt、divisor、apply-loop 六个 phase；
- VL1 标量只允许 single-use、相同 storage root、相同 offset 和 mask；
- `VL1 VST -> full-VL BRC VLD` 转换为 vreg `vdup LOWEST`；
- scalar chain 和 broadcast divisor 在 vreg/SSA 中传递；
- 任一 alias、escape、额外 use、sync 或 unknown call 使整个 promotion 拒绝。

### 6.3 效果

它与 accumulator promotion 共同删除 4096 宽 RMSNorm reduction 后的 compact UB
往返，并让 apply-loop 直接消费寄存器 divisor。该优化的收益已经包含在上一节
`[8,4096]` 的 D 路径数据中。

### 6.4 适用场景

- reduction 后的单标量 epilogue；
- scalar arithmetic 后马上广播回一个 VL；
- RMSNorm、LayerNorm、variance/standard-deviation 类计算。

### 6.5 相关实现

- `037fd1818`：VL1 mask 正确性；
- `c8ab401e3`：scalar phase planning；
- `bcee73e70`：scalar vreg promotion；
- `7d45afb0b`：删除被覆盖的 tail store。

## 7. 优化五：principal row domain 与多阶段 row-loop fusion

### 7.1 问题

`N x 64` RMSNorm 的完整 shape 在不同阶段发生变化：

```text
N x 64 -> N x 1 -> N x 64
```

若 FusionPlan 只比较完整 tile shape，convert/elementwise、row reduction、scalar
chain、broadcast divide 会被拆成多个 region 和多个 row loop，虽然它们实际共享
完全相同的 `row = 0..N step 1` 迭代。

### 7.2 设计

内部分析为受支持的 selected VMI op 推导 principal row domain：

```text
start=0, end=N, step=1, one full FP32 VL per row
```

只有在以下条件成立时才允许跨完整 shape domain 聚合：

- 静态且相同的 N；
- row-preserving producer/consumer dependency；
- reduction 为 `N x 64 -> N x 1`；
- broadcast 为同一行的 `N x 1 -> N x 64`；
- gamma `[1,64]` 被证明为只读 loop-invariant side input；
- 无 tail、alias、escape、DMA、sync、unknown call 或跨行依赖。

低层 loop fusion 随后只合并具有相同 loop header 的已证明 row loop，并保持每行
算子顺序和 reduction 次数不变。

### 7.3 效果

RMSNorm `[64,64]` 是当前最明显的收益：

| 指标 | 优化前分支 D | 性能优化分支 D | 改善 |
| --- | ---: | ---: | ---: |
| Vector median | 4.700 us | 0.518 us | 88.98% / 9.07x |
| Task median | 9.731 us | 5.443 us | 44.07% / 1.79x |
| row loop | 8 | 1 | 合并为一个主 VLOOP |

只做 loop fusion 的 C 路径仍保留中间 UB 往返，因此 loop 数下降本身不代表性能
收益；它必须与下一节的 forwarding/elision 配合。

### 7.4 适用场景

- `N x VL -> N x scalar -> N x VL` 的逐行 pipeline；
- RMSNorm、LayerNorm、row-wise normalize、row reduction + broadcast；
- 静态 full-row one-VL 计算。

不适用于不同 N、不同 row step、transpose、跨行依赖、dynamic/tail 或多 VL 行。

### 7.5 相关实现

- `862dbfb0a`：principal row domain；
- `5a830f372`：multi-stage one-VL row-loop fusion。

## 8. 优化六：row-pipeline forwarding 与 invariant hoist

### 8.1 问题

即使 9 个 row loop 已合并，如果每个 TileOp 仍把结果写入 UB、下一个 TileOp 再
加载，最终 VF 仍包含大量 VLD/VST 和内部 membar。gamma 若在每行重复转换，也会
产生不必要的固定开销。

### 8.2 设计

- x、square、sum、mean、mean+epsilon、rms、normalized、scaled 通过 SSA/vreg
  直接传递；
- 删除完全匹配的 producer VST / consumer VLD；
- gamma BF16 在 preheader 加载、转换一次，以只读 vreg 进入 row loop；
- dead `alloc_tile` 仅作为地址元数据，不视为真实 UB 访问；
- 仅在 exact byte range、offset、dtype、mask、single-use、alias、escape 和唯一
  pipeline 均可证明时转换；证明失败完整保留 C 路径。

### 8.3 效果

RMSNorm `[64,64]` 的最终物理结构：

| 分支 | loop | VLD | VST | VF 内部 membar |
| --- | ---: | ---: | ---: | ---: |
| 优化前分支 D | 8 | 11 | 10 | 10 |
| 性能优化分支 D | 1 | 2 | 1 | 0 |

静态只保留 gamma/x 输入 VLD 和 y 输出 VST；所有计算阶段中间 UB 流量归零。这是
该用例 Vector 9.07x 提升的主要直接原因。

### 8.4 适用场景

- 已经融合为单一 principal loop 的 producer/consumer chain；
- 中间 tile 只用于相邻计算，没有外部可观察 use；
- 存在只读、loop-invariant 的 gamma/scale/bias vector。

### 8.5 相关实现

- `6cdb5990d`：row forwarding、scalar broadcast rewrite、gamma hoist。

## 9. 优化七：alias-aware membar 精化

### 9.1 问题

VMIToVPTO 后的 VLD/VST 使用指针和 affine offset 表达地址。若分析无法证明两个
访问不重叠，编译器必须保守插入 vecscope membar；仅比较表面 pointer SSA 值会
产生大量假冲突。

### 9.2 设计

- 追踪 normalized storage root、整数指针和 affine offset；
- 计算访问 byte range、dtype 宽度和 mask 覆盖；
- 证明 disjoint 后删除内部 membar；
- unpack/pack、未知动态地址、未知 mask 或无法归一化的 pointer 保守保留 barrier；
- GM↔UB pipeline 同步独立统计，不作为 VF 内部 barrier 删除。

### 9.3 效果

该分析使 RMSNorm `[64,64]` 最终达到 0 个 VF 内部 membar，并避免 forwarding 后
残留的假依赖重新串行化 vector pipeline。4096 宽用例当前仍有 6 个 membar，说明
它只删除有证明的 barrier，不以激进假设换取结构数字。

### 9.4 适用场景

- 同一 UB root 上具有静态 affine offset 的 chunk/row 访问；
- load/store elision 后需要重新判断残余物理访问关系的 vecscope；
- ping/pong 或 disjoint slice 能通过 byte range 证明的 pipeline。

### 9.5 相关实现

- `39dbb0666`：vecscope memory-barrier pass；
- `eb423c621`：affine-disjoint vector traffic 证明。

## 10. 优化八：Softmax column phase-state forwarding

### 10.1 问题

Softmax Dn 的数据流为：

```text
x[M,N]
  -> max(axis=M)[1,N]
  -> exp(x-max)[M,N]
  -> sum(axis=M)[1,N]
  -> exp/sum[M,N]
```

max 和 sum 是跨 M loop 保持的 N-lane accumulator。普通 lowering 会在 max、exp、
sum、divide phase 之间把状态写回 UB，再重新加载。

### 10.2 设计

- 为 `tcolmax`、`tcolexpandexpdif`、`tcolsum`、`tcolexpanddiv` 提供受限 VMI
  pipeline；
- 保留 max/sum accumulator 的稳定 phase provenance；
- loop fusion 后把完成的 max 和 sum vreg 直接转发到下一 phase；
- 删除对应 UB state round-trip；
- 保留 exp matrix 物化，因为 denominator 只能在 sum loop 完成后获得，最终 divide
  必须重新读取 exp；
- 保留与该 VST→VLD 真依赖对应的 barrier。

### 10.3 效果

同设备横向比较：

| Softmax shape | Vector：优化前分支 D | Vector：性能优化分支 D | Vector 变化 | Task 变化 |
| --- | ---: | ---: | ---: | ---: |
| `[4,16,32]` | 1.578 us | 1.848 us | 回退 17.11% | 回退 6.32% |
| `[4,16,64]` | 0.636 us | 0.410 us | 提升 35.53% / 1.55x | 提升 1.62% |
| `[4,16,128]` | 0.780 us | 0.668 us | 提升 14.36% / 1.17x | 提升 0.82% |

N=64/128 的 vector 阶段改善明确，但 vector 在总 task 中占比较小，所以端到端
收益有限。N=32 仍回退，不能列为已完成的性能优化场景。

### 10.4 适用场景

- 沿 M 维 reduction、每个 N lane 独立累加的 Softmax/归约子图；
- max/sum phase state 具有唯一 storage root 和明确 consumer；
- N 为 one-VL 或可证明的固定 multi-chunk 宽度。

当前不适用于自动把多个独立短行打包进一个 VL，也不会自动把四个独立 tile 的
循环交错为一个手写 VF 风格的 pipeline。Base32 需要后续增加 short-row packing
和 cross-independent-tile loop coalescing。

### 10.5 相关实现

- `d9bb892e6`：Base32/Base64/Base128 column candidates；
- `b3c5e6d35`：Softmax max/sum phase-state forwarding；
- `957f00670`：三宽度 A5 六路径 fixture/harness。

## 11. 场景适用矩阵

| 优化 | 4096 宽 RMSNorm | `N x 64` RMSNorm | Softmax Dn | RoPE/普通 elementwise |
| --- | --- | --- | --- | --- |
| selected-VMI-only region | 是 | 是 | 是 | 可复用 |
| one-VL candidate | chunk 内 | 每行 | N 方向 | 满足 shape 时可复用 |
| accumulator promotion | 核心 | 不需要跨 chunk | max/sum 采用独立 phase forwarding | 通常不适用 |
| scalar phase promotion | 核心 | 行内 scalar chain | 非主要路径 | 通常不适用 |
| principal row domain | 外层行不是本轮重点 | 核心 | 当前不是主要规划域 | row-preserving 时可复用 |
| row-loop fusion | 有限 | 核心 | 部分 loop | 同域 elementwise 可复用 |
| load/store forwarding | 核心 | 核心 | max/sum state | 同地址 producer/consumer 可复用 |
| invariant hoist | gamma 可继续增强 | gamma 核心 | 常量/只读 side input 可扩展 | sin/cos 等需独立证明 |
| affine membar 精化 | 部分生效 | 核心 | 部分生效 | 静态地址时可复用 |

## 12. 正确性和保守回退契约

所有 promotion/elision 必须保持：

- shape、dtype、layout 和 valid mask；
- reduction 次数与 FP32 association；
- narrowing rounding，例如 FP32→BF16 RINT；
- alias 和可观察存储语义；
- DMA/sync/unknown-call 边界；
- VMI candidate 在关闭 fusion 时可独立正确运行。

无法证明以下任一项时必须完整保留非优化路径：

```text
candidate legality
principal iteration relation
exact byte range and mask
single-use / escape
alias freedom
phase provenance
loop-invariant side input
```

禁止只删除部分 store、只改写部分 iter_arg 或在结果可观察时静默丢弃 UB
materialization。

## 13. 验证结果摘要

当前同设备、同输入的横向结论：

- RMSNorm `[64,64]`：Vector 提升 88.98%，Task 提升 44.07%；
- RMSNorm `[8,4096]`：Vector 提升 33.90%，Task 提升 27.72%；
- Softmax N=64：Vector 提升 35.53%；
- Softmax N=128：Vector 提升 14.36%；
- Softmax N=32：仍回退，是明确待办而非已验收收益。

验证同时检查 independent golden、AC-U、AC-F、PTO A/B/C/D、完整 lowering dump、
loop/VLD/VST/membar 统计以及每次 profile 后的输出。

## 14. 后续工作

1. 为实际单行 `[1,4096]` 重新构建六路径 workload；当前性能表是 `[8,4096]`，
   其中每行使用 `1x4096` VF body，不能简单除以 8 代替单行延迟。
2. Softmax Base32 增加 short-row full-VL packing。
3. Softmax 四个独立 tile 增加 cross-tile loop coalescing/interleave。
4. 继续分析 4096 宽 RMSNorm 残留 6 个 membar，并以 alias/mask 证明为前提清理。
5. 将 principal domain 从静态 one-VL row 推广到受限 multi-VL、dynamic/tail，
   但不得弱化 selected-VMI-only 和 conservative fallback 契约。

## 15. 共享与 cherry-pick

共享分支：

```text
https://github.com/TaoTao-real/PTOAS/tree/codex/rmsnorm-softmax-vmi-vf-performance-delta
```

RMSNorm 完整链：

```bash
git cherry-pick c3bc02cba^..e13e1021a
```

Softmax：

```bash
git cherry-pick d9bb892e6 b3c5e6d35 957f00670
```

目标分支若与当前实现历史已分叉，应按 candidate contract、RMSNorm cross-phase、
principal row pipeline、Softmax phase forwarding 的顺序移植，并在每组完成后重新
运行对应 A/B/C/D correctness 和 performance matrix。
