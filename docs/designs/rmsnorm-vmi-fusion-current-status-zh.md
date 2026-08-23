# RMSNorm VMI VF Fusion 现状、收益归因与泛化边界

## 文档目的

本文是 RMSNorm VMI VF Fusion 实验的现状分析，不改变编译器行为。它在已有
`1 x 4096` 和 `N x 256B` A5 acceptance 的基础上回答三个问题：

1. 当前实现已经验证了什么；
2. C（Region + loop fusion）到 D（promotion + elision）的主要收益来自哪里；
3. 哪些能力已经可以复用，哪些结果仍依赖 PTO 用例手工准备好的计算框架。

原始正确性、lowering 和性能数据分别记录在：

- [RMSNorm 1VL A5 acceptance](rmsnorm-a5-vmi-fusion-acceptance.md)；
- [RMSNorm `N x 256B` A5 acceptance](rmsnorm-nx256b-a5-vmi-fusion-acceptance.md)。

## 已验证的两个计算形态

### `1 x 4096`：单 token、宽 feature 维

输入的一行表示一个 token 的 4096 个 BF16 特征值。PTO 用例显式将 feature
维切成 64 个 `1 x 64xf32` chunk，每个 chunk 正好是一个 256B FP32 VL：

```text
reduction phase, 64 chunks:
  BF16 x -> FP32 -> square -> lane-wise FP32 accumulator

one final reduction:
  trowsum -> tmuls -> tadds -> tsqrt

apply phase, 64 chunks:
  BF16 x -> FP32
  BF16 gamma -> FP32
  divide -> multiply -> RINT BF16
```

这一形态验证了：在两个显式 phase 之间，PTOAS 可以将 one-VL accumulator 和
compact scalar chain 从 UB materialization 提升为跨循环/跨 region 的 vreg
数据流。

### `N x 256B`：多个 token、每行一个 VL

输入为 BF16 `[N,64]`，每一行表示一个 token 的 64 个特征值。BF16 转为 FP32
后，每行 `64 x 4B = 256B`，恰好是一个 VL。主验收规模为 `N=64`：

```text
for row in 0..N:
  BF16 x[row] -> FP32
  square -> trowsum -> mean -> epsilon -> sqrt
  divide -> gamma multiply -> RINT BF16 y[row]
```

这一形态验证了：PTOAS 可以证明 `N x 64 -> N x 1 -> N x 64` 具有相同的
principal row domain，将多段 row loop 合并为一个 loop，并在行内转发中间
vreg，同时将只读 gamma 提升到 preheader。

两个用例都遵守 RMSNorm 对每个 token 独立归一化的算法语义。区别是：
`1 x 4096` 检验宽 feature 维的 chunk/phase 处理，`N x 256B` 检验多 token 的
逐行 one-VL pipeline。后者不是一次跨 token 做 reduction；只是将多个独立行
放在同一个 row loop 中执行。

## C 到 D 为什么有显著收益

### `1 x 4096` 的结构和性能

| 路径 | 能力 | 静态 VLD | 静态 VST | 内部 `mem_bar` | Vector median |
| --- | --- | ---: | ---: | ---: | ---: |
| C | selected VMI + Region + loop fusion | 15 | 13 | 14 | 59.720 us |
| D | C + accumulator/scalar promotion + elision | 3 | 1 | 0 | 2.325 us |

Vector 时间从 `59.720 us` 降到 `2.325 us`，降低约 `96.1%`，约为
`25.7x`。原 acceptance 中 `D/C = 0.210114` 是 task median
`72.294 us -> 15.190 us` 的比值，不能与 vector 比值混用。

### `N=64` 的结构和性能

| 路径 | Row loop | 静态 VLD | 静态 VST | 内部 `mem_bar` | Vector median |
| --- | ---: | ---: | ---: | ---: | ---: |
| C | 1 | 12 | 10 | 10 | 151.130 us |
| D | 1 | 2 | 1 | 0 | 8.470 us |

这里 C 已经把九个 row loop 合成一个，但 vector 时间反而高于未融合的 B。
它说明 Region 数量和 loop 数量不能单独作为 VF 收益指标。

### C 只合并控制流，D 才改变数据流

每个 VMI candidate 必须能够独立正确执行，因此 candidate lowering 默认会将
输出写入 UB，后继 candidate 再从 UB 加载：

```text
producer -> VST temporary UB -> mem_bar -> VLD temporary UB -> consumer
```

C 将 candidate 放入同一个 Region 或 loop，但仍保留上述通信。D 在证明地址、
offset、byte range、mask、dtype、迭代关系和无逃逸之后，将它改成：

```text
producer vreg -> consumer
```

显著收益来自以下组合，而不是单纯的 Region 规划：

1. **Accumulator promotion**：将 reduction chunk loop 中每轮的 accumulator
   VLD/VST 改为 `scf.for` loop-carried vreg，并让最终 `trowsum` 直接消费 loop
   result。对 `1 x 4096`，仅这一 accumulator 原路径每行就产生约 32.5KiB 的
   UB 往返。
2. **Scalar promotion**：将 `trowsum -> tmuls -> tadds -> tsqrt` 的 compact
   scalar 中间值用 SSA/vreg 直传，只在 apply phase 前做一次寄存器广播。
3. **Region-local load/store forwarding**：将完全匹配的 `VST -> VLD` 配对替换
   为 producer SSA value，并在没有可观察使用时删除 dead store。
4. **Row-pipeline forwarding and invariant hoisting**：在 `N x 256B` 路径中，
   x、square、sum、rms、normalized 和 scaled 在同一行迭代中直传；gamma
   load/convert 只在 preheader 执行一次。
5. **Predicate/barrier cleanup**：当物理 UB 依赖已经被 vreg 数据流替代，删除
   不再需要的 VF 内部 predicate 和 `mem_bar`。GM/UB DMA 边界同步仍然保留，
   不计入 VF 内部消除结果。

被删除的静态指令位于 64 次 chunk loop 或 64 次 row loop 内时，动态代价会被
迭代次数放大；`mem_bar` 还会阻断 vector pipeline。因此算术量没有明显变化，
vector 时间仍可能出现数量级下降。

## 哪些能力具有复用价值

以下思想不专属于 RMSNorm，当前实现也已经具备一定的通用性：

- selected-VMI-only FusionRegion invariant；
- 相同 principal iteration domain 的保守 loop fusion；
- Region 内同地址、同 offset、同 mask、同 dtype 的 VST/VLD forwarding；
- 无外部可观察使用的 overwritten/dead store 删除；
- 无 alias、escape、sync 或 unknown effect 时的只读 invariant forwarding；
- 物理 byte range 不相交时的冗余 vector-scope barrier 删除；
- 证明失败时完整保留未优化路径，不做部分危险改写。

这些能力适合 elementwise producer/consumer、同迭代 row pipeline，以及短生命周期
scalar/vector 中间值。它们仍需要后续 workload 验证，不能只凭 RMSNorm 结果宣称
对所有 TileOp 计算图都有效。

## 当前结果依赖的 RMSNorm 特化条件

### PTO 输入已经手工完成的工作

`1 x 4096` 用例不是裸的整行 TileOp 描述。PTO 源码已经显式完成：

- 将 4096 feature 拆成 64 个 FP32 one-VL chunk；
- 将计算拆成 reduction 和 apply 两个 phase；
- 选择 64-lane lane-wise accumulator，并在循环结束后只做一次 `trowsum`；
- apply phase 重新加载 BF16 x 并重新执行 BF16-to-FP32 conversion，以计算换取
  不物化完整 FP32 row，即手工 rematerialization；
- 为 accumulator、scalar scratch 和输出指定可证明的静态 UB root；
- 使用固定 step、完整 mask、无 tail 的循环结构。

`N x 256B` 用例也已经把每行限定为一个完整 FP32 VL，并显式提供 row-preserving
的 `N x 64 -> N x 1 -> N x 64` 算法框架。

因此当前 PTOAS 的主要贡献是：在已经准备好的 chunk/phase/row 框架内进行
candidate selection、Region/loop planning、vreg promotion 和物理流量消除。当前
实现还不能从任意裸的整 tile RMSNorm 自动发现所有这些算法改写。

### 当前 promotion pattern 的窄约束

- accumulator promotion 只识别静态 one-VL `1 x 64xf32` sum accumulator；
- scalar promotion 识别固定的
  `trowsum -> tmuls -> tadds -> default tsqrt -> broadcast -> apply` 链；
- row forwarding 要求静态、完整的一 VL 行和唯一 principal row pipeline；
- compact scalar broadcast 要求单 lane store、完整 `BRC_B32` load 和唯一使用；
- gamma hoist 要求唯一静态 UB root、唯一 VST/VLD pair 和只读 loop-invariant use；
- 地址、mask、loop step、use count 或 alias 关系无法证明时，优化必须拒绝。

## 在其他计算流或 tiling 下可能收益很小

| 场景 | 预期限制 |
| --- | --- |
| 单 chunk 或很短的 loop | UB 往返没有足够的动态放大，promotion 收益有限 |
| dynamic shape、tail 或 partial mask | 现有 exact-range/mask 证明通常拒绝 |
| 多 VL accumulator 或其他 reduction | 当前 RMSNorm phase pattern 无法直接匹配 |
| 多 consumer 或可观察临时 tensor | 必须保留 materialization，不能删除最终 store |
| 不同 loop domain 或 dtype rate | candidate loop 不能直接按相同 iteration 合并 |
| alias、escape、sync、unknown call | effect 无法证明，forwarding 和 barrier elision 失效 |
| PTODSL fallback 位于主链 | selected-VMI-only Region 被切断 |
| Softmax 等完整中间 tile 跨 phase 存活 | 寄存器容量不足时仍需 UB，或需要额外 rematerialization 策略 |
| exp/div/sqrt 等计算主导的链 | UB 流量不再是主要瓶颈，结构改善不一定转化为同等性能收益 |
| 融合后寄存器压力过高 | 可能产生 spill 或调度退化，性能可能下降 |
| GM/UB `tload`、`tstore` 边界 | 属于 DMA hard boundary，不应由 VF fusion 删除 |

## 从裸 TileOp 实现走向自动优化仍缺少的能力

要让用户只写整 tile 的 RMSNorm/Softmax 等算法，而不手写 chunk loop、phase 和
rematerialization，PTOAS 还需要逐步补齐：

1. **内部物理 chunk 规划**：保持上层 logical tile shape 不变，根据 dtype、
   下游共同计算量子和 mask 选择物理 chunk，而不是让每个 candidate 独立选择
   不兼容的循环域。
2. **统一 iteration-domain 建模**：显式表示 logical row、physical byte range、
   dtype expansion ratio 和 tail，支持不同 candidate loop 的对齐、split 或合并。
3. **自动 phase 识别**：从 reduction 依赖和最终值可用时刻识别必须分开的 phase，
   并区分普通同域 fusion 与跨 phase state promotion。
4. **成本驱动 rematerialization**：比较重新 VLD/convert/compute 与保存完整中间
   tensor 的 UB 容量、流量、barrier 和寄存器压力，不能仅依赖固定 RMSNorm pattern。
5. **multi-consumer fusion**：对同一 producer 的多个消费者选择 clone、共同融合、
   rematerialize 或保留 materialization，避免只支持单 use 链。
6. **通用 reduction/scalar DAG promotion**：从固定 RMSNorm op 序列扩展为带有
   association、rounding 和 escape 证明的 reduction state/compact value 模型。
7. **资源与压力模型**：在 Region planning、loop fusion 和 load/store elision 前后
   估算 vreg/preg、spill 和 code size，避免“消 UB、增 spill”。
8. **dynamic/tail legality**：支持运行时 valid shape 和 partial mask，同时保持
   candidate 独立正确与 fusion 前后 byte/rounding 语义一致。

其中“内部物理 chunk 规划”不等于修改上层语言决定的公共 tile shape。PTOAS 可以
保留 logical tile 合同，只在 candidate lowering 和 fusion scheduling 内选择物理
执行分块；是否允许这一内部变换仍必须由 layout、mask、rounding 和依赖证明约束。

## 现有证据能够支持和不能支持的结论

现有 A5 数据可以支持：

- 两个固定 RMSNorm workload 均保持独立 golden 和所有实现路径 BF16 byte-exact；
- C 只减少 Region/loop 数量时没有可靠收益；
- D 的 promotion/elision bundle 与中间 UB traffic、VF 内部 `mem_bar` 消失以及
  vector 时间显著下降一致；
- 对当前固定 workload，生成的 D 路径已经接近或达到手写 VF 的 vector 量级。

现有数据不能支持：

- 将 D 的收益外推到 arbitrary shape、dynamic/tail、多 VL reduction 或任意算子图；
- 精确声称 accumulator promotion、scalar promotion、load/store elision 各自贡献
  多少微秒；
- 声称 PTOAS 已能从裸整 tile RMSNorm 自动生成当前手写 chunk/phase/rematerialized
  数据流；
- 仅根据 Region 数、loop 数或静态 VLD/VST 数宣称性能验收。

若要进一步细分 C 到 D 的收益，应增加相同 source/candidate set 下的 ablation：

```text
C0: Region + loop fusion
C1: C0 + accumulator promotion
C2: C1 + scalar promotion
C3: C2 + Region-local load/store elision
C4: C3 + post-flatten invariant/predicate/barrier cleanup (current D)
```

每档都应检查 byte-exact correctness，并同时记录动态 VLD/VST、内部 `mem_bar`、
spill、vector median 和离散度。只有这种拆分采样才能把当前 D 的整体收益继续归因
到单个优化阶段。

## 当前结论

RMSNorm 实验已经证明：仅有 VMI candidate、FusionRegion 和 loop fusion 不足以
获得 VF 性能；真正关键的是在严格证明下，将跨 TileOp、跨循环甚至跨 phase 的
中间值保持为 vreg，并删除由 independently-correct candidate 引入的 UB 往返和
同步。

同时，当前优异结果仍部分依赖 PTO 用例手工准备的 one-VL chunk、phase 划分和
rematerialization。下一阶段的核心不是继续为固定 RMSNorm 序列增加更多特判，而是
将 physical chunk、iteration domain、phase、rematerialization、multi-consumer 和
resource pressure 提升为可组合、可成本评估且保守失败的统一优化模型。
