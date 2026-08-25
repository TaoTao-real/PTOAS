# RMSNorm / Softmax VMI VF 性能优化增量汇总

- 状态：Implemented / A5 validated
- 汇总分支：`codex/rmsnorm-softmax-vmi-vf-performance-delta`
- 日期：2026-08-25
- 对外对比名称：`优化前分支`、`性能优化分支`

## 1. 结论

本分支把已经完成并通过 A5 验证的 RMSNorm 和 Softmax VMI VF 工作整理为一条
可独立看护、可 cherry-pick 的提交链。它不包含 RoPE 跟踪提交，也不改写已验证的
RMSNorm 历史提交。

性能提升并非来自“把 TileOp 换一种方式展开”这一件事，而来自完整的优化链：

```text
前置选择合法 VMI candidate
  -> 只用 selected VMI TileOp 规划 FusionRegion
  -> 证明共同的 chunk/row/column 物理迭代域
  -> 融合低层循环
  -> accumulator、scalar 和 phase state 保持在 vreg/SSA
  -> 删除中间 UB VLD/VST、提升只读 gamma
  -> 用地址与 byte-range 证明精化内部 membar
```

同设备直接横向结果中，性能优化分支的最终 VF 路径在 RMSNorm `[64,64]` 上相对
优化前分支将 Vector 时间降低 88.98%，在 RMSNorm `[8,4096]` 上降低 33.90%；
Softmax `[4,16,64]` 和 `[4,16,128]` 分别降低 35.53% 和 14.36%。Softmax
`[4,16,32]` 回退 17.11%，当前不作为已验收收益场景。

## 2. 分支和证据口径

- 优化前分支：内部复现 SHA `6b4809b06d6893915b845d8ce14498af7247f197`；
  对外表格统一称为“优化前分支”。
- 性能优化源码验证 SHA：`149405c569b068daa738fc5e865066d1c9a7dbcd`。
- 本汇总分支将 Softmax 三个提交无语义改动地移植为
  `d9bb892e6`、`b3c5e6d35`、`957f00670`；此前 RMSNorm 已验证提交保持原 SHA。
- 设备和环境：同一 A5 device 2、CANN 9.2、固定输入；每档 1 次 warmup、5 次
  串行 profile，profile 后立即校验输出。
- 正确性：RMSNorm 使用 BF16 byte-exact；Softmax exact-onehot 使用 byte-exact，
  finite-sensitive 使用预先固定的独立 golden 容差。

完整 dump、binary、assembly 和 profiler 原始数据只保存在 immutable private lab，
不提交源码仓。

## 3. 本分支新增提交

### 3.1 Candidate 与 FusionRegion 契约

| Commit | 内容 | 性能作用 |
| --- | --- | --- |
| `c3bc02cba` | 定义 selected-candidate 与跨 phase 契约 | 建立保守优化边界 |
| `489e8753e` | candidate selection 前置 | FusionPlan 使用最终选择结果 |
| `cbd2161ca` | selected-VMI-only FusionRegion | 阻止 PTODSL/DMA 污染 VF region |

### 3.2 RMSNorm one-VL candidate 与跨 phase 优化

| Commit | 内容 |
| --- | --- |
| `e67fa1214` | 将合法 `tsqrt` 纳入 scalar fusion chain |
| `282b66f43` | one-VL BF16→FP32 conversion |
| `d72914f30` | one-VL accumulator `texpands` 初始化 |
| `6e8a17734` | 受限 FP32 `trowexpanddiv` candidate |
| `cabf6d140` | 规划跨 chunk accumulator phase |
| `6b14104e9` | accumulator vreg promotion |
| `037fd1818` | 保持 compact VL1 mask 语义 |
| `c8ab401e3` | 规划 reduction 后 scalar phase |
| `bcee73e70` | scalar/divisor vreg promotion |
| `7d45afb0b` | 删除被覆盖的 VMI tail store |
| `39dbb0666` | 新增 vecscope memory-barrier pass |
| `eb423c621` | 证明 affine-disjoint vector traffic |

### 3.3 RMSNorm `N×64` row pipeline

| Commit | 内容 |
| --- | --- |
| `3e2265d4b` | padded one-VL row reduction |
| `ffa3aba0e` | `N×64 / N×1` trowexpanddiv |
| `862dbfb0a` | principal row domain 规划 |
| `5a830f372` | 多阶段 one-VL row-loop fusion |
| `6cdb5990d` | row value forwarding、gamma hoist、UB 流量消除 |

### 3.4 Softmax column pipeline

| Commit | 内容 |
| --- | --- |
| `d9bb892e6` | Base32/Base64/Base128 column candidate |
| `b3c5e6d35` | max/sum 跨 phase state forwarding |
| `957f00670` | 三宽度 A5 六路径 fixture 与性能矩阵 |

### 3.5 测试、验收和设计说明

相关提交还包括 `d16ebd3b6`、`fe6fdf1f9`、`21dcfb97a`、`0e57d1572`、
`e13e1021a` 和 `075822f56`，覆盖 aligned RMSNorm fixture、残余流量分类、
`N×256B` fixture、A5 harness、验收结果和关键优化设计。

## 4. 新增和增强的优化 Pass

### 4.1 相对优化前分支新增的独立 Pass

| Pass | 主要职责 | 适用场景 |
| --- | --- | --- |
| `pto-select-tilelib-candidate` | FusionPlan 前唯一选择 VMI/PTODSL candidate | 所有 TileLib/VMI 图；主要是正确性和优化前提 |
| `pto-plan-vmi-accumulator-phases` | 证明 init→chunk update→final reduction 的 provenance | 长行 RMSNorm/LayerNorm one-VL lane accumulator |
| `pto-vmi-accumulator-promotion` | 用 loop-carried vreg 替代 accumulator UB 往返 | 固定 chunk、完整 mask、无 alias/escape 的 accumulator |
| `pto-plan-vmi-scalar-phases` | 证明 reduction 后 scalar chain 与 broadcast divisor | RMSNorm/variance 类 scalar epilogue |
| `pto-vmi-scalar-promotion` | scalar chain 用 SSA/vreg 直传并只广播一次 | VL1 compact state→full-VL apply loop |

### 4.2 增强的既有阶段

| 阶段/Pass | 增强内容 | 直接效果 |
| --- | --- | --- |
| FusionAnalysis / FusionPlan | selected-VMI-only；principal row domain；column phase provenance | 合法跨 `N×64→N×1→N×64` shape 规划一个 region |
| `pto-low-level-loop-fusion` | 合并已证明共享 header 的 row/chunk/column loop | RMSNorm `[64,64]` 主 row loop 从 8 个降到 1 个 |
| `pto-fusion-load-store-elision` | exact range/mask 的 producer→consumer forwarding；dead storage 清理；invariant hoist | 中间 UB VLD/VST 归零或显著减少 |
| VMIToVPTO / vecscope inference | 保持 phase vreg、compact mask、广播与布局语义 | 让高层 promotion 真正落到 VPTO/汇编 |
| `pto-insert-vecscope-mem-bar` / hazard analysis | 优化前已有早期框架；本分支补充 normalized root、affine offset、byte range disjoint 证明 | 删除假冲突 membar，保留真实 hazard |

这些 Pass 的边界是保守的：dynamic address、tail、mask 不一致、alias、escape、
unknown call、sync 或多个不唯一 pipeline 任一无法证明时，完整保留未优化路径。

## 5. 实际性能效果

### 5.1 VF Fusion 的增量收益：D 相对 B

B 表示“VMI candidate，仅展开，不开 region/loop/elision”；D 表示完整的
“VMI VF Fusion”。正数为时间降低，负数为回退。

| 用例 | 优化前分支 Task D/B | 性能优化分支 Task D/B | 优化前分支 Vector D/B | 性能优化分支 Vector D/B |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm `[64,64]` | -0.32% | **49.83% / 1.99×** | 0.21% | **91.45% / 11.70×** |
| RMSNorm `[8,4096]` | -0.67% | **27.14% / 1.37×** | -1.26% | **33.07% / 1.49×** |
| Softmax `[4,16,32]` | 3.46% | 4.55% | -0.19% | 10.81% / 1.12× |
| Softmax `[4,16,64]` | 3.21% | 4.59% | 2.15% | **30.39% / 1.44×** |
| Softmax `[4,16,128]` | 1.15% | 1.65% | 2.01% | **11.99% / 1.14×** |

该表说明 RMSNorm 的收益确实来自新增的 fusion/promotion/elision，而不是 candidate
expand：优化前分支的 B→D 基本没有收益，性能优化分支的 B→D 才出现大幅下降。

### 5.2 最终 VF 路径 D 的分支间直接对比

| 用例 | 优化前 Task | 优化后 Task | Task 改善 | 优化前 Vector | 优化后 Vector | Vector 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSNorm `[64,64]` | 9.731 us | 5.443 us | **44.07% / 1.79×** | 4.700 us | 0.518 us | **88.98% / 9.07×** |
| RMSNorm `[8,4096]` | 73.296 us | 52.982 us | **27.72% / 1.38×** | 60.471 us | 39.971 us | **33.90% / 1.51×** |
| Softmax `[4,16,32]` | 6.092 us | 6.477 us | **回退 6.32%** | 1.578 us | 1.848 us | **回退 17.11%** |
| Softmax `[4,16,64]` | 5.785 us | 5.691 us | 1.62% / 1.02× | 0.636 us | 0.410 us | **35.53% / 1.55×** |
| Softmax `[4,16,128]` | 6.724 us | 6.669 us | 0.82% / 1.01× | 0.780 us | 0.668 us | **14.36% / 1.17×** |

五个用例中四个最终 D 更快。Softmax N=32 因尚未支持短行 full-VL packing 和
多个独立 tile 的 loop coalescing 而回退，必须作为后续优化项保留。

### 5.3 结构变化与性能归因

| 用例/分支 D | loop | VLD | VST | VF 内部 membar |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm `[64,64]` 优化前 | 8 | 11 | 10 | 10 |
| RMSNorm `[64,64]` 性能优化后 | **1** | **2** | **1** | **0** |
| RMSNorm `[8,4096]` 优化前 | 3 | 13 | 13 | 4 |
| RMSNorm `[8,4096]` 性能优化后 | 3 | **8** | **6** | 6 |

`[64,64]` 的 9.07× Vector 改善可以直接由一个主 row loop、计算中间 UB 流量
归零和内部 membar 归零解释。`[8,4096]` 的主要收益来自 accumulator/scalar
跨 phase 保持和 VLD/VST 减少；其 membar 尚未完全清理，因此仍有继续优化空间。

Softmax N=64/N=128 的 Vector 改善来自 max/sum phase-state forwarding 和循环
数量下降，但 Vector 在总 Task 中占比较小，所以端到端收益只有 1.62%/0.82%。
这不是采样矛盾，而是非 Vector 固定开销占主导后的 Amdahl 限制。

## 6. 应用场景与泛化边界

- RMSNorm `[8,4096]`：每个 row body 是 `1×4096`，64 个 one-VL chunk 跨循环
  保持 FP32 lane accumulator，循环后只 reduction 一次。
- RMSNorm `[64,64]`：典型 `N×1VL` row pipeline，通过 principal row domain
  融合 reduction 前后不同逻辑 shape。
- Softmax `[4,16,N]`：沿 M 维维护 N-lane max/sum accumulator，并跨 phase
  转发状态；当前验证 N=32/64/128。

当前没有宣称支持 dynamic/tail、任意 multi-VL row、任意 reduction、自动短行
packing 或四个独立 tile 的通用交错调度。现有 Pass 使用可复用的证明框架，但
pattern 和合法性范围仍是逐步扩展的；不能把已验证的 RMSNorm/Softmax 子集等同于
完整图级自动向量调度。

## 7. 获取与 cherry-pick

共享分支：

```text
https://github.com/TaoTao-real/PTOAS/tree/codex/rmsnorm-softmax-vmi-vf-performance-delta
```

完整 RMSNorm 链：

```bash
git cherry-pick c3bc02cba^..e13e1021a
```

Softmax 增量：

```bash
git cherry-pick d9bb892e6 b3c5e6d35 957f00670
```

设计文档可再 cherry-pick 当前分支末尾的文档提交。目标分支若已经覆盖其中部分
提交，应按 candidate contract、RMSNorm cross-phase、principal row pipeline、
Softmax phase forwarding 的顺序移植，并在每组完成后重跑对应 correctness 和
performance matrix。
