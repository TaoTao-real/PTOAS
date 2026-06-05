# PTOAS 自动同步内存一致性建模需求与设计说明

## 1. 背景

PTOAS 的自动同步当前主要围绕两类信息工作：

- op 所属的执行 pipe，例如 `PIPE_MTE2`、`PIPE_V`、`PIPE_MTE3`、`PIPE_S`。
- op 的 MemoryEffects 与 alias 关系，用于判断 RAW、WAR、WAW 数据依赖。

这套模型可以覆盖很多 tile op 的跨 pipe 数据依赖。例如 `TLOAD` 写入 UB 后由
vector op 读取，或者 vector op 写入 UB 后由 `TSTORE` 搬回 GM，PTOAS 可以根据
`MTE2 -> V`、`V -> MTE3` 的 pipe pair 生成 `set_flag/wait_flag`。

但是硬件的内存一致性要求并不只由 pipe 顺序决定。UB 和 GM 会被 AICore 的多个
组件访问，包括 scalar、SIMT/SIMD vector、MTE2、MTE3 等。尤其在 GM 访问中，
cacheable scalar/SIMT 访问、write-back/write-through 策略、DMA/MTE 访问和
跨 rank 通信信号之间存在额外的可见性约束。硬件资料中的 RAW/WAR/WAW consistency
表明确要求某些 hazard 除了 pipe sync 以外，还需要 `dcci`、`dsb`、`mem_bar`
或 pipe-local barrier。

近期 issue #711 和 #744 暴露了这个缺口：

- #711：`pto.comm.tnotify` 的 signal store 发生在内部 trailing
  `pipe_barrier(PIPE_ALL)` 之前。若 `TNotify` 之前还有 pending MTE2/MTE3 操作，
  remote `TWait` 可能先看到 signal，再读取尚未完成的数据。
- #744：#711 补了 MTE pipe drain 后，跨 rank 场景仍然失败。原因是
  `pipe_barrier(PIPE_MTE3)` 只能约束本核 MTE3 pipe 的执行完成，不能等价保证
  前序 GM/DDR store 已经在 remote/DDR visibility domain 中可见。因此
  `TNotify` 前如果存在前序 MTE3 store，还需要 `dsb(DSB_DDR)` 作为 release fence。

这说明 `pipe + alias -> set/wait/barrier` 不足以完整表达硬件 memory consistency
contract。PTOAS 需要引入显式的内存一致性建模能力。

## 2. 问题陈述

当前 InsertSync 的核心问题是：同步分析只知道“哪个 pipe 读写了哪个 buffer”，
但不知道“这个访问通过哪种内存一致性路径完成”。

具体表现为：

1. Cache policy 没有进入依赖分析。

   例如 scalar/SIMT GM 访问可能是 cacheable、non-cacheable、write-back 或
   write-through。硬件表格中同样的 RAW/WAR/WAW hazard 在不同 cache policy 下
   需要不同动作。当前 MemoryEffects 只表达 Read/Write，不表达 cache policy。

2. 同一个 pipe pair 不一定对应同一种同步动作。

   `sync V -> MTE2` 可以表达 vector producer 和 MTE2 consumer 的执行顺序，但如果
   vector producer 是 write-back GM store，还可能需要 `simt.dcci` 或对应 cache
   maintenance。只插 event sync 不能保证 reader 看到最新数据。

3. Alias 粒度不覆盖硬件 hazard 粒度。

   对 scalar cacheable store，硬件 WAW hazard 可能以 64B scalar data cache line
   为冲突单位。当前 alias 分析更接近 buffer/range overlap，不会自动扩展到
   cacheline-level conflict。

4. Publish/consume 语义不是普通 alias 依赖。

   `TNotify` 的 signal buffer 和 payload buffer 通常不 alias。普通
   `payload write -> signal write` 依赖无法通过 MemoryEffects alias 直接发现。
   但通信语义要求 payload 在 signal 发布前可见。类似地，`TWait` 返回后是否需要
   acquire fence，也应由通信语义建模，而不是依赖 payload 与 signal 的地址关系。

5. Macro op 内部多组件访问难以被单 pipe 描述。

   `TPut/TGet/TGather/TScatter/TBroadcast/TReduce` 等复杂 op 可能在内部经过 MTE2、
   MTE3、V、S 等多个 component phase。若只把 op 建模成一个 pipe，就可能漏掉内部
   component 与外部 op 之间的 consistency action。

## 3. 目标

本需求的目标是为 PTOAS 建立一套可扩展的自动内存一致性模型，使自动同步能够正确
处理硬件 RAW/WAR/WAW 表中要求的 pipe sync、cache maintenance 和 memory fence。

具体目标包括：

- 能区分 memory space：GM、UB 以及后续可能需要的 L1/L0/ACC 等空间。
- 能区分 access component：scalar、SIMT/SIMD vector、MTE2、MTE3、macro phase。
- 能区分 access kind：read、write、read-write、atomic、signal publish/consume。
- 能表达 cache policy：cacheable/non-cacheable、write-back/write-through、
  L1/L2 cache control 等。
- 能根据 hazard type：RAW、WAR、WAW 查表生成 action list。
- 能生成不止一种同步动作：`set_flag/wait_flag`、`pipe_barrier`、`mem_bar`、
  `dcci`、`dsb(DSB_DDR)` 等。
- 能将通信 op 的 publish/consume 语义独立建模，覆盖 payload 与 signal 不 alias 的场景。
- 能为 macro op 暴露内部 component phase，使外部同步分析可以看到真实访问路径。

## 4. 非目标

第一阶段不要求做到全局最优同步数量。

以下内容暂不作为首个实现阶段的目标：

- 精确消除所有由 consistency action 引入的冗余 fence。
- 完整处理所有硬件架构变体的所有特殊 cache policy。
- 对所有动态地址做完美 alias 证明。
- 替代现有 event id 分配和 RemoveRedundantSync 机制。
- 自动推断跨 rank 通信拓扑或 remote address ownership。

第一阶段应优先保证正确性，并让模型结构可扩展，后续再做精细化和性能优化。

## 5. 需求拆解

### 5.1 访问描述模型

需要新增一个统一的 memory consistency access descriptor，作为 MemoryEffects
之外的补充信息。建议字段包括：

| 字段 | 说明 |
| --- | --- |
| `memorySpace` | GM、UB、L1、L0、ACC 等 |
| `component` | scalar、simt/simd vector、MTE2、MTE3、macro phase |
| `accessKind` | read、write、read_write、atomic、signal_publish、signal_consume |
| `cachePolicy` | cacheable、non-cacheable、write-back、write-through、L1/L2 控制 |
| `pipe` | 现有 `PIPE_*`，用于 event/barrier 生成 |
| `value` | 参与 alias 的 SSA value 或 memory region |
| `granularity` | byte range、cacheline、whole buffer、unknown |

MemoryEffects 继续用于发现读写对象和 alias；access descriptor 用于说明“这次访问的
硬件一致性属性”。

### 5.2 Hazard 判定

对于两个可能 alias 的访问，先按 MemoryEffects 判定 hazard type：

- prior write + later read：RAW。
- prior read + later write：WAR。
- prior write + later write：WAW。

再结合两个 access descriptor 查询 consistency matrix，得到需要生成的动作序列。

对于 `signal_publish/consume` 这类通信语义，不要求 payload 与 signal alias。
通信 op 应显式声明：

- publish 前需要哪些 producer access 可见。
- consume 后哪些 consumer access 需要 acquire 保证。

### 5.3 Action 表达

需要新增同步动作抽象，不能只把结果表示成 set/wait pair。

建议动作类型包括：

| 动作 | 示例 | 用途 |
| --- | --- | --- |
| pipe event | `set_flag(PIPE_V, PIPE_MTE3)` / `wait_flag(...)` | 跨 pipe 执行顺序 |
| pipe barrier | `pipe_barrier(PIPE_MTE3)` | 同 pipe drain/order |
| memory barrier | `mem_bar(VST_VLD)` 等 | SIMD/UB 内存顺序 |
| cache maintenance | `dcci(...)` | cache line clean/invalidate |
| DDR fence | `dsb(DSB_DDR)` | GM/DDR visibility release/acquire |
| architecture fallback | `pipe_barrier(PIPE_ALL)` | 无法精确建模时的保守兜底 |

动作需要携带 insertion point、target value/range、以及是否可以被冗余删除的信息。

### 5.4 Matrix 驱动

硬件 RAW/WAR/WAW consistency 表应转换成 PTOAS 内部的 matrix。matrix 输入为：

```text
memorySpace
hazardType
priorAccess(component, accessKind, cachePolicy)
laterAccess(component, accessKind, cachePolicy)
```

matrix 输出为：

```text
ordered action list
```

例如：

```text
GM + RAW + prior MTE3 write + later scalar read
  -> pipe/order action if needed
  -> scalar cache action if required by later access policy

GM + publish + prior MTE3 write + TNotify
  -> pipe_barrier(PIPE_MTE3)
  -> dsb(DSB_DDR)
```

具体表项需要与硬件团队确认后固化，并按架构版本区分。

### 5.5 通信 publish/consume

`TNotify/TWait` 需要从普通 op pipe 模型中独立出来，作为通信同步语义处理。

Producer 侧：

```cpp
payload write
release actions
TNotify(signal)
```

已知需要覆盖：

- pending MTE2：`pipe_barrier(PIPE_MTE2)`。
- pending MTE3：`pipe_barrier(PIPE_MTE3); dsb(DSB_DDR)`。
- scalar/SIMT/SIMD GM write：需要按 cache policy 查询 matrix，决定是否补
  `dcci`、`dsb` 或其它 fence。

Consumer 侧：

```cpp
TWait(signal)
acquire actions
payload read
```

需要确认 `TWAIT_IMPL` 或硬件 wait 是否已经提供 acquire 语义。如果没有，需要为
`TWait -> GM read` 建模 acquire fence。

### 5.6 Macro op 建模

对于内部包含多个 component phase 的 op，应通过 Sync Macro Model 暴露内部访问：

```text
phase 0: MTE2 read/write staging
phase 1: V/S compute or index generation
phase 2: MTE3 writeback
hidden events / reserved event ids
```

每个 phase 都应带 access descriptor，而不只是 def/use。这样外部 op 与 macro 内部
phase 的 hazard 能查到正确 consistency action。

## 6. 对现有 PTOAS 的影响

### 6.1 IR / Op 定义

需要逐步为相关 op 补充一致性 metadata：

- scalar GM load/store/atomic。
- SIMT/SIMD GM load/store/atomic。
- MTE2/MTE3 tile load/store。
- `pto.comm.*` 通信 op。
- macro op：`TPut/TGet/TGather/TScatter/TBroadcast/TReduce` 等。

这些 metadata 可以先在 C++ helper 中推导，后续再视需要暴露成 op interface。

### 6.2 SyncIR 翻译

`PTOIRTranslator` 当前从 MemoryEffects 提取 def/use，并从 OpPipeInterface 提取 pipe。
后续需要额外提取 access descriptor，存入 `CompoundInstanceElement` 或并行 side table。

### 6.3 InsertSyncAnalysis

依赖判断不再只返回“需要 sync 或不需要 sync”，而应返回一个 action request：

```text
hazard
prior access
later access
required actions
```

event sync 只是 action list 的一种结果。

### 6.4 Move / RemoveRedundantSync

同步移动和冗余删除需要知道 action 类型：

- event pair 可以继续按 pipe pair 与 must-path 规则删除。
- `dsb/dcci/mem_bar` 不能随意按 pipe pair 删除。
- publish/consume fence 需要保持在 signal op 附近，不能被普通 alias sync 吞掉。

### 6.5 EventIdAllocation

event id 分配只处理 event action。`dsb/dcci/mem_bar/pipe_barrier` 不应占用 event id。
但 macro op 内部固定 event id 或 reserved event id 仍需保持现有隔离机制。

### 6.6 Codegen

`SyncCodegen` 或 EmitC lowering 需要支持输出多种 action：

- `set_flag/wait_flag`
- `pipe_barrier`
- `mem_bar`
- `dcci`
- `dsb(DSB_DDR)`

对于 `TNotify` 这类 op-local publish action，可以继续在 EmitC 阶段插入；长期建议将其
前移到统一 action model 中。

## 7. 分阶段实施计划

### 阶段 1：文档与需求确认

- 将硬件 RAW/WAR/WAW 表整理成 PTOAS 内部需求。
- 与硬件团队确认每个表项的动作语义、作用域和架构差异。
- 明确 `TWait` 是否自带 acquire 语义。

### 阶段 2：TNotify publish 补齐

- 已覆盖 MTE2/MTE3 pending drain。
- 已覆盖 MTE3 publish 前 `dsb(DSB_DDR)`。
- 继续补齐 scalar/SIMT/SIMD GM write -> TNotify 的 release action。

### 阶段 3：Access Descriptor 基础设施

- 为关键 op 建立 access descriptor。
- 在 SyncIR 翻译阶段携带 descriptor。
- 保持旧 event sync 逻辑兼容。

### 阶段 4：Matrix-driven action 生成

- 将 GM/UB RAW/WAR/WAW 表转换成内部 matrix。
- 先覆盖高风险路径：GM cacheable scalar/SIMT、MTE3、MTE2、V/S。
- 增加 lit 回归和板级验证用例。

### 阶段 5：优化与冗余删除

- 对 `dsb/dcci/mem_bar` 做 must-path 冗余删除。
- 对重复 publish fence 做合并。
- 将保守 fallback 逐步替换成精确 action。

## 8. 风险与取舍

- 正确性风险：如果 action 表项缺失或 op metadata 错误，仍可能漏同步。
- 性能风险：第一阶段保守插入 `dsb/dcci` 可能增加同步开销。
- 编译复杂度：同步分析从 pipe pair 扩展为 matrix-driven，会增加实现和调试复杂度。
- 维护成本：硬件表格变化时，需要同步更新 PTOAS matrix 和测试。

建议以正确性优先，先在高风险通信和 GM cacheable 场景补齐，再逐步优化冗余。

## 9. 验证需求

需要新增以下类型测试：

- `MTE3 store -> TNotify`：检查 `pipe_barrier(PIPE_MTE3); dsb(DSB_DDR)`。
- `MTE2 load -> TNotify`：检查不插 DDR fence。
- `scalar cacheable GM store -> scalar/SIMT/MTE read`：检查 `dcci/dsb`。
- `SIMT write-back GM store -> MTE2/scalar read`：检查 cache maintenance。
- `UB SIMD write -> SIMD read/write`：检查 `mem_bar`。
- scalar cacheline WAW：两个 store 地址不同但落在同一 64B cacheline 时仍触发 hazard。
- `TWait -> payload read`：根据 acquire 语义确认是否需要 fence。
- macro op 内部 phase 与外部 GM/UB access 的 consistency action。

板级验证应覆盖跨 rank notify/wait 场景，尤其是 payload 和 signal 不 alias 的 publish/consume。

## 10. 当前结论

PTOAS 现有自动同步模型解决了大部分“同一 buffer 上跨 pipe 的执行顺序”问题，但不能完整
覆盖硬件 memory consistency contract。后续需要把自动同步从单一的 event/barrier 模型
扩展为：

```text
alias/memory-effect analysis
  + access descriptor
  + RAW/WAR/WAW consistency matrix
  + communication publish/consume semantics
  -> ordered action list
```

#744 中 `TNotify` 前补 `dsb(DSB_DDR)` 是该模型的一个具体特例。完整方案需要进一步
覆盖 cache maintenance、memory barrier、cacheline-level hazard 和 macro-op 内部访问。
