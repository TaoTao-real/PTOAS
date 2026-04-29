# PTOAS InsertSync: Memory-Phi / Canonical-Writer 去重合并架构设计（ADR）

- Status: Proposed
- Date: 2026-04-29
- Owners: PTOAS InsertSync maintainers
- Scope: `InsertSync` 分析与生成层（不改前端 IR 语义，不改 CLI）

---

## 1. 背景与问题

当前 `InsertSync` 在复杂控制流（尤其 `loop + if/else`）下，常出现“依赖语义等价但同步链重复”：

1. 分支内有两个物理 writer（例如 `if: tmatmul` / `else: tmatmul.acc`）。
2. 汇合后 consumer 只关心“该逻辑结果已可见”，但当前算法仍按两个物理 writer 各建一套同步边。
3. 即便已有 `if/else` 路径同步状态取交集，交集机制只作用于 `alreadySync` 传播，不会合并“语义等价依赖边”。

结果是：

1. loop-carried `set/wait`、seed/drain 数量偏多。
2. 某些 pipe-pair（如 `M -> MTE1`）在分支场景下出现双链或多链。
3. 同步数量与手工版本差距明显，影响性能和 event-id 压力。

---

## 2. 设计目标

1. 在 `InsertSync` 中建立 Memory-SSA 风格的逻辑写点建模（memory-phi / canonical-writer）。
2. 以“逻辑 writer”而非“分支内物理 writer”作为依赖源，消除语义重复建边。
3. 对 loop-carried 仅保留真实跨迭代依赖，同步链按 canonical key 去重。
4. 在不放松安全边界前提下降量，确保 `#428/#454` 等历史高风险场景零回退。

---

## 3. 非目标

1. 不改变 PTO 前端 IR 语义与用户写法。
2. 本 ADR 不直接引入 multi-buffer(`eventIdNum>1`) 激进删减策略。
3. 不在“不可证明”别名场景做乐观放宽（unknown 仍保守）。

---

## 4. 核心决策

### 4.1 从“物理 writer”切换到“逻辑 writer”

对每个 `(rootBuffer, groupId)` 建立逻辑定义链：

1. `MemoryDef`: 普通写操作定义。
2. `BranchPhi`: `if/else` 汇合处的逻辑定义（代表互斥分支写入的统一结果）。
3. `LoopPhi`: 循环头的逻辑定义（承载 loop-carried 关系）。

consumer 不再直接依赖 `writer_A/writer_B`，而是依赖其最近支配的 canonical writer（可能是 `BranchPhi/LoopPhi`）。

### 4.2 去重基于 canonical key，而非路径状态交集

保留既有路径交集逻辑用于保守传播，但新增语义级去重主键：

`(producerCanonicalWriter, consumerAnchor, srcPipe, dstPipe, depKind, loopScope, slotClass)`

同 key 的同步边只保留一条。

### 4.3 loop-carried seed/drain 与 eventId 绑定 canonical 依赖

seed/drain 的生成单位由“具体物理边”改为“canonical 回边依赖”。  
eventIdNum 不再由 `MAT/VEC => 2` 推断，而由 canonical 依赖关联的真实槽位信息决定。

---

## 5. 架构设计

### 5.1 新增分析层：`CanonicalWriterAnalysis`

建议新增模块：

1. `include/PTO/Transforms/InsertSync/CanonicalWriterAnalysis.h`
2. `lib/PTO/Transforms/InsertSync/CanonicalWriterAnalysis.cpp`

职责：

1. 构建控制流区域树（loop/branch scope）。
2. 构建每个 `(rootBuffer, groupId)` 的 MemoryDef/Phi 图。
3. 回答“某 consumer 的 canonical writer 是谁”。
4. 提供 canonical 去重 key 所需 metadata。

### 5.2 关键类型

建议新增或扩展如下类型：

1. `CanonicalMemKey { rootBuffer, groupId, depKind, scopeId }`
2. `CanonicalWriterId { kind=Def|BranchPhi|LoopPhi, id }`
3. `MemoryPhiNode { kind, incomingDefs, mergeScopeId }`
4. `CanonicalDepEdge { producer, consumer, srcPipe, dstPipe, depKind, loopScopeId }`
5. `SyncDedupKey { producer, consumerAnchor, srcPipe, dstPipe, depKind, loopScopeId, slotClass }`

### 5.3 与现有 `InsertSyncAnalysis` 交互

`InsertSyncAnalysis` 流程调整为：

1. 先做 `MemoryDependentAnalyzer` 得到候选依赖。
2. 用 `CanonicalWriterAnalysis` 把候选依赖映射到 canonical 依赖边。
3. 用 `SyncDedupKey` 去重。
4. 再进入 `InsertSyncOperation` 生成同步原语。

---

## 6. 控制流与移动不变量

为防止分支/循环场景错配，同步移动必须满足：

1. `set/wait` 必须保持在共同执行谓词域内，不能跨越最内层共同控制区域边界。
2. 分支一致性同步只能锚定在 `IF_BEGIN/IF_END` 内侧合法锚点。
3. 对 `if/else` 互斥路径，只能在已建 `BranchPhi` 的前提下合并。
4. 对 `loop-carried`，仅在存在真实 `LoopPhi` 回边依赖时才生成 seed/drain。
5. unknown alias 或不满足上述条件时，一律保守不合并。

---

## 7. 端到端算法流程

### Phase A: Canonical 图构建

1. 收集 `def/use` 与控制流作用域。
2. 按 `(rootBuffer, groupId)` 建立 `MemoryDef` 序列。
3. 在 `if/else` 汇合创建 `BranchPhi`。
4. 在循环头创建 `LoopPhi`（若存在 loop-carried 候选）。

### Phase B: 依赖归一化

1. 将候选依赖 `(frontCompound -> nowCompound)` 投影为 canonical 依赖。
2. 以 canonical writer 替代具体 writer 作为 producer。
3. 形成 `CanonicalDepEdge`。

### Phase C: 建边与去重

1. 使用 `SyncDedupKey` 去重 canonical 边。
2. 对剩余边执行 `set/wait/barrier` 插入。
3. 对 loop-carried 边只生成一套 seed/drain（按 canonical 边）。

### Phase D: event-id 与后处理

1. `eventIdNum` 基于槽位信息与 canonical 边属性决策。
2. `RemoveRedundantSync` 在 canonical 依赖签名层面继续做保守裁剪。

---

## 8. 与现有“分支交集”机制的区别

现有机制（`MergeAlreadySync`）：

1. 解决的是“路径状态是否同时成立”。
2. 不改变边语义身份，不会折叠两条等价依赖边。

本设计：

1. 先把依赖边身份提升到 canonical writer 层。
2. 再在 canonical 层去重，源头消除重复同步链。

两者关系：交集保留，但角色从“主要降量手段”降级为“保守状态传播机制”。

---

## 9. 对 FA 场景的预期收益

在 `loop + if(tmatmul)/else(tmatmul.acc)` 的链路上：

1. `M -> MTE1` 回边双链可收敛为单链。
2. seed/drain 与 loop 内对应 `set/wait` 可按链路减少一组。
3. 静态同步数可下降，动态每迭代同步开销可下降，event-id 压力降低。

注：`PIPE_V` 同 pipe barrier 过量属于另一条优化线（同 pipe barrier 判定精化），不与本 ADR 绑定。

---

## 10. 正确性风险与缓解

主要风险：

1. 嵌套 if/else + nested loop 下错误跨域合并。
2. 分支非等价写入被误合并。
3. loop-carried 误判导致漏同步。

缓解策略：

1. 合并前置条件必须包含控制域一致性、writer 等价性、alias 可证明性。
2. 不满足条件时立即降级为保守路径（不合并）。
3. 增加结构化回归与 debug 统计（见下一节）。

---

## 11. 验证计划与门禁

### 11.1 必过回归

1. `issue428_cube_sync_regression`
2. `issue454_nested_loop_same_pipe_pair_regression`
3. `issue454_loop_if_else_loop_carried_sync_regression`
4. `issue226/issue233` 代表用例

### 11.2 新增结构化 case

1. 双层嵌套 if + loop-carried 同 pipe-pair
2. loop 内 if/else 分支写同 root、汇合后单 consumer
3. 分支写不同 root（应禁止合并）
4. unknown alias（应保持保守）

### 11.3 Debug 指标

新增统计建议：

1. `rawDepEdges`
2. `canonicalDepEdges`
3. `mergedDepEdges`
4. `rejectedByPredicateDomain`
5. `rejectedByAliasUnknown`

用于证明“降量来源于分析增强，而非误删”。

---

## 12. 实施计划

### Step 1: 分析与类型落地

1. 新增 `CanonicalWriterAnalysis` 及类型定义。
2. 打通 `InsertSyncAnalysis` 查询接口。

### Step 2: canonical 建边与去重

1. 在依赖构建处替换为 canonical producer。
2. 引入 `SyncDedupKey` 去重容器。

### Step 3: seed/drain 与 event-id 对齐

1. 以 canonical 回边依赖生成 seed/drain。
2. 将 eventIdNum 决策对齐到真实槽位。

### Step 4: 回归与性能评估

1. 完整回归门禁。
2. 对 `#226/#233/FA` 输出同步计数对比与编译日志对比。

---

## 13. 兼容性与外部行为

1. 不改 CLI 参数。
2. 不改前端 IR 语义。
3. 对外仅表现为同步插入数量与位置更优，正确性约束不放宽。

---

## 14. 备选方案与取舍

### 方案A：只靠路径交集和本地规则继续打补丁

优点：改动小。  
缺点：无法从语义层合并重复依赖边，长期维护成本高。

### 方案B：直接做激进 multi-buffer 删除

优点：短期降量快。  
缺点：高风险，易触发历史回归，不利于解释性与可维护性。

### 方案C（本决策）：先引入 canonical writer 架构

优点：语义清晰、可扩展、正确性边界可审计。  
缺点：前期工程量较大。

---

## 15. 结论

本 ADR 决策为：  
采用 Memory-Phi / Canonical-Writer 架构，对 `InsertSync` 的依赖建模与去重机制进行语义级升级。  
该方向可系统性解决 `loop + if/else` 场景的重复同步链问题，并为后续 event-id 与多缓冲优化提供稳定基础。

