# ADR: Loop 控制流与首迭代特化场景的路径敏感自动同步收敛

- Status: Proposed
- Date: 2026-04-30
- Owners: PTOAS InsertSync maintainers
- Scope: `InsertSync` 分析层与同步移动层（不改 CLI/IR 语义）

---

## 1. 背景

在 FA 等 kernel 中，loop 内常见首迭代特化：

1. `i == 0` 走初始化路径（如 `TMATMUL`）。
2. `i > 0` 走累加路径（如 `TMATMUL_ACC`）。

当前 InsertSync 在该类控制流上仍偏保守，主要表现为：

1. 将“同 rootBuffer + 跨迭代可能相关”统一视为 loop-carried 候选。
2. 对分支路径的依赖以 union 近似，缺少谓词可满足性约束。
3. seed/drain（loop 头尾补偿同步）按潜在依赖补齐，导致自动同步数量高于手工版本。

这类过保守同步会与已有问题叠加：

1. 影响 `set_flag/wait_flag` 数量和 eventId 压力。
2. 放大 loop 回边链路，稀释前续 ADR（canonical-writer、VF barrier 收敛）的收益。

---

## 2. 问题定义

目标问题不是“删除所有 loop 同步”，而是回答以下可证明问题：

1. 某条依赖是否在同一执行谓词下可达？
2. 某条依赖是否是真实跨迭代（loop-carried）而不是首迭代/同迭代依赖？
3. 某条同步在移动（hoist/sink）后是否仍与其配对同步处于同一控制流执行域？

只有三者都成立时，才允许生成或保留对应的跨迭代同步链与 seed/drain。

---

## 3. 目标与非目标

### 3.1 目标

1. 对 loop 内 if/else 与首迭代特化场景进行路径敏感依赖判定。
2. 仅为真实 loop-carried 依赖生成 seed/drain 和回边事件链。
3. 约束同步移动不得越过最紧密控制流执行域，保证 set/wait 谓词匹配。
4. 在 FA/#233 类场景降低冗余同步数量，保持 `#428/#454/#533` 正确性。

### 3.2 非目标

1. 不在本 ADR 中修改 `SyncEventIdAllocation` 的硬件资源策略。
2. 不在本 ADR 中引入 multi-buffer (`eventIdNum>1`) 删除/合并。
3. 不依赖用户新增注解作为必要条件（注解仅可作为后续增强）。

---

## 4. 关键决策

### 决策A：在 SyncIR 层显式建模执行谓词域（Predicate Domain）

为每个 `InstanceElement` 维护谓词域，编码其控制流可达条件：

1. `if` 的 then/else 谓词。
2. loop 迭代类谓词（首迭代、稳态迭代、单次迭代）。

依赖边仅在谓词交集可满足时建立。

### 决策B：引入迭代类（Iteration Class）区分首迭代与真正跨迭代

在分析层区分：

1. `FirstIter`：`i == lb`
2. `SteadyIter`：`i > lb`
3. `SingleTrip`：静态可证 tripCount <= 1

`SingleTrip` 场景禁用 loop-carried 依赖与 seed/drain 生成。

### 决策C：基于 memory-phi/canonical-writer 建立逻辑写点

分支汇合后不再对每个物理 writer 重复建边，而是先映射到逻辑写点：

1. 路径内 writer -> canonical logical writer。
2. 回边 RAW/WAR/WAW 按 logical writer 判重与收敛。

### 决策D：同步移动受“共同控制域”约束

`MoveSyncState` 进行 wait 外提 / set 下沉时，必须满足：

1. 不越过 set/wait 的最紧密共同控制域（LCA control scope）。
2. 若跨域将导致谓词不匹配，则禁止移动。

---

## 5. 架构设计

## 5.1 新增/扩展分析组件

建议在 `InsertSync` 子系统新增以下 analysis-only 组件：

1. `ControlPredicateAnalysis`
2. `LoopIterationClassAnalysis`
3. `PredicateAwareDepProver`
4. `LogicalWriterResolver`（复用 canonical-writer 框架）

建议文件：

1. `include/PTO/Transforms/InsertSync/ControlPredicateAnalysis.h`
2. `lib/PTO/Transforms/InsertSync/ControlPredicateAnalysis.cpp`
3. `include/PTO/Transforms/InsertSync/LoopIterationClassAnalysis.h`
4. `lib/PTO/Transforms/InsertSync/LoopIterationClassAnalysis.cpp`

## 5.2 关键数据结构

1. `PredicateToken`：原子谓词（如 `if#k.then`、`if#k.else`、`loop#j.first`、`loop#j.steady`）。
2. `PredicateDomain`：`SmallBitVector` 或 CNF-lite 集合，用于 `IntersectIsSat()`。
3. `IterationClass`：`FirstIter / SteadyIter / SingleTrip / UnknownTrip`。
4. `LogicalWriterId`：canonical writer/memory-phi 节点标识。
5. `DepEdgeMeta`：`{depKind, rootBuffer, srcPipe, dstPipe, predDomain, iterClass}`。

## 5.3 与现有流水线集成点

现有链路：

`Translator -> Analysis -> MoveSyncState -> RemoveRedundant -> EventIdAlloc -> Codegen`

集成建议：

1. 在 `InsertSyncAnalysis::Run` 前准备 `ControlPredicateAnalysis` 与 `LoopIterationClassAnalysis`。
2. 在 `InsertSyncAnalysis::InsertSync/InsertBackForSync` 中使用 `PredicateAwareDepProver`。
3. 在 `MoveSyncState` 的 move-out 逻辑增加 `CanMoveAcrossControlScope` 检查。
4. 其余阶段保持接口兼容，unknown 回退为现有保守行为。

---

## 6. 规则细化

### 6.1 依赖建边必要条件

仅当以下条件同时满足才建边：

1. alias/slice 判定存在冲突可能（或已证冲突）。
2. `PredicateDomain(src) ∩ PredicateDomain(dst)` 可满足。
3. 在同迭代或跨迭代拓扑上可达。

任一条件 unknown 时保守保留。

### 6.2 loop-carried 与 seed/drain 判定

仅当以下条件成立才生成回边链与 seed/drain：

1. 迭代类不是 `SingleTrip`。
2. 存在 `iter(k)` 到 `iter(k+1)` 的真实可达冲突。
3. 该冲突在谓词域上可满足。

若只在 `FirstIter` 内成立，不视为 loop-carried。

### 6.3 首迭代特化分支处理

典型模式：

1. `if (i==0) W0 else W1`
2. 后续 `Read` 依赖前述写。

处理方式：

1. 在迭代内建立 `W0/W1 -> memory-phi -> Read` 逻辑链。
2. 首迭代与稳态迭代分别评估依赖，不将首迭代特有边错误提升为回边依赖。

### 6.4 同步移动约束

对 `MoveSyncState` 增加强约束：

1. wait 外提不得越过其匹配 set 的共同控制域入口。
2. set 下沉不得越过其匹配 wait 的共同控制域出口。
3. 若移动后存在“可能执行 wait 但不执行 set”或反向情况，禁止移动。

---

## 7. 正确性不变量与风险控制

## 7.1 不变量

1. 谓词匹配不变量：每个 wait 在任意可执行路径上必须存在可先行匹配 set。
2. 回边真实性不变量：seed/drain 仅对应真实跨迭代依赖。
3. 控制域不变量：同步移动不能破坏 set/wait 的共同执行域。

## 7.2 风险与防护

主要风险是“过度收敛导致漏同步”。防护策略：

1. predicate 不可判定时回退保守。
2. 单独 debug 统计 `suppressedByPredicate`、`suppressedBySingleTrip`、`keptByUnknown`。
3. 先灰度开关，再默认启用。

---

## 8. 预期收益

在与 canonical-writer ADR、VF barrier ADR 叠加后，本 ADR 主要贡献：

1. 消除首迭代特化被误判为 loop-carried 的冗余事件链。
2. 降低 seed/drain 数量与类型扩散。
3. 降低 if/else 场景下由移动导致的配对不稳定风险。

预计对 FA 类场景体现为：

1. `set_flag/wait_flag` 进一步下降（主要是回边链路）。
2. 为 eventId 压力收敛创造条件（间接收益）。

---

## 9. 验证计划

### 9.1 必过回归

1. `issue428_cube_sync_regression`
2. `issue454_nested_loop_same_pipe_pair_regression`
3. `issue454_loop_if_else_loop_carried_sync_regression`
4. `issue533_loop_zero_trip_sync_regression`

### 9.2 场景增强

1. loop 首迭代特化（`i==0` vs `i>0`）matmul/matmul_acc case。
2. loop 内 if/else 且仅一侧写、汇合后读取 case。
3. `tripCount<=1` 静态可证 case（应无 seed/drain）。
4. `tripCount unknown` case（应保守保留）。

### 9.3 指标验收

新增 debug 指标：

1. `loopCarriedCandidates`
2. `loopCarriedConfirmed`
3. `seedDrainInserted`
4. `seedDrainSuppressedBySingleTrip`
5. `depSuppressedByUnsatPredicate`

验收原则：

1. 关键回归零失败。
2. FA/#233 代表样例同步数量下降且无行为退化。
3. 无新增编译错误与死锁。

---

## 10. 落地计划

### Phase 1（安全落地）

1. 引入 `IterationClass` 与 `tripCount<=1` 抑制回边。
2. 在回边链路启用 predicate 可满足性检查。
3. 增加 debug 统计与测试看护。

### Phase 2（核心收益）

1. 全面接入 memory-phi/canonical-writer 到回边依赖建模。
2. 扩展到 if/else 汇合点的逻辑写点去重。

### Phase 3（完备收敛）

1. 将控制域约束接入 `MoveSyncState` 全路径移动规则。
2. 与 VF barrier 收敛策略联合调优，形成统一同步收敛框架。

---

## 11. 架构影响评估（侵入性/兼容性/性能）

### 11.1 是否基于现有 SyncIR 扩展

本方案明确基于当前 `SyncIR` 线性框架扩展，而非重写同步框架：

1. 保留 `InstanceElement` 及 `LOOP/BRANCH/PLACE_HOLDER` 控制流骨架。
2. 新增的是 analysis-only 元数据（谓词域、迭代类、逻辑写点映射）。
3. `SyncOperation` 对外语义与 codegen 指令形态不变。

### 11.2 对现有框架的破坏性

侵入性评估为“低到中”：

1. 不改前端 IR 语义，不改 CLI 兼容面。
2. 主要修改集中在 `InsertSyncAnalysis` 与 `MoveSyncState` 判定逻辑。
3. `SyncEventIdAllocation`、`SyncCodegen` 仅消费收敛后的边，不需结构性重写。

### 11.3 功能正确性风险与控制

主要风险是过度收敛导致漏同步。控制策略：

1. `predicate unknown => 保守保留`。
2. `tripCount unknown => 维持当前 loop-carried 判定`。
3. 默认灰度开关，回归稳定后再默认启用。
4. 所有删除/抑制决策必须可解释（debug reason）。

### 11.4 编译时与运行时性能影响

1. 编译时：新增谓词/迭代类分析，预期小幅增加 pass 开销。
2. 运行时：目标是减少冗余同步，常见场景性能应提升或持平。
3. 最坏情况：回退到保守路径，行为与当前基线一致，不应引入系统性性能回退。

### 11.5 回退与应急策略

1. 保留 feature flag，可一键关闭路径敏感收敛逻辑。
2. 若发现风险，可仅保留 `SingleTrip` 抑制（Phase 1）并关闭后续阶段。
3. PR 颗粒度按 Phase 拆分，便于快速回滚单阶段改动。

---

## 12. 备选方案与取舍

### 方案A：仅加模式特判（例如识别 `i==0`）

优点：实现快。  
缺点：覆盖窄、不可扩展、易回归。

### 方案B：仅依赖后端调度/融合消化保守同步

优点：前端改动小。  
缺点：前端已插入同步会抑制后端优化上限。

### 方案C（本决策）：路径敏感 + 迭代类 + 逻辑写点统一建模

优点：可解释、可扩展、与现有架构兼容。  
缺点：分析复杂度与调试成本上升。

---

## 13. 结论

采用“路径敏感谓词域 + 首迭代/稳态迭代分层 + canonical logical writer + 受控同步移动”方案，在不放宽正确性边界的前提下收敛 loop 内控制流场景的过保守同步，为 FA 等复杂 kernel 的自动同步数量接近手工版本提供架构基础。
