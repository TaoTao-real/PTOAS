# ADR: VF 融合场景下 `PIPE_V` Barrier 精简优化

- Status: Proposed
- Date: 2026-04-29
- Owners: PTOAS InsertSync maintainers
- Scope: `InsertSync` 分析与生成层（`PIPE_V` 同 pipe barrier）

---

## 1. 背景

当前 `InsertSync` 在同 pipe（`PIPE_V -> PIPE_V`）的同步策略较保守：  
一旦识别到依赖，通常直接插入 `pipe_barrier(PIPE_V)`。  
在 FA/softmax 等向量链路上，会出现“几乎每步一条 barrier”的过串行化现象。

这与手工优化版本存在差异：手工版本常在“关键边界”插 barrier，而在可连续执行的 VF 链中不逐步插入。

---

## 2. 问题定义

问题不是“完全移除 `PIPE_V` barrier”，而是区分：

1. 必要 barrier：防止真实读写冲突、跨阶段可见性问题、特殊算子内部临时缓冲风险。
2. 冗余 barrier：可由 VF 连续执行顺序天然保证，或可通过更粗粒度边界同步替代。

目标是让自动同步在 VF 可融合链上“少插且不漏插”。

---

## 3. 目标与非目标

### 3.1 目标

1. 在 `PIPE_V` 可连续链路上减少冗余 barrier 插入。
2. 在 FA/softmax 代表场景明显降低 `pipe_barrier(PIPE_V)` 数量。
3. 保持正确性约束不放宽：unknown/不可证明场景仍保守。

### 3.2 非目标

1. 不在本 ADR 里修改 cross-pipe (`set_flag/wait_flag`) 主策略。
2. 不在本 ADR 里引入激进 multi-buffer 删除。
3. 不依赖上层用户新增语义注解才能工作（可选注解仅增强，不作为必要条件）。

---

## 4. 关键决策

### 决策A：从“点对点依赖即插 barrier”升级为“VF 段级建模 + 边界同步”

对同一控制域内的 `PIPE_V` 操作构建 VF 连续段（VF Segment）：

1. 段内仅保留必要 barrier（通常 0 或极少数）。
2. 在段边界按风险规则放置 barrier。

### 决策B：引入可证明安全的算子分类与风险矩阵

将 `PIPE_V` 算子按融合风险分层（白名单优先）：

1. `VF_PURE_ELEMWISE`: 逐元素纯算，无隐藏状态。
2. `VF_LAYOUT_LIGHT`: reshape/cast/轻布局变换。
3. `VF_REDUCTION`: row/col reduce 类。
4. `VF_GATHER_SCATTER`: 地址/掩码相关访问（较高风险）。
5. `VF_STATEFUL_OR_UNKNOWN`: 含潜在临时缓冲或无法证明安全。

仅在 “低风险 -> 低风险” 且 alias/切片可证明安全时允许无 barrier 跨越。

### 决策C：unknown 保守

任一条件未知时，回退到当前保守策略，不做冒险消减。

---

## 5. 架构设计

## 5.1 新增分析模块：`VFBarrierCoarseningAnalysis`

建议新增：

1. `include/PTO/Transforms/InsertSync/VFBarrierCoarseningAnalysis.h`
2. `lib/PTO/Transforms/InsertSync/VFBarrierCoarseningAnalysis.cpp`

职责：

1. 在 `syncIR` 中扫描 `PIPE_V` 节点，构建候选 VF 段。
2. 查询每对相邻 V 节点是否可“无 barrier 连续”。
3. 输出 `VFSegmentPlan`（段内去 barrier、段边界保 barrier 建议）。

### 5.2 关键数据结构

1. `VFOpClass`：算子分类枚举（如上 5 类）。
2. `VFPairDecision`：`KEEP_BARRIER / DROP_BARRIER / UNKNOWN`。
3. `VFSegment { beginIdx, endIdx, droppedEdges[], keptEdges[] }`
4. `VFDecisionReason`：记录为何保留/删除（用于 debug 可解释性）。

### 5.3 与现有 InsertSync 集成点

集成策略：

1. `InsertSyncAnalysis` 仍先建全量候选同步边（保持正确性基线）。
2. 在 `RemoveRedundantSync` 之前增加“VF 段精简”步骤：
   - 仅处理 `PIPE_V -> PIPE_V` barrier。
   - 按 `VFSegmentPlan` 标记可删 barrier。
3. 其余同步链路不受影响。

这样能降低改造风险，并保留回退路径。

---

## 6. 判定规则（核心）

设相邻两条 `PIPE_V` 操作为 `A -> B`，当前有候选 barrier：

### 必须保留（任一命中）

1. `A/B` 任一属于 `VF_STATEFUL_OR_UNKNOWN`。
2. `A/B` 属于高风险跨类转换（如 `REDUCTION -> GATHER_SCATTER`）且无额外证明。
3. alias/slice 无法证明安全（unknown）。
4. 跨控制域边界（if/else 汇合、loop back-edge、unlikely scope 锚点）。
5. 存在 op-level 强制同步标记（如后续引入 `must_barrier_before/after` trait）。

### 可移除（需同时满足）

1. `A/B` 在同一控制域、同一 VF 候选段内。
2. 分类组合在“可连续执行矩阵”中允许。
3. 依赖为 SSA 前向可见，不要求额外内存可见性屏障。
4. alias/slice 可证明无额外冲突。

---

## 7. 与现有问题的关系

本 ADR 直接针对：

1. FA 等向量长链条中的 `PIPE_V` barrier 过密问题。
2. 自动同步相对手工同步“逐点串行化”偏保守问题。

本 ADR 不直接解决：

1. `eventIdNum` 双ID推测问题。
2. `if/else` 逻辑写点重复链（memory-phi/canonical-writer）问题。

三者可并行推进，互相增益但边界清晰。

---

## 8. 正确性约束与风险控制

主要风险：误删 barrier 导致隐藏读写冲突。

控制策略：

1. 默认白名单策略：先支持低风险算子组合。
2. unknown 一律保留 barrier。
3. 新增 debug trace：每个被删 barrier 必须有“可解释原因”。
4. 关键回归集必须全通过后再逐步扩白名单。

---

## 9. 验证计划

### 9.1 功能回归（必须）

1. `issue428_cube_sync_regression`
2. `issue454_nested_loop_same_pipe_pair_regression`
3. `issue454_loop_if_else_loop_carried_sync_regression`

### 9.2 场景回归（新增）

1. FA softmax/init/not-init 两路径。
2. 含 causal mask 分支的 V 链。
3. `gather/scatter + elementwise` 混合链。
4. unknown alias 场景（验证“保守不删”）。

### 9.3 指标与验收

输出以下统计：

1. `pipeVBarrierBefore / pipeVBarrierAfter`
2. `pipeVDroppedByReason`
3. `pipeVKeptByUnknown`

验收标准（第一阶段）：

1. FA 代表样例中 `PIPE_V` barrier 数显著下降。
2. 正确性回归零失败。
3. 无新增编译/运行异常。

---

## 10. 分阶段落地

### Phase 1（低风险）

1. 只对 `VF_PURE_ELEMWISE + VF_LAYOUT_LIGHT` 组合做段内去 barrier。
2. 不跨 if/else、不跨 loop back-edge。

### Phase 2（中风险）

1. 增加 `REDUCTION` 相关可融合规则。
2. 引入更精细 alias/slice 证明接入。

### Phase 3（高收益）

1. 与 memory-phi/canonical-writer 结合，跨分支做逻辑级段合并。
2. 与 event-id/loop-carried 收敛协同优化。

---

## 11. 实施接口建议

建议新增 pass 选项（先默认关闭，灰度）：

1. `--enable-vf-barrier-coarsening`
2. `--vf-barrier-coarsening-level=[safe|balanced|aggressive]`

建议 debug 选项：

1. `--vf-barrier-debug`

---

## 12. 备选方案与取舍

### 方案A：仅靠手工规则特判 FA

优点：见效快。  
缺点：不可扩展，难维护。

### 方案B：完全依赖后端融合器，前端不做 barrier 精简

优点：实现简单。  
缺点：前端保守 barrier 已经固化，会直接抑制融合机会。

### 方案C（本决策）：插入阶段做可证明的 VF 段级精简

优点：可解释、可控、可扩展，且能与现有框架兼容。  
缺点：需要新增分析模块与验证体系。

---

## 13. 结论

采用“VF 段级建模 + 风险矩阵 + unknown 保守”的 `PIPE_V` barrier 精简架构。  
该方案可在保证正确性的前提下，把自动同步从“逐点插 barrier”升级为“边界化同步”，在 FA 等高频场景显著降低冗余 `pipe_barrier(PIPE_V)`。

