# ADR: 先模型选择，再完成 CV 候选编译

日期：2026-10-27
状态：proposed；本文件定义下一轮实现，不表示能力已经交付
计划：[实施与验收计划](../designs/cv-vf-model-first-implementation-plan.md)
上层设计：[VF 与延迟存储绑定](adr-cv-vf-late-storage-binding.md)

## 背景

2026-10-23 固定 FA 验收先完整编译 P=0/1/2/3，再让 TileSim 读取最终物理程序并选择。
它证明了模型选择、原生 CV 变换和上板收益，但没有实现“先寻优、后最终编译”。
当前 FA 工具还逐候选建立 prepared 包，没有形成跨候选的一次 VF 准备与成本复用。

已有 `--cv-analyze-schedule` 会在 CV 调度分析后退出，跳过后续 PlanMemory、同步落实和对象生成。
它是轻量候选入口。当前 TileSim `ptoas_prepared.py` 依赖最终地址和编译反馈，不能直接消费这个入口。

## 决策

正式显式优化路径按以下顺序执行：

```text
准备并融合 VF 一次
  → 导出逻辑 Buffer、VF 接口及适用条件
  → 编译器生成轻量候选调度、版本、资源需求和必要完成约束
  → TileSim/VfSimulator 计时、排序并冻结完整 Plan
  → 原生 CV pass 消费选定 Plan
  → PlanMemory、同步落实、最终对象编译
  → 核对编译反馈；必要时重评受影响候选
  → 独立 G2/G3 与性能验收
```

PTOAS 定义合法候选与执行语义。TileSim 负责成本与排序。
外部 orchestrator 负责文件交换、独立进程与有限重试，不将 Python 模型嵌入 C++ pass。
CV 分析与实际应用共享调度构造逻辑，禁止在 adapter 中再实现一套 preload 排序规则。

所有初始推荐冻结前，不生成候选设备对象、LLVM 代码或 launcher，不运行 Bisheng 最终编译。
允许对 prepared 模块的新副本作调度、依赖、活跃版本和资源分析。
必要的 layout、stride、对齐和 VF 内部 scratch 条件在准备阶段确定；跨阶段绝对地址仍由选中后的 PlanMemory 决定。

模型使用逻辑 Buffer / version / slot 身份与相对访问范围。
模拟所需的代表性基址只能是带条件的模型内部量，不是 PTO 物理分配，也不能写回 Plan。
成本依赖布局、相对地址、bank、mask 或别名条件时，将这些条件纳入缓存键与覆盖报告。
无法表达的必需成本为 unknown，不能用零替代或凭旧物理候选补齐。

最终编译必须核对逻辑 schedule 指纹、VF 指令/布局条件、搬运、完成约束和实际资源。
逻辑 schedule 指纹与最终物理产物指纹分别记录，不要求不同层 IR 的文本摘要相等。
新增实质搬运、同步或不满足 VF 成本条件时，旧预测失效，重评选中候选。
若需更换候选，则从原始 prepared 模块重新开始，记录新 attempt，不能修改原冻结 Plan。
模型预测时间不参与内存安全证明；G2 仍由编译器及独立证据完成。

## 版本和兼容

沿用现有 preload、逐 Buffer 槽数和 Buffer ID annotation，不增加用户 annotation。
既有 v1/2.0 和 physical prepared adapter 保持原行为。
逻辑搜索输入使用独立版本 `pto.cv.prepared_search.v1`，显式声明所需能力。
旧 adapter 不支持该能力时明确拒绝，不偷偷退回全候选完整编译。
普通编译路径与默认 annotation-only 行为保持不变。

首版仅搜索 preload，槽位沿用已验证的固定配置或编译器给出的最低合法需求。
已有固定 FA 仍使用受限兼容导入器，不能因此宣称通用 FA 导入已完成。

## 备选方案和影响

- 保留全候选完整编译再选择：保留作显式诊断/离线参考，不能计入本方案完成条件。
- 在 CV C++ pass 内直接调用 TileSim：不采用，避免绑定 Python/MLIR 环境并阻塞未来模型替换。
- 完全在 adapter 中猜测调度：不采用，会使模型和编译器执行语义分离。

本方案减少重复 VF 准备及最终编译次数，但前置预测必须描述完成约束和成本适用条件。
编译后反馈是必需校验，不是要求先完整编译所有候选的理由。
基线和排序诊断候选可在推荐冻结后为实验单独编译，明确标为 evaluation-only。
历史 33.04% 收益和 3.36% 单点误差保留其原验收范围，不为新流程自动背书。
