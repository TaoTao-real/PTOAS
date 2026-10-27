# PTOAS × TileSim：先选 preload、后最终编译的实施与验收计划

日期：2026-10-27
状态：planned；本轮仅制定计划，未实现新搜索路径
决策：[model-first ADR](../adr/adr-cv-model-first-selection.md)

## 1. 目标和范围

将当前固定 FA 的“全候选完整编译→模型选择”改成：

```text
PTO → VF 准备一次 → 轻量候选 → TileSim 推荐与 Plan 冻结
    → 仅编译选中候选 → 最终反馈核对 → G2/G3 → 新 A/B
```

正常成功路径只进行一次选中候选的完整设备编译。
基线及额外排序对照只能在推荐冻结后，为独立验收生成，不属于在线搜索前置条件。
遇到失败或反馈改变成本时有限处理，不能重新枚举并完整编译全部候选。

首版保持 A5、静态 shape、1 AIC＋2 AIV、核内调度关闭。
先运行 basic/crossing，再运行同事提供的 M32 FA，P=0/1/2/3。
固定 FA 的形状、递推、输入 SHA、共享容量、同步策略及物理尾部保护沿用现有证据定义。
本轮不扩展通用 FA 识别、动态 shape、multi-buffer 搜索和通用地址复用。
不增加用户 annotation，不改变旧协议含义，不在普通 pipeline 中默认启用。

## 2. 起点、分支和现状

- PTOAS：PR #1676，冻结起点 `b2d06862dad3974ecfa0a8476ab6c14ed2db248b`。
- 开发分支：`codex/cv-vf-model-first-search`，从上述 MR289 集成基础开始。
- TileSim：PR #142，起点 `5a7fcba433627a31c1d825f4feedd88f5d4dfa49`，基于 `dev_zzh/tilesim_refactor`。
- VfSimulator 暂以 `f19615c6951f6c5210fbe06f1579bc90dc0737e7` 为兼容起点；正式实验重新记录实际冻结版本。
- `cv_vf.analyze_schedule` 与 `--cv-analyze-schedule` 已有轻量基础，不需要从头重写 CV pipeline。
- 现有 TileSim `NativeEvents` 仅支持最终物理程序；FA 工具逐候选 export、compile、link，是主要改动对象。

不把主线重基作为核心开发前置。合入前独立处理 #1676 的主线冲突并运行合入版本回归。
旧源码、已冻结 Plan、对象和原始样本保持只读。

## 3. 输入契约

新增独立版本 `pto.cv.prepared_search.v1`，通过 required_features 协商。
它引用一个共享 prepared 包，而不是每个 P 复制一个完整程序。

| 内容 | 必须提供的信息 |
| --- | --- |
| 共享程序 | 配对 C/V、prepared VF、输入/绑定/目标/编译器指纹 |
| VF 描述 | 指令体、shape、dtype、layout、stride、mask、scratch、读写及成本适用条件 |
| 候选 | candidate ID、P/Pe、编译器生成的实例顺序、依赖、迭代距离和 schedule 指纹 |
| 存储 | Buffer/alias 根、version、slot、ownership、所属物理核及静态需求；不含最终物理地址 |
| 通信与完成 | split、有效容量、事务、push/pop/free 或对应低层交互、释放/访问完成条件 |
| 模型结果 | 完整 Plan、排序、候选时延、分项/等待、覆盖、近似及模型/adapter/config 指纹 |

结构化描述来自 MLIR 对象。adapter 不解析 MLIR 字符串，不从 SSA 名或绝对地址反推身份。
Cube 和搬运成本所需的维度、stride、layout、方向、lane/split 从描述读取，不能沿用固定维度常量。
未知 alias、必要完成条件或操作成本明确拒绝；预测不能代替安全证明。
资源保守上界超预算而未证明不可行时标为 unresolved，不伪装成最终内存规划失败。
首轮只选择预检查可覆盖的候选；通用有界规划试跑另列后续。

## 4. 按依赖实施

| 步骤 | PTOAS / TileSim 改动 | 完成条件 |
| --- | --- | --- |
| S1 契约与可行性 | 冻结 schema、能力、参考包；验证无最终地址的 prepared VF 可被模型消费 | 一个 micro 和一个 FA VF 可得到有来源、有适用条件的有限成本；不读取旧 final feedback |
| S2 共享 VF 与轻量调度 | PTOAS export 一次；复用 analyze 入口，补全版本、相对访问、通信完成和资源描述；分析与应用共享构造逻辑 | 所有 P 从同一 prepared 包分析，无最终 PlanMemory/同步 lowering/设备对象；不修改原模块 |
| S3 TileSim 逻辑 adapter | 新增逻辑事件转换和 VF 成本缓存，复用 EventEvaluator/成本后端；保留物理 adapter 作诊断 | 全候选只用 S2 产物完成 evaluate/propose；没有最终反馈也能选择；旧输入/adapter 明确区分 |
| S4 推荐驱动编译 | 编排 export→analyze→propose→freeze→annotate→compile；实现最终反馈核对和有限失败处理 | 首次设备编译发生在选择冻结之后；正常只编译一个候选，实际 CV、PlanMemory 与同步通过 |
| S5 契约与编译验证 | 执行正负例、执行顺序、缓存、协议兼容、同一调度及资源/完成检查 | 新测试与既有回归通过；未知、旧 Plan、预测失效均不能静默放行 |
| S6 新上板验收与 PR | 冻结新版本/Plan，运行新 G3 与配对 A/B，记录模型误差及编译成本；更新双方 PR | 新流程证据完整；收益通过或失败如实报告，历史样本不冒充本次验收 |

S1/S2 完成后才能接 S3；S3 的初选不能依赖 S4 的产物。S4/S5 完成后才能进入 S6。

### 主要代码位置

- `lib/PTO/Transforms/CVSchedule.cpp`：从同一构造逻辑导出候选语义，并应用选中的 Plan；不在 C++ pass 中启动模型。
- `lib/PTO/Transforms/CVVF.cpp`、`tools/ptoas/ptoas_pipeline.cpp`：共享 VF 边界、效果与完成契约，分析路径的提前退出、最终反馈。
- `ptodsl/ptoas/cv_vf.py`：共享包、候选导出、严格 Plan 校验与最终反馈核对。
- TileSim `core/frontend/adaptor`：新增逻辑搜索 adapter，保留 `ptoas_prepared.py` 的物理诊断模式。
- `test/samples/CVCostModel/fa_model_ab.py`：拆开准备、搜索/冻结、选中候选构建和实验对照构建；旧全编译流程显式标为 diagnostic。

## 5. 成本复用与最终反馈

VF 准备一次不等于所有调用都只计算一个固定成本。
按 VF 内容、ISA、模型配置、dtype、有效 shape/mask、layout、stride、对齐、alias 及地址敏感条件缓存。
不同调用满足同一条件时复用成本；不同条件分别评估。用至少两组合法代表性绑定检查成本与缓存边界。
模型内部的代表性地址不进入 Plan，不冒充最终物理布局。
资源占用与两个 AIV 私有存储分开核算；临时 scratch 不伪装成可回填 Buffer。

编译后记录三个不同身份：逻辑候选 schedule、VF/成本条件、最终物理程序。
先核对逻辑执行语义，再核对实际地址与资源、VF 条件和新增操作/完成约束。

- 仅绑定地址且所有成本条件成立：记录预测可复用的依据。
- 有实质搬运、同步或地址敏感成本变化：使旧预测失效，使用最终反馈仅重评当前候选。
- 编译器证明候选非法/不可行：按冻结排名取下一候选，从原始 prepared 模块的新副本开始。
- 包损坏、身份不符或协议违规：终止，不能冒充正常回退。
- 自动优化模式最多尝试两个优化候选，最多重评两次；达到预算或没有可接受候选则保留基线，记录原因。
- 显式用户 Plan 失败即报错，不自动换参数。任何新 Plan 在对应编译前单独冻结，不覆盖旧 attempt。

保留现有 0.5% tie、固定相等存储下优先较小 Pe/ID、预测收益至少 2% 的规则。
本轮存储固定；以后扩展不同槽数搜索时才按资源需求补全 tie-break，不能直接套用相等存储假设。
无优化推荐时记录 BASELINE_RETAINED，不伪造 B。

建议增加统一显式 `vf optimize` 入口，复用已有 export/analyze/annotate/compile API。
模型仍是独立进程，用户可离线重放每个阶段。
每次运行保存 prepared、候选、VF 成本、初选、冻结 Plan、attempt、编译反馈、最终评估及完整调用事件记录。

## 6. 必须新增的验收

| 用例 | 预期 |
| --- | --- |
| 首次推荐 | 记录 freeze→compile 的因果顺序；不是仅比较文件修改时间 |
| 禁止提前设备编译 | 调用跟踪证明初选前无 Bisheng/链接/对象生成；只检查文件后缀不算通过 |
| VF 复用 | 每个输入仅 export/prepare 一次；相同成本键只评估一次，修改条件必定失效 |
| 未选候选 | 正常路径下没有设备对象/launcher；设备编译次数为 1 |
| 调度一致 | N/P 边界、阶段顺序和版本与原生应用一致；FA 的 m/l/O 递推不被当作独立迭代 |
| 资源与完成 | 不足槽、容量、unknown、borrowed entry、两 AIV 分域、完成条件与 FIFO/依赖环按原要求检查 |
| 最终反馈变化 | 注入新增搬运/同步/布局条件变化后，旧预测不可复用；重评仅针对当前候选 |
| 失败处理 | 新副本、有界尝试、baseline retained、显式 Plan 报错及协议损坏终止均可重放 |
| 兼容 | v1/2.0、现有 annotation、旧物理诊断 adapter 和普通 EmitC/VPTO 行为不变 |

保留现有协议、native CV、内存与同步回归；按实际源码重新枚举，不沿用历史通过数。
涉及 VF/同步/后端代码时，按 PR 规范补当前 HEAD 的 VF A/B/C/D 子图矩阵。
旧冻结编译器数据不替代新编译器门禁。

## 7. 新实验和报告

首个性能 workload 沿用同事的 M32 FA。TileSim 在任何新候选设备编译和新上板数据前给出初选。
沿用已授权的 PR 时序参数代理 DT，明确记录覆盖限制；暂无 DT 参数不阻止代理对比，也不签发目标匹配 G4。
主机、健康/ACL、CANN、Bisheng、输入和同步重新检查，所有新数据使用独立目录。

G2 先验证选中候选；通用 G2 为 unknown 时，必须有覆盖此输入的独立受限证明才可送板。
推荐冻结后另行生成 baseline 对照。在任何 baseline 运行前先执行一次 B 冷启动，
随后六组输入、两个反向顺序各测 A/B：共 25 次 G3。
独立 FP64 golden、常量 atol=1e-6、pattern atol=1e-4/rtol=1e-3、输入/guard/稳定性及单次 60 秒限制沿用现有口径。
正确性失败立即停止性能阶段。

性能 A/B 各预热一次，20 个交替 AB/BA block，共 40 个正式样本；每次恢复并校验输入输出，严格核对 MIX_AIC task。
按固定种子 20260916、10000 次 bootstrap 报告收益和 95% 区间。
平均收益至少 2%、区间下界大于零、最坏配对回退不超过 2% 才判 AB_BENEFIT_PASS。
模型候选/基线时延误差、收益预测误差及实际收益分别报告；不能预设必须复现 33.04%。

同时测量 VF 准备次数、成本缓存命中、候选分析时间、完整编译次数及总寻优时间。
正常路径验收要求 VF prepare=1、初选前设备编译=0、选中候选设备编译=1；实际时间报告测量值，不预设加速倍数。

推荐冻结后，可另建 evaluation-only 批次编译其余 P，验证排名；禁止用这些设备结果重选后复用同批验收。
绝对性能对齐另建同批对照，运行同事交付版冻结 P1/P4 与我们的原生候选，使用同板、同输入、同预热和同 task 口径。
保留两条路线各自的同步实现，作为待分析差异，不能为制造相同条件随意改掉对方冻结程序。
比较绝对时延和生成指令/等待/搬运/布局，不因原生组的相对收益更大就宣布替换性能通过。

## 8. 完成定义和交付

1. TileSim 仅凭共享 prepared 包和轻量候选返回真实 Plan，无任何 final feedback 前置依赖。
2. 顺序、单次 VF 准备、单候选正常编译以及分析/应用一致性均有自动化证据。
3. 最终资源、完成条件、反馈有效性和兼容回归通过；不能把编译成功当 G2。
4. 新 G3、A/B、模型误差和寻优成本报告完成，收益或精度失败均明确呈现。
5. ADR/schema/代码/fixture/Plan/调用记录/原始 profiler/重放命令齐备；更新 PTOAS #1676 和 TileSim #142 的最终范围及待办。

截至本计划创建，上述 S1–S6 均为待实施；历史“编译后评估”流程及其收益保留为诊断参考。
