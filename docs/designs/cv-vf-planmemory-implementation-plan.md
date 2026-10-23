# 支持融合 VF 的 PTOAS × Cost Model 实施与测试计划

日期：2026-10-22
状态：MR289 基线上的受限 native 实现与回归进行中；完整签收尚未完成
设计：[adr-cv-vf-late-storage-binding.md](../adr/adr-cv-vf-late-storage-binding.md)

## 1. 交付目标

同一份无 local 物理地址的 PTO，通过一次可重放流程完成：

```text
PTO → 自有 VF 准备 → 模型评估/推荐
    → preload 与逐 Buffer 槽数实际变换
    → 自动 PlanMemory 与同步校验
    → 优化代码 → 独立正确性和 A/B 测试
```

输入同时覆盖 Cube TileOp＋自动融合 VF，以及 Cube TileOp＋前端手写 VMI/VF。
自动与显式地址模式分别验收。固定 FA 地址脚本不进入正式优化路径。

## 2. 基线与分支

- 正式实现分支：`codex/cv-vf-planmemory-integration`，从已验证的 `f4df5091f` 开始。
- 该基线已包含主线 `37f5f25e3`、MR 289 头 `73c820689` 的移植，以及本轮兼容修复。
- 按用户确认，主线 rebase 不作为本轮前置门槛。优先在 MR 289 功能基线上实现。
- 2026-10-22 核对的主线是 `eea8f91ac`，MR 289 尚未合入。
  主线新增崩溃回溯与 scalar cast 修复单独核对，低风险修复可移植；主线同步不阻挡核心功能。
- 老实验分支 `codex/cv-vf-costmodel-integration` 和已冻结证据保留。
- 既有 v1/2.0 与 G2 代码参照 `cv-costmodel-stage1` 的 `630e7ea3b`，按接口移植，
  不整分支覆盖新 driver 或恢复旧 PTO 语义。旧测试必须在新分支重新运行。
- 交付包参照为 `cv-vf-delivery-reproduction` 的 `988f723ba`。
- 核对的 TileSim 分支 `dev_zzh/tilesim_refactor` 为 `b27ebf863`。
- 核对的 VfSimulator master 为 `f19615c69`；MR 289 记录的 submodule 为 `f4ba4ecaf`。
  两者分别做 ABI/接口兼容检查，最终使用版本写入构建与模型清单。

正式候选编译与设备验收前重新冻结源代码、模型、目标和输入。
本计划中的版本是观察/起点，不自动代表将来验收所使用的版本。

## 3. 首发范围

- A5 VPTO，静态 shape，1 AIC＋2 AIV，tile entry。
- 无地址 level2 为自动规划主验收；level3 为显式地址兼容路径。
- 旧 FA 已有地址时，默认保留约束；显式重新规划模式只重绑经证明可重定位的 local。
  同源无地址 FA fixture 或编译器受控导入用于自动规划验收，不能只换 level 开关。
- basic/crossing 四阶段 micro，逐轮不同输入；M32 FA 原始文件直接进入正式入口。
- FA 支持明确的 Q×KV 静态投影、组边界和已识别 m/l/O 状态递推。
- 至少第二种合法 shape 和不同 Buffer 次序，证明实现没有绑定旧 FA 地址表。
- 默认不启用核内联合搜索；先冻结 VF 分组、layout 与 unroll，再搜索 CV。
- A3、EmitC 和不启用 cost model 的普通路径保留既有能力。
  950PR/CANN9.1 只有在对应环境实际通过后才列为已验证。

## 4. 按依赖实施

| 阶段 | PTOAS 改动 | 可验收产物与通过门槛 |
| --- | --- | --- |
| P0：契约与最小可行性 | 固定 VF 调用 ABI、效果查询、完成条件和检查点；审计各 pass 对地址的依赖 | 一个无绝对地址的 VF 先融合，再真实运行 PlanMemory；以两组合法规划结果编译验证布局和向量语义；原始包装作用域不用外部脚本修改 |
| P1：VF 边界和来源 | 显式 VF 参数、读写/alias/scratch 摘要、Buffer/stage ID；自动/手写统一；延迟 C/V job 拆分 | Cube＋VF 混合 IR、实际 VF 指令包、来源表；未识别调用或跨界值明确拒绝；不会重复融合 |
| P2：原生 CV 变换 | 基于统一依赖/效果接口消费 Plan，构造 preload 调度及逐 Buffer 版本/槽位 | 正式 pass/API 输出实际结构；slot 1/2/3 精确对应 Buffer；支持范围内无需 fixture materializer |
| P3：自动分配与同步 | 分离逻辑存储分析和物理绑定；扩展 PlanMemory/同步识别 VF；核算 backing/scratch/预留 | native 证据证明 PlanMemory 真正运行；自动地址、完整资源报告、完成事件、最终依赖/FIFO 检查；unknown 不通过 G2 |
| P4：模型与反馈 | 结构化混合程序/VF 导出；兼容旧协议；接 TileSim/VfSim；返回最终编译反馈并重评 | 真实模型返回绑定 Plan，正式变换消费；不支持成本为 unknown；候选、调度、最终程序与预测指纹一致 |
| P5：设备正确性 | 正式路径构建 micro、FA 与手写 VF；独立 golden、guard、重复和反向顺序 | 通过新 G3；原始源码无需人工改地址；旧实验只能作为参照 |
| P6：性能和交付 | 同批消融、模型推荐 A/B、新旧整链对照；冻结版本、脚本和原始数据 | 分开签收收益、预测误差和替换性能；交付 reviewer 可复现 PR、命令、报告及证据包 |

P0 是结构风险门槛。未通过时，先修复参数化/布局或调用接口，不能继续以 level3
绕过自动规划后宣布 P3 完成。模型成本覆盖开发可与 P2/P3 并行，但真实联合签收依赖 P4。

### 核心实现落点

- `tools/ptoas/driver.cpp`：完整 C/V 视图、阶段入口、后端拆分时机与产物身份。
- `tools/ptoas/ptoas_pipeline.cpp`：在显式优化模式组装新流程，保留默认流程。
- `lib/PTO/Transforms/TileFusion`、`VMI`、`VPTO`：VF 参数化、准备与保护边界。
- `lib/PTO/Analysis` 和必要的接口声明：统一 VF 读写、alias、完成条件与资源摘要。
- `lib/PTO/Transforms/Passes/PTOPlanMemoryModern.cpp`：VF 调用的存储根、槽、scratch 和复用分析。
- `lib/PTO/Transforms/InsertSync` 及 pipe lowering：实际完成依赖和最后释放的同步证据。
- `ptodsl/ptoas` 与协议包：CLI/Python 编排、导出/导入/重放，语义变换由编译器校验。
- TileSim adapter：混合程序、两 AIV、VF 成本及 channel 生命周期；VfSimulator：实际 VF 评估。

先评估并复用交付包 `_cv_vf_first`、`_cv_preload`、`_cv_storage` 的契约和测试，
去除固定模式假设。补齐缺失的私有脚本依赖，正式复现命令只依赖冻结仓库和声明的工具链。

## 5. 必须补齐的接口

1. **prepared VF**：参数与来源、类型/layout/alias/对齐、指令、scratch、完成条件、条件指纹。
2. **混合程序**：Cube/VF/COPY/pipe 节点、读写、循环域、递推边和迭代距离、两 AIV 实例。
3. **候选计划**：输入身份、统一调度语义、preload、完整逐 Buffer 槽数和 schedule 指纹。
4. **最终反馈**：同次编译的 memory/validation 报告、地址复用及完成证据、临时存储、
   新增等待、最终执行约束、C++/LLVM/对象的共同身份。
5. **模型报告**：同一最终候选的时延、单位、资源依据、覆盖和未知项、模型/adapter/config 版本。

由 MLIR 对象直接生成结构化 `program.json` 和 VF 描述，不要求 adapter 重新解析
SSA/MLIR 文本。输入包、VF 指令包、候选 schedule、memory/validation 报告和最终产物
分别保留版本与摘要。最终文件名及 schema 在 P0 冻结。

保留现有 preload/槽数 annotation。模型排名、预测和搜索预算放在外部 Plan/报告。
默认 annotation-only 不变；显式 apply 失败时报错。
自动调优可以淘汰不可行候选并返回基线；协议损坏或身份不匹配必须终止。
未认证范围不默认自动启用。

preload 使用编译器定义的调度。#1292 的迭代距离与旧 `preload_depth` 必须独立记录。
以 schedule 评估和核对为准，不能默认使用 `depth=P+1`。

## 6. 编译与契约测试

| ID | 测试 | 必须满足的预期 |
| --- | --- | --- |
| T01 | 自动融合、手写 VF 两种输入 | 使用相同效果/资源契约，生成 Cube＋VF 混合描述 |
| T02 | 原始作用域和边界 | 无嵌套错误、mask 不逃逸；strict/指定边界保持；不调用 fixture 预处理 |
| T03 | 无地址 allocation | native PlanMemory 实际执行并为所有活跃存储赋址；关闭规划则明确报错 |
| T04 | level3 显式地址 | 原能力保留；冲突被拒绝；报告不能标为自动规划 |
| T05 | 换地址、第二 shape、不同 allocation 次序 | 参数化 VF 正确，布局/条件核对通过；没有旧地址或 SSA 名依赖 |
| T06 | SSA 改名、alias、被融合消除的 Buffer | 身份映射正确；alias 按根核算；消除对象有记录、不伪造分配 |
| T07 | 1/2/3 槽与非法槽数 | local 根准确展开；borrowed entry 不被错误展开；不足槽数拒绝 |
| T08 | preload 改变存活重叠 | 按最终调度分配；不覆盖仍在使用的数据；复用所需等待可追溯 |
| T09 | 异步读已发射但未完成 | 不允许覆盖；调用返回不单独作为完成证明 |
| T10 | 加等待导致依赖/FIFO 环 | 最终检查拒绝，不能仅凭原调度通过 |
| T11 | 两 AIV、pipe backing、VF scratch、编译临时存储 | 分物理域核算，不漏算或重复算；单项可放但总体超额被拒绝 |
| T12 | 求和超预算但复用可放 | 有界真实规划成功才确认可行；超时/策略失败不等同确定不可行 |
| T13 | N=0/1/2/4/8、P=0/N/>N | 按声明语义生成；N=0 不调用空图 policy；跨 Q 状态和组边界正确 |
| T14 | stale/未知/重复 ID、目标或绑定变化、地址注入 | 严格拒绝；重复导入幂等 |
| T15 | VF、layout、unroll、同步或后端变化 | 成本/反馈缓存正确失效；不能继续使用旧完整预测 |
| T16 | 不支持的操作、alias、完成条件或成本 | 明确 unsupported/unknown，不按零成本继续推荐或通过 G2 |
| T17 | 候选失败和后续尝试 | 从原始模块新副本重试；不直接减槽或发布失败后的半成品 |
| T18 | 最终报告与产物不匹配 | 拒绝复用或签收；反馈来自同一次 native 编译 |
| T19 | 默认编译和旧接口 | A3/EmitC/VPTO、嵌套 C/V、原始 FA、旧 v1/2.0 保持既有行为 |
| T20 | 旧显式地址重新规划 | 必须显式选择策略；alias/ownership 已核对；pinned/地址可观察对象不能被静默搬动 |

既有 12 项基础、38 项 v1、31 项 v2、15 项 G2 和 4 项计时边界测试作为迁移清单，
在新分支重新收集并执行，记录实际数量及跳过原因，不沿用历史通过数。
执行相关及完整可用 lit/Python/sample 回归。分别测试 VfSim 构建开关 OFF/ON。

## 7. 设备正确性

主验收设备优先 `ptoas-a5-39` device 0，CANN9.2。每次先核对实际 SOC、工具链来源、
ACL 探测、独占队列与目录权限。更换主机则新建实验、重跑全部对照。

- micro 的 basic/crossing 使用逐轮不同 Q/K/V，固定三个种子，独立 NumPy golden。
  在 FP32 整数精确范围内要求精确相等；错误版本/槽位必须能被检出。
- FA 覆盖 zero、one_v、one_qkv 和三个随机种子。保持已声明的 FA 数值口径，
  不以放宽容差修复错误。相同算法及变换顺序下要求重复稳定。
- 自动/手写 VF、PlanMemory 自动/显式地址、VF on/off、P1/P4 均有正式路径覆盖。
- 各候选至少两次、两种反向顺序；包含候选先运行、重置输入/哨兵和独立目录。
- 检查所有迭代输出、输入未改、GM guard 未改、运行错误和超时。
  guard 不单独替代静态访问范围及完成条件证明。
- 单次 60 秒超时；任一错误停止该 workload，保留证据，不采集性能。
- 共享内存/driver 改动增加 A3 默认路径最小板端 smoke；环境未通过时列为待验，不能沿用旧通过数。

## 8. 性能与模型分别验收

先做两个层次，所有参与统计的程序都先通过 G2/G3：

**消融层：** VF on/off × P1/P4 × 自动/显式地址，最多 8 个变体。
按平衡顺序在 20 个 block 中运行，同 block 同种子。
比较 VF、preload 和自动绑定各自影响。显式地址只作对照，不替代自动规划。

**真实推荐及替换层：** TileSim 在新设备计时前返回完整排名和推荐，并冻结摘要。
模型 baseline-retained 是合法功能结果，但不能记作优化收益通过。
对自动规划的新候选、同条件新链路基线和重新构建的交付包参考做同批交替测量。
不能以旧 62.28 μs 或旧 77.49 μs 代替此次样本。

先预热，每个主比较保留 20 个配对。保存原始 profiler、task duration、wall time、
实际频率、顺序、种子和输出摘要。每次检查唯一匹配的 MIX_AIC task、1 AIC＋2 AIV。

统计保持 mean(1-B/A)、bootstrap 10000 次、seed=20260916、95% 区间。

- 优化收益：平均至少 2%，区间下界大于 0，任意配对回退不超过 2%，全部正确。
- 整链替换：对同批交付包最优参考单独报告平均及最坏回退，目标均不超过 2%；
  达不到时不能宣布性能替换通过，保留功能与正确性结论。
- 模型精度：对同一最终候选的绝对时延误差不超过 10%，并报告收益预测偏差和排序。
- 模型精度、覆盖或认证配置不足时，G4 未认证；不能用编译成功替代性能签收。

若使用独立原语校准，冻结其版本，并在新的推荐和设备实验前完成。
用本轮计时调整候选后，需要新实验，不能复用本轮样本签收。

## 9. 分工与可审查改动

PTOAS 负责 VF/Buffer 语义、实际变换、PlanMemory、同步、安全证明及最终反馈。
TileSim/VfSimulator 负责合法程序的成本覆盖、候选搜索、预测与模型配置。
双方共同维护 schema、输入包、命令和同一候选的联合签收记录。
参考 adapter 可做契约测试，不能代替真实模型验收。

建议拆成四组 PR：

1. VF 参数接口、效果分析、作用域与来源维护。
2. CV 正式变换、PlanMemory 和同步验证。
3. 模型交换兼容、真实 adapter 与反馈消费。
4. 自动化回归、板端工具、性能报告及用户说明。

## 10. 最终完成定义

以下条件全部满足，才能称为“满足 cost model 要求的 VF 支持版本”：

1. 原始支持输入直接走正式入口，无人工编辑地址及固定 FA helper。
2. 自动和手写 VF 都能进入真实模型评估，模型 Plan 能作用于最终结构。
3. 至少主验收组确实运行 PlanMemory，所有槽、backing、scratch 有合法绑定。
4. 实际完成、释放和复用依赖满足，G2 不含影响安全的 unknown。
5. 新 G3、回归及负例通过，兼容差异有明确范围。
6. 新推荐候选的收益与同批整链替换门槛通过；模型精度单独签收，不合格则不签发 G4。
7. 冻结源码、模型、配置、输入、Plan、最终反馈、对象、launcher、原始数据与复现说明。

阶段进度按上述门槛更新。2026-10-23 已实现参数化 VF、受限单轴 native 调度、
候选最低槽数分析、实际 PlanMemory、同步分析的 VF 调用识别和同次编译反馈。
已迁入旧 v1/2.0 SDK 和回归，新增独立的 prepared VF 实验扩展。
使用说明见 [prepared interface](cv-vf-prepared-interface.md)。

上一轮编译阶段状态（后续进度见下方验收补充）：P0/P1 已有自动与手写 VF 的编译证据；P2 仅支持独立单轴四阶段循环；
P3 已执行实际规划，但完整异步完成与 C/V FIFO 联合证明尚缺，G2 保持 evidence_incomplete；
P4 只有新交换与 schedule/产物绑定，尚无 TileSim/VfSimulator 的真实联合推荐；
P5/P6 的新设备正确性与性能验收未运行。
FA 投影/递推、旧地址重定位、VfSim ON 构建和 A3 新板测仍未完成。
旧固定 FA 实验不计入这些新门槛。
若只满足接口、G2/G3 与复现交付，可签收“编译器功能支持”，不能宣布上述完整性能验收完成。


## 2026-10-23：固定 micro 验收补充

本轮继续执行至设备验收，结果见 [上板消融报告](cv-vf-native-micro-acceptance-20261023.md)。

- P3：固定 crossing micro 的受限 G2 通过；真实 PlanMemory、完成屏障、绝对同步覆盖、联合 FIFO 和不重叠物理分配均有绑定证据。通用异步与复用证明仍未完成。
- P5：N=8、P=2、P 三槽的 A/P0/M/B 正式 G3 24/24 通过。
- P6：20 个四组交替 block、80 个 profiler 样本完成，机制收益通过；B/A 15.46%，B/P0 14.81%，M/P0 0.73%。独立离线复核通过。
- P4：新 prepared-VF 接口的真实 TileSim 推荐及编译后模型重评仍未完成；本轮候选为预固定配置，G4 未认证。
- FA、完整 M01–M12、通用地址复用、其他 shape/target 和自动启用继续保持未完成。

## 2026-10-23：固定 FA 真实代理模型闭环

结果见 [固定 FA 验收报告](cv-vf-fa-model-acceptance-20261023.md)。本补充更新上一轮“FA/真实推荐未完成”的状态，不扩大通用支持范围。

- P2：新增原生静态 stage graph 与共享存储绑定。SHA 限定的 M32 FA importer 保留 32 个 token 的 Q/KV 投影、m/l/O 递推和相对行视图；不是任意 FA 识别器。
- P3：同次对象编译导出最终 VPTO；真实 PlanMemory；私有 row 读取尾部、不复用布局、完整事件与访存见证、flag/完成条件及预算检查通过。通用复用证明仍未完成。
- P4：真实 TileSim EventEvaluator＋VfSimulator 消费最终程序，在首次 FA 上板前冻结 P=1 推荐；模型 Plan 回填与编译产物一致。DT 参数缺失，用户授权 PR_9599 代理评估 DT_9592。
- P5：新 FA G3 49/49，通过两套 FP64 golden、全 1、三个种子、候选先运行及反向顺序。
- P6：新正式批次 20 block、80 profiler 样本全部正确；配对平均降时 33.04%，95% 区间 32.52%–33.55%，收益通过。候选时延误差 3.36%、基线误差 13.31%，收益预测少 7.75 个百分点。
- 本轮固定槽位，只搜索 preload；模型数据和原始证据冻结。G4、DT 参数、逐 Buffer 搜索、通用地址复用、任意 FA 正式入口和同批交付包整链替换继续未完成。
