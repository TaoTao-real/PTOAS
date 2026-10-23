# VF × CV × PlanMemory 实现状态（2026-10-23）

当前结论：**固定 M32 FA 已完成真实代理模型推荐、原生 VF＋PlanMemory、受限 G2、G3 和同板 A/B，配对平均降时 33.04%。通用 FA 支持与目标匹配 G4 仍未完成。**

新增验收见 [固定 FA 代理模型与实测报告](cv-vf-fa-model-acceptance-20261023.md)。本轮 49 次 G3、80 个正式 profiler 样本全部正确。模型推荐 P=1（本图旧 depth=2），候选绝对时延误差 3.36%，基线误差 13.31%，收益预测少 7.75 个百分点。PR 参数仅作 DT 代理，不签 G4。

新增验收见 [固定 micro 上板报告](cv-vf-native-micro-acceptance-20261023.md)。实验代码冻结于 `f8c9cbcd0`，B/A 配对平均时延下降 15.46%，B/P0 下降 14.81%；正式 G3 24 次、profiler 80 次全部正确。

以下编译记录来自先前 `4e3a50728`，不将旧回归改记为本轮执行。
实现基于 `f4df5091f`（主线 `37f5f25e3`＋MR289 `73c820689`＋兼容修复），
分支为 `codex/cv-vf-planmemory-integration`。

## 已实现的行为

| 工作 | 实际结果 |
| --- | --- |
| 融合前的逻辑存储 | local 无最终地址；VF 用参数访问存储，不再要求预先填写地址表 |
| VF 准备 | 自动 TileOp 和手写 VMI/VF 统一为 private prepared VF；实际执行循环融合与访存转发 |
| 完整 C/V 视图 | 配对函数保留到 CV 变换、规划和同步之后，后端边界再划分编译单元 |
| 导出与回填 | 结构化程序、稳定 ID、指纹、完整 Plan、严格字段检查、幂等 annotation |
| 候选分析 | 编译器生成 Prefix/Suffix 顺序、源操作实例、写版本和最低槽数；不猜测模型 depth 的含义 |
| 原生变换 | 根据 preload 和逐 Buffer 配置实际展开调度、选择槽位，不再依赖旧固定 Python materializer |
| 自动 PlanMemory | 真正运行 native pass，核算 local 与 pipe backing；关闭规划则无地址活跃对象报错 |
| 来源与反馈 | 槽位保留来源 ID；同一次编译反馈实际布局、两 AIV 实例及产物摘要；核对分析和最终 schedule |
| 兼容 | 迁入旧 v1/v2 SDK；旧 checkpoint 保留原规范化语义；默认普通编译流程不切换到新实验路径 |

附带修复：reserved buffer 分配到空洞时，旧算法没有登记新占用范围。
新算法登记所有成功分配，并检查容量、重叠和对齐溢出。这里是静态复现及单元验证，
不把它描述为已复现某个 FA 板端错误的根因。

32 列 ND→NZ 的 VMI template 原先会选择 PTODSL 不支持的逻辑 mask 宽度。
现在先过滤该候选，使用已有 MI template；没有放宽 mask 或写越界检查。

## 具体示例

固定 crossing micro：N=4、P=2，32×32 FP32，1 AIC＋2 AIV。
本例是固定候选编译测试，没有经过 TileSim 排名或推荐。

- 编译器对两个 core projection 都生成 `Pe=2`。
- P local Buffer 需要 3 槽，实际地址为 0、2048、4096，每槽 2048 字节。
- 其他 7 个 local allocation 均为 1 槽。
- 每个 AIV 的 VEC 地址高水位为 43008 字节，包含两个 pipe backing；两个 AIV 分开核算。
- AIC 的 MAT 地址高水位为 24576 字节；LEFT、RIGHT、ACC 各为 4096 字节。
- 生成了 LLVM 和 CANN 9.2 device object。对象含共同 `native_cv_vf` 入口，
  以及 `native_cv_vf_mix_aic` / `native_cv_vf_mix_aiv` kernel 元数据。
- Plan、schedule、native feedback 和对象的 SHA-256 已逐项核对。

完整样例对象 SHA-256：
`bda7826b97b4fdb39263f7bf550d0e18464038dec7a52da4f27b1401e7b12cd7`。

## 验证与证据

- 本地相关 lit：67/67，通过。涵盖 memory、multi-buffer、VF、同步 oracle 和旧接口。
- A5 编译主机集成测试：6/6，通过；随后补充了共同 kernel ABI 的 LLVM 集成测试并通过。
- 新纯契约单元：12/12，通过。
- 旧接口的四个 lit 入口覆盖当前源码中的 38 个 v1、33 个 v2、16 个固定 G2、8 个运行/统计测试。
  旧固定 G2 的通过不能转记成新 prepared VF 的 G2。
- native 阶段测试覆盖 20 个 N/P 组合、非零下界、非单位步长、不足槽数和递推拒绝。
- crossing 固定候选的同步 ID、互扰、依赖覆盖检查通过；AIC 394 个依赖全部覆盖，
  AIV 71 个依赖中 58 个显式覆盖、13 个按架构排序，无 uncovered 或 unmappable。
  这是局部同步检查，不是完整 G2。
- N=0 和 N=1 的公开重放入口都生成了 LLVM；N=0 无阶段实例。
- A5 编译主机全量 lit：2281 项中通过 2278，3 项按 VfSim OFF 配置不支持，失败 0。
  首轮缺测试工具的中断已修复；不将当时未执行的测试计为通过。
- 原始 FA 输入通过普通 level2 EmitC 对象构建；用户提供的 M32 VPTO 输入通过普通
  level3 VF 融合路径生成 LLVM。这两项只验证原编译能力，没有启用新 CV 路径。

本机证据目录：
`/Users/lishengtao/Documents/PTO/_ptoas_tech_lab_records/profiles/cv-costmodel/experiments/20261022-005-vf-cv-automatic-memory`。

远端编译证据目录：
`~/private-ptoas-lab/experiments/20261022-cv-vf-late-memory-build`（`ptoas-a5-39`）。
上述旧记录仅为编译验证，没有执行设备 kernel；本轮新增的板端证据与结论见本页开头的验收报告。

## 验收状态与剩余条件

| 阶段 | 状态与缺口 |
| --- | --- |
| G1 | 固定 FA 的真实 TileSim/VfSimulator 返回 Plan 已回填；annotation 与已编译候选逐字节相同 |
| G2 | 固定 micro 与固定 M32 FA 均通过各自受限证书；FA 增加独立物理尾部、事件/访存完整性、flag 与不重叠布局检查。通用反馈仍为 `evidence_incomplete` |
| G3 | micro 24 次与 FA 49 次分别通过；FA 使用 C++/NumPy FP64 golden，常量、随机、反向顺序及 guard 检查均通过 |
| 性能 | micro 固定候选 B/A 15.46%、B/P0 14.81%；FA 真实推荐 A/B 降时 33.04%，95% 区间 32.52%–33.55%。G4 不认证 |
| FA | SHA 限定的 M32 原件已完成 32-token 投影、m/l/O 版本和旧地址相对视图导入；不是任意 FA 的正式通用入口 |
| 模型反馈 | 固定候选同次编译的最终 VPTO/布局/同步已由真实模型消费；通用反馈迭代、DT 参数认证与自动启用未完成 |
| 环境 | VfSim ON、CANN9.1/950PR 和 A3 新板测尚未验收 |

固定 micro 是手工候选的机制验收；本轮固定 FA 是真实代理模型推荐验收。两类证据分别保留，不能互相替代，也不能扩展到其他输入或目标。

使用方式与约束见 [prepared interface](cv-vf-prepared-interface.md)；
后续门槛仍按 [实施计划](cv-vf-planmemory-implementation-plan.md) 执行。
