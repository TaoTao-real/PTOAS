# Prepared VF 与延迟内存绑定接口

本页描述 MR289 集成分支的新实现。它不替代已有 v1/2.0 文件协议。
新接口的实验扩展名为 `pto.cv.prepared_exchange.v1`。TileSim adapter 必须显式支持
该扩展后，才可消费这里的 VF 和 schedule；原 v2 adapter 不能假装已支持它。

## 执行顺序

```text
level2 无地址 PTO
  → 前端分组 / singleton VF 边界
  → 参数化 VF：TileLib 展开、VMI 循环融合、访存转发、VPTO 准备
  → 导出 canonical.pto + 结构化 program.json
  → 编译器分析指定 preload 的阶段顺序和最低 local 槽数
  → cost model 评估该 schedule 并返回完整 Plan
  → 回填 annotation
  → 原生 CV pass：写版本、槽位和实际外层调度
  → 原生 PlanMemory：local / VF scratch / pipe backing
  → InsertSync、vecscope membar、VPTO / LLVM / device object
  → 同次编译反馈
```

普通编译流程保持原有顺序。新流程必须显式选择；不是把全局 PlanMemory 移到后端。
VF 的外部 Buffer 通过指针参数传入。借入 FIFO entry 是访问接口，不能增加 local 槽数。
编译器从 VF 指令体重新推导效果，不信任外部填入的 read/write 字符串。

本版支持 A5、静态 shape、单轴独立四阶段循环。自动 TileOp 和手写 VMI/VF 使用同一
参数接口。当前指针边界承诺 32 字节对齐；需要更强证明的地址敏感指令会被拒绝。
未知 alias、未知调用效果、循环递推、嵌套调度与不足槽数均不能静默优化。

**当前不支持原始 FA 的 Q×KV 投影、m/l/O 递推及旧显式地址的自动重定位。**
这些输入继续走原编译路径。level3 显式地址不能作为新自动 PlanMemory 的通过证据。

## 文件与命令

先提供源 PTO、bindings 和目标文件。目标最小格式：

```json
{"arch": "a5", "cann_version": "9.2.0"}
```

使用构建树的 `ptoas` wrapper 和匹配的 Python（PTODSL 运行需要 Python ≥ 3.10）：

```sh
ptoas costmodel vf capabilities
ptoas costmodel vf export source.pto bindings.json target.json package --compiler /absolute/path/to/ptoas
```

输入包包含 `canonical.pto`、`program.json`、`bindings.json`、`target.json` 和 `manifest.json`。
程序来自 MLIR 对象，包含类型、操作数/结果身份、区域、循环、VF 效果和 borrowed entry。
未知语义显式为 unknown；它不等于零成本或零资源。
Manifest 绑定这些文件、原始源文件、CANN 和实际选中编译器的二进制身份。

`program.json` 的非 VF 内部循环有稳定 `id`。例如，将所有配对循环的 preload 设为 2：

```json
{"cube:loop:0": 2, "vector:loop:0": 2}
```

实际 ID 以导出文件为准。不能从 SSA 名或最终地址推导 ID。

```sh
ptoas costmodel vf analyze package preload.json analysis --compiler /absolute/path/to/ptoas
```

`analysis/schedule.json` 返回编译器构造的 Prefix/Suffix 实例、源操作 ID、写版本、槽位、
最低槽数及完整 Plan 模板。其状态为 `schedule_only`，G2 为 `not_checked`。
`scheduled.pto` 是供评估的候选描述，不能作为新的 prepared checkpoint 直接回放。

语义固定为 `Pe=min(P,N)`，同一步先 Prefix(t)，再 Suffix(t-Pe)。槽数由编译器按实际
写版本求得。模型评估这些实例，不自行假定 `preload_depth=P+1`。

模型可更改 candidate ID，并在合法范围内增加槽数；必须返回所有 local Buffer 的配置。
Plan 必须携带输入指纹、schedule 语义版本和分析所得的 `schedule_fingerprint`。
编译后将实际 schedule 与该指纹比较，不一致时拒绝签收该产物。模型预测和搜索参数保留在模型的外部报告。

```sh
ptoas costmodel vf annotate package plan.json annotated.pto
ptoas costmodel vf compile package plan.json candidate --compiler /absolute/path/to/ptoas --emit llvm
```

`annotate` 只回填 annotation；重复执行不会叠加属性。`compile` 明确执行原生变换与内存规划，
输出 `annotated.pto`、产物、`compiler_feedback.json`、`compilation.json`、Plan 和日志。
`--emit` 可选择 `vpto`、`llvm`、`object`。输出目录必须是新目录。
设备对象要求显式 `pto.entry`；C/V 混合入口必须使用相同的参数顺序和逻辑 kernel 名。
示例工具会生成共同 Q/K/V/O ABI 的 `_mix_aic` / `_mix_aiv` 入口，不把普通 device 函数误报为可启动 kernel。
所有配置、旧输入、未知 ID、重复 ID、非法数量和物理地址注入均严格校验。

## 同次编译反馈与安全边界

`compiler_feedback.json` 的格式为 `pto.cv.compiler_feedback.v1`，独立于模型 Plan。
它记录实际 native PlanMemory 的调用、Buffer/槽位的地址、对齐、compiler budget、
pipe backing、两 AIV 的私有存储实例、最终 schedule、降低后 IR 和产物摘要。
两个物理核的相同数值地址不合并。关闭 PlanMemory 后，活跃的无地址 allocation 会报错。

反馈的 `g2_status` 当前固定为 **`evidence_incomplete`**。当前实现尚未提供完整异步访问完成、
释放和跨核 FIFO 环的联合证明。因此，产物编译成功只证明该编译路径执行成功，不能签收 G2/G3，
不能据此启动性能认证或自动启用。`prediction_reusable_without_final_evaluation=false`
要求模型核对最终编译约束，不能继续沿用初评时延。

已有同步 oracle 已补充 prepared VF 调用的效果和 pipe 识别；这不单独构成完整 G2。
真实 TileSim/VfSimulator 推荐、最终反馈消费、FA 支持和新设备 G3/A/B 仍是后续验收项。
旧实验数据不能替代这一分支的验收。

## 回归入口

- `test/lit/vpto/cv_vf_auto_fusion.pto`：自动 TileOp VF 先实际融合，再自动规划。
- `test/lit/vpto/cv_vf_late_memory.pto`：手写 VMI、规划关闭负例、旧 checkpoint 拒绝、反馈绑定。
- `test/lit/vpto/cv_vf_exchange.pto`：四阶段 crossing 的真实导出、最低槽数分析和 LLVM 编译。
- `test/lit/pto/cv_static_schedule*.pto`：N/P 边界、非零下界、非单位步长、递推和不足槽数拒绝。
- `ptodsl/tests/test_cv_prepared_contract.py`：严格 Plan/JSON/身份契约。
- `pto-memory-range-test`：reserved backing 的 hole allocation 占用登记、容量、对齐和溢出。
- 原 v1/v2、旧固定 G2 和计时边界测试继续独立运行，不把其通过数算成新 prepared G2。
