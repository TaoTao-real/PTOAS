# PTOAS 自动同步（InsertSync）测试串讲（面向测试同事）

## 1. 文档目的

这份文档给测试同事提供一套可落地的自动同步验证方法，回答 4 个核心问题：

1. 自动同步到底在做什么，测试要看哪些信号。
2. 如何快速判断“正确性风险”与“仅同步数量偏多”。
3. 如何稳定复现与对比（A3/A5、InsertSync/GSS、自动/手工）。
4. 如何给研发提交高质量问题单，缩短定位时间。

本文聚焦 `PTOAS` 当前代码基线，不要求测试同学先读完整实现代码。

---

## 2. 一页结论（给测试执行前先对齐）

1. 自动同步的第一目标是正确性，不是最少指令数。
2. “同步数量变多”不等于错误；“结果错/挂死/死等”才是 P0。
3. 验证顺序固定为：先正确性、再数量、最后性能。
4. 回归重点不是单条指令，而是控制流边界上的成对语义是否匹配（尤其 loop/if-else）。

---

## 2.1 设计目标（测试验收基线）

InsertSync 的设计目标按优先级分三层：

1. 正确性目标（必须满足）
   - 对真实数据依赖提供可执行顺序约束，避免跨 pipe 竞争导致的数据错误。
   - 在 loop/if-else 等控制流下保持同步语义匹配，避免死等/挂死。
2. 稳定性目标（必须满足）
   - event id 分配合法；资源不足时可退化到保守同步（如 `pipe_barrier`），但不允许产生非法代码。
   - zero-trip loop、嵌套 loop、分支汇合等高风险路径行为稳定。
3. 优化目标（尽力满足）
   - 在保证正确性的前提下减少冗余同步，向手工同步收敛。
   - 优先通过“依赖证明增强”降量，而不是激进删除。

---

## 2.2 接口定义（测试可见接口）

### 2.2.1 输入接口

1. 输入文件：PTO IR（`*.pto`）。
2. 输入内容：包含 compute/memory op、`scf.for`/`scf.if` 等控制流、tile/view/subview 等内存视图关系。
3. 前置要求：IR 语义合法，可被 `ptoas` 正常解析。

### 2.2.2 控制开关接口（CLI）

1. `--enable-insert-sync`
   - 启用 InsertSync 自动同步路径。
2. `--pto-insert-sync-debug=<N>`
   - 输出同步分析日志，常用 `N=2/3`。
3. `--enable-graph-sync-solver`
   - 切换到 GSS 对照路径（用于对比，不要求与 InsertSync 文本一致）。
4. `--pto-arch=<a3|a5>`
   - 选择目标架构，影响同步落地形态与校验口径。
5. `--enable-inject-barrier-all-sync`（PR #623 引入）
   - 启用“保守全屏障模式”：不做精细依赖分析，直接在关键 pipe memory-effect op 前插入 `pto.barrier <PIPE_ALL>`，并在函数返回前插入尾部 drain。
   - 与 `--enable-insert-sync`、`--enable-graph-sync-solver` 互斥。

### 2.2.3 输出接口

1. 主输出：生成 C++（含 `set_flag` / `wait_flag` / `pipe_barrier` 等同步指令）。
2. 调试输出：各阶段统计（`nodes` / `syncGroups` / `activeOps`）。
3. 可观测指标：
   - 指令计数：`set_flag`、`wait_flag`、`pipe_barrier`。
   - 结构语义：同步是否在同一可执行控制域内匹配。

---

## 2.3 功能预期（必须项 / 允许项 / 不承诺项）

### 2.3.1 必须项（测试失败即提 P0）

1. 正确性：不允许数值错误、挂死、死等。
2. 同步匹配：不允许出现可执行路径上的 set/wait 失配。
3. 控制流安全：同步移动不能越过会破坏谓词一致性的控制流边界。
4. 事件合法性：event id 必须合法；分配失败应安全退化，不得生成非法同步组合。

### 2.3.2 允许项（可作为 P1/P2 优化）

1. 同步数量多于手工版本，但结果正确。
2. 在 alias/切片不可证明时采取保守同步。
3. 不同架构（A3/A5）生成不同同步形态，但语义正确。

### 2.3.3 不承诺项（测试判定时避免误报）

1. 不承诺与手工同步“逐条一致”。
2. 不承诺 event id 编号稳定不变（同语义下编号可变）。
3. 不承诺自动同步数量必然等于最优下界。

---

## 2.4 三种自动同步模式（测试口径）

1. `InsertSync`（`--enable-insert-sync`）
   - 依赖分析 + set/wait/barrier 混合同步，目标是正确性优先下尽量降量。
2. `GraphSyncSolver`（`--enable-graph-sync-solver`）
   - 图求解路径，事件着色失败时可回退 `PIPE_ALL`。
3. `InjectBarrierAllSync`（`--enable-inject-barrier-all-sync`，PR #623）
   - 保守模式，核心目标是稳定正确，不追求同步数量和性能最优。

测试建议：

1. 把 `InjectBarrierAllSync` 作为“保底正确性对照组”。
2. 若 `InsertSync/GSS` 出现疑似漏同步，可先用该模式验证问题是否与精细分析相关。

---

## 3. 基础概念与术语（为什么需要同步）

1. `set_flag / wait_flag`：跨 pipe 事件同步，生产者 set，消费者 wait。
2. `pipe_barrier`：pipe 内或全局屏障，同步更重，通常更保守。
3. `seed / drain`：回边（loop-carried）依赖补偿同步。
4. `syncGroups`：分析阶段识别出的依赖组数量。
5. `activeOps`：最终还活跃的同步操作数（set/wait/barrier 等）。
6. `eventIdNum`：单条同步占用的 event id 数；在 loop 场景可能 > 1。
7. `zero-trip loop`：循环可能一次都不执行；这是同步移动最容易出错的场景之一。

---

### 3.1 为什么必须同步

PTO kernel 在硬件上是多 pipe 并行执行（例如 `MTE2` 搬运、`V` 向量、`M` 矩阵、`MTE3` 回写），并不是严格按源码“逐行串行”。

当多条 pipe 访问同一底层内存（同一 tile/rootBuffer）时，如果没有显式同步，就会出现经典数据竞争：

1. RAW（Read After Write）
   - 消费者读到“尚未写完”的数据。
2. WAR（Write After Read）
   - 生产者提前覆写，消费者还没读完旧数据。
3. WAW（Write After Write）
   - 两次写入顺序不确定，最终结果不稳定。

所以自动同步的本质是：把“逻辑依赖关系”转成“硬件可执行顺序约束”。

---

### 3.2 `set_flag / wait_flag` 的基础语义

1. `set_flag(srcPipe, dstPipe, eventId)`
   - 由生产者侧发信号，表示“这阶段已完成”。
2. `wait_flag(srcPipe, dstPipe, eventId)`
   - 由消费者侧等待该信号，未到达则阻塞。

可理解为“生产-消费握手”：

1. 生产者完成后 `set`。
2. 消费者在使用数据前 `wait`。
3. `set/wait` 必须在可执行路径上成对匹配。

测试关注点：

1. 控制流域是否一致（同一 loop/if 谓词域内可匹配）。
2. 是否存在“只 set 不 wait”或“只 wait 不 set”的潜在路径。
3. loop 场景的 `seed/drain` 是否与体内同步链路语义一致。

---

### 3.3 `pipe_barrier` 的基础语义

`pipe_barrier` 是比 set/wait 更保守、更重的同步方式：

1. `pipe_barrier(PIPE_X)`
   - 约束某条 pipe 在屏障前后的执行顺序。
2. `pipe_barrier(PIPE_ALL)`
   - 全局收敛，影响最大，但最稳妥。

适用场景通常是：

1. 无法稳定分配 event id，退化到保守屏障。
2. 分析无法证明不冲突，需要正确性优先。
3. 指定保守模式（如 PR #623 的 barrier-all 模式）。

测试判定时要区分：

1. `pipe_barrier` 增多：通常是性能/收敛问题（P1/P2）。
2. 缺少必要 `pipe_barrier` 导致错算/挂死：正确性问题（P0）。

---

### 3.4 一个最小例子（帮助测试同学快速理解）

设 `MTE2` 把 GM 数据搬到本地 tile，`V` 读取该 tile 做计算：

1. 无同步时：`V` 可能先读，读到未完成搬运的数据（RAW 风险）。
2. 加同步后：
   - `MTE2` 完成搬运 `set_flag(MTE2->V, id)`
   - `V` 使用前 `wait_flag(MTE2->V, id)`
3. 若 event 资源不足或模式保守，可能退化成 `pipe_barrier(PIPE_ALL)` 来保证顺序正确。

这就是测试里“正确性优先、数量次之、性能最后”的根本原因。

---

## 4. InsertSync 流水线与测试观察点

自动同步主流程：

`PTOIRTranslator -> InsertSyncAnalysis -> MoveSyncState -> RemoveRedundantSync -> SyncEventIdAllocation -> SyncCodegen`

测试同学无需逐行看代码，但要在日志里关注每阶段统计变化：

1. `After Analysis`
   - 看依赖识别是否异常膨胀或异常减少（`syncGroups`、`activeOps`）。
2. `After Remove Redundant Sync`
   - 看是否有合理删减；如果完全不降，可能是分析过保守或去重条件没触发。
3. `After EventId Allocation`
   - `activeOps` 可能上升（补偿同步/事件分配），这是正常现象，需要结合 C++ 落地代码判断。

---

## 5. 建议的标准执行命令

### 5.1 单用例编译 + 自动同步 debug

```bash
ptoas \
  --enable-insert-sync \
  --pto-insert-sync-debug=2 \
  test/lit/pto/issue454_nested_loop_same_pipe_pair_regression.pto \
  -o /tmp/issue454_auto.cpp 2>&1 | tee /tmp/issue454_auto.log
```

### 5.2 提取阶段统计

```bash
grep -E "After (Translator|Analysis|Sync Motion|Remove Redundant Sync|EventId Allocation)|nodes=|syncGroups=|activeOps=" /tmp/issue454_auto.log
```

### 5.3 统计生成 C++ 中同步指令数量（统一口径）

```bash
rg -n "set_flag\\(|wait_flag\\(|pipe_barrier\\(" /tmp/issue454_auto.cpp
rg -o "set_flag\\(" /tmp/issue454_auto.cpp | wc -l
rg -o "wait_flag\\(" /tmp/issue454_auto.cpp | wc -l
rg -o "pipe_barrier\\(" /tmp/issue454_auto.cpp | wc -l
```

### 5.4 样例目录批量跑（与 CI 口径一致）

`test/samples/runop.sh` 默认追加 `--enable-insert-sync`：

```bash
bash test/samples/runop.sh --enablebc -t Sync
```

如需透传调试参数：

```bash
PTOAS_FLAGS="--pto-insert-sync-debug=2" bash test/samples/runop.sh --enablebc -t Sync
```

### 5.5 保守全屏障模式（PR #623）执行示例

先确认 `ptoas` 是否支持该开关：

```bash
ptoas --help | rg "enable-inject-barrier-all-sync"
```

若未匹配到该开关，说明当前本地分支可能未包含 PR #623，请先同步到包含该 PR 的基线再执行本节测试。

单用例（输出 IR，便于检查 barrier 插入点）：

```bash
ptoas \
  --pto-arch=a3 \
  --enable-inject-barrier-all-sync \
  --emit-pto-ir \
  test/lit/pto/inject_barrier_all_sync_tpush_tpop.pto
```

单用例（输出 C++，便于统计）：

```bash
ptoas \
  --pto-arch=a3 \
  --enable-inject-barrier-all-sync \
  test/lit/pto/inject_barrier_all_sync_tpush_tpop.pto \
  -o /tmp/inject_barrier_all_sync.cpp
```

验收要点：

1. 关键 pipe memory-effect op 前应出现 `PIPE_ALL` 屏障。
2. 函数尾部应有 drain 语义（return 前收敛）。
3. 紧邻已有 `PIPE_ALL` 的位置不应重复插入（去重）。

---

## 6. 回归测试分层（建议执行顺序）

## 6.1 P0 正确性守护（必须先过）

优先执行以下 lit 回归：

1. `test/lit/pto/issue428_cube_sync_regression.pto`
2. `test/lit/pto/issue454_nested_loop_same_pipe_pair_regression.pto`
3. `test/lit/pto/issue454_loop_if_else_loop_carried_sync_regression.pto`
4. `test/lit/pto/issue533_loop_zero_trip_sync_regression.pto`
5. `test/lit/pto/issue564_k_loop_mte1_mte2_wait_regression.pto`

这组覆盖：

1. 单 loop 回边
2. 嵌套 loop 同 pipe-pair
3. if/else + loop-carried
4. zero-trip loop 路径匹配风险
5. MTE1/MTE2 事件链复杂路径

## 6.2 P1 同步数量收敛（正确后再看）

推荐观察：

1. `test/lit/pto/issue226_remove_redundant_pipe_pair.pto`
2. `issue #233` 对应 hadamard/子视图场景（通常在 issue 附件或专项目录）

判断规则：

1. 先确保结果正确，再比较数量。
2. “自动多于手工”不直接判错，需要结合依赖是否真实存在。
3. 若存在明显冗余（例如无体内匹配的 seed/drain），标记为优化机会并附日志。

## 6.3 P2 Solver 对照（可选）

`*_gss.pto` 用于 GraphSyncSolver 对照，不要求与 InsertSync 输出逐行一致，但要满足基本同步语义。

参考说明：`test/lit/pto/README_gss_lit_companions.txt`。

## 6.4 P2 保守全屏障模式对照（可选）

建议增加：

1. `test/lit/pto/inject_barrier_all_sync_tpush_tpop.pto`（PR #623）

该组主要验证：

1. 全屏障模式可用性（编译/落地稳定）。
2. 与 InsertSync/GSS 的互斥开关行为正确。
3. 作为问题分流基线：若保守模式正确而精细模式错误，优先排查依赖分析/同步移动。

---

## 7. 如何判断“高风险问题”与“可优化问题”

## 7.1 高风险问题（直接 P0）

1. 结果错误（数值 mismatch）。
2. 运行挂死（常见于 set/wait 控制域不匹配）。
3. 明显越控制流外提导致的潜在不匹配（例如 set 在外层，wait 仅在条件内）。
4. `event id` 分配失败后行为异常（不仅仅是数量增加）。

## 7.2 可优化问题（P1/P2）

1. 同步数量偏多但结果正确。
2. `pipe_barrier(PIPE_V)` 在可融合 VF 链路上出现过量。
3. 动态切片/`bind_tile + subview` 证明能力不足导致保守同步。
4. loop 回边被保守识别导致 seed/drain 偏多。

---

## 8. 常见误判与解释口径（测试沟通建议）

1. 误判：自动同步比手工多，一定是 bug。
   - 正解：手工可能漏同步；必须以正确性和可证明依赖为准。

2. 误判：`After EventId Allocation` 活跃同步变多是回退。
   - 正解：该阶段可能引入补偿同步或多 ID 展开，需要看最终代码语义是否正确。

3. 误判：seed/drain 一定冗余。
   - 正解：只有确认 loop-carried 不成立或控制域不匹配时，才可判冗余/风险。

4. 误判：A3/A5 同步形态必须一致。
   - 正解：架构与 lowering 约束不同，需在各自语义下验证。

---

## 9. 问题单模板（建议原样复制）

### 9.1 基本信息

1. 分支/commit：
2. 目标架构：`a3` / `a5`
3. 命令行（完整）：
4. 用例路径：

### 9.2 现象分类

1. 正确性错误 / 挂死 / 编译失败 / 同步数量异常 / 性能退化
2. 首次出现版本与可复现概率

### 9.3 必附材料

1. 生成 C++（自动同步结果）
2. `--pto-insert-sync-debug=2` 日志
3. 同步计数统计（`set_flag` / `wait_flag` / `pipe_barrier`）
4. 对照版本（手工或历史版本）同口径统计

### 9.4 建议附加材料（复杂 case）

1. 最小复现 PTO（可裁剪）
2. 关键控制流说明（loop/if-else 嵌套层级）
3. 预期依赖关系图（哪对 op 应该/不应该同步）

---

## 10. 面向测试同事的执行清单（Checklist）

1. 跑 P0 回归并确认全绿。
2. 对目标 issue case 生成 auto C++ 与 debug 日志。
3. 统一口径统计三类同步指令数量。
4. 对比历史版本或手工版本，记录变化量。
5. 若数量上升，先确认是否引入正确性修复，再决定是否归类为优化项。
6. 按模板提交问题单，附最小复现与日志。

---

## 11. 参考文档

1. `docs/designs/ptoas-auto-sync-design.md`
2. `docs/designs/ci-board-validation-guide.md`
3. `test/lit/pto/README_gss_lit_companions.txt`
4. `test/samples/runop.sh`
5. `https://github.com/hw-native-sys/PTOAS/pull/623`

---

## 12. 备注

本文是测试串讲文档，强调“如何测、怎么判、如何提单”。  
具体算法细节（例如 alias 证明、回边建模、event-id 分配策略）请以设计文档和对应 PR/Issue 讨论为准。
