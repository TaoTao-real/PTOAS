# DeepSeek V4 VMI TileLib 与 VF Fusion 能力补齐计划

## 1. 文档信息

- 状态：静态主链 VMI producer 已实现；后端物理化与 A5 验收单独跟踪
- 日期：2026-07-23
- 目标分支：PTOAS `feature-vmi`
- DS v4 基线：pypto-lib `0999dba`
- PTOAS 开发基线：`9f17313dd`
- 关联设计：
  - [VMI VF Fusion RFC](vmi-vf-fusion-design.md)
  - [ADR-0001](adr/0001-vmi-vf-fusion-rfc-minimal-pipeline.md)
  - [PTODSL TileLib 模板选择设计](ptodsl-tilelib-template-selection-design.md)

本文面向 DeepSeek V4 算子落地，定义 VMI TileLib、VMI VF Fusion、VMI
mem2reg 的能力缺口、正确性约束、分阶段计划和验证方案。

### 1.1 2026-07-23 实现快照

当前批次只交付 VMI TileLib 静态主链、VMI conversion 语义闭环和分析型
Fusion-ready checker，不实现实际 VMI loop fusion、mem2reg 或后端物理化：

- 已从 8 个真实 DS v4 生成 PTO 文件固化 manifest，共 146 类签名；113 类为
  `PIPE_V`，其中 111 类已有 canonical VMI candidate，`textract` 和 `tfillpad`
  明确标记为边界，没有 `UNKNOWN` pipe。
- `LogicalRowMap` 已支持静态 `1/8/16/32/64/128/256/448/512` logical lanes；每个
  TileOp 只生成一个 principal row loop，不生成 physical-chunk 内层循环。
- 已打通 P0 convert 矩阵、wide elementwise、wide row reduce、compact row state、
  row/col broadcast、`trecip`、两种 `trsqrt` form 和 `tgather:index`。
- RMSNorm、QKV/RoPE、Decode FA、Prefill FA 四个静态 harness 用于验证 Expand、Inline
  和 Fusion-ready；到达语义完整 surface VMI 即关闭 producer Gate。
- Wide logical vreg 不因宽度在 VMI checker 中直接判定不可融合；gather 标记为
  `layout_phase`。物理可行性和寄存器压力等待后端 capability/cost 接口。
- 动态 `valid_shape`、tail mask、实际 Fusion、mem2reg、sort、通信 lowering 仍是后续
  批次。本批次对动态 valid shape 保持显式拒绝并有负向测试。
- 当前分支已通过 Python 语法检查和 diff hygiene；完整 lit 需要匹配的 MLIR Python
  binding 环境后复跑。
- LayoutAssignment、VMIToVPTO、intrinsic 和 A5 数值问题整理为独立后端 backlog，不在
  本批次通过修改 VMI contract 规避。

## 2. 需求背景

DeepSeek V4 的 RMSNorm、QKV/RoPE、Attention、Compressor、Gate 和 Indexer 中有大量
连续向量计算。当前这些计算通过 PyPTO 生成 PTO TileOp，再由 PTOAS TileLib 展开到底层
实现。现有 MI/VPTO TileLib 可以独立执行，但每个 TileOp 通常各自生成循环和 UB
load/store，中间结果难以在逻辑向量寄存器中直接传递。

VMI VF Fusion 的目标不是简单替换 TileOp 名称，而是建立以下链路：

```text
PyPTO DS v4 kernel
  -> PTO TileOp
  -> Tile 层粗粒度 fusion_region
  -> PIPE_V TileOp 展开并 inline 为 canonical VMI loops
  -> VMI phase/loop compatibility 分析
  -> VMI loop fusion
  -> VMI mem2reg 消除中间 UB 往返
  -> VMI layout assignment / VMIToVPTO
  -> A5 object / assembly / runtime
```

最终价值体现在三点：

1. DS v4 关键向量链可以由 VMI TileLib 无语义损失地表达，并为后端提供稳定输入。
2. FA/Online Softmax 等 producer-consumer 链能够形成可证明安全的深融合数据流。
3. 融合后减少循环、UB load/store 和同步开销，同时保持动态 Shape、mask、精度和内存
   依赖语义。

## 3. 范围与成功标准

### 3.1 完整一期目标（分批交付）

- 首先服务 PyPTO 生成的 PTO-ISA/TileLib 代码，不优化任意用户手写 VMI 控制流。
- 保持 `tload/tstore`、Cube matmul 等非 `PIPE_V` TileOp 的既有实现。
- 补齐 DS v4 关键 `PIPE_V` TileOp 的 canonical VMI implementation。
- 复用 Tile 层 `pto.fusion_region` 作为粗粒度分析边界，在其内部执行 VMI phase
  识别、loop fusion 和 mem2reg。
- 支持一个静态 logical row 使用多种已验证 logical width；candidate 只生成一个 row
  主循环，动态 `valid_inner <= logical_inner` 通过 typed mask 表达。
- 后端和真实 A5 功能/性能验证作为端到端独立 Gate，不以 VMI IR 编译通过替代硬件正确性。

### 3.2 非目标

- 首期不做多 schedule candidate、cost model、autotuning 和任意 Shape 自动重切。
- 首期不跨 `pto.fusion_region`、sync、DMA、Cube、未知 call 或未知副作用移动代码。
- 首期 mem2reg 不做跨迭代 promotion，不自动构造复杂 `scf.for iter_args`。
- 不在 VMI 模板中固化 physical vreg、interleave、pack 或 MI post-update 策略。

### 3.3 成功标准

VMI producer 交付需要满足：

- **覆盖**：选定 DS v4 entrypoint 的所有 `PIPE_V` TileOp 均有合法 VMI form，或被明确
  列为尚未完成的阻塞项；不能静默回退。
- **语义完整**：关闭 Fusion 时，每个 VMI TileLib implementation 生成完整
  load/compute/store、round/sat、mask 和 memory effects，且 VMI verifier 通过。
- **融合正确**：打开 Fusion 后，循环合并和 mem2reg 不改变结果、mask、内存顺序和同步
  语义。

端到端系统另设以下后端 Gate，不作为 VMI TileLib PR 的完成判据：

- **物理化正确**：VMIToVPTO 后无残留 VMI，layout、mask 和物理指令合法。
- **端到端正确**：RMSNorm、QKV/RoPE、Attention 等目标 kernel 在动态 Shape 和边界值下
  通过 golden 对比。
- **性能可解释**：能用 IR、汇编和真机数据说明收益来自哪些 loop/load/store/sync 的消除。

## 4. 实施前基线

本节保留启动本轮补齐前的基线和缺口，用于解释改造动机；当前实现状态以 1.1 节为准。

### 4.1 已具备的编译链

当前 `--tile-lib-backend=ptodsl-vmi` 已实现按 Pipe 路由：

- `PIPE_V` TileOp 使用 canonical VMI provider。
- MTE/Cube 等 TileOp 继续使用现有 PTODSL TileLib provider。
- Expand 后生成 `func.call @__pto_ptodsl_vmi_*`，Inline 后得到 `scf.for + pto.vmi.*`。
- 现有 VMI semantic/layout pipeline 可以继续 lower 到 VPTO。

现有测试已经验证静态 Softmax compute harness 和一个静态 FA 多项式样例可以完成
Expand、Inline 和 VMIToVPTO，但尚未执行 VMI loop fusion，也不代表完整 DS v4 覆盖。

### 4.2 实施前 VMI TileLib

当前注册 28 个 canonical implementation，覆盖 27 个 TileOp；`tdivs` 按操作数顺序包含
两个 semantic form：

```text
tadd tsub tmul tmov texp tmax tmin tabs tneg
tadds tsubs tmuls tmaxs tmins tdivs:tile_scalar tdivs:scalar_tile
tdiv texpands
trowmax trowsum trowexpandsub
tcolmax tcolsum
tcolexpandsub tcolexpandadd tcolexpandmul tcolexpanddiv
tcvt
```

静态 full-1VL 基础族已具备以下 dtype 能力：

- `tadd/tmov/tadds/tmuls/tsubs/texpands`：`f32/f16/bf16/i32/i16/i8`。
- `tsub`：`f32/f16/i32/i16/i8`；A5 TileOp 本身不接受 `bf16`。
- `tmul`：`f32/f16/i32/i16`；A5 TileOp 本身不接受 `bf16/i8`。
- `texp/tmax/tmin/tmaxs/tmins/tdiv/tdivs/tabs/tneg`：当前 canonical VMI form
  只开放 `f32/f16`，不把 unified VMI 尚不能正确物理化的整数 form 注册进来。

剩余主要限制：

- Elementwise Dense Tile 要求每行严格为一个 dtype-native VL；当前不接受 multi-VL inner。
- Row/Col Reduce 和 Expand 仍只接受 FP32 的静态 full-1VL form。
- `tcvt` 只支持 `f32 -> f16/bf16 + RINT`，P0 完整转换矩阵尚未补齐。
- provider 拒绝动态 `valid_shape`，也拒绝 `valid_shape != physical shape`。
- provider 已按 `(target, op, semantic_form)` 保证 canonical 唯一性，但 `tgather`、
  `trsqrt` optional tmp 和 `tmrgsort` 的具体 semantic form 尚未实现。
- 当前只验证 standalone VMI lowering；未达到 `numeric_unfused_pass` 的 dtype/form 不能计入
  DS v4 coverage，更不能直接标记为 `fusion_eligible`。

### 4.3 当前 Fusion 状态

- `ptodsl-vmi` 当前禁止 `--enable-op-fusion`，因此实际编译链不会生成 Tile 层
  `pto.fusion_region`。
- `VMIIdentifyFusionUnits`、VMI phase planner、VMI loop fusion 和 `VMIMem2Reg` 目前只有
  设计，没有实现。
- 现有 `FusionPlan/FusionRegionGen` 面向 Tile-native PTO IR，可以作为粗粒度 region
  规划能力复用。
- 现有 `PTOLowLevelLoopFusion` 面向已物理化 VPTO/MI，不应作为 VMI Fusion 实现。
- 不需要新增 `fusion_scope` 包裹每个 TileOp。TileOp 展开后仍应位于原
  `pto.fusion_region` 内；轻量 provenance attribute 只用于诊断和 unit 映射，不作为
  融合合法性的唯一依据。

### 4.4 DS v4 需求画像

源码静态调用统计显示主要压力集中在：

```text
cast 437, mul 213, add 184, full 146, sub 85,
row_expand_mul 83, col_expand_mul 70, exp 48, gather 47,
row_sum 43, row_max 30, recip 27, arange 23,
fillpad 22, set_validshape 31, sort32 3, mrgsort 9
```

源码计数只用于估算，不是最终 coverage 依据。正式清单必须来自 PyPTO lowering 后的
PTO IR，并记录每个 TileOp 的 Pipe、operand form、dtype、physical/valid shape、layout
和 context attrs。

### 4.5 关键阻塞

1. **TileLib 尚未使用 VMI wide logical vreg**：VMI core 已能把
   `vreg<128xf32>` 映射为两个物理 FP32 vreg，但当前 `CanonicalBlockMap` 仍拒绝
   `cols != dtype-native VL`，`tcvt` 也未使用 BF16/FP16 widening `extf` 路径。
2. **动态 valid shape 未接入模板**：DS v4 decode/prefill 和 fillpad 依赖运行时尾块。
3. **semantic form 无法分派**：`tgather/trsqrt/tmrgsort` 不能用一个固定签名覆盖。
4. **sort 缺少 VMI 语义**：PTO/VPTO 有 sort 指令，但 unified VMI dialect 没有完整表达。
5. **Fusion/mem2reg 尚未实现**：当前只能独立 lowering，不能证明深融合收益。
6. **VPTO 缺少 pipe communication lowering**：`tpush/tpop` 目前只在 EmitC 路径直接调用
   PTO-ISA C++ 模板；VMI、VMIToVPTO 和 VPTO backend 都没有完整 pipe lifecycle lowering。
   混合 Cube/Vector DS kernel 即使补齐所有向量 TileLib，仍会被这一缺口阻塞。

## 5. 必须冻结的设计协议

### 5.1 Fusion region 协议

复用现有 TileFusion 的 `pto.fusion_region` 作为粗粒度所有权和 escape 边界：

- Tile 层负责构建 DFG、形成候选 region、描述 region inputs/outputs。
- Expand/Inline 在 region 内完成，非 `PIPE_V` 实现仍然保留。
- VMI 层基于实际 SCF、VMI memory effects、mask 和 alias 将一个 region 划分为若干
  phase 和真正可融合的 loop groups。
- region 成员关系不等于所有 loop 都必须融合；失败时保留独立实现。

VMI 路由只复用 `FusionPlan/OpScheduling/FusionRegionGen` 的 Tile 层规划部分，跳过 legacy
`PTOLowLevelLoopFusion`，改为运行 VMI-specific fusion 和 mem2reg。

### 5.2 Canonical form 协议

把“每个 `(target, op)` 一个固定 candidate”修正为：

```text
每个 (target, op, semantic_form) 恰好一个 canonical implementation
```

`semantic_form` 只描述语义合法性，不是性能 schedule，例如：

```text
tgather:index
tgather:mask_pattern
trsqrt:basic
trsqrt:high_precision_with_tmp
tmrgsort:format1
tmrgsort:format2
tcvt:<src,dst,round,sat>
```

同一 semantic form 首期仍不允许多个性能 candidate。

### 5.3 Logical-row 与 wide vreg 协议

这里必须区分硬件物理 VL、Tile logical row 和 VMI logical vreg。冻结协议如下：

- 一个 row-major canonical unit 只生成一次 row iteration，不生成动态 inner
  `scf.for`；ColReduce 也只生成一个携带 wide accumulator 的 row loop。
- Tile 的静态 inner extent 定义 logical lane domain。模板直接构造
  `!pto.vmi.vreg<inner x dtype>`；inner 可以小于、等于或大于 dtype-native VL。
- Widening/narrowing convert 保持 logical lane count。例如 `128xbf16 -> 128xf32` 仍是一
  个 128-lane SSA value；VMI layout lowering 将结果映射为 `K=2` 个物理 vreg。
- Fusion 兼容性比较 row domain、logical lane mapping、mask 和 phase，不比较 convert
  前后的物理 vreg 数。
- `valid_inner` 可以动态小于静态 logical width，并由 typed mask 表达；VMI element count
  本身保持静态。
- 每个 semantic form 必须声明已验证的 logical width 和最大 physical chunk 数。第一批
  至少覆盖 `64xf32`、`128xf32`、`256xf32`、`128xbf16/f16` 以及 DS v4 的 compact
  `1/8/16-lane f32` state。
- 超出已验证 width/chunk/layout 矩阵时明确拒绝或要求 PyPTO 重切，不能在 TileLib 中
  临时生成动态 inner-block loop。

这意味着用户切 Tile 的真正契约是“一个静态 logical row + 一个动态主循环”，而不是
“所有中间 Tile 都必须恰好占一个物理 vreg”。只有在 wide value 的物理 chunk 压力超过
预算、VMI op/layout 不支持或 logical width 动态不可界定时，才要求上游重新切 Tile。

### 5.4 Fallback 协议

- VMI implementation 缺失必须产生明确 coverage error，不能静默回退到 MI。
- 单个 unit 不满足 Fusion 条件时，保留其独立 VMI implementation，不是编译失败。
- mem2reg 无法证明安全时保留 store/load，不是编译失败。
- 若短期需要运行尚未覆盖 sort 的 Gate/Indexer，只能整 kernel 显式选择 legacy backend；
  不在一个 VMI region 内混入未声明的物理 MI fallback。

## 6. 能力补齐要求

### 6.1 Provider 与 tracing 基础能力

| 能力 | 具体要求 | 正确性要求 |
|---|---|---|
| Semantic-form dispatch | 按 operand schema、context attrs、dtype 选择唯一 form | ambiguity 和 no-match 必须可诊断 |
| Dtype 泛化 | BF16/FP16/FP32/INT32/INT8 及相应 logical lanes | verifier 和 VMIToVPTO 均支持 |
| Logical-row mapping | 从 Tile shape 构造 `vreg<inner x dtype>`，不生成 inner-block loop | 1/2/4 physical chunks 均保持一个 row loop |
| Wide load/store | logical load/store 覆盖整行，由 layout lowering 拆分物理访问 | unfused 独立执行正确，地址 span 精确 |
| Widening/narrowing | 使用 `extf/truncf` 等 surface VMI，保持 logical lane count | layout relation 与 EVEN/ODD/pack lowering 正确 |
| Dynamic valid shape | 把运行时 valid row/col 作为 SSA 传入模板 | 检查 `0 <= valid <= physical` |
| Dynamic mask | 从 `valid_inner` 构造 typed prefix mask | inactive lane 不得产生可观察写入 |
| Context attrs | round/sat/precision/mask pattern 等完整传递 | specialization key 必须包含全部语义 attrs |
| Tracing surface | iota、unary、min、gather、shuffle、masked access、group op | 只暴露 surface VMI，不固化 physical layout |
| Provenance | region 内保留 provider/op/form/unit 的轻量信息 | Inline 后仍可诊断来源，不新增 scope op |

### 6.2 P0：DS v4 FA/RMSNorm/QKV 主链

新增 canonical forms：

```text
texpands
tdiv tmin tsubs
tabs tneg
trsqrt trecip
trowexpandmul trowexpanddiv
tcolexpandmul tcolexpandadd
tgather:index
```

`tci` 的 `OpPipeInterface` 为 `PIPE_S`，不进入 canonical VMI provider；它继续使用现有
PTODSL TileLib/标量 lowering。VMI 可以在其它 Vector semantic form 内部使用 `vci`，但这
不等于为 TileOp `tci` 注册 VMI implementation。

扩展已有 forms：

```text
tadd tsub tmul tmax texp
tadds tmuls tmaxs tmins tdivs
trowmax trowsum trowexpandsub
tcvt
```

P0 转换矩阵至少覆盖：

```text
BF16 -> FP32
FP16 -> FP32
FP32 -> BF16 (RINT)
FP32 -> FP16 (ROUND/RINT)
FP32 -> INT32 (RINT/TRUNC/NONE as required)
INT32 -> FP32
INT32 -> FP16
FP16 -> INT8 (TRUNC)
```

P0 logical-width 矩阵至少覆盖：

```text
RMSNorm:  128xbf16 -> 128xf32（1 -> 2 physical chunks）
QKV:      256xf32（4 physical chunks）
Attention:128xf32（2 physical chunks）
Compact:  1/8/16xf32 reduce/scale state
```

Elementwise、Convert、RowReduce 和 RowBroadcast 必须共享 `iterationDomain = rows`。
禁止把 wide Elementwise 展平为 `rows * physicalChunks`，否则无法与 RowReduce/ColReduce
按同一 row phase 融合。ColReduce 使用一个 `scf.for row` 携带 wide accumulator；最终
Reduce 结果的 consumer 从下一 phase 开始。

本批次两种 `trsqrt` form 都严格复用当前 A5 PTO-ISA 的 `vsqrt + 1/vdiv` 实现。带 tmp
的 form 保留参数、operand schema 和语义身份，但不虚构额外精度算法，也不宣称其结果
高于现有 PTO-ISA 实现。若后续引入专用高精度算法，必须单独完成 PTO-ISA 对齐和数值验收。

### 6.3 P1：Compressor、HC 和完整 Attention 辅助路径

```text
tsqrt tlog
trowexpand
tcolexpand
tcolmax tcolsum
tcolexpandexpdif
tfillpad
ttrans
tgather:mask_pattern
```

要求：

- `tfillpad` 支持 zero/min 等 pad value，并正确消费动态 valid shape。
- `tcolmax/tcolsum` 保持跨 row 归约 phase，不能错误并入依赖最终归约结果的 consumer loop。
- `ttrans` 使用 VMI layout/rearrangement 语义，不在模板里直接指定 `vintlv/vdintlv`。
- mask-pattern gather 使用静态 shuffle/select 或合法 gather form，保持输出 dtype 和 lane 顺序。

### 6.4 P2：Gate/Indexer TopK

```text
tsort32
tmrgsort:format1
tmrgsort:format2
```

这部分需要先补 unified VMI sort 语义、verifier、layout support 和 VMIToVPTO lowering，不能
只在 TileLib 中拼接 VPTO 指令。P2 是 DS v4 全量 VMI coverage 的阻塞项，但不应阻塞
FA/RMSNorm/QKV 主链的阶段验收。

### 6.5 VMI phase 与 loop fusion

在每个 `pto.fusion_region` 内识别：

```text
MapPhase           elementwise / scalar / convert
RowReducePhase     每个 row 内归约，可与同 row consumer 深融合
RowBroadcastPhase  [rows,1] compact state 广播回同 row
ColReducePhase     跨 rows 归约，结果完成前必须形成 phase boundary
LayoutPhase        transpose/gather/pack 等重排
OpaquePhase        未知或暂不支持，作为 hard boundary
```

两个 loop 只有同时满足以下条件才能融合：

- 位于同一 region、同一 block，并保持原顺序。
- iteration domain 等价，或有经过验证的 row/compact-state 映射。
- 中间无 sync、DMA、Cube、call、barrier 或未知副作用。
- VMI memory access 可规范化，依赖方向和 distance 可证明。
- mask obligation 兼容，consumer mask 被完整保留。
- 不违反 Reduce phase 的结果可用时机。

### 6.6 VMI mem2reg 与 alias

mem2reg 输入是已经融合的 VMI loop/group。每次 VMI memory access 建模为：

```text
LocationKey = (
  storage_root,
  normalized_offset,
  per_iteration_span,
  element_type,
  logical_shape,
  mask_coverage,
  layout/access_mode
)
```

首期允许：

- 同一迭代、精确相同 location 的 RAW forwarding。
- forwarding 后，如果 store 无 region 外可观察用户，删除 store/load。
- store 仍需对外可见时，只替换内部 load，保留 store。

首期拒绝：

- partial overlap、动态 pointer arithmetic 或无法恢复 storage root。
- 未证明安全的 WAW/WAR、跨迭代依赖和已规划 UB 地址重叠。
- producer/consumer mask 覆盖关系无法证明。
- layout 或 logical value footprint 不一致。

内存规划可能让不同 Tile handle 复用同一物理 UB 地址，因此不能仅凭 Tile SSA 不同判定
NoAlias。当前 PlanMemory 和 auto-sync 已各自具备 alias/address-range 建模，但不是可直接
消费的统一 analysis result。应抽取共享的 root/range utility，或由 VMI pass 从已规划 IR
重建保守结果，不能依赖前序 pass 的隐藏进程状态。`reserve_buffer/import_reserved_buffer`
是跨 kernel 管理缓冲区，首期直接作为边界；它不同于普通 Tile 的 PlanMemory 地址复用。

### 6.7 目标 pass 顺序

```text
Tile-native PTO IR
  -> InsertTemplateAttributes
  -> FusionPlan / OpScheduling / FusionRegionGen
  -> PTOViewToMemref
  -> PlanMemory
  -> ResolveReservedBuffers
  -> AutoSync / BufidSync / GraphSyncSolver（按现有配置三选一）
  -> ResolveBufferSelect
  -> ExpandTileOp(ptodsl-vmi composite routing)
  -> PTOInlineLibCall
  -> FoldTileBufIntrinsics(shape-only)
  -> VMIIdentifyPhasesAndAccesses
  -> VMIPlanLoopFusion
  -> VMIFuseCompatibleLoops
  -> canonicalize / CSE
  -> VMIMem2Reg
  -> canonicalize / CSE
  -> PTOFlattenFusionRegion
  -> FoldTileBufIntrinsics(addr-only)
  -> existing VMI semantic/layout pipeline
  -> VMIToVPTO
  -> LowerPipeCommunicationToVPTO
  -> PTOInferVPTOVecScope
```

关键要求：

- `ptodsl-vmi + --enable-op-fusion` 改为进入上述 VMI lifecycle，不再触发当前报错。
- Tile 层 region 在 PlanMemory/auto-sync 前形成，VMI Fusion 必须尊重它们后来插入的物理
  地址和同步边界。
- VMI Fusion 取代该路由下的 `PTOLowLevelLoopFusion`、legacy predicate/store-load
  elision；不能两套低层 fusion 同时运行。
- `PTOFlattenFusionRegion` 在 VMI mem2reg 和 escape 检查完成后运行。
- Tile `tload/tstore` 是 MTE 搬运边界；模板内部 `pto.vmi.vload/vstore` 是 UB 与 logical
  vreg 之间的访问，两者不能混为一类。

### 6.8 `tpush/tpop` 与 VMI 的边界

`tpush/tpop` 不是 VMI 计算指令，也不是一条可直接补齐的 MI。A5 PTO-ISA 实现包含：

```text
producer: acquire/free-slot wait -> slot address -> TSTORE/TMOV/TINSERT -> publish signal
consumer: data-ready wait -> slot address/TLOAD or bind -> release signal
shared:   ring index、slot modulo、split/subblock、flag id、talloc/tfree lifecycle
```

因此 VMI 侧只负责以下语义：

- `tpop` 是 incoming memory/ownership boundary，后续 VMI load 不得移动到它之前。
- `tpush` 读取并发布完整 producer Tile，前序 VMI store 不得移动到它之后或被错误删除。
- `tfree` 结束 popped entry 生命周期，任何访问不得越过它。
- VMI Fusion 可以优化 `tpop` 与 `tpush/tfree` 之间的纯向量链，但不得跨通信边界融合。
- mem2reg 不跨 pipe 边界 forwarding；tpush 可观察的最终 store 必须保留。
- popped FIFO slot 的地址是动态 alias root。除非 slot/index 可证明，否则与其它 pipe entry
  按 MayAlias 处理。

VPTO backend 需要独立的 `LowerPipeCommunicationToVPTO`，覆盖完整 lifecycle，而不是新增
VMI TileLib。建议先把高层 pipe op 分解为通信 micro-ops，再复用已有物理能力：

```text
initialize_pipe / talloc / tpush / tpop / tfree
  -> comm.acquire_slot / comm.slot_addr / comm.advance
  -> pto.sync.wait / pto.sync.set
  -> existing GM<->UB、GM<->MAT、ACC->VEC、VEC->MAT movement ops
  -> existing VPTO LLVM intrinsic emission
```

第一阶段只实现 DS v4 实际 lowering 出现的 `(direction, fifo kind, split, dtype, quant)` forms；
不应凭源码猜测。完整通用矩阵包括 C2V/V2C、local/GM FIFO、split、dual-AIV 和 fixpipe quant，
工作量显著大于增加普通 elementwise TileLib。

短期若 pipe lowering 尚未完成：

- 可以继续用 vector-only extracted harness 验证 VMI TileLib/Fusion/mem2reg。
- 含 `tpush/tpop` 的完整 kernel 只能显式走 EmitC baseline。
- 不能宣称 DS v4 VPTO/VMI 端到端已经打通。Mixed backend 只在 module child 边界工作，且
  cross-child ABI 不传 Tile，不能自动解决同一 mixed kernel 内的 pipe lowering。

## 7. 正确性要求

### 7.1 TileOp 语义

- VMI unfused 结果必须先与 TileOp/PTO-ISA 语义一致，再允许参与 Fusion。
- Scalar、layout、round、sat、precision、tmp 和 inplace/DPS 语义必须进入 form 合法性判断。
- Reduce 不得静默改变结合顺序；任何 reassociation 必须有显式属性和数值验收。

### 7.2 Shape 与 mask

- physical shape 静态，valid shape 可静态或动态。
- `valid=0`、`valid=1`、`valid=VL-1`、`valid=VL` 都要有定义明确的行为。
- masked store 不得写 inactive lanes；fillpad 必须写入规定的 pad value。
- 同一动态 bound 派生的 mask 可以融合，不同或无法证明等价的 mask 保守拒绝。

### 7.3 Convert 与精度

- 对整数转换逐 bit/逐元素验证 rounding、truncation、saturation 和边界值。
- FP transcendental 使用 PTO-ISA/legacy hardware 结果和 golden 双重对比。
- High-precision rsqrt/recip 不以普通 FP32 相对误差测试替代指令语义验证。

### 7.4 Fusion 与内存顺序

- Fusion 前后 VMI op 的程序顺序、RAW/WAR/WAW 约束和 sync 边界一致。
- MayAlias 默认不融合或不提升，不能用“不同 SSA”代替 NoAlias 证明。
- region output、外部用户和异常/边界路径上的必要 store 必须保留。
- PlanMemory 和自动同步当前早于 Expand/VMI Fusion。Loop fusion 改变迭代交错前，必须证明
  已规划的 UB 地址复用在新 schedule 下仍不产生同时存活或覆盖；否则拒绝融合。

### 7.5 Lowering 完整性

- VMIToVPTO 后不得残留 `pto.vmi.*`。
- VMI Fusion 前不得生成物理 `pto.vecscope`；最终由现有 VPTO scope inference 生成。
- layout assignment 不得引入 verifier error、非法 partial group 或不支持的 mask granularity。
- 自动同步已经生成的 sync/barrier 必须作为 Fusion hard boundary 并原样保留。
- `tpush/tpop/tfree` 自带跨 core FIFO handshake，不能由普通 auto-sync 重复替代或删除。

## 8. 验证测试方案

### 8.1 四路径、三组差分原则

每个关键测试尽量同时比较：

```text
A. CPU/PyTorch golden
B. legacy TileLib / MI or VPTO implementation
C. VMI TileLib unfused
D. VMI TileLib fused + mem2reg
```

- `B vs C` 隔离 TileLib 改写错误。
- `C vs D` 隔离 Fusion/mem2reg 错误。
- `A vs C/D` 防止 legacy implementation 本身成为错误 oracle。

### 8.2 测试分层

| 层级 | 测试内容 | 主要断言 |
|---|---|---|
| L0 Coverage manifest | PyPTO lowering 后扫描 DS v4 PTO IR | 无未分类 `PIPE_V` op/form |
| L1 PTODSL unit | template specialize/render/helper | form 选择、dtype、shape、attrs、诊断正确 |
| L2 PTOAS lit | Expand/Inline/Fusion/mem2reg/VMIToVPTO | IR 结构、拒绝原因、无残留 VMI |
| L3 单 op 数值 | camodel/A5，随机和边界输入 | VMI unfused 与 golden/legacy 一致 |
| L4 融合链数值 | VMI unfused vs fused | 输出一致，必要 store/sync 保留 |
| L5 DS kernel E2E | PyPTO runtime harness | 动态 Shape、decode/prefill、golden 通过 |
| L6 性能与汇编 | A5 真机、VPTO/LLVM/ASM artifacts | loop/UB 访存减少且 latency 不回退 |

### 8.3 L0 Coverage manifest

为选定 DS v4 entrypoints 固化机器可读清单，每一项包含：

```text
source API
PTO TileOp
Pipe
operand semantic form
dtype signature
physical shape / valid shape
layout / memory space
context attrs
VMI form name
implementation status
test status
```

源码正则计数只作为预估。清单必须由实际 PyPTO lowering IR 生成，并纳入回归，防止
pypto-lib 更新后新增 TileOp 但 PTOAS coverage 未同步。

### 8.4 L1/L2 模板与 IR 测试矩阵

每个 op/form 至少包含：

- 一个合法静态 native-VL case；支持 wide 的 form 还必须包含2VL/4VL和超出矩阵的负向
  case，支持 compact 的 form 必须包含实际 DS state width。
- 当前静态批次包含动态 `valid_shape`/tail 的明确拒绝用例；B3 开启后再把该项替换为合法
  dynamic tail case，并覆盖 `0/1/VL-1/VL`。
- 所有支持 dtype/round/precision 的代表 case。
- 非法 dtype、shape、layout、form 和 context attr 的负向 case。
- Expand 后 call 指向正确 VMI provider。
- Inline 后包含预期 surface VMI op 和主循环，不含显式 vecscope。
- VMIToVPTO 后包含预期物理 op，不含残留 VMI。

Fusion/mem2reg 专项 lit cases：

```text
正向：
  elementwise -> elementwise
  bf16/f16 widen -> wide f32 elementwise -> rowreduce
  2VL/4VL wide logical store-load exact RAW forwarding
  rowmax -> row_expand_sub -> exp -> rowsum
  map -> colreduce wide accumulator，consumer 被切到下一 phase
  convert at region output
  dynamic bound SSA 相同
  exact RAW same offset
  forwarding with externally visible store retained

负向：
  region 外或用户手写 VMI
  trip count/step 不同
  sync/DMA/Cube/call boundary
  exact offset 不同或 partial overlap
  MayAlias / pointer arithmetic
  两个 Tile 被 PlanMemory 复用到同一 UB range，融合后 lifetime 冲突
  mask coverage 不兼容
  WAW/WAR 未证明
  compact state 有 region 外用户
  physical chunk/liveness 超出已冻结预算
  colreduce 结果未完成即被 consumer 使用
  VMI load/store 被移动到 tpop 之前、tpush/tfree 之后
  tpush 可观察的最终 UB store 被 mem2reg 删除
```

### 8.5 单 op 数值数据集

- 普通值、零、正负极值、NaN、Inf、subnormal。
- Convert 的上下界、半 ULP、溢出、饱和和负数 truncation。
- Reduce 的全相等、单峰、交替极值和 NaN 行为。
- Gather 的重复 index、逆序、边界 index 和 mask pattern。
- Tail 的 `0/1/VL-1/VL` active lanes。
- Fillpad 的 zero/min pad value。

整数、index、mask 和 lane permutation 应逐元素精确比较。浮点 elementwise 和 reduce 使用
op-specific `atol/rtol/ULP`；Softmax/RMSNorm 使用最终输出容差并额外检查归一化不变量。

### 8.6 DS v4 端到端用例

第一批必须固定：

1. `rmsnorm.py`：BF16/FP32 convert、mul、row_sum、rsqrt、row/col expand。
2. `qkv_proj_rope.py`：arange、convert、gather、RoPE、amax 和 INT8 quant。
3. `decode_sparse_attn.py`：Online Softmax 的 rowmax/sub/exp/rowsum/div 与状态 merge。
4. `prefill_sparse_attn.py`：prefill 动态长度和 mask。

第二批：

5. Compressor/HC：colreduce、transpose、fillpad。
6. Gate/Indexer：gather、sort32、mrgsort、动态 top-k tail。

对包含 Cube/Vector pipe 的 kernel，还必须增加：

- C2V 和 V2C producer/consumer 配对。
- FIFO 首次迭代、稳定状态和超过 `slot_num` 后的 ring wraparound。
- split=none/up-down/left-right 中实际被 DS v4 使用的形式。
- `tpop -> vector chain -> tpush/tfree` 边界保持测试。
- A5 多轮压力运行，验证无死锁、无提前 free、无 slot 数据串扰。

每个用例至少覆盖最小 Shape、典型 Shape、最大计划 Shape和非整齐动态有效长度。

### 8.7 性能验收

性能比较必须使用同一算法、同一 Shape 和同一编译选项，记录：

- Fusion 前后 `scf.for` 数量。
- 中间 UB `vload/vstore` 数量和消除位置。
- VMI layout materialization、interleave/pack 数量。
- VPTO/LLVM/ASM 的关键指令序列。
- physical vreg pressure 或编译器 spill/materialization 迹象。
- A5 kernel latency、方差和相对 legacy baseline。

首期不要求每个单 op 都优于 legacy，但关键融合链不得因 layout materialization 抵消 UB
访存收益。若性能回退，必须能定位到 TileLib schedule、Fusion、layout 或物理 lowering。
后端 capability/cost 接口具备后，Fusion 规划可以记录：

```text
physical_cost(v) = backend.estimate(layout, dtype, logical_lanes)
peak_live_cost = backend.estimate_peak(live logical values)
```

VMI pass 不硬编码 2048-bit chunk 公式、特定 width 阈值或 A5 寄存器预算。layout、group
slot 或 lane stride 可能改变实际成本，最终以后端 IR、汇编和 A5 数据为准。

测量协议：

- 每个 case 固定 Shape、dtype、编译参数、频率设置和输入分布。
- 至少 20 次预热、100 次测量，报告 median、P90 和变异系数；不能只报最优单次。
- 以 legacy 和 VMI unfused 为双基线。关键链 fused latency 相对 VMI unfused 不得回退，
  相对 legacy 的允许波动先按 `3%` 设置，Phase 0 可根据机器噪声校准后冻结。
- 性能 Gate 必须同时附带 IR/ASM 计数；若 latency 改善但数值、同步或边界 case 未通过，
  仍视为失败。

## 9. 分阶段实施计划

### 9.1 VMI TileLib 分批交付与 PR 拆分

TileLib 不按现有 MI 库逐项翻译，只为实际 `PIPE_V` semantic form 提供 canonical VMI
implementation。每个 VMI producer 批次必须形成以下闭环后才能合入：

```text
PyPTO/PTO operand form
  -> provider 唯一匹配
  -> Expand/Inline 后得到 surface VMI
  -> round/sat/mask/memory effects 语义完整
  -> VMI verifier 和结构检查通过
```

`LayoutAssignment -> VMIToVPTO -> intrinsic/A5` 作为独立 backend 状态线。它必须在系统
合入前闭环，但其失败通过 backend issue 跟踪，不反向削弱 producer 语义。

以下内容不进入 VMI TileLib：`PIPE_MTE` 的 `tload/tstore`、`PIPE_M` Cube 计算、
`PIPE_FIX`、纯标量控制，以及 `tpush/tpop/talloc/tfree` 通信协议。它们继续走现有
PTODSL/VPTO 或专项 lowering，只作为 VMI Fusion 边界。

| 批次 | 时间盒 | 合入内容 | 主要 Gate |
|---|---:|---|---|
| B0 Coverage 与协议 | 2 天 | 从 DS v4 实际 PTO IR 生成 `(op, pipe, semantic_form, dtype, logical_width, valid_shape, attrs)` manifest；冻结 logical-row、form key、fallback 和诊断协议 | 所有 `PIPE_V` form 均归类，无未知项；非 Vector op 不进入 VMI provider |
| B1 静态基础族（当前基线） | 已具备 | semantic-form dispatch；基础 unary/binary/scalar 模板；静态 full-1VL Elementwise 和现有 Reduce/Expand/Convert | 保持现有 PTODSL 和 VMI lit 回归通过 |
| B1W Wide logical-row 基础 | 3 天 | 用 logical-row map 替换 `cols == dtype-native VL`；wide/compact load-store；`tcvt` 接入 `extf/truncf`；Elementwise 支持已验证的 wide logical width | `[8,128]xbf16 -> [8,128]xf32 -> tmul` 和 `[8,256]xf32` Expand/Inline 语义正确；每个 TileOp 只有一个 row loop |
| B2 FA/RMSNorm Reduce/Expand | 4 天 | wide `trowmax/trowsum`，compact state，`trowexpandsub/trowexpandmul/trowexpanddiv`，`trecip/trsqrt`，P0 convert 矩阵 | Online Softmax、RMSNorm 生成语义完整 VMI；Reduce 结合顺序和 precision/tmp form 有测试 |
| B3 动态 valid shape 与 mask | 3 天 | valid row/inner SSA 进入模板；typed prefix mask；masked store；tail-safe load 契约；`tfillpad` 最小 form | `valid=0/1/VL-1/VL`、随机动态值通过；inactive lane 不越界且无可观察写入 |
| B4 RoPE/Attention 辅助与重排 | 4 天 | `tgather:index`、`tgather:mask_pattern`、`trowexpand/tcolexpand*`、`tcolmax/tcolsum`、`tcolexpandexpdif`、`ttrans` | QKV/RoPE、Attention 辅助路径无未知 `PIPE_V` form；lane 顺序、layout 和 col-reduce phase 正确 |
| B5 Gate/Indexer 专项 | 独立评审 | `tsort32`、`tmrgsort:format1/format2`；先补 VMI sort 语义和 verifier | 设计评审通过后再写模板；不允许模板内直接拼物理 VPTO，layout/VMIToVPTO 单独跟踪 |

批次依赖为：

```text
B0 -> B1W -> B2 -> B3 -> B4
                  \
                   -> VMI Fusion / mem2reg 可以基于 wide P0 主链并行开展

B5 独立于 FA 主链，不阻塞 B2/B3 的深融合验收。
```

截至 2026-07-23 的 producer 状态：B0、B1W、B2 已具备静态 VMI 实现；B4 中
`tgather:index` 的 DSv4 64-lane vertical slice 已完成。其 backend 和 A5 状态保持独立
未关闭。B3 dynamic valid shape、B4 其余重排 form、实际 VMI Fusion 和 mem2reg 尚未
开始，不能计入本轮已交付能力。

其中 B1W-B4 应分别提交 PR，不把 wide row contract、十余个算子、动态 mask 和 sort 混在同一个
PR。每个算子/form 在 coverage manifest 中维护以下状态：

```text
missing
-> registered
-> expand_inline_pass
-> vmi_semantics_pass
-> structural_fusion_candidate 或 boundary_only
-> fusion_eligible
```

Backend 另行维护：

```text
not_evaluated -> vmi_to_vpto_pass -> numeric_unfused_pass -> a5_pass
```

`registered` 不能计为“producer 已支持”。至少达到 `vmi_semantics_pass` 才计入 VMI TileLib
producer coverage；单层 logical-row 检查只能标记为 `structural_fusion_candidate`，只有
通过访问、alias、mask、escape 和 phase 合法性检查后，才标记为 `fusion_eligible`。

本部分归入 VMI TileLib 与 mem2reg 责任域，不与 Fusion 规划责任域拆分：

- 负责 B0 manifest、provider/form dispatch、logical-row 公共模板族和 B1W-B4 candidate 实现。
- 负责逐 form 的 surface VMI 语义和 VMI verifier/结构测试；后端 closure 以 issue 交接。
- 负责维护 producer coverage 状态，并把达到 `vmi_semantics_pass` 的 form 分类为
  `structural_fusion_candidate` 或 `boundary_only`；最终 `fusion_eligible` 由后续
  Fusion 合法性分析产生。
- B2 完成后冻结 FA/Online Softmax P0 TileLib surface；后续 B3/B4 以增量 PR 合入，不能持续
  改写 P0 模板契约而阻塞 Fusion 和 mem2reg。

由于 TileLib 和 mem2reg 位于同一责任域，一个月内的硬承诺范围收敛为 B0-B3。B4 只补 DS v4
关键 FA/RoPE 路径实际命中的阻塞 form；B5 只完成 coverage、语义设计和必要时的最小
vertical slice，不承诺全量 sort/mrgsort。不能以追求 MI 库逐项对齐为由挤占 FA 深融合和
mem2reg 的正确性验证时间。

### 9.2 代码改造落点与 PR 顺序

| PR | 主要文件/模块 | 内容 | 完成判据 |
|---|---|---|---|
| W0 LogicalRowMap | `ptodsl/ptodsl/_tile_template_tracing.py` | logical row/offset、wide/compact mask 和 load/store；保留旧1VL回归 | `64/128/256xf32` 均只生成一个 row loop |
| W1 Wide Map/Convert | `ptodsl/ptodsl/vmi_tilelib.py`、provider helper | Elementwise 默认 lanes=Tile inner；`tcvt` 按 dtype relation 选择 `extf/truncf`；logical-width 诊断 | RMSNorm widening 和 QKV wide VMI 通过 |
| W2 Reduce/Broadcast | `vmi_tilelib.py`、tracing surface | wide RowReduce、compact state、Row/Col Broadcast、ColReduce accumulator | Softmax/RMSNorm 独立 lowering 正确，ColReduce phase IR 正确 |
| W3 Dynamic Mask | operand spec/helper、tracing、candidate validators | dynamic valid row/inner、typed prefix mask、masked store、tail contract | `0/1/max-1/max` 与随机 tail 正确 |
| F0 Region/Phase | TileFusion 路由和 VMI analysis | ptodsl-vmi 复用粗 region，识别 Map/Reduce/Broadcast/Layout phase | 稳定 phase/group dump 和拒绝原因 |
| F1 Loop Fusion | VMI-specific transform | 同 row loop merge、ColReduce producer 融入 accumulator loop | 循环数按预期下降，控制/内存顺序保持 |
| M0 Mem2Reg | VMI access analysis/transform | wide/compact exact RAW forwarding、escape 和 mask 检查 | 中间 UB 往返消除，负向 alias 用例保守 |
| I0 Backend integration | Layout/VMIToVPTO、CA model、A5、artifact tools | 后端物理化、数值、IR/ASM/latency/pressure | 独立 backend Gate 有可审计结果 |

W0-W3 和 M0 属于 VMI TileLib 与 mem2reg 责任域；F0-F1 属于 Fusion 规划责任域。接口
冻结点是 Inline 后的 canonical logical-row IR：每个 unit 提供一个主循环、静态 logical
width、完整 mask/memory effects
和可追踪 Tile storage root。W1 合入后 F0/F1 可用2VL Elementwise链并行开发；W2 合入后
再打开 RowReduce/ColReduce 深融合，不要求 Fusion 侧等待 W3 的完整动态 mask。

### Phase 0：基线与协议冻结（7 月 23 日 - 7 月 24 日）

**输入**：最新 pypto-lib DS v4、PyPTO、PTOAS feature-vmi。

**工作**：

- 生成实际 lowering coverage manifest，而不是只统计源码 API。
- 固化 legacy、VMI unfused 的 IR、输出和真机基线。
- 更新 ADR：fusion_region 复用、semantic form、phase logical lanes、fallback 协议。
- 从 mixed-kernel lowering 后的 PTO IR 统计 DS v4 实际 pipe forms，并决定通信 lowering 的
  最小 vertical slice。
- 选定 RMSNorm、QKV/RoPE、decode/prefill sparse attention 四个首批 harness。

**输出/验收**：

- 所有 `PIPE_V` op/form 均被分类为已有、P0、P1 或 P2。
- 所有 `tpush/tpop` lifecycle form 均被分类为已支持、VPTO communication blocker 或非本期。
- logical-row/wide-vreg 协议影响点、支持宽度矩阵和必要的 PyPTO retile 清单明确。
- 每个首批 harness 都有可重复的 legacy baseline。

### Phase 1：Wide logical-row 与 P0 基础 TileLib（7 月 27 日 - 7 月 31 日）

**工作**：

- 用 `LogicalRowMap` 替换当前只接受 dtype-native VL 的 `CanonicalBlockMap` 路径。
- 扩展 tracing 的 wide/compact load、store、mask、`extf/truncf` surface。
- 先打通 `[8,128]xbf16 -> [8,128]xf32`，再验证 `[8,256]xf32` Elementwise。
- 补 elementwise、full、convert、abs/neg/div/min 基础 forms 的 logical-width 泛化；
  `PIPE_S tci` 不进入 VMI TileLib。
- 为 VMI provider 接通 TileFusion region 规划，但暂不执行 legacy low-level fusion。

**输出/验收**：

- P0 基础 op 的 1VL/2VL/4VL 与 compact L1/L2 正负向测试通过。
- BF16/FP16 widening、FP32 narrowing 和 DS v4 关键转换通过单 op数值测试。
- Expand/Inline 后每个 wide TileOp 只有一个 row loop，VMIToVPTO 后物理 chunk 数正确。
- region 内 Expand/Inline 后结构稳定，MTE/Cube 仍为 hard boundary。

### Phase 2：Wide Reduce/Expand 与动态尾块（8 月 3 日 - 8 月 7 日）

**工作**：

- 补 wide row reduce、compact state、row/col expand、rsqrt/recip。
- 完成动态 `valid_inner` mask、fillpad 前置能力。
- 按冻结后的 logical-row 协议检查 RMSNorm/QKV/Attention；只在 physical chunk 压力、
  op/layout 不支持或 width 动态不可界定时要求 PyPTO retile。
- 实现 region 内 VMI phase 识别和兼容性诊断。

**输出/验收**：

- RMSNorm 和 QKV/RoPE 的 VMI unfused E2E 正确。
- `[32,128]xf32` Softmax 和 `[8,256]xf32` QKV 主链具备稳定单 row-loop IR。
- decode/prefill 的 tail mask case 可编译并通过数值测试。
- phase dump 能解释每个 loop 为什么融合或拒绝。

### Phase 3：FA/Online Softmax 深融合与 mem2reg（8 月 10 日 - 8 月 14 日）

**工作**：

- 实现兼容 VMI loop fusion。
- 实现 LocationKey、alias 判定和同迭代 RAW forwarding。
- 直接 forwarding wide logical store/load，消除 convert、elementwise、row compact state
  中间值的安全 UB 往返。
- 完成 external user、mask、WAW/WAR、MayAlias 的负向测试。
- 统计 fused loop 的 peak live physical chunks，对2VL/4VL设置首版保守 Gate。

**输出/验收**：

- `rowmax -> row_expand_sub -> exp -> rowsum/div/convert` 关键链形成预期深融合 phase。
- `bf16 widen -> f32 elementwise -> rowreduce` 在 VMI 层保持一个 wide SSA 数据流。
- VMI unfused 与 fused 的四路径、三组差分通过。
- IR 中预期中间 load/store 被消除，region output 和 sync 保留。
- decode/prefill sparse attention 首批场景在 A5 上数值正确。

### Phase 4：P1 关键覆盖、真机性能与 P2 设计（8 月 17 日 - 8 月 21 日）

**工作**：

- 补 colreduce、transpose、fillpad、mask-pattern gather 等 P1 forms。
- 完成 Compressor/HC E2E。
- 收集 fused/unfused/legacy VPTO、LLVM、ASM 和 A5 latency。
- 检查 wide logical op 的物理 chunk 排列、materialization、vecscope 和寄存器压力；必要时
  对超大 width 增加 retile/boundary 策略，不在 TileLib 中恢复动态 inner loop。
- 完成 VMI sort/mrgsort 语义与 lowering 设计，并实现可收敛的最小 P2 vertical slice。
- 若 DS v4 pipe manifest 只包含少量 form，完成最小
  `LowerPipeCommunicationToVPTO` vertical slice；否则保持独立 blocker，不压缩 Fusion
  和 mem2reg 的正确性验证。

**输出/验收**：

- FA/RMSNorm/QKV/Compressor 关键路径通过回归和真机验证。
- 性能收益或回退有可追踪的 IR/ASM 解释。
- Gate/Indexer 的剩余阻塞只允许是已登记的 P2 sort forms，不得有未知缺口。
- 混合 Cube/Vector kernel 的 pipe forms 有明确支持矩阵；未支持时不得计入 VMI E2E 覆盖。

> 按当前投入规模，一个月内同时完成全部 P0/P1、VMI Fusion、mem2reg、动态 tail 和完整
> sort/mrgsort、通用 tpush/tpop VPTO lowering 风险过高。承诺范围应是 Phase 0-3 和关键
> P1；P2 与 pipe communication 至少完成实际 form 清单、设计和最小 vertical slice，完整
> 覆盖作为紧接的后续里程碑。

## 10. 责任域与接口

### Fusion 规划责任域：Fusion region、phase 与 loop fusion

- TileFusion region 路由、VMI phase 识别和 loop fusion。
- 输出稳定的 fusion group/phase、loop mapping 和拒绝原因。

### VMI TileLib 与 mem2reg 责任域：VMI 语义、Alias 与 promotion

- DS v4 lowering manifest 和 TileOp/form coverage。
- PTODSL tracing、provider semantic-form dispatch、dtype/valid shape 支持。
- P0/P1 VMI TileLib implementation、surface VMI 语义和 verifier/结构验证。
- 定义并实现 LocationKey、storage root、offset/span/mask 规范化。
- VMI mem2reg、external escape 和必要 store 保留。
- alias/WAW/WAR/MayAlias 正负向测试。
- VMI-unfused/VMI-fused 的语义与结构差分。
- 后端失败最小复现、预期 VMI contract 和问题清单交接。

### 责任域接口

Fusion 规划责任域交付给 mem2reg 的输入必须包含：

- 一个已验证合法的 `pto.fusion_region` 和内部 fused VMI loop/group。
- 每个 access 可追踪的 storage root 和线性 offset SSA。
- 保留的 mask、element type、logical shape 和 region escape 信息。
- 不跨 sync/DMA/Cube/unknown-effect boundary。

VMI TileLib 与 mem2reg 责任域输出给后续 lowering 的结果必须保证：

- IR verifier 通过，VMI SSA dominance 和 memory effects 合法。
- 无法证明的访问保持原状。
- 被删除的 store/load 有可打印的 promotion reason。
- 输入满足已发布的 VMI producer contract；后端是否支持该 dtype/width/layout 组合由后端
  capability 和测试独立判断。

## 11. 里程碑 Gate

| Gate | 必须满足 | 不满足时处理 |
|---|---|---|
| G0 Coverage | lowering manifest 无未知 `PIPE_V` form | 不进入批量实现 |
| G1 TileLib | P0 的1VL/2VL/4VL/compact 单 op unfused 数值正确，且每个 form 只有一个主循环 | 不允许进入 Fusion |
| G2 Dynamic | tail mask/valid shape 边界正确 | 不进入动态 DS E2E |
| G3 Fusion | 正负向依赖/alias/mask 测试通过 | 保持独立 loops |
| G4 mem2reg | exact RAW promotion 与 escape 保留正确 | 保留 UB load/store |
| G5 E2E | RMSNorm、QKV、decode/prefill Attention 的四路径、三组差分通过 | 不做性能结论 |
| G6 Pipe | mixed kernel 所需 pipe forms 已 lower，或明确排除出 E2E | 不宣称完整 DS E2E |
| G7 Hardware | A5 正确、无 hang、无非法指令 | 不合入默认路径 |
| G8 Performance | 关键链 latency 不回退，peak physical chunks/materialization 可接受且收益可解释 | 限制 width/Fusion group 或定位 schedule/layout |

## 12. 主要风险与控制

| 风险 | 影响 | 控制措施 |
|---|---|---|
| TileLib 继续把 logical vreg 当成一个物理 vreg | widening/multi-VL 被错误拒绝 | 使用 logical-row/wide-vreg 协议，并增加 extf/truncf 与2VL/4VL专项测试 |
| Wide vreg 类型合法但物理寄存器压力过高 | spill/materialization 或性能回退 | 估算 peak live chunks，检查最终 ASM，超预算时缩小 group 或要求 retile |
| 源码统计替代 lowering 清单 | 漏 form、漏 attrs | G0 强制实际 PTO IR manifest |
| fusion_region 被当成融合证明 | 错误合并不同 phase | region 只做粗边界，VMI 重新做兼容性分析 |
| 只按 Tile SSA 做 alias | UB 地址复用时误优化 | 使用 storage root + offset/span + physical provenance |
| PlanMemory 早于 VMI Fusion | loop 交错后原地址复用失效 | 检查 planned range 与新 schedule lifetime，无法证明则拒绝 |
| 动态 mask 仅验证典型长度 | 尾块越界或脏写 | 固化 0/1/VL-1/VL 和动态随机长度 |
| 直接拼物理 sort 指令 | 污染 VMI 抽象和 layout | P2 新增 unified VMI 语义与 lowering |
| 把 tpush/tpop 当 VMI TileLib | 无法表达 FIFO、跨 core sync 和所有权 | 独立 communication lowering，VMI 只建模边界 |
| pipe lowering 缺失却做完整 E2E | VPTO 编译失败或协议不完整 | extracted vector harness 与 mixed-kernel Gate 分开验收 |
| 以编译通过代替数值正确 | 精度/rounding 回归 | 四路径差分 + A5 真机 Gate |
| 融合收益被 layout 开销抵消 | 性能目标不达成 | 记录 materialization/vreg/ASM，按阶段归因 |

## 13. 最终交付物

- DS v4 PTO TileOp/VMI form coverage manifest。
- 更新后的 VMI VF Fusion ADR 和 pass pipeline。
- P0/P1 VMI TileLib、provider/tracing 和 dynamic mask 支持。
- logical-row/wide-vreg 公共模板、支持宽度矩阵和物理 chunk 压力报告。
- Tile fusion region 到 VMI phase/loop fusion 的完整链路。
- VMI alias analysis 与 mem2reg。
- 单 op、lit、差分、DS kernel 和 A5 真机测试集。
- fused/unfused/legacy 的 IR、VPTO、LLVM、ASM、数值和性能报告。
- Gate/Indexer P2 sort/mrgsort 的设计、vertical slice 和剩余工作清单。
