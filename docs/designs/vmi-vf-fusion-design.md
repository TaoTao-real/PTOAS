# VMI VF Fusion RFC 最小实现设计

## 1. 文档状态

- 状态: Proposed
- 日期: 2026-07-23
- 决策记录: [ADR-0001](adr/0001-vmi-vf-fusion-rfc-minimal-pipeline.md)
- 参考设计: 上游 `RFC-vf-fusion-on-vmi.md`

本文定义 PTOAS `feature-vmi` 分支上第一阶段 VMI VF Fusion 的编译器边界和 pass
协议。第一阶段只建立 RFC 的正确性闭环，不引入多 candidate、算法专项 schedule、
cost model 或 autotuning。

## 2. 目标与非目标

### 2.1 目标

1. `PIPE_V` TileOp 通过 PTODSL VMI TileLib 展开为独立正确的 canonical VMI 实现。
2. Expand 和 Inline 后，从真实 VMI IR 中识别由 TileLib 产生的独立 fusion unit。
3. 保守合并结构兼容的相邻 VMI 循环。
4. 在融合后通过 VMI mem2reg 消除可证明安全的中间 UB store-load。
5. 利用 logical-row contract，把同一 row 的 Convert -> Elementwise -> Reduce ->
   Broadcast 链合并进一个 row runtime loop；logical row 可以映射为一个或多个物理 VL。
6. VMI 模板和 Fusion 保持 scope-free；`VMIToVPTO` 后由现有
   `PTOInferVPTOVecScope` 统一推断物理 VPTO vecscope。
7. 任何分析失败均保持原独立实现，不能改变程序语义。

### 2.2 非目标

- 不优化任意用户手写 VMI 或任意用户控制流。
- 不为一个 TileOp 注册多个 schedule candidate。
- 不做 candidate selection、candidate locking 或 region-aware specialization。
- 不在 TileLib 中生成动态 inner-block loop，不做多 schedule candidate、代价模型或自动调优。
- 不承诺任意宽度都具备最优物理调度；首期只开放 DS v4 验证过的静态 logical width。
- 不替换 VMI layout assignment、物理 vreg 分配或 VMI-to-VPTO lowering。

## 3. 当前基础与缺口

当前已有链路为：

```text
TileOp
  -> ExpandTileOp(--tile-lib-backend=ptodsl-vmi)
  -> func.call @__pto_ptodsl_vmi_...
  -> PTOInlineLibCall
  -> 多个独立 scf.for + pto.vmi.*
  -> FoldTileBufIntrinsics
  -> VMI semantic/layout pipeline
  -> VPTO
```

现有 canonical VMI TileLib 已能让静态 Softmax compute harness 中的一组基础 TileOp
独立 lower，并已打通静态 `[64,64]xf32` FlashAttention 多项式样例。该样例不包含标准
Softmax 的 RowReduce 归一化，尚不代表完整 FA/Online Softmax 已覆盖。Expand + Inline
后每个 TileOp 仍有自己的循环。例如 `tadd -> texp` 会得到：

```mlir
scf.for %i = %c0 to %blocks step %c1 {
  %a = pto.vmi.vload %src0[%off_i] ...
  %b = pto.vmi.vload %src1[%off_i] ...
  %x = pto.vmi.vadd %a, %b, %mask ...
  pto.vmi.vstore %x, %tmp[%off_i], %mask ...
}
scf.for %j = %c0 to %blocks step %c1 {
  %x = pto.vmi.vload %tmp[%off_j] ...
  %y = pto.vmi.vexp %x, %mask ...
  pto.vmi.vstore %y, %dst[%off_j], %mask ...
}
```

缺口不是 VMI op lowering，而是：

- Inline 后缺少稳定的 TileLib fusion-unit provenance。
- 没有 VMI 循环兼容性、依赖和 alias 分析。
- 没有 VMI loop merge。
- 没有 fusion-after mem2reg。
- VMI core 已支持 wide logical vreg、widening layout 和多物理 vreg lowering，但当前
  TileLib `CanonicalBlockMap` 仍把 dense row 限制为一个 dtype-native VL。

## 4. Canonical VMI TileLib 协议

### 4.1 唯一实现

RFC 模式下，每个 `(target, op, semantic_form)` 必须恰好注册一个 canonical VMI
`TileTemplate`：

```text
(a5, tadd, tile_tile) -> exactly one canonical VMI implementation
(a5, tdivs, scalar_tile) -> exactly one canonical VMI implementation
```

同一个模板可以根据静态 dtype/shape 做合法特化，但 provider 不在多个模板之间进行
性能选择。零个实现是 coverage error；多个实现是 provider contract error。

provider module 必须暴露独立的 `VMI_TILELIB_REGISTRY`，canonical 模板通过
`@canonical_vmi_template` 显式注册。helper 只查询该 registry，不通过扫描 module
全局变量发现模板，也不与普通 PTODSL TileLib 的 VPTO/MI registry 混用。

### 4.2 独立正确性

每个 canonical 实现必须包含完整 load/compute/store，在没有 fusion pass 时也能生成
语义完整、verifier 合法的 surface VMI。Fusion 只能删除已证明冗余的控制和访存，不能
成为 TileOp 语义完整的前置条件。LayoutAssignment、VMIToVPTO 和 A5 数值闭环由后端
独立验收；后端缺口不能通过改变或削弱 VMI 语义来掩盖。

### 4.3 单主循环与 logical-row 调度

一个 canonical fusion unit 应具有一个主 `scf.for`，循环遍历 row 或该 semantic form
声明的等价逻辑单元。对于 row-major dense Tile，模板按一整行构造一个
`!pto.vmi.vreg<logical_lanes x element_type>`，不得因为该值占用多个物理 vreg 而新增
动态 `for col` / `for block`：

```text
iterationDomain = rows
logicalLanes = physicalShape.cols
validLanes <= logicalLanes
```

物理寄存器拆分不是 VMI Fusion 的循环维度。LayoutAssignment 和 VMIToVPTO 在 Fusion
之后根据目标、layout 和 lowering capability 物理化 wide logical vreg。VMI 层只保留
以下 logical value，不规定精确物理 vreg 数：

```text
128xbf16 -> !pto.vmi.vreg<128xbf16>
128xf32  -> !pto.vmi.vreg<128xf32>
256xf32  -> !pto.vmi.vreg<256xf32>
```

Convert 保持 logical lane count。`128xbf16 -> 128xf32` 在 VMI 层仍是一个
128-lane SSA value，使用 `pto.vmi.extf`；目标 layout 可为 `deinterleaved = 2`，但该
layout 不得成为 TileLib candidate 的手工选择条件。

Compact Reduce 状态同样使用 logical vreg 表达，例如 `[rows,1]` 对应每 row 一个
`vreg<1xf32>`，`[1,8]xf32` 对应 `vreg<8xf32>`。它们不是非法的“小于 1VL Tile”，但
必须通过 Reduce/Broadcast/compact semantic form 保持清晰的 row/lane 映射。

动态 Shape 使用静态最大 logical width 和 `validLanes` typed mask 表达；VMI type 的
element count 仍为静态。若运行时需要的 logical width 可能超过 candidate 的静态上限，
必须由前端重新切 Tile 或选择另一个已注册 form，不能在模板中临时生成动态 inner loop。

首期不无限开放任意 wide width。每个 form 必须声明并测试支持的 dtype、logical width、
logical layout relation 和 mask policy；未验证组合明确拒绝。后端再独立声明哪些组合
能够物理化，以及相应的寄存器压力限制。这不是“VMI vreg 固定为 1VL”。

### 4.4 不暴露物理 layout

canonical 模板只能表达 surface VMI 类型和语义 op。以下内容不能成为模板选择维度：

- contiguous / deinterleaved physical layout
- physical vreg 编号和数量
- MI post-update 地址模式
- interleave/pack/materialization 指令序列

这些信息继续由 VMI semantic/layout pipeline 决定。

### 4.5 TileLib 实现结构

当前 `CanonicalBlockMap` 同时承担 logical iteration 和一个物理 VL 的限制。实现时应拆成
以 row 为中心的逻辑映射，概念接口为：

```text
LogicalRowMap {
  rows
  logical_lanes       // static Tile inner
  valid_rows          // static or SSA
  valid_lanes         // static or SSA
  element_type
  row_offset(row)     // normalized linear element offset
}
```

PTODSL tracing/runtime helper至少提供以下 surface 能力：

```text
load_row(tile, row, logical_lanes) -> vreg<logical_lanes x dtype>
store_row(value, tile, row, mask)
create_typed_prefix_mask(valid_lanes, logical_lanes, dtype)
extf/truncf/cast preserving logical_lanes
compact load/store and row/column broadcast
```

公共 candidate family 按语义组织，不按 MI 指令逐项复制：

| Family | 主循环与 logical mapping | 代表 TileOp |
|---|---|---|
| Map | `for row`，输入输出同 lane domain | `tadd/tmul/texp/tabs` |
| Convert | `for row`，lane count 不变，dtype/chunks 可变 | `tcvt` |
| RowReduce | `for row`，wide row -> compact row state | `trowmax/trowsum` |
| RowBroadcast | `for row`，compact row state -> wide row | `trowexpand*` |
| ColReduce | 一个 `for row iter_args(wide_acc)`，输出 `[1,cols]` | `tcolmax/tcolsum` |
| ColBroadcast | `for row`，共享 `[1,cols]` wide value | `tcolexpand*` |
| Layout/Opaque | 显式 lane mapping 或 hard boundary | `tgather/ttrans/sort` |

每个 `(target, op, semantic_form)` descriptor 除 correctness attrs 外，还应声明：

```text
iteration_kind
logical_lane_relation       // same/reduce/broadcast/permutation
supported dtype signatures
verified logical widths
mask policy
phase kind
```

这些字段用于 specialization、诊断和 Fusion compatibility，不允许编码具体 physical
layout、vreg 编号或 MI post-update 策略。

## 5. Fusion region 与 provenance 协议

当前 ExpandTileOp 已给实例化函数添加：

```mlir
attributes {pto.tileop.instance = "ptodsl-vmi"}
```

Tile 层 `FusionPlan/FusionRegionGen` 负责构造粗粒度 `pto.fusion_region`。该 region 是
后续 VMI 分析的所有权、escape 和硬边界；Expand/Inline 后的 VMI 循环仍位于原 region
内部，不为每个 TileOp 再创建 `fusion_scope`。

设计实现时，ExpandTileOp 还应给实例化函数或 call 添加显式 TileOp 名称，不能依赖
解析私有函数名恢复来源。函数 inline 后 function/call attribute 会消失，因此
`PTOInlineLibCall` 可以把来源信息转移到 canonical unit 的主 `scf.for`：

```mlir
scf.for ... attributes {
  pto.vmi.fusion.source = "tilelib",
  pto.vmi.fusion.tileop = "tadd",
  pto.vmi.fusion.unit_id = 7 : i64
}
```

这些属性是编译器内部 provenance，用于恢复 TileOp/unit 对应关系和输出诊断；它们不是
candidate lock，也不承诺一定融合。融合合法性必须同时依赖 region 成员关系、实际 SCF
结构、VMI memory effects、alias、mask 和 phase，不能仅凭 marker 跨越 region 或忽略
真实副作用。

## 6. Fusion unit 与宽松分组

### 6.1 Fusion unit

一个首期 fusion unit 是满足以下条件的 TileLib 代码段：

- 带合法 provenance。
- 位于一个 Tile 层生成的 `pto.fusion_region` 内。
- 包含一个主 `scf.for`。
- 不包含显式 `pto.vecscope`；物理 vecscope 由 emission boundary 的现有 pass 推断。
- 主循环 step 可证明一致，首期要求正的常量 step。
- 循环体中的 VMI memory effect 可枚举。
- unit 外 setup 仅包含常量、mask、pointer/address 计算等可分析操作。
- 不包含未知 call、barrier、sync、DMA、Cube 或未知副作用。

### 6.2 宽松分组

`VMIPlanLoopFusion` 在每个 `pto.fusion_region` 内按 block、控制边界和数据依赖收集一段
连续 fusion units，形成宽松分组。宽松分组只表示“可以共同分析”，不表示组内所有循环
都必须合并。

随后按相邻循环逐对检查兼容性，得到一个或多个真正的 fusion groups：

```text
[unit0, unit1, unit2, unit3]
      loose analysis span

unit0 ~= unit1    -> fusion group A
unit2 incompatible -> standalone
unit3             -> standalone or later group
```

这样可以在 Tile 层粗 region 内支持部分融合；region 成员关系不等于所有 unit 都必须
融合。

## 7. Pass 设计

### 7.1 `VMIIdentifyFusionUnits`

类型：analysis / validation pass，建议作用于 `func::FuncOp`。

职责：

- 查找 TileLib provenance marker。
- 遍历 `pto.fusion_region`，拒绝 region 外 VMI 和未知嵌套 region。
- 验证 canonical unit 结构。
- 记录主循环、logical-row contract、wide/compact value、setup、storage accesses、mask uses
  和硬边界。
- 为诊断输出稳定的拒绝原因。

它不做 group selection，不修改循环，也不重新选择 TileOp 实现。跨 pass 的分析结果
必须通过 AnalysisManager 缓存或显式、可打印的 IR metadata 共享，不能使用隐藏全局
状态；若采用 AnalysisManager，`VMIPlanLoopFusion` 应直接请求同一个
`VMIFusionUnitAnalysis`，而不是依赖前一个 pass 的进程内副作用。

### 7.2 `VMIPlanLoopFusion`

类型：analysis + metadata pass，建议作用于 `func::FuncOp`。

职责：

- 在每个 Tile fusion region 内构建宽松分组并识别 Map、RowReduce、RowBroadcast、
  ColReduce、Layout 和 Opaque phase。
- 对相邻 unit 做兼容性检查。
- 生成确定性的 group id/order 或 analysis result。
- 保持原程序顺序，不做算法级重排。

可复用现有 TileFusion 的 block-local DFG、liveness、iteration-domain 代码思路；若直接
复用代码，应抽取与具体 TileOp 类型无关的 utility，而不是让 VMI pass 消费
`FusionPlan` 的 TileOp metadata。

### 7.3 `VMIFuseCompatibleLoops`

类型：transform pass，建议作用于 `func::FuncOp`。

职责：

- 为 fusion group 创建一个共享 `scf.for`。
- 将后续循环 IV 映射到第一个循环 IV。
- 按原程序顺序克隆/移动循环体。
- 保留每个 op 的 mask、属性和内存顺序。
- 删除被合并的旧循环。

首个闭环先支持同 row domain 的 resultless Map/RowReduce/RowBroadcast 循环。DS v4
ColReduce 使用带 `iter_args` 的 wide accumulator loop，作为紧随其后的明确扩展：允许
producer Map body 融入 accumulator loop，但最终 Reduce 结果的 consumer 必须进入下一
phase。复杂 region branch 和无法证明的跨迭代依赖继续拒绝。

### 7.4 `VMIMem2Reg`

类型：transform pass，建议作用于 `func::FuncOp`。

必须在 loop fusion 后运行。首期处理同一融合循环、同一迭代中的：

```mlir
pto.vmi.vstore %x, %tmp[%off], %mask
...
%reload = pto.vmi.vload %tmp[%off]
```

当 location、值 shape、访问覆盖范围和 mask obligation 均可证明兼容时：

```mlir
// %reload users 改用 %x
// 删除冗余 vstore/vload
```

如果中间 Tile 仍有 fusion group 外用户，或 store 可能被其他访问覆盖，则不能删除
对外可观察的 store。跨迭代 promotion 到 `scf.for iter_args` 属于后续扩展。

### 7.5 VecScope 插入边界

不新增 VMI vecscope coalescing pass。canonical VMI 模板和 Fusion transform 都不显式
生成 `pto.vecscope`，只维护合法的 SCF、VMI SSA 和内存顺序。现有
`PTOInferVPTOVecScope` 在 `VMIToVPTO` 之后、LLVM emission 之前基于物理 VPTO 操作自动
划分 resultless vecscope；DMA、sync、barrier 和无法安全移动的操作继续作为其边界。

## 8. 兼容性与安全判据

两个 unit 只有全部满足以下条件才可合并。

### 8.1 控制边界

- 位于同一 block，且保持原顺序。
- 中间没有 call、sync、barrier、DMA、Cube、未知副作用或 region boundary。
- 首期不跨 `scf.if`、外层 `scf.for` 边界移动 unit。

### 8.2 迭代域

- lower、upper、step 相同 SSA，或可由简单 affine/canonical expression 证明等价。
- 动态 shape 可以支持，但两个循环必须共享同一动态 bound SSA 或可证明等价。
- logical width 可以对应 1/2/4 个物理 VL，但不参与主循环 trip count；Map、RowReduce 和
  RowBroadcast 都应保持 row trip count，禁止 Elementwise 私自平坦化为 `rows * chunks`。
- Convert 前后 dtype 可以不同，但 logical lane count、row mapping 和 active-lane
  obligation 必须兼容；物理拆分变化不进入该判据。
- trip count 或 logical row mapping 不同则不融合；ColReduce 的 loop-carried accumulator
  与最终结果 consumer 按 phase 规则处理。

### 8.3 Alias 与依赖

VMI load/store 的首期 location key 定义为：

```text
LocationKey = (
  storage root,
  normalized linear offset,
  per-iteration accessed span,
  VMI value shape and element type,
  dist/group/block-stride mode
)
```

`storage root` 需要穿透 `tile_buf_addr`、合法 cast 和可规范化 addptr；offset 只处理
常量、IV 和简单 affine arithmetic。规则为：

- 可证明 NoAlias：允许保持顺序后融合。
- 精确 RAW：允许融合，mem2reg 可进一步判断是否提升。
- WAW/WAR：只有保持顺序且可证明每迭代访问关系安全时允许。
- MayAlias 或无法规范化：保守拒绝。
- 不允许把同一迭代依赖误判成跨迭代依赖，反之亦然。

现有 VMI 使用线性 offset；后续若引入 shaped pointer / multidimensional index，可替换
LocationKey 的构造方式，不改变 pass 顺序和保守原则。

### 8.4 Mask

- 融合不能丢失任何 consumer mask。
- A5 `vload` 不可谓词化，promotion 后 consumer 的 mask obligation 仍存在。
- store 的 mask/pmode 与 load 后所有 consumer 的有效 lane 关系无法证明时，不做
  mem2reg。
- 动态 tail mask 只要由同一 bound/remaining SSA 推导且逐 use 保留，可以参与融合；
  首期 provider 尚未生成动态 valid-shape tail，因此先完成静态 mask 用例。

## 9. Pipeline 顺序

VMI provider 的目标顺序为：

```text
InsertTemplateAttributes
  -> FusionPlan / OpScheduling / FusionRegionGen
  -> existing memory planning / auto-sync lifecycle
  -> ExpandTileOp(--tile-lib-backend=ptodsl-vmi)
  -> PTOInlineLibCall
  -> FoldTileBufIntrinsics(shape-only)
  -> VMIIdentifyFusionUnits
  -> VMIPlanLoopFusion
  -> VMIFuseCompatibleLoops
  -> canonicalize / CSE
  -> VMIMem2Reg
  -> canonicalize / CSE
  -> FoldTileBufIntrinsics(addr-only)
  -> existing VMI semantic/layout pipeline
  -> VMIToVPTO
  -> existing PTOInferVPTOVecScope at the VPTO emission boundary
```

关键顺序约束：

- Expand + Inline 之前看不到真实 VMI 循环，不能做 VMI loop compatibility 分析。
- Tile 层 region 只提供粗粒度边界，不能代替 VMI phase/loop/alias/mask 合法性分析。
- shape-only folding 先暴露静态/动态 loop bound。
- mem2reg 必须在 fusion 之后，才能看到原本位于不同循环体的 store-load。
- layout assignment 必须在 fusion/mem2reg 之后，避免物理 layout 细节污染判据。
- addr-only folding 放在分析之后，以保留 Tile handle/storage provenance；分析需要能
  追踪 `tile_buf_addr` 的 root。
- vecscope inference 必须在 VMI 物理化之后统一运行，VMI Fusion 不分析或合并 scope。

## 10. 与现有 Fusion pipeline 的关系

### 10.1 复用粗粒度 region 的 pass

- `FusionPlan` / `OpScheduling` / `FusionRegionGen` 继续在 Tile-native PTO IR 上构建 DFG、
  规划顺序并产生 `pto.fusion_region`。VMI 路由复用 region 所有权和边界，但不让这些 pass
  判断展开后的 VMI loop 是否真正可融合。

### 10.2 不直接复用的低层 pass

- `PTOLowLevelLoopFusion`：输入是已经展开到 VPTO/MI 的低层循环。
- `PTOFusionLoadStoreElision`：不是 fusion-after 的 VMI SSA promotion。

### 10.3 可以复用的能力

- block-local DFG 构建框架。
- value liveness、external user、write-instance escape 的建模思路。
- iteration-domain equivalence 的部分 solver/utility。
- 确定性的 group id/order、打印和测试方式。

### 10.4 CLI 路由

当前 `--tile-lib-backend=ptodsl-vmi --enable-op-fusion` 会报错，防止误入 legacy
VPTO fusion。新 passes
完成后，`--enable-op-fusion` 应按 provider 路由：

```text
--tile-lib-backend=tilelang   -> existing Tile/VPTO fusion lifecycle
--tile-lib-backend=ptodsl-vmi -> new VMI fusion lifecycle
```

在 VMI pipeline 可用前应保留当前拒绝逻辑。

## 11. 失败与 fallback

- 缺少 VMI implementation：ExpandTileOp 明确报 coverage error，不静默回退到 MI。
- 同一 `(target, op, semantic_form)` 存在多个 VMI implementation：provider contract error。
- unit 不符合 canonical 结构：保持独立，输出可诊断拒绝原因。
- loop/domain/alias/mask 无法证明：不融合。
- mem2reg 无法证明：保留原 store/load。
- 任一 unit 独立 lowering 必须始终有效。

部分融合示例：

```text
tadd(loop=rows) + texp(loop=rows) + trowmax(loop=rows, reduce phase)

首期结果：
  依赖和 mask 兼容时，[tadd + texp + trowmax] 合并到同一个 row loop
```

这不是错误，而是 RFC 保守闭环的预期行为。

## 12. 验证计划

### 12.1 正向 lit tests

- 两个相邻 elementwise canonical loops 合并为一个 `scf.for`。
- 三个 elementwise loops 连续合并且保持 op 顺序。
- 同 location、同 offset 的中间 `vstore -> vload` 被 mem2reg 消除。
- `128xbf16 -> 128xf32 -> Mul -> RowSum` 使用 wide logical vreg，在同一个 row loop 内完成。
- `256xf32` Elementwise 链保持一个 row loop；物理拆分由后端单独验收。
- RowMax -> Broadcast -> Exp -> RowSum/Convert 在同一个 row loop 内完成。
- Map -> ColReduce producer 融入带 `iter_args` 的 accumulator loop，最终 consumer 位于下一
  phase。
- 动态 upper bound 使用同一 SSA 时可融合。
- VMI Fusion 输出不包含显式 `pto.vecscope`，最终 VPTO emission 自动推断合法 scope。
- VMI producer boundary verifier 通过；VMI-to-VPTO 无残留检查作为独立后端 Gate。

### 12.2 负向 lit tests

- 用户手写、无 provenance 的 VMI loop 不处理。
- trip count、step 或 offset mapping 不一致时不融合。
- 中间存在 sync/call/unknown side effect 时不融合。
- MayAlias、WAW/WAR 无法证明时不融合。
- mask/pmode 不兼容时不做 mem2reg。
- 中间 Tile 有 group 外用户时保留必要 store。
- logical width、dtype、logical layout relation 或 mask policy 超出该 form 的已验证矩阵时拒绝。
- 模板生成动态 inner-block loop、Elementwise 平坦化为 `rows * chunks` 时拒绝 canonical
  contract。

### 12.3 端到端基线

- 现有 PTODSL VMI TileTemplate Python test 保持通过。
- composite provider 和 no-vector-fallback lit tests 保持通过。
- 静态 Softmax compute-op coverage harness 完成 Expand、Inline、VMI-to-VPTO。
- 静态 `[64,64]xf32` FlashAttention 多项式样例完成 Expand、Inline、VMI-to-VPTO。
- DS v4 RMSNorm `[8,128]xbf16 -> [8,128]xf32` 完成 unfused/fused 差分验证。
- DS v4 Attention `[32,128]xf32` Softmax 主链完成 wide row-loop 深融合。
- DS v4 QKV `[8,256]xf32` 完成 wide logical lowering；物理压力检查由后端 Gate 覆盖。
- M3 要求 RowMax -> Broadcast -> Exp -> RowSum/Convert 关键链在同一 row loop 内深融合；
  动态 tail 和任意 Shape 泛化不作为门槛。

## 13. 后续迭代

基本闭环稳定后，再分别设计和评审：

1. 动态 valid shape 和 tail mask 完整覆盖。
2. Gather/Transpose/Pack 等非 identity logical-lane mapping 泛化。
3. 多 canonical schedule candidate 与 region-aware selection。
4. outer-row unroll、重读/保活选择和 physical vreg pressure cost。
5. 超大 wide vreg 的 chunk-wise 物理调度和必要的重新切 Tile 策略。
6. FA/Softmax 专项 schedule、cost model 和性能验收。
