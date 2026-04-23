# PTOAS Auto Sync Subview Multibuffer V1

## 1. 结论

本 PR 只支持一种 multibuffer 使能方式：

- 用户显式分配一块已经扩好的 workspace。
- 用户用 `pto.subview` 或 lowering 后的 `memref.subview` 切出 leaf buffer。
- 每个参与 multibuffer 的 leaf subview 显式标注：
  - `pto.multi_buffer_factor`
  - `pto.multi_buffer_slot`
  - 可选 `pto.multi_buffer_group`
- 用户自己在 IR 中写出 selector/control-flow，例如 `%iv % factor` + `if/else`。
- `PTOInsertSync` 基于这些显式槽位信息分析依赖、分配 event id、插入同步。

本 PR 不支持：

- 在 root buffer 或 alloc 上标 `pto.multi_buffer`
- 通过 `EnableMultiBuffer` 或其他 pass 自动展开 ping/pong/triple-buffer 逻辑
- 自动把单 buffer 扩成 multibuffer workspace
- 通过启发式把两块独立 `pto.alloc_tile` 推断成同一组 ping/pong

换句话说，这次 PR 的语义非常收敛：`multibuffer = 显式 annotated subview`。

## 2. 用户契约

### 2.1 共同 root workspace

用户先显式分配一块已经扩好的 workspace。共同 root 本身就是隐式 family，不再额外暴露单独的 family annotation。

```mlir
%workspace = pto.alloc_tile : !pto.tile_buf<vec, 32x16xf16>
```

### 2.2 leaf subview 上显式标槽位

每个 leaf subview 必须显式带 `factor + slot`，多 group 场景再额外带 `group`：

```mlir
%ping = pto.subview %workspace[%c0, %c0] sizes [16, 16]
  {pto.multi_buffer_factor = 2 : i32,
   pto.multi_buffer_slot = 0 : i32}
  : !pto.tile_buf<vec, 32x16xf16> -> !pto.tile_buf<vec, 16x16xf16>

%pong = pto.subview %workspace[%c16, %c0] sizes [16, 16]
  {pto.multi_buffer_factor = 2 : i32,
   pto.multi_buffer_slot = 1 : i32}
  : !pto.tile_buf<vec, 32x16xf16> -> !pto.tile_buf<vec, 16x16xf16>
```

规则：

- `pto.multi_buffer_factor` 是显式契约，不做内部默认。
- 当前支持 `2 <= factor < 8`。
- 不按“同组 subview 个数”反推 factor。
- `pto.multi_buffer_slot` 必须落在 `[0, factor)`。
- 单 group 场景下，`pto.multi_buffer_group` 可省略，默认视为 `0`。
- 多 group 共享同一个 root workspace 时，必须显式标 `pto.multi_buffer_group`。

### 2.3 selector 也必须由用户显式写出

PTOAS 这次不会自动生成 ping/pong 切换逻辑。用户需要自己写 `%iv % factor` 和分支：

```mlir
scf.for %iv = %c0 to %cN step %c1 {
  %slot = arith.remui %iv, %c2 : index
  %is_ping = arith.cmpi eq, %slot, %c0 : index
  scf.if %is_ping {
    // use slot 0
  } else {
    // use slot 1
  }
}
```

更完整的 group + triple-buffer 示例见 [multibuffer-root-group-slot-demo.pto](./multibuffer-root-group-slot-demo.pto)。

## 3. 为什么这次要收敛到显式 subview

当前自动同步真正需要的是“槽位语义”，而不是“地址个数恰好是 2”。

对 autosync 来说，必须先回答三个问题：

1. 这些 buffer 是否属于同一个逻辑 multibuffer root。
2. 它们是否属于同一个 group。
3. 它们在 group 内分别是哪个 slot。

只有这三点都能被稳定证明，event id 才能和 slot 绑定，迭代之间才能复用正确的 lane。

如果只是看到：

- root 上有一个模糊的 multibuffer intent
- 或者 IR 里恰好出现了两个地址
- 或者用户手工写了两块独立 alloc

编译器并不能稳定证明“哪个 loop selector 对应哪个 slot”，也无法保证 event id 在跨迭代时始终绑到同一个物理槽位。

所以这次 PR 明确只接受显式 annotated leaf subview，先把正确性边界收紧，再在这个边界内把 event lane 绑定做好。

## 4. IR 约束

当前 V1 对显式 subview multibuffer 的静态要求如下：

1. 参与 multibuffer 的 leaf 必须来自同一个 root workspace。
2. `pto.subview` / `memref.subview` 的 offset、size 必须静态可知。
3. `memref.subview` 的 stride 必须是静态正整数。
4. 同一个 group 内：
   - factor 必须一致
   - 每个 leaf 的 shape 必须一致
   - slot 必须覆盖 `[0, factor)` 中的合法编号
   - 几何上必须能解释为“同一 group region 被 factor 等分后的各个 slot”
5. 不同 group 在同一个 root 下必须静态不重叠。

任何条件不满足时，整组 annotated root 会被判定为非法 multibuffer root，然后整体回退到普通 autosync。

这个“整 root 失效”的策略是刻意的：

- 它避免一半 leaf 走 slot-aware、另一半 leaf 又退回普通地址分析，造成语义撕裂。
- overlap、factor 不一致、slot 越界这类负用例都会稳定回退，不会半途产生错误的多 lane event 分配。

## 5. 实现流程

### 5.1 `PTOViewToMemref`

`pto.subview` lowering 到 `memref.subview` 时，PTOAS 会把以下 attrs 原样传递下去：

- `pto.multi_buffer_factor`
- `pto.multi_buffer_slot`
- `pto.multi_buffer_group`

因此 `PTOInsertSync` 同时支持两种 IR 形态：

- 前端原始 `pto.subview`
- lowering 后的 `memref.subview`

### 5.2 `PTOIRTranslator`

`PTOIRTranslator` 只在“显式 annotated subview”上生成 multibuffer 槽位元数据，并写入 `BaseMemInfo`：

- `multibufferRoot`
- `multibufferGroup`
- `multibufferSlot`
- `multibufferFactor`
- `isMultibufferSlotValid`

然后在 `FinalizeExplicitSubviewMultibufferGroups()` 中按 root 做一次整体验证：

- 检查每个 group 的 leaf 几何是否自洽
- 检查不同 group 是否静态不重叠

验证失败就把整个 root 标记成 invalid，所有 leaf 清空 slot 元数据，后续统一回退。

### 5.3 `MemoryDependentAnalyzer`

在普通 alias/range 分析之前，额外消费 `(root, group, slot)` 语义：

- 同 root、不同 group：直接视为不相关
- 同 root、同 group、不同 slot：视为静态不重叠

其余情况仍然回到原有地址区间分析主逻辑。

也就是说，slot 语义不是替代 alias/range，而是给 alias/range 增加“这两个 leaf 明确属于同一组不同槽位”的额外证明。

### 5.4 `InsertSyncAnalysis`

对每一条候选同步边，分析逻辑先尝试提取共享 identity：

- `(multibufferRoot, multibufferGroup, multibufferFactor)`

如果依赖对里的 `BaseMemInfo` 不能共享同一个 identity，就直接按普通 autosync 处理：

- `eventIdNum = 1`
- 不启用 slot-aware event lane

如果 identity 成立，再继续判断是哪一种槽位使用形态：

- `SINGLE`
  - 只有一个 slot 真正参与依赖
  - 仍然是普通单 id
- `BRANCH`
  - 各 slot 在互斥分支中静态出现
  - 保持静态 lane 绑定
- `SELECTOR`
  - 存在 owner loop，且 loop 内能识别到 round-robin selector 族
  - `eventIdNum = factor`
  - 进入动态 event-id 选择路径

这里的核心目标是：

- event id 绑定到 slot，而不是绑定到“最近父循环”或者“偶然出现的地址”
- 同一物理 slot 在不同 iteration 中必须复用同一条 lane
- 不同 slot 之间不制造伪依赖

### 5.5 `SyncEventIdAllocation`

一旦某条同步边被判定为 `SELECTOR`，分配阶段就按该边的 `eventIdNum = factor` 分配一组稳定 lane：

- ping/pong 会拿到 2 条 lane
- triple-buffer 会拿到 3 条 lane
- 多 group 会各自独立拿自己的 lane bundle

对等价的 branch selector family，会复用同一组 lane，避免一组互斥分支重复占用 event id。

### 5.6 `SyncCodegen`

codegen 阶段只保留显式 subview 这条 multibuffer 路径：

- `SINGLE` / `BRANCH`
  - 发静态 `pto.set_flag` / `pto.wait_flag`
- `SELECTOR`
  - 在 owner loop 上构造 round-robin slot index
  - 按 slot index 从 event lane bundle 中选择当前 event id
  - 发 `pto.set_flag_dyn` / `pto.wait_flag_dyn`

当前 `GetBufferSelected()` 不再对“无显式 slot metadata 的多地址 buffer”做 selector 推断。

## 6. 一个典型 ping/pong 的中间状态

用户 IR：

```mlir
%workspace = pto.alloc_tile : !pto.tile_buf<vec, 32x16xf16>
%ping = pto.subview %workspace[%c0, %c0] sizes [16, 16]
  {pto.multi_buffer_factor = 2 : i32, pto.multi_buffer_slot = 0 : i32}
%pong = pto.subview %workspace[%c16, %c0] sizes [16, 16]
  {pto.multi_buffer_factor = 2 : i32, pto.multi_buffer_slot = 1 : i32}
```

translator 后的 `BaseMemInfo` 关键字段可理解为：

```text
ping: root=workspace, group=0, slot=0, factor=2, valid=true
pong: root=workspace, group=0, slot=1, factor=2, valid=true
```

如果 loop 内存在 `%iv % 2` 的 round-robin 选择，并且依赖边跨 iteration：

```text
shared identity = (workspace, group0, factor2)
slot mode = SELECTOR
eventIdNum = 2
event lanes = [eid0, eid1]
```

codegen 时：

- slot 0 永远选择 `eid0`
- slot 1 永远选择 `eid1`
- `%iv` 轮转时只是在两条稳定 lane 之间切换

这就是“event id 绑定 buffer 槽位”的具体落点。

## 7. group 的意义

`group` 是同步隔离域，不是装饰性属性。

当同一个 root workspace 里存在多套互不相关的 multibuffer 时：

- group 0 拥有自己的一组 slot
- group 1 拥有自己的一组 slot
- 两组之间不应该共享 lane，也不应该制造跨组依赖

因此实现里始终按 `(root, group, slot)` 管理，而不是只按 `(root, slot)` 管理。

## 8. 当前不支持的场景

以下场景当前都会保守回退到普通 autosync：

- root buffer 上标 intent，想让编译器自动展开 ping/pong
- 两块独立 `pto.alloc_tile` 想让编译器猜它们是一组 ping/pong
- 动态 offset / 动态 size 的 subview
- 非法或不完整 annotation
- 同一 root 下 group 几何重叠
- factor 超出当前支持范围

这不是能力缺失，而是当前 PR 的刻意边界：先把显式 subview multibuffer 的 correctness 做扎实。

## 9. 对未来的升级空间

这次的内部抽象已经是按 `(root, group, slot, factor)` 建模的，所以未来如果要支持更自动化的用法，可以在此之上增加前置 materialization pass：

1. 用户在更高层只声明 multibuffer intent。
2. 前置 pass 负责扩 workspace、生成 leaf subview、改写 selector/control-flow。
3. `PTOInsertSync` 继续消费统一的 slot 元数据。

也就是说，未来如果要做更自动化的 multibuffer 改造，应该新增前置规范化/materialization pass，而不是重新发明一套 autosync 语义。

但那不在本 PR 范围内。
