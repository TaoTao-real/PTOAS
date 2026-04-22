# PTOAS InsertSync Multibuffer 支持机制

本文围绕三个核心问题展开：multibuffer 如何表达、slot 如何识别、event id 如何按 slot 分配。

---

## 1. 问题的背景：ping/pong 优化带来的同步新需求

自动同步的基础问题是 producer 和 consumer 是否访问同一底层内存。
multibuffer 引入以后，同一个逻辑 tile buffer 会被拆成多个物理 slot，典型形态是 ping/pong：

- 当前迭代使用 `ping` 做搬运或计算。
- 下一迭代切换到 `pong`。
- 两个 slot 在物理地址上不重叠，但语义上属于同一个循环流水缓冲区。

这会带来两个容易混淆的问题：

- 如果只按同一个 root buffer 看待，会把 ping 和 pong 误认为互相依赖，插入过多同步。
- 如果只按地址不重叠看待，虽然能减少伪依赖，但仍然缺少“它们属于同一个 multibuffer group 的 slot 0/1”这一层语义，因此无法把 event id 稳定绑定到槽位，也无法覆盖未来由 pass 自动物化 slot 的场景。

因此，multibuffer 的自动同步不能只依赖原来的地址区间分析，还需要在 `BaseMemInfo` 上补充 slot 语义：它们是否来自同一逻辑 multibuffer root、属于哪个 slot、factor 是多少，以及该槽位选择归属于哪个 loop。

---

## 2. 目标和非目标

本设计的目标是支持 `SubView V1`，同时保持已有自动生成 multibuffer 路径不回归：

1. 当前 V1 正式支持的用户契约是：用户先声明一块已经扩好的 root workspace，并显式切出 ping/pong slot。
2. 用户可以在同一根 tile workspace 上标记 `pto.multi_buffer = 2`。
3. 用户可以通过 `pto.subview` 或 lowering 后的 `memref.subview` 切出 ping/pong，并显式写出稳定的 slot 选择逻辑，例如 parity 分支。
4. `PTOInsertSync` 能识别这两个 slice 是同一 multibuffer root 下的两个 slot。
5. event id 分配可以为 ping/pong 提供稳定的 slot-local event lane，并把具体 event id 绑定到槽位。
6. 当几何关系无法静态证明时，必须回退到普通 autosync，保证正确性优先。
7. 内部抽象要建立在 slot 语义上，而不是建立在 `pto.subview` 这一种语法上，为后续 annotation-driven 自动 multibuffer 留出演进空间。

本设计明确不解决以下问题：

- 不新增 CLI 参数、Python API 或新的用户侧构造。
- 不支持 `pto.multi_buffer > 2` 的新语义。
- 不支持动态 offset、动态 size 或动态 stride 的 slot 证明。
- 不通过启发式推断两块独立 `pto.alloc_tile` 是一组 ping/pong。
- 不在本 PR 中把单 buffer 自动扩成 ping/pong workspace，也不在本 PR 中自动改写计算通路的流水并行结构。

最后一点很重要：两块独立 alloc tile 在 IR 上没有“同一逻辑缓冲区的 slot 0/1”这个不变量。编译器可以保守地保证同步正确性，但不能无条件把它们当作 multibuffer slot 来分配 dynamic event id。

---

## 3. 当前 V1 的用户契约、已有路径与未来演进

当前 PTOAS 中有两条与 multibuffer 相关的路径，它们解决的问题不同。V1 要正式支持的是“用户显式声明 slot”的路径，同时保持已有 legacy 自动生成路径不回归。

### 3.1 手工定义：`pto.multi_buffer=2` + `pto.subview/memref.subview`

SubView V1 面向的是用户显式切分 workspace 的写法：

```mlir
%root = pto.alloc_tile ... {pto.multi_buffer = 2}
%ping = pto.subview %root[%c0, %c0] sizes [128, 256] : !pto.tile_buf<...> -> !pto.tile_buf<...>
%pong = pto.subview %root[%c128, %c0] sizes [128, 256] : !pto.tile_buf<...> -> !pto.tile_buf<...>
```

这里的用户契约是：

- `ping` 和 `pong` 必须来自同一个 root buffer。
- root buffer 必须带 `pto.multi_buffer = 2`。
- 两个 slice 必须静态可证明为非重叠、等大小、二等分。
- 只能沿一个维度切分，其他维度必须 full-span 或完全一致。
- loop 中必须存在稳定的 slot 选择逻辑，例如 `%iv % 2` 对应 ping/pong。
- 用户只负责表达 workspace、slot 和选择逻辑；自动同步和 event id 插入仍然由 PTOAS 完成。

`PTOViewToMemref` 会把 `pto.multi_buffer` 属性从 `pto.alloc_tile` 传递到 lowering 后的 `memref.alloc`，并把 `pto.subview` lowering 成 `memref.subview + pto.bind_tile`。
因此 `PTOInsertSync` 同时识别原始 `pto.subview` 和 lowering 后的 `memref.subview`。

### 3.2 自动生成：`PTOPlanMemory` + `pointer_cast(addrs=[...])`

已有路径面向的是 PlanMemory 自动生成 multibuffer 的场景：

1. 用户或上游 pass 在局部 buffer 上标记 `pto.multi_buffer = 2`。
2. `PTOPlanMemory` 把它规划成两个物理 tile buffer。
3. IR 中出现 `pto.pointer_cast(addrs=[ping, pong])` 形态。
4. `EnableMultiBuffer` 后续把它改写成 loop-local 地址选择。

这条路径已经存在，InsertSync 的 legacy 逻辑通过 `baseAddresses.size() == 2` 识别 double-buffer 地址组。
SubView V1 不替换这条主路径，只是在 slot 元数据存在时优先使用 slot-aware 语义；否则继续走 legacy 逻辑。

### 3.3 不自动推断：两块独立 `alloc_tile`

下面这种写法目前不被当作 slot-aware multibuffer：

```mlir
%ping = pto.alloc_tile ...
%pong = pto.alloc_tile ...

scf.for ... {
  scf.if %is_ping {
    pto.tload ... outs(%ping)
    pto.tstore ins(%ping) ...
  } else {
    pto.tload ... outs(%pong)
    pto.tstore ins(%pong) ...
  }
}
```

原因不是这段 IR 一定不能正确同步，而是它缺少可证明的 group 语义：

- `%ping` 和 `%pong` 是两个独立 root。
- 编译器不知道它们是否必须按同一个 loop selector 交替使用。
- 编译器不知道是否还有其他路径把其中一块当作普通临时 buffer 使用。

因此当前策略是 correctness-first：把它们作为普通 buffer 分析，不强行给它们绑定同一个 dynamic event selector。
如果后续要正式支持这种写法，应该引入显式 group/slot 契约，例如 `pto.multi_buffer_group`、`pto.multi_buffer_slot`、`pto.multi_buffer_factor`，而不是依靠名字或控制流形态做启发式推断。

### 3.4 未来演进：从手工 slot 走向 annotation-driven 自动 multibuffer

当前 V1 不是“把单 buffer 自动扩成 ping/pong”，而是“用户先把 ping/pong workspace 和 slot 显式写出来，autosync 再按 slot 分 event id”。

但这条路线并不妨碍未来升级。关键在于内部抽象必须建立在 slot 语义上，而不是建立在 `pto.subview` 这一种来源上：

1. 现在由 `pto.subview/memref.subview` 产生 slot 元数据。
2. 未来可以新增一个前置 multibuffer materialization pass，让用户只标 `pto.multi_buffer = 2` 或类似 annotation。
3. 该 pass 负责把单 buffer 扩成 root workspace、生成 slot view、补齐 selector，必要时再配合 loop/pipeline 改写。
4. `PTOInsertSync`、`SyncEventIdAllocation` 和 `SyncCodegen` 继续消费统一的 slot 元数据，不需要重新设计同步核心。

换句话说，手工 subview 只是当前 V1 的第一种 slot 来源，不应被当作 multibuffer 模型本身。

---

## 4. 流程并不复杂，但 slot 信息必须提前进入 `BaseMemInfo`

与 multibuffer 相关的关键阶段如下：

`PTOViewToMemref -> PTOPlanMemory -> PTOInsertSync -> EnableMultiBuffer -> PTOToEmitC`

其中 `PTOInsertSync` 内部仍然沿用原有流水线：

`PTOIRTranslator -> InsertSyncAnalysis -> MoveSyncState -> RemoveRedundantSync -> SyncEventIdAllocation -> SyncCodegen`

当前 V1 中，slot 的直接来源是 `pto.subview/memref.subview`；未来如果新增自动 multibuffer materialization pass，它也应该产出同样的 slot 元数据。

新增 slot-aware 语义主要落在三处：

1. `PTOIRTranslator`：从 `pto.subview/memref.subview` 计算 slot 元数据。
2. `InsertSyncAnalysis`：基于 slot 元数据决定一条 sync edge 需要几个 event id lane。
3. `SyncCodegen`：当 sync edge 有多个 event id 时，生成 dynamic event-id set/wait。

整体关系可以概括为：

```mermaid
flowchart LR
  A["手工 root workspace + subview"] --> C["slot producer"]
  B["未来 auto multibuffer materialization pass"] --> C
  C --> D["PTOIRTranslator"]
  D --> E["BaseMemInfo slot metadata"]
  E --> F["InsertSyncAnalysis"]
  F --> G{"slot-aware?"}
  G -->|"yes"| H["eventIds = lanes by (edge, root, slot, ownerLoop)"]
  G -->|"no"| I["普通 autosync / legacy multibuffer"]
  H --> J["SyncCodegen static/dynamic event id"]
  I --> J
```

---

## 5. 依赖识别：alias/range 仍是主体，slot 是额外语义

普通自动同步依赖 `BaseMemInfo` 上的四类基础信息：

- `rootBuffer`：底层分配源。
- `scope`：GM、L1、UB 等地址空间。
- `baseAddresses`：静态可知的地址或 offset。
- `allocateSize`：当前 view 的静态大小。

SubView V1 在此基础上增加以下字段：

- `multibufferRoot`：slot 所属的逻辑 multibuffer root。
- `multibufferSlot`：当前 view 属于哪个 slot，ping/pong 分别是 0/1。
- `multibufferFactor`：当前只支持 2。
- `isMultibufferSlotValid`：slot 元数据是否可用。
- `suppressLegacyMultibuffer`：该 root 已被证明不是合法 subview multibuffer 时，禁止半路回退到 legacy double-buffer。

注意：slot 元数据不替代 alias/range 分析。
RAW、WAR、WAW 依赖是否存在，仍然由原来的 def/use 和地址区间规则判定。slot 元数据只参与 event-id lane 数量和 loop/back-edge 相关的动态选择。

这些字段的设计应保持 source-agnostic：当前可以由 subview 填充，未来也可以由自动 multibuffer 物化 pass、显式 group/slot annotation 或其他 view-like IR 产生。

---

## 6. slot 识别：只接受静态可证明的 ping/pong 几何

当前 V1 中，`PTOIRTranslator::TryComputeSubviewSlotInfo` 是第一种 slot producer。它只在以下条件全部满足时标记 slot：

1. 定义 op 是 `pto.subview` 或 `memref.subview`。
2. root buffer 带 `pto.multi_buffer = 2`，或者来自已有 legacy double-buffer root。
3. offset、size 都是静态整数。
4. `memref.subview` 的 stride 必须全为 1。
5. 所有维度都在 root shape 范围内。
6. 只有一个维度被切分，其他维度必须保持 full-span。
7. 被切分维度必须能被 2 等分。
8. 当前 slice size 必须正好等于该维度的一半。
9. offset 必须落在 slot 边界上，因此 slot 只能是 0 或 1。
10. root 的总静态字节数必须等于当前 slice 字节数的 2 倍。

满足这些条件后，translator 会写入：

```text
multibufferRoot   = root
multibufferSlot   = 0 or 1
multibufferFactor = 2
isMultibufferSlotValid = true
```

### 6.1 为什么失败时要按 root 失效

一个 root 下如果出现了非法 subview，例如两个 slice 重叠、不是二等分、或只识别到了其中一半，不能只让失败的那条 view 回退。
否则会出现一部分 use/def 带 slot 元数据，另一部分 use/def 没有 slot 元数据，event-id 分配可能半路进入 dynamic path。

因此当前策略是：

1. 如果 root-level subview multibuffer candidate 识别失败，标记整个 root invalid。
2. 清空该 root 下已记录的 slot 元数据。
3. 设置 `suppressLegacyMultibuffer = true`，避免 overlap 等负向用例又被 legacy `baseAddresses.size()==2` 当成合法 double-buffer。
4. 后续全部按普通 alias/range autosync 处理。

这条规则的目的不是优化，而是防止“半合法 multibuffer”生成错误 dynamic event id。

---

## 7. event id 分配：从地址组升级到 slot-aware edge

普通同步只需要一个 event id：

```text
set(MTE2 -> V, event_id = k)
wait(MTE2 -> V, event_id = k)
```

multibuffer ping/pong 需要两个 event lane：

```text
even iteration -> lane0
odd  iteration -> lane1
```

这里的 event lane 不是硬件 `EVENT_IDk` 本身，而是分析阶段的逻辑槽位通道。更准确地说，一条 multibuffer lane 应绑定到：

```text
(sync edge, multibuffer root/group, slot index, ownerLoop)
```

`SyncEventIdAllocation` 再把这些逻辑 lane 映射成具体的硬件 event id。

`InsertSyncAnalysis::GetEventIdNum` 的判断顺序应当是：

1. 先证明一条 sync edge 是否真的拥有多个稳定 slot。
2. 如果能证明所有依赖对都来自同一个 root、同一个 factor，且 slot 几何一致，则返回 `factor`，当前即 2。
3. 如果不能证明，就返回 1，按普通同步处理。
4. 只有在 slot 元数据不存在时，才尝试已有 legacy double-buffer 判断。

这条顺序的核心不是“尽量多给两个 id”，而是“event id 必须先绑定到已证明的槽位，再决定是否扩张到多个 id”。

### 7.1 slot-aware 判断到底证明什么

对于一组依赖 pair，`IsSlotAwareMultibufferPair` 要证明：

- 两侧都不是 GM buffer。
- 两侧都带合法 slot 元数据。
- `multibufferRoot` 相同。
- `multibufferFactor` 相同。
- `multibufferSlot` 在 `[0, factor)` 范围内。
- 同 slot 的两个 view 必须地址重叠。
- 不同 slot 的两个 view 必须地址不重叠。

这组规则表达的是：同一个 slot 内的 producer/consumer 仍然需要同步，不同 slot 之间不制造伪依赖。

### 7.2 slot 证明结果如何进入 codegen

一条 sync edge 被证明为 multibuffer 后，后续 codegen 还需要知道两件事：

1. 槽位选择是静态分支还是 loop parity。
2. 这个槽位选择真正归属于哪个 loop，而不是当前 op 最近的父循环。

因此 analysis 阶段应进一步沉淀最小元数据，例如：

- `slotMode`：`single / branch / parity`
- `slotCount`
- `ownerLoopBegin/End`

这些元数据的作用是让 `SyncCodegen` 消费已经证明过的槽位语义，而不是在 codegen 阶段重新猜。

### 7.3 static / dynamic event-id 如何落代码

分析阶段先决定一条 sync edge 究竟属于 single、branch 还是 parity：

1. `single` 路径只有一个 proven slot，直接生成静态 `set/wait`。
2. `branch` 路径虽然整体上属于 ping/pong，但具体到 branch-local edge 时每条 edge 只绑定一个已证明 slot，因此也生成静态 `set/wait`。
3. `parity` 路径只有在一条 edge 真实跨越多个 slot，且 slot 选择明确归属于某个 `ownerLoop` 时，才生成 dynamic event-id。
4. `parity` 的偶数迭代选择 `eventIds[0]`，奇数迭代选择 `eventIds[1]`。
5. 只有 `parity` 走 `pto.set_flag_dyn` 或 `pto.wait_flag_dyn`。

伪代码如下：

```text
cond = (loop_parity == odd)
event = select cond, eventIds[1], eventIds[0]
set_flag_dyn(srcPipe, dstPipe, event)
wait_flag_dyn(srcPipe, dstPipe, event)
```

因此 ping 与 pong 拥有稳定的 slot-local event lane。对显式 `if/else` 写出的 ping/pong，编译器会把 event id 静态绑定到对应 branch-local slot；对真正的 parity edge，loop 迭代交替时再复用对应 slot 的 dynamic event id。

这条规则的关键不是“只要在 loop 里就用 `%iv % 2`”，而是“只有当 slot 选择已被证明属于该 loop 时，才能把动态 event 绑定到该 loop 上”。否则必须保守退回单 id。

---

## 8. 两条路径如何共存

SubView V1 和 legacy 自动生成 multibuffer 的优先级如下：

1. 如果 `BaseMemInfo` 已经带合法 `isMultibufferSlotValid`，优先走 slot-aware path。
2. 如果 slot-aware 不成立，且没有 `suppressLegacyMultibuffer`，再尝试 legacy `baseAddresses.size()==2` path。
3. 如果 root 被标记为 invalid subview multibuffer，直接禁用 legacy fallback。
4. 如果以上都不成立，回到单 event id 的普通 autosync。

这个顺序保证了三件事：

- 手工 subview pingpong 能得到按槽位绑定的 static/dynamic event-id 语义。
- 旧的 `pointer_cast(addrs=[...])` 主路径不回归。
- 非法 subview 不会因为地址数量看起来像 double-buffer 而误进 multibuffer codegen。

---

## 9. 例子：合法 subview、非法 overlap、缺少属性

### 9.1 合法 subview ping/pong

```text
root shape = [256, 256]
ping offset = [0,   0], size = [128, 256]
pong offset = [128, 0], size = [128, 256]
```

识别结果：

- `ping.multibufferSlot = 0`
- `pong.multibufferSlot = 1`
- `multibufferFactor = 2`
- 如果 edge 是 branch-local 单槽位访问，则静态绑定到对应 slot 的 event id。
- 如果未来出现真实的 parity edge，再按 `ownerLoop` 生成两个 dynamic event lane。

### 9.2 非法 overlap

```text
root shape = [256, 256]
ping offset = [0,  0], size = [160, 256]
pong offset = [96, 0], size = [160, 256]
```

两个 slice 重叠，且不是二等分。
识别结果：

- 整个 root 被标记为 invalid subview multibuffer。
- 清空 slot 元数据。
- 禁用 legacy multibuffer fallback。
- 按普通 autosync 保守插同步。

### 9.3 合法切片但缺少 `pto.multi_buffer`

```text
root shape = [256, 256]
ping offset = [0,   0], size = [128, 256]
pong offset = [128, 0], size = [128, 256]
root has no pto.multi_buffer attribute
```

几何关系看起来像 ping/pong，但用户没有声明 multibuffer intent。
识别结果：

- 不标记 slot。
- 不进入 slot-aware event 分配。
- 按普通 autosync 处理。

这能避免普通 workspace 恰好被二等分时被误识别成 multibuffer。

---

## 10. 正确性原则

可以把当前实现归纳成三条不变量：

1. 没有显式 multibuffer intent，就不启用 slot-aware 语义。
2. 没有静态可证明的二等分非重叠几何，就不启用 slot-aware 语义。
3. 一旦某个 root 的 subview multibuffer 识别失败，该 root 下所有 view 都回退到普通 autosync。

这些不变量会牺牲一部分优化机会，但能保证不会因为误识别 multibuffer 而漏同步或错误复用 event id。

---

## 11. 测试与看护

本设计对应的看护用例覆盖三类场景：

- `test_inject_sync_multibuf_subset_pingpong.py`：合法 `pto.subview` + `pto.multi_buffer=2`，期望生成按槽位静态绑定的 event-id 形态，且不再产生 orphan dynamic lane，并防止回退到 tile pointer-cast 风格 lowering。
- `test_inject_sync_multibuf_subset_overlap.py`：带 `pto.multi_buffer=2` 但 subview 重叠或不等分，期望只走普通 autosync，不进入 slot-aware multibuffer。
- `test_inject_sync_multibuf_subset_no_attr.py`：ping/pong 几何合法但缺少 `pto.multi_buffer`，期望仍走普通 autosync。

A3 上板最小用例为 `multibuffer_subset_pingpong_a3.py`。
它使用单输入单输出和确定性 golden/compare，用于验证 subview-based ping/pong lowering、自动同步插入、按槽位静态 event-id 绑定和远端 NPU 执行链路。

---

## 12. 演进路径

后续演进可以分为四步：

1. 扩展到 `pto.multi_buffer > 2`，把 factor 从 ping/pong 推广到 N-buffer。
2. 支持显式 group/slot 属性，让两块独立 `alloc_tile` 也能安全表达为同一 multibuffer group。
3. 新增 annotation-driven multibuffer materialization pass，让用户只标 multibuffer intent，由 pass 自动扩 root workspace、生成 slot view 和 selector，再复用同一套 slot-aware autosync。
4. 在保持 correctness-first 的前提下，提高动态 shape/offset 场景下的 slot 证明能力。

其中第二、三条是从“用户手工声明 ping/pong workspace”升级到“编译器自动把单 buffer 改造成 ping/pong”的关键。
在没有显式 group/slot 契约或没有前置 materialization pass 前，编译器不应该仅凭变量名、if 分支或两块 buffer 大小相同就推断 multibuffer，因为这会把普通双缓冲临时变量误绑定到同一个 dynamic event selector 上。

---

## 13. 结论

SubView V1 的核心是把 multibuffer 从“两个地址”提升为“同一 root 下的多个 slot”。
当前正式支持的用户契约不是“编译器自动把单 buffer 扩成 ping/pong”，而是“用户先把 ping/pong workspace 和 slot 显式写出来，autosync 再把 event id 绑定到槽位”。

依赖识别仍然保持原来的 alias/range correctness-first 策略，slot 元数据只在 event-id lane 统计以及必要时的 dynamic event-id codegen 中生效。
这样既能覆盖用户手工 `pto.subview/memref.subview` 定义 ping/pong 的场景，又不会破坏已有 `PTOPlanMemory + pointer_cast(addrs=[...])` 自动生成 multibuffer 路径。更重要的是，只要内部保持 slot-aware 抽象，未来就可以在 autosync 前面增加 annotation-driven 自动 multibuffer materialization pass，而不需要重写同步核心。
