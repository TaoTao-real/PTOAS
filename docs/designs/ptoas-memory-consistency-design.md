# PTOAS 内存一致性设计

本文说明 PTOAS 如何建模并校验 GM payload 与 signal 之间的内存一致性要求。

这里讨论的是内存一致性，不是自动同步。自动同步负责 pipe 之间的执行顺序，例如
`set_flag`、`wait_flag` 和 `pipe_barrier`。内存一致性负责回答另一个问题：当 signal
已经被对端观察到时，signal 之前发布的 payload 是否已经对正确的观察方可见。

## 1. 背景

`pto.comm.tnotify` 用来发布一个 signal。对端通过 `pto.comm.twait` 或
`pto.comm.ttest` 观察这个 signal，然后读取对应的 payload。

signal ready 不等价于 payload 一定已经可见。原因是 signal 和 payload 可能走不同的硬件路径：

- signal 通常是一个较小的通信同步标记。
- payload 通常是更大的 GM 数据，可能由 MTE3、FIX 或 comm macro op 写出。
- 不同路径之间只靠源码顺序不一定形成完整的 GM 可见性关系。

因此，PTOAS 需要在发布 signal 前校验 release 侧动作，在消费 signal 后校验 acquire 侧动作。

## 2. 当前暴露的 IR

当前 PR 只暴露两条内存一致性 IR：

| PTO IR | 语义 | EmitC lowering |
| --- | --- | --- |
| `pto.fence.barrier_all #pto.fence_scope<gm>` | GM visibility barrier，可用于 publish 或 acquire 边界 | `dsb(DSB_DDR)` |
| `pto.cmo.cacheinvalid all #pto.address_space<gm>` | whole-cache GM cache maintenance 边界，可用于 release 或 acquire | `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)` |
| `pto.cmo.cacheinvalid %addr single_cache_line : !pto.ptr<T, gm>` 或 `!pto.partition_tensor_view<...>` | 指定 GM payload 地址。cacheable 路径上是真实 cache maintenance 边界；non-cacheable 路径上可以作为 marker-only IR 驱动精准 pipe drain | cacheable 路径 lower 到 `dcci(addr, cache_line_t::SINGLE_CACHE_LINE)`；marker-only 路径由 pass 消费并消除 |

对应 PyPTO 写法：

```python
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM, addr=payload_ptr)
```

`FenceScopeAttr` 当前定义三个语义 scope：

- `local_memory`
- `gm`
- `all`

本 PR 只为 `gm` 和 `all` 提供 EmitC lowering；它们当前都 lower 成 `dsb(DSB_DDR)`。
`local_memory` 预留给后续 UB/local-memory 语义。

`cmo.cacheinvalid` 支持两种粒度：

```mlir
pto.cmo.cacheinvalid all #pto.address_space<gm>
pto.cmo.cacheinvalid %payload_ptr single_cache_line : !pto.ptr<i32, gm>
```

`all` 表示 whole-cache 粒度，并且一定会 lower 成真实 `dcci`。在 release 侧，
它也是一个保守 payload marker：PTOAS 会认为用户希望发布到这条 CMO 为止已经 pending 的
所有 GM payload 访问，并为这些访问补齐必要的 pipe drain。`single_cache_line` 表示
`%payload_ptr` 所在 cache line，同时也是 PTOAS 用来精准识别 TNotify payload 的地址
marker。当前还没有 range 形式；如果 payload 横跨多条 cache line，PyPTO 或用户可以
生成多条 single-line CMO，或者使用 whole-cache 形式做保守处理。

对 non-cacheable 的 MTE2、MTE3 或 FIX payload path，`single_cache_line` 主要用作精准地址
marker。MemoryConsistency pass 会用它和前序 pending payload access 做 alias 匹配，决定
是否需要插入 `PIPE_MTE2`、`PIPE_MTE3` 或 `PIPE_FIX` drain。若匹配到的路径不经过 scalar
cache，EmitC lowering 会直接消除这条 `cmo.cacheinvalid`，不生成 `dcci`。`all` 形式不做
地址匹配，而是保守选择该 op 之前已经 pending 的所有 GM payload access，并保留真实
whole-cache `dcci`。

## 3. DCCI 语义边界

`dcci` 是 cache maintenance 指令，不是 pipe drain，也不是 GM visibility fence。

PTOAS 当前对齐 PTO-ISA 的写法，只生成 two-argument CCE builtin：

```cpp
dcci(addr, cache_line);
```

其中 `cache_line_t::SINGLE_CACHE_LINE` 表示处理传入地址所在 cache line，
`cache_line_t::ENTIRE_DATA_CACHE` 表示处理整个 Data Cache。

公开 AscendC API 和部分底层 CCE 头文件还存在带 destination 参数的形式，例如
`CACHELINE_OUT`。但是 PTO-ISA 主线的 `TNotify`、`TWait`、`TTest` 和 ready-queue
实现都使用 two-argument `dcci`。因此本 PR 不把三参数 `dcci(..., CACHELINE_OUT)`
作为 PTOAS 的生成代码契约，也不在 PTO IR 中暴露 destination 选择。

当前 public IR 使用 two-argument CCE builtin。whole-cache 形式：

```cpp
dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
```

single-cache-line 形式：

```cpp
dcci((__gm__ void*)addr, cache_line_t::SINGLE_CACHE_LINE);
```

它们都是 PTOAS 暴露给上游的显式 CMO 边界。虽然 op 名称沿用
`cacheinvalid`，但当前 PTO-ISA 对齐的 two-argument `dcci` 同时服务两类上下文：

- release 侧：放在 cacheable scalar GM store 之后、`fence.barrier_all` 之前，作为发布
  payload 前的 cache maintenance。
- acquire 侧：放在 `TWait` 或 `TTest` 之后、cacheable GM load 之前，避免读取本地 stale
  cache line。

本 PR 不新增 `pto.cmo.clean`。原因是当前 release 侧和 acquire 侧都对齐到 PTO-ISA
two-argument `dcci`，新增一个 lowering 相同的 public op 会让 PyPTO 和用户难以区分。
后续如果确认需要暴露不同 destination 或精确 writeback 语义，再单独扩展 CMO IR。

MemoryConsistency pass 在当前阶段不会证明 single-cache-line CMO 的地址是否覆盖所有
pending payload。也就是说，`single_cache_line` 是一个精确 CMO 操作，但其正确使用需要
PyPTO 或用户保证 `%addr` 确实落在需要发布或消费的 payload cache line 上。

## 4. MemoryConsistency Pass

`pto-memory-consistency` 是一个 Module pass，运行在 shared mainline 上，因此 EmitC 和
VPTO backend 都会先经过这一步。

这个 pass 的职责是：

- 识别 signal publish 前是否存在 `cmo.cacheinvalid %addr` 精准 marker，或
  `cmo.cacheinvalid all #pto.address_space<gm>` 保守全量 marker。
- 识别 signal acquire 后是否存在 cacheable GM payload read。
- 校验用户或 PyPTO 是否已经插入必要的 `fence.barrier_all` 或 `cmo.cacheinvalid`。
- 在显式 `barrier_all` 前自动补齐 marker 对应的 MTE3 或 FIX pipe drain。
- 在 `TNotify` 前自动补齐 marker 对应的 MTE2 pipe drain。
- 对缺失或顺序错误的场景报编译错误。

这个 pass 不负责分配 event id，也不属于 InsertSync 自动同步流水线。

## 5. 场景规则

### 5.1 MTE3、FIX 或 TPUT 写 Payload 后发布 Signal

适用场景：

- `TStore` 通过 `PIPE_MTE3` 写 GM。
- `TStore` 通过 `PIPE_FIX` 写 GM，例如 ACC tile 写回 GM。
- `TStoreFP` 通过 `PIPE_FIX` 写 GM。
- `TPUT` macro op 内部通过 MTE3 写 peer GM。
- 其他 macro op phase 中存在 MTE3 GM write。

PyPTO 或用户需要生成：

```mlir
// payload producer
pto.cmo.cacheinvalid %payload single_cache_line : !pto.partition_tensor_view<...>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

PTOAS 会用 `cmo.cacheinvalid %payload` 的地址和前序 pending payload access 做 alias 匹配。
也可以使用 `pto.cmo.cacheinvalid all #pto.address_space<gm>`，表示不做精确地址指定，
保守发布到该 op 为止已经 pending 的全部 GM payload access。如果 marker 选择到 pending MTE3 或
FIX GM write，就在
`pto.fence.barrier_all #pto.fence_scope<gm>` 前自动插入对应 pipe 的 drain：

```mlir
pto.barrier #pto.pipe<PIPE_MTE3>
// or
pto.barrier #pto.pipe<PIPE_FIX>
```

最终 EmitC 关键顺序是：

```cpp
pipe_barrier(PIPE_MTE3);
dsb(DSB_DDR);
pto::comm::TNOTIFY(...);
```

`pipe_barrier` 用来排空实际执行 GM write 的 pipe。`barrier_all` lower 出来的
`dsb(DSB_DDR)` 用来保证这些 GM 写入在 signal 发布前进入 GM visibility domain。

如果 matching payload 缺少 `pto.fence.barrier_all #pto.fence_scope<gm>`，PTOAS 会报错。
PTOAS 可以推导低层 pipe drain，但 payload marker 和 visibility fence 仍由 PyPTO 或用户
显式表达。

### 5.2 MTE2 工作后发布 Signal

`TLoad` 或其他 `PIPE_MTE2` 工作出现在 `TNotify` 前时，PyPTO 应生成 matching payload
marker：

```mlir
pto.tload ...
pto.cmo.cacheinvalid %payload single_cache_line : !pto.partition_tensor_view<...>
pto.comm.tnotify ...
```

PTOAS 会用 marker 地址匹配前序 MTE2 GM read，并在 EmitC lowering 中生成
`PIPE_MTE2` barrier。MTE2 是 GM read 方向，只需要 signal 前不要越过前序 MTE2 工作，
不需要 `barrier_all`。这条 marker 对 MTE2 non-cacheable path 不生成 `dcci`。

### 5.3 TWait 或 TTest 后读取 Cacheable GM Payload

适用场景：

- `TWait` 返回后执行 `load_scalar` 读取 GM payload。
- `TTest` 成功观察到 signal 后执行 `load_scalar` 读取 GM payload。

需要的顺序：

```mlir
pto.comm.twait ...
pto.cmo.cacheinvalid %payload_ptr single_cache_line : !pto.ptr<i32, gm>
%value = pto.load_scalar ...
```

`cacheinvalid` 用来避免读取到本地 stale cache line。也可以使用 whole-cache 形式：
`pto.cmo.cacheinvalid all #pto.address_space<gm>`。如果缺少该 op，PTOAS 会报错。

Acquire 侧的 `single_cache_line` 形式是用户或 PyPTO 对 cache line 覆盖关系的显式承诺：
`%payload_ptr` 必须覆盖后续 `load_scalar` 实际读取的 GM 地址。PTOAS 当前只检查
`TWait` 或 ready `TTest` 后、cacheable GM `load_scalar` 前存在 `cmo.cacheinvalid`，
不会证明该 CMO 地址和后续 load 地址是否 alias。如果生成方无法保证精确地址正确，
应使用 `pto.cmo.cacheinvalid all #pto.address_space<gm>` 做保守 whole-cache invalidate。

### 5.4 Cacheable Scalar GM Store 后发布 Signal

适用场景：

```mlir
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32>, i32
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

最终 EmitC 关键顺序是：

```cpp
payload[idx] = value;
dcci((__gm__ void*)payload, cache_line_t::SINGLE_CACHE_LINE);
dsb(DSB_DDR);
pto::comm::TNOTIFY(...);
```

这里 `cmo.cacheinvalid` 表示 publish 前的显式 CMO 边界，`fence.barrier_all` 表示
GM visibility fence。二者顺序不能交换。如果 payload 覆盖多条 cache line，可以改用
whole-cache 形式，或者生成多条 single-line CMO。如果缺少 `cmo.cacheinvalid`，或者把它
放在 `fence.barrier_all` 后面，PTOAS 会报错。

## 6. PyPTO 对接说明

PyPTO 不需要手动生成 `pto.barrier #pto.pipe<PIPE_MTE3>` 或
`pto.barrier #pto.pipe<PIPE_FIX>`。这是低层 pipe drain 细节，由 PTOAS 根据
`pto.cmo.cacheinvalid %payload single_cache_line` marker 匹配到的 GM payload access 自动插入。
这样可以保证最终顺序是对应 pipe barrier 先于 `dsb(DSB_DDR)`，不会出现先 fence、后 drain
的错误顺序。没有 marker 的 memory access 不会被 PTOAS 当作 TNotify payload；也就是说，
marker 是 signal 和 payload 的显式关联。`single_cache_line` 只关联匹配地址的 payload；
`all` 是显式兜底 marker，会关联该 op 之前已经 pending 的所有 GM payload access。

PyPTO 生成规则：

| 场景 | PyPTO 需要生成 | PTOAS 自动补齐 |
| --- | --- | --- |
| `TStore`、`TStoreFP` 或 `TPUT` 后发布 signal | `pto.cmo.cacheinvalid %payload single_cache_line` 作为精准 payload marker，或 `pto.cmo.cacheinvalid all #pto.address_space<gm>` 作为保守全量 marker，随后 `pto.fence.barrier_all #pto.fence_scope<gm>` | 若 marker 选择到 pending GM write，则补 `PIPE_MTE3` 或 `PIPE_FIX` drain；non-cacheable single-line marker 不生成 `dcci`，whole-cache marker 生成 whole-cache `dcci` |
| `TLoad` 后发布 signal | `pto.cmo.cacheinvalid %payload single_cache_line` 作为精准 payload marker，或 `pto.cmo.cacheinvalid all #pto.address_space<gm>` 作为保守全量 marker；不需要显式 fence | 若 marker 选择到 pending GM read，则补 `PIPE_MTE2` drain；non-cacheable single-line marker 不生成 `dcci`，whole-cache marker 生成 whole-cache `dcci` |
| `TWait` 后读取 cacheable scalar GM payload | payload load 前生成 `pto.cmo.cacheinvalid %addr single_cache_line : !pto.ptr<T, gm>`，或使用 whole-cache 形式；single-line 地址必须由 PyPTO 或用户保证覆盖后续 load | PTOAS 检查 CMO 存在和顺序，不校验 acquire CMO 与 load 的 alias 关系 |
| `TTest` ready path 后读取 cacheable scalar GM payload | 在 ready path 的 payload load 前生成 single-line 或 whole-cache `pto.cmo.cacheinvalid`；single-line 地址必须由 PyPTO 或用户保证覆盖后续 load | PTOAS 检查 CMO 存在和顺序，不校验 acquire CMO 与 load 的 alias 关系 |
| cacheable scalar GM store 后发布 signal | 在 payload store 后生成 `pto.cmo.cacheinvalid %payload single_cache_line`，或使用 `pto.cmo.cacheinvalid all #pto.address_space<gm>`；随后生成 `pto.fence.barrier_all #pto.fence_scope<gm>` | single-line 只覆盖指定 cache line；whole-cache 覆盖全部 scalar D-cache，并保守选择全部 pending GM payload |

如果某个 TNotify 只是发布 signal，而不表示某块 GM payload 已经准备好，可以不生成 payload
marker。PTOAS 不会尝试从所有前序 memory op 里推断“可能相关”的 payload。

`pto.entry` launcher 可以调用多个 kernel 函数；每个 kernel 函数会被
`pto-memory-consistency` 独立分析。kernel body 内部可以通过 `func.call` 调用只包含
payload 访问的 helper，PTOAS 会把 callee 中的 payload producer 摘要回 caller，并继续由
caller 侧的 `cmo.cacheinvalid` marker、`fence.barrier_all` 和 signal op 决定 release
边界。如果 helper 内部已经插入 `pto.barrier`，callee 摘要也会消费对应 pipe drain，避免
caller 侧重复补同一个 pipe barrier。

但 CMO、fence、`TNotify`、`TWait` 和 `TTest` 这类 memory-consistency boundary op 不能藏在
non-inline helper 中。PyPTO 应在 `pto-memory-consistency` 前将包含 boundary op 的 helper
inline，或者把这些 boundary op 保持在 caller 中；否则 pass 会报错，避免 caller 侧看不到
callee 内部的 release 或 acquire 边界。

### 6.1 Issue #872：TPUT 发布 Signal

```mlir
pto.comm.tput ...
pto.cmo.cacheinvalid %peer_payload single_cache_line : !pto.partition_tensor_view<...>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

对应 PyPTO 写法：

```python
pto.TPutOp(...)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM, addr=peer_payload)
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.TNotifyOp(...)
```

这个形态对应 #872 中的 `TPUT -> TNotify` 问题。`TPUT` macro op 内部会通过 MTE3 写
peer GM payload；如果直接发布 `TNotify`，receiver 可能先观察到 signal ready，但 payload
写入还没有完成 pipe drain 或进入 GM visibility domain。

PTOAS 会识别 `TPUT` macro model 中的 MTE3 GM write phase，并在 `barrier_all` 前自动补齐
低层 pipe drain：

```mlir
pto.comm.tput ...
pto.cmo.cacheinvalid %peer_payload single_cache_line : !pto.partition_tensor_view<...>
pto.barrier #pto.pipe<PIPE_MTE3>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

EmitC 最终生成的关键顺序是：

```cpp
TPUT(...);
pipe_barrier(PIPE_MTE3);
dsb(DSB_DDR);
TNOTIFY(...);
```

这里的 `cmo.cacheinvalid %peer_payload` 对 `TPUT` 是 marker-only：它告诉 PTOAS 本次
`TNotify` 发布的是哪一块 peer payload。因为 `TPUT` 内部 MTE3 store 是 non-cacheable
路径，最终不会生成 `dcci`，只会让 PTOAS 在 `fence.barrier_all` 前精准补上匹配 payload
的 MTE3 drain。

### 6.2 TWait 后读取 Scalar Payload

```mlir
pto.comm.twait ...
pto.cmo.cacheinvalid %payload_ptr single_cache_line : !pto.ptr<i32, gm>
%value = pto.load_scalar ...
```

对应 PyPTO 写法：

```python
pto.TWaitOp(...)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM, addr=payload_ptr)
pto.LoadScalarOp(...)
```

### 6.3 Scalar Store 发布 Signal

```mlir
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32>, i32
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

对应 PyPTO 写法：

```python
pto.StoreScalarOp(...)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM, addr=payload_ptr)
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.TNotifyOp(...)
```

这对应 cacheable scalar GM store 发布 payload 的场景。`CmoCacheInvalidOp` 和
`FenceBarrierAllOp` 都是必需的，且顺序必须是 CMO 在 fence 之前。release 侧的
`pto.cmo.cacheinvalid all #pto.address_space<gm>` 同时表示 whole-cache CMO 边界和保守全量
payload marker。使用 whole-cache 形式时，不需要再额外生成 matching
`pto.cmo.cacheinvalid %payload single_cache_line`；代价是 PTOAS 会保守选择该 op 之前已经
pending 的所有 GM payload access，并为这些访问补齐对应 pipe drain。

## 7. Backend Lowering 状态

### 7.1 EmitC

EmitC backend 当前支持：

- `pto.cmo.cacheinvalid all #pto.address_space<gm>` lower 到 `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)`。
- `pto.cmo.cacheinvalid %addr single_cache_line` 在 cacheable scalar release 或 acquire 路径 lower 到 `dcci((__gm__ void*)addr, cache_line_t::SINGLE_CACHE_LINE)`。
- `pto.cmo.cacheinvalid %addr single_cache_line` 在 non-cacheable TLoad、TStore、TStoreFP 或 comm macro release 路径只作为 payload marker，被 MemoryConsistency pass 消费后消除。
- `pto.cmo.cacheinvalid all #pto.address_space<gm>` 在 release 路径作为保守全量 marker，选择该 op 之前已经 pending 的所有 GM payload access；它不会被消除。
- `pto.fence.barrier_all #pto.fence_scope<gm>` lower 到 `dsb(DSB_DDR)`。

### 7.2 VPTO

VPTO backend 当前已经支持低层 `pto.dsb` 和 `pto.dcci` 到 HIVM intrinsic 的 lowering。

本 PR 暴露的 `pto.cmo.cacheinvalid` 和 `pto.fence.barrier_all` 是较高层的
signal/payload 一致性契约 IR。当前还没有在 VPTO pipeline 中把这两类高层 op 自动降成
低层 `pto.dcci` 和 `pto.dsb`，因此它们进入 VPTO LLVM lowering 时仍然 fail-fast：

- `pto.cmo.cacheinvalid`
- `pto.fence.barrier_all`

如果这些 op 进入 VPTO LLVM lowering，PTOAS 会报错，提示 VPTO backend 尚不支持这些
high-level memory-consistency op。后续需要补一层 VPTO memory-consistency lowering，
把明确的 CMO 和 fence 语义转换为 `pto.dcci` 与 `pto.dsb`。

## 8. 当前限制

- `cmo.cacheinvalid` 支持 whole-cache 和 single-cache-line 粒度，但还没有连续 range 形式。
- MemoryConsistency pass 当前不证明 acquire 侧 single-line CMO 是否覆盖后续 `load_scalar` payload，地址正确性由 PyPTO 或用户保证；不确定时应使用 whole-cache `cmo.cacheinvalid all #pto.address_space<gm>`。
- `TWait` 和 `TTest` acquire 侧当前只覆盖 `load_scalar`。
- VPTO 暂不支持 high-level `cmo.cacheinvalid` 和 `fence.barrier_all` 的真实 lowering；
  低层 `pto.dcci` 和 `pto.dsb` 已有 VPTO lowering。
- 对复杂 CFG 的分析仍是保守近似，不做完整 path-sensitive 数据流。

## 9. 后续工作

1. 在 VPTO pipeline 中把 high-level `cmo.cacheinvalid` 与 `fence.barrier_all` 降到
   low-level `pto.dcci` 与 `pto.dsb`。
2. 将多条 single-line `cacheinvalid` 优化成精确 GM address range CMO。
3. 如果后续确认 release writeback 需要不同 destination 或更精确语义，再决定是否引入新的 CMO op。
4. 扩展 acquire 侧 consumer 范围，从 `load_scalar` 扩展到更多 cacheable GM read。
5. 将 macro op phase 的 memory descriptor 做得更精细，减少误报。
