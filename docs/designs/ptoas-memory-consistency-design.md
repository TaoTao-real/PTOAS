# PTOAS 内存一致性设计

本文说明 PTOAS 当前暴露的内存一致性 IR，以及为什么暂时移除自动
MemoryConsistency 分析 pass。

## 当前状态

PTOAS 目前保留显式内存一致性 IR 和 EmitC lowering：

| PTO IR | 语义 | EmitC lowering |
| --- | --- | --- |
| `pto.cmo.cacheinvalid all #pto.address_space<gm>` | whole-cache GM cache maintenance，保守 drain 当前 core 的 pipe | `pipe_barrier(PIPE_ALL); dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)` |
| `pto.cmo.cacheinvalid %addr single_cache_line : !pto.ptr<T, gm>` | 指定地址所在 cache line 的 GM cache maintenance，并作为 `PIPE_S` 边界参与 InsertSync 地址依赖分析 | 必要时先由 InsertSync 生成 `MTE2/MTE3 <-> PIPE_S` set/wait，再生成 `dcci((__gm__ void*)addr, cache_line_t::SINGLE_CACHE_LINE)` |
| `pto.fence.barrier_all #pto.fence_scope<gm>` | GM visibility fence | `dsb(DSB_DDR)` |

`pto.cmo.cacheinvalid` 和 `pto.fence.barrier_all` 由 PyPTO 或用户显式插入。
PTOAS 不再通过独立 `pto-memory-consistency` pass 自动扫描 signal/payload
关系、消除 marker-only CMO，或诊断缺失的 CMO/fence。为了避免复杂分析带来的
编译时退化，地址型 CMO 只作为已有自动同步 solver 的局部依赖节点参与分析。
因此若使用 `single_cache_line` 精确模式并希望 PTOAS 自动插入
`MTE2/MTE3 <-> PIPE_S` set/wait，需要在编译时启用 `--enable-insert-sync`
或覆盖该场景的同步 solver；只跑默认 EmitC lowering 时，CMO 只生成 DCCI。

## 为什么移除自动分析 pass

issue #950 暴露了 `pto-memory-consistency` 在复杂单 block 控制流中的编译时
间退化。该 pass 的 state `merge()` 直接追加状态向量；处理单块 `scf.if` 时，
子 region 返回值已经包含入口状态，父层又把它合并回原状态，导致每经过一个
`if`，历史状态近似翻倍。

issue #950 的 repro 中包含约 80 个连续 `scf.if`、160 次 GM scalar load、163 次
GM scalar store，因此状态规模接近指数增长，触发 ptoas 0.50 相比 0.48 的确定性
编译超时。

在重新设计 bounded、去重的数据流模型之前，默认 pipeline 不应继续运行这个
自动分析 pass。当前策略是先回到显式 IR 模式，保证编译时间可控。

## 使用要求

PyPTO 或用户需要显式表达内存一致性动作：

```mlir
// cacheable scalar GM store 发布 payload
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32, gm>, i32
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

```mlir
// TWait 后读取 cacheable scalar GM payload
pto.comm.twait ...
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
%value = pto.load_scalar %payload[%idx] : !pto.ptr<i32, gm> -> i32
```

如果涉及 non-cacheable MTE/comm macro payload，上游需要在 publish 点显式生成
`pto.cmo.cacheinvalid` 和 `pto.fence.barrier_all #pto.fence_scope<gm>`。推荐优先使用
`single_cache_line` 精确地址形式，PTOAS 会根据该地址和 MTE2/MTE3 payload 访问的
alias 关系插入 `PIPE_S` set/wait。该精确 pipe ordering 依赖自动同步 solver，
issue #872 这类 `TPUT` macro phase 目前由 legacy InsertSync 的
`SyncMacroModel` 覆盖。如果上游无法或不想精确指定地址，可以使用
`pto.cmo.cacheinvalid all #pto.address_space<gm>`，EmitC lowering 会在 whole-cache
`dcci` 前生成 `pipe_barrier(PIPE_ALL)`。`fence.barrier_all` 本身只生成
`dsb(DSB_DDR)`，不会再生成 `PIPE_FIX` drain。

## 后续重新引入自动分析的要求

后续若恢复 MemoryConsistency pass，需要先满足以下条件：

- state merge 必须 bounded，不能通过简单追加导致指数增长。
- state key 至少应按 payload identity、pipe、cacheability 和 action kind 去重。
- `scf.if` 的 region exit state 不能重复携带父层入口 state。
- loop 和多 block region 需要明确固定点或保守 summary 上界。
- 新 pass 需要包含 issue #950 规模的 compile-time 回归测试。
