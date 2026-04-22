# PTOAS TileOp Expand Design

## 1. Goal

`Expand TileOp` 的目标是把前端易写的 tile 级算子展开为一组更接近 PTO-ISA 语义的基础操作，同时保持用户可见 IR 始终停留在 `!pto.ptr`、`!pto.tensor_view`、`!pto.partition_tensor_view` 和 `!pto.tile_buf` 这套类型体系内。

这条分支的约束很明确：

- 不再依赖旧式 buffer bridge
- 不再要求用户输入额外的中间表示
- Expand、PlanMemory、InsertSync、EmitC 使用统一的 tile/view 语义模型

## 2. 输入与输出

### 输入

Expand pass 接收的输入主要由以下对象组成：

- 全局地址：`!pto.ptr<T>`
- 全局逻辑视图：`!pto.tensor_view<...>`
- 全局切片视图：`!pto.partition_tensor_view<...>`
- 本地 tile：`!pto.tile_buf<...>`

常见入口形态：

```mlir
func.func @kernel(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>, %c: !pto.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  %a_tv = pto.make_tensor_view %a, shape = [%c16, %c64], strides = [%c64, %c1]
    : !pto.tensor_view<16x64xf32>
  %b_tv = pto.make_tensor_view %b, shape = [%c16, %c64], strides = [%c64, %c1]
    : !pto.tensor_view<16x64xf32>
  %c_tv = pto.make_tensor_view %c, shape = [%c16, %c64], strides = [%c64, %c1]
    : !pto.tensor_view<16x64xf32>

  %a_pt = pto.partition_view %a_tv, offsets = [%c0, %c0], sizes = [%c16, %c64]
    : !pto.tensor_view<16x64xf32> -> !pto.partition_tensor_view<16x64xf32>
  %b_pt = pto.partition_view %b_tv, offsets = [%c0, %c0], sizes = [%c16, %c64]
    : !pto.tensor_view<16x64xf32> -> !pto.partition_tensor_view<16x64xf32>
  %c_pt = pto.partition_view %c_tv, offsets = [%c0, %c0], sizes = [%c16, %c64]
    : !pto.tensor_view<16x64xf32> -> !pto.partition_tensor_view<16x64xf32>

  %ta = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %tb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %tc = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>

  pto.tload ins(%a_pt : !pto.partition_tensor_view<16x64xf32>)
            outs(%ta : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  pto.tload ins(%b_pt : !pto.partition_tensor_view<16x64xf32>)
            outs(%tb : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  pto.tadd ins(%ta, %tb : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
                          !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%tc : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  pto.tstore ins(%tc : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
             outs(%c_pt : !pto.partition_tensor_view<16x64xf32>)
  return
}
```

### 输出

Expand 之后，算子会被拆成：

- 更细粒度的 tile intrinsic
- 显式的 view / tile 传递关系
- 便于 InsertSync 和 EmitC 直接消费的 SSA 依赖图

pass 不再生成额外的 legacy bridge 产物。

## 3. Core Rule

### 3.1 统一语义对象

Expand pass 内部只承认四类对象：

- `ptr`：全局基地址
- `tensor_view`：完整逻辑视图
- `partition_tensor_view`：切片视图
- `tile_buf`：本地 tile

### 3.2 Tile 操作数

对于 tile 输入，Expand 直接保留原对象，不引入中间桥接类型。

### 3.3 View 操作数

对于 view 输入，Expand 只追踪以下链路：

- `pto.make_tensor_view`
- `pto.partition_view`
- `pto.subview`
- `pto.bitcast`
- `pto.treshape`
- `pto.bind_tile`

这些对象统一进入 `Utils` 的 tile/view 语义抽取逻辑，供：

- PlanMemory alias 分析
- InsertSync buffer 依赖构造
- EmitC lowering

## 4. Expand Strategy

### 4.1 加载类

`pto.tload` / `pto.tprefetch` 直接消费 `partition_tensor_view`，目标是 `tile_buf`。

### 4.2 存储类

`pto.tstore` / `pto.tstore_fp` 直接把 `tile_buf` 写回 `partition_tensor_view`。

### 4.3 纯 tile 计算类

诸如 `pto.tadd`、`pto.tmul`、`pto.tmatmul`、`pto.tmov` 等算子只在 tile 世界里流转，不需要额外桥接。

### 4.4 视图变换类

`pto.subview`、`pto.bitcast`、`pto.treshape` 被当作 view/tile 语义的一部分保留到后续分析，而不是先降成另一套历史形式。

## 5. Folding Rule

Expand 之后还需要做一轮清理，把可以静态求值的内容折叠掉：

- `pto.tile_buf_addr`
- `pto.tensor_view_addr`
- `pto.get_tensor_view_dim`
- `pto.get_tensor_view_stride`
- 可静态确定的 shape / stride / valid 信息

目标是让 EmitC 阶段看到的是：

- 清晰的 base pointer
- 清晰的 shape / stride 常量
- 无多余中转节点的 view/tile 依赖链

## 6. InsertSync Cooperation

InsertSync 不能只看 tile 本体，还必须看 view 链带来的别名传播。

因此 translator 必须把以下对象全部纳入同一套 root/alias 图：

- `pto.make_tensor_view`
- `pto.partition_view`
- `pto.subview`
- `pto.bitcast`
- `pto.treshape`
- `pto.bind_tile`

否则在 `--enable-insert-sync` 下，经 view 传递的 use/def 会丢失，最终导致 barrier / event 漏插。

## 7. PlanMemory Cooperation

PlanMemory 只规划 tile 世界的本地对象：

- `pto.alloc_tile`
- `pto.declare_tile`

它依赖统一的 tile/view alias 抽取来做：

- root traceback
- liveness
- reuse
- address assignment

用户路径上不保留任何历史兼容分支。

## 8. EmitC Cooperation

EmitC 只接收 tile/view/pointer 三类 public 语义：

- `tile_buf` 映射为本地 tile 变量
- `partition_tensor_view` 映射为 GM 逻辑视图
- `ptr` 映射为基地址

因此 Expand 的职责不是制造新的中间表示，而是把 tile 语义整理到 EmitC 能稳定消费的状态。

## 9. Example

下面给出一个 tile 计算的典型展开方向：

```mlir
%src_tv = pto.make_tensor_view %src_ptr, shape = [%c16, %c64], strides = [%c64, %c1]
  : !pto.tensor_view<16x64xf32>
%dst_tv = pto.make_tensor_view %dst_ptr, shape = [%c16, %c64], strides = [%c64, %c1]
  : !pto.tensor_view<16x64xf32>
%src_pt = pto.partition_view %src_tv, offsets = [%c0, %c0], sizes = [%c16, %c64]
  : !pto.tensor_view<16x64xf32> -> !pto.partition_tensor_view<16x64xf32>
%dst_pt = pto.partition_view %dst_tv, offsets = [%c0, %c0], sizes = [%c16, %c64]
  : !pto.tensor_view<16x64xf32> -> !pto.partition_tensor_view<16x64xf32>

%tile0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tile1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>

pto.tload ins(%src_pt : !pto.partition_tensor_view<16x64xf32>)
          outs(%tile0 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
pto.tadd ins(%tile0, %tile0 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
                        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%tile1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
pto.tstore ins(%tile1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%dst_pt : !pto.partition_tensor_view<16x64xf32>)
```

## 10. Result

这条分支的最终状态应当是：

- 用户输入只写 tile/view/pointer
- 中间分析只看 tile/view/pointer
- EmitC 只消费 tile/view/pointer
- 仓库里不再保留旧式 buffer 语义作为兼容路径
