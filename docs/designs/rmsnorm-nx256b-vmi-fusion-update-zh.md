# RMSNorm `N x 256B` VMI VF Fusion 飞书同步材料

## 一句话结论

我们已经在 A5 上完成 `N=64`、每行一个完整 FP32 VL（BF16 输入
`[64,64]`）的 RMSNorm 六路径闭环。VMI VF（D）相对 VMI 非 VF（B）的
vector median 从 `100.260 us` 降到 `8.470 us`，加速 `11.84x`；输出与独立
golden、AscendC 非 VF、AscendC 手写 VF、PTO A/B/C/D 全部 BF16
byte-exact。

## 为什么该用例是典型逐行 VF

- 输入/输出为 BF16 `[N,64]`，gamma 为 BF16 `[1,64]`。
- BF16 转为 FP32 后，每行 `64 x 4B = 256B`，恰好是 A5 一个完整 VL。
- 外层遍历 N 行；每行内部依次完成 convert、square、row reduction、标量
  RMS 链、broadcast divide、gamma multiply 和 RINT BF16 convert。
- 不依赖 4096 宽行内 chunk 累加，因此直接检验“多行、每行一 VL”的 VF
  主行域融合能力。

## 四条核心路径

| 路径 | Vector median | Task median | AIV cycles | 每行 vector | 相对 AscendC 非 VF | 相对 VMI 非 VF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AscendC 非 VF（AC-U） | 276.719 us | 367.715 us | 35,219 | 4.3237 us | 1.00x | 0.36x |
| AscendC 手写 VF（AC-F） | 7.503 us | 90.769 us | 7,464 | 0.1172 us | 36.88x | 13.36x |
| VMI 非 VF（B） | 100.260 us | 158.243 us | 14,816 | 1.5666 us | 2.76x | 1.00x |
| VMI VF（D） | 8.470 us | 71.740 us | 5,901 | 0.1323 us | 32.67x | 11.84x |

每条路径 warmup 1 次、串行 profile 5 次。最大 vector CV 为 `0.633%`，低于
2% 门槛。D/B 比值的 bootstrap 95% 区间为 `[0.08390, 0.08494]`，明确排除
1.0。D 的 vector 阶段距手写 AC-F 约 12.9%，但本次测量的 task/AIV 总时间
更低。

## 正确性与计算/数据等价性

六条路径共享同一输入文件、输出初始化、launch、BF16 边界、gamma、epsilon、
FP32 算子顺序、单 VL row reduction 和最终 RINT BF16 conversion。CPU golden
独立实现，不复用任何 kernel。

采用两套固定数据：

- `exact-association`：使用可精确表示的数，保证 FP32 reduction association
  可稳定对比；
- `layout-sensitive`：每行具有不同的正负与幅值标记，用于发现跨行、mask、
  layout 和 gamma broadcast 错误。

六路径、两数据集各执行一次 cold 和两次 non-profile；layout-sensitive 还覆盖
全部 warmup/profile。最终 `42/42` 次执行 PASS，30 个 profile 后也逐次验证输出，
两个 BF16 golden hash 在所有路径完全一致。

## 本次 PTOAS 优化

1. `FusionRegion` 只允许 selected VMI TileOp，PTODSL fallback、DMA、sync 和未知
   op 作为边界。
2. 增加 `N x 64 / N x 1` 的受限 `trowexpanddiv` VMI candidate。
3. 引入可证明的 principal row domain，使 `N x 64 -> N x 1 -> N x 64` 在保持
   row-preserving 的前提下规划为同一主 region；gamma conversion 留在 preheader。
4. 将九段逐行 loop 合并为一个主 row loop。
5. 在 flatten 后进行 vreg forwarding：x、square、sum、mean、rms、normalized、
   scaled 全部 SSA/vreg 直传。
6. 将单 lane scalar store + full-VL broadcast load 转换为寄存器 duplicate，并
   将只读 gamma load/convert 提到 preheader。
7. 只有地址、offset、mask、alias、escape、sync、unknown call 和唯一 pipeline
   证明全部成立时才消除 UB 流量；证明失败完整保留旧路径。

结构变化与性能归因如下：

| 路径 | Row loop | VLD | VST | VF 内部 membar | Vector median |
| --- | ---: | ---: | ---: | ---: | ---: |
| B：candidate，无 fusion | 9 | 12 | 10 | 13 | 100.260 us |
| C：Region + loop fusion | 1 | 12 | 10 | 10 | 151.130 us |
| D：C + forwarding/elision | 1 | 2 | 1 | 0 | 8.470 us |

C 说明“region 数量变少/loop 合并”本身不能代表收益；在保留中间 UB round-trip
时反而会变慢。D 的收益来自真正删除计算阶段的中间 VLD/VST 和 membar。D 的
动态物理访问只剩一次 gamma VLD、每行一次 x VLD 和每行一次 y VST，即 65 次
输入 VLD、64 次输出 VST；GM/UB 边界所需同步单独统计，不算作 VF 内部开销。

## PR 与证据

- 分支：`codex/vmi-fusion-elementwise-coverage`
- Commit 8：`0e57d1572 test(rmsnorm): add A5 row-VF comparison harness`
- 最终 PR 保留从基线 `a9797c468` 开始的 9 个原子 commit。
- 完整证据：远端 finalized private-lab experiment
  `20260821-rmsnorm-nx256b-sixway-0e57d1572-a5-n64-r1`。
- 仓库内只提交 fixture、harness、ADR 和摘要，不提交 binary、完整 IR、profile。

按“性能闭环优先”的约定，`N=1/8/32` 扩展采样、assembly branch/spill census
以及四个历史 TileFusion lit 期望更新留到后续，不影响本次 `N=64` 正确性和
性能验收结论。
