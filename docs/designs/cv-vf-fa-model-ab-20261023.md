# 固定 FA：代理模型推荐与原生 VF + PlanMemory 实测

本轮工作绑定输入 SHA256
`3c67b7a42b05b7fa1dd705e38892a335dc336bd6f14e7c025a05340e4fa00e9b`。
输入是 M32 FA，Q/KV/D=128/1024/128，FP16 输入、FP32 输出，1 AIC + 2 AIV。

## 比较对象

四个候选采用相同的原生 VF、自动 PlanMemory、固定存储容量和 PTOAS 同步。
仅改变迭代距离 P=0/1/2/3。对这份 32-token 静态图逐项核对后，才能将其
称为旧工具的 warm-up depth=1/2/3/4；不推广为任意程序的换算公式。
P=0 是相同静态展开下的串行阶段图，不是未经展开的源循环。

兼容导入器只接受上述源码摘要。它保留原算法、Q/KV 递推和行别名，
把旧地址转为逻辑根和相对视图，不把旧物理地址作为 Plan 输入。
原生 pass 完成 CV 重排、共享存储绑定、PlanMemory 和同步。
为限制本轮证明范围，禁用地址复用，但实际地址仍由 PlanMemory 自动分配。
通信容量固定 QK=1/P=3/PV=1；alpha 有四个版本，Q 的 O/L 状态分别保留。
本轮没有逐 Buffer 槽数搜索，也不评价 VF 融合单独贡献。

## 模型与安全边界

TileSim adapter 读取同一次对象编译反馈中的结构化最终 VPTO。
COPY/MMAD 采用 TileSim 模型，低层 VF 采用独立进程的上游 VfSimulator。
所有未知操作必须拒绝；VSLDB VL1 的精确范围与 VLDS 时序近似、
逐标量一周期和目标时钟假设均单独报告。

用户已确认没有 DT 参数包，授权先比较 PR 参数代理模型与 DT 实测。
硬件是 Ascend950DT_9592，时序参数是 950PR_9599，不能据此签发目标匹配 G4。
禁止用本轮 FA 实测拟合成本或重新挑选候选。

原生检查覆盖 SSA、RAW/WAR/WAW、地址与事件合法性以及同步覆盖。
受限 G2 另检查不重叠布局、访问范围、跨核 flag 代际、DAG 进展和完成屏障。
G2 证明只使用依赖，不使用模型时间作为安全依据。
通用编译反馈中的 `evidence_incomplete` 不自动改为通过；单独签发 SHA 绑定的 FA 证据。

## 重放顺序与判定

1. `fa_model_ab.py prepare SOURCE MATRIX --compiler PTOAS`：编译四个独立符号。
2. TileSim `core.frontend.adaptor.ptoas_prepared`：离线计时并选择候选。
3. `fa_model_ab.py freeze MATRIX REQUEST RESULT`：校验并冻结模型与产物。
4. 独占队列内 `fa_model_ab.py run MATRIX G3_EXPERIMENT --mode g3`。
5. 全部 G3 通过后，在新实验目录使用 `--mode performance --g3-report REPORT`。

G3 覆盖零、全 V=1、全 Q/K/V=1、三个 pattern 种子，两种相反执行顺序。
每次重新初始化输入、NaN 输出和 guard，使用独立 C++ FP64 与 NumPy FP64 golden。
常量误差不超过 1e-6；pattern 误差不超过 1e-4 + 1e-3 |golden|。
任一错误或 60 秒超时立即停止，不进行性能签收。

性能使用四组平衡顺序、20 个 block、80 个同板 profiler 样本；每次验证输出。
主结论分别报告：实际 B/A 收益、模型选择与四组实测排序、绝对时延预测误差。
平均收益 >=2%、配对 bootstrap 95% 下界 >0、最坏配对回退 <=2% 才通过收益门。
预测误差另与 10% 比较，不用模型误差否定已验证的优化收益。

实施和原始证据记录：`cv-costmodel/20261023-002-fa-native-model-ab`。
结果完成后补入同一输入、候选、对象和设备绑定的报告，不复用历史 130 个样本。
