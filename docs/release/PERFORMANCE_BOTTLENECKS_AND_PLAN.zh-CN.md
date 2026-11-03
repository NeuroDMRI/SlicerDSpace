<div align="center">

# 性能瓶颈与下一轮优化方案

**2026-10-29 · 代码审阅与实测证据整理 · 后续 Agent 实施交接**

[项目](../../README.zh-CN.md) | [已有优化与实测](PERFORMANCE_OPTIMIZATIONS.zh-CN.md) | [运行时间](benchmarks/runtime_devices_20261024.json)

</div>

---

## 1. 结论与实施边界

**10 月 30 日更新：**先完成输入执行门禁、DDSeg 输出凭据和批次终态修复，再补包含报告阶段的计时及目录扫描画像。
验证范围与原始小型测量见[可靠性与计时复核](RELIABILITY_AND_PROFILING_20261030.zh-CN.md)。
这些工程修复不计作新的 MRI 算法加速，下面的长模块候选仍须完整配对。

**11 月 3 日范围澄清：**`8868f74` 完成模型级队列调度、CPU 位置约束等改进，没有改动 `mkcurve.py`，没有实现新的 voxel chunk 动态分发或最优线程数搜索。MK-Curve 原有的多 worker 路径不等于已完成新的调优。后续须分别测试 MK-Curve 的 worker/线程/chunk、DKI 的线程/chunk、WMA 的线程与同预算并发；参见[提交范围与后续配置试验](COHORT_SCHEDULING_REFINEMENT_20261103.zh-CN.md)。此前计算实现优化的证据仍归属于各自的历史记录。

优先处理 **MK-Curve、DKI、WMA** 的整模块瓶颈；面向无 GPU 用户，另把 **DDParcel CPU 推理**列为重点。先取得当前实现的分阶段耗时，再选择占比高、可保持结果一致的计算路径修改。后续还可以通过共用只读输入、复用已完成的依赖和资源感知调度缩短整条路线的等待时间。

本文初稿供后续 Agent（用户指定的 6.1sol）执行。**候选不代表已取得收益**；10 月 29 日首轮实现和测量见第 6 节，其余任务仍是待实施方案。本地修改尚未推送 GitHub。

审阅基线为 `1898f627c16ce45b651854182709e63862d61eb5`。工作树中已有论文及其测试的未提交修改，须保留；实际测速必须另存算法源码、依赖和工具快照的 SHA256，不能只引用提交号。文中的历史绝对路径属于证据来源，不应成为新实现的默认路径。

实施时遵守以下边界：

- 优先保持估计公式、求解器、初始化、体素选择、流线数、采样规则和 QC 阈值不变。
- 原作者代码对照、当前插件前后对照、Python 移植版前后对照分别记录；多线程、设备选择和任务并发单独归类。
- 不用减少 synthetic-b0 网格、缩小脑掩膜、减少流线或迭代次数换取速度，不将无效拟合填补为有效。
- 不把 WLS DKI 换为 MK-Curve、NLS 换为 WLS、UKF 换为 CSD 后称作等价加速。
- 默认路线要求数值与几何一致；仅达到误差容限的候选保留为独立实验，不自动取代默认实现。
- 默认运行用户选定的设备；显式 CUDA 请求失败时保留错误，不静默转 CPU。无需为本轮优化增加大型运行环境。
- 不全局修改 Slicer 进程中的 DIPY、PyTorch 或 VTK；后端适配保持隔离、源码门禁和原路径回退。

---

## 2. 哪些任务真正值得优先投入

下表来自已保存记录；不同输入、线程和计时范围之间不能相加，也不能直接排出跨设备算法速度名次。

| 路线 | 已存耗时 | 测量范围及状态 | 下一步 |
| --- | ---: | --- | --- |
| DDParcel / CPU | 218.57 min | 三视图预测循环，16 CPU 线程；含批次传输/概率累加，不含加载权重及最终清理 | 无 GPU 场景重点 profiling，区分卷积、拼接和累加 |
| MK-Curve / CPU | 153.53 min | 全脑 Python 移植版，`batched`、1 worker、chunk 256；早期配对 165.90 → 153.53 min | 最新 QR/调度消冗之后的全脑配对仍需重测；默认 `reference` 另测 |
| DKI / CPU | 91.60 min | 全脑 WLS 单次记录，740,916 掩膜体素；不是当前新旧实现的完整配对 | 补当前 dense-metrics 全脑对照，读取已有阶段计时 |
| WMA / CPU | 74.22 min | 完整模块，早期原作者配对 102.10 → 74.22 min，74 个束文件一致 | 新的点变换/数组转换优化之后尚缺完整配对 |
| CSD/ACT + SIFT2 / CPU | 36.34 min | 一次 cohort task 记录，含依赖/验收范围；不是孤立新鲜 DTI 拟合时间 | 利用现有 `phase_timings`，拆分 FOD、追踪、SIFT2、转换 |
| Free-water WLS / CPU | 11.89 min | 最新全脑原作者配对 15.64 → 11.89 min，双方 1 数值线程 | 已有 23.99% 降耗；进一步批处理需证明额外收益 |
| 表面后处理 / CPU | 10.87 min | **仅左半球**完整原生组件适配器调用 | 先补两半球阶段画像，不能当作双侧总时间 |
| NODDI / CPU | 5.19 min | 完整 AMICO 8 线程；同输入 1 线程为 29.15 min | 已属上游并行能力，暂不作为新算法优化重点 |

DDParcel 同输入 CUDA 预测循环为 **68.5 s**，DDSeg CPU/CUDA 预测记录为 **669.32 / 7.83 s**。它们提示设备路线的重要性，但 CPU/GPU 比值不能作为本插件相对原作者实现的优化收益。CPU 和 CUDA 必须分别对照自身修改前后的实现。

证据入口：

- [运行设备与时间记录](benchmarks/runtime_devices_20261024.json)与[设备记录](benchmarks/runtime_devices_20261024.json)。
- [完整模块配对](benchmarks/complete_models_20261020.json)：NODDI、WMA、早期 MK-Curve 的完成记录，以及 DKI ROI 对照。
- [最新 Free-water 全脑配对](benchmarks/freewater_protocol_full_20261029.json)。
- [MK-Curve 最新 SciPy 小块对照](benchmarks/mkcurve_qr_scipy1181_20261022.json)：不能外推到全脑。

**证据缺口：**`complete_models_20261020.json` 仍含 DKI `running` 标记；已有报告记载其保存进度停在 710,656 / 740,916。它不是已完成的全脑配对，不能将该标记理解为当前仍有活跃任务，也不能据此填入通过或推算剩余时间。

### 已完成的内容不要重复开发

| 已有优化 | 当前入口 | 后续关注点 |
| --- | --- | --- |
| Free-water 去对角矩阵、作者辅助函数直调、协议数组复用 | `microstructure/freewater_wls.py`、`freewater_dispatch.py`、`freewater_protocol.py` | 每体素权重/伪逆依赖信号，不能跨体素缓存 |
| MK-Curve 设计/积分基缓存、LAPACK 工作空间查询缓存 | `microstructure/mkcurve_tensor.py` | 已缓存的是工作空间大小，不是每次 WLLS 的数值分解 |
| WMA 长度筛选批量化、确定性预处理缓存、VTK 点变换及纤维数组批处理 | `tractography/wma_*.py` | 完整模块的新计时、门禁回退次数及核心热点 |
| DKI dense-metrics | `microstructure/dki.py::_compute_maps` | 已重建 `DiffusionKurtosisFit` 做数组指标；不要再次把它包装为新优化 |
| 多产品 DTI/配准准备复用 | `shared/pipeline.py`、[现有说明](PERFORMANCE_OPTIMIZATIONS.zh-CN.md) | DDSeg LS 与 DDParcel/DDSurfer WLS、各自训练网格仍分开 |
| FW/NODDI/DKI 并发及 CLI 多被试批处理 | `shared/compute_schedule.py`、`scripts/run_batch_pipeline.py` | 已支持并发，下一轮应优化资源准入与嵌套预算 |
| DWI 单次顺序解压后 mmap、ROI 分组统计 | `shared/diffusion_io.py`、`microstructure/pipeline.py::_compute_roi_stats` | 跨消费者共享及真实长路线收益，而非重复报告局部倍数 |

---

## 3. 实施任务与验收条件

优先级中 P0 为实施前置，P1 为长任务及主要用户场景，P2 为按画像决定的后续项。

| ID | 优先级 | 候选 | 类型 | 当前证据 |
| --- | --- | --- | --- | --- |
| R0 | P0 | 当前整模块计时与可切换基准 | 测量基础 | 部分完整配对过期或未完成 |
| R1 | P1 | MK-Curve 网格内不变量复用与尾部映射 | 同算法计算实现 | 已确认重复调用；实际耗时占比待测 |
| R2 | P1 | DKI 拟合/指标包装与重复计算 | 上游 API 适配 / 计算实现 | 已有阶段计时及 dense 路线，剩余热点待测 |
| R3 | P1 | WMA 真实配准/分类热点及批处理回退 | 作者计算实现 / 适配 | 全模块仍长；大数组存在回退分支 |
| R4 | P1，CPU 用户优先 | DDParcel 特征拼接与概率写入 | 同网络计算实现 | 连续 `cat` 和大概率体已确认 |
| R5 | P2 | Free-water 小矩阵批处理 | 高风险数值实现候选 | 每体素伪逆已确认；批处理收益与精确性未测 |
| R6 | P1 画像、P2 实施 | CSD 中间依赖复用与转换按需化 | 工作流复用 | 已有命令阶段计时；全流程前后收益未测 |
| R7 | P2 | 多种 connectome 指标共用端点分配 | 空间查询 / 工作流复用 | Python 路线逐端点查询、指标独立执行 |
| R8 | P2 | 跨微结构模型共享已解压 DWI | I/O 复用 | 当前各阶段/上下文各自准备临时文件 |
| R9 | P2 | 跨被试、跨模型统一资源预算 | 吞吐 / 调度 | 已有 CPU/RAM 准入；批处理存在嵌套并发 |

### R0. 先让基准能够回答“整模块节省了多少分钟”

入口：[完整基准](../../scripts/benchmark_complete_models.py)、[DKI](../../microstructure/dki.py)、[CSD](../../tractography/csd.py)、[WMA](../../tractography/wma.py)。

1. 冻结 `author`、`current`、`candidate` 三条适用路线，记录各条实际启用的优化及源码哈希。MK-Curve 的 `author` 运行时间不能用旧 Python 版本冒充。
2. 先审查 benchmark 的开关范围。现有 WMA 对照主要切换 `registration_preprocess_cache`；MK-Curve reference 只替换部分计算函数且固定 `batched`。它们不能自动成为下一轮所有改动的完整开/关对照。
3. 复用 DKI 的 `phase_times` 和 CSD 的 `phase_timings`。给缺少画像的路线补边界计时和调用计数，记录子进程 CPU 时间、进程树峰值 RSS、磁盘读写量及可用时的 GPU 利用率/显存。
4. WMA 拆为读取/筛选、配准准备、优化器目标函数、图谱分类、离群处理、原空间回写、导出/QC；MK-Curve 拆为准备、信号修复、初始拟合、网格拟合/积分、异常检测、校正、参数图、导出/QC。
5. 先低开销采样定位，再在一个有界真实块上用细粒度 profiler。正式计时关闭重 profiler；父进程等待时间与子阶段时间不可重复相加。
6. 在独立目录补最新 WMA、MK-Curve 两种求解器、DKI 整阶段配对。长任务结果逐阶段落盘；失败或中断保留状态，不能导出为成功汇总。

**产出：**当前绝对耗时、互斥阶段占比、调用数、门禁命中/回退数、峰值资源、关键路径；核函数、完整模块、完整路线三种计时范围分开。

### R1. MK-Curve：减少 synthetic-b0 扫描中的重复工作

入口：[mkcurve.py](../../microstructure/mkcurve.py) 的 `_fit_chunk`、`repair_nonpositive`、`run_mkcurve`；[mkcurve_tensor.py](../../microstructure/mkcurve_tensor.py) 的 `fit_wlls`、`parameter_maps`。

**已确认：**`_fit_chunk` 对每个 synthetic-b0 值再次调用 `fit_wlls`。扫描期间仅信号第 0 列变化，但每次仍对全部测量执行有限性检查、截断及取对数。WLLS 后的 `parameter_maps` 逐体素构建张量、求特征值、生成径向积分基。设计和固定积分基已缓存。

1. 将“信号预处理”与“同一求解器求解”拆开，在 chunk 内复用不变的非 b0 列及其逐元素对数。第 0 列、有效行、权重及每体素数值求解仍按该次输入计算；原始 b0 非有限、网格 b0 有限等情况必须正确区分。
2. 保持矩阵形状、dtype、布局和算术顺序；先不合并不同网格点的 RHS，也不把当前 QR 改为正规方程。缓存 QR 分解虽然数学上可行，但可能改变浮点路径，先列为实验而非默认实施。
3. 单独 profile `parameter_maps`。若占比明显，试作有界批量张量组装、同一 `eigh` 调用和方向基生成；固定轴向符号、重根、极点、零 MD、NaN 传播规则。分块归约与 BLAS 改变可能影响结果，须逐项通过精确比较才启用。
4. `ProcessPoolExecutor` 已限制待处理块数；先测 pickle/等待/返回量，只有其确实影响关键路径时才引入 worker 初始化的只读设计/输入映射。继续保留 Windows/macOS 的 `spawn` 路线。

**验收：**全部网格曲线、22 个原始/校正系数、异常检测、所有标量与专属有效掩膜、残差、Fortran 体素顺序一致；`reference`、`batched` 分别测。`repair_nonpositive` 是按 MATLAB 顺序原地利用已修复邻域的算法，不能用一次卷积或同步插值替换。局部提升后仍须全脑配对，不按 64 体素比例推算分钟。

### R2. DKI：定位真正剩下的拟合或指标成本

入口：[dki.py](../../microstructure/dki.py) 的 `_compute_maps`；[dense-metrics 基准](../../scripts/benchmark_dki_dense_metrics.py)。

**已确认：**每 chunk 调用 `model.fit(observed)`；dense-metrics 已把拟合结果重建为数组版 `DiffusionKurtosisFit`，随后单独调用预测、MK/AK/RK/MKT/RTK/KFA。解压、读取、拟合、指标/预测、赋值、导出已有计时。

1. 先完成当前 full-brain 配对并根据 `tensor_fit`、`metrics_prediction` 判断主攻方向，不继续笼统优化“DKI 很慢”。
2. 检查当前安装 DIPY 的逐体素包装、WLS 内部调用和协议矩阵准备是否已被上游消冗。仅对仍重复的只读准备或 Python 包装做源码门禁适配；信号相关 WLS 权重/伪逆不能复用。
3. 若指标占主导，核对多个指标是否重新计算相同的张量旋转/中间量，再在局部私有上下文共享已相同的中间结果。保持原 DIPY 指标公式、原始不截断结果和无效值语义。
4. chunk 大小只作为内存/包装开销试验，固定线程并记录；改变矩阵批次可能改变数值后端，必须重复做完整结果比较。

**验收：**27 参数、S0、原始/显示六指标、预测残差、fit/valid mask、shell 选择及 QC 一致。不能因为显示图接近而跳过原始参数比较。未认证 DIPY 版本继续走原实现。

### R3. WMA：优化真实热点，避免只追局部大倍数

入口：[wma.py](../../tractography/wma.py)、[wma_transform.py](../../tractography/wma_transform.py) 的 `AuthorCompatibleFiberTransform`、`wma_registration.py` 的 `RegistrationPreprocessCache`、`wma_fibers.py`。

**已确认：**点变换辅助内存估算超过 64 MiB 时，当前实现直接退回作者逐点路径；这与已经存在的分块纤维数组转换不是同一问题。该分支在当前完整 WMA 中是否触发、占时多少尚未测定。

1. 在真实完整运行中保存各适配器命中/回退原因、数组规模和耗时。只有大数组回退实际发生且占比可观，才将点变换改为有界 `TransformPoints` 分块，固定原 VTK 变换和 float64 点顺序。
2. 预算应包含不可避免的结果数组与辅助块；不能为了符合“64 MiB”丢结果或把全部输入复制后再分块。每个块使用独立目标 `vtkPoints`，避免 `TransformPoints` 的追加语义造成重复点。
3. profile 作者配准目标函数及分类中的距离计算。如果瓶颈是 Python↔VTK 桥接或相同确定性数组重复构造，再做批量化/复用；若主要时间已在原生距离核，先提交热点报告，不盲目重写。
4. 所有缓存都不得跨优化器状态复用目标值、改变采样、优化器预算或距离/阈值。保留作者特殊长度筛选定义，不能改成“更合理”的实际弧长筛选。

**验收：**先覆盖跨块边界、不同 mode、精度/布局、内存回退；再比较固定样本的完整小优化器轨迹、变换和分类；最后完整 74 束产物、流线顺序、点/单元属性及 QC。几何一致与小束采样/人工解剖审核分别记录。

### R4. DDParcel：减少不必要的拼接和大体积累加开销

入口：[DDSurfer_Pred.py](../../anatomy/parcellation/DDSurfer_Pred.py) 的 `run_network`、`fastsurfercnn`；[networks.py](../../anatomy/parcellation/models/networks.py) 的 `FastSurferCNN_Fuse_Unet_v3_extended.forward`。

**已确认：**五个编码/瓶颈融合处均通过循环连续 `torch.cat`，构造越来越大的临时张量，再做同一维度 `max`。预测累计体为 `(1, 256, 256, 256, 82)` float32，单这一数组理论大小为 **5.125 GiB**，不含网络和临时量。三视图循环还需 permute、CPU 传输和切片累加。

1. 先测算子时间、内存峰值及 cache/page-fault 情况，确认 CPU 是否计算受限或内存带宽受限。
2. 将每层多次递增 `cat` 改成**相同操作数顺序的一次 `cat`**，再调用原 `max`。不先改成两两 `maximum`，以免 tie、NaN 或 signed-zero 语义改变。
3. 检查 permute 后写入累计体的实际开销；可试作保持视图顺序/0.4、0.4、0.2 权重和 float32 累加的有界块写入或磁盘后备。任何整 GPU 概率累计须先判断显存，不把 5 GiB 数组强行放入小显卡。
4. 评估 `inference_mode` 相对现有 `no_grad` 的实际收益和兼容性。小片段对照只作为筛选，CPU/CUDA 各做自身整推理循环对照。

**特别注意：**当前 `forward` 把 `logits` 放进 `logits_list[0]` 后又对其原地相加，因此该位置含最终叠加结果。不能因调用处只用 `temp_list[:1]` 就删掉 backbone 分类输出或改变其求和顺序。backbone 的中间特征也参与融合，不能直接删除 backbone。

当前 `options_parse` 最后还把 `batch_size` 固定为 4；若之后做批次试验，须先让参数真正生效并记录解析后的配置，不能只传入 `--batch_size` 就宣称已测不同批次。

**验收：**先逐批 logits、再完整三视图累计与最终 labels、原空间重采样、ACT/区域分析所消费的结果一致。改变 batch size、channels-last、AMP、量化、`torch.compile` 或卷积后端另列实验；本轮不默认采用，也不把加线程作为新计算方法。

### R5. Free-water：小矩阵批处理是候选，不保证等价

入口：[freewater_wls.py](../../microstructure/freewater_wls.py) 的 `wls_iter`；[协议缓存](../../microstructure/freewater_protocol.py)。

**已确认：**每体素仍构造自身权重、7×7 正规矩阵和伪逆，然后执行分数网格细化。已有最新完整全脑收益；继续缓存不变设计的边际空间已缩小。

1. profile 伪逆、网格预测/残差、张量分解和 Python 调用各自占比。
2. 若小矩阵调用占主导，试作小块、相同 `pinv` 参数的堆栈版本；先保持每个矩阵与原 `dot` 结果完全一致，单独验证堆栈求解自身是否改变结果。
3. 不改用 `inv`、Cholesky、`hermitian=True` 或不同 rcond 来追速度；这些会改变退化系统处理。超限、非有限或不支持布局仍执行作者路径。
4. 自适应分数区间对每体素不同；若做分组，按精确区间键且保留结果回填顺序，不能量化区间键或合并近似区间。

**验收：**每体素 13 参数、四个输出图/掩膜、排除体素和 QC 一致；若只能数值近似，停止默认集成。批处理支持是 NumPy 现有能力，报告贡献应是实际消除的重复开销及完整阶段收益。

### R6. CSD/ACT/SIFT2：沿依赖复用，转换按消费者需要准备

入口：[csd.py](../../tractography/csd.py) 的 `build_csd_act_commands`、`run_csd_act`、`_timed_step`；[act_prep.py](../../tractography/act_prep.py)；[orchestrator.py](../../shared/orchestrator.py)。

1. 先读取或重测命令级时间，不假设流线生成必然比 SIFT2/FOD 更慢。
2. 核对响应函数、FOD、归一化 FOD、5TT/GMWMI 的缓存是否已涵盖各自独立依赖。只在存在真实重复执行时增加阶段级、完整内容签名的复用和断点恢复。
3. 追踪参数/流线目标数改变时，可复用与这些参数无关且输入相同的 FOD；DWI、梯度、脑掩膜、响应方法、壳层或归一化设置改变时必须失效。
4. 每个消费方可按需请求 TCK/TRK/VTK。已有格式与当前源一致则复用；GUI 必需的 VTK 和用户声明导出仍要生成，不删除默认交付结果。

**验收：**当前默认固定 seed 的 `tckgen` 有单线程重复性约束，不能仅放开线程并称精确加速。SIFT2 权重必须绑定原流线顺序；重跑追踪后旧权重必须失效。区分“从零冷启动加速”和“参数实验/续跑复用收益”。

### R7. Connectome：共用端点分配，保持边权累加语义

入口：[connectome/pipeline.py](../../connectome/pipeline.py) 的 `_build_endpoint_label_assigner`、`_write_python_connectome`、`run_connectome`；[orchestrator.py](../../shared/orchestrator.py) 的 connectome 路由。

**已确认：**Python 路线逐流线调用两次单点 `cKDTree.query`，不同指标分别执行 `run_connectome`。SciPy 支持批量查询；MRtrix 支持导出 `-out_assignments`。当前这些 API 本身均不是新算法。

1. 对相同 tractogram、分区、节点集合、坐标约定、分配方法/半径构建只读端点分配记录，供 count、SIFT2 等指标复用；不同 backend 的分配不可直接视为相同。
2. Python 路线试作有界批量查询，固定 `workers=1`、`eps=0` 和原欧氏距离/上界。覆盖等距、边界、无匹配和越界；不能默认假设批量/逐点的 tie 选择必然相同。
3. 仍按原流线索引顺序累加浮点边权；若用 `bincount`/并行 reduction，必须单独证明舍入一致。长度/标量沿线采样保留各自定义，不仅凭端点替代。
4. 需要 MRtrix 分配复用时先与原命令的矩阵独立比对；不能用 Python 的径向搜索结果替代官方分配后仍声称同一 backend。

**验收：**端点标签、被排除轨迹、边计数、SIFT2 一一对应、节点 LUT 顺序、所有权重矩阵及 QC 一致。这主要改善多指标组合，当前证据还不能证明其为整条路线的大头。

### R8. 输入共享：从“每个阶段一次解压”到“同一快照只读复用”

入口：[diffusion_io.py](../../shared/diffusion_io.py)、`microstructure/dki.py::_compute_maps`、`microstructure/pipeline.py`、`microstructure/mkcurve_inputs.py`。

**已确认：**`diffusion_image` 每个上下文创建独立解压文件；DKI 也自行解压。MK-Curve 另有选壳、合成 b0 及非正信号修复，这一工作副本不能和其他模型共享写入。

1. 用画像统计跨 FW/DKI/NODDI/解剖准备的解压次数、总 I/O 和同时占用磁盘，确认额外副本是否显著。
2. 如值得实施，在单次 subject run 生命周期内创建经过内容校验的只读 DWI 快照，供支持的后端 mmap；模型专属选壳/缩放/修复用独立派生副本。
3. 保留 NIfTI slope/inter、header、dtype、affine 和字节顺序的原语义；不把缩放后数组与 unscaled 数组混用。不兼容的外部工具继续使用标准输入文件。
4. 使用原子发布、进程锁、活跃使用者生命周期和磁盘预算；取消、异常、Windows mmap 未关闭时不能误删活跃文件。不能以 path/mtime 替代内容完整性校验。

**验收：**同一数据从 gzip/proxy/快照读出的值与几何一致，改变源文件及参数时缓存失效；串行/并行均无交叉污染。分别记录新增快照成本与后续消费收益，不能把热缓存优势写成冷启动算法提升。

### R9. 并发：优化整条路线关键路径，而非一味增加进程

入口：[compute_schedule.py](../../shared/compute_schedule.py) 的 `ComputeJob`、`run_independent_jobs`；`microstructure/pipeline.py::_run_independent_models`；[run_batch_pipeline.py](../../scripts/run_batch_pipeline.py)。

**已确认：**微结构模型已有 CPU/RAM 保守预约及独立进程；多个模型目前使用相同形式的保守内存估算。CLI 批处理另有 `--jobs`，可能与单被试内部并发叠加。GPU 配额不在 `ComputeJob` 现有字段内。

1. 先测并发时峰值内存、页错误、I/O 饱和和线程总量；比较固定整机预算下的串行、模型并发、被试并发，不能让并发组获得更多 CPU 后称算法加速。
2. 建立全批次 CPU/RAM/GPU 资源预算，向子工作流传递剩余配额；使用实测保守上界区分不同模型，未知资源时保留顺序运行回退。
3. GPU 推理阶段按设备/显存预约，CPU 分析可与其重叠；不能同时启动多个大概率体或多个整脑内存副本导致 OOM。预约属于调度，不能忽略外部已有负载。
4. 待前置产品校验并原子发布后，才放行独立分支。共享目录、最终 ROI 表、场景报告均保持单写者；Qt/MRML 加载在 GUI 主线程执行。
5. 维持默认保守行为和现有参数兼容；不要把默认流程改成所有可选算法同时运行。

**验收：**相同输入和预算下的组合 makespan、被试/小时、峰值资源、等待原因有记录；取消和子任务失败能停止完整进程树，无混合旧新产物。并发吞吐收益单列，不能乘进单算法加速比。

---

## 4. 基准、测试与交付约定

### 4.1 推荐执行顺序

1. **R0 基线与画像。**先补当前长阶段计时和有效的开关对照，不直接启动多被试全算法长跑。
2. **R1 / R4 小块候选。**MK-Curve 不变量、DDParcel 一次拼接最适合先做受控实现；根据 profile 结果决定先后。
3. **R2 / R3 长模块。**以当前热点占比决定修改；WMA 大数组回退若未命中，跳过该小项，转向真实配准/分类热点。
4. **R6 / R8 / R9 路线收益。**组合场景验收依赖复用与资源预约；R5 / R7 只在画像显示值得时投入。
5. 小块通过后补完整模块，再补一个代表性完整路线。最后才把已经证实的通用计算优化整理进 Supplementary；论文未提交修改保持不动，新增结论必须来自新证据。

每个候选先用三个不同采集的原生 ROI/真实纤维块，至少一次预热、三次轮换重复。全模块先一名被试配对作为 pilot，通过后再扩展被试与重复。独立重估配准存在随机波动的路线，先固定同一变换隔离计算差异，再另验从零路线的科学 QC；两类实验不要混写。

停止条件：候选阶段占比很低、完整模块收益落入测量波动、增加复杂依赖或破坏精确结果时，保留结论并停止默认集成。无需为凑优化项数继续修改。

### 4.2 先复用现有命令，再扩展真实开关

以下为现有脚本参数模板，变量须由执行 Agent 自动定位并核验，不要求普通用户填这些实验路径。`SOURCE_A/B/C` 指含四个标准输入文件的 **input 目录**，`NPZ_*` 指已有 `signals/design/grid` 真实信号块。使用正确科学环境的 Python，输出目录必须全新。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 已有基准：用于核对当前协议复用，不自动覆盖新的候选实现。
python scripts/benchmark_freewater_protocol.py \
  --source "$SOURCE_A" --source "$SOURCE_B" --source "$SOURCE_C" \
  --output "$FRESH_ROOT/freewater-roi" --crop-size 12 --repeats 3

python scripts/benchmark_dki_dense_metrics.py \
  --input "$SOURCE_A" --input "$SOURCE_B" --input "$SOURCE_C" \
  --output "$FRESH_ROOT/dki-roi" --side 24 --repeats 3

# 基准只替换 tensor 模块；若改动 _fit_chunk，本基准也必须扩展为双快照。
python scripts/benchmark_mkcurve_qr.py \
  --reference "$FROZEN_TENSOR_PY" \
  --input "$NPZ_A" --input "$NPZ_B" --input "$NPZ_C" \
  --output "$FRESH_ROOT/mkcurve-numeric.json" --repeats 3

# 现有完整 DKI 配对；执行前完成资源与预计磁盘/时间检查。
python scripts/benchmark_complete_models.py \
  --source "$SOURCE_A" --subject-id "$SUBJECT_ID" \
  --stages dki --output "$FRESH_ROOT/dki-full"

# WMA：当前小优化器基准固定 1000 条采样，不能替代新大数组跨块用例。
python scripts/benchmark_wma_transform.py \
  --wma-root "$WMA_ROOT" --input "$REAL_TRACTOGRAM" \
  --output "$FRESH_ROOT/wma-transform" \
  --report "$FRESH_ROOT/wma-transform.json"
```

`benchmark_complete_models.py` 当前固定 optimized→reference 顺序且无重复参数。后续需增加轮换/重复及准确的候选快照开关，不能在文档中虚构脚本已有 `--repeats` / `--candidate` 等参数。DDParcel 现有 `benchmark_runtime_devices.py` 主要比较设备和已存标签；新候选仍须增加同设备源码对照与 logits 比较。

推荐本地回归入口：

```bash
python -m pytest -q \
  Testing/Python/test_mkcurve_python.py \
  Testing/Python/test_mkcurve_qr_optimization.py \
  Testing/Python/test_microstructure_dki.py \
  Testing/Python/test_dki_dense_metrics.py \
  Testing/Python/test_wma_transform.py \
  Testing/Python/test_wma_fibers.py \
  Testing/Python/test_wma_registration_cache.py \
  Testing/Python/test_freewater_wls_optimization.py \
  Testing/Python/test_freewater_protocol.py \
  Testing/Python/test_parcellation_pipeline.py \
  Testing/Python/test_connectome_pipeline.py \
  Testing/Python/test_diffusion_chunk_io.py \
  Testing/Python/test_compute_schedule.py \
  Testing/Python/test_run_batch_pipeline.py
```

以上测试只能作为已有回归；每个新候选补专属边界测试、完整产物比较及复制安装副本验收。全部本地执行，不启动 GitHub Actions。

### 4.3 结果记录格式

原始材料写入 `ignore/validation/performance_next_<date>/`，公开脱敏汇总写入 `docs/release/benchmarks/`。不要提交受试者数据、权重副本、大型 profiler dump 或硬编码本机依赖路径。

| 记录 | 必需内容 |
| --- | --- |
| `environment.json` | OS、CPU、内存、GPU、Python、库/外部程序版本、线程限制、负载、源码/模型/输入/基准快照哈希 |
| `profile.json` | 互斥阶段耗时、调用数、缓存/回退、I/O、进程树峰值 RSS、可用时的 GPU 指标 |
| `measurements.csv` | case、variant、comparison_kind、scope、solver、device、线程、运行次序、warm/cold、wall/CPU 秒 |
| `comparison.json` | 数组/原始参数/几何/顺序/掩膜/QC 比较、错误上界、首个失败位置、产物完整性 |
| `summary.md` | 核函数、整模块、路线的分钟数与节省时间；失败候选、数据范围和下一步 |

数组比较包含 shape、dtype、affine、有效/无效掩膜及 NaN 位置；NaN 相同位置可视为语义相同，精确性结论需说明是否检查位模式。VTK 比较包含连接拓扑、点和 cell data、流线顺序。普通日志、绝对路径、时间戳和 gzip 元数据等非科学差异单独列出。

新旧两组要使用相同导出和校验开销；GPU 计时在边界同步，CPU 异步调度不得只量提交时间。缓存复用、warm OS cache 和完整 cold run 分开标注；不需要通过清空整台共享服务器缓存影响其他用户。

### 4.4 最终应回答的四个问题

1. 哪个真正热点被消除，原作者最新版是否仍存在该开销？
2. 同输入、同参数、同资源预算下，完整模块节省多少秒/分钟？
3. 原始数值、几何、流线/权重顺序和 QC 是否保持一致？
4. 对代表性完整路线的 makespan 和资源峰值有什么实际影响？

只在这些证据齐全后，把“候选”改为“已实现并验证”，更新已有性能报告、网页数据与 Supplementary。未测过的收益留空，不外推局部倍数。

---

## 5. 上游核对与参考依据

实施前重新核对已安装版本、上游最新稳定发布和涉及函数的源码哈希；升级在隔离环境中进行。本文对代码路径的判断基于审阅工作树，既有报告中 10 月 22/29 日的上游检查不能替代实施当天复核，也不应一次性升级所有依赖。

- [既有性能与原作者对照](PERFORMANCE_OPTIMIZATIONS.zh-CN.md)：已完成优化、源码门禁、历史成功/失败范围。
- [SciPy cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)：支持成组查询、距离上界、精确/近似与 worker 参数；不提供本插件 tie 一致性保证。
- [NumPy pinv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html)：支持矩阵堆栈；伪逆阈值和 Hermitian 路线会影响数值行为，不能据 API 支持就断言批处理精确一致。
- [VTK vtkAbstractTransform](https://vtk.org/doc/nightly/html/classvtkAbstractTransform.html)：`TransformPoints` 接口及其追加输出语义；本插件仍须测试实际支持的 VTK 版本。
- [MRtrix tck2connectome](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html)：端点分配、SIFT2 权重及分配导出接口；改用另一种分配方法属于方法变化。

这些文档提供 API 依据，未证明任何候选已加速或已通过科学验收。

---

## 6. 首轮实施记录：2026-10-29

R0 已补双源码快照的 MK-Curve 配对、独立于正式计时的 profiler，以及使用真实权重/真实输入的 DDParcel 整批 forward 基准。两者均保留原始重复耗时和输入/源码哈希。[脱敏结果](benchmarks/performance_next_20261029.json)。

R1 已实现 chunk 内非 b0 信号对数/设计准备复用，并将 QR 路线每体素重复的权重有效性与最大值检查合并为数组检查；未改变求解器及归一化数值。SciPy 1.18.1、单数值线程、一次预热及三次轮换重复的完整 64 体素计算块结果如下，涵盖全部 synthetic-b0 曲线、校正、参数图、残差和掩膜：

| 实例 | 原 Python 实现 | 当前 QR 路线 | 降耗 | 数值结果 |
| --- | ---: | ---: | ---: | --- |
| 100206 | 1.193 s | 1.059 s | 11.26% | 全部一致 |
| 149741 | 1.126 s | 1.000 s | 11.24% | 全部一致 |
| 877168 | 1.102 s | 0.989 s | 10.24% | 全部一致 |

早期 batched 缓存候选出现负收益，因此默认仅 QR 路线启用信号复用；batched 保留原信号处理路线。最终 batched 配对约在 -1.50% 至 +3.07% 之间，不记为收益。上述局部数字不推算为全脑分钟数，也不与此前加速比相乘。

R4 已完成相同操作数顺序的一次拼接，并保持最大值与最终 logits 原地相加语义。真实训练权重、原生准备特征、轴位 256×256 单批、4 CPU 线程的三次中位数为 **59.814 → 59.851 s**，所有返回 logits 的字节哈希一致；没有可报告的整批速度收益。该修改仅减少中间拼接复制，不标为新的算法加速；暂不继续昂贵的全体积 CPU 对照。

Slicer Python 下的 MK-Curve、DKI、DDParcel 相关回归共 **53 项通过**。全脑新旧配对、WMA 核心热点、DKI 剩余热点及路线级调度仍待后续处理；网页性能图和论文不加入这些未完成的结论。

## 2026-11-03 调度候选补充

已完成 Nipype 1.11.0 / 主分支核查，采用其资源声明检查和 `mem_thread` 候选排序，并直接复用 Python `graphlib` 与 `concurrent.futures` 完成模型就绪及完成事件处理。历史 RAM/时长反馈改为每次准入刷新；保留队首保护、独立科学子进程和显式预算。没有新增工作流框架依赖，没有修改科学拟合或实现 MK-Curve chunk 动态分发，也未自动寻找最优线程数。

新候选已冻结并进入现有完整队列之后的验证队列。软件 1,831 项回归通过（9 项跳过）；完整科学对照仍待完成，不能从单元测试推算吞吐收益。来源、实现范围、快照和运行记录见 [调度改进记录](COHORT_SCHEDULING_REFINEMENT_20261103.zh-CN.md#nipype-对照与本轮候选)。

后续最新候选已补上可选的有界线程/worker/块大小搜索，以及 MK-Curve 的完成驱动体素块分发。
新增目标解析直接复用单被试 CLI，按所选算法子集规划。全套软件回归为 1,853 项（9 项跳过），
Slicer 内定向回归 47 项通过。固定配置与自动选参的完整对照分组计时，尚未报告新加速率。
实现边界、校准代价及队列位置见上述记录的“最新候选”一节。
