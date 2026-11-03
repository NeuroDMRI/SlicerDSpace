# 队列调度改进与完整体积验证

日期：2026-11-03。此记录区分实现、软件回归及尚未完成的真实队列测量。

## 已实施

| 问题 | 改动 | 保持的科学计算约定 |
| --- | --- | --- |
| Microstructure 占用一个长阶段预约 | DTI 导出、Free-water、NODDI、DKI/MK-Curve 分别获取 CPU/RAM 预约；adaptive 允许有界独立模型子进程 | 拟合方法、输入、有效掩膜及共享统计表不变 |
| 线程变量不能约束部分原生程序 | Linux 按阶段分配互不重叠的 CPU 亲和性集合，原生子进程继承；退出后恢复原范围 | 不修改 SlicerDMRI 的估计器或求解公式 |
| UKF 自动线程数可能超出预约 | 队列模式的自动线程数采用阶段预算；超额显式请求报错 | 最终线程数写入实际参数与缓存签名 |
| 超时估计被误视为资源释放 | 未完成的超时任务继续占用预约；保留队首保护、120 秒等待及 8 次越过屏障 | 不抢占或截断正在拟合的任务 |
| 缓存命中可能污染历史画像 | 缓存复用记录不参与拟合时长和 RAM 高水位学习；补充原生/配准 DTI 与标量导出的命中标记 | 继续验证产物凭据，不降低缓存完整性检查 |
| 并发解压竞争 | DWI 解压采用单独的两许可 FIFO，解压完成即释放 | 不改变图像 dtype、缩放、affine 或数据值 |
| 缺少整批资源曲线 | 独立基准记录进程树 CPU、RSS、I/O、任务等待与自有进程显存；另存整张 GPU 的负载背景 | 整张 GPU 的占用不归因于本研究 |

RAM 和 GPU 槽为准入预约，不是 cgroup 内存限额或按设备显存分配。阶段级 I/O 槽与短时解压许可分别计数。环境检查、报告生成及等待进程本身仍会占用资源；基准的整树监测包含这些进程。

### 提交范围与未完成项

`8868f74` 的主要改动是模型级资源预约及队列运行控制，没有修改 `microstructure/mkcurve.py`。因此，这个提交不包含新的 voxel chunk 调度优化，也不能作为 MK-Curve 拟合核心加速的证据。此前已有的 chunk 内信号复用及其历史测量继续按原记录保留，不能归入本次提交。

MK-Curve 原有 `num_workers > 1` 路径已使用进程池及有界的待处理块，但主进程按块索引等待 `result()`，再补充下一块；慢的前序块可能推迟后续任务补充。按完成事件补充任务、按原始体素索引写回的动态分发仍是独立候选。当前标准 DKI/MK-Curve 路径没有传入多 worker 参数，使用单 worker，并将 MK-Curve 块大小限制为最多 64 个体素。

当前也没有自动寻找最优 CPU 线程数。初始 CPU 预约采用经验规则或用户覆盖；历史反馈更新 RAM 高水位与时长估计，不搜索最优 worker、线程数或 chunk size。当前基准中的每任务 4 个 CPU 是固定对照条件，不代表算法最优设置，更不等于 4 个 MK-Curve worker。

## 是否引入全局 DAG

本轮保留现有受试者依赖编排，将微结构模型下沉为资源预约单位，并分开管理活跃受试者窗口和模型 worker 上限。这样先解决长阶段占用及嵌套并发，避免同时维护第二套依赖、缓存与失败状态。

目前不是任意任务可同时就绪的全局 DAG：受试者内其余阶段仍按已有编排暴露，追踪变体等仍可能顺序执行。若完整队列的等待时间证明其限制关键路径，再将这些已明确依赖的计算节点抽取到全局 DAG。没有测量依据时不重写整个执行器。

历史反馈用于后续批次的估计，不声称自动寻找算法的最优线程数。线程效率曲线、按设备 VRAM 准入、统一跨阶段 I/O 预算及共享输入 mmap 仍需独立验证。

下一阶段单独测试运行配置，不与当前调度主比较混在一起：

| 模块 | 待测试参数 | 主要约束 |
| --- | --- | --- |
| MK-Curve | Worker 数、每 worker 数值线程数、voxel chunk size；另比较原顺序等待与完成驱动补充 | 固定求解器、synthetic-b0 网格及信号修复策略；按原始体素索引写回 |
| DKI | 数值线程数、现有 voxel chunk size | 当前没有多 worker 拟合参数；新增进程并行须另行实现和验收 |
| WMA | 现有线程数、同预算下的被试并发度；仅在热点成立时测试对应批处理粒度 | 没有与 MK-Curve 通用的 voxel chunk 参数；固定采样、优化器预算及分类阈值 |

先用有限候选筛选，避免对 worker、线程数和 chunk size 做昂贵的全排列；入选配置再做完整受试者与固定总资源预算的队列配对。分别报告单受试者延迟、整批吞吐、峰值内存及等待时间，不以单任务最快配置代替队列吞吐最优配置。比较保留原始参数、曲线、有效掩膜、流线顺序及束产物的验收要求。当前后台队列不修改这些参数，以优先完成已有主对照。

## 真实队列设计

入口为 `scripts/benchmark_cohort_workflows.py`。选择 12 名不同 HCP 参与者的完整多壳层 DWI，保留完整网格与采集体积；核对梯度、掩膜几何及输入 SHA256。外部已预处理输入不重复去噪或畸变校正。

主比较固定 32 个逻辑 CPU、96 GiB 整树内存监测上限、每任务 4 个 CPU 位置、1 个 GPU 型任务许可。Subject-level、Stage-level、Adaptive 的活跃受试者上限均为 4；Sequential 为 1。Subject-level 通过整受试者预约保持同样的 CPU 位置约束，Stage-level 和 Adaptive 分阶段释放。

各策略独立创建输入工作目录与输出目录，避免输入旁的原生 DTI 缓存跨组复用；只读原始 DWI 内容相同。冻结运行代码与哈希，分开保存环境、原始时间序列、批次与被试状态、独立 QC 及产物指纹。

| 路线 | 计算内容 | 当前验收条件 |
| --- | --- | --- |
| DKI | 原生 DTI、标量导出、完整 WLS DKI | 四种策略全部结束后，逐被试比较原始参数、图像、affine、dtype、无效掩膜及自动 QC |
| MK-Curve | 完整 WLS 准备及 MK-Curve | 独立路线计时，不把 MK-Curve 与 DKI 当作同一拟合方法比较 |
| Tracts | DDSeg、DDParcel、CSD/ACT、SIFT2、UKF、WMA | 同一流线预算及固定 MRtrix 随机种子；比较流线顺序、拓扑、属性、权重和束产物 |

预检中，DKI 路线通过。追踪路线最初未发现 UKF，且 PyTorch 设备检查超时；指定已有 UKF 组件并重新检查后，所选路线无阻塞。两次早期试启动分别发现输入旁缓存共享和原生程序线程变量无效，均保留为取消/失败记录，不纳入性能统计。

本轮完整队列实验尚未完成。首轮完整比较为每策略一次运行，不能据此给出稳定加速区间；还需轮换重复、线程数、工作负载组合与资源预算消融。旧的三 ROI pilot 继续作为独立的小规模证据保留。

## 本机运行与验收记录

- 软件全套回归：1,815 项，9 项跳过，其余通过，包含 SIFT2 权重比较回归。原始日志位于 `/data/chengjin/SlicerDDSurfer-test/cohort-refinement-unittest-20261103-final5.log`。
- 论文回归：180 项通过；中英正文与补充 PDF 均已重新编译至 `/data/chengjin/AAA-Paper/SlicerDSpace`。论文依赖审查通过：322 个文件、40 项嵌入素材。
- 正式首轮 DKI 记录：`/data/chengjin/SlicerDDSurfer-test/cohort-full-20261103-dki-v3/campaign.json`。已实测确认原生 `DWIToDTIEstimation` 进程继承四个 CPU 位置；完整批次尚在运行。
- 后续 MK-Curve 与追踪路线的冻结记录：`/data/chengjin/SlicerDDSurfer-test/cohort-full-20261103-long-models/campaign.json`，共八个策略/路线组合。
- 持续运行队列：`/data/chengjin/SlicerDDSurfer-test/cohort-full-20261103-queue.json`。前一实验结束后才启动后续路线，不同时叠加两套计算预算；执行失败时停止并保留原始日志。
- 队列只在完整 QC 与输出比较通过后调用论文绘图器；不会自动改写论文中的数值或将未完成结果替换到网站。

SIFT2 比较忽略包含不同输出路径的命令历史注释，严格比较数值权重及其顺序。此规则在追踪实验执行前更新并登记哈希；正在运行的 DKI 快照不被修改。

## 绘图与写作门禁

论文仓库的 `scripts/figures/build_cohort_campaign.py` 只接受至少 10 名完整体积参与者、四种策略均完成且 QC/一致性通过的数据。导出 PNG、PDF、可编辑 SVG，涵盖墙钟时间、QC 吞吐量、峰值 RSS、CPU 容量使用及任务/资源时间轴。

一个资源预算不能生成吞吐量—预算关系曲线；未完成或失败的数据不生成论文对比图。新的完整队列结果通过后再替换补充材料数值和图，网站也不提前新增加速率。

## 15:55 优先级调整

为尽早得到有解释力的主比较，保留已完成的 12 人 Sequential 基线及正在运行的 Subject-level，DKI 下一组直接执行 Adaptive；Stage-level 留作后续消融。MK-Curve 与追踪路线只先做 Subject-level / Adaptive，暂缓各自的 Sequential 与 Stage-level。尚未开始的组合由 10 组减少到 5 组。

新顺序为 DKI 主对照 → 包含 CSD/ACT、UKF 与 WMA 的混合路线主对照 → MK-Curve 主对照。混合路线更适合考察不同资源需求的阶段交错；单一 DKI 路线的调度收益可能较小。所有路线仍为相同 12 人完整体积、相同算法与导出、相同 32 CPU / 96 GiB 预算、相同最多 4 名活跃被试及每任务 4 CPU，不用缩小数据或增加并发数量换取更早结果。

新的控制记录为 `/data/chengjin/SlicerDDSurfer-test/cohort-priority-20261103/priority.json`。旧驱动不支持运行中改序，因此在未启动的 Stage-level 输出位置设置显式边界保留目录。它在当前 Subject-level 完成、QC 与指纹写入后停止旧驱动，随后启动独立的优先队列。旧日志中预期的 `FileExistsError` 是该计划交接，不是科学计算失败；当前运行不中断，原始算法快照与既有测量不改写，旧目录全部保留。

优先队列使用单独冻结的控制工具与绘图器。已完成证据仅被引用并核对哈希；新比较保存到新目录。主比较参考改为 Subject-level，从而区分调度机制增益与单纯增加被试并发的增益。每条路线完成并通过 QC 和精确产物比较后立即导出原始主比较 JSON 与 PNG/PDF/SVG，不再等待其余长路线全部结束。未运行的策略不在图中补点或填零。

本次调整后的回归为软件 1,821 项（9 项跳过，其余通过）、论文 181 项全部通过；52 项调度定向测试覆盖已完成证据复用、拒绝迁移活跃任务、边界交接及缺少参与者时拒绝生成加速率。控制工具、绘图器与当前运行的科学快照均通过哈希复核。新原始日志为 `/data/chengjin/SlicerDDSurfer-test/cohort-priority-tests-20261103-final.log` 及 `cohort-priority-paper-tests-20261103.log`。

## Nipype 对照与本轮候选

本轮核查 Nipype 最新稳定版 1.11.0 及主分支，保留官方源码 URL、文件和选定方法的 SHA256，见 `benchmarks/cohort_upstreams_20261103.json`；可用 `scripts/audit_cohort_upstreams.py` 重新核查。稳定版与本次主分支的 MultiProc 文件相同。以下采用机制和标准库实现，不新增 Nipype、Pydra 或 Slurm 运行依赖。

| 已核查来源 | 本次采用方式 | 边界 |
| --- | --- | --- |
| [Nipype MultiProc 1.11.0](https://github.com/nipy/nipype/blob/1.11.0/nipype/pipeline/plugins/multiproc.py) | 启动前检查资源声明；可回填候选按 `(memory_gb, cpu)` 升序稳定排列 | 队首预约、120 秒与八次越过保护优先；不是把整个队列按小任务无限插队 |
| [Python graphlib](https://docs.python.org/3.12/library/graphlib.html) | 模型执行器直接复用 `TopologicalSorter`，启动前检查缺失依赖及循环 | 仅管理一次模型调用内的计算；未重写为任意节点的全局队列 DAG |
| [Python futures](https://docs.python.org/3.12/library/concurrent.futures.html) 与 [Pydra 执行模型](https://nipype.github.io/pydra/tutorial/2-advanced-execution.html) | 标准线程池等待独立科学子进程完成，使用 `FIRST_COMPLETED` 持续补充就绪任务 | 不将不同被试信号合批，不修改 MK-Curve 的 voxel chunk 分发；科学计算仍在独立进程 |
| [Slurm 预约回填原则](https://slurm.schedmd.com/sched_config.html) | 保留已有受保护回填与未知/超时任务的保守处理 | 不用运行时估计提前释放仍在运行的资源 |

具体改动还包括：每次阶段准入重读匹配的成功画像，使同批前序被试的实测峰值可用于后续任务；显式固定 RAM 预约不被自动覆盖。画像实现签名覆盖调度与主要拟合内核，避免更改实现后误用旧记录。非法的非整数 CPU、非有限 RAM/时长及超预算申请在启动前报错；未选模型的合法高内存默认值不阻塞本次路线。队列等待者核查协调器 PID 与创建时间，协调器退出后停止等待；子进程失败或取消时释放预约，不发布成功的聚合状态。

候选冻结在 `/data/chengjin/SlicerDDSurfer-test/cohort-mature-20261103/candidate_runtime/`。与原科学快照相比，266 项科学文件逐字节相同；两处共享缓存命中标记仅改变 `mark_reused(...)` 的控制记录参数，经 AST 归一化核查，未改变数值代码。完整清单与哈希见同目录的 `candidate_provenance.json`。冻结工具为 `scripts/benchmark_cohort_candidate.py`。

旧 DKI Adaptive 测量继续使用原快照；新队列只接管监督进程，不向科学驱动或进程组发送信号。旧组结束后，新候选按 DKI、混合追踪、MK-Curve 顺序运行。DKI 引用已经完成且通过 QC/一致性检查的 Subject-level 基线；新 Adaptive 单独计时，避免重复耗时基线。控制记录为 `/data/chengjin/SlicerDDSurfer-test/cohort-mature-20261103/queue.json`，每条路线独立验收并出图。

本轮软件回归为 **1,831 项，9 项跳过，其余通过**；日志为 `/data/chengjin/SlicerDDSurfer-test/cohort-mature-tests-20261103-final.log`。新完整队列结果尚未完成，本节不提供新候选的加速率。仍未自动搜索最优线程/worker/chunk 配置，也未新增按设备显存硬限额。

## 最新候选：运行配置搜索与体素完成驱动

上述“未实现”描述保留为前一候选的历史状态。本轮新增 `scripts/autotune_batch.py`、
`shared/compute_tuning.py` 及 `microstructure/chunk_dispatch.py`，不改拟合核、网格或求解器。
MK-Curve 的多 worker 路径使用标准 futures 完成事件补充工作，最多保留 worker 数量的待处理块；
结果按原始 Fortran 体素索引写回曲线、原始/校正系数、各参数图及有效掩膜。
`ordered` 控制路径保留，单 worker 默认不新增进程池。每个 worker 默认一个数值线程，防止嵌套并发。

可选 CPU 校准对 DKI 搜索 1/2/4/8 以内的有界线程候选；MK-Curve 额外搜索 worker 与
32/64/128 体素块，受每候选 CPU 上限限制。先预热，再轮换重复三次；候选须与基线的
全部采样产物精确一致。低于 5% 的估计收益保留基线。选择区分单任务延迟和整批容量估计，
同时考虑完整输入的 RAM 下限、总 CPU 预算及活跃被试数。记录保留原始时间、RSS、启动耗时、
逐产物指纹、输入来源及排除情况；不把采样时间外推为完整模型分钟数。

校准结果匹配硬件、软件实现、数据维度及方法，按实际所选任务应用；显式线程/worker/块参数
和阶段预约优先。Batch 目标解析直接复用单被试 CLI，覆盖算法子集、重复目标与跳过目标，
不为未选算法保留执行槽或启动校准。已预处理 NIfTI 可以直接校准；需要预处理或格式转换的
输入先保留声明配置。WMA 可对已有流线进行完整模块线程校准；没有所需输入时明确延期，不
额外运行上游追踪，也不声称已经找出 WMA 最优线程数。

新的完整队列对比将固定配置的 Subject-level、固定配置的最新 Adaptive、以及完成校准后的
最新 Adaptive 分开：前一对比考察调度，后一对比考察同一调度器下的运行配置选择。总 CPU、
RAM、被试集合与活跃被试上限保持一致。旧候选与已完成证据保留，不把不同实现的耗时拼成
新版本指标。入口为 `scripts/benchmark_cohort_latest.py`；在完整 QC 及科学产物比较通过后出图。

实现参考：[Python futures](https://docs.python.org/3.12/library/concurrent.futures.html)、
[threadpoolctl](https://github.com/joblib/threadpoolctl)、
[Nipype 节点资源声明](https://nipype.readthedocs.io/en/latest/api/generated/nipype.pipeline.plugins.multiproc.html)
及 [Snakemake 目标与资源规则](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html)。

最新验收：软件全套 **1,853 项，9 项跳过，其余通过**（`cohort-latest-tests-20261103-final.log`）；
Slicer Python 内的模型、调参及队列定向回归 **47 项全部通过**（`cohort-latest-slicer-tests-20261103.log`）。
同一次验收还通过 88 份活跃文档检查、扩展元数据与 37 份 Skill Shell 模板检查。
日志均位于 `/data/chengjin/SlicerDDSurfer-test/`。旧候选的待启动监督器已停止，科学驱动未中断。
最新冻结版本及接续队列位于 `/data/chengjin/SlicerDDSurfer-test/cohort-latest-20261103-r2/`；
`queue.json` 记录控制状态，先完成现有 DKI 历史组，再进行新校准和新版本完整对照。
当前尚无这一最新候选的完整队列加速百分比。
