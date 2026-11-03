<div align="center">

# 资源感知队列处理

[文档](../README.zh-CN.md) | [CLI 模板](CLI_TEMPLATES.zh-CN.md) | [English](COHORT_RESOURCES.md)

</div>

---

## 使用方式

CPU 密集型队列可使用新的自适应准入模式：

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/study --output-root /path/to/results \
  --scheduler adaptive --resource-budget 80% --subject-limit 20 \
  --targets microstructure --microstructure-stages dti,noddi,dki --execute
```

省略 `--subject-limit` 即处理所有合格输入；省略 `--jobs` 自动确定活跃被试窗口。
窗口由受试者总数、CPU 预算及等待解释器的 RAM 余量推导，不再固定限制为 4 或 8；计算并发由阶段预约决定，不再按最重阶段
限制整个活跃窗口。默认使用可用 CPU/RAM 的 80%，显式预算可以覆盖；这是预约策略，
不是操作系统强制使用上限。一般阶段默认最多预约 4 线程，默认 DTI 导出、串行自由水、DKI、MK-Curve、WMA、WMQL 与连接组
路线为 1 线程，保留用户的明确覆盖。这些是保守初值，不代表实测最优线程数。

被试内仍由现有编排器保证依赖；微结构模型分别取得资源预约。自适应模式可用
`--max-parallel-models 2` 或 `3` 启用有界独立模型 worker，默认仍为一个模型。
有历史时长时，回填不得推迟队首的预计启动时间；队首等待达到 120 秒或被越过 8 次后，
暂停后续回填，等待在运行阶段释放足够资源。不会抢占长任务。时长估计可能超时，
等待保护仍会阻止持续插队。配准、预处理、WMQL 与连接组阶段另共享 2 个 I/O 准入槽，覆盖整个阶段，
并非实时磁盘带宽限额。DWI 解压另设两个 FIFO 许可，拟合前即释放。
超时的时长估计不会释放仍在运行的任务预约。

`schedule_<id>/events.jsonl` 保存阶段进程树 RSS、CPU 时间、I/O 及执行时间采样；
批记录分开汇总预约与测量。`00_batch_metadata/resource_history.json` 每个工作负载
最多保留 12 条记录、共 512 个键，`--resource-history` 可另选路径。键包含硬件、
Python、实现哈希、数值参数、输入维度及线程数。成功且未命中缓存的记录可用 30% 安全余量提高 RAM
估计，以 25% 余量估计时长；不会自动降低原始 RAM 估计，也不会自动重试失败的科学计算。
RSS 求和可能重复计算共享页，也可能漏掉采样间峰值。

`docs/release/benchmarks/cohort_cpu_20261103.json` 是三份真实 ROI 的 WLS 张量拟合
配对 pilot，验证数值一致性及调度开销，不代表全脑或整条队列工作流加速。

## 兼容模式

默认仍按被试滚动并发。新增 `--scheduler resources` 在计算阶段边界共享资源预约，
保留各被试自身的依赖、输出、运行记录、缓存与 QC。

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/study --output-root /path/to/results \
  --scheduler resources --jobs 3 \
  --targets segmentation,parcellation,surface,microstructure \
  --microstructure-stages dti,freewater,noddi --execute
```

使用已准备的被试目录与正常依赖配置。省略 `--jobs` 时，自动选择保守的被试并发数，
不设固定人数上限。初始计划参考 CPU 亲和性与容器限制、可用 RAM、所选方法和 DWI 头信息，
不加载整套 DWI。可在批次记录的 `resource_plan` 查看计划，并用试运行的峰值复核。
不需要新增服务、容器或推理框架；工作流 Python 需要 `psutil` 检查进程身份并回收失效预约。

## 调度规则

- `--jobs` 可选，限制活跃被试进程数，包括等待资源的进程；资源模式省略时自动规划，原逐被试模式省略时仍为一。
- 计算阶段的 CPU、RAM 预约总和不得超过批次预算；RAM 单位为 GiB。
- 默认估计随方法和各被试影像维度变化；`--batch-stage-cpus` 和 `--batch-stage-memory-gb` 可将其覆盖为统一预约值。
- GPU 型阶段共享 `--batch-gpu-slots`，默认一个；CPU 阶段可以与其他被试的 GPU 阶段重叠。阶段暴露设备设置且明确为 CPU 时，不占 GPU 槽位。
- 采用先满足条件先准入的队列；等待 GPU 的请求不阻塞有资源可用的 CPU 请求。被试自身的依赖仍决定阶段就绪顺序。
- 嵌套调用和所属子进程复用父阶段预约；通过 PID 和进程创建时间识别失效预约，异常会释放预约。

覆盖掩膜生成、共享原生及配准 DTI、预处理、解剖推理与后处理、微结构、追踪、
纤维束提取及连接组计算。环境检查、输入校验和报告不单独预约资源。
DTI 导出、自由水、NODDI 及 DKI/MK-Curve 分别获取预约，控制器不持有父级预约。
这是任务级准入，不进行逐体素或迭代级合批。

## 线程与设备

高级用户仍可覆盖 `--batch-cpus`、`--batch-memory-gb`、`--batch-stage-cpus`、
`--batch-stage-memory-gb` 和 `--batch-gpu-slots`。
内存估计参考 float64 信号大小及各方法的工作空间。无法读取头信息时，采用并记录
160 x 192 x 160 x 128 的回退尺寸；正常输入校验仍会执行。
若所选单个阶段的估计需求已超过预算，启动前明确报错，不缩小预约或更换分析方法。
显式算法线程数在总预算允许时保留。估计为保守启发式，并非实测峰值。

为子进程设置通用数值库、ITK、MRtrix 的线程默认值，提供 surface/MRtrix/eddy
默认线程数，限制自动 NODDI 线程数。显式线程参数超过预约时拒绝启动。
UKF 自动线程数采用阶段预约。兼容 `resources` 模式要求被试内部模型并发为一，
`adaptive` 也支持有界独立模型 worker。自由水并行 worker 默认一个。改变线程设置时，
应按方法需要重新建立可重复性基线。

Linux 阶段获得互不重叠的 CPU 亲和性集合，原生子程序也继承该范围；完成或异常退出后恢复。
RAM 仍是声明预约，不是操作系统强制上限。第三方工具可以在其 CPU 范围内创建辅助线程。
GPU 槽位统计具有 GPU 能力的调用，不测量真实利用率或显存；当前不分配 GPU、
不限制单卡显存、不常驻模型，也不将多被试合成一个推理批次。
精度、随机种子、求解定义及拟合网格不作改变，设备仍通过既有算法参数选择。

## 记录与验证

批次记录的 `resource_schedule` 指向包含逐被试阶段估计和规划输入的 `config.json`、实时 `state.json` 与
追加式 `events.jsonl`，保存被试、阶段、等待时间和执行事件。
阶段完成表示调用返回，科学验收以各被试 QC 为准。默认失败后停止启动新被试，
已活跃的被试继续完成；原有续跑与 `--continue-on-error` 语义保留。

`Testing/Python/test_cohort_schedule.py` 验证真实子进程调度、重叠、预算、嵌套调用、
异常回收与合成输出一致性。`scripts/benchmark_cohort_schedule.py --output /path/to/new_directory`
在相同声明预算下，交替顺序重复三次比较整被试预约与阶段预约。
这些等待型合成负载只验证调度，不代表 MRI 工作流加速；真实队列的墙钟、资源峰值、
产物一致性和 QC 仍需配对测量。
