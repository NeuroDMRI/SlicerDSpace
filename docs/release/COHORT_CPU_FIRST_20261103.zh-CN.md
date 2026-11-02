<div align="center">

# CPU 优先的队列调度

[使用说明](../setup/COHORT_RESOURCES.zh-CN.md) | [原始测量](benchmarks/cohort_cpu_20261103.json)

</div>

---

## 已实现

`--scheduler adaptive --resource-budget 80%` 启用新的准入策略；
`--subject-limit N` 可限制此次处理人数，省略时处理所有合格输入。
`--jobs` 单独限制活跃参与者进程，省略时自动选择，最多八个。
原 `subjects` 与 `resources` 模式保留。

| 内容 | 实现 |
| --- | --- |
| 依赖 | 沿用单被试编排器的前置检查、缓存及产物凭据；每名参与者一次准入一个封装阶段 |
| CPU/RAM | 自适应默认预算为可用资源的 80%；显式预算优先；一般阶段初始最多四线程，默认 WMA/WMQL/连接组为一线程 |
| 回填 | 有预计时长时不推迟队首的预计启动；缺少时长时检查与队首联合预约是否兼容 |
| 等待保护 | 等待 120 秒或被越过八次后，暂停继续越过，等待在运行阶段释放资源；不抢占科学计算 |
| I/O | 配准、预处理、WMQL、连接组共享两个粗粒度准入槽；不让整个长时间 DKI/WMA/CSD 计算持续占有 I/O 槽 |
| 资源画像 | 每 0.2 秒采样阶段进程树 RSS、CPU 时间和 I/O，记录实际执行时长；批记录分开汇总预约与测量 |
| 历史 | 按硬件、Python/包版本、实现、输入维度、参数和线程建立索引；每键最多十二条，共最多 512 键 |
| 反馈 | 成功记录的 RSS 高水位加 30% 可提高原始 RAM 估计；时长加 25% 用于预约估计；失败记录不参与反馈 |
| 恢复 | 异常与退出进程释放预约；保留各被试输出和 QC；不自动重试数值失败 |

资源历史默认保存在批次输出的 `00_batch_metadata/resource_history.json`；
`--resource-history` 可指定路径。历史不会自动降低 RAM 初值。采样 RSS 可能重复计算共享页，
也可能漏掉采样间峰值。预约覆盖计算边界，环境检查、导入和报告生成在边界之外。

## 实测 Pilot

三个真实 dMRI ROI，各 12×12×12 体素、288 次测量。使用相同 DIPY WLS 张量拟合、
相同两个 CPU 线程及 4 GiB 总预约预算，每次拟合一个数值线程，三次重复轮换策略顺序。
每次使用新的输出目录，未清空 OS 缓存。

| 策略 | 中位墙钟时间 |
| --- | ---: |
| 逐受试者执行 | 3.27 s |
| First-fit 阶段准入 | 2.25 s |
| 新自适应准入 | 1.80 s |

全部十三个张量参数、FA、MD、affine、掩膜和自动 QC 输出的字节哈希一致。
原始记录同时保留父进程及子进程的 RSS 采样，新活跃窗口使用更多解释器内存。
这项测试验证 ROI 任务的调度与数值一致性，不作为全脑、完整多方法流程或大队列加速率。

复现：`scripts/benchmark_cohort_cpu.py --inputs <ROI输入目录...> --output <新目录> --repeats 3`。
输入目录包含 `data.nii.gz`、`bvals` 与 `bvecs`。

## 后续测量

完整 DKI、MK-Curve、WMA 的一、二、四、八线程效率曲线尚未在本轮建立。
当前线程初值不声称最优。十至二十名真实被试的完整路线配对、长期内存峰值、
存储竞争及设备级显存准入仍需专项验收。当前实现不做同一参与者内部的全局 DAG 任意重排，
也不引入持久 GPU 模型服务或跨参与者张量合批。

## 设计来源

[Snakemake 资源说明](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#resources)、
[Slurm 回填说明](https://slurm.schedmd.com/sched_config.html)、
[Nextflow 动态资源说明](https://docs.seqera.io/nextflow/process#dynamic-task-resources)
提供工作流资源预约与动态请求的工程参考。相关方法论文为
[Köster 与 Rahmann，2012](https://doi.org/10.1093/bioinformatics/bts480)、
[Mu'alem 与 Feitelson，2001](https://doi.org/10.1109/71.932708)、
[Di Tommaso 等，2017](https://doi.org/10.1038/nbt.3820)。
插件不要求安装这些框架，也不把后续动态资源特性归因于 2017 年论文的全部原始实现。
