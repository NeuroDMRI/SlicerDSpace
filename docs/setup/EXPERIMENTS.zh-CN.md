<div align="center">

# 轻量依赖与批量实验

[首页](../../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [English](EXPERIMENTS.md)

</div>

---

## 减少外部环境

优先复用 Slicer 自带 Python 与已安装扩展，兼容 Python 包按需安装，
不需要为每种算法另外建 Conda 环境。已有的有效脑掩膜优先复用。

在 **环境与安装 > 使用 Slicer 原生轻量路线...** 确认后切换当前配置：

| 部分 | 方法 | 额外独立环境 |
| --- | --- | --- |
| 脑掩膜 | mean b0 上的 DIPY median-Otsu | 无外部程序或权重 |
| 可选预处理 | DIPY | 不需要 |
| DTI / FW / NODDI / Python DKI | Slicer Python 中所选模型 | 不需要；按需安装 Python 包 |
| 解剖重建 | DDSeg / DDParcel / DDSurfer | PyTorch 扩展、Python 包及下载资源 |
| 追踪 | DTI deterministic | 兼容 Python 包（张量拟合在 Slicer 内完成） |
| 束分析 | WMQL | Python 实现及查询文件 |
| 连接组 | Python | 兼容 Python 包 |
| 可选表面后处理 | 原生组件运行包 | 小型授权组件，无需完整版 FreeSurfer |

默认 PNL 掩膜和 CSD/ACT 追踪不自动改变。PNL 仍需 PyTorch/ANTs；
CSD/ACT 和 SAF 仍需 MRtrix；MRtrix/FSL 预处理仍需 FSL；UKF 调用安装的
UKFTractography 扩展。保留这些算法的按需入口，不将其他方法伪装为等价替代。
轻量预设不修改 GPU 算法的设备偏好。

DIPY median-Otsu 是 CPU 算法，采用半径 4、中值滤波 4 次、膨胀 1 次、不裁剪。
原生网格、算法版本、参数、输入指纹与 QC 均保留。需检查脑边界、脑干和小脑；
它不等同于 PNL 模型。见 [DIPY 官方说明](https://docs.dipy.org/stable/reference/dipy.segment.html#dipy.segment.mask.median_otsu)
与 [Slicer Python 文档](https://slicer.readthedocs.io/en/latest/developer_guide/python_faq.html)。

---

## GUI 实验配置

默认仍运行当前单个被试，批量实验需主动启用：

1. 主界面选择产品和算法。
2. 打开 **高级设置 > 实验与批处理**，勾选 **启用实验运行器**。
3. 选择“当前被试”，或“已准备好的被试文件夹”并填写批处理根目录。
   每个子目录对应一个已导入被试；先完成 DICOM/NRRD 导入。
4. 并发被试数建议从 1 开始。同一被试的参数组合依次执行，不同被试可并发，
   共享 CPU、内存及 GPU 显存。
5. 填写可选 CLI 参数覆盖与 JSON 命名组合。优先级为 GUI 配置、公共参数覆盖、
   当前组合参数，后者优先。
6. **预览实验计划**后，点击 **Run SlicerDSpace** 确认任务数和输出路径。
   预览检查参数和路径，每个任务执行前独立检查最终配置所需依赖。

每个实验最多 64 个命名组合、4,096 个任务；更大的研究可拆分提交。
GUI 预览显示前 50 项，导出配置及 CLI 计划保留全部任务。

选择 Microstructure 后，可对比两种 DTI 拟合：

```json
[
  {"name": "dti_wls", "args": ["--micro-fit-method", "wls"]},
  {"name": "dti_ls", "args": ["--micro-fit-method", "ls"]}
]
```

选择 Fiber bundles 后，可保留当前方案并增加 UKF 来源的 WMA 实验：

```json
[
  {"name": "baseline", "args": []},
  {"name": "ukf_wma", "args": [
    "--tractography-backend", "ukf",
    "--bundle-backend", "wma",
    "--bundle-tractography-backend", "ukf"
  ]}
]
```

后一示例需准备 UKF/WMA 依赖，相关科学 QC 不变。公共参数可填
`--mrtrix-seed 42 --mrtrix-nthreads 8`；`--ukf-parameters` 接受 CLI 同款
校验 JSON。`--max-parallel-models` 控制单被试内部模型并发，与并发被试数不同，
叠加时需评估总内存和显存。

环境路径仍在“环境与安装”配置。实验参数不接受 shell 命令、安装操作、被试身份、
输出路径或跳过依赖检查的选项。参数及算法组合使用 CLI 同一套校验函数。

---

## 输出与记录

```text
<output>/experiments/<campaign>/
  00_batch_metadata/
    experiments.json
    logs/<job>.stdout.log
    logs/<job>.stderr.log
  subject_0001/
    dti_wls/<正常产品目录>
    dti_ls/<正常产品目录>
  subject_0002/...
```

配置 JSON 保存在实验目录旁，总记录持续保存最终配置、命令数组、状态、耗时、
日志及每个任务的运行记录路径。输入、工具及条件满足的被试缓存可复用，
不同组合的输出彼此隔离，不覆盖既有实验。取消停止本次后台进程树并保留已完成记录；
失败及未启动任务仍可追溯。

**Results and QC** 中可查看整个实验或单个运行。完成时只自动显示一个已完成运行，
其他组合通过运行列表查看，避免全部加载到场景。自动 QC 与人工审核分别记录。

**导出配置...** 后可通过同一运行器执行：

```bash
/path/to/Slicer --launch PythonSlicer scripts/run_experiments.py \
  --request /path/to/request.json
```

不带 `--execute` 时仅验证并输出计划，不创建结果；加上后才执行。再次运行时
生成新的实验输出路径，安装依赖仍使用已有的显式环境准备工具。
