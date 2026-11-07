<div align="center">

# GPU 执行策略

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [脑掩膜](brain_mask_gui_config.md) | [English](GPU_EXECUTION.md)

</div>

---

## 默认原则

支持 GPU 的神经网络推理默认 `auto`，优先使用**实际执行推理的 Python 环境**
中可见的第一个 CUDA GPU。环境没有可用 CUDA，或用户明确选择 CPU 时才用 CPU。
设备探测失败、CUDA 推理报错不能冒充“没有 GPU”并静默重跑 CPU。

尊重用户、容器和调度器的 GPU 可见性，不擅自取消限制、不同时占满所有 GPU，
不在分析过程中重新安装 CUDA。机器有 GPU 但框架只有 CPU 版本时，必须先配置
对应后端的 CUDA 环境；硬件存在不等于当前 Python 支持 GPU。

---

## 各环节

| 环节 | 实际策略 |
| --- | --- |
| PNL CNN 掩膜 | 外部 PyTorch `auto`；优先调用方可见的首个 CUDA GPU，否则 CPU，可显式选择 |
| DDSeg | GUI、CLI、直接 API 默认 `auto`；在执行推理的环境中判断设备 |
| DDParcel | 默认 `auto`；继承 GUI 的 PyTorch 选择，旧版独立入口也转发选择；显式 CUDA 不静默变 CPU |
| DDSurfer 表面 | 默认 `auto`；用实际外部 Python 探测，保留显存与精度检查 |
| TractCloud | 用其实际 Python 判断；继承 GUI 选择，CUDA 出错不自动重跑 CPU |
| MRtrix/FSL eddy | CUDA 版和 GPU 可用时优先使用；启动器禁止 CUDA 失败后自动重跑 CPU |
| DIPY MP-PCA 去噪、Gibbs 去除 | 当前 CPU 实现 |
| Slicer DTI、DIPY FW/DKI、Python MK-Curve | 当前 CPU 实现 |
| AMICO NODDI | 当前 CPU 拟合后端，线程设置与 GPU 是两回事 |
| ANTs/BRAINSFit、FreeSurfer 表面后处理 | 当前 CPU 配准与几何后端 |
| MRtrix CSD/ACT/SIFT2、WMQL/WMA、SAF | 当前 CPU 后端与数值筛选 |
| 连接组、区域统计、文件转换 | 当前 CPU 数值和 I/O 操作 |
| Slicer 渲染 | Slicer 的图形上下文，独立于推理设备 |

加载权重时的 CPU 校验、张量转 NumPy/文件不等于 CPU 推理，不能一看到 `.cpu()`
就删掉。没有 GPU 实现的算法也不能仅修改设备字符串或换算法来宣称加速。

---

## 高级设置与记录

“高级设置 > 预处理与掩膜”中选择掩膜的 `auto/cuda/cpu`；“高级设置 > 运行与性能”
中选择 DDSeg、DDParcel、DDSurfer、TractCloud 共用的 PyTorch 设备。选择持久化，
不会覆盖用户已保存的明确 CPU 偏好。CLI 对应 `--brain-mask-device`、
`--ddseg-device`、`--surface-device`、`--tractcloud-device`。

掩膜 manifest 记录 `device_execution`，表面 manifest 记录 `resolved_device`，
DDSeg QC 记录执行设备，TractCloud 保存独立设备记录并写入 manifest。

PNL 的 `DEVICE_ID_LIST` 未定义分支由启动器
在内存中修正，不改外部源文件、模型权重、科学计算、配准或后处理逻辑。
DDSurfer 的固定上游快照保持不变；插件预先用 PNL CNN 生成并明确传入原生掩膜，避开上游
独立辅助脚本的 CPU 默认值。

---

## 环境边界

[PNLBrainMask-Torch](https://github.com/ChengjinLii/PNLBrainMask-Torch)
使用转换后的 PNL 权重与 PyTorch 推理；受管安装默认使用已做配对测试的 CPU 构建，显式 CUDA 路线需要兼容的 PyTorch/CUDA。
GPU 可用不代表每次端到端运行更快。

官方 CUDA eddy 2602.0 已装到本机独立 FSL 环境，完成真实 31-volume 数据的预处理验收，
原 CPU 环境保持不变。[插件内自动配置](MANAGED_SETUP.zh-CN.md)现可按需安装该后端并自动保存路径。
实测发现 CUDA eddy 只接受一个 CPU 辅助线程，启动器会自动设为 `--nthr=1`；
CPU eddy 仍保留用户的线程数，同时记录请求参数与实际参数。
启动器已阻止 MRtrix 上游 `dwifslpreproc` 的 GPU 错误后 CPU 重试。没有 CUDA eddy
后端时允许 CPU 路线；这是运行环境能力不足，不是强行禁用 GPU。其选择独立于
网络设备选项；若要主动限制 eddy 使用 GPU，可在启动环境设置 `CUDA_VISIBLE_DEVICES=-1`。
实际证据和未完成项见[核查记录](../release/GPU_POLICY_20261013.md)。
