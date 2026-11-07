<div align="center">

# 能力与平台

**只安装当前路线需要的依赖**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [发布](../release/EXTENSION_READINESS.zh-CN.md) | [English](SUPPORT_MATRIX.md)

</div>

能力检查检测包、程序和资源是否可用，不认证每种采集、每个平台的数值正确性。

---

## 依赖层级

| 层级 | 产品 | 前提 |
| --- | --- | --- |
| 检查与计划 | 导入预览、输入检查、计划 | 基础影像环境；格式转换需要对应工具 |
| 最小 DTI | 张量参数和彩色 FA 图、可选确定性流线 | Slicer 自带的 Python（Teem 张量拟合，无需扩展）及 Python 包 |
| 解剖模型 | DDSurfer/DDSeg/DDParcel | PyTorch、匹配资源及对应准备过程 |
| 微结构 | DTI/FW/iFW/NODDI/DKI/MK-Curve | 对应数值包和适用采集；默认 DTI |
| 高级通路 | CSD/ACT/SIFT2/SAF/白质束 | 仅配置所选外部后端 |
| 连接组 | 区域矩阵与场景 QC | 流线/分区；加权路线才需权重 |

不强制一次安装所有依赖，资源授权与环境可用性分别检查。

---

## 平台证据

| 检查 | Linux | Windows | macOS |
| --- | --- | --- | --- |
| 资源/CMake 合同 | 有本地证据；已配置 CI | 已配置 CI，仍需运行证据 | 已配置 CI，仍需运行证据 |
| 真实 Slicer GUI/源码安装 | 有带日期记录 | 未认证 | 未认证 |
| 真实 SDK/CTest/CPack | 2026-10-22 候选版本已在 Linux/Slicer 5.10 验证 | 待真实 SDK 验收 | 待真实 SDK 验收 |
| 科学路线 | 仅报告所列路线/数据 | 原生未认证；WSL 设置不等于原生验收 | 未认证 |
| 高级后端 | 仅报告所列路线 | 受工具/WSL 限制 | 受工具限制 |

Linux 验证包含实际 CPack 包和隔离设置的本地扩展安装，范围与版本见
[官方集成记录](../release/OFFICIAL_SLICER_VALIDATION_20261022.zh-CN.md)；
不能代表当前修改、空白机器、在线商店安装或其他平台已通过。资源检查不等于
三平台 Slicer 构建。详见[发布门禁](../release/EXTENSION_READINESS.zh-CN.md)。

---

## 功能与平台

**原生候选**表示源码有对应路线，不代表该系统已通过科学验收。
WSL 是 Linux 后端，不是 Windows 原生计算；Windows/macOS 不继承 Linux 的 QC 或时间。

| 功能 | Linux | Windows | macOS |
| --- | --- | --- | --- |
| 界面、NIfTI/NRRD、校验、结果加载 | 有限定范围实测 | 原生候选，需影像包 | 原生候选，需影像包 |
| DICOM 转换 | 需要转换器 | 原生转换器/Slicer 候选 | 原生转换器/Slicer 候选 |
| DTI、确定性追踪 | Slicer 内 Teem 张量拟合；DIPY 追踪 | 同左，原生候选 | 同左，原生候选 |
| DDSeg/DDParcel/DDSurfer 原生重建 | 指定数据/路线已测，需匹配模型 | 原生 PyTorch 候选，需可用掩膜 | 原生 CPU 候选，需可用掩膜及匹配模型 |
| FW、NODDI | 指定输入已测 | 原生 Python 候选，需兼容 DIPY/AMICO | 原生 Python 候选，需匹配架构的包 |
| DKI/Python MK-Curve | 指定被试全脑已有独立检查与计时；无效体素须排除 | 原生数值候选，需适用的多壳数据 | 原生数值候选，需适用的多壳数据 |
| PNL CNN 掩膜 | PyTorch/ANTs 路线；CPU 概率/掩膜配对检查；Linux x86_64 自动配置 | Windows Slicer Python 内的 native 路线（SimpleITK + PyTorch），**实验性**；此处未在 Windows 上运行 | native 路线（SimpleITK + PyTorch），**实验性**；无 CUDA，Metal 未实现，未实测 |
| 皮层表面后处理 | 合格原生组件或已有 FreeSurfer；已测离线预编译包安装，在线发布待完成；[配置说明](SURFACE_COMPONENTS.zh-CN.md) | 无合格原生预编译包；WSL／完整安装桥接待验收 | 已有原生 FreeSurfer 候选；无合格预编译包 |
| MRtrix/FSL 预处理 | CPU、CUDA 有记录 | WSL 2 桥接；设备策略在实际 WSL Python 内运行 | 原生 CPU MRtrix/FSL 候选；**不能使用 CUDA eddy** |
| CSD/ACT/SIFT2 | 外部 MRtrix 路线已测 | MRtrix WSL 桥接，全流程待验收 | 原生 MRtrix 候选，全流程待验收 |
| 完整 SAF | 指定输入的 Python 筛选/几何已核验 | 已补 argv/PATH 适配；需 MRtrix 桥接和准备好的表面，待验收 | POSIX/MRtrix 及准备好的表面，待验收 |
| WMQL | 规则通过，稀疏 AF 为 **PARTIAL** | 固定 TractQuerier WSL 桥接候选 | 配置作者 Python 后端的候选路线 |
| WMA | 149741 限定范围通过；旧 100307 覆盖失败仍排除 | WSL 作者 shell/Slicer 反向桥接，待验收 | 作者 shell/display/Slicer CLI 前提，待验收 |
| TractCloud | 指定输入的固定种子 CUDA/重复检查通过 | WSL 按设备选 CPU/CUDA；原生 Python 未认证 | 外部 Python CPU 候选，未实现 MPS |
| Python 连接组/ROI 统计/VTK 预览 | 有限定范围实测 | 有效输入产品下的原生候选 | 有效输入产品下的原生候选 |
| 插件内 MRtrix/FSL/PNL 自动配置 | Linux x86_64（含 managed PNL 运行环境） | 独立 Windows/WSL 安装器配置 MRtrix3 和 FSL（脑掩膜与 Python 阶段仍在 Windows 版 Slicer 的 Python 中）；PNL 使用 native 路线（PyTorch 扩展与脑掩膜资源） | MRtrix3 3.0.8 官方 tarball 与 FSL CPU 运行环境，Intel 和 Apple Silicon，macOS 14.5 或更高；PNL 使用 native 路线 |

---

## 使用边界

- 各平台优先保留已有、人工检查过的原生掩膜，后端失败不会静默换算法。
- PNL 掩膜路线：Linux 用 managed conda 运行环境（ANTs 2.3.0）；macOS 和 Windows 用 native 路线，标注为“native（实验性）”。Linux 默认 PNL 生成掩膜。显式 WSL 安装器创建的新 Windows 配置默认 MRtrix 掩膜；已有默认设置不变。无 WSL 时，可提供已核验掩膜，或明确选择纯 Python 的 DIPY 掩膜候选路线。见[轻量路线](EXPERIMENTS.zh-CN.md)。
- PyTorch Auto 只选择 CUDA 或 CPU。**Apple MPS 尚未实现/验证**，并不是 Mac 没有 GPU。PNL 同样选择 PyTorch CPU 或调用方可见的 CUDA，不选择 MPS。
- Windows 安装器默认 `-Device auto`，在 WSL 内检测 GPU 并验证真实 PyTorch CUDA 分配。明确 CUDA 失败不会重试 CPU。
- FSL 安装需明确接受许可；不会修改 NVIDIA 驱动或安装替代 Linux 显卡驱动。
- WSL 子进程取消及 Windows/macOS 场景回载仍需真机验收。源码/单元测试不能证明取消后所有 Linux 子进程都已终止。
- 全脑 DKI 与 MK-Curve 记录见[模块计时](../release/MODULE_RUNTIMES_20261017.zh-CN.md)；全脑运行与有效体素筛查不等于全脑 MATLAB 对照或所有采集方案均已验证。

旧 DDSeg 入口已识别 `.exe` 和 `Slicer.app/Contents`；macOS 外部 Python 不再继承
Slicer dylib 路径。详见[本轮平台复核](../release/PLATFORM_AUDIT_20261015.md)、
[Windows](WINDOWS.md)、[macOS](MACOS.md) 指南。

---

## 上游限制

FSL 支持 Linux/macOS，Windows 通过 WSL；本插件仍需相应系统验收。
[FSL 安装](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html)、
[Windows FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)。
FreeSurfer 的 Windows 路线也是托管 Linux，不是原生 Windows 程序。
[FreeSurfer 要求](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements)。
PyTorch 本身支持 MPS，但本插件目前没有接入。
[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)。
