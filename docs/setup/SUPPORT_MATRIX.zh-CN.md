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
| 资源/CMake 合同 | 有本地证据；已配置 CI | 有本地证据（`SlicerDSpace-v0.1.1-Windows-x64.zip` 审计通过）；已配置 CI | 已配置 CI，仍需运行证据 |
| 真实 Slicer GUI/安装包安装 | 有带日期记录（`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`） | **已通过 3D Slicer 5.10.0 验收**（Extension Manager 安装 + 4 项 GUI smoke + 46 项自检） | 未认证（原生依赖全部验收前不宣称完整支持） |
| 真实 SDK/CTest/CPack | 2026-10-22 候选版本已在 Linux/Slicer 5.10 验证 | 统一扩展包已在 Slicer 5.10.0 验证；完整 C++ superbuild SDK 待选配 | 待真实 SDK 验收 |
| 科学路线 | 仅报告所列路线/数据 | **已在 3D Slicer 5.10.0（`HCP 100307` 全脑体数据，`SLICERDSPACE_DISABLE_WSL=1`）完成原生验收**；见 [Windows 5.10.0 验收报告](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md) | 未认证 |
| 高级后端 | 仅报告所列路线 | **原生 MRtrix3（`3.0.8`，68 个命令）、FSL（`2602.0`）、Surface（`0.1.0`，7 个命令）、WMA、WMQL、TractCloud、SAF 均已通过验收** | 受工具限制，待验收 |

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
| 界面、NIfTI/NRRD、校验、结果加载 | 有限定范围实测 | **已在 3D Slicer 5.10.0 原生通过**（GUI 冒烟测试 + 场景回载） | 原生候选，需影像包 |
| DICOM 转换 | 需要转换器 | **已在 3D Slicer 5.10.0 原生通过**（`dcm2niix.exe`） | 原生转换器/Slicer 候选 |
| DTI、确定性追踪 | Slicer 内 Teem 张量拟合；DIPY 追踪 | **已在 3D Slicer 5.10.0 原生通过**（多壳 `HCP 100307`、`HCP 100408` 全脑 + 临床单壳 `sub-001` 全脑） | 同左，原生候选 |
| DDSeg/DDParcel/DDSurfer 原生重建 | 指定数据/路线已测，需匹配模型 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307`、`HCP 100408`、`sub-001` 全脑体数据） | 原生 CPU 候选，需可用掩膜及匹配模型 |
| FW、NODDI | 指定输入已测 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑 FW & NODDI；`sub-001` 全脑 FW） | 原生 Python 候选，需匹配架构的包 |
| DKI/Python MK-Curve | 指定被试全脑已有独立检查与计时；无效体素须排除 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307`、`HCP 100408` 全脑 WLS + 整层轴位切片与多层分层抽样 MK-Curve） | 原生数值候选，需适用的多壳数据 |
| PNL CNN 掩膜 | PyTorch/ANTs 路线；CPU 概率/掩膜配对检查；Linux x86_64 自动配置 | Windows Slicer Python 内的 native 路线（SimpleITK + PyTorch），**实验性** | native 路线（SimpleITK + PyTorch），**实验性**；无 CUDA，Metal 未实现，未实测 |
| 皮层表面后处理 | 合格原生组件或已有 FreeSurfer；已内置于 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`；[配置说明](SURFACE_COMPONENTS.zh-CN.md) | **已在 3D Slicer 5.10.0 原生通过**（7 个原生命令内置于 `surface-runtime-windows-x86_64`，随统一安装包分发；运行需自备 FreeSurfer 许可证） | 已有原生 FreeSurfer 候选；无合格预编译包 |
| MRtrix/FSL 预处理 | CPU、CUDA 有记录；原生运行包内置于 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` | **已在 3D Slicer 5.10.0 原生通过**（`topup.exe`、`applytopup.exe`、`eddy_cpu.exe` 内置于 `fsl-2602.0-windows-x86_64`，与 Linux 数值完全一致；WSL 2 仅作可选回退） | 原生 CPU MRtrix/FSL 候选；**不能使用 CUDA eddy** |
| CSD/ACT/SIFT2 | 外部 MRtrix 路线已测 | **已在 3D Slicer 5.10.0 原生通过**（68 个命令内置于 `mrtrix-3.0.8-windows-x86_64`；免 T1 ACT 使用 `act_prep.py` `DDSeg + DDParcel -> 5TT -> 5ttcheck + 5tt2gmwmi` 及 `5ttgen freesurfer`） | 原生 MRtrix 候选，全流程待验收 |
| 完整 SAF | 指定输入的 Python 筛选/几何已核验 | **已在 3D Slicer 5.10.0 原生通过**（`tckgen.exe`、`tckedit.exe`、`tckresample.exe`、`tckmap.exe`、`tcksift2.exe` + 原生 `trimesh`/`rtree`） | POSIX/MRtrix 及准备好的表面，待验收 |
| WMQL | 规则通过，稀疏 AF 为 **PARTIAL** | **已在 3D Slicer 5.10.0 原生通过**（原生 `tract_querier`，37/37 非空纤维束与 Linux 完全一致） | 配置作者 Python 后端的候选路线 |
| WMA | 149741 限定范围通过；旧 100307 覆盖失败仍排除 | **已在 3D Slicer 5.10.0 原生通过**（`whitematteranalysis` + 原生 VTK 标量统计，74/74 ORG 束 100% 覆盖） | 作者 shell/display/Slicer CLI 前提，待验收 |
| TractCloud | 指定输入的固定种子 CUDA/重复检查通过 | **已在 3D Slicer 5.10.0 原生通过**（Slicer PyTorch CPU/CUDA，42/42 束与 Linux 100% 一致） | 外部 Python CPU 候选，未实现 MPS |
| Python 连接组/ROI 统计/VTK 预览 | 有限定范围实测 | **已在 3D Slicer 5.10.0 原生通过**（多被试 `84×84` 对称连接矩阵） | 有效输入产品下的原生候选 |
| 插件内运行包分发与损坏自动修复 | `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`（含 `bundled-runtimes` 与 `tree_sha256` 自动修复） | **`SlicerDSpace-v0.1.1-Windows-x64.zip`**（首次启动自动识别内置 `bundled-runtimes`，缺失或损坏时通过 `tree_sha256` 自动校验修复） | 尚未发布完整原生统一安装包 |

---

## 使用边界与分发说明

- **安装方式**：当前版本从 GitHub Releases 下载平台统一安装包（`SlicerDSpace-v0.1.1-Windows-x64.zip` 或 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`），通过 3D Slicer 5.10.0 `Extensions Manager -> Install Extensions -> Install Extension from File` 本地安装（尚未上架 Slicer 在线扩展商店目录）。
- **模型权重与大型图谱按需下载**：统一安装包内置扩展代码与三大原生运行环境（`MRtrix3 3.0.8`、`FSL 2602.0` 扩散预处理、`FreeSurfer Surface Runtime 0.1.0`）。深度学习模型权重（`DDSeg`、`DDParcel`、`DDSurfer`、`TractCloud`、`PNL`）与大型白质图谱（`ORG`）仍在首次使用时按需下载并缓存至本机。
- **ACT / 5TT 准备路线说明**：SlicerDSpace 原生免 T1 ACT 流程直接由 `DDSeg + DDParcel` 合成 5TT 与灰白质交界掩膜（`tractography/act_prep.py`，经 `5ttcheck` 与 `5tt2gmwmi` 校验），或对分区体积运行 `5ttgen freesurfer`。MRtrix3 上游的 `5ttgen fsl` 模式需要 T1w 输入及完整外部 FSL（`bet`、`fast`、`run_first_all`），SlicerDSpace 不依赖该模式，且仅含 3 个扩散预处理命令（`topup`、`applytopup`、`eddy_cpu`）的内置 FSL 运行包不支持 `5ttgen fsl`。
- **FreeSurfer 许可证要求**：内置的 7 个 FreeSurfer 皮层表面后处理命令（`mris_anatomical_stats`、`mris_ca_label` 等）不包含任何用户专属 FreeSurfer 许可证或激活文件（`activation_key_included: false`）。运行皮层表面后处理时，用户需自备免费的个人 FreeSurfer 许可证文件（`FS_LICENSE`）。
- **掩膜策略与 PyTorch 设备**：各平台优先保留已有、人工检查过的原生掩膜，后端失败不会静默替换算法。PyTorch Auto 优先选择可用 CUDA，否则回退 CPU；**Apple MPS 尚未实现/验证**。
- **DKI / MK-Curve 验收范围**：已在多壳被试（`HCP 100307`、`HCP 100408`）上完成全脑 DKI（`WLS`）及整层轴位切片/多层抽样 Python `MK-Curve` 验证；有效体素筛查不等于任意临床采集协议下的全脑 MATLAB 对照。

详见 [Windows 5.10.0 验收报告](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)、[Windows 配置指南](WINDOWS.md) 与 [macOS 指南](MACOS.md)。

---

## 上游说明

- **FSL**：FSL 官方在 Windows 上仅提供 WSL 安装路线（[FSL 安装](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html)、[Windows FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)）。SlicerDSpace `v0.1.1` 依据 Oxford University Innovation Academic Licence 内置了扩散预处理所需的 3 个 Windows x64 原生程序（`topup.exe`、`applytopup.exe`、`eddy_cpu.exe`）；扩散预处理以外的完整 FSL 组件（如 `bet`、`fast`、`FIRST`、GPU `eddy_cuda`）不在内置范围内，WSL 2 仅保留为可选回退。
- **FreeSurfer**：FreeSurfer 官方在 Windows 上仅提供 Linux 托管路线（[FreeSurfer 要求](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements)）。SlicerDSpace `v0.1.1` 内置了 `DDSurfer` 皮层后处理所需的 7 个 Windows x64 原生命令，无需 WSL；完整 `recon-all` 不在内置范围内，且仍需用户自备 FreeSurfer 许可证。
- **PyTorch MPS**：PyTorch 支持 Apple MPS（[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)），但本插件目前尚未启用或认证 MPS。
