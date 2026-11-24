<div align="center">

# 能力与平台

**只安装当前路线需要的依赖**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [发布](../release/EXTENSION_READINESS.zh-CN.md) | [English](SUPPORT_MATRIX.md)

</div>

能力检查检测包、程序和资源是否可用，不认证每种采集、每个平台的数值正确性。

**支持的 3D Slicer 版本：** 5.10 至 5.12.4（参考版本 5.10.0）。
不支持 3D Slicer 5.8 与 5.9：Slicer 5.8 内置 Python 3.9，而打包的 Python 依赖（`dipy 1.12.1`、`matplotlib 3.11.2`、`networkx 3.7`）需要 Python 3.11 或更高版本。

**macOS（`arm64`）** 指 Apple Silicon 主机。macOS 版 3D Slicer 是在 Rosetta 2 下运行的 `x86_64` 构建，打包的命令行运行时为原生 `arm64`。

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

| 检查 | Linux | Windows | macOS（Apple Silicon `arm64` 已验收；Intel `x86_64` 未验证） |
| --- | --- | --- | --- |
| 资源/CMake 合同 | 有本地证据；已配置 CI | 有本地证据（`SlicerDSpace-v0.1.1-Windows-x64.zip` 审计通过）；已配置 CI | 有本地证据（`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` 审计通过）；已配置 CI |
| 真实 Slicer GUI/安装包安装 | 有带日期记录（`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`） | **已通过 3D Slicer 5.10.0 验收**（Extension Manager 安装 + 4 项 GUI smoke + 46 项自检） | **已在 Apple Silicon 3D Slicer 5.10.0 通过验收**（Extension Manager 安装 + 4 项 GUI smoke + 48 项自检；Intel `x86_64` 真机未验证） |
| 真实 SDK/CTest/CPack | 2026-10-22 候选版本已在 Linux/Slicer 5.10 验证 | 统一扩展包已在 Slicer 5.10.0 验证；完整 C++ superbuild SDK 待选配 | 统一扩展包已在 Slicer 5.10.0（`arm64`）验证；完整 C++ superbuild SDK 待选配 |
| 科学路线 | 仅报告所列路线/数据 | **已在 3D Slicer 5.10.0（`HCP 100307` 全脑体数据，`SLICERDSPACE_DISABLE_WSL=1`）完成原生验收**；见 [Windows 5.10.0 验收报告](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md) | **已在 Apple Silicon 3D Slicer 5.10.0（`HCP 100307` 全脑体数据）完成原生验收**；见 [macOS 5.10.0 验收报告](../release/MACOS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md) |
| 高级后端 | 仅报告所列路线 | **原生 MRtrix3（`3.0.8`，68 个命令）、FSL（`2602.0`）、Surface（`0.1.0`，7 个命令）、WMA、WMQL、TractCloud、SAF 均已通过验收** | **原生 `arm64` MRtrix3（`3.0.8`，67 个命令）、FSL（`2602.0`）、Surface（`0.1.0`，7 个命令）、WMA、WMQL、TractCloud、SAF 均已通过验收** |

Linux 验证包含实际 CPack 包和隔离设置的本地扩展安装，范围与版本见
[官方集成记录](../release/OFFICIAL_SLICER_VALIDATION_20261022.zh-CN.md)；
不能代表当前修改、空白机器、在线商店安装或其他平台已通过。资源检查不等于
三平台 Slicer 构建。详见[发布门禁](../release/EXTENSION_READINESS.zh-CN.md)。

---

## 功能与平台

**原生候选**表示源码有对应路线，不代表该硬件架构已通过科学验收（例如 Intel `x86_64` Mac 真机尚未实测验证）。
WSL 是 Linux 后端，不是 Windows 原生计算；Windows 与 Apple Silicon macOS 均通过真实数据独立完成数值一致性验收。

| 功能 | Linux | Windows | macOS（Apple Silicon `arm64` 已验收；Intel `x86_64` 未验证） |
| --- | --- | --- | --- |
| 界面、NIfTI/NRRD、校验、结果加载 | 有限定范围实测 | **已在 3D Slicer 5.10.0 原生通过**（GUI 冒烟测试 + 场景回载） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（GUI 冒烟测试 + 场景回载） |
| DICOM 转换 | 需要转换器 | **已在 3D Slicer 5.10.0 原生通过**（`dcm2niix.exe`） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`dcm2niix`） |
| DTI、确定性追踪 | Slicer 内 Teem 张量拟合；DIPY 追踪 | **已在 3D Slicer 5.10.0 原生通过**（多壳 `HCP 100307`、`HCP 100408` 全脑 + 临床单壳 `sub-001` 全脑） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑，与 Linux `max_abs = 0.0`） |
| DDSeg/DDParcel/DDSurfer 原生重建 | 指定数据/路线已测，需匹配模型 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307`、`HCP 100408`、`sub-001` 全脑体数据） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑：DDSeg `99.89%`、DDParcel `101/101` 标签、DDSurfer `100%` 面片拓扑一致） |
| FW、NODDI | 指定输入已测 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑 FW & NODDI；`sub-001` 全脑 FW） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑 FW `r = 1.0`、NODDI `r >= 0.999999`） |
| DKI/Python MK-Curve | 指定被试全脑已有独立检查与计时；无效体素须排除 | **已在 3D Slicer 5.10.0 原生通过**（`HCP 100307`、`HCP 100408` 全脑 WLS + 整层轴位切片与多层分层抽样 MK-Curve） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`HCP 100307` 全脑 WLS `r >= 0.999999999999996` + 多层抽样 MK-Curve `r = 0.9919–0.9999`） |
| PNL CNN 掩膜 | PyTorch/ANTs 路线；CPU 概率/掩膜配对检查；Linux x86_64 自动配置 | Windows Slicer Python 内的 native 路线（SimpleITK + PyTorch），**实验性** | native 路线（SimpleITK + PyTorch），**实验性**；无 CUDA；在 MPS 上运行（PyTorch `2.2.2`、`x86_64`、Rosetta 2） |
| 皮层表面后处理 | 合格原生组件或已有 FreeSurfer；已内置于 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`；[配置说明](SURFACE_COMPONENTS.zh-CN.md) | **已在 3D Slicer 5.10.0 原生通过**（7 个原生命令内置于 `surface-runtime-windows-x86_64`，随统一安装包分发；运行需自备 FreeSurfer 许可证） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（7 个 `arm64` 原生命令内置于 `surface-runtime-darwin-arm64`，与 Linux `r >= 0.9999999`；运行需自备 FreeSurfer 许可证） |
| MRtrix/FSL 预处理 | CPU、CUDA 有记录；原生运行包内置于 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` | **已在 3D Slicer 5.10.0 原生通过**（`topup.exe`、`applytopup.exe`、`eddy_cpu.exe` 内置于 `fsl-2602.0-windows-x86_64`，与 Linux 数值完全一致；WSL 2 仅作可选回退） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`topup`、`applytopup`、`eddy_cpu` 内置于 `fsl-2602.0-darwin-arm64`；**不能使用 CUDA eddy**） |
| CSD/ACT/SIFT2 | 外部 MRtrix 路线已测 | **已在 3D Slicer 5.10.0 原生通过**（68 个命令内置于 `mrtrix-3.0.8-windows-x86_64`；免 T1 ACT 使用 `act_prep.py` `DDSeg + DDParcel -> 5TT -> 5ttcheck + 5tt2gmwmi` 及 `5ttgen freesurfer`） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（67 个命令内置于 `mrtrix-3.0.8-darwin-arm64`；免 T1 ACT + `iFOD2` + `SIFT2`） |
| 完整 SAF | 指定输入的 Python 筛选/几何已核验 | **已在 3D Slicer 5.10.0 原生通过**（`tckgen.exe`、`tckedit.exe`、`tckresample.exe`、`tckmap.exe`、`tcksift2.exe` + 原生 `trimesh`/`rtree`） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（原生 `mrtrix-3.0.8-darwin-arm64` + `trimesh`/`rtree`，`HCP 100307` 共 `12,745` 条流线） |
| WMQL | 规则通过，稀疏 AF 为 **PARTIAL** | **已在 3D Slicer 5.10.0 原生通过**（原生 `tract_querier`，37/37 非空纤维束与 Linux 完全一致） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（原生 `tract_querier`，37/37 非空纤维束与 Linux 完全一致） |
| WMA | 149741 限定范围通过；旧 100307 覆盖失败仍排除 | **已在 3D Slicer 5.10.0 原生通过**（`whitematteranalysis` + 原生 VTK 标量统计，74/74 ORG 束 100% 覆盖） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`whitematteranalysis` + 原生 VTK 标量统计，74/74 ORG 束 100% 覆盖） |
| TractCloud | 指定输入的固定种子 CUDA/重复检查通过 | **已在 3D Slicer 5.10.0 原生通过**（Slicer PyTorch CPU/CUDA，42/42 束与 Linux 100% 一致） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（Slicer PyTorch CPU，42/42 束完整生成；推理也可在 Rosetta 2 下用 PyTorch `2.2.2` 的 MPS 运行） |
| Python 连接组/ROI 统计/VTK 预览 | 有限定范围实测 | **已在 3D Slicer 5.10.0 原生通过**（多被试 `84×84` 对称连接矩阵） | **已在 Apple Silicon (`arm64`) 3D Slicer 5.10.0 原生通过**（`python` 与 `mrtrix` `84×84` 对称连接矩阵 + `101` 脑区 ROI 统计表） |
| 插件内运行包分发与损坏自动修复 | `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`（含 `bundled-runtimes` 与 `tree_sha256` 自动修复） | **`SlicerDSpace-v0.1.1-Windows-x64.zip`**（首次启动自动识别内置 `bundled-runtimes`，缺失或损坏时通过 `tree_sha256` 自动校验修复） | **`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`**（首次启动自动识别内置 `bundled-runtimes`，缺失或损坏时通过 `tree_sha256` 自动校验修复） |

---

## 各阶段 / 模块硬件内存与显存需求参考

下表中的主机内存峰值综合了 `HCP 100307` 全脑多壳层数据（`145 × 174 × 145 × 288`，`1.25 mm` 各向同性，未压缩 `float64` DWI 信号约 `7.85 GiB`）在个人 Windows PC（`i7-13700K`、32 GB 内存、纯 CPU）与 Linux `x86_64` M1 探针（`shared/cohort_defaults.py`）上的实测峰值，以及批处理规划启发值（`shared/cohort_planning.py`）和显存 / OOM 保护门限（`shared/hardware_adapt.py`、`shared/environment.py`、`anatomy/surface/ddsurfer_backend/inference/predict.py`）。常规临床单壳层采集（`<= 64` 卷）的实际内存占用通常显著更低。

- **主机内存基线**：`16 GiB` 内存可覆盖全部 22 个模块在 HCP `288` 卷全脑数据上的单被试运行（已在 Apple M4 16 GB 统一内存上完成实测）；若运行完整多壳层全流程、`MK-Curve`、`UKF`、`100 万`流线 `CSD/ACT`、`WMA` 或多被试队列批处理，建议配置 `32 GiB` 及以上内存；常规临床（`<= 64` 卷）DTI 及基础路线在 `8 GiB` 内存上即可运行。
- **消费级 GPU 与 Windows WDDM（`4–8 GiB` 显存）**：在 `auto` 设备模式下，CUDA 算子会先查询可用显存（`torch.cuda.mem_get_info`），按 Windows TDR 上限（`windows_tdr_cap` / `consumer_fp64_tdr_cap`）自适应缩放分块大小，在 Windows WDDM 可用显存低于 `8.0 GiB`（Linux 低于 `6.0 GiB`）时自动禁用 `cudnn.benchmark` 以避免工作区溢出至共享内存，并在显存不足（OOM）时自动将分块减半或回退至 CPU。`--low-memory` 模式会限制单阶段内存预留、并发作业数（`1` 个作业 / `<= 4` 线程）与体素分块，已在 `3.5 GiB` 笔记本显存上限下通过验证（峰值显存 `<= 2.4 GiB`）。
- **显式上限控制**：可通过环境变量 `SLICERDSPACE_MAX_VRAM_GIB` 与 `SLICERDSPACE_MAX_RAM_GIB` 手动设定保守显存 / 内存上限，或在批处理模式下传入 `--low-memory`。

| 阶段 / 模块 | 典型峰值内存（HCP 288 卷） | 推荐主机内存 | GPU 显存需求（CUDA / MPS） | 说明（OOM 防护 / 自动回退） |
| --- | --- | --- | --- | --- |
| **输入与校验**（`DICOM / NIfTI / NRRD`） | `0.2–1.5 GiB`（临床 31 卷实测 `0.20 GiB`） | `8 GiB` | 仅 CPU（`—`） | 单遍流式 gzip 分层校验，避免一次性将完整 4D 数据解压至内存。 |
| **脑掩膜**（`PNL CNN`） | `2.2 GiB`（PC CPU 实测 `2.15 GiB`） | `8 GiB` | 可选 `>= 2 GiB` CUDA 或 Apple MPS（`16 GB` 统一内存） | 通过 `adaptive_cuda_scaled_batch` / `adaptive_gpu_scaled_batch` 缩放 2D 切片批大小；CUDA/MPS OOM 时自动减半批大小并在 `auto` 下回退 CPU。 |
| **预处理**（`DIPY MP-PCA / Gibbs`） | 约 `1–8 GiB`（临床 31 卷实测 `0.92 GiB`） | `16 GiB`（临床 `<= 64` 卷 `8 GiB`） | 可选 `>= 2 GiB` CUDA（约 `150 MB` 5 切片环形缓冲区）；Gibbs `float32` FFT 支持 Apple MPS | CPU 端由 `adaptive_cpu_slab_z` 限制轴位分块内存；CUDA/MPS OOM 时自动减半分块并在 `auto` 下回退多线程 CPU。 |
| **预处理**（`FSL Eddy / Topup`） | 约 `2.2–12 GiB`（临床 31 卷 `eddy_cpu` 实测 `2.17 GiB`） | `16 GiB`（多壳队列推荐 `32 GiB`） | 可选 `>= 4–6 GiB` CUDA（仅外部 Linux FSL `eddy_cuda`）；Windows/macOS 内置版仅 CPU | Windows/macOS 内置原生 FSL 使用多线程 `topup` + `eddy_cpu`，无需独立 GPU。 |
| **解剖先验**（`DDSeg`） | `2.8 GiB`（Linux M1 实测 `2.79 GiB`） | `8–16 GiB` | 可选 `>= 2–4 GiB` CUDA 或 Apple MPS（`16 GB` 统一内存） | CUDA/MPS OOM 时自动将 2D 切片批大小减半（最低至 `2`）并在 `auto` 下回退 CPU；Windows WDDM 低于 `8.0 GiB` 可用显存时自动关闭 cuDNN benchmark（Linux 为 `6.0 GiB`）。 |
| **解剖先验**（`DDParcel`） | `10.8 GiB`（Linux M1 实测 `10.82 GiB`） | `16 GiB`（队列推荐 `32 GiB`） | 可选 `>= 4 GiB` CUDA 或 Apple MPS（`16 GB` 统一内存） | CUDA/MPS OOM 时自动退化为单切片重试并在 `auto` 下回退 CPU；Windows WDDM 低于 `8.0 GiB` 可用显存时禁用 cuDNN benchmark。 |
| **解剖先验**（`DDSurfer`） | `3.6 GiB`（Linux M1 实测 `3.63 GiB`） | `16 GiB` | 可选 CUDA：需 `>= 7.0 GiB` 可用显存（`5.5 GiB` 单表面按需预算 + `1.5 GiB` 余量，峰值约 `4–5 GiB`）；`>= 18.0 GiB` 可用显存启用双表面预载；macOS Rosetta PyTorch `2.2.2` 无 MPS `Conv3d`，走 CPU | 预检门限（`SURFACE_MIN_FREE_GPU_GIB = 7.0`）在 `auto` 模式下将低于 `7.0 GiB` 可用显存的设备（含常见 `< 8 GiB` Windows WDDM 显卡）自动转至 CPU；显式指定 `cuda` 时直接给出清晰报错提示。 |
| **解剖先验**（`FreeSurfer` 表面形态学后处理） | `3.4 GiB`（PC CPU 实测 `3.42 GiB`） | `8–16 GiB` | 仅 CPU（`—`） | 内置 7 命令原生表面运行包（`mris_inflate`、`mris_sphere`、`mris_register` 等），支持半球内 DAG 并行及 OpenMP（`macOS 13.0+` 静态 `libomp.a`）；需自备 `FS_LICENSE`。 |
| **微结构**（`Teem DTI`） | 首次全脑 `6.6–9.7 GiB`（PC `6.60 GiB` / Linux M1 `9.75 GiB`；复用已备 DTI 约 `2.4 GiB`） | `16 GiB`（临床 `<= 64` 卷 `8 GiB`） | 仅 CPU（`—`，按设计执行 `fp64` WLS） | 零信号体素跳过与共享准备回执避免重复张量拟合。 |
| **微结构**（`Free-Water` 自由水） | `2.4–6.9 GiB`（Linux M1 `2.37 GiB` / PC CPU `6.93 GiB`） | `16 GiB` | 可选 `>= 4 GiB` CUDA（`fp64`）；macOS 走 CPU | `adaptive_cuda_scaled_batch` 自动施加 Windows WDDM / 消费级 FP64 TDR 上限，OOM 时减半分块并在 `auto` 下回退 CPU。 |
| **微结构**（`AMICO NODDI`） | `8.6–17.2 GiB`（PC 有效集 QP 实测 `8.55 GiB` / Linux M1 峰值 `17.22 GiB`） | `16–32 GiB` | 可选 `>= 4 GiB` CUDA（`fp64`）；macOS 走 CPU | CUDA 有效集 QP 应用 `windows_tdr_cap=8192` / `consumer_fp64_tdr_cap=2048`，OOM 时减半分块并在 `auto` 下回退 CPU。 |
| **微结构**（`DKI`） | `5.9–6.9 GiB`（Linux M1 `5.94 GiB` / PC CPU `6.88 GiB`） | `16 GiB` | 可选 `>= 4 GiB` CUDA（`fp64` WLS）；macOS 走 CPU | `adaptive_spawn_workers` 限制 Windows/macOS 派生工作进程数；`--low-memory` 默认 `--dki-voxels-per-chunk 8192`；CUDA OOM 时减半分块并回退 CPU。 |
| **微结构**（`MK-Curve`） | `3.3–9.8 GiB`（Apple M4 8 线程流式 CPU 实测 `3.29 GiB`；早期 PC 4 进程 `9.78 GiB`） | `16 GiB` | 可选 `>= 4 GiB` CUDA（`fp64`）；macOS 走 CPU | Windows/macOS 在全脑扫描（`len(grid) >= 64`）下采用单遍流式输入与释放 GIL 的 `ThreadPoolExecutor`（`<= 10 GiB` 主机限 4 线程），免除多进程拷贝开销；Linux 保持多进程；CUDA OOM 自动减半分块。 |
| **纤维追踪**（`DTI` 确定性追踪） | 约 `4–8 GiB`（分块受限；规划启发值 `2 + signal`） | `16 GiB`（临床 `<= 64` 卷 `8 GiB`） | 可选 `>= 4 GiB` CUDA（张量与主方向提取）；macOS 走 CPU | `adaptive_cpu_slab_z` 约束 CPU 峰值内存；CUDA 峰值提取在 OOM 时减半分块并在 `auto` 下回退 CPU。 |
| **纤维追踪**（`UKF` 双张量追踪） | `10.1 GiB`（PC CPU 实测 `10.09 GiB`，`25 万`条流线） | `16–32 GiB` | 仅 CPU（`—`，多线程 C++ `UKFTractography`） | 内存峰值随流线密度与每体素种子数增长；VTK/TRK 导出采用单缓冲复用以降低峰值。 |
| **纤维追踪**（`MRtrix3 CSD / ACT`） | `10.6 GiB`（Linux M1 实测 `10.58 GiB`，`100 万`条流线） | `16–32 GiB` | 仅 CPU（`—`，原生 MRtrix3 `3.0.8`） | 峰值随目标流线数（`tckgen` + `tcksift2`）增长；`16 GiB` 内存主机多任务运行时可适当降低目标流线数。 |
| **纤维追踪**（`SAF` 浅表联合纤维） | `0.6–4 GiB`（对既有 `10 万`条流线与 DDSurfer 网格重筛选实测 `0.63 GiB`） | `16 GiB` | 仅 CPU（`—`，向量化 BVH 射线-三角面求交 CPU 算子） | 全流程在优化后的多线程 CPU 路径运行（`tckgen`/`tckedit` + BVH 几何过滤）。 |
| **白质纤维束**（`WMQL`） | 约 `4–8 GiB`（`100 万` CSD/ACT 流线；规划启发值 `8 GiB`） | `16 GiB` | 仅 CPU（`—`，多线程扁平 CSR CPU 算子） | 扁平 CSR 体素标签直方图避免逐流线构建稠密 Python 字典。 |
| **白质纤维束**（`WMA` ORG 图谱聚类） | `5.7–12 GiB`（PC CPU `8 万`条流线 / 8 线程实测 `5.74 GiB`；稠密流线规划启发值 `16 GiB`） | `16–32 GiB` | 可选 `>= 4–6 GiB` CUDA（PyTorch/Triton 算子）；macOS 走 CPU | Windows 下自动配置 `PYTORCH_CUDA_ALLOC_CONF`；`auto` 模式下如遇 CUDA OOM 自动切换 `WMA_GPU=off` 于 CPU 重试。 |
| **白质纤维束**（`TractCloud`） | `2.7–8 GiB`（PC CPU `5 万`条流线实测 `2.70 GiB`；`100 万`条流线时随数量升至约 `8–12 GiB`） | `16 GiB` | 可选 `>= 2–4 GiB` CUDA 或 Apple MPS（`16 GB` 统一内存） | 根据可用显存自动缩放批大小（`< 2.2 GiB` 限 `256`、`< 4.5 GiB` 限 `512`、Windows `< 7.5 GiB` 限 `1024`；MPS 为 `256/512/1024`）；显式 GPU 报错不静默切回 CPU（必要时手动选择 `CPU`）。 |
| **结构连接组**（`Streamline Count`） | `0.3–2.6 GiB`（PC Python `100 万`流线实测 `0.27 GiB`；Linux M1 `2.61 GiB`） | `8–16 GiB` | 仅 CPU（`—`，分批 `cKDTree` / `tck2connectome`） | 按 `4,096` 条流线分批端点匹配，即使在 `100 万`条流线上也保持极低内存占用。 |
| **结构连接组**（`SIFT2-Weighted`） | `0.5–2.6 GiB`（Linux M1 `100 万`流线 `<= 2.61 GiB`） | `8–16 GiB` | 仅 CPU（`—`，原生 MRtrix3 / 向量化 Python） | 需配对流线与 `sift2_weights.txt`，相对端点索引仅增加极少量权重累加内存。 |

---

## 使用边界与分发说明

- **安装方式**：当前版本从 GitHub Releases 下载平台统一安装包（`SlicerDSpace-v0.1.1-Windows-x64.zip`、`SlicerDSpace-v0.1.1-Linux-x64.tar.gz` 或 `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`），通过 3D Slicer 5.10.0 `Extensions Manager -> Install Extensions -> Install Extension from File` 本地安装（尚未上架 Slicer 在线扩展商店目录；Intel `x86_64` Mac 真机硬件未验证）。
- **模型权重与大型图谱按需下载**：统一安装包内置扩展代码与三大原生运行环境（`MRtrix3 3.0.8`、`FSL 2602.0` 扩散预处理、`FreeSurfer Surface Runtime 0.1.0`）。深度学习模型权重（`DDSeg`、`DDParcel`、`DDSurfer`、`TractCloud`、`PNL`）与大型白质图谱（`ORG`）仍在首次使用时按需下载并缓存至本机。
- **ACT / 5TT 准备路线说明**：SlicerDSpace 原生免 T1 ACT 流程直接由 `DDSeg + DDParcel` 合成 5TT 与灰白质交界掩膜（`tractography/act_prep.py`，经 `5ttcheck` 与 `5tt2gmwmi` 校验），或对分区体积运行 `5ttgen freesurfer`。MRtrix3 上游的 `5ttgen fsl` 模式需要 T1w 输入及完整外部 FSL（`bet`、`fast`、`run_first_all`），SlicerDSpace 不依赖该模式，且仅含 3 个扩散预处理命令（`topup`、`applytopup`、`eddy_cpu`）的内置 FSL 运行包不支持 `5ttgen fsl`。
- **FreeSurfer 许可证要求**：内置的 7 个 FreeSurfer 皮层表面后处理命令（`mris_anatomical_stats`、`mris_ca_label` 等）不包含任何用户专属 FreeSurfer 许可证或激活文件（`activation_key_included: false`）。运行皮层表面后处理时，用户需自备免费的个人 FreeSurfer 许可证文件（`FS_LICENSE`）。
- **掩膜策略与 PyTorch 设备**：各平台优先保留已有、人工检查过的原生掩膜，后端失败不会静默替换算法。PyTorch Auto 优先选择可用 CUDA。在 Apple Silicon 上，打包的 PyTorch `2.2.2`（`x86_64`，Rosetta 2）在 **MPS** 上运行 DDSeg、DDParcel、PNL CNN 掩膜和 TractCloud 推理；DDSurfer 表面体积网络（`Conv3d`）在 CPU 上运行，因为该 PyTorch 构建没有 MPS `Conv3d`；`fp64` 模型拟合（DTI、DKI、NODDI、自由水、MK-Curve）按设计在 CPU 上运行。
- **DKI / MK-Curve 验收范围**：已在多壳被试（`HCP 100307`、`HCP 100408`）上完成全脑 DKI（`WLS`）及整层轴位切片/多层抽样 Python `MK-Curve` 验证；有效体素筛查不等于任意临床采集协议下的全脑 MATLAB 对照。

详见 [Windows 5.10.0 验收报告](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)、[macOS 5.10.0 验收报告](../release/MACOS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)、[Windows 配置指南](WINDOWS.md) 与 [macOS 指南](MACOS.md)。

---

## 上游说明

- **FSL**：FSL 官方在 Windows 上仅提供 WSL 安装路线（[FSL 安装](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html)、[Windows FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)）。SlicerDSpace `v0.1.1` 依据 Oxford University Innovation Academic Licence 内置了扩散预处理所需的 3 个 Windows x64 原生程序（`topup.exe`、`applytopup.exe`、`eddy_cpu.exe`）；扩散预处理以外的完整 FSL 组件（如 `bet`、`fast`、`FIRST`、GPU `eddy_cuda`）不在内置范围内，WSL 2 仅保留为可选回退。
- **FreeSurfer**：FreeSurfer 官方在 Windows 上仅提供 Linux 托管路线（[FreeSurfer 要求](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements)）。SlicerDSpace `v0.1.1` 内置了 `DDSurfer` 皮层后处理所需的 7 个 Windows x64 原生命令，无需 WSL；完整 `recon-all` 不在内置范围内，且仍需用户自备 FreeSurfer 许可证。
- **PyTorch MPS**：PyTorch 支持 Apple MPS（[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)），本插件在 Apple Silicon 上将其用于上文“掩膜策略与 PyTorch 设备”所列的推理路线。
