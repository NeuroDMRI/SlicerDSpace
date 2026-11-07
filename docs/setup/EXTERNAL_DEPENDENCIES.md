<div align="center">

# SlicerDSpace 外部环境与配置说明

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

本文档说明运行 SlicerDSpace 需要用户本地安装或配置哪些外部环境。原则是：**不要把所有后端都视为必装项**。很多功能是可选后端，只有选择对应 target 或 backend 时才需要安装。

Slicer GUI 的 Linux x86_64 用户可先用[插件内自动配置](MANAGED_SETUP.zh-CN.md)：
按需安装 MRtrix/FSL，成功后自动填写路径，无需终端或手动激活环境。
以下 CLI 安装计划和高级路径配置仍适用于其他后端、离线与管理员管理的环境。

---

## 0. 推荐策略：先环境体检，再安装，再运行

插件和统一 CLI 都应在真正执行 pipeline 前先做 dependency doctor：

- 告诉用户当前机器能跑哪些 target。
- 告诉用户哪些 target 会被 blocked。
- 列出缺失的 Python 包、外部命令、环境变量、模型权重和资源文件。
- 自动写出安装计划，而不是跑到一半才失败。

命令行体检示例：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets microstructure,tractography,bundles,connectome \
  --tractography-backend csd_act \
  --connectome-backend mrtrix \
  --doctor
```

执行前 run plan 示例：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets validation,preprocessing,microstructure,tractography,bundles,connectome \
  --plan
```

`--plan` 会先运行 dependency doctor，再写出 `<outputs>/00_run_metadata/<run_name>_run_plan.json`。每个 target 会标记为 `will_run`、`skipped` 或 `blocked`，并记录 `reason_category`：`environment`、`data`、`upstream`、`configuration`、`reuse` 等。

体检报告默认写到：

```text
<subject-root>/outputs/00_run_metadata/environment/
  environment_doctor.json
  dependency_profile.json
  install_missing_python.sh
  install_missing_python.ps1
  install_external_tools.md
```

真正执行时，统一 CLI 默认会先跑同样的环境体检。当前策略是 **partial execution 默认开启**：

- 对 selected targets 逐个判断依赖。
- 依赖齐备的 target 继续运行。
- 缺依赖的 target 写入 orchestration manifest，状态为 `blocked`，并记录明确原因。
- 数据依赖仍然会正常 gate。例如 connectome 即使环境依赖齐备，如果缺 tractogram 或 parcellation，也会在对应步骤标记 `blocked`。

如果希望采用严格模式，即任何 selected target 缺必需依赖就完全停止，可加：

```bash
--strict-environment
```

`--ignore-environment-blockers` 仍保留为兼容旧命令的 deprecated 参数；现在不需要常规使用。

示例：缺 MRtrix3 时，`microstructure` 仍可运行；`tractography`、`saf` 或 MRtrix backend connectome 会被标记为 `blocked`。缺 TractQuerier 时，`bundles` 可被 blocked，但已有 tractogram 和 parcellation 仍可继续用于 connectome。

---

## 0.1 一键安装脚本

仓库提供跨平台包装脚本：

```bash
# Linux
scripts/install_deps_linux.sh --targets microstructure,tractography

# macOS
scripts/install_deps_macos.sh --targets microstructure,tractography

# Windows PowerShell
scripts/install_deps_windows.ps1 --targets microstructure,tractography

# If local policy blocks script execution:
powershell -ExecutionPolicy Bypass -File scripts/install_deps_windows.ps1 --targets microstructure,tractography
```

Each wrapper passes an explicit platform profile (`Linux`, `Darwin`, or
`Windows`) into `scripts/install_dependencies.py`. You can also call the core
planner directly:

```bash
PythonSlicer scripts/install_dependencies.py \
  --all \
  --platform Linux
```

默认行为是**只生成安装计划**。如果要自动安装缺失 Python 包，加 `--execute`：

```bash
scripts/install_deps_linux.sh \
  --targets microstructure,tractography \
  --execute
```

也可以指定 Slicer Python 或其他 Python：

```bash
PYTHON_BIN=/path/to/Slicer/bin/PythonSlicer \
  scripts/install_deps_linux.sh --targets microstructure --execute
```

或直接传给安装器：

```bash
PythonSlicer scripts/install_dependencies.py \
  --targets microstructure,segmentation \
  --python-executable /path/to/Slicer/bin/PythonSlicer \
  --execute
```

注意：

- Python 包可以由脚本安装。
- MRtrix3、FSL、FreeSurfer、WMA、TractCloud、模型数据和 license 不会被脚本静默安装，因为它们涉及管理员权限、平台差异、license、模型下载或用户本地路径选择。
- 这些外部项会写入 `install_external_tools.md`，用户按平台说明安装后重新运行 `--doctor` 即可。

---

## 0.1.1 Dependency profile

`dependency_profile.json` 是“路径配置 + 本机扫描结果”的组合，不是跨机器
免检通行证。它包含：

- 当前平台和 Python executable。
- `environment_doctor.json` 的完整扫描结果。
- Python 安装 helper 和外部工具 notes 路径。
- portability policy：导入后必须在当前机器重新扫描。

GUI 的 `Import dependency config` 会读取 `external_dependencies`，保存到
Slicer user settings，然后自动重新运行环境扫描。这样换机器时可以复用路径
提示，但不会把旧机器的扫描结果当作当前机器事实。

CLI 仍然支持同一份 JSON：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets tractography,bundles \
  --doctor \
  --dependency-config ~/.slicerdsurfer-deps.json
```

如果导入 profile 后 doctor 仍显示 `not found`，以新的 doctor 结果为准，修正
路径或环境变量后再次扫描。

---

## 0.1.2 Linux/macOS/Windows dependency profile matrix

发布前或给用户排查环境时，可以生成三平台 dependency profile 矩阵：

```bash
PythonSlicer scripts/dependency_profile_matrix.py \
  --output-dir install_plan/dependency_profile_matrix
```

默认矩阵覆盖完整依赖面，包括常规产品、`surface_postprocess`、
`microstructure_freewater`、`microstructure_noddi`、`saf`、`wma` 和
`tractcloud`。默认 backend profile 会选择更严格的外部依赖组合：
`tractography=csd_act`、`connectome=mrtrix`；预处理只有一条路线（DIPY MP-PCA、DIPY Gibbs、FSL eddy/topup），不再有 backend 选项。

输出结构：

```text
install_plan/dependency_profile_matrix/
  dependency_profile_matrix.json
  linux/
    environment_doctor.json
    dependency_profile.json
    install_missing_python.sh
    install_missing_python.ps1
    install_external_tools.md
  darwin/
    ...
  windows/
    ...
```

注意：如果在 Linux 主机上生成 Darwin/Windows 行，这些行是
**platform-rule dry-run**，不是原生 macOS/Windows 通过证明。它们会使用对应
平台的常见路径、命令后缀和 install hint，但文件系统可用性仍然只能反映当前
主机。公开发布前仍应在真实 macOS/Windows 设备上运行对应 wrapper 或核心
planner。

---

## 0.2 用户设备与 Python 环境

SlicerDSpace 需要区分三件事：

- Slicer GUI 使用的是 Slicer Python。
- `run_subject_pipeline.py` 默认检查和运行的是当前 shell 的 Python。
- 完整 subject pipeline 需要 Slicer CLI 模块时，应通过 `Slicer --launch PythonSlicer` 启动统一入口。
- 外部后端，例如 TractCloud，可以使用单独的 Python。

因此，如果目标是让 Slicer 插件内运行，请优先用 Slicer Python 做检查和安装：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets segmentation,surface \
  --doctor \
  --doctor-python-executable /path/to/Slicer/bin/PythonSlicer
```

完整执行推荐命令形式：

```bash
/path/to/Slicer --launch PythonSlicer \
  /path/to/SlicerDSpace/run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets validation,preprocessing,segmentation,parcellation,surface,microstructure,tractography,bundles,connectome \
  --execute
```

直接用系统 `python3 run_subject_pipeline.py --execute` 不能运行张量拟合：它需要 Slicer 内核中的 Teem 库，环境检查会报告 `native_tensor_fit` 缺失，并提示使用 Slicer 的 `PythonSlicer`。
不要通过 `pip install slicer` 处理该提示；应改用 3D Slicer 或其 `bin/PythonSlicer`。

设备策略：

- 默认使用 `auto`，即有可用 CUDA GPU 时使用 GPU，否则回退 CPU。
- `auto` 只会解析成一个具体设备，例如 `cuda:0` 或 `cpu`，不会同时跑 CPU 和 GPU 两套结果。
- 如果用户选择 `cuda` 或 `cuda:0`，dependency doctor 会在目标 Python 中检查 `torch.cuda.is_available()` 和 GPU 数量。
- 如果 CUDA 不可用，目标会在运行前被标记为 `blocked`，而不是运行到深度学习推理阶段才失败。
- MRtrix3、FSL、FreeSurfer 通常仍是 CPU 命令行工具；GPU 检查主要影响 PyTorch/DDSurfer/DDSeg/TractCloud。

MRtrix3 特别说明：

- MRtrix3 仍作为外部 backend，不打包进 SlicerDSpace。
- `dwi2response` 等 MRtrix Python 脚本通过 `#!/usr/bin/env python` 启动；如果继承 Slicer 的 `PYTHONHOME` / `PYTHONPATH`，会在 `PythonSlicer` 里出现 Python 标准库冲突。
- SlicerDSpace 的 MRtrix CSD/ACT backend 会在执行 MRtrix 命令前隔离这些 Slicer Python 变量，并把用户配置的 MRtrix `bin` 放到 `PATH` 最前面。
- 用户仍需要通过 `--mrtrix-bin-dir`、`SLICERDSPACE_MRTRIX_BIN_DIR` 或 `PATH` 提供本机 MRtrix3。

---

## 0.3 外部路径发现顺序

SlicerDSpace 不要求用户“上传依赖”。外部软件和数据采用以下发现顺序：

1. 用户保存的路径：Slicer UI 中 `Environment & Dependencies` 的路径输入框，或 CLI 的 `--dependency-config`。
2. 环境变量：例如 `SLICERDSPACE_MRTRIX_BIN_DIR`、`FSLDIR`、`FREESURFER_HOME`。
3. `PATH`：适用于 `mrconvert`、`tckgen`、`eddy`、`recon-all` 等命令。
4. 常见系统路径：Linux、macOS 和 Windows/WSL 的 MRtrix3、FSL、FreeSurfer、Slicer、WMA、TractCloud 常见安装位置。

在 Windows 上，命令扫描会同时检查无扩展名命令和 `.exe`、`.cmd`、`.bat`
变体。例如配置 `C:\Program Files\MRtrix3\bin` 后，`mrconvert.exe` 会被识别为
`mrconvert`。

体检结果会在 `environment_doctor.json` 的每个 dependency 下写出：

- `details.detected_source`: `user setting`、`env:...`、`PATH`、`common path`、`not found` 等。
- `details.resolved`: 实际命中的命令路径。
- `details.checked_locations`: 对命令类依赖检查过的显式路径。

Slicer UI 中的路径输入框会保存到 Slicer user settings，下次打开 SlicerDSpace 会自动恢复。

CLI 可保存和复用同样的外部路径配置：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets tractography,bundles \
  --doctor \
  --mrtrix-bin-dir /path/to/mrtrix3/bin \
  --tract-querier-root /path/to/tract_querier \
  --save-dependency-config ~/.slicerdsurfer-deps.json

PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets tractography,bundles \
  --doctor \
  --dependency-config ~/.slicerdsurfer-deps.json
```

---

## 1. 最小可用环境

如果只需要启动 Slicer 插件、做输入检查、运行已有 Python pipeline 的轻量部分，至少需要：

- 3D Slicer
  - 推荐使用 Slicer 5.10 系列；本机验证路径为 `/data/chengjin/software/Slicer-5.10.0-linux-amd64`。
  - GUI 加载插件时不要把整个仓库根目录直接作为 Slicer module path；开发 smoke 应使用只包含 `SlicerDSpace.py` 和 `Resources/` 的临时 module 目录。
- Slicer PyTorch Extension
  - DDSurfer / DDParcel / DDSeg 的深度学习推理依赖 PyTorch。
  - 插件代码会通过 Slicer 的 `PyTorchUtils` 检查/安装 `torch>=2.0.0`，但用户仍需要先在 Slicer Extension Manager 安装 PyTorch 扩展。
- Git LFS
  - 模型权重和部分大文件走 Git LFS。
  - 克隆后推荐用统一资源下载脚本，而不是手动记路径：

```bash
git lfs install
scripts/download_model_assets.sh --profile core
```

  - `core` 会下载 DDSurfer / DDSeg / DDParcel 所需模型权重和模板。
  - 只跑某个深度学习后端时，可改用 `--profile surface`、`--profile segmentation` 或 `--profile parcellation`。
  - 详细 profile、路径和外部 asset 覆盖方式见 [MODEL_ASSETS.md](../assets/MODEL_ASSETS.md)。

---

## 2. Python 包依赖

这些包需要安装在实际运行 pipeline 的 Python 环境中。GUI 内运行时通常是 Slicer Python；命令行运行 `run_subject_pipeline.py` 时是当前 shell 的 Python。

### 2.1 公共基础包

- `numpy`
- `scipy`
- `nibabel>=5.0`
- `SimpleITK`
- `trimesh`
- `rtree`
- `tqdm`
- `networkx`
- `pynrrd`
- `h5py`
- `scikit-image`
- `plumbum`
- `vtk`

### 2.2 深度学习相关

- `torch>=2.0.0`
- `torchvision`

建议在 Slicer 内通过 PyTorch Extension 管理 PyTorch，而不是手动混装到 Slicer Python。

### 2.3 Microstructure

- DTI / Free Water / preprocessing（DIPY MP-PCA 和 Gibbs）：
  - `dipy>=1.10,<1.13` (DIPY 1.10 is the compatible choice for Slicer 5.6 / Python 3.9)
- NODDI：
  - `dmri-amico`，Python import 名为 `amico`

说明：
- NODDI 不依赖 Free Water 结果。
- Free Water / NODDI 在统一入口中是显式 gate，不是默认必跑项。

### 2.4 SAF Python backend

SAF 转写后端已内置在 `tractography/saf_backend/`，但仍需要：

- `numpy`
- `scipy`
- `scikit-learn`，Python import 名为 `sklearn`
- `nibabel`
- MRtrix3 命令，见第 4 节

SAF 只接受已经位于原生 dMRI 空间的表面。DDSurfer postprocess 是默认表面来源；
插件不提供 T1w/FreeSurfer 到 dMRI 的 ANTs 配准路线。

---

## 3. Slicer 相关配置

### 3.1 Slicer executable

统一 orchestrator 和 WMA wrapper 可通过以下方式找到 Slicer：

- CLI 参数：`--slicer-executable /path/to/Slicer`
- 环境变量：`SLICERDSPACE_SLICER_EXECUTABLE=/path/to/Slicer`

本机例子：

```bash
export SLICERDSPACE_SLICER_EXECUTABLE=/data/chengjin/software/Slicer-5.10.0-linux-amd64/Slicer
```

### 3.2 SlicerDMRI

张量拟合不再需要 SlicerDMRI：它由 Slicer 内核中的 Teem 库完成（`shared/teem_dti.py`）。
勾选 DTI、微结构、追踪等目标时，准备步骤不会要求 SlicerDMRI。以下场景仍使用 SlicerDMRI：

- 勾选 WMA 时，准备步骤会请求 SlicerDMRI（WMA 步骤仍加载它的模块）。
- 在 Slicer scene 中直接加载/显示 fiber bundle。
- WMA diffusion measures 导出时使用 `FiberTractMeasurements` CLI。
- DICOM 导入时，`DWIConvert` 是 dcm2niix 和 MRtrix3 `mrconvert` 之外的可选转换器。

UKF 的错误提示同时提到 UKFTractography 和 SlicerDMRI；准备步骤只为 UKF 请求 UKFTractography。

---

## 4. MRtrix3

MRtrix3 是外部后端，插件不打包 MRtrix3。

### 4.1 用途

MRtrix3 用于：

- CSD/FOD/ACT tractography
- SIFT2
- `tck2connectome`
- SAF 的 tractogram 生成、streamline count 和 `SAF.tck -> SAF.vtk`
- 可选 MRtrix/FSL DWI preprocessing wrapper

### 4.2 需要的命令

CSD/FOD/ACT 主线：

- `mrconvert`
- `dwi2response`
- `dwi2fod`
- `mtnormalise`
- `tckgen`
- `tcksift2`

SAF / connectome 常用：

- `tckstats`
- `tckconvert`
- `tcksample`，可选
- `tck2connectome`

preprocessing（唯一路线；去噪和 Gibbs 去除由 DIPY 完成，不再使用 `dwidenoise` 和 `mrdegibbs`）：

- `dwifslpreproc`
- `dwi2mask`

### 4.3 配置方式

优先用 CLI 显式传入：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --execute \
  --mrtrix-bin-dir /path/to/mrtrix3/bin
```

或使用环境变量：

```bash
export SLICERDSPACE_MRTRIX_BIN_DIR=/path/to/mrtrix3/bin
export MRTRIX_BIN_DIR=/path/to/mrtrix3/bin
```

也可以直接把 MRtrix3 加入 `PATH`。

---

## 5. FSL

FSL 只在选择 `preprocessing` 目标时需要（W2a）：预处理只有一条路线，即 DIPY MP-PCA、DIPY Gibbs 去除，然后通过 `dwifslpreproc` 运行 FSL eddy / topup。该目标还需要 MRtrix3 和 FSL 许可。没有 FSL 构建的平台（例如 Linux aarch64）不提供 `preprocessing`，请导入已预处理的数据。

需要的命令或工具：

- `eddy`
- `topup`
- MRtrix3 的 `dwifslpreproc` 和 `dwi2mask`

配置方式：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets preprocessing \
  --execute \
  --preprocessing-mrtrix-bin-dir /path/to/mrtrix3/bin \
  --preprocessing-fsl-bin-dir /path/to/fsl/bin \
  --preprocessing-reverse-phase-b0 /path/to/reverse_b0.nii.gz \
  --preprocessing-phase-encoding-direction AP \
  --preprocessing-readout-time 0.05 \
  --preprocessing-eddy-threads 8
```

相位编码方向是必需的：先取 `--preprocessing-phase-encoding-direction`，否则取 `input/dwi.json` 的 `PhaseEncodingDirection`（按图像方向换算为 AP、PA、LR、RL、IS、SI）；两者都没有时，运行被阻止，不会猜测（大多数轴位临床 DWI 为 AP）。读出时间先取用户值，再取 sidecar 的 `TotalReadoutTime`；都没有时 dwifslpreproc 使用 0.1 s，并记为 `readout_assumed`。

未指定 `--preprocessing-eddy-threads` 时，CPU eddy 使用当前进程可用 CPU
数量，上限为 8。若 `dwifslpreproc` 报告 diffusion-direction sampling strongly
asymmetric，可按其建议添加 `--preprocessing-eddy-slm linear`；不要在没有该
警告或方法学依据时任意改变 EC model。

FSL 校正完成后，最终脑掩膜由 PNL CNN 基于校正后的 DWI 和旋转后的
梯度重新生成（选择“仅使用提供的掩膜”时复制该掩膜），不复用校正前输入掩膜。MRtrix `dwi2mask` 只由 `dwifslpreproc` 内部使用。恢复运行会自动修复缺失或时间早于最终
DWI 的旧掩膜。

预处理只有一条路线，没有“不需要 FSL”的 DIPY 路线；需要导入已预处理数据时，请不要选择 `preprocessing` 目标。

---

## 6. FreeSurfer

DDSurfer postprocess 生成 FreeSurfer-style `surf/label/stats`，当前原生 OBJ 路线可使用
本机从固定源码编译的七个原始组件，**不强制安装完整 FreeSurfer**。已有完整安装仍可复用。
组件并非 Python 重写，仍适用 FreeSurfer 许可及个人激活要求。
详见[表面组件与首次准备](SURFACE_COMPONENTS.zh-CN.md)。

### 6.1 用途

使用这些组件或已有 FreeSurfer 的功能：

- DDSurfer postprocess
- surface annotation 投射
- stats 计算
- fsaverage resampling

SAF 不直接调用 FreeSurfer 命令，但通常会消费 DDSurfer postprocess 生成的
FreeSurfer-style subject 目录。

### 6.2 需要的资源/命令

本机组件路线需要有效组件目录和个人激活文件。完整安装兼容路线需要：

- `FREESURFER_HOME`
- `FREESURFER_HOME/subjects/fsaverage`
- FreeSurfer license
- `FreeSurferColorLUT.txt`

当前 surface-only postprocess 检查或调用的命令包括：

- `mris_sphere`
- `mris_curvature`
- `mris_inflate`
- `mris_register`
- `mri_label2label`
- `mri_surf2surf`
- `mris_place_surface`；旧版安装不支持时需要 `mris_thickness`

区域标签和 surface-only 区域统计在 Python 内完成，fsaverage 的四个指标一次映射。
不再将 `recon-all`、`mri_annotation2label`、`mri_segstats` 作为本路线的必要条件。
这不是完整 `recon-all` 或 `mris_anatomical_stats` 的替代实现；不会新增灰质体积或 eTIV。

### 6.3 配置方式

可在 Slicer 的环境与安装中配置表面组件目录和激活文件，也可使用
`--postprocess-runtime`、`--postprocess-license-file`，或 `SLICERDSPACE_SURFACE_RUNTIME`
与 `FS_LICENSE`。缺少组件时明确确认后可后台编译；环境检查和分析本身不会下载或编译。

已有完整安装可继续配置：

```bash
export FREESURFER_HOME=/path/to/freesurfer
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FREESURFER_LICENSE=/path/to/license.txt
```

也可以通过 CLI 参数覆盖：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets surface_postprocess \
  --execute \
  --postprocess-freesurfer-home /path/to/freesurfer
```

---

## 7. SAF 表面空间约束

SAF 的白质面和软脑膜面必须已经与 FOD、掩膜和待分析 tractogram 位于同一
原生 dMRI 空间。SlicerDSpace 不再提供 ANTs 驱动的 T1w/FreeSurfer 到 dMRI
配准分支，也不会扫描或安装 ANTs。使用外部表面时，应在导入前验证空间一致性；
推荐直接使用 DDSurfer postprocess 输出。

---

## 8. WMA / WhiteMatterAnalysis

WMA 是可选外部 bundle backend，插件不把 WMA 作为内置运行依赖。

### 8.1 需要的软件和数据

- 外部 WhiteMatterAnalysis checkout
- ORG atlas 目录，必须包含：
  - `ORG-RegAtlas-100HCP`
  - `ORG-800FC-100HCP`
- 原生 RAS 坐标中的全脑 tractogram（`.vtk`、`.vtp` 或 `.tck`），并具有适合
  ORG atlas 的长程白质覆盖。推荐使用 UKF 全脑 tractography；`.tck` 输入会通过
  配置的 MRtrix3 `tckconvert` 自动转换。SAF 短联络纤维、已拆分的单束文件以及
  为 ACT/connectome 优化但长程覆盖不足的 tractogram 不适合作为 ORG 输入
- Slicer executable
- 可选：SlicerDMRI `FiberTractMeasurements` CLI，用于 diffusion measures 导出
- headless 环境可选：`xvfb-run` 或可用 `DISPLAY`

### 8.2 WMA discovery

配置方式任选其一：

- CLI：`--wma-root /path/to/whitematteranalysis`
- 环境变量：`SLICERDSPACE_WMA_ROOT=/path/to/whitematteranalysis`
- 环境变量：`WMA_ROOT=/path/to/whitematteranalysis`
- 把 `wm_apply_ORG_atlas_to_subject.sh` 加入 `PATH`

其他常用参数：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets wma \
  --execute \
  --wma-input-tractography /path/to/tractogram.vtk \
  --wma-atlas-dir /path/to/ORG-Atlases \
  --wma-root /path/to/whitematteranalysis \
  --slicer-executable /path/to/Slicer \
  --wma-use-virtual-x
```

自动发现只接受文件名明确包含 `ukf`、`wma_input` 或 `org_input` 的候选文件，
并排除 `saf/`、`bundles/`、WMA、WMQL 和 TractCloud 派生目录。普通 CSD/ACT
tractogram 不会被静默复用：代表性受试者实测表明，增加其长程流线数量仍未满足
ORG 解剖束覆盖 QC。其他算法生成的全脑 tractogram 可通过
`--wma-input-tractography` 显式选择，运行后仍须通过非空解剖束检查。

---

## 9. WMQL / Tract Querier

WMQL 是规则化 bundle 提取后端，默认使用随扩展打包的未修改作者多束规则，执行 48 个输出查询。
Advanced 可选束子集；空束和低采样束单独记录，不宣称全束定量通过。见 [WMQL](WMQL.zh-CN.md)。

### 9.1 需要的软件

- `tract_querier` Python package 或 tract_querier checkout
- `nibabel`
- `numpy`

输入 tractogram 支持：

- `.vtk`
- `.vtp`
- `.trk`

### 9.2 配置方式

可通过以下方式指定：

- CLI：`--tract-querier-root /path/to/tract_querier`
- 环境变量：`SLICERDSPACE_TRACT_QUERIER_ROOT=/path/to/tract_querier`
- 环境变量：`TRACT_QUERIER_ROOT=/path/to/tract_querier`

如果 `tract_querier` 已经在当前 Python 和 `PATH` 中可用，可以不额外配置。

---

## 10. TractCloud

TractCloud 是可选外部 deep-learning bundle backend。

### 10.1 需要的软件和数据

- TractCloud source checkout 或已安装的 `tractcloud` Python package
- `torch`
- TractCloud 模型/数据目录
- 输入 tractogram：`.vtk` 或 `.vtp`

### 10.2 配置方式

- CLI：`--tractcloud-root /path/to/TractCloud`
- CLI：`--tractcloud-python-executable /path/to/python`
- CLI：`--tractcloud-data-dir /path/to/model-data`
- 环境变量：`SLICERDSPACE_TRACTCLOUD_ROOT`
- 环境变量：`TRACTCLOUD_ROOT`
- 环境变量：`SLICERDSPACE_TRACTCLOUD_PYTHON_EXECUTABLE`
- 环境变量：`TRACTCLOUD_PYTHON_EXECUTABLE`
- 环境变量：`SLICERDSPACE_TRACTCLOUD_DATA_DIR`
- 环境变量：`TRACTCLOUD_DATA_DIR`

示例：

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets tractcloud \
  --execute \
  --tractcloud-input-tractography /path/to/tractogram.vtp \
  --tractcloud-root /path/to/TractCloud \
  --tractcloud-data-dir /path/to/tractcloud-data \
  --tractcloud-device cuda
```

---

## 11. Docker

Docker 不是当前主线必需项。

仅在使用 legacy helper `run_postprocess_docker.py` 或外部容器化 workflow 时需要：

- Docker
- FreeSurfer license
- 对应 docker image 或 postprocess-release repo

当前内置 DDSurfer postprocess backend 默认不要求 Docker。

---

## 12. 模型权重与资源文件

推荐入口：

```bash
scripts/download_model_assets.sh --profile core
```

也可以按需下载：

```bash
scripts/download_model_assets.sh --profile surface
scripts/download_model_assets.sh --profile segmentation
scripts/download_model_assets.sh --profile parcellation
```

完整说明见 [MODEL_ASSETS.md](../assets/MODEL_ASSETS.md)。

以下资源必须存在，否则对应深度学习流程会报错：

- DDSurfer checkpoints：
  - `Resources/Data/ckpts/hcp/*.pt`
  - 或通过 `DDSURFER_CHECKPOINT_DIR` 指向外部目录
- DDSurfer template：
  - `Resources/Data/template/100307-FA.nii.gz`
  - 或通过 `DDSURFER_TEMPLATE_DIR` 指向外部目录
- DDSurfer registration atlas：
  - `Resources/Data/100HCP-population-mean-T2.nii.gz`
  - 或通过 `DDSURFER_REGISTRATION_ATLAS` 指向外部文件
- DDParcel weights：
  - `parcellation/weights/*.pkl`
- DDSeg weights：
  - `segmentation/weights/*.pt`

---

## 13. 主要环境变量汇总

| 变量 | 用途 |
| --- | --- |
| `DDSURFER_TEMPLATE_DIR` | DDSurfer template 目录 |
| `DDSURFER_CHECKPOINT_DIR` | DDSurfer checkpoint 根目录 |
| `DDSURFER_REGISTRATION_ATLAS` | DDSurfer/DDParcel 注册 atlas |
| `SLICERDSPACE_SLICER_EXECUTABLE` | Slicer executable |
| `SLICERDSPACE_MRTRIX_BIN_DIR` / `MRTRIX_BIN_DIR` | MRtrix3 `bin` 目录 |
| `SLICERDSPACE_MRTRIX_HOME` / `MRTRIX_HOME` / `MRTRIX3_HOME` | MRtrix3 安装根目录 |
| `SLICERDSPACE_FSL_BIN_DIR` / `FSL_BIN_DIR` / `FSLDIR` | FSL 安装或 `bin` 目录 |
| `SLICERDSPACE_FREESURFER_HOME` / `FREESURFER_HOME` | FreeSurfer 安装目录 |
| `SLICERDSPACE_SURFACE_RUNTIME` | 可选本机源码编译的表面组件目录 |
| `FS_LICENSE` | 本机组件或完整安装使用的个人激活文件 |
| `FREESURFER_LICENSE` | FreeSurfer license |
| `SLICERDSPACE_WMA_ROOT` / `WMA_ROOT` | WhiteMatterAnalysis checkout |
| `SLICERDSPACE_WMA_ATLAS_DIR` / `WMA_ATLAS_DIR` | WMA ORG atlas 目录 |
| `SLICERDSPACE_FIBER_TRACT_MEASUREMENTS_CLI` | SlicerDMRI FiberTractMeasurements CLI |
| `SLICERDSPACE_DISPLAY` / `DISPLAY` | WMA/Slicer headless GUI display |
| `SLICERDSPACE_TRACT_QUERIER_ROOT` / `TRACT_QUERIER_ROOT` | TractQuerier checkout |
| `SLICERDSPACE_TRACTCLOUD_ROOT` / `TRACTCLOUD_ROOT` | TractCloud checkout |
| `SLICERDSPACE_TRACTCLOUD_PYTHON_EXECUTABLE` / `TRACTCLOUD_PYTHON_EXECUTABLE` | TractCloud Python |
| `SLICERDSPACE_TRACTCLOUD_DATA_DIR` / `TRACTCLOUD_DATA_DIR` | TractCloud 模型/数据目录 |

---

## 14. 按目标功能的依赖速查

| 目标 | 必需依赖 | 可选/说明 |
| --- | --- | --- |
| 启动 Slicer GUI | 3D Slicer, PyTorch Extension | SlicerDMRI 用于 WMA 和 fiber bundle 显示 |
| Input validation | `nibabel`, `SimpleITK` | 可检查 MRtrix/FSL/Microstructure 依赖状态 |
| DDSurfer surface | PyTorch, `torchvision`, DDSurfer weights/templates, `SimpleITK`, `trimesh` | GPU 可选 |
| DDSurfer postprocess | 本机源码组件或已有 FreeSurfer、个人许可证、`nibabel`、`numpy`、`trimesh`、`rtree` | 当前原生 OBJ 路线不需完整版；不需要外部 T1w，可用 FA/b0 生成 `brain.mgz` |
| DDParcel | PyTorch, `torchvision`, `nibabel`, `pynrrd`, `h5py`, `scikit-image`, `SimpleITK`, weights | Slicer CLI DTI extraction |
| DDSeg | PyTorch, `torchvision`, `nibabel`, `pynrrd`, weights | GPU 可选 |
| Microstructure DTI | Slicer 内核 Teem（`PythonSlicer`）、`dipy`、`nibabel`、`numpy` | 同时输出彩色 FA 图；ROI-wise 需要 parcellation label map |
| Free Water | `dipy` | 显式启用 |
| NODDI | `dmri-amico` | 显式启用 |
| preprocessing（唯一路线） | `dipy`, `nibabel`, `numpy`; MRtrix3, FSL | DIPY 做 MP-PCA 和 Gibbs；需要 `dwifslpreproc`, `dwi2mask`, `eddy`, `topup` 和 FSL 许可 |
| CSD/FOD/ACT | MRtrix3, 5TT, GMWMI | 当前 5TT/GMWMI 可由标签近似生成 |
| SAF | 内置 SAF backend, MRtrix3, DDSurfer postprocess surface | 表面必须位于原生 dMRI 空间 |
| WMQL | Tract Querier, `nibabel`, `numpy` | 默认官方多束规则，支持束子集 |
| WMA ORG atlas | external WMA, ORG atlas, Slicer | 可选 xvfb/SlicerDMRI measurements |
| TractCloud | TractCloud, `torch`, model data | 当前是可选后端 |
| Connectome | Python 或 MRtrix3 `tck2connectome`（默认 `auto`：`.tck` 来源且指标为流线计数时用 MRtrix，其他情况用 Python） | 两个后端默认采用 4 mm 径向端点分配、FreeSurfer/DDParcel 灰质节点和零对角矩阵；Python 支持 `.trk/.tck/.vtk/.vtp` |

---

## 15. 推荐安装策略

建议分阶段安装：

1. 先安装 Slicer、PyTorch Extension、Git LFS 和公共 Python 包。
2. 拉取模型权重，确认 DDSurfer/DDParcel/DDSeg 资源完整。
3. 如需 Microstructure，安装 DIPY 和 AMICO。
4. 如需 CSD/ACT/SAF/connectome MRtrix 后端，安装 MRtrix3 并配置 `SLICERDSPACE_MRTRIX_BIN_DIR`。
5. 如需高质量 preprocessing，安装 FSL 并配置 FSL bin。
6. 如需 DDSurfer postprocess，可按[组件指南](SURFACE_COMPONENTS.zh-CN.md)准备本机组件及个人激活文件；已有 FreeSurfer 仍可使用，历史 STL/shell 路线有额外依赖。
7. 如需 bundle 后端，再分别安装 WMA / Tract Querier / TractCloud。

不要在不需要对应功能时强制安装所有可选后端；统一命令会通过 manifest/QC 报告当前缺少的依赖和 blocked reason。
