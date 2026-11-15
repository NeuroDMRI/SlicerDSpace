# SlicerDSpace v0.1.1

SlicerDSpace `v0.1.1` provides self-contained, offline-ready extension packages per platform. The **Windows (`x64`)** and **Linux (`x64`)** packages bundle the SlicerDSpace module together with all required Python 3.12 packages (including CPU PyTorch, nibabel, DIPY, AMICO, scikit-image, scikit-learn, trimesh, dcm2niix, etc.), all deep-learning model weights and atlases (`DDSeg`, `DDParcel`, `DDSurfer`, `PNL` brain mask, `TractCloud`, and extracted `ORG` white-matter atlas), pinned algorithm sources (`WMQL`, `WMA`, `TractCloud`), `UKFTractography` CLI, and native binary runtimes (`MRtrix3 3.0.8`, `FSL 2602.0` preprocessing, and FreeSurfer surface postprocessing runtime; no personal FreeSurfer activation key is bundled).

## 1. Download

Choose the package for your operating system:

- **Windows (x64, All-in-One offline bundle)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
- **Linux (x64, All-in-One offline bundle)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`
- **macOS (Apple Silicon arm64)**: `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`

> Recommended version: **3D Slicer 5.10.0** (Windows 10/11 x64, Linux x86_64, or macOS Apple Silicon arm64; physical Intel x86_64 Mac hardware is unverified).

## 2. Installation

1. Open **3D Slicer 5.10.0**.
2. Go to **View -> Extensions Manager** (or click the Extensions Manager icon on the toolbar).
3. Switch to the **Install Extensions** tab, click **Install Extension from File...**, and select the downloaded `SlicerDSpace-v0.1.1-Windows-x64.zip` (or `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` on Linux / `.tar.gz` on macOS).
4. Restart 3D Slicer when prompted.

## 3. Quick Start

1. Open the **SlicerDSpace** module (**Modules -> Diffusion -> SlicerDSpace**, or press `Ctrl + F` / `Cmd + F` and search `SlicerDSpace`).
2. **Ready out of the box (Windows & Linux)**: All model weights, atlases, Python dependencies, algorithm sources, and native runtimes (MRtrix3, FSL, UKFTractography, Surface Runtime) are pre-bundled and ready immediately after restart—no additional model weight downloads, `pip` installation, Conda/WSL setup, or extra Slicer extensions are needed. *(Note: Optional FreeSurfer cortical surface postprocessing still requires your personal FreeSurfer `license.txt` activation file; on macOS, use **Environment and installation -> Prepare selected analysis environment** on first use to fetch Python packages and model weights.)*
3. **Run analysis**:
   - Under **1. Inputs**, select your input format (with automatic layout detection) and **Subject root** (or raw DICOM / NIfTI / NRRD inputs).
   - Under **2. Products**, check the analyses you want to run (use `...` to choose specific algorithms) and select your compute device.
   - Click **Run SlicerDSpace**.

---

## 中文安装与使用说明

### 1. 下载安装包

请根据您的操作系统下载对应的单文件安装包。**Windows (`x64`)** 与 **Linux (`x64`)** 版本现已升级为**全量离线一体包（All-in-One）**，包内已完整内置 SlicerDSpace 模块、全部 Python 3.12 依赖库（含 CPU 版 PyTorch、nibabel、DIPY、AMICO、dcm2niix 等）、全部深度学习模型权重与图谱（`DDSeg`、`DDParcel`、`DDSurfer`、`PNL` 脑掩膜、`TractCloud` 及解压就绪的 `ORG` 白质图谱）、算法源码（`WMQL`、`WMA`、`TractCloud`）、`UKFTractography` CLI 以及原生二进制运行组件（`MRtrix3 3.0.8`、`FSL 2602.0` 预处理、FreeSurfer 表面后处理组件；不含个人 FreeSurfer 激活密钥），安装重启后即可开箱即用，无需再拉取模型权重或配置外部环境：

- **Windows (64 位，全量开箱即用包)**：`SlicerDSpace-v0.1.1-Windows-x64.zip`
- **Linux (64 位，全量开箱即用包)**：`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`
- **macOS (Apple Silicon arm64)**：`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`

### 2. 安装步骤

1. 打开 **3D Slicer**（推荐使用 **5.10.0**）。
2. 点击顶部菜单 **View -> Extensions Manager**（扩展管理器）。
3. 点击 **Install Extension from File...**（从文件安装），选择下载好的压缩包。
4. 安装完成后按提示重启 3D Slicer。

### 3. 快速使用

1. 进入模块：点击顶部放大镜图标（或按 `Ctrl + F`）搜索 **`SlicerDSpace`**，或在 **Modules -> Diffusion -> SlicerDSpace** 中打开。右上角可切换 **中文 / English** 界面。
2. **开箱即用（Windows 与 Linux）**：安装重启后无需点击“准备所选分析环境”、无需联网下载模型权重或安装额外扩展，直接即可运行全流程分析。（注：若勾选可选的“皮层表面后处理”，仍需按 FreeSurfer 官方要求提供您个人的 `license.txt` 激活文件；macOS 版本首次使用时可在 **环境与安装** 中一键准备依赖与权重。）
3. **开始分析**：
   - 在 **1. 输入** 中选择输入格式（支持自动检测目录结构）与受试者目录。
   - 在 **2. 分析任务** 中勾选需要运行的功能（点击右侧 `...` 可多选切换具体算法），并选择计算设备。
   - 点击 **运行 SlicerDSpace** 即可在后台运行并自动加载三维与切片结果。
