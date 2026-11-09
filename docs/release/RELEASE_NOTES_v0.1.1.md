# SlicerDSpace v0.1.1

SlicerDSpace `v0.1.1` provides a single all-in-one extension package per platform (bundling the SlicerDSpace module along with the native MRtrix3, FSL preprocessing, and FreeSurfer surface postprocessing runtimes).

## 1. Download

Choose the package for your operating system:

- **Windows (x64)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
- **Linux (x64)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`

> Recommended version: **3D Slicer 5.10.0** (Windows 10/11 x64 or Linux x86_64).

## 2. Installation

1. Open **3D Slicer**.
2. Go to **View -> Extensions Manager** (or click the Extensions Manager icon on the toolbar).
3. Switch to the **Install Extensions** tab, click **Install Extension from File...**, and select the downloaded `SlicerDSpace-v0.1.1-Windows-x64.zip` (or `.tar.gz` on Linux).
4. Restart 3D Slicer when prompted.

## 3. Quick Start

1. Open the **SlicerDSpace** module (**Modules -> Diffusion -> SlicerDSpace**, or press `Ctrl + F` and search `SlicerDSpace`).
2. **First-time setup**: Expand **Environment and installation** and click **Prepare selected analysis environment** to automatically install required Python packages and download deep-learning model weights on demand. Built-in native tools (MRtrix3, FSL, Surface Runtime) are configured automatically without WSL.
3. **Run analysis**:
   - Under **1. Inputs**, select your **Subject root** (containing `input/data.nii.gz`, `input/bvals`, `input/bvecs`; or use **Advanced settings -> Input import / conversion** to convert raw DICOM/NIfTI/NRRD files).
   - Under **2. Products**, check the analyses you want to run (use `...` to choose specific algorithms).
   - Click **Run SlicerDSpace**.

---

## 中文安装与使用说明

### 1. 下载安装包

请根据您的操作系统下载对应的单文件完整安装包（已内置 SlicerDSpace 插件及 MRtrix3、FSL 预处理、FreeSurfer 表面后处理原生运行组件，Windows 下无需配置 WSL）：

- **Windows (64 位)**：`SlicerDSpace-v0.1.1-Windows-x64.zip`
- **Linux (64 位)**：`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`

### 2. 安装步骤

1. 打开 **3D Slicer**（推荐使用 **5.10.0**）。
2. 点击顶部菜单 **View -> Extensions Manager**（扩展管理器）。
3. 点击右上角或底部的 **Install Extension from File...**（从文件安装），选择下载好的压缩包。
4. 安装完成后按提示重启 3D Slicer。

### 3. 快速使用

1. 进入模块：点击顶部放大镜图标（或按 `Ctrl + F`）搜索 **`SlicerDSpace`**，或在 **Modules -> Diffusion -> SlicerDSpace** 中打开。右上角可切换 **中文 / English** 界面。
2. **首次环境准备**：展开底部的 **环境与安装**，点击 **准备所选分析环境**，插件会自动检查并按需下载所需的 Python 依赖与深度学习模型权重。
3. **开始分析**：
   - 在 **1. 输入** 中选择受试者根目录（若原始数据为 DICOM 或普通 NIfTI/NRRD，可在 **高级设置 -> 输入导入 / 格式转换** 中一键转换）。
   - 在 **2. 分析任务** 中勾选需要运行的功能（点击右侧 `...` 可多选切换具体算法）。
   - 点击 **运行 SlicerDSpace** 即可在后台运行并自动加载结果。
