# SlicerDSpace v0.1.1

SlicerDSpace `v0.1.1` provides **100% self-contained, offline-ready All-in-One extension packages** for **Windows (`x64`)**, **Linux (`x64`)**, and **macOS (Apple Silicon `arm64`)** on **3D Slicer 5.10.0**. Each package bundles the SlicerDSpace module (`depends NA`), all required Python 3.12 packages (`bundled-python/`: CPU PyTorch, nibabel, DIPY, AMICO, scikit-image, scikit-learn, trimesh, dcm2niix, etc.), all deep-learning model weights and atlases (`bundled-model-assets/`: `DDSeg`, `DDParcel`, `DDSurfer`, `PNL` brain mask, `TractCloud`, FreeSurfer `atlas.tar.gz`, HCP T2 template, and all three extracted WMA white-matter atlases—`ORG`, `East-West`, and `NABA`), pinned algorithm sources (`bundled-sources/`: `WMQL`, `WMA`, `TractCloud`), companion `UKFTractography` CLI, and native binary runtimes (`bundled-runtimes/`: `MRtrix3 3.0.8`, `FSL 2602.0` `topup`/`applytopup`/`eddy_cpu`, and FreeSurfer 7-command surface runtime; no personal FreeSurfer activation key is bundled).

> **Recommended for clinicians**: Use the **[Interactive Website Installer Customizer](https://neurodmri.github.io/SlicerDSpace/#start)** ([中文版](https://neurodmri.github.io/SlicerDSpace/zh.html#start)) to generate a double-click `Install-SlicerDSpace.bat` / `.command` / `.sh` script that configures your 3D Slicer path, selects only the single-file modular weight packs (`Pack-BrainMask-PNL.zip`, `Pack-Anatomy.zip`, `Pack-Surface.zip`, `Pack-TractCloud.zip`, `Pack-WMA-ORG.zip`) you need, or auto-merges All-in-One `.001`, `.002`, ... split volumes—with built-in mirror (`ghfast.top` / `gh-proxy.com`), HTTP/HTTPS proxy, and offline folder support, plus hot in-Slicer on-demand supplement whenever new analyses are ticked later. Because the full All-in-One archives exceed GitHub Releases' **2.0 GiB per-file limit**, each platform full package is split into `.001`, `.002`, ... volumes below. Verify SHA-256 against `SHA256SUMS.txt` (optional).

## 1. Download & Merge Split Volumes

- **Windows (`x64`, All-in-One offline bundle)**:
  Download all `SlicerDSpace-v0.1.1-Windows-x64.zip.001`, `.002`, ... volumes into the same folder, then merge them in **PowerShell** (or right-click `.001` in **7-Zip -> Combine Files...**):
  ```powershell
  cmd /c ("copy /b " + ((Get-ChildItem SlicerDSpace-v0.1.1-Windows-x64.zip.[0-9]* | Sort-Object Name | ForEach-Object Name) -join "+") + " SlicerDSpace-v0.1.1-Windows-x64.zip")
  ```
- **Linux (`x64`, All-in-One offline bundle)**:
  Download all `SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001`, `.002`, ... volumes into the same directory, then merge:
  ```bash
  cat SlicerDSpace-v0.1.1-Linux-x64.tar.gz.[0-9]* > SlicerDSpace-v0.1.1-Linux-x64.tar.gz
  ```
- **macOS (Apple Silicon `arm64`, All-in-One offline bundle)**:
  Download all `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.001`, `.002`, ... volumes into the same directory, then merge:
  ```bash
  cat SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.[0-9]* > SlicerDSpace-v0.1.1-macOS-arm64.tar.gz
  ```

> Recommended version: **3D Slicer 5.10.0** (Windows 10/11 x64, Linux x86_64, or macOS Apple Silicon arm64; physical Intel x86_64 Mac hardware is unverified). Do **not** extract the merged archive before installing.

## 2. Installation

1. Open **3D Slicer 5.10.0**.
2. Go to **View -> Extensions Manager** (or click the Extensions Manager wrench icon on the toolbar).
3. Switch to the **Install Extensions** tab, click **Install Extension from File...**, and select the merged `SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, or `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`.
4. Restart 3D Slicer when prompted.

## 3. Quick Start

1. Open the **SlicerDSpace** module (**Modules -> Diffusion -> SlicerDSpace**, or press `Ctrl + F` / `Cmd + F` and search `SlicerDSpace`).
2. **Ready out of the box (Windows, Linux & macOS arm64)**: All model weights, atlases, Python dependencies, algorithm sources, and native runtimes (`MRtrix3`, `FSL`, `UKFTractography`, `Surface Runtime`) are pre-bundled and ready immediately after restart—no additional model weight downloads, `pip` installation, Conda/WSL/Docker setup, or extra Slicer extensions are needed. *(Note: Optional FreeSurfer cortical surface postprocessing still requires your personal FreeSurfer `license.txt` activation file.)*
3. **Run analysis**:
   - Under **1. Inputs**, select your input format (with automatic layout detection) and **Subject root** (or raw DICOM / NIfTI / NRRD inputs).
   - Under **2. Products**, check the analyses you want to run (use `...` to choose specific algorithms) and select your compute device.
   - Click **Run SlicerDSpace**.

---

## 中文安装与使用说明

**Windows (`x64`)**、**Linux (`x64`)** 与 **macOS (Apple Silicon `arm64`)** 三个平台现已全部提供**全量离线开箱即用一体包（All-in-One）**，包内已完整内置 SlicerDSpace 模块（`depends NA`）、全部 Python 3.12 科学计算依赖（含 CPU 版 PyTorch、nibabel、DIPY、AMICO、dcm2niix 等）、全部深度学习模型权重与图谱（`DDSeg`、`DDParcel`、`DDSurfer`、`PNL` 脑掩膜、`TractCloud` 及全部三套解压就绪的 WMA 白质图谱：`ORG`、`East-West` 与 `NABA`）、算法源码（`WMQL`、`WMA`、`TractCloud`）、`UKFTractography` CLI 以及原生二进制运行组件（`MRtrix3 3.0.8`、`FSL 2602.0` 预处理、FreeSurfer 表面后处理组件；不含任何个人 FreeSurfer 激活密钥）。

此外，在项目官网（<https://neurodmri.github.io/SlicerDSpace/zh.html#start>）提供了面向临床用户的**可视化安装脚本定制器**，支持填写 Slicer 路径、按日常需要勾选单文件模块化功能包（免手工合并分卷）或全自动合并 `.001`、`.002` ... 分卷，内置加速镜像源 / 代理 / 本地离线目录选项，并在 Slicer 内勾选未装模型/图谱时支持随时热补装。如需手工合并离线一体包，完整安装包因超过 GitHub Releases 单文件 `2.0 GiB` 上传限制，已切分为 `.001`、`.002` ... 分卷：

### 1. 下载并合并分卷

- **Windows (`x64`，全量开箱即用包)**：
  将同平台全部 `.001`、`.002` ... 分卷下载到同一文件夹后在 PowerShell 执行（或使用 7-Zip 右键 `.001` 选择“合并文件”）：
  ```powershell
  cmd /c ("copy /b " + ((Get-ChildItem SlicerDSpace-v0.1.1-Windows-x64.zip.[0-9]* | Sort-Object Name | ForEach-Object Name) -join "+") + " SlicerDSpace-v0.1.1-Windows-x64.zip")
  ```
- **Linux (`x64`，全量开箱即用包)**：
  将同平台全部 `.001`、`.002` ... 分卷下载到同一目录后执行：
  ```bash
  cat SlicerDSpace-v0.1.1-Linux-x64.tar.gz.[0-9]* > SlicerDSpace-v0.1.1-Linux-x64.tar.gz
  ```
- **macOS (`Apple Silicon arm64`，全量开箱即用包，Intel 真机硬件未验证)**：
  将同平台全部 `.001`、`.002` ... 分卷下载到同一目录后在终端（Terminal）执行：
  ```bash
  cat SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.[0-9]* > SlicerDSpace-v0.1.1-macOS-arm64.tar.gz
  ```
  *（注：合并还原 `.zip` / `.tar.gz` 后**请勿手动解压**，直接在 3D Slicer 中选择合并后的文件安装即可。）*

### 2. 安装步骤

1. 打开 **3D Slicer**（推荐使用 **5.10.0**）。
2. 点击顶部菜单 **View -> Extensions Manager**（扩展管理器）。
3. 点击 **Install Extension from File...**（从文件安装），选择合并后的平台安装包。
4. 安装完成后按提示重启 3D Slicer。

### 3. 快速使用

1. 进入模块：点击顶部放大镜图标（或按 `Ctrl + F` / `Cmd + F`）搜索 **`SlicerDSpace`**，或在 **Modules -> Diffusion -> SlicerDSpace** 中打开。右上角可切换 **中文 / English** 界面。
2. **全平台开箱即用**：安装重启后无需点击“准备所选分析环境”、无需联网下载模型权重、无需安装额外扩展或配置 Docker/WSL，直接即可运行全流程分析。（注：若勾选可选的“皮层表面后处理”，仍需按 FreeSurfer 官方要求提供您个人的 `license.txt` 激活文件。）
3. **开始分析**：
   - 在 **1. 输入** 中选择输入格式（支持自动检测目录结构）与受试者目录。
   - 在 **2. 分析任务** 中勾选需要运行的功能（点击右侧 `...` 可多选切换具体算法），并选择计算设备。
   - 点击 **运行 SlicerDSpace** 即可在后台运行并自动加载三维与切片结果。
