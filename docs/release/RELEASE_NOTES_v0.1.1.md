# SlicerDSpace v0.1.1

> **Recommended Installation**: Open **[https://neurodmri.github.io/SlicerDSpace/#start](https://neurodmri.github.io/SlicerDSpace/#start)** ([中文官网向导](https://neurodmri.github.io/SlicerDSpace/zh.html#start)) to package a lightweight one-click setup archive (`SlicerDSpaceSetup-macOS.zip` / `SlicerDSpaceSetup-Windows.zip` / `SlicerDSpaceSetup-Linux.zip`) or download one directly from the Assets below, then install it in **3D Slicer 5.10.0** via **Extensions Manager -> Install Extension from File...** (with built-in multi-threaded Range resume, mirror/proxy, and offline folder support).

## Package Structure (`v0.1.1`)

To eliminate multi-part manual archive merging for everyday use while keeping all core workflows self-contained, `v0.1.1` provides:

1. **Universal Graphical Setup Bootstrappers (`~13 KB`, 1 per OS — direct GitHub Release Assets)**:
   - **macOS (Apple Silicon `arm64`)**: `SlicerDSpaceSetup-macOS.zip`
   - **Windows (`x86_64`)**: `SlicerDSpaceSetup-Windows.zip`
   - **Linux (`x86_64`)**: `SlicerDSpaceSetup-Linux.zip`
   - Import the downloaded `.zip` in **3D Slicer 5.10.0 -> Extensions Manager -> Install Extension from File...** and restart Slicer. After restart, the **SlicerDSpace Setup** window opens automatically (without auto-starting downloads) so you can inspect the included core package size, select optional WMA atlases or mirror/proxy/offline paths, and click **Start Install**.

2. **Unified OS Base Packages (`~1.4–1.7 GB`, 1 per OS — no manual merge needed)**:
   - **Windows (`x86_64`)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
   - **Linux (`x86_64`)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`
   - **macOS (Apple Silicon `arm64`)**: `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` *(physical Intel `x86_64` Mac hardware is unverified)*
   - Each OS Base package bundles the extension (`depends NA`), Python 3.12 packages (`bundled-python/`: CPU PyTorch, DIPY, AMICO, nibabel, dcm2niix, etc.), companion `UKFTractography` CLI, `bundled-sources/` (`WMQL`, `WMA`, `TractCloud`), native runtimes (`MRtrix3 3.0.8`, `FSL 2602.0`, `FreeSurfer Surface Runtime` with `activation_key_included: false`), and **all non-WMA model weights** (`DDSeg`, `DDParcel`, `DDSurfer` + FreeSurfer `atlas.tar.gz`, `PNL` weights, and `TractCloud`). *(`IITmean_b0_256.nii.gz` stays `upstream-only` from NITRC and is fetched automatically by the installer or on first mask run; personal FreeSurfer `license.txt` is required only for cortical surface postprocessing.)*

3. **Optional Standalone WMA White-Matter Atlases (selected in Setup or hot-installed inside Slicer)**:
   - **`WMA ORG` (Default, 74 tracts)**: `SlicerDSpace-v0.1.1-WMA-ORG.tar.gz` *(single file `< 2.0 GiB`)*
   - **`WMA NABA` (Lifespan atlas, optional)**: `SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001..`
   - **`WMA East-West` (Cross-population atlas, optional)**: `SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001..`

---

## 中文快速安装说明

1. **GitHub Release 通用一键引导包（推荐）**：直接在本页 Assets 下载对应系统的轻量引导包 **`SlicerDSpaceSetup-macOS.zip`** / **`SlicerDSpaceSetup-Windows.zip`** / **`SlicerDSpaceSetup-Linux.zip`**（仅约 13 KB），或在官网向导 **<https://neurodmri.github.io/SlicerDSpace/zh.html#start>** 预配置图谱与镜像后生成下载。在 **3D Slicer 5.10.0 -> Extensions Manager -> Install Extension from File...** 导入该 `.zip` 并重启 Slicer；重启后会自动弹出 **SlicerDSpace Setup** 图形安装窗口（默认不自动开下，便于先核对底座体积、按需勾选 WMA 图谱或设置镜像源/代理/本地离线目录），点击 **`Start Install`** 后即开始多线程分片断点续传下载并自动部署，全程无需命令行。
2. 底座包默认已包含 **基础预处理、PNL 脑掩膜权重、Anatomical Priors 三大深度学习模型（`DDSeg`/`DDParcel`/`DDSurfer`）、微结构、全脑追踪、`WMQL`、`TractCloud` 与连接组**；**WMA 白质图谱**默认仅选中标准 **`ORG` 图谱**（单文件），超大 **`NABA` / `East-West`** 图谱默认不下载，日后在 Slicer 内切换时随时支持一键热补装。
3. **全量底座离线手动安装方式**：也可直接下载对应平台的完整底座包（`.zip` / `.tar.gz`），通过 **3D Slicer 5.10.0 -> Extensions Manager -> Install Extension from File...** 安装并重启，在 **Modules -> Diffusion -> SlicerDSpace** 打开使用。
