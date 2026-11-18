# SlicerDSpace v0.1.1

> **Recommended Installation**: Open **[https://neurodmri.github.io/SlicerDSpace/#start](https://neurodmri.github.io/SlicerDSpace/#start)** ([中文官网向导](https://neurodmri.github.io/SlicerDSpace/zh.html#start)) to generate a one-click custom installer script (`Install-SlicerDSpace.bat` / `.command` / `.sh`) with built-in mirror (`ghfast.top` / `gh-proxy.com`), proxy, and offline folder support.

## Package Structure (`v0.1.1`)

To eliminate multi-part manual archive merging for everyday use while keeping all core workflows self-contained, `v0.1.1` splits the extension into:

1. **Unified OS Base Packages (`~1.4–1.7 GB`, 1 per OS — no manual merge needed)**:
   - **Windows (`x86_64`)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
   - **Linux (`x86_64`)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`
   - **macOS (Apple Silicon `arm64`)**: `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` *(physical Intel `x86_64` Mac hardware is unverified)*
   - Each OS Base package bundles the extension (`depends NA`), Python 3.12 packages (`bundled-python/`: CPU PyTorch, DIPY, AMICO, nibabel, dcm2niix, etc.), companion `UKFTractography` CLI, `bundled-sources/` (`WMQL`, `WMA`, `TractCloud`), native runtimes (`MRtrix3 3.0.8`, `FSL 2602.0`, `FreeSurfer Surface Runtime` with `activation_key_included: false`), and **all non-WMA model weights** (`DDSeg`, `DDParcel`, `DDSurfer` + FreeSurfer `atlas.tar.gz`, `PNL` weights, and `TractCloud`). *(`IITmean_b0_256.nii.gz` stays `upstream-only` from NITRC and is fetched automatically by the installer script or on first mask run; personal FreeSurfer `license.txt` is required only for cortical surface postprocessing.)*

2. **Optional Standalone WMA White-Matter Atlases (selected on the website or hot-installed inside Slicer)**:
   - **`WMA ORG` (Default, 74 tracts)**: `SlicerDSpace-v0.1.1-WMA-ORG.tar.gz` *(single file `< 2.0 GiB`)*
   - **`WMA NABA` (Lifespan atlas, optional)**: `SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001..`
   - **`WMA East-West` (Cross-population atlas, optional)**: `SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001..`

---

## 中文快速安装说明

1. 访问官网定制安装页 **<https://neurodmri.github.io/SlicerDSpace/zh.html#start>**，选择系统并填写本机 **3D Slicer 5.10.0** 路径（留空自动查找），下载定制脚本双击运行即可。
2. 底座包默认已包含 **基础预处理、PNL 脑掩膜权重、Anatomical Priors 三大深度学习模型（`DDSeg`/`DDParcel`/`DDSurfer`）、微结构、全脑追踪、`WMQL`、`TractCloud` 与连接组**；**WMA 白质图谱**默认仅选中标准 **`ORG` 图谱**（单文件），超大 **`NABA` / `East-West`** 图谱默认不下载，日后在 Slicer 内切换时随时支持一键热补装。
3. 手动安装方式：下载对应平台的统一底座包（`.zip` / `.tar.gz`），在 **3D Slicer 5.10.0 -> Extensions Manager -> Install Extension from File...** 直接安装并重启，在 **Modules -> Diffusion -> SlicerDSpace** 打开使用。
