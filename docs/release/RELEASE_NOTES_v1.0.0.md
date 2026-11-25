# SlicerDSpace v1.0.0

## Highlights

- **Modular Cross-Platform Model Weight Architecture**: Deep-learning model weights are now cleanly decoupled from the platform-specific OS runtime bundles into five standalone, SHA-256-verified cross-platform archives (`DDSurfer`, `DDSeg`, `DDParcel`, `PNL-CNN`, `TractCloud`), shrinking each OS Base runtime archive from ~1.5 GB down to ~500–650 MB while laying the foundation for future selective plugin-style model installation.
- **Unchanged Out-of-the-Box Setup Experience**: `SlicerDSpaceSetup-{Windows,Linux,macOS}.zip` and the generated CLI installers automatically install the slimmed OS runtime archive together with all five core model-weight packs and the upstream NITRC IIT template, preserving the exact same zero-configuration experience as `v0.1.1`. Optional WMA white-matter atlases remain on-demand inside Slicer when WMA is selected.
- **Updated DDSurfer Weights**: Incorporates the updated DDSurfer cortical surface reconstruction checkpoints from `NeuroDMRI/DDSurfer@7d28dbddd7` (`ddsurfer_{lh,rh}_{wm,pial}.pt` and `SHA256SUMS`), packaged together with the surface templates and FreeSurfer post-processing atlas in `SlicerDSpace-v1.0.0-Model-DDSurfer.tar.gz`.
- **In-Slicer Guided CUDA PyTorch Setup**: When an NVIDIA GPU is detected on a system with CPU-only PyTorch, CUDA device entries appear with a guided `?` setup prompt that previews the CUDA wheel download size, installs CUDA PyTorch into Slicer's environment, and guides a clean restart.

---

## 更新亮点（中文）

- **跨平台模型权重模块化拆包**：将五大深度学习模型权重（`DDSurfer`、`DDSeg`、`DDParcel`、`PNL-CNN`、`TractCloud`）从三平台系统底座包中解耦为独立、跨平台复用、带 SHA-256 校验的模型分包；各平台系统底座包体积由约 1.5 GB 精简至约 500–650 MB，为后续“按需拉取/一切皆插件”扩展形态奠定基础。
- **保持与 v0.1.1 一致的开箱即用体验**：`SlicerDSpaceSetup-{Windows,Linux,macOS}.zip` 图形安装引导包及命令行安装脚本仍默认一键安装系统底座包、全部 5 个核心模型权重分包及 NITRC IIT 参考模板，安装完成即可使用全部核心功能；大型 WMA 白质图谱保持按需在插件内拉取。
- **更新 DDSurfer 模型权重**：同步更新至 `NeuroDMRI/DDSurfer@7d28dbddd7` 最新版皮层表面重建权重（`ddsurfer_{lh,rh}_{wm,pial}.pt` 与 `SHA256SUMS`），打包于 `SlicerDSpace-v1.0.0-Model-DDSurfer.tar.gz`。
- **Slicer 内置 CUDA 版 PyTorch 引导安装**：在检测到 NVIDIA 独立显卡但当前为 CPU 版 PyTorch 时，设备下拉列表置灰并附带 `?` 引导按钮，安装前提示体积并自动安装 CUDA 版 PyTorch 及引导重启。
