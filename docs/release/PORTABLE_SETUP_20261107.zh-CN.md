<div align="center">

# 可移植安装与默认方法，2026 年 11 月 7 日

[项目](../../README.zh-CN.md) | [自动配置](../setup/MANAGED_SETUP.zh-CN.md) | [平台](../setup/SUPPORT_MATRIX.zh-CN.md) | [English](PORTABLE_SETUP_20261107.md)

</div>

---

本说明列出对用户可见的变化，对应 `r5-native-tensor` 分支的代码。

## 部署提示

更新后，每个 Linux 部署都会把脑掩膜和预处理（含 eddy）重新生成一次。
原因：掩膜签名现在包含掩膜路线，旧版本保存的结果与它不再匹配。
更新后的第一次运行请预留计算时间。

## 变化：默认方法

- DTI 拟合在 Slicer 内部用 Teem 库完成。张量拟合不再需要 SlicerDMRI。
  仅当勾选 WMA 时，准备步骤才会要求 SlicerDMRI。
- DTI 产品新增彩色 FA 图（`<id>-dti-DEC.nrrd`）。
- DKI 默认使用 Python MK-Curve。
- 连接组后端为 `auto`：`.tck` 来源且指标为流线计数时用 MRtrix `tck2connectome`，其他来源或指标用 Python。
- 新增可选的 iFW 阶段（需要回波时间），它同时运行 Free Water 和 DDSeg。
- 每次运行写出 `00_run_metadata/run_record.json`，列出所用方法、脑掩膜来源和已接受的许可。
  重新运行时，上一次的记录保存为 `run_record-<started_at>.json`。

## 变化：Linux、macOS 与 Windows 的安装

- 勾选所需产品，插件会自动加入前置步骤。
- 一个对话框列出每项下载的大小和许可。点击一次 **接受许可并安装** 即可。
  命令行用 `PythonSlicer scripts/setup_selected.py --targets <targets> --execute --accept-licences` 完成同样的事。
- 安装结束后，对每项功能做一次自检。自检失败时不会标记为就绪。
- dcm2niix：插件把固定版本的 dcm2niix 轮子装入 Slicer 的 Python。
  如果该轮子或你自己设置的 dcm2niix 路径未通过检查，则准备未完成。
  如果只在 `PATH` 中找到的 dcm2niix 未通过检查，只给出警告。NIfTI 输入不需要 dcm2niix。
- macOS 把 MRtrix3 3.0.8 和 FSL 安装为托管运行环境（两者都需要 macOS 14.5 或更高）。
- Windows 在 WSL 2 Ubuntu 中运行 MRtrix3 和 FSL。脑掩膜和 Python 阶段在 Windows 版 Slicer 的 Python 中运行。
- 准备命令使用 `PythonSlicer`（Slicer 自带的 Python），不再用 `python3`。

## 变化：PNL 脑掩膜

- 新设置 `pnl_mask_route`：`auto`、`managed` 或 `native`（图形界面高级设置，或 `--pnl-mask-route`）。
- Linux 上 `auto` 即 `managed`（带 ANTs 2.3.0 的 conda 运行环境）。
- macOS 和 Windows 上 `auto` 即 `native`：在 Slicer 的 Python 中用 SimpleITK 配准和 PyTorch。
  界面标注为 "native (experimental)"。
- native 路线是实验性的：在 24 个测试例中有 2 例，其掩膜未达到与 managed 路线的端到端一致性规则。
- 掩膜回执和运行记录写明生成掩膜所用的路线。
- PNL 许可显示在许可对话框中，全文见 [docs/licenses/pnl_brain_mask.md](../licenses/pnl_brain_mask.md)。

## 旧运行环境文件夹

准备步骤替换已有的 macOS MRtrix3 运行环境时，会把旧文件夹重命名为
`mrtrix-3.0.8-macos.replaced-<ns>`，位于 runtimes 文件夹（`<模型资源目录的上一级>/runtimes`；
macOS 上为 `~/Library/Application Support/SlicerDSpace/runtimes`）。每个这样的文件夹约 169 MB。
插件从不删除它们。运行成功后，你可以手动删除。
