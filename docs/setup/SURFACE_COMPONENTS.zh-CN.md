<div align="center">

# 表面后处理组件

**可选原生运行组件，不再强制安装完整 FreeSurfer**

[项目](../../README.zh-CN.md) | [入门](QUICKSTART.zh-CN.md) | [平台](SUPPORT_MATRIX.zh-CN.md) | [English](SURFACE_COMPONENTS.md)

</div>

已同步 DDSurfer `8a8699c0fda32be7e0a3cbca2903680b9f0c950b`：皮层后处理可调用
本机编译的七个原始 FreeSurfer 命令。不是 Python 重写、新算法或完整 FreeSurfer 的
等功能替代；后处理仍为可选任务。

---

## 路线选择

| 配置 | 实际行为 |
| --- | --- |
| 填写表面组件目录 | 使用该经校验运行包，不再搜索完整 FreeSurfer |
| 未填组件目录，检测到已有 FreeSurfer | 复用已有安装 |
| 两者都没有 | 查找与上游构建配方匹配的用户缓存 |

在 **环境与安装 > 更多选项 > 高级路径** 中填写可选组件目录及个人激活文件。
要沿用已有 FreeSurfer，可清空组件目录。环境变量 `SLICERDSPACE_SURFACE_RUNTIME`
也可选择本机组件；`FS_LICENSE` 指向个人激活文件。显式选择的组件不完整或损坏时
明确报错，不静默换用其他算法或安装。

支持当前保留顶点对应关系的原生 `lh/rh.white.obj`、`lh/rh.pial.obj`。
历史 STL／外部 shell 路线还有其他依赖，不会强行改成七组件流程。

---

## 首次准备

选择表面后处理后，点击 **准备所选分析环境**。插件先查找已配置路径、
`FS_LICENSE`／`FREESURFER_LICENSE`、用户目录的 `.licenses/freesurfer/license.txt`、
`.freesurfer/license.txt`、`license.txt`，以及已选 FreeSurfer 目录的许可证。
显式指定的路径无效时不会静默换用其他文件。
缺少许可证时显示 **免费申请** 和 **导入许可证**；申请按钮打开官方注册页面，
收到文件后可在同一窗口导入。插件保存文件路径，不复制、上传或打包个人许可证。
请保留原文件位置；之后的准备和运行会复用此路径。取消不启动安装。
只有选择表面后处理才出现该提示。文本可读性检查不代表激活有效，原生命令仍执行激活校验。

**“检查”不安装依赖，GUI 的准备流程不编译源码。**
在 **`v0.1.1`** 中，包含 7 个原生命令与图谱模板的 `Surface Runtime 0.1.0` 已**直接内置**于各平台统一安装包中：
- **Windows x64**：`SlicerDSpace-v0.1.1-Windows-x64.zip`（`bundled-runtimes/surface-runtime-windows-x86_64`）
- **Linux x86_64**：`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`（`bundled-runtimes/surface-runtime-linux-x86_64`）

首次启动自动发现路径并通过 `tree_sha256` 校验（缺失或损坏时自动从发布包修复）。内置运行包不包含任何个人 FreeSurfer 激活文件（`activation_key_included: false`），运行时仍需提供您的个人 FreeSurfer 许可证文件（`FS_LICENSE`）。

各平台内置与离线运行包状态如下：

| 平台 | 内置 / 离线压缩包 | 状态 |
| --- | --- | --- |
| Windows x86_64 | 已内置于 `SlicerDSpace-v0.1.1-Windows-x64.zip`（`surface-runtime-windows-x86_64`） | **已通过 3D Slicer 5.10.0 原生验收** |
| Linux x86_64 | 已内置于 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`（`surface-runtime-linux-x86_64`） | **已通过 3D Slicer 5.10.0 原生验收** |
| macOS 15 或更高版本，Intel | `surface-runtime-darwin-x86_64.tar.gz` | 离线候选包，科学结果对照待完成 |
| macOS 15 或更高版本，Apple Silicon | `surface-runtime-darwin-arm64.tar.gz` | 离线候选包，科学结果对照待完成 |

缺少压缩包时，准备窗口会将该运行环境列为手动处理项。

阅读组件与数据许可后安装离线包：

```bash
PythonSlicer scripts/setup_surface_runtime.py --execute --accept-terms \
  --asset-root /path/to/assets --offline
```

不加 `--execute` 只打印计划。损坏或配方不同的已有安装不会被覆盖；默认使用上游按配方
区分的用户缓存，也可通过 `--prefix` 选择新的安装位置。

Linux 维护者可使用 `--build-from-source --jobs 4`。该路线需要 Git、C/C++ 开发工具和
CMake >=3.24,<4，保持上游构建器与数值源码不变。
`scripts/package_surface_runtime.py` 将合格构建打包，不收录激活文件。
`models/runtime_bundles.json` 记录平台、版本、配方、哈希和大小；
只有完成运行包发布及分发授权核对后，才启用在线下载。

---

## 运行与检查

```bash
PythonSlicer run_subject_pipeline.py --subject-root /data/sub-01 --subject-id sub-01 \
  --targets surface_postprocess --postprocess-runtime /data/local-components \
  --postprocess-license-file /data/private/freesurfer-license.txt --execute
```

使用已保存原生 OBJ 的同一被试目录，也可指定 `--postprocess-native-dir`。
完整安装仍可通过 `--postprocess-freesurfer-home` 使用。这七个表面组件不提供脑掩膜。

独立工作进程检查组件清单、源码版本、动态库和校验和。激活文件预检查仅确认文件
存在，不代表取得授权；原生命令仍执行其激活检查。几何、标量和分区完整性 QC 独立进行，
不代替人工解剖审核。清单及缓存签名记录运行环境身份；切换显式组件目录时不会沿用
上一运行环境的“已完成”状态。

Apple Silicon 候选包在 macOS 15.8.1 上用固定的 FreeSurfer 7.4.1 源码构建。
构建时在 `CFLAGS` 和 `CXXFLAGS` 中加入上游 Apple Silicon 选项
`-DARM64 -DPNG_ARM_NEON_OPT=0`，并设置 `MACOSX_DEPLOYMENT_TARGET=15.0`。
七个二进制均为 arm64，并在干净的进程环境下通过 `--version` 检查。
全新离线安装重复通过了这七项检查。此候选包未链接 OpenMP。
macOS 科学结果对照仍待验证。

Windows x86_64（`v0.1.1`）已完成全部 7 个原生 FreeSurfer 表面后处理程序（`mris_convert.exe`、`mris_inflate.exe`、`mris_curv.exe`、`mris_smooth.exe`、`mris_sphere.exe`、`mris_register.exe`、`mris_ca_label.exe`、`mris_anatomical_stats.exe`）的 Windows 原生移植、编译与打包（内置于 `surface-runtime-windows-x86_64`），并在 3D Slicer 5.10.0 中通过真实数据与 Linux 参考结果的一致性验收（`Activation Key Included: false`，运行仍需用户个人 FreeSurfer 许可证）。

---

## 来源与许可

原始命令、图谱及第三方库保留 [FreeSurfer 协议](../../licenses/FreeSurfer.txt)
和[图谱许可声明](../../anatomy/surface/ddsurfer_backend/postprocessing/assets/LICENSE.txt)，
不改为项目 MIT 许可。个人激活文件不会进入扩展包、模型镜像或自动下载。
见[实现来源说明](../architecture/IMPLEMENTATION_PROVENANCE.zh-CN.md)。
