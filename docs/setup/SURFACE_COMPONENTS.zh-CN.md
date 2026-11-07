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
“准备所选分析环境”优先安装合格预编译包，核对压缩包及逐文件哈希、构建配方、动态库，
成功后保存组件路径。Linux x86_64 候选包包含七个不变的原生命令及图谱，
压缩大小为 40,994,566 字节，解压约 85.2 MiB，不包含个人激活文件。
分析时仍需您自己的激活文件。

该运行包尚待发布到模型仓库。发布前可将合格压缩包放到
`<asset-root>/runtime/surface/surface-runtime-linux-x86_64.tar.gz`，
或配置已有的合格运行目录／FreeSurfer 安装。缺少尚未发布的包会明确列为待处理项，
不会自动转为源码编译。

阅读组件与数据许可后安装离线包：

```bash
PythonSlicer scripts/setup_surface_runtime.py --execute --accept-terms \
  --asset-root /path/to/assets --offline
```

不加 `--execute` 只打印计划。损坏或配方不同的已有安装不会被覆盖；默认使用上游按配方
区分的用户缓存，也可通过 `--prefix` 选择新的安装位置。

维护者可明确使用 `--build-from-source --jobs 4`，该专家路线需要 Git、C/C++ 开发工具和
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

合格预编译包目前仅覆盖 **Linux x86_64**；未认证 macOS 编译或原生 Windows，
WSL 也不等于原生 Windows 支持。支持的平台上仍可使用已有原生 FreeSurfer。
不宣称核心算法或完整计算流程因此提速。

---

## 来源与许可

原始命令、图谱及第三方库保留 [FreeSurfer 协议](../../licenses/FreeSurfer.txt)
和[图谱许可声明](../../anatomy/surface/ddsurfer_backend/postprocessing/assets/LICENSE.txt)，
不改为项目 MIT 许可。个人激活文件不会进入扩展包、模型镜像或自动下载。
见[实现来源说明](../architecture/IMPLEMENTATION_PROVENANCE.zh-CN.md)。
