<div align="center">

# 下载与安装范围

**用户安装包、轻量源码与完整研究记录分别提供**

[项目](../../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [文档](../README.zh-CN.md) | [English](DOWNLOADS.md)

</div>

仅使用插件，不需要下载网站源码、历史性能报告或测试代码。它们保留在完整仓库中，
供开发、审计和论文复现使用，不应成为普通用户安装的前提。
仓库目前仍为私有预发布；以下不表示已经上架官方 Extension Manager，
也不表示已有公开可下载的正式安装包。

---

## 选择下载方式

| 用途 | 推荐方式 | 包含内容 |
| --- | --- | --- |
| 在 Slicer 中使用 | 维护者提供并完成对应验收的插件安装包 | 运行代码、UI、查询文件、许可证、使用指南和必要参考文件 |
| 有仓库权限、从源码安装 | 下文的轻量检出，再 staging | 可更新的运行源码；不检出网站、测试和历史优化报告 |
| 开发或独立复现 | 完整源码 checkout | 测试、性能证据、发布工具、网页源码及文档 |
| 阅读结果与性能依据 | 登录 GitHub 浏览对应报告 | 不需要把报告复制到插件目录 |

安装包默认不包含 `docs/site/`、`docs/website/`、`website/`、`Testing/`、
历史优化与 QC 报告、原始基准 JSON 或测速脚本。
少数发布说明和运行时间 CSV 是显式保留的参考文件；第三方许可证、
WMQL 查询、WSL 辅助文件及模型资源清单不会因此删除。
模型权重、模板、外部软件环境仍按所选路线单独准备，不是安装包内的代码。

---

## 轻量源码安装

以下命令用于**新的 checkout**，需已有私有仓库访问权限和支持 partial clone /
sparse checkout 的 Git。Bash、Git Bash 或 PowerShell 均可执行：

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/NeuroDMRI/SlicerDSpace.git
cd SlicerDSpace
git show HEAD:scripts/extension_runtime/runtime_checkout.txt | git sparse-checkout set --no-cone --stdin
git checkout
PythonSlicer scripts/stage_slicer_extension.py --install-root ~/SlicerDSpace-extension --link-mode copy --clean
```

`PythonSlicer` 是 Slicer 自带的 Python，位于 Slicer 的 `bin` 文件夹，无需系统 Python。根据实际 Slicer 版本设置
`--slicer-version`，并按[项目安装说明](../../README.zh-CN.md#安装)添加模块路径。
上述 staging 是源码安装，不是官方 SDK/CPack 发布认证。

`--depth 1` 限制历史；`--filter=blob:none` 按需获取文件内容；稀疏规则决定实际检出的
文件。三者作用不同，**只有浅克隆仍会下载当前版本的网站和报告**。
Git 服务端需要支持过滤；若提示 `filtering not recognized by server`，只能保证工作区
不显示被排除文件，不能保证它们没有被传输。不要加 `--also-filter-submodules` 或
自动下载模型来扩大安装范围。原理见 [Git clone](https://git-scm.com/docs/git-clone)
与 [sparse checkout](https://git-scm.com/docs/git-sparse-checkout) 官方说明。

更新轻量 checkout：

```bash
git pull --ff-only
git show HEAD:scripts/extension_runtime/runtime_checkout.txt | git sparse-checkout set --no-cone --stdin
```

随后重新 staging 并重启 Slicer。轻量 checkout 不包含完整测试与科学证据，
不用于宣称已独立复现。需要完整源码时，先保存自己的修改，再运行
`git sparse-checkout disable`；这会按需下载被排除的文件。
如还需要完整提交历史，再执行 `git fetch --unshallow`，不要把这作为普通安装步骤。
部分文档链接指向未打包的历史报告，请在完整 GitHub 仓库中查看。

---

## 安装包与源代码归档不同

GitHub 自动生成的 **Source code (zip/tar.gz)** 不是插件安装包；不要将其当作
Slicer Extension Manager 的安装文件。经过对应验收的 CPack 扩展包优先；
维护者也可生成明确标为本地预发布测试的 staging 包：

```bash
PythonSlicer scripts/stage_slicer_extension.py --install-root /path/to/staging --link-mode copy --clean
PythonSlicer scripts/package_slicer_extension.py --extension-root /path/to/staging/SlicerDSpace --output /path/to/packages/SlicerDSpace.tar.gz
PythonSlicer scripts/audit_extension_package.py /path/to/packages/SlicerDSpace.tar.gz
```

CMake 与 staging 共用 `scripts/extension_runtime/runtime_resources.json`，
不再递归收集整个 `docs/`。添加运行时文件后，同时检查轻量检出规则和安装包测试。
本地打包通过不能替代完整 SDK 构建、干净安装和科学验收。

整模块测速会将开发用脚本单独冻结在基准目录的 `harness/` 中，与插件运行时快照
分开保存。不能为了运行测速，再把这些脚本加回面向用户的插件安装包。

不通过 `.gitignore` 隐藏已经提交的报告，也不删除历史证据、重写 Git 历史或移除
网页贡献者信息。网站继续独立部署，安装用户不需要它的构建工具。
