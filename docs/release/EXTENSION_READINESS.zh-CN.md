<div align="center">

# 扩展准备状态

**先形成证据，再分发**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [平台](../setup/SUPPORT_MATRIX.zh-CN.md) | [发布流程](RELEASE.md) | [English](EXTENSION_READINESS.md)

</div>

Windows 5.10.0 原生全流程实测见[Windows 5.10.0 原生验收报告](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)，
本机表面组件集成见[2026-10-24 实测](SURFACE_COMPONENTS_20261024.zh-CN.md)，
GUI 后续验证见[2026-10-23 优化](GUI_RESPONSIVENESS_20261023.zh-CN.md)，
官方集成实测见[2026-10-22 验收](OFFICIAL_SLICER_VALIDATION_20261022.zh-CN.md)，
科学流程等历史记录见[2026-10-16 复核](QC_FOLLOWUP_20261016.zh-CN.md)，
历史记录见[2026-10-12 复核](DISTRIBUTION_VALIDATION_20261012.md)，
前一轮见[2026-10-11 分发验证](DISTRIBUTION_VALIDATION_20261011.md)。
分别列出已通过的本地检查和各轮尚未完成的验收。

目标是通过门禁后申请 Tier 1，不代表已获 Slicer 批准。
仓库保持私有；构建工具不自动公开仓库、不提交 Index PR、不上传扩展包。

---

## 资源打包

CMake 与 staging 共同读取
<code>scripts/extension_runtime/runtime_resources.json</code>。
清单包含 WMQL 查询、WSL C# 桥接及运行 requirements、LUT、shell/config 和许可证；
排除测试、参考源码与本地模型目录。不再直接复制整个运行目录。

~~~bash
python3 scripts/audit_slicer_extension_metadata.py --json
python3 scripts/stage_slicer_extension.py --install-root ignore/release-test/install --clean --link-mode copy
python3 scripts/package_slicer_extension.py --extension-root ignore/release-test/install/SlicerDSpace --output ignore/release-test/SlicerDSpace.tar.gz
python3 scripts/audit_extension_package.py ignore/release-test/SlicerDSpace.tar.gz --report ignore/release-test/package_audit.json
~~~

安装包检查核对文件及源文件哈希，报告缺失、额外或重复资源。
通过不代表真实 SDK 构建成功，也不代表算法科学有效性。

---

## 真实 SDK 构建

需要含 SlicerConfig.cmake 的真实 Slicer 构建树，不能使用普通二进制安装目录代替。
每个平台、SDK 版本使用新的构建目录：

~~~bash
python3 scripts/build_slicer_sdk.py --slicer-dir /path/to/Slicer-build \
  --build-dir ignore/sdk-release-linux --package-generator TGZ \
  --unit-test-python /path/to/test-env/bin/python
~~~

Windows 使用 ZIP。脚本执行配置、构建、CTest、CPack 和包内容检查，
在 sdk_build_report.json 保留命令、返回码与日志位置。
SDK 缺失标记为 BLOCKED，不能显示为通过；不会隐式下载模型。

从终端运行，并在指定的独立 Python 中准备后端测试依赖。后端单元测试不在嵌入式
Slicer 内执行，通用／原生模块测试仍使用真实 SDK 应用。Linux 无显示环境需提供
显示服务，例如在构建命令前使用 `xvfb-run -a`。

手动 **Slicer SDK** Actions workflow 需要可信 self-hosted Linux/Windows/macOS
runner，标签含 <code>slicer-sdk</code>，并配置 SLICER_SDK_DIR。
不同平台目录不同可分别运行。不要让不可信 PR 在这些机器执行。
常规 CI 中三平台轻量资源检查与真实 SDK 验收分开。

**10 月 16 日证据：** 真实 Linux Slicer v5.10.0 superbuild、扩展配置／构建、
128 项 CTest、CPack 及 460 项资源哈希审计通过。实际 CPack 包通过隔离设置的
Extension Manager 安装，不加入开发模块路径；DTI 缓存／区域表及真实束加载、
失败结果拒绝检查通过。科学依赖和 SlicerDMRI 已预配置，不能说是空白机器全流程、
Windows／macOS 原生验收或解剖认证。最小 SDK stub 仍仅是合同测试夹具。

---

## 独立干净安装

1. 使用新的 Slicer 安装、独立设置，不加入开发目录。
2. 通过 Extension Manager 本地文件入口安装 CPack 生成的包。
3. 确认导入只来自安装目录，不依赖开发 checkout 或旧模型缓存。
4. 仅配置最小 DTI 依赖，按教程运行明确标识的合成安装测试。
5. 独立获得授权、去标识化的真实数据，检查 DTI 图、梯度/掩膜 QC 与运行记录。
6. WMQL 应实际配置后端、执行包内查询并记录白质束数量及对齐；有文件不等于提取成功。
7. 解剖模型另行显式安装资源，并核对匹配版本的 SHA256。
8. 对每个声称支持的平台/后端独立重复。

记录 OS、SDK/Slicer 版本、包 SHA256、路径、工具版本、输入哈希和人审状态。
staging、旧源码测试或已配置好的工作站不能替代这种验收。

---

## 发布门禁

| 项目 | 状态 |
| --- | --- |
| 共用资源清单与安装包回归检查 | 已实现，本地保留运行证据 |
| LICENSE.txt 与 Index JSON | 已准备；Tier 1 仅为草稿 |
| 简洁界面中英文指南及截图来源 | 已准备 |
| 仓库公开、公开图标/截图 URL、GitHub topic | 按用户要求暂缓；公开提交前必须处理 |
| Linux 真实 SDK／CPack 与隔离安装 | 已通过记录所述限定路线，不等于空白机器全流程 |
| Windows 5.10.0 原生运行包及全流程真实数据验收 | 已在 3D Slicer 5.10.0（`HCP 100307`，`SLICERDSPACE_DISABLE_WSL=1`）通过；详见 [Windows 5.10.0 原生验收报告](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md) |
| 第三方源码许可 | BWH conversion、MGH FreeSurfer、DDSurfer MIT 与 PyTorch3D BSD 声明已补齐并校验；不等于全面法律认证 |
| 权重/模板/测试数据再分发 | 未确认组不能进入公开资产归档 |
| 真实数据教程及 Sample Data 注册 | [Slicer DWIVolume 教程](../setup/SLICER_SAMPLE_DTI.zh-CN.md)已验证 DTI 路线；独立可再分发数据注册仍待完成 |
| 专利披露 | 维护者需审查，不能未经确认声称不存在 |

<code>docs/release/extension-index/SlicerDSpace.json</code> 是目录条目草稿，
不是正式提交。需与 CMake、s4ext 保持一致；仓库私有时资源 URL 不对公众可用。

---

## 官方参考

[扩展开发指南](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)、
[提交清单](https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)、
[目录 schema](https://github.com/Slicer/Slicer/blob/main/Schemas/slicer-extension-catalog-entry-schema-v1.0.2.json)。
用户指南结构参考 [SlicerTotalSegmentator](https://github.com/lassoan/SlicerTotalSegmentator)
和 [SlicerDMRI](https://dmri.slicer.org/)，不表示其背书或相同维护等级。
