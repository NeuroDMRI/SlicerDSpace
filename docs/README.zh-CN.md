<div align="center">

# SlicerDSpace 文档

**安装、运行、检查与复现**

[项目](../README.zh-CN.md) | [快速开始](setup/QUICKSTART.zh-CN.md) | [平台](setup/SUPPORT_MATRIX.zh-CN.md) | [发布](release/EXTENSION_READINESS.zh-CN.md) | [English](README.md)

</div>

首次使用请阅读简洁界面的快速教程。技术参考说明各条路线；
带日期的验证报告保留历史证据，不是当前代码的通用保证。

---

## 用户指南

| 指南 | 内容 |
| --- | --- |
| [下载范围](setup/DOWNLOADS.zh-CN.md) | 精简插件包、可选轻量源码与完整研究仓库的区别 |
| [快速开始](setup/QUICKSTART.zh-CN.md) | 最小 DTI 工作流与合成安装测试 |
| [CLI 模板](setup/CLI_TEMPLATES.zh-CN.md) | 各路线命令、追踪/束来源、PowerShell、脑区分析和报告 |
| [批量实验](setup/EXPERIMENTS.zh-CN.md) | 高级参数覆盖、命名组合与被试并发 |
| [Agent skill](../.agents/skills/slicerdspace/SKILL.md) | 安装、输入准备、按请求分析和结果检查 |
| [输出目录](setup/OUTPUT_LAYOUT.zh-CN.md) | 新 `03_anatomy` 产品及旧 `03_structural` 结果的只读兼容 |
| [能力与平台](setup/SUPPORT_MATRIX.zh-CN.md) | 按路线配置依赖 |
| [结果检查](setup/GUI_RESULT_REVIEW.md) | 按运行加载结果、取消任务与 QC |
| [脑掩膜](setup/brain_mask_gui_config.md) | 默认 PNL CNN（Linux 为 managed，macOS 和 Windows 为实验性的 native）、替代后端 |
| [GPU 策略](setup/GPU_EXECUTION.zh-CN.md) | 神经网络优先 GPU、明确 CPU 选择与 CPU-only 路线边界 |
| [插件内自动配置](setup/MANAGED_SETUP.zh-CN.md) | 一次确认含全部许可、自检、Python/模型、MRtrix/FSL 运行环境、修复与离线资源 |
| [微结构分析](setup/MICROSTRUCTURE.zh-CN.md) | DTI 默认（含彩色 FA 图）、独立 FW/iFW/NODDI/DKI 与原作者 MK-Curve 边界 |
| [UKF 纤维追踪](setup/UKF.zh-CN.md) | 可选作者扩展、默认参数、无损输入及自动 WMA 衔接 |
| [WMA 图谱](setup/WMA_ATLASES.zh-CN.md) | ORG、East-West 与 NABA 选择、按需资源及图谱对应的 QC |
| [WMQL 小脑区域查询](setup/WMQL_CEREBELLAR.zh-CN.md) | 可选区域规则、CLI/GUI 预设与检查 |
| [脑区分析](setup/REGIONAL_ANALYSIS.zh-CN.md) | 高级设置中按需选择脑区，进行参数统计、流线筛选和原权重连接分析 |
| [自动加载](setup/AUTOMATIC_RESULTS.md) | 当前运行记录与 RAS VTK 显示预览 |
| [配置](setup/USER_CONFIG.md) | GUI/CLI 共用设置 |
| [Linux](setup/LINUX_HPC.md)、[Windows](setup/WINDOWS.md)、[macOS](setup/MACOS.md) | 平台设置与边界 |
| [模型资源](assets/MODEL_ASSETS.md) | Profile、下载与校验 |
| [资源授权](assets/MODEL_ASSET_TERMS.md) | 文件校验与再分发许可分别审查 |
| [代码与第三方授权](assets/CODE_LICENSES.zh-CN.md) | 原许可证、固定来源与分发条件 |
| [实现来源](architecture/IMPLEMENTATION_PROVENANCE.zh-CN.md) | DDSeg/SAF/MK-Curve 移植及修改后的 PNL 推理代码与未修改上游的区别 |
| [外部依赖](setup/EXTERNAL_DEPENDENCIES.md) | 外部软件与路线前提 |

---

## 维护者指南

- [扩展准备状态](release/EXTENSION_READINESS.zh-CN.md)：资源打包、SDK 构建与发布门禁。
- [发布流程](release/RELEASE.md)：代码归档、资源、标签与证据。
- [目录结构](architecture/REPOSITORY_LAYOUT.md)：解剖、微结构与连接分析源码。
- [产品架构](architecture/PRODUCT_ARCHITECTURE.zh-CN.md)：原生客户端、按需依赖、隔离后端与发布边界。
- [性能优化与实测](release/PERFORMANCE_OPTIMIZATIONS.zh-CN.md)：原作者 WMA 同输入实测与逐字节一致性；旧插件比较和 GPU／缓存收益单独列出。

---

## 历史验证记录

[可移植安装与默认方法](release/PORTABLE_SETUP_20261107.zh-CN.md)列出对用户的变化：
原生张量拟合、一次安装确认、各平台的 PNL 掩膜路线、运行记录，以及更新后的一次性重新生成。

[所选依赖准备验证](release/DEPENDENCY_SETUP_20261021.zh-CN.md)记录原生安装/取消检查、
PNL 新隔离环境 GPU 掩膜、离线资源校验，以及平台和空白机器验收边界。

[10 月 21 日束工作流复核](release/BUNDLE_WORKFLOW_20261021.zh-CN.md)记录自动输入衔接、
三被试 WMQL 重审、WMA 保存结果重审、TractCloud 实际重跑与采样警告，以及隔离场景验证。

[10 月 16 日后续验证](release/QC_FOLLOWUP_20261016.zh-CN.md)记录已核实的 MK-Curve
排除与逐指标覆盖率、新版 ACT／WMA 及 Linux SDK／CPack 检查，以及仍未通过的
端点 WMQL 采样与解剖人审门禁。

[CSD／ACT 专项排查](release/CSD_ACT_DIAGNOSIS_20261015.zh-CN.md)更正此前 CSD
验收结论：发现 VentralDC 组织映射缺陷，已完成固定 FOD 对照，旧 ACT 依赖结果需重算。

[本轮平台复核](release/PLATFORM_AUDIT_20261015.md)记录 Windows/WSL 设备与桥接修复、
macOS 路径适配、计时边界及尚未验收的功能。

[本轮白质束 QC](release/BUNDLE_QC_20261015.zh-CN.md)说明旧绿色状态、采样门禁、
真实输入诊断和跨平台未验收边界。
[WMQL CST 专项排查](release/WMQL_CST_20261015.zh-CN.md)区分历史经过查询和论文端点定义，
记录配对输入对照及固定导入文件。

[白质束后端核查](release/BUNDLE_BACKENDS_20261013.zh-CN.md) 区分 WMA 失败结果、
稀疏 WMQL 候选束和固定种子验证的 TractCloud 输出。
[模块运行时间](release/MODULE_RUNTIMES_20261017.zh-CN.md) 汇总历史与补测计时，
不把缓存复用或小 ROI 测量冒充全脑计算耗时。

[SAF Python 核验](release/SAF_PYTHON_20261013.md)记录原版核心函数对照、
有意保留的修正，以及真实纤维的几何检查与可靠性边界。

[最新 DDSurfer 同步](release/DDSURFER_SYNC_20261013.md)记录 RAS 标记、Python 标签导出、
批量 fsaverage 映射及真实结果对照。
[Pipeline QC 与性能](release/PIPELINE_QC_20261015.zh-CN.md)记录最新失败门禁、
全脑 NODDI 重复性、已完成的 DKI/后处理检查，以及剩余解剖与平台验收边界。
[科学验证补充](release/SCIENTIFIC_CLOSURE_20261015.zh-CN.md)记录全脑 MK-Curve
参考对照、UKF 输入上的未修改作者 WMQL，以及新版 WMA／GUI 联合运行状态；人工审核仍独立保留。
[Python MK-Curve 验证](release/MKCURVE_PYTHON_20261012.md)记录 Python 校正路线与独立作者代码对照。

[脑掩膜、微结构与场景验证](release/PIPELINE_MASK_MICROSTRUCTURE_20261012.md)
记录默认 PNL、DKI 适用性、Advanced 独立开关与自动加载；真实 MATLAB MK-Curve
尚未验证，不以标准 MK 或接口测试冒充。

[10 月 12 日复核](release/DISTRIBUTION_VALIDATION_20261012.md)记录许可、干净克隆、
隔离安装和 SDK 阻塞。[10 月 11 日记录](release/DISTRIBUTION_VALIDATION_20261011.md)
保留前一轮工作树验证；其测试数量与本轮限定范围的 Git 快照不同。

[科学计算与 GUI 验证](release/SLICER_ACCEPTANCE_20261007.md)、
[简洁界面验证](release/UI_SIMPLIFICATION_20261008.md)、
[DDSurfer 同步](release/DDSURFER_SYNC_20261005.md)为带日期记录。
旧报告中的 Beginner/向导描述不再指导当前界面。

文档采用 DDSurfer 的居中标题、导航和分隔线风格。固定同步的上游 Markdown
及第三方许可证原文不改写，避免破坏源码哈希与授权记录。
