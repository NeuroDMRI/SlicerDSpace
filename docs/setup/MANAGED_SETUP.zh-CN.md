<div align="center">

# 插件内自动配置

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [GPU](GPU_EXECUTION.zh-CN.md) | [English](MANAGED_SETUP.md)

</div>

---

## 用户操作

选择分析任务后，展开独立的“环境与安装”区域，点击 **准备所选分析环境**。
勾选一个产品会加入它的前置步骤（预处理不会被自动加入），计划只覆盖这些步骤。
一个对话框列出每项下载的大小和许可（名称、链接和本地文件），点击一次 **接受许可并安装** 即可。
环境报告的组合入口使用同一逻辑。检查与最终复核均在可取消的后台进程中进行。
只有存在待处理项目时才确认；依赖已能导入时不重复安装，仍执行功能自检。
离线资源、修复和手工路径位于环境区的“更多选项”，不混入算法高级设置。
只处理本次所需依赖：

1. 通过 Slicer 扩展管理器安装所勾选产品缺少的扩展：神经网络模型和 native 脑掩膜需要 PyTorch，UKF 需要 UKFTractography，WMA 需要 SlicerDMRI。张量拟合不需要扩展（它使用 Slicer 内核中的 Teem）。
2. 新安装或启用扩展后，保存工作并重启 Slicer；重新进入 SlicerDSpace 后提示继续准备，复用已有扩展。
3. 后台准备固定版本的 WMQL/WMA/TractCloud 源码，无需 Git；安装缺少的 Python wheel、下载模型并准备支持的隔离工具。
4. 校验成功后保存路径，重新检查依赖并执行小型功能测试。自检失败时不能进入分析，须重新准备并通过。

同一配置的功能自检失败会跨重启保留。命令行可用 `scripts/check_setup_functionality.py` 运行同样的自检。
dcm2niix（用于 DICOM 导入）遵循一条规则：插件自带的固定版本轮子或你自己设置的 dcm2niix 路径未通过检查时，准备未完成；
只在 `PATH` 中找到的 dcm2niix 未通过检查时只给出警告，因为 NIfTI 输入不需要它。源码发现与实际运行使用同一份经校验缓存，
未保存显式源码路径时也能复用。Windows 的 WMA 不会把 Bash 源码目录当作 WSL 可执行桥接。

启用“首次使用准备”后，点击运行时发现缺少依赖会进入同一个确认流程，安装或重启后不自动启动科学计算。
确认页展示资源目录、已知模型与源码大小及许可。Python/Conda 依赖的下载量由求解器决定，
不计入这些大小总计，可能另外需要数 GB。

这些自动配置路线不需要打开终端、管理员权限、激活 Conda 或手动寻找可执行文件。
如果没有 Conda，插件会从 conda-forge 官方 GitHub 下载固定版本的 Miniforge，
先校验 SHA256，再运行引导安装。环境放在用户目录，不修改 Slicer Python、系统 CUDA 或 shell 配置。
FSL 与 PNL 源码、模型、模板条款均需明确确认。
检测到 NVIDIA 工具时默认安装 GPU eddy，CUDA profile 同时保留 CPU 程序；分析设备偏好仍在高级设置调整。
实际 GPU 能否运行要在后端执行时检查，不能由“包安装成功”推断。

安装在后台执行，提供取消按钮和日志。取消或校验失败不替换已配置路径；
已成功安装的包和部分下载可能保留供重试。
已经校验且未发生变化的环境直接复用，不重复安装。

**修复所选受管工具** 在明确确认后重新安装所选的插件受管外部环境，保留修复前的就绪回执，
拒绝修改非插件管理的环境。下载或安装中断后通常直接重试“准备”；已有环境损坏时再用“修复”。
取消不回滚已经安装成功的包，可通过“打开安装日志”定位失败阶段。

Python 安装检测 `slicer.packaging.pip_ensure` 是否存在；Slicer 5.10 没有该模块，
使用可取消的后台 Python 进程。常规 pip 安装约束现有 NumPy、SciPy、VTK、SimpleITK、
PyTorch 等核心版本，不为解决冲突而静默升级基础栈。PyTorch 使用官方扩展辅助器，
保留所选算法中最严格的最低版本要求。DDSeg、DDParcel、DDSurfer、TractCloud 默认复用兼容的
PyTorch；WMQL、DIPY、AMICO 默认使用工作流 Python，明确配置的其他解释器仍保留。
已加载的包发生变更时，必须重启后再运行分析。
新版 packaging API 当前通过契约测试验证，并非已实测更新版 Slicer 二进制。
普通包请求只使用二进制 wheel；缺少兼容 wheel 时报告失败，不在用户机器上启动编译器。
专家手动指定的 VCS URL 和本地工程使用原生 `pip_install`，避免传入仅解析标准包名的
`pip_ensure`；同样应用版本约束并在最后检查实际导入。

---

## 模型与已有环境

模型使用已配置目录或默认用户数据目录，不再强制弹出目录选择器。
缺少某个产品模型时，仅下载该产品的 profile，而不是三个解剖产品全部下载。
主动选择核心资源下载时仍下载 core profile。
资源通过校验后自动保存目录；文件校验不代表授权确认或模型准确性认证。

**使用离线模型资源** 选择已有资源目录，后台按插件自带的可信 SHA256 清单验证所选产品。
不下载、不执行资源，失败保留原路径；复制资源时须保留目录结构。
导入的是模型目录，不是可移植 Conda 环境；受保护的模型下载仍需相应访问授权，不会绕过权限。

PNL 模型/模板、TractCloud 推理参考、SynthStrip 权重和 WMA ORG 图谱使用统一资源目录。
准备所选环境时，按缺少的分组获取资源、核对 SHA256 并安全解压 ORG，成功后保存路径。
算法源码与数据分阶段准备；源码压缩包固定提交和 SHA256，保留作者许可，校验后自动保存路径。
用户明确配置的其他源码目录不会被替换。
详见[可选资源与离线导入](../assets/EXTERNAL_RESOURCES.zh-CN.md)。

已有软件、离线部署或集群管理员管理的环境仍可在高级路径设置中配置。
自动安装是明确发起的操作，不会在科学分析过程中静默替换已有环境。
默认环境目录位于模型资源旁的 `SlicerDSpace/runtimes`。

准备步骤替换已有的 macOS MRtrix3 运行环境时，会把旧文件夹重命名为该 runtimes 文件夹中的
`mrtrix-3.0.8-macos.replaced-<ns>`（每个约 169 MB，`<ns>` 为时间戳）。插件从不删除这些文件夹，
运行成功后你可以手动删除。

---

## 当前覆盖范围

| 工具 | 自动配置范围 |
| --- | --- |
| MRtrix3 3.0.8 | Linux x86_64：官方预编译 Conda 包。macOS（Intel 和 Apple Silicon，macOS 14.5 或更高）：官方通用 tarball，校验 SHA-256 后解压到 runtimes 文件夹 |
| FSL eddy 2602.0 / topup 2203.5 | Linux x86_64、macOS Intel 和 Apple Silicon（macOS 14.5 或更高，仅 CPU eddy），官方 FSL 源，安装前确认许可 |
| CUDA eddy 2602.0 | 可选 FSL profile，需要 NVIDIA 驱动，不要求安装完整 CUDA Toolkit |
| DDSurfer / DDSeg / DDParcel 资源 | 已有后台下载，改为按产品选择并使用默认目录 |
| Python 科学计算包 | 所选解释器、核心版本约束、可用时使用原生 packaging API；官方 PyTorch 辅助器 |
| PNL diffusion CNN | Linux x86_64；固定 PNLBrainMask-Torch 包、转换后的 v0.3 三方向权重、IIT b0 模板、Python 3.11、PyTorch 2.5.1 CPU、ANTs 2.3.0，独立环境 |
| 可选皮层后处理 | Linux x86_64 [合格预编译运行包](SURFACE_COMPONENTS.zh-CN.md)；已测离线安装，在线发布待完成；个人激活文件独立提供，GUI 不编译源码 |
| WMA / TractCloud / SynthStrip 非代码资源 | 所选分组下载/导入、校验、ORG 解压及自动保存路径；私有镜像仍需访问权限 |
| WMA/WMQL/TractCloud 源码 | 固定提交压缩包、安全解压、逐文件哈希、自动发现并保存路径，无需 Git，也不重复实现算法 |
| macOS 与 Windows 的 PNL 扩散 CNN | 无 managed 运行环境。native 路线（“native（实验性）”）在 Slicer 的 Python 中运行，需要 PyTorch 扩展和 pnl-cnn 脑掩膜资源，本安装器会规划它们（含 PNL 许可） |
| Windows | 由 `scripts/install_deps_windows.ps1` 安装 WSL 2 Ubuntu 运行环境，用于 MRtrix3、FSL 以及可选的 FreeSurfer 和算法源码；准备对话框会列出它及其许可。脑掩膜和 Python 阶段在 Windows 版 Slicer 的 Python 中运行，不在 WSL 中 |
| macOS CUDA eddy、Windows 原生 MRtrix3/FSL | 不提供。CUDA eddy 需要 Linux（或带 NVIDIA GPU 的 WSL） |

功能自检按所选依赖执行 NumPy 求解、小型 NIfTI 与 VTK 流线写入读回、所选 PyTorch 设备计算、
解剖模型反序列化及 DDSeg、TractCloud 小型前向推理；MRtrix 转换小体积图像，
PNL 受管环境通过 ANTs ImageMath 执行恒等运算后读回影像。
回执保存在 `setup-history`，不使用患者数据。
安装校验还检查命令、指定包版本、ELF 库和可执行文件哈希，并保存实际环境的显式锁定文件。
PNL 在反序列化前核对固定源码、模型与模板哈希，并执行三方向模型有限值前向检查，保存 pip 包清单。
CUDA wheel 的库和工具仅对该 Python 子进程可见，不修改全局 PATH、LD_LIBRARY_PATH 或驱动。
默认优先可用 CUDA，不统一禁用 GPU；固定作者兼容框架首次运行时可能需编译 GPU 内核。
这些检查不能代替科学正确性验证、真实干净安装、SDK 打包和跨平台验收。
实际 GPU 证据见[核查记录](../release/GPU_POLICY_20261013.md)。

---

## 重启后继续

经过用户确认但尚未完成的准备操作会跨重启保留。现有按钮显示为“继续准备所选分析环境”，
重新进入模块会提示续接，也可点击按钮继续；重新检查当前选择，必要时再次确认，不会重放保存的命令、把旧许可确认用于不同计划，
或自动启动科学计算。已有的合格资源与环境会复用，最终就绪检查成功后才清除待续接标记。

仓库的完整 Skill 新增 `prepare_in_slicer.py`，供 Agent 导出计划并进入同一 GUI 准备流程，
保留真实确认弹窗，不自动接受许可或启动患者数据分析。
见 [Agent 安装参考](https://github.com/NeuroDMRI/SlicerDSpace/blob/main/.agents/skills/slicerdspace/references/setup.md)。

---

## 维护者接口

`scripts/setup_selected.py --targets <targets>` 输出与图形界面相同的计划；`--execute` 执行，`--accept-licences` 接受计划中的全部许可，用 `PythonSlicer` 运行。
`scripts/setup_managed_runtime.py` 默认只生成计划；`--execute` 才执行安装，
FSL 另需 `--accept-fsl-license`，PNL 另需 `--accept-pnl-terms`，
`--repair` 显式重装所选受管环境。
`--root` 或 `SLICERDSPACE_RUNTIME_ROOT` 用于隔离测试。
后台 worker 成功后返回经过校验的设置，GUI 再保存路径。
不修改用户 shell、GPU 驱动或非插件管理的 Conda 目录。
