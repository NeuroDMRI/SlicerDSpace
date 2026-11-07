<div align="center">

# SlicerDSpace 产品架构

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [外部依赖](../setup/EXTERNAL_DEPENDENCIES.md) | [发布](../release/EXTENSION_READINESS.zh-CN.md) | [English](PRODUCT_ARCHITECTURE.md)

</div>

---

## Slicer 原生客户端与经过核验的后端

插件负责 Qt 界面、MRML 参数、任务执行与取消、日志、按运行记录加载结果及场景检查。
科学计算由支持包和经过核验的后端负责，并保留来源、版本、参数和 QC。

[官方扩展文档](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
允许依赖其他扩展和第三方库。“Slicer 插件风格”应落实在依赖管理、执行管理和场景集成，
**不是必须把 FSL eddy、MRtrix 或神经网络重新写一遍**。
替换算法需要独立科学验证，不能仅因输出文件名一致就称为等价实现。

---

## 依赖分层

| 层级 | 内容 | 管理方式 |
| --- | --- | --- |
| Slicer 集成 | Qt、MRML、读写和显示、Slicer 内核中的 Teem 张量拟合；PyTorch 扩展（已声明），UKFTractography 和 SlicerDMRI（仅在所勾选产品需要时由准备步骤请求） | 只声明一个扩展依赖，使用 Slicer API，不另外创建桌面客户端 |
| Python 科学计算 | DDSeg/DDParcel/DDSurfer、DIPY、AMICO、区域统计 | 按所选功能安装，保护 Slicer 已有环境，共用兼容 PyTorch |
| 隔离 Python worker | managed PNL 运行环境（Python 3.11、PyTorch、ANTs 2.3.0；Linux）、独立模型环境 | 在实际执行解释器中检查和运行，不强行向 Slicer 混装冲突框架。macOS 和 Windows 上 PNL 掩膜改在 Slicer 自带的 Python 中运行（native 路线） |
| 可选外部工具 | MRtrix/FSL、FreeSurfer 后处理、WMA/WMQL/TractCloud 资源 | 明确版本与路径，执行前体检，采用官方来源并核对许可 |
| 模型资源 | 权重、模板和图谱 | 按需下载、校验哈希、记录来源；授权单独验收 |

目前默认脑掩膜后端仍为 PNL CNN。没有输入掩膜时，它需要 PyTorch、模型资源，Linux 上还需要 managed 的 ANTs 运行环境；macOS 和 Windows 上由 native 路线（Slicer Python 中的 SimpleITK 配准和 PyTorch，标注为“native（实验性）”）取代 managed 运行环境。
**不能把它描述成零外部依赖的默认路线**。用户提供掩膜，或主动选择替代后端，
可以避免调用该 worker。局部验收用过的 FreeSurfer Docker 大镜像不是插件必装依赖，
PNL 本身不要求安装这个镜像。替换后端必须明确选择，不能静默切换。

基础 DTI 使用不应要求用户装齐所有高级工具。CUDA eddy 是 MRtrix/FSL 预处理的可选
Linux NVIDIA 组件，与插件的 PyTorch 设备选择、脑掩膜设备选择相互独立。

---

## 模块与结果

| 源码目录 | 职责 |
| --- | --- |
| `shared/` | 输入与目录契约、掩膜适配、依赖体检、编排、QC、设置及结果回载 |
| `preprocessing/` | 预处理（唯一路线：DIPY MP-PCA 和 Gibbs，然后通过 `dwifslpreproc` 运行 FSL eddy/topup）、校正梯度和诊断产物 |
| `anatomy/segmentation/` | DDSeg 组织分割 |
| `anatomy/parcellation/` | DDParcel 分区及格式转换 |
| `anatomy/surface/` | 固定 DDSurfer 后端、原生表面及可选后处理 |
| `microstructure/` | 默认 DTI，可选 FW/NODDI/DKI/MK-Curve、适用性检查和区域统计 |
| `tractography/` | DTI 或 MRtrix CSD/ACT、SAF、WMA/WMQL/TractCloud 封装 |
| `connectome/` | Python/MRtrix 矩阵、节点表和定量/可视化产物 |

关键结果按运行记录注册，并加载到当前 Slicer 场景。TCK 使用派生的 RAS VTK 显示预览，
不改原始纤维。执行完成、自动 QC 和待人工检查是不同状态，不能视为科学正确性的同义词。

---

## 已实现管理与剩余发布工作

已有 `PyTorch` 扩展依赖声明（不再声明 SlicerDMRI；张量拟合使用 Slicer 内核中的 Teem），使用 PyTorch 扩展管理框架，
并检测可用的 `slicer.packaging` API 安装 Python 依赖。
依赖配置、外部 worker 路径和模型哈希管理已经实现。
[插件内自动配置](../setup/MANAGED_SETUP.zh-CN.md)现支持 Linux x86_64 的 MRtrix/FSL
按需安装与路径自动保存；macOS 14.5 或更高时还可安装 MRtrix3 和 FSL CPU 工具；Windows 上 WSL 2 运行环境列入同一个确认对话框；其他外部工具仍需显式配置。
**尚未实现并验收所有后端的跨平台一键安装**。

正式分发前还需要可选环境的干净安装验收、可信且固定版本的下载、必要的授权确认入口，
以及安装过程的进度和取消验证。
[FSL 许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)对多数组件限制非商业使用，
不能直接把二进制归入插件的 MIT 许可。
[Extension Index 清单](https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
要求二进制来源可靠，并在 Tier 3 要求用户明确批准非宽松许可的可选组件。

真实 SDK/CPack、受支持平台验收仍是独立门禁，本机 GPU 运行或资源检查通过不能代替。
仓库可见性保持不变，本说明不代表已经公开发布或获官方目录收录。

---

## 对标方式

[SlicerTotalSegmentator](https://github.com/lassoan/SlicerTotalSegmentator)
也在 Slicer 内安装 AI 依赖并下载权重。应借鉴这种方式：保留成熟算法、减少必装依赖、
在插件内清楚管理安装和执行，而不是为“原生化”重新实现所有算法。
该案例不能作为本插件尚未完成的安装验收证据。
