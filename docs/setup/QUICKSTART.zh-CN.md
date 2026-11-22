<div align="center">

# 快速开始

**一条最小、便于检查的工作流**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [平台](SUPPORT_MATRIX.zh-CN.md) | [QC](GUI_RESULT_REVIEW.md) | [English](QUICKSTART.md)

</div>

## 安装 3D Slicer 5.10.0 与 SlicerDSpace

默认界面不显示首次向导；当前项目尚未进入官方 Extensions Catalog。

1. **安装 3D Slicer 5.10.0**：从 <https://download.slicer.org/> 下载对应操作系统的安装包（Windows `x86_64`、Linux `x86_64` 或 Apple Silicon macOS `arm64`），默认安装与启动器路径如下：
   - **Windows**：`%LOCALAPPDATA%\slicer.org\Slicer 5.10.0\Slicer.exe`（或 `C:\Program Files\Slicer 5.10.0\Slicer.exe`；内置 Python 为 `bin\PythonSlicer.exe` 或 `PythonSlicer.exe`）
   - **Linux**：解压目录如 `~/Slicer-5.10.0-linux-amd64/Slicer`（或 `/opt/Slicer-5.10.0-linux-amd64/Slicer`；内置 Python 为 `bin/PythonSlicer`）
   - **macOS**：`/Applications/Slicer.app`（CLI 启动器为 `/Applications/Slicer.app/Contents/MacOS/Slicer`；内置 Python 为 `/Applications/Slicer.app/Contents/bin/PythonSlicer`）
2. **下载 SlicerDSpace 发布包**：前往 <https://github.com/NeuroDMRI/SlicerDSpace/releases> 下载对应系统的发布归档（`SlicerDSpace-v*-Windows-x64.zip`、`SlicerDSpace-v*-Linux-x64.tar.gz` 或 `SlicerDSpace-v*-macOS-arm64.tar.gz`；请勿下载 GitHub 自动生成的 `Source code` 归档）。
3. **在 3D Slicer 中安装**：打开 **View > Extensions Manager**，点击右上角扳手图标并选择 **Install Extension from File...**，选中下载好的发布包，重启 3D Slicer 后打开 **Modules > Diffusion > SlicerDSpace**。
4. **发布包解压后的磁盘路径**（内含 `lib/Slicer-5.10/qt-scripted-modules/shared/run_subject_pipeline_cli.py` 与 `bundled-runtimes/`，也可在 **Edit > Application Settings > Modules** 中查看）：
   - **Windows**：`%LOCALAPPDATA%\slicer.org\Slicer 5.10.0\slicer.org\Extensions-34045\SlicerDSpace`
   - **Linux**：`<Slicer-5.10.0-dir>/slicer.org/Extensions-34045/SlicerDSpace`
   - **macOS**：`/Applications/Slicer.app/Contents/Extensions-34045/SlicerDSpace`（或 `~/Library/Application Support/slicer.org/Extensions-34045/SlicerDSpace`）

若需开发者本地源码部署而非发布包安装，请参见[项目中文主说明](../../README.zh-CN.md#installation)。

---

## 更新代码

发布包内是打包当天的代码。之后的修复以同一版本的增量代码更新发布。安装后请应用一次更新；有新的更新发布时再应用一次。

- **界面**：在 SlicerDSpace 模块中点击 **检查更新**，确认下载并应用，然后重启 3D Slicer。
- **仅命令行**（例如没有显示器的服务器）：运行下面的命令。`MODULE_ROOT` 是扩展解压目录中的 `lib/Slicer-5.10/qt-scripted-modules` 文件夹（见上面第 4 步）。示例使用 Linux 路径。

~~~bash
SLICER_DIR="$HOME/Slicer-5.10.0-linux-amd64"
MODULE_ROOT="$SLICER_DIR/slicer.org/Extensions-34045/SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules"
"$SLICER_DIR/bin/PythonSlicer" - "$MODULE_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from shared import code_updater
check = code_updater.check_for_code_update(module_root=root)
print("status:", check["status"])
if check["status"] == "update_available":
    print(code_updater.download_and_apply_code_overlay(module_root=root, manifest=check["remote"]))
PY
~~~

第一次运行先输出 `status: update_available`，再输出含 `'status': 'applied'` 的结果。再次运行输出 `status: up_to_date`。更新使用已保存的网络设置（镜像、代理）。更新只改代码文件，不下载模型或运行时。状态 `runtime_upgrade_required` 表示需要安装新的发布包。

---

## 准备输入

使用获授权、去标识化的 NIfTI（含匹配的 bval/bvec）、NRRD/NHDR 体数据或 DICOM 文件夹（含多序列时可在提示中选择目标序列）。
在 **Inputs（输入）** 中直接选择 DWI 数据源即可，点击运行后会自动导入并校验至 `<subject>/input/`。脑掩膜是可选的：
既可在 **Inputs** 中提供已有掩膜，也可留空交由所选掩膜后端（默认 PNL CNN）生成。DTI 拟合在 Slicer 内部完成，不需要额外扩展。

在 **Analysis（分析）** 下勾选所需阶段，插件会自动加入必需的前置步骤；预处理不会在未勾选时被加入。

点击 **Run SlicerDSpace** 时，系统会先对所选阶段执行依赖预检（仅尝试 DTI 不必安装 MRtrix3、FSL 或 FreeSurfer；
依赖不足时会如实提示，不会悄悄换成另一种科学算法）。若缺少软件包、外部工具或模型权重，可通过弹出的
**准备所选分析环境（Prepare selected analysis environment）** 对话框查看下载项与许可，点击一次 **接受许可并安装** 即可下载、校验并缓存复用
（已经就绪时不再重复确认或安装；如提示重启，请重启 Slicer 后再次点击运行）。
准备提示中的 **更多选项（离线、修复、路径）** 提供依赖详情、**使用离线模型资源**、
**修复所选受管工具** 与手工路径设置，支持范围与平台限制见[自动配置说明](MANAGED_SETUP.zh-CN.md)。

**Analysis** 区域各阶段行显示当前算法配置（默认分别为 **DTI**、**MRtrix3 CSD / ACT** 和 **WMQL**），
修改设置后标签会同步更新（仅代表运行配置，不表示结果已完成或 QC 已通过）。点击各阶段旁的 **...** 可多选算法并调整阶段参数
（如在 DTI 之外按需启用 FW、NODDI、DKI 或需提供回波时间的 iFW，后者同时运行 FW 与 DDSeg），并在 **Compute device** 中选择 Auto、CPU 或指定 CUDA GPU。

多种追踪方法顺序运行，以 `<run>__<backend>` 分别保存，共用解剖与微结构只运行一次。
束分析与连接组针对各追踪结果分别计算，不混合流线；依赖与 QC 要求不变。
多选将增加耗时和磁盘占用。

---

## 仅运行 DTI

1. 在 **Inputs** 中选择 Subject root（或直接选择 NIfTI/NRRD/DICOM 数据源）；空白被试 ID 自动取目录名，已有自定义 ID 不被覆盖。
   误选规范目录的 `input/` 子目录会自动回到其被试根目录。Output directory 留空时使用被试下的 `outputs/`。
2. 在 **Analysis** 中只勾选 **Microstructure**，取消其他阶段及预处理。
3. 点击 **Microstructure** 旁的 **...**，仅保留 DTI。
4. 点击 **Run SlicerDSpace**，查看日志。系统会先执行运行前依赖检查；
   如缺少依赖，请按准备提示完成配置后再重新运行，不会静默遗漏已选阶段。
5. 运行生成的输出按需加载，不会自动载入场景。请在 **Results and QC（结果与 QC）** 中选择已完成运行，点击 FA/MD 旁的眼睛图标按需加载至场景或切换显隐（右键可 `Unload from scene` 卸载或直接打开表格/JSON 文件），并点击 **Open report** 查看 HTML QC 报告。
6. 检查脑掩膜、数值有限性、空间对齐和 QC 摘要，再接受结果。

只生成计划、不执行算法：

~~~bash
PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets validation,microstructure --microstructure-stages dti --plan
~~~

MD 使用灰度；显示设置不修改真实参数。执行完成、自动 QC 和人审是不同状态。
默认只跑 DTI；FW、NODDI、DKI、iFW 在该行的选择框分别启用。DTI 产品包含彩色 FA 图。缺少掩膜时使用所选后端
（默认 PNL CNN），需要先配置其外部环境；已有掩膜不需要生成器依赖。

---

## 在 CLI 中选择束分析的追踪来源

各路线的 Bash/PowerShell 启动方式、掩膜、后处理、可选模型、多被试及报告命令，
见[完整命令行模板](CLI_TEMPLATES.zh-CN.md)。

GUI 的参数覆盖、命名实验组合和多被试调度见[轻量依赖与批量实验](EXPERIMENTS.zh-CN.md)。
默认 GUI 仍运行单个被试（或在 **Inputs** 下切换 **Batch** 批处理多被试目录；命名参数变体与批量调度详见命令行及实验说明）。

CLI 可以分别选择全脑追踪来源和束分析方法。例如，同时运行 CSD/ACT 和 UKF，
但仅对 UKF 结果运行 WMQL 和 TractCloud：

~~~bash
PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets segmentation,parcellation,tractography,bundles \
  --tractography-backend csd_act,ukf --bundle-backend wmql,tractcloud \
  --bundle-tractography-backend ukf --plan
~~~

检查输入和依赖后将 `--plan` 换成 `--execute`。来源可选 `ukf`、`csd_act`、
`dti_det`，必要时会自动加入追踪任务。所有选中的束分析方法共享这一来源；
如需不同方法使用不同来源，请使用不同的 `--run-name` 分别运行。未指定新参数时，
保留原有逐追踪路线执行方式。不能同时指定该参数和 WMA/TractCloud 的外部输入文件。
SAF 是短程联络纤维结果，不作为全脑来源。WMA 使用非 UKF 来源时尤其需要关注束覆盖，
现有 QC 门槛和人工解剖审核仍然适用；能够运行不等于束覆盖合格。正式执行前可用
`--doctor` 检查所选方法的依赖。

---

## 合成安装测试

在具备 NumPy/nibabel 的环境、从仓库根目录执行：

~~~bash
PythonSlicer -c "from pathlib import Path; from shared.demo_data import create_synthetic_demo_subject; print(create_synthetic_demo_subject(Path('ignore/demo')))"
~~~

使用生成的 <code>ignore/demo/slicerdsurfer-demo</code> 作为 Subject root。
这是七个测量的小型恒定张量场，**不是人体脑数据**，只检查安装与 DTI 链路，
不能验证表面精度、NODDI 或临床有效性。

旧 Demo 回调保留给测试，不是公开按钮。真实 DWI 入门可参考
[Slicer DWIVolume 教程](SLICER_SAMPLE_DTI.zh-CN.md)，复用 Slicer 已有的
Sample Data 条目，不镜像原始影像。本插件独立可再分发数据的注册仍待完成。

---

## 失败处理

缺字段时 waiting 提示变红。依赖不足请先通过运行前准备提示完成配置后再试。
取消不会回滚已安装的包；模型部分下载可续传。共享日志前删除私人路径和被试信息。
