<div align="center">

# 快速开始

**一条最小、便于检查的工作流**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [平台](SUPPORT_MATRIX.zh-CN.md) | [QC](GUI_RESULT_REVIEW.md) | [English](QUICKSTART.md)

</div>

先按 README 完成源码安装。默认界面不显示首次向导；
当前项目尚未进入官方 Extensions Catalog。

---

## 准备输入

使用获授权、去标识化的 DWI 及匹配的 bval/bvec，或 DICOM 序列。脑掩膜是可选的：
没有时由所选掩膜后端（默认 PNL CNN）生成。按 README 的规范目录存放。DICOM/NRRD 转换位于
**高级设置 > 脑区与输入导入 > 输入导入 / 转换**，先预览后写入。DTI 拟合在 Slicer 内部完成，不需要额外扩展。

勾选所需产品，插件会自动加入每个产品的前置步骤；预处理不会在未勾选时被加入。

在独立的 **环境与安装** 区域检查环境。仅尝试 DTI 不必安装 MRtrix3、FSL 或
FreeSurfer。依赖不足会报告阻碍，不会悄悄换成另一种科学算法。

检查在可取消的后台进程中进行，结果直接显示在环境区；无需逐次关闭详情弹窗。
点击 **准备所选分析环境**。一个对话框列出下载项及其许可，点击一次 **接受许可并安装** 即可；已经就绪时不再确认或安装。
如提示重启，保存工作并重启后再次准备。安装结束后，会对每项功能做自检。
**更多选项（离线、修复、路径）** 中提供依赖详情、**使用离线模型资源**、
**修复所选受管工具** 和手工路径。
支持范围与平台限制见[自动配置说明](MANAGED_SETUP.zh-CN.md)。

主界面的 Microstructure、Tractography 和 Fiber bundles 三行显示当前算法，默认分别为
**DTI**、**MRtrix3 CSD / ACT** 和 **WMQL**。修改设置后提示会同步更新；这是运行
配置，不表示结果已完成或 QC 已通过。点击每行旁的 **...**，可勾选一种或多种算法。
DTI 为基础步骤；FW、NODDI、DKI、iFW 可分别勾选（iFW 需要回波时间，并同时运行 FW 和 DDSeg）。高级设置按类别组织，每次只显示当前类别参数，
环境安装独立于高级设置。

多种追踪方法顺序运行，以 `<run>__<backend>` 分别保存，共用解剖与微结构只运行一次。
束分析与连接组针对各追踪结果分别计算，不混合流线；依赖与 QC 要求不变。
多选将增加耗时和磁盘占用。

---

## 仅运行 DTI

1. 选择 Subject root；空白的被试 ID 自动取目录名，已有自定义 ID 不被覆盖。
   误选规范目录的 `input/` 子目录会自动回到其被试根目录。Output directory 留空时使用被试下的 `outputs/`。
2. 只勾选 **Microstructure**，取消其他产品及预处理。
3. 点击 Microstructure 旁的 **...**，仅保留 DTI。
4. 点击 **Run SlicerDSpace**，查看日志。必要的环境检查在后台自动完成；
   缺少依赖会停止提交并展开环境区，准备好后再运行，不会静默遗漏勾选产品。
5. 成功阶段的 FA/MD 自动加载；在独立的 **结果与 QC** 区域检查报告，或用 **加载** 再次加载而不堆积重复节点。
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
默认 GUI 仍运行单个被试，实验功能在高级设置中主动启用。

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

缺字段时 waiting 提示变红。依赖不足应先到“环境与安装”配置。
取消不会回滚已安装的包；模型部分下载可续传。共享日志前删除私人路径和被试信息。
