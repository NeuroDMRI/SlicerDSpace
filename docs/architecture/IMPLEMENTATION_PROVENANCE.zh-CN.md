<div align="center">

# 实现来源与移植边界

**区分原始方法、Python 移植和外部调用适配**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [SAF 核验](../release/SAF_PYTHON_20261013.md) | [English](IMPLEMENTATION_PROVENANCE.md)

</div>

代码改写、程序跑通、数值等价和解剖学正确是不同层次。
以下说明插件实际使用的实现，不表示原作者背书或临床有效性认证。

---

## 实现总览

| 模块 | 原始实现 | 插件执行方式 | 应采用的表述 |
| --- | --- | --- | --- |
| DDSeg | [原作者 MATLAB 流程及 U-Net](https://github.com/zhangfanmark/DDSeg) | `anatomy/segmentation/`：Python 预处理/推理、转换后的 PyTorch/TorchScript 网络 | **MATLAB 到 Python/PyTorch 的改写实现**，不是重新训练的新组织模型 |
| SAF | [原作者 MATLAB 框架](https://github.com/dmitrishastin/SAF/tree/7e2a9d152caf1cf3e6bface64e7e176faac0c14f) | `tractography/saf_backend/`：NumPy/SciPy/scikit-learn、NiBabel、外部 MRtrix | **非官方 Python 移植版**，限已对齐的原生扩散空间表面，并保留明确记录的几何修正 |
| PNL diffusion CNN 脑掩膜 | [原作者 Python/TensorFlow 代码](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation)，经 [PyTorch 版 PNLBrainMask-Torch](https://github.com/ChengjinLii/PNLBrainMask-Torch) | `shared/pnl_cnn/` 是 PNLBrainMask-Torch 的修改副本（固定提交，哈希见 `UPSTREAM.json`）；`shared/brain_mask.py` 在 Slicer Python 中原生运行它（SimpleITK 配准），Linux 上则在 managed 运行环境中用 ANTs 运行已安装的包 | **作者 PNL 网络的修改版 PyTorch 重新实现，保留 BWH 许可条款**（[许可页](../licenses/pnl_brain_mask.md)）；不是 MATLAB 移植，也不是重写整套 PNLpipeline；native 路线为实验性 |
| MK-Curve | [原作者 MATLAB 算法](https://github.com/zhangfanmark/MK-Curve/tree/9a213e1a5850fe50c998793f31ed852df00242f8) | `microstructure/mkcurve.py`、`mkcurve_tensor.py`，可选 MATLAB 对照 | **非官方 MATLAB 到 Python 移植版**，不能与普通 DIPY DKI 混称 |
| DDSurfer 表面重建 | 上游 Python/PyTorch 项目 | 固定的 `anatomy/surface/ddsurfer_backend/` 快照与 Slicer 调度 | 同步上游 Python，不是 MATLAB 改写 |
| 皮层后处理 | 原始 FreeSurfer C++ 算法 | 固定上游配方在本机编译七个命令，或复用已有 FreeSurfer | **原生算法复用**，不是 Python 重写；本机组件无需完整套件；[配置与边界](../setup/SURFACE_COMPONENTS.zh-CN.md) |
| DTI 张量拟合 | SlicerDMRI `DWIToDTIEstimation` 与 Teem 库（Slicer 内核） | `shared/teem_dti.py` 在 Slicer Python 中调用 Teem，写出张量、FA 和彩色 FA 图 | **SlicerDMRI 估计器的修改版 Python 重新实现**，遵循 Slicer 许可（[声明](../../THIRD_PARTY_NOTICES.md)）；插件不再分发 Teem |
| Free-water / NODDI | 所选路线的 DIPY / AMICO Python 实现 | 输入验证、拟合适配、参数记录与 QC | 调用科学计算库，不能笼统称为本项目 MATLAB 移植 |
| iFW | 已发表公式（Cho 等预印本，补充方法 1） | `microstructure/ifw.py` | 已发表公式的 Python 实现；遵循论文的参考函数 |
| Free-water WLS 优化 | 固定来源的 DIPY `wls_iter` Python 估计器 | 源码匹配时用 `microstructure/freewater_wls.py` 消除大对角矩阵；串行 NLS 沿用同一初始化 | **修改后的等价 Python 计算实现**，不是新估计器或 MATLAB 移植；见[配对实测](../release/PERFORMANCE_OPTIMIZATIONS.zh-CN.md)及[DIPY 授权](../../licenses/DIPY.txt) |
| WMA / WMQL / TractCloud | 外部作者 Python/shell 流程 | 独立运行目录、原始入口和溯源记录；TractCloud 增加固定种子/logging 兼容启动器 | **外部作者代码适配**，不是算法移植；[真实核查](../release/BUNDLE_BACKENDS_20261013.zh-CN.md) 区分执行、完整性和人工解剖核验 |

代码许可和模型再分发权限仍需分别审查，见[第三方声明](../../THIRD_PARTY_NOTICES.md)
与[模型资源条款](../assets/MODEL_ASSET_TERMS.md)。

---

## DDSeg

原始 MATLAB `DAGNetwork` 的权重、偏置和输入均值图先导出为数值文件，
再映射到 PyTorch 层并转换为 TorchScript。Python 负责掩膜、归一化、
三视图推理与融合、组织概率和原生网格导出。仍需引用
[Zhang 等，NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.117934)。

当前组织路线使用 **5 种 DTI 特征**，不是作者另一个 10 特征 MK-Curve
组织网络。新增 DKI 参数图不意味着自动替换 DDSeg 模型。
已有概率、坐标和真实被试 QC，不能替代相同输入下 MATLAB/PyTorch
逐视图预测对照；这里不声称已完成这种完整原始运行时等价验证。

---

## SAF

Python 移植保留皮层端点条件（GG）、同侧半球筛选（HH）、可选软脑膜过滤
（PG）、初始纤维长度上限、白质表面穿行与裁剪（GWG）及表面统计。
但它不是原 MATLAB 环境的全功能复刻：插件要求表面已经位于原生扩散空间，
不执行结构像到扩散像的自动配准。

明确差异包括固定种子的 scikit-learn k-means、端点与三角面索引修正、
complete 模式保留边/顶点交点、极短纤维原始索引的保留，以及 hard-split
权重按原纤维索引映射。MATLAB 和 scikit-learn 的聚类结果可能不同，
不能承诺全流程逐位一致。

原版默认初始长度上限为 **90 mm**、PG 关闭、GWG 保留最外侧交点之间的链。
代表性验证运行明确使用 **40 mm**。soft GWG 不保证链的每个内部点都位于
白质；`gwg_hard` 的拆段含义不同。两者都没有“相邻脑回标签”或“必须呈 U 形”
的分类器。因此应称为**表面约束的 SAF 纤维追踪候选**，不能只凭截图保证
每条都是典型 U-fibre。仍需引用
[Shastin 等，NeuroImage 2022](https://doi.org/10.1016/j.neuroimage.2022.119423)。

旧的端点密度、长度和标量表面投影应使用 `python-native-20261013`
重新计算。续跑检查拒绝缺失或旧版实现标识，避免将旧投影当成已修复结果。
核验范围和未完成项见[验证记录](../release/SAF_PYTHON_20261013.md)。

---

## PNL 脑掩膜

PNL 上游已经是 Python。插件新增的是 Python 调用适配层：准备原生 DWI/b0，
调用所配置的官方外部运行环境，记录资源哈希，恢复并检查掩膜几何，报告失败。
CNN 架构、学习权重和 ANTs 配准仍来自配置的作者后端。
**不应写成“我们把整套 PNLpipeline 从 MATLAB 改成 Python”。**

已有输入掩膜时保留原掩膜；需要生成时才用 PNL 默认路线。
替代方法在 Advanced 明确选择，不静默切换。见[配置说明](../setup/brain_mask_gui_config.md)。

---

## 验证表述

每次运行记录实现标识、外部版本、参数、输入哈希与 QC 范围。
自动几何/QC 和专家解剖复核分别说明。[MK-Curve](../release/MKCURVE_PYTHON_20261012.md)
和 [SAF](../release/SAF_PYTHON_20261013.md) 的原版 `.m` 对照在注明范围内
由 GNU Octave 执行，不是 MATLAB 本身，也不能推广为所有可选分支、所有被试均已验证。
