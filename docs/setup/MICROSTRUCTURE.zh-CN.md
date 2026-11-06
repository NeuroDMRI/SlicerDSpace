<div align="center">

# 微结构分析

[项目](../../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [结果与 QC](GUI_RESULT_REVIEW.md) | [English](MICROSTRUCTURE.md)

</div>

---

## 选择模型

主界面勾选 Microstructure 后，**默认仅运行 DTI**。
点击 Microstructure 旁的 **...**，分别启用 FW、NODDI、DKI，不要求逐级同时开启。
DTI 为基础步骤；DKI 拟合方法在“高级设置 > 模型参数”调整。

| 模型 | 主要结果 | 实现 |
| --- | --- | --- |
| DTI | FA、MD、AD、RD、三个特征值、Trace | 共用 native-space 张量拟合 |
| FW | 自由水比例、组织 FA/MD、有效性掩膜与 QC | DIPY |
| NODDI | NDI、ODI、IsoVF、残差、有效性掩膜与 QC | AMICO |
| 标准 DKI | MK、AK、RK、KFA、MKT、RTK、残差与原始张量参数 | 内置 Python MK-Curve（默认），或 DIPY WLS/OLS |
| MK-Curve | 原始及校正 MK/AK/RK、FA/MD/AD/RD、特征值、有效性与残差 | 内置作者算法 Python 移植版，不是普通 DIPY MK |

AD/RD 已存在；本次补充 DKI 的 MKT/RTK/KFA，参考
[DIPY 官方说明](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html)。
不为增加指标数量而默认拟合需要特殊采集的 IVIM 等模型。

---

## 数据检查与 QC

DKI 在拟合前检查 4D 数据及匹配梯度、b0、至少两个非零壳层、
每壳层至少 15 个独立方向和角度矩阵秩 15、完整拟合矩阵秩 22。
还检查脑掩膜的网格、仿射和二值内容。名义 b0 的方向只在工作梯度表中清零，
记录处理方式，不修改输入。单壳层或秩不足会阻止 DKI，而非阻止可用的 DTI。

b0 阈值为 50 s/mm²，壳层容差 100 s/mm²；默认仅纳入中位 b 值不超过
2500 s/mm² 的完整壳层，避免在名义壳层边界零散剔除测量。使用的 b 值上限、被排除的测量和
高 b 值偏倚警告会记录。可辨识不等于信噪比充分，也不等于科学结果已验证。

标准拟合不额外平滑、去噪、重采样或填洞。`*_raw.nii.gz` 保留未截断估计；
显示副本采用峰度 0–3、KFA 0–1，**不是校正结果**。
未拟合的脑内体素保留 NaN，并提供拟合掩膜、有效性掩膜及残差。
统计使用原始值和独立有效性条件，不用显示截断制造“正常”结果。

---

## Python MK-Curve

在高级设置选择 `MKCURVE` 即使用内置 Python 移植版，依赖 NumPy、SciPy、nibabel，
不需要 MATLAB 或另行下载作者代码。按[作者源码](https://github.com/zhangfanmark/MK-Curve)
提交 `9a213e1a5850fe50c998793f31ed852df00242f8` 移植：OLS 初始化、预测信号权重 WLLS、
原始 256 方向数值 MK 积分、合成 b0 扫描、异常曲线检测和阈值 0.5 的 b0 校正。
这是项目维护的移植版，不是作者正式发布的 Python 版本；保留完整许可证和来源。

多个 b0 仅在工作副本中取均值；非正/非有限信号按作者的正值邻域均值策略修复，
记录修复和未能修复的数量，原始输入不变。原始及校正参数不截断，另存有效性掩膜；
区域统计使用校正原始值与其有效掩膜。保存曲线、b0 网格、拟合系数和哈希。
KFA/MKT/RTK 仍是独立 DIPY 参考结果，不称为 MK-Curve 校正指标。

移植版明确使用作者的自适应整数信号网格，不根据输出目录名称猜测 HCP/Ndyx；
不适用的归一化信号尺度会报错，不会静默重标定。增加径向积分极点保护，
非有限曲线标为无效，不强行纠正。拟合分块，输入为磁盘映射 float64 工作副本，
全脑耗时与内存仍需另行验收。实测范围见[移植验证](../release/MKCURVE_PYTHON_20261012.md)。

`MKCURVE_MATLAB` 仅作为可选作者实现对照后端，需要在高级外部运行路径设置 MATLAB
及作者代码目录；缺少依赖时明确阻止，不回退为标准 DKI。

---

## 运行与结果

```bash
python3 run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets microstructure --microstructure-stages dti,dki \
  --dki-fit-method WLS --plan
```

正式执行用 `--execute`。可另加 `--run-free-water`、`--run-noddi`、`--run-dki`；
`all` 明确表示四种模型全部启用，不是默认值。
Python MK-Curve 只需 `--dki-fit-method MKCURVE`。
仅可选对照后端使用 `--dki-fit-method MKCURVE_MATLAB --dki-matlab-executable /path/to/matlab
--dki-mkcurve-root /path/to/MK-Curve`。

输出位于 `02_microstructure/dki/`，附带 manifest、QC、progress。
缓存检查输入、配置、库版本和结果哈希。GUI 在成功阶段后自动加载代表图，
执行完成、自动 QC 和人工核验仍然是三个独立状态。
