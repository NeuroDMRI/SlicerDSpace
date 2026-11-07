<div align="center">

# UKF 纤维追踪

**可选官方扩展 · 明确、可追溯的作者默认参数**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [作者项目](https://github.com/pnlbwh/ukftractography) | [English](UKF.md)

</div>

---

## 使用方式

默认追踪仍为 **MRtrix CSD/ACT/SIFT2**。需要 UKF 时，在 Extension Manager
安装 **UKFTractography**（UKF 的错误提示还会提到 SlicerDMRI，WMA 需要它），重启 Slicer，再到
点击 Tractography 旁的 **...** 勾选 `UKF`。
束分析旁同样选择 `WMA` 后，自动执行 **UKF → WMA**，不必先手动运行 UKF 再导入。
UKF 参数位于 **高级设置 > 模型参数**。如同时选择其他追踪算法，各自生成独立的运行结果。
ORG 图谱及 WMA 后端仍需配置；已有流线文件可在 Advanced 中作为可选覆盖输入，
不是标准流程的必需步骤。GUI 同时发现 UKF 和 DWIConvert，并向后台显式传递两者
的路径；不要求用户复制文件、手动运行转换器或在另一个模块先生成流线。

插件调用已安装的作者 CLI，**不复制、不重写 UKF 算法**。缺少依赖则明确阻断，
不静默换算法。UKF 本身不要求 DDSeg、DDParcel 或表面；WMQL 和连接组仍需脑区分区。

---

## 默认参数

取值依据为[已复核版本的作者模块 XML](https://github.com/pnlbwh/ukftractography/blob/2d2b661539700aa825c1949823119601e11e1823/UKFTractography/UKFTractography.xml)。
这里称为**作者模块默认值**，不宣称它们是所有采集协议的统一最优值，也不等同于
已经通过 WMA 逐束采样验证的专用协议，更不是针对当前被试调出的参数。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `numTensor` | 2 | 双张量 |
| `seedsPerVoxel` | 1 | 每个符合条件体素的种子数，不是目标流线总数 |
| `seedingThreshold` | 0.18 | 播种 FA 下限 |
| `stoppingFA` | 0.15 | 终止 FA |
| `stoppingThreshold` | 0.10 | 终止平均归一化信号，不是已废弃的 GA 阈值 |
| `stepLength` | 0.3 mm | 追踪步长 |
| `recordLength` | 0.9 mm | 输出点间距 |
| `maxHalfFiberLength` | 250 mm | **半束**最大长度，不能当作整条流线 250 mm 上限 |
| `freeWater` | false | 可选 UKF 自由水项，与微结构模块逐体素 FW 拟合不同 |
| `numThreads` | −1 | 作者 CPU 自动线程；界面中 0 表示自动 |
| `Qm / Ql / Qw / Rs` | 0 | 作者按模型选择噪声，**不是零噪声拟合** |
| `recordFA / recordTensors` | true | VTK 保留作者默认 FA／张量数组 |

UKF 参数仅在选择该后端时显示，默认折叠；可一键恢复作者默认值。
CSD 的目标流线数隐藏且不参与 UKF。作者实现使用 CPU 多线程，不是插件禁用其他
算法的 GPU。使用全部输入壳层，不自动筛选壳层，也不在 QC 失败后偷偷增加密度或放宽阈值。

---

## 无损输入与结果管理

拟合前检查 DWI 有限值、梯度长度与方向设计矩阵秩，以及原生网格二值掩膜。
插件将输入掩膜与平均 b0 信号为有限正值的区域取交集，单独保存为
`<subject>_ukf_valid_mask.nii.gz`，用于转换、追踪及掩膜重合 QC。
排除体素的数量、比例和提示写入运行记录及 QC，不修改原始 DWI 或用户提供的掩膜。
有效掩膜为空或排除比例超过 1% 时停止并要求检查输入；1% 是保守的输入完整性保护线，
不代表解剖准确性阈值。派生掩膜也参与缓存完整性核验。
这些检查不能代替采集质量及双张量模型适用性评估。

运行前读取**当前安装版本**的 CLI 参数描述。有些作者构建的 `--xml` 会先完整输出
XML，再进入空输入读取而返回非零；只在 XML 完整且可解析时接受描述，并记录返回码。
真正的转换和追踪必须成功退出，不能忽略其错误。
FA／张量输出开关显式传入：SEM 生成的 CLI 可能将布尔初值设为 false，即使 GUI
XML 默认是 true。结果缺少请求的 FA／张量数组也判 QC 失败，不能只看到流线就算完成。

实测 DWIConvert 只输出 short，直接处理预处理后的浮点 DWI 会失败；开启
`allowLossyConversion` 又可能截断信号。因此采用**作者转换器决定梯度／空间规则，
原始信号无损写入**：先转换微型整数编码探针，验证其仅做轴置换／翻转、体积次序
不变，再沿相同映射写入原始 DWI 和掩膜。不重新采样、不做 short 量化、不自行猜测
FSL 梯度到世界坐标的规则。无法验证无损映射则报错。

VTK 为主结果，保留原始张量数组；另导出供 WMQL 使用的几何 TRK，逐条、逐点核对
与 VTK 坐标差不超过 0.0001 mm。保存输入／二进制／转换器 SHA256、实际命令、
CLI XML、耗时和原生掩膜几何 QC；严格匹配才复用缓存。UKF 失败时不把旧流线
继续送入下游，也不会把旧 CSD 的 SIFT2 权重用于 UKF。
目前 UKF 连接组使用 Python 后端。

**几何 QC 通过不等于解剖正确或 WMA 每束采样充足。** WMA、WMQL、TractCloud
保留独立门禁及人工审核要求。Windows／macOS 还需匹配的官方扩展构建，不能用
Linux 验证冒充三平台完整验收。

---

## 命令行

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject --subject-id subject \
  --targets tractography,bundles --tractography-backend ukf \
  --bundle-backend wma --execute
```

WMA 的图谱与环境参数仍按原方式传入。GUI 自动发现 UKF 并向后台传递路径；
独立命令行可指定 `--ukf-executable` 或 `SLICERDSPACE_UKF_EXECUTABLE`。
DWIConvert 自动从 Slicer 发现，完全独立的环境也可指定
`--dwiconvert-executable` 或 `SLICERDSPACE_DWICONVERT_EXECUTABLE`。调整参数使用
`--ukf-parameters '{"seedsPerVoxel":2,"numThreads":8}'`，修改与默认值一起记录，
无需训练或复制 UKF 源码。

如需明确使用本次 CSD/ACT 结果进行 WMA 分类，可指定
`--tractography-backend csd_act --bundle-tractography-backend csd_act --bundle-backend wma`。
大批次运行前先用 `--plan` 检查并解决受阻任务；`--doctor` 检查依赖，
并不保证所有数据路由均已连接。

---

## 验证范围

[2026-10-21 工作流复核](../release/BUNDLE_WORKFLOW_20261021.zh-CN.md)补充
转换器后台传递、TractCloud QC／采样警告、DTI 自动 VTK 导出以及真实束文件复核。
旧失败、当前自动检查与人工解剖审核分别记录，不把历史 UKF 结果视为新版默认协议的完整验收。

[2026-10-20 证据摘要](../release/benchmarks/ukf_integration_20261020.json)记录
1,143 项测试通过、5 项跳过，其中包含 23 项 UKF 专项测试。
隔离 Slicer 5.10 的 UKF 控件与紧凑布局测试均通过；实际作者源码构建的 UKF
二进制通过显式路径调用，在浮点合成张量场上运行，其余采用默认参数、线程限定为 2，
得到 1,728 条流线、100% 掩膜重合，FA／张量数组、全部 VTK/TRK 坐标及原生
SlicerDMRI 加载均通过。**这不是人体解剖／WMA 逐束验收，也不是新安装市场 UKF
包或跨平台验收。** 初始失败及修复后的记录分别保留，不覆盖旧失败。
