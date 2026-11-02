<div align="center">

# 脑区分析

**按需复用已完成且 QC 可用的结果，不改变默认工作流**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [结果查看](AUTOMATIC_RESULTS.md) | [English](REGIONAL_ANALYSIS.md)

</div>

---

## 操作入口

展开 **高级设置 > 脑区与输入导入 > 脑区分析（可选）**。默认折叠，不会自动增加主工作流的任务。

1. 展开面板后自动读取最近一次运行；也可点击“读取脑区与可用结果”刷新。
2. 搜索名称、左右半球或标签号并勾选脑区。支持“左海马”“右中央前回”等常用中文别名；纯数字按完整标签号匹配。搜索后按回车可勾选当前可见结果。
3. 勾选需要的分析，点击“分析所选脑区”。界面只显示本次已有可用结果的分析类型。
4. 关键结果自动加载到场景；点击“打开脑区分析报告”查看汇总。

**更多选项与输入文件**中可修改运行记录、分区图/LUT、指标子集、流线来源、筛选规则、
目标/途经/排除脑区和输出目录。只读取来源记录明确声明、已完成或可复用、QC 可用且文件签名未过期的产品。

默认采用 DDParcel 分区与 FreeSurfer 脑区名称。也可指定自定义整数 NIfTI 分区图
及 FreeSurfer 格式的 LUT。参数图、脑掩膜和分区图必须具有一致的形状与空间仿射；
不会静默配准或重采样。连接组必须对应原运行记录的同一分区图，不能仅凭标签号相同混用。

---

## 支持的分析

| 分析 | 复用输入 | 输出 |
| --- | --- | --- |
| 微结构统计 | 已有 DTI、FW、NODDI、DKI/MK-Curve 参数图 | 均值、标准差、中位数、四分位数/IQR、范围、有效/排除体素数及覆盖率 |
| 组织构成 | 已有 DDSeg 硬标签分割 | WM/GM/CSF 和未分类体素的数量、体积及占整个选区的比例 |
| 皮层形态 | 已完成后处理的 aparc 分区、厚度、面积、曲率 | 顶点数、面积、平均/中位厚度、厚度标准差、平均曲率 |
| 选区相关流线 | 已完成的完整流线文件，以及明确记录的配套 SIFT2 权重 | 完整筛选流线、原始索引、对应权重和独立显示预览 |
| 脑区连接分析 | 已有对称连接矩阵与节点表 | 所选脑区的诱导子矩阵，以及各选区到原矩阵其他节点的非零连接 |

可选指标由原运行已经生成的结果决定；这里**不重新拟合模型**。DTI 包括
FA/MD/AD/RD 及已有特征值、Trace。FW 组织参数、NDI/ODI 必须有明确的组织有效性
掩膜；DKI 采用原始参数图与有效性掩膜，不使用为显示截断后的值；MK-Curve
优先使用各参数自己的有效性掩膜。有限的真实零值仍参与统计；无效拟合不进入
分母。脑区没有有效值时留空/null，不填零。

流线支持五种明确的几何规则：

- **经过任一所选脑区**：各脑区相交流线的并集。
- **经过全部所选脑区**：各脑区匹配的交集；单次最多 16 个脑区。
- **至少一个端点在选区内**：至少一个端点属于所选脑区的并集。
- **两个端点都在选区内**：两端都属于并集，不要求端点位于不同脑区。
- **A ↔ B 连接**：主列表所选脑区为 A，在更多选项中输入 B；一个端点在 A，另一个在 B，不区分流线存储方向，A/B 必须非空且互不重叠。

更多选项中还可设置必须途经的脑区（逐区同时满足，最多 16 个）和排除脑区（进入任一个即排除）。
名称/标签号用逗号分隔；左右半球不明确的名称会列出候选编号，由用户确认。A/B 的统计和连接分析
覆盖两组脑区的并集。可用来源包括 DTI、CSD/ACT、UKF 和 SAF 的完整流线，取决于原运行实际完成的结果。

皮层统计通过 `ctx-lh-*` / `ctx-rh-*` 的完整结构名与对应半球的 `aparc.annot` 精确匹配，
读取实际表面顶点数组。未匹配区域记录为不可用。厚度/曲率采用区域顶点均值，面积为顶点面积之和。
同时选中一对左右同名、同组织类型脑区时，自动导出左右差值和不对称指数
`2(L-R)/(|L|+|R|)`；两侧都为零或任一侧无有效值时指数留空。

“经过”规则采用 DIPY 的 [line-based ROI targeting](https://docs.dipy.org/stable/reference/dipy.tracking.html#dipy.tracking.utils.target_line_based)，
可处理两个已保存顶点之间线段穿过脑区的情况。输出是**几何子集，不自动等于
WMQL/WMA/TractCloud 解剖束识别**；流线数量也不是轴突数量。端点规则采用明确的
体素归属，不等于原连接组可能使用的径向端点分配。

连接组保持原权重，不额外归一化、阈值化、平滑或重算。按节点表中的实际分区
标签匹配，而不是按连续行号匹配。不存在的边沿用原矩阵的零边约定；有向或不一致
的矩阵会拒绝，不会悄悄对称化。

---

## 输出与审核

分析在可取消的后台进程运行，关键结果自动进入当前 Slicer 场景，按被试、区域运行、
产品分组。重复加载复用已有结果节点，不替换主工作流的“最近一次运行”引用。

默认写入 `05_statistics/selected_regions/<原运行>/<时间戳-ID>/`，每次使用独立目录：

| 文件 | 内容 |
| --- | --- |
| `request.json` | 脑区、指标、规则和来源配置 |
| `region_analysis_manifest.json` / `region_analysis_qc.json` | 输入/输出哈希、任务状态、耗时及验证范围 |
| `selected_regions.nii.gz` / `.csv` | 保留原标签的选区图及名称、体素数、物理体积 |
| `selected_microstructure.csv` / `.json` | 所选脑区的有效参数统计 |
| `selected_tissue_statistics.csv` | 选区内 DDSeg 组织构成 |
| `selected_surface_statistics.csv` | 匹配皮层脑区的实际表面统计 |
| `bilateral_statistics.csv` | 可配对脑区的左右对照 |
| `regional_report.html` | 可离线打开的区域联合报告 |
| `selected_streamlines.vtk` | 非空时生成的完整匹配几何 |
| `selected_streamlines_preview.vtk` | 仅用于显示的有界预览 |
| `selected_streamline_ids.csv` | 原流线索引；存在配套 SIFT2 权重时保留对应权重 |
| `selected_tractography.json` | 规则、数量、坐标系及预览策略 |
| `selected_connectome_matrix.csv` | 所选脑区的原权重子矩阵 |
| `selected_region_connections.csv` | 所选脑区到其他节点的非零连接 |
| `selected_connectome.json` | 原连接指标与导出策略 |

VTK 保存 RAS-mm 几何，不复制原文件的张量或逐点数组；可用索引回溯原文件。
完整几何以双精度写出，不裁剪、不重采样。可复现预览最多 30,000 条流线、
1,000,000 个点，**不能用于定量测量**；场景中另行记录完整定量来源。

显示时将 FreeSurfer LUT 的前景色适配为不透明，不修改原 LUT 或标签。
未提供 LUT 时使用确定性的显示配色。此处 Slicer 稠密颜色表最多支持标签 ID
1,000,000；更大的定量 ID 仍可导出，但不会为了显示而静默重编号。

没有匹配流线时如实记录 0，不补线。开始与结束均核对来源/QC 文件哈希，输入改变
即不验收；不覆盖已有结果。取消或未完成任务不能成为完成状态。
**这里的 QC 仅验证后处理一致性，不代表重新验证上游拟合，也不替代人工解剖审核。**

---

## CLI 与验证

常规分析直接传参，无需手写请求文件；GUI 与 CLI 共用同一后端：

```bash
python scripts/analyze_regions.py --source-manifest /absolute/run_orchestration_manifest.json \
  --regions 1024 2024 --metrics fa md --output-dir /absolute/new_region_results
python scripts/validate_region_analysis.py /absolute/region_analysis_manifest.json \
  --report /absolute/independent_check.json
```

来源运行记录是工作流已经生成的文件，不是额外编写的配置。
`--regions` 接受标签号或名称，多词名称加引号；支持逗号、空格分隔和重复参数。
未选择任何分析类型时，默认汇总 FA。只选择组织、皮层、流线或连接组时，不隐式添加标量指标。
所选指标必须存在于该运行且满足来源 QC 检查。

| 参数 | 作用 |
| --- | --- |
| `--source-manifest RUN [RUN ...]` | 单个或多个运行记录；可重复传入，多个来源时导出队列表 |
| `--regions NAME_OR_ID [...]` | 待分析脑区的名称或标签号 |
| `--metrics fa md [...]` | 可用指标键名，例如 `fwf`、`noddi_ndi`、`noddi_odi`、`dki_mk` |
| `--tissue`、`--surface` | 添加组织构成或匹配的皮层形态统计 |
| `--tractogram TASK` | 流线来源任务，例如 `tractography.ukf` |
| `--tract-rule RULE` | `through_any`（默认）、`through_all`、`endpoint_any`、`endpoint_both`、`endpoint_pair` |
| `--target-regions [...]` | A-B 端点筛选的目标组 |
| `--include-regions [...]`、`--exclude-regions [...]` | 必经区和排除区 |
| `--connectome TASK` | 连接组来源任务，例如 `connectome.streamline_count` |
| `--atlas FILE`、`--lut FILE` | 可选替代图谱和标签名称表 |
| `--output-dir DIR` | 新输出目录；多来源时为队列结果根目录 |

例如，筛选连接双侧中央前回的 UKF 流线：

```bash
python scripts/analyze_regions.py --source-manifest /absolute/run_orchestration_manifest.json \
  --regions 1024 --target-regions 2024 --tractogram tractography.ukf \
  --tract-rule endpoint_pair --output-dir /absolute/new_region_tracks
```

需要查询可用脑区、指标或任务键名时，使用 `--inspect /absolute/run_orchestration_manifest.json`
直接打印清单；可选 `--catalog /absolute/catalog.json` 保存。
名称从当前图谱/LUT 解析；歧义名称需指定标签号。

同一组参数可直接应用于多个已完成的被试：

```bash
python scripts/analyze_regions.py --source-manifest /absolute/subject1/run.json /absolute/subject2/run.json \
  --regions 1024 2024 --metrics fa md --output-dir /absolute/new_region_cohort
```

高级保存配置仍可使用 `--request /absolute/request.json` 和 `--batch /absolute/region_batch.json`；
不要与直接分析参数混用。GUI 自动生成所需请求，无需用户手写。
[批处理模板](examples/region_batch.json)包含共享 `request`、每被试的 `source_manifest`（可覆盖 atlas/LUT）
和新输出目录。逐被试执行并导出 `cohort_*.csv` 长表，保留被试 ID、来源和分区哈希；
所选标签号/名称语义必须一致。个别被试失败时保留失败原因并继续其他被试，CLI 以非零状态退出。
不自动增加组间检验或正常参考范围。

真实 Slicer 验证覆盖后台执行、中英文、440 像素面板、1000 以上标签、自定义 LUT、
自动加载和重复加载复用。独立核对将统计值与布尔掩膜直接计算比较，逐条比较导出
流线及原索引对应几何，并与原矩阵逐项核对连接权重；这些测试不等于临床有效性验证。

[10 月 20 日验证记录](../release/benchmarks/regional_analysis_20261020.json)包含真实
HCP-YA 百万流线案例，单独标明后处理耗时，不将其当作上游算法加速对照实验。

[10 月 29 日联动验证](../release/benchmarks/regional_analysis_20261029.json)覆盖简化面板、组织/皮层
接入、A↔B 流线几何/权重一致性，以及三名真实被试的区域统计独立核对。
