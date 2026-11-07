<div align="center">

# WMQL 多束自动提取

[项目](../../README.zh-CN.md) | [入门](QUICKSTART.md) | [English](WMQL.md)

</div>

## 默认协议

默认 `official` 直接使用作者仓库未修改的 `freesurfer_queries.qry` 和
`FreeSurfer.qry`，固定提交 `5c69c594d489859e08485110678dca7f6f4cba40`。
2026-10-24 核对上游 master，两个文件的字节、SHA256 与随包副本完全一致。
规则、标签定义和原许可证随扩展安装；执行器仍使用作者 Tract Querier，
无需用户编写查询，不加入 DDParcel 专用 AF 改写，也不自动换规则或放宽阈值。

该文件声明 **48 个输出查询**，覆盖双侧联络束、投射束与连合束。
不保证每人都产生 48 个非空束，不要求全部输入流线强制归类，且查询允许重叠。
作者 CLI 对空结果不写束文件；中间 `|=` 区域定义不当作束导出。

---

## 高级设置

**Advanced settings > Model parameters > WMQL 束子集**：

| 输入 | 行为 |
| --- | --- |
| 留空或 `all` | 执行全部官方输出查询 |
| `AF,CST,UF` | 三种束各取左右两侧 |
| `af.right,ilf.left,cc_4` | 指定精确查询名 |

不区分大小写，错误名称会阻止执行。单被试、批处理及 GUI 后台进程传递同一选择；
选择或规则依赖改变后不会复用旧提取缓存。

旧 `author`、`ddparcel_act`、`published_endpoints` 仅保留显式 CLI 历史复现和审计能力，
不作为 GUI 选项或推荐预设。已保存的 GUI 默认值迁移为 `official`；载入的旧场景
将原值记在 `SlicerDSpace.LegacyWMQLPreset`，既有运行记录不会重写。

---

## 原生 Python 执行器

要选择原生执行器，在 **环境与安装 > 核心 Python 与运行时路径** 设置
**TractQuerier 根目录**和 **WMQL Python**。命令行对应
`--tract-querier-root <源码目录>` 和 `--wmql-python-executable <Python>`。
源码目录须包含 `scripts/cli_tract_querier.py`。执行器需要 NumPy、NiBabel 和 VTK。
依赖检查和 WMQL 执行都使用所选解释器。更换执行器后，如需用同一数据
验证新执行器，请使用新的运行名或 `--no-resume`。

留空 **WMQL Python** 时，已有配置沿用默认路径。Windows 为其他工具修复
WSL 运行时时，所选环境准备流程会保留已配置的原生 WMQL 和 TractCloud 路径。
手动运行 Windows 安装脚本时，可用 `-PreserveWmqlSettings` 或
`-PreserveTractCloudSettings` 保留对应的已保存路径。

---

## 兼容性与质量控制

执行前检查标签是否为三维、有限、非负整数，是否具备 FreeSurfer/DK 区域编号及有效仿射矩阵。
只有 WM/GM/CSF 的组织图不能替代区域分区。记录皮层 ID、区域白质 ID 和缺失核心 ID，
但不制造白质分区、不膨胀标签，也不修改作者谓词。

**DDParcel 文件叫 wmparc，不等于它与 FreeSurfer wmparc 完全等价。**
当前数据可能只有 DK 皮层/aseg 标签，缺少 3000/4000 系列区域白质，
在作者穿行规则和 ACT 边界截断流线上可能得到空束或稀疏束。编号兼容不代表空间、解剖已验证。

每条查询分别标记：未提取、空、无效、采样不足（少于 20 条）或待解剖审核。
20 条是插件本地采样合理性门槛，不是作者默认值，也不代表统计效能。
文件损坏或执行失败会判失败；局部覆盖不足可以完成提取，但显示 QC 警告，
`automatic_qc_passed=false`，不能宣称全部通过。未完成人工解剖审核的束不作为已验收定量结果。

---

## 实测记录

2026-10-24，HCP 100307 已有 UKF 流线及其真实 DDParcel 分区：

| 项目 | 实际结果 |
| --- | --- |
| 官方查询 | 48 |
| 非空输出 | 41 |
| 达到 20 条采样门槛 | 31 |
| 未提取/空 | 7 |
| 非空但采样不足 | 10 |
| AF 左/右 | 1,184 / 1,216 |
| CST 左/右 | 168 / 252 |
| 封装执行与几何检查 | 36.69 秒 |
| 直接调用作者执行器 | 19.81 秒 |

与直接使用作者原始规则的执行结果比较，**48 条查询的流线多重集合全部一致**，
包括双方均为空的查询；全部导出流线可精确回溯到输入，输入哈希未变化。
两个耗时范围不同，不能解释为算法加速或减速率。

同一被试 CSD/ACT 输入：34 个非空输出，其中 28 个达到采样门槛、6 个稀疏、14 个缺失；
含封装检查耗时 54.41 秒。两组均不是“全束 QC 通过”，人工解剖审核仍待完成；
这是同一被试两种追踪输入，不是多被试泛化验收。

指定 `AF,CST,UF` 的 UKF 子集实际产生全部六个输出，计数与全套运行一致，
含封装检查耗时 18.71 秒；即使该子集采样足够，分区兼容性警告仍保留。
独立审计确认几何和源流线精确对应，但全套请求的采样检查仍未通过。
在独立设置、独立 X 显示的 Slicer 5.10 中，安装副本已通过 Advanced 默认项、
参数双向绑定、后端切换显隐及旧场景迁移测试，无 traceback；未改动当前用户 Slicer 会话。

本地原始证据保存在 `ignore/validation/wmql_official_20261024/`；
仓库汇总：[验证 JSON](../release/benchmarks/wmql_official_20261024.json)。
可用 `scripts/validate_wmql_official.py` 复现封装与直接作者调用对照，不改动既有科学输入。

参考：[作者教程](https://tract-querier.readthedocs.io/en/latest/example_script.html)、
[固定版本规则](https://github.com/demianw/tract_querier/blob/5c69c594d489859e08485110678dca7f6f4cba40/tract_querier/data/freesurfer_queries.qry)。
