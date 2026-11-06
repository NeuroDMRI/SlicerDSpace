<div align="center">

# WMA 图谱

[配置](../README.zh-CN.md) | [CLI 模板](CLI_TEMPLATES.zh-CN.md) | [资源仓库](https://github.com/ChengjinLii/SlicerDSpace-models) | [English](WMA_ATLASES.md)

</div>

---

## 选择图谱

| 参数值 | 图谱 | 人群与来源 |
| --- | --- | --- |
| `org`（默认） | ORG 纤维聚类图谱 | 成人参考图谱，应用覆盖多个年龄段；[论文](https://doi.org/10.1016/j.neuroimage.2018.06.027)、[数据](https://doi.org/10.5281/zenodo.2648292) |
| `east-west` | East-West 白质图谱 | 东西方成人联合图谱；[论文](https://doi.org/10.1038/s41597-024-03624-2)、[数据](https://doi.org/10.57760/sciencedb.17354) |
| `naba` | 新生儿/成人脑图谱 | 新生儿与成人联合图谱；[论文](https://doi.org/10.1016/j.fmre.2025.12.030)、[数据](https://doi.org/10.5281/zenodo.18227691)、[作者实现](https://github.com/azurezhangwei/NABA-Atlas) |

NABA 针对新生儿与成人，不宜笼统称为覆盖所有儿童的图谱。根据研究人群和追踪协议选择，
并检查配准及各束走行。选择图谱与选择流线来源是两个参数；UKF 仍是 WMA 的参考路线。

## 按需准备

图谱放在 `SlicerDSpace-models`，不打入扩展代码。只下载当前所选图谱，并校验分片及每个
解压文件的 SHA256。运行包保留原始数据字节、署名和离群流线剔除所需的全部 800 个参考簇。

新增运行包下载量约为 East-West 11.1 GB、NABA 3.3 GB。首次安装包含分片、临时合并包和
解压文件，分别预留约 39.4 GB、10.8 GB 空间；安装计划按所选图谱计算具体需求。

[公开镜像检查](../release/benchmarks/wma_public_mirror_20261030.json)覆盖全部资源地址和大小，
并对两套解压索引及各一个分片完成匿名下载和 SHA256 核验；完整解压另通过同字节离线副本验证。

```bash
python scripts/setup_optional_assets.py --profiles wma-east-west --asset-root /path/to/models
python scripts/setup_optional_assets.py --profiles wma-east-west --asset-root /path/to/models \
  --execute --accept-terms
```

NABA 使用 `wma-naba`，ORG 使用 `wma`。可用 `--from-directory /path/to/offline-mirror`
导入离线副本。GUI 在 **高级设置 > 模型参数 > WMA 图谱** 选择，然后到独立的
**环境与安装 > 准备所选分析环境** 下载。切换图谱不需要重装 WMA。

## 运行

```bash
"$SLICER" --launch PythonSlicer run_subject_pipeline.py --no-user-config \
  --subject-root /path/to/subject --subject-id SUBJECT \
  --output-root /path/to/results/east-west --run-name east_west \
  --model-assets-dir /path/to/models --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-tractography-backend ukf \
  --bundle-backend wma --wma-atlas east-west --slicer-executable "$SLICER" --plan
```

确认计划后将 `--plan` 改为 `--execute`。同一参数还支持 `org` 和 `naba`；
`--wma-atlas-dir` 可指定所选图谱的已有目录。不同图谱对照使用不同输出和运行名。

Linux 未设置 `DISPLAY` 或 `SLICERDSPACE_DISPLAY` 时，WMA 自动通过 `xvfb-run -a`
为依赖 Slicer 的回变换提供虚拟显示。预检会在分析前检查 `xvfb-run`、`Xvfb` 和
`xauth`。已有显示会直接复用；macOS 和原生 Windows 不会自动启用 Xvfb。
运行记录保存实际选择；无显示的 Linux 批处理无需另外填写 `--wma-use-virtual-x`，
该参数仍可用于显式请求。

## 实现与检查

配准、谱聚类、离群剔除、回到原空间与束合并仍调用 WMA。新增适配选择相应目录、
半球标记和束清单；非 ORG 图谱不再套用 WMA 对 ORG 五个 CPC 簇编号的特殊修正。
ORG 现有路线不变。运行记录保存图谱身份、来源、束清单及对应的输出要求。
切换图谱不会复用另一套图谱的 QC；覆盖及逐束采样检查保持启用，解剖复核另行记录。

NABA 发布的辅助脚本将 CPC 放在连合组，但其图谱中的 7 个 CPC 参考簇全部标为半球束
（`h`）。适配按图谱标记输出左、右 CPC，共 78 个解剖束，避免从已经分到左右半球的簇中
再合并出空的连合 CPC。原始簇数据不修改。

在源码目录中，可选的资源测试读取配准几何、检查坐标有效性，读取已校验的谱图谱，
并调用实际 WMA 束合并代码；检查 32 条参考流线和
全部已发布束定义，以及合成样例的点坐标、属性和流线顺序：

```bash
"$SLICER" --launch PythonSlicer Testing/Python/run_wma_atlas_resource_smoke.py \
  --family east-west --asset-root /path/to/models \
  --wma-root /path/to/whitematteranalysis --report /path/to/atlas-check.json
```

NABA 使用 `--family naba`。检查记录：[East-West](../release/benchmarks/wma_east-west_resources_20261030.json)、
[NABA](../release/benchmarks/wma_naba_resources_20261030.json)。

### WMQL 为什么没有小脑束

默认沿用的官方预设有 48 个输出查询，没有专门的小脑脚或小脑内部纤维束查询。
其中 `CB` 是扣带束（cingulum bundle），不是小脑。DDParcel 和输入流线包含小脑，
也不意味着这些默认规则会输出小脑束；需要这类结果时可选择相应的 WMA 图谱。
WMQL 现也提供可选的[小脑区域扩展](WMQL_CEREBELLAR.zh-CN.md)，与未修改的默认规则及
精细小脑脚图谱标签分别记录。

[本次核查](../release/benchmarks/wmql_coverage_20261030.json)确认四个小脑标签均存在，
输入流线也经过这些标签。保留的 WMQL 实例有 41 个非空输出，其中 31 个通过采样检查；
空束与采样不足仍保留记录。默认规则缺少小脑查询不代表输入覆盖失败；更新的 Results 素材
另行加入并注明了可选的区域查询结果。

---

图谱数据保留 CC BY 4.0 署名，WMA 使用其自身的软件许可；扩展未复制 NABA 算法源码。
