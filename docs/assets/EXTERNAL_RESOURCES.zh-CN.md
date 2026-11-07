<div align="center">

# 非代码运行资源归档

[项目](../../README.zh-CN.md) | [自动配置](../setup/MANAGED_SETUP.zh-CN.md) | [授权状态](MODEL_ASSET_TERMS.md) | [English](EXTERNAL_RESOURCES.md)

</div>

---

## 范围

[SlicerDSpace-models](https://github.com/ChengjinLii/SlicerDSpace-models)
统一管理权重、模板、图谱及必要推理参考文件。已有本地资源目录继续可用，无需更名或重新下载。
不是把所有“大文件”都迁进去：用户影像、分析结果、训练/测试数据、Conda/Python 环境、
CUDA 库、外部程序和逐被试生成的拟合字典均排除。
默认 core 仍是原有 31 个解剖资源，可选后端不随默认安装全量下载。

| 分组 | 统一目录 | 处理 |
| --- | --- | --- |
| DDSurfer / DDSeg / DDParcel | 原有 `surface/`、`segmentation/`、`parcellation/` | 保持原有索引，不改变核心权重 |
| PNL CNN | `brain_mask/pnl_cnn/` | 三方向 HDF5、结构 JSON、完整原许可；约 216 MiB |
| IIT b0 模板 | 同目录，约 7.5 MiB | 仅本地保留，Git 忽略；未经书面许可不得电子镜像 |
| TractCloud | `bundles/tractcloud/` | 权重、推理参数、308 字节参考、原许可；约 8.5 MiB，不复制训练/测试包 |
| WMA / ORG | `bundles/wma/archives/` 与本地解压目录 | 两个官方 ZIP 共约 1.68 GiB；按需获取，解压副本不重复提交 Git |
| WMQL / SAF / MK-Curve | 插件运行资源 | 查询、方向表等小文件随代码包；无需另传大型权重 |
| DTI / FW / NODDI / MRtrix / UKF | 算法或 runtime | 不因算法名含 model 就假定存在预训练权重；缓存和软件环境分别管理 |

---

## 来源与授权

PNL 的权重来自作者 v0.3，TractCloud 来自 v1.0.0 的最小推理子集，均保留原完整协议与引用。
TractCloud 权重及参考字节与已保存的官方发布包一致。

ORG 压缩包与 [Zenodo 固定记录](https://zenodo.org/records/2648292) 的文件大小和 MD5 一致，
另记录本地 SHA256；官方记录声明 CC BY 4.0，并保留 ORG 仓库的完整 Slicer 协议及方法引用。

[IIT 官方条款](https://www.nitrc.org/projects/iit/) 明确要求电子再发布须取得书面许可。
本地副本可用于获授权的研究，但不能因“下载成功”或“放在私有仓库”就认定可再上传。
导出工具会拒绝完整 `pnl-cnn` 组合，镜像只选 `pnl-cnn-weights`；受限模板下载仅指向上游。
以上是工程分发记录，不是法律认证或临床有效性证明。

---

## 工程接口

可选资源的可信清单是 `models/external_assets_manifest.json`，包含大小、SHA256、
官方来源、许可及分发策略；不与 core 的历史 `asset_mirror_manifest.json` 混写。
本次准备了独立的 `optional_assets_mirror_manifest.json` 和 TSV，包含 14 个可镜像文件，
包括固定 SHA256 的 ORG 解压文件索引。
资源提交 `11167a5` 已推送，包含 8 个 LFS 二进制对象，下载 URL 固定到该提交。
核对 GitHub 时，代码仓库仍为私有，独立资源仓库已是公开状态；本轮没有更改任何仓库的可见性。

```bash
PythonSlicer scripts/download_model_assets.py \
  --manifest models/external_assets_manifest.json --profile mirrorable-inference \
  --asset-root /path/to/SlicerDSpace-models --verify-only
```

`scripts/import_external_resources.py --help` 提供维护者/离线导入接口，默认仅输出计划，
`--execute` 才复制。只处理明确指定来源，核对固定校验值，不删除原文件、不覆盖不同内容。
ORG 解压防止越界与链接攻击，排除 `__MACOSX` 元数据；849 个运行文件逐一核对官方 ZIP 内的字节，
并生成 `bundles/wma/extracted_resource_index.json`。

PNL、TractCloud、WMA 在未指定专用目录时查找统一资源目录，先校验再使用；
显式路径和旧环境变量仍优先。WMA 按固定索引核对全部解压文件，不能仅凭 ZIP 或空目录判定就绪。

---

## 按需准备

Slicer 中选择算法后，使用 **环境与安装 > 准备所选分析环境**。
只准备当前路线缺少的资源，确认窗口显示完整分组大小及条款；下载、SHA256 校验、ORG 解压均在可取消
的后台进程中执行。成功后保存路径，再做一次新的环境检查；失败或取消的资源阶段不替换原有配置。
已有且校验正确的文件复用；不同内容的既有文件不静默覆盖，应先检查或另选干净资源目录。

PNL 环境安装会复用共享目录中校验通过的模型和模板；缺少时从作者模型压缩包和 IIT 官方来源获取，
校验后归档到同一共享目录。Python、ANTs、源码保留在隔离环境，旧的 prefix 内置模型仍兼容。
不改 GPU 选择、算法参数或科学 QC 门槛。

命令行默认仅规划，明确执行才下载：

```bash
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma \
  --asset-root /path/to/resources --execute --accept-terms
PythonSlicer scripts/setup_optional_assets.py --profiles tractcloud wma \
  --asset-root /path/to/resources --from-directory /path/to/offline-resources --execute
```

离线可通过以上命令只导入选定资源并解压图谱；GUI 的“使用离线模型资源”校验所选 core 及缺失的可选分组，
不联网。下载权重不会安装 Python 库或外部程序，WMA/TractCloud 的源码/运行环境缺项仍保留为缺项。

公开资源镜像支持匿名下载，GUI 的 URL 下载不要求用户安装 Git 或 Git LFS。
已在干净目录中匿名下载 TractCloud 四个推理资源，文件大小及 SHA256 全部通过。
ORG、IIT 以及 PNL 运行环境安装仍保留官方来源。
如果将来使用私有镜像，须标记 `access: authorized-private`，准备流程将提前说明权限要求，
不会先下载一部分再因权限失败；可改用获授权的完整 Git LFS 副本或离线导入。

本地迁移证据：`ignore/validation/external_resources_20261021/import.json` 和各 profile 的 lockfile。
迁移验证的是字节一致和安装资源完整性，不代表重新验收所有人体算法结果的 QC。

本轮复核：可镜像 14 文件通过 SHA256 检查，完整本地 PNL 6 文件通过检查，
原 core 31 文件未变；PNL 从新目录读取后三方向 GPU 前向通过。
当前 Slicer 的 PNL、WMA、TractCloud 资源路径已保存到统一目录，所选默认环境检查 0 项缺失。
Python 回归共 1,287 项，1,284 通过、3 跳过；独立 Slicer 环境检查与依赖准备 GUI 测试通过。
另一次真实 GUI 测试验证了可选资源后台 worker 和成功后的自动保存路径，测试容器禁用网络。
本地 HTTP 实际下载测试验证首次获取与再次复用不联网；公开镜像的真实匿名获取也通过。
安装包资源审计未发现缺失或不匹配；从 GitHub 全新检出 `11167a5` 后，14 个远端文件的
大小与 SHA256 全部通过，不是 LFS 指针空壳。
记录：`ignore/validation/external_resources_20261021/remote-verified.lock.json`。

公开 URL 仍需网络可达，不保证防火墙或隔离容器能直连。下载器遵循标准代理环境变量，保留可续传文件；
超时仍是失败，不会转成 ready。离线导入是不依赖联网的替代方式。
Slicer Python 的首次隔离桥接网络尝试超时，保留为失败；同一个 Slicer 5.10 worker 在可达的
host 网络与已有标准代理环境下匿名获取 TractCloud 四文件通过，没有禁用 TLS 校验，插件没有写死代理地址。
记录：`ignore/validation/slicer_anonymous_resources_20261021/`。
