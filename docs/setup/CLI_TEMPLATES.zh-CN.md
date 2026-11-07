<div align="center">

# 命令行模板

**选择路线、检查计划，再执行分析**

[首页](../../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [批量实验](EXPERIMENTS.zh-CN.md) | [平台支持](SUPPORT_MATRIX.zh-CN.md) | [English](CLI_TEMPLATES.md)

</div>

---

## 从这里开始

| 任务 | 模板 |
| --- | --- |
| 导入 DICOM、NIfTI 或 NRRD | [输入准备](#输入准备) |
| 生成掩膜或校正 DWI | [掩膜和预处理](#掩膜和预处理) |
| DDSeg、DDParcel、DDSurfer 或皮层后处理 | [解剖重建](#解剖重建) |
| DTI、FW、NODDI、DKI 或 MK-Curve | [微结构](#微结构) |
| DTI、UKF、CSD/ACT 或 SAF | [纤维追踪](#纤维追踪) |
| WMQL、WMA、TractCloud 及来源选择 | [束分析](#束分析和显式来源) |
| 连接组或组合路线 | [连接组](#连接组和组合路线) |
| 多被试或参数对照 | [批量实验](#多被试和参数实验) |
| 脑区分析或报告 | [结果](#脑区分析和报告) |

以下使用 Linux/macOS 的 Bash，或已明确配置的 WSL 环境；原生 PowerShell 示例见后文。
先替换路径。`REPO` 可以是源码仓库，也可以是安装后包含 `shared/` 和 `scripts/`
的 `qt-scripted-modules` 目录。`SLICER` 指向真正的 Slicer 启动程序。

[Agent Skill 算法目录](../../.agents/skills/slicerdspace/references/shell_templates.md)
提供从下方单被试命令生成的独立 Bash 模板。默认只生成计划，显式选择 `execute`
才开始分析。Skill 文件位于已授权的源码副本中，不随插件运行时安装。

<!-- cli-setup: bash -->
```bash
export SLICER="/path/to/Slicer"
export REPO="/path/to/SlicerDSpace"
export SUBJECT="/path/to/prepared/sub-01"
export SUBJECT_ID="sub-01"
export OUTPUTS="/path/to/analysis/sub-01"
export ASSETS="/path/to/SlicerDSpace-models"
DEPENDENCY_ARGS=()
# If needed, export dependency paths from the GUI, then uncomment:
# DEPENDENCY_ARGS=(--dependency-config "/path/to/dependency_paths.json")

pipeline() {
  local run_name="$1"
  shift
  "$SLICER" --launch PythonSlicer "$REPO/shared/run_subject_pipeline_cli.py" \
    --no-user-config --strict-environment \
    --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" \
    --output-root "$OUTPUTS/$run_name" --run-name "$run_name" \
    --model-assets-dir "$ASSETS" "${DEPENDENCY_ARGS[@]}" "$@"
}
```

每条路线使用独立输出目录。保持**同一条命令、同一运行名和参数**：先将 `--plan`
换成 `--doctor` 检查依赖，再检查 `--plan`，确认后换成 `--execute`。
Doctor/plan 可以写入元数据，不执行科学模型。`--strict-environment` 在依赖缺失时停止，
避免只执行所选路线的一部分。

模板不读取已保存的 GUI 运行默认值。如需复用 GUI 配置的外部路径，在
**环境与安装 > 更多选项 > 导出依赖配置** 中导出，并设置 `DEPENDENCY_ARGS`；
也可用 `--help` 中列出的显式路径参数。按[托管环境配置](MANAGED_SETUP.zh-CN.md)
只准备选定依赖和模型。资源下载、许可确认与正式分析分开进行。

大多数模板使用已检查的原生掩膜（`provided`）；需要生成时明确改选 PNL CNN。
已有掩膜优先使用。掩膜及可复用缓存位于被试根目录，仅换输出目录不能隔离掩膜生成对照；
比较掩膜方法时使用独立输入副本。

---

## 输入准备

DICOM 目录应只包含目标扩散序列；NIfTI 需要配套 bval/bvec；NRRD/NHDR 应保留梯度元数据。
使用新的目标被试目录。先预览，确认后移除 `--dry-run` 导入；已有标准化输入受覆盖保护。

<!-- cli-template: import-dicom -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/diffusion_DICOM" --source-kind dicom \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

<!-- cli-template: import-nifti -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/dwi.nii.gz" --source-kind nifti \
  --bval "/path/to/dwi.bval" --bvec "/path/to/dwi.bvec" \
  --mask "/path/to/reviewed_mask.nii.gz" \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

<!-- cli-template: import-nrrd -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/dwi.nhdr" --source-kind nrrd --backend dwiconvert \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

标准输入包括 `input/data.nii.gz`、`input/bvals`、`input/bvecs`，以及可选的
`input/dwi.json`、`input/nodif_brain_mask.nii.gz`。检查体积数、梯度方向、空间几何和有限值。
NRRD uint16 只在已检查的 int16 数值范围内无损转换；溢出时导入器拒绝转换。

---

## 掩膜和预处理

脑掩膜只有两种方法：`pnl_cnn` 和 `provided`；缺少掩膜时才生成。PNL 按 `--pnl-mask-route auto|managed|native` 选择路线：Linux 上 `auto` 即 managed（带 ANTs 的 conda 运行环境），macOS 和 Windows 上即 Slicer Python 中的 native 路线（SimpleITK 和 PyTorch，“native（实验性）”）；
SynthStrip、MRtrix `dwi2mask` 和 DIPY 掩膜方法已移除（W5）；`--brain-mask-backend` 写已移除的名称会失败，并提示 `pnl_cnn` 和 `provided`。
一种方法失败时不会自动改用另一种。

<!-- cli-template: mask-pnl -->
```bash
pipeline mask_pnl --targets validation --brain-mask-backend pnl_cnn \
  --brain-mask-device auto --plan
```

预处理只有一条路线：DIPY MP-PCA 去噪、DIPY Gibbs 去除，然后通过 MRtrix3 `dwifslpreproc` 运行 FSL eddy 和 topup。
所有系统都需要 MRtrix3、FSL（eddy、topup）和 FSL 许可。没有 FSL 构建的平台（例如 Linux aarch64）不提供该目标，请导入已预处理的数据。
`--preprocessing-skip-gibbs` 是唯一可以跳过某一步的选项。旧的 `--preprocessing-backend`、`--preprocessing-dipy-denoise-method` 和 `--preprocessing-skip-motion` 已移除（W2a）；使用它们会失败，并提示这条唯一路线。
每次预处理都需要相位编码方向。命令先取 `--preprocessing-phase-encoding-direction`（AP、PA、LR、RL、IS、SI，或 `j-` 这类 NIfTI 轴代码），否则取 `input/dwi.json` 中的 `PhaseEncodingDirection`。两者都没有时，`--plan` 和 `--execute` 以退出码 2 停止，并提示 "Phase-encoding direction needed: choose AP, PA, LR, RL, IS or SI in Preprocessing (most axial clinical DWI: AP)"。
读出时间先取 `--preprocessing-readout-time`，否则取 sidecar 的 `TotalReadoutTime`；两者都没有时，dwifslpreproc 使用 0.1 s，记录中标为 `readout_assumed`。
下面的 `j-`、`0.05` 和反向相位编码 b0 仅为示例，必须按采集协议替换。

<!-- cli-template: preproc-fsl -->
```bash
pipeline preproc_fsl --targets validation,preprocessing \
  --brain-mask-backend provided \
  --preprocessing-phase-encoding-direction j- --preprocessing-readout-time 0.05 \
  --preprocessing-reverse-phase-b0 "/path/to/matched_reverse_b0.nii.gz" --plan
```

需要将校正后数据用于后续计算时，把下游 targets 放进**同一条命令**，例如
`validation,preprocessing,segmentation,parcellation,surface`。
独立的新运行不会仅凭早先存在的预处理结果就自动采用它；以计划/运行记录的实际输入路径为准。

---

## 解剖重建

需要匹配的模型资源和兼容 PyTorch。工作流自动处理共享原生 DTI 准备步骤。
按需要选择组织分割、脑区分区或皮层表面。

<!-- cli-template: anatomy-seg -->
```bash
pipeline ddseg --targets validation,segmentation --brain-mask-backend provided --plan
```

<!-- cli-template: anatomy-parcel -->
```bash
pipeline ddparcel --targets validation,parcellation --brain-mask-backend provided --plan
```

<!-- cli-template: anatomy-surface -->
```bash
pipeline ddsurfer --targets validation,surface --brain-mask-backend provided \
  --surface-device auto --surface-precision fp32 --plan
```

<!-- cli-template: anatomy-combined -->
```bash
pipeline anatomy --targets validation,segmentation,parcellation,surface \
  --brain-mask-backend provided --ddseg-device auto --surface-device auto --plan
```

表面重建加皮层后处理、厚度/曲率和图谱映射。配置许可组件运行包及个人激活文件；
该路线不强制安装完整版 FreeSurfer。

<!-- cli-template: anatomy-postprocess -->
```bash
pipeline anatomy_post --targets validation,segmentation,parcellation,surface,surface_postprocess \
  --brain-mask-backend provided --postprocess-mode whole \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

复用已完成的原生表面时，通过 `--postprocess-native-dir`，以及需要时的
`--postprocess-mri-dir`，传入运行记录中的准确路径。参见[表面组件](SURFACE_COMPONENTS.zh-CN.md)。
不要把模板空间表面当成原生输入。

---

## 微结构

DTI 为默认项。可选 FW/NODDI/DKI 有各自的采样与拟合检查；NODDI 和 DKI 需要适用的多壳层数据。
检查适用性记录、有效体素掩膜和排除的拟合，不要用显示用图像替代有效性判断。

<!-- cli-template: micro-dti -->
```bash
pipeline dti --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method wls --plan
```

显式选择普通最小二乘拟合进行对照：

<!-- cli-template: micro-dti-ols -->
```bash
pipeline dti_ols --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method ls --plan
```

<!-- cli-template: micro-fw -->
```bash
pipeline freewater --targets validation,microstructure \
  --microstructure-stages dti,freewater --brain-mask-backend provided --plan
```

iFW 用 `--microstructure-stages dti,ifw`（或 `--run-ifw`）选择，同时运行 Free Water 和 DDSeg。
用 `--ifw-echo-time-ms` 给出回波时间；不给时从 DWI sidecar 读取 `EchoTime`，否则停止。`all` 不含 iFW。

<!-- cli-template: micro-noddi -->
```bash
pipeline noddi --targets validation,microstructure \
  --microstructure-stages dti,noddi --brain-mask-backend provided --plan
```

<!-- cli-template: micro-dki -->
```bash
pipeline dki_wls --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method WLS --brain-mask-backend provided --plan
```

<!-- cli-template: micro-dki-ols -->
```bash
pipeline dki_ols --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method OLS --brain-mask-backend provided --plan
```

<!-- cli-template: micro-mkcurve -->
```bash
pipeline mkcurve --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE --brain-mask-backend provided --plan
```

`MKCURVE` 是 Python 移植版，无需 MATLAB；不给出 `--dki-fit-method` 时它也是默认的 DKI 方法。`MKCURVE_MATLAB` 是另行配置路径的作者参考路线。

<!-- cli-template: micro-mkcurve-matlab -->
```bash
pipeline mkcurve_matlab --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE_MATLAB --brain-mask-backend provided \
  --dki-matlab-executable "/path/to/matlab" --dki-mkcurve-root "/path/to/MK-Curve" --plan
```

<!-- cli-template: micro-parallel -->
```bash
pipeline micro_multi --targets validation,microstructure \
  --microstructure-stages dti,freewater,noddi,dki --dki-fit-method WLS \
  --max-parallel-models 2 --brain-mask-backend provided --plan
```

最后一条明确请求四种模型。先完成共同前提，再并发适用且独立的可选拟合；这不是默认行为。

---

## 纤维追踪

DTI 确定性追踪使用原生 Teem 张量拟合（Slicer 内核）及 DIPY。
`--select-streamlines` 不等同于 UKF 的追踪预算。

<!-- cli-template: tract-dti -->
```bash
pipeline tracks_dti --targets validation,tractography --tractography-backend dti_det \
  --brain-mask-backend provided --select-streamlines 1000000 \
  --dti-min-length-mm 10 --dti-max-length-mm 250 --plan
```

UKF 自动调用已安装的 UKFTractography 扩展/CLI。
采用作者默认参数，下面仅调整线程预算：

<!-- cli-template: tract-ukf -->
```bash
pipeline tracks_ukf --targets validation,tractography --tractography-backend ukf \
  --brain-mask-backend provided --ukf-parameters '{"numThreads":8}' --plan
```

CSD/ACT 需要 MRtrix、DDSeg 组织概率图及 DDParcel 解剖标签，并计算与流线匹配的 SIFT2 权重。
选择下方两种解剖产物，或提供已有输出；标准路线在缺少 DDParcel 时阻止运行。

<!-- cli-template: tract-csd -->
```bash
pipeline tracks_csd --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act --brain-mask-backend provided \
  --select-streamlines 1000000 --tckgen-seeding dynamic --mrtrix-nthreads 8 --plan
```

`--allow-act-ddseg-only` 可显式启用缺少 DDParcel 的探索性回退。ACT QC 会记录缺少皮层下灰质、
脑室及指定白质结构先验；这些输出不会被标准路线复用。两条路线的第五个 5TT 通道均设为零，
不估计病灶专属组织先验。

SAF 使用 CSD 输入和通过检查的原生表面/后处理：

<!-- cli-template: tract-saf -->
```bash
pipeline tracks_saf --targets validation,segmentation,parcellation,surface,surface_postprocess,tractography \
  --tractography-backend saf --brain-mask-backend provided \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

<!-- cli-template: tract-multiple -->
```bash
pipeline tracks_compare --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act,ukf --brain-mask-backend provided --plan
```

多追踪后端依次执行，各有 `<run>__<backend>` 记录，不合并流线。
MRtrix 可用 `--mrtrix-seed 42` 进行确定性对照，这会强制 `tckgen -nthreads 0`，可能变慢；
SIFT2 仍使用配置的线程数。种子不意味着其他后端也完全确定。

---

## 束分析和显式来源

以下示例自动生成 UKF 全脑流线。全脑输入、束分析方法和图谱/资源分别选择。

<!-- cli-template: bundle-wmql -->
```bash
pipeline bundles_wmql --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-preset official --brain-mask-backend provided --plan
```

官方 WMQL 规则默认执行全部输出查询。需要小脑区域查询时，将
`--wmql-query-preset official` 改为 `--wmql-query-preset official_cerebellar`
（原有 48 条加 3 条区域查询），或 `cerebellar`（仅区域查询）。新增结果为左、右小脑—脑干及
双侧小脑相关候选流线，不直接赋予小脑上、中、下脚标签。
参见[规则与检查](WMQL_CEREBELLAR.zh-CN.md)。选择束子集示例：

<!-- cli-template: bundle-wmql-subset -->
```bash
pipeline bundles_wmql_subset --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-selection AF,CST,UF,cc_4 --brain-mask-backend provided --plan
```

<!-- cli-template: bundle-wma -->
```bash
pipeline bundles_wma --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wma --bundle-tractography-backend ukf \
  --wma-atlas org \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

WMA 默认使用 ORG。改用 `--wma-atlas east-west` 或 `--wma-atlas naba` 可选择东西方成人图谱或新生儿/成人图谱。直接调用 CLI 时传入 `--wma-atlas org|east-west|naba`；`--wma-atlas-dir` 可指定已有图谱。仅在确认的资源目录下准备所选 profile（`wma`、`wma-east-west` 或 `wma-naba`），不用下载全部三套。参见 [WMA 图谱指南](WMA_ATLASES.zh-CN.md)。

<!-- cli-template: bundle-tractcloud -->
```bash
pipeline bundles_tractcloud --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend tractcloud --bundle-tractography-backend ukf \
  --tractcloud-device auto --tractcloud-batch-size 2048 --brain-mask-backend provided --plan
```

<!-- cli-template: bundle-multiple -->
```bash
pipeline bundles_compare --targets validation,parcellation,tractography,bundles \
  --tractography-backend csd_act,ukf --bundle-backend wmql,wma,tractcloud \
  --bundle-tractography-backend ukf --slicer-executable "$SLICER" \
  --brain-mask-backend provided --plan
```

最后一条的全部束方法都使用 UKF；独立 CSD 路线不接收束分析任务。
比较其他来源时改 `--bundle-tractography-backend`，并使用新运行名。
非 UKF 来源的 WMA 覆盖率需单独检查。稀疏/空束保留在 QC 中；SAF 不作为全脑束分析来源。

复用已有兼容全脑 VTK/VTP 输入：

<!-- cli-template: bundle-existing -->
```bash
pipeline wma_existing --targets validation,bundles --bundle-backend wma \
  --wma-input-tractography "/path/to/reviewed_whole_brain.vtp" \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

显式 WMA/TractCloud 输入文件不可与 `--bundle-tractography-backend` 同时指定。
使用完整定量流线，不能用经过数量裁剪的场景预览替代；检查坐标与协议兼容性。

---

## 连接组和组合路线

DTI 流线加 Python 计数/标量加权连接组：

<!-- cli-template: connectome-python -->
```bash
pipeline connectome_python --targets validation,microstructure,parcellation,tractography,connectome \
  --tractography-backend dti_det --connectome-backend python \
  --brain-mask-backend provided --plan
```

CSD/ACT 加匹配的 SIFT2 权重和 MRtrix 连接组：

<!-- cli-template: connectome-sift2 -->
```bash
pipeline connectome_sift2 --targets validation,segmentation,parcellation,tractography,connectome \
  --tractography-backend csd_act --connectome-backend mrtrix \
  --brain-mask-backend provided --select-streamlines 1000000 --plan
```

CSD/ACT 路线使用自动连接组后端（`.tck` 计数用 MRtrix，其他来源或指标用 Python）：

<!-- cli-template: connectome-auto -->
```bash
pipeline connectome_auto --targets validation,segmentation,parcellation,tractography,connectome \
  --tractography-backend csd_act --connectome-backend auto \
  --brain-mask-backend provided --select-streamlines 1000000 --plan
```

解剖、微结构、UKF、WMQL 与 Python 连接组组合路线：

<!-- cli-template: combined -->
```bash
pipeline combined --targets validation,microstructure,segmentation,parcellation,surface,tractography,bundles,connectome \
  --microstructure-stages dti --tractography-backend ukf --bundle-backend wmql \
  --bundle-tractography-backend ukf --connectome-backend python \
  --brain-mask-backend provided --plan
```

需要预处理时，在同一条命令 targets 中加 `preprocessing`，并配置后端/采集参数。
后处理和可选微结构模型仍需明确选择，本组合不会隐含运行它们。

---

## 多被试和参数实验

每名已准备被试使用同一配置；根据总 RAM/显存需求，从一到两个并发被试开始。
默认生成计划，加 `--execute` 才执行计算。未指定 `--continue-on-error` 时，
一次失败后停止提交新任务，等待已运行任务结束。
批次 manifest 汇总每个被试的实际运行状态及任务数量。即使单被试 CLI 允许部分执行，
受阻子任务也会使批次记为 `completed_with_blocks`，默认返回非零退出码。
明确接受部分结果时才使用 `--allow-partial-success`；它允许退出码为 0，但仍保留受阻状态。

<!-- cli-template: cohort -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root "/path/to/prepared_study" --output-root "/path/to/cohort_results" \
  --run-name cohort_dti --jobs 2 --no-user-config --strict-environment \
  --targets validation,microstructure --microstructure-stages dti --brain-mask-backend provided
```

命名参数组合参见[实验请求示例](examples/dti_experiments.json)。替换其中的绝对路径，
JSON 不展开 `$VARIABLES`。使用新的实验输出目录；所有组合先校验再提交，
分别保存输出并记录失败和未启动的任务。
需要时在请求的 `base_args` 中设置共用的显式依赖路径（例如 `--model-assets-dir`）。
`options` 和每个组合的 `args` 用于科学参数，身份、输出路径和执行模式由调度器管理。

<!-- cli-template: experiments -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_experiments.py" \
  --request "/path/to/edited_experiment_request.json"
```

加 `--execute` 才执行。一个被试的不同组合串行，被试间可以并发。
跨被试计算阶段的共享准入可使用 `--scheduler resources`；省略资源参数时自动规划，
可用 `--jobs 3` 指定活跃被试上限。高级资源覆盖见[资源感知队列处理](COHORT_RESOURCES.zh-CN.md)。该模式将被试内部模型并发设为一，
保留既有运行记录，不常驻模型，也不把多被试合成一个推理批次。
`--scheduler adaptive` 增加受保护回填与实测资源反馈，也可通过
`--max-parallel-models 2` 或 `3` 在共享模型级 CPU/RAM 准入预算下并发独立模型。
首次使用建议保留单模型默认值，先检查少量被试。

GUI **高级设置 > 实验与批处理** 支持相同请求。被试并发数 `--jobs`
与单被试模型并发数 `--max-parallel-models` 是两个层级，资源占用可能相乘。
详见[批量实验](EXPERIMENTS.zh-CN.md)。

---

## 脑区分析和报告

已完成的单被试运行使用其记录路径，不扫描公共输出目录。
上述封装把元数据写到 `$OUTPUTS/<run>/00_run_metadata/<run>_orchestration_manifest.json`。
多追踪变体和实验各有子运行记录。

直接传入脑区与指标，不需要编写请求文件：

<!-- cli-template: region-run -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --source-manifest "/path/to/run_orchestration_manifest.json" \
  --regions 1024 2024 --metrics fa md \
  --output-dir "/path/to/new_region_results"
```

脑区分析复用适用的参数图、完整流线和矩阵，不重新拟合模型；
图谱中不存在的脑区、失败或过时来源会被拒绝。

`--regions` 接受名称或标签号，可用空格、逗号分隔，也可重复传入；多词名称加引号。
`--tissue`、`--surface` 添加已有 DDSeg 组织构成和皮层形态统计。
`--tractogram`、`--connectome` 指定记录中的任务键名。
`--tract-rule endpoint_pair --target-regions ...` 筛选 A-B 流线，
`--include-regions`、`--exclude-regions` 添加途经和排除区。
未指定任何分析类型时默认汇总 FA；歧义名称需指定标签号。需要查看脑区、指标及任务键名时：

<!-- cli-template: region-inspect -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --inspect "/path/to/run_orchestration_manifest.json"
```

此命令直接打印目录；可选 `--catalog /path/to/catalog.json` 保存文件。

传入多个已完成的单被试运行记录，即可导出队列 CSV：

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --source-manifest "/path/to/subject1/run.json" "/path/to/subject2/run.json" \
  --regions 1024 2024 --metrics fa md \
  --output-dir "/path/to/new_region_cohort"
```

JSON 作为可选高级接口保留：`--request` 接受[区域请求](examples/region_request.json)
或 [A-B 请求](examples/region_pair_request.json)，`--batch` 接受[逐被试覆盖配置](examples/region_batch.json)。
JSON 模式不与直接分析参数混用。完整参数见[脑区分析说明](REGIONAL_ANALYSIS.zh-CN.md)。

<!-- cli-template: report -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/generate_subject_report.py" \
  --subject-root "$SUBJECT" --output-root "$OUTPUTS/dti" --run-name dti \
  --manifest "$OUTPUTS/dti/00_run_metadata/dti_orchestration_manifest.json"
```

检查任务状态、QC、排除的拟合数、束覆盖率和声明的输出签名。
在 **结果与 QC** 中选择相应运行并加载/复用节点。CLI 不直接修改已打开的 Slicer 场景；
重新生成报告不会将失败 QC 改为通过。

---

## 原生 PowerShell

参数使用数组，不直接复制 Bash 的续行符和 JSON 引号写法。
PNL、MRtrix/FSL、SAF 和原生表面后处理需先核对平台支持。

<!-- cli-setup: powershell -->
```powershell
$Slicer = 'C:\Apps\Slicer\Slicer.exe'
$Repo = 'C:\work\SlicerDSpace'
$Subject = 'D:\study\sub-01'
$SubjectId = 'sub-01'
$Outputs = 'D:\analysis\sub-01'
$Assets = 'D:\SlicerDSpace-models'
$DependencyArgs = @()
# $DependencyArgs = @('--dependency-config', 'D:\configs\dependencies.json')

function Invoke-DDSurfer {
    param([string]$Run, [string[]]$RouteArgs)
    & $Slicer --launch PythonSlicer "$Repo\shared\run_subject_pipeline_cli.py" `
        --no-user-config --strict-environment `
        --subject-root $Subject --subject-id $SubjectId `
        --output-root (Join-Path $Outputs $Run) --run-name $Run `
        --model-assets-dir $Assets @DependencyArgs @RouteArgs
    if ($LASTEXITCODE -ne 0) { throw "SlicerDSpace exited with $LASTEXITCODE" }
}

Invoke-DDSurfer -Run 'dti' -RouteArgs @(
    '--targets', 'validation,microstructure', '--microstructure-stages', 'dti',
    '--brain-mask-backend', 'provided', '--plan'
)
```

---

## 复用、取消和帮助

默认可复用匹配的已完成结果；比较方法时用新的运行名/输出路径。
`--no-resume` 请求重算，不消除 QC 失败。不要把删缓存或调低科学阈值当作通用修复。
GUI 取消只停止所属任务树；Agent/终端应保留自己启动的进程/进程组句柄，
先等待清理，避免终止其他 Slicer/Python 会话。

选项有差异时，以**已安装版本**的帮助为准：

```bash
"$SLICER" --launch PythonSlicer "$REPO/shared/run_subject_pipeline_cli.py" --help
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" --help
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" --help
```

本页标记的模板由本地回归测试读取并检查参数及路线选择；这不代替每条路线的真实数据验收。
## 按所选任务进行可选 CPU 校准

Batch 与单被试 CLI 复用同一套目标选择规则，支持重复的 `--targets`、算法子集与
`--skip-targets`；不会因为选择单个模型就运行完整 pipeline。已有有效中间结果按原凭据复用。

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/preprocessed_subjects --output-root /path/to/results \
  --scheduler adaptive --jobs 3 --autotune \
  --targets microstructure --microstructure-stages dki --execute
```

`--autotune` 在所选计算前进行有界校准，默认每个候选最多使用四个 CPU，可由
`--tuning-max-cpus` 调整。DKI 搜索数值线程数；MK-Curve 另外搜索 worker 数与体素块大小。
预热后轮换执行三次重复，全部采样科学产物精确一致的候选才可入选，低于 5% 的收益保留基线。
整批容量估算仍使用完整输入的 RAM 下限。匹配的硬件、实现、维度及方法配置通过
`--compute-profile /path/to/compute_profile.json` 复用。校准耗时与原始试验单独记录，不当作整批加速率。

用户显式设置的线程、worker、块大小及阶段预约优先。MK-Curve 动态补充完成的体素块，
按原体素位置写回；worker×线程数不得超过阶段 CPU 预约。`--mkcurve-chunk-schedule ordered`
保留顺序控制路径。搜索只保证在已测试且通过一致性检查的候选中选择，不保证全局最优。

校准需要最终预处理后的 NIfTI 输入；包含预处理的批次先保留声明配置。WMA 完整模块校准需要
已有流线、图谱、WMA 后端与 Slicer launcher，可能耗时较长；缺失时保留预约，不为调参额外触发追踪。
未选算法、没有适配器的算法以及 dry run 均不启动校准。
