<div align="center">

# 输出目录

[首页](../../README.zh-CN.md) | [快速开始](QUICKSTART.zh-CN.md) | [网站](../website/README.zh-CN.md) | [English](OUTPUT_LAYOUT.md)

</div>

---

## 当前目录

```text
<subject_root>/
  input/
  preprocessed/                         # 在自动生成脑掩膜 (nodif_brain_mask.nii.gz) 或运行预处理 (<sub_id>_dwi_final.*、<sub_id>_brain_mask_final.nii.gz) 时创建
  cache/                                # 阶段工作区、日志、manifest 与 Slicer 降采样预览
    stages/
    display_previews/
  outputs/
    00_run_metadata/
    01_anatomical_priors/
      tissue_segmentation/              # SegmentationMap_GMWMCSF.nii.gz, ProbabilisticMap_{WM,GM,CSF}.nii.gz
      brain_parcellation/               # <sub_id>_parcellation.nii.gz
      cortical_surface/                 # lh.white.obj, lh.pial.obj, rh.white.obj, rh.pial.obj
      surface_morphometry/              # {lh,rh}.{aparc,aparc.a2009s}.stats, {thickness,curv,sulc,area,inflated}, *.annot
    02_microstructure/
      <sub_id>_ROIWise-Microstructure.csv
      diffusion_tensor/                 # <sub_id>_{FA,MD,AD,RD}.nii.gz, <sub_id>_DEC.nrrd
      free_water/                       # <sub_id>_FW.nii.gz, <sub_id>_fw_FA.nii.gz, <sub_id>_fw_MD.nii.gz
      interstitial_free_water/          # <sub_id>_ifw.nii.gz, <sub_id>_ROIWise-iFW.csv
      neurite_microstructure/           # <sub_id>_{NDI,ODI,IsoVF}.nii.gz
      diffusion_kurtosis/               # <sub_id>_{MK,AK,RK,KFA,MKT,RTK}.nii.gz
    03_tractography/
      single_tensor_tractography/       # <sub_id>_dti.trk[, .vtk]
      multi_tensor_tractography/        # <sub_id>_ukf.trk, <sub_id>_ukf.vtk
      anatomically_constrained_tractography/ # <sub_id>_act.{tck,trk,vtk}
      short_association_fibers/         # <sub_id>_saf.{tck,vtk}
    04_bundles/
      query_bundle_extraction/<source>/ # WMQL: *.trk
      fiber_clustering/<source>/        # WMA: T_*.vtp, bundle_measurements.csv
      bundle_identification/<source>/   # TractCloud: *.vtp
    05_connectome/                      # <sub_id>_{streamline_count,sift2_weighted,<metric>}.csv, <sub_id>_nodes.tsv, <sub_id>_node_parcellation.nii.gz
    06_qc/                              # 仅保存已运行或复用任务的简洁 TXT 质控摘要
```

`00_run_metadata/run_record.json` 在每次运行结束时写出，列出运行 manifest、参数、软件和资源版本、
已接受的许可、脑掩膜来源（请求的后端及本次运行所用掩膜的路线）、QC 摘要，以及每个最终产品文件的 SHA-256。
它不复制任何输出文件。重新运行时，上一次的记录保存为 `run_record-<started_at>.json`，不删除任何文件。

`outputs/` 下仅保留扁平、无 symlink 的最终科学交付物与 `06_qc/` 的简洁 TXT 质控文件；阶段缓存、日志与降采样预览统一存放于 `cache/`。

每个纤维束文件夹按输入纤维追踪结果分子文件夹。`<source>` 是生成该追踪结果的追踪方法 id（`csd_act`、`ukf`、`dti_det`、
`saf`），只有一种追踪时也一样。用户以文件形式提供追踪结果时（WMA 或 TractCloud 输入文件），`<source>` 为 `supplied`。
有两种或更多追踪时，每个追踪变体写入自己的子文件夹，因此一个变体不会覆盖另一个变体的纤维束文件。每个纤维束算法使用哪种追踪，
由该算法的“纤维追踪来源”设置决定（GUI 设置弹窗；CLI `--bundle-source ALG=SOURCE`）；默认行为不变：WMQL 和 TractCloud
使用每个已选追踪，WMA 使用 UKF。此布局之前的结果在这些文件夹中保留扁平文件，仍可读取；恢复运行时会把已完成阶段的输出重新发布到子文件夹，
不会重新计算。

---

## 兼容已有结果

旧 `03_anatomy`、`03_structural`、`01_preproc`、`04_tractography`、`05_statistics`、`06_slicer_scene_report` 目录继续支持只读兼容与缓存复用，亦可通过 `migrate_subject_output_tree` 迁移至新目录树。Python 接口 `SubjectIOLayout` 保留 `anatomy`、`structural`、`ddseg`、`ddparcel`、`ddsurfer` 等兼容属性别名。

按产品目录选择：新目录存在时优先使用新目录，否则再考虑旧目录。
不从旧文件补齐一个不完整的新产品。原有 QC、输入签名和版本检查不变；
发现旧目录不等于科学结果已验收。报告保留实际路径，并能解释两种目录名称。

历史验收报告和已冻结的插图来源记录保留当时的路径，避免改写实验事实。
它们不是新输出写入旧目录的指引。用户显式选择的第三方旧后端可能仍返回旧路径，
封装层会兼容读取其实际结果。

---

## 本地验证

GitHub Actions 仅保留手动触发。使用 `make test-fast` 在本地运行目录兼容与回归测试；
数据处理不依赖 Actions。
