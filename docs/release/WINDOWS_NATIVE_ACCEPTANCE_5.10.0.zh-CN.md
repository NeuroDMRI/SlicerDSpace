<div align="center">

# Windows 原生运行验收报告（3D Slicer 5.10.0）

**全流程真实数据验收、与 Linux 参考结果数值一致性比对及原生运行包校验**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [扩展准备状态](EXTENSION_READINESS.zh-CN.md) | [English](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md)

</div>

本报告记录 **SlicerDSpace** 在 **3D Slicer 5.10.0**（`revision 34045`，内置 Python `3.12.10`，Windows `AMD64`）环境下，显式禁用 WSL（`SLICERDSPACE_DISABLE_WSL=1`）后的正式原生运行验收结果。所有工作流均使用真实 Human Connectome Project（`HCP 100307`）多壳层 dMRI 数据完成计算，并与 Linux 参考环境（`Ubuntu x86_64`，3D Slicer `5.10.0` + MRtrix `3.0.8` + FSL `2602.0` + FreeSurfer `7.4.1` / `7eb8460`）进行逐项数值一致性对比。

---

## 1. 目标版本与测试环境对齐

| 项目 | 验收配置 |
| --- | --- |
| **目标 Slicer 版本** | **3D Slicer 5.10.0**（`r34045`，Python `3.12.10`） |
| **多版本兼容性设计** | 同时兼容 **3D Slicer 5.6.2**（Python `3.9`）与 **3D Slicer 5.10.0**（Python `3.12`）；`stage_slicer_extension.py` 与 `CMakeLists.txt` 动态适配 `lib/Slicer-<major>.<minor>/qt-scripted-modules`，Windows 多线程并行自动规避 `SlicerApp-real.exe` 子进程派生问题 |
| **操作系统与环境** | Windows 10/11 `x86_64`（`os.name == "nt"`，全程强制 `SLICERDSPACE_DISABLE_WSL=1`） |
| **真实测试数据** | `HCP 100307` 多壳层 dMRI（`145 × 174 × 145 × 288`，`b ∈ {0, 1000, 2000, 3000} s/mm²`，`1.25 mm` 各向同性体素） |
| **Linux 参考环境** | Ubuntu Linux `x86_64`（`3D Slicer 5.10.0`、MRtrix3 `3.0.8`、FSL `2602.0`、FreeSurfer `7.4.1` / `7eb8460`） |

---

## 2. 全算法逐项真实数据验收矩阵（零 WSL 依赖）

| 优先级 | 算法 / 工作流 | 是否使用 Windows 原生程序（未调用 WSL） | 是否使用真实 dMRI 数据（`HCP 100307`） | 输出文件与自动 QC 状态 | 与 Linux 参考结果的数值一致性 |
| --- | --- | --- | --- | --- | --- |
| **P0** | **FSL 预处理**（`topup`、`applytopup`、`eddy_cpu`） | 是 — 原生 `topup.exe`、`applytopup.exe`、`eddy_cpu.exe`（`fsl-2602.0-windows-x86_64`），通过 `preprocessing/fsl_native.py` 调度 | 是（`HCP 100307` 反向相位编码 `b0` 对 + dMRI） | 通过（生成 `dwi_preprocessed.nii.gz`、旋转 `.bvec`、`.eddy_parameters`、`.eddy_outlier_map`，`qc_passed = true`） | **`max_abs = 0.0`、`rmse = 0.0`、`relative_l2 = 0.0`**（`topup fieldcoef`、`movpar`、`eddy_cpu` 输出体数据、旋转 `bvecs` 及 `eddy_parameters` 完全一致） |
| **P0** | **MRtrix3 5TT 与灰白质交界**（`5ttgen fsl`、`5ttgen freesurfer`、`5tt2gmwmi`、`mrcrop`） | 是 — 原生 `5ttgen`、`5tt2gmwmi.exe`、`mrconvert.exe`、`mrmath.exe`、`mrthreshold.exe`、`mrcalc.exe`、`mrtransform.exe`、`mrcrop.exe`、`labelconvert.exe` + 内置 `share/mrtrix3/_5ttgen/FreeSurferColorLUT.txt`（`mrtrix-3.0.8-windows-x86_64`，共 68 个命令） | 是（`HCP 100307` 全脑 `145 × 174 × 145`） | 通过（生成 `5tt.mif`、`5tt_fs.mif`、`gmwmi.mif`、`dwi_cropped.mif`；`5ttgen freesurfer` 无需 `FREESURFER_HOME` 即可通过 `5ttcheck.exe`） | **`max_abs = 0.0`、`rmse = 0.0`、`pearson_r = 1.0`** |
| **P0** | **MRtrix3 MSMT-CSD 与多组织归一化**（`dwi2response dhollander`、`dwi2fod msmt_csd`、`mtnormalise`） | 是 — 原生 `dwi2response`、`dwi2fod.exe`、`mtnormalise.exe`（`mrtrix-3.0.8-windows-x86_64`） | 是（`HCP 100307` 多壳层 dMRI） | 通过（生成 `wm/gm/csf_response.txt`、`wmfod/gm/csf.mif`、`wmfod_norm/gm_norm/csf_norm.mif`） | 响应函数：**`max_abs <= 1.0e-10`、`r = 1.0`**；FOD 与归一化 FOD：**`max_abs <= 4.17e-7`、`rmse <= 2.34e-9`、`pearson_r = 1.0`** |
| **P0** | **MRtrix3 ACT + iFOD2 + SIFT2 + Connectome + 流线工具**（`tckgen`、`tckedit`、`tckresample`、`tckmap`、`tcksift2`、`tck2connectome`） | 是 — 原生 `tckgen.exe`、`tckedit.exe`、`tckresample.exe`、`tckmap.exe`、`tcksift2.exe`、`tck2connectome.exe`、`labelconvert.exe`（`mrtrix-3.0.8-windows-x86_64`） | 是（`HCP 100307`） | 通过（生成 `tracks.tck`、`tracks_edited.tck`、`tracks_resampled.tck`、`tdi.mif`、`sift2_weights.txt`、`84 × 84` 对称连接矩阵，`zero_diagonal = true`） | 确定性 `SD_STREAM`：**`max_abs = 0.0` mm、`rmse = 0.0` mm**；相同 `.tck` 下 `tcksift2` 权重：**`max_abs = 3.21e-6`、`rmse = 1.85e-7`、`r = 1.0`**；`tck2connectome` 矩阵：**`max_abs = 9.09e-5`、`rmse = 3.16e-6`、`r = 1.0`** |
| **P0** | **FreeSurfer 表面后处理**（7 个原生命令） | 是 — 原生 `mris_curvature.exe`、`mris_inflate.exe`、`mris_sphere.exe`、`mris_register.exe`、`mris_place_surface.exe`、`mri_label2label.exe`、`mri_surf2surf.exe`（`surface-runtime-windows-x86_64`） | 是（`HCP 100307` 左右半球皮层网格 + `wmparc.mgz`） | 通过（生成 `lh/rh.thickness`、`curv`、`sulc`、`area`、`area.pial`、`white.H`、`white.K`、`inflated`、`sphere`、`sphere.reg`、`cortex.label`、`aparc.annot`、`aparc.a2009s.annot`） | 同源码版本 `mris_place_surface`（`7eb8460`）：**`lh/rh.thickness` 与 `lh/rh.curv` `max_abs = 0.0`、`rmse = 0.0`、`r = 1.0`**；`area`/`sulc`/`white.H`：**`r >= 0.99999999`**；`inflated`：**`rmse = 2.4e-5` mm**；`aparc` / `aparc.a2009s` 脑区映射与 Linux FreeSurfer 7.4.1 一致率 **`97.55%–98.79%`** |
| **P1** | **DDSeg**（组织概率图与四分类分割） | 是 — Slicer 5.10.0 原生 PyTorch（`dti_axial/coronal/sagittal.pt`）+ Teem（`anatomy/segmentation/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`881,299` 脑内体素） | 通过（生成 `PredLabels_4classes.nii.gz`、`ProbabilisticMap_GM/WM/CSF/NoBrain.nii.gz`，`qc_passed = true`） | **脑内体素分类标签一致率 `100.00%`（`881,299 / 881,299` 体素完全一致）**；概率图 **`max_abs <= 1.01e-6`、`rmse <= 4.14e-8`、`pearson_r = 1.0`** |
| **P1** | **DDParcel**（102 脑区解剖分区） | 是 — Slicer 5.10.0 原生 `BRAINSFit.exe` + `ResampleScalarVectorDWIVolume.exe` + 三视角 Fuse-UNet v3（`anatomy/parcellation/pipeline.py`） | 是（`HCP 100307` 全脑体数据） | 通过（生成 `100307-DDSurfer-wmparc.nii.gz`，完整包含 `102/102` 个标签，`qc_passed = true`） | **`102/102` 脑区标签集合完全一致**，脑内体素一致率 **`91.71%`**（边界差异仅源于 `BRAINSFit` 多线程仿射配准随机采样） |
| **P1** | **DDSurfer**（皮层白质面与软脑膜面重建） | 是 — Slicer 5.10.0 原生 PyTorch CMU-NeXt + 网格形变（`ddsurfer_lh/rh_wm/pial.pt`） | 是（`HCP 100307` 全脑双半球） | 通过（生成 `lh.white` `164,239` 顶点、`rh.white` `164,694` 顶点及 `lh/rh.pial`） | **`max_abs = 0.0` mm、`rmse = 0.0` mm、三角面片拓扑 `100%` 完全一致** |
| **P1** | **DTI**（`FA`、`MD`、`AD`、`RD`、`L1`、`L2`、`L3`、`Trace` + ROI 统计表） | 是 — Slicer 5.10.0 原生 Teem `tend estim` WLS + `tend anvol/eval`（`shared/teem_dti.py`、`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`881,299` 脑内体素） | 通过（8 个全脑标量 NIfTI 图 + `101` 脑区 ROI 统计 CSV，`status = completed`） | **`FA`、`MD`、`AD`、`L1`：`max_abs = 0.0`、`rmse = 0.0`**；**`L2`、`L3`、`RD`、`Trace`：`max_abs <= 3.64e-12`、`rmse <= 1.90e-15`、`pearson_r = 1.0`** |
| **P1** | **Free-Water 自由水消除**（`fwf`、`fw_tissue_fa`、`fw_tissue_md`） | 是 — Slicer 5.10.0 原生 DIPY/SciPy 正则化非线性拟合（`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`881,299` 脑内体素） | 通过（生成全脑 `fwf.nii.gz` `881,042` 非零体素、`fw_tissue_fa.nii.gz` 与 `fw_tissue_md.nii.gz` `846,558` 非零体素，耗时 `710.20 s`，`automatic_qc_passed = true`） | **`max_abs = 0.0`、`rmse = 0.0`、`pearson_r = 1.0`** |
| **P1** | **NODDI**（`noddi_ndi`、`noddi_odi`、`noddi_fwf`） | 是 — Slicer 5.10.0 原生 AMICO 凸优化求解（`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`881,299` 脑内体素） | 通过（生成全脑 `noddi_odi.nii.gz` `881,299` 体素、`noddi_ndi.nii.gz` `873,290` 体素、`noddi_fwf.nii.gz` `516,440` 体素，耗时 `536.86 s`，`automatic_qc_passed = true`） | **`max_abs <= 2.28e-4`、`rmse <= 6.13e-6`、`pearson_r >= 0.9999999997`** |
| **P1** | **DKI**（`WLS` 与 `MK-Curve` `MK`、`AK`、`RK`、`FA`、`MD`、`AD`、`RD`） | 是 — Slicer 5.10.0 原生 DIPY WLS（`microstructure/dki.py`）与原生 Python 多线程 MK-Curve（`microstructure/mkcurve.py`） | 是（`HCP 100307` 全脑 `881,299` 脑内体素 WLS + 全脑多层 7 层切片 `87,092` 体素 WLS + 多层全脑采样 `1,500` 脑内体素 MK-Curve） | 通过（全脑 `881,299` 体素 WLS 耗时 `522.36 s`，`792,249` 有效体素；多层全脑采样 `1,500` 体素 MK-Curve `201` 合成 `b0` 网格耗时 `101.36 s`，`1,389` 有效体素，`automatic_qc_passed = true`） | `test_mkcurve_python.py` 全部 `15/15` 单元测试通过；真实全脑与多层切片数据峰度参数 `100%` 有限且与 Linux 参考分布一致 |
| **P1** | **DTI 确定性纤维追踪与 Python Connectome** | 是 — Slicer 5.10.0 原生 DIPY EuDX（`tractography/dti.py`）与 SciPy `cKDTree` 连接组（`connectome/pipeline.py`） | 是（`HCP 100307` 全脑体数据） | 通过（`9,825` 条流线，`84 × 84` `streamline_count` 与 `mean_fa` 矩阵，`564` 条有效连接边） | **流线数量 `9,825 / 9,825` 完全一致（`ratio = 1.0`）**；连接矩阵权重 **`pearson_r = 0.9922`** |
| **P1** | **SAF 浅表联合纤维** | 是 — 原生 MRtrix3（`tckgen.exe`、`tckedit.exe`、`tckresample.exe`、`tckmap.exe`、`tcksift2.exe`）+ 原生 `trimesh`/`rtree`（`tractography/saf.py`） | 是（`HCP 100307` + DDSurfer 皮层表面） | 通过（生成 `100307_saf_final.tck`、`lh_saf.tck`、`rh_saf.tck`，共 `2,350` 条流线） | **流线数量 `2,350 / 2,350` 完全一致（`lh = 1,158`、`rh = 1,192`，`ratio = 1.0`）** |
| **P1** | **WMA ORG 全流程 + 原生标量测量** | 是 — Slicer 5.10.0 原生 `whitematteranalysis` + 原生 VTK `tractography/wma_native_measures.py` | 是（`HCP 100307` 全脑纤维束） | 通过（两阶段配准 + `800` 纤维簇 + `74` 条 ORG 解剖束 + 标量统计 CSV） | **`74/74` 条 ORG 解剖白质束完整生成（`100%` 覆盖率）**；原生 VTK 标量统计与 `FiberTractMeasurements` CLI 交叉验证通过 |
| **P1** | **TractCloud**（深度学习白质束分割） | 是 — Slicer 5.10.0 原生 PyTorch PointNet+DGCNN（`tractography/tractcloud.py`） | 是（`HCP 100307` `9,825` 条流线） | 通过（生成 `42` 个解剖纤维束 `.vtp`，`non_empty_bundles = 42`，`qc_passed = true`） | **`42 / 42` 纤维束分类、`9,825 / 9,825` 流线归类 `100.0%` 完全一致** |
| **P1** | **WMQL**（白质查询语言） | 是 — Slicer 5.10.0 原生 `tract_querier`（`tractography/wmql.py`） | 是（`HCP 100307` + `wmparc.nii.gz`） | 通过（执行 `39` 条官方查询，`non_empty_tracts = 37`，`qc_passed = true`） | **`37 / 37` 非空纤维束及各束流线数与 Linux `100.0%` 完全一致** |

---

## 3. 原生运行依赖包（Windows + Linux）登记与 SHA-256 校验表

在 `v0.1.1` 中，每个平台仅向用户提供一个完整的统一安装包（`SlicerDSpace-v0.1.1-Windows-x64.zip` 与 `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`），内部将 MRtrix3、FSL 与 FreeSurfer Surface Runtime 模块化置于 `SlicerDSpace/bundled-runtimes/` 下，首次启动即自动识别，无需手动配置路径或分别下载多个依赖包。内部各模块归档的大小与 SHA-256 校验值登记于 [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json) 与 [`models/runtime_bundles.json`](../../models/runtime_bundles.json)：

| 运行包标识 | 平台 | 归档文件名 | 压缩包大小（字节） | 解压后大小（字节） | SHA-256 校验值 |
| --- | --- | --- | --- | --- | --- |
| `mrtrix`（`3.0.8`，68 个命令） | `windows-x86_64` | `mrtrix-3.0.8-windows-x86_64.zip` | `30,968,289` | `59,517,417` | `a0d5c3f12e6e54a914bb8690edf6b795ae432b82a1e99d741094f8460db8a20a` |
| `mrtrix`（`3.0.8`，68 个命令） | `linux-x86_64` | `mrtrix-3.0.8-linux-x86_64.tar.gz` | `27,145,376` | — | `c5b4ad88c20c569638ca9d49779f439a156247a417dc8f15d33cee0543aa5c7b` |
| `fsl`（`2602.0`，3 个命令） | `windows-x86_64` | `fsl-2602.0-windows-x86_64.zip` | `16,331,282` | `23,315,729` | `1a50663922099c3207e02a45f3416a6f4093fc28c982a123e8451f94f005816f` |
| `fsl`（`2602.0`，3 个命令） | `linux-x86_64` | `fsl-2602.0-linux-x86_64.tar.gz` | `42,914,629` | `54,487,316` | `327d92c8be1f09ebb9d22357de390b0ebfecd72ae5a917875fa48b3e6ad200ab` |
| `surface`（`0.1.0`，7 个命令） | `windows-x86_64` | `surface-runtime-windows-x86_64.tar.gz` | `66,211,294` | `258,835,483` | `bbc97ea1c89117713bf36db2e4f9dd0b23ad4b69b9e27cba1b7050c0bc71f742` |
| `surface`（`0.1.0`，7 个命令） | `linux-x86_64` | `surface-runtime-linux-x86_64.tar.gz` | `40,995,581` | `254,461,401` | `9e6005a114c7d1ff58f247ee2e8977f9eb80f48f525172e15d7e7174ab30fb10` |

> **第三方软件再分发与许可证合规检查**：所有内置组件均附带上游开源许可证文本（`third_party/LICENSES/`、各运行包 `LICENSE.txt` 及 `manifest.json`），且绝不打包任何个人或机构专属的 FreeSurfer 许可证或激活文件（`activation_key_included: false`）。

---

## 4. 干净环境安装与 3D Slicer 5.10.0 GUI 验收

1. **通过 3D Slicer 5.10.0 Extension Manager 安装统一扩展包验收**：
   - `Testing/Python/slicer_extension_manager_install_smoke.py` 在干净的 3D Slicer 5.10.0（`r34045`）配置下（开启 `SLICERDSPACE_DISABLE_WSL=1` 且清空用户运行包缓存）验证 `slicer.app.extensionsManagerModel().installExtension(...)` 安装 `SlicerDSpace-v0.1.1-Windows-x64.zip`：
     - `passed: true`、`archive_rooted: true`、`archive_clean: true`、`installed: true`、`manager_reports_installed: true`、`registered: true`、`module_loaded: true`、`module_available: true`、`runtime_resources_ready: true`、`installed_imports_ready: true`、`bundled_runtimes_ready: true`（自动识别内置的 `fsl-2602.0-windows-x86_64`、`mrtrix-3.0.8-windows-x86_64`、`surface-runtime-windows-x86_64`）、`self_tests_passed: true`。
2. **无 WSL 干净环境默认原生配置与功能自检**：
   - Windows GUI 默认启用原生运行环境（`preprocessing_execution_backend = "native"`、`use_native_runtime = True`、`wsl_required = False`）。
   - `shared/setup_selftest.py`（`run_checks`）全部 **46 项**合成安装自检（Python 核心库、Teem 自检、NumPy/NIfTI 读写、PyTorch 算子、全部 21 个解剖深度学习模型权重校验、原生 MRtrix3 / FSL / dcm2niix 命令可执行性）**100% 通过**。
3. **3D Slicer 5.10.0 真实 GUI Smoke 测试套件**：
   - `Testing/Python/slicer_extension_install_smoke.py`：**通过**（`exit_code = 0`）
   - `Testing/Python/slicer_environment_button_smoke.py`：**通过**（`exit_code = 0`，Qt 主线程心跳不阻塞）
   - `Testing/Python/slicer_gui_operations_smoke.py`：**通过**（`exit_code = 0`，结果加载、场景节点管理、安装队列、失败处理、任务取消与组件销毁全部通过）
   - `Testing/Python/slicer_cancel_background_smoke.py`：**通过**（`exit_code = 0`，Windows 后台进程树取消与 `cancelled` 清单落盘通过）
