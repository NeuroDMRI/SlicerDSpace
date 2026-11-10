<div align="center">

# macOS 原生运行验收报告（3D Slicer 5.10.0 — Apple Silicon `arm64`）

**全流程真实数据验收、与 Linux 参考结果数值一致性比对及原生运行包校验**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [扩展准备状态](EXTENSION_READINESS.zh-CN.md) | [English](MACOS_NATIVE_ACCEPTANCE_5.10.0.md)

</div>

本报告记录 **SlicerDSpace** 在 **3D Slicer 5.10.0**（`revision 34045`，内置 Python `3.12.10`，Apple Silicon `darwin-arm64`，macOS `15.7.1`）环境下，无需 Docker、无需 WSL、无需手动编译 C/C++ 工具链的正式原生运行验收结果。所有工作流均使用真实 Human Connectome Project（`HCP 100307`）多壳层 dMRI 数据完成计算，并与 Linux 参考环境（`Ubuntu x86_64`，3D Slicer `5.10.0` + MRtrix `3.0.8` + FSL `2602.0` + FreeSurfer `7.4.1` / `7eb8460`）进行逐项数值一致性对比。

> **硬件验证范围说明**：本报告中的全部真实数据运行、数值一致性比对与 GUI 冒烟测试均在 **Apple Silicon（`arm64`）** 真机上完成验证。虽然 `models/native_runtime_bundles.json` 已登记 `mrtrix-3.0.8-darwin-x86_64.tar.gz`，但 **Intel（`x86_64`）Mac 真机尚未实测**，明确标注为**未验证（unverified）**。

---

## 1. 目标版本与测试环境对齐

| 项目 | 验收配置 |
| --- | --- |
| **目标 Slicer 版本** | **3D Slicer 5.10.0**（`r34045`，Python `3.12.10`，`darwin-arm64`） |
| **多版本兼容性设计** | 同时兼容 **3D Slicer 5.6.2**（Python `3.9`）与 **3D Slicer 5.10.0**（Python `3.12`）；`stage_slicer_extension.py` 与 `CMakeLists.txt` 动态适配 `lib/Slicer-<major>.<minor>/qt-scripted-modules` |
| **操作系统与硬件** | macOS `15.7.1`（`24G231`，Apple Silicon `arm64`，8 核 CPU，16 GB 内存）；Intel（`x86_64`）Mac 真机硬件**未验证** |
| **真实测试数据** | `HCP 100307` 多壳层 dMRI（`145 × 174 × 145 × 288`，`b ∈ {0, 1000, 2000, 3000} s/mm²`，`1.25 mm` 各向同性体素，脑内掩膜 `740,916` 体素） |
| **Linux 参考环境** | Ubuntu Linux `x86_64`（`3D Slicer 5.10.0`、MRtrix3 `3.0.8`、FSL `2602.0`、FreeSurfer `7.4.1` / `7eb8460`） |

---

## 2. 全算法逐项真实数据验收矩阵（macOS `arm64` 原生）

| 优先级 | 算法 / 工作流 | 是否使用 macOS 原生程序（无 Docker / 无手动编译） | 是否使用真实 dMRI 数据（`HCP 100307`） | 输出文件与自动 QC 状态 | 与 Linux 参考结果的数值一致性 |
| --- | --- | --- | --- | --- | --- |
| **P0** | **FSL 与 DIPY 预处理**（`DIPY MP-PCA`、`DIPY Gibbs`、`topup`、`applytopup`、`eddy_cpu`） | 是 — 原生 `topup`、`applytopup`、`eddy_cpu`（`fsl-2602.0-darwin-arm64`）+ `mrconvert`/`dwiextract`/`mrcat`/`dwi2mask`（`mrtrix-3.0.8-darwin-arm64`），通过 `preprocessing/fsl_native.py` 调度 | 是（`HCP 100307` 反向相位编码 `b0` 对 + dMRI） | 通过（生成 `100307_dwi_final.nii.gz`、旋转 `100307_dwi_final.bvec`、`100307_eddy_parameters.txt`、`100307_eddy_outlier_map.txt`，耗时 `70.46 s`，`status = completed`） | `MP-PCA`：**`r = 1.0`（`rmse = 2.36e-4`）**；`Gibbs`：**`r = 1.0`（`rmse = 0.0309`）**；`topup fieldcoef`：**`r = 0.9999988`（`rmse = 0.0063`）**；`topup movpar`：**`r = 0.9999980`（`rmse = 1.32e-5`）**；`applytopup`：**`r = 0.999999995`**；旋转 `bvec`：**`r = 0.999995`（`rmse = 0.00177`）**；`eddy_movement_rms`：**`r = 0.99998`**；`eddy_parameters`：**`r = 0.9946`**；`100307_dwi_final.nii.gz`：**`r = 0.9926`** |
| **P0** | **MRtrix3 5TT 与灰白质交界**（`act_prep.py`、`5ttcheck`、`5tt2gmwmi`、`5ttgen freesurfer`） | 是 — 原生 `5ttcheck`、`5tt2gmwmi`、`5ttgen`、`mrconvert`、`mrmath`、`mrthreshold`、`mrcalc`、`mrtransform`、`mrcrop`、`labelconvert` + 内置 `FreeSurferColorLUT.txt`（`mrtrix-3.0.8-darwin-arm64`，共 67 个命令） | 是（`HCP 100307` 全脑 `145 × 174 × 145`） | 通过（生成 `100307_5tt.nii.gz`、`100307_gmwmi.nii.gz`，`5ttcheck` 校验通过，`status = completed`） | 相同输入下 **`max_abs = 0.0`、`rmse = 0.0`、`pearson_r = 1.0`** |
| **P0** | **MRtrix3 MSMT-CSD 与多组织归一化**（`dwi2response dhollander`、`dwi2fod msmt_csd`、`mtnormalise`） | 是 — 原生 `dwi2response`、`dwi2fod`、`mtnormalise`（`mrtrix-3.0.8-darwin-arm64`） | 是（`HCP 100307` 多壳层 dMRI） | 通过（生成 `wm/gm/csf_response.txt`、`wmfod/gm/csf.mif`、`wmfod_norm/gm_norm/csf_norm.mif`，`status = completed`） | 响应函数：**`max_abs <= 1.0e-10`、`r = 1.0`**；归一化 FOD：**`pearson_r = 1.0`** |
| **P0** | **MRtrix3 ACT + iFOD2 + SIFT2 + Connectome**（`tckgen`、`tcksift2`、`tck2connectome`） | 是 — 原生 `tckgen`、`tcksift2`、`tck2connectome`、`labelconvert`、`tckedit`、`tckresample`、`tckmap`（`mrtrix-3.0.8-darwin-arm64`） | 是（`HCP 100307` 全脑 `100,000` 条 ACT 流线） | 通过（生成 `100307_csd_act.tck`、`.trk`、`.vtk`、`sift2_weights.txt`、`sift2_mu.txt` 及 `84 × 84` 对称 `streamline_count` 与 `sift2_weighted` 连接组矩阵，`status = completed`） | `84 × 84` 连接组节点集合 **`100%` 完全一致**（对称、`zero_diagonal = true`）；随机概率追踪 `iFOD2` 下 `sift2_mu` 差异 `< 1.2%` |
| **P0** | **FreeSurfer 表面后处理**（7 个 `arm64` 原生命令） | 是 — 原生 `mris_curvature`、`mris_inflate`、`mris_sphere`、`mris_register`、`mris_ca_label`、`mris_anatomical_stats`、`mri_surf2surf`（`surface-runtime-darwin-arm64`） | 是（`HCP 100307` 左右半球皮层网格 + `wmparc.mgz`） | 通过（生成 `lh/rh.thickness`、`curv`、`sulc`、`area`、`area.pial`、`white.H/K`、`pial.H/K`、`inflated`、`sphere`、`sphere.reg`、`aparc.annot`、`aparc.a2009s.annot`、`lh/rh.aparc*.stats` 及 `fsaverage` 映射结果，耗时 `2142.7 s`，`automatic_qc_passed = true`） | **`lh/rh.thickness`、`curv`、`sulc`、`area`、`area.pial`、`white.H/K`、`pial.H/K`、`inflated`：`pearson_r >= 0.9999999`**；`aparc` / `aparc.a2009s` 顶点标签与 Linux 一致率 **`98.1%–99.2%`** |
| **P1** | **DDSeg**（组织概率图与四分类分割） | 是 — Slicer 5.10.0 原生 PyTorch（`dti_axial/coronal/sagittal.pt`）+ Teem（`anatomy/segmentation/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`740,916` 脑内体素） | 通过（生成 `SegmentationMap_GMWMCSF.nii.gz`、`ProbabilisticMap_GM/WM/CSF.nii.gz`，耗时 `272.4 s`，`status = completed`） | **脑内体素分类标签一致率 `99.8900%`（`740,101 / 740,916` 体素）、全图体素一致率 `99.9777%`**；概率图 **`rmse <= 0.00222`、`pearson_r >= 0.999985`** |
| **P1** | **DDParcel**（101 脑区解剖分区） | 是 — Slicer 5.10.0 原生 `BRAINSFit` + `ResampleScalarVectorDWIVolume` + 三视角 Fuse-UNet v3（`anatomy/parcellation/pipeline.py`） | 是（`HCP 100307` 全脑体数据） | 通过（生成 `100307-DDSurfer-wmparc.nii.gz`，完整包含 `101/101` 个非零标签，耗时 `3603.4 s`，`status = completed`） | **`101/101` 非零脑区标签集合 `100%` 完全一致（`exact_label_set_match = true`）**，脑内体素一致率 **`92.8852%`**（全图 `98.5589%`），`DDSeg WM vs DDParcel WM Dice = 0.860` |
| **P1** | **DDSurfer**（皮层白质面与软脑膜面重建） | 是 — Slicer 5.10.0 原生 PyTorch CMU-NeXt + 网格形变（`ddsurfer_lh/rh_wm/pial.pt`） | 是（`HCP 100307` 全脑双半球） | 通过（生成 `lh.white` 与 `lh.pial` `164,239` 顶点 / `328,474` 面片、`rh.white` 与 `rh.pial` `164,694` 顶点 / `329,384` 面片，耗时 `484.3 s`，`automatic_qc_passed = true`） | **三角面片拓扑 `100%` 完全一致（`faces_identical = true`、`watertight = true`、`euler_number = 2`）**，平均顶点位移 `<= 0.14 mm`（源于上游 `DDParcel` `BRAINSFit` 多线程仿射采样） |
| **P1** | **DTI**（`FA`、`MD`、`AD`、`RD`、`L1`、`L2`、`L3`、`Trace` + ROI 统计表） | 是 — Slicer 5.10.0 原生 Teem `tend estim` WLS + `tend anvol/eval`（`shared/teem_dti.py`、`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`740,916` 脑内体素） | 通过（8 个全脑标量 NIfTI 图 + `101` 脑区 `144` 列 ROI 统计 CSV，`status = completed`） | **`FA`、`AD`、`RD`、`L1`、`L2`、`L3`、`Trace`：`max_abs = 0.0`、`rmse = 0.0`、`pearson_r = 1.0`**；**`MD`：`max_abs = 2.33e-10`、`rmse = 2.00e-12`、`pearson_r = 1.0`** |
| **P1** | **Free-Water 自由水消除**（`fraction`、`FA`、`MD`） | 是 — Slicer 5.10.0 原生 DIPY/SciPy 正则化非线性拟合（`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`740,916` 脑内体素） | 通过（生成全脑 `100307_fw_fraction.nii.gz`、`100307_fw_FA.nii.gz`、`100307_fw_MD.nii.gz`，耗时 `319.12 s`，`status = completed`） | **`fraction` 与 `MD`：`max_abs = 0.0`、`rmse = 0.0`、`pearson_r = 1.0`**；**`FA`：`max_abs = 5.96e-8`、`rmse = 7.22e-11`、`pearson_r = 1.0`** |
| **P1** | **NODDI**（`NDI`、`ODI`、`IsoVF`、`RMSE`） | 是 — Slicer 5.10.0 原生 AMICO 凸优化求解（`microstructure/pipeline.py`） | 是（`HCP 100307` 全脑体数据，`740,916` 脑内体素） | 通过（生成全脑 `100307_noddi_NDI.nii.gz`、`ODI.nii.gz`、`IsoVF.nii.gz`、`RMSE.nii.gz`，耗时 `511.54 s`，`status = completed`） | `NDI`：**`rmse = 1.52e-4`、`r = 0.99999978`**；`ODI`：**`rmse = 2.38e-4`、`r = 0.99999945`**；`IsoVF`：**`rmse = 5.75e-5`、`r = 0.99999997`**；`RMSE`：**`rmse = 2.73e-5`、`r = 0.99999999999`** |
| **P1** | **DKI**（`WLS` 与 `MK-Curve` `MK`、`AK`、`RK`、`KFA`、`MKT`、`RTK`） | 是 — Slicer 5.10.0 原生 DIPY WLS（`microstructure/dki.py`）与原生 Python 多线程 MK-Curve（`microstructure/mkcurve.py`） | 是（`HCP 100307` 全脑 `740,916` 脑内体素 WLS 耗时 `1473.13 s`；多层全脑采样 `1,500` 脑内体素 MK-Curve 耗时 `125.01 s`） | 通过（全脑 WLS `735,319` 有效体素 + `101` 脑区 42 列 DKI ROI 统计；多层 `1,500` 体素 MK-Curve `202` 合成 `b0` 网格，`1,483` 有效体素，`status = completed`） | `WLS`（`MK`、`AK`、`RK`、`KFA`、`MKT`、`RTK`）：**`max_abs <= 2.38e-7`、`rmse <= 7.45e-10`、`pearson_r >= 0.999999999999996`**；`MKCURVE`：**`pearson_r = 0.9919–0.9999`**（`zero_mk_b0`: `r = 0.9999`，`corrected_fa`: `r = 0.9991`，`corrected_md/ad/rd`: `r = 0.9993`，`corrected_mk`: `r = 0.9957`，`corrected_rk`: `r = 0.9944`，`corrected_ak`: `r = 0.9919`） |
| **P1** | **DTI 确定性追踪、UKF 追踪与 Python Connectome** | 是 — Slicer 5.10.0 原生 DIPY EuDX（`tractography/dti.py`）、`UKFTractography` CLI（`tractography/ukf.py`）与 SciPy `cKDTree` 连接组（`connectome/pipeline.py`） | 是（`HCP 100307` 全脑体数据） | 通过（`dti_det` 耗时 `137.35 s`；`UKF` 耗时 `151.62 s`，`22,113` 条流线，`automatic_qc_passed = true`；Python `84 × 84` 连接组矩阵） | `UKF`：`effective_mask` SHA256 **`100%` 完全一致**（`ca500368...`），流线数 **`22,113` vs `22,023`（`ratio = 1.0041`）**，中位长度 **`34.487` vs `34.490` mm**，`points_inside_brain_fraction = 1.0` |
| **P1** | **SAF 浅表联合纤维** | 是 — 原生 MRtrix3（`tckgen`、`tckedit`、`tckresample`、`tckmap`、`tcksift2`）+ 原生 `trimesh`/`rtree`（`tractography/saf.py`） | 是（`HCP 100307` + DDSurfer 皮层表面） | 通过（生成 `SAF.tck`、`SAF.vtk`，耗时 `273.42 s`，共 `12,745` 条流线，`status = completed`） | 完整 8 阶段 SAF 流程通过端到端验证，皮层网格过滤与几何交集校验 `100%` 通过 |
| **P1** | **WMA ORG 全流程 + 原生标量测量** | 是 — Slicer 5.10.0 原生 `whitematteranalysis` + 原生 VTK `tractography/wma_native_measures.py` | 是（`HCP 100307` 全脑 UKF 纤维束） | 通过（两阶段配准 + `800` 纤维簇 + `74/74` 条 ORG 解剖束 + 4 个扩散标量统计 CSV，`automatic_qc_passed = true`，`all_tract_sampling_passed = true`） | **`74 / 74` 条 ORG 解剖白质束完整生成（`100%` 覆盖率，`excluded_from_quantification = []`）**；原生 VTK 标量统计 CSV 生成并校验通过 |
| **P1** | **TractCloud**（深度学习白质束分割） | 是 — Slicer 5.10.0 原生 PyTorch PointNet+DGCNN（`tractography/tractcloud.py`） | 是（`HCP 100307` 全脑流线） | 通过（生成 `42` 个解剖纤维束 `.vtp` + `6` 个 `.mrb` 场景包，耗时 `322.77 s`，`automatic_qc_passed = true`） | **`42 / 42` 解剖纤维束 `100%` 完整生成** |
| **P1** | **WMQL**（白质查询语言） | 是 — Slicer 5.10.0 原生 `tract_querier`（`tractography/wmql.py`） | 是（`HCP 100307` + `wmparc.nii.gz`） | 通过（耗时 `8.01 s`，执行 `39` 条官方查询，`non_empty_tracts = 37`，`execution_passed = true`，`output_integrity_passed = true`） | **`37 / 37` 非空纤维束集合与 Linux `100.0%` 完全一致** |

---

## 3. macOS 原生运行依赖包登记与 SHA-256 校验表

在 `v0.1.1` 中，Apple Silicon macOS 仅向用户提供一个完整的统一安装包（`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`），内部将 MRtrix3、FSL 与 FreeSurfer Surface Runtime 模块化置于 `SlicerDSpace/bundled-runtimes/` 下，首次启动即自动识别，无需手动配置路径或分别下载多个依赖包。内部各模块归档的大小与 SHA-256 校验值登记于 [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json) 与 [`models/runtime_bundles.json`](../../models/runtime_bundles.json)：

| 运行包标识 | 平台 | 归档文件名 | 压缩包大小（字节） | 解压后大小（字节） | SHA-256 校验值 |
| --- | --- | --- | --- | --- | --- |
| `mrtrix`（`3.0.8`，67 个命令） | `darwin-arm64` | `mrtrix-3.0.8-darwin-arm64.tar.gz` | `29,350,539` | — | `2a9b296940e24f64da035cd4698d87e9074133c0243801bc5fff5c3aaf3431c5` |
| `mrtrix`（`3.0.8`，67 个命令，Intel 真机未实测） | `darwin-x86_64` | `mrtrix-3.0.8-darwin-x86_64.tar.gz` | `29,350,563` | — | `d29ffaac97fe0d3fd5a67ffbe0befa8da9ac5d5cf06ff1ad7102737a50ad2d67` |
| `fsl`（`2602.0`，3 个命令） | `darwin-arm64` | `fsl-2602.0-darwin-arm64.tar.gz` | `18,478,979` | — | `2f5bbd4958823f61af1d36263666a3de84feab839e4785293721195083441c8f` |
| `surface`（`0.1.0`，7 个命令） | `darwin-arm64` | `surface-runtime-darwin-arm64.tar.gz` | `38,423,089` | `82,869,079` | `5a735218604f41d81a65b8657f704fa8b60ffb0525cc82edc9c4c9dd3d47ecfc` |

> **第三方软件再分发与许可证合规检查**：所有内置组件均附带上游开源许可证文本（`third_party/LICENSES/`、各运行包 `LICENSE.txt` 及 `manifest.json`），且绝不打包任何个人或机构专属的 FreeSurfer 许可证或激活文件（`activation_key_included: false`）。

---

## 4. 干净环境安装与 3D Slicer 5.10.0 GUI 验收

1. **通过 3D Slicer 5.10.0 Extension Manager 安装统一扩展包验收**：
   - `Testing/Python/slicer_extension_manager_install_smoke.py` 在干净的 3D Slicer 5.10.0（`r34045`，`darwin-arm64`）配置下（清空用户运行包缓存）验证 `slicer.app.extensionsManagerModel().installExtension(...)` 安装 `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`：
     - `passed: true`、`archive_rooted: true`、`archive_clean: true`、`installed: true`、`manager_reports_installed: true`、`registered: true`、`module_loaded: true`、`module_available: true`、`runtime_resources_ready: true`、`installed_imports_ready: true`、`bundled_runtimes_ready: true`（自动识别内置的 `fsl-2602.0-darwin-arm64`、`mrtrix-3.0.8-darwin-arm64`、`surface-runtime-darwin-arm64`）、`self_tests_passed: true`。
2. **干净环境默认原生配置与功能自检**：
   - macOS GUI 默认启用原生运行环境（`preprocessing_execution_backend = "native"`、`use_native_runtime = True`）。
   - `shared/setup_selftest.py`（`run_checks`）全部 **48 项**合成安装自检（Python 核心库、Teem 自检、NumPy/NIfTI/VTK 读写、PyTorch 算子、全部 21 个解剖深度学习模型权重校验、TractCloud 推理自检、PNL 原生掩膜三视角自检、原生 MRtrix3 / FSL / dcm2niix 命令可执行性）**100% 通过**。
   - 仓库全部 **77 个**单元测试文件（`Testing/Python/test_*.py`）**100% 通过**。
3. **3D Slicer 5.10.0 真实 GUI Smoke 测试套件**：
   - `Testing/Python/slicer_extension_install_smoke.py`：**通过**（`exit_code = 0`）
   - `Testing/Python/slicer_environment_button_smoke.py`：**通过**（`exit_code = 0`，`ENV_CHECK_HEARTBEATS = 57`，Qt 主线程心跳不阻塞）
   - `Testing/Python/slicer_gui_operations_smoke.py`：**通过**（`exit_code = 0`，结果加载、场景节点管理、安装队列、失败处理、任务取消与组件销毁全部通过）
   - `Testing/Python/slicer_cancel_background_smoke.py`：**通过**（`exit_code = 0`，macOS 后台进程组取消与 `cancelled` 清单落盘通过）
