<div align="center">

# macOS Native Acceptance Report (3D Slicer 5.10.0 — Apple Silicon `arm64`)

**Full-workflow real-data acceptance, numerical parity against Linux, and native runtime bundle verification**

[Project](../../README.md) | [Documentation](../README.md) | [Extension Readiness](EXTENSION_READINESS.md) | [中文](MACOS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)

</div>

This report records the formal macOS native acceptance of **SlicerDSpace** on **3D Slicer 5.10.0** (`revision 34045`, embedded Python `3.12.10`, Apple Silicon `darwin-arm64`, macOS `15.7.1`) without Docker, WSL, or manual C/C++ compilation, evaluated on real Human Connectome Project (`HCP 100307`) dMRI data and compared numerically against the Linux reference environment (`Ubuntu x86_64`, 3D Slicer `5.10.0` + MRtrix `3.0.8` + FSL `2602.0` + FreeSurfer `7.4.1` / `7eb8460`).

> **Hardware Scope**: All real-data execution, numerical parity comparisons, and GUI smoke tests in this report were verified on physical **Apple Silicon (`arm64`)** hardware. While `mrtrix-3.0.8-darwin-x86_64.tar.gz` is registered in `models/native_runtime_bundles.json`, **physical Intel (`x86_64`) Mac hardware has not been tested** and remains marked as **unverified**.

---

## 1. Environment & Target Version Alignment

| Item | Verified Value |
| --- | --- |
| **Target Slicer Version** | **3D Slicer 5.10.0** (`r34045`, Python `3.12.10`, `darwin-arm64`) |
| **Multi-Version Compatibility** | Verified compatible with **3D Slicer 5.6.2** (Python `3.9`) and **3D Slicer 5.10.0** (Python `3.12`); `stage_slicer_extension.py` and `CMakeLists.txt` dynamically target `lib/Slicer-<major>.<minor>/qt-scripted-modules` |
| **Operating System & Hardware** | macOS `15.7.1` (`24G231`, Apple Silicon `arm64`, 8 cores, 16 GB RAM); physical Intel (`x86_64`) Mac hardware is **unverified** |
| **Real Subject Dataset** | `HCP 100307` multi-shell dMRI (`145 × 174 × 145 × 288`, `b ∈ {0, 1000, 2000, 3000} s/mm²`, `1.25 mm` isotropic, `740,916` brain voxels) |
| **Linux Reference Host** | Ubuntu Linux `x86_64` (`3D Slicer 5.10.0`, MRtrix3 `3.0.8`, FSL `2602.0`, FreeSurfer `7.4.1` / `7eb8460`) |

---

## 2. Algorithm-by-Algorithm Acceptance Matrix (Real Data, Native macOS `arm64`)

Every workflow below was executed natively on macOS (`darwin-arm64`) in **3D Slicer 5.10.0** on real `HCP 100307` data:

| Priority | Workflow / Algorithm | Native macOS Program (No Docker / No Manual Compilation) | Real dMRI Data (`HCP 100307`) | Output & Automatic QC | Numerical Parity vs Linux Reference |
| --- | --- | --- | --- | --- | --- |
| **P0** | **FSL & DIPY Preprocessing** (`DIPY MP-PCA`, `DIPY Gibbs`, `topup`, `applytopup`, `eddy_cpu`) | Yes — Native `topup`, `applytopup`, `eddy_cpu` (`fsl-2602.0-darwin-arm64`) + `mrconvert`/`dwiextract`/`mrcat`/`dwi2mask` (`mrtrix-3.0.8-darwin-arm64`) via `preprocessing/fsl_native.py` | Yes (`HCP 100307` RPE `b0` pair + dMRI) | Passed (`100307_dwi_final.nii.gz`, rotated `100307_dwi_final.bvec`, `100307_eddy_parameters.txt`, `100307_eddy_outlier_map.txt`, `status = completed` in `70.46 s`) | `MP-PCA`: **`r = 1.0` (`rmse = 2.36e-4`)**; `Gibbs`: **`r = 1.0` (`rmse = 0.0309`)**; `topup fieldcoef`: **`r = 0.9999988` (`rmse = 0.0063`)**; `topup movpar`: **`r = 0.9999980` (`rmse = 1.32e-5`)**; `applytopup`: **`r = 0.999999995`**; rotated `bvec`: **`r = 0.999995` (`rmse = 0.00177`)**; `eddy_movement_rms`: **`r = 0.99998`**; `eddy_parameters`: **`r = 0.9946`**; `100307_dwi_final.nii.gz`: **`r = 0.9926`** |
| **P0** | **MRtrix3 5TT & GM/WM Interface** (`act_prep.py`, `5ttcheck`, `5tt2gmwmi`, `5ttgen freesurfer`) | Yes — Native `5ttcheck`, `5tt2gmwmi`, `5ttgen`, `mrconvert`, `mrmath`, `mrthreshold`, `mrcalc`, `mrtransform`, `mrcrop`, `labelconvert` (`mrtrix-3.0.8-darwin-arm64`, **67 commands** + bundled `FreeSurferColorLUT.txt`) | Yes (`HCP 100307`) | Passed (`100307_5tt.nii.gz`, `100307_gmwmi.nii.gz`, `5ttcheck` passed, `status = completed`) | **`max_abs = 0.0`, `rmse = 0.0`, `pearson_r = 1.0`** for `5tt` and `gmwmi` on shared tissue segmentation |
| **P0** | **MRtrix3 MSMT-CSD & Normalization** (`dwi2response dhollander`, `dwi2fod msmt_csd`, `mtnormalise`) | Yes — Native `dwi2response`, `dwi2fod`, `mtnormalise` (`mrtrix-3.0.8-darwin-arm64`) | Yes (`HCP 100307` multi-shell dMRI) | Passed (`wm/gm/csf_response.txt`, `wmfod/gm/csf.mif`, `wmfod_norm/gm_norm/csf_norm.mif`, `status = completed`) | Response functions: **`max_abs <= 1.0e-10`, `r = 1.0`**; normalized FODs: **`pearson_r = 1.0`** |
| **P0** | **MRtrix3 ACT + iFOD2 + SIFT2 + Connectome** (`tckgen`, `tcksift2`, `tck2connectome`) | Yes — Native `tckgen`, `tcksift2`, `tck2connectome`, `labelconvert`, `tckedit`, `tckresample`, `tckmap` (`mrtrix-3.0.8-darwin-arm64`) | Yes (`HCP 100307`, `100,000` ACT streamlines) | Passed (`100307_csd_act.tck`, `.trk`, `.vtk`, `sift2_weights.txt`, `sift2_mu.txt`, `84 × 84` symmetric `streamline_count` & `sift2_weighted` connectome matrices, `status = completed`) | `84 × 84` connectome node set **`100%` identical** (`zero_diagonal = true`, symmetric); `sift2_mu` within `1.2%` across stochastic `iFOD2` runs |
| **P0** | **FreeSurfer Surface Postprocessing** (7 native `arm64` commands) | Yes — Native `mris_curvature`, `mris_inflate`, `mris_sphere`, `mris_register`, `mris_ca_label`, `mris_anatomical_stats`, `mri_surf2surf` (`surface-runtime-darwin-arm64`) | Yes (`HCP 100307` `lh`/`rh` cortical surfaces + `wmparc.mgz`) | Passed (`lh/rh.thickness`, `curv`, `sulc`, `area`, `area.pial`, `white.H/K`, `pial.H/K`, `inflated`, `sphere`, `sphere.reg`, `aparc.annot`, `aparc.a2009s.annot`, `lh/rh.aparc*.stats`, `fsaverage` mapped overlays in `2142.7 s`, `automatic_qc_passed = true`) | **`lh/rh.thickness`, `curv`, `sulc`, `area`, `area.pial`, `white.H/K`, `pial.H/K`, `inflated`: `pearson_r >= 0.9999999`**; `aparc` / `aparc.a2009s`: **`98.1%–99.2%`** vertex label agreement vs Linux |
| **P1** | **DDSeg** (Tissue Probability & 4-Class Segmentation) | Yes — Native Slicer 5.10.0 PyTorch (`dti_axial/coronal/sagittal.pt`) + Teem (`anatomy/segmentation/pipeline.py`) | Yes (`HCP 100307`, `740,916` brain voxels) | Passed (`SegmentationMap_GMWMCSF.nii.gz`, `ProbabilisticMap_GM/WM/CSF.nii.gz` in `272.4 s`, `status = completed`) | **`99.8900%` (`740,101 / 740,916` brain voxels)** and **`99.9777%` all-voxel** exact label agreement; probability maps: **`rmse <= 0.00222`, `pearson_r >= 0.999985`** |
| **P1** | **DDParcel** (101-Region Parcellation) | Yes — Native Slicer 5.10.0 `BRAINSFit` + `ResampleScalarVectorDWIVolume` + 3-view Fuse-UNet v3 (`anatomy/parcellation/pipeline.py`) | Yes (`HCP 100307`) | Passed (`100307-DDSurfer-wmparc.nii.gz`, `101 / 101` non-zero FreeSurfer LUT labels in `3603.4 s`, `status = completed`) | **`101 / 101` exact label set match (`exact_label_set_match = true`)**, **`92.8852%` brain-voxel (`98.5589%` all-voxel)** agreement, `DDSeg WM vs DDParcel WM Dice = 0.860` |
| **P1** | **DDSurfer** (Cortical White/Pial Surface Reconstruction) | Yes — Native Slicer 5.10.0 PyTorch CMU-NeXt + surface deformation (`ddsurfer_lh/rh_wm/pial.pt`) | Yes (`HCP 100307`) | Passed (`lh.white` & `lh.pial`: `164,239` vertices / `328,474` faces; `rh.white` & `rh.pial`: `164,694` vertices / `329,384` faces in `484.3 s`, `automatic_qc_passed = true`) | **`100%` face topology identity (`faces_identical = true`, `watertight = true`, `euler_number = 2`)**, mean vertex displacement `<= 0.14 mm` (from upstream `DDParcel` `BRAINSFit` affine thread sampling) |
| **P1** | **DTI** (`FA`, `MD`, `AD`, `RD`, `L1`, `L2`, `L3`, `Trace` + ROI CSV) | Yes — Native Slicer 5.10.0 Teem `tend estim` WLS + `tend anvol/eval` (`shared/teem_dti.py`, `microstructure/pipeline.py`) | Yes (`HCP 100307`, full `740,916` brain voxels) | Passed (8 NIfTI scalar maps + `101`-region ROI statistics table with `144` columns, `status = completed`) | **`max_abs = 0.0`, `rmse = 0.0`, `pearson_r = 1.0`** across `FA`, `AD`, `RD`, `L1`, `L2`, `L3`, `Trace`; `MD`: **`max_abs = 2.33e-10`, `rmse = 2.00e-12`, `pearson_r = 1.0`** |
| **P1** | **Free-Water Elimination** (`fraction`, `FA`, `MD`) | Yes — Native Slicer 5.10.0 DIPY/SciPy regularized Beltrami/NLS fit (`microstructure/pipeline.py`) | Yes (`HCP 100307`, full `740,916` brain voxels in `319.12 s`) | Passed (`100307_fw_fraction.nii.gz`, `100307_fw_FA.nii.gz`, `100307_fw_MD.nii.gz`, `status = completed`) | `fraction` & `MD`: **`max_abs = 0.0`, `rmse = 0.0`, `pearson_r = 1.0`**; `FA`: **`max_abs = 5.96e-8`, `rmse = 7.22e-11`, `pearson_r = 1.0`** |
| **P1** | **NODDI** (`NDI`, `ODI`, `IsoVF`, `RMSE`) | Yes — Native Slicer 5.10.0 AMICO convex optimization solver (`microstructure/pipeline.py`) | Yes (`HCP 100307`, full `740,916` brain voxels in `511.54 s`) | Passed (`100307_noddi_NDI.nii.gz`, `ODI.nii.gz`, `IsoVF.nii.gz`, `RMSE.nii.gz`, `status = completed`) | `NDI`: **`rmse = 1.52e-4`, `r = 0.99999978`**; `ODI`: **`rmse = 2.38e-4`, `r = 0.99999945`**; `IsoVF`: **`rmse = 5.75e-5`, `r = 0.99999997`**; `RMSE`: **`rmse = 2.73e-5`, `r = 0.99999999999`** |
| **P1** | **DKI** (`WLS` & `MK-Curve` `MK`, `AK`, `RK`, `KFA`, `MKT`, `RTK`) | Yes — Native Slicer 5.10.0 DIPY WLS (`microstructure/dki.py`) & native Python MK-Curve (`microstructure/mkcurve.py`) | Yes (`HCP 100307`, full `740,916` brain voxels for `WLS` in `1473.13 s`; `1,500` multi-slice brain voxels for `MKCURVE` in `125.01 s`) | Passed (`WLS`: `735,319` valid voxels + 42 DKI ROI columns across 101 regions; `MKCURVE`: `1,483` valid voxels, `202` synthetic `b0` grid samples, `status = completed`) | `WLS` (`MK`, `AK`, `RK`, `KFA`, `MKT`, `RTK`): **`max_abs <= 2.38e-7`, `rmse <= 7.45e-10`, `pearson_r >= 0.999999999999996`**; `MKCURVE`: **`pearson_r = 0.9919–0.9999`** (`zero_mk_b0`: `r = 0.9999`, `corrected_fa`: `r = 0.9991`, `corrected_md/ad/rd`: `r = 0.9993`, `corrected_mk`: `r = 0.9957`, `corrected_rk`: `r = 0.9944`, `corrected_ak`: `r = 0.9919`) |
| **P1** | **DTI Deterministic & UKF Tractography + Python Connectome** | Yes — Native Slicer 5.10.0 DIPY EuDX (`tractography/dti.py`), `UKFTractography` CLI (`tractography/ukf.py`), & SciPy `cKDTree` connectome (`connectome/pipeline.py`) | Yes (`HCP 100307`) | Passed (`dti_det` in `137.35 s`; `UKF` in `151.62 s` with `22,113` streamlines, `automatic_qc_passed = true`; Python `84 × 84` `streamline_count` & `sift2_weighted` connectomes) | `UKF`: `effective_mask` SHA256 **`100%` identical** (`ca500368...`), streamline count **`22,113` vs `22,023` (`ratio = 1.0041`)**, median length **`34.487` vs `34.490` mm**, `points_inside_brain_fraction = 1.0` |
| **P1** | **SAF** (Superficial Association Fibers) | Yes — Native MRtrix3 (`tckgen`, `tckedit`, `tckresample`, `tckmap`, `tcksift2`) + native `trimesh`/`rtree` (`tractography/saf.py`) | Yes (`HCP 100307` + DDSurfer surfaces) | Passed (`SAF.tck`, `SAF.vtk`, `lh/rh` SAF surfaces & filtering in `273.42 s`, `12,745` streamlines, `status = completed`) | Full 8-step SAF pipeline verified end-to-end with `100%` valid cortical surface intersection and streamline filtering |
| **P1** | **WMA ORG Atlas + Native Scalar Summary** | Yes — Native Slicer 5.10.0 `whitematteranalysis` + native VTK `tractography/wma_native_measures.py` | Yes (`HCP 100307` whole-brain UKF tractography) | Passed (2-stage registration + `800` anatomic clusters + `74/74` ORG commissural/hemispheric tracts + 4 diffusion measurement CSVs, `automatic_qc_passed = true`, `all_tract_sampling_passed = true`) | **`74 / 74` ORG tracts generated (`100%` tract coverage, `excluded_from_quantification = []`)** + native VTK point scalar summary CSVs verified |
| **P1** | **TractCloud** (Deep Learning Tract Parcellation) | Yes — Native Slicer 5.10.0 PyTorch PointNet+DGCNN (`tractography/tractcloud.py`) | Yes (`HCP 100307` whole-brain tractography) | Passed (`42` anatomical VTP bundles + `6` `.mrb` scene bundles in `322.77 s`, `automatic_qc_passed = true`) | **`42 / 42` anatomical bundles generated (`100%` bundle coverage)** |
| **P1** | **WMQL** (White Matter Query Language) | Yes — Native Slicer 5.10.0 `tract_querier` (`tractography/wmql.py`) | Yes (`HCP 100307` + `wmparc.nii.gz`) | Passed (`39` queried tracts across `official` FreeSurfer queries in `8.01 s`, `non_empty_tracts = 37`, `execution_passed = true`, `output_integrity_passed = true`) | **`37 / 37` non-empty tracts (`100.0%` tract set match vs Linux)** |

---

## 3. Native Runtime Bundles & Checksum Registration

In `v0.1.1`, each verified platform is distributed as a **single user-facing installer package** (`SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, and `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`) that embeds the modular native runtime bundles inside `SlicerDSpace/bundled-runtimes/`. Internally, all native binary dependencies are registered in [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json) and [`models/runtime_bundles.json`](../../models/runtime_bundles.json):

| Bundle ID | Platform | Archive Filename | Compressed Size (Bytes) | Unpacked Size (Bytes) | SHA-256 Checksum |
| --- | --- | --- | --- | --- | --- |
| `mrtrix` (`3.0.8`, 67 commands) | `darwin-arm64` | `mrtrix-3.0.8-darwin-arm64.tar.gz` | `29,350,539` | — | `2a9b296940e24f64da035cd4698d87e9074133c0243801bc5fff5c3aaf3431c5` |
| `mrtrix` (`3.0.8`, 67 commands, unverified on Intel hardware) | `darwin-x86_64` | `mrtrix-3.0.8-darwin-x86_64.tar.gz` | `29,350,563` | — | `d29ffaac97fe0d3fd5a67ffbe0befa8da9ac5d5cf06ff1ad7102737a50ad2d67` |
| `fsl` (`2602.0`, 3 commands) | `darwin-arm64` | `fsl-2602.0-darwin-arm64.tar.gz` | `18,478,979` | — | `2f5bbd4958823f61af1d36263666a3de84feab839e4785293721195083441c8f` |
| `surface` (`0.1.0`, 7 commands) | `darwin-arm64` | `surface-runtime-darwin-arm64.tar.gz` | `38,423,089` | `82,869,079` | `5a735218604f41d81a65b8657f704fa8b60ffb0525cc82edc9c4c9dd3d47ecfc` |

> **Third-Party Redistribution & License Hygiene**: All bundled components include their upstream license notices (`third_party/LICENSES/`, `LICENSE.txt`, and `manifest.json` metadata). No user-specific FreeSurfer license or activation file is ever bundled (`activation_key_included: false`).

---

## 4. Clean-Environment Installation & GUI Smoke Acceptance (3D Slicer 5.10.0)

1. **Unified Package Installation via Slicer 5.10.0 Extension Manager**:
   - `Testing/Python/slicer_extension_manager_install_smoke.py` verified installing `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` via `slicer.app.extensionsManagerModel().installExtension(...)` in a clean Slicer 5.10.0 (`r34045`, `darwin-arm64`) profile with an empty user runtime cache:
     - `passed: true`, `archive_rooted: true`, `archive_clean: true`, `installed: true`, `manager_reports_installed: true`, `registered: true`, `module_loaded: true`, `module_available: true`, `runtime_resources_ready: true`, `installed_imports_ready: true`, `bundled_runtimes_ready: true` (`fsl-2602.0-darwin-arm64`, `mrtrix-3.0.8-darwin-arm64`, `surface-runtime-darwin-arm64`), `self_tests_passed: true`.
   - On first run, `find_installed_native_runtime("mrtrix" | "fsl")` and `resolve_surface_runtime()` automatically discover `SlicerDSpace/bundled-runtimes/...` without manual path configuration or separate runtime archive downloads, and automatically repair corrupted files using `tree_sha256`.
2. **Clean Setup Plan & Functional Self-Test**:
   - The macOS GUI defaults to `preprocessing_execution_backend = "native"` and `use_native_runtime = True`.
   - `shared/setup_selftest.py` (`run_checks`) passes all **48** synthetic installation checks (`import:*`, `teem:self_check`, `numpy:linear_solve`, `nifti:write_read`, `vtk:streamline_write_read`, `torch:tensor_operation:cpu`, `torch:convolution:cpu`, all 21 PyTorch models, `tractcloud:load_and_synthetic_forward`, `pnl_native_mask:*`, `mrtrix:mrconvert_version`, `mrtrix:dwi2response_help`, `mrtrix:5ttgen_help`, `fsl:eddy_cpu_help`, `fsl:topup_help`, `dcm2niix:version`).
   - All **77** unit test files in `Testing/Python/test_*.py` pass (`100%` pass rate).
3. **Real Slicer 5.10.0 GUI Smoke Suite**:
   - `Testing/Python/slicer_extension_install_smoke.py`: **PASSED** (`exit_code = 0`)
   - `Testing/Python/slicer_environment_button_smoke.py`: **PASSED** (`exit_code = 0`, `ENV_CHECK_HEARTBEATS = 57`, non-blocking Qt thread heartbeat verified)
   - `Testing/Python/slicer_gui_operations_smoke.py`: **PASSED** (`exit_code = 0`, scene result loading, setup job sequence, failure handling, cancellation, and teardown verified)
   - `Testing/Python/slicer_cancel_background_smoke.py`: **PASSED** (`exit_code = 0`, macOS process-group cancellation and `cancelled` manifest writing verified)
