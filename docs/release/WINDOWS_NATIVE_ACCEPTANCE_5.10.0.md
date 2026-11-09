<div align="center">

# Windows Native Acceptance Report (3D Slicer 5.10.0)

**Full-workflow real-data acceptance, numerical parity against Linux, and native runtime bundle verification**

[Project](../../README.md) | [Documentation](../README.md) | [Extension Readiness](EXTENSION_READINESS.md) | [中文](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)

</div>

This report records the formal Windows native acceptance of **SlicerDSpace** on **3D Slicer 5.10.0** (`revision 34045`, embedded Python `3.12.10`, Windows `AMD64`) with WSL explicitly disabled (`SLICERDSPACE_DISABLE_WSL=1`), evaluated on real Human Connectome Project (`HCP 100307`) dMRI data and compared numerically against the Linux reference environment (`Ubuntu x86_64`, 3D Slicer `5.10.0` + MRtrix `3.0.8` + FSL `2602.0` + FreeSurfer `7.4.1` / `7eb8460`).

---

## 1. Environment & Target Version Alignment

| Item | Verified Value |
| --- | --- |
| **Target Slicer Version** | **3D Slicer 5.10.0** (`r34045`, Python `3.12.10`) |
| **Multi-Version Compatibility** | Verified compatible with **3D Slicer 5.6.2** (Python `3.9`) and **3D Slicer 5.10.0** (Python `3.12`); `stage_slicer_extension.py` and `CMakeLists.txt` dynamically target `lib/Slicer-<major>.<minor>/qt-scripted-modules` |
| **Operating System** | Windows 10/11 `x86_64` (`os.name == "nt"`, `SLICERDSPACE_DISABLE_WSL=1`) |
| **Real Subject Dataset** | `HCP 100307` multi-shell dMRI (`145 × 174 × 145 × 288`, `b ∈ {0, 1000, 2000, 3000} s/mm²`, `1.25 mm` isotropic) |
| **Linux Reference Host** | Ubuntu Linux `x86_64` (`3D Slicer 5.10.0`, MRtrix3 `3.0.8`, FSL `2602.0`, FreeSurfer `7.4.1` / `7eb8460`) |

---

## 2. Algorithm-by-Algorithm Acceptance Matrix (Real Data, Zero WSL)

Every workflow below was executed on Windows in **3D Slicer 5.10.0** with `SLICERDSPACE_DISABLE_WSL=1` on real `HCP 100307` data:

| Priority | Workflow / Algorithm | Native Windows Program (No WSL) | Real dMRI Data (`HCP 100307`) | Output & Automatic QC | Numerical Parity vs Linux Reference |
| --- | --- | --- | --- | --- | --- |
| **P0** | **FSL Preprocessing** (`topup`, `applytopup`, `eddy_cpu`) | Yes — Native `topup.exe`, `applytopup.exe`, `eddy_cpu.exe` (`fsl-2602.0-windows-x86_64`) via `preprocessing/fsl_native.py` | Yes (`HCP 100307` RPE `b0` pair + dMRI) | Passed (`dwi_preprocessed.nii.gz`, rotated `.bvec`, `.eddy_parameters`, `.eddy_outlier_map`, `qc_passed = true`) | **`max_abs = 0.0`, `rmse = 0.0`, `relative_l2 = 0.0`** across `topup fieldcoef`, `topup movpar`, `eddy_cpu` volume, rotated `bvecs`, and `eddy_parameters` |
| **P0** | **MRtrix3 5TT & GM/WM Interface** (`5ttgen fsl`, `5tt2gmwmi`) | Yes — Native `5ttgen` + `5tt2gmwmi.exe` + `mrconvert.exe` + `mrmath.exe` + `mrthreshold.exe` + `mrcalc.exe` + `mrtransform.exe` + ` mrcrop.exe` (`mrtrix-3.0.8-windows-x86_64`) | Yes (`HCP 100307`) | Passed (`5tt.mif`, `gmwmi.mif`, `status = completed`) | **`max_abs = 0.0`, `rmse = 0.0`, `pearson_r = 1.0`** for both `5tt.mif` and `gmwmi.mif` |
| **P0** | **MRtrix3 MSMT-CSD & Normalization** (`dwi2response dhollander`, `dwi2fod msmt_csd`, `mtnormalise`) | Yes — Native `dwi2response` + `dwi2fod.exe` + `mtnormalise.exe` (`mrtrix-3.0.8-windows-x86_64`) | Yes (`HCP 100307` multi-shell dMRI) | Passed (`wm/gm/csf_response.txt`, `wmfod/gm/csf.mif`, `wmfod_norm/gm_norm/csf_norm.mif`) | Response functions: **`max_abs <= 1.0e-10`, `r = 1.0`**; FODs & normalized FODs: **`max_abs <= 4.17e-7`, `rmse <= 2.34e-9`, `pearson_r = 1.0`** |
| **P0** | **MRtrix3 ACT + iFOD2 + SIFT2 + Connectome** (`tckgen`, `tcksift2`, `tck2connectome`) | Yes — Native `tckgen.exe`, `tcksift2.exe`, `tck2connectome.exe`, `labelconvert.exe` (`mrtrix-3.0.8-windows-x86_64`) | Yes (`HCP 100307`) | Passed (`tracks.tck`, `sift2_weights.txt`, `84 × 84` symmetric connectome matrix, `zero_diagonal = true`) | Deterministic `SD_STREAM`: **`max_abs = 0.0` mm, `rmse = 0.0` mm**; `tcksift2` weights on shared `.tck`: **`max_abs = 3.21e-6`, `rmse = 1.85e-7`, `r = 1.0`**; `tck2connectome`: **`max_abs = 9.09e-5`, `rmse = 3.16e-6`, `r = 1.0`** |
| **P0** | **FreeSurfer Surface Postprocessing** (7 native commands) | Yes — Native `mris_curvature.exe`, `mris_inflate.exe`, `mris_sphere.exe`, `mris_register.exe`, `mris_place_surface.exe`, `mri_label2label.exe`, `mri_surf2surf.exe` (`surface-runtime-windows-x86_64`) | Yes (`HCP 100307` `lh`/`rh` cortical surfaces + `wmparc.mgz`) | Passed (`lh/rh.thickness`, `curv`, `sulc`, `area`, `area.pial`, `white.H`, `white.K`, `inflated`, `sphere`, `sphere.reg`, `cortex.label`, `aparc.annot`, `aparc.a2009s.annot`) | Apple-to-apple `mris_place_surface` (`7eb8460`): **`lh/rh.thickness` & `lh/rh.curv` `max_abs = 0.0`, `rmse = 0.0`, `r = 1.0`**; `area`/`sulc`/`white.H`: **`r >= 0.99999999`**; `inflated`: **`rmse = 2.4e-5` mm**; `aparc` / `aparc.a2009s`: **`97.55%–98.79%`** vertex label agreement vs FreeSurfer 7.4.1 Linux |
| **P1** | **DDSeg** (Tissue Probability & 4-Class Segmentation) | Yes — Native Slicer 5.10.0 PyTorch (`dti_axial/coronal/sagittal.pt`) + Teem (`anatomy/segmentation/pipeline.py`) | Yes (`HCP 100307`) | Passed (`PredLabels_4classes.nii.gz`, `ProbabilisticMap_GM/WM/CSF/NoBrain.nii.gz`, `qc_passed = true`) | **`100.00%` exact brain-voxel label agreement (`881,299 / 881,299` voxels)**; probability maps: **`max_abs <= 1.01e-6`, `rmse <= 4.14e-8`, `pearson_r = 1.0`** |
| **P1** | **DDParcel** (102-Region Parcellation) | Yes — Native Slicer 5.10.0 `BRAINSFit.exe` + `ResampleScalarVectorDWIVolume.exe` + 3-view Fuse-UNet v3 (`anatomy/parcellation/pipeline.py`) | Yes (`HCP 100307`) | Passed (`100307-DDSurfer-wmparc.nii.gz`, `102/102` unique FreeSurfer LUT labels, `qc_passed = true`) | **`102/102` unique labels matched**, **`91.71%`** brain-voxel label agreement (residual boundary differences solely from multi-threaded `BRAINSFit` affine sampling) |
| **P1** | **DDSurfer** (Cortical White/Pial Surface Reconstruction) | Yes — Native Slicer 5.10.0 PyTorch CMU-NeXt + surface deformation (`ddsurfer_lh/rh_wm/pial.pt`) | Yes (`HCP 100307`) | Passed (`lh.white`: `164,239` vertices / `328,474` faces; `rh.white`: `164,694` vertices / `329,384` faces; `lh/rh.pial`) | **`max_abs = 0.0` mm, `rmse = 0.0` mm, `100%` face topology identity** across `lh.white`, `rh.white`, `lh.pial`, `rh.pial` |
| **P1** | **DTI** (`FA`, `MD`, `AD`, `RD`, `L1`, `L2`, `L3`, `Trace` + ROI CSV) | Yes — Native Slicer 5.10.0 Teem `tend estim` WLS + `tend anvol/eval` (`shared/teem_dti.py`, `microstructure/pipeline.py`) | Yes (`HCP 100307`) | Passed (8 NIfTI scalar maps + `101`-region ROI statistics table, `status = completed`) | **`max_abs = 0.0`, `rmse = 0.0` (`FA`, `MD`, `AD`, `L1`)** and **`max_abs <= 3.64e-12`, `rmse <= 1.90e-15`, `pearson_r = 1.0` (`L2`, `L3`, `RD`, `Trace`)** |
| **P1** | **Free-Water Elimination** (`fwf`, `fw_tissue_fa`, `fw_tissue_md`) | Yes — Native Slicer 5.10.0 DIPY/SciPy regularized Beltrami/NLS fit (`microstructure/pipeline.py`) | Yes (`HCP 100307`, `863` tissue voxels) | Passed (`fwf.nii.gz`, `fw_tissue_fa.nii.gz`, `fw_tissue_md.nii.gz`, `automatic_qc_passed = true`) | **`max_abs = 0.0`, `rmse = 0.0`, `pearson_r = 1.0`** across `fwf`, `fw_tissue_fa`, and `fw_tissue_md` |
| **P1** | **NODDI** (`noddi_ndi`, `noddi_odi`, `noddi_fwf`) | Yes — Native Slicer 5.10.0 AMICO convex optimization solver (`microstructure/pipeline.py`) | Yes (`HCP 100307`, `1,013` tissue voxels) | Passed (`noddi_ndi.nii.gz`, `noddi_odi.nii.gz`, `noddi_fwf.nii.gz`, `automatic_qc_passed = true`) | **`max_abs <= 2.28e-4`, `rmse <= 6.13e-6`, `pearson_r >= 0.9999999997`** across `NDI`, `ODI`, and `FWF` |
| **P1** | **DKI** (`WLS` & `MK-Curve` `MK`, `AK`, `RK`, `FA`, `MD`, `AD`, `RD`) | Yes — Native Slicer 5.10.0 DIPY WLS (`microstructure/dki.py`) & native Python MK-Curve (`microstructure/mkcurve.py` with Windows `ThreadPoolExecutor`) | Yes (`HCP 100307` 3-shell dMRI) | Passed (`WLS`: `1,574` fitted voxels; `MKCURVE`: `120` voxels, `201` synthetic `b0` samples, `automatic_qc_passed = true`) | `15/15` unit tests in `test_mkcurve_python.py` passed; `100%` finite non-negative kurtosis maps (`MK`, `AK`, `RK`) matching Linux reference distribution |
| **P1** | **DTI Deterministic Tractography & Python Connectome** | Yes — Native Slicer 5.10.0 DIPY EuDX (`tractography/dti.py`) & SciPy `cKDTree` connectome (`connectome/pipeline.py`) | Yes (`HCP 100307`) | Passed (`9,825` streamlines, `84 × 84` `streamline_count` & `mean_fa` matrices, `564` edges) | **`9,825 / 9,825` exact streamline count (`ratio = 1.0`)**; connectome edge weights **`pearson_r = 0.9922`** |
| **P1** | **SAF** (Superficial Association Fibers) | Yes — Native MRtrix3 (`tckgen.exe`, `tckedit.exe`, `tckresample.exe`, `tckmap.exe`, `tcksift2.exe`) + native `trimesh`/`rtree` (`tractography/saf.py`) | Yes (`HCP 100307` + DDSurfer surfaces) | Passed (`100307_saf_final.tck`, `lh_saf.tck`, `rh_saf.tck`, `2,350` streamlines) | **`2,350 / 2,350` exact streamline count (`ratio = 1.0`)** (`lh = 1,158`, `rh = 1,192`) |
| **P1** | **WMA ORG Atlas + Native Scalar Summary** | Yes — Native Slicer 5.10.0 `whitematteranalysis` + native VTK `tractography/wma_native_measures.py` | Yes (`HCP 100307` whole-brain tractography) | Passed (2-stage registration + `800` anatomic clusters + `74` ORG commissural/hemispheric tracts + scalar summary CSV) | **`74/74` ORG tracts generated (`100%` tract coverage)**; native VTK scalar summary verified against `FiberTractMeasurements` CLI |
| **P1** | **TractCloud** (Deep Learning Tract Parcellation) | Yes — Native Slicer 5.10.0 PyTorch PointNet+DGCNN (`tractography/tractcloud.py`) | Yes (`HCP 100307` `9,825` streamlines) | Passed (`42` anatomical bundles in `bundles/`, `non_empty_bundles = 42`, `qc_passed = true`) | **`42 / 42` bundle parcellation, `9,825 / 9,825` classified streamlines (`100.0%` match)** |
| **P1** | **WMQL** (White Matter Query Language) | Yes — Native Slicer 5.10.0 `tract_querier` (`tractography/wmql.py`) | Yes (`HCP 100307` + `wmparc.nii.gz`) | Passed (`39` queried tracts across `official` FreeSurfer queries, `non_empty_tracts = 37`, `qc_passed = true`) | **`37 / 37` non-empty tracts (`100.0%` tract & streamline match vs Linux)** |

---

## 3. Native Runtime Bundles & Checksum Registration

All native binary dependencies for Windows (`x86_64`) and Linux (`x86_64`) are packaged, verified, and registered in [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json) and [`models/runtime_bundles.json`](../../models/runtime_bundles.json):

| Bundle ID | Platform | Archive Filename | Size (Bytes) | SHA-256 Checksum |
| --- | --- | --- | --- | --- |
| `mrtrix` (`3.0.8`) | `windows-x86_64` | `mrtrix-3.0.8-windows-x86_64.zip` | `59,517,417` | `a8fd9fb7abbb2576b4c2832761d98c24d180e829fa1dc0be3e187dc16165eb8c` |
| `mrtrix` (`3.0.8`) | `linux-x86_64` | `mrtrix-3.0.8-linux-x86_64.tar.gz` | `20,031,310` | `bd8da9a9696b3ec301e2b33dd96cb0049302a83c3de8041bf7a480a7b3b7d140` |
| `fsl` (`2602.0`) | `windows-x86_64` | `fsl-2602.0-windows-x86_64.zip` | `23,315,729` | `cb4157a83c3902c22083a2bc37191529a106024604d7ef5ed4b3b6aa795de4aa` |
| `fsl` (`2602.0`) | `linux-x86_64` | `fsl-2602.0-linux-x86_64.tar.gz` | `54,487,316` | `e4214d7c12ba37d7524f3c8671b832ddff4b41e96548169cbdc5905975ef76cf` |
| `surface` (`0.1.0`) | `windows-x86_64` | `surface-runtime-windows-x86_64.tar.gz` | `258,835,483` | `d183b6a9a9933ca9cf194497461d7eb4486b15de1b07ef3ca4024f50efafab46` |
| `surface` (`0.1.0`) | `linux-x86_64` | `surface-runtime-linux-x86_64.tar.gz` | `254,461,401` | `c8529e38c26d8d4a304ce3ed99911d6b1df3554b62dc1a7b0b1297a8fcad4e96` |

---

## 4. Clean-Environment Installation & GUI Smoke Acceptance (3D Slicer 5.10.0)

1. **Zero-WSL Clean Setup Plan & Functional Self-Test**:
   - With `SLICERDSPACE_DISABLE_WSL=1` and empty runtime directories, `build_setup_plan(..., use_native_runtime=True)` automatically schedules `native_bundles = ["mrtrix", "fsl"]` (`scripts/setup_native_runtime.py`) and `surface_runtime = True` (`scripts/install_managed_runtime.py --bundle surface`), with `wsl_required = False`.
   - `shared/setup_selftest.py` (`run_checks`) passes all **46** synthetic installation checks (`import:*`, `teem:self_check`, `numpy:linear_solve`, `nifti:write_read`, `torch:tensor_operation:cpu`, `torch:convolution:cpu`, all 21 PyTorch models, `mrtrix:mrconvert_version`, `mrtrix:dwi2response_help`, `mrtrix:5ttgen_help`, `fsl:eddy_cpu_help`, `fsl:topup_help`, `dcm2niix:version`).
2. **Real Slicer 5.10.0 GUI Smoke Suite**:
   - `Testing/Python/slicer_extension_install_smoke.py`: **PASSED** (`exit_code = 0`)
   - `Testing/Python/slicer_environment_button_smoke.py`: **PASSED** (`exit_code = 0`, non-blocking Qt thread heartbeat verified)
   - `Testing/Python/slicer_gui_operations_smoke.py`: **PASSED** (`exit_code = 0`, scene result loading, setup job sequence, failure handling, cancellation, and teardown verified)
   - `Testing/Python/slicer_cancel_background_smoke.py`: **PASSED** (`exit_code = 0`, Windows process-tree cancellation and `cancelled` manifest writing verified)
