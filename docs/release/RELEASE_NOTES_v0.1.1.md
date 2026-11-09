<div align="center">

# SlicerDSpace v0.1.1 Release Notes

**Unified Platform Installers, Native Windows Default, Expanded 68-Command MRtrix3 Runtime, and Full-Volume Acceptance on 3D Slicer 5.10.0**

[Project](../../README.md) | [Windows 5.10.0 Acceptance](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md) | [Windows Setup](../setup/WINDOWS.md) | [Support Matrix](../setup/SUPPORT_MATRIX.md) | [中文验收报告](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.zh-CN.md)

</div>

---

## 1. Unified Single-File Release Packages (One User-Facing Installer per Platform)

In `v0.1.1`, each supported desktop platform provides **one complete installer archive** on the GitHub Release page so users no longer have to download and manage separate extension, MRtrix3, FSL, and FreeSurfer Surface Runtime files:

- **Windows (`x86_64`)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
- **Linux (`x86_64`)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`

> **macOS Status**: Native macOS scientific dependencies have not yet completed full end-to-end validation; `v0.1.1` does **not** claim full native macOS release support.

### Modular Internal Runtime Architecture (`SlicerDSpace/bundled-runtimes/`)

While distributed as a single archive per platform, internal runtime management remains strictly modular under `SlicerDSpace/bundled-runtimes/`:

| Component | Windows (`x86_64`) Subdirectory | Linux (`x86_64`) Subdirectory | Contents |
| --- | --- | --- | --- |
| **MRtrix3 (`3.0.8`)** | `mrtrix-3.0.8-windows-x86_64` | `mrtrix-3.0.8-linux-x86_64` | **68 commands** (`mrconvert`, `dwi2fod`, `mtnormalise`, `tckgen`, `tckedit`, `tckresample`, `tckmap`, `tcksift2`, `tck2connectome`, `5tt2gmwmi`, `5ttcheck`, `labelconvert`, `mrcrop`, `mrgrid`, `mrresize`, `dwi2response`, `5ttgen`, `dwifslpreproc`, etc.) + `share/mrtrix3/_5ttgen/FreeSurferColorLUT.txt` |
| **FSL (`2602.0`)** | `fsl-2602.0-windows-x86_64` | `fsl-2602.0-linux-x86_64` | **3 commands** (`topup`, `applytopup`, `eddy_cpu`) + `etc/flirtsch/b02b0.cnf` (`b02b0_1.cnf`, `b02b0_2.cnf`, `b02b0_4.cnf`) |
| **FreeSurfer Surface Runtime (`0.1.0`)** | `surface-runtime-windows-x86_64` | `surface-runtime-linux-x86_64` | **7 commands** (`mris_curvature`, `mris_inflate`, `mris_sphere`, `mris_register`, `mris_place_surface`, `mri_label2label`, `mri_surf2surf`) + `average/` spherical atlas templates & `FreeSurferColorLUT.txt` |

- **3D Slicer 5.10.0 Extension Manager Compatibility**: Install directly via **View > Extensions Manager > Install Extensions > Install Extension from File**.
- **Automatic First-Run Discovery**: `shared/native_runtimes.py` (`find_installed_native_runtime`) and `shared/surface_runtime.py` (`resolve_surface_runtime`) automatically discover `SlicerDSpace/bundled-runtimes/` on first launch without manual path configuration.
- **On-Demand Model & Atlas Downloads**: Deep-learning weights (`DDSeg`, `DDParcel`, `DDSurfer`, `TractCloud`, `PNL`) and `WMA` atlases remain downloaded on demand via **Prepare selected analysis environment**.
- **Third-Party License Compliance**: All bundled runtimes include their upstream license notices (`third_party/LICENSES/`, `LICENSE.txt`, and `manifest.json`). **No user-specific FreeSurfer license or activation file is bundled** (`activation_key_included: false`).

---

## 2. Fixes for `v0.1.0` Issues

1. **Windows Native Runtime by Default (Zero WSL Requirement)**:
   - Updated [`shared/module_widget.py`](../../shared/module_widget.py), [`shared/ui_run_controller.py`](../../shared/ui_run_controller.py), [`shared/module_logic.py`](../../shared/module_logic.py), [`shared/ui_selected_setup.py`](../../shared/ui_selected_setup.py), and [`shared/setup_components.py`](../../shared/setup_components.py) so Windows GUI and CLI sessions default to `preprocessing_execution_backend = "native"` and `use_native_runtime = True`.
2. **Expanded MRtrix3 Runtime (`34 -> 68` Commands) & Complete `5ttgen` Dependencies**:
   - Added 34 missing MRtrix3 C++ binaries to [`scripts/native/prepare_mrtrix_candidate.py`](../../scripts/native/prepare_mrtrix_candidate.py) and [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json), including **`tckedit`**, **`tckresample`**, **`tckmap`**, **`mrcrop`**, `5ttcheck`, `mrgrid`, `mrresize`, `mrfilter`, `mrstats`, `mrinfo`, `mrhistmatch`, `mrdump`, `maskfilter`, `label2colour`, `label2mesh`, `labelstats`, `tckconvert`, `tckdfc`, `tckinfo`, `tcknormalise`, `tcksample`, `tcksift`, `tckstats`, `tcktransform`, `connectome2tck`, ` connectomeedit`, `connectomestats`, `fixel2peaks`, `fixel2sh`, `fixel2tsf`, `fixel2voxel`, `fod2dec`, `fod2fixel`, and `peaks2amp`.
   - Bundled `FreeSurferColorLUT.txt` into `share/mrtrix3/_5ttgen/FreeSurferColorLUT.txt` so `5ttgen freesurfer` runs out of the box without requiring `FREESURFER_HOME`.
3. **Self-Contained Algorithm Command Resolution**:
   - Updated [`shared/external_process.py`](../../shared/external_process.py) (`resolve_external_command`), [`tractography/csd.py`](../../tractography/csd.py), [`tractography/act_prep.py`](../../tractography/act_prep.py), [`connectome/pipeline.py`](../../connectome/pipeline.py), and [`tractography/saf_backend/Support/sys_exec.py`](../../tractography/saf_backend/Support/sys_exec.py) so MRtrix3 and FSL commands automatically resolve from the bundled/managed runtime even when `mrtrix_bin_dir` is not explicitly passed or not on the system `PATH`.
4. **Updated Checksums, File Sizes, and Documentation**:
   - Synchronized compressed and unpacked byte sizes and SHA-256 checksums across [`models/native_runtime_bundles.json`](../../models/native_runtime_bundles.json), [`docs/setup/WINDOWS.md`](../setup/WINDOWS.md), [`docs/setup/SUPPORT_MATRIX.md`](../setup/SUPPORT_MATRIX.md), and [`docs/release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md`](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md).

---

## 3. Modular Runtime Bundle Checksums (`v0.1.1`)

| Bundle ID | Platform | Archive Filename | Compressed Size (Bytes) | Unpacked Size (Bytes) | SHA-256 Checksum |
| --- | --- | --- | --- | --- | --- |
| `mrtrix` (`3.0.8`, 68 commands) | `windows-x86_64` | `mrtrix-3.0.8-windows-x86_64.zip` | `30,968,289` | `59,517,417` | `a0d5c3f12e6e54a914bb8690edf6b795ae432b82a1e99d741094f8460db8a20a` |
| `mrtrix` (`3.0.8`, 68 commands) | `linux-x86_64` | `mrtrix-3.0.8-linux-x86_64.tar.gz` | `27,145,376` | — | `c5b4ad88c20c569638ca9d49779f439a156247a417dc8f15d33cee0543aa5c7b` |
| `fsl` (`2602.0`, 3 commands) | `windows-x86_64` | `fsl-2602.0-windows-x86_64.zip` | `16,331,282` | `23,315,729` | `1a50663922099c3207e02a45f3416a6f4093fc28c982a123e8451f94f005816f` |
| `fsl` (`2602.0`, 3 commands) | `linux-x86_64` | `fsl-2602.0-linux-x86_64.tar.gz` | `42,914,629` | `54,487,316` | `327d92c8be1f09ebb9d22357de390b0ebfecd72ae5a917875fa48b3e6ad200ab` |
| `surface` (`0.1.0`, 7 commands) | `windows-x86_64` | `surface-runtime-windows-x86_64.tar.gz` | `66,211,294` | `258,835,483` | `bbc97ea1c89117713bf36db2e4f9dd0b23ad4b69b9e27cba1b7050c0bc71f742` |
| `surface` (`0.1.0`, 7 commands) | `linux-x86_64` | `surface-runtime-linux-x86_64.tar.gz` | `40,995,581` | `254,461,401` | `9e6005a114c7d1ff58f247ee2e8977f9eb80f48f525172e15d7e7174ab30fb10` |

---

## 4. Clean-Environment & Full-Volume Real-Data Acceptance Summary

- **Clean Slicer 5.10.0 Extension Manager Installation (`slicer_extension_manager_install_smoke.py`)**: Installed `SlicerDSpace-v0.1.1-Windows-x64.zip` into a clean Slicer 5.10.0 (`r34045`) profile with `SLICERDSPACE_DISABLE_WSL=1` and empty user runtime cache; verified `bundled_runtimes_ready = true`, **46/46** functional self-tests passed, and **4/4** GUI smoke suites passed (`exit_code = 0`).
- **Full-Volume Real-Data Microstructure (`HCP 100307`, `145 × 174 × 145 × 288`, `881,299` Brain Voxels)**:
  - **DTI**: Full `881,299`-voxel brain volume + 101-region ROI CSV (`FA`/`MD`/`AD`/`L1` `max_abs = 0.0` vs Linux).
  - **Free-Water**: Full `881,299`-voxel brain volume (`881,042` nonzero `fwf` voxels, `846,558` tissue `FA`/`MD` voxels in `710.20 s`, `max_abs = 0.0` vs Linux).
  - **NODDI**: Full `881,299`-voxel brain volume (`881,299` `ODI`, `873,290` `NDI`, `516,440` `IsoVF` voxels in `536.86 s`, `pearson_r >= 0.9999999997` vs Linux).
  - **DKI**: Full `881,299`-voxel brain volume `WLS` fit (`792,249` valid voxels in `522.36 s`, `automatic_qc_passed = true`) + multi-slice whole-brain subsampled `MK-Curve` (`1,500` brain voxels, `201` synthetic `b0` grid points per voxel in `101.36 s`, `automatic_qc_passed = true`).
- **All Other Native Workflows Verified on Real `HCP 100307` Data**:
  - FSL `topup`/`applytopup`/`eddy_cpu`, MRtrix3 `5ttgen fsl` & `5ttgen freesurfer`, `5tt2gmwmi`, `mrcrop`, `dwi2response dhollander`, `dwi2fod msmt_csd`, `mtnormalise`, `tckgen`, `tckedit`, `tckresample`, `tckmap`, `tcksift2`, `tck2connectome`, FreeSurfer 7-command surface postprocessing, `DDSeg`, `DDParcel`, `DDSurfer`, `SAF`, `WMA ORG` (74/74 tracts), `TractCloud` (42/42 bundles), and `WMQL` (37/37 non-empty tracts).
