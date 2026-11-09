<div align="center">

# Capabilities And Platforms

**Install what the selected route needs**

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文](SUPPORT_MATRIX.zh-CN.md)

</div>

Capability checks detect packages, executables and assets. They do not certify
numerical accuracy for every acquisition or platform.

---

## Dependency Levels

| Level | Products | Prerequisites |
| --- | --- | --- |
| Inspect/plan | Input preview, validation, plans | Basic imaging runtime; converter for DICOM/NRRD |
| Minimal DTI | Tensor scalars and colour FA map, optional deterministic tracks | Slicer's own Python (Teem tensor fit, no extension) and Python packages |
| Anatomical models | DDSurfer/DDSeg/DDParcel | PyTorch, matching assets, route-specific preparation |
| Microstructure | DTI/FW/iFW/NODDI/DKI/MK-Curve | Configured numerical packages and eligible acquisition; DTI is default |
| Advanced pathways | CSD/ACT/SIFT2/SAF/fiber bundles | External backend only as selected |
| Connectome | Regional matrices, scene QC | Tractogram/parcellation; weights only for weighted routes |

Install only selected dependencies. Review external-tool and model terms
separately from runtime readiness.

---

## Evidence By Platform

| Check | Linux (`x86_64`) | Windows (`x86_64`) | macOS |
| --- | --- | --- | --- |
| Unified `v0.1.1` Release Package (3D Slicer 5.10.0) | Verified (`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`: extension + modular MRtrix3 `3.0.8` 68 cmds + FSL `2602.0` + Surface Runtime) | Verified (`SlicerDSpace-v0.1.1-Windows-x64.zip`: extension + modular MRtrix3 `3.0.8` 68 cmds + FSL `2602.0` + Surface Runtime) | Not yet packaged / native dependencies unvalidated (do not claim complete native support) |
| Resource/CMake contract | Local evidence; CI configured | Verified (`audit_extension_package.py` pass); CI configured | CI configured; execution evidence needed |
| Real Slicer 5.10.0 GUI & Extension Manager install | Dated local records (`slicer_extension_manager_install_smoke.py` pass) | Verified on 3D Slicer 5.10.0 (`slicer_extension_manager_install_smoke.py` + 4 GUI smoke suites pass with `SLICERDSPACE_DISABLE_WSL=1`) | Not certified |
| Real SDK/CTest/CPack | Linux/Slicer 5.10 candidate verified 2026-10-22 | Unified extension archive verified in Slicer 5.10.0 Extension Manager | Requires real SDK acceptance |
| Scientific workflows (Real Data `HCP 100307`) | Recorded reference routes/data (`145×174×145×288`) | Native Windows verified across all P0/P1 workflows with `SLICERDSPACE_DISABLE_WSL=1`; see [Windows 5.10.0 Acceptance Report](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md) | Not certified |
| Advanced backends | Bundled native MRtrix3 `3.0.8`, FSL `2602.0`, Surface Runtime | Bundled native MRtrix3 `3.0.8`, FSL `2602.0` (`eddy_cpu`/`topup`), Surface Runtime (default native; optional WSL 2) | Tool-specific limitations; native validation pending |

The Linux record includes a real CPack archive and local installation with
isolated settings; see the [official integration record](../release/OFFICIAL_SLICER_VALIDATION_20261022.md).
It does not establish acceptance of later changes, a clean machine, online
store installation or other platforms. Resource checks are not three-platform
Slicer builds. See the [release gate](../release/EXTENSION_READINESS.md).

---

## Feature Routes

**Native verified** indicates real-data execution verified without WSL (`SLICERDSPACE_DISABLE_WSL=1`).
**Native candidate** describes source feasibility where full platform certification is still pending.
WSL is a Linux backend, not native Windows computation; on Windows `v0.1.1`, native execution is the default and WSL 2 is only an optional fallback.

| Feature | Linux (`x86_64`) | Windows (`x86_64`) | macOS |
| --- | --- | --- | --- |
| UI, NIfTI/NRRD, validation, result loading | Verified in Slicer 5.10.0 | **Verified native in 3D Slicer 5.10.0** (Extension Manager install + 4 GUI smoke suites + scene reload) | Native candidate with imaging packages |
| DICOM conversion | Converter required (`dcm2niix`) | **Verified native in 3D Slicer 5.10.0** (`dcm2niix.exe`) | Native converter/Slicer candidate |
| DTI, deterministic tracking | Teem tensor fit in Slicer; DIPY tracking | **Verified native in 3D Slicer 5.10.0** (full-brain multi-shell `HCP 100307` & `HCP 100408` + clinical single-shell `sub-001`) | Same, native candidate |
| DDSeg/DDParcel/DDSurfer native reconstruction | Verified on recorded data/routes with matching models | **Verified native in 3D Slicer 5.10.0** (full-brain `HCP 100307`, `HCP 100408`, and `sub-001`) | Native CPU candidate with usable mask and matching models |
| FW, NODDI | Verified on recorded inputs | **Verified native in 3D Slicer 5.10.0** (full-brain `HCP 100307` FW & NODDI; full-brain `sub-001` FW) | Native Python candidate; architecture-compatible packages |
| DKI/Python MK-Curve | Full-brain WLS and MK-Curve checks recorded; invalid voxels excluded | **Verified native in 3D Slicer 5.10.0** (full-brain WLS on `HCP 100307` & `HCP 100408`; whole-axial-slab + stratified multi-slice MK-Curve) | Native numerical candidate; eligible multi-shell data |
| PNL CNN mask | PyTorch/ANTs route; paired CPU probability/mask checks | Native route in Windows Slicer's Python (SimpleITK + PyTorch), **experimental** | Native route (SimpleITK + PyTorch), **experimental**; no CUDA, Metal unimplemented |
| Cortical postprocessing | Bundled native 7-command Surface Runtime in `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`; [setup](SURFACE_COMPONENTS.md) | **Verified native in 3D Slicer 5.10.0** (7 native `.exe` tools bundled in `surface-runtime-windows-x86_64`; requires user FreeSurfer license) | Existing native FreeSurfer candidate; no prebuilt bundle yet |
| MRtrix/FSL preprocessing | Bundled native MRtrix3/FSL in `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` | **Verified native in 3D Slicer 5.10.0** (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe` in `fsl-2602.0-windows-x86_64`, numerically matched to Linux; WSL 2 optional fallback only) | Native CPU MRtrix/FSL candidate; **no CUDA eddy** |
| CSD/ACT/SIFT2 | Bundled native MRtrix3 `3.0.8` route verified | **Verified native in 3D Slicer 5.10.0** (68 commands in `mrtrix-3.0.8-windows-x86_64`; T1-free ACT via `act_prep.py` `DDSeg + DDParcel -> 5TT -> 5ttcheck + 5tt2gmwmi` and `5ttgen freesurfer`) | Native MRtrix candidate; full acceptance pending |
| Full SAF | Python filter/geometry checks on recorded inputs | **Verified native in 3D Slicer 5.10.0** (`tckgen.exe`, `tckedit.exe`, `tckresample.exe`, `tckmap.exe`, `tcksift2.exe` + native `trimesh`/`rtree`) | POSIX/MRtrix with prepared surfaces; acceptance pending |
| WMQL | Verified (`37/37` non-empty bundles) | **Verified native in 3D Slicer 5.10.0** (native `tract_querier`, `37/37` non-empty bundles matching Linux) | Configured author Python backend candidate |
| WMA | Verified (`74/74` ORG bundles) | **Verified native in 3D Slicer 5.10.0** (`whitematteranalysis` + native VTK scalar stats, `74/74` ORG bundles 100% coverage) | Author shell/display/Slicer CLI requirements; acceptance pending |
| TractCloud | Seeded CPU/CUDA checks on recorded inputs | **Verified native in 3D Slicer 5.10.0** (Slicer PyTorch CPU/CUDA, `42/42` bundles 100% parity with Linux) | External Python CPU candidate; MPS not implemented |
| Python connectome/ROI statistics/VTK previews | Verified on recorded inputs | **Verified native in 3D Slicer 5.10.0** (`84×84` symmetric matrices across multiple subjects) | Native candidate given valid input products |
| Bundled runtime distribution & auto-recovery | `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` (`bundled-runtimes` + `tree_sha256` auto-repair) | **`SlicerDSpace-v0.1.1-Windows-x64.zip`** (`bundled-runtimes` auto-discovered on first start + `tree_sha256` auto-repair if deleted/corrupted) | Not yet packaged as a unified archive |

---

## Practical Limits & Distribution Scope

- **Installation method**: Install `SlicerDSpace-v0.1.1-Windows-x64.zip` or `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` from GitHub Releases via 3D Slicer 5.10.0 `Extensions Manager -> Install Extensions -> Install Extension from File` (not yet listed in the online Slicer Extension Catalog).
- **On-demand model weights & atlases**: The unified platform archive bundles the extension code and native C/C++ runtimes (`MRtrix3 3.0.8`, `FSL 2602.0` diffusion preprocessing, and `FreeSurfer Surface Runtime 0.1.0`). Deep-learning model weights (`DDSeg`, `DDParcel`, `DDSurfer`, `TractCloud`, `PNL`) and large atlases (`ORG`) remain downloaded on demand on first use and cached locally.
- **ACT / 5TT preparation scope**: SlicerDSpace's native T1-free ACT workflow generates 5TT and GMWMI directly from `DDSeg + DDParcel` (`tractography/act_prep.py`, validated by `5ttcheck` and `5tt2gmwmi`) or via `5ttgen freesurfer` on parcellation volumes. Upstream MRtrix3 `5ttgen fsl` requires T1w input plus external full FSL (`bet`, `fast`, `run_first_all`) and is not used by SlicerDSpace or supported by the 3-command FSL diffusion preprocessing bundle (`topup`, `applytopup`, `eddy_cpu`).
- **FreeSurfer license requirement**: The 7 bundled FreeSurfer cortical surface postprocessing commands (`mris_anatomical_stats`, `mris_ca_label`, etc.) do not include any user-specific FreeSurfer license file (`activation_key_included: false`). Users running cortical surface postprocessing must supply their own free FreeSurfer license (`FS_LICENSE`).
- **Mask policy & PyTorch devices**: Existing reviewed native masks are preserved on all platforms without silent algorithm substitution. Auto PyTorch selects CUDA when available and otherwise CPU; **Apple MPS is not implemented/validated**.
- **DKI / MK-Curve scope**: Full-brain DKI (`WLS`) and whole-axial-slab / multi-slice Python `MK-Curve` runs are recorded on multi-shell subjects (`HCP 100307`, `HCP 100408`); valid-voxel screening is not a full-brain MATLAB runtime match across arbitrary clinical protocols.

See the [Windows 5.10.0 Acceptance Report](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md), [Windows Setup Guide](WINDOWS.md), and [macOS Guide](MACOS.md).

---

## Upstream Boundaries

- **FSL**: Upstream FSL officially distributes full Linux/macOS suites and recommends WSL on Windows ([FSL installation](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html), [FSL Windows](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)). SlicerDSpace `v0.1.1` bundles native Windows x64 builds of the 3 diffusion preprocessing executables (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe`) under the Oxford University Innovation Academic Licence; full FSL tools outside diffusion preprocessing (`bet`, `fast`, `FIRST`, GPU `eddy_cuda`) are not bundled.
- **FreeSurfer**: Upstream FreeSurfer officially distributes Linux/macOS suites ([FreeSurfer requirements](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements)). SlicerDSpace `v0.1.1` bundles native Windows x64 builds of the 7 cortical surface postprocessing commands required by `DDSurfer`; full `recon-all` is not bundled and a user FreeSurfer license is still required.
- **PyTorch MPS**: PyTorch supports Apple MPS ([PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)), but SlicerDSpace does not yet enable or certify MPS execution.
