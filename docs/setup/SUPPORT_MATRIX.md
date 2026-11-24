<div align="center">

# Capabilities And Platforms

**Install what the selected route needs**

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文](SUPPORT_MATRIX.zh-CN.md)

</div>

Capability checks detect packages, executables and assets. They do not certify
numerical accuracy for every acquisition or platform.

**Supported 3D Slicer versions:** 5.10 to 5.12.4 (reference version 5.10.0).
3D Slicer 5.8 and 5.9 are not supported: Slicer 5.8 embeds Python 3.9, and the
bundled Python stack (`dipy 1.12.1`, `matplotlib 3.11.2`, `networkx 3.7`) needs Python 3.11 or newer.

**macOS (`arm64`)** means an Apple Silicon host. The macOS 3D Slicer release is an
`x86_64` build that runs under Rosetta 2. The bundled CLI runtimes are native `arm64`.

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

| Check | Linux (`x86_64`) | Windows (`x86_64`) | macOS (`arm64` verified; Intel `x86_64` unverified) |
| --- | --- | --- | --- |
| Unified `v0.1.1` Release Package (3D Slicer 5.10.0) | Verified (`SlicerDSpace-v0.1.1-Linux-x64.tar.gz`: extension + modular MRtrix3 `3.0.8` 68 cmds + FSL `2602.0` + Surface Runtime) | Verified (`SlicerDSpace-v0.1.1-Windows-x64.zip`: extension + modular MRtrix3 `3.0.8` 68 cmds + FSL `2602.0` + Surface Runtime) | Verified on Apple Silicon (`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`: extension + modular MRtrix3 `3.0.8` 67 cmds + FSL `2602.0` + Surface Runtime `arm64`; Intel `x86_64` hardware unverified) |
| Resource/CMake contract | Local evidence; CI configured | Verified (`audit_extension_package.py` pass); CI configured | Verified on `arm64` (`audit_extension_package.py` pass); CI configured |
| Real Slicer 5.10.0 GUI & Extension Manager install | Dated local records (`slicer_extension_manager_install_smoke.py` pass) | Verified on 3D Slicer 5.10.0 (`slicer_extension_manager_install_smoke.py` + 4 GUI smoke suites pass with `SLICERDSPACE_DISABLE_WSL=1`) | Verified on Apple Silicon 3D Slicer 5.10.0 (`slicer_extension_manager_install_smoke.py` + 4 GUI smoke suites pass) |
| Real SDK/CTest/CPack | Linux/Slicer 5.10 candidate verified 2026-10-22 | Unified extension archive verified in Slicer 5.10.0 Extension Manager | Unified extension archive verified in Slicer 5.10.0 Extension Manager (`arm64`); full C++ superbuild SDK optional |
| Scientific workflows (Real Data `HCP 100307`) | Recorded reference routes/data (`145×174×145×288`) | Native Windows verified across all P0/P1 workflows with `SLICERDSPACE_DISABLE_WSL=1`; see [Windows 5.10.0 Acceptance Report](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md) | Native Apple Silicon (`arm64`) verified across all P0/P1 workflows; see [macOS 5.10.0 Acceptance Report](../release/MACOS_NATIVE_ACCEPTANCE_5.10.0.md) |
| Advanced backends | Bundled native MRtrix3 `3.0.8`, FSL `2602.0`, Surface Runtime | Bundled native MRtrix3 `3.0.8`, FSL `2602.0` (`eddy_cpu`/`topup`), Surface Runtime (default native; optional WSL 2) | Bundled native `arm64` MRtrix3 `3.0.8`, FSL `2602.0` (`eddy_cpu`/`topup`), Surface Runtime `0.1.0` |

The Linux record includes a real CPack archive and local installation with
isolated settings; see the [official integration record](../release/OFFICIAL_SLICER_VALIDATION_20261022.md).
It does not establish acceptance of later changes, a clean machine, online
store installation or other platforms. Resource checks are not three-platform
Slicer builds. See the [release gate](../release/EXTENSION_READINESS.md).

---

## Feature Routes

**Native verified** indicates real-data execution verified without WSL or Docker.
**Native candidate** describes source feasibility where full platform certification is still pending (such as physical Intel `x86_64` Mac hardware).
WSL is a Linux backend, not native Windows computation; on Windows `v0.1.1`, native execution is the default and WSL 2 is only an optional fallback.

| Feature | Linux (`x86_64`) | Windows (`x86_64`) | macOS (`arm64` verified; Intel `x86_64` unverified) |
| --- | --- | --- | --- |
| UI, NIfTI/NRRD, validation, result loading | Verified in Slicer 5.10.0 | **Verified native in 3D Slicer 5.10.0** (Extension Manager install + 4 GUI smoke suites + scene reload) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (Extension Manager install + 4 GUI smoke suites + scene reload) |
| DICOM conversion | Converter required (`dcm2niix`) | **Verified native in 3D Slicer 5.10.0** (`dcm2niix.exe`) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (`dcm2niix`) |
| DTI, deterministic tracking | Teem tensor fit in Slicer; DIPY tracking | **Verified native in 3D Slicer 5.10.0** (full-brain multi-shell `HCP 100307` & `HCP 100408` + clinical single-shell `sub-001`) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (full-brain `HCP 100307`, `max_abs = 0.0` vs Linux) |
| DDSeg/DDParcel/DDSurfer native reconstruction | Verified on recorded data/routes with matching models | **Verified native in 3D Slicer 5.10.0** (full-brain `HCP 100307`, `HCP 100408`, and `sub-001`) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (full-brain `HCP 100307`: DDSeg `99.89%`, DDParcel `101/101` labels, DDSurfer `100%` mesh topology) |
| FW, NODDI | Verified on recorded inputs | **Verified native in 3D Slicer 5.10.0** (full-brain `HCP 100307` FW & NODDI; full-brain `sub-001` FW) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (full-brain `HCP 100307` FW `r = 1.0`, NODDI `r >= 0.999999`) |
| DKI/Python MK-Curve | Full-brain WLS and MK-Curve checks recorded; invalid voxels excluded | **Verified native in 3D Slicer 5.10.0** (full-brain WLS on `HCP 100307` & `HCP 100408`; whole-axial-slab + stratified multi-slice MK-Curve) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (full-brain WLS `r >= 0.999999999999996` + multi-slice MK-Curve `r = 0.9919–0.9999` on `HCP 100307`) |
| PNL CNN mask | PyTorch/ANTs route; paired CPU probability/mask checks | Native route in Windows Slicer's Python (SimpleITK + PyTorch), **experimental** | Native route (SimpleITK + PyTorch), **experimental**; no CUDA; runs on MPS (PyTorch `2.2.2`, `x86_64`, under Rosetta 2) |
| Cortical postprocessing | Bundled native 7-command Surface Runtime in `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`; [setup](SURFACE_COMPONENTS.md) | **Verified native in 3D Slicer 5.10.0** (7 native `.exe` tools bundled in `surface-runtime-windows-x86_64`; requires user FreeSurfer license) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (7 native `arm64` tools bundled in `surface-runtime-darwin-arm64`, `r >= 0.9999999` vs Linux; requires user FreeSurfer license) |
| MRtrix/FSL preprocessing | Bundled native MRtrix3/FSL in `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` | **Verified native in 3D Slicer 5.10.0** (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe` in `fsl-2602.0-windows-x86_64`, numerically matched to Linux; WSL 2 optional fallback only) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (`topup`, `applytopup`, `eddy_cpu` in `fsl-2602.0-darwin-arm64`; **no CUDA eddy**) |
| CSD/ACT/SIFT2 | Bundled native MRtrix3 `3.0.8` route verified | **Verified native in 3D Slicer 5.10.0** (68 commands in `mrtrix-3.0.8-windows-x86_64`; T1-free ACT via `act_prep.py` `DDSeg + DDParcel -> 5TT -> 5ttcheck + 5tt2gmwmi` and `5ttgen freesurfer`) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (67 commands in `mrtrix-3.0.8-darwin-arm64`; T1-free ACT + `iFOD2` + `SIFT2`) |
| Full SAF | Python filter/geometry checks on recorded inputs | **Verified native in 3D Slicer 5.10.0** (`tckgen.exe`, `tckedit.exe`, `tckresample.exe`, `tckmap.exe`, `tcksift2.exe` + native `trimesh`/`rtree`) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (native `mrtrix-3.0.8-darwin-arm64` + `trimesh`/`rtree`, `12,745` streamlines on `HCP 100307`) |
| WMQL | Verified (`37/37` non-empty bundles) | **Verified native in 3D Slicer 5.10.0** (native `tract_querier`, `37/37` non-empty bundles matching Linux) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (native `tract_querier`, `37/37` non-empty bundles matching Linux) |
| WMA | Verified (`74/74` ORG bundles) | **Verified native in 3D Slicer 5.10.0** (`whitematteranalysis` + native VTK scalar stats, `74/74` ORG bundles 100% coverage) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (`whitematteranalysis` + native VTK scalar stats, `74/74` ORG bundles 100% coverage) |
| TractCloud | Seeded CPU/CUDA checks on recorded inputs | **Verified native in 3D Slicer 5.10.0** (Slicer PyTorch CPU/CUDA, `42/42` bundles 100% parity with Linux) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (Slicer PyTorch CPU, `42/42` bundles; inference also runs on MPS with PyTorch `2.2.2` under Rosetta 2) |
| Python connectome/ROI statistics/VTK previews | Verified on recorded inputs | **Verified native in 3D Slicer 5.10.0** (`84×84` symmetric matrices across multiple subjects) | **Verified native on Apple Silicon (`arm64`) in 3D Slicer 5.10.0** (`84×84` symmetric `python` and `mrtrix` connectome matrices + `101`-region ROI CSV) |
| Bundled runtime distribution & auto-recovery | `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` (`bundled-runtimes` + `tree_sha256` auto-repair) | **`SlicerDSpace-v0.1.1-Windows-x64.zip`** (`bundled-runtimes` auto-discovered on first start + `tree_sha256` auto-repair if deleted/corrupted) | **`SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`** (`bundled-runtimes` auto-discovered on first start + `tree_sha256` auto-repair if deleted/corrupted) |

---

## Hardware Memory & VRAM Requirements (Per Stage / Module)

Host RAM peaks below combine measured single-subject runs on `HCP 100307` (`145 × 174 × 145 × 288`, `1.25 mm` isotropic; `~7.85 GiB` uncompressed `float64` DWI signal) across Windows x64 CPU (`i7-13700K`, 32 GB RAM) and Linux `x86_64` M1 probes (`shared/cohort_defaults.py`), plus planner heuristics (`shared/cohort_planning.py`) and runtime VRAM/OOM safeguards (`shared/hardware_adapt.py`, `shared/environment.py`, `anatomy/surface/ddsurfer_backend/inference/predict.py`). Typical clinical single-shell acquisitions (`<= 64` volumes) require significantly less host RAM.

- **General host baseline**: `16 GiB` RAM covers single-subject full-brain HCP `288`-volume execution across all 22 modules (verified on Apple M4 16 GB unified memory); `32 GiB` RAM is recommended for full multi-shell pipelines, `MK-Curve`, `UKF`, `1M`-track `CSD/ACT`, `WMA`, or multi-subject batches; `8 GiB` RAM suffices for clinical (`<= 64`-volume) DTI and basic routes.
- **Consumer GPUs & Windows WDDM (`4–8 GiB` VRAM)**: Under `auto` device selection, CUDA kernels query free VRAM (`torch.cuda.mem_get_info`), scale chunk/batch sizes with Windows TDR caps (`windows_tdr_cap` / `consumer_fp64_tdr_cap`), disable `cudnn.benchmark` below `8.0 GiB` free on Windows WDDM (`6.0 GiB` on Linux), and halve chunks or fall back to CPU on OOM. `--low-memory` caps stage reservations, limits concurrency to `1` job / `<= 4` threads, and was verified under a `3.5 GiB` laptop VRAM cap (`<= 2.4 GiB` peak).
- **Explicit caps**: Set `SLICERDSPACE_MAX_VRAM_GIB` or `SLICERDSPACE_MAX_RAM_GIB` to enforce conservative budgets, or pass `--low-memory` in batch mode.

| Stage / Module | Typical Peak RAM (HCP 288-vol) | Recommended Host RAM | GPU VRAM (CUDA / MPS) | Notes (OOM / Auto-Fallback) |
| --- | --- | --- | --- | --- |
| **Input & Validation** (`DICOM / NIfTI / NRRD`) | `0.2–1.5 GiB` (`0.20 GiB` clinical 31-vol) | `8 GiB` | CPU only (`—`) | Single-pass gzip slab streaming avoids decompressing the full 4D volume into RAM. |
| **Brain Mask** (`PNL CNN`) | `2.2 GiB` (`2.15 GiB` PC CPU) | `8 GiB` | Optional `>= 2 GiB` CUDA or Apple MPS (`16 GB` unified) | `adaptive_cuda_scaled_batch` / `adaptive_gpu_scaled_batch` sizes 2D slice batches; halves batch on CUDA/MPS OOM and falls back to CPU under `auto`. |
| **Preprocessing** (`DIPY MP-PCA / Gibbs`) | `~1–8 GiB` (`0.92 GiB` clinical 31-vol) | `16 GiB` (`8 GiB` clinical `<= 64`-vol) | Optional `>= 2 GiB` CUDA (`~150 MB` rolling 5-slice ring buffer); Apple MPS for Gibbs `float32` FFT | `adaptive_cpu_slab_z` bounds CPU slab RAM; CUDA/MPS halves chunks on OOM and falls back to multi-threaded CPU under `auto`. |
| **Preprocessing** (`FSL Eddy / Topup`) | `~2.2–12 GiB` (`2.17 GiB` clinical 31-vol `eddy_cpu`) | `16 GiB` (`32 GiB` large multi-shell cohorts) | Optional `>= 4–6 GiB` CUDA (`eddy_cuda`, external Linux FSL only); CPU only on bundled Windows/macOS | Bundled native Windows/macOS FSL runs multi-threaded `topup` + `eddy_cpu`; no GPU required. |
| **Anatomical Priors** (`DDSeg`) | `2.8 GiB` (`2.79 GiB` Linux M1) | `8–16 GiB` | Optional `>= 2–4 GiB` CUDA or Apple MPS (`16 GB` unified) | Halves 2D slice batch down to `2` on CUDA/MPS OOM and falls back to CPU under `auto`; cuDNN benchmark gated at `>= 8.0 GiB` free on Windows WDDM (`>= 6.0 GiB` Linux). |
| **Anatomical Priors** (`DDParcel`) | `10.8 GiB` (`10.82 GiB` Linux M1) | `16 GiB` (`32 GiB` cohorts) | Optional `>= 4 GiB` CUDA or Apple MPS (`16 GB` unified) | Automatic single-slice retry on CUDA/MPS OOM with CPU fallback under `auto`; cuDNN benchmark gated at `>= 8.0 GiB` free on Windows WDDM. |
| **Anatomical Priors** (`DDSurfer`) | `3.6 GiB` (`3.63 GiB` Linux M1) | `16 GiB` | Optional CUDA: `>= 7.0 GiB` free (`5.5 GiB` budget + `1.5 GiB` margin, lazy single-surface `~4–5 GiB` peak); `>= 18.0 GiB` free for eager dual-surface; CPU on macOS Rosetta PyTorch `2.2.2` (no MPS `Conv3d`) | Preflight gate (`SURFACE_MIN_FREE_GPU_GIB = 7.0`) automatically routes `< 7.0 GiB` free CUDA (including typical `< 8 GiB` Windows WDDM GPUs) to CPU under `auto`; explicit `cuda` exits cleanly with guidance. |
| **Anatomical Priors** (`FreeSurfer Surface Morphometry`) | `3.4 GiB` (`3.42 GiB` PC CPU) | `8–16 GiB` | CPU only (`—`) | Bundled 7-command native runtime (`mris_inflate`, `mris_sphere`, `mris_register`, etc.) with intra-hemisphere DAG parallelism and OpenMP (`macOS 13.0+` static `libomp.a`). Requires user `FS_LICENSE`. |
| **Microstructure** (`Teem DTI`) | `6.6–9.7 GiB` fresh (`6.60 GiB` PC / `9.75 GiB` Linux M1; `2.4 GiB` reusing prepared DTI) | `16 GiB` (`8 GiB` clinical `<= 64`-vol) | CPU only (`—`, `fp64` WLS by design) | Zero-voxel skipping and shared preparation receipts avoid redundant tensor recomputation. |
| **Microstructure** (`Free-Water`) | `2.4–6.9 GiB` (`2.37 GiB` Linux M1 / `6.93 GiB` PC CPU) | `16 GiB` | Optional `>= 4 GiB` CUDA (`fp64`); CPU on macOS | `adaptive_cuda_scaled_batch` applies Windows WDDM / consumer-FP64 TDR caps, halves chunks on CUDA OOM, and falls back to CPU under `auto`. |
| **Microstructure** (`AMICO NODDI`) | `8.6–17.2 GiB` (`8.55 GiB` PC active-set QP / `17.22 GiB` Linux M1 peak) | `16–32 GiB` | Optional `>= 4 GiB` CUDA (`fp64`); CPU on macOS | CUDA active-set QP uses `windows_tdr_cap=8192` / `consumer_fp64_tdr_cap=2048`, halves chunks on OOM, and falls back to CPU under `auto`. |
| **Microstructure** (`DKI`) | `5.9–6.9 GiB` (`5.94 GiB` Linux M1 / `6.88 GiB` PC CPU) | `16 GiB` | CPU only (`—`, vectorized `fp64` WLS/OLS CPU solver) | `adaptive_spawn_workers` bounds spawn workers to available RAM on Windows/macOS (and Linux fork workers throttle down under low available RAM); `--low-memory` sets `--dki-voxels-per-chunk 8192`. |
| **Microstructure** (`MK-Curve`) | `3.3–9.8 GiB` (`3.29 GiB` Apple M4 8-thread streaming CPU; `9.78 GiB` legacy 4-process PC CPU) | `16 GiB` | Optional `>= 4 GiB` CUDA (`fp64`); CPU on macOS | Uses single-pass streaming staging and GIL-releasing `ThreadPoolExecutor` on Windows/macOS (`len(grid) >= 64`, `<= 4` threads on `<= 10 GiB` hosts); Linux retains process workers; CUDA halves chunks on OOM. |
| **Tractography** (`DTI Det`) | `~4–8 GiB` (slab-bounded; header heuristic `2 + signal`) | `16 GiB` (`8 GiB` clinical `<= 64`-vol) | CPU only (`—`, vectorized multi-threaded CPU tensor fit & EuDX) | `adaptive_cpu_slab_z` bounds slab loading RAM and worker thread count scales down to available RAM (`1` worker under tight memory). |
| **Tractography** (`UKF`) | `10.1 GiB` (`10.09 GiB` PC CPU, `250k` streamlines) | `16–32 GiB` | CPU only (`—`, multi-threaded C++ `UKFTractography`) | Peak scales with streamline density and seeds-per-voxel; uses single-buffer VTK/TRK serialization. |
| **Tractography** (`MRtrix3 CSD / ACT`) | `10.6 GiB` (`10.58 GiB` Linux M1, `1,000,000` tracks) | `16–32 GiB` | CPU only (`—`, native MRtrix3 `3.0.8`) | Peak scales with streamline count (`tckgen` + `tcksift2`); reduce target track count on `16 GiB` hosts if multitasking. |
| **Tractography** (`SAF`) | `0.6–4 GiB` (`0.63 GiB` re-filtering `100k` tracks with DDSurfer meshes) | `16 GiB` | CPU only (`—`, vectorized BVH ray–triangle CPU kernels) | Runs natively on multi-threaded CPU (`tckgen`/`tckedit` + BVH geometric filtering). |
| **Fibre Bundles** (`WMQL`) | `~4–8 GiB` (`1M` CSD/ACT tracks; planner heuristic `8 GiB`) | `16 GiB` | CPU only (`—`, multi-threaded flat CSR CPU kernels) | Flat CSR voxel-label crossing histograms avoid dense per-streamline Python dictionaries. |
| **Fibre Bundles** (`WMA`) | `5.7–12 GiB` (`5.74 GiB` PC CPU at `80k` tracks / `8` threads; planner heuristic `16 GiB`) | `16–32 GiB` | Optional `>= 4–6 GiB` CUDA (PyTorch/Triton kernels); CPU on macOS | Configures Windows `PYTORCH_CUDA_ALLOC_CONF`; automatically retries on CPU (`WMA_GPU=off`) if CUDA OOM occurs under `auto`. |
| **Fibre Bundles** (`TractCloud`) | `2.7–8 GiB` (`2.70 GiB` PC CPU on `50k` tracks; scales up with track count toward `~8–12 GiB` on `1M` tracks) | `16 GiB` | Optional `>= 2–4 GiB` CUDA or Apple MPS (`16 GB` unified) | Auto-caps batch size to free memory (`256` if `< 2.2 GiB`, `512` if `< 4.5 GiB`, `1024` if Windows `< 7.5 GiB`; MPS `256/512/1024`); explicit GPU errors do not silently switch to CPU (select `CPU` if needed). |
| **Connectome** (`Streamline Count`) | `0.3–2.6 GiB` (`0.27 GiB` PC Python on `1M` tracks; `2.61 GiB` Linux M1) | `8–16 GiB` | CPU only (`—`, batched `cKDTree` / `tck2connectome`) | Batched `4,096`-streamline endpoint assignment keeps peak RAM low even on `1M`-track tractograms. |
| **Connectome** (`SIFT2-Weighted`) | `0.5–2.6 GiB` (`<= 2.61 GiB` Linux M1 on `1M` tracks) | `8–16 GiB` | CPU only (`—`, native MRtrix3 / vectorized Python) | Requires matched tractogram and `sift2_weights.txt`; minimal memory overhead beyond endpoint indexing. |

---

## Practical Limits & Distribution Scope

- **Installation method**: Install `SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, or `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` from GitHub Releases via 3D Slicer 5.10.0 `Extensions Manager -> Install Extensions -> Install Extension from File` (not yet listed in the online Slicer Extension Catalog; physical Intel `x86_64` Mac hardware is unverified).
- **On-demand model weights & atlases**: The unified platform archive bundles the extension code and native C/C++ runtimes (`MRtrix3 3.0.8`, `FSL 2602.0` diffusion preprocessing, and `FreeSurfer Surface Runtime 0.1.0`). Deep-learning model weights (`DDSeg`, `DDParcel`, `DDSurfer`, `TractCloud`, `PNL`) and large atlases (`ORG`) remain downloaded on demand on first use and cached locally.
- **ACT / 5TT preparation scope**: SlicerDSpace's native T1-free ACT workflow generates 5TT and GMWMI directly from `DDSeg + DDParcel` (`tractography/act_prep.py`, validated by `5ttcheck` and `5tt2gmwmi`) or via `5ttgen freesurfer` on parcellation volumes. Upstream MRtrix3 `5ttgen fsl` requires T1w input plus external full FSL (`bet`, `fast`, `run_first_all`) and is not used by SlicerDSpace or supported by the 3-command FSL diffusion preprocessing bundle (`topup`, `applytopup`, `eddy_cpu`).
- **FreeSurfer license requirement**: The 7 bundled FreeSurfer cortical surface postprocessing commands (`mris_anatomical_stats`, `mris_ca_label`, etc.) do not include any user-specific FreeSurfer license file (`activation_key_included: false`). Users running cortical surface postprocessing must supply their own free FreeSurfer license (`FS_LICENSE`).
- **Mask policy & PyTorch devices**: Existing reviewed native masks are preserved on all platforms without silent algorithm substitution. Auto PyTorch selects CUDA when available. On Apple Silicon, the bundled PyTorch `2.2.2` (`x86_64`, under Rosetta 2) runs DDSeg, DDParcel, PNL CNN mask and TractCloud inference on **MPS**. The DDSurfer surface volume network (`Conv3d`) runs on CPU because this PyTorch build has no MPS `Conv3d`. The `fp64` model fits (DTI, DKI, NODDI, free water, MK-Curve) run on CPU by design.
- **DKI / MK-Curve scope**: Full-brain DKI (`WLS`) and whole-axial-slab / multi-slice Python `MK-Curve` runs are recorded on multi-shell subjects (`HCP 100307`, `HCP 100408`); valid-voxel screening is not a full-brain MATLAB runtime match across arbitrary clinical protocols.

See the [Windows 5.10.0 Acceptance Report](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md), [macOS 5.10.0 Acceptance Report](../release/MACOS_NATIVE_ACCEPTANCE_5.10.0.md), [Windows Setup Guide](WINDOWS.md), and [macOS Guide](MACOS.md).

---

## Upstream Boundaries

- **FSL**: Upstream FSL officially distributes full Linux/macOS suites and recommends WSL on Windows ([FSL installation](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html), [FSL Windows](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)). SlicerDSpace `v0.1.1` bundles native Windows x64 builds of the 3 diffusion preprocessing executables (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe`) under the Oxford University Innovation Academic Licence; full FSL tools outside diffusion preprocessing (`bet`, `fast`, `FIRST`, GPU `eddy_cuda`) are not bundled.
- **FreeSurfer**: Upstream FreeSurfer officially distributes Linux/macOS suites ([FreeSurfer requirements](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements)). SlicerDSpace `v0.1.1` bundles native Windows x64 builds of the 7 cortical surface postprocessing commands required by `DDSurfer`; full `recon-all` is not bundled and a user FreeSurfer license is still required.
- **PyTorch MPS**: PyTorch supports Apple MPS ([PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)), and SlicerDSpace uses it on Apple Silicon for the inference routes listed under **Mask policy & PyTorch devices**.
