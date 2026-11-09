<div align="center">

# Windows Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

## Recommended Default: Unified Windows Native Package (`v0.1.1`, No WSL Required)

Starting in **`v0.1.1`**, SlicerDSpace provides a single unified Windows release package (**`SlicerDSpace-v0.1.1-Windows-x64.zip`**) for **3D Slicer 5.10.0** that bundles the extension together with modularly managed native Windows runtimes under `SlicerDSpace/bundled-runtimes/`:

- **MRtrix3 `3.0.8`** (`bundled-runtimes/mrtrix-3.0.8-windows-x86_64`, **68 native commands** including `dwi2response`, `5ttgen`, `5ttcheck`, `5tt2gmwmi`, `dwi2fod`, `mtnormalise`, `tckgen`, `tcksift2`, `tck2connectome`, `tckedit`, `tckresample`, `tckmap`, `mrcrop`, plus bundled `FreeSurferColorLUT.txt` fallback). SlicerDSpace prepares 5TT and GMWMI natively without a T1w scan via `tractography/act_prep.py` (`DDSeg + DDParcel -> 5TT -> 5ttcheck.exe + 5tt2gmwmi.exe`) or `5ttgen freesurfer` (upstream `5ttgen fsl` requires T1w + external full FSL `bet`/`fast`/`FIRST` and is not used by SlicerDSpace).
- **FSL `2602.0`** (`bundled-runtimes/fsl-2602.0-windows-x86_64`, native `topup.exe`, `applytopup.exe`, `eddy_cpu.exe` for diffusion distortion and motion/eddy correction).
- **FreeSurfer Surface Runtime** (`bundled-runtimes/surface-runtime-windows-x86_64`, 7 native surface postprocessing commands and atlas templates; `activation_key_included: false` — supply your personal FreeSurfer license file when running cortical postprocessing).

**Installation, First Run & Automatic Integrity Recovery:**
1. Open **3D Slicer 5.10.0** → **View > Extension Manager** → wrench icon (**Install Extension from File…**) and select `SlicerDSpace-v0.1.1-Windows-x64.zip`, then restart Slicer.
2. On first launch, SlicerDSpace defaults to the **Native** execution backend on Windows (`preprocessing_execution_backend = "native"`) and automatically discovers the bundled MRtrix3, FSL, and Surface Runtime paths without manual path entry or WSL.
3. Each bundled runtime is validated against its per-file content digest (`tree_sha256` in `models/native_runtime_bundles.json`). If any file or component inside `bundled-runtimes/` is accidentally deleted or corrupted, **Prepare selected analysis environment** automatically detects the mismatch and repairs the missing runtime from the release archive into `%LOCALAPPDATA%\SlicerDSpace\runtimes`.
4. Deep-learning model weights (`DDSeg`, `DDParcel`, `DDSurfer`, `PNL`, `TractCloud`) and large atlases (`WMA ORG`) remain downloaded on demand on first use via **Environment and installation > Prepare selected analysis environment**.

See the [Windows 5.10.0 Native Acceptance Report](../release/WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md) for multi-subject real-data (`HCP 100307`, `HCP 100408`, clinical `sub-001`) validation and numerical parity against Linux.

---

## Optional Legacy Route: WSL 2 Bridge Setup

The sections below document the optional legacy Ubuntu WSL 2 bridge setup (useful if you specifically require Linux CUDA `eddy` inside WSL 2). The current bridge supervises the owned Linux process group with a
Windows heartbeat. If the bridge is killed, only that WSL task is terminated;
the distribution and unrelated jobs are not shut down. Existing bridges need
the runtime installer rerun to obtain this change.
See [current bundle/platform follow-up](../release/BUNDLE_QC_20261015.md).

**What runs where.** In native mode (default on Windows), MRtrix3, FSL (`topup`/`applytopup`/`eddy_cpu`), FreeSurfer surface postprocessing, WMA ORG clustering, WMQL, TractCloud, SAF, DTI, Free-Water, NODDI, DKI (`WLS` & `MK-Curve`), and DDSeg/DDParcel/DDSurfer all run natively on Windows without WSL. In the optional WSL 2 mode, WSL 2 Ubuntu runs MRtrix3, FSL, and optional Linux tools. Use **Prepare
selected analysis environment** with WMA selected to prepare the ORG atlas
and pinned WMA source. An existing bridge remains usable. East-West, NABA and
the FiberTractMeasurements export use the legacy route.

The tensor fit, the PNL brain mask and the other Python stages run in Windows
Slicer's own Python. The
PNL mask on Windows uses the native route (SimpleITK and PyTorch), labelled
"native (experimental)"; the managed conda route is refused on Windows. In the
GUI, **Prepare selected analysis environment** lists the WSL runtime, with its
licences, in the same single dialog as the other downloads. If WSL 2 Ubuntu is
missing, the dialog shows the install command; it does not install WSL itself.

---

## 1. Prerequisites

Install these once:

- 3D Slicer for Windows.
- WSL 2 with an Ubuntu distribution and a normal (non-root) default user.
- A FreeSurfer license file if FreeSurfer postprocessing is required.

The installer deliberately uses `wsl.exe --user root` only for Ubuntu `apt`.
The scientific runtime and downloaded sources are installed as the normal WSL
user; the Linux script never calls `sudo`.

Confirm the distribution name from PowerShell:

```powershell
wsl.exe --list --verbose
```

---

## 2. Inspect the Reproducible Plan

Plan mode performs no WSL, package, registry or Slicer changes. It is also safe
to use before WSL has been installed.

```powershell
Set-Location D:\SlicerDSpace
.\scripts\install_deps_windows.ps1 `
  -Distro Ubuntu `
  -PlanOnly `
  -PlanOutput D:\SlicerDSpace-test\windows-install-plan.json
```

The plan records every pinned version, source commit, resource URL and digest.

---

## 3. Install the Full Runtime

Pass the local Slicer executable and your own FreeSurfer license:

```powershell
.\scripts\install_deps_windows.ps1 `
  -Distro Ubuntu `
  -AcceptFslLicense `
  -SlicerExecutable "C:\path\to\Slicer.exe" `
  -FreeSurferLicensePath "C:\path\to\license.txt"
```

This command really executes the following idempotent operations:

1. Installs fixed Ubuntu packages as WSL root.
2. Verifies and installs Miniforge `26.3.2-3`.
3. Creates a Python 3.10 runtime with MRtrix3 `3.0.8` and the pinned runtime
   requirements in `scripts/wsl/requirements-runtime.txt`. Default `-Device auto`
   detects the GPU inside WSL: PyTorch `2.11.0+cu126` for CUDA,
   `2.11.0+cpu` without a GPU or with explicit `-Device cpu`. A fresh interpreter
   verifies the installed wheel and GPU allocation; explicit CUDA failure has no CPU retry.
4. Creates a separate FSL environment from the official FSL conda channel,
   pinned to Eddy `2602.0` (`hb35a746_0`) and topup `2203.5`
   (`ha905fa9_6`). The generated command bridge sets `FSLDIR` and
   `FSLOUTPUTTYPE=NIFTI_GZ` for every MRtrix/FSL invocation. GPU selection also
   installs `fsl-eddy-cuda=2602.0=h1a432fe_1`. CPU/CUDA use separate FSL prefixes.
   `-AcceptFslLicense` is required before actual installation, not plan mode.
5. Checks out exact TractQuerier, WhiteMatterAnalysis and TractCloud commits.
6. Downloads TractCloud v1.0.0 data and both ORG atlas archives with SHA-256
   verification.
7. Downloads FreeSurfer `7.4.1`, verifies its official MD5, and installs the
   supplied license.
8. Downloads and SHA-256-verifies the core neural model/template profile. An
   existing `model_assets_dir` in `user_config.json` is reused; otherwise the
   assets are stored below the Windows install root.
9. Compiles native Windows-to-WSL command bridges and writes a dependency JSON
   that the SlicerDSpace environment panel and CLI can consume.
10. Merges the bridge paths into `%APPDATA%\SlicerDSpace\user_config.json` so
   subsequent Slicer sessions discover them without re-entering paths.
11. Downloads Windows wheels for Slicer's exact Python ABI through the managed
    WSL runtime, records their SHA-256 values, installs them into PythonSlicer
    offline, and requires the final environment doctor to pass.

The default Linux root is `~/.local/share/slicerdsurfer`. The default Windows
bridge/config root is `%LOCALAPPDATA%\SlicerDSpace\wsl-runtime`. Re-running
the command verifies caches, pinned commits and versions instead of blindly
duplicating them.

The installer refuses to adopt an existing Miniforge or FreeSurfer directory
that has no matching SlicerDSpace provenance marker. Preserve that directory
and choose a clean managed location instead, for example
`-WslInstallRoot ~/.local/share/slicerdsurfer-managed`.

FreeSurfer is approximately 9 GB and the two ORG archives total approximately
1.8 GB. Make sure the WSL virtual disk has sufficient free space.

---

## 4. Defer Large Downloads

To install Miniforge, MRtrix3, the Python runtime, pinned source checkouts,
TractCloud and the Windows bridge while deferring FreeSurfer and ORG data:

```powershell
.\scripts\install_deps_windows.ps1 `
  -Distro Ubuntu `
  -AcceptFslLicense `
  -SlicerExecutable "C:\path\to\Slicer.exe" `
  -SkipLargeDownloads
```

Re-run later without `-SkipLargeDownloads` and supply
`-FreeSurferLicensePath`; existing verified components are reused. Use
`-SkipFreeSurfer` only when FreeSurfer is intentionally out of scope.

`-SkipApt` is available for a pre-provisioned/offline Ubuntu image.
`-SkipSlicerPythonInstall` leaves PythonSlicer unchanged while still building
the WSL runtime and bridges. `-SkipUserConfigUpdate` writes the standalone
dependency JSON without changing the persistent GUI/CLI user configuration.

---

## 5. Verified Mirrors

Every ORG or TractCloud resource URL override requires a matching explicit
SHA-256. The installer refuses a URL-only override:

```powershell
.\scripts\install_deps_windows.ps1 `
  -Distro Ubuntu `
  -AcceptFslLicense `
  -OrgRegUrl "https://mirror.example/ORG-RegAtlas-100HCP.zip" `
  -OrgRegSha256 "<64 hexadecimal characters>" `
  -Org800Url "https://mirror.example/ORG-800FC-100HCP.zip" `
  -Org800Sha256 "<64 hexadecimal characters>" `
  -SlicerExecutable "C:\path\to\Slicer.exe" `
  -FreeSurferLicensePath "C:\path\to\license.txt"
```

Equivalent override pairs are available for `TractCloudTrainUrl` /
`TractCloudTrainSha256` and `TractCloudModelUrl` /
`TractCloudModelSha256`.

---

## 6. Use the Generated Dependency Config

The installer writes:

```text
%LOCALAPPDATA%\SlicerDSpace\wsl-runtime\dependency-config.json
```

For command-line runs, pass it explicitly:

```powershell
PythonSlicer -m shared.run_subject_pipeline_cli `
  --subject-root D:\data\sub-01 `
  --targets tractography,wma,tractcloud,connectome `
  --dependency-config "$env:LOCALAPPDATA\SlicerDSpace\wsl-runtime\dependency-config.json" `
  --execute
```

The bridge translates drive-letter and `\\wsl.localhost\...` paths before
calling Linux tools. A protected `bash.exe` bridge is placed beside the chosen
Slicer executable so the bundled FreeSurfer shell workflow uses WSL reliably.
The installer refuses to overwrite a pre-existing `bash.exe` that it does not
own. Restart Slicer once after installation so it reloads the persistent JSON;
the installer never terminates an open Slicer process.

---

## 7. Install Model Assets

The WSL installer covers executable backends and their atlas/model resources.
DDSeg, DDParcel and DDSurfer checkpoints remain managed by the model-asset
installer. `PythonSlicer.exe` is Slicer's own Python (in the Slicer `bin` folder). No system Python is needed:

```powershell
PythonSlicer.exe scripts\download_model_assets.py `
  --profile core `
  --asset-root D:\SlicerDSpace-model-assets `
  --source url `
  --source-id slicerdsurfer_models_git
```

Set `Model assets root` to the same directory in SlicerDSpace. Existing
verified assets are reused.

---

## 8. Prepare Subject Data

Recommended layout:

```text
D:\data\sub-01\
  input\
    data.nii.gz
    bvals
    bvecs
    nodif_brain_mask.nii.gz
```

In SlicerDSpace, set the subject ID and subject root, then run `Check
environment dependencies`. If a stage is still blocked, include the generated
environment report and install plan when reporting the issue.

For preprocessing (DIPY MP-PCA, DIPY Gibbs removal, then FSL eddy/topup through `dwifslpreproc`), provide the real phase-encoding direction (needed; the run is blocked without it) and
the total readout time from the acquisition metadata. Both can come from `input/dwi.json`;
a direction you type wins over the sidecar. Most axial clinical DWI is AP. Susceptibility correction
with topup additionally requires the subject's reverse-phase spin-echo b0 data.
If those acquisition files are absent, `-rpe_none` can exercise Eddy but cannot
replace a scientifically valid topup correction; do not guess these values for
production processing.
