<div align="center">

# Managed Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](QUICKSTART.md) | [GPU](GPU_EXECUTION.md) | [中文](MANAGED_SETUP.zh-CN.md)

</div>

---

## Inside Slicer

Select your analyses, then open the separate **Environment and installation** section
and choose **Prepare selected analysis environment**. Ticking a product adds the
prerequisites of that product (preprocessing is never added), and the plan covers
only those. One dialog lists every download with its size and its licence (name,
link and local file). Press **Accept licences and install** once. The dependency
report uses the same entry. Checks and final verification run in cancellable background processes.
An already-importable environment skips installation and runs functional checks. Offline import,
repair and manual paths live under Environment > More options, not Advanced
algorithm settings. It prepares required selected items, not every optional backend:

1. Slicer's Extension Manager installs the missing extensions that the ticked products need: PyTorch for the neural models and the native brain mask, UKFTractography for UKF, SlicerDMRI for WMA. The tensor fit needs none (it uses Teem in Slicer core).
2. If an extension was installed or enabled, save your work and restart Slicer. Reopening SlicerDSpace offers to continue preparation; extensions are reused.
3. Background workers prepare pinned WMQL/WMA/TractCloud sources without Git, install missing Python wheels, download selected models, and prepare supported isolated tools.
4. Paths are saved after verification. A fresh dependency check and synthetic functional tests determine readiness. A failed functional test prevents analysis until preparation passes again.
The same self-checks run from the command line with `scripts/check_setup_functionality.py`.
dcm2niix (used for DICOM import) follows one rule: if the plugin's own pinned wheel,
or a dcm2niix path that you set, fails its check, the setup is not ready. If a
dcm2niix found only on `PATH` fails, you get a warning, because NIfTI input does not need it.

Functional failures are retained across restarts for the same configuration.
Source discovery is shared with runtime execution, so verified prepared sources
remain usable if an explicit saved source path is absent. On Windows `v0.1.1`,
native execution is the default (`MRtrix3 3.0.8`, `FSL 2602.0` diffusion preprocessing,
and `FreeSurfer Surface Runtime 0.1.0` are bundled inside `SlicerDSpace-v0.1.1-Windows-x64.zip`
under `bundled-runtimes/`, and `WMA`, `WMQL`, `TractCloud`, and `SAF` run natively in
Windows Slicer's Python without WSL; WSL 2 remains an optional fallback).

With **First-use setup** enabled, clicking Run with missing dependencies opens this
same preparation plan. It does not start analysis automatically after installation
or restart. The confirmation lists resource locations, known model/source sizes
and applicable licenses. Solver-dependent Python/Conda downloads are not included
in those size totals; they may require several additional GB.

No terminal, administrator account, shell activation or manual executable-path
selection is needed for these managed routes. If Conda is unavailable, a pinned
Miniforge bootstrap is downloaded from conda-forge's official GitHub release and
SHA256-verified before execution. Installation happens in a user-local directory,
not in Slicer's Python or the system CUDA installation. FSL and PNL
source/model/template terms require explicit confirmation. GPU eddy is selected
by default when NVIDIA tools are detected; the CUDA profile also retains the CPU
executable. Analysis device preferences remain editable under Advanced.
Actual GPU usability is checked when
running the selected backend, not certified by package installation.

Setup runs in a background process with a Cancel control and an install log.
Cancellation/failed verification does not replace configured paths. Successfully
installed packages and partial downloads may remain for retry. A verified,
unchanged managed runtime is reused without reinstalling it.

**Repair selected managed tools** explicitly reinstalls selected plugin-owned
external runtimes and rechecks them. It preserves the preceding ready receipt,
refuses unowned prefixes, and does not repair arbitrary existing environments.
Retry ordinary preparation after an interrupted download/install; use Repair
when a completed managed installation is damaged. Cancellation does not roll
back installed packages. Open install logs to inspect the failed phase.

Python installation feature-detects `slicer.packaging.pip_ensure`. Slicer 5.10,
which lacks that module, uses a cancellable Python subprocess. Existing Slicer
NumPy/SciPy/VTK/SimpleITK/PyTorch versions are constrained during ordinary pip
installation. PyTorch uses the official PyTorch extension helper, retaining the
strongest selected minimum version. Compatible DDSeg/DDParcel/DDSurfer/TractCloud
tasks share it by default; WMQL, DIPY and AMICO use the workflow Python. Explicit
interpreter overrides are retained. Conflicts fail
visibly instead of upgrading Slicer's core stack. Changed loaded packages require
a restart. The newer packaging API has contract tests; a newer Slicer binary
has not been validated in this campaign.
Ordinary dependency requests use binary wheels only. Missing wheels fail with a
setup log instead of starting a compiler. Expert VCS URLs and local projects use native `pip_install`, not the name-only
`pip_ensure` parser; the same constraints and final import checks apply.

---

## Models And Existing Installations

Model download uses the configured asset root or the default user data directory;
it no longer requires a folder picker. A missing-model row downloads that product's
profile rather than all three anatomical products. The explicit core download
action still downloads the core profile. Checksums are verified before the asset
path is saved. Authorization and model quality are separate from checksums.

**Use offline model resources** selects an existing asset directory and verifies
selected profiles against the bundled trusted SHA256 manifest in a background
worker. It neither downloads nor executes the files; failed checks preserve the
previous path. Preserve the documented directory layout when transferring
resources. This imports models, not a portable Conda environment. Protected
model downloads may still require source authorization; setup does not bypass it.

Optional PNL models/templates, TractCloud inference references
and WMA ORG atlases use the shared resource layout. Selected preparation acquires
missing profiles, verifies SHA256 and safely extracts ORG before saving paths.
Algorithm sources and their data are separate preparation phases. Pinned source
archives retain upstream licenses and checksums; prepared roots are saved
automatically. Existing custom roots are not replaced. See
[optional resources and offline import](../assets/EXTERNAL_RESOURCES.md).

External-path fields in Environment and installation remain available for existing installations, offline
environments and administrator-managed clusters. Managed setup is opt-in and does
not silently replace an explicitly configured environment while an analysis runs.
User-local defaults live alongside model assets under `SlicerDSpace/runtimes`.

When setup replaces an existing macOS MRtrix3 runtime, it renames the old folder
to `mrtrix-3.0.8-macos.replaced-<ns>` in that runtimes folder (about 169 MB each;
`<ns>` is a time stamp). The plugin never deletes these folders. You can delete
them by hand after a successful run.

---

## Current Coverage

| Component | Managed path |
| --- | --- |
| MRtrix3 3.0.8 | Windows x64 & Linux x86_64: bundled natively in the unified `v0.1.1` release archive (`bundled-runtimes/mrtrix-3.0.8-*`, 68 commands, `tree_sha256` auto-repair) or installed via prebuilt packages. macOS (Intel and Apple Silicon, macOS 14.5 or later): official universal tarball, SHA-256 verified, unpacked into the runtimes folder (full scientific acceptance pending) |
| FSL eddy 2602.0 / topup 2203.5 | Windows x64 (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe`) & Linux x86_64: bundled natively in the unified `v0.1.1` release archive (`bundled-runtimes/fsl-2602.0-*`, `tree_sha256` auto-repair) or installed from official FSL channel with explicit license approval. macOS Intel and Apple Silicon (macOS 14.5 or later; CPU eddy only): official FSL channel candidate |
| CUDA eddy 2602.0 | Optional Linux FSL profile (or optional WSL 2 with NVIDIA GPU on Windows); NVIDIA driver required; no full CUDA Toolkit installation |
| DDSurfer / DDSeg / DDParcel assets | On-demand background download with profile-specific selection and default directory (`SHA-256` verified) |
| Python scientific packages | Selected interpreter, protected Slicer core packages, native packaging API when available; official PyTorch helper |
| PNL diffusion CNN | Linux x86_64: pinned PNLBrainMask-Torch package, converted v0.3 three-view weights, IIT b0 template, Python 3.11, PyTorch 2.5.1 CPU and ANTs 2.3.0 in a separate prefix. Windows & macOS: native route (`SimpleITK` + `PyTorch`) in Slicer's Python (`experimental`) |
| Optional cortical postprocessing | Bundled 7-command native [Surface Runtime](SURFACE_COMPONENTS.md) (`0.1.0`) in both `SlicerDSpace-v0.1.1-Windows-x64.zip` (`surface-runtime-windows-x86_64`) and `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` (`surface-runtime-linux-x86_64`); personal FreeSurfer license (`FS_LICENSE`) required; GUI never compiles sources |
| WMA / TractCloud non-code resources | On-demand profile download/import, checksums, ORG extraction and automatic path saving |
| WMA/WMQL/TractCloud sources | Pinned archives, safe extraction, file hashes, automatic discovery/path saving; runs natively on Windows and Linux without Git or WSL |
| Windows (`x86_64`) | **Default native execution** (`SlicerDSpace-v0.1.1-Windows-x64.zip`): bundled native `MRtrix3 3.0.8` (68 commands), `FSL 2602.0` (`topup.exe`, `applytopup.exe`, `eddy_cpu.exe`), and `Surface Runtime 0.1.0` (7 commands) auto-discovered on first launch with `tree_sha256` auto-repair; `WMA`, `WMQL`, `TractCloud`, `SAF`, and microstructure models run natively in Windows Slicer's Python. WSL 2 (`scripts/install_deps_windows.ps1`) is an optional fallback only |
| macOS | Native candidate only; unified release archive and full native runtime validation remain pending (no CUDA eddy) |

Functional preparation tests execute a NumPy solve, write/read a small NIfTI and
VTK streamline where selected, exercise the selected PyTorch device, deserialize
selected anatomical checkpoints, and run synthetic DDSeg and TractCloud inference. MRtrix
converts a small image; the managed PNL runtime runs an ANTs ImageMath identity
operation and reads the image back. Receipts are saved in `setup-history`.
These tests use no patient data. Installation verification also checks command availability, selected package versions,
ELF libraries and executable hashes, and saves an explicit environment lockfile.
PNL also verifies pinned source/model/template hashes before deserialization,
checks finite synthetic forward outputs for all three views, and saves a pip
package inventory. Its CUDA wheel libraries and tools are exposed only to its
subprocess, not global PATH/LD_LIBRARY_PATH or the driver. Visible CUDA is
preferred where usable; a pinned framework may need first-use kernel compilation.
It does not establish scientific correctness or replace real clean-install,
SDK/package and platform acceptance. See the dated
[GPU verification](../release/GPU_POLICY_20261013.md).

---

## Continue After Restart

An unfinished, explicitly confirmed setup is remembered across Slicer restarts.
The existing preparation button becomes **Continue preparing selected analysis
environment**. Reentering the module offers a recheck; the button also remains
available. Continuing rechecks current selections and requests any needed
confirmation again; it does not replay stored commands, reuse a license approval
as permission for a different plan, or automatically run scientific analysis.
Verified resources and environments are reused. The pending marker is cleared
only after the final readiness check succeeds.

For Agent-assisted installation, the complete repository Skill includes
`prepare_in_slicer.py`: a plan/export and GUI preparation helper that keeps the
real confirmation dialog. It never starts patient analysis or accepts terms
automatically. See the [Agent setup reference](https://github.com/NeuroDMRI/SlicerDSpace/blob/main/.agents/skills/slicerdspace/references/setup.md).

---

## Maintainer Interface

`scripts/setup_selected.py --targets <targets>` prints the same plan as the GUI;
`--execute` runs it and `--accept-licences` accepts every licence in the plan.
Run it with `PythonSlicer`. `scripts/setup_managed_runtime.py` plans by default; `--execute` performs setup.
FSL requires `--accept-fsl-license`; PNL requires `--accept-pnl-terms`.
`--repair` explicitly reinstalls owned selected prefixes. `--root` and
`SLICERDSPACE_RUNTIME_ROOT` support isolated test environments. The worker returns
verified settings; the GUI applies them only after successful completion.
It never modifies the user's shell, GPU driver or an unmanaged Conda prefix.
