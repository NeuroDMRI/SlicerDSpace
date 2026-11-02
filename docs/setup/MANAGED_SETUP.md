<div align="center">

# Managed Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](QUICKSTART.md) | [GPU](GPU_EXECUTION.md) | [中文](MANAGED_SETUP.zh-CN.md)

</div>

---

## Inside Slicer

Select your analyses, then open the separate **Environment and installation** section
and choose **Prepare selected analysis environment**. Review the plan and
approve downloads and applicable licenses. The dependency report uses the same
entry. Checks and final verification run in cancellable background processes.
An already-importable environment skips installation and runs functional checks. Offline import,
repair and manual paths live under Environment > More options, not Advanced
algorithm settings. It prepares required selected items, not every optional backend:

1. Slicer's Extension Manager installs missing SlicerDMRI/PyTorch extensions; UKFTractography is requested only for a selected UKF route.
2. If an extension was installed or enabled, save your work and restart Slicer. Reopening SlicerDSpace offers to continue preparation; extensions are reused.
3. Background workers prepare pinned WMQL/WMA/TractCloud sources without Git, install missing Python wheels, download selected models, and prepare supported isolated tools.
4. Paths are saved after verification. A fresh dependency check and synthetic functional tests determine readiness. A failed functional test prevents analysis until preparation passes again.

Functional failures are retained across restarts for the same configuration.
Source discovery is shared with runtime execution, so verified prepared sources
remain usable if an explicit saved source path is absent. Windows WMA is not
offered a Bash-only source checkout as a substitute for its WSL executable bridge.

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

Optional PNL models/templates, TractCloud inference references, SynthStrip weights
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

---

## Current Coverage

| Component | Managed path |
| --- | --- |
| MRtrix3 3.0.8 | Official prebuilt Conda packages: Linux x86_64; macOS Intel package/installer adaptation, requiring macOS 14.5+, not yet run on a Mac |
| FSL eddy 2602.0 / topup 2203.5 | Linux x86_64, official FSL channel; explicit license approval |
| CUDA eddy 2602.0 | Optional FSL profile; NVIDIA driver required; no full CUDA Toolkit installation |
| DDSurfer / DDSeg / DDParcel assets | Existing background download with profile-specific selection and default directory |
| Python scientific packages | Selected interpreter, protected Slicer core packages, native packaging API when available; official PyTorch helper |
| PNL diffusion CNN | Linux x86_64; pinned official source, v0.3 three-view weights, IIT b0 template, Python 3.11, TensorFlow 2.15.1/Keras 2 and ANTs 2.3.0 in a separate prefix |
| Optional cortical postprocessing | [Qualified prebuilt bundle](SURFACE_COMPONENTS.md), Linux x86_64; offline installation tested. Bundle publication is pending. Personal activation is separate; the GUI never compiles sources |
| WMA / TractCloud / SynthStrip non-code resources | Selected optional profile download/import, checksums, ORG extraction and automatic path saving; private-mirror authorization still applies |
| WMA/WMQL/TractCloud sources | Pinned archives, safe extraction, file hashes, automatic discovery/path saving; no Git or duplicated algorithm implementation |
| Native Windows, macOS ARM, macOS FSL/PNL | No qualified automatic native profiles in this installer; existing Windows/WSL and administrator-provided routes remain |

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

`scripts/setup_managed_runtime.py` plans by default; `--execute` performs setup.
FSL requires `--accept-fsl-license`; PNL requires `--accept-pnl-terms`.
`--repair` explicitly reinstalls owned selected prefixes. `--root` and
`SLICERDSPACE_RUNTIME_ROOT` support isolated test environments. The worker returns
verified settings; the GUI applies them only after successful completion.
It never modifies the user's shell, GPU driver or an unmanaged Conda prefix.
