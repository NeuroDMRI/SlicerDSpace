<div align="center">

# SlicerDSpace native-portability implementation and acceptance

[Project](../../README.md) | [Documentation](../README.md) | [Platform support](../setup/SUPPORT_MATRIX.md)

</div>

---

**Status: engineering work in progress. This document does not certify any new platform.**

Goal: one SlicerDSpace GUI and one algorithm configuration schema on Windows, macOS and Linux; no end-user WSL, Miniconda/Anaconda, MSYS2 or compiler setup. Build-time dependencies installed on CI runners **do not** become user prerequisites.

## Existing infrastructure

- `shared/native_runtimes.py`: safe, SHA256-checked installer for precompiled platform-native archives, with explicit licence acceptance. No WSL fallback.
- `models/native_runtime_bundles.json`: intentionally empty until an accepted archive exists. An empty list is **not** an installable MRtrix/FSL/FreeSurfer backend.
- `.github/workflows/native-runtime-candidates.yml` runs Python/VTK unit tests and builds candidates on manual dispatch. The workflow uploads the build and relocation reports.

## Windows x86_64: build MRtrix3 executables on a runner

1. Trigger **Native runtime candidates (not a release)** in GitHub Actions. The Windows job runs on `windows-2022` with `msys2/setup-msys2@v2` using `MSYSTEM=MINGW64`. These tools are installed **only on the build runner**.
2. Checkout the exact upstream MRtrix3 `3.0.8` tag under `mrtrix3/`.
3. `scripts/native/build_mrtrix_windows.sh` runs upstream `./configure -nogui` and `./build`. The Windows candidate stages 26 executable commands and the Dhollander `dwi2response` entry point. `prepare_mrtrix_candidate.py` copies transitive DLLs using `objdump -p` and includes the upstream licence file.
4. The local candidate completed Dhollander, MSMT-CSD, ACT and SIFT2 on synthetic DWI in Wine. The Windows runner tests compilation and command startup after relocation. Release qualification still needs a clean Windows installation and a comparison on real DWI data.
5. `5ttgen` and `dwifslpreproc` remain outside this candidate. The unified FSL correction pilot invokes the native programs directly.

The native WMA ORG runner uses Slicer core to harden transforms. Windows end-to-end testing still needs to cover registration, clustering and anatomical tract output.

## macOS arm64 and x86_64

1. The macOS matrix (`macos-15`, `macos-15-intel`) invokes `scripts/native/build_mrtrix_macos.sh`. This is an **upstream native binary repackaging workflow**, not a source compilation: it fetches the MRtrix3 3.0.8 universal tarball and verifies its pinned SHA256 before staging.
2. `prepare_mrtrix_candidate.py` packages the target binaries with the original licence and platform library tree. Each architecture is labelled only on the matching host.
3. Before qualification, run relocatability, library/code-signing checks and end-to-end ACT/iFOD2/SIFT2/Connectome jobs on physical macOS hosts. Apple Silicon GPU/Metal and Intel differences must be evaluated separately; code compilation is not a substitute for scientific parity.
4. The existing macOS FSL CPU managed runtime remains separate, and FreeSurfer's 7-command postprocessing bundle is **not** approved for macOS by this patch.

## Linux x86_64

Keep the existing managed original MRtrix/FSL routes as the scientific baseline; do not delete them until native packaging on other hosts has passed paired-image tests.

## Release gates (all required, not currently met)

- Clean Slicer install on Windows/macOS/Linux; no end-user build environment, WSL or Conda requirement. Test with no PATH to these tools.
- All targets and settings present with the same workflow semantics. No silent algorithm swaps.
- Cross-OS model inference and data import; original-core algorithm versions and parameters pinned.
- Native `dwi2response`, `5ttgen`, FSL Eddy/Topup equivalent scientific method and FreeSurfer postprocessors qualified or explicitly reimplemented with a new method name and parity experiment.
- WMA: end-to-end 800-cluster transform and anatomical tracts, optional scalar/tensor metrics matched against the original FiberTractMeasurements CLI on real inputs.
- CPU/GPU choices, numerical tolerances, timestamps, SHA256, licences and QC records reviewed before enabling `published:true` in any catalog.
- Slicer SDK Extension Manager install/restart, cancellation, cleanup and release package checks validated on each host.

### How to read the CI results

A green `platform-unit` job records a portable Python test result. The Windows and macOS reports record candidate build, packaging and relocation results. Clean installation and scientific acceptance have separate release gates.

## Integration on server 93 (2026-11-07)

The source checkout is `/data04/chengjin/SlicerDSpace`. This integration starts
from the patch's exact upstream base `f09110a964697622d57dbefa8499cf45109cd15d`.
The original ZIP and the previous checkout files are retained outside the repo.

WMA hardening now uses only Slicer core Model/Transform classes. This removes
the default WMA setup's SlicerDMRI dependency.

The ORG workflow also has a native Python runner. It calls the pinned WMA
scripts with Slicer's PythonSlicer and uses a separate Slicer process to harden
transforms. Windows selects this runner when no WMA WSL bridge is configured.
Linux and macOS default to the author shell. Set
`DSP_WMA_EXECUTION_BACKEND=native` to exercise the native runner on any platform,
or set it to `legacy` to use the author shell or configured WSL bridge. Native
runs record each stage command and bypass the POSIX registration cache. Resume
checks the recorded execution backend before reusing results. East-West, NABA
and the FiberTractMeasurements export continue to use the legacy runner.

The optional measurement methods have separate selection. In
`WMAOrgAtlasConfig`, `export_diffusion_measures=True` retains the existing
`measurement_backend="fibertractmeasurements"` default and its explicitly
configured `fiber_tract_measurements_cli`. To request the experimental scalar
summary in Python, set both:

```python
export_diffusion_measures=True,
measurement_backend="native_vtk_point_scalar_summary",
```

The latter requires no FiberTractMeasurements CLI, rejects tensor arrays, records
its method in the manifest and sidecar, and does not certify equivalence to the
original tensor measurements. The GUI does not silently opt users into it.
Failures in this export prevent publication of the run's staged products.

The ZIP's application script required a newline-escape correction before it
could match the pinned source. Integration also fixes the candidate smoke
script's import path for direct CLI use. The workflow follows this repository's
manual-only Actions policy, including the portable unit jobs.
A pre-existing WMA test fixture now invokes the PythonSlicer launcher instead of
symlinking to `python-real`, which cannot start in a cleaned environment without
Slicer's library paths.

Real Slicer 5.10 checks on Linux cover forward/inverse VTK and VTP transforms,
point scalar preservation, unchanged empty clusters and removal of created MRML
nodes with SlicerDMRI unloaded. These are synthetic geometry checks, not a full
atlas or cross-platform scientific validation. Native bundle publication stays
disabled until the release gates above pass.

## Unified FSL correction pilot

DIPY MP-PCA, Gibbs removal, phase-encoding resolution and output management
use the existing preprocessing entry point. The default correction runner is
MRtrix's dwifslpreproc. The experimental runner uses native Eddy CPU on all
hosts and requires Topup, Applytopup and b02b0.cnf only for paired phase encoding.
It uses MRtrix image commands to prepare the FSL input and export the rotated
gradients. It writes a separate cache identity and records the executed commands.

Select the pilot with --preprocessing-execution-backend native or set
DSP_PREPROCESSING_EXECUTION_BACKEND=native before launching Slicer. Point the
preprocessing binary-directory settings at native MRtrix and FSL installations.
The pilot needs same-version correction and end-to-end validation before any
default change. macOS Intel and Apple Silicon need separate runtime checks.

Local validation on server 93 used a 66-volume synthetic DWI. The Windows x64
Topup 2203.5 and Eddy CPU 2602.0 build completed all 16 correction stages in
Wine. On Linux, both dispatchers generated identical Eddy inputs with and
without reverse phase encoding. Fixed-seed Eddy replays produced identical
outputs for both cases. The default unseeded runs differed. The separate Windows
Topup 2203.6 build passed a voxel-spacing regression against the same Linux
version. Windows host and real DWI acceptance are still pending.
