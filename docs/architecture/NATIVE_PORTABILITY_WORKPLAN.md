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
- `.github/workflows/native-runtime-candidates.yml`: manually dispatched three-host Python/VTK unit tests and candidate generation. Candidate artifacts are uploaded to Actions; this workflow must **not** upload approved release assets.

## Windows x86_64: build MRtrix3 executables on a runner

1. Trigger **Native runtime candidates (not a release)** in GitHub Actions. The Windows job runs on `windows-2022` with `msys2/setup-msys2@v2` using `MSYSTEM=MINGW64`. These tools are installed **only on the build runner**.
2. Checkout the exact upstream MRtrix3 `3.0.8` tag under `mrtrix3/`.
3. `scripts/native/build_mrtrix_windows.sh` runs upstream `./configure -nogui` and `./build` and stages the six named C++ CLI programs. `prepare_mrtrix_candidate.py` copies transitive DLLs from the MinGW/MSYS2 directories using `objdump -p`, failing if a non-system DLL is missing. The upstream licence file is carried into the archive.
4. The artifact is **unpublished** and **scientifically unqualified**. Before release, extract to a fresh Windows machine **without MSYS2/WSL installed**, clear build-only PATH entries, and invoke each binary (`mrconvert`, `dwi2fod`, `tckgen`, `tcksift2`, `tck2connectome`, `tckconvert`) on a small synthetic input and actual DWI data.
5. Extend beyond these six tools. `dwi2response`, `5ttgen`, `dwifslpreproc` and other MRtrix Python scripts are not yet covered. `dwifslpreproc` still depends on FSL Eddy/Topup, which lacks an approved native Windows build.

**Windows WMA**: the hardening call now uses Slicer core Model/Transform/ModelStorage, no SlicerDMRI reader. WMA's upstream `wm_apply_ORG_atlas_to_subject.sh` is still a POSIX workflow and must be ported to a native Python orchestration with argv-based process invocation. The native point-scalar measurement method is identified separately and must not be represented as the upstream FiberTractMeasurements tensor algorithm. Add Windows real-host tests of all atlas registration/clustering stages.

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

A green `platform-unit` job proves *only* that portable Python unit tests passed. The Windows and macOS artifact uploads are **build candidates**, not clean-install or scientific acceptance. A missing/blocked build is a failure to satisfy the native-only requirement, not a reason to fall back to WSL.

## Integration on server 93 (2026-11-07)

The source checkout is `/data04/chengjin/SlicerDSpace`. This integration starts
from the patch's exact upstream base `f09110a964697622d57dbefa8499cf45109cd15d`.
The original ZIP and the previous checkout files are retained outside the repo.

WMA hardening now uses only Slicer core Model/Transform classes. This removes
the default WMA setup's SlicerDMRI dependency; WMA's original POSIX atlas
orchestration and existing Linux backends remain in use.

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
