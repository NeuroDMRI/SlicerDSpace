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

| Check | Linux | Windows | macOS |
| --- | --- | --- | --- |
| Resource/CMake contract | Local evidence; CI configured | CI configured; execution evidence needed | CI configured; execution evidence needed |
| Real Slicer GUI/source install | Dated local records | Not certified | Not certified |
| Real SDK/CTest/CPack | Linux/Slicer 5.10 candidate verified 2026-10-22 | Requires real SDK acceptance | Requires real SDK acceptance |
| Scientific workflows | Only recorded routes/data | Native not certified; WSL setup is not native acceptance | Not certified |
| Advanced backends | Only recorded routes | Tool/WSL limitations | Tool-specific limitations |

The Linux record includes a real CPack archive and local installation with
isolated settings; see the [official integration record](../release/OFFICIAL_SLICER_VALIDATION_20261022.md).
It does not establish acceptance of later changes, a clean machine, online
store installation or other platforms. Resource checks are not three-platform
Slicer builds. See the [release gate](../release/EXTENSION_READINESS.md).

---

## Feature Routes

**Native candidate** describes source feasibility, not native scientific acceptance.
WSL is a Linux backend, not native Windows computation. Windows/macOS do not
inherit Linux QC or timings.

| Feature | Linux | Windows | macOS |
| --- | --- | --- | --- |
| UI, NIfTI/NRRD, validation, result loading | Scoped tests recorded | Native candidate with imaging packages | Native candidate with imaging packages |
| DICOM conversion | Converter required | Native converter/Slicer candidate | Native converter/Slicer candidate |
| DTI, deterministic tracking | Teem tensor fit in Slicer; DIPY tracking | Same, native candidate | Same, native candidate |
| DDSeg/DDParcel/DDSurfer native reconstruction | Recorded data/routes and matching models | Native PyTorch candidate with usable mask | Native CPU candidate with usable mask and matching models |
| DIPY preprocessing, FW, NODDI | Scoped inputs measured | Native Python candidate; compatible DIPY/AMICO | Native Python candidate; architecture-compatible packages |
| DKI/Python MK-Curve | Specified-subject full-brain checks and timing recorded; invalid voxels excluded | Native numerical candidate; eligible multi-shell data | Native numerical candidate; eligible multi-shell data |
| PNL CNN mask | PyTorch/ANTs route; paired CPU probability/mask checks; Linux x86_64 managed setup | Native route in Windows Slicer's Python (SimpleITK + PyTorch), **experimental**; not run on Windows here | Native route (SimpleITK + PyTorch), **experimental**; no CUDA, Metal unimplemented; unmeasured here |
| DIPY median-Otsu mask | Python-only generation and DTI sample route measured | Native Python candidate; no POSIX tool required, unmeasured here | Native Python candidate, unmeasured here |
| Full FreeSurfer SynthStrip environment | CPU/CUDA measured | **Not native Windows in this wrapper**; full Linux/WSL environment needed | FreeSurfer/Python CPU candidate, unmeasured here |
| Cortical postprocessing | Verified native components or existing FreeSurfer; offline prebuilt installation tested, publication pending; [setup](SURFACE_COMPONENTS.md) | No qualified native prebuilt bundle; WSL/full-install bridge acceptance pending | Existing native FreeSurfer candidate; no qualified prebuilt bundle |
| MRtrix/FSL preprocessing | CPU/CUDA recorded | WSL 2 bridges; device policy runs in actual WSL Python | Native CPU MRtrix/FSL candidate; **no CUDA eddy** |
| CSD/ACT/SIFT2 | External MRtrix route recorded | MRtrix WSL bridges; acceptance pending | Native MRtrix candidate; acceptance pending |
| Full SAF | Python filter/geometry checks on recorded inputs | Windows argv/PATH adaptation added; MRtrix bridges/prepared surfaces; acceptance pending | POSIX/MRtrix with prepared surfaces; acceptance pending |
| WMQL | Rule checks pass; sparse AF is **PARTIAL** | Pinned TractQuerier WSL bridge candidate | Configured author Python backend candidate |
| WMA | 149741 scoped pass; old 100307 coverage failures remain excluded | WSL author shell/Slicer reverse bridge; acceptance pending | Author shell/display/Slicer CLI requirements; acceptance pending |
| TractCloud | Seeded CUDA/repeat checks on recorded inputs | WSL device-selected CPU/CUDA; native Python not certified | External Python CPU candidate; MPS not implemented |
| Python connectome/ROI statistics/VTK previews | Scoped tests recorded | Native candidate given valid input products | Native candidate given valid input products |
| Managed MRtrix/FSL/PNL setup | Linux x86_64 (managed PNL runtime) | Separate Windows/WSL installer for MRtrix3 and FSL (the mask and the Python stages stay in Windows Slicer's Python); PNL uses the native route (PyTorch extension and the brain-mask assets) | MRtrix3 3.0.8 official tarball and FSL CPU runtime, Intel and Apple Silicon, macOS 14.5 or later; PNL uses the native route |

---

## Practical Limits

- Preserve existing reviewed native masks on all platforms. No error-triggered silent algorithm substitution.
- PNL mask route: Linux uses the managed conda runner (ANTs 2.3.0); macOS and Windows use the native route, labelled "native (experimental)". Linux defaults to PNL generation. Fresh Windows configs from the explicit WSL installer default to MRtrix masking; existing defaults are preserved. Without WSL, supply a reviewed mask or explicitly select the Python-only DIPY mask candidate. See [lightweight routes](EXPERIMENTS.md).
- Auto PyTorch uses CUDA when available, otherwise CPU. **Apple MPS is not implemented/validated**; this is not lack of Mac GPU hardware. PNL also selects PyTorch CPU or caller-visible CUDA; MPS is not selected.
- Windows setup defaults to `-Device auto`, detects the GPU inside WSL and verifies actual PyTorch CUDA allocation. Explicit CUDA errors are not retried on CPU.
- FSL requires explicit license acceptance. Installers do not change NVIDIA drivers or install replacement Linux display drivers.
- WSL process-tree cancellation and Windows/macOS scene reintegration require real-host acceptance. Unit/source checks do not prove every Linux descendant stops on Windows cancellation.
- The [full-brain DKI/MK-Curve records](../release/MODULE_RUNTIMES_20261017.md) concern a specified subject; valid-voxel screening is not a full-brain MATLAB comparison or validation of all acquisitions.

The legacy DDSeg entry now handles `.exe` and `Slicer.app/Contents`; macOS
external Python removes Slicer dylib-path contamination. Details:
[platform audit](../release/PLATFORM_AUDIT_20261015.md), [Windows](WINDOWS.md), [macOS](MACOS.md).

---

## Upstream Boundaries

FSL supports Linux/macOS and Windows through WSL; our wrappers still require host acceptance.
[FSL installation](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html),
[FSL Windows](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html).
FreeSurfer's Windows route hosts Linux, not native Windows binaries.
[FreeSurfer requirements](https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements).
PyTorch has MPS support, but this plugin does not yet implement it.
[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html).
