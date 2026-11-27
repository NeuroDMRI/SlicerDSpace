# SlicerDSpace v1.0.0

**Supported 3D Slicer versions:** **3D Slicer 5.10.x / 5.12.x**

SlicerDSpace is a unified 3D Slicer extension for T1-weighted-image-free anatomical reconstruction, diffusion MRI microstructural mapping, tractography, white-matter fiber bundle analysis, and structural connectomics—bringing preprocessing, deep-learning anatomical priors, tracking, and scene-native visualization into a single module.

## Overview

- **Input Preparation & Brain Masking**: DICOM, NIfTI, and NRRD/NHDR import with gradient checks, PNL CNN / BET / Otsu brain masking, and optional DWI denoising, Gibbs ringing removal, and Eddy/Topup correction.
- **Microstructural Mapping**: DTI (FA, MD, AD, RD, color FA), Free-Water / iFW, AMICO NODDI, and DKI (Python MK-Curve).
- **Anatomical Reconstruction (T1w-Free)**: DDSeg tissue segmentation, DDParcel 101-ROI brain parcellation, DDSurfer white/pial cortical surface reconstruction, and optional FreeSurfer surface postprocessing.
- **Tractography, Bundles & Connectome**: DTI deterministic tracking, UKFTractography, MRtrix3 CSD/ACT with SIFT2, SAF short association fibers, WMQL / WMA / TractCloud bundle extraction, and structural connectivity matrices.
- **Cross-Platform & Modular Delivery**: Supports Windows (`x64`), Linux (`x64`), and macOS (Apple Silicon `arm64`) with slimmed OS runtime archives, shared cross-platform model-weight packs, and guided in-Slicer CUDA PyTorch setup on NVIDIA GPU systems.

## Installation

### Option A: One-Click Graphical Setup Extension (Recommended)

1. Download the Setup `.zip` archive for your operating system from **Assets** below (or customize mirror / proxy / offline directory on the [SlicerDSpace website](https://neurodmri.github.io/SlicerDSpace/#start)):
   - **Windows (`x86_64`)**: `SlicerDSpaceSetup-Windows.zip`
   - **Linux (`x86_64`)**: `SlicerDSpaceSetup-Linux.zip`
   - **macOS (Apple Silicon `arm64`)**: `SlicerDSpaceSetup-macOS.zip`
2. Open **3D Slicer 5.10.x / 5.12.x**, go to **Extensions Manager -> Install Extension from File...**, select the downloaded `.zip` file, and restart Slicer when prompted.
3. After restarting, the **SlicerDSpace Setup** window opens automatically. Click **Start Install** to download and verify the OS runtime archive, core model packs, and required templates. Once complete, restart Slicer and open **Modules -> Diffusion -> SlicerDSpace**.

### Option B: Standalone OS Base Archives + Shared Model & Atlas Packs

- **OS Base Packages (`depends NA`, bundled Python 3.12 + PyTorch + UKF + native MRtrix3 / FSLcpu / FreeSurfer Surface Runtime)**:
  - **Windows (`x86_64`)**: `SlicerDSpace-v1.0.0-Windows-x64.zip`
  - **Linux (`x86_64`)**: `SlicerDSpace-v1.0.0-Linux-x64.tar.gz`
  - **macOS (Apple Silicon `arm64`)**: `SlicerDSpace-v1.0.0-macOS-arm64.tar.gz`
- **Shared Cross-Platform Model Packs, Atlases & Example Data**:
  - Cross-platform modular model weight packs (`SlicerDSpace-Model-DDSurfer-v1.tar.gz`, `SlicerDSpace-Model-DDSeg-v1.tar.gz`, `SlicerDSpace-Model-DDParcel-v1.tar.gz`, `SlicerDSpace-Model-PNL-CNN-v1.tar.gz`, `SlicerDSpace-Model-TractCloud-v1.tar.gz`), optional WMA white-matter atlases (`ORG`, `NABA`, `East-West`), prebuilt runtimes, and the skull-intact Sherbrooke example DWI dataset (`SlicerDSpace-Example-Sherbrooke-NRRD.zip`, DWI-only for automatic PNL CNN brain masking) are hosted in the shared [`assets`](https://github.com/NeuroDMRI/SlicerDSpace/releases/tag/assets) release.
  - Option A downloads and verifies all core model packs and the NITRC `IITmean_b0_256.nii.gz` template automatically; optional WMA atlases can be downloaded on demand inside Slicer.
- **Incremental Updates & Verification**:
  - Existing installations can apply code updates from **Modules -> Diffusion -> SlicerDSpace -> Update -> Check for updates** (`SlicerDSpace-v1.0.0-CodeOverlay.tar.gz`).
  - Verify release archive integrity against `SHA256SUMS.txt`.
