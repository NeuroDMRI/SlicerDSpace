# SlicerDSpace v0.1.1

**Required 3D Slicer version:** **3D Slicer 5.10.0**

## Highlights & Improvements

- **Apple Silicon MPS Acceleration & Graceful Fallback**: Native Metal (`scatter_`) unpooling in DDParcel (`CompetitiveDecoderBlock`), MPS unified-memory slice-batch scaling and single-slice OOM retry in DDParcel / DDSeg / DDSurfer / TractCloud / PNL brain mask, `PYTORCH_ENABLE_MPS_FALLBACK=1` preset, float32 MPS FFT path in DIPY Gibbs removal, end-to-end `compute_device` propagation across orchestrator and cohort scheduler, and graceful CPU fallback for classical/non-MPS stages. *(Native Intel `x86_64` Macs are unsupported by the macOS bundle; Apple Silicon M1–M4 is required.)*
- **Cross-Platform Hardening (Windows / Linux / macOS)**:
  - **Windows `x64`**: Console-window suppression (`CREATE_NO_WINDOW`) and full `taskkill /F /T /PID` process-tree cancellation across Surface/MRtrix/compute stages, UTF-8 decoding across subprocesses and JSON/gradient I/O, UTF-8 BOM + `(N, 3)` `.bvec` auto-transpose on import, Windows Registry (`WinINet`) proxy parsing, and BLAS thread-oversubscription prevention in AMICO NODDI.
  - **macOS (`arm64`)**: Space-free default managed runtime root (`~/.local/share/SlicerDSpace/runtimes`), Gatekeeper quarantine probe across extracted `.so`/`.dylib` payloads, `DYLD_*` library isolation inside FreeSurfer postprocessing, `scutil` system proxy support, and Intel vs Rosetta detection in the installer.
  - **Linux (`x64`) & Installer / On-Demand Resources**: `/tmp` disk-space check with automatic `~/.cache` fallback, HTTP `400`/`405`/`501`/timeout Range probe fallback to streaming download, `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `certifi` SSL context support, fixed on-demand Surface Runtime detection, and preserved Chinese (`zh`) UI language setting.

## Installation

### Option A: One-Click Graphical Setup Extension (Recommended)

1. Download the Setup `.zip` archive for your operating system from **Assets** below (or customize mirror / proxy / offline directory / WMA atlases on the [SlicerDSpace website](https://neurodmri.github.io/SlicerDSpace/#start)):
   - **macOS (Apple Silicon `arm64`)**: `SlicerDSpaceSetup-macOS.zip`
   - **Windows (`x86_64`)**: `SlicerDSpaceSetup-Windows.zip`
   - **Linux (`x86_64`)**: `SlicerDSpaceSetup-Linux.zip`
2. Open **3D Slicer 5.10.0**, go to **Extensions Manager -> Install Extension from File...**, select the downloaded `.zip` file, and restart Slicer when prompted.
3. After restarting, the **SlicerDSpace Setup** window opens automatically. Review your options (or optional WMA white-matter atlases: `ORG` checked by default, `NABA` and `East-West` optional) and click **Start Install**. Once installation finishes, restart Slicer and open **Modules -> Diffusion -> SlicerDSpace**.

> **Note (2026-11-22):** The setup published after this release installs the core only (no WMA atlas is preselected); when you select WMA, SlicerDSpace offers the atlas download inside the plugin.

### Option B: Standalone Offline OS Base Archives + Optional Assets

- **OS Base Packages (`depends NA`, bundled Python 3.12 + PyTorch + core weights + UKF + native MRtrix3 / FSLcpu / FreeSurfer Surface Runtime)**:
  - **Windows (`x86_64`)**: `SlicerDSpace-v0.1.1-Windows-x64.zip`
  - **Linux (`x86_64`)**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`
  - **macOS (Apple Silicon `arm64`)**: `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`
- **Optional Assets & Example Data**:
  - WMA Atlases (`SlicerDSpace-v0.1.1-WMA-ORG.tar.gz`, split `WMA-NABA.tar.gz.001..002`, split `WMA-EastWest.tar.gz.001..007`) — omitted atlases and missing NITRC `IITmean_b0_256.nii.gz` can be downloaded or imported on demand inside Slicer.
  - Preprocessed HCP Example DWI: `SlicerDSpace-Example-HCP-100307-NIfTI.zip` (`1.17 GiB`).
  - Verify integrity against `SHA256SUMS.txt`. Check the pinned **[FAQ]** issue on GitHub Issues before opening a support request.
