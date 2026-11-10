<div align="center">

# macOS Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This page is for users who want to run SlicerDSpace from 3D Slicer on macOS.
Use the Slicer UI first; use Terminal only when troubleshooting.

**Platform acceptance status**: On **Apple Silicon (`arm64`)** macOS, SlicerDSpace `v0.1.1` has completed full-workflow real-data acceptance on **3D Slicer 5.10.0** (`HCP 100307`) with the unified installer `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` (see the [macOS 5.10.0 Native Acceptance Report](../release/MACOS_NATIVE_ACCEPTANCE_5.10.0.md)). **Physical Intel (`x86_64`) Mac hardware has not been tested and remains unverified.**

See the [feature/platform matrix](SUPPORT_MATRIX.md) for verified routes,
external-tool requirements and explicitly unavailable features.

---

## 1. Install 3D Slicer & SlicerDSpace

1. Download and install **3D Slicer 5.10.0** (macOS `arm64`) from <https://www.slicer.org/>.
2. Download `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` from GitHub Releases.
3. In 3D Slicer, open **View -> Extensions Manager -> Install Extensions -> Install Extension from File...**, select `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`, and restart 3D Slicer.
4. Open **Modules -> Diffusion -> SlicerDSpace**, then use **Environment and installation -> Prepare selected analysis environment** to install required Python packages and download deep-learning model weights on demand. No automatic wizard is shown on startup.

---

## 2. Choose A Model Asset Folder

Recommended example:

```text
/Users/yourname/SlicerDSpace-model-assets
```

Command-line equivalent. `PythonSlicer` is Slicer's own Python
(`/Applications/Slicer.app/Contents/bin/PythonSlicer`). No system Python is needed:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /Users/yourname/SlicerDSpace-model-assets \
  --source url \
  --source-id slicerdsurfer_models_git
```

---

## 3. Prepare Subject Data

Recommended layout:

```text
/Users/yourname/data/sub-01/
  input/
    data.nii.gz
    bvals
    bvecs
    nodif_brain_mask.nii.gz
```

In SlicerDSpace:

1. Set `Subject ID` to `sub-01`.
2. Set `Subject root` to `/Users/yourname/data/sub-01`.
3. Leave `Output directory` empty unless you need a custom output folder.

---

## 4. Built-In Native Runtimes & External Tools

When installed from `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`, the three native binary runtime bundles for Apple Silicon (`darwin-arm64`) are pre-packaged inside `SlicerDSpace/bundled-runtimes/` and discovered automatically without manual path configuration or separate runtime downloads:

- **MRtrix3 `3.0.8`** (`mrtrix-3.0.8-darwin-arm64`, 67 commands + bundled `FreeSurferColorLUT.txt`)
- **FSL Diffusion Preprocessing `2602.0`** (`fsl-2602.0-darwin-arm64`, `topup`, `applytopup`, `eddy_cpu` + `b02b0.cnf`)
- **FreeSurfer Surface Postprocessing Runtime `0.1.0`** (`surface-runtime-darwin-arm64`, 7 native `arm64` surface commands + spherical registration templates; requires a user-supplied FreeSurfer license file `FS_LICENSE` because no personal activation key is bundled)

The unified preparation entry (`Prepare selected analysis environment`) also acquires pinned WMA/WMQL/TractCloud sources and Python wheels without Git, and automatically repairs any missing or corrupted bundled runtime files using `tree_sha256` verification. If you stage from source without `bundled-runtimes/`, on macOS 14.5 or later setup can also download the native runtime bundles or install MRtrix3 3.0.8 and FSL CPU tools into `~/Library/Application Support/SlicerDSpace/runtimes`. Apple Silicon is never silently switched to an Intel/Rosetta environment. Your own architecture-matching MRtrix3, FSL, or FreeSurfer installation is also accepted ([Official macOS FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/macos.html)).

CUDA `eddy` is unavailable on macOS; diffusion preprocessing uses native multi-threaded `eddy_cpu`. Auto PyTorch currently uses CPU because Apple MPS is not yet implemented or certified ([PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)). The PNL brain mask runs on macOS through its native route (SimpleITK registration and PyTorch in Slicer's Python), labelled "native (experimental)"; it also uses CPU and does not select MPS.

Common custom override examples (only needed if overriding the bundled runtimes):

```text
MRtrix bin: /opt/mrtrix3/bin
FSL bin: /usr/local/fsl/bin
FreeSurfer home: /Applications/freesurfer
```

---

## 5. If Something Fails

Click `Copy diagnostic report` in SlicerDSpace and paste it into your support
message. Include the exact target you tried to run.

