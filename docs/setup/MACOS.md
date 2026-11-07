<div align="center">

# macOS Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This page is for users who want to run SlicerDSpace from 3D Slicer on macOS.
Use the Slicer UI first; use Terminal only when troubleshooting.

Native macOS scientific execution and real SDK packaging are not certified.
These are setup instructions, not a completed platform acceptance record.

See the [feature/platform matrix](SUPPORT_MATRIX.md) for native candidates,
external-tool requirements and explicitly unavailable features.

---

## 1. Install 3D Slicer

Download and install 3D Slicer from <https://www.slicer.org/>.

Install the module following the project README, open SlicerDSpace, then use
**Environment and installation**. No automatic wizard is shown.

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

## 4. External Tools

MRtrix3, FSL, FreeSurfer, WMA and TractCloud are optional depending on selected
targets. The unified preparation entry can acquire pinned WMA/WMQL/TractCloud
sources and Python wheels without Git. On macOS 14.5 or later, on Intel and
on Apple Silicon, it also installs MRtrix3 3.0.8 (the official universal
tarball, SHA-256 verified) and the FSL CPU tools eddy and topup (the official
FSL conda channel, built for your architecture, with an architecture-matched
Miniforge bootstrap). Apple Silicon is not silently switched to an Intel/Rosetta
environment. Both runtimes go to the runtimes folder next to the model assets,
for example `~/Library/Application Support/SlicerDSpace/runtimes`. If setup
replaces an existing MRtrix3 runtime there, it renames the old folder to
`mrtrix-3.0.8-macos.replaced-<ns>` (about 169 MB). The plugin never deletes it;
you can delete it by hand after a successful run. FreeSurfer and other
unqualified native routes still use an existing compatible installation; Linux
Conda prefixes cannot be copied to macOS. The managed macOS runtimes have
planning and unit tests; real macOS scientific acceptance is still open.
WMA path discovery accepts `Slicer.app/Contents` libraries and installed
SlicerDMRI modules rather than assuming Linux layout. This has app-layout
regression tests, not real macOS scientific acceptance.
Your own FSL installation is still accepted: use an architecture-matching one.
[Official macOS FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/macos.html).

CUDA eddy is unavailable on macOS. MRtrix/FSL correction uses the CPU eddy
executable; Auto PyTorch also currently uses CPU because this plugin has not
implemented MPS. The PNL brain mask runs on macOS through its native route
(SimpleITK registration and PyTorch in Slicer's Python), labelled "native
(experimental)"; it also uses CPU and does not select MPS. [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html).

For the least external dependencies, provide a reviewed native brain mask and
select DTI without the `preprocessing` target (it needs FSL and MRtrix3). FW/NODDI/DKI/MK-Curve additionally
need suitable acquisition and compatible numerical packages; they remain
unverified on real macOS data. WMA, SAF and FreeSurfer-style postprocessing need
their shell/CLI/display prerequisites and separate end-to-end validation.

Common examples:

```text
MRtrix bin: /opt/mrtrix3/bin
FSL bin: /usr/local/fsl/bin
FreeSurfer home: /Applications/freesurfer
```

---

## 5. If Something Fails

Click `Copy diagnostic report` in SlicerDSpace and paste it into your support
message. Include the exact target you tried to run.

