<div align="center">

# Linux / HPC Setup

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This page is for Linux workstations and HPC environments. Use the Slicer UI for
interactive setup when possible. On clusters, use command-line preflight before
submitting long jobs.

---

## 1. Install 3D Slicer

Download 3D Slicer from <https://www.slicer.org/> or use the Slicer build
provided by your lab/HPC module system.

Install the module using the source/staging instructions in the project README.
Open **Environment and installation** to configure the selected
route; the default compact interface has no automatic first-run wizard.

---

## 2. Choose A Model Asset Folder

For a personal workstation:

```text
~/SlicerDSpace-model-assets
```

For shared HPC storage:

```text
/shared/software/SlicerDSpace-model-assets
```

Command-line equivalent. `PythonSlicer` is Slicer's own Python (`<Slicer>/bin/PythonSlicer`).
No system Python is needed:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /shared/software/SlicerDSpace-model-assets \
  --source url \
  --source-id slicerdsurfer_models_git
```

---

## 3. Prepare Subject Data

Recommended layout:

```text
/data/project/sub-01/
  input/
    data.nii.gz
    bvals
    bvecs
    nodif_brain_mask.nii.gz
```

---

## 4. External Tools

External tools are usually managed by the system, conda, Spack or environment
modules. SlicerDSpace scans saved paths, environment variables, `PATH` and
common install locations.

Common examples:

```text
MRtrix bin: /opt/mrtrix3/bin
FSL bin: /usr/local/fsl/bin
FreeSurfer home: /opt/freesurfer
WMA root: /shared/software/whitematteranalysis
TractCloud root: /shared/software/TractCloud
```

If your HPC uses modules, load them before opening Slicer or before running the
CLI:

```bash
module load mrtrix3
module load fsl
module load freesurfer
```

On Linux the PNL brain mask uses the managed route (`--pnl-mask-route auto`
resolves to `managed`: a conda runtime with ANTs 2.3.0). After you update to
this version, each deployment regenerates the brain masks and the preprocessing
(eddy included) once, because the mask signature now holds the route. See the
[7 November release note](../release/PORTABLE_SETUP_20261107.md).

---

## 5. Command-Line Preflight

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /data/project/sub-01 \
  --targets validation,microstructure,tractography \
  --tractography-backend dti_det \
  --doctor
```

Do not submit a long job until preflight says the selected targets are runnable.

---

## 6. If Something Fails

Click `Copy diagnostic report` in SlicerDSpace, or attach the generated
`environment_doctor.json` from:

```text
<subject-root>/outputs/00_run_metadata/environment/
```

