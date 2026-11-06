<div align="center">

# Dependency Setup Workflow

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

SlicerDSpace treats Python packages, external software and model assets as
different resource classes. They can all be assisted by scripts, but the path
fields mean different things.

---

## Resource Classes

| Resource class | User-facing setting | Automated action | Why the path still matters |
| --- | --- | --- | --- |
| Python packages | Python executable or conda environment | Run `python -m pip install ...` in the selected environment | The path identifies the Python runtime to modify, normally Slicer `PythonSlicer`. |
| External software | Existing install path, such as MRtrix `bin`, FSL `bin` or `FREESURFER_HOME` | Show setup notes; optional scripts may be run only after explicit user action | System tools depend on OS, package manager, permissions, license terms and existing lab environments. |
| Model weights/templates | Model assets root | Download manifest assets, verify checksum and save the selected root | The path is a writable cache outside the extension code repository. |

---

## Recommended UI Flow

1. Open SlicerDSpace from a staged extension or installed module.
2. Use the header language selector for English/Chinese.
3. Open **Advanced settings > Environment and models**; Beginner controls and the automatic wizard are hidden.
4. Choose a `Model assets root`, download or verify
   model assets, and scan the local runtime. Folder selection and download are
   separate actions; cancelling folder selection does not start a download.
5. Click `Check environment dependencies`.
6. Review the capability dashboard. It tells you which targets can run now and
   which targets still need setup.
7. Review the dependency dialog. It separates Python packages, model resources
   and external tools.
8. Click `Install/download auto items` to process safe automatic actions.
9. Use row-level `Install` for a Python package if only one package is missing.
10. Use row-level `Download / choose path` for model resources. Select a user or
   shared `Model assets root`, then let the downloader verify checksums.
11. Use row-level `Choose path / setup` for external software. Select an existing
   installation path or read the setup notes.
12. Re-run preflight. Saved paths are stored in Slicer user settings and restored
   after reopening Slicer.

During a URL model download, the progress
dialog reports the current file, downloaded bytes, total bytes and whether a
partial file is being resumed. Cancel pauses safely and leaves the `.part`
file in the selected asset root for the next attempt.

For a safe installation test without human data, use the synthetic-subject
instructions in [Quick Start](../setup/QUICKSTART.md). The legacy Demo handler
remains available to tests but is not a visible button in the default UI.

---

## What Is Scanned Automatically

Preflight checks saved Slicer settings, explicit UI fields, environment
variables, `PATH` and common install locations. Common paths are used as
discovery hints only; a found dependency is reported, and a missing dependency
is shown with setup guidance.

External software is not installed silently. This avoids mutating a user's OS,
conda installation, lab module environment or licensed FreeSurfer/FSL setup
without consent.

---

## Command-Line Equivalents

Download model assets into a user-selected cache:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /srv/slicerddsurfer/model-assets \
  --source url \
  --source-id slicerdsurfer_models_git
```

Test the public download route without large model files:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile sentinel \
  --asset-root /tmp/slicerdsurfer-model-sentinel \
  --source url \
  --source-id slicerdsurfer_models_git
```

Generate an install plan without changing the system:

```bash
python3 scripts/install_dependencies.py \
  --profile core \
  --model-assets-dir /srv/slicerddsurfer/model-assets \
  --output-dir /tmp/slicerdsurfer-install-plan
```

Install Python packages into a selected Slicer Python:

```bash
python3 scripts/install_dependencies.py \
  --profile core \
  --python-executable /path/to/Slicer/bin/PythonSlicer \
  --execute
```

Build a preflight/run plan with an explicit model asset root:

```bash
python3 shared/run_subject_pipeline_cli.py \
  --subject-root /path/to/sub-01 \
  --targets validation,preprocessing,segmentation,parcellation,surface \
  --model-assets-dir /srv/slicerddsurfer/model-assets \
  --doctor
```

---

## Current Validation Evidence

- `make test-fast` passes locally.
- Related dependency/model tests pass with `python3 -m unittest
  Testing.Python.test_model_assets Testing.Python.test_release_validation
  Testing.Python.test_ui_environment_panel`.
- `Testing/Python/slicer_dependency_workflow_smoke.py` runs in a real Slicer
  session and verifies `Check environment dependencies`, top-level
  `Install/download auto items`, row-level model download/path selection,
  external path selection, non-silent external setup notice and user-settings
  restore.
- `Testing/Python/slicer_extension_install_smoke.py` verifies that the beginner
  onboarding panel is installed in a real staged Slicer module session.

An HTML rendering for local review is kept at
`ignore/DEPENDENCY_SETUP_WORKFLOW.html` and is intentionally not tracked.
