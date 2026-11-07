# Dependency Workflow Release Checklist

This checklist tracks the dependency setup and model-asset workflow that must be
verified before a public SlicerDDSurfer release.

## Scope

- Fresh clone can run the repository smoke checks without vendored model weights.
- Missing model assets produce a clear prompt and a documented download command.
- Model assets can be downloaded into a user-selected asset root, not the Slicer
  extension code directory.
- Slicer UI preflight distinguishes Python packages, external software and model
  resources.
- Per-item UI actions persist selected paths and survive Slicer restart.
- A minimal subject run can build and review a processing plan with the selected
  asset root.
- User-facing setup documentation is available in
  `docs/assets/DEPENDENCY_SETUP_WORKFLOW.md`; the local HTML rendering is kept
  under `ignore/` and is not tracked by git.

## Dependency Policy

| Resource class | User-facing setting | Automated action | Safety rule |
| --- | --- | --- | --- |
| Python packages | Python executable or conda environment | `python -m pip install ...` into the selected environment | Default target is Slicer Python; never install into an unrelated system Python unless selected. |
| External software | Existing install path, such as MRtrix `bin` or `FREESURFER_HOME` | Generate setup notes; optional installer scripts may be run only after explicit user action | Do not silently run apt, brew, conda, source builds or license-restricted setup. |
| Model weights/templates | Model assets root | Download manifest assets, verify size and checksum, then save the selected root | Keep large assets outside the extension repository by default. |

## Current Local Validation

| Check | Status | Evidence |
| --- | --- | --- |
| Repository unit tests | Passed | `make test-fast` passed with 257 tests after adding the dependency-workflow smoke and user documentation. |
| Slicer extension smoke | Passed | Slicer 5.10 loaded the staged extension and reported `SLICERDDSURFER_EXTENSION_TARGET_REGISTERED=True`, `LOGIC_READY=True`, `ENV_PATHS_READY=True`, `MODEL_ASSETS_READY=True`. |
| Dependency UI smoke | Passed | `Testing/Python/slicer_dependency_workflow_smoke.py` exercised the dependency check entry point, top-level `Install/download auto items`, per-item model download/path selection, MRtrix path selection, external-tool setup notice and user-settings restore in a real Slicer session. |
| Fresh clone smoke | Passed | Fresh clone smoke was run in plan mode after moving model assets out of the code repository. |
| GitHub push | Passed | `origin/main` was updated to commit `891d5f210e0d37961895911a24cfabd5c9715318`. |
| GitHub Actions readback | Not verified from this host | GitHub Actions API and connector returned 404 for this repository/commit in the current environment; verify in the GitHub web UI or with an authenticated `gh` CLI. |

## Commands To Re-run Before Tagging

```bash
cd /data/chengjin/SlicerDDSurfer
make test-fast
git diff --check
```

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /srv/slicerddsurfer/model-assets \
  --source url \
  --source-id slicerdsurfer_models_git \
  --verify-only
```

```bash
PythonSlicer scripts/install_dependencies.py \
  --profile core \
  --model-assets-dir /srv/slicerddsurfer/model-assets \
  --output-dir /tmp/slicerdsurfer-install-plan
```

```bash
python3 scripts/stage_slicer_extension.py \
  --install-root /tmp/slicerdsurfer-extension-release-smoke \
  --clean \
  --link-mode copy
```

```bash
DISPLAY=:99 \
LD_LIBRARY_PATH=/data/chengjin/software/libpcre2-16-0/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
SLICERDDSURFER_EXTENSION_MODULE_DIR=/tmp/slicerdsurfer-extension-release-smoke/SlicerDDSurfer/lib/Slicer-5.10/qt-scripted-modules \
/data/chengjin/software/Slicer-5.10.0-linux-amd64/Slicer \
  --no-splash \
  --disable-cli-modules \
  --additional-module-paths /tmp/slicerdsurfer-extension-release-smoke/SlicerDDSurfer/lib/Slicer-5.10/qt-scripted-modules \
  --python-script Testing/Python/slicer_dependency_workflow_smoke.py
```

## Manual UI Checks

- Open SlicerDDSurfer from a staged extension, not from the repository root.
- Click `Check environment dependencies`.
- Confirm the dialog shows separate actions for Python packages, external tools
  and model resources.
- Select a custom `Model assets root`, download or verify assets, close and
  reopen Slicer, then confirm the path is restored.
- Select an MRtrix/FSL/FreeSurfer/WMA/TractCloud path and confirm preflight
  reruns without silently installing system software.
- Build a minimal run plan with `--model-assets-dir` or the UI model-root field.

## Release Gate

Do not tag a release until GitHub Actions is green and at least one real model
asset download has been verified against the public asset host.
