# SlicerDSpace v0.1.0 Release Notes

SlicerDSpace v0.1.0 is the first code/documentation release candidate for the
integrated 3D Slicer workflow for diffusion MRI-derived anatomical
reconstruction, tractography, bundle analysis, connectome construction and
provenance capture.

## Highlights

- Added a manifest-driven model asset download and verification route:
  `docs/assets/MODEL_ASSETS.md`, `models/assets_manifest.json` and
  `scripts/download_model_assets.py`.
- Added explicit model-asset terms governance in `docs/assets/MODEL_ASSET_TERMS.md`.
- Added standard developer and release checks through `Makefile`,
  `pyproject.toml`, release metadata files and release validation scripts.
- Improved dependency handling for Slicer Python, conda environments and
  external runtime paths.
- Added conservative dependency installation behavior: Python packages require
  explicit user action; external tools are never silently installed.
- Split the Slicer UI implementation into smaller helper modules for
  environment panels, run control, status tables, styles, output loading and
  error dialogs.
- Added representative output collection and release validation reporting.

## Validation

The v0.1.0 tag was created from commit
`4f97e4fa518d22042ed1d893223eac4a87e99bda`.

Checks run locally before tagging:

```bash
make test-fast
PythonSlicer scripts/download_model_assets.py --profile core --verify-only
python3 scripts/audit_slicer_extension_metadata.py --json
SLICERDSPACE_RELEASE_SUBJECT_ROOT=/data/chengjin/SlicerDDSurfer-test/100307-release-candidate make release-check
```

Results:

- `make test-fast`: 207 tests passed.
- Model assets: 29 files verified for the `core` profile.
- Extension metadata audit: zero errors and zero warnings.
- `release-check`: `release_ready=True`, `blockers=0`, `warnings=0`.

## Asset And DOI Boundary

This release can be treated as a code/documentation release. Model weights,
templates and validation fixtures remain controlled assets. Local checkout
verification does not grant public redistribution or DOI-archive rights.

Do not include model weights, templates or subject-derived fixtures in a public
DOI archive unless the source, owner, license, redistribution permission and
archive permission for each asset group have been confirmed.

## Known Gaps

- `make test-slicer` requires a local Slicer launcher and `xvfb-run`; it was not
  run in the v0.1.0 tagging environment because `xvfb-run` was unavailable.
- DOI publication requires an external archive service such as Zenodo, Figshare
  or an institutional repository.
- Native Windows and macOS dependency execution should be verified on real
  Windows/macOS hosts before claiming platform-level installation success.
