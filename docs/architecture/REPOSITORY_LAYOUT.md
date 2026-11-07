<div align="center">

# Repository Layout

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This repository is organized as a 3D Slicer scripted extension with a stable
top-level module entry point and Python package directories for the processing
domains.

---

## Root Directory

The root directory should stay small because it is the first view users see on
GitHub and because 3D Slicer scans top-level scripted-module files.

Keep only:

- `SlicerDSpace.py`: the Slicer scripted-module entry point.
- `SlicerDSpace.s4ext`: Slicer extension metadata.
- `CMakeLists.txt`: extension packaging.
- `README.md` and `README.zh-CN.md`: short user entry points.
- `LICENSE.txt`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `codemeta.json`,
  `.zenodo.json`, `.gitattributes`, `.gitignore`: licensing and project metadata.
- `Makefile`, `pyproject.toml`: development and test configuration.
- `run_subject_pipeline.py`: stable CLI entry point documented for users.
- `run_postprocess_docker.py`: thin compatibility entry point; implementation
  lives in `anatomy/surface/postprocess_docker.py`.
- `workflow_catalog.py`: stable source-tree compatibility entry point for
  workflow metadata in `shared/workflow_catalog.py`. It is intentionally not
  installed as a top-level Slicer module resource.

Do not add generated reports, manuscripts, release drafts, validation outputs,
or long design notes to the root directory. Put durable documentation under
`docs/`; put local reports and generated evidence under ignored `ignore/`.

The root compatibility files contain no processing implementations. Existing
`python3 run_subject_pipeline.py ...`, `python3 run_postprocess_docker.py ...`
and `import workflow_catalog` calls remain supported.

---

## Documentation

- `docs/assets/`: model-asset manifest, download policy, redistribution notes.
- `docs/setup/`: dependency and installation guidance.
- `docs/release/`: release workflow, release notes and DOI/archive guidance.
- `docs/architecture/`: design and repository-structure notes.

This mirrors the SlicerDMRI pattern of keeping project documentation in a
dedicated documentation tree instead of crowding the repository root.

---

## Source Packages

Files are retained according to runtime, packaging, testing, compatibility and
licensing needs, not their last-edit date. The 2026-10-11 unused-source cleanup
removed the historical May copy, superseded surface networks and unused
parcellation training helpers. See [source hygiene](SOURCE_HYGIENE.md) for the
evidence and retained-file policy.

- `shared/`: common Slicer UI, dependency, IO, reporting and orchestration code.
- `preprocessing/`: diffusion preprocessing (DIPY denoising and Gibbs correction,
  FSL eddy/topup and associated QC). Input import and standardisation use
  `shared/input_import.py` with the stable `scripts/prepare_inputs.py` CLI.
- `anatomy/`: anatomical reconstruction, grouped into `segmentation/` (DDSeg),
  `parcellation/` (DDParcel) and `surface/` (DDSurfer). The pinned upstream
  DDSurfer snapshot and surface postprocessing also live inside this group.
- `microstructure/`: DTI, Free-water, NODDI and regional quantitative summaries
  share one pipeline and Slicer facade; they are not separate root packages.
- `tractography/`: tracking, SAF and bundle analysis.
- `connectome/`: structural connectivity and network visualisation.
- `models/assets_manifest.json`: canonical model/template asset manifest.

The scientific source tree is:

```text
preprocessing/
anatomy/
  segmentation/
  parcellation/
  surface/
microstructure/
tractography/
connectome/
```

Python integrations now import `anatomy.segmentation`, `anatomy.parcellation`
and `anatomy.surface`. No duplicate root packages or directory symlinks are
retained. CMake packaging, local extension staging, dynamic imports and the
workflow catalog all use these paths. The root Slicer and CLI entry points
remain unchanged.

This is a source-only reorganisation: scientific task IDs (for example,
`surface.ddsurfer`), subject output directories and the independent model-asset
namespace (`surface/weights`, `surface/template`, `segmentation/weights`,
`parcellation/weights`) deliberately retain their original names. Existing
weights, result manifests and QC records do not need conversion.

An existing copied extension must be restaged; it does not update itself when
the source tree changes. Use `scripts/stage_slicer_extension.py --clean` for a
managed installation to remove the old flat packages, then restart Slicer.
Back up any manually added files before cleaning the installation; external
model assets and subject outputs should never be stored inside that disposable
installation directory.

---

## Developer Tools And Local Outputs

### Distribution Boundary

Repository contents and installed plugin contents are different. The shared
runtime manifest packages processing code, licenses, user-facing guides and
explicit release/reference files, not the entire documentation tree.
Website source, dated performance/QC reports, benchmark JSON and timing scripts
stay in the full source repository rather than the user installation.
Three module-runtime CSV files are intentionally retained as small references.
See [download scope](../setup/DOWNLOADS.md) for an optional sparse/partial source
checkout. A normal `git clone` still includes tracked reports and website files;
neither `.gitignore` nor package exclusions change that behavior.
No historical evidence, commit history or contributor attribution is deleted.

### Local Material

- `scripts/`: command-line maintenance, installation, release and audit tools.
- `scripts/dev/requirements-ci.txt`: pinned fast-test dependencies, separate
  from scientific runtime installation. Install with
  `python3 -m pip install -r scripts/dev/requirements-ci.txt`.
- `ignore/releases/<tag>/`: generated release manifests, lockfiles and copied
  validation reports. `scripts/build_release_artifacts.py` writes here by
  default; `scripts/create_github_release.py` consumes the same location.
  The uploader can still read an existing legacy `release_artifacts/<tag>/`
  workspace if the new tag directory does not exist.
- `ignore/cache/`: pytest cache and Python bytecode produced through `make`.
  Direct Python invocations may create ignored `__pycache__/` directories;
  they are disposable caches, not source files.
- `ignore/archives/`: local material moved aside without deleting it.
- `ignore/validation/` and `ignore/local_reports/`: reproducibility evidence
  and local diagnostics. They are not shipped as runtime code.
- `website/`: public presentation assets, separate from the Slicer module.

`make clean-local` removes caches only. It does not remove release evidence,
scientific outputs, local reports or a configured smoke-test workspace.
The 2026-10-10 root cleanup archived the existing caches and moved the release
directory without altering its contents or regenerating QC records.

---

## Resources And Assets

- `Resources/Icons/` and `Resources/UI/` are normal Slicer extension resources.
- Large model weights and templates are not tracked in this code repository.
  They are restored from the independent `SlicerDSpace-models` repository or
  another manifest-compatible mirror.
- Keep only `models/assets_manifest.json`, checksums and download tooling in
  this code repository.

---

## Testing

- `Testing/Python/test_*.py` contains unittest-discovered tests.
- `Testing/Python/run_*_smoke.py`, `slicer_*_smoke.py`, `summarize_*.py` and
  related helpers are manual or release-candidate smoke tools.

The flat `Testing/Python` layout is retained for simple unittest discovery and
Slicer extension compatibility. If the helper set grows further, split it into
`Testing/Python/smoke/` and `Testing/Python/tools/` in a dedicated refactor.

---

## Third-Party Or Reference Code

- `tractography/saf_backend/` is bundled runtime code used by SAF tracking
  (MIT License; from https://github.com/dmitrishastin/SAF).
- `tractography/slicer_dmri_reference/` is reference material and should not be
  packaged as runtime extension code. Both CMake builds and local extension
  staging explicitly exclude it; `tractography/saf_backend/` remains packaged.
- `anatomy/parcellation/models/` and `anatomy/parcellation/data_loader/` are derived from
  FastSurfer (Apache License 2.0; https://github.com/Deep-MI/FastSurfer).
- `anatomy/parcellation/conversion/` is attributed in-repo to
  https://github.com/pnlbwh/conversion; its license is not recorded in this
  repository and must be confirmed against that project before redistribution.
- `anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt` is copied from
  FreeSurfer and is governed by the FreeSurfer Software License, not this
  repository's BSD-3-Clause license.
- `third_party/` is ignored and must not be used for packaged runtime content.

Workflow packages expose `SUPPORTED_WORKFLOWS`. The older
`PLANNED_WORKFLOWS` names remain aliases for downstream compatibility, but the
root catalog now marks tractography and connectome as implemented because their
Python/MRtrix execution paths are part of the product pipeline.

Any bundled third-party code must include provenance and license information;
see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) at the repository
root for the full inventory.
