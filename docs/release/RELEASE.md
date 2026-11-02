<div align="center">

# SlicerDSpace Release and Archive Checklist

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This checklist defines the release path for code, model assets, validation
metadata, and DOI-backed archival records. It is intended for maintainers who
prepare a public release or a manuscript-linked release.

---

## Release Scope

User extension packages are distinct from source/DOI archives. Build extension
resources through the shared runtime manifest: keep runtime, third-party
licenses and user guides, but omit website source, historical optimization
reports, raw benchmark JSON and timing scripts. Keep those records in the
complete source/DOI release for audit and reproducibility. Explicit release
documents and the three small module-runtime CSV references remain packaged.
The [download guide](../setup/DOWNLOADS.md) distinguishes validated CPack
packages, local staging packages and optional lightweight source checkouts.
Do not label GitHub's automatic source archive as an installable extension.

SlicerDSpace release artifacts are split into three classes:

1. Source code and documentation in this repository.
2. Manifest-managed model weights and templates in the independent
   `SlicerDSpace-models` repository, downloaded to a user-selected asset root
   and verified by size and SHA256.
3. External software installed separately by users, including 3D Slicer,
   SlicerDMRI, MRtrix3, FSL, FreeSurfer-compatible tools, WMA, TractQuerier,
   and TractCloud.

The top-level BSD-3-Clause license applies to SlicerDSpace source code unless
a file or bundled third-party component states different terms. It does not
grant redistribution rights for external tools, model weights, templates, or
subject-derived data whose terms must be reviewed separately.

---

## Pre-Tag Checks

First complete the SDK packaging and independent clean-install gates in
[Extension readiness](EXTENSION_READINESS.md). Resource/metadata checks and
local staging do not replace those gates. The Index JSON is a draft only; keep
the repository private until an explicit maintainer decision to publish.

Run these checks from a clean worktree:

```bash
make test-fast
python3 scripts/release_audit.py
python3 scripts/download_model_assets.py --profile core --asset-root /path/to/model-assets --verify-only
python3 scripts/download_model_assets.py --profile all --asset-root /path/to/model-assets --verify-only
SMOKE_ASSET_MODE=download scripts/smoke_fresh_clone.sh
SLICERDSPACE_RELEASE_SUBJECT_ROOT=/path/to/subject-root \
  SLICERDSPACE_RELEASE_ARGS="--skip-slicer-smoke --warn-only --asset-mirror-dir /path/to/core-mirror --all-asset-mirror-dir /path/to/all-mirror" \
  make release-plan
```

The release gate treats code checkout and model-asset download as separate
operations. Fresh-clone smoke uses `GIT_LFS_SKIP_SMUDGE=1` only to avoid any
legacy/test-fixture LFS materialization in the code checkout.
The CI-safe mode (`SMOKE_ASSET_MODE=plan`) resolves the manifest and runs fast
tests without downloading large model files. The release gate must use
`SMOKE_ASSET_MODE=download`, which runs `scripts/download_model_assets.py` with
an explicit profile and checksum verification. Do not treat a developer checkout
that already has materialized LFS files as proof that new users can install the
release.

If Slicer is available, also run:

```bash
SLICERDSPACE_SLICER_LAUNCHER=/path/to/Slicer \
SLICERDSPACE_XVFB_RUN=/path/to/xvfb-run \
  make test-slicer
```

On a workstation or VNC session that already has a display server, use:

```bash
SLICERDSPACE_SLICER_LAUNCHER=/path/to/Slicer \
SLICERDSPACE_USE_EXISTING_DISPLAY=1 \
  make test-slicer
```

All GitHub Actions workflows are currently manual-only (`workflow_dispatch`),
including CI, Website, Release and Slicer SDK. Pushes and pull requests do not
start them. Run local validation during development; a maintainer may explicitly
request a release or cross-platform run when needed. A manually dispatched CI
downloads only the small sentinel asset, not the model profiles.

The same manual `Release` workflow can create or update the draft GitHub Release
page by setting `create_github_release=true`. To attach fresh-clone evidence to
the release in the same run, also set `run_fresh_clone_smoke=true`.

If GitHub reports `some jobs were not successful`, first confirm that the latest
commit is being inspected and that branch protection no longer requires old
manual release jobs. With a token that can read repository Actions and branch
protection, run:

```bash
GITHUB_TOKEN=<token> \
  python3 scripts/check_github_ci_status.py \
  --repository NeuroDMRI/SlicerDSpace \
  --ref $(git rev-parse HEAD) \
  --branch main
```

If automatic CI is re-enabled later, the normal required check should be
`CI / fast-checks` or `fast-checks`. While automation is manual-only, do not require
an automatic check that will never start; branch protection is an owner-managed
setting and is not changed by the workflow edit. Remove stale required checks such as `fresh-clone-smoke`, `create-github-release`, or
`create-github-release-with-smoke` from GitHub branch protection. The manual
`Release` workflow is still available for fresh-clone smoke and draft release
creation, but it should not be required for every push or pull request.

---

## Asset Terms Review

Before archiving model or validation assets, review:

- `docs/assets/MODEL_ASSETS.md`
- `docs/assets/MODEL_ASSET_TERMS.md`
- `docs/assets/MODEL_ASSET_HOSTING_PLAN.md`
- `models/assets_manifest.json`

Do not include an LFS-managed asset in a DOI archive until its source,
ownership, license, redistribution rights, and archive permissions are known.
If an asset is restricted, document the restriction in the release notes and
availability statement.

If `docs/assets/MODEL_ASSET_TERMS.md` still marks redistribution or DOI-archive permission
as unconfirmed, publish only a code/documentation release or exclude the affected
assets from the DOI archive.
Use `docs/assets/MODEL_ASSET_HOSTING_PLAN.md` to prepare a separate asset repository or
DOI-backed mirror without moving large weights into the code archive.

---

## Tagging

Use a semantic or manuscript-linked tag after the code, documentation, and
validation route are frozen:

```bash
git tag -a vX.Y.Z -m "SlicerDSpace vX.Y.Z"
git push origin vX.Y.Z
```

For Nature Protocols preparation, use a stable manuscript-linked tag only after
the manuscript scope, validation subject, timing evidence, source-data package,
and asset terms are finalized.

---

## DOI Archive

Archive the release through Zenodo, Figshare, an institutional repository, or
another DOI-backed service. The included `.zenodo.json` is a metadata starting
point. Before publishing the DOI record:

- Confirm the release tag being archived.
- Confirm that restricted model weights, templates, and validation data are not
  included unless redistribution is allowed.
- Include `CITATION.cff`, `codemeta.json`, `LICENSE.txt`, and the release
  validation report.
- Record the DOI in the release notes and manuscript Code Availability section.

Use `docs/release/DOI_ARCHIVE_PREP.md` and the files generated by `make release-artifacts`
as the code-only archive checklist and asset-exclusion record.

For the existing `v0.1.0` tag, `docs/release/GITHUB_RELEASE_DRAFT_v0.1.0.md` contains a
ready-to-review GitHub Release body and attachment list. If post-tag changes on
`main` should be included, create a new tag after rerunning the release gates
rather than moving `v0.1.0`.

To create the draft GitHub Release page and upload the default code-only support
files from a maintainer machine instead of GitHub Actions:

```bash
python3 scripts/create_github_release.py --dry-run \
  --expected-commit <full-tested-commit-sha> \
  --fresh-clone-report /path/to/fresh_clone_smoke_report.json
# Configure GITHUB_TOKEN or GH_TOKEN securely before the write operation.
  python3 scripts/create_github_release.py \
  --expected-commit <full-tested-commit-sha> \
  --fresh-clone-report /path/to/fresh_clone_smoke_report.json
```

Both commands validate the actual report content. Evidence must cover a real
`core` download and verification, all required successful steps, and the exact
commit being released. Plan-only and sentinel checks are insufficient. Existing
tags must resolve to that commit. The helper refuses to replace assets on an
already public release; use a new version for corrections. Draft creation is
not a substitute for the SDK build, packaged installation and scientific review
required before public release.

---

## Post-Release Updates

After a release tag and DOI exist:

1. Update `CITATION.cff` with the final version and DOI.
2. Update `codemeta.json` with the final version and identifier.
3. Update release notes and availability text with the repository URL, release
   tag, DOI, license, and asset restrictions.
