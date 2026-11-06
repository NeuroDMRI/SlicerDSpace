<div align="center">

# Model Assets and Weight Downloads

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

SlicerDSpace keeps the code repository lightweight by storing large
model/template files outside the main source tree. The preferred external
source is the independent [SlicerDSpace-models](https://github.com/ChengjinLii/SlicerDSpace-models) repository. The manifest and
downloader verify every asset by size and SHA256 checksum before use.

Users do not need to clone model files with the code repository. A code-only
checkout can restore the selected model profile with the downloader. Git LFS is
only needed if users clone the separate model repository directly.

Use the unified downloader instead of asking users to remember individual LFS paths:

```bash
scripts/download_model_assets.sh --profile core
```

To test network access, mirror routing and checksum verification without
downloading large model weights, use the tiny sentinel profile:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile sentinel \
  --asset-root /tmp/slicerdsurfer-model-sentinel \
  --source url \
  --source-id slicerdsurfer_models_git
```

By default, downloaded assets are treated as user data and are not written into
the Slicer extension checkout:

| Platform | Default asset root |
| --- | --- |
| Linux | `$XDG_DATA_HOME/SlicerDSpace/model-assets` or `~/.local/share/SlicerDSpace/model-assets` |
| macOS | `~/Library/Application Support/SlicerDSpace/model-assets` |
| Windows | `%LOCALAPPDATA%\\SlicerDSpace\\model-assets` |

Users can override this with the Slicer UI `Model assets root` field, the
`SLICERDSPACE_MODEL_ASSETS_DIR` environment variable, or `--asset-root`.
Only the per-user default above is ever scanned automatically; any other
location (e.g. a shared cache on a multi-user cluster) must be explicit --
callers do not silently trust well-known/guessable shared paths:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /srv/slicerddsurfer/model-assets
```

### Runtime sha256 Verification at Load Time

This is separate from the CLI's own `--verify-only` pass described below.
Whenever the module code (not the downloader script) resolves a model
checkpoint or template for use -- e.g. before handing a path to
`torch.load()` -- it checks the file's sha256 against the recorded digest
in `models/assets_manifest.json`:

- An **auto-discovered** root -- the per-user default above, or the
  extension's own checkout used as a legacy fallback -- is only trusted
  once every manifest-recorded file present under the requested path
  verifies. A file that exists but does not match its recorded sha256
  raises a `ModelAssetIntegrityError`, which the Slicer module and the
  environment doctor surface as a status message rather than letting it
  abort module load or the whole doctor scan.
- An **explicitly configured** root -- via `--asset-root`, the Slicer UI's
  `Model assets root` field, or the `SLICERDSPACE_MODEL_ASSETS_DIR`
  environment variable -- is used as-is, **without** this sha256 check.
  Pointing SlicerDSpace at an explicit root is treated as the user's own
  informed choice; a tampered or stale file there is not caught until you
  explicitly run `--verify-only` (or the Slicer UI's `Verify` action)
  yourself. Setting `SLICERDSPACE_MODEL_ASSETS_DIR` therefore comes with
  the same responsibility as passing `--asset-root` directly -- only point
  it at a location you trust or have verified.

The downloader reads `models/assets_manifest.json`, resolves the selected
profile, downloads from the independent repository or configured URL mirrors, and verifies that the
selected files are real materialized objects with the expected file size and
SHA256 checksum. URL downloads write to `.part` files first, print file-size and
progress updates, and resume partial downloads when the server supports HTTP
Range requests. It also prints profile-level asset groups and conservative
terms notices so users know when model weights, templates, or subject-derived
fixtures require separate license or redistribution confirmation.

The Slicer downloader uses the same implementation. Its progress dialog shows
the current file and byte count. `Cancel` preserves the `.part` file; choosing
Download again resumes it when the server supports Range requests. Cancelling
the destination-folder dialog leaves the previous setting unchanged and does
not start a download.

To inspect the resolved profile without downloading anything:

```bash
PythonSlicer scripts/download_model_assets.py --profile core --print-plan
```

To verify an existing checkout without downloading:

```bash
PythonSlicer scripts/download_model_assets.py --profile core --verify-only
```

CLI verification performs size and SHA256 checks. The Slicer UI uses a lighter
presence/pointer check for passive status text and performs full checksum
verification when the user clicks `Verify core`.

To download from an external mirror while preserving the same repository-local
paths:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --asset-root /srv/slicerddsurfer/model-assets \
  --source url \
  --source-id slicerdsurfer_models_git
```

The `slicerdsurfer_models_git` source resolves files from
`https://github.com/ChengjinLii/SlicerDSpace-models/raw/main/{path}`.
Existing configured asset directories and verified downloads remain reusable.
For an ad-hoc mirror, use `--base-url`;
the downloader appends each relative asset path to that URL and verifies every
download against the manifest size and SHA256.

To prepare an upload-ready mirror after the selected assets are materialized and
verified locally:

```bash
python3 scripts/export_model_asset_mirror.py \
  --profile core \
  --output-dir /tmp/SlicerDSpace-models-v0.1.0
```

The exporter verifies size and SHA256 first, then writes the files using the
same repository-relative paths plus `asset_mirror_manifest.json` and
`asset_mirror_index.tsv`. Upload that directory to `SlicerDSpace-models`,
Hugging Face, Zenodo, OSF, Figshare, or institutional object storage. Use
`--dry-run` to inspect the export plan without writing files, or `--mode
hardlink` for local staging when the target filesystem supports hard links.

To freeze a release or manuscript-validation asset state, write a lockfile
after verification:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --verify-only \
  --write-lockfile outputs/00_run_metadata/model_assets_core.lock.json
```

The lockfile records the Git commit, resolved profile, asset groups, terms
notices, required paths, file sizes, SHA256 checksums, and whether any file is
still only a Git LFS pointer. Use it as the source of truth when building an
offline archive or mirror.

For a fast clone followed by a targeted model download:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NeuroDMRI/SlicerDSpace.git
cd SlicerDSpace
scripts/download_model_assets.sh --profile core
```

---

## Profiles

Profiles are configured in `models/assets_manifest.json`. Update that manifest when adding, removing, or moving model assets; `scripts/download_model_assets.sh` is a compatibility wrapper around the Python manifest reader.

| Profile | Includes | Use when |
| --- | --- | --- |
| `surface` | DDSurfer checkpoints, surface templates, registration atlas | Running DDSurfer surface reconstruction. |
| `segmentation` | DDSeg CNN weights | Running WM/GM/CSF segmentation. |
| `parcellation` | DDParcel weights | Running DDParcel parcellation. |
| `core` | `surface` + `segmentation` + `parcellation` | Recommended default for local users and release quick-gate runs. |
| `quick-gate` | Alias for `core` | Matching the release smoke terminology. |
| `test-fixtures` | Parcellation conversion test data | Running conversion tests only. |
| `all` | Every LFS object tracked in the repository | Full offline mirror or release validation. |

Examples:

```bash
# Recommended after clone
scripts/download_model_assets.sh --profile core

# Equivalent explicit Python entrypoint
PythonSlicer scripts/download_model_assets.py --profile core --asset-root /srv/slicerddsurfer/model-assets

# Only run DDSeg
scripts/download_model_assets.sh --profile segmentation

# Full offline checkout, including test fixtures
scripts/download_model_assets.sh --profile all
```

Use `--no-verify` only for debugging slow or interrupted downloads.

---

## Asset Layout

The core model assets are expected at:

```text
Resources/Data/100HCP-population-mean-T2.nii.gz
Resources/Data/ckpts/hcp/*.pt
Resources/Data/template/*
surface/template/*
segmentation/weights/*.pt
parcellation/weights/*.pkl
```

DDSurfer also supports lower-level external asset locations through environment variables:

| Variable | Purpose |
| --- | --- |
| `DDSURFER_CHECKPOINT_DIR` | Directory containing `hcp/*.pt` DDSurfer checkpoints. |
| `DDSURFER_TEMPLATE_DIR` | Directory containing DDSurfer surface templates. |
| `DDSURFER_REGISTRATION_ATLAS` | Path to `100HCP-population-mean-T2.nii.gz` or equivalent atlas. |

The preferred unified layout under the model asset root is:

```text
Resources/Data/
  100HCP-population-mean-T2.nii.gz
  ckpts/
  template/
segmentation/weights/
parcellation/weights/
```

---

## Manifest Format

The manifest has one top-level `profiles` object. Each profile may define:

| Field | Meaning |
| --- | --- |
| `include_patterns` | Git LFS include patterns passed to `git lfs pull --include=...`. |
| `required_paths` | Concrete files that verification checks for existence and non-pointer contents. |
| `include_profiles` | Other profiles to merge into this profile. |
| `alias_for` | Profile alias, used by `quick-gate`. |
| `all_lfs` | Pull every Git LFS object tracked by the checkout. |
| `external_overrides` | Environment variables users may set instead of repository-local DDSurfer assets. |
| `asset_groups` | Human-readable asset groups included by the profile, used for manuscript and release audits. |
| `terms_notice` | Conservative license/redistribution notice printed by the downloader. This is not a license grant. |

The manifest also contains source and checksum metadata:

| Field | Meaning |
| --- | --- |
| `asset_sources` | Named sources. `github_lfs` is the compatibility source, `slicerdsurfer_models_git` is the preferred independent repository source, and release/Hugging Face/Zenodo entries are reserved mirrors until their URLs and terms are finalized. |
| `default_asset_source_ids` | Ordered source ids attached to asset records when no per-file source list is supplied. |
| `asset_records` | Per-file `size_bytes` and `sha256` records used by verification, URL downloads, and release lockfiles. |

URL-template sources may use these placeholders:

| Placeholder | Meaning |
| --- | --- |
| `{path}` | Repository-relative asset path. |
| `{path_urlencoded}` | URL-encoded repository-relative path, preserving `/`. |
| `{filename}` | URL-encoded basename only. Useful for flat DOI records. |
| `{sha256}` | Manifest SHA256 for content-addressed mirrors. |

---

## Recommended Asset Hosting Architecture

For a public release, keep this code repository lightweight and publish large
weights/templates through a governed asset source:

1. Use `ChengjinLii/SlicerDSpace-models` or an equivalent
   Hugging Face/Zenodo/OSF record.
2. Preserve the same relative paths as `asset_records`.
3. Upload only files whose redistribution and archive permissions are confirmed.
4. Generate or reuse the manifest `size_bytes` and `sha256` values.
5. Keep `slicerdsurfer_models_git` active for the independent repository, or
   pass an explicit `--base-url` during validation.
6. Run:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --source url \
  --source-id slicerdsurfer_models_git
PythonSlicer scripts/download_model_assets.py --profile core --verify-only
```

If a DOI platform stores all files flat rather than preserving directories,
configure per-record `urls` or a URL template using `{filename}`. Do not publish
asset-inclusive DOI archives until the terms decision record allows it.

---

## Release Governance

For public releases and manuscript supplements, treat model assets as a
controlled dependency rather than as incidental files.

Required release checks:

1. Run `scripts/download_model_assets.sh --profile core` or the intended
   profile.
2. Run `PythonSlicer scripts/download_model_assets.py --profile <profile>
   --verify-only --write-lockfile <path>`.
3. Confirm the lockfile has no `exists=false` records and no
   `lfs_pointer=true` records.
4. Record the lockfile path in the release-validation report.
5. Confirm redistribution terms for every `asset_groups` entry before DOI or
   supplementary upload.

Mirrors and offline packages:

- A mirror must preserve the same relative paths, file sizes, SHA256 checksums
  and terms notices as the lockfile.
- Offline packages should be generated only from a verified checkout and should
  include the lockfile next to the archive.
- A mirror URL is not a license grant; keep the original source and terms
  notice visible in release notes and manuscript availability statements.

---

## Recommended User Flow

1. Clone the repository.
2. Run `scripts/download_model_assets.sh --profile core`.
3. Run dependency doctor:

```bash
python3 run_subject_pipeline.py \
  --subject-root /path/to/subject \
  --targets validation,preprocessing,segmentation,parcellation,surface,microstructure \
  --doctor
```

4. If the doctor reports missing model assets, rerun the downloader for the relevant profile or configure the DDSurfer environment variables above.

---

## Asset Terms Boundary

The downloader proves that selected files are present and materialized from Git
LFS. It does not prove that model weights, templates, or subject-derived test
fixtures can be redistributed through GitHub, a DOI archive, or supplementary
materials. Before submission or public archival release, confirm asset source,
ownership, license, GitHub LFS redistribution permission, and DOI-archive
permission for each asset group.

The current release-facing asset terms status is tracked in
`MODEL_ASSET_TERMS.md`. Treat that file as the decision record for whether a
release may include model assets or must remain code/documentation-only.

---

## Notes for Maintainers

- Do not commit large `.pt`, `.pkl`, `.nii.gz`, `.mgz`, `.obj`, `.stl`, `.trk`, `.tck`, `.vtk`, or `.vtp` assets as normal Git blobs.
- Run `python3 scripts/release_audit.py` before pushing release changes; it checks the expected Git LFS patterns and pointer state.
- `scripts/docker/run_rc_smoke.sh` defaults `SMOKE_STAGE=quick-gate` to `SMOKE_LFS_MODE=profile` with `SMOKE_ASSET_PROFILE=quick-gate`, so fresh-clone quick-gate runs download only the manifest-selected `core` assets by default. Use `SMOKE_LFS_MODE=full` if you explicitly want every tracked LFS object.
