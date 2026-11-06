<div align="center">

# Model Asset Terms Status

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

This file records the release-facing status of model weights, templates and
validation fixtures used by SlicerDSpace. It separates local availability
checks from redistribution and DOI-archive permission.

---

## Current Decision

The repository has a unified model-asset download and verification route through
`MODEL_ASSETS.md`, `models/assets_manifest.json` and
`scripts/download_model_assets.py`.

The current checkout can verify the required core assets locally, but the
redistribution and DOI-archive terms for bundled model weights, templates and
subject-derived fixtures are not yet confirmed. Until those terms are confirmed,
the safe release mode is a code-and-documentation release that documents how
users obtain or verify assets, without treating the assets as unrestricted
archive contents.

---

## Status By Asset Group

| Asset group | Manifest profile | Local verification status | Public redistribution / DOI archive status |
| --- | --- | --- | --- |
| DDCSR surface model weights | `surface`, `core` | Checked by the manifest when `--profile core` is verified. | Requires author or rights-holder confirmation before public archive inclusion. |
| HCP-like registration/template images | `surface`, `core` | Checked by the manifest when `--profile core` is verified. | Requires source, ownership and archive-permission confirmation. |
| Surface initialization meshes | `surface`, `core` | Checked by the manifest when `--profile core` is verified. | Requires source, ownership and archive-permission confirmation. |
| DDSeg tissue-segmentation weights | `segmentation`, `core` | Checked by the manifest when `--profile core` is verified. | Requires redistribution-term confirmation and citation of the peer-reviewed DDSeg method paper. |
| DDParcel parcellation weights | `parcellation`, `core` | Checked by the manifest when `--profile core` is verified. | Requires redistribution-term confirmation and citation of the peer-reviewed DDParcel method paper. |
| Conversion test NIfTI fixtures | `test-fixtures`, `all` | Checked only when the fixture profile is selected. | Requires confirming that the local license covers modified fixtures and DOI/archive use. |
| PNL CNN weights and architecture | External catalog: `pnl-cnn-weights` | Fixed v0.3 SHA256; unmodified files and full upstream agreement. | Follow PNL/BWH notice and attribution terms; does not cover IIT data. |
| IIT b0 template | External catalog: `pnl-cnn`, upstream-only | Hash-verified for local authorized use. | **No electronic mirror without express written IIT permission.** Ignored by Git and blocked by the mirror exporter. |
| TractCloud minimal inference resources | External catalog: `tractcloud` | Matches the saved v1.0.0 distribution bytes; no subject datasets. | Preserve the original Slicer/BWH agreement and upstream citations. |
| ORG registration/clustering atlases | External catalog: `wma` | Fixed official ZIPs; size/MD5 match the Zenodo record, SHA256 pinned. | CC BY 4.0 per doi:10.5281/zenodo.2648292; retain attribution and original upstream agreement. |
| East-West WM atlas | External catalog: `wma-east-west` | Byte-preserved runtime subset of ScienceDB `10.57760/sciencedb.17354`; all 800 reference clusters retained, split archives and extracted files SHA256-pinned. | CC BY 4.0 per the dataset record; cite Li et al., Scientific Data 11:787 (2024), doi:10.1038/s41597-024-03624-2. |
| NABA neonatal/adult atlas | External catalog: `wma-naba` | Byte-preserved runtime subset of Zenodo `10.5281/zenodo.18227691`; original published MD5 verified before packaging. | CC BY 4.0 per the dataset record; cite Zhang et al., Fundamental Research 6:1838-1847 (2026), doi:10.1016/j.fmre.2025.12.030. No NABA source code is copied. |
| SynthStrip main model v1 | External catalog: `synthstrip` | Matches author's v1 download SHA256. | Author's CC BY 4.0 option; preserve attribution and license link. Does not cover the rest of FreeSurfer. |

Optional-resource layout, official sources and migration boundaries are recorded
in [Non-Code Runtime Resources](EXTERNAL_RESOURCES.md).

---

## Required Checks Before Asset-Inclusive Release

Before publishing a release archive or DOI record that includes model or
validation assets, maintainers must confirm:

1. The original source and owner for every asset group.
2. The license or permission statement covering redistribution.
3. Whether GitHub LFS distribution is permitted.
4. Whether Zenodo, Figshare or another DOI-backed archive may redistribute the
   asset.
5. Whether attribution, citation or access restrictions must appear in release
   notes and manuscript availability statements.
6. A lockfile generated from the final release checkout:

```bash
PythonSlicer scripts/download_model_assets.py \
  --profile core \
  --verify-only \
  --write-lockfile outputs/00_run_metadata/model_assets_core.lock.json
```

The lockfile must have no missing files and no Git LFS pointer-only files before
it is used as release evidence.

---

## Tag And DOI Policy

A code-only tag may be created after the worktree is clean and all code release
checks pass, provided release notes state that model assets remain controlled
dependencies with separately reviewed terms.

An asset-inclusive DOI archive must not be created until every asset group above
has confirmed redistribution and DOI-archive permission. If any asset group
cannot be redistributed, the DOI archive should exclude that asset and document
the approved access route instead.
