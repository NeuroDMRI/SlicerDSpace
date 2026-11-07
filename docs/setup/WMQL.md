<div align="center">

# WMQL Bundle Extraction

[Project](../../README.md) | [Setup](QUICKSTART.md) | [中文](WMQL.zh-CN.md)

</div>

## Official Protocol

The default `official` protocol uses the unmodified author repository's
`freesurfer_queries.qry` and `FreeSurfer.qry`, pinned at
`5c69c594d489859e08485110678dca7f6f4cba40`. Upstream master was checked on
2026-10-24; the bundled files match it byte for byte. Rules and their upstream
license ship in the extension; the Tract Querier engine is a managed dependency.

The script declares **48 output queries**, including left/right association,
projection and commissural fiber bundles. It does not force every input streamline
into a class. Queries can overlap, and the author CLI does not write empty results.
No DDParcel-specific AF override, fallback query or threshold relaxation is used.

---

## Controls

In **Advanced settings > Model parameters**, `WMQL bundles` is optional:

| Value | Action |
| --- | --- |
| Empty or `all` | All official output queries |
| `AF,CST,UF` | Those three families, both sides |
| `af.right,ilf.left,cc_4` | Exact official output names |

Names are case-insensitive; unknown names fail before execution. Single-subject,
batch and background GUI runs pass the same selection. Changing the selection or
rule dependencies invalidates the old extraction cache.

```bash
PythonSlicer shared/run_subject_pipeline_cli.py --subject-id SUBJECT --subject-root /data/SUBJECT \
  --targets bundles --wmql-query-selection AF,CST,UF --execute
```

Other required upstream inputs/dependencies still apply. Deprecated `author`,
`ddparcel_act` and `published_endpoints` CLI values remain solely for explicit
historical reproduction; they are not GUI choices or automatic fallback routes.
Saved GUI defaults migrate to `official`; migrated scene parameters retain the
previous preset in `SlicerDSpace.LegacyWMQLPreset`. Run records are not rewritten.

---

## Compatibility and QC

The atlas must contain finite nonnegative integer 3-D FreeSurfer/DK region IDs.
A tissue-only WM/GM/CSF map is rejected. The check records cortical and regional
WM IDs, absent core IDs and the affine's validity. No atlas is relabeled or dilated.
DDParcel's legacy `wmparc` filename does not prove FreeSurfer wmparc equivalence:
its cortical/aseg labels can lack the 3000/4000 regional-WM territories, which
changes coverage under author crossing rules, especially on ACT-cropped tracks.
Numeric compatibility is not a check of spatial or anatomical accuracy.

Each query receives an explicit status: `not_extracted`, `empty`, `invalid`,
`insufficient_sampling` (<20 streamlines), or `anatomical_review_pending`.
The 20-streamline policy is a local sampling sanity check, not an author default
or a statistical-power guarantee. All fiber bundles remain ineligible for accepted
quantification until anatomical review. Damaged geometry or a broken invocation
fails the run; partial coverage can complete extraction with a visible QC warning
and `automatic_qc_passed=false`. This is not an all-bundle scientific pass.

---

## Recorded Validation

2026-10-24, HCP 100307, archived UKF input with its actual DDParcel atlas:

| Item | Observed |
| --- | --- |
| Requested queries | 48 |
| Nonempty outputs | 41 |
| At least 20 streamlines | 31 |
| Empty/not extracted | 7 |
| Nonempty but below 20 | 10 |
| AF left/right | 1,184 / 1,216 |
| CST left/right | 168 / 252 |
| Wrapper execution + geometry checking | 36.69 s |
| Direct author invocation | 19.81 s |

All **48 query streamline multisets**, including empty results, match direct
execution using the unmodified author resources. Every exported streamline
matches the input exactly; scientific source hashes are unchanged. These runtimes
have different scopes and are not an algorithm speedup comparison.
Local evidence: `ignore/validation/wmql_official_20261024/100307_ukf/` and
`100307_ukf_parity/parity.json`. The published test summary is
[wmql_official_20261024.json](../release/benchmarks/wmql_official_20261024.json).

The same subject's archived CSD/ACT input produced 34 nonempty outputs: 28 meet
the sampling threshold, 6 are sparse and 14 absent. Runtime including wrapper
checks: 54.41 s. Neither run is all-bundle QC-passed; both need anatomical review.
This is two tracking inputs from one subject, not multi-subject generalization.

An explicit `AF,CST,UF` UKF subset produced all six requested outputs, with
identical counts to the full query run (18.71 s including wrapper checks).
The atlas compatibility warning remains visible even for this complete subset.
The independent audit confirms geometry and exact source membership, while its
all-requested-sampling check remains false for the full run.
An isolated 3D Slicer 5.10 session loaded the staged extension and passed the
Advanced selector, parameter binding, backend visibility and legacy-scene
migration checks without tracebacks. The current live Slicer session was not changed.

Reproduce the wrapper/direct-author comparison with
`scripts/validate_wmql_official.py --manifest <wmql_manifest.json> --output <new-directory>`.
The validator does not alter source inputs or existing run records.

Sources: [author usage](https://tract-querier.readthedocs.io/en/latest/example_script.html),
[pinned rules](https://github.com/demianw/tract_querier/blob/5c69c594d489859e08485110678dca7f6f4cba40/tract_querier/data/freesurfer_queries.qry).
