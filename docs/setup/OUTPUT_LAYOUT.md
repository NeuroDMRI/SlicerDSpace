<div align="center">

# Output Layout

[Home](../../README.md) | [Quick Start](QUICKSTART.md) | [Website](../website/README.md) | [中文](OUTPUT_LAYOUT.zh-CN.md)

</div>

---

## Canonical Directories

```text
outputs/
  00_run_metadata/
  01_preproc/
  02_microstructure/
  03_anatomy/
    ddseg/
    ddparcel/
    ddsurfer/
      postprocess/
  04_tractography/
  05_statistics/
  06_slicer_scene_report/
```

`00_run_metadata/run_record.json` is written at the end of every run. It lists the run
manifest, the parameters, the software and asset versions, the licences that were
accepted, the brain-mask provenance (the requested backend and the route of the
mask that the run used), a QC summary and the SHA-256 of every final product file.
It copies no output file. A re-run keeps the previous record as
`run_record-<started_at>.json` and nothing is deleted.

New anatomical results use `03_anatomy`. Segmentation, parcellation and surface
reconstruction remain separate products. The directory name does not prescribe
the execution order or imply that all products were selected or passed QC.

---

## Existing Results

The previous `03_structural` name remains supported for reading and eligible
cache reuse. Existing files, manifests and scientific provenance are not moved,
copied or rewritten merely to rename the directory. Explicit output paths remain
honoured. The Python `SubjectIOLayout.structural` attribute is retained as an
alias for the new canonical `anatomy` path.

Selection is per product directory: use the new directory if it exists;
otherwise consider its legacy counterpart. An incomplete new product is not
filled from old files. Original QC, input signatures and current-version cache
checks still apply; finding an old directory is not a scientific acceptance.
Reports describe both names while retaining the real recorded paths.

Archived validation reports and frozen figure provenance retain their historical
paths. They are evidence of those runs, not instructions to write new results
under the old name. Third-party backends explicitly selected by the user may
still return legacy paths; the adapter reads those paths compatibly.

---

## Local Validation

GitHub Actions is manual-only. Directory compatibility and normal regression
tests run locally with `make test-fast`; no Actions run is needed to process data.
