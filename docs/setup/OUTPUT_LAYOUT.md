<div align="center">

# Output Layout

[Home](../../README.md) | [Quick Start](QUICKSTART.md) | [Website](../website/README.md) | [中文](OUTPUT_LAYOUT.zh-CN.md)

</div>

---

## Canonical Directories

```text
<subject_root>/
  input/
  preprocessed/                         # Created when brain mask is generated (nodif_brain_mask.nii.gz) or preprocessing runs (<sub_id>_dwi_final.*, <sub_id>_brain_mask_final.nii.gz)
  cache/                                # Internal stage workspaces, logs, manifests, display previews
    stages/
    display_previews/
  outputs/
    00_run_metadata/
    01_anatomical_priors/
      tissue_segmentation/              # SegmentationMap_GMWMCSF.nii.gz, ProbabilisticMap_{WM,GM,CSF}.nii.gz
      brain_parcellation/               # <sub_id>_parcellation.nii.gz
      cortical_surface/                 # lh.white.obj, lh.pial.obj, rh.white.obj, rh.pial.obj
      surface_morphometry/              # {lh,rh}.{aparc,aparc.a2009s}.stats, {thickness,curv,sulc,area,inflated}, *.annot
    02_microstructure/
      <sub_id>_ROIWise-Microstructure.csv
      diffusion_tensor/                 # <sub_id>_{FA,MD,AD,RD}.nii.gz, <sub_id>_DEC.nrrd
      free_water/                       # <sub_id>_FW.nii.gz, <sub_id>_fw_FA.nii.gz, <sub_id>_fw_MD.nii.gz
      interstitial_free_water/          # <sub_id>_ifw.nii.gz, <sub_id>_ROIWise-iFW.csv
      neurite_microstructure/           # <sub_id>_{NDI,ODI,IsoVF}.nii.gz
      diffusion_kurtosis/               # <sub_id>_{MK,AK,RK,KFA,MKT,RTK}.nii.gz
    03_tractography/
      single_tensor_tractography/       # <sub_id>_dti.trk[, .vtk]
      multi_tensor_tractography/        # <sub_id>_ukf.trk, <sub_id>_ukf.vtk
      anatomically_constrained_tractography/ # <sub_id>_act.{tck,trk,vtk}
      short_association_fibers/         # <sub_id>_saf.{tck,vtk}
    04_bundles/
      query_bundle_extraction/<source>/ # WMQL: *.trk
      fiber_clustering/<source>/        # WMA: T_*.vtp, bundle_measurements.csv
      bundle_identification/<source>/   # TractCloud: *.vtp
    05_connectome/                      # <sub_id>_{streamline_count,sift2_weighted,<metric>}.csv, <sub_id>_nodes.tsv, <sub_id>_node_parcellation.nii.gz
    06_qc/                              # Concise per-algorithm plain-text QC summaries for tasks that ran/reused
```

`00_run_metadata/run_record.json` is written at the end of every run. It lists the run
manifest, the parameters, the software and asset versions, the licences that were
accepted, the brain-mask provenance (the requested backend and the route of the
mask that the run used), a QC summary and the SHA-256 of every final product file.
It copies no output file. A re-run keeps the previous record as
`run_record-<started_at>.json` and nothing is deleted.

Primary deliverables under `outputs/` are published as flat, symlink-free real files or hardlinks organized by scientific stage (`01_anatomical_priors` through `05_connectome`), while concise `.txt` quality-control summaries sit under `06_qc/`. Internal workspaces and intermediate files reside under `cache/`.

Each bundle folder has one subfolder per input tractogram. `<source>` is the id of the tracking that made the
tractogram (`csd_act`, `ukf`, `dti_det`, `saf`), also when the run has one tracking only. It is `supplied` when the user
gave the tractogram as a file (WMA or TractCloud input file). With two or more trackings, each tracking variant writes
into its own subfolder, so one variant cannot overwrite the bundle files of another. Which tracking each bundle
algorithm uses is the "Tractography source" setting of that algorithm (GUI settings popup, CLI
`--bundle-source ALG=SOURCE`); the default is unchanged: WMQL and TractCloud use every ticked tracking, WMA uses UKF.
Results from before this layout keep their flat files in these folders and stay readable; a resumed run publishes the
finished stage outputs again into the subfolders and does not compute them again.

---

## Existing Results

Previous `03_anatomy`, `03_structural`, `01_preproc`, `04_tractography`, `05_statistics`, and `06_slicer_scene_report` layouts remain supported for reading and cache reuse, and `migrate_subject_output_tree` migrates legacy subjects cleanly without symlinks. The Python `SubjectIOLayout` retains legacy property aliases (`anatomy`, `structural`, `ddseg`, `ddparcel`, `ddsurfer`, etc.).

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
