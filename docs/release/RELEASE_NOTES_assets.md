# SlicerDSpace Shared Assets (`assets`)

This permanent release hosts version-independent binary assets shared across SlicerDSpace software releases (`v1.0.0` and later):

- **Core Modular Model Weight Packs**:
  - `SlicerDSpace-Model-DDSurfer-v1.tar.gz` — DDSurfer cortical surface reconstruction checkpoints, templates, and postprocessing atlas
  - `SlicerDSpace-Model-DDSeg-v1.tar.gz` — DDSeg multi-view tissue-segmentation weights
  - `SlicerDSpace-Model-DDParcel-v1.tar.gz` — DDParcel 101-ROI fused U-Net + DTI backbone weights and T2 template
  - `SlicerDSpace-Model-PNL-CNN-v1.tar.gz` — PNL CNN PyTorch brain-masking weights *(excludes third-party NITRC `IITmean_b0_256.nii.gz`, which is fetched directly from NITRC)*
  - `SlicerDSpace-Model-TractCloud-v1.tar.gz` — TractCloud inference weights and reference mass center
- **Optional White-Matter Atlases (WMA)**:
  - `SlicerDSpace-WMA-ORG-v1.tar.gz` — ORG 800-cluster atlas (74 tracts)
  - `SlicerDSpace-WMA-NABA-v1.tar.gz.001..002` — NABA lifespan atlas
  - `SlicerDSpace-WMA-EastWest-v1.tar.gz.001..007` — East-West cross-population atlas
- **Example Data**:
  - `SlicerDSpace-Example-HCP-100307-NRRD.zip` (~32 MB) — Preprocessed dual-shell HCP subject 100307 NRRD quickstart example DWI (2.0 mm isotropic, 66 volumes: b = 0, 1000, 2000; `100307_dwi.nrrd` + `nodif_brain_mask.nii.gz`, `DWIConvert` / 3D Slicer native; ~1 min end-to-end DTI + UKF + TractCloud)
- **Prebuilt Runtimes & Individual-File Mirror**:
  - Standalone `UKFTractography` binaries (`ukf-standalone-*`), prebuilt FreeSurfer surface runtimes (`surface-runtime-*`), and SHA-256-indexed individual model files for manifest verification and per-file downloads.

> **Do not delete this release.** SlicerDSpace installers and in-app on-demand resource downloads pull directly from this release tag. For software installers and release archives, see the [latest SlicerDSpace release](https://github.com/NeuroDMRI/SlicerDSpace/releases/latest).
