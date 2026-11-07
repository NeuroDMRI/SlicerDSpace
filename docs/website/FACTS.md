# SlicerDDSurfer project site: fact sheet

Checked 2026-10-20. Facts and contradictions only. The site makes no claim that is not on this sheet.

Path keys:
- `R/` = this repository's `main` branch at a48fc95 (dated 2026-10-16).
- `P/` = the promo film project, a separate local folder that is not in this repository. Its film, subtitles and credits are copied into `docs/` on this branch.
- Line numbers are from `cat -n`. Probe facts name the command.

Read in full: all files named in the brief. Not read (linked from them): docs/release/SLICER_ACCEPTANCE_20261007.md, UI_SIMPLIFICATION_20261008.md, DISTRIBUTION_VALIDATION_20261012.md, MODEL_ASSETS.md, P/film/TREATMENT.md, P/docs/research/FACTS_product.md (only lines 1-40 seen), the papers themselves. Facts about the papers' text below come from CITATION.cff or from P/film/FACTS.md. They were not re-checked against the papers.

---

## 1. Identity

| Fact | Source |
|---|---|
| Name: SlicerDDSurfer | R/README.md:3 |
| Tagline: "Diffusion MRI workflows in 3D Slicer" | R/README.md:5 |
| README description: "SlicerDDSurfer brings diffusion-MRI anatomical reconstruction, microstructural mapping and white-matter connectivity analysis into one Slicer module. Select the products you need, run a traceable workflow, and review its images, QC reports and execution records." | R/README.md:15-18 |
| s4ext/CMake description: "Integrated diffusion MRI workflows for anatomical reconstruction, tractography, bundle analysis, and structural connectivity." | R/SlicerDDSurfer.s4ext:11; R/CMakeLists.txt:10 |
| CITATION.cff abstract: "SlicerDDSurfer is a 3D Slicer scripted module and command-line workflow for diffusion MRI-derived anatomical reconstruction, segmentation, parcellation, microstructure analysis, tractography, bundle analysis, connectome construction, quality-control review, and provenance capture." | R/CITATION.cff:13-17 |
| codemeta description: "3D Slicer workflow for diffusion MRI-derived anatomical reconstruction, connectivity analysis, quality-control review, and provenance capture." | R/codemeta.json:5 |
| .zenodo.json description: "SlicerDDSurfer is a 3D Slicer scripted module and command-line workflow for diffusion MRI-derived anatomical reconstruction, connectivity analysis, quality-control review, and provenance capture." | R/.zenodo.json:5 |
| Full title (CITATION.cff and .zenodo.json): "SlicerDDSurfer: an integrated 3D Slicer workflow for anatomical reconstruction and connectivity analysis from diffusion MRI" | R/CITATION.cff:3; R/.zenodo.json:2 |
| Version: 0.1.0 (CITATION.cff, .zenodo.json, codemeta). date-released 2026-06-24; codemeta dateModified 2026-06-24. | R/CITATION.cff:11-12; R/.zenodo.json:4; R/codemeta.json:36-37 |
| Code licence: BSD-3-Clause. Copyright line: "Copyright (c) 2026, Chengjin Li and the SlicerDDSurfer contributors". | R/LICENSE.txt:1-3; R/README.md:207; R/CITATION.cff:10 |
| Author as recorded: one person, Chengjin Li (CITATION.cff, .zenodo.json "Li, Chengjin", codemeta). s4ext "contributors Chengjin Li". No affiliation is recorded in any file read. | R/CITATION.cff:5-7; R/.zenodo.json:6-10; R/codemeta.json:21-27; R/SlicerDDSurfer.s4ext:10 |
| Repo URL: https://github.com/ChengjinLii/SlicerDDSurfer (origin, fetch and push) | R/README.md:55; R/CITATION.cff:8-9; `git remote -v` |
| Extension category: Diffusion | R/SlicerDDSurfer.s4ext:9; R/CMakeLists.txt:8 |
| Extension dependencies (declared): PyTorch only. SlicerDMRI and UKFTractography are requested at setup only for a ticked product that needs them (WMA; UKF) | R/SlicerDSpace.s4ext:6; R/CMakeLists.txt:15; R/shared/selected_setup.py:25-32, 422 |
| Slicer version used for local GUI validation: "Local GUI validation uses Slicer 5.10." Badge: 3D_Slicer-5.10. | R/README.md:9, 64; R/docs/images/README.md:14 |
| Release status, exact: "**Research software.** Private pre-release; not currently distributed through the official Extensions Catalog. A completed run is not a clinical assessment or proof of scientific correctness." | R/README.md:20-22 |
| Same, Chinese: "科研软件。当前为私有预发布项目，尚未通过官方 Extensions Catalog 分发。任务完成不等于科学结果正确，也不构成临床诊断依据。" | R/README.zh-CN.md:18-19 |
| "this project is not yet in the official Extensions Catalog" | R/docs/setup/QUICKSTART.md:11-12 |
| "this document does not announce a public release or official catalog acceptance" | R/docs/architecture/PRODUCT_ARCHITECTURE.md:91-92 |
| "The repository stays private until the maintainer explicitly chooses otherwise." "Target: a Tier 1 submission after release gates pass. This is a preparation record, not Slicer approval." | R/docs/release/EXTENSION_READINESS.md:11-13 |
| "keep the repository private until an explicit maintainer decision to publish." "The Index JSON is a draft only" | R/docs/release/RELEASE.md:38-39 |
| "Use an authorized checkout while the repository is private." | R/README.md:51 |
| Public repo access is a release gate: "Public repository, resolvable icon/screenshot URLs, GitHub topic: Deferred by maintainer; required before public Index submission" | R/docs/release/EXTENSION_READINESS.md:99 |
| Dependencies shown in the film: SlicerDMRI extension (DWIToDTIEstimation, DiffusionTensorScalarMeasurements), PyTorch extension | P/CREDITS.md:14 |

## 2. The core claim: "without a T1-weighted scan"

What the repo files say:
- The strings "T1" and "T1-weighted" do NOT appear in R/README.md, README.zh-CN.md, QUICKSTART.md, SUPPORT_MATRIX.md, IMPLEMENTATION_PROVENANCE.md, PRODUCT_ARCHITECTURE.md, MODEL_ASSET_TERMS.md, EXTENSION_READINESS.md, CITATION.cff, codemeta.json, .zenodo.json or the s4ext (command: `grep -n -i t1` over those files; no hit).
- What the README does state is the input layout: `data.nii.gz`, `bvals`, `bvecs`, optional `nodif_brain_mask.nii.gz`. No T1 file appears. (R/README.md:100-109; R/README.zh-CN.md:90-97)
- QUICKSTART: "Use authorized, de-identified DWI, matching bval/bvec and a brain mask on the same voxel grid." (R/docs/setup/QUICKSTART.md:18-19)
- DDSeg route uses "five DTI features". (R/docs/architecture/IMPLEMENTATION_PROVENANCE.md:43)

What P/film/FACTS.md says (the film's claim table, P/film/FACTS.md:10):
- Claim allowed: "SlicerDDSurfer reconstructs anatomy directly from diffusion MRI, without a T1-weighted scan." and on-screen "No T1-weighted scan needed at analysis time."
- Evidence it cites: DDSurfer graphical abstract "reconstructs cortical surfaces directly from diffusion MRI without requiring T1-weighted scans."; DDParcel "the DDParcel network no longer requires anatomical MRI data but only the dMRI data."; DDSeg "predict a tissue segmentation directly from new dMRI data ... without requiring anatomical data"; "Repo README input layout: data.nii.gz + bvals + bvecs (+ mask); no T1w input in the GUI."
- Caveat, quoted: "models were trained with T1w-derived labels, so the film says 'at analysis time', never 'T1w-free'."
- Ruled out: "T1w-free" (P/film/FACTS.md:10). MAKING_OF repeats: "'Without a T1-weighted scan' is true **at analysis time**. DDSeg and DDParcel were trained with T1w-derived labels." (P/MAKING_OF.md:113-114)
- Also ruled out in the film ("Not claimed anywhere"): "no registration", "no preprocessing", "fully automatic". The film allows the wording "native diffusion space" (results), not "no registration". (P/film/FACTS.md:13, 21-25)
- The "problem" sentence in the film ("Mapping anatomy has usually meant a second, T1-weighted scan, registered into diffusion space.") is sourced to DDSurfer and DDSeg paper quotes. (P/film/FACTS.md:9)
- FACTS.md records that the spoken line itself omits "at analysis time"; the on-screen line carries it (P/film/FACTS.md:10).
- The line "All in native diffusion space." is sourced to `shared/workflow_catalog.py` ("native-space mesh export") and the DDSurfer paper (results transformed back to native dMRI space after template-space inference) (P/film/FACTS.md:13). Those two sources were not read here.

## 3. Capabilities

Five workflow families (README table, verbatim; R/README.md:28-36):

| Workflow | Products | Main prerequisites |
|---|---|---|
| Input preparation | DICOM/NIfTI/NRRD import, gradient validation, optional mask generation and preprocessing | Appropriate import/preprocessing backend |
| Anatomical reconstruction | DDSeg tissue labels, DDParcel regions, DDSurfer white/pial surfaces | Matching model assets and PyTorch for the selected route |
| Microstructural analysis | DTI FA/MD/AD/RD, optional Free-water, NODDI and DKI, regional summaries | Route-specific packages and acquisition suitability checks |
| White-matter pathways | DTI/CSD tractography, ACT/SIFT2, SAF, WMQL/WMA/TractCloud bundles | Backend-specific tools and assets |
| Structural connectivity | Regional matrices, Slicer scene inspection, QC and provenance | Compatible parcellation and tractogram; SIFT2 weights for weighted routes |

Opening line: "One module, five workflow families:" (R/README.md:28).

Optional and default behaviour:
- "Install dependencies **for the selected route**, not for every possible product. MRtrix3, FSL, FreeSurfer and advanced bundle backends are optional and are not silently installed." (R/README.md:38-40)
- "Microstructure runs **DTI only by default**. Enable Free-water, NODDI or DKI independently under Advanced." (R/README.md:120-121)
- DKI "checks baseline measurements, nonzero shells, gradient directions and design rank before fitting." (R/README.md:121-122)
- MK-Curve: "uses a bundled Python port of the author's algorithm, distinct from standard DIPY WLS/OLS. MATLAB is only an optional reference backend." (R/README.md:122-124)
- Mask: an existing input mask is retained. If one must be generated, the default is "PNL diffusion CNN masking"; SynthStrip, MRtrix and supplied-mask alternatives are selectable. "A missing backend is reported, never silently replaced." (R/README.md:113-118)
- PNL CNN default is not dependency-free: if no mask is supplied it needs PyTorch and model resources; on Linux also the managed ANTs 2.3.0 runtime. On macOS and Windows the native route (SimpleITK and PyTorch in Slicer's Python) is used and is labelled "native (experimental)". (R/docs/architecture/PRODUCT_ARCHITECTURE.md; R/shared/brain_mask.py:142-162)
- Tractography display: "Automatic tractography display uses a capped VTK preview, preserving the full TCK/TRK source; a preview is not the analysis tractogram." (R/README.md:125-126)
- GUI language: English or Chinese, chosen in the header. (R/README.md:89; R/README.zh-CN.md:79)
- "There is no visible Beginner section or automatic first-run wizard in the default interface." (R/README.md:96-97)
- Key results of successful stages load automatically into the current scene; QC reload/replace/clear under Advanced settings > Results and QC. (R/README.md:94)
- Dependency levels (Inspect/plan, Minimal DTI, Anatomical models, Microstructure, Advanced pathways, Connectome): R/docs/setup/SUPPORT_MATRIX.md:18-25. "Capability checks detect packages, executables and assets. They do not certify numerical accuracy for every acquisition or platform." (SUPPORT_MATRIX.md:11-12)
- CLI exists: `run_subject_pipeline.py ... --doctor` or `--plan`. (R/README.md:144-152)
- Slicer client design: "a Slicer-native workflow client, not a replacement implementation of every external scientific tool." (R/docs/architecture/PRODUCT_ARCHITECTURE.md:13-14)
- Managed MRtrix/FSL setup: Linux x86_64 (conda) and macOS 14.5 or later on Intel and Apple Silicon (MRtrix3 3.0.8 official tarball; FSL CPU tools). On Windows the WSL 2 Ubuntu runtime holds MRtrix3 and FSL; the brain mask and the Python stages run in Windows Slicer's Python. (R/docs/setup/MANAGED_SETUP.md; R/shared/managed_runtimes.py:31-33, 50-57, 77-84)
- GPU: "Auto PyTorch uses CUDA when available, otherwise CPU. **Apple MPS is not implemented/validated**". (R/docs/setup/SUPPORT_MATRIX.md:80)
- Compact-UI product list as shown in the screenshot (probe: viewed R/docs/images/slicerdds-compact-en.png): Preprocessing; Surface (native); Surface postprocess; Parcellation; Segmentation; Microstructure "DTI / Free-water / NODDI"; Tractography; Bundles; Connectome. In that image Microstructure, Tractography, Bundles and Connectome are ticked.

## 4. Implementation provenance hedges

Quoted (R/README.md:96-99): "DDSeg uses converted Python/PyTorch models; SAF and MK-Curve are unofficial Python ports of author MATLAB methods. PNL CNN masking runs a modified copy of the author's PNL inference code (licence terms in the PNL licence page)."

Provenance page (R/docs/architecture/IMPLEMENTATION_PROVENANCE.md):
- Preamble: "Language translation, successful execution, numerical equivalence and anatomical validity are different claims. This page ... does not assert author endorsement or clinical validation." (:11-13)
- DDSeg: "**MATLAB-to-Python/PyTorch reimplementation**, not a newly trained tissue model" (:21). "That complete original-runtime comparison is not claimed here." (a same-input, per-view MATLAB/PyTorch parity study is not claimed) (:45-47). The tissue route uses "five DTI features", not the author's ten-feature MK-Curve tissue network (:43-44).
- SAF: "**Unofficial Python port**, restricted to already co-registered native diffusion surfaces, with documented corrections" (:22). "whole-pipeline bitwise equivalence is **not** guaranteed." (:62-63). Upstream default 90 mm initial-track limit; validated runs used 40 mm (:65-67). "Describe outputs as **surface-constrained tractographic SAF candidates**. A curved display subset is not evidence that all output fibres are U-shaped." (:72-74). Neither mode "imposes adjacent-gyri labels or a U-shaped curvature classifier." (:69-70)
- PNL mask: "**Modified PyTorch re-implementation of the author's PNL network, with BWH licence terms kept** ... not a MATLAB port or a rewrite of the entire PNL pipeline; the native route is experimental" (R/docs/architecture/IMPLEMENTATION_PROVENANCE.md:23). The code is `shared/pnl_cnn/`, a modified copy of PNLBrainMask-Torch at commit 8d201a5 (R/shared/pnl_cnn/UPSTREAM.json).
- MK-Curve: "**Unofficial MATLAB-to-Python port**; distinct from standard DIPY DKI" (:24).
- DDSurfer: "Bundled upstream Python plus Slicer orchestration, not a MATLAB rewrite" (:25).
- DTI/Free-water/NODDI: "Library-backed computation" (DIPY/AMICO) (:26).
- WMA/WMQL/TractCloud: "**External author-code integration**, not local algorithm ports" (:27).
- Verification limits: reference checks use original `.m` kernels run in GNU Octave, "not MATLAB itself and not a blanket validation of every optional branch or subject." (:100-103)
- Third-party notice repeats: SAF "author endorsement and whole-pipeline bitwise equivalence are not implied." (R/THIRD_PARTY_NOTICES.md:55-56); MK-Curve "is not an official author-endorsed Python implementation." (:179-180)
- The English and the Chinese README now say the same about PNL: a modified copy of the author's PNL inference code (R/README.md:97; R/README.zh-CN.md, Implementation origins). The earlier "官方" (official) wording is gone.

## 5. Platform and validation status

- Validated: "Linux real-data and GUI evidence is documented in the acceptance campaign (SLICER_ACCEPTANCE_20261007.md); the compact interface has a separate UI validation record. These records describe specific checkouts, routes and datasets, not all future configurations." (R/README.md:158-163)
- Not certified, verbatim: "Native Windows/macOS scientific workflows and three-platform SDK builds are not certified." (R/README.md:174)
- "Fast CI and local installation checks are not substitutes for a real SDK/CTest/CPack build." (R/README.md:173)
- "The Extension Index JSON is a **Tier 1 proposal**, not acceptance by Slicer." (R/README.md:175)
- "Public source access, third-party licensing clearance and asset redistribution review remain release gates." (R/README.md:176)
- "**Current limitation:** no complete genuine SDK build or native Windows/macOS scientific acceptance is asserted by this change." (R/docs/release/EXTENSION_READINESS.md:69-71). Contract tests use "a clearly identified minimal SDK stub". (:70-71)
- Platform matrix (R/docs/setup/SUPPORT_MATRIX.md:34-40): Real Slicer GUI/source install: Linux "Dated local records"; Windows "Not certified"; macOS "Not certified". Scientific workflows: Linux "Only recorded routes/data"; Windows "Native not certified; WSL setup is not native acceptance"; macOS "Not certified". Real SDK/CTest/CPack: all three "Requires real SDK acceptance".
- PNL CNN mask: available on all three systems. Linux: managed conda runtime with ANTs 2.3.0. macOS and Windows: native route in Slicer's Python (SimpleITK and PyTorch), **experimental**: its masks did not meet the parity rule on 2 of 24 test cases; not run on Windows or macOS here. (R/docs/setup/SUPPORT_MATRIX.md, PNL CNN mask row; R/shared/brain_mask.py:142-162)
- "Windows/macOS do not inherit Linux QC or timings." (SUPPORT_MATRIX.md:51-52)
- Release gate table (R/docs/release/EXTENSION_READINESS.md:94-104): real SDK/CPack + clean install "Requires independent evidence"; model/template/fixture redistribution "Unconfirmed groups remain excluded from public asset archives"; real-data tutorial and Sample Data registration "Not completed"; patent: "Maintainer review required; no absence claim is made".
- Extension Index descriptor `docs/release/extension-index/SlicerDSpace.json`: tier 1, scm_revision main, build_dependencies PyTorch; "a schema-compatible draft, not a submitted catalog entry." (R/docs/release/extension-index/SlicerDSpace.json; EXTENSION_READINESS.md:106-109)
- CI: the README badge tracks main push runs; "the live badge ... is not externally accessible without authorization." (R/README.md:165-168)
- Documented real-data scope is Linux. Later validation records (CSD/ACT, platform audit, etc.) are listed in R/docs/README.md:47-92 but were not read. One recorded caveat: the CSD/ACT investigation "supersedes the earlier CSD acceptance ... old ACT-dependent outputs require regeneration." (R/docs/README.md:49-51)
- Film run platform: macOS, Apple M4, x86_64 Slicer 5.12.4 under Rosetta, CPU only. P/MAKING_OF.md:73-74, 144-145 ("SlicerDDSurfer is validated on Linux. This run used macOS ... and needed the run-time workarounds in §10.")

## 6. Install / quick start (as written)

Source/staging install (R/README.md:51-67). It is "a **source/staging installation**, not an official CPack release".
```bash
git clone https://github.com/ChengjinLii/SlicerDDSurfer.git
cd SlicerDDSurfer
PythonSlicer scripts/stage_slicer_extension.py \
  --install-root ~/SlicerDDSurfer-extension --link-mode copy --clean
```
- Then in Edit > Application Settings > Modules, add `~/SlicerDDSurfer-extension/SlicerDDSurfer/lib/Slicer-5.10/qt-scripted-modules` (expanded to an absolute path) to Additional module paths, then restart Slicer. Set `--slicer-version` to the actual version. Updating the source needs restaging and a restart. (R/README.md:61-67)
- Install PyTorch from Slicer's Extension Manager; setup adds UKFTractography and SlicerDMRI only for the products that need them. (R/README.md, Installation and Prepare sections)
- Model files live outside the checkout; choose a writable directory in Advanced settings > Environment and models. Review the asset terms first. (R/README.md:69-71)
```bash
PythonSlicer scripts/download_model_assets.py --profile surface \
  --asset-root ~/SlicerDDSurfer-model-assets --source url \
  --source-id slicerdsurfer_models_git
```
- Broader profile: `scripts/download_model_assets.sh --profile core`. "Availability and SHA256 verification do **not** establish redistribution rights." (R/README.md:79-81)
- Quick start steps (R/README.md:89-94): 1 select SlicerDDSurfer, choose English or Chinese; 2 enter Subject ID, Subject root, optional Output directory; 3 select only needed products; 4 open Advanced settings, configure routes, dependencies, models, check environment; 5 click Run SlicerDDSurfer (missing inputs show in red); 6 key results appear in the scene; review under Advanced settings > Results and QC.
- Canonical input layout (R/README.md:100-109):
```text
sub-01/
  input/
    data.nii.gz
    bvals
    bvecs
    nodif_brain_mask.nii.gz   # optional; generation needs a supported backend
```
- "DWI and mask must share a voxel grid. Gradient files must match the DWI measurements. Corrected gradients must accompany motion-corrected DWI." (R/README.md:111-113)
- Minimal DTI-only run and synthetic installation check: R/docs/setup/QUICKSTART.md:29-69. The synthetic subject is "a small constant tensor field with seven measurements, **not human anatomy**". "A redistributable real-data tutorial and Sample Data registration remain future milestones; neither is claimed as complete." (QUICKSTART.md:63-69)
- Release procedure files (RELEASE.md:17-25): three classes: source in this repo; model assets in the separate `SlicerDDSurfer-models` repo; external software installed separately.

## 7. Citation

Software entry (R/CITATION.cff:1-25):
- message: "If you use SlicerDDSurfer, please cite the tagged software release and the relevant component-method papers."
- type: software; title as in section 1; author Chengjin Li; version "0.1.0"; date-released "2026-06-24"; license BSD-3-Clause.
- preferred-citation: type article, same title, author Chengjin Li, year 2026, notes: "Manuscript in preparation; replace this entry with the final publication metadata after acceptance."
- README: "The SlicerDDSurfer protocol manuscript is in preparation. Cite the tested software version using CITATION.cff, together with the methods actually used". (R/README.md:197-199)
- No software DOI is recorded anywhere read. .zenodo.json has no DOI. RELEASE.md says the DOI is recorded after a release tag and archive (R/docs/release/RELEASE.md:158, 193-196).

Three method papers, exactly as CITATION.cff gives them (R/CITATION.cff:32-111):

1. Title: "DDSurfer: A Weakly‐Supervised Dual‐Stream Deep Learning Framework for Cortical Surface Reconstruction From Diffusion MRI" (the file uses non-ASCII hyphens). Authors: Chengjin Li, Wei Zhang, Xi Zhu, Yuqian Chen, Nir A. Sochen, Jarrett Rushmore, Carl‐Fredrik Westin, Yogesh Rathi, Lauren J. O'Donnell, Ofer Pasternak, Fan Zhang. Journal: Advanced Science. Year 2026. Article number e76596 (recorded as `start`). DOI 10.1002/advs.76596. (:32-61)
2. Title: "Deep learning based segmentation of brain tissue from diffusion MRI". Authors: Fan Zhang, Anna Breger, Kang Ik Kevin Cho, Lipeng Ning, Carl-Fredrik Westin, Lauren J. O’Donnell, Ofer Pasternak. Journal: NeuroImage. Year 2021. Volume 233. Article 117934. DOI 10.1016/j.neuroimage.2021.117934. (:62-83)
3. Title: "DDParcel: Deep Learning Anatomical Brain Parcellation From Diffusion MRI". Authors: Fan Zhang, Kang Ik Kevin Cho, Johanna Seitz-Holland, Lipeng Ning, Jon Haitz Legarreta, Yogesh Rathi, Carl-Fredrik Westin, Lauren J. O’Donnell, Ofer Pasternak. Journal: IEEE Transactions on Medical Imaging. Year 2024. Volume 43, issue 3, pages 1191-1202. DOI 10.1109/TMI.2023.3331691. (:84-111)

- CITATION.cff comment: the three papers "were verified individually against Crossref" (:27-31). THIRD_PARTY_NOTICES.md:152-164 repeats the three.
- Other paper cited in the repo: SAF, Shastin et al., "Surface-based tracking for short association fibre tractography", NeuroImage 2022, doi:10.1016/j.neuroimage.2022.119423 (R/THIRD_PARTY_NOTICES.md:45-49).
- Promo CREDITS gives the DDSurfer issue as "Advanced Science 13(56):e76596 (2026)" (P/CREDITS.md:12). CITATION.cff has no volume or issue for it. Crossref (api.crossref.org/works/10.1002/advs.76596, read 2026-10-24) gives volume 13, issue 56, article number e76596, as does anatomy/surface/ddsurfer_backend/README.md:34.
- Model-asset terms require citing the DDSeg and DDParcel papers (R/docs/assets/MODEL_ASSET_TERMS.md:37-38).

## 8. Licences

- Project code: BSD-3-Clause, the single root licence file R/LICENSE.txt. (R/README.md:207-208)
- "Bundled components retain their own licenses and notices." (R/README.md:208-209)
- "Model weights, templates and subject-derived data are not relicensed by the project license. Some component permissions remain unresolved; do not publish asset-inclusive archives until those permissions are documented." (R/README.md:211-213)
- LICENSE.txt:31-36: the licence "applies to the SlicerDDSurfer source code unless a file or bundled third-party component states different terms. Model weights, templates, example neuroimaging data, external tools, and Git LFS-managed assets may have separate source, redistribution, citation, or access restrictions."
- Model asset status (R/docs/assets/MODEL_ASSET_TERMS.md:21-26): "the redistribution and DOI-archive terms for bundled model weights, templates and subject-derived fixtures are not yet confirmed." The safe release mode is code and documentation only. Groups listed: DDCSR surface weights; HCP-like registration/template images; surface initialization meshes; DDSeg weights; DDParcel weights; conversion test NIfTI fixtures (MODEL_ASSET_TERMS.md:32-39).
- "An asset-inclusive DOI archive must not be created until every asset group above has confirmed redistribution and DOI-archive permission." (MODEL_ASSET_TERMS.md:75-78)
- Third-party notices summary (R/THIRD_PARTY_NOTICES.md):
  - FastSurfer-derived code (5 files under anatomy/parcellation): Apache-2.0; DZNE copyright header (:18-32).
  - SAF backend `tractography/saf_backend/`: MIT, from Dmitri Shastin's SAF; unofficial MATLAB-to-Python port, pinned to 7e2a9d15... (:36-56).
  - `anatomy/parcellation/conversion/` (from pnlbwh/conversion, Tashrif Billah): "BWH Conversion Contribution and Software License Agreement, Version 1.0", a BSD-style agreement, "not the project's BSD-3-Clause license"; modified integration (:60-88).
  - `FreeSurferColorLUT.txt`: FreeSurfer Software License Agreement, Version 1.0 (February 2011); covers the LUT only (:92-113).
  - DDSurfer backend snapshot `anatomy/surface/ddsurfer_backend/`: MIT (upstream licence added in commit 0c399ef); PyTorch3D-derived helpers: Meta BSD-3-Clause; conversion helpers keep the BWH terms (:117-134).
  - MK-Curve port: translated from Fan Zhang's MK-Curve (commit 9a213e1a...); upstream licence preserved in `licenses/MK-Curve-LICENSE.txt` (type not stated in the file); `mkcurve_tensor.py` and `mkcurve_directions.json` derive from DKI routines (c) 2017 NYU and University of Antwerp, MPL-2.0; "MATLAB I/O utilities and the proprietary MathWorks helper are not redistributed." (:166-181)
  - Limits: the audit "does not claim that every third-party right, patent or asset redistribution condition has been cleared." (:144-147)
- FSL: "limits most components to non-commercial use"; do not bundle FSL binaries (R/docs/architecture/PRODUCT_ARCHITECTURE.md:82-84).
- Film data and tools: see section 9 (EDDEN CC0, MIT, Apache-2.0, OFL).

## 9. Promo film facts

Shorter render of 4 November 2026 (user request: a shorter introduction and the new wordmark at the end): `P/out/SlicerDSpace_promo_1080p.mp4` (45,452,363 bytes, SHA-256 a28e5980f72f26004b845a91e9b8ebb2f3df3c5dfcf097dbf1ce092b07aa1973; 64.800 s, 1944 frames, -14.0 LUFS, true peak -1.2 dBFS; no black segment of 0.25 s or more with `blackdetect=d=0.25:pix_th=0.04`, the `P/build.sh check` setting; with the ffmpeg default `pix_th` 0.10 the dark beat between the near-silence and the wordmark, 57.53-57.93 s, counts as black, as the same beat did in the 72.0 s render at 64.73-65.27 s) and `P/out/SlicerDSpace.en.srt` (12 cues). Narration line 2 ("Mapping anatomy has usually meant a second, T1-weighted scan, registered into diffusion space."), the T1-weighted card and the registration arrow are removed; line 1 starts at 0.5 s, and every later line, cue and section is 7.2 s earlier than in the 28 October render below. The last frames show only the SlicerDSpace wordmark (`Resources/Icons/SlicerDSpace-wordmark.svg`, black set to the film's white) and the last subtitle; the cover images are unchanged. Site copy `docs/site/media/SlicerDSpace_promo_1080p.mp4`: H.264 CRF 24 encode, 23,356,794 bytes, SHA-256 0531f24d91f8abedc278899b43649eee3469fcdd71b762e6d6cdb5afe204abb7, 1944 frames, AAC payload unchanged, full-frame SSIM 0.993841 (`scripts/optimize_website_video.py`; receipt in `published_assets.json`). The narration list and the tables below describe the 72.0 s renders; for the current times subtract 7.2 s from every time after 12.0 s, and drop line 2.

Rename render of 28 October 2026: `P/out/SlicerDSpace_promo_1080p.mp4` (47,767,254 bytes, SHA-256 4ead178f6d1165b98d0c5b61b1c94e250ac53d7893c5fae2d7847d6cdd1e13b5; 72.000 s, 2160 frames, -14.0 LUFS, LRA 3.1 LU, true peak -1.1 dBFS) and `P/out/SlicerDSpace.en.srt` (13 cues, 1,131 B). Only the name changed: on-screen and spoken "SlicerDDSurfer" became "SlicerDSpace" (spoken "Slicer D Space"), shot 11 shows a new capture of the renamed module, and the end-card icon is the SlicerDSpace logo. Cue 13 ends at 1:11.649. The facts below describe the 24 October render; in the narration, read SlicerDSpace for SlicerDDSurfer.

File `P/out/SlicerDDSurfer_promo_1080p.mp4`, render of 24 October 2026 (72.0 s, 2160 frames). Earlier deliveries: the first cut (64.8 s, 41,070,370 bytes, 1944 frames, 12 cues) and the 23 October render (72.0 s, 50,757,754 bytes). Re-measured on 24 October 2026 with the commands below.

| Fact | Value | Source |
|---|---|---|
| Duration | 72.000000 s | `ffprobe -v error -show_entries format=duration,size,bit_rate:stream=index,codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,color_range,color_space,sample_rate,channels,bit_rate,nb_frames -of compact` |
| File size | 47,838,648 bytes (about 45.6 MiB) | same command; `ls -l` |
| SHA-256 | 123e1d20ebb637aca1d8a8d2e02ac5967a1e7ebe7a34956edd19b2916d4ba4a1 | `shasum -a 256` |
| Video | h264, 1920x1080, yuv420p, 30/1 fps | same ffprobe command |
| Video detail | profile High, color_range tv (limited), bt709, 2160 frames, about 5.04 Mb/s (file about 5.32 Mb/s) | same; P/MAKING_OF.md:11 |
| Audio | aac (LC), 48000 Hz, 2 channels, about 264 kb/s | same |
| Streams | exactly 2: video and audio. No subtitle stream. | same (stream index and codec_type) |
| Subtitles | burned into the picture, plus sidecar SRT with 13 cues | P/MAKING_OF.md:3, 11-12; cue count from P/out/SlicerDDSurfer.en.srt (13 numbered cues; build log: "13 cues", P/logs/build_20261024_type.log:7-8) |
| Loudness | -14.0 LUFS integrated, LRA 3.0 LU, true peak -1.2 dBFS (`ffmpeg -af ebur128=peak=true` on the final MP4); no black segment of 0.25 s or more (`blackdetect`) | P/MAKING_OF.md:133. The build log P/logs/build_20261024_type.log:88-98 records the same values, but it was written before the motion-blur step (log 00:42, final MP4 01:51 on 24 October); the values were measured again on the final file |
| moov / mdat order | ftyp@0 (32 B), moov@32 (80,111 B), free@80143 (8 B), mdat@80151 (47,758,497 B). moov is BEFORE mdat (faststart OK). | `python3 -I` top-level box scan |
| Cover images | cover_1920x1080.png = 1920x1080 (1,267,791 B); cover_1280x640.png = 1280x640 (607,718 B); the 1280x640 is "GitHub social-preview size (centre crop of the cover)" | `sips -g pixelWidth -g pixelHeight`; `ls -l`; P/MAKING_OF.md:13-14 |
| Motion blur | On the two fast Slicer-window moves only (pull-back 55.2-57.4 s, glide 57.6-59.2 s). Frames 1653-1778 (55.1-59.3 s, 126 frames) are replaced. 180 degree shutter; per frame 8 to 47 sub-frames (minimum 8, maximum used 47), so that no visible point of the window moves more than 1 px between sub-frames; averaged in linear light. Only the GOPs that hold a replaced frame (frames 1620-1869) are encoded again. The renderer itself has no motion blur. | P/MAKING_OF.md:24-35; settings and per-frame values in P/logs/motion_blur.json (`n_min` 8, `n_max_used` 47, `shutter_deg` 180, `frames_replaced` 126, `reencoded_frames` [1620, 1869]); tools `P/tools/motion_blur.mjs` and `P/tools/motion_blur_avg.py`. Frames 0-1578 and 1959-2159 are bit-identical to the film before the blur (P/MAKING_OF.md:135) |

Narration, 13 lines verbatim (P/out/SlicerDDSurfer.en.srt; times in file; the narration names no bundle method):
1. (0:00.8-0:04.95) "Diffusion MRI records how water moves through the living brain." (srt:3)
2. (0:05.0-0:11.55) "Mapping anatomy has usually meant a second, T1-weighted scan, registered into diffusion space." (srt:7)
3. (0:12.1-0:19.25) "SlicerDDSurfer reconstructs anatomy directly from diffusion MRI, without a T1-weighted scan." (srt:11)
4. (0:19.3-0:24.35) "It maps microstructure, from DTI to free water, NODDI and kurtosis." (srt:15)
5. (0:24.4-0:26.2) "Tissue labels." (srt:19)
6. (0:28.0-0:29.8) "Brain regions." (srt:23)
7. (0:31.4-0:33.35) "White and pial surfaces," (srt:27)
8. (0:33.4-0:35.722) "all in native diffusion space." (srt:31)
9. (0:37.0-0:41.018) "Whole-brain tractography, traced in the same native space." (srt:35)
10. (0:46.0-0:49.224) "White-matter bundles, from the same tractogram." (srt:39)
11. (0:50.8-0:54.531) "And structural connectomes, built on the same anatomy." (srt:43)
12. (0:55.8-1:01.3) "One 3D Slicer module, with automatic QC and a record of every run." (srt:47)
13. (1:05.4-1:11.95) "SlicerDDSurfer. Brain anatomy from diffusion MRI, inside 3D Slicer." (srt:51)

Voice: Kokoro TTS reads respelled text ("T one weighted", "Slicer D D Surfer", "nod-ee", "pie-al" ...); subtitles show the written text (P/MAKING_OF.md:143-145). "The ASR check is automatic; no person has listened to the final mix yet." (P/MAKING_OF.md:149; also :131: "No person has listened to the mix")

Bundles scene (45.6-50.4 s):
- Model and tractogram: TractCloud v1.0.0 on the film's own DTI deterministic tractogram (the same 130,780-streamline file that the Tractography scene samples). The model ran on 23 October 2026. 42 bundles; 53,522 classified streamlines; 8,994 drawn (display subset). (P/MAKING_OF.md:99; P/film/FACTS.md:15)
- What the viewer sees: the whole-brain fibres dim to grey; the 42 bundles arrive family by family (commissural, projection, association, cerebellar, superficial), one colour each. (P/MAKING_OF.md:73-76, 99; P/film/TREATMENT.md:7-10)
- On-screen text: title "White-matter bundles", note "42 tracts · display subset". (P/film/content.json:65-69)
- The Bundles scene names no method, in the narration (line 10, srt:39) or on screen (user request, 24 October 2026). (P/MAKING_OF.md:164-166; P/film/FACTS.md:15)
- The 23 October cut also showed WMQL tracts from the multi-shell run (two bundle methods). The 24 October cut's Bundles scene does not. The 3D Slicer screenshot later in the film still shows the module's Bundles row ("WMQL + WMA + TractCloud") and, in its 3D view, the CSD/ACT tractogram with bundles (P/runtime/shots/final_ms3/window.png). The scene's fibre nodes are that tractogram, the WMQL AF and CST bundles (left and right) and four TractCloud bundles: AF, CC1, CC2 and CST (P/runtime/shots_workflow_ms3/capture.log:13). (P/MAKING_OF.md:99; P/film/TREATMENT.md:20-21)

Data used:
- EDDEN (Evaluating Diffusion MRI Denoising), OpenNeuro ds004666, subject sub-01, session ses-1p5mm, licence CC0. Used: topup + eddy corrected `AVG_mag` DWI (5 repeats averaged), bval/bvec, brain mask, and the T1-weighted image "shown only on the 'problem' card". (P/CREDITS.md:4-5)
- Citation: Manzano-Patron, J. P. et al., Imaging Neuroscience (2024). (P/CREDITS.md:5). Primary-source wording: "Manzano-Patron, J.P. et al, Denoising Diffusion MRI: Considerations and implications for analysis, Imaging Neuroscience, 2024 (https://doi.org/10.1162/imag_a_00060)". (P/docs/research/DATA_LICENSES.md:21-22)
- Licence evidence: `dataset_description.json` "License": "CC0"; "No attribution is legally required (CC0); cite the paper as a courtesy." (DATA_LICENSES.md:11, 20-22). Dataset DOI doi:10.18112/openneuro.ds004666.v1.0.8 (snapshot 1.0.8, 2026-01-25) (DATA_LICENSES.md:19).
- Subject: one healthy volunteer (DATA_LICENSES.md:24). Defacing of the T1w is not stated in the dataset README: "UNVERIFIED" (DATA_LICENSES.md:35).
- Volumes used in the pipeline: only b0 and b about 1000: 27 + 90 = 117 of 297 volumes; the hook readout shows all 297 (P/MAKING_OF.md:90, 110-113).
- End-card credit: "Data: EDDEN, OpenNeuro ds004666 (CC0) · Manzano-Patron et al. 2024" (P/film/content.json:91; P/MAKING_OF.md:179; visible in the cover PNGs).

Software shown:
- SlicerDDSurfer "repo HEAD a48fc95", run headless on the EDDEN data (P/CREDITS.md:9). The run started from the command line, not the Run button (P/MAKING_OF.md:61-62).
- 3D Slicer 5.12.4 (Slicer License, BSD-style), SlicerDMRI, PyTorch extension (P/CREDITS.md:14). x86_64 Slicer under Rosetta on an Apple M4, CPU only (P/MAKING_OF.md:114-115).
- Methods cited: DDSeg, DDParcel, DDSurfer (as the model-asset terms ask) and TractCloud, whose bundles the film shows (P/CREDITS.md:9-13).
- The Slicer section (55.2-64.8 s) is a real screenshot of the module in 3D Slicer 5.12.4, from `runtime/shots/final_ms3/window.png`. It shows Results and QC open on the multi-shell run `ms_micro`. The film shows it as a floating window that zooms out and then glides beside a step list ("Select products", "Run", "Review results & QC"). (P/MAKING_OF.md:46-53, 77-81, 101; P/film/content.json:79-86)

Production credits (P/CREDITS.md:17-24):
- lemo-opuscar: MIT, (c) 2026 LemoLab, vendored.
- three.js r170: MIT.
- Voice: Kokoro TTS (Apache 2.0), voice `af_heart`, offline.
- Speech check (QA only, not in film): faster-whisper `base.en`.
- Fonts, SIL Open Font License 1.1: Inter Tight and Inter ((c) The Inter Project Authors); IBM Plex Mono ((c) IBM Corp.).
- Music and effects: original, synthesized by code (`film/audio/mix.py`, numpy; physical-model plucks). "No samples, no stock audio."
- Icon: `Resources/Icons/SlicerDDSurfer.png` from the repo.

Display-subset caveats (P/MAKING_OF.md:36-45):
- "Tractography: 24,000 of 130,780 streamlines (length >= 30 mm, random subset, seed 7), resampled every 2.5 mm. This is our display subset; it is not the module's own VTK preview and not the analysis tractogram." (:43-44)
- "Connectome: the 260 strongest of 854 edges are drawn; the matrix panel shows all 84 x 84 nodes." (:45)
- Bundles: 8,994 of 53,522 classified streamlines are drawn (P/MAKING_OF.md:99). The film's note reads "42 tracts · display subset" (P/film/content.json:68).
- Volumes upsampled 2x for textures; DEC colour = |principal eigenvector| x FA; white matter labels 2 and 41 drawn grey; GM orange, WM ivory, CSF blue (:39-42).
- The Slicer window is "a **real screenshot** of 3D Slicer 5.12.4 with the SlicerDDSurfer module." Edits to the first screenshot: status bar cropped; 3D view box and axis labels hidden; lateral camera; "Advanced settings > Results and QC" opened; module inputs and ticked products set by script to match the run (six ran: microstructure, segmentation, parcellation, surface, tractography, connectome). (:54-62) The 23 October screenshot replaced it and the status bar is cropped as before (:46-53).
- The tractography shown is DTI deterministic. The film labels it "DTI tractography" with the note "Display subset · 24,000 of 130,780 streamlines" (P/film/content.json:60-63; P/MAKING_OF.md:161-163). The 23 October cut labelled it "DTI tractography · preview" (P/MAKING_OF.md:162-163).
- Not run for the film's main run (`promo`) and not shown: Free-water, NODDI, DKI, CSD, ACT, SIFT2, SAF, WMQL/WMA, surface postprocess (P/MAKING_OF.md:159-163, 199-200). TractCloud did run on the film's DTI tractogram and its bundles are shown (P/MAKING_OF.md:99, 199-200). The 23 October website-gallery runs added UKF, CSD, ACT, SIFT2, free water, NODDI, DKI and WMQL/WMA/TractCloud on other tractograms; the film does not show them (P/MAKING_OF.md:200-202).
- The narration names free water, NODDI and kurtosis. On screen, the rows Free-water, NODDI and DKI carry the note "optional" (P/film/content.json:22-33; P/film/FACTS.md:11; P/MAKING_OF.md:159-160). Until 23 October the film showed them as chips; there are no chips now (P/MAKING_OF.md:63-65, 159-160).
- One subject, one session; EDDEN AVG_mag averages 5 repeats so SNR is higher than a typical single scan (P/MAKING_OF.md:195). EDDEN publishes no eddy-rotated bvecs; original bvecs were used (:196).
- QC state of the run: automatic QC passed for DTI, DDParcel, DDSurfer; DDSeg world-space QC passed (DDSeg-vs-DDParcel WM Dice 0.848, gate >= 0.70); tractography and connectome "completed"; `human_review` is "pending": "nobody reviewed the outputs as an expert." (P/MAKING_OF.md:117-119)
- Run times (one subject, Rosetta, CPU): DTI 44 s, tractography 50 s, DDSeg 95 s, DDParcel 1956 s, DDSurfer 1089 s, connectome 12 s (P/MAKING_OF.md:116). Own checks: FA vs dataset FA Pearson r = 0.826; principal direction mean |cos| = 0.952 (:125-126).
- Render note: renders are not always bit-identical (P/MAKING_OF.md:212-214).
- Film end card text: "Diffusion MRI workflows in 3D Slicer" and "Research software · Not for clinical use" (P/film/content.json:89-90; P/film/FACTS.md:18-19; visible in cover PNGs). The repository URL is not shown (P/film/FACTS.md:26; `endcard.url` is empty in P/film/content.json:92).
- Claims the film says it does NOT make: "Open source / available now / Extension Manager / version or date; speed or accuracy numbers; clinical use; 'first' or 'only'; cross-platform support; 'U-shaped fibres'; 'no registration'; 'no preprocessing'; 'fully automatic'." (P/film/FACTS.md:24-25)

## 10. Image assets

| File | Size (px) / bytes | Shows | Source and licence |
|---|---|---|---|
| R/Resources/Icons/SlicerDDSurfer.png | 1024x1024; 1,389,926 B (`sips`, `ls`) | White brain with two wave lines on a blue square; text "DDSURFER". Byte-identical to P/film/assets/icon.png (`cmp`). Used as the Slicer module icon (s4ext:12). | Project file. Project licence is BSD-3-Clause unless a file states otherwise (LICENSE.txt:31-32). No separate record of the icon's author or origin in the files read. |
| R/docs/images/slicerdds-compact-en.png | 440x651; 74,455 B | Real Qt widget capture of the compact module panel (EN). Empty inputs shown in red. Log: "Validation failed. Missing entries: Subject ID, Subject directory". Not a result. | Genuine capture from the isolated Slicer 5.10 compact-UI smoke test; source tree 4993990b..., main commit 37a543c; "copied without generative modification"; "no patient data/private paths" (R/docs/images/README.md:13-21). Project licence applies. Listed as the extension screenshot (s4ext:13). |
| R/docs/images/slicerdds-compact-zh.png | 426x684; 84,280 B | Same panel in Chinese. | Same as above. |
| P/out/cover_1920x1080.png | 1920x1080; 1,267,791 B | Dark cover: icon, "SlicerDDSurfer", "Diffusion MRI workflows in 3D Slicer", "Research software · Not for clinical use", a DEC-coloured tractography brain in a glass surface on the right, credit line "Data: EDDEN, OpenNeuro ds004666 (CC0) · Manzano-Patron et al. 2024". | Rendered by the film page (`?cover=1`); data from EDDEN sub-01 (CC0); tractography is the display subset (24,000 of 130,780). Film tools and fonts per section 9 (P/MAKING_OF.md:43-44). No licence for the cover itself is stated in P/CREDITS.md. |
| P/out/cover_1280x640.png | 1280x640; 607,718 B | Centre crop of the same cover. | Same. |
| P/runtime/shots/final_ms3/window.png | 3840x2054; 2,999,948 B | Real 3D Slicer 5.12.4 screenshot with the SlicerDDSurfer module (24 October film; the 23 October and earlier cuts used `runtime/shots/final/window.png`): Subject ID sub-01, products ticked, Results and QC on the run "2026-10-23 20:26 · sub-01 · ms_micro · completed" ("Recorded QC: 4 passed, 0 issues, 0 unavailable | Visual review: pending"), DTI FA in the three slice views ("sub-01_microstructure.dti_fa") and the CSD/ACT tractogram with bundles in the 3D view. | Source of the film's Slicer section (P/MAKING_OF.md:46-53). The file shows a red error indicator at the bottom right (SlicerDMRI "No data to glyph!" and macOS messages; `docs/review_backlog.md`). |
| P/runtime/shots/final/view3d.png | 3000x886; 1,960,731 B | 3D view capture (not opened). | Same run. |
| P/out/SlicerDDSurfer.en.srt | 13 cues, 1,135 B | Subtitles | Section 9. |
| P/out/picture.mp4 | 109,619,433 B (24 October; `out/picture_blur.mp4` is the motion-blurred copy) | Intermediate render without final audio mix. Listed in P/.gitignore:22-23 ("the muxed film in out/ is the deliverable"). | Do not publish. |

Items NOT to publish (P/MAKING_OF.md:147-152; P/.gitignore:1-17): `runtime/model-assets/` (weights and templates), `runtime/stage/` (copy of private code), `data/edden_ds004666/`, `runtime/subjects/`, local environments. "No bundle has been made or checked yet." (MAKING_OF.md:151-152)

## 11. External URLs found in the sources

| URL | Where |
|---|---|
| https://github.com/ChengjinLii/SlicerDDSurfer (repo; private, section 12) | R/README.md:55; R/CITATION.cff:8-9; R/codemeta.json:6; R/SlicerDDSurfer.s4ext:4, 8 |
| https://github.com/ChengjinLii/SlicerDDSurfer/actions/workflows/ci.yml (CI badge; not public) | R/README.md:7 |
| https://www.slicer.org/ | R/README.md:9 |
| https://doi.org/10.1002/advs.76596 (DDSurfer) | R/README.md:199; R/CITATION.cff:60 (doi field) |
| https://doi.org/10.1016/j.neuroimage.2021.117934 (DDSeg) | R/README.md:200; R/docs/architecture/IMPLEMENTATION_PROVENANCE.md:41 |
| https://doi.org/10.1109/TMI.2023.3331691 (DDParcel) | R/README.md:201 |
| https://doi.org/10.1016/j.neuroimage.2022.119423 (SAF paper) | R/THIRD_PARTY_NOTICES.md:49; IMPLEMENTATION_PROVENANCE.md:74 |
| https://doi.org/10.1162/imag_a_00060 (EDDEN paper, Imaging Neuroscience 2024) | P/docs/research/DATA_LICENSES.md:22; P/film/FACTS.md:19 |
| https://openneuro.org/datasets/ds004666 (EDDEN) | P/docs/research/DATA_LICENSES.md:19 |
| doi:10.18112/openneuro.ds004666.v1.0.8 (dataset DOI, as written, no URL form) | P/docs/research/DATA_LICENSES.md:19 |
| https://raw.githubusercontent.com/ChengjinLii/SlicerDDSurfer/main/Resources/Icons/SlicerDDSurfer.png and .../docs/images/slicerdds-compact-en.png (not resolvable while the repo is private) | R/SlicerDDSurfer.s4ext:12-13; R/CMakeLists.txt:11-12; R/docs/images/README.md:21-22; R/docs/release/EXTENSION_READINESS.md:109 |
| Upstream projects: https://github.com/zhangfanmark/DDSeg ; https://github.com/dmitrishastin/SAF ; https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation ; https://github.com/zhangfanmark/MK-Curve ; https://github.com/Deep-MI/FastSurfer ; https://github.com/pnlbwh/conversion ; https://github.com/ChengjinLii/DDSurfer | R/docs/architecture/IMPLEMENTATION_PROVENANCE.md:21-24; R/THIRD_PARTY_NOTICES.md:28, 41, 64, 122, 170 |
| https://dmri.slicer.org/ (SlicerDMRI) ; https://github.com/lassoan/SlicerTotalSegmentator | R/docs/release/EXTENSION_READINESS.md:118-119; R/docs/architecture/PRODUCT_ARCHITECTURE.md:98 |
| Slicer extension guide https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html ; Extensions Index checklist https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md | R/docs/architecture/PRODUCT_ARCHITECTURE.md:18, 85; EXTENSION_READINESS.md:115-117 |
| FSL, FreeSurfer, PyTorch MPS pages | R/docs/setup/SUPPORT_MATRIX.md:94-98; PRODUCT_ARCHITECTURE.md:83 |
| Slicer SDK / HPC links not listed. No project homepage other than the repo exists in the files read (`homepage` = repo URL). | R/SlicerDDSurfer.s4ext:8 |

No URL was probed (no network call was made). The repo, CI badge and raw.githubusercontent links are stated by the sources to be private or unreachable. The DOIs, slicer.org, dmri.slicer.org, OpenNeuro and the upstream GitHub projects are public services whose resolution was not checked.

## 12. Contradictions and gaps

Contradictions between sources:
1. Slicer version. README says 5.10 (README.md:9, 62, 64; images README:14). The film credits 3D Slicer 5.12.4 (P/CREDITS.md:14; MAKING_OF.md:34, 73). The 5.12.4 run was macOS/Rosetta and is not the Linux validation.
2. Licence. R/docs/architecture/PRODUCT_ARCHITECTURE.md:82 says "the extension's MIT license". LICENSE.txt, README, CITATION.cff, .zenodo.json, codemeta say BSD-3-Clause. (The DDSurfer backend inside is MIT; THIRD_PARTY_NOTICES.md:117-126.)
3. Platforms. codemeta.json:12-18 lists runtimePlatform "Linux, macOS, Windows" and developmentStatus "active". README:174 and SUPPORT_MATRIX say Windows/macOS scientific workflows are not certified.
4. Tag. RELEASE.md:163 mentions "the existing `v0.1.0` tag" and a draft release file. `git tag -l` in R/ lists no tag. CITATION.cff:2 asks users to cite "the tagged software release". HEAD is dated 2026-10-16; date-released is 2026-06-24.
5. Clinical wording. The end card says "Not for clinical use" and P/film/FACTS.md:18 attributes it to the README. The README wording is "A completed run is not a clinical assessment or proof of scientific correctness." (README.md:21-22). The words "Not for clinical use" do not appear in the README.
6. "Official". Settled: both READMEs now describe PNL as a modified copy of the author's inference code; neither calls it "official".
7. DDSurfer citation detail. CITATION.cff has year 2026 and article number e76596 only; P/CREDITS.md:12 adds "13(56)". Settled: Crossref gives 13(56) (§7); the site's [1] gives it since 2026-10-24.
8. Description text differs across README, s4ext, CITATION.cff, codemeta, .zenodo.json (section 1). No file defines one official one-line description. README tagline: "Diffusion MRI workflows in 3D Slicer".
9. Final narration line 12 ("Brain anatomy from diffusion MRI, inside 3D Slicer.") has no row of its own in P/film/FACTS.md. The FACTS row for the end card covers the tagline "Diffusion MRI workflows in 3D Slicer" only.

Gaps / claims the sources do NOT support:
10. T1w-independence is not stated in the repo. It appears in P/film/FACTS.md and rests on the three papers (not re-read here) plus the README input layout. Allowed qualifier: "at analysis time". Models were trained with T1w-derived labels (P/film/FACTS.md:10). The sources read do not say whether any optional route (for example "Surface postprocess" FreeSurfer-style refinement, or FreeSurfer) needs a T1w. The README never says "no T1" for the whole module.
11. Availability and openness. The sources say private pre-release, not in the Extensions Catalog, repository private, Index entry only a draft (section 1). Not supported: "open source", "available now", "install from Extension Manager", "download", a release version or date, or any public repo/raw-image link. README:66 says to install the dependencies SlicerDMRI and PyTorch from Extension Manager; SlicerDDSurfer itself is not installed that way. A public site that links the repo or says "available" would contradict RELEASE.md:38-39 and EXTENSION_READINESS.md:11-13 (keep private until an explicit maintainer decision).
12. Performance numbers. No speed, accuracy or benchmark claim is made for the product in the sources. Film numbers (run times, r = 0.826, Dice 0.848, |cos| = 0.952) are one subject, CPU, Rosetta, not benchmarks; the film does not claim them. "Full-brain timing missing" for DKI/MK-Curve (SUPPORT_MATRIX.md:61). Paper speed claims are not in the files read.
13. Clinical use. Not supported. "A completed run is not a clinical assessment" (README.md:21-22). Human review of the film run is "pending" (MAKING_OF.md:78).
14. Cross-platform. Not supported (README.md:174). Linux only for recorded real-data/GUI evidence. The film run was on macOS and needed workarounds (MAKING_OF.md:144-145, 166-181: six maintainer findings, none fixed in the repo).
15. Features shown versus run. Free-water, NODDI, DKI, CSD, ACT, SIFT2, SAF, WMQL/WMA/TractCloud and surface postprocess were not run for the film. Update, 23 October 2026: the 72 s film adds TractCloud (run that day on the film run's DTI tractogram) and WMQL (multi-shell CSD/ACT run of that day); free water, NODDI and DKI ran that day but the film does not show them. The film's tractography is a display subset of a DTI deterministic run (24,000 of 130,780 streamlines); connectome edges 260 of 854. The film run used the CLI, not the Run button.
16. Screenshots. The docs images show a missing-input state, not a result, and the Microstructure label there reads "DTI / Free-water / NODDI" (no DKI). The window.png screenshot shows the label "DTI + optional models". The UI labels differ between the two captures (Slicer 5.10 doc image vs 5.12.4 film image).
17. Acceptance records not read (SLICER_ACCEPTANCE_20261007.md, UI_SIMPLIFICATION_20261008.md, DISTRIBUTION_VALIDATION_20261012.md). The datasets, subject counts and test counts behind "Linux real-data validation" are not on this sheet. Do not state them.
18. "First" or "only", "fully automatic", "no registration", "no preprocessing", "U-shaped fibres": not supported; film FACTS.md rules them out (:21-25), provenance page rules out U-shaped (:72-74).
19. Authors and affiliation. One author, Chengjin Li. No lab or affiliation in the files read. Co-authors appear only on the method papers.
20. Asset licences. Weights, templates and fixtures: redistribution unconfirmed (MODEL_ASSET_TERMS.md:21-26). Do not offer asset downloads on the site. The cover and window screenshots are derived outputs of CC0 data (DATA_LICENSES.md:192-195), but no licence is stated for the images themselves, and window.png shows local paths.
21. Software DOI: none exists in the files read. No Zenodo record is recorded.

## 13. Integration addendum: measured performance (22 October 2026)

The observations above remain the historical film/site source review. The new
`performance.html` subpage is based on separate, subsequently checked benchmark
records, not film timings. Its seven public records are selected from
`freewater_tensor_helpers_full_20261022.json`, `complete_models_20261020.json` and
`author_wma_optimizations_20261020.json` under `docs/release/benchmarks/`.
`scripts/build_performance_site.py --check` verifies the source hashes and
rendered data. Only the curated public summary is deployed.

Full-module, ROI, substage, multithreading and Python-port references must remain
distinct. Output equivalence is not an assertion that scientific QC or expert
anatomical review has passed. WMA small-bundle sampling and invalid model voxels
are not hidden. No whole-workflow, original-MATLAB-port speedup or improved
scientific validity is claimed. Repository visibility and film bytes are unchanged.
