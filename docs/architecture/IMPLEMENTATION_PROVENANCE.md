<div align="center">

# Implementation Provenance

**Original methods, Python ports and external adapters**

[Project](../../README.md) | [Documentation](../README.md) | [SAF validation](../release/SAF_PYTHON_20261013.md) | [中文](IMPLEMENTATION_PROVENANCE.zh-CN.md)

</div>

Language translation, successful execution, numerical equivalence and anatomical
validity are different claims. This page identifies what the extension actually
runs; it does not assert author endorsement or clinical validation.

---

## Implementation Map

| Component | Original implementation | SlicerDSpace execution | Correct description |
| --- | --- | --- | --- |
| DDSeg | [Author MATLAB pipeline and trained U-Nets](https://github.com/zhangfanmark/DDSeg) | `anatomy/segmentation/`: Python preprocessing/inference and converted PyTorch/TorchScript networks | **MATLAB-to-Python/PyTorch reimplementation**, not a newly trained tissue model |
| SAF | [Author MATLAB framework](https://github.com/dmitrishastin/SAF/tree/7e2a9d152caf1cf3e6bface64e7e176faac0c14f) | `tractography/saf_backend/`: NumPy/SciPy/scikit-learn, NiBabel and external MRtrix | **Unofficial Python port**, restricted to already co-registered native diffusion surfaces, with documented corrections |
| PNL diffusion CNN mask | [Author Python/TensorFlow script](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation), via the [PyTorch version PNLBrainMask-Torch](https://github.com/ChengjinLii/PNLBrainMask-Torch) | `shared/pnl_cnn/` is a modified copy of PNLBrainMask-Torch (pinned commit, hashes in `UPSTREAM.json`); `shared/brain_mask.py` runs it natively in Slicer's Python (SimpleITK registration) or, on Linux, runs the installed package with ANTs in the managed runtime | **Modified PyTorch re-implementation of the author's PNL network, with BWH licence terms kept** ([licence page](../licenses/pnl_brain_mask.md)); not a MATLAB port or a rewrite of the entire PNL pipeline; the native route is experimental |
| MK-Curve | [Author MATLAB implementation](https://github.com/zhangfanmark/MK-Curve/tree/9a213e1a5850fe50c998793f31ed852df00242f8) | `microstructure/mkcurve.py` and `mkcurve_tensor.py`; optional external MATLAB reference | **Unofficial MATLAB-to-Python port**; distinct from standard DIPY DKI |
| DDSurfer surface reconstruction | Author Python/PyTorch project | Pinned `anatomy/surface/ddsurfer_backend/` snapshot, recorded in `UPSTREAM.json` | Bundled upstream Python plus Slicer orchestration, not a MATLAB rewrite |
| Cortical postprocessing | Original FreeSurfer C++ algorithms | Pinned DDSurfer recipe compiles seven commands locally, or uses an existing FreeSurfer installation | **Original native-code reuse**, not a Python rewrite; no full suite required for the source-built route; [setup and limits](../setup/SURFACE_COMPONENTS.md) |
| DTI tensor fit | SlicerDMRI `DWIToDTIEstimation` and the Teem library (Slicer core) | `shared/teem_dti.py` calls Teem in Slicer's Python and writes the tensor, FA and the colour FA map | **Modified Python re-implementation of the SlicerDMRI estimator**, under the Slicer licence ([notice](../../THIRD_PARTY_NOTICES.md)); the plugin does not redistribute Teem |
| Free-water / NODDI | DIPY / AMICO Python implementations for the selected route | Input validation, fitting adapters, saved parameters and QC | Library-backed computation; do not describe these as local MATLAB translations |
| iFW | Published equations (Cho et al. preprint, Supplementary Methods 1) | `microstructure/ifw.py` | Python implementation of the published formula; follows the manuscript's reference function |
| Free-water WLS optimization | Pinned DIPY `wls_iter` Python estimator | Source-qualified `microstructure/freewater_wls.py` removes the materialized diagonal; serial NLS uses the same initialization | **Modified equivalent Python implementation**, not a new estimator or MATLAB translation; [paired evidence](../release/PERFORMANCE_OPTIMIZATIONS.md) and [DIPY license](../../licenses/DIPY.txt) |
| WMA / WMQL / TractCloud | External author Python/shell pipelines | Fresh-run adapters, original entry points and saved provenance; TractCloud has a seed/logger compatibility bootstrap | **External author-code integration**, not local algorithm ports; [real-output checks](../release/BUNDLE_BACKENDS_20261013.md) distinguish execution, integrity and anatomical review |

Source-code licenses and model redistribution permissions remain separate. See
[third-party notices](../../THIRD_PARTY_NOTICES.md) and
[model terms](../assets/MODEL_ASSET_TERMS.md).

---

## DDSeg

The original `DAGNetwork` parameters are exported from MATLAB as numeric weights,
biases and input-mean images, then mapped to PyTorch layers and TorchScript
models. Python handles masking, normalization, three-view inference, view
combination, class probabilities and native-grid export. The original method
must still be cited: [Zhang et al., NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.117934).

The current tissue route uses **five DTI features**, not the author's separate
ten-feature MK-Curve tissue network. Adding DKI maps does not silently change the
DDSeg model. Current probability/world-coordinate checks and real-subject
validation do not substitute for a same-input, per-view MATLAB/PyTorch parity
study. That complete original-runtime comparison is not claimed here.

---

## SAF

The Python port follows cortical endpoint eligibility (GG), same-hemisphere
selection (HH), optional pial exclusion (PG), the initial-track length limit,
white-surface crossing/truncation (GWG) and surface summaries. It is not the
entire MATLAB execution environment: the facade requires native-diffusion-space
surfaces and does not perform structural-to-diffusion registration.

Documented differences include deterministic scikit-learn k-means, corrected
endpoint/face indexing, retained complete-mode edge/vertex intersections, safe
bookkeeping for one-/two-point tracks and original-index mapping for hard-split
weights. Clustering implementations can select different nearest clusters;
whole-pipeline bitwise equivalence is **not** guaranteed.

The upstream defaults include a **90 mm** initial-track limit, disabled pial
filtering and a soft outer-crossing GWG rule. The validated representative runs
instead explicitly used **40 mm**. Soft GWG preserves the chain between the
outer crossings; it does not prove that every interior point stays in white
matter. `gwg_hard` has different segmenting semantics. Neither mode imposes
adjacent-gyri labels or a U-shaped curvature classifier.

Describe outputs as **surface-constrained tractographic SAF candidates**. A
curved display subset is not evidence that all output fibres are U-shaped. Cite
[Shastin et al., NeuroImage 2022](https://doi.org/10.1016/j.neuroimage.2022.119423).
Old endpoint-density/length/scalar projections must be recomputed with revision
`python-native-20261013`. Resume rejects missing/older implementation identities.
See the [scoped verification record](../release/SAF_PYTHON_20261013.md).

---

## PNL Masking

PNL masking is already Python upstream. Our adapter prepares the native DWI/b0
input, invokes the selected external runtime, records resource hashes, restores
and checks mask geometry, and reports failures. The CNN architecture, learned
weights and ANTs registration remain those of the configured author backend.
It does **not** execute the entire PNL preprocessing pipeline or replace PNL's
CNN with a locally reimplemented model.

The PNL route is used when a mask needs generating; an existing supplied mask is
retained. Alternative backends are explicit Advanced settings, not silent
fallbacks. See [setup](../setup/brain_mask_gui_config.md).

---

## Reporting Rule

Record implementation identity, external versions, parameters, input hashes and
QC scope for each run. Report automatic geometry/QC separately from expert
anatomical review. The [MK-Curve reference checks](../release/MKCURVE_PYTHON_20261012.md)
and [SAF checks](../release/SAF_PYTHON_20261013.md) use original `.m` kernels
executed by GNU Octave for their stated scopes, not MATLAB itself and not a
blanket validation of every optional branch or subject.
