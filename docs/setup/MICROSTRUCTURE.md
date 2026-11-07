<div align="center">

# Microstructural Analysis

[Project](../../README.md) | [Quick Start](QUICKSTART.md) | [Results and QC](GUI_RESULT_REVIEW.md) | [中文](MICROSTRUCTURE.zh-CN.md)

</div>

---

## Select A Route

Select Microstructure in Products. **DTI is the only default model**. Click
**...** beside Microstructure to enable FW, NODDI, DKI and iFW independently;
choosing DKI does not imply enabling FW or NODDI. iFW is the exception: it
needs the FW result, so the plugin runs FW and the DDSeg segmentation for it. DTI remains required preparation.
Choose the DKI fitting method in **Advanced settings > Model parameters**.

| Model | Outputs | Implementation |
| --- | --- | --- |
| DTI | FA, MD, AD, RD, L1/L2/L3, Trace, colour FA map (`<id>-dti-DEC.nrrd`) | Shared native-space tensor fit with the Teem library in Slicer's Python; no SlicerDMRI needed |
| Free-water | FWF, tissue FA/MD, validity mask and QC | DIPY free-water tensor model |
| NODDI | NDI, ODI, IsoVF, RMSE/NRMSE, validity mask and QC | AMICO NODDI |
| iFW | Free-water fraction corrected for T2 weighting, minus the CSF fraction, with regional table | Python implementation of the published equations (fixed T2 of 95 ms for tissue and 1500 ms for CSF); needs the echo time (typed in Microstructure, or read from the DWI sidecar) and at least three distinct b-values, b=0 counted |
| Standard DKI | MK, AK, RK, KFA, MKT, RTK, residuals and raw tensor estimates | Python MK-Curve port (default DKI method; stops with a WLS re-run hint when the signal scale or data cannot support it), or DIPY WLS or OLS |
| MK-Curve | Original and corrected MK/AK/RK, FA/MD/AD/RD, eigenvalues, validity and residuals | Bundled Python port of the author algorithm, not a DIPY approximation |

AD/RD and tensor eigenvalues are already available; DKI adds the complementary
MKT/RTK/KFA measures described in the [DIPY DKI tutorial](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html).
Other models such as IVIM or specialized restricted-diffusion models require
additional acquisition assumptions and are not silently added to a generic run.

---

## DKI Eligibility

Before fitting, the pipeline checks a 4D volume and matching finite gradients,
b0 measurements, at least two nonzero shells, at least 15 independent gradient
directions and angular rank 15 per shell, and total scaled design rank 22.
Mask shape, affine and binary contents are checked before fitting. Nominal b0
direction entries are cleared only in the working gradient table and recorded.
Input files are not edited. A failed eligibility check blocks DKI, not a valid
DTI route.

Shell tolerance is 100 s/mm², b0 threshold 50 s/mm². The default upper shell
median is 2500 s/mm², retaining complete shells rather than dropping isolated
measurements at a nominal-shell boundary. The selected upper b-value
and any excluded measurements are recorded in the DKI manifest; high-b data
can bias the truncated cumulant expansion. These gates check identifiability,
not adequate SNR or independent biological correctness. Inspect preprocessing,
signal residuals and anatomical priors before interpreting results.

The fit does not add smoothing, resampling, denoising or spatial hole filling.
Quantitative `*_raw.nii.gz` maps preserve unbounded estimates. Separate display
maps bound kurtosis to 0–3 and KFA to 0–1; these are **not corrected estimates**.
Unfitted in-mask voxels remain NaN, and explicit fit/validity masks accompany
the maps. Invalid-voxel counts and residuals are reported, not hidden.

---

## Python MK-Curve

Select `MKCURVE` under Advanced. The bundled Python port uses NumPy, SciPy
and nibabel; no MATLAB or separate author checkout is required. It follows
the [author source](https://github.com/zhangfanmark/MK-Curve), pinned at
`9a213e1a5850fe50c998793f31ed852df00242f8`:
OLS initialization, predicted-signal WLLS weights, the original 256-direction
numerical MK integration, synthetic-b0 scanning, curve-based abnormal-voxel
detection and b0 correction at threshold 0.5. It is not an official upstream
Python release. Upstream licenses and attribution ship with the extension.

Multiple selected b0 measurements are averaged in a working copy. The author's
positive-neighbour repair of nonpositive/nonfinite signals is explicitly counted;
original DWI, gradients and mask are never edited. Raw original and corrected
maps are untruncated. Separate validity masks govern corrected regional statistics.
Curve arrays, their b0 grid, coefficients, input/source hashes and QC are saved.
Invalid values remain visible in QC; correction is not a guarantee of biological
correctness. KFA/MKT/RTK belong to the separate DIPY reference, not corrected maps.

The grid uses the author's adaptive integer signal scale, independent of output
directory names; it refuses unusable normalized signal scales rather than silently
rescaling. The radial integration has an explicit pole fix, and nonfinite curves
are marked invalid. Voxel fitting is chunked and selected DWI volumes are staged
as disk-backed float64 arrays; full-brain runtime/memory still need separate validation.
See [port validation](../release/MKCURVE_PYTHON_20261012.md) for measured scope.

`MKCURVE_MATLAB` retains the original external implementation for optional
reference execution. Only this choice requires Advanced MATLAB executable and
author checkout paths; missing dependencies block it without falling back.

---

## CLI And Outputs

```bash
PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets microstructure --microstructure-stages dti,dki \
  --dki-fit-method WLS --plan
```

Use `--execute` instead of `--plan` for actual processing in the required
runtime. Optional `--run-free-water`, `--run-noddi` and `--run-dki` flags are
also available. `all` explicitly includes all four models; it is not the default.
For Python MK-Curve use `--dki-fit-method MKCURVE`.
Only the optional reference uses `--dki-fit-method MKCURVE_MATLAB`
with `--dki-matlab-executable /path/to/matlab --dki-mkcurve-root /path/to/MK-Curve`.

DKI files live under `02_microstructure/dki/`, with `dki_manifest.json`,
`dki_qc.json` and `dki_progress.json`. Reuse requires matching inputs/settings,
library versions and artifact hashes. Successful GUI tasks automatically load
representative maps; execution completion still leaves human review pending.
