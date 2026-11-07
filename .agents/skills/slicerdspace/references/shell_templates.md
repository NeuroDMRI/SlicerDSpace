# Algorithm Shell Templates

Generated from `docs/setup/CLI_TEMPLATES.md`; do not edit this catalogue or its generated scripts.
Regenerate with `python scripts/build_skill_shell_templates.py` from the authorized checkout.

Source SHA-256 (UTF-8, LF): `a2ed39878940d95110f203dc9dddd55fc310704d47b9ae3edcd03525207c67f3`.

## Contents

- [Environment and modes](#environment-and-modes)
- [Brain masks](#brain-masks)
- [Preprocessing](#preprocessing)
- [Anatomical prior](#anatomical-prior)
- [Microstructure](#microstructure)
- [Tractography](#tractography)
- [Fiber bundles](#fiber-bundles)
- [Connectome and combined routes](#connectome-and-combined-routes)
- [Import, cohorts, regional analysis and reports](#other-workflows)

## Environment And Modes

Use Bash on Linux/macOS or an explicitly configured WSL environment. Check backend/platform
support first; a template does not install or make every backend cross-platform. The Skill and
scripts live in the source checkout; `REPO` may instead point to the installed runtime.

Replace the paths and use the confirmed resource root. These exports support the standalone
scripts; the `pipeline` function also makes the inline snippets below usable in this shell.

```bash
export SLICER="/path/to/Slicer"
export REPO="/path/to/SlicerDSpace"
export SUBJECT="/path/to/prepared/sub-01"
export SUBJECT_ID="sub-01"
export OUTPUTS="/path/to/analysis/sub-01"
export ASSETS="/path/to/SlicerDSpace-models"
DEPENDENCY_ARGS=()
# If needed, export dependency paths from the GUI, then uncomment:
# DEPENDENCY_ARGS=(--dependency-config "/path/to/dependency_paths.json")

pipeline() {
  local run_name="$1"
  shift
  "$SLICER" --launch PythonSlicer "$REPO/shared/run_subject_pipeline_cli.py" \
    --no-user-config --strict-environment \
    --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" \
    --output-root "$OUTPUTS/$run_name" --run-name "$run_name" \
    --model-assets-dir "$ASSETS" "${DEPENDENCY_ARGS[@]}" "$@"
}
```

For standalone scripts, export `DEPENDENCY_CONFIG=/absolute/path/to/dependencies.json`
when reusing GUI-configured external paths. Inline snippets use `DEPENDENCY_ARGS` above.

```bash
export SKILL="/path/to/checkout/.agents/skills/slicerdspace"
bash "$SKILL/assets/shell/ddparcel.sh" doctor
bash "$SKILL/assets/shell/ddparcel.sh" plan
# After reviewing the same inputs, parameters, dependencies and plan:
bash "$SKILL/assets/shell/ddparcel.sh" execute
```

Omitting the mode means `plan`. `doctor` and `plan` may write metadata but do not fit models.
Only an explicit `execute` starts analysis. Unknown modes, extra positional arguments and
unresolved `/path/to/` placeholders in execute mode are rejected before launching Slicer.
Do not change methods between these modes. Edit a separate script copy for custom settings;
the maintained CLI guide documents additional flags and PowerShell syntax.

Most templates use an existing reviewed mask (`provided`); choose a mask-generation recipe
explicitly if needed. Optional FW/NODDI/DKI fits require acquisition eligibility checks.
Multi-model and multi-subject concurrency are separate resource budgets. Preserve invalid fits
and sparse-bundle warnings. UKF is called automatically, and bundle sources remain explicit.

## Brain masks

### PNL CNN brain mask

[Standalone Bash](../assets/shell/mask_pnl.sh) - Recipe `mask-pnl`.

```bash
pipeline mask_pnl --targets validation --brain-mask-backend pnl_cnn \
  --brain-mask-device auto --plan
```

## Preprocessing

### DIPY preprocessing with MP-PCA

[Standalone Bash](../assets/shell/preproc_dipy.sh) - Recipe `preproc-dipy`.

```bash
pipeline preproc_dipy --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method mppca --plan
```

### DIPY preprocessing with Local PCA

[Standalone Bash](../assets/shell/preproc_localpca.sh) - Recipe `preproc-localpca`.

```bash
pipeline preproc_localpca --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method localpca --plan
```

### DIPY preprocessing with Patch2Self

[Standalone Bash](../assets/shell/preproc_patch2self.sh) - Recipe `preproc-patch2self`.

```bash
pipeline preproc_patch2self --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method patch2self --plan
```

### MRtrix/FSL preprocessing

[Standalone Bash](../assets/shell/preproc_fsl.sh) - Recipe `preproc-fsl`.

```bash
pipeline preproc_fsl --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend mrtrix_fsl \
  --preprocessing-phase-encoding-direction j- --preprocessing-readout-time 0.05 \
  --preprocessing-reverse-phase-b0 "/path/to/matched_reverse_b0.nii.gz" --plan
```

## Anatomical prior

### DDSeg tissue segmentation

[Standalone Bash](../assets/shell/ddseg.sh) - Recipe `anatomy-seg`.

```bash
pipeline ddseg --targets validation,segmentation --brain-mask-backend provided --plan
```

### DDParcel brain parcellation

[Standalone Bash](../assets/shell/ddparcel.sh) - Recipe `anatomy-parcel`.

```bash
pipeline ddparcel --targets validation,parcellation --brain-mask-backend provided --plan
```

### DDSurfer native surfaces

[Standalone Bash](../assets/shell/ddsurfer.sh) - Recipe `anatomy-surface`.

```bash
pipeline ddsurfer --targets validation,surface --brain-mask-backend provided \
  --surface-device auto --surface-precision fp32 --plan
```

### Combined anatomical priors

[Standalone Bash](../assets/shell/anatomy.sh) - Recipe `anatomy-combined`.

```bash
pipeline anatomy --targets validation,segmentation,parcellation,surface \
  --brain-mask-backend provided --ddseg-device auto --surface-device auto --plan
```

### Native cortical postprocessing and atlas mapping

[Standalone Bash](../assets/shell/anatomy_post.sh) - Recipe `anatomy-postprocess`.

```bash
pipeline anatomy_post --targets validation,segmentation,parcellation,surface,surface_postprocess \
  --brain-mask-backend provided --postprocess-mode whole \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

## Microstructure

### DTI with WLS

[Standalone Bash](../assets/shell/dti.sh) - Recipe `micro-dti`.

```bash
pipeline dti --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method wls --plan
```

### DTI with OLS

[Standalone Bash](../assets/shell/dti_ols.sh) - Recipe `micro-dti-ols`.

```bash
pipeline dti_ols --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method ls --plan
```

### Free-water imaging

[Standalone Bash](../assets/shell/freewater.sh) - Recipe `micro-fw`.

```bash
pipeline freewater --targets validation,microstructure \
  --microstructure-stages dti,freewater --brain-mask-backend provided --plan
```

### NODDI with AMICO

[Standalone Bash](../assets/shell/noddi.sh) - Recipe `micro-noddi`.

```bash
pipeline noddi --targets validation,microstructure \
  --microstructure-stages dti,noddi --brain-mask-backend provided --plan
```

### DKI with WLS

[Standalone Bash](../assets/shell/dki_wls.sh) - Recipe `micro-dki`.

```bash
pipeline dki_wls --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method WLS --brain-mask-backend provided --plan
```

### DKI with OLS

[Standalone Bash](../assets/shell/dki_ols.sh) - Recipe `micro-dki-ols`.

```bash
pipeline dki_ols --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method OLS --brain-mask-backend provided --plan
```

### MK-Curve Python port

[Standalone Bash](../assets/shell/mkcurve.sh) - Recipe `micro-mkcurve`.

```bash
pipeline mkcurve --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE --brain-mask-backend provided --plan
```

### MK-Curve optional MATLAB reference

[Standalone Bash](../assets/shell/mkcurve_matlab.sh) - Recipe `micro-mkcurve-matlab`.

```bash
pipeline mkcurve_matlab --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE_MATLAB --brain-mask-backend provided \
  --dki-matlab-executable "/path/to/matlab" --dki-mkcurve-root "/path/to/MK-Curve" --plan
```

### Independent microstructure models in parallel

[Standalone Bash](../assets/shell/micro_multi.sh) - Recipe `micro-parallel`.

```bash
pipeline micro_multi --targets validation,microstructure \
  --microstructure-stages dti,freewater,noddi,dki --dki-fit-method WLS \
  --max-parallel-models 2 --brain-mask-backend provided --plan
```

## Tractography

### DTI deterministic tractography

[Standalone Bash](../assets/shell/tracks_dti.sh) - Recipe `tract-dti`.

```bash
pipeline tracks_dti --targets validation,tractography --tractography-backend dti_det \
  --brain-mask-backend provided --select-streamlines 1000000 \
  --dti-min-length-mm 10 --dti-max-length-mm 250 --plan
```

### UKF tractography

[Standalone Bash](../assets/shell/tracks_ukf.sh) - Recipe `tract-ukf`.

```bash
pipeline tracks_ukf --targets validation,tractography --tractography-backend ukf \
  --brain-mask-backend provided --ukf-parameters '{"numThreads":8}' --plan
```

### CSD/ACT tractography and SIFT2

[Standalone Bash](../assets/shell/tracks_csd.sh) - Recipe `tract-csd`.

```bash
pipeline tracks_csd --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act --brain-mask-backend provided \
  --select-streamlines 1000000 --tckgen-seeding dynamic --mrtrix-nthreads 8 --plan
```

### Short association fibres

[Standalone Bash](../assets/shell/tracks_saf.sh) - Recipe `tract-saf`.

```bash
pipeline tracks_saf --targets validation,segmentation,parcellation,surface,surface_postprocess,tractography \
  --tractography-backend saf --brain-mask-backend provided \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

### Separate CSD/ACT and UKF runs

[Standalone Bash](../assets/shell/tracks_compare.sh) - Recipe `tract-multiple`.

```bash
pipeline tracks_compare --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act,ukf --brain-mask-backend provided --plan
```

## Fiber bundles

### WMQL official multi-bundle queries

[Standalone Bash](../assets/shell/bundles_wmql.sh) - Recipe `bundle-wmql`.

```bash
pipeline bundles_wmql --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-preset official --brain-mask-backend provided --plan
```

### WMQL selected queries

[Standalone Bash](../assets/shell/bundles_wmql_subset.sh) - Recipe `bundle-wmql-subset`.

```bash
pipeline bundles_wmql_subset --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-selection AF,CST,UF,cc_4 --brain-mask-backend provided --plan
```

### WMA with automatically generated UKF tracks

[Standalone Bash](../assets/shell/bundles_wma.sh) - Recipe `bundle-wma`.

```bash
pipeline bundles_wma --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wma --bundle-tractography-backend ukf \
  --wma-atlas org \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

### TractCloud with automatically generated UKF tracks

[Standalone Bash](../assets/shell/bundles_tractcloud.sh) - Recipe `bundle-tractcloud`.

```bash
pipeline bundles_tractcloud --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend tractcloud --bundle-tractography-backend ukf \
  --tractcloud-device auto --tractcloud-batch-size 2048 --brain-mask-backend provided --plan
```

### WMQL, WMA and TractCloud with an explicit source

[Standalone Bash](../assets/shell/bundles_compare.sh) - Recipe `bundle-multiple`.

```bash
pipeline bundles_compare --targets validation,parcellation,tractography,bundles \
  --tractography-backend csd_act,ukf --bundle-backend wmql,wma,tractcloud \
  --bundle-tractography-backend ukf --slicer-executable "$SLICER" \
  --brain-mask-backend provided --plan
```

### WMA with existing reviewed tracks

[Standalone Bash](../assets/shell/wma_existing.sh) - Recipe `bundle-existing`.

```bash
pipeline wma_existing --targets validation,bundles --bundle-backend wma \
  --wma-input-tractography "/path/to/reviewed_whole_brain.vtp" \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

## Connectome and combined routes

### Count/scalar-weighted Python connectome

[Standalone Bash](../assets/shell/connectome_python.sh) - Recipe `connectome-python`.

```bash
pipeline connectome_python --targets validation,microstructure,parcellation,tractography,connectome \
  --tractography-backend dti_det --connectome-backend python \
  --brain-mask-backend provided --plan
```

### MRtrix connectome with matching SIFT2 weights

[Standalone Bash](../assets/shell/connectome_sift2.sh) - Recipe `connectome-sift2`.

```bash
pipeline connectome_sift2 --targets validation,segmentation,parcellation,tractography,connectome \
  --tractography-backend csd_act --connectome-backend mrtrix \
  --brain-mask-backend provided --select-streamlines 1000000 --plan
```

### Connectome with the automatic backend choice

[Standalone Bash](../assets/shell/connectome_auto.sh) - Recipe `connectome-auto`.

```bash
pipeline connectome_auto --targets validation,segmentation,parcellation,tractography,connectome \
  --tractography-backend csd_act --connectome-backend auto \
  --brain-mask-backend provided --select-streamlines 1000000 --plan
```

### Combined anatomical priors, DTI, UKF, WMQL and connectome

[Standalone Bash](../assets/shell/combined.sh) - Recipe `combined`.

```bash
pipeline combined --targets validation,microstructure,segmentation,parcellation,surface,tractography,bundles,connectome \
  --microstructure-stages dti --tractography-backend ukf --bundle-backend wmql \
  --bundle-tractography-backend ukf --connectome-backend python \
  --brain-mask-backend provided --plan
```

<a id="other-workflows"></a>
## Import, Cohorts, Regional Analysis And Reports

Use the maintained [CLI guide](../../../../docs/setup/CLI_TEMPLATES.md) or
[Chinese guide](../../../../docs/setup/CLI_TEMPLATES.zh-CN.md) for these non-model workflows:

| Workflow | Recipe / reference |
| --- | --- |
| DICOM, NIfTI, NRRD/NHDR import | `import-dicom`, `import-nifti`, `import-nrrd`; preview with `--dry-run` |
| Multi-subject scheduling | `cohort`; review RAM/GPU demand before increasing `--jobs` |
| Named algorithm/parameter comparisons | `experiments`; validate the request before `--execute` |
| Region/tract/connectome queries | `region-run` takes direct region/metric arguments; `region-inspect` is optional |
| Regional cohort summaries | `scripts/analyze_regions.py --source-manifest RUN1 RUN2 ...`; optional saved JSON via `--batch` |
| Result report | `report`; select the exact run manifest |

Regional postprocessing and report commands can write outputs without fitting models; they
are not plan-only operations. Follow their documented preview/review steps and authorization
instead of appending unsupported `--plan` or `--doctor` flags.

The optional `MKCURVE_MATLAB` script is an author-reference route and requires separately
configured MATLAB and source paths; the Python `MKCURVE` route does not. Native surface
postprocessing needs its verified component runtime and the applicable personal licence.
