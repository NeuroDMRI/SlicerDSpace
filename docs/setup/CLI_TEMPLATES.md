<div align="center">

# Command-Line Templates

**Choose a route, inspect its plan, then execute**

[Home](../../README.md) | [Quick start](QUICKSTART.md) | [Experiments](EXPERIMENTS.md) | [Platforms](SUPPORT_MATRIX.md) | [简体中文](CLI_TEMPLATES.zh-CN.md)

</div>

---

## Start Here

| Task | Recipes |
| --- | --- |
| Import DICOM, NIfTI or NRRD | [Input preparation](#input-preparation) |
| Generate masks or correct DWI | [Masks and preprocessing](#masks-and-preprocessing) |
| DDSeg, DDParcel, DDSurfer or cortical postprocessing | [Anatomical prior](#anatomical-prior) |
| DTI, FW, NODDI, DKI or MK-Curve | [Microstructure](#microstructure) |
| DTI, UKF, CSD/ACT or SAF | [Tractography](#tractography) |
| WMQL, WMA, TractCloud and source selection | [Fiber bundles](#fiber-bundles-and-explicit-sources) |
| Connectivity or a combined route | [Connectome](#connectome-and-combined-routes) |
| Cohorts or parameter comparisons | [Experiments](#cohorts-and-parameter-experiments) |
| Regional analysis or reports | [Results](#regional-analysis-and-reports) |

Use Bash on Linux/macOS or inside an explicitly configured WSL environment.
Native PowerShell is covered below. Replace the paths before running anything.
`REPO` is either a checkout or the installed `qt-scripted-modules` directory
containing `shared/` and `scripts/`. Use the real Slicer launcher.

The [Agent Skill's algorithm catalogue](../../.agents/skills/slicerdspace/references/shell_templates.md)
provides standalone Bash templates generated from the subject-level recipes below.
Their default is plan-only; an explicit `execute` mode starts analysis. The Skill
files belong to the authorized source checkout, not the installed extension runtime.

<!-- cli-setup: bash -->
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

Each recipe uses a separate output directory. For the **same recipe and run
name**, first replace `--plan` with `--doctor`, then inspect `--plan`, and
finally replace it with `--execute` when ready. Doctor/plan can write metadata
but do not run the scientific models. `--strict-environment` stops missing
dependencies instead of silently running only part of a requested route.

Saved GUI run defaults are excluded. To reuse GUI-configured external paths,
use **Environment and installation > More options > Export dependency config**
and set `DEPENDENCY_ARGS`. You can instead pass the explicit path flags shown
by `--help`. Prepare only selected dependencies/models using
[managed setup](MANAGED_SETUP.md); model availability and license acceptance
are separate from analysis execution.

Most recipes assume an existing reviewed native mask (`provided`). Select the
DIPY or PNL alternatives explicitly when generation is needed. Supplied masks
take priority. Mask generation and eligible caches live under the subject root;
separate output folders alone do not isolate mask-generation comparisons. Use
independent input copies for those comparisons.

---

## Input Preparation

Isolate one diffusion DICOM series. NIfTI needs matching bval/bvec sidecars;
NRRD/NHDR must retain diffusion gradient metadata. The destination must be a
fresh prepared subject folder. Preview first; remove `--dry-run` for an approved
import. Existing canonical files are protected unless replacement is requested.

<!-- cli-template: import-dicom -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/diffusion_DICOM" --source-kind dicom \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

<!-- cli-template: import-nifti -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/dwi.nii.gz" --source-kind nifti \
  --bval "/path/to/dwi.bval" --bvec "/path/to/dwi.bvec" \
  --mask "/path/to/reviewed_mask.nii.gz" \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

<!-- cli-template: import-nrrd -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" \
  --source "/path/to/dwi.nhdr" --source-kind nrrd --backend dwiconvert \
  --subject-root "$SUBJECT" --subject-id "$SUBJECT_ID" --dry-run
```

The canonical bundle is `input/data.nii.gz`, `input/bvals`, `input/bvecs`,
optional `input/dwi.json` and `input/nodif_brain_mask.nii.gz`. Review volume
counts, gradient orientation, geometry and finite values. NRRD uint16 conversion
is lossless only within the checked int16 range; the importer refuses overflow.

---

## Masks And Preprocessing

Choose one mask method; these commands generate a mask only if needed. PNL
needs its configured TensorFlow/ANTs runtime; DIPY needs no weights or external
mask program. MRtrix and SynthStrip retain their own requirements. None is an
error-triggered fallback for another method.

<!-- cli-template: mask-pnl -->
```bash
pipeline mask_pnl --targets validation --brain-mask-backend pnl_cnn \
  --brain-mask-device auto --plan
```

<!-- cli-template: mask-dipy -->
```bash
pipeline mask_dipy --targets validation --brain-mask-backend dipy --plan
```

<!-- cli-template: mask-mrtrix -->
```bash
pipeline mask_mrtrix --targets validation --brain-mask-backend mrtrix --plan
```

<!-- cli-template: mask-synthstrip -->
```bash
pipeline mask_synthstrip --targets validation --brain-mask-backend synthstrip \
  --postprocess-freesurfer-home "/path/to/freesurfer" --brain-mask-device auto --plan
```

Optional DIPY denoising/Gibbs/motion correction:

<!-- cli-template: preproc-dipy -->
```bash
pipeline preproc_dipy --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method mppca --plan
```

Alternative denoisers keep the remaining preprocessing route unchanged:

<!-- cli-template: preproc-localpca -->
```bash
pipeline preproc_localpca --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method localpca --plan
```

<!-- cli-template: preproc-patch2self -->
```bash
pipeline preproc_patch2self --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend dipy \
  --preprocessing-dipy-denoise-method patch2self --plan
```

MRtrix/FSL with acquisition-specific phase encoding and total readout time.
Replace `j-`, `0.05` and the reverse-PE image with the actual protocol; they are
examples, not universal defaults:

<!-- cli-template: preproc-fsl -->
```bash
pipeline preproc_fsl --targets validation,preprocessing \
  --brain-mask-backend provided --preprocessing-backend mrtrix_fsl \
  --preprocessing-phase-encoding-direction j- --preprocessing-readout-time 0.05 \
  --preprocessing-reverse-phase-b0 "/path/to/matched_reverse_b0.nii.gz" --plan
```

To feed corrected data into reconstruction, include the downstream targets
in the **same command**, e.g. `validation,preprocessing,segmentation,parcellation,surface`.
Do not assume a later independent run automatically adopts an earlier run's
preprocessed images. Review the analysis-input paths recorded in the plan/run.

---

## Anatomical prior

Model assets and compatible PyTorch are required. Shared native DTI preparation
is handled by the workflow. Choose only the products needed by the study.

<!-- cli-template: anatomy-seg -->
```bash
pipeline ddseg --targets validation,segmentation --brain-mask-backend provided --plan
```

<!-- cli-template: anatomy-parcel -->
```bash
pipeline ddparcel --targets validation,parcellation --brain-mask-backend provided --plan
```

<!-- cli-template: anatomy-surface -->
```bash
pipeline ddsurfer --targets validation,surface --brain-mask-backend provided \
  --surface-device auto --surface-precision fp32 --plan
```

<!-- cli-template: anatomy-combined -->
```bash
pipeline anatomy --targets validation,segmentation,parcellation,surface \
  --brain-mask-backend provided --ddseg-device auto --surface-device auto --plan
```

Native surfaces plus cortical postprocessing, thickness/curvature and atlas
mapping. Configure the licensed component runtime and personal activation file;
a full FreeSurfer installation is optional for this route:

<!-- cli-template: anatomy-postprocess -->
```bash
pipeline anatomy_post --targets validation,segmentation,parcellation,surface,surface_postprocess \
  --brain-mask-backend provided --postprocess-mode whole \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

For previously completed native surfaces, use the exact recorded inputs with
`--postprocess-native-dir` and, where required, `--postprocess-mri-dir`.
See [surface components](SURFACE_COMPONENTS.md); do not guess template-space
surfaces as native inputs.

---

## Microstructure

DTI is the default. Optional FW/NODDI/DKI require their own sampling/fit checks.
NODDI and DKI require eligible multi-shell acquisitions. Inspect the eligibility
record, validity masks and excluded fits; do not replace them with display maps.

<!-- cli-template: micro-dti -->
```bash
pipeline dti --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method wls --plan
```

An explicit ordinary least-squares comparison:

<!-- cli-template: micro-dti-ols -->
```bash
pipeline dti_ols --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend provided --micro-fit-method ls --plan
```

Explicit lightweight masking plus DTI:

<!-- cli-template: micro-dti-light -->
```bash
pipeline dti_light --targets validation,microstructure --microstructure-stages dti \
  --brain-mask-backend dipy --plan
```

<!-- cli-template: micro-fw -->
```bash
pipeline freewater --targets validation,microstructure \
  --microstructure-stages dti,freewater --brain-mask-backend provided --plan
```

<!-- cli-template: micro-noddi -->
```bash
pipeline noddi --targets validation,microstructure \
  --microstructure-stages dti,noddi --brain-mask-backend provided --plan
```

<!-- cli-template: micro-dki -->
```bash
pipeline dki_wls --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method WLS --brain-mask-backend provided --plan
```

<!-- cli-template: micro-dki-ols -->
```bash
pipeline dki_ols --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method OLS --brain-mask-backend provided --plan
```

<!-- cli-template: micro-mkcurve -->
```bash
pipeline mkcurve --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE --brain-mask-backend provided --plan
```

`MKCURVE` is the Python port and does not need MATLAB. `MKCURVE_MATLAB` is
an optional author-reference route with separately configured paths.

<!-- cli-template: micro-mkcurve-matlab -->
```bash
pipeline mkcurve_matlab --targets validation,microstructure --microstructure-stages dti,dki \
  --dki-fit-method MKCURVE_MATLAB --brain-mask-backend provided \
  --dki-matlab-executable "/path/to/matlab" --dki-mkcurve-root "/path/to/MK-Curve" --plan
```

<!-- cli-template: micro-parallel -->
```bash
pipeline micro_multi --targets validation,microstructure \
  --microstructure-stages dti,freewater,noddi,dki --dki-fit-method WLS \
  --max-parallel-models 2 --brain-mask-backend provided --plan
```

The last command explicitly requests all four models. Only the independent
eligible optional fits can overlap after prerequisites; it is not the default.

---

## Tractography

DTI deterministic tracking uses SlicerDMRI/native DTI preparation and DIPY.
`--select-streamlines` is not an equivalent UKF budget.

<!-- cli-template: tract-dti -->
```bash
pipeline tracks_dti --targets validation,tractography --tractography-backend dti_det \
  --brain-mask-backend provided --select-streamlines 1000000 \
  --dti-min-length-mm 10 --dti-max-length-mm 250 --plan
```

UKF uses the installed UKFTractography extension/CLI and SlicerDMRI. Its own
author defaults apply; this example changes only its thread budget:

<!-- cli-template: tract-ukf -->
```bash
pipeline tracks_ukf --targets validation,tractography --tractography-backend ukf \
  --brain-mask-backend provided --ukf-parameters '{"numThreads":8}' --plan
```

CSD/ACT requires MRtrix, DDSeg tissue probabilities and DDParcel anatomical labels;
it also computes matching SIFT2 weights. Select both anatomical products below or
provide their existing outputs. The standard route blocks when DDParcel is missing.

<!-- cli-template: tract-csd -->
```bash
pipeline tracks_csd --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act --brain-mask-backend provided \
  --select-streamlines 1000000 --tckgen-seeding dynamic --mrtrix-nthreads 8 --plan
```

`--allow-act-ddseg-only` explicitly permits an exploratory fallback without
DDParcel. Its ACT QC records the missing subcortical GM, ventricular and selected
WM priors. These outputs are not reused by the standard route. The fifth 5TT
channel is zero in both routes: lesion-specific tissue priors are not estimated.

SAF uses CSD inputs and accepted native surfaces/postprocessing:

<!-- cli-template: tract-saf -->
```bash
pipeline tracks_saf --targets validation,segmentation,parcellation,surface,surface_postprocess,tractography \
  --tractography-backend saf --brain-mask-backend provided \
  --postprocess-runtime "/path/to/verified/surface-components" \
  --postprocess-license-file "/path/to/personal/license.txt" --plan
```

<!-- cli-template: tract-multiple -->
```bash
pipeline tracks_compare --targets validation,segmentation,parcellation,tractography \
  --tractography-backend csd_act,ukf --brain-mask-backend provided --plan
```

Multiple tracking backends run sequentially with separate `<run>__<backend>`
records. They do not merge streamlines. For deterministic MRtrix comparisons,
`--mrtrix-seed 42` forces `tckgen -nthreads 0`; this can be slower. The configured
thread budget still applies to SIFT2. A seed does not imply determinism for
other backends.

---

## Fiber bundles And Explicit Sources

UKF is generated automatically for the examples below. Whole-brain input,
bundle method and atlas/resource requirements are distinct selections.

<!-- cli-template: bundle-wmql -->
```bash
pipeline bundles_wmql --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-preset official --brain-mask-backend provided --plan
```

The official WMQL rules run all output queries by default. For the optional
regional cerebellar selections, replace `--wmql-query-preset official` with
`--wmql-query-preset official_cerebellar` (48 original queries plus three regional
selections), or `cerebellar` (regional selections only). These select left/right
cerebellar--brainstem and bilateral cerebellar candidates, without assigning
individual cerebellar-peduncle labels. See [definitions and checks](WMQL_CEREBELLAR.md).
To choose families:

<!-- cli-template: bundle-wmql-subset -->
```bash
pipeline bundles_wmql_subset --targets validation,parcellation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wmql --bundle-tractography-backend ukf \
  --wmql-query-selection AF,CST,UF,cc_4 --brain-mask-backend provided --plan
```

<!-- cli-template: bundle-wma -->
```bash
pipeline bundles_wma --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend wma --bundle-tractography-backend ukf \
  --wma-atlas org \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

WMA defaults to ORG. Use `--wma-atlas east-west` or `--wma-atlas naba` for the Eastern/Western adult or neonatal/adult atlas. In direct CLI calls, pass `--wma-atlas org|east-west|naba`; `--wma-atlas-dir` optionally selects an existing atlas. Prepare only the selected resource profile (`wma`, `wma-east-west` or `wma-naba`) under the confirmed model root. See [WMA atlas guide](WMA_ATLASES.md).

<!-- cli-template: bundle-tractcloud -->
```bash
pipeline bundles_tractcloud --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-backend tractcloud --bundle-tractography-backend ukf \
  --tractcloud-device auto --tractcloud-batch-size 2048 --brain-mask-backend provided --plan
```

<!-- cli-template: bundle-multiple -->
```bash
pipeline bundles_compare --targets validation,parcellation,tractography,bundles \
  --tractography-backend csd_act,ukf --bundle-backend wmql,wma,tractcloud \
  --bundle-tractography-backend ukf --slicer-executable "$SLICER" \
  --brain-mask-backend provided --plan
```

All selected bundle methods in the last recipe use UKF; the separate CSD route
does not receive bundle jobs. To compare a different source, change
`--bundle-tractography-backend` and use a new run name. WMA coverage on a
non-UKF source needs its own review. Sparse/empty fiber bundles stay visible in QC.
SAF is not a whole-brain bundle source.

Advanced reuse of an existing compatible whole-brain VTK/VTP input:

<!-- cli-template: bundle-existing -->
```bash
pipeline wma_existing --targets validation,bundles --bundle-backend wma \
  --wma-input-tractography "/path/to/reviewed_whole_brain.vtp" \
  --slicer-executable "$SLICER" --brain-mask-backend provided --plan
```

Do not combine explicit WMA/TractCloud input files with
`--bundle-tractography-backend`. Preserve the full quantitative tractogram,
not a capped scene preview, and verify coordinate/protocol compatibility.

---

## Connectome And Combined Routes

Count/scalar-weighted Python connectivity with DTI tracks:

<!-- cli-template: connectome-python -->
```bash
pipeline connectome_python --targets validation,microstructure,parcellation,tractography,connectome \
  --tractography-backend dti_det --connectome-backend python \
  --brain-mask-backend provided --plan
```

CSD/ACT connectivity with matching SIFT2 weights and MRtrix:

<!-- cli-template: connectome-sift2 -->
```bash
pipeline connectome_sift2 --targets validation,segmentation,parcellation,tractography,connectome \
  --tractography-backend csd_act --connectome-backend mrtrix \
  --brain-mask-backend provided --select-streamlines 1000000 --plan
```

A combined anatomical priors/microstructure/UKF/WMQL/Python-connectome route:

<!-- cli-template: combined -->
```bash
pipeline combined --targets validation,microstructure,segmentation,parcellation,surface,tractography,bundles,connectome \
  --microstructure-stages dti --tractography-backend ukf --bundle-backend wmql \
  --bundle-tractography-backend ukf --connectome-backend python \
  --brain-mask-backend provided --plan
```

To use optional preprocessing, add `preprocessing` to that command's targets
and choose its backend/acquisition settings. Postprocessing and optional
microstructure models remain explicit choices, not implied by this recipe.

---

## Cohorts And Parameter Experiments

One configuration per prepared subject; start with one or two concurrent
subjects according to combined RAM/GPU requirements. This writes plans by
default; add `--execute` to run the cohort. Without `--continue-on-error`, the
batch runner stops new submissions after a failure and drains active jobs.
The batch manifest includes each subject's actual run status and task counts.
Blocked child tasks produce `completed_with_blocks` and a nonzero batch exit
code by default, even when the single-subject CLI allows partial execution.
Use `--allow-partial-success` only when that partial result is acceptable; it
allows exit code 0 but retains the blocked state in the manifest.

<!-- cli-template: cohort -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root "/path/to/prepared_study" --output-root "/path/to/cohort_results" \
  --run-name cohort_dti --jobs 2 --no-user-config --strict-environment \
  --targets validation,microstructure --microstructure-stages dti --brain-mask-backend provided
```

For shared compute-stage admission across active subjects, select
`--scheduler resources`. Subject concurrency and stage reservations are planned
automatically; `--jobs 3` optionally sets the active-subject limit. Advanced resource
overrides are documented in [cohort resource scheduling](COHORT_RESOURCES.md).
That mode uses one within-subject model at a time and keeps the existing subject
manifests. It does not keep models resident or combine subjects into inference batches.
`--scheduler adaptive` adds protected backfilling and measured resource feedback;
it also supports `--max-parallel-models 2` or `3` under the shared model-level
CPU/RAM admission budget. Start with the default single model and a small cohort.

Named variants use [the example request](examples/dti_experiments.json).
Replace its literal absolute paths; JSON does not expand `$VARIABLES`.
Use a fresh campaign output path. The runner validates all variants before
submitting, keeps variant outputs separate and records failed/unlaunched jobs.
Set common explicit dependency paths (for example `--model-assets-dir`) in
the request's `base_args` when needed. `options` and per-variant `args` accept
scientific parameters; identity, output paths and execution mode belong to the runner.

<!-- cli-template: experiments -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_experiments.py" \
  --request "/path/to/edited_experiment_request.json"
```

Add `--execute` to run. Variants of one subject are sequential; concurrency
is across subjects. GUI **Advanced settings > Experiments & batch** supports
the same requests. `--jobs` and per-subject `--max-parallel-models` are different
levels and can multiply resource use. See [experiments](EXPERIMENTS.md).

---

## Regional Analysis And Reports

For a completed single-subject run, read its recorded paths instead of scanning
shared output directories. The wrapper above writes metadata under
`$OUTPUTS/<run>/00_run_metadata/<run>_orchestration_manifest.json`. Multiple
tracking variants and experiments have their own child records.

Pass the region selection and metrics directly; no request file is needed:

<!-- cli-template: region-run -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --source-manifest "/path/to/run_orchestration_manifest.json" \
  --regions 1024 2024 --metrics fa md \
  --output-dir "/path/to/new_region_results"
```

Regional analysis reuses eligible maps, full tracks and matrices; it does not
refit models. A region absent from the atlas or a failed/stale source is rejected.

`--regions` accepts names or IDs, separated by spaces/commas, or repeated.
Quote multiword names. `--tissue` and `--surface` add existing DDSeg tissue composition
and cortical morphology. `--tractogram` and `--connectome` select recorded task keys.
Use `--tract-rule endpoint_pair --target-regions ...` for A-B tracks; add
`--include-regions` and `--exclude-regions` for waypoints/exclusions.
Without an explicit analysis choice, the default metric is FA. Ambiguous names
require an explicit ID. To view available labels, metric keys and task keys:

<!-- cli-template: region-inspect -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --inspect "/path/to/run_orchestration_manifest.json"
```

This prints the catalog; `--catalog /path/to/catalog.json` optionally saves it.

For cohort CSV export, give multiple completed subject manifests:

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --source-manifest "/path/to/subject1/run.json" "/path/to/subject2/run.json" \
  --regions 1024 2024 --metrics fa md \
  --output-dir "/path/to/new_region_cohort"
```

JSON remains optional: `--request` accepts [a regional request](examples/region_request.json)
or [A-B request](examples/region_pair_request.json); `--batch` accepts
[per-subject overrides](examples/region_batch.json). Do not mix JSON modes with direct
analysis flags. See [Regional analysis](REGIONAL_ANALYSIS.md) for all parameters.

<!-- cli-template: report -->
```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/generate_subject_report.py" \
  --subject-root "$SUBJECT" --output-root "$OUTPUTS/dti" --run-name dti \
  --manifest "$OUTPUTS/dti/00_run_metadata/dti_orchestration_manifest.json"
```

Inspect task completion, QC, excluded-fit counts, bundle coverage and the
declared output signatures. In **Results and QC**, choose that run and load/reuse
its nodes. CLI execution does not directly modify an already-open Slicer scene.
Regenerating a report does not change failed QC into a pass.

---

## Native PowerShell

Define the corresponding wrapper; pass route arguments as an array rather than
copying Bash continuation characters or JSON quoting. Check platform support
before choosing PNL, MRtrix/FSL, SAF or native postprocessing.

<!-- cli-setup: powershell -->
```powershell
$Slicer = 'C:\Apps\Slicer\Slicer.exe'
$Repo = 'C:\work\SlicerDSpace'
$Subject = 'D:\study\sub-01'
$SubjectId = 'sub-01'
$Outputs = 'D:\analysis\sub-01'
$Assets = 'D:\SlicerDSpace-models'
$DependencyArgs = @()
# $DependencyArgs = @('--dependency-config', 'D:\configs\dependencies.json')

function Invoke-DDSurfer {
    param([string]$Run, [string[]]$RouteArgs)
    & $Slicer --launch PythonSlicer "$Repo\shared\run_subject_pipeline_cli.py" `
        --no-user-config --strict-environment `
        --subject-root $Subject --subject-id $SubjectId `
        --output-root (Join-Path $Outputs $Run) --run-name $Run `
        --model-assets-dir $Assets @DependencyArgs @RouteArgs
    if ($LASTEXITCODE -ne 0) { throw "SlicerDSpace exited with $LASTEXITCODE" }
}

Invoke-DDSurfer -Run 'dti_light' -RouteArgs @(
    '--targets', 'validation,microstructure', '--microstructure-stages', 'dti',
    '--brain-mask-backend', 'dipy', '--plan'
)
```

---

## Optional CPU Calibration For Selected Batch Tasks

Targets and algorithm selections are identical to the subject CLI. Repeated
`--targets`, `--microstructure-stages` and `--skip-targets` are resolved by the
same selector; choosing one model does not schedule the other models. Existing
valid intermediate products retain their normal cache checks.

```bash
# DKI only (plus its shared DTI preparation), with bounded measured CPU selection.
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/preprocessed_subjects --output-root /path/to/results \
  --scheduler adaptive --jobs 3 --autotune \
  --targets microstructure --microstructure-stages dki --execute

# MK-Curve only; reuse an existing checked calibration profile.
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/preprocessed_subjects --output-root /path/to/mkcurve_results \
  --scheduler adaptive --jobs 3 --compute-profile /path/to/compute_profile.json \
  --targets microstructure_dki --dki-fit-method MKCURVE --execute
```

`--autotune` is optional and performs a bounded calibration before the selected
analysis. Matching hardware/implementation/shape/method profiles are reusable.
The default search ceiling is four CPUs (`--tuning-max-cpus`). DKI searches
numerical threads; MK-Curve additionally searches worker counts and voxel block
sizes. Three rotating repeats follow a warm run. Candidates must match all
sampled scientific products exactly; gains below 5% retain the reference setting.
Capacity estimates use the full-input RAM reservation, not just sample RSS.
This selects among tested configurations, rather than guaranteeing a global
optimum. Calibration time and its raw trials are recorded separately.

Explicit `--dki-threads`, `--mkcurve-workers`, `--mkcurve-voxels-per-chunk`,
`--wma-threads` and per-stage reservations take precedence. MK-Curve uses bounded
completion-driven voxel work and indexed output writes. Worker/thread products
must fit the reserved CPU count. `--mkcurve-chunk-schedule ordered` retains the
control path for paired comparisons.

Calibration uses final preprocessed NIfTI inputs; a batch requesting preprocessing
keeps declared settings until corrected inputs are available. Unsupported or
unselected stages are not calibrated. Full WMA calibration is available in
`scripts/autotune_batch.py` when an existing tractogram, WMA root, atlas and Slicer
launcher are supplied. It reruns the actual module and can be expensive; without
those inputs WMA retains its declared reservation, rather than triggering upstream
tracking just for tuning. Dry runs do not perform calibration.

## Reuse, Cancellation And Help

Matching completed outputs can be reused by default; use a new run/output path
for method comparisons. `--no-resume` requests recomputation but does not remove
QC failures. Do not delete caches or change scientific thresholds as a generic
repair step. GUI Cancel stops its owned job tree. In an Agent/terminal run,
retain the owned process/group handle and allow cleanup before escalation;
never terminate unrelated Slicer/Python sessions.

Inspect the **installed revision's** help when options differ:

```bash
"$SLICER" --launch PythonSlicer "$REPO/shared/run_subject_pipeline_cli.py" --help
"$SLICER" --launch PythonSlicer "$REPO/scripts/prepare_inputs.py" --help
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" --help
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_experiments.py" --help
```

Local regression tests parse the tagged recipes against the CLI schema and
analysis resolver. This checks syntax/configuration, not that every dataset,
optional backend or operating system has passed scientific validation.
