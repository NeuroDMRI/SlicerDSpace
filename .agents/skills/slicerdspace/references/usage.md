# Using SlicerDSpace With An Agent

Use this reference after installation or when a user requests an analysis.
Read project files from the authorized checkout/runtime root. The detailed
commands live in `docs/setup/CLI_TEMPLATES.md` and `.zh-CN.md`. The generated
[Shell catalogue](shell_templates.md) includes their algorithm snippets and
links to standalone `../assets/shell/*.sh` templates; do not maintain an
independent hand-edited set of commands.

## Establish The Run

For CPU-heavy cohort requests, consult `docs/setup/COHORT_RESOURCES.md`. The
implemented `--scheduler adaptive --resource-budget 80%` uses shared stage
reservations and protected backfilling; `--subject-limit N` optionally selects
the first N eligible inputs. Leave `--jobs` automatic unless the user asks for a
specific active-subject window. Preview the plan and obtain execution approval.
Resource history is local to the batch output metadata by default. Never claim
that the three-ROI scheduling pilot establishes full-cohort speedup, and do not
raise memory budgets or retry numerical failures without reviewing the cause.

Find the real Slicer launcher, project/runtime root, authorized input, subject
ID, writable output root, intended products and any existing run to reuse.
Inspect the installed CLI's `--help` when its revision differs from the checkout.
Do not infer that every visible product should run. A request to install,
inspect, preview or explain is not a request to execute an analysis.

Check `docs/setup/SUPPORT_MATRIX.md` for the user's OS and backend. Preserve
explicit CPU/CUDA choices. Auto prefers usable CUDA in supported neural methods;
DIPY masking, tensor fitting and several other numerical methods are CPU-based.
Plan concurrency against the combined RAM/GPU demand, not just CPU count.

## Prepare Inputs

Identify canonical inputs, NIfTI with gradients, DICOM, or NRRD/NHDR. For DICOM,
isolate the intended diffusion series; clarify a genuinely ambiguous series
choice rather than combining acquisitions. Use `scripts/prepare_inputs.py`
with `--dry-run` before an authorized import. Preserve original files and avoid
`--force` unless replacement is explicitly requested.

Canonical inputs are `input/data.nii.gz`, `input/bvals`, `input/bvecs`, optional
`input/dwi.json`, and a supplied or generated `input/nodif_brain_mask.nii.gz`.
Verify volume/gradient counts, finite values, gradient orientation and mask
shape/affine. Valid supplied masks are retained. PNL remains the default missing
mask generator (`--brain-mask-backend pnl_cnn`; route `--pnl-mask-route auto|managed|native`,
native is experimental); DIPY is an explicit model-free alternative, not a fallback.

For a small real installation example, use the project's
`docs/setup/SLICER_SAMPLE_DTI.md`. A synthetic tensor-field demo only checks
plumbing; do not present its outputs as anatomical validation.

## Choose A Route

| User goal | Configuration to inspect |
| --- | --- |
| DTI maps | Microstructure, stage `dti`; choose the mask route explicitly |
| Tissue/region/surface reconstruction | DDSeg, DDParcel and/or DDSurfer; selected weights and native DTI preparation |
| Cortical metrics/atlas mapping | Native reconstruction plus `surface_postprocess`; licensed native components |
| Free water / NODDI / DKI | Requested independent stages; acquisition eligibility and validity masks |
| Whole-brain tractography | Explicit `dti_det`, `ukf` or `csd_act`; confirm prerequisites and resources |
| Short association fibres | `saf`; accepted surfaces/postprocessing and CSD inputs |
| Bundle identification | Explicit WMQL/WMA/TractCloud selection and whole-brain source |
| Structural connectivity | Matching tractogram, native atlas and backend; SIFT2 weights only with their source CSD tracks |
| Region-specific analysis | Completed, QC-eligible run via `scripts/analyze_regions.py`; no model refitting |
| Parameter comparisons/cohort | Advanced experiments or documented batch CLI; separate output roots |

NODDI/DKI require eligible multi-shell sampling. Do not enable every model just
because a flag exists. Check gradients and the actual eligibility report;
do not fill invalid fits with zero or lower thresholds to make a run green.
Check the selected FW estimator's protocol suitability too.

For bundles, select `--bundle-tractography-backend` so the source is explicit.
UKF is automatically called through the installed extension/CLI; the standard
route does not require the user to generate UKF tracks manually. An existing
tractogram is an advanced input, not a mandatory handoff. Do not combine an
explicit WMA/TractCloud file with the automatic source option. WMA commonly
uses UKF; runnable CSD/DTI sources still need their own coverage/QC review.
SAF is not a whole-brain bundle source.

WMQL keeps the unchanged 48-query `official` preset as the default. Use
`--wmql-query-preset official_cerebellar` to add three SlicerDSpace regional
cerebellar selections, or `cerebellar` for those selections only. Subsets such as
`--wmql-query-selection cerebellar_brainstem,cerebellar_bilateral` belong to these
new presets. They select regional candidates, not independently identified
ICP/MCP/SCP bundles. Preserve the query provenance and per-output QC; see
`docs/setup/WMQL_CEREBELLAR.md`.

WMA atlas selection is independent of tractography source: `--wma-atlas org`
is the default; `east-west` selects the Eastern/Western adult atlas, and `naba`
selects the neonatal/adult atlas. NABA is not a general paediatric atlas.
Prepare only `wma`, `wma-east-west` or `wma-naba` under the confirmed asset root
using `scripts/setup_optional_assets.py`; inspect its plan and terms before
downloading. `--wma-atlas-dir` can point to an existing copy of the selected
atlas. Keep atlas comparisons in separate runs and retain the atlas-specific
group definitions, registration and per-bundle QC. GUI: Advanced settings >
Model parameters > WMA atlas. See `docs/setup/WMA_ATLASES.md` for sources.

## Prepare, Plan, Execute

In the GUI: enter Inputs, choose Products and their `...` algorithms, then
**Environment and installation > Prepare selected analysis environment**.
Advanced settings handles parameters, including **Experiments & batch**;
environment paths/installations stay in the separate Environment panel.
Reuse the confirmed model/resource root. If none is configured and downloads
are needed, follow the Skill's storage-choice step before preparing resources.
Follow its installation rules for downloads, licenses and restarts.

In the CLI, use the actual Slicer launcher with `--launch PythonSlicer` and
`shared/run_subject_pipeline_cli.py`. The recipe wrapper works from a checkout
or installed runtime tree and isolates output by route/run name. Saved GUI
defaults are excluded by the templates. Explicitly pass any required existing
dependency paths, including the confirmed `--model-assets-dir`; the documented
`--dependency-config` can restore those paths.

Use the selected recipe with `--doctor`, inspect blockers, then `--plan`.
Both can write metadata. Review the chosen inputs, model versions, tasks,
outputs and resource budget. Only change to `--execute` when the user's
requested analysis and scope are clear. Long/increased-scope downloads or a
new expensive route need confirmation, not silent expansion.

Retain the exact argv, logs, run/plan manifests, input/model signatures, device
and timing. `00_run_metadata/run_record.json` holds the parameters, the brain-mask
route and backend that was used, the accepted licences and the software versions; a
re-run keeps the old file as `run_record-<started_at>.json`. Default resume reuses compatible recorded outputs; use new run/output
paths for controlled method comparisons. `--no-resume` is an explicit rerun,
not a repair for an unexplained failed QC record.

## Batch And Experiments

GUI experiments are opt-in. Public CLI parameters go in the override field;
named JSON variants are supported for the current subject or prepared cohort.
Preview the complete job count. Different subjects may run concurrently;
variants of one subject run sequentially, with isolated outputs.

The basic `scripts/run_batch_pipeline.py` runs one configuration per subject.
For resource-aware cross-subject stage admission, use `--scheduler resources`
with automatic resource estimates; omit `--jobs` for automatic subject concurrency,
or set a limit such as `--jobs 3`. Keep explicit CPU/RAM and stage overrides for
shared-host policies or pilot-informed adjustments. See `docs/setup/COHORT_RESOURCES.md` for the CLI
example and limitations. Keep `--max-parallel-models 1` in this mode; do not
describe it as resident-model inference or per-device VRAM enforcement.
The `scripts/run_experiments.py` request format runs explicit named variants.
Its JSON paths are literal absolute paths, not shell variables. Start with one
level of concurrency; `--jobs` and `--max-parallel-models` multiply resource use.
Read each child manifest and the batch task/status counts, not just process exit
codes. Blocked analyses make the batch exit nonzero unless the user explicitly
accepts partial results with `--allow-partial-success`; that flag does not clear
the blocked state or certify missing outputs.
Preserve failed, cancelled and unlaunched jobs. Never select only the best seed
or relabel an unavailable method as a reproduced result.

Use the GUI's Cancel for GUI-owned processes. For an Agent-launched CLI job,
retain the process/group handle and terminate only that owned tree, allowing
cleanup before escalation. Do not kill all Slicer/Python processes or interrupt
other agents. Existing scenes and sessions stay open unless the user requests
otherwise.

## Review And Hand Back

For routine regional analyses, pass options directly instead of asking the user
to prepare JSON. The source manifest is already produced by the subject workflow:

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/analyze_regions.py" \
  --source-manifest "/path/to/run_orchestration_manifest.json" \
  --regions 1024 2024 --metrics fa md --output-dir "/path/to/new_region_results"
```

Names/IDs, repeated region flags and comma-separated lists are supported. Use
`--inspect` only when the available names/metric/task keys need checking; it
prints to the terminal without requiring a catalog file. Add `--tissue`,
`--surface`, `--tractogram`, `--connectome` or tract-selection options as requested.
Multiple `--source-manifest` paths use the same choices for cohort export. Keep
JSON requests optional for saved configurations or subject-specific overrides.

Read the exact child run manifest, not a shared-directory scan or latest filename
guess. Verify task status, requested products, declared output existence, QC
records/signatures and scientific eligibility. Keep execution completion,
automatic QC and human anatomical review separate. Check sparse bundles,
excluded fits and empty regions explicitly; passing file integrity is not
an anatomical approval.

GUI runs load declared key results; repeated loads reuse run-owned nodes.
Use **Results and QC** to choose a run and load/reuse or replace that run's
nodes. Bulk runs show one successful experiment automatically. If operating a
current scene through an available GUI/Python channel, preserve unrelated nodes.
Without that channel, hand back exact paths/reports or offer a separate viewer;
do not pretend that a CLI process loaded the user's current MRML scene.

Generate the subject report from the recorded manifest when needed. Return the
selected methods/source, actual outcome, output/manifest/report/log paths,
runtime/device and unresolved review items. For failure, report the first
actionable cause and stop blind retries or method substitution.
## Optional Batch CPU Calibration

Use the same subject CLI targets and algorithm subsets for batches; do not
expand a user's selected models into the full pipeline. `--scheduler adaptive`
supports optional `--autotune` on final preprocessed NIfTI inputs, or reuse of a
validated `--compute-profile`. Explain the extra calibration cost and confirm it
before starting. Explicit thread, MK-Curve worker and chunk parameters take
precedence. Never describe a sampled calibration as full-cohort speedup or a
globally optimal thread count. See `docs/setup/CLI_TEMPLATES.md` for examples,
unsupported-input handling and full WMA calibration requirements.
