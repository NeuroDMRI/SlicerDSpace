<div align="center">

# Quick Start

**A small, inspectable first workflow**

[Project](../../README.md) | [Documentation](../README.md) | [Platforms](SUPPORT_MATRIX.md) | [QC](GUI_RESULT_REVIEW.md) | [中文](QUICKSTART.zh-CN.md)

</div>

Use the source installation in the README. No automatic first-run wizard is
shown; this project is not yet in the official Extensions Catalog.

---

## Prepare A Subject

Use authorized, de-identified DWI with matching bval/bvec, or a DICOM series.
A brain mask is optional: without one, the selected mask backend (PNL CNN by
default) makes it. Follow the canonical input layout in the README. DICOM/NRRD
conversion is under **Advanced settings > Regions & input import > Input import / conversion**; preview
before writing. The DTI fit runs inside Slicer; no extra extension is needed for it.

Tick the products you need. The plugin adds the prerequisites of each product
by itself; preprocessing is never added without your tick.
Check the runtime in the separate **Environment and installation** section.
Choose **Prepare selected analysis environment**. One dialog lists the downloads
with their licences; press **Accept licences and install** once.
Restart when requested, then click Prepare again to continue. After the install,
a self-check runs for each feature.
Checks run in a cancellable background process and update the inline summary.
Preparing an already-ready environment does not prompt, download or reinstall.
Dependency details, offline import, repair and manual paths are under
**More options (offline, repair, paths)** in the Environment section.
Use **Use offline model resources** for a predownloaded model directory, or
**Repair selected managed tools** for a damaged plugin-managed runtime.
See [managed setup](MANAGED_SETUP.md) for coverage and platform limits.
MRtrix3, FSL and FreeSurfer are not needed just to try a DTI-only route.
Missing dependencies are reported, not silently replaced by a different method.

The Microstructure, Tractography and Fiber bundles rows show the current algorithm
selection: **DTI**, **MRtrix3 CSD / ACT** and **WMQL** by default. These labels
update when settings change; they describe the configuration, not completed or
QC-approved results. Click **...** beside each of these three products to
choose one or several methods. DTI remains required preparation; FW, NODDI, DKI and
iFW are independent options (iFW needs the echo time, and also runs FW and DDSeg). Advanced settings uses categories, showing
only the selected category's parameters. Environment installation is separate.

Multiple tracking methods run sequentially as separate `<run>__<backend>`
variants. Shared anatomical priors/microstructure runs once; selected bundle methods and
connectomes are computed for each tracking variant. Streamlines are not combined;
dependency checks and QC gates still apply. More methods require more time and disk space.

---

## Run DTI Only

1. Select Subject root. An empty subject ID is filled from the folder name;
   an existing custom ID is preserved. Selecting its canonical `input/` directory
   resolves to the subject root. An empty output directory uses `<subject>/outputs/`.
2. Select **Microstructure** and uncheck all other products, including preprocessing.
3. Click **...** beside Microstructure and retain only DTI.
4. Click **Run SlicerDSpace** and inspect the log. Necessary preflight checks
   run in the background. Missing dependencies stop submission and open the
   Environment section; prepare them and retry rather than silently omitting products.
5. FA/MD automatically load as successful stages finish. In the separate **Results and QC** section, select the completed run to inspect QC or use **Load** again without duplicating nodes.
6. Inspect the mask, finite-value checks, spatial alignment and QC summary before accepting results.

Generate a plan without execution:

~~~bash
PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets validation,microstructure --microstructure-stages dti --plan
~~~

MD loads in grayscale. Display settings do not change voxel values.
Execution completion, automated QC and human review are separate states.
DTI is the default; enable FW, NODDI, DKI and iFW with the main-screen selector only when
their acquisition requirements are met. The DTI products include a colour FA map. A missing input mask uses the selected
backend (PNL CNN by default); an existing mask avoids that dependency. Configure
mask alternatives and external runtimes before running.

---

## Select A Bundle Source In The CLI

For all routes, Bash/PowerShell launchers, masks, postprocessing, optional models,
cohorts and reports, use the [command-line templates](CLI_TEMPLATES.md).

For GUI parameter overrides and named variants across subjects, see
[Lightweight setup and experiments](EXPERIMENTS.md). Default GUI runs remain
single-subject; experiments are opt-in under Advanced settings.

The CLI can choose the whole-brain tracking source independently of the bundle
methods. For example, run CSD/ACT and UKF, then apply WMQL and TractCloud only
to the UKF result:

~~~bash
PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 \
  --targets segmentation,parcellation,tractography,bundles \
  --tractography-backend csd_act,ukf --bundle-backend wmql,tractcloud \
  --bundle-tractography-backend ukf --plan
~~~

Replace `--plan` with `--execute` after checking dependencies and inputs. The
source can be `ukf`, `csd_act`, or `dti_det`; it is run automatically if needed.
All selected bundle methods use that source. To use different sources for
different bundle methods, run separate commands with distinct `--run-name` values.
Omitting the new option preserves the existing per-tracking-variant behaviour.
Explicit WMA/TractCloud input-file options cannot be combined with this source
option. SAF is a short-association-fibre product, not a whole-brain source.
WMA results from non-UKF sources require particular care: the original QC
thresholds and anatomical review still apply, and a runnable combination is not
a guarantee of adequate bundle coverage. Install the dependencies for the
selected methods and use `--doctor` before execution.

---

## Synthetic Installation Check

From the repository root, in a runtime with NumPy and nibabel:

~~~bash
PythonSlicer -c "from pathlib import Path; from shared.demo_data import create_synthetic_demo_subject; print(create_synthetic_demo_subject(Path('ignore/demo')))"
~~~

Use the resulting <code>ignore/demo/slicerdsurfer-demo</code> as Subject root.
This is a small constant tensor field with seven measurements, **not human
anatomical priors**. It checks installation/DTI plumbing, not surface accuracy, NODDI or
clinical validity.

The legacy Demo handler is retained for tests, not an advertised visible button.
For a real DWI example, use the [Slicer DWIVolume walkthrough](SLICER_SAMPLE_DTI.md).
It uses Slicer's existing Sample Data entry without mirroring the images.
Registration of a separately redistributable SlicerDSpace dataset remains pending.

---

## If A Run Fails

Missing fields turn the waiting message red. Configure missing dependencies in
Environment and installation before retrying. Cancel stops active jobs; installed packages are not
rolled back and partial downloads may resume. Redact identifiers before sharing logs.
