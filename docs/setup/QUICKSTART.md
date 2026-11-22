<div align="center">

# Quick Start

**A small, inspectable first workflow**

[Project](../../README.md) | [Documentation](../README.md) | [Platforms](SUPPORT_MATRIX.md) | [QC](GUI_RESULT_REVIEW.md) | [中文](QUICKSTART.zh-CN.md)

</div>

## Install 3D Slicer 5.10.0 & SlicerDSpace

No automatic first-run wizard is shown; this project is not yet in the official Extensions Catalog.

1. **Install 3D Slicer 5.10.0** from <https://download.slicer.org/> for Windows (`x86_64`), Linux (`x86_64`), or Apple Silicon macOS (`arm64`). Default application and launcher locations:
   - **Windows**: `%LOCALAPPDATA%\slicer.org\Slicer 5.10.0\Slicer.exe` (or `C:\Program Files\Slicer 5.10.0\Slicer.exe`; Python launcher: `bin\PythonSlicer.exe` or `PythonSlicer.exe`)
   - **Linux**: extracted folder such as `~/Slicer-5.10.0-linux-amd64/Slicer` (or `/opt/Slicer-5.10.0-linux-amd64/Slicer`; Python launcher: `bin/PythonSlicer`)
   - **macOS**: `/Applications/Slicer.app` (CLI launcher: `/Applications/Slicer.app/Contents/MacOS/Slicer`; Python launcher: `/Applications/Slicer.app/Contents/bin/PythonSlicer`)
2. **Download the SlicerDSpace release archive** from <https://github.com/NeuroDMRI/SlicerDSpace/releases> (`SlicerDSpace-v*-Windows-x64.zip`, `SlicerDSpace-v*-Linux-x64.tar.gz`, or `SlicerDSpace-v*-macOS-arm64.tar.gz`; do not use GitHub's automatic `Source code` archive).
3. **Install in 3D Slicer 5.10.0**: open **View > Extensions Manager**, click the top-right wrench icon, choose **Install Extension from File...**, select the downloaded archive, and restart 3D Slicer. Open **Modules > Diffusion > SlicerDSpace**.
4. **Unpacked extension paths on disk** (containing `lib/Slicer-5.10/qt-scripted-modules/shared/run_subject_pipeline_cli.py` and `bundled-runtimes/`, also shown under **Edit > Application Settings > Modules**):
   - **Windows**: `%LOCALAPPDATA%\slicer.org\Slicer 5.10.0\slicer.org\Extensions-34045\SlicerDSpace`
   - **Linux**: `<Slicer-5.10.0-dir>/slicer.org/Extensions-34045/SlicerDSpace`
   - **macOS**: `/Applications/Slicer.app/Contents/Extensions-34045/SlicerDSpace` (or `~/Library/Application Support/slicer.org/Extensions-34045/SlicerDSpace`)

For developer source staging instead of a release archive, see [the project README](../../README.md#installation).

---

## Update The Code

A release archive holds the code of its build day. Later fixes are published as an incremental code update for the same release. Apply the update once after the installation, and again when a new update is published.

- **GUI**: in the SlicerDSpace module, click **Check for updates**, confirm **Download and apply now?**, then restart 3D Slicer.
- **Command line only** (for example a server without a display): run the commands below. `MODULE_ROOT` is the `lib/Slicer-5.10/qt-scripted-modules` folder of the unpacked extension (step 4 above). The example uses the Linux paths.

~~~bash
SLICER_DIR="$HOME/Slicer-5.10.0-linux-amd64"
MODULE_ROOT="$SLICER_DIR/slicer.org/Extensions-34045/SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules"
"$SLICER_DIR/bin/PythonSlicer" - "$MODULE_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from shared import code_updater
check = code_updater.check_for_code_update(module_root=root)
print("status:", check["status"])
if check["status"] == "update_available":
    print(code_updater.download_and_apply_code_overlay(module_root=root, manifest=check["remote"]))
PY
~~~

The first run prints `status: update_available` and then a result with `'status': 'applied'`. A second run prints `status: up_to_date`. The update uses the saved network settings (mirror, proxy). It changes code files only; it does not download models or runtimes. The status `runtime_upgrade_required` means that a new release archive must be installed.

---

## Prepare A Subject

Use authorized, de-identified DWI in NIfTI (with matching bval/bvec), NRRD/NHDR,
or a DICOM folder (choosing the target series if multiple series are present).
Under **Inputs**, select the DWI source directly; import and validation into
`<subject>/input/` run automatically when execution starts. A brain mask is optional:
supply an existing mask under **Inputs** or let the selected mask backend (PNL CNN by
default) generate one. The DTI fit runs inside Slicer; no extra extension is needed for it.

Under **Analysis**, tick the stages you need. The plugin adds required upstream
prerequisites automatically; preprocessing is never added unless ticked.
When you click **Run SlicerDSpace**, preflight dependency checks run for the selected
stages first. If required packages, external tools or model weights are missing,
use **Prepare selected analysis environment** when prompted: one dialog lists the
required downloads and licences, and pressing **Accept licences and install** downloads,
verifies and caches them for reuse (restart Slicer if requested, then run again).
Preparing an already-ready environment does not prompt, download or reinstall.
Dependency details, **Use offline model resources**, **Repair selected managed tools**,
and manual path overrides are available under **More options (offline, repair, paths)**
in the setup prompt (see [managed setup](MANAGED_SETUP.md) for coverage and platform limits).
MRtrix3, FSL and FreeSurfer are not needed just to try a DTI-only route.
Missing dependencies are reported rather than silently replaced by a different method.

Each stage row under **Analysis** shows the current algorithm configuration (with **DTI**,
**MRtrix3 CSD / ACT** and **WMQL** as defaults); these labels reflect settings rather than
completed or QC-approved results. Click **...** beside a stage to choose one or several
algorithms and adjust stage parameters (for example, FW, NODDI, DKI and iFW alongside DTI;
iFW requires the echo time and also runs FW and DDSeg), and use **Compute device** to select
Auto, CPU or a specific CUDA GPU.

Multiple tracking methods run sequentially as separate `<run>__<backend>`
variants. Shared anatomical priors/microstructure runs once; selected bundle methods and
connectomes are computed for each tracking variant. Streamlines are not combined;
dependency checks and QC gates still apply. More methods require more time and disk space.

---

## Run DTI Only

1. Under **Inputs**, select the Subject root (or direct NIfTI/NRRD/DICOM source).
   An empty subject ID is filled from the folder name; an existing custom ID is
   preserved. Selecting its canonical `input/` directory resolves to the subject root,
   and an empty output directory defaults to `<subject>/outputs/`.
2. Under **Analysis**, select **Microstructure** and uncheck all other stages, including preprocessing.
3. Click **...** beside **Microstructure** and retain only DTI.
4. Click **Run SlicerDSpace** and inspect the log. Necessary preflight checks
   run first; if any dependency is missing, prepare it via the setup prompt and
   retry rather than silently omitting selected stages.
5. Completed outputs remain on disk for on-demand inspection rather than loading
   automatically into the scene. In **Results and QC**, select the completed run,
   click the eye icon beside FA or MD to load or toggle them in 3D Slicer (right-click
   to `Unload from scene` or open table/JSON files), and click **Open report** to view
   the HTML QC summary.
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
single-subject (or toggle **Batch** under **Inputs** to process a cohort root; see the CLI/experiment guide for named parameter variants).

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

Missing fields turn the waiting message red. Resolve missing dependencies via the preflight setup prompt before retrying. Cancel stops active jobs; installed packages are not
rolled back and partial downloads may resume. Redact identifiers before sharing logs.
