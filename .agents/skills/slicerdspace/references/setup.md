# SlicerDSpace Setup

Use this reference for installation, activation, selected dependency preparation,
offline import and repair. Project paths refer to the authorized checkout or
installed runtime containing `shared/` and `scripts/`, not the Skill directory.
The ordinary path is to enable the module, confirm storage, then use the single
GUI preparation entry. Shell workers below are optional diagnostics, not a
second installation workflow. User-facing example prompts belong in the
README/website, not this Skill.

Read as needed: [activation](#stage-and-enable-the-module),
[storage](#confirm-storage), [GUI preparation](#prepare-through-the-gui),
[Agent requests](#agent-request-inside-slicer),
[worker diagnostics](#optional-worker-diagnostics),
[platform limits](#platform-and-delivery-limits),
[retry and handoff](#retry-offline-import-and-handoff).

## Inspect Before Installing

- Find the actual Slicer launcher, version, OS/architecture, authorized checkout
  and intended install location. Do not download another Slicer, change drivers
  or invent credentials to resolve a setup failure.
- Read `docs/setup/QUICKSTART.md`, `docs/setup/MANAGED_SETUP.md` (or its
  `MANAGED_SETUP.zh-CN.md` counterpart) and `docs/setup/SUPPORT_MATRIX.md`.
  Also read `docs/setup/WINDOWS.md` or `docs/setup/MACOS.md` on those systems.
- For a web link, obtain an authorized checkout before using relative scripts.
  An inaccessible private repository requires access or a local checkout, not
  installation of an unrelated dependency. Record `git rev-parse HEAD` and local
  modifications; preserve dirty work and private repository access boundaries.
- Reuse a working module; do not reinstall it just to prepare another route.
  Prefer a trusted official package when available. Current Quick Start says
  the project is not in the official Extensions Catalog; source staging is not
  an Extension Manager release.
- Preserve open scenes and other sessions. Ask before closing/restarting them.
  An isolated test process needs isolated settings and a separate display if
  required; do not persist activation into a test profile when the user wants
  their normal profile enabled.

## Stage And Enable The Module

For a source install, use normal Python from the authorized checkout. Substitute
the approved install root and installed Slicer major/minor version:

```bash
python3 scripts/stage_slicer_extension.py \
  --install-root /path/to/user/extensions --slicer-version Slicer-5.10 \
  --link-mode copy
```

The module directory is
`<install-root>/SlicerDSpace/lib/Slicer-<major>.<minor>/qt-scripted-modules`.
Verify `SlicerDSpace.py`, declared runtime resources and
`<install-root>/SlicerDSpace/extension_staging_manifest.json`. Do not copy the
whole development repository. Staging replaces runtime files even without
`--clean`; require upgrade approval before targeting an existing installation.
Use `--clean` only for an approved upgrade of that exact staged extension,
retaining its previous manifest/settings.

Use `scripts/enable_in_slicer.py` from the complete Skill directory, with these
process environment variables:

| Variable | Value |
| --- | --- |
| `SLICERDSPACE_AGENT_MODULE_DIR` | Absolute staged module directory |
| `SLICERDSPACE_AGENT_RECEIPT` | Writable JSON activation receipt path |
| `SLICERDSPACE_AGENT_PERSIST` | `1` only for requested persistent enablement |
| `SLICERDSPACE_AGENT_EXIT` | `1` only for an isolated activation-only test; otherwise unset |

Launch Slicer itself, not `SlicerApp-real`, PythonSlicer or system Python:

```bash
/path/to/Slicer --no-splash \
  --additional-module-paths /absolute/staged/module-directory \
  --python-script /absolute/path/to/slicerdspace/scripts/enable_in_slicer.py
```

The helper checks the import path, opens the widget, runs registered native
Self Tests and optionally persists the additional-module path. It refuses
conflicting installations. It does not prepare dependencies, change algorithm
settings, clear the scene or run analysis. Inspect the receipt: activation and
native Self Tests do not establish dependency readiness or scientific QC.

PowerShell uses `$env:NAME = 'value'` and `&` to invoke the launcher. For a macOS
`.app`, use `Contents/MacOS/Slicer`. Do not pass shell syntax as subprocess
arguments. Keep the complete Skill folder when copying/installing the Skill.

## Confirm Storage

Before downloading models, atlases or other large resources, confirm storage
outside the extension installation so upgrades cannot remove it. Apply this
step even when the installation request says nothing about storage. The GUI's
automatic default is not user confirmation.

- If the user already supplied a path, validate and use it without asking them
  to choose again. Otherwise inspect saved `external_dependencies.model_assets_dir`,
  `SLICERDSPACE_MODEL_ASSETS_DIR` and
  `shared.model_assets.default_user_model_assets_root()`. Consider user-approved
  data locations/mounted user storage, not unrelated medical data or another
  account's cache.
- Offer two or three usable absolute-path candidates when available, stating
  free space, writability, verified reusable assets and the reason for the
  recommended option. If only one is usable, explain that; allow a custom path.
  For a new directory, inspect its nearest existing parent without creating it.
- **Ask the user to choose and wait for their answer before creating the
  directory, moving files or downloading resources.** A recommendation or
  preselected option is not consent. Never silently pick a developer path,
  default, system drive or largest disk.
- Estimate the selected profiles' download/extraction needs from
  `models/assets_manifest.json` and optional resource manifests. Python/Conda
  solver downloads are additional and can require several GB. Verify backend
  access, including Windows/WSL mapping. Reuse only manifest-verified resources;
  do not acquire every profile or move caches without approval.
- Save the confirmed absolute root as `external_dependencies.model_assets_dir`
  with `shared.user_config.update_user_config`, preserving other settings, or
  use **Environment and installation > More options > Model assets root**.
  Use the same root for downloader `--asset-root` and analysis
  `--model-assets-dir`. Resolve a conflicting `SLICERDSPACE_MODEL_ASSETS_DIR`
  for these processes without editing global shell profiles unasked.
- Review the plan's source-cache and runtime destinations too:
  `model_assets_dir` does not relocate every backend source/runtime cache.
  Confirm those destinations and available space before approving downloads.

For later requests, reuse this root and ask about new downloads/terms. Ask for a
new root only if requested or the current one is no longer suitable.

## Prepare Through The GUI

Select only the requested products and scientific options, then choose
**Environment and installation > Prepare selected analysis environment**.
Dependency reports and First-use setup route to this same preparation logic.
Do not use Run as an installation shortcut: setup permission is not permission
to submit an analysis.

1. Inspect the plan, destinations, sizes and terms with the user. Obtain
   explicit approval for downloads and applicable algorithm, resource, FSL,
   PNL source/model/template and FreeSurfer/component terms. The user must make
   the real dialog approvals. Never autoclick checkboxes/Prepare, monkeypatch
   confirmation, inject acceptance or treat hashes as a license grant.
2. Let Slicer's Extension Manager install/enable the needed SlicerDMRI/PyTorch
   extensions; UKFTractography is requested only for UKF. An enabled extension
   is not necessarily loaded. If restart is required, stop, ask the user to save
   work and approve restarting, then reopen SlicerDSpace.
3. Continue with **Continue preparing selected analysis environment** or the
   reentry offer. It rechecks current selections and asks for needed approval
   again, reusing verified extensions/resources. It does not replay commands,
   transfer old terms approval to a changed plan or automatically run analysis.
   Loaded-package changes can also require restart. Only final readiness clears
   the pending setup marker.
4. Background workers prepare missing sources, Python wheels, selected model
   profiles and supported external tools. Preserve explicit interpreter/source
   overrides. Compatible DDSeg/DDParcel/DDSurfer/TractCloud share Slicer Python
   and official-extension PyTorch by default; WMQL/DIPY/AMICO use workflow Python.
   Protect bundled NumPy/SciPy/VTK/SimpleITK/PyTorch. Conflicts or unavailable
   wheels must fail visibly, not trigger core upgrades or local compilation.
5. Prepare PNLBrainMask-Torch with its converted weights and ANTs in the managed
   Python 3.11 prefix. Reuse a compatible PyTorch there; new prefixes use the
   paired-tested PyTorch 2.5.1 CPU build. TensorFlow/Keras are not runtime dependencies. Selected supported external
   runtimes use user-local managed prefixes and, when needed, a hash-verified
   pinned Miniforge bootstrap; do not replace unmanaged environments, drivers or
   global shell/CUDA settings.
6. Inspect the final dependency report and synthetic functional results, not
   just package imports or worker completion. Failed functional checks block
   readiness. Receipts/logs live in `setup-history` beside
   `shared.user_config.default_user_config_path()`. A `ready` receipt means
   installation readiness for that selection, not anatomical correctness.

Prefer pinned WMQL/WMA/TractCloud archives through `shared.backend_sources`:
exact revisions/SHA256, safe extraction, retained licenses and per-file receipts,
without Git or duplicated algorithm implementations. Sources and model/atlas
profiles are separate phases. Do not replace a custom configured source root.

If fewer external environments are requested, consult `docs/setup/EXPERIMENTS.md`
and explain the Slicer-native preset before offering it. It changes methods;
never silently replace PNL/CSD or relax scientific options to make setup pass.
An existing valid supplied mask avoids automatic mask extraction.

### Agent Request Inside Slicer

Use `scripts/prepare_in_slicer.py` from the Skill in an **already enabled Slicer**
process. It does not install/register the module; first use
`enable_in_slicer.py` or an already loaded widget. It needs Qt, so use Slicer's
`--python-script`, **not `--launch PythonSlicer`**.

After storage is confirmed, provide a JSON request such as this structural
example, substituting only the user's requested targets/settings:

```json
{
  "mode": "plan",
  "targets": ["segmentation"],
  "model_assets_dir": "/user/confirmed/path",
  "parameters": {}
}
```

- `mode`: `plan` (default) or `prepare`. Plan performs read-only dependency
  checking/export and writes reports, without installation or analysis. The
  helper applies the requested GUI target/scientific selections and displays
  the root, so it is not a snapshot/restore of widget settings. Prepare invokes
  the real plugin confirmation, not unattended acceptance.
  Normal check metadata (including `lastEnvironmentCheck`) can be saved in
  Slicer settings; reports use the current output/subject metadata directory
  when present. Planning is not a no-filesystem-write inspection mode.
- `targets`: a nonempty list from `validation`, `preprocessing`, `segmentation`,
  `parcellation`, `surface`, `surface_postprocess`, `microstructure`,
  `tractography`, `bundles`, `wma`, `tractcloud`, `connectome`.
- `model_assets_dir`: the user-confirmed absolute path outside the extension.
- `parameters`: optional MRML field names with string values, restricted to
  `brainMaskBackend`, `brainMaskDevice`, `ddsegDevice`, `device`,
  `preprocessingBackend`, `orchestrationMicrostructureStages`,
  `orchestrationDkiFitMethod`, `orchestrationTractographyBackend`,
  `orchestrationBundleBackend`, `wmaAtlas`, `connectomeBackend`.
  Choice validation applies. Include only requested scientific options, not
  paths, installation flags or consent. Bundle targets select their respective
  methods unless an explicit bundle setting is provided; inspect the plan.
- `prepare_mask`: optional boolean, default `false`. Set `true` only to
  explicitly prepare the chosen mask backend's dependencies without patient
  input (for example, PNL if selected). Otherwise the GUI may defer PNL until
  input is selected. This does not change the masking method or generate a mask.

Set the request/receipt paths for the actual Slicer process:

```bash
SLICERDSPACE_SETUP_REQUEST=/absolute/setup-request.json \
SLICERDSPACE_SETUP_RECEIPT=/absolute/setup-receipt.json \
/path/to/Slicer --no-splash \
  --python-script /absolute/path/to/slicerdspace/scripts/prepare_in_slicer.py
```

Use the enabled normal profile, or pass the previously verified staged directory
via `--additional-module-paths` for a nonpersistent install. For an existing
session, execute the helper inside that Slicer's Python console with those
environment variables set; do not terminate the session to attach automation.
Do not pass subject/input/output paths: the helper accepts no patient-analysis
request. It neither clears the scene, installs every package, runs data nor
restarts Slicer. Leave `SLICERDSPACE_AGENT_EXIT` unset when continuing from
activation in the same process. `SLICERDSPACE_SETUP_EXIT=1` is optional only for
an isolated process to exit after completion; it does not bypass confirmation.

Wait for the terminal JSON receipt, not merely process startup. Plan returns
`status: planned` with `plan` and `environment`; preparation follows the plugin's
`ready`, `blocked`, `restart_required`, `failed` or `cancelled` outcome.
Treat an in-progress receipt as unfinished. Preserve error reports and use the
normal restart/resume path for `restart_required`; never chain analysis onto it.

## Optional Worker Diagnostics

Use the actual runtime's workers and their current `--help`. These interfaces
were checked with `--help` on Slicer 5.10/Linux; that is interface verification,
not a clean-install or cross-platform acceptance test. Ordinary GUI preparation
already invokes these workers and saves verified settings.

For selected algorithm sources, plan first (writes only the result report):

```bash
PythonSlicer /path/to/runtime/scripts/setup_backend_sources.py \
  --profiles wmql --root /user/confirmed/algorithm-sources \
  --result /absolute/source-plan.json
```

Profiles are space-separated choices `wmql`, `wma`, `tractcloud`; select only
needed ones. After the user explicitly approves that plan, source terms and
destinations, the executable worker form is:

```bash
/path/to/Slicer --launch PythonSlicer \
  /path/to/runtime/scripts/setup_backend_sources.py \
  --profiles wmql --root /user/confirmed/algorithm-sources \
  --execute --accept-terms --result /absolute/source-result.json
```

`--accept-terms` records already-obtained explicit user approval, never inferred
from an installation request. For offline import add
`--offline-root /absolute/trusted-archives` containing the requested
`wmql.tar.gz`, `wma.tar.gz` or `tractcloud.tar.gz`; trusted pinned hashes still
apply. `--root` is a source-cache root, not the model asset root. This worker
neither installs Python requirements nor persists returned
`external_dependencies`; use GUI preparation for verified path saving and the
remaining phases.

For a standalone installation-function check, supply a request JSON such as:

```json
{"imports": ["numpy", "nibabel", "vtk"], "devices": [], "models": []}
```

```bash
/path/to/Slicer --launch PythonSlicer \
  /path/to/runtime/scripts/check_setup_functionality.py \
  --request /absolute/functional-request.json \
  --result /absolute/functional-result.json
```

`shared.setup_selftest.requests_for_report(...)` builds route-specific requests
grouped by the actual selected interpreter; `run_checks(request)` executes
them. Respect interpreter overrides and explicit CPU/device choices rather than
testing every group in Slicer or silently selecting CUDA.
Requests can carry verified `pythonpath`, selected PyTorch `devices` and
manifest-qualified `models`; prefer the GUI-generated requests for those.
Checks exercise a NumPy solve, NIfTI and VTK streamline round trips, and, where
selected, PyTorch allocation, DDParcel/DDSurfer checkpoint loading, synthetic
inference for all selected DDSeg views, and TractCloud's actual
`tractcloud.inference.load_model` plus a finite-output synthetic forward pass.
Functional failures persist across restarts for the matching environment
configuration and block analysis until preparation passes again. Rerun Prepare
and inspect its receipts rather than bypassing the persistent failure gate.
These checks do not process patient data, install missing packages or certify scientific
accuracy. Activation Self Tests, dependency reports and these functional tests
are distinct evidence.

## Platform And Delivery Limits

- Linux x86_64/Slicer 5.10 is the verified reference, not an all-platform
  zero-configuration guarantee. MRtrix3 3.0.8, FSL and managed PNL have scoped
  Linux routes; CUDA still requires a usable existing NVIDIA driver/device.
- Optional cortical postprocessing is separate from neural surface inference.
  `shared.prebuilt_surface.bundle_for_platform()` reads
  `models/runtime_bundles.json`; its Linux x86_64 entry currently has
  `published=false`. Only offline prebuilt installation was tested. A listed
  URL is not an available published download: supply the matching qualified
  archive under the confirmed asset root at
  `runtime/surface/surface-runtime-linux-x86_64.tar.gz`, or configure an existing
  compatible runtime. `shared.prebuilt_surface.prepare(...)` verifies the
  recipe, hashes and commands; no personal activation key is bundled.
  User-provided activation and explicit component/data terms approval remain
  necessary. The surface-only GUI preparation discovers existing licenses or offers
  official registration and import. Save only the selected path; never print,
  upload or bundle license contents, register on the user's behalf, or reuse the
  maintainer's key. An invalid explicit path must be corrected, not bypassed.
  See `docs/setup/SURFACE_COMPONENTS.md`. The GUI never compiles;
  `scripts/setup_surface_runtime.py` defaults to prebuilt, with
  `--build-from-source` an expert opt-in, not an automatic failure fallback.
- Mac Intel MRtrix requires macOS **14.5+**. The package/installer adaptation has
  plan tests, not real-Mac execution or native scientific acceptance. No
  qualified managed ARM/FSL/PNL profiles or surface bundle are supplied for
  macOS; do not silently switch Apple Silicon to Intel/Rosetta. Plugin MPS is
  not implemented/validated.
- Windows' existing WSL installer/bridges are a separate path, not native Windows
  acceptance or the same unified Linux managed installer. Read
  `docs/setup/WINDOWS.md` before proposing it; it has broader downloads, storage
  and authorization needs. Do not launch it implicitly to repair a missing item.

## Retry, Offline Import And Handoff

Use **Environment and installation > More options** for offline resources,
repair and manual paths, not Advanced algorithm settings. **Use offline model
resources** verifies selected profiles against trusted manifests without
downloading/executing the resources; preserve the documented directory layout.
It is not portable Conda import or permission to bypass protected downloads.

Cancellation/failure preserves configured paths but can leave installed packages
or partial downloads for retry. Retry ordinary preparation after an interruption;
use **Repair selected managed tools** only with explicit approval for a damaged
plugin-owned runtime. It refuses unowned prefixes and does not roll back all
installed packages. Keep receipts/logs and fix the specific failed phase rather
than repeatedly reinstalling or replacing algorithms/environments.

Report revision/local changes, Slicer version, staged path, activation receipt
and persistence, confirmed asset root, selected targets/profiles, setup receipt
and functional results, plus remaining restart/manual/platform blockers. If
real-data QC or runtime readiness was not checked, say so. Only an explicitly
authorized analysis proceeds to [usage](usage.md) and its doctor/plan/execute
workflow, retaining the user's scientific selections.
