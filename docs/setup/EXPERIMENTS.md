<div align="center">

# Lightweight Setup And Experiments

[Home](../../README.md) | [Quick start](QUICKSTART.md) | [简体中文](EXPERIMENTS.zh-CN.md)

</div>

---

## Fewer External Environments

Slicer's Python and installed extensions provide the main runtime. Compatible
Python packages are prepared there on demand, without a separate Conda
environment per algorithm. Supplied valid masks are reused.

Open **Environment and installation > Use Slicer-native route...** to review
and explicitly apply the lighter preset:

| Part | Method | Standalone environment |
| --- | --- | --- |
| Brain mask | PNL CNN (the only generating method) | PyTorch extension and the pnl-cnn assets |
| Optional preprocessing | Not part of this route (one route only: DIPY MP-PCA and Gibbs, then FSL eddy/topup) | MRtrix3, FSL and the FSL licence |
| DTI / FW / NODDI / Python DKI | Selected models in Slicer's Python | None; model-specific Python packages as needed |
| Anatomical prior | DDSeg / DDParcel / DDSurfer | PyTorch extension, Python packages and downloaded assets |
| Tracking | DTI deterministic | Compatible Python packages (the tensor fit runs in Slicer) |
| Fiber bundles | WMQL | Python implementation and queries |
| Connectome | Python | Compatible Python packages |
| Optional surface postprocessing | Licensed native components | Small component runtime; no full FreeSurfer |

PNL and CSD/ACT defaults stay unchanged until explicitly changed. PNL needs
PyTorch and the pnl-cnn assets (ANTs only on the managed route); CSD/ACT and SAF need MRtrix; preprocessing needs
MRtrix3 and FSL. UKF uses its installed extension. GPU-capable stages retain their device
preference. Optional methods remain available on demand.

The preset sets the brain mask to PNL CNN. The DIPY median-Otsu mask was removed (W5); a supplied
mask (`provided`) stays available. Inspect brain boundaries, brainstem and cerebellum. See the
[Slicer Python documentation](https://slicer.readthedocs.io/en/latest/developer_guide/python_faq.html).

---

## Advanced Experiments

1. Select products and algorithms on the main screen.
2. Open **Advanced settings > Experiments & batch**, then **Enable experiments**.
3. Choose **Current subject** or **Prepared subject folders** with a cohort root.
   Each cohort child folder contains prepared subject inputs. Import DICOM/NRRD
   first. Empty paths and duplicate subject IDs are rejected.
4. Set concurrent subjects, initially 1. Each subject runs its variants
   sequentially; different subjects may run concurrently and share GPU memory.
5. Enter optional CLI overrides and named variants. Precedence is GUI settings,
   then common overrides, then the variant's arguments.
6. Use **Preview experiments**, then **Run SlicerDSpace** and confirm the job
   count/output path. Preview validates parameters and paths; each job checks
   its own dependencies before execution.

Each campaign accepts up to 64 variants and 4,096 jobs. Larger studies can be
split into campaigns. Preview displays the first 50 jobs; the exported request
and CLI plan retain all jobs.

With Microstructure selected, compare two DTI estimators:

```json
[
  {"name": "dti_wls", "args": ["--micro-fit-method", "wls"]},
  {"name": "dti_ls", "args": ["--micro-fit-method", "ls"]}
]
```

With Fiber bundles selected, compare the current route with UKF feeding WMA:

```json
[
  {"name": "baseline", "args": []},
  {"name": "ukf_wma", "args": [
    "--tractography-backend", "ukf",
    "--bundle-backend", "wma",
    "--bundle-tractography-backend", "ukf"
  ]}
]
```

The second example requires UKF/WMA dependencies and the usual scientific QC.
Common overrides can be `--mrtrix-seed 42 --mrtrix-nthreads 8`.
`--ukf-parameters` accepts the CLI's validated JSON. `--max-parallel-models`
controls within-subject model concurrency separately; combining concurrency
levels increases memory demand.

Environment paths stay under **Environment and installation**. Parameter
overrides do not accept shell commands, installation actions, identity/output
paths or options that bypass dependency checks. Analysis parameters use the
same parser and validation as the subject CLI.

---

## Outputs And Reuse

```text
<output>/experiments/<campaign>/
  00_batch_metadata/
    experiments.json
    logs/<job>.stdout.log
    logs/<job>.stderr.log
  subject_0001/
    dti_wls/<normal product hierarchy>
    dti_ls/<normal product hierarchy>
  subject_0002/...
```

The request JSON is retained beside the campaign. The live record stores
effective settings, complete command arrays, status, duration, logs and child
run paths. Inputs, installed tools and eligible subject caches are reused;
variant outputs are isolated. An existing campaign is never overwritten.
Cancel stops the owned background process tree and preserves completed results.
Failed and unlaunched jobs remain recorded.

**Results and QC** lists campaign and individual runs. One completed run loads
automatically; select other runs for review rather than filling the scene with
every variant. Automatic QC and human review remain separate.

**Export request...** saves a JSON request for the same CLI runner:

```bash
/path/to/Slicer --launch PythonSlicer scripts/run_experiments.py \
  --request /path/to/request.json
```

Omit `--execute` to print a validated plan without creating outputs. Add it to
execute. Use a fresh campaign/output path for another execution. Environment
installation remains an explicit preparation step.
