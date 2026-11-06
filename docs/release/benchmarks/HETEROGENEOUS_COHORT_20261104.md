# Heterogeneous cohort scheduling experiment

## Workloads

Each campaign uses the same 12 distinct, full-volume HCP acquisitions recorded
in its `campaign.json`. Inputs are already corrected DWI, matching gradients and
brain masks; acquisition preprocessing is not repeated or included in timing.

| Campaign | Selected computations | Per-arm budget | Execution order |
| --- | --- | --- | --- |
| CPU | DTI, two-tensor UKF tractography, WMA with the ORG atlas | 24 CPU threads on CPUs 32-55; 160 GiB RAM; no CUDA | Adaptive, Subject-level |
| Hybrid | DDSeg, DDParcel, **DDSurfer**, DTI, Free-water WLS, NODDI, CSD/ACT with SIFT2, WMA with the ORG atlas, MRtrix connectome | 16 CPU threads on CPUs 56-71; 128 GiB RAM; one admitted GPU stage | Subject-level, Adaptive |

DDSurfer includes WM and pial mesh inference; optional surface postprocessing
is not selected. CSD/ACT requests one million streamlines and fixes the MRtrix
seed at 20261104. The hybrid WMA input is the CSD/ACT tractogram; the CPU
campaign classifies UKF streamlines. The two workloads are separate comparisons.

## Comparison contract

- Reuse the existing `benchmark_cohort_workflows.py` runner, resource guards,
  scientific audit and process-tree telemetry rather than another scheduler.
- Freeze a copied installation and its source hashes before each campaign.
  Retain input, launcher, model, atlas and backend checksums in the record.
- Use fresh output directories, the same selected computations and the same
  four-thread task configuration in both arms. Internal independent-model
  concurrency is one in both arms, isolating cross-subject scheduling.
- Do not supply an arbitrary four-subject limit. Subject-level derives its
  concurrency from whole-subject resource estimates; Adaptive admits ready
  tasks across the cohort within its CPU/RAM/I/O/GPU reservations. Record the
  resulting concurrency alongside throughput.
- Measure complete batch wall time, process-tree CPU/RAM/I/O, owned VRAM,
  task timings and reservation wait times. Whole-device GPU utilisation is
  separately labelled context and can include other users' work.
- Audit all 12 subject outputs and compare exact scientific fingerprints,
  including arrays/dtypes/affines, ordered streamlines and weights, mesh
  geometry/topology/attributes and connectome matrix/node tables. A successful exit
  alone does not accept the comparison. Differences retain failed/pending
  acceptance status and are investigated before any speedup is reported.
- These are single paired runs on a shared host. Arm order differs between
  workloads, but does not replace repeated, order-balanced measurements.

## Execution Records

Entry point: `scripts/benchmark_heterogeneous_cohort.py prepare|run`.

Active CPU record:
`/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261104-cpu-v4/`

Original hybrid record (retained with failed sampling QC):
`/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261104-hybrid-v7/`

The new 5 November hybrid comparison has a distinct workload scope without
WMA; UKF/WMA remains in the CPU pair. Its record is
`/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261105-hybrid-v8/`.
See [repair and workload revision](COHORT_RECOVERY_20261105.md) for the
diagnosis, unchanged QC thresholds and frozen validation follow-up.

Each directory contains `campaign.json`, `controller.json`, preflight reports,
controller/driver logs, a frozen installation and separate trial directories.
After both arms finish, `primary_comparison.json` records the accepted timing
comparison and `comparisons/` contains per-subject product comparisons.

At launch on 4 November, CPU dependency checks passed and Adaptive began
computation. All hybrid dependency checks also passed. Both H100 devices were
occupied, so the hybrid controller waits for three consecutive 30-second
checks with utilisation at most 5%, used VRAM at most 1024 MiB and at least
30000 MiB free. Device discovery during preflight does not run inference.
No existing GPU job is terminated and no busy device is treated as idle merely
because it has free VRAM. The existing MK-Curve experiment on CPUs 0-31 remains
unchanged. CPU sets are disjoint across these experiments, although the host,
memory bandwidth and storage remain shared.

The first CPU setup attempt used an incorrect installed CLI path. An early
hybrid setup attempt hid CUDA during device discovery. A subsequent queue was
superseded before execution to include node-table fingerprints. Their records
remain intact; no scientific trial ran in those attempts. They are not timing
samples.

### Corrected Input Preparation And Route Selection

The CPU v2 run finished its DTI computations but UKF rejected isolated masked
voxels with zero mean b0, blocking all WMA tasks. The hybrid v4 run completed
some anatomical/model products but omitted the explicit CSD-to-WMA source
selection. It was intentionally cancelled through its own driver; partial
products and telemetry remain available, but neither run is a valid performance
pair for the intended workload. MK-Curve was not interrupted.

Both corrected routes explicitly pass `--bundle-tractography-backend`. Preflight
now checks the actual task plan, rejects any blocked or missing target, and
validates the full UKF input bundle for every subject before timing begins.
Dependency-probe timeouts receive at most two retries, with each report retained;
missing dependencies and persistent timeouts still stop the campaign.

The UKF adapter retains the original DWI/mask and writes a separate positive-b0
tracking mask. Exclusions are recorded in native space and included in QC and
output fingerprints. All 12 acquisitions passed the revised input checks;
exclusions range from 18 to 1269 voxels in the completed input report, with a
maximum excluded fraction of approximately 0.159%. The adapter rejects an empty
effective mask or an exclusion above the conservative 1% input-integrity guard.
The mask policy and generated artifact are covered by cache identity/integrity.
This is input handling, not a change to UKF scientific parameters or a relaxation
of tractogram QC. Both arms use the same corrected implementation and fresh
outputs. The earlier timings are not spliced into these comparisons.

Hybrid v5 stopped during a transient PyTorch dependency-probe timeout before
any scientific trial started; v6 adds the bounded diagnostic retries above.

### Output Validation Follow-Up

The original independent auditor recognises DTI/CSD tracking but not UKF. The
corrected auditor also checks UKF TRK geometry, counts, automatic QC and the
native input/backend/artifact receipt, including the new effective mask.

The already frozen CPU v3 run is kept intact, rather than repeating its
scientific computations for this validation-only correction. A separately
launched `verify` action waits for both timed arms to finish, runs the corrected
auditor and recomputes scientific fingerprints against the recorded products.
It keeps the original campaign, measurements and failed/partial audit records
unchanged. Its validation-code hashes and reports are saved under
`post_validation/`; the resulting `primary_comparison.json` in that directory
is accepted only if fresh QC, exact product comparison and input integrity all
pass. Incomplete scientific execution or changed products cannot pass this
follow-up. A verification watcher is also attached to the hybrid campaign.

The actual official CLI was exercised by
`Testing/Python/ukf_valid_mask_smoke.py`: a synthetic tensor field with two zero
b0 mask voxels produced 1726 streamlines; effective-mask handling, tensor/FA
arrays, VTK/TRK coordinates, cache integrity and unchanged source files passed.
The receipt is `ukf_valid_mask_20261104.json`. This tests the input adapter and
official executable; it is not a full-brain performance result.

No new cohort speedup or scientific acceptance is claimed before the paired
execution and comparison records are complete.

### Headless WMA Back-Transformation Fix

The CPU v3 run reaches WMA registration, clustering and outlier removal, but
the first native-space back-transformation fails. Its retained Slicer log says
`qt.qpa.xcb: could not connect to display`. This is a missing display context,
not an input-gradient error or a failure to produce the 800 atlas clusters.
Slicer's `--no-main-window` hides its window but does not remove its Linux X
display requirement. The old driver reports the downstream output-count error;
the absent products remain a failure, even though the author shell returns zero.

The adapter now selects `xvfb-run -a` automatically when Linux has neither
`DISPLAY` nor `SLICERDSPACE_DISPLAY`. Both route validation and the environment
doctor require `xvfb-run`, `Xvfb` and `xauth` before computation. Existing displays
are reused, and macOS/native Windows do not automatically acquire Linux display
dependencies. The run manifest distinguishes requested from selected virtual-X
mode. Registration, clustering, transforms and output QC criteria are unchanged.

`Testing/Python/wma_headless_transform_smoke.py` exercises the archived real
subject's 800 clusters and saved affine using the actual Slicer/WMA programs.
It writes recovery products to a separate directory, checks source hashes,
streamline connectivity and native-space coordinates against the ITK affine,
then checks bundle outputs with the existing WMA contract. ITK files use LPS
and the from-parent convention; WMA inverts the loaded Slicer node, so the
independent point check uses the file's forward affine converted to RAS.
See [Slicer transform conventions](https://slicer.readthedocs.io/en/5.8/developer_guide/modules/transforms.html).
The recovery directory is
`/data/chengjin/SlicerDDSurfer-test/wma-headless-recovery-20261104/`.
Reused registration/clustering work makes this a recovery/correctness check,
not a full WMA timing observation.

The [recovery receipt](wma_headless_recovery_20261104.json) records 800/800
transformed clusters and 74 readable, nonempty anatomical bundles. All cluster
connectivity arrays and native coordinates pass the independent check, with
maximum coordinate error about 0.00000382 mm; all 801 archived cluster/transform
files retain their hashes. The existing major-tract output contract passes;
per-tract sampling and human anatomical review keep their separate status.
The full software suite runs 1880 tests with nine skips and no failures.

CPU v3 retains its frozen implementation and is not eligible for a successful
paired speedup. Its remaining expensive UKF work is not interrupted. A scoped
cancellation watcher stops only its obsolete WMA author-shell process trees,
recording each event under `obsolete_wma_cancellation.json`, so known-invalid
WMA calls do not occupy resources until the display failure recurs. Corrected CPU v4 uses a
new frozen installation and fresh outputs, queued behind v3 so their CPU
budgets do not overlap. Hybrid v6 is superseded before scientific execution;
hybrid v7 passes preflight with the corrected installation and waits for an
idle GPU. Both corrected campaigns have independent post-validation watchers.
No failed or recovered timing is spliced into their complete-route comparisons.
