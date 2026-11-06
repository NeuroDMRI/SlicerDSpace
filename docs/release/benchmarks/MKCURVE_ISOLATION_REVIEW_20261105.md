# MK-Curve slowdown and cohort timing isolation

## Conclusions

The CPU UKF/WMA comparison has not established an accepted cohort speedup.
The completed Adaptive arm took 584.07 minutes and its independent automatic
QC passed for all 12 participants, but the reference arm is still running.
More importantly, inspection of the live reference processes found an incorrect
CPU placement that invalidates a matched-resource timing comparison.

The apparent MK-Curve regression is also contaminated by this overlap. It
cannot currently be attributed to an intrinsic scheduler or numerical-kernel
slowdown. Scientific outputs, elapsed-time records and failed measurements are
retained; moving a running process does not make its complete timing fair.

## Confirmed Causes and Scope

| Finding | Evidence | Interpretation |
| --- | --- | --- |
| Reference CPU-placement guard selected the wrong CPU range | CPU v4 requests CPUs 32-55, but its Subject-level guard records CPUs 0-23. Actual descendant-thread affinity confirmed use of those CPUs. MK-Curve uses CPUs 0-31. | Unintended cross-campaign contention affects both comparisons. This is a benchmark isolation defect, not evidence of faster UKF/WMA. |
| Fixed MK-Curve arm has not enabled the tuned worker configuration | Current production settings are reference solver, one worker, 64-voxel chunks, maximum b-value 2500 and four available numerical threads. `adaptive_tuned` is still pending. | Scheduling-only and worker-tuning effects must not be conflated. |
| Four numerical threads were inefficient in an earlier sample | With 1024 real signals, earlier medians were 40.78 s / 40.66 CPU-s for one worker x one thread, versus 41.73 s / 161.98 CPU-s for one worker x four threads. Four single-thread workers took 10.96 s / 37.57 CPU-s, at the same 64-voxel chunk size. | This motivates a bounded worker candidate. The baseline and fixed Adaptive arms share the inefficient setting, so it alone does not explain their difference. |
| Calibration selected a different acquisition subset | The old tuner hardcoded a maximum b-value of 3000; production `DkiConfig` uses 2500. For this acquisition the prepared measurement counts are 271 and 181, respectively, after averaging baseline volumes. | The old sample timings are not a matched-protocol production benchmark. Revalidate rather than reuse them as full-volume evidence. |

The earlier sample contained three real acquisitions and three measured repeats
per setting. These values describe sampled signals and include neither a
full-volume wall-clock comparison nor independently measured cohort throughput.

For the first four completed MK-Curve participants, retained coefficient and
curve fingerprints agree between reference and fixed Adaptive. Their per-module
times vary in both directions (reference -> Adaptive: 361.9 -> 332.2,
298.3 -> 365.9, 256.4 -> 576.5 and 330.6 -> 545.9 minutes). This pattern is
consistent with changing contention, but does not isolate its causal share.
Queue admission waits were about 0.15 seconds, and sampled monitoring overhead
was approximately 0.07-0.09 CPU core per process; neither explains the observed
multi-hour difference. Twelve OS threads per process do not mean twelve busy
numerical threads: live sampling identified approximately four active threads.

## Implemented Repairs

- Guards now receive the explicit campaign CPU allocation instead of selecting
  the first host CPUs in the broad-affinity driver process.
- `taskset` applies the allocation before launcher execution, including new
  threads and descendants. Stage placement may narrow, but not expand, the
  inherited allocation.
- Timed benchmark execution checks live descendant-thread affinity every ten
  seconds and stops its own trial if the allocation is escaped.
- Comparison acceptance checks the recorded guard allocation, observed escapes
  and explicit timing-condition records. Scientific equivalence and QC alone
  cannot authorise a speed ratio for an invalid timing comparison.
- Calibration preparation and probes now share the production DKI/MK-Curve
  numerical protocol. The protocol participates in the tuning key; old profiles
  are not silently reused under the corrected implementation.

No solver, gradient table, synthetic-b0 grid, numerical threshold or scientific
QC acceptance criterion is changed. Worker counts remain a separately measured
runtime choice, not a claim that a globally optimal setting has been found.

## Treatment of Running Experiments

At 23:03 Beijing time on 5 November, the two remaining CPU reference worker
trees were moved to their originally requested CPU ranges. Their scientific
computations were not restarted. A journal records process identities and every
changed thread mask; explicit `performance_valid: false` markers accompany the
affected CPU reference and MK-Curve Adaptive trials. Original frozen source,
scientific products and raw elapsed times are untouched.

The short post-move observation is diagnostic only: not all MK-Curve processes
sped up, and other shared-host jobs remain active. Removing our known overlap
is necessary but is not a clean before/after performance experiment.

The waiting Hybrid v8 controller and old acceptance watchers were superseded
before starting new scientific work. Hybrid v9 has a fresh frozen installation
with the corrected harness, and retains the same explicit CPUs 56-71, 16-CPU
budget, 128 GiB budget and distinct hybrid workload. No other user's CPU or GPU
jobs are stopped.

Hybrid v9 dependency and task-plan preflight passed with zero blockers. Its
controller is waiting for an idle GPU; frozen-code validators are waiting for
v9 and CPU v4 to finish. At the final check, CPU v4 reference had 11 completed
participants and one running, while fixed MK-Curve Adaptive had five completed,
four running and three not yet launched.

Evidence locations:

- `ignore/validation/cohort_affinity_20261105/live_affinity_repair.json`
- `ignore/validation/cohort_affinity_20261105/waiting_controllers_superseded.json`
- `/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261104-cpu-v4/`
- `/data/chengjin/SlicerDDSurfer-test/cohort-latest-20261103-r2/mkcurve-latest/`
- `/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261105-hybrid-v9/`

## Validation and Remaining Evidence

The final full software suite passed 1,892 tests, with nine skips (175.415 seconds).
An additional targeted run in the official Slicer interpreter passed 25 tests,
including real pre-exec affinity inheritance, refusal to widen CPU placement,
invalid-timing acceptance and production-shell selection in the actual sample
preparer. `git diff --check` passed.

A bounded production-protocol real-signal check is recorded under
`ignore/validation/cohort_affinity_20261105/mkcurve_b2500/`. It compares one
worker x four threads, one worker x one thread and four single-thread workers,
all using 64-voxel chunks and the unchanged reference solver. One warm-up and
three rotating measured repeats are retained per candidate. Its full-volume
grid is computed from the original acquisition, not from a cropped brain.
All twelve runs (three warm-ups and nine measured trials) completed with exact
scientific fingerprints. Measured medians, excluding warm-ups, are:

| Runtime configuration | Sample wall-clock seconds | Sampled process-tree CPU-seconds |
| --- | ---: | ---: |
| One worker, four numerical threads | 22.188 | 79.46 |
| One worker, one numerical thread | 20.172 | 20.10 |
| Four workers, one numerical thread each | 11.051 | 36.27 |

The first and third candidates use the same four-CPU allocation. Their sampled
latency differs by 50.19%; this is not a full-volume or cohort gain. Input
preparation is recorded separately and is not included in these fit latencies.
All candidates use the complete 200-point synthetic-b0 grid and 181 prepared
measurements at the production cutoff. The single-worker/single-thread
candidate is restricted to one CPU, so its row is a CPU-efficiency diagnostic,
not a matched-four-CPU scheduler comparison.

The probe driver's summary initially used the wrong CPU-time dictionary key
after all twelve numerical runs had already completed. A separate aggregator
verified the retained result files and source hashes, repaired the summary
without rerunning computations, and retained that repair in the evidence.
Full trial records and fingerprints are available in
[the probe evidence](mkcurve_protocol_probe_20261105.json). A corrected,
implementation-specific tuning profile is retained alongside the local probe;
it is not injected into an already running frozen campaign.

A clean full-volume paired measurement is still required before changing the
paper, website or any accepted speedup claim. The current long-running MK-Curve
outputs can support correctness checks, but their contaminated elapsed times
must remain excluded from a matched-isolation performance conclusion.
