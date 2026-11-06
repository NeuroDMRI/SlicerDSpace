<div align="center">

# Resource-Aware Cohort Processing

[Documentation](../README.md) | [CLI templates](CLI_TEMPLATES.md) | [中文](COHORT_RESOURCES.zh-CN.md)

</div>

---

## Run A Cohort

For CPU-first cohort processing, use the adaptive admission mode:

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/study --output-root /path/to/results \
  --scheduler adaptive --resource-budget 80% --subject-limit 20 \
  --targets microstructure --microstructure-stages dti,noddi,dki --execute
```

Omit `--subject-limit` to process every eligible input. Omit `--jobs` to choose
an active-subject window independently of the heaviest stage; waiting processes
do not reserve an entire subject's compute budget. The window is derived from
cohort size, the CPU budget and estimated interpreter RAM headroom, without a
fixed four/eight-subject ceiling. Explicit CPU/RAM budgets override
the automatic fraction. This is a reservation policy, not a system utilisation cap.

Adaptive mode reserves up to four threads per typical stage initially, one for
the default DTI-export/serial-Free-water/DKI/MK-Curve/WMA/WMQL/connectome routes,
and honours explicit thread overrides.
These are conservative admission defaults, not measured optimal thread counts.
Dependencies remain enforced by each subject's existing orchestrator. Microstructure
models acquire individual reservations; adaptive mode supports bounded independent
model workers with `--max-parallel-models 2` or `3`. The default remains one model.

Backfilling protects the oldest waiting stage's estimated start. The estimate
comes from runtime feedback, else from the measured defaults in
`shared/cohort_defaults.py`; a stage with neither has an unknown duration. A later
stage starts early only if the protected start does not move later. There is no
ageing barrier: the reservation is recomputed on every poll. Long-running stages are
not preempted, and a stage that overruns its estimate never releases its reservation.
Registration, preprocessing, WMQL and connectome stages also share two I/O admission slots. Reservations cover whole
wrapped stages, rather than instantaneous disk bandwidth.
DWI decompression has a separate two-permit FIFO gate, released before fitting.
An overrun runtime estimate never releases a still-running task's reservation.

Every adaptive stage records sampled process-tree RSS, CPU time, I/O and execution
time in `schedule_<id>/events.jsonl`. The batch manifest summarises reservations and
measurements separately. `00_batch_metadata/resource_history.json` retains at most
12 records per workload key and 512 keys. `--resource-history` selects another file.
Keys include hardware, Python, implementation hashes, numerical arguments, image
dimensions and threads. Successful uncached observations can raise the heuristic RAM
reservation with a 30% margin and estimate duration with a 25% margin. They never
silently lower the initial RAM estimate or retry a failed scientific computation.
Sampled RSS sums can double-count shared pages and miss between-sample peaks.

The paired pilot in `docs/release/benchmarks/cohort_cpu_20261103.json` uses three
real-data ROIs and WLS tensor fitting. It verifies numerical outputs and records
scheduling overhead; it is not a full-workflow or large-cohort speedup estimate.

## Compatibility Mode

The existing subject scheduler remains the default. `--scheduler resources` adds
shared admission at compute-stage boundaries across active subject processes.
It preserves each subject's dependencies, manifests, cached outputs and QC.

```bash
"$SLICER" --launch PythonSlicer "$REPO/scripts/run_batch_pipeline.py" \
  --batch-root /path/to/study --output-root /path/to/results \
  --scheduler resources --jobs 3 \
  --targets segmentation,parcellation,surface,microstructure \
  --microstructure-stages dti,freewater,noddi --execute
```

Use prepared subject directories and the normal dependency configuration. Omit
`--jobs` to choose a conservative subject limit automatically from the budget.
CPU affinity/container limits, available RAM, selected methods and DWI header
dimensions determine the initial plan. No full DWI is loaded for planning.
Review `resource_plan` in the batch manifest and check a pilot's peak usage. No new service,
container or inference framework is required. `psutil` is required in the workflow
Python for process identity and stale-reservation cleanup.

## Admission Rules

- `--jobs` optionally limits active subject processes, including processes waiting for resources; it is automatic when omitted in resource mode. The subject scheduler still defaults to one.
- Total declared CPU and RAM reservations must fit `--batch-cpus` and `--batch-memory-gb`.
- Stage estimates vary by method and subject image dimensions. `--batch-stage-cpus` and `--batch-stage-memory-gb` optionally replace these estimates with fixed reservations; RAM is in GiB.
- GPU-capable stages share `--batch-gpu-slots`, default one. CPU stages can run while another subject holds that slot. Explicit CPU devices release the slot where the stage exposes its device setting.
- First-fitting queue admission allows a CPU stage to pass a GPU stage waiting for its slot. Dependencies within a subject still determine which stage becomes ready.
- Nested calls and owned descendant workers reuse the parent reservation. An exited owner is detected by PID and process creation time. Exceptions release the reservation.

Covered boundaries include mask generation, shared native/registered DTI,
preprocessing, anatomical inference and postprocessing, microstructure, tracking,
bundle extraction and connectome construction. Validation, environment checks and
report generation are not separately reserved. DTI export, Free-water, NODDI and
DKI/MK-Curve have separate leases; their controller does not retain a parent lease.
This is task-level admission, not per-voxel or iteration-level batching.

## Threads And Devices

Advanced users can override `--batch-cpus`, `--batch-memory-gb`,
`--batch-stage-cpus`, `--batch-stage-memory-gb` and `--batch-gpu-slots`.
Automatic memory estimates use float64 signal size and model-specific workspace
allowances. Missing/unreadable headers use a recorded 160 x 192 x 160 x 128 fallback;
normal input validation still applies. If even one selected stage is estimated to
exceed the budget, preparation stops with a clear message rather than under-reserving
memory or changing the analysis. Explicit method thread requests are retained if
they fit the total budget. Estimates are conservative heuristics, not measured peaks.

The scheduler sets common numerical/ITK/MRtrix thread defaults for child processes,
provides surface/MRtrix/eddy defaults and bounds automatic NODDI threads. In `adaptive`
mode the batch driver adds no plan-wide `--surface-cpu-threads`, `--mrtrix-nthreads` or
`--preprocessing-eddy-threads`: each stage uses its own reservation
(`SLICERDSPACE_STAGE_CPUS`) unless the user gave a value, subject processes start with
one OpenMP/BLAS thread, and an imported `torch` follows the reservation of each
admitted stage. Explicit
thread options above the reservation are rejected. UKF automatic threads use the
stage reservation. The compatibility `resources` mode requires one within-subject
model; `adaptive` also accepts bounded independent model workers. Free-water
parallel workers default to one. To change these thread
settings, establish a new reproducibility baseline where appropriate.

On Linux, stages receive disjoint CPU affinity sets and child programs inherit
them, including native tools that ignore thread variables. Affinity is restored
after completion or failure. RAM remains a declared reservation, not an OS cap.
Third-party tools can create helper threads within their CPU set. GPU slots conservatively count
GPU-capable calls, not measured GPU use or VRAM. This version does not assign GPUs,
enforce per-device VRAM, keep models resident or merge different subjects into one
model batch. It does not change precision, seeds, solver definitions or fitting grids.
Choose the desired devices through the existing analysis options.

## Records And Validation

The batch manifest links `resource_schedule`. Its directory contains `config.json`,
including the per-subject stage estimates and planning inputs,
live `state.json` and append-only `events.jsonl` with stage/subject identifiers,
waiting durations and execution events. Stage completion means the callable returned;
scientific acceptance still comes from the subject's QC. Subject failure stops new
subject launches by default, while active subjects drain. Existing skip/resume and
`--continue-on-error` semantics are retained.

`Testing/Python/test_cohort_schedule.py` checks real subprocess admission, overlap,
resource accounting, nested workers, failure cleanup and deterministic fixture
outputs. `scripts/benchmark_cohort_schedule.py --output /path/to/new_directory`
compares whole-subject and stage reservations with three repeated, alternating-order
synthetic workloads under the same declared budgets. These sleep-based fixtures
test scheduling and do not measure MRI throughput or algorithm speedup. Real-cohort
wall time, peak memory, output consistency and QC still require matched measurements.
