<div align="center">

# Resource-Aware Cohort Processing

[Documentation](../README.md) | [CLI templates](CLI_TEMPLATES.md) | [中文](COHORT_RESOURCES.zh-CN.md)

</div>

---

## Run A Cohort

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
`--jobs` to choose a conservative subject limit automatically (up to eight).
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
report generation are not separately reserved. A grouped microstructure call keeps
its reservation until it returns; this is not per-voxel or iteration-level batching.

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
provides surface/MRtrix/eddy defaults and bounds automatic NODDI threads. Explicit
thread options above the reservation are rejected. Within-subject model concurrency
is set to one; Free-water parallel workers default to one. To change these thread
settings, establish a new reproducibility baseline where appropriate.

Reservations are admission estimates, not OS-enforced CPU or RAM caps. Third-party
tools can use additional helper threads or memory. GPU slots conservatively count
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
