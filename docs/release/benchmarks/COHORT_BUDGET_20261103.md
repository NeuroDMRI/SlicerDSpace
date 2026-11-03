# Budget-Only Cohort Experiment

## Motivation

The previous matched DTI/DKI experiment constrained all arms to four active
participants. Its adaptive arm reserved at most 16 of 32 CPU threads and showed
no throughput gain over subject-level execution. Removing that experimental
ceiling is a new experiment, not a reinterpretation of the earlier result.

Automatic planning previously also contained an eight-participant ceiling.
The updated planner derives the frontier from cohort size, CPU budget and
0.5-GiB interpreter headroom, while preserving explicit user overrides and
per-stage CPU/RAM admission. The first-fit planner's automatic conservative
capacity is also no longer truncated at eight.

## Frozen Experiment

- Output: `/data/chengjin/SlicerDDSurfer-test/cohort-budget-20261103-v1`
- Twelve distinct HCP participants, complete native input grids.
- Forty CPU threads on logical CPUs 32-71, all distinct physical cores; the
  existing experiment retains CPUs 0-31. RAM budget: 192 GiB.
- No GPU-heavy anatomical inference: both GPUs were occupied by unrelated work.
- One frozen implementation for Subject, first-fit Stage and Adaptive arms.
- No manually selected `jobs`: Subject derives capacity from whole-subject
  reservations; Stage and Adaptive omit `--jobs` and plan their own frontier.
- Two independent microstructure model workers per participant in every arm,
  with identical numerical arguments. Whole-subject reservations account for
  concurrent model CPU and RAM rather than assuming only one fitting model.
- Fresh outputs per arm; no changed scientific fitting definitions or grids.

Routes, in order:

1. DTI + Free-water + AMICO-NODDI.
2. DTI + Free-water + AMICO-NODDI + WLS DKI.
3. DTI + two-tensor UKF + WMA.
4. DTI + MK-Curve, retaining the original full calculation grid.

The first route runs Adaptive, Subject and Stage; the next reverses the primary
order. These are single-repeat paired pilots, not confidence intervals or an
isolated estimate of the backfilling algorithm. Automatic subject concurrency,
model concurrency, observed resource use and waiting time must be reported
alongside total wall time. Numerical arrays, affine/dtype, streamline ordering
and attributes are compared with the subject-level arm; independent output QC
is also required. Stochastic geometry may fail exact comparison and must not
be silently treated as equivalent.

The coordinator writes `campaign.json`, per-arm logs, resource time series,
independent QC and scientific fingerprints. No completed results are claimed
by this preparation record. Original experiments and frozen installations are
left untouched and continue independently.
