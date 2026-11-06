"""Measured defaults for the adaptive cohort scheduler.

Every value in this module comes from the M1 measurement run (one subject, four
1-subject adaptive cohorts at 16, 8, 4 and 2 CPUs per stage). The run folder is
``/data04/chengjin/r5_cohort/m1_bcdbc95``. Until M1 fills the tables below, the code
keeps its previous behaviour: CPU requests equal today's built-in values, the RAM
header heuristic is not replaced, and stage durations are unknown (0).

The planner reads these tables at call time, so tests can replace them.

``STAGE_CPU_REQUESTS``
    stage -> CPU request in adaptive mode (S1.1). A stage that is not listed keeps
    the built-in request.
``RAM_FIT``
    stage -> ``(P_gib, signal_ref)`` (S2). P_gib is the maximum peak RSS of the
    stage over all M1 probes. The RAM estimate is
    ``1 + 1.25 * P_gib * (signal / signal_ref)`` GiB, rounded up to 0.25 GiB.
    A stage that is not listed keeps the header heuristic.
``DURATION_DEFAULTS``
    stage -> ``{'signal_ref': float, 'probes': {probed_cpus: (t_p_seconds,
    c_p_mean_cores)}}`` (S3). t_p is the summed wall time of the stage's
    non-reused invocations in the probe with ``probed_cpus`` CPUs; c_p is its mean
    core use. A stage that is not listed has an unknown duration (0).

In all three tables, ``signal`` is the planner's ``voxels * volumes * 8 / 1024**3``.
"""

# Today's built-in adaptive request for these stages is min(4, CPU budget).
# The planner caps every value here at the CPU budget.
STAGE_CPU_REQUESTS = {
    'native_dti': 4,
    'registered_dti': 4,
    'segmentation': 4,
    'parcellation': 4,
    'surface': 4,
    'microstructure_noddi': 4,
}

RAM_FIT = {}

DURATION_DEFAULTS = {}
