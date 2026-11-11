"""Measured defaults for the adaptive cohort scheduler.

Every value in this module comes from the M1 measurement run (one subject, four
1-subject adaptive cohorts at 16, 8, 4 and 2 CPUs per stage). The run folder is
``/data04/chengjin/r5_cohort/m1_bcdbc95`` (code bcdbc95, fan93, HCP 100307, finished
2026-11-07; all four probes exit 0). The values were copied from its summary
``m1_bcdbc95_summary.json`` (``m1_summary.py``): wall times (s), mean cores and peak
RSS (GiB) rounded to three decimals (relative change at most 0.06 %).

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

# DWI signal size of the M1 subject, voxels * volumes * 8 / 1024**3 GiB
# (HCP 100307: 145 x 174 x 145 x 288).
SIGNAL_REF = 7.8499674797058105

# S1.1 M1 rule: the largest candidate whose parallel efficiency against the smallest
# candidate is >= 0.75. The planner caps every value at the CPU budget.
STAGE_CPU_REQUESTS = {
    'native_dti': 16,  # efficiency 4: 1.000, 8: 1.000, 16: 0.935
    'registered_dti': 2,  # efficiency 2: 1.000, 4: 0.700, 8: 0.346
    'segmentation': 2,  # efficiency 2: 1.000, 4: 0.474
    'parcellation': 2,  # efficiency 2: 1.000, 4: 0.610
    'surface': 2,  # efficiency 2: 1.000, 4: 0.523
    'microstructure_noddi': 8,  # efficiency 4: 1.000, 8: 0.966
}

# S2: (maximum peak RSS over the four probes in GiB, signal_ref).
RAM_FIT = {
    'connectome': (2.608, SIGNAL_REF),
    'csd_act': (10.583, SIGNAL_REF),
    'microstructure_dki': (5.944, SIGNAL_REF),
    'microstructure_dti': (2.366, SIGNAL_REF),
    'microstructure_freewater': (2.367, SIGNAL_REF),
    'microstructure_noddi': (17.220, SIGNAL_REF),
    'native_dti': (9.749, SIGNAL_REF),
    'parcellation': (10.824, SIGNAL_REF),
    'registered_dti': (2.270, SIGNAL_REF),
    'segmentation': (2.785, SIGNAL_REF),
    'surface': (3.633, SIGNAL_REF),
}

# S3: probed CPUs -> (summed non-reused wall seconds, mean cores).
DURATION_DEFAULTS = {
    'connectome': {'signal_ref': SIGNAL_REF,
        'probes': {2: (17.539, 1.598), 4: (15.825, 1.751), 8: (16.650, 1.999), 16: (15.652, 2.041)}},
    'csd_act': {'signal_ref': SIGNAL_REF,
        'probes': {2: (14779.576, 1.971), 4: (7629.035, 3.828), 8: (4406.515, 7.235), 16: (2144.138, 12.615)}},
    'microstructure_dki': {'signal_ref': SIGNAL_REF,
        'probes': {2: (38.000, 1.201), 4: (38.000, 1.204), 8: (38.000, 1.212), 16: (38.000, 1.218)}},
    'microstructure_dti': {'signal_ref': SIGNAL_REF,
        'probes': {2: (0.761, 1.116), 4: (0.726, 1.199), 8: (0.795, 1.119), 16: (0.725, 1.187)}},
    'microstructure_freewater': {'signal_ref': SIGNAL_REF,
        'probes': {2: (185.000, 1.184), 4: (185.000, 1.176), 8: (185.000, 1.180), 16: (185.000, 1.180)}},
    'microstructure_noddi': {'signal_ref': SIGNAL_REF,
        'probes': {2: (1615.589, 1.953), 4: (809.103, 3.636), 8: (418.590, 6.583), 16: (594.260, 4.123)}},
    'native_dti': {'signal_ref': SIGNAL_REF,
        'probes': {2: (4920.204, 1.984), 4: (2440.108, 3.816), 8: (1220.131, 7.394), 16: (652.404, 13.600)}},
    'parcellation': {'signal_ref': SIGNAL_REF,
        'probes': {2: (176.776, 1.460), 4: (144.849, 1.694), 8: (138.840, 2.065), 16: (150.342, 2.658)}},
    'registered_dti': {'signal_ref': SIGNAL_REF,
        'probes': {2: (36.826, 1.474), 4: (26.288, 1.735), 8: (26.597, 2.126), 16: (22.737, 2.200)}},
    'segmentation': {'signal_ref': SIGNAL_REF,
        'probes': {2: (14.350, 1.203), 4: (15.140, 1.147), 8: (13.104, 1.216), 16: (17.498, 1.008)}},
    'surface': {'signal_ref': SIGNAL_REF,
        'probes': {2: (77.901, 1.282), 4: (74.543, 1.342), 8: (72.620, 1.493), 16: (77.018, 1.819)}},
}
