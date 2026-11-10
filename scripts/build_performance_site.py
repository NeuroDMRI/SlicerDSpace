#!/usr/bin/env python3
"""Export explicitly selected, verified measurements without private runtime paths."""

import argparse
from datetime import date
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / 'docs/release/benchmarks'
SITE = ROOT / 'docs/site'
BEGIN, END = '<!-- measured-evidence:start -->', '<!-- measured-evidence:end -->'
REVIEWED_ON = '2026-11-10'


def runtime_optimizations_20261109_records(report):
    """Qualify the three-subject Teem WLS, DEC FA, endpoint-selection and connectome-radial records."""
    environment = report.get('environment', {})
    if (report.get('status') != 'passed' or report.get('source_inputs_unchanged') is not True
            or any(environment.get('thread_env', {}).get(key) != '1'
                   for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))):
        raise ValueError('Runtime optimizations 2026-11-09 report header or environment is incomplete')
    sections = {}
    for name in ('teem_dti_wls', 'teem_dec_fa', 'regional_endpoints', 'connectome_radial'):
        records = (report.get(name) or {}).get('records', [])
        if len(records) != 3 or [row.get('subject_id') for row in records] != ['100206', '100408', '149741']:
            raise ValueError(f'Expected three ordered acquisitions in {name}')
        for row in records:
            if row.get('exact_output_parity') is not True:
                raise ValueError(f'Output equivalence verification failed in {name}')
            if 'times_seconds' in row:
                if row.get('repeats') != 3 or row.get('warmups_per_method') != 1:
                    raise ValueError(f'Unexpected repetition count in {name}')
                for variant in ('reference', 'optimized'):
                    values = row['times_seconds'][variant]
                    if (len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values)
                            or not math.isclose(median(values), row[variant + '_median_seconds'], rel_tol=1e-12)):
                        raise ValueError(f'Median mismatch in {name}')
                before, after = row['reference_median_seconds'], row['optimized_median_seconds']
            else:
                before, after = row['reference_seconds'], row['optimized_seconds']
                if not all(math.isfinite(v) and v > 0 for v in (before, after)):
                    raise ValueError(f'Invalid duration in {name}')
            if (not math.isclose(before / after, row['speed_ratio'], rel_tol=1e-12)
                    or not math.isclose(100 * (1 - after / before), row['time_reduction_percent'], abs_tol=1e-10)):
                raise ValueError(f'Speed ratio or reduction mismatch in {name}')
        sections[name] = records
    return sections


def five_module_subalgorithm_20261110_records(report):
    """Qualify the 10 November 2026 five-module sub-algorithm optimization records."""
    if report.get('status') != 'passed' or report.get('kind') != 'five_module_subalgorithm_optimizations':
        raise ValueError('Five-module sub-algorithm optimization report header is incomplete')
    records = report.get('records', [])
    if len(records) != 11:
        raise ValueError('Expected 11 five-module sub-algorithm records')
    for row in records:
        if row.get('exact_output_parity') is not True or row.get('repeats') != 3 or row.get('warmups_per_method') != 1:
            raise ValueError('Five-module sub-algorithm parity or repetition check failed')
        before, after = row['reference_median_seconds'], row['optimized_median_seconds']
        if not all(math.isfinite(v) and v > 0 for v in (before, after)):
            raise ValueError('Invalid five-module duration')
        if (not math.isclose(before / after, row['speed_ratio'], rel_tol=1e-12)
                or not math.isclose(100 * (1 - after / before), row['time_reduction_percent'], abs_tol=1e-10)):
            raise ValueError('Five-module speed ratio or reduction mismatch')
    return records


def mkcurve_signal_records(report):
    """Qualify all six incremental comparisons, including the non-winning solver."""
    evidence = report.get('mkcurve', {})
    records = evidence.get('records', [])
    environment = evidence.get('environment', {})
    if (report.get('status') != 'passed' or evidence.get('source_inputs_unchanged') is not True
            or environment.get('scipy') != '1.18.1'
            or any(environment.get('thread_env', {}).get(key) != '1'
                   for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
            or len(records) != 6):
        raise ValueError('MK-Curve signal-reuse evidence is incomplete')
    subjects = {row['subject_id'] for row in records}
    if len(subjects) != 3 or any(
            sorted(row['solver'] for row in records if row['subject_id'] == subject) != ['batched', 'reference']
            for subject in subjects):
        raise ValueError('Expected both MK-Curve solvers in each of three acquisitions')
    for row in records:
        if (row.get('exact_output_parity') is not True or row.get('repeats') != 3
                or row.get('warmups_per_method') != 1 or row.get('voxels') != 64
                or row.get('measurements') != 181 or row.get('grid_samples') not in (200, 201)):
            raise ValueError('MK-Curve signal-reuse output or scope verification failed')
        for variant in ('reference', 'optimized'):
            values = row['times_seconds'][variant]
            if (len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values)
                    or not math.isclose(median(values), row[variant + '_median_seconds'], rel_tol=1e-12)):
                raise ValueError('MK-Curve signal-reuse median does not match its repetitions')
        before, after = row['reference_median_seconds'], row['optimized_median_seconds']
        if (not math.isclose(before / after, row['speed_ratio'], rel_tol=1e-12)
                or not math.isclose(100 * (1 - after / before), row['time_reduction_percent'], abs_tol=1e-10)):
            raise ValueError('MK-Curve signal-reuse ratio does not match its measurements')
    return records


def build_evidence():
    sources = []

    def load(name):
        path = BENCHMARKS / name
        sources.append({'file': name, 'sha256': hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()})
        return json.loads(path.read_text(encoding='utf-8'))

    full_name = ('freewater_protocol_full_20261029.json' if (BENCHMARKS / 'freewater_protocol_full_20261029.json').is_file()
                 else 'freewater_latest_stable_full_20261022.json')
    protocol_reuse = full_name == 'freewater_protocol_full_20261029.json'
    full = load(full_name)
    if protocol_reuse and (
        (full.get('fit_implementation') or {}).get('protocol_reuse_enabled') is not True
        or (full.get('reference_implementation') or {}).get('enabled') is not False
        or (full.get('reference_implementation') or {}).get('implementation') != 'dipy_original_dispatch'
    ):
        raise ValueError('Protocol timing must record enabled reuse and an unmodified author reference')
    if not all(full.get(key) is True for key in ('inputs_unchanged', 'snapshot_unchanged', 'harness_unchanged', 'qc_equal')) or full['status'] != 'passed':
        raise ValueError('Full Free-water evidence has not passed its independent verification')
    previous = load('complete_models_20261020.json')
    rows = []

    def add(identifier, title, category, scope, reference, optimized, method, baseline, unchanged, limitation, sample, evidence_date='2026-10-20'):
        if not all(math.isfinite(value) and value > 0 for value in (reference, optimized)):
            raise ValueError('Invalid measured duration')
        rows.append(dict(id=identifier, title=title, category=category, scope=scope, evidence_date=evidence_date,
                         reference_seconds=reference, optimized_seconds=optimized,
                         speed_ratio=reference / optimized, seconds_saved=reference - optimized,
                         reduction_percent=100 * (1 - optimized / reference), method=method,
                         baseline=baseline, unchanged=unchanged, limitation=limitation, sample=sample))

    case = full['records'][0]
    if not case['exact_raw_parameters'] or not case['exact_exported_maps_and_qc'] or not case['automatic_qc_passed']:
        raise ValueError('Free-water numerical equivalence is incomplete')
    add('freewater', 'Free-water WLS', 'implementation', 'Full brain',
        case['reference_seconds'], case['optimized_seconds'],
        ('Reuse protocol-only attenuation/spectrum/grid arrays in a bounded private cache, alongside diagonal-free weights and source-qualified tensor helpers.'
         if protocol_reuse else 'Diagonal-free weight multiplication and source-qualified direct tensor-helper calls remove repeated allocation and compatibility checks.'),
        'Original installed DIPY WLS, not an earlier plugin. One numerical thread in both paths.',
        f"All 13 raw fit parameters, four maps/masks, affines, dtypes and automatic QC match. The same {case['validity']['excluded_tissue_voxels']} voxels remain excluded.",
        'One subject, one pair on a shared Linux host; optimized first. OS caches were not flushed. Both calls include raw-parameter evidence export.',
        f"{case['mask_voxels']} mask voxels; {case['measurements']} measurements; automatic QC passes, anatomical review pending.",
        evidence_date='2026-10-29' if protocol_reuse else '2026-10-22')

    pairs = {row['stage']: row for row in previous['complete_model_pairs']}
    for key in ('wma', 'noddi'):
        pair = pairs[key]
        if pair['status'] != 'passed' or not pair['products'] or not all(item['exact'] for item in pair['products']):
            raise ValueError('Incomplete output-equivalence evidence: ' + key)
    pair = pairs['wma']
    add('wma', 'WMA registration & classification', 'implementation', 'Complete module',
        pair['reference_seconds'], pair['optimized_seconds'],
        'Bulk geometry copying and verified registration-input reuse reduce repeated Python/VTK traversal and preparation.',
        'Original-author processing versus equivalent bulk/cache implementation, with identical UKF input and atlas.',
        'All 74 anatomical-tract output files are byte-identical; source inputs and author code are unchanged.',
        'One full-module pair on a shared Linux host. Matching files do not certify small-bundle sampling or anatomical validity. Later transform optimizations are not measured by this pair.',
        'Full registration and 74-tract output; no streamline downsampling introduced for timing.')
    pair = pairs['noddi']
    add('noddi', 'NODDI / AMICO', 'parallel', 'Full brain',
        pair['reference_seconds'], pair['optimized_seconds'],
        'Use AMICO\'s existing native multithreading: one thread versus eight.',
        'The same author implementation and model settings. This is a thread-configuration comparison, not a new algorithm.',
        'NDI, ODI, IsoVF, RMSE, NRMSE and the tissue-validity mask match voxelwise, including affines.',
        'One shared-host pair. Hardware-specific multithreading benefit; not a universal speed or whole-workflow guarantee.',
        'Original 740916-voxel brain mask; fresh output directories.')
    qr = load('mkcurve_qr_scipy1181_20261022.json')
    if qr['status'] != 'passed' or qr['environment']['scipy'] != '1.18.1' or not all(row['exact_output_parity'] for row in qr['records']):
        raise ValueError('Current SciPy QR verification did not pass')
    current_qr = [row for row in qr['records'] if row['solver'] == 'reference']
    case = current_qr[0]
    add('mkcurve', 'MK-Curve QR / Python port', 'port', '64-voxel numeric chunk',
        case['reference_median_seconds'], case['optimized_median_seconds'],
        'Reuse protocol/design checks and LAPACK workspace sizes while keeping the same pivoted QR and full b0 grid.',
        'Current SciPy 1.18.1 public gelsy solver versus qualified direct-driver/cache use in the same Python port. NOT original MATLAB versus Python.',
        'Raw coefficients, full curves, every map and validity decision match in all three subjects; both reference and batched solvers checked.',
        'Table shows subject 100206 default-QR medians, three alternating repeats after warmup. Numeric chunks exclude input preparation and file export; not whole ROI or full brain. SciPy already supplies native batching.',
        'Three real 64-voxel chunks; 181 measurements; complete 200-201 point b0 grids; one numerical thread.', evidence_date='2026-10-22')
    rows[-1]['solver_results'] = [dict(subject=item['subject_id'], solver=item['solver'],
        reference_seconds=item['reference_median_seconds'], optimized_seconds=item['optimized_median_seconds'],
        speed_ratio=item['speed_ratio']) for item in qr['records']]

    signal_report = load('performance_next_20261029.json')
    signal_records = mkcurve_signal_records(signal_report)
    reference_records = [item for item in signal_records if item['solver'] == 'reference']
    case = reference_records[0]
    reductions = [item['time_reduction_percent'] for item in reference_records]
    add('mkcurve-signal', 'MK-Curve signal reuse', 'port', '64-voxel numeric chunk / incremental QR change',
        case['reference_median_seconds'], case['optimized_median_seconds'],
        'Within each chunk, reuse non-b0 signals and their logarithms across the full synthetic-b0 scan; update only the candidate b0 column.',
        'The preceding qualified-QR Python implementation versus added signal reuse; solver, grid, correction rules and one numerical thread stay fixed.',
        'Original/corrected coefficients, complete curves, parameter maps, residuals and validity masks match exactly in all three acquisitions and both solvers.',
        'Displayed medians use acquisition 1, selected consistently with the earlier QR record. Three alternating repeats after one warmup. Batched-path changes range from -1.50% to +3.07%, so signal reuse remains enabled only for the reference solver. These incremental timings overlap the earlier QR implementation and are not additive.',
        f'Three acquisitions, 64 voxels each, 181 measurements and 200-201 synthetic-b0 samples. Reference-QR time falls {min(reductions):.2f}-{max(reductions):.2f}%. Numeric computation only; input preparation and file export excluded.',
        evidence_date=signal_report['reviewed_on'])
    rows[-1]['solver_results'] = [dict(subject=item['subject_id'], solver=item['solver'],
        reference_seconds=item['reference_median_seconds'], optimized_seconds=item['optimized_median_seconds'],
        speed_ratio=item['speed_ratio'], reduction_percent=item['time_reduction_percent'],
        times_seconds=item['times_seconds'], repeats=item['repeats'], warmups_per_method=item['warmups_per_method'],
        exact_output_parity=item['exact_output_parity']) for item in signal_records]

    roi = load('dki_latest_stable_integration_20261022.json')
    if roi['status'] != 'passed' or not all(item['all_output_arrays_exact'] and item['source_unchanged'] for item in roi['cases']):
        raise ValueError('DKI ROI equivalence did not pass')
    case = roi['cases'][0]
    add('dki', 'DKI WLS metrics', 'integration', 'Native 24-cubed ROI',
        case['median_seconds']['reference'], case['median_seconds']['optimized'],
        'Evaluate the author\'s existing array-capable kurtosis metrics once per block, avoiding repeated per-voxel result wrappers.',
        'Previous plugin per-voxel integration versus existing DIPY array usage. No improvement over the author\'s native vectorized API is claimed.',
        'All exported voxel arrays and affines match in three subjects, with three alternating repeats each.',
        'Table shows subject 100307 medians. This adopts existing native DIPY functionality, not a new upstream algorithm improvement. No completed full-brain DKI pair; do not scale ROI times to whole brain.',
        'Three subjects; three repeats per path; native-grid ROI, no resampling; DIPY 1.12.1 with SciPy 1.18.1.', evidence_date='2026-10-22')
    rows[-1]['subject_results'] = [dict(subject=item['subject'], reference_seconds=item['median_seconds']['reference'],
                                       optimized_seconds=item['median_seconds']['optimized'], speed_ratio=item['speed_ratio'])
                                  for item in roi['cases']]
    schedule = previous['concurrency_evidence']
    if schedule['status'] != 'passed' or not all(item['all_arrays_exact'] for item in schedule['measurements']):
        raise ValueError('Concurrent output equivalence did not pass')
    times = {item['mode']: item['elapsed_seconds'] for item in schedule['measurements']}
    add('scheduling', 'Independent FW + DKI jobs', 'workflow', 'Native 24-cubed ROI',
        times['sequential'], times['concurrent'],
        'Overlap independent model subprocesses after their shared inputs are ready; preserve downstream dependencies.',
        'The same model computations, serial scheduling versus two concurrent workers.',
        'All exported scalar and mask arrays match. Process startup is included.',
        'One ROI pair. This reduces schedule wall time, not the cost of an individual scientific model. Default model concurrency remains one; memory/CPU budgets may prevent overlap.',
        'Two independent models; CPU/memory-constrained scheduling; not full-brain timing.')

    cohort = load('cohort_multi_pipeline_20261109.json')
    cohort_by_workload = {item['workload']: item for item in cohort.get('pipelines', [])}
    cohort_specs = (
        ('cpu', 'cohort-microstructure', 'Cohort scheduling: Microstructure', '16 HCP subjects / 32-core budget',
         'Stage-level resource-aware scheduling across subjects overlaps single-thread Free-water/DKI and multithreaded NODDI/MK-Curve stages within a shared 32-core CPU and memory budget.',
         'Subject-level fixed concurrency (4 concurrent subjects, 8 threads each) versus Adaptive stage-level scheduling under the same 32-core CPU ceiling.',
         'All 16 subjects pass output and Rule-1 numerical equivalence checks across DTI, Free-water, NODDI, DKI and MK-Curve maps.',
         '16-subject HCP cohort on a shared Linux host (32-core CPU budget per arm). Reduces cohort makespan by raising mean CPU utilization from 12.6 to 21.2 cores, without changing individual model estimators.',
         '16 full-brain HCP subjects; complete Microstructure route (DTI + FW + NODDI + DKI + MK-Curve).'),
        ('anatomy', 'cohort-anatomy', 'Cohort scheduling: Anatomical priors', '16 HCP subjects / 32-core budget + GPU',
         'Coordinate GPU-backed DDSeg/DDParcel/DDSurfer inference and CPU-bound surface post-processing across subjects using per-stage CPU, RAM and GPU tokens.',
         'Subject-level fixed concurrency versus Adaptive stage-level scheduling across the same 16-subject cohort.',
         'All 16 subjects produce identical tissue segmentations, parcellations, cortical surfaces and QC outcomes.',
         '16-subject HCP cohort. Speedup reflects overlapping CPU surface post-processing with GPU inference across subjects rather than shortening single-subject neural network inference.',
         '16 full-brain HCP subjects; complete Anatomical priors route (DDSeg + DDParcel + DDSurfer).'),
        ('tractography', 'cohort-tractography', 'Cohort scheduling: Tractography & Bundles', '16 HCP subjects / 32-core budget + GPU',
         'Overlap CPU-bound DTI deterministic tracking, WMQL anatomical queries and structural connectome generation with GPU-assisted DDSeg/DDParcel/TractCloud stages across subjects under a unified resource budget.',
         'Subject-level fixed concurrency versus Adaptive stage-level scheduling across the same 16-subject cohort.',
         'All 16 subjects pass tractogram, bundle extraction (WMQL and TractCloud) and structural connectome equivalence and QC checks.',
         '16-subject HCP cohort. Stage-level scheduling overlaps GPU-assisted parcellation/TractCloud inference with CPU-bound tracking, WMQL and connectome stages across subjects.',
         '16 full-brain HCP subjects; Tractography & Bundles route (DDSeg + DDParcel + DTI + DTI-Det + WMQL + TractCloud + Connectome).'),
    )
    for workload, identifier, title, scope, method, baseline, unchanged, limitation, sample in cohort_specs:
        entry = cohort_by_workload.get(workload)
        if not entry or entry['arms']['subject']['subjects_passed_rule1'] != 16 or entry['arms']['adaptive']['subjects_passed_rule1'] != 16:
            raise ValueError('Incomplete 16-subject cohort scheduling evidence: ' + workload)
        add(identifier, title, 'workflow', scope,
            entry['arms']['subject']['makespan_seconds'], entry['arms']['adaptive']['makespan_seconds'],
            method, baseline, unchanged, limitation, sample, evidence_date='2026-11-09')

    filtering = load('author_wma_optimizations_20261020.json')
    sample = filtering['full_inputs']['cases'][0]
    if filtering['status'] != 'passed' or filtering['full_inputs']['status'] != 'passed':
        raise ValueError('Author WMA filtering comparison did not pass')
    add('wma-filter', 'WMA atlas length filtering', 'implementation', 'Single substage',
        sample['author_median_seconds'], sample['optimized_median_seconds'],
        'Replace per-point Python geometry copying with equivalent bulk coordinate/topology operations.',
        'Unmodified author filter on the full 1989106-streamline registration atlas.',
        'Output coordinates/topology and fixed-seed samples match exactly.',
        'One full-atlas substage pair, not complete WMA. The large local ratio must not be multiplied by or added to the full WMA result.',
        'Full registration-atlas length-filter substage only.')

    nls = load('freewater_dispatch_slicer_20261021.json')
    if (nls['status'] != 'passed' or not nls['source_inputs_unchanged'] or
            not all(item['exact_raw_parameters'] and item['exact_exported_maps_and_qc'] and
                    item['automatic_qc_passed'] for item in nls['records'])):
        raise ValueError('Free-water NLS equivalence did not pass')
    case = nls['records'][0]
    add('freewater-nls', 'Free-water NLS dispatch', 'implementation', '1728-voxel ROI',
        case['reference_median_seconds'], case['optimized_median_seconds'],
        'Call source-qualified author NLS, residual and Jacobian bodies directly, removing repeated keyword-compatibility dispatch.',
        'Installed original DIPY 1.12.1 serial NLS versus the same function bodies; one numerical thread in both paths.',
        'All 13 raw parameters, exported maps, affine and QC decisions match in all three subjects.',
        'Subject 100307 medians, three alternating repeats after warmup. ROI, not full brain. These timings precede later WLS-initialization/helper changes and cannot be added to them.',
        'Three native 12-cubed ROIs; 288 measurements; complete ROI stage including fresh export.', evidence_date='2026-10-21')

    transform = load('wma_transform_20261021.json')
    if (transform['status'] != 'passed' or not transform['source_inputs_unchanged'] or
            not transform['compatibility']['supported'] or
            not all(item['exact_all_points'] and item['exact_objective_history'] and item['exact_transform']
                    for item in transform['records'])):
        raise ValueError('WMA transform equivalence did not pass')
    case = transform['records'][0]
    add('wma-transform', 'WMA point transformation', 'implementation', '1000 fibers x 15 points',
        case['transform_author_seconds'], case['transform_optimized_seconds'],
        'Use the same author vtkTransform with a bounded bulk point call, instead of one Python/VTK call per point.',
        'Unmodified author point-transform loop versus bulk calls with the same float64 point order and transform.',
        'All transformed points, complete small-optimizer trajectories and final transforms match in three subjects.',
        'Subject 100206 median of five point-transform repeats, not full registration. Small-optimizer gains were only about 1.27-1.28x; the point-kernel ratio cannot be applied to whole WMA.',
        'Three real-subject fixtures; explicit synthetic offset for the separate optimizer check.', evidence_date='2026-10-21')

    fibers = load('wma_fibers_bounded_20261021.json')
    if (fibers['status'] != 'passed' or not fibers['sources_unchanged'] or
            not fibers['compatibility']['supported'] or
            not all(item['exact_all_arrays_and_hemispheres'] and item['exact_entire_optimizer_trajectory'] and
                    item['exact_final_transform'] for item in fibers['records'])):
        raise ValueError('WMA array-conversion equivalence did not pass')
    case = fibers['records'][0]
    add('wma-fibers', 'WMA fiber-array conversion', 'implementation', '20000 fibers x 15 points',
        case['author_seconds'], case['optimized_seconds'],
        'Gather the same author nearest-neighbor fiber samples from VTK connectivity in bounded NumPy blocks.',
        'Unmodified author FiberArray conversion versus equivalent bulk conversion; no interpolation or streamline reduction added.',
        'All arrays, hemisphere flags and separate small-optimizer trajectories/transforms match in three subjects.',
        'Subject 100206 median of five conversion repeats. Not whole WMA; small-optimizer gains were only about 1.02-1.05x. Historical source sampling failures remain failed.',
        '20000 identical sampled fibers per subject; original nearest-neighbor rule and ordering.', evidence_date='2026-10-21')

    historical = load('performance_optimizations_20261020.json')
    comparisons = {row['id']: row for row in historical['comparisons']}
    selected = (
        ('wma_registration_preprocessing', 'wma-cache', 'WMA registration-input reuse', 'implementation', 'Three preparation calls',
         'Reuse verified deterministic atlas/subject length-filter results across seeds; keep fresh author random sampling.',
         'Three original author preparation calls versus first cache fill plus reuse, with the same input and fixed seeds.',
         'One subject, one pair; includes initial cache construction. This overlaps WMA filtering and the full-module pair; savings cannot be added.'),
        ('surface_four_overlay_mapping', 'surface-mapping', 'DDSurfer surface-map reuse', 'inherited', 'Four left-hemisphere overlays',
         'Pack four overlays into one multi-frame mri_surf2surf call, sharing the same surface registration.',
         'Four separate FreeSurfer commands versus one multi-frame command. Inherited from DDSurfer upstream c3b2d40, not plugin-original work.',
         'One subject, command timing only; extra packing/validation excluded. Not surface reconstruction or a new FreeSurfer algorithm.'),
        ('diffusion_three_slab_io', 'dwi-io', 'Compressed DWI block reading', 'io', 'Three slabs',
         'Decode gzip once to temporary disk, then memory-map the same requested blocks; leave the source file unchanged.',
         'Repeated compressed-array reads versus one sequential decode and disk-backed block reads; no indexed_gzip in this measurement.',
         'One subject, one pair, uncontrolled OS page cache; needs temporary disk. I/O only, not model fitting; no universal storage-performance guarantee.'),
    )
    for key, identifier, title, category, scope, method, baseline, limitation in selected:
        case = comparisons[key]
        if (case['category'] not in ('same_input_stage_comparison', 'same_input_io_comparison') or
                not case['output_check'] or not case['evidence'] or
                not all(len(historical['evidence_records'][name]['sha256']) == 64 for name in case['evidence'])):
            raise ValueError('Incomplete preserved paired evidence: ' + key)
        add(identifier, title, category, scope, case['reference_seconds'], case['optimized_seconds'],
            method, baseline, case['output_check'], limitation, case['scope'])

    rt_20261109 = load('runtime_optimizations_20261109.json')
    rt_sections = runtime_optimizations_20261109_records(rt_20261109)
    wls_case = rt_sections['teem_dti_wls'][0]
    add('teem-dti-wls', 'Teem DTI WLS estimation', 'implementation', 'Full brain',
        wls_case['reference_seconds'], wls_case['optimized_seconds'],
        'Evaluate one all-zero signal template once and run Teem tenEstimateUpdate only on nonzero-signal voxel rows; fill zero-signal rows directly from the template.',
        'Previous full-volume Teem tenEstimateUpdate loop across all image-grid rows versus nonzero-row estimation with the same thread pool and gradient table.',
        'All 7 tensor/mask components, baseline b0 volume and failure counts match voxelwise across all three acquisitions.',
        'Displayed values use subject 100206 on a shared Linux host with the default host thread pool. Gains depend on the background zero-voxel fraction of the masked DWI volume.',
        f"{wls_case['voxels']} image-grid voxels ({wls_case['nonzero_voxels']} nonzero); {wls_case['gradients']} gradient rows; three real acquisitions verified.",
        evidence_date=rt_20261109['reviewed_on'])
    rows[-1]['subject_results'] = [dict(subject=item['subject_id'], reference_seconds=item['reference_seconds'],
                                        optimized_seconds=item['optimized_seconds'], speed_ratio=item['speed_ratio'])
                                   for item in rt_sections['teem_dti_wls']]

    dec_case = rt_sections['teem_dec_fa'][0]
    add('teem-dec-fa', 'Direction-encoded colour FA', 'implementation', 'Full brain',
        dec_case['reference_median_seconds'], dec_case['optimized_median_seconds'],
        'Restrict 3x3 measurement-frame rotation and symmetric eigendecomposition to voxels with positive fractional anisotropy; assign zero RGB directly elsewhere.',
        'Full-grid 3x3 matrix rotation and NumPy eigh across every image voxel versus active-FA voxel indexing with one numerical thread.',
        'Exported uint8 RGB direction-encoded colour maps are byte-identical in all three acquisitions.',
        'Subject 100206 medians of three alternating repeats after one warmup. Gains depend on the proportion of zero-FA background voxels.',
        f"{dec_case['voxels']} total voxels ({dec_case['nonzero_fa_voxels']} with FA > 0); three acquisitions, three alternating repeats each.",
        evidence_date=rt_20261109['reviewed_on'])
    rows[-1]['subject_results'] = [dict(subject=item['subject_id'], reference_seconds=item['reference_median_seconds'],
                                        optimized_seconds=item['optimized_median_seconds'], speed_ratio=item['speed_ratio'])
                                   for item in rt_sections['teem_dec_fa']]

    ep_case = rt_sections['regional_endpoints'][0]
    add('regional-endpoints', 'Streamline endpoint ROI selection', 'implementation', '1M streamlines',
        ep_case['reference_median_seconds'], ep_case['optimized_median_seconds'],
        'Transform start and end streamline points, check grid bounds and evaluate inclusion/target masks in 4096-streamline NumPy blocks; invert the atlas affine once.',
        'Per-streamline Python loop and repeated affine inversion versus batched 4096-streamline endpoint array evaluation.',
        'Selected streamline indices and output point arrays match identically in all three 1000000-streamline tractograms.',
        'Subject 100206 medians of three alternating repeats after one warmup on 4096-streamline batches; excludes file loading and writing.',
        'Three 1000000-streamline whole-brain tractograms; bilateral thalamus/precentral to superior-frontal endpoint query; one numerical thread.',
        evidence_date=rt_20261109['reviewed_on'])
    rows[-1]['subject_results'] = [dict(subject=item['subject_id'], reference_seconds=item['reference_median_seconds'],
                                        optimized_seconds=item['optimized_median_seconds'], speed_ratio=item['speed_ratio'])
                                   for item in rt_sections['regional_endpoints']]

    conn_case = rt_sections['connectome_radial'][0]
    add('connectome-radial', 'Connectome radial endpoint assignment', 'implementation', '1M streamlines',
        conn_case['reference_seconds'], conn_case['optimized_seconds'],
        'Batch direct voxel label lookup and SciPy cKDTree 2 mm radial search across 4096-streamline endpoint blocks instead of per-endpoint Python calls.',
        'Per-endpoint Python voxel lookup and single-point cKDTree queries versus batched 8192-point array queries with the same KD-tree and 2 mm radius.',
        'Direct voxel labels, radially assigned node labels and all 84x84 connectome matrices match identically across all three acquisitions.',
        'Displayed values use subject 100206 on 1000000 streamlines (2000000 endpoints); excludes tractogram file parsing and matrix export.',
        'Three 1000000-streamline tractograms; 84 grey-matter nodes; 2.0 mm radial assignment search; one numerical thread.',
        evidence_date=rt_20261109['reviewed_on'])
    rows[-1]['subject_results'] = [dict(subject=item['subject_id'], reference_seconds=item['reference_seconds'],
                                        optimized_seconds=item['optimized_seconds'], speed_ratio=item['speed_ratio'])
                                   for item in rt_sections['connectome_radial']]

    fm_20261110 = load('five_module_subalgorithm_optimizations_20261110.json')
    fm_records = five_module_subalgorithm_20261110_records(fm_20261110)
    fm_by_opt = {(r['sub_algorithm'], r.get('device'), r['subject_id']): r for r in fm_records}

    ddseg_cuda = fm_by_opt[('DDSeg', 'cuda', '100408')]
    add('ddseg-cuda', 'DDSeg tissue segmentation (CUDA)', 'implementation', 'Full 3-view CUDA',
        ddseg_cuda['reference_median_seconds'], ddseg_cuda['optimized_median_seconds'],
        'Cache predictions for uniform background slices within each anatomical view and execute TorchScript models under torch.inference_mode().',
        'Uncached per-slice TorchScript evaluation across all 256 slices per view under torch.no_grad() on the same GPU.',
        'All 3-view class probability volumes and final discrete 6-class tissue segmentations match voxelwise.',
        'Subject 100408 medians of three alternating 3-view repeats after warmup on NVIDIA RTX 3090; excludes NIfTI file I/O.',
        'Full 3-view 256-slice DTI feature stacks; 6 tissue classes; CUDA execution.',
        evidence_date=fm_20261110['reviewed_on'])

    ddseg_cpu = fm_by_opt[('DDSeg', 'cpu', '100408')]
    add('ddseg-cpu', 'DDSeg slice inference (CPU)', 'implementation', 'Strided full-view CPU',
        ddseg_cpu['reference_median_seconds'], ddseg_cpu['optimized_median_seconds'],
        'Skip redundant CNN forward passes on uniform background slices via per-view slice caching and torch.inference_mode().',
        'Uncached per-slice TorchScript CPU evaluation across a 32-slice strided sample spanning the full 256-slice sagittal view.',
        'Per-slice softmax probability maps match identically across all sampled slices.',
        'Subject 100408 medians of three alternating repeats after warmup on CPU. Gains reflect the background slice fraction around the brain mask.',
        '32-slice strided sagittal view sample spanning background and brain slices; CPU execution.',
        evidence_date=fm_20261110['reviewed_on'])

    ddparcel_cuda = fm_by_opt[('DDParcel', 'cuda', '100408')]
    add('ddparcel-cuda', 'DDParcel 3-view inference (CUDA)', 'implementation', 'Full 3-view CUDA',
        ddparcel_cuda['reference_median_seconds'], ddparcel_cuda['optimized_median_seconds'],
        'Build 7-slice windows with zero-copy sliding_window_view, concatenate 4 modalities per slice on demand, cache uniform background batches and apply a precomputed sagittal-to-full label LUT.',
        'Iterative np.append thick-slice construction, eager 4D modality concatenation, uncached background batches and per-batch sagittal label loops on the same GPU.',
        'Full 256x256x256x82 fused 3-view probability volumes match bitwise across all three views.',
        'Subject 100408 medians of three alternating full 3-view repeats after warmup on NVIDIA RTX 3090; eliminates 10.75 GB of temporary 4D array allocations.',
        'Full 256x256x256 4-modality DTI input volumes across Axial, Coronal and Sagittal FastSurferCNN networks.',
        evidence_date=fm_20261110['reviewed_on'])

    ddparcel_cc = fm_by_opt[('DDParcel', 'cpu', '100408')]
    add('ddparcel-cc', 'DDParcel 3D connected-component cleanup', 'implementation', '256-cubed parcellation',
        ddparcel_cc['reference_median_seconds'], ddparcel_cc['optimized_median_seconds'],
        'Label connected components once per parcel with return_num=True, identify the largest component via bincount and clear disconnected islands inside local region.slice bounding boxes.',
        'Duplicate full-volume measure.label calls and repeated 256-cubed boolean array comparisons per region in fastsurfercnn post-processing.',
        'Cleaned 256x256x256 parcellation volumes match voxelwise across all 81 non-zero labels.',
        'Subject 100408 medians of three alternating repeats after warmup on CPU.',
        'Full 256x256x256 DDParcel parcellation volume with 81 non-zero anatomical labels.',
        evidence_date=fm_20261110['reviewed_on'])

    mk_ols_records = [r for r in fm_records if r['sub_algorithm'] == 'MK-Curve']
    mk_ols_case = mk_ols_records[0]
    add('mkcurve-ols-eigh', 'MK-Curve OLS QR & batched eigh', 'port', '64-voxel numeric chunk',
        mk_ols_case['reference_median_seconds'], mk_ols_case['optimized_median_seconds'],
        'Reuse cached dgeqp3 QR factorizations of the equilibrated design matrix via dormqr/dtrsm for unweighted OLS initialization across the synthetic-b0 grid, vectorize tensor-basis construction and batch 3x3 eigendecomposition in parameter_maps.',
        'Preceding signal-reuse QR Python implementation calling dgelsy on every synthetic-b0 OLS initialization and per-voxel np.linalg.eigh.',
        'All fitted coefficients, MK-curves, parameter maps and validity masks match identically across all three acquisitions.',
        'Displayed medians use subject 100206 (1.10x; up to 1.37x on subject 149741), three alternating repeats after warmup with one numerical thread. Incremental over signal reuse.',
        'Three real 64-voxel chunks (100206, 149741, 877168); 181 measurements; 200 synthetic-b0 grid samples.',
        evidence_date=fm_20261110['reviewed_on'])
    rows[-1]['subject_results'] = [dict(subject=item['subject_id'], reference_seconds=item['reference_median_seconds'],
                                        optimized_seconds=item['optimized_median_seconds'], speed_ratio=item['speed_ratio'])
                                   for item in mk_ols_records]

    csd_qc = fm_by_opt[('CSD / ACT / SIFT2', 'cpu', '100408')]
    add('csd-qc', 'CSD/ACT tractogram QC validation', 'implementation', 'Full-brain tractogram',
        csd_qc['reference_median_seconds'], csd_qc['optimized_median_seconds'],
        'Validate vertex finiteness and compute per-streamline Euclidean segment lengths directly on flat ArraySequence coordinate and offset buffers using np.add.reduceat.',
        'Per-streamline Python loop converting each streamline to float64 and calling np.diff and np.linalg.norm individually.',
        'Streamline counts, finiteness flags, <20 mm fractions and all length percentiles match identically.',
        'Subject 100408 medians of three alternating repeats after warmup; includes tractogram loading and summary generation.',
        'Full-brain tractogram validation stage in CSD/ACT/SIFT2 quality control.',
        evidence_date=fm_20261110['reviewed_on'])

    wma_meas = next(r for r in fm_records if r['sub_algorithm'] == 'WMA')
    add('wma-measures', 'WMA native tract length & scalar summary', 'implementation', '50000 streamlines',
        wma_meas['reference_median_seconds'], wma_meas['optimized_median_seconds'],
        'Extract VTK cell offsets and connectivity arrays once via vtk_to_numpy and vectorize Euclidean segment-length summation across all streamlines.',
        'Per-cell vtkIdList traversal and pointwise math.dist accumulation across every streamline vertex in measure_polydata.',
        'Point counts, fiber counts, clamp exclusion counts and mean tract lengths match within IEEE-754 float64 summation precision (rtol < 1e-10).',
        'Subject 100408 medians of three alternating repeats after warmup on 50000 streamlines.',
        '50000 whole-brain streamlines in VTK polydata format.',
        evidence_date=fm_20261110['reviewed_on'])

    conn_fa = next(r for r in fm_records if r['sub_algorithm'] == 'Connectome (mean_fa)')
    add('connectome-mean-fa', 'Connectome along-tract FA sampling', 'implementation', '50000 streamlines',
        conn_fa['reference_median_seconds'], conn_fa['optimized_median_seconds'],
        'Batch endpoint voxel/KD-tree node assignment across all streamlines and vectorize along-tract scalar map sampling over flat ArraySequence vertices via np.add.reduceat.',
        'Per-streamline endpoint assignment and per-vertex Python affine transformation and scalar voxel lookup in _write_python_connectome.',
        'All exported node tables, edge lists and mean_fa connectivity matrix entries match within float64 precision (rtol < 1e-12).',
        'Subject 100408 medians of three alternating repeats after warmup on 50000 streamlines with 4 mm radial assignment.',
        '50000 streamlines; 84 DDParcel nodes; full-brain DTI FA scalar map.',
        evidence_date=fm_20261110['reviewed_on'])

    audit = load('upstream_freshness_verified_20261022.json')
    upstream = audit['repositories']
    references = {
        'freewater': f"Measured DIPY {full['environment']['dipy']}; latest stable at review: {upstream['dipy/dipy']['pypi']['version']}. " +
            ('Current stable-release full-brain pair.' if full['environment']['dipy'] == upstream['dipy/dipy']['pypi']['version'] else 'Historical full-brain timing, not latest-release timing; new stable ROI pairs verified separately.'),
        'wma': 'Author commit ' + upstream['SlicerDMRI/whitematteranalysis']['measured_reference_sha'][:12] +
            (': matches current HEAD and measured source bytes.' if upstream['SlicerDMRI/whitematteranalysis']['measured_reference_is_current_head'] else ': historical commit, no longer current HEAD.'),
        'wma-filter': 'Same immutable current WMA source audit as the full-module record.',
        'noddi': 'AMICO ' + upstream['daducci/AMICO']['pypi']['version'] +
            (' verified in both timed configurations; latest stable at review. Threads are already upstream functionality.' if upstream['daducci/AMICO']['measured_versions_match_latest_stable'] else '; timed configuration versions are not verified against current stable.'),
        'dki': 'DIPY 1.12.1 is current stable at review. Array metrics already exist upstream; this is plugin integration, not an author-algorithm speedup.',
        'mkcurve': 'Current MATLAB reference ' + upstream['zhangfanmark/MK-Curve']['head_sha'][:12] +
            '; measured on current stable SciPy 1.18.1. This port-only comparison does not claim to outperform MATLAB or invent upstream native batching.',
        'mkcurve-signal': 'Recorded 29 October with SciPy 1.18.1; source/input checksums and one-thread settings are preserved. This is an incremental comparison within the Python port, following the qualified QR change.',
        'scheduling': 'Historical plugin scheduler comparison, not a latest-upstream algorithm performance claim.',
        'cohort-microstructure': 'Plugin batch scheduler comparison (scripts/run_batch_pipeline.py); upstream DIPY 1.12.1, AMICO 2.1.1 and SciPy 1.18.1 estimators are unchanged.',
        'cohort-anatomy': 'Plugin batch scheduler comparison (scripts/run_batch_pipeline.py); upstream DDSeg, DDParcel, DDSurfer and FreeSurfer routines are unchanged.',
        'cohort-tractography': 'Plugin batch scheduler comparison (scripts/run_batch_pipeline.py); upstream DDSeg, DDParcel, DIPY deterministic tracking, WMQL, TractCloud and connectome routines are unchanged.',
        'freewater-nls': 'Measured DIPY 1.12.1 in Slicer; author NLS/residual/Jacobian source hashes match the dated stable-source audit.',
        'wma-transform': 'Author register_two_subjects.py SHA256 matches the 22 October WMA source audit; measured bulk vtkTransform path.',
        'wma-fibers': 'Author fibers.py at 4532b911386d independently verified; native VTK array sampling, not a new registration algorithm.',
        'wma-cache': 'Preserved same-author preparation pair. Source-qualified reuse, not a fresh latest-version benchmark.',
        'surface-mapping': 'Optimization inherited from DDSurfer c3b2d40; no plugin-original credit or latest-FreeSurfer speed claim.',
        'dwi-io': 'Generic compression/I/O control, not an improvement to a scientific estimator or an upstream-freshness claim.',
        'teem-dti-wls': 'Measured with Slicer 5.10.0 bundled Teem 1.12.0; preserves Teem tenEstimate WLS numerics and thread count.',
        'teem-dec-fa': 'Same measurement-frame rotation and NumPy eigh eigenvector convention; skips zero-FA background voxels where RGB is identically zero.',
        'regional-endpoints': 'Internal regional-tractography endpoint selector; preserves nearest-voxel rounding and endpoint-pair logic.',
        'connectome-radial': 'Measured with SciPy 1.18.1 cKDTree; preserves exact tie-breaking and 2 mm radial distance cutoff.',
        'ddseg-cuda': 'Measured with Slicer 5.10.0 PyTorch 2.14.1+cu130 on official DDSeg TorchScript weights; preserves exact per-slice CNN outputs.',
        'ddseg-cpu': 'Measured with Slicer 5.10.0 PyTorch 2.14.1 CPU on official DDSeg TorchScript weights; preserves exact per-slice CNN outputs.',
        'ddparcel-cuda': 'Measured with Slicer 5.10.0 PyTorch 2.14.1+cu130 on official DDSurfer FastSurferCNN weights; preserves exact 3-view fused probabilities.',
        'ddparcel-cc': 'Measured with scikit-image 0.25.2 measure.label/regionprops; preserves exact FastSurferCNN connected-component cleanup rules.',
        'mkcurve-ols-eigh': 'Measured in the Python MK-Curve port using LAPACK dgeqp3/dormqr/dtrsm and NumPy batched eigh; preserves exact WLLS coefficients and maps.',
        'csd-qc': 'Measured with NiBabel ArraySequence tractogram buffers; preserves exactCSD/ACT streamline finiteness and length-distribution QC.',
        'wma-measures': 'Measured with VTK 9.2 GetOffsetsArray/GetConnectivityArray buffers; preserves exact native WMA fiber length and scalar summary statistics.',
        'connectome-mean-fa': 'Measured with SciPy cKDTree and NiBabel ArraySequence buffers; preserves exact connectome edge weights and node assignments.',
    }
    if protocol_reuse:
        latest_fw = load('freewater_protocol_upstream_20261029.json')
        if latest_fw['pypi_version'] != full['environment']['dipy'] or latest_fw['qualified_bodies_match'] != [True, True]:
            raise ValueError('New Free-water pair lacks a matching stable/head source audit')
        references['freewater'] = ('Measured DIPY ' + full['environment']['dipy'] + '; latest stable on 29 October. '
            'The stable/head fitting definitions match qualified hashes; bounded protocol reuse is included in this full-brain pair.')
    measurement_sources = {
        'freewater': full_name,
        'wma': 'complete_models_20261020.json', 'noddi': 'complete_models_20261020.json',
        'mkcurve': 'mkcurve_qr_scipy1181_20261022.json', 'dki': 'dki_latest_stable_integration_20261022.json',
        'mkcurve-signal': 'performance_next_20261029.json',
        'scheduling': 'complete_models_20261020.json',
        'cohort-microstructure': 'cohort_multi_pipeline_20261109.json',
        'cohort-anatomy': 'cohort_multi_pipeline_20261109.json',
        'cohort-tractography': 'cohort_multi_pipeline_20261109.json',
        'wma-filter': 'author_wma_optimizations_20261020.json',
        'freewater-nls': 'freewater_dispatch_slicer_20261021.json', 'wma-transform': 'wma_transform_20261021.json',
        'wma-fibers': 'wma_fibers_bounded_20261021.json',
        **{identifier: 'performance_optimizations_20261020.json' for _, identifier, *_ in selected},
        'teem-dti-wls': 'runtime_optimizations_20261109.json',
        'teem-dec-fa': 'runtime_optimizations_20261109.json',
        'regional-endpoints': 'runtime_optimizations_20261109.json',
        'connectome-radial': 'runtime_optimizations_20261109.json',
        'ddseg-cuda': 'five_module_subalgorithm_optimizations_20261110.json',
        'ddseg-cpu': 'five_module_subalgorithm_optimizations_20261110.json',
        'ddparcel-cuda': 'five_module_subalgorithm_optimizations_20261110.json',
        'ddparcel-cc': 'five_module_subalgorithm_optimizations_20261110.json',
        'mkcurve-ols-eigh': 'five_module_subalgorithm_optimizations_20261110.json',
        'csd-qc': 'five_module_subalgorithm_optimizations_20261110.json',
        'wma-measures': 'five_module_subalgorithm_optimizations_20261110.json',
        'connectome-mean-fa': 'five_module_subalgorithm_optimizations_20261110.json',
    }
    for row in rows:
        row['upstream_reference'] = references[row['id']]
        row['measurement_source'] = next(source for source in sources if source['file'] == measurement_sources[row['id']])
    return dict(kind='slicerddsurfer_public_performance_evidence', version=4, reviewed_on=REVIEWED_ON,
                sources=sources, records=rows,
                upstream_review=dict(checked_at_utc=audit['checked_at_utc'],
                    stable_versions={key: upstream[repo]['pypi']['version'] for key, repo in (
                        ('dipy', 'dipy/dipy'), ('scipy', 'scipy/scipy'), ('amico', 'daducci/AMICO'))},
                    policy='Latest release and default-branch source identity are checked separately; source review is not a development-branch benchmark.'),
                policy='Output equivalence is not anatomical or clinical validation. No whole-workflow speedup has been established.',
                pending=['Repeated multi-subject full-module pairs', 'Completed full-brain DKI comparison',
                         'Full-brain validation for later default-QR MK-Curve changes',
                         'Matched CSD/ACT/SIFT2 complete-module baseline', 'Native Windows/macOS scientific timing',
                         'Optional persistent-model worker: proposed, not implemented or benchmarked'])


LABELS = dict(implementation='Same-method implementation', integration='Existing upstream capability', parallel='Native threads',
              port='Python-port computation', workflow='Workflow scheduling', inherited='Inherited upstream optimization', io='Compressed data I/O')


def duration(seconds):
    if seconds < .1:
        return f'{seconds * 1000:.3g} ms'
    return f'{seconds / 60:.2f} min' if seconds >= 180 else f'{seconds:.2f} s'


def render(evidence):
    rows = []
    for row in evidence['records']:
        identifier, category = escape(row['id']), escape(row['category'])
        rows.append(f'''<tr class="perf-record" data-category="{category}" data-record="{identifier}"><th scope="row"><button class="perf-toggle" type="button" aria-controls="method-{identifier}" aria-expanded="false" disabled><span class="perf-toggle-icon" aria-hidden="true">+</span><span>{escape(row['title'])}</span></button><small>{escape(row['scope'])} / record {row['evidence_date']}</small></th><td><span class="perf-kind {category}">{LABELS[row['category']]}</span></td><td>{duration(row['reference_seconds'])}</td><td>{duration(row['optimized_seconds'])}</td><td>{row['speed_ratio']:.2f}x<small>{row['reduction_percent']:.2f}% less time</small></td></tr>''')
        rows.append(f'''<tr class="perf-detail-row" id="method-{identifier}" hidden><td colspan="5"><div class="perf-method-body"><dl><dt>Upstream check</dt><dd>{escape(row['upstream_reference'])}</dd><dt>What changed</dt><dd>{escape(row['method'])}</dd><dt>Reference</dt><dd>{escape(row['baseline'])}</dd><dt>Equivalence</dt><dd>{escape(row['unchanged'])}</dd><dt>Measured scope</dt><dd>{escape(row['sample'])}</dd><dt>Limits</dt><dd>{escape(row['limitation'])}</dd></dl></div></td></tr>''')
    return '\n'.join([
        '<div class="perf-table-wrap" tabindex="0" role="region" aria-label="Measured performance comparisons, scroll horizontally on small screens"><table class="perf-table"><caption>Different models and scopes are not ranked against each other. Ratios are reference time divided by optimized time.</caption><thead><tr><th scope="col">Computation / scope</th><th scope="col">Type of change</th><th scope="col">Reference</th><th scope="col">Optimized</th><th scope="col">Measured ratio</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if checked-in website evidence is stale')
    args = parser.parse_args()
    evidence = build_evidence()
    payload = json.dumps(evidence, indent=2) + '\n'
    page = SITE / 'index.html'
    original = page.read_text(encoding='utf-8')
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise ValueError('Expected one performance evidence region in docs/site/index.html')
    before, middle = original.split(BEGIN)
    _, after = middle.split(END)
    expected = before + BEGIN + '\n' + render(evidence) + '\n' + END + after
    reviewed = date.fromisoformat(evidence['reviewed_on'])
    expected = re.sub(r'Catalogue updated \d{1,2} [A-Za-z]+ \d{4}',
                      f'Catalogue updated {reviewed.day} {reviewed:%B %Y}', expected)
    data = SITE / 'assets/data/performance.json'
    if args.check:
        if not data.is_file() or data.read_text(encoding='utf-8') != payload or original != expected:
            raise SystemExit('Website performance evidence in index.html is stale; run scripts/build_performance_site.py')
        print('PASS public performance data, source hashes and rendered measurements')
    else:
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(payload, encoding='utf-8', newline='\n')
        page.write_text(expected, encoding='utf-8', newline='\n')
        print(f"Exported {len(evidence['records'])} scoped measurements; private paths and runtime snapshots excluded")


if __name__ == '__main__':
    main()
