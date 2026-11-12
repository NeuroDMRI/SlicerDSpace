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
REVIEWED_ON = '2026-11-12'


def remaining_pipeline_accelerations_20261112_records(report):
    """Qualify the 12 November 2026 remaining-pipeline CPU and CUDA acceleration records."""
    if report.get('status') != 'completed' or report.get('kind') != 'remaining_pipeline_accelerations_benchmark':
        raise ValueError('Remaining-pipeline accelerations 2026-11-12 header is incomplete')
    rcpts = report.get('receipts') or {}
    expected_keys = {
        'input_nifti', 'brainmask_pnl_cnn', 'preprocessing_mrtrix_fsl',
        'surface_ddsurfer', 'surface_ddsurfer_postprocess', 'microstructure_noddi',
        'tractography_ukf', 'tractography_csd_act', 'bundles_wmql', 'bundles_tractcloud',
    }
    if set(rcpts.keys()) != expected_keys:
        raise ValueError('Unexpected receipt keys in remaining_pipeline_accelerations_20261112.json')
    for key, val in rcpts.items():
        if val.get('status') != 'completed':
            raise ValueError(f'Incomplete stage benchmark in {key}')
    if not (rcpts['input_nifti']['sub_dicom']['parity']['dwi_sha256_exact']
            and rcpts['input_nifti']['hcp_100307']['parity']['dwi_sha256_exact']):
        raise ValueError('NIfTI import parity check failed')
    if not (rcpts['brainmask_pnl_cnn']['cpu']['parity_vs_baseline']['exact_voxels']
            and rcpts['brainmask_pnl_cnn']['cuda']['parity_vs_baseline']['exact_voxels']):
        raise ValueError('PNL brain-mask binary parity check failed')
    if not rcpts['surface_ddsurfer_postprocess']['all_payloads_equal']:
        raise ValueError('DDSurfer post-processing payload parity check failed')
    if not rcpts['tractography_ukf']['adapter_io_qc_trk_pair']['trk_sha256_exact']:
        raise ValueError('UKF TRK export parity check failed')
    if not (rcpts['bundles_wmql']['parity']['cpu_counts_exact_vs_baseline']
            and rcpts['bundles_wmql']['parity']['cuda_counts_exact_vs_baseline']):
        raise ValueError('WMQL bundle streamline-count parity check failed')
    if not (rcpts['bundles_tractcloud']['cpu']['parity_vs_baseline']['counts_exact_all']
            and rcpts['bundles_tractcloud']['cuda']['parity_vs_baseline']['counts_exact_all']):
        raise ValueError('TractCloud bundle streamline-count parity check failed')
    return rcpts


def full_pipeline_accelerations_20261111_records(report):
    """Qualify the 11 November 2026 full-pipeline CPU and CUDA acceleration records."""
    if (report.get('status') != 'passed' or report.get('kind') != 'full_pipeline_cpu_and_cuda_accelerations'
            or report.get('source_inputs_unchanged') is not True):
        raise ValueError('Full-pipeline accelerations 2026-11-11 header is incomplete')
    modest = report.get('modest_hardware_verification') or {}
    if modest.get('all_peak_vram_under_cap') is not True:
        raise ValueError('Modest-hardware VRAM cap verification failed')
    records = report.get('records', [])
    if len(records) != 24:
        raise ValueError(f'Expected 24 full-pipeline acceleration records, found {len(records)}')
    for row in records:
        if row.get('exact_or_verified_parity') is not True:
            raise ValueError(f"Parity check failed for {row.get('id')}")
        before, after = row['reference_seconds'], row['optimized_seconds']
        if not all(math.isfinite(v) and v > 0 for v in (before, after)):
            raise ValueError(f"Invalid duration in {row.get('id')}")
        if (not math.isclose(before / after, row['speed_ratio'], rel_tol=1e-12)
                or not math.isclose(100 * (1 - after / before), row['time_reduction_percent'], abs_tol=1e-9)):
            raise ValueError(f"Speed ratio or reduction mismatch in {row.get('id')}")
    return records


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
         '16-subject HCP cohort on a shared Linux host (32-core CPU budget per arm). Reduces cohort makespan by raising mean CPU utilization from 12.9 to 19.9 cores, without changing individual model estimators.',
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
            method, baseline, unchanged, limitation, sample, evidence_date='2026-11-11')

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

    accel_20261111 = load('full_pipeline_accelerations_20261111.json')
    accel_by_id = {r['id']: r for r in full_pipeline_accelerations_20261111_records(accel_20261111)}
    accel_specs = (
        ('dipy-mppca-cpu', 'MP-PCA + Gibbs denoising (CPU)', 'implementation', 'Complete module (104x104x60x31)',
         'Extract sliding-window 3D patch blocks, solve batched covariance eigendecompositions with automatic z-slab RAM capping and vectorize 2D rFFT/irFFT Gibbs shifts.',
         'Initial unoptimized per-voxel DIPY localpca SVD/eigh loop and slice-by-slice FFT Gibbs suppression on the same acquisition.',
         'Denoised and degibbsed 4D volume matches unoptimized DIPY within float32/float64 numerical tolerance; automatic QC passes.',
         'Complete sub-dicom acquisition including NIfTI read/write; HCP 100307 (145x174x145x288) runs in 32.65 min on CPU.',
         'Full 104x104x60x31 clinical diffusion volume; patch_radius=2; CPU 8 numerical threads.'),
        ('dipy-mppca-cuda', 'MP-PCA + Gibbs denoising (CUDA)', 'implementation', 'Complete module (104x104x60x31; CUDA)',
         'Execute VRAM-adaptive batched PyTorch/cuSOLVER eigh MP-PCA patch kernels and batched 2D rFFT/irFFT Gibbs sub-voxel shifts on CUDA with automatic OOM fallback.',
         'Initial unoptimized per-voxel DIPY localpca SVD/eigh loop and slice-by-slice FFT Gibbs suppression on the same acquisition.',
         'Denoised and degibbsed 4D volume matches both unoptimized DIPY and optimized CPU within 3.9e-6 relative mean difference.',
         'Complete sub-dicom acquisition including NIfTI I/O on NVIDIA H100; HCP 100307 (288 volumes) completes in 4.80 min on CUDA and stays <2.4 GiB under a 3.5 GiB laptop VRAM cap.',
         'Full 104x104x60x31 clinical diffusion volume; patch_radius=2; CUDA execution.'),
        ('ddseg-full-cpu', 'DDSeg complete prediction (CPU)', 'implementation', 'Full 3-view predict_model_space (CPU)',
         'Batch non-uniform brain slices (16 slices/pass on CPU), cache uniform background slice predictions per view and run TorchScript under torch.inference_mode().',
         'Initial 3-view TorchScript slice-by-slice CPU evaluation without background caching or slice batching.',
         'Discrete 6-class tissue segmentation matches saved reference (>99.98% voxel agreement; exact match vs single-slice CPU).',
         'Subject 877168 median of three complete 3-view predict_model_space runs on CPU (8 threads); 4-thread laptop mode verified separately.',
         'Full 3-view 5-channel DTI feature volume; 6 tissue classes; CPU execution.'),
        ('ddseg-full-cuda', 'DDSeg complete prediction (CUDA)', 'implementation', 'Full 3-view predict_model_space (CUDA)',
         'Batch up to 64 non-uniform brain slices per forward pass (scaled to available free VRAM), cache uniform background slices and fuse 3-view probabilities under inference_mode.',
         'Initial 3-view TorchScript slice-by-slice GPU evaluation without multi-slice batching.',
         'Discrete 6-class tissue segmentation matches saved reference (>99.98% voxel agreement) while peaking at 1.15 GiB VRAM under a 3.5 GiB cap.',
         'Subject 877168 median of three complete 3-view predict_model_space runs including TorchScript loading on NVIDIA H100.',
         'Full 3-view 5-channel DTI feature volume; 6 tissue classes; CUDA execution.'),
        ('ddparcel-full-cuda', 'DDParcel complete 3-view inference (CUDA)', 'implementation', 'Full 3-view FastSurferCNN (CUDA)',
         'Fuse zero-copy thick-slice views, VRAM-adaptive 32-slice GPU batches, uniform background batch caching, sagittal LUT remapping and GPU probability accumulation.',
         'Initial unoptimized 3-view FastSurferCNN CUDA inference with iterative np.append thick slices, 8-slice batches and host-side sagittal label loops.',
         'Fused 256-cubed 82-class parcellation matches saved reference wmparc-Reg.mgz (>99.99% voxel agreement).',
         'Subject 100408 pure 3-view FastSurferCNN inference on NVIDIA H100; total time including 12-checkpoint initialization is also recorded in the benchmark file.',
         'Full 256x256x256 4-modality DTI volumes across Axial, Coronal and Sagittal FastSurferCNN ensembles; CUDA execution.'),
        ('ddparcel-roi-cc', 'DDParcel padded-ROI cleanup & filtering', 'implementation', 'Full 256-cubed post-processing',
         'Crop the 256-cubed parcellation volume to a padded non-zero brain bounding box before single-pass connected-component island cleanup and 3D median/Gaussian filtering.',
         'Unoptimized full-grid 256-cubed connected-component labeling and full-volume 3D median/Gaussian post-processing.',
         'Exported 256x256x256 post-processed parcellation volume is bit-for-bit identical across all 16,777,216 voxels (0 differing voxels).',
         'Subject 100408 median on CPU; gains reflect the ~85% background-voxel fraction outside the conformed brain bounding box.',
         'Complete 256x256x256 DDParcel post-processing stage (81 non-zero anatomical labels).'),
        ('native-dti-full', 'Fresh SlicerDMRI/Teem DTI + 6 scalars', 'implementation', 'Complete fresh DTI stage',
         'Combine single-template zero-voxel skipping in Teem WLS estimation with active-FA DEC generation and consolidated scalar export.',
         'Initial unoptimized fresh full-volume SlicerDMRI/Teem WLS tensor estimation and full-grid scalar/DEC export.',
         'All exported tensor components, 6 scalar maps (FA, MD, AD, RD, Trace, DEC) and masks match reference identically (max abs diff 0.0).',
         'Subject 100307 fresh end-to-end stage execution including compressed NIfTI/NRRD I/O on HCP 288-volume dMRI.',
         'Full-brain HCP 100307 (145x174x145x288); fresh uncached execution.'),
        ('freewater-full-cpu', 'Free-water complete WLS + NLS fit (CPU)', 'implementation', 'Complete full-brain fit (CPU)',
         'Combine protocol/attenuation reuse and column-scaled WLS initialization with vectorized analytical-Jacobian Gauss-Newton/Levenberg-Marquardt NLS refinement chunks.',
         'Initial unoptimized DIPY 1.12.1 full-brain WLS grid search and per-voxel SciPy leastsq NLS refinement.',
         'All 4 exported maps (FWF, tissue FA, tissue MD, validity mask) and QC metrics match DIPY 1.12.1 reference across 623,981 valid tissue voxels.',
         'Subject 100307 complete end-to-end full-brain execution including NIfTI I/O and QC.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); CPU execution.'),
        ('freewater-full-cuda', 'Free-water complete WLS + NLS fit (CUDA)', 'implementation', 'Complete full-brain fit (CUDA)',
         'Execute VRAM-adaptive batched FP64 CUDA Cholesky WLS grid search, analytical Cardano 3x3 eigensolver and vectorized GPU Gauss-Newton NLS refinement.',
         'Initial unoptimized DIPY 1.12.1 full-brain WLS grid search and per-voxel SciPy leastsq NLS refinement.',
         'FWF, tissue MD and validity mask match CPU identically (0.0 diff); tissue FA max abs diff is 5.96e-8 across 623,981 valid voxels.',
         'Subject 100307 complete end-to-end full-brain execution including NIfTI I/O and QC on NVIDIA H100.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); CUDA execution.'),
        ('dki-full-cpu', 'DKI complete WLS + 10 maps (CPU)', 'implementation', 'Complete full-brain fit (CPU)',
         'Replace per-voxel DIPY WLS and metric loops with chunked normal-equation WLS tensor/kurtosis estimation and vectorized analytical MK/AK/RK/KFA/MKT evaluation.',
         'Initial unoptimized full-brain DIPY 1.12.1 per-voxel WLS 22-parameter kurtosis fitting and metric wrappers (91.60 min).',
         'All 10 exported DKI/DTI maps, residuals and 681,769-voxel validity mask pass automatic QC and match CUDA within float32 precision.',
         'Subject 100307 complete end-to-end full-brain DKI stage including compressed 288-volume NIfTI I/O (tensor fit alone takes 24.21 s).',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 multi-shell volumes); CPU execution.'),
        ('dki-full-cuda', 'DKI complete WLS + 10 maps (CUDA)', 'implementation', 'Complete full-brain fit (CUDA)',
         'Solve column-equilibrated batched FP64 PyTorch/cuSOLVER Cholesky WLS normal equations and GPU-vectorized analytical kurtosis invariants in VRAM-bounded chunks.',
         'Initial unoptimized full-brain DIPY 1.12.1 per-voxel WLS 22-parameter kurtosis fitting and metric wrappers (91.60 min).',
         'All 10 exported maps match optimized CPU within <= 1.19e-7 (KFA, RMSE, NRMSE and masks exact); peaks at 0.47 GiB VRAM under a 3.5 GiB laptop cap.',
         'Subject 100307 complete end-to-end full-brain DKI stage including NIfTI I/O on NVIDIA H100 (CUDA tensor fit alone takes 3.74 s).',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 multi-shell volumes); CUDA execution.'),
        ('mkcurve-full-cpu1', 'MK-Curve complete signal correction (1-thread CPU)', 'port', 'Complete full-brain fit (1 worker)',
         'Combine basis/signal/QR reuse with coarse-to-fine synthetic-b0 grid search and batched 3x3 Cardano/eigh eigendecomposition across 740,916 brain voxels.',
         'Initial unoptimized Python MK-Curve full-brain implementation (165.90 min; prior basis-reuse version 153.53 min).',
         'All 27 exported uncorrected/corrected DKI & DTI maps, synthetic-b0 thresholds and 736,294-voxel validity masks pass full QC verification.',
         'Subject 100307 complete single-worker full-brain execution including NIfTI export of all 27 volumetric maps.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes; 200 synthetic-b0 grid samples); 1 CPU worker.'),
        ('mkcurve-full-cpu8', 'MK-Curve complete signal correction (8-worker CPU)', 'parallel', 'Complete full-brain fit (8 workers)',
         'Partition independent in-mask voxel chunks across 8 multi-process workers on top of the coarse-to-fine QR and batched-eigensolver CPU port.',
         'Initial unoptimized Python MK-Curve full-brain implementation (165.90 min).',
         'All 27 exported NIfTI volumes are bit-for-bit identical to the 1-worker CPU execution across every voxel.',
         'Subject 100307 complete 8-worker execution including NIfTI export; combines algorithmic port speedups with 8-process chunk parallelism.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); 8 CPU workers.'),
        ('mkcurve-full-cuda', 'MK-Curve complete signal correction (CUDA)', 'port', 'Complete full-brain fit (CUDA)',
         'Execute VRAM-adaptive batched FP64 PyTorch/cuSOLVER Cholesky WLS and analytical closed-form Cardano 3x3 eigensolvers across the synthetic-b0 grid on GPU.',
         'Initial unoptimized Python MK-Curve full-brain implementation (165.90 min; prior basis-reuse version 153.53 min).',
         'All 27 exported NIfTI maps match single-worker CPU within rtol=1e-5, atol=1e-6 (validity masks and b0 thresholds match identically).',
         'Subject 100307 complete end-to-end CUDA execution including host-to-device streaming and export of 27 NIfTI maps on NVIDIA H100.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); CUDA execution.'),
        ('dti-det-cpu', 'DTI deterministic tracking (CPU)', 'implementation', 'Complete module (CPU)',
         'Fit active-mask WLS diffusion tensors in vectorized chunks and extract exact analytical principal eigenvectors directly into DIPY EuDX tracking.',
         'Initial DIPY TenModel full-grid tensor fit and 362-vertex discrete sphere ODF projection before EuDX tracking.',
         'Produces the exact same 3,244 valid streamlines and automatic QC pass as the reference pipeline.',
         'Subject 100307 complete deterministic DTI tracking stage including DWI read and TCK export.',
         'Full-brain HCP 100307 (288 volumes); deterministic DTI EuDX tracking; CPU execution.'),
        ('dti-det-cuda', 'DTI deterministic tracking (CUDA)', 'implementation', 'Complete module (CUDA)',
         'Run batched CUDA Cholesky WLS tensor estimation and cuSOLVER eigh principal-direction extraction before deterministic EuDX streamline integration.',
         'Initial DIPY TenModel full-grid tensor fit and 362-vertex discrete sphere ODF projection before EuDX tracking.',
         'Produces the exact same 3,244 streamlines matching the optimized CPU execution.',
         'Subject 100307 complete deterministic DTI tracking stage including DWI read and TCK export on NVIDIA H100.',
         'Full-brain HCP 100307 (288 volumes); deterministic DTI EuDX tracking; CUDA execution.'),
        ('saf-full-cpu', 'SAF surface-guided filtering (CPU)', 'port', 'Complete module (100k tracks; CPU)',
         'Combine vectorized endpoint, intra-hemispheric and length prefiltering with spatial-hash bounding-volume-hierarchy (BVH) ray-triangle intersection tests.',
         'Initial unoptimized surface-guided short association fibre geometric filtering implementation on 100,000 whole-brain streamlines.',
         'Retains 13,043 valid superficial white-matter streamlines passing all anatomical surface-correspondence QC checks.',
         'Subject 100307 complete SAF stage including cortical surface loading, filtering and TCK export on CPU.',
         '100,000 CSD/ACT streamlines + DDSurfer bilateral white/pial cortical meshes; CPU execution.'),
        ('wma-full-cpu', 'WMA complete ORG-atlas pipeline (CPU)', 'implementation', 'Complete 74-bundle pipeline (CPU)',
         'Fuse bulk VTK geometry/input reuse and native Slicer transform hardening with vectorized bilateral registration, pairwise streamline distances and outlier removal.',
         'Initial unmodified WhiteMatterAnalysis complete pipeline (102.10 min; prior geometry-copy optimization 74.22 min) on identical UKF tractography and ORG atlas.',
         'All 74/74 anatomical white-matter bundles are non-empty (21,955 classified streamlines) and match CUDA per-bundle streamline counts.',
         'Subject 100307 complete end-to-end WMA registration, clustering, outlier removal and 74-tract MRML/VTP assembly on CPU.',
         'Full whole-brain UKF tractogram + ORG-800FC-100HCP atlas; 74 anatomical tracts; CPU execution.'),
        ('wma-full-cuda', 'WMA complete ORG-atlas pipeline (CUDA)', 'implementation', 'Complete 74-bundle pipeline (CUDA)',
         'Accelerate multi-scale rigid/affine fiber registration, bilateral pairwise fiber distances and cluster outlier removal via Triton and PyTorch CUDA kernels (whitematteranalysis_gpu).',
         'Initial unmodified WhiteMatterAnalysis complete pipeline (102.10 min; prior geometry-copy optimization 74.22 min) on identical UKF tractography and ORG atlas.',
         'All 74/74 anatomical bundles match optimized CPU per-bundle streamline counts identically (21,955 classified streamlines).',
         'Subject 100307 complete end-to-end WMA pipeline on NVIDIA H100 with automatic VRAM batch sizing and CPU fallback on memory pressure.',
         'Full whole-brain UKF tractogram + ORG-800FC-100HCP atlas; 74 anatomical tracts; CUDA execution.'),
        ('connectome-count-cpu', 'Connectome streamline count (CPU)', 'implementation', 'Complete 84-node connectome (CPU)',
         'Batch direct voxel label lookup, 4 mm SciPy cKDTree boundary radial search and flat np.add.at symmetric matrix accumulation across 1,000,000 streamlines.',
         'Initial unbatched Python endpoint lookup and per-streamline matrix accumulation over 1,000,000 streamlines.',
         '84x84 structural connectome matrix matches saved reference identically (max abs diff 0.0; 946,019 assigned streamlines).',
         'Subject 100408 complete connectome stage including 1M-streamline TCK loading, 4 mm radial assignment and CSV/JSON export on CPU.',
         '1,000,000 CSD/ACT streamlines; 84 DDParcel grey-matter nodes; 4.0 mm radial search; CPU execution.'),
        ('connectome-sift2-cpu', 'Connectome SIFT2-weighted (CPU)', 'implementation', 'Complete 84-node SIFT2 connectome (CPU)',
         'Combine batched direct voxel + 4 mm cKDTree endpoint assignment with vectorized flat np.bincount/add.at SIFT2 streamline-weight accumulation.',
         'Initial unbatched Python endpoint lookup and per-streamline SIFT2 weight accumulation over 1,000,000 streamlines.',
         '84x84 SIFT2-weighted connectome matrix matches saved reference within float64 machine precision (max abs diff 0.0).',
         'Subject 100408 complete SIFT2 connectome stage including 1M-streamline TCK + weights loading and export on CPU.',
         '1,000,000 CSD/ACT streamlines + SIFT2 weights; 84 DDParcel nodes; 4.0 mm radial search; CPU execution.'),
    )
    for identifier, title, category, scope, method, baseline, unchanged, limitation, sample in accel_specs:
        rec = accel_by_id[identifier]
        add(identifier, title, category, scope, rec['reference_seconds'], rec['optimized_seconds'],
            method, baseline, unchanged, limitation, sample, evidence_date=accel_20261111['reviewed_on'])

    accel_20261112 = load('remaining_pipeline_accelerations_20261112.json')
    rem = remaining_pipeline_accelerations_20261112_records(accel_20261112)
    rem_specs = (
        ('input-nifti-import', 'NIfTI import & validation (clinical 4D)', 'io',
         'Complete clinical 4D import (104x104x60x31)',
         rem['input_nifti']['sub_dicom']['baseline_seconds'], rem['input_nifti']['sub_dicom']['optimized_seconds'],
         'Validate finite non-zero voxel slabs in a single sequential gzip pass and stream standardized NIfTI output with fast deflate compression (compresslevel=1).',
         'Repeated slice-by-slice dataobj[..., idx] random gzip seeks and default level-9 gzip recompression during NIfTI import.',
         'Imported data.nii.gz, bvals and bvecs are byte-for-byte SHA-256 identical across payload values and affine matrices.',
         'Median of three clinical sub-dicom (104x104x60x31) imports; 288-volume HCP import verified separately.',
         'Complete run_nifti_import execution on clinical 4D dMRI (104x104x60x31).'),
        ('input-nifti-hcp', 'NIfTI import & validation (HCP 288-vol)', 'io',
         'Complete HCP 288-volume import (145x174x145x288)',
         rem['input_nifti']['hcp_100307']['baseline_seconds'], rem['input_nifti']['hcp_100307']['optimized_seconds'],
         'Eliminate O(N^2) gzip random-access re-decompression across 288 diffusion volumes via single-pass sequential slab streaming and fast deflate export.',
         'Slice-by-slice dataobj[..., idx] validation over compressed 288-volume NIfTI (re-decompressing earlier gzip streams on every frame).',
         'Exported 288-volume NIfTI payload, affine, bvals and bvecs match identically.',
         'Subject 100307 mean of two complete fresh imports; reflects compressed-stream I/O complexity rather than an estimator change.',
         'Full HCP 100307 diffusion volume (145x174x145x288; ~1.2 GB gzip).'),
        ('pnl-cnn-cpu', 'PNL CompNet brain masking (CPU)', 'port',
         'Complete module incl. ANTs & QC (CPU)',
         rem['brainmask_pnl_cnn']['cpu']['baseline_seconds'], rem['brainmask_pnl_cnn']['cpu']['optimized_seconds'],
         'Combine single-pass gzip b0 accumulation, concurrent SHA-256 hashing, overlapped 3-view CompNet weight preloading during ANTs registration, zero-slice deduplication and 16-slice CPU batching.',
         'Initial sequential b0 extraction, serial weight hashing/loading and 8-slice uncached CompNet inference alongside ANTs SyN registration.',
         'Exported native-space binary brain mask matches baseline bit-for-bit (0 differing voxels; Dice = 1.000000).',
         'Subject 100307 complete brain-mask stage on CPU (4 threads, fixed ANTs seed); ANTs registration accounts for ~65 s of both paths.',
         'Full HCP 100307 (288 volumes -> mean b0 -> ANTs template registration -> 3-view CompNet -> native mask + QC); CPU execution.'),
        ('pnl-cnn-3view-cuda', 'PNL 3-view CompNet inference (CUDA)', 'port',
         '3-view CompNet CNN pass (CUDA)',
         rem['brainmask_pnl_cnn']['cnn_3view_predict']['cuda']['baseline_seconds'],
         rem['brainmask_pnl_cnn']['cnn_3view_predict']['cuda']['optimized_seconds'],
         'Compile a static CompNet forward execution plan, deduplicate uniform background slices and evaluate 32-slice IEEE FP32 batches under torch.inference_mode().',
         'Per-step dynamic module-dict dispatch and 8-slice uncached 3-view CompNet evaluation on registered b0 volume.',
         'Fused 3-view brain mask agrees on >99.998% of voxels (32 boundary voxels over 2.22M mask voxels; native warped mask is 100% identical).',
         'Median of three warmed 3-view CompNet forward passes on registered HCP 100307 b0 volume on NVIDIA H100; excludes ANTs registration.',
         'Registered 3D b0 volume across axial, coronal and sagittal CompNet checkpoints; CUDA execution.'),
        ('preproc-fsl-cpu', 'MRtrix3/FSL distortion & eddy pipeline (CPU)', 'implementation',
         'Complete denoising + topup + eddy (CPU)',
         rem['preprocessing_mrtrix_fsl']['cpu']['baseline_seconds'],
         rem['preprocessing_mrtrix_fsl']['cpu']['optimized_seconds'],
         'Fuse in-RAM vectorized MP-PCA + Gibbs preprocessing before a single fast-gzip handoff and forward explicit multi-thread flags (-nthreads / --nthr) to topup and eddy_cpu.',
         'Separate DIPY subprocesses with intermediate level-9 gzip NIfTI round-trips and single-threaded topup execution before eddy_cpu.',
         'Final distortion- and eddy-corrected 4D DWI volume matches baseline (Pearson r = 0.99983; identical shape, affine and automatic QC pass).',
         'Complete sub-dicom acquisition with reverse-phase b0 topup + eddy_cpu (4 threads); iterative susceptibility/eddy solvers dominate remaining CPU runtime.',
         'Clinical 4D dMRI (104x104x60x31) + reverse-phase b0; MP-PCA + Gibbs + topup + eddy_cpu.'),
        ('preproc-fsl-cuda', 'MRtrix3/FSL distortion & eddy pipeline (CUDA)', 'implementation',
         'Complete denoising + topup + eddy_cuda (CUDA)',
         rem['preprocessing_mrtrix_fsl']['cuda']['baseline_seconds'],
         rem['preprocessing_mrtrix_fsl']['cuda']['optimized_seconds'],
         'Execute fused in-RAM CUDA MP-PCA + 2D FFT Gibbs suppression, concurrent fast-gzip NIfTI export and multi-threaded topup (--nthr=4) feeding FSL eddy_cuda.',
         'Unoptimized CPU DIPY subprocesses with intermediate level-9 gzip I/O and single-threaded topup feeding FSL eddy_cuda.',
         'Final corrected 4D DWI volume matches baseline (Pearson r = 0.99999; 0.28% relative mean difference within multi-threaded topup floating-point tolerance).',
         'Complete sub-dicom acquisition with reverse-phase b0 on NVIDIA H100 + 4 CPU threads.',
         'Clinical 4D dMRI (104x104x60x31) + reverse-phase b0; CUDA MP-PCA/Gibbs + multi-threaded topup + eddy_cuda.'),
        ('ddsurfer-cpu', 'DDSurfer bilateral surface reconstruction (CPU)', 'implementation',
         'Complete lh/rh white & pial meshes (CPU)',
         rem['surface_ddsurfer']['cpu']['baseline_seconds'], rem['surface_ddsurfer']['cpu']['optimized_seconds'],
         'Pack 18 feature channels into a contiguous channels-last [X*Y*Z, 18] buffer for single-pass trilinear vertex gathering, reuse conformal volumes across hemispheres and buffer .obj mesh serialization.',
         '18 separate per-channel volume interpolations per ODE step, duplicate cross-hemisphere NIfTI loading and per-vertex Python string formatting.',
         'All four cortical meshes (lh/rh white & pial) preserve exact triangle topology with <= 6.4e-4 mm maximum vertex displacement (mean 1.2e-5 mm).',
         'Subject 100408 complete bilateral white and pial reconstruction on CPU (4 threads).',
         'Full HCP 100408 conformed feature volumes; bilateral white + pial ODE deformation; CPU execution.'),
        ('ddsurfer-cuda', 'DDSurfer bilateral surface reconstruction (CUDA)', 'implementation',
         'Complete lh/rh white & pial meshes (CUDA)',
         rem['surface_ddsurfer']['cuda']['baseline_seconds'], rem['surface_ddsurfer']['cuda']['optimized_seconds'],
         'Stack all 18 spatial feature volumes into one [1, 18, D, H, W] GPU tensor for single-kernel F.grid_sample trilinear sampling, reuse volume cache across hemispheres and vectorize native RAS transforms.',
         '18 separate 1-channel F.grid_sample kernel launches per ODE step, repeated NIfTI loading per hemisphere and unbuffered .obj export.',
         'All four reconstructed cortical meshes (lh/rh white & pial) match baseline bit-for-bit (0.0 mm maximum vertex distance; exact topology).',
         'Subject 100408 median of three complete bilateral white and pial reconstructions on NVIDIA H100.',
         'Full HCP 100408 conformed feature volumes; bilateral white + pial ODE deformation; CUDA execution.'),
        ('ddsurfer-postproc-dag', 'Cortical surface morphometry DAG (CPU)', 'implementation',
         'Complete FreeSurfer post-processing (4 threads)',
         rem['surface_ddsurfer_postprocess']['baseline_seconds'],
         rem['surface_ddsurfer_postprocess']['optimized_seconds'],
         'Schedule independent intra-hemisphere FreeSurfer C++ branches (white morphometry, pial morphometry, inflation/sphere and thickness; then cortex label, aparc/a2009s annotations and fsaverage transfer) concurrently across stage threads and skip redundant symlink re-checks.',
         'Strictly serial intra-hemisphere execution of all 16 FreeSurfer surface, curvature, inflation, spherical registration, annotation and stats stages.',
         'All 133/133 exported surf/, label/, mri/ and stats/ files match baseline payloads identically (0 mismatches).',
         'Subject 100307 left-hemisphere complete post-processing (4 threads, sphere_seed=1234); single-threaded mris_register (~4 min) bounds maximum intra-hemisphere parallelism.',
         'DDSurfer native white/pial meshes -> FreeSurfer thickness, curvature, sulc, sphere.reg, aparc + aparc.a2009s annotations/stats and fsaverage overlays.'),
        ('noddi-full-cpu', 'NODDI active-set QP microstructure fit (CPU)', 'implementation',
         'Complete full-brain fit (8-thread CPU)',
         rem['microstructure_noddi']['baseline_8t_seconds'], rem['microstructure_noddi']['cpu_seconds'],
         'Replace per-voxel Python/Joblib scipy.optimize.nnls loops with a batched multi-threaded C++/PyTorch FP64 active-set Lawson-Hanson QP solver over the exact AMICO dictionary and concurrent fast-gzip NIfTI writes.',
         'Default 8-thread AMICO 2.1.1 full-brain NODDI fit on the same 740,916 mask voxels and precomputed dictionary.',
         'NDI and validity mask match AMICO within 5.96e-8 (0 voxels > 1e-6); 99.9995% of all 740,916 voxels agree within 1e-6 across ODI, IsoVF and RMSE.',
         'Subject 100307 complete end-to-end NODDI stage on 8 CPU threads including dictionary generation, 27-file export and QC.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); AMICO NODDI dictionary; 8 CPU threads.'),
        ('noddi-full-cuda', 'NODDI active-set QP microstructure fit (CUDA)', 'implementation',
         'Complete full-brain fit (CUDA)',
         rem['microstructure_noddi']['baseline_8t_seconds'], rem['microstructure_noddi']['cuda_seconds'],
         'Execute a vectorized FP64 CUDA active-set Lawson-Hanson NNLS solver across all 121x12 dictionary atoms per voxel in VRAM-bounded chunks alongside concurrent fast-gzip NIfTI export.',
         'Default 8-thread AMICO 2.1.1 CPU full-brain NODDI fit on the same 740,916 mask voxels (AMICO provides no GPU solver).',
         'Matches optimized CPU identically on NDI, RMSE, NRMSE and validity mask (0.0 max diff) and within 5.96e-8 on ODI/IsoVF (0 voxels > 1e-6 across 740,916 voxels).',
         'Subject 100307 complete end-to-end NODDI stage on NVIDIA H100 including AMICO LUT setup, GPU solve and 27 NIfTI exports.',
         'Full-brain HCP 100307 (740,916 mask voxels; 288 volumes); AMICO NODDI dictionary; CUDA execution.'),
        ('ukf-full', 'UKF 2-tensor whole-brain tractography (CPU)', 'implementation',
         'Complete 16-thread UKF stage',
         rem['tractography_ukf']['full_task_pair']['baseline_seconds'],
         rem['tractography_ukf']['full_task_pair']['optimized_seconds'],
         'Stream uncompressed scratch NRRDs into UKFTractography and reuse a single validated VTK polydata buffer across streamline validation, vectorized mask-overlap QC, flat ArraySequence TRK export and TRK verification.',
         'Gzip-compressed temporary NRRD staging and four redundant full-file VTK/TRK parses with per-streamline Python QC loops around UKFTractography.',
         'Produces the exact same 276,924 whole-brain streamlines and passes all automatic QC checks.',
         'Subject 100307 complete 16-thread 2-tensor UKF stage; C++ UKFTractography Kalman filtering (~10.3 min) accounts for >93% of runtime.',
         'HCP 100307 b0+b1000 shell; 2-tensor UKF (1 seed/voxel, 16 threads) + QC + VTK/TRK export.'),
        ('ukf-adapter-io', 'UKF NRRD staging, QC & TRK export', 'implementation',
         'Adapter pre/post-CLI I/O & QC (277,875 tracks)',
         rem['tractography_ukf']['adapter_io_qc_trk_pair']['baseline_seconds'],
         rem['tractography_ukf']['adapter_io_qc_trk_pair']['optimized_seconds'],
         'Write uncompressed scratch NRRDs, parse output VTK polydata once in memory, vectorize streamline length/mask-overlap QC and build TRK ArraySequence buffers directly from flat VTK offset/connectivity arrays.',
         'Compressed temporary NRRD writes plus separate VTK reads for contract check, Python per-streamline QC, nibabel GeneratorWrapper TRK export and TRK re-verification.',
         'Exported 100307_ukf.trk file is byte-for-byte SHA-256 identical (277,875 streamlines) with identical QC metrics.',
         'Isolated deterministic adapter staging + post-CLI validation/QC/TRK export on saved HCP 100307 UKF tractogram (excludes C++ UKFTractography tracking).',
         'HCP 100307 DWI staging + 277,875-streamline UKF VTK validation, QC and TRK conversion.'),
        ('csd-act-full', 'MRtrix3 CSD/ACT + SIFT2 whole-brain pipeline', 'implementation',
         'Complete 8-thread 1M-streamline stage',
         rem['tractography_csd_act']['baseline_seconds'], rem['tractography_csd_act']['optimized_seconds'],
         'Stage contiguous volume-axis strides (-strides 0,0,0,1) on dwi.mif, overlap post-tckgen tcksift2, tckconvert (.vtk) and TRK export concurrently, and vectorize 1M-streamline validation and SIFT2 weight parsing.',
         'Default dwi.mif strides, sequential post-tracking execution of tcksift2 -> tckconvert -> export_tck_to_trk and Python-loop SIFT2/streamline validation.',
         'Generates 1,000,000 valid ACT iFOD2 streamlines (.tck, .vtk, .trk) and 1,000,000 finite positive SIFT2 weights passing all QC checks.',
         'Subject 100408 complete 8-thread MSMT-CSD + ACT iFOD2 (1M tracks) + SIFT2 + VTK/TRK export; stochastic iFOD2 tckgen (~21 min) dominates total time.',
         'Full HCP 100408 (288 volumes) + ACT 5TT/GMWMI; 1,000,000 dynamic-seed iFOD2 streamlines; 8 CPU threads.'),
        ('csd-post-export', 'CSD/ACT post-tracking SIFT2 & VTK/TRK export', 'implementation',
         'Post-tckgen SIFT2 + dual export (1M tracks)',
         rem['tractography_csd_act']['post_tracking_sift2_export_pair']['baseline_seconds'],
         rem['tractography_csd_act']['post_tracking_sift2_export_pair']['optimized_seconds'],
         'Execute independent consumers of the finished 1M-streamline .tck file (tcksift2, multi-threaded tckconvert to VTK and streaming TCK-to-TRK conversion) concurrently in a bounded thread pool.',
         'Sequential execution of tcksift2 followed by tckconvert (.vtk) followed by Python TCK-to-TRK export on the same 1,000,000-streamline tractogram.',
         'All three outputs (SIFT2 weights/mu, 3.44 GB VTK polydata and 1.21 GB TRK tractogram) pass full 1,000,000-streamline integrity validation.',
         'Measured phase wall time from tckgen completion to completion of SIFT2, VTK and TRK exports on HCP 100408 (8-thread budget).',
         '1,000,000 whole-brain ACT streamlines + MSMT-CSD wmfod.mif on HCP 100408.'),
        ('wmql-full-cpu', 'WMQL 48-bundle rule-based queries (CPU)', 'implementation',
         'Complete 48-query segmentation (CPU)',
         rem['bundles_wmql']['baseline_seconds'], rem['bundles_wmql']['cpu_seconds'],
         'Load VTK/TRK streamlines into flat CSR arrays, compute voxel-label crossings via multi-threaded vectorized bincount histograms with lazy CSR sets and export binary VTK bundles concurrently.',
         'Upstream tract_querier per-streamline Python point loops, dense dict/set construction and sequential VTK bundle writing.',
         'All 48 official queries and 41 non-empty white-matter bundles match upstream tract_querier streamline counts identically (0 mismatches).',
         'Subject 100307 complete WMQL stage on 8 CPU threads including VTK load, 48 query evaluations, 41 bundle exports and QC.',
         'Full HCP 100307 UKF tractogram + DDParcel wmparc atlas; 48 official WMQL queries; CPU execution.'),
        ('tractcloud-full-cpu', 'TractCloud point-cloud parcellation (CPU)', 'implementation',
         'Complete 42-tract parcellation (CPU)',
         rem['bundles_tractcloud']['cpu']['baseline_seconds'], rem['bundles_tractcloud']['cpu']['optimized_seconds'],
         'Vectorize CSR polyline arc-length resampling for 15-point local + 20-neighbor global features, pre-stage contiguous batches and write compressed .vtp bundles concurrently from cached CSR coordinates.',
         'Per-streamline DIPY set_number_of_points loops, per-batch NumPy/tensor re-packing, duplicate whole-brain tractogram reload and serial level-9 VTP compression.',
         'All 39/39 non-empty anatomical bundles match baseline per-bundle streamline counts identically (exact_bundle_count = 39/39).',
         'Subject 100307 complete TractCloud stage on 4 CPU threads with fixed random_seed=0.',
         'HCP 100307 CSD/ACT tractogram + TractCloud 800-cluster atlas; 42 target tracts; CPU execution.'),
        ('tractcloud-full-cuda', 'TractCloud point-cloud parcellation (CUDA)', 'implementation',
         'Complete 42-tract parcellation (CUDA)',
         rem['bundles_tractcloud']['cuda']['baseline_seconds'], rem['bundles_tractcloud']['cuda']['optimized_seconds'],
         'Combine vectorized CSR feature resampling with GPU-resident dataset slicing (preserving exact torch.Generator batch order), IEEE FP32 cuDNN DGCNN+PointNet forward passes and parallel fast-zlib .vtp export.',
         'Unoptimized TractCloud pipeline with Python per-streamline resampling, per-batch host-to-GPU DataLoader collation, disabled cuDNN and serial VTP export.',
         'All 39/39 non-empty anatomical bundles match baseline CUDA streamline counts identically (exact_bundle_count = 39/39).',
         'Subject 100307 median of three complete TractCloud runs on NVIDIA H100 + 4 CPU threads (random_seed=0).',
         'HCP 100307 CSD/ACT tractogram + TractCloud 800-cluster atlas; 42 target tracts; CUDA execution.'),
    )
    for identifier, title, category, scope, before, after, method, baseline, unchanged, limitation, sample in rem_specs:
        add(identifier, title, category, scope, before, after, method, baseline, unchanged, limitation, sample,
            evidence_date=accel_20261112['reviewed_on'])

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
        'dipy-mppca-cpu': 'Compared directly against unmodified DIPY 1.12.1 localpca/mppca and gibbs_removal on the same 4D dMRI volume.',
        'dipy-mppca-cuda': 'Custom PyTorch/cuSOLVER CUDA implementation compared directly against unmodified DIPY 1.12.1 mppca and gibbs_removal.',
        'ddseg-full-cpu': 'Measured on official DDSeg TorchScript weights (NeuroDMRI/DDSeg-Torch); preserves exact 3-view softmax fusion.',
        'ddseg-full-cuda': 'Measured on official DDSeg TorchScript weights with VRAM-adaptive slice batching; preserves exact 3-view softmax fusion.',
        'ddparcel-full-cuda': 'Measured on official DDSurfer FastSurferCNN weights with VRAM-adaptive GPU batching; preserves exact 3-view fusion.',
        'ddparcel-roi-cc': 'Measured against unmodified FastSurferCNN 256-cubed post-processing; padded bounding-box crop is mathematically exact.',
        'native-dti-full': 'Measured against Slicer 5.10.0 Teem tenEstimate full-volume baseline; preserves exact WLS tensor and scalar maps.',
        'freewater-full-cpu': 'Compared directly against unmodified DIPY 1.12.1 fwdti WLS + NLS full-brain execution on HCP 100307.',
        'freewater-full-cuda': 'Custom PyTorch/cuSOLVER FP64 CUDA WLS + Cardano/Gauss-Newton solver compared against unmodified DIPY 1.12.1.',
        'dki-full-cpu': 'Compared directly against unmodified DIPY 1.12.1 DiffusionKurtosisModel WLS full-brain fit (91.60 min) on HCP 100307.',
        'dki-full-cuda': 'Custom column-equilibrated FP64 PyTorch/cuSOLVER CUDA Cholesky WLS + kurtosis invariant kernel compared against DIPY 1.12.1.',
        'mkcurve-full-cpu1': 'Compared against the initial full-brain Python port of zhangfanmark/MK-Curve (165.90 min) with identical synthetic-b0 grid.',
        'mkcurve-full-cpu8': '8-worker multi-process execution of the optimized Python MK-Curve port compared against the initial 165.90 min baseline.',
        'mkcurve-full-cuda': 'Custom batched FP64 CUDA Cholesky + analytical Cardano eigensolver port compared against the initial 165.90 min baseline.',
        'dti-det-cpu': 'Compared against DIPY 1.12.1 TenModel + discrete sphere peak extraction + EuDX deterministic tracking on HCP 100307.',
        'dti-det-cuda': 'Custom CUDA Cholesky WLS + cuSOLVER eigh peak extractor feeding DIPY EuDX compared against initial DIPY baseline.',
        'saf-full-cpu': 'Compared against the initial Python port of Shastin et al. SAF surface-guided filtering on 100,000 CSD/ACT streamlines.',
        'saf-full-cuda': 'Custom batched PyTorch CUDA Möller-Trumbore ray-triangle kernel compared against the initial SAF surface-filtering baseline.',
        'wma-full-cpu': 'Fuses plugin bulk-geometry/input reuse with NeuroDMRI/whitematteranalysis_gpu vectorized CPU kernels vs initial WMA (102.10 min).',
        'wma-full-cuda': 'Fuses plugin bulk-geometry/input reuse with NeuroDMRI/whitematteranalysis_gpu Triton/PyTorch CUDA kernels vs initial WMA (102.10 min).',
        'connectome-count-cpu': 'Compared against initial unbatched 1M-streamline endpoint assignment and matrix assembly on HCP 100408.',
        'connectome-count-cuda': 'Custom tiled PyTorch CUDA torch.cdist radial endpoint assignment compared against initial 1M-streamline baseline.',
        'connectome-sift2-cpu': 'Compared against initial unbatched 1M-streamline SIFT2-weighted connectome assembly on HCP 100408.',
        'connectome-sift2-cuda': 'Custom tiled PyTorch CUDA torch.cdist radial assignment + vectorized SIFT2 accumulation vs initial 1M-streamline baseline.',
        'input-nifti-import': 'Plugin NIfTI import/validation optimization (shared/input_import.py); eliminates repeated gzip random-access seeks and level-9 recompression.',
        'input-nifti-hcp': 'Plugin NIfTI import/validation optimization on 288-volume HCP dMRI; compressed-stream I/O fix, not an upstream estimator change.',
        'pnl-cnn-cpu': 'Measured on the ChengjinLii/PNLBrainMask-Torch CompNet port alongside ANTs registration; official pretrained CompNet weights unchanged.',
        'pnl-cnn-cuda': 'Measured on the ChengjinLii/PNLBrainMask-Torch CompNet port on CUDA alongside ANTs registration; official CompNet weights unchanged.',
        'pnl-cnn-3view-cuda': 'Measured on official converted CompNet weights (axial/coronal/sagittal .pt); static execution plan and slice batching preserve model math.',
        'preproc-fsl-cpu': 'Plugin workflow handoff and multi-thread forwarding around DIPY 1.12.1, MRtrix3 and FSL topup/eddy_cpu; external FSL binaries unchanged.',
        'preproc-fsl-cuda': 'Fuses CUDA MP-PCA/Gibbs kernels with multi-threaded FSL topup and unmodified FSL eddy_cuda on reverse-phase clinical dMRI.',
        'ddsurfer-cpu': 'Measured on official DDSurfer conformal feature checkpoints; channels-last trilinear gather and volume caching preserve mesh ODE integration.',
        'ddsurfer-cuda': 'Measured on official DDSurfer checkpoints; packed 18-channel F.grid_sample kernel matches per-channel GPU interpolation identically.',
        'ddsurfer-postproc-dag': 'Schedules independent FreeSurfer C++ surface/label/overlay commands concurrently within each hemisphere; FreeSurfer binaries unchanged.',
        'noddi-full-cpu': 'Custom multi-threaded FP64 active-set Lawson-Hanson QP solver over the exact AMICO 2.1.1 NODDI dictionary vs 8-thread AMICO.',
        'noddi-full-cuda': 'Custom batched FP64 PyTorch CUDA active-set Lawson-Hanson QP solver over the exact AMICO 2.1.1 NODDI dictionary vs 8-thread AMICO.',
        'ukf-full': 'Plugin staging, VTK validation, QC and TRK export around unmodified C++ UKFTractography (16 threads, 2-tensor model).',
        'ukf-adapter-io': 'Plugin pre/post-CLI NRRD/VTK/TRK I/O and vectorized QC around saved UKFTractography output; produces byte-identical TRK.',
        'csd-act-full': 'Plugin stride staging, concurrent post-tracking SIFT2/VTK/TRK export and vectorized QC around unmodified MRtrix3 binaries.',
        'csd-post-export': 'Concurrent execution of unmodified MRtrix3 tcksift2, tckconvert and streaming TRK export after tckgen completes.',
        'wmql-full-cpu': 'Custom CSR spatial indexing and parallel binary VTK export feeding upstream tract_querier query evaluation; 48 official queries unchanged.',
        'wmql-full-cuda': 'Custom CUDA crossing-histogram spatial indexing and parallel binary VTK export feeding upstream tract_querier query evaluation.',
        'tractcloud-full-cpu': 'Vectorized CSR streamline resampling and parallel fast-zlib VTP export around official TractCloud DGCNN+PointNet weights.',
        'tractcloud-full-cuda': 'Vectorized CSR resampling, GPU-resident batching and IEEE FP32 cuDNN execution on official TractCloud weights.',
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
        **{identifier: 'full_pipeline_accelerations_20261111.json' for identifier, *_ in accel_specs},
        **{identifier: 'remaining_pipeline_accelerations_20261112.json' for identifier, *_ in rem_specs},
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
                pending=['Repeated multi-subject full-module pairs',
                         'Native Windows/macOS scientific timing',
                         'Optional persistent-model worker: proposed, not implemented or benchmarked'])


LABELS = dict(implementation='Same-method implementation', integration='Existing upstream capability', parallel='Native threads',
              port='Python-port computation', workflow='Workflow scheduling', inherited='Inherited upstream optimization', io='Compressed data I/O')


def duration(seconds):
    if seconds < .1:
        return f'{seconds * 1000:.3g} ms'
    return f'{seconds / 60:.2f} min' if seconds >= 180 else f'{seconds:.2f} s'


def render(evidence):
    _ = len(evidence['records'])
    return '\n'.join([
        '<ul class="doc-links perf-index-links">',
        '  <li><a class="btn btn-secondary" href="https://github.com/NeuroDMRI/SlicerDSpace/blob/main/docs/release/PERFORMANCE_OPTIMIZATIONS.md" target="_blank" rel="noopener">Performance optimization index (Markdown)</a></li>',
        '  <li><a class="btn btn-secondary" href="assets/data/performance.json" download>Download measurement data (JSON)</a></li>',
        '</ul>',
        '<figure class="shot fig">',
        '  <a href="assets/img/fig-runtime-optimizations.png" target="_blank" rel="noopener" aria-label="Open the full-size runtime optimizations figure">',
        '    <picture>',
        '      <source type="image/webp" srcset="assets/img/fig-runtime-optimizations.webp">',
        '      <img src="assets/img/fig-runtime-optimizations.png" width="2400" height="2428" loading="lazy" decoding="async" fetchpriority="low" alt="Thirty-nine paired single-subject runtime benchmarks across all twenty-two protocol tasks, showing initial reference open circles and optimized filled circles on a shared log-time axis with aligned Reference, Optimized and Ratio columns.">',
        '    </picture>',
        '  </a>',
        '  <figcaption>Single-subject runtime comparisons across all six workflow stages (rows 01–39). Open circles mark initial or reference baselines and filled circles mark optimized paths on the same input acquisition; ratios give reference time divided by optimized time.</figcaption>',
        '</figure>',
        '<figure class="shot fig">',
        '  <a href="assets/img/fig-cohort-scheduling.png" target="_blank" rel="noopener" aria-label="Open the full-size parallel and cohort scheduling figure">',
        '    <picture>',
        '      <source type="image/webp" srcset="assets/img/fig-cohort-scheduling.webp">',
        '      <img src="assets/img/fig-cohort-scheduling.png" width="2400" height="703" loading="lazy" decoding="async" fetchpriority="low" alt="Three-panel parallel execution summary: stage-specific CPU scaling across 2 to 16 allocated CPUs, schematic subject-level waiting versus adaptive task backfilling, and 16-subject cohort time savings across all five downstream modules under a 32-CPU budget.">',
        '    </picture>',
        '  </a>',
        '  <figcaption>Parallel core scaling and multi-subject cohort scheduling. Panel a profiles per-stage runtime across 2–16 allocated CPUs and marks selected core quotas; panel b illustrates adaptive stage-level task backfilling across subjects; panel c reports measured makespan reductions across 16 Human Connectome Project subjects under a shared 32-CPU budget.</figcaption>',
        '</figure>',
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
