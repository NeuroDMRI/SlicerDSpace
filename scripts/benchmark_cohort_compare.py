"""Strict scientific-product fingerprints for a frozen cohort experiment."""

import hashlib
import json
from pathlib import Path


def _array(digest, value):
    import numpy as np
    value = np.asarray(value)
    digest.update(json.dumps([value.shape, value.dtype.str]).encode())
    # Hash stored values, shape and dtype without silently rounding differences.
    flat = value.ravel(order='C')
    for start in range(0, flat.size, 1024*1024):
        digest.update(flat[start:start+1024*1024].tobytes())


def product_hash(path, *, allow_empty=False):
    import numpy as np
    digest = hashlib.sha256()
    if path.name.endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
        import nibabel as nib
        image = nib.load(path)
        digest.update(str(image.header.get_data_dtype()).encode())
        _array(digest, image.affine)
        _array(digest, np.asanyarray(image.dataobj))
    elif path.suffix == '.npy':
        _array(digest, np.load(path, mmap_mode='r'))
    elif path.suffix == '.txt' and any(word in path.name.lower() for word in ('sift','weight')):
        _array(digest, np.loadtxt(path, comments='#', ndmin=1))
    elif path.suffix in ('.tck', '.trk'):
        import nibabel as nib
        tractogram = nib.streamlines.load(str(path), lazy_load=True).tractogram
        for track in tractogram:
            _array(digest, track.streamline)
            for kind in (track.data_for_streamline, track.data_for_points):
                for key in sorted(kind):
                    digest.update(key.encode())
                    _array(digest, kind[key])
    elif path.suffix in ('.vtk', '.vtp', '.stl'):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = {'.vtp': vtk.vtkXMLPolyDataReader, '.vtk': vtk.vtkPolyDataReader,
                  '.stl': vtk.vtkSTLReader}[path.suffix]()
        errors = []
        reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
        if path.suffix == '.vtp' and not reader.CanReadFile(str(path)):
            raise ValueError('Unreadable polydata: '+str(path))
        reader.SetFileName(str(path))
        reader.Update()
        data = reader.GetOutput()
        if errors or reader.GetErrorCode() or (data.GetNumberOfPoints() == 0 and not allow_empty):
            raise ValueError('Empty or unreadable polydata: '+str(path))
        if data.GetPoints() is None:
            digest.update(b'empty-polydata-without-points')
        else:
            _array(digest, vtk_to_numpy(data.GetPoints().GetData()))
        for cells in (data.GetVerts(), data.GetLines(), data.GetPolys(), data.GetStrips()):
            _array(digest, vtk_to_numpy(cells.GetOffsetsArray()))
            _array(digest, vtk_to_numpy(cells.GetConnectivityArray()))
        for attributes in (data.GetPointData(), data.GetCellData(), data.GetFieldData()):
            for i in range(attributes.GetNumberOfArrays()):
                array = attributes.GetAbstractArray(i)
                digest.update(str(array.GetName()).encode())
                if isinstance(array, vtk.vtkDataArray):
                    _array(digest, vtk_to_numpy(array))
                else:
                    digest.update(str([array.GetVariantValue(j).ToString()
                                       for j in range(array.GetNumberOfValues())]).encode())
    else:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def scientific_paths(root):
    """Select the same delivered arrays, geometry and saved cluster intermediates."""
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] not in ('02_microstructure','03_anatomy','04_tractography'):
            continue
        if any(part.startswith(('.', '_')) or part in ('cache', 'logs') for part in relative.parts):
            continue
        if path.name.endswith(('.nii','.nii.gz','.mgh','.mgz','.npy','.tck','.trk','.vtk','.vtp','.stl','.tfm')) or (
                path.suffix in ('.csv', '.tsv') and any('connectome' in part.lower() for part in relative.parts)) or (
                path.suffix == '.txt' and any(word in path.name.lower() for word in ('sift', 'weight', 'transform'))):
            yield path


def scientific_inventory(root):
    rows = {}
    for path in scientific_paths(root):
        relative = path.relative_to(root)
        # Atlas cluster slots may be legitimately empty after classification,
        # outlier removal or hemisphere separation. Final-product QC remains strict.
        cluster = ('FiberClustering' in relative.parts and path.name.startswith('cluster_')
                   and path.suffix in ('.vtk', '.vtp'))
        rows[str(relative)] = product_hash(path, allow_empty=cluster)
    if not rows:
        raise ValueError('No scientific arrays or tractograms found: '+str(root))
    return rows


def compare_inventories(reference, candidate):
    missing = sorted(reference.keys()-candidate.keys())
    extra = sorted(candidate.keys()-reference.keys())
    different = sorted(key for key in reference.keys() & candidate.keys() if reference[key] != candidate[key])
    return dict(status='passed' if reference and not (missing or extra or different) else 'failed',
                product_count=len(reference), missing=missing, extra=extra, different=different,
                scope='Exact array/dtype/affine, streamline ordering, mesh topology/attributes, weights and connectome tables; excludes logs and intermediate MIF files')


# ---------------------------------------------------------------- H8: paired-arm QC rules 2 and 3 (T3 S4)
# Ops code: it applies the class table of the spec and adds no metric, threshold or rule of its own.

import csv
import io
import math
import re

MASK_TOLERANCE_MM = 1e-4         # "grid matches": affines equal within 1e-4 mm
LABEL_FRACTION = 1e-4            # class L
FLOAT_FRACTION = 1e-4            # class T: 1e-4 x p99
SURFACE_MM = 0.01                # class S
MIN_SPEARMAN_EDGES = 10          # rule 3
MANIFEST_NAME = 'benchmark_orchestration_manifest.json'
DKI_METRIC_NAMES = ('mk', 'ak', 'rk', 'kfa', 'mkt', 'rtk')
DTI_MAP_NAMES = ('fa', 'md', 'ad', 'rd', 'l1', 'l2', 'l3', 'trace')
KEY_ALIASES = {'fw_tissue_fa': 'fw_fa', 'fw_tissue_md': 'fw_md', 'noddi_ndi': 'ndi', 'noddi_odi': 'odi',
               'noddi_fwf': 'isovf', 'noddi_rmse': 'rmse', 'noddi_nrmse': 'nrmse'}
# task name -> {canonical output key: class}; a key outside the table is class R.
TASK_CLASSES = {
    'microstructure.dti': {name: 'T' for name in DTI_MAP_NAMES},
    'microstructure.freewater': {'fwf': 'T', 'fw_fa': 'T', 'fw_md': 'T'},
    'microstructure.noddi': {name: 'T' for name in ('ndi', 'odi', 'isovf', 'rmse', 'nrmse')},
    'microstructure.dki': {name: 'T' for name in (*DKI_METRIC_NAMES, *(m + '_raw' for m in DKI_METRIC_NAMES))},
    'segmentation.ddseg': {'segmentation': 'L', 'gm_probability': 'T', 'wm_probability': 'T', 'csf_probability': 'T'},
    'parcellation.ddparcel': {'wmparc': 'L', 'wmparc_reg': 'L', 'fa_reg_norm': 'T', 'trace_reg_norm': 'T',
                              'min_eig_reg_norm': 'T', 'mid_eig_reg_norm': 'T'},
    'surface.ddsurfer': {name: 'S' for name in ('white_lh', 'white_rh', 'pial_lh', 'pial_rh')},
}
# Expected entries per workload, from its targets (the spec: "built from the workload's targets").
EXPECTED_TASKS = {
    'cpu': ('microstructure.dti', 'microstructure.freewater', 'microstructure.noddi', 'microstructure.dki'),
    'hybrid': ('microstructure.dti', 'microstructure.noddi', 'segmentation.ddseg', 'parcellation.ddparcel',
               'surface.ddsurfer'),
}
INPUT_MASK = 'input:nodif_brain_mask.nii.gz'
EXPECTED_CACHE = {'cpu': (INPUT_MASK,), 'hybrid': (INPUT_MASK, 'csd:five_tt')}
NATIVE_EXPECTED_FILES = ('DTI.nrrd', 'b0.nrrd')     # in every native_dti bundle (LS and WLS) seen in either arm


def _nrrd_hash(path, digest):
    import nrrd
    data, header = nrrd.read(str(path))
    fields = {key: (header[key].tolist() if hasattr(header[key], 'tolist') else header[key])
              for key in ('space', 'space directions', 'space origin', 'kinds', 'sizes', 'type') if key in header}
    digest.update(json.dumps(fields, sort_keys=True, default=str).encode())
    _array(digest, data)


_base_product_hash = product_hash


def product_hash(path, *, allow_empty=False):       # noqa: F811  (extends the base hash with .nrrd)
    if path.name.endswith('.nrrd'):
        digest = hashlib.sha256()
        _nrrd_hash(path, digest)
        return digest.hexdigest()
    return _base_product_hash(path, allow_empty=allow_empty)


def canonical_key(key):
    return KEY_ALIASES.get(key, key)


def _manifest_tasks(output):
    path = Path(output)/'00_run_metadata'/('benchmark_orchestration_manifest.json')
    return json.loads(path.read_text(encoding='utf-8')).get('tasks') or []


def subject_entries(trial_output, subject):
    """({entry id: {'cls': E/L/T/S, 'path': Path}}, class-R names) of one subject in one arm."""
    trial_output = Path(trial_output)
    output, root = trial_output/'outputs'/subject, trial_output/'inputs'/subject
    entries, report_only = {}, []
    for task in _manifest_tasks(output):
        if task.get('status') not in ('completed', 'skipped', 'reused'):
            continue
        name = task.get('name')
        table = TASK_CLASSES.get(name, {})
        for key, value in (task.get('outputs') or {}).items():
            canonical = canonical_key(key)
            if canonical in table and value:
                entries[f'{name}:{canonical}'] = dict(cls=table[canonical], path=Path(str(value)))
            elif name == 'parcellation.ddparcel' and key.endswith('_reg_norm') and value:
                entries[f'{name}:{key}'] = dict(cls='T', path=Path(str(value)))
            else:
                report_only.append(f'{name}:{key}')
    entries[INPUT_MASK] = dict(cls='E', path=root/'input/nodif_brain_mask.nii.gz')
    cache = root/'cache'
    native = cache/'native_dti'
    for path in sorted(native.rglob('*')) if native.is_dir() else ():
        name = path.name
        if path.is_file() and (name in ('DTI.nrrd', 'b0.nrrd') or re.fullmatch(r'DTI_.*\.nrrd', name) or (
                re.search(r'-dti-.*\.nii\.gz$', name) and 'Reg' not in name)):
            entries['cache/native_dti/'+path.relative_to(native).as_posix()] = dict(cls='E', path=path)
    registered = cache/'registered_dti'
    for path in sorted(registered.rglob('*')) if registered.is_dir() else ():
        relative = 'cache/registered_dti/'+path.relative_to(registered).as_posix()
        if path.is_file() and re.search(r'-mask-Reg\.nii\.gz$', path.name):
            entries[relative] = dict(cls='L', path=path)
        elif path.is_file() and re.search(r'-dti-.*-Reg\.nii\.gz$', path.name):
            entries[relative] = dict(cls='T', path=path)
    for manifest in sorted(output.rglob('csd_act_manifest.json')):
        five_tt = (json.loads(manifest.read_text(encoding='utf-8')).get('dependencies') or {}).get('five_tt') or {}
        if five_tt.get('path'):
            entries['csd:five_tt'] = dict(cls='T', path=Path(five_tt['path']))
        break
    return entries, report_only


def expected_entries(workload):
    keys = set(EXPECTED_CACHE[workload])
    for task in EXPECTED_TASKS[workload]:
        keys.update(f'{task}:{key}' for key in TASK_CLASSES[task]
                    if not (task == 'microstructure.dki' and key.endswith('_raw')))
    return keys


def _grid_matches(image, other):
    import numpy as np
    return tuple(image.shape[:3]) == tuple(other.shape[:3]) and bool(
        np.allclose(image.affine, other.affine, atol=MASK_TOLERANCE_MM, rtol=0))


def domain_mask(image, native_mask_path, registered_mask_paths):
    """The Subject-arm brain mask on the output's grid; all voxels when no mask grid matches."""
    import nibabel as nib
    import numpy as np
    for path in (native_mask_path, *registered_mask_paths):
        try:
            mask = nib.load(str(path))
        except Exception:
            continue
        if _grid_matches(image, mask):
            return np.asarray(mask.dataobj) > 0, str(path)
    return np.ones(tuple(image.shape[:3]), dtype=bool), 'all voxels'


def compare_label(first_image, second_image, mask):
    import numpy as np
    first, second = np.asarray(first_image.dataobj), np.asarray(second_image.dataobj)
    differ = first != second
    if differ.ndim > 3:
        differ = differ.reshape(differ.shape[:3] + (-1,)).any(axis=-1)
    count = int(mask.sum())
    if count == 0:
        return dict(status='not_compared', reason='empty domain mask')
    fraction = float(differ[mask].sum()) / count
    return dict(status='passed' if fraction <= LABEL_FRACTION else 'mismatch', differing_fraction=fraction,
                voxels=count, limit=LABEL_FRACTION)


def _flat(array):
    return array.reshape(array.shape[:3] + (-1,)) if array.ndim > 3 else array[..., None]


def compare_float(first_image, second_image, mask):
    import numpy as np
    first = np.asarray(first_image.dataobj, dtype=np.float64)
    second = np.asarray(second_image.dataobj, dtype=np.float64)
    finite_first, finite_second = np.isfinite(first), np.isfinite(second)
    if not np.array_equal(finite_first, finite_second):
        return dict(status='mismatch', reason='non-finite voxels at different positions',
                    differing_positions=int((finite_first != finite_second).sum()))
    domain = mask[..., None] & _flat(finite_first)
    selected_first, selected_second = _flat(first)[domain], _flat(second)[domain]
    if selected_first.size == 0:
        return dict(status='not_compared', reason='no finite voxel in the domain mask')
    # SPEC-GAP: p99 is taken over the finite in-domain Subject-arm voxels.
    p99 = float(np.percentile(np.abs(selected_first), 99))
    worst = float(np.max(np.abs(selected_first - selected_second)))
    limit = FLOAT_FRACTION*p99
    passed = worst == 0. if p99 == 0. else worst <= limit
    return dict(status='passed' if passed else 'mismatch', max_abs_difference=worst, p99=p99, limit=limit,
                voxels=int(selected_first.size))


def read_surface(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = {'.vtp': vtk.vtkXMLPolyDataReader, '.vtk': vtk.vtkPolyDataReader, '.stl': vtk.vtkSTLReader}[
        Path(path).suffix]()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data.GetNumberOfPoints() == 0:
        raise ValueError('Empty or unreadable surface: '+str(path))
    return (vtk_to_numpy(data.GetPoints().GetData()).astype('float64'),
            vtk_to_numpy(data.GetPolys().GetConnectivityArray()).copy())


def compare_surface(first_file, second_file):
    import numpy as np
    points_a, faces_a = read_surface(first_file)
    points_b, faces_b = read_surface(second_file)
    if points_a.shape != points_b.shape or faces_a.shape != faces_b.shape:
        return dict(status='mismatch', reason='vertex or face count differs',
                    vertices=[len(points_a), len(points_b)], faces=[len(faces_a), len(faces_b)])
    if not np.array_equal(faces_a, faces_b):
        return dict(status='mismatch', reason='faces differ')
    if not (np.isfinite(points_a).all() and np.isfinite(points_b).all()):
        return dict(status='mismatch', reason='non-finite vertex position')
    displacement = float(np.max(np.linalg.norm(points_a-points_b, axis=1)))
    return dict(status='passed' if displacement <= SURFACE_MM else 'mismatch', max_displacement_mm=displacement,
                limit_mm=SURFACE_MM)


def compare_entry(cls, first_path, second_path, domain_args):
    """Apply the class rule to one output of two arms. Geometry that differs is a mismatch."""
    import nibabel as nib
    import numpy as np
    if cls == 'E':
        return dict(status='passed' if product_hash(first_path) == product_hash(second_path) else 'mismatch')
    if cls == 'S':
        return compare_surface(first_path, second_path)
    first, second = nib.load(str(first_path)), nib.load(str(second_path))
    if first.shape != second.shape or not np.allclose(first.affine, second.affine, atol=MASK_TOLERANCE_MM, rtol=0):
        return dict(status='mismatch', reason='geometry differs')
    mask, mask_source = domain_mask(first, *domain_args)
    result = compare_label(first, second, mask) if cls == 'L' else compare_float(first, second, mask)
    result['domain'] = mask_source
    return result


def read_connectome(path):
    """{(i, j): weight} over the upper triangle (i < j) of an edge list or a dense matrix."""
    import numpy as np
    text = Path(path).read_text(encoding='utf-8')
    first = text.lstrip().splitlines()[0] if text.strip() else ''
    edges = {}
    if first.startswith('source_node'):
        for row in csv.DictReader(io.StringIO(text)):
            i, j = sorted((int(float(row['source_node'])), int(float(row['target_node']))))
            if i != j:
                edges.setdefault((i, j), float(row['metric']))
        return edges
    matrix = np.loadtxt(io.StringIO(text), delimiter=',' if ',' in first else None, ndmin=2)
    for i, j in zip(*np.triu_indices(matrix.shape[0], k=1)):
        edges[(int(i), int(j))] = float(matrix[i, j])
    return edges


def rule3(first_path, second_path):
    from scipy.stats import spearmanr
    try:
        first, second = read_connectome(first_path), read_connectome(second_path)
    except Exception as exc:
        return dict(status='undefined', reason=f'unreadable connectome: {type(exc).__name__}: {exc}')
    keys = sorted(key for key in first.keys() | second.keys() if first.get(key, 0.) != 0. or second.get(key, 0.) != 0.)
    if len(keys) < MIN_SPEARMAN_EDGES:
        return dict(status='undefined', reason=f'{len(keys)} edges < {MIN_SPEARMAN_EDGES}', edges=len(keys))
    rho = float(spearmanr([first.get(key, 0.) for key in keys], [second.get(key, 0.) for key in keys])[0])
    if not math.isfinite(rho):
        return dict(status='undefined', reason='constant weights', edges=len(keys))
    return dict(status='computed', spearman=rho, edges=len(keys))


def _connectome_matrix(trial_output, subject):
    for task in _manifest_tasks(Path(trial_output)/'outputs'/subject):
        if task.get('name') == 'connectome.sift2_weighted':
            value = (task.get('outputs') or {}).get('matrix_path')
            return Path(value) if value else None
    return None


def compare_subject(subject, subject_output, adaptive_output, workload):
    """Rule 2 (coverage plus class rules) and rule 3 for one subject whose two arms passed rule 1."""
    first, report_only = subject_entries(subject_output, subject)
    second, _ = subject_entries(adaptive_output, subject)
    results, mismatches = {}, []
    expected = expected_entries(workload)
    bundles = {key.split('/')[2] for key in (*first, *second) if key.startswith('cache/native_dti/')
               and key.count('/') >= 3} or {'<no bundle>'}
    expected.update(f'cache/native_dti/{bundle}/{name}' for bundle in bundles for name in NATIVE_EXPECTED_FILES)
    registered =[entry['path'] for key, entry in sorted(first.items())
                  if key.startswith('cache/registered_dti/') and entry['cls'] == 'L']
    for key in sorted(expected | first.keys() | second.keys()):
        usable_first = key in first and first[key]['path'].is_file()
        usable_second = key in second and second[key]['path'].is_file()
        if not (usable_first and usable_second):
            if key in expected or usable_first != usable_second:
                results[key] = dict(status='missing', subject_arm=usable_first, adaptive_arm=usable_second)
                mismatches.append(key)
            continue
        try:
            result = compare_entry(first[key]['cls'], first[key]['path'], second[key]['path'],
                                   (first[INPUT_MASK]['path'], registered))
        except Exception as exc:
            result = dict(status='mismatch', reason=f'{type(exc).__name__}: {exc}')
        result['class'] = first[key]['cls']
        results[key] = result
        if result['status'] == 'mismatch':
            mismatches.append(key)
    matrix_a, matrix_b = _connectome_matrix(subject_output, subject), _connectome_matrix(adaptive_output, subject)
    connectome = (rule3(matrix_a, matrix_b) if matrix_a and matrix_b and matrix_a.is_file() and matrix_b.is_file()
                  else dict(status='undefined', reason='no SIFT2 connectome in one arm'))
    return dict(rule2=dict(status='mismatch' if mismatches else 'passed', mismatches=mismatches, entries=results,
                           class_r=sorted(set(report_only)), class_r_note='report only; tractography is unseeded'),
                rule3=connectome)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def compare_pair(subject_campaign, adaptive_campaign, output):
    """H8: source_hashes check, QC rules 2-3, metric table; writes `pair_report.json` (never overwrites)."""
    from scripts.benchmark_cohort_harness import trial_metrics
    records, trials = {}, {}
    for policy, folder in (('subject', subject_campaign), ('adaptive', adaptive_campaign)):
        record = json.loads((Path(folder)/'campaign.json').read_text(encoding='utf-8'))
        matching = [row for row in record['trials'] if row['policy'] == policy and 'output' in row]
        if len(matching) != 1:
            raise ValueError(f'{policy} campaign must hold exactly one started {policy} trial')
        records[policy], trials[policy] = record, matching[0]
    hashes_equal = records['subject']['source_hashes'] == records['adaptive']['source_hashes']
    same_cohort = ([row['subject'] for row in records['subject']['subjects']] ==
                   [row['subject'] for row in records['adaptive']['subjects']]
                   and trials['subject']['route'] == trials['adaptive']['route'])
    workload = trials['subject']['route']
    report = dict(workload=workload, source_hashes_equal=hashes_equal, same_cohort_and_route=same_cohort,
                  limitations=['Shared host: other users share the CPUs, GPUs, disk and NFS; see the host and GPU samples',
                               'Claims are limited to the observed conditions; no confidence interval'])
    outcomes = {}
    for policy in records:
        path = Path(trials[policy]['output'])/'subject_outcomes.json'
        outcomes[policy] = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    metrics = {}
    for policy in records:
        passed = sum(row.get('rule1') == 'passed' for row in outcomes[policy].values())
        metrics[policy] = trial_metrics(trials[policy], records[policy], passed)
    report.update(metrics=metrics, subject_outcomes=outcomes)
    first, second = (metrics[p].get('throughput_subjects_per_hour') for p in ('subject', 'adaptive'))
    report['throughput_ratio'] = (second/first) if first and second else None
    report['throughput_ratio_definition'] = 'Adaptive throughput / Subject throughput; the numerator counts rule-1 passes'
    subjects = {}
    if hashes_equal and same_cohort:
        for row in records['subject']['subjects']:
            name = row['subject']
            if not all(outcomes[p].get(name, {}).get('rule1') == 'passed' for p in outcomes):
                subjects[name] = dict(rule2=dict(status='skipped_rule1', mismatches=[]), rule3=dict(status='skipped'))
                continue
            try:
                subjects[name] = compare_subject(name, trials['subject']['output'], trials['adaptive']['output'], workload)
            except Exception as exc:
                subjects[name] = dict(rule2=dict(status='mismatch', mismatches=['compare_error'],
                                                 reason=f'{type(exc).__name__}: {exc}'), rule3=dict(status='undefined'))
    report['subjects'] = subjects
    mismatched = sorted(name for name, row in subjects.items() if row['rule2']['status'] == 'mismatch')
    report['rule2_mismatch_subjects'] = mismatched
    report['flagged'] = bool(mismatched) or not hashes_equal or not same_cohort
    path = Path(output)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(_jsonable(report), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return report
