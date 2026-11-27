#!/usr/bin/env python3
"""Decode real bundle products and test their relationship to the source tracks.

The default audit is read-only. Optional QC updates archive the old records.
Exact/sub-millimetric source membership does not prove
the predicted anatomical bundle label; manual review remains explicit.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tracks_from_file(path):
    if path.suffix.lower() in ('.trk', '.tck'):
        import nibabel as nib
        return list(nib.streamlines.load(str(path)).streamlines)
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLPolyDataReader() if path.suffix.lower() == '.vtp' else vtk.vtkPolyDataReader()
    errors = []
    reader.AddObserver('ErrorEvent', lambda *_: errors.append(True))
    reader.SetFileName(str(path))
    reader.Update()
    if errors or reader.GetErrorCode():
        raise ValueError(f'Cannot decode {path}')
    poly = reader.GetOutput()
    if poly.GetNumberOfVerts() or poly.GetNumberOfPolys() or poly.GetNumberOfStrips():
        raise ValueError(f'Non-streamline cells in {path}')
    if not poly.GetNumberOfPoints():
        return []
    points = vtk_to_numpy(poly.GetPoints().GetData())
    offsets = vtk_to_numpy(poly.GetLines().GetOffsetsArray())
    connectivity = vtk_to_numpy(poly.GetLines().GetConnectivityArray())
    if np.any(connectivity < 0) or np.any(connectivity >= len(points)):
        raise ValueError(f'Invalid connectivity in {path}')
    return [points[connectivity[offsets[i]:offsets[i + 1]]] for i in range(len(offsets) - 1)]


def line_hash(track):
    a = np.ascontiguousarray(track, dtype=np.float32)
    forward, reverse = a.tobytes(), a[::-1].tobytes()
    return hashlib.sha256(min(forward, reverse)).digest()


def stats(values):
    return dict(zip(('min', 'median', 'p95', 'max'), map(float, np.percentile(values, (0, 50, 95, 100))))) if len(values) else None


def endpoint_rule_check(manifest, files):
    """Independent voxel-space check of the shipped endpoint rules and aliases."""
    from tractography.wmql import bundled_wmql_query_path, wmql_query_dependencies, wmql_query_provenance
    query = Path(manifest.get('staged_query') or manifest['query_path'])
    bundled = bundled_wmql_query_path('published_endpoints')
    if query.read_bytes() not in (bundled.read_bytes(), bundled.with_name('CST_WMQL2016.qry').read_bytes(),
                                  bundled_wmql_query_path('ddparcel_act').read_bytes()):
        return {'status': 'not_checked', 'reason': 'Custom query; fixed bundled-rule audit does not apply.'}
    try:
        dependencies = wmql_query_dependencies(query, [query.parent])
        provenance = wmql_query_provenance(query, dependencies)
    except (OSError, ValueError, SyntaxError):
        return {'status': 'not_checked', 'reason': 'Regional alias snapshot unavailable; fixed bundled-rule audit does not apply.'}
    if provenance['preset'] not in {'published_cst_endpoints', 'af_candidates_published_cst',
                                     'ddparcel_af_candidates_author_cst'}:
        return {'status': 'not_checked', 'reason': 'Modified regional aliases; fixed bundled-rule audit does not apply.'}
    import nibabel as nib
    atlas = nib.load(manifest['atlas_path'])
    labels = np.asarray(atlas.dataobj)
    inverse = np.linalg.inv(atlas.affine)
    results, not_checked = [], []
    for file in files:
        name = file.stem.split('_desc-wmql_', 1)[-1].lower()
        if (name not in {'af.left', 'af.right', 'cst.left', 'cst.right'}
                or (provenance['preset'] == 'ddparcel_af_candidates_author_cst' and not name.startswith('af.'))):
            not_checked.append({'query': name, 'reason': 'Not a certified endpoint predicate in this preset.'})
            continue
        shift = 1000 if name.endswith('.right') else 0
        frontal = {1003, 1018, 1019, 1020, 1024, 1027}
        temporal = {1015, 1030}
        cortex = {1022, 1024, 3022, 3024}
        frontal, temporal, cortex = ({v + shift for v in group} for group in (frontal, temporal, cortex))
        valid, invalid, outside, pairs = 0, 0, 0, Counter()
        for track in tracks_from_file(file):
            voxels = np.rint(nib.affines.apply_affine(inverse, track[[0, -1]])).astype(int)
            if np.any(voxels < 0) or np.any(voxels >= labels.shape):
                outside += 1
                invalid += 1
                continue
            a, b = (int(labels[tuple(v)]) for v in voxels)
            pairs[str((a, b))] += 1
            matches = ((a in frontal and b in temporal) or (b in frontal and a in temporal)) if name.startswith('af.') else \
                      ((a == 16 and b in cortex) or (b == 16 and a in cortex))
            valid += int(matches)
            invalid += int(not matches)
        results.append(dict(query=name, matches=valid, violations=invalid, outside_fov=outside, endpoint_label_pairs=dict(pairs)))
    return {'status': ('passed' if all(r['violations'] == 0 for r in results) else 'failed') if results else 'not_checked',
            'reason': None if results else 'No applicable endpoint products.',
            'bundles': results, 'checks_not_run': not_checked,
            'scope': 'Exact selected endpoint label rules, not anatomical AF/CST ground truth; no image resampling or endpoint dilation.'}


def applicable_checks(kind, contract_error, geometry_ok, duplicate_count, endpoint_rules, *, wma_sampling=None):
    checks = {'current_output_contract': contract_error is None,
              'finite_positive_geometry_and_source_membership': geometry_ok}
    not_run = {}
    if kind == 'wma_org_atlas':
        checks['wma_all_tract_sampling'] = wma_sampling is True
    if kind == 'tractcloud_parcellation':
        checks['tractcloud_disjoint_assignment'] = duplicate_count == 0
    else:
        not_run['tractcloud_disjoint_assignment'] = 'Only TractCloud assignment is required to form a disjoint partition.'
    if endpoint_rules is not None and endpoint_rules['status'] in {'passed', 'failed'}:
        checks['bundled_wmql_endpoint_rules'] = endpoint_rules['status'] == 'passed'
    else:
        not_run['bundled_wmql_endpoint_rules'] = (endpoint_rules or {}).get('reason', 'Not a bundled WMQL endpoint query.')
    return checks, not_run


def audit(manifest_path, tolerance=.001):
    started = time.monotonic()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent
    kind = manifest['kind']
    qc_name = {'wma_org_atlas': 'wma_org_atlas_qc.json', 'wmql_bundle_segmentation': 'wmql_qc.json',
               'tractcloud_parcellation': 'tractcloud_qc.json'}[kind]
    qc = json.loads((directory / qc_name).read_text()) if (directory / qc_name).is_file() else {}
    recorded = qc.get('output_summary') or {}
    # Audit the exact retained failure, never unrelated old published products.
    if qc.get('status') in {'failed', 'partial_failed'}:
        root = recorded.get('root')
        if kind == 'wma_org_atlas' and recorded.get('case_dir'):
            root = str(Path(recorded['case_dir']).parent)
        if root:
            candidate = Path(root).resolve()
            candidate.relative_to(directory.resolve())
            directory = candidate
    source_path = Path(manifest.get('prepared_input_tractography') or manifest['input_tractography'])
    if kind == 'wma_org_atlas':
        from tractography.wma import summarize_wma_outputs, _wma_output_contract_error
        summary = summarize_wma_outputs(directory, source_path)
        contract_error = _wma_output_contract_error(summary)
        files = list(map(Path, summary['anatomical_tract_files']))
    elif kind == 'wmql_bundle_segmentation':
        from tractography.wmql import summarize_wmql_outputs, _wmql_output_contract_error
        summary = summarize_wmql_outputs(directory, manifest['subject_id'])
        requested = tuple(manifest.get('requested_queries') or manifest.get('query_selection', ()))
        contract_error = _wmql_output_contract_error(summary, len(requested), requested,
                                                    require_sampling=not (manifest.get('official_multibundle', False)
                                                                          or manifest.get('coverage_report_mode', False)))
        files = list(map(Path, summary['bundle_files']))
    elif kind == 'tractcloud_parcellation':
        from tractography.tractcloud import summarize_tractcloud_outputs, _tractcloud_output_contract_error
        summary = summarize_tractcloud_outputs(directory)
        contract_error = _tractcloud_output_contract_error(summary)
        files = list(map(Path, summary['bundle_files']))
    else:
        raise ValueError(f'Unsupported bundle manifest {kind}')
    source = tracks_from_file(source_path)
    index = defaultdict(list)
    endpoints = []
    for i, track in enumerate(source):
        index[line_hash(track)].append(i)
        endpoints.append(np.r_[track[0], track[-1]])
    endpoints = np.asarray(endpoints)
    tree = cKDTree(np.vstack([endpoints, endpoints[:, [3, 4, 5, 0, 1, 2]]]))
    all_assigned = Counter()
    records = []
    for file in files:
        tracks = tracks_from_file(file)
        matched, unmatched, errors, lengths = 0, 0, [], []
        finite = True
        for track in tracks:
            if len(track) < 2 or not np.isfinite(track).all():
                finite = False
                unmatched += 1
                continue
            lengths.append(float(np.linalg.norm(np.diff(track.astype(float), axis=0), axis=1).sum()))
            ids = index.get(line_hash(track))
            if ids:
                i, delta = ids[0], 0.
            else:
                _, nearest = tree.query(np.r_[track[0], track[-1]], k=min(8, len(endpoints) * 2))
                i, delta = -1, float('inf')
                for candidate in np.atleast_1d(nearest):
                    j = int(candidate) % len(source)
                    original = source[j]
                    if original.shape != track.shape:
                        continue
                    difference = min(float(np.max(np.abs(original - track))), float(np.max(np.abs(original[::-1] - track))))
                    if difference < delta:
                        i, delta = j, difference
            if delta <= tolerance:
                matched += 1
                all_assigned[i] += 1
                errors.append(delta)
            else:
                unmatched += 1
        records.append({'filename': file.name, 'streamlines': len(tracks), 'all_points_finite': finite,
                        'length_mm': stats(lengths), 'source_matched': matched, 'source_unmatched': unmatched,
                        'source_coordinate_delta_mm': stats(errors)})
    geometry_ok = bool(records) and all(r['all_points_finite'] and r['source_unmatched'] == 0 and
                                        (r['streamlines'] == 0 or r['length_mm']['min'] > 0) for r in records)
    endpoint_rules = endpoint_rule_check(manifest, files) if kind == 'wmql_bundle_segmentation' else None
    duplicate_count = sum(v - 1 for v in all_assigned.values())
    checks, not_run = applicable_checks(kind, contract_error, geometry_ok, duplicate_count, endpoint_rules,
        wma_sampling=summary.get('tract_sampling', {}).get('all_tract_sampling_passed'))
    if kind == 'wmql_bundle_segmentation' and manifest.get('official_multibundle'):
        from tractography.wmql_protocol import bundle_assessment
        from tractography.wmql import MIN_BUNDLE_STREAMLINES
        assessment = bundle_assessment(summary, requested, MIN_BUNDLE_STREAMLINES)
        summary['bundle_assessment'] = assessment
        checks['wmql_all_requested_sampling'] = assessment['all_requested_sampling_passed']
    return {'status': 'integrity_pass_anatomical_review_pending' if all(checks.values()) else 'needs_review',
            'subject_id': manifest['subject_id'], 'backend': kind, 'manifest': str(manifest_path.resolve()),
            'source': str(source_path.resolve()), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'input_streamlines': len(source), 'current_contract_error': contract_error,
            'recorded_status': manifest.get('status'), 'output_summary': summary, 'bundles': records,
            'checks': checks, 'checks_not_run': not_run,
            'wmql_endpoint_rules': endpoint_rules,
            'source_match_tolerance_mm': tolerance, 'duplicate_source_assignments': duplicate_count,
            'audit_seconds': time.monotonic() - started,
            'limitations': ['Source membership and QC do not establish correctness of anatomical labels.',
                            'WMA/WMQL may intentionally overlap anatomical queries; only TractCloud is required to be a partition.',
                            'A small WMQL result can satisfy rules yet be inadequate for robust anatomical interpretation.']}


def update_qc_records(manifest_path, result, report_path):
    """Revalidate metadata only; archive old QC/provenance, never alter tracts."""
    from datetime import datetime, timezone
    from shared.product_staging import archive_run_records
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') != 'wma_org_atlas' or not manifest.get('executed'):
        raise ValueError('QC updates currently support only executed WMA records.')
    if manifest.get('status') == 'running':
        # Opus P2 (WMA audit): a second writer of 'completed' on a killed-rerun folder would let _wma_complete accept old products.
        print(f'Refused --update-qc: {manifest_path} has status "running" (a killed or unfinished run); nothing was written.', file=sys.stderr)
        return None
    qc_path = manifest_path.with_name('wma_org_atlas_qc.json')
    qc = json.loads(qc_path.read_text())
    passed = all(result['checks'].values())
    failed_checks = [key for key, value in result['checks'].items() if not value]
    error = result['current_contract_error'] or (None if passed else 'Independent checks failed: ' + ', '.join(failed_checks))
    archive = archive_run_records(manifest_path.parent, (manifest_path.name, qc_path.name))
    record = {'checked_at': datetime.now(timezone.utc).isoformat(), 'previous_records': archive,
              'original_recorded_status': qc.get('status'), 'source_sha256': result['source_sha256'],
              'audit_report': str(Path(report_path).resolve()),
              'audit_sha256': hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
              'scope': 'Current automatic sampling/identity/source-geometry contract; NOT anatomical certification'}
    from tractography.wma import REQUIRED_MAJOR_TRACTS, MIN_MAJOR_TRACT_STREAMLINES, WMA_QC_VERSION
    qc.update(status='completed' if passed else 'partial_failed', automatic_qc_passed=passed,
              version=WMA_QC_VERSION, tract_sampling=result['output_summary'].get('tract_sampling', {}),
              quantification_eligible=False, human_review='pending', revalidation=record,
              error=error, output_summary=result['output_summary'])
    qc.setdefault('acceptance_criteria', {}).update(required_major_tracts=list(REQUIRED_MAJOR_TRACTS),
                         minimum_streamlines_per_major_tract=MIN_MAJOR_TRACT_STREAMLINES)
    manifest.update(status=qc['status'], error=error, qc_revalidation=record)
    manifest.setdefault('outputs', {})['summary'] = result['output_summary']
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + '\n')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--tolerance-mm', type=float, default=.001)
    parser.add_argument('--previous-audit', type=Path)
    parser.add_argument('--update-qc', action='store_true', help='Archive and revalidate executed WMA QC metadata; tract files remain unchanged')
    args = parser.parse_args()
    result = audit(args.manifest, args.tolerance_mm)
    if args.previous_audit:
        previous = json.loads(args.previous_audit.read_text())
        if previous['source_sha256'] != result['source_sha256']:
            raise ValueError('Repeat comparison requires the identical source tractogram.')
        def assignments(record):
            summary = record['output_summary']
            files = summary.get('anatomical_tract_files') or summary.get('bundle_files') or []
            assigned = defaultdict(set)
            for file in files:
                path = Path(file)
                for track in tracks_from_file(path):
                    assigned[line_hash(track)].add(path.stem)
            return assigned
        before, after = assignments(previous), assignments(result)
        common = before.keys() & after.keys()
        changed = sum(before[key] != after[key] for key in common)
        union = before.keys() | after.keys()
        result['repeat_comparison'] = {
            'previous_audit': str(args.previous_audit), 'assigned_before': len(before), 'assigned_after': len(after),
            'shared_assignments': len(common), 'changed_labels_in_shared': changed,
            'only_before': len(before.keys() - after.keys()), 'only_after': len(after.keys() - before.keys()),
            'same_labels_on_assigned_union_fraction': (len(common) - changed) / len(union) if union else None,
            'exact_same_assignment_sets': before == after,
            'note': 'Identity of emitted tracks/labels, not anatomical accuracy; excluded Other tracks are not emitted.',
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    refused = args.update_qc and update_qc_records(args.manifest, result, args.report) is None
    print(json.dumps({k: result[k] for k in ('status', 'input_streamlines', 'current_contract_error', 'checks', 'audit_seconds')}, indent=2))
    if refused:
        return 2  # Opus P2 (fix round): the refusal is reported on stderr; the exit code must not say success
    return 0 if all(result['checks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
