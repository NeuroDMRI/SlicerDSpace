#!/usr/bin/env python3
"""Audit query coverage against the actual atlas and retained WMQL evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tractography.wmql_protocol import OFFICIAL_DIR, official_query_names


def audit(manifest_path, source_coverage=False):
    import nibabel as nib
    import numpy as np

    manifest = json.loads(manifest_path.read_text())
    atlas = nib.load(manifest['atlas_path'])
    labels = np.asarray(atlas.dataobj)
    names = official_query_names()
    def digest(path):
        hasher = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    record = {
        'schema': 'wmql_query_coverage.v1',
        'query_sha256': hashlib.sha256((OFFICIAL_DIR / 'freesurfer_queries.qry').read_bytes()).hexdigest(),
        'official_outputs': list(names),
        'dedicated_cerebellar_output_queries': [],
        'cb_definition': 'cingulum bundle, not cerebellum',
        'atlas_sha256': digest(manifest['atlas_path']),
        'source_tractogram_sha256': digest(manifest['input_tractography']),
        'cerebellar_label_voxels': {str(label): int(np.count_nonzero(labels == label)) for label in (7, 8, 46, 47)},
        'finding': 'Cerebellar labels are present; the official output-query inventory has no dedicated cerebellar bundle rules.',
        'action': 'Keep the official rules and scientific display unchanged. Use a cerebellar-capable atlas route when these bundles are requested.',
        'source': 'https://github.com/demianw/tract_querier/blob/5c69c594d489859e08485110678dca7f6f4cba40/tract_querier/data/freesurfer_queries.qry',
    }
    if any(any(term in name for term in ('cerebell', 'icp', 'mcp', 'scp')) for name in names):
        raise ValueError('The upstream inventory changed; reassess the cerebellar coverage finding')
    if source_coverage:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(manifest['input_tractography'])
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfLines() == 0:
            raise ValueError('Source tractogram is empty or unreadable')
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        inverse = np.linalg.inv(atlas.affine)
        hits = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 500000):
            stop = min(start + 500000, len(points))
            voxels = np.rint(nib.affines.apply_affine(inverse, points[start:stop])).astype(np.int64)
            inside = np.all((voxels >= 0) & (voxels < np.array(labels.shape)), axis=1)
            values = np.zeros(stop - start, dtype=labels.dtype)
            values[inside] = labels[tuple(voxels[inside].T)]
            hits[start:stop] = np.isin(values, (7, 8, 46, 47))
        cells = polydata.GetLines()
        offsets = vtk_to_numpy(cells.GetOffsetsArray())
        connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        per_track = np.zeros(len(offsets) - 1, dtype=bool)
        for index, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            per_track[index] = bool(hits[connectivity[a:b]].any())
        record['source_tractogram'] = {
            'streamlines': int(polydata.GetNumberOfLines()),
            'streamlines_visiting_cerebellar_labels': int(per_track.sum()),
            'coordinates': 'RAS millimetres to atlas voxel centres, nearest-neighbour labels',
            'scope': 'Input coverage check only; no query redefinition or anatomical acceptance',
        }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-coverage', action='store_true')
    args = parser.parse_args()
    record = audit(args.manifest, args.source_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'official_outputs'}, indent=2))


if __name__ == '__main__':
    main()
