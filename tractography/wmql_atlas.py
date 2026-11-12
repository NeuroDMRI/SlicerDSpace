"""Explicit experimental DDParcel-to-WMQL white-matter territory preparation.

Native cortical and subcortical labels remain untouched. This volume-based
nearest-cortex construction is distinct from FreeSurfer's surface-based wmparc.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

METHOD = 'ddparcel_adjacent_wm_5mm.v1'
RADIUS_MM = 5.0
SUBCORTICAL = {0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54,
               58, 60, 63, 192}
CORTICAL = {base + number for base in (1000, 2000) for number in range(1, 36) if number != 4}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def nearest_labels(tree, seed_labels, points):
    """Exact physical nearest neighbour; equal-distance ties choose the lowest ID."""
    import numpy as np

    distances, indices = tree.query(points, k=2, eps=0, workers=-1)
    labels = seed_labels[indices[:, 0]].copy()
    ties = np.isclose(distances[:, 0], distances[:, 1], atol=1e-10, rtol=0)
    for index in np.flatnonzero(ties & (distances[:, 0] <= RADIUS_MM)):
        neighbours = np.asarray(tree.query_ball_point(points[index], distances[index, 0] + 1e-10))
        squared = np.sum((tree.data[neighbours] - points[index]) ** 2, axis=1)
        equal = np.isclose(squared, squared.min(), atol=1e-9, rtol=0)
        labels[index] = seed_labels[neighbours[equal]].min()
    return distances[:, 0], labels, int(ties.sum())


def derive_territories(image):
    import nibabel as nib
    import numpy as np
    from scipy.spatial import cKDTree

    data = np.asarray(image.dataobj)
    affine = image.affine
    if (data.ndim != 3 or not np.isfinite(data).all() or not np.all(data == np.floor(data))
            or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError('Expected finite integer 3-D labels and an invertible affine')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Experimental WMQL territory preparation requires explicit millimetre units')
    unexpected = set(map(int, np.unique(data))) - SUBCORTICAL - CORTICAL
    if unexpected:
        raise ValueError('Not the supported DDParcel label schema: ' + ', '.join(map(str, sorted(unexpected))))
    output = data.astype(np.int16)
    counts, wm_trees = {}, []
    for hemi, wm, base, deep in (('left', 2, 1000, 5001), ('right', 41, 2000, 5002)):
        voxels = np.argwhere(data == wm)
        cortex = np.argwhere((data > base) & (data <= base + 35) & (data != base + 4))
        if len(voxels) == 0 or len(cortex) < 2:
            raise ValueError('Missing cerebral WM or cortical seeds in the ' + hemi + ' hemisphere')
        points = nib.affines.apply_affine(affine, voxels)
        wm_trees.append(cKDTree(points))
        seed_labels = data[tuple(cortex.T)].astype(np.int16)
        tree = cKDTree(nib.affines.apply_affine(affine, cortex))
        distances, labels, ties = nearest_labels(tree, seed_labels, points)
        regional = distances <= RADIUS_MM
        output[tuple(voxels.T)] = np.where(regional, labels + 2000, deep)
        counts[hemi] = dict(generic_wm_voxels=len(voxels), regional_voxels=int(regional.sum()),
                            unsegmented_voxels=int((~regional).sum()), nearest_seed_ties=ties)
    callosal = np.argwhere(data == 192)
    if len(callosal):
        points = nib.affines.apply_affine(affine, callosal)
        dl, dr = (tree.query(points, eps=0, workers=-1)[0] for tree in wm_trees)
        # Equidistant central voxels retain the original label rather than an arbitrary side.
        equal = np.isclose(dl, dr, atol=1e-10, rtol=0)
        labels = np.where(equal, 192, np.where(dl < dr, 3004, 4004))
        output[tuple(callosal.T)] = labels
        counts['callosal'] = {str(label): int((labels == label).sum()) for label in (192, 3004, 4004)}
    protected = ~np.isin(data, (2, 41, 192))
    if not np.array_equal(data[protected], output[protected]):
        raise AssertionError('Non-WM source labels changed')
    return output, counts


def prepared_atlas_current(record, source=None):
    """Verify provenance and decoded geometry before reusing a derived atlas."""
    import nibabel as nib
    import numpy as np

    try:
        source = Path(source or record['source_path']).expanduser().resolve()
        output = Path(record['output_path'])
        if (record['method'] != METHOD or record['radius_mm'] != RADIUS_MM
                or record['implementation_sha256'] != digest(__file__)
                or record['source_sha256'] != digest(source)
                or record['output_sha256'] != digest(output)):
            return False
        original, prepared = nib.load(source), nib.load(output)
        return (original.shape == prepared.shape == tuple(record['shape'])
                and np.array_equal(original.affine, prepared.affine)
                and prepared.header.get_xyzt_units()[0] == 'mm'
                and np.array_equal(original.affine, record['affine']))
    except (OSError, KeyError, TypeError, ValueError, nib.filebasedimages.ImageFileError):
        return False


def prepare_ddparcel_atlas(source, cache_dir):
    """Content-addressed, locked preparation; corrupt cache files are regenerated."""
    import nibabel as nib
    import numpy as np
    from shared.cache_lock import cache_lock

    start = time.monotonic()
    source = Path(source).expanduser().resolve()
    source_sha, implementation_sha = digest(source), digest(__file__)
    key = hashlib.sha256((METHOD + source_sha + implementation_sha).encode()).hexdigest()
    directory = Path(cache_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    output, receipt = directory / 'wmql_territories.nii.gz', directory / 'atlas_preparation.json'
    with cache_lock(directory / '.lock'):
        try:
            record = json.loads(receipt.read_text(encoding='utf-8'))
            if prepared_atlas_current(record, source):
                return output, dict(record, cache_reused=True)
        except (OSError, ValueError):
            pass
        image = nib.load(source)
        data, counts = derive_territories(image)
        with tempfile.TemporaryDirectory(prefix='build-', dir=directory) as staging:
            candidate = Path(staging) / 'wmql_territories.nii.gz'
            header = image.header.copy()
            header.set_data_dtype(np.int16)
            header.set_slope_inter(1, 0)
            prepared = nib.Nifti1Image(data, image.affine, header)
            prepared.set_qform(image.get_qform(), int(image.header['qform_code']))
            prepared.set_sform(image.get_sform(), int(image.header['sform_code']))
            nib.save(prepared, candidate)
            if digest(source) != source_sha:
                raise ValueError('Source parcellation changed during WMQL atlas preparation')
            record = dict(method=METHOD, experimental=True, radius_mm=RADIUS_MM,
                          source_path=str(source), source_sha256=source_sha,
                          implementation_sha256=implementation_sha, output_path=str(output.resolve()),
                          output_sha256=digest(candidate), shape=list(data.shape), affine=image.affine.tolist(),
                          counts=counts, protected_labels_unchanged=True,
                          distance_definition='Euclidean distance between voxel centres in scanner RAS mm',
                          callosum_definition='Nearest original WM hemisphere; equal distances retain 192',
                          anatomical_equivalence_verified=False, human_review='pending',
                          cache_reused=False, preparation_seconds=time.monotonic() - start)
            staging_receipt = Path(staging) / 'atlas_preparation.json'
            staging_receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            os.replace(candidate, output)
            os.replace(staging_receipt, receipt)
        if not prepared_atlas_current(record, source):
            raise ValueError('Prepared atlas failed its provenance/geometry check')
        return output, record
