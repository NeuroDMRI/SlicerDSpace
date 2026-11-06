"""Exercise the official UKF CLI with isolated invalid b0 voxels, without a GUI."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(destination, binary):
    import nibabel as nib
    import numpy as np
    from shared.pipeline import RawInputBundle
    from tractography.ukf import current, digest, paths, run_ukf

    destination.mkdir(parents=True, exist_ok=False)
    gradients = np.random.default_rng(42).normal(size=(30, 3))
    gradients /= np.linalg.norm(gradients, axis=1)[:, None]
    gradients = np.vstack((np.zeros((1, 3)), gradients))
    b = np.array([0]+[1000]*30)
    signal = 1000*np.exp(-b*np.einsum('ni,ij,nj->n', gradients,
                                     np.diag([0.0017, 0.0003, 0.0003]), gradients))
    data = np.broadcast_to(signal, (14, 14, 14, 31)).astype(np.float32).copy()
    data[1, 1, 1, 0] = 0
    data[12, 12, 12, 0] = 0
    mask = np.zeros((14, 14, 14), np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    affine = np.array([[0, -2, 0, 14], [2, 0, 0, -14], [0, 0, 2, -14], [0, 0, 0, 1.]])
    nib.save(nib.Nifti1Image(data, affine), destination/'data.nii.gz')
    nib.save(nib.Nifti1Image(mask, affine), destination/'mask.nii.gz')
    np.savetxt(destination/'bvals', b[None])
    np.savetxt(destination/'bvecs', gradients.T)
    raw = RawInputBundle(destination, 'synthetic', destination, destination/'data.nii.gz',
                         destination/'bvals', destination/'bvecs', destination/'mask.nii.gz')
    before = {str(p): digest(p) for p in (raw.dwi_path, raw.mask_path, raw.bval_path, raw.bvec_path)}
    options = dict(numThreads=1)
    result = run_ukf(raw, destination/'outputs', options=options, executable=binary, log=print)
    outputs = paths('synthetic', destination/'outputs')
    qc = json.loads(outputs['qc'].read_text())
    manifest = json.loads(outputs['manifest'].read_text())
    effective = nib.load(outputs['effective_mask'])
    assert result['status'] == 'completed' and qc['automatic_qc_passed']
    assert qc['input_validation']['excluded_nonpositive_b0_voxels'] == 2
    assert np.asanyarray(effective.dataobj).sum() == mask.sum()-2
    assert np.array_equal(effective.affine, affine)
    assert before == {p: digest(p) for p in before}
    assert current(raw, destination/'outputs', options=options, executable=binary)
    receipt = dict(status='passed', scope='Synthetic native-grid input-adapter/official-CLI smoke, not cohort performance or anatomical validation',
                   streamline_count=qc['streamline_count'], excluded_voxels=2,
                   valid_voxels=int(np.asanyarray(effective.dataobj).sum()),
                   sources_unchanged=True, cache_current=True, automatic_qc=qc['checks'],
                   vtk_trk_verified_streamlines=qc['vtk_trk_verified_streamlines'],
                   implementation_sha256=digest(ROOT/'tractography/ukf.py'),
                   runtime_seconds=manifest['runtime_seconds'], executable=manifest['signature']['executable'])
    (destination/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ukf-executable', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.ukf_executable)
