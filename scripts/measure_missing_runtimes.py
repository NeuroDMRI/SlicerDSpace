#!/usr/bin/env python3
"""Time previously unrecorded operations on existing real inputs, in new paths."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Measured outputs must be created in a new directory.')
    args.output_dir.mkdir(parents=True)
    from shared.input_import import InputImportConfig, prepare_inputs
    from microstructure.pipeline import _compute_roi_stats
    from scripts.generate_protocol_benchmarks import _report_generation_system

    source_root = Path('/data/chengjin/SlicerDDSurfer-validation/dicom_import_publishable_20260725')
    native = source_root / 'subject' / 'input'
    original = native / 'data.nii.gz'
    old_hash = sha(original)
    source_img = nib.load(str(original))
    records = []
    for kind, cfg in (
        ('nifti', dict(source=original, backend='copy', bval_path=native / 'bvals', bvec_path=native / 'bvecs')),
        ('dicom', dict(source=source_root / 'dicom_source' / 'ap', backend='dcm2niix',
                       dcm2niix_executable=Path('/data/chengjin/software/dcm2niix/dcm2niix'))),
        ('nrrd', dict(source=ROOT / 'ignore/validation/slicer_acceptance_20261006/input_verified_gui/input_preparation/real_dwi.nrrd',
                      backend='dwiconvert', dwiconvert_executable=Path('/data/chengjin/software/Slicer-5.10.0-linux-amd64/lib/Slicer-5.10/cli-modules/DWIConvert'))),
    ):
        manifest = prepare_inputs(InputImportConfig(subject_id='sub-dicom', source_kind=kind,
                                                   subject_root=args.output_dir / kind, **cfg))
        img = nib.load(manifest['outputs']['dwi'])
        bvals = np.loadtxt(manifest['outputs']['bvals']).reshape(-1)
        bvecs = np.loadtxt(manifest['outputs']['bvecs'])
        if bvecs.shape == (len(bvals), 3):
            bvecs = bvecs.T
        checks = dict(shape_matches=img.shape == source_img.shape, finite_gradient_table=bool(np.isfinite(bvecs).all()),
                      bvals_match=bool(np.allclose(bvals, np.loadtxt(native / 'bvals'), atol=1e-3)),
                      voxel_affine_matches=bool(np.allclose(img.affine, source_img.affine, atol=1e-3)))
        assert all(checks.values()), (kind, checks)
        records.append(dict(module='input.' + kind, subject_id='sub-dicom', status='completed',
                            runtime_seconds=manifest['runtime_seconds'], shape=list(img.shape),
                            source=str(cfg['source']), checks=checks,
                            source_record=str((args.output_dir / kind / 'input' / 'input_import_manifest.json').resolve()),
                            scope=manifest['timing_scope']))

    subject = ROOT / 'ignore/validation/slicer_acceptance_20261006/science/subjects/100307/outputs'
    micro = subject / '02_microstructure'
    metrics = {
        'FA': micro / 'dti/100307_dti_FA.nii.gz', 'MD': micro / 'dti/100307_dti_MD.nii.gz',
        'FWF': micro / 'freewater/100307_fw_fraction.nii.gz', 'FW_FA': micro / 'freewater/100307_fw_FA.nii.gz',
        'NDI': micro / 'noddi/100307_noddi_NDI.nii.gz', 'ODI': micro / 'noddi/100307_noddi_ODI.nii.gz',
    }
    masks = {}
    for name in ('FW_FA', 'NDI', 'ODI'):
        m = micro / ('freewater/100307_fw_tissue_valid_mask.nii.gz' if name == 'FW_FA' else 'noddi/100307_noddi_tissue_valid_mask.nii.gz')
        masks[name] = np.asarray(nib.load(str(m)).dataobj) > 0
    from shared.io_layout import anatomy_input_dir
    atlas = anatomy_input_dir(subject, 'ddparcel') / '100307-DDSurfer-wmparc.nii.gz'
    started = time.monotonic()
    count = _compute_roi_stats(metric_paths=metrics, roi_label_map=atlas, lut_path=None,
                               csv_path=args.output_dir / 'roi_stats.csv', json_path=args.output_dir / 'roi_stats.json',
                               validity_masks=masks)
    seconds = time.monotonic() - started
    stats_record = json.loads((args.output_dir / 'roi_stats.json').read_text())
    assert count > 0 and all(np.isfinite(row['FA_mean']) for row in stats_record['rows'] if row.get('FA_mean') is not None)
    records.append(dict(module='statistics.regional_microstructure', subject_id='100307', status='completed',
                        runtime_seconds=seconds, roi_count=count, source=str(atlas), source_record=str((args.output_dir / 'roi_stats.json').resolve()),
                        scope='Read six real scalar maps and DDParcel labels; apply FW/NDI/ODI validity masks; write ROI statistics. No model refitting.'))
    assert sha(original) == old_hash, 'Source input changed'
    result = dict(status='passed', measured_at=datetime.now(timezone.utc).isoformat(), records=records,
                  source_dwi_sha256=old_hash, source_unchanged=True, system=_report_generation_system())
    (args.output_dir / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
