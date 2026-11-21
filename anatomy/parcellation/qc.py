"""Decoded DDParcel output QC; integrity is not anatomical accuracy."""

import json
from pathlib import Path


def label_semantics(values) -> dict:
    """Describe DK label content without certifying anatomy or FS equivalence."""
    import numpy as np
    labels = np.asarray(values)
    cortical = ((labels >= 1001) & (labels <= 1035)) | ((labels >= 2001) & (labels <= 2035))
    left_wm = (labels >= 3001) & (labels <= 3035)
    right_wm = (labels >= 4001) & (labels <= 4035)
    regional_wm = left_wm | right_wm
    missing = bool(cortical.any() and not regional_wm.any())
    callosal = np.isin(labels, (192, 251, 252, 253, 254, 255, 1004, 2004, 3004, 4004))
    return {'schema': 'dk_label_content.v1', 'cortical_region_voxels': int(cortical.sum()),
            'regional_white_matter_voxels': int(regional_wm.sum()),
            'left_regional_white_matter_voxels': int(left_wm.sum()),
            'right_regional_white_matter_voxels': int(right_wm.sum()),
            'generic_cerebral_white_matter_voxels': int(np.count_nonzero((labels == 2) | (labels == 41))),
            'regional_white_matter_label_ids': [int(value) for value in np.unique(labels[regional_wm])],
            'corpus_callosum_voxels': int(callosal.sum()),
            'corpus_callosum_label_ids': [int(value) for value in np.unique(labels[callosal])],
            'corpus_callosum_192_voxels': int(np.count_nonzero(labels == 192)),
            'cortical_labels_without_regional_wm': missing,
            'scope': 'DK label content only; neither anatomical validation nor FreeSurfer wmparc equivalence',
            'warnings': (['Cortical/aseg labels are present without DK regional WM territories (3000/4000 series). '
                          'This is expected for the DDParcel model, which predicts generic left/right WM (2/41). '
                          'The legacy wmparc filename does not establish FreeSurfer wmparc equivalence; '
                          'WMQL crossing/only rules need compatible label territories.'] if missing else [])}


def validate_outputs(subject_id: str, output_dir: Path, native_mask: Path) -> dict:
    import nibabel as nib
    import numpy as np
    from microstructure.pipeline import _file_signature, _write_json

    output_dir = Path(output_dir)
    qc_path = output_dir / "ddparcel_qc.json"
    paths = [output_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz",
             output_dir / f"{subject_id}-DDSurfer-wmparc-Reg.mgz"]
    paths.extend(output_dir / f"{subject_id}-dti-{metric}-Reg-NormMasked.nii.gz"
                 for metric in ("FractionalAnisotropy", "Trace", "MinEigenvalue", "MidEigenvalue"))
    _write_json(qc_path, {"status": "running", "automatic_qc_passed": False})
    try:
        images = [nib.load(str(path)) for path in paths]
        arrays = [np.asarray(image.dataobj) for image in images]
        mask_img = nib.load(str(native_mask))
        mask_data = np.asarray(mask_img.dataobj)
        mask = mask_data > 0
        native = arrays[0]
        grid = native.shape == mask.shape and np.allclose(images[0].affine, mask_img.affine, rtol=0, atol=1e-4)
        checks = {
            "finite_nonempty_brain_mask": bool(np.isfinite(mask_data).all() and mask.any()),
            "all_products_decoded_finite_3d": all(array.ndim == 3 and bool(np.isfinite(array).all()) for array in arrays),
            "native_grid_matches_mask": bool(grid),
            "integer_nonnegative_labels": all(bool(np.all((array >= 0) & (array == np.floor(array)))) for array in arrays[:2]),
            "nonempty_labels_in_brain": bool(grid and np.any(native[mask] > 0)),
            "nonempty_registered_labels": bool(np.any(arrays[1] > 0)),
        }
        semantics = label_semantics(native)
        payload = {"kind": "ddparcel_output_qc", "version": 1, "subject_id": subject_id,
                   "status": "completed" if all(checks.values()) else "failed",
                   "automatic_qc_passed": all(checks.values()), "human_review": "pending",
                   "checks": checks, "source_signature": {"native_mask": _file_signature(Path(native_mask))},
                   "output_artifacts": {str(path.resolve()): _file_signature(path) for path in paths},
                   "label_ids": [int(value) for value in np.unique(native) if np.isfinite(value)],
                   "label_semantics": semantics,
                   "warnings": ["Decoded geometry/label integrity only. Atlas coverage and anatomical accuracy require independent review.",
                                *semantics['warnings']]}
        _write_json(qc_path, payload)
        if not payload["automatic_qc_passed"]:
            raise ValueError("DDParcel output QC failed: " + ", ".join(key for key, value in checks.items() if not value))
        return payload
    except Exception as exc:
        if not qc_path.is_file() or json.loads(qc_path.read_text(encoding="utf-8")).get("status") != "failed":
            _write_json(qc_path, {"kind": "ddparcel_output_qc", "status": "failed",
                                  "automatic_qc_passed": False, "subject_id": subject_id, "error": str(exc)})
        raise
