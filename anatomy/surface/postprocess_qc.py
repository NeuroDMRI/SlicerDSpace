"""Decoded integrity checks, not anatomical certification of surface outputs."""

from pathlib import Path

from shared.scalar_qc import _signature


def postprocess_output_paths(subject_dir, hemispheres, atlases, stats_dir=None):
    root = Path(subject_dir)
    stats = Path(stats_dir) if stats_dir is not None else root / "stats"
    return [root / "mri/brain.mgz",
            *(root / "surf" / f"{hemi}.{name}" for hemi in hemispheres for name in
              ("white", "pial", "inflated", "sphere", "sphere.reg", "thickness", "curv", "sulc", "area")),
            *(root / "label" / f"{hemi}.{atlas}.annot" for hemi in hemispheres for atlas in atlases),
            *(stats / f"{hemi}.{atlas}.stats" for hemi in hemispheres for atlas in atlases)]


def validate_postprocess_outputs(subject_dir, hemispheres, atlases, stats_dir=None):
    import nibabel as nib
    import numpy as np

    subject_dir = Path(subject_dir)
    stats_dir = Path(stats_dir) if stats_dir is not None else subject_dir / "stats"
    checks, summaries, artifacts = {}, {}, {}

    def remember(path):
        artifacts[str(path.resolve())] = _signature(path)

    brain_path = subject_dir / "mri/brain.mgz"
    brain = nib.load(brain_path)
    checks["brain_reference_geometry"] = bool(len(brain.shape) == 3 and np.isfinite(brain.affine).all()
        and abs(np.linalg.det(brain.affine[:3, :3])) > 1e-12)
    remember(brain_path)
    for hemi in hemispheres:
        reference_vertices, reference_faces = None, None
        details = {}
        for kind in ("white", "pial", "inflated", "sphere", "sphere.reg"):
            path = subject_dir / "surf" / f"{hemi}.{kind}"
            vertices, faces = nib.freesurfer.read_geometry(path)
            finite = (vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0
                      and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                      and np.isfinite(vertices).all() and np.all((faces >= 0) & (faces < len(vertices))))
            checks[f"{hemi}.{kind}.finite_mesh"] = bool(finite)
            if reference_vertices is None:
                reference_vertices, reference_faces = vertices, faces
            checks[f"{hemi}.{kind}.correspondence"] = (vertices.shape == reference_vertices.shape
                                                        and bool(np.array_equal(faces, reference_faces)))
            remember(path)
        for name in ("thickness", "curv", "sulc", "area"):
            path = subject_dir / "surf" / f"{hemi}.{name}"
            values = nib.freesurfer.read_morph_data(path)
            checks[f"{hemi}.{name}.finite_vertex_values"] = bool(values.size == len(reference_vertices)
                                                                  and np.isfinite(values).all())
            if name in {"thickness", "area"}:
                checks[f"{hemi}.{name}.nonnegative_nonempty"] = bool(np.all(values >= 0) and np.any(values > 0))
            details[name] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}
            remember(path)
        for atlas in atlases:
            path = subject_dir / "label" / f"{hemi}.{atlas}.annot"
            labels, color_table, names = nib.freesurfer.read_annot(path)
            checks[f"{hemi}.{atlas}.vertex_labels"] = bool(len(labels) == len(reference_vertices)
                and len(names) > 1 and len(color_table) == len(names) and np.all((labels >= -1) & (labels < len(names)))
                and np.any(labels > 0))
            remember(path)
            path = stats_dir / f"{hemi}.{atlas}.stats"
            rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
            checks[f"{hemi}.{atlas}.finite_statistics"] = bool(rows) and all(
                len(row) > 1 and np.isfinite(np.asarray(row[1:], dtype=float)).all() for row in rows)
            remember(path)
        summaries[hemi] = dict(vertices=len(reference_vertices), faces=len(reference_faces), metrics=details)
    passed = all(checks.values())
    return {"kind": "surface_postprocess_integrity", "version": 1,
            "status": "completed" if passed else "failed", "automatic_qc_passed": passed,
            "checks": checks, "output_artifacts": artifacts, "hemispheres": summaries,
            "human_review": "pending", "scope": "Decoded geometry, correspondence, morphometry and atlas/statistics integrity; not anatomical accuracy"}
