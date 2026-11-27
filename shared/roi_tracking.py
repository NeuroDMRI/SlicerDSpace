"""Standalone DDParcel-guided ROI tractography (side task under outputs/03_tractography/roi_tracking).

Supports all four SlicerDSpace tractography algorithms:
- ``ukf``: official UKFTractography CLI with ``--seedsFile`` / ``--seedLabels 1``
- ``csd_act``: MRtrix3 ``tckgen -seed_image`` (reusing cached FOD and 5TT when available)
- ``dti_det``: DIPY deterministic EuDX tracking seeded inside the ROI mask
- ``saf``: short-range ROI-seeded association fiber tracking (10-80 mm)

Outputs are stored under ``outputs/03_tractography/roi_tracking/<algorithm>/<region_slug>/``
and never feed into whole-brain bundle segmentation (WMQL/WMA/TractCloud) or connectomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Optional

ROI_TRACKING_MANIFEST_KIND = "slicerdspace_roi_tracking"
SUPPORTED_ROI_ALGORITHMS = ("ukf", "csd_act", "dti_det", "saf")
ROI_ALGORITHM_LABELS = {
    "ukf": ("UKF", "UKF"),
    "csd_act": ("CSD / ACT", "CSD / ACT"),
    "dti_det": ("DTI", "DTI"),
    "saf": ("CSD + SAF", "CSD + SAF"),
}


@dataclass(frozen=True)
class ROITrackingConfig:
    subject_id: str
    subject_root: Path
    parcellation_path: Path
    algorithm: str
    label_ids: tuple[int, ...]
    region_names: tuple[str, ...] = ()
    output_root: Optional[Path] = None
    device: str = "auto"
    select_streamlines: int = 10000
    dilate_cortical_voxels: int = 1
    ukf_parameters: Optional[dict[str, Any]] = None
    ukf_executable: Optional[Path] = None
    dwiconvert_executable: Optional[Path] = None
    csd_act_parameters: Optional[dict[str, Any]] = None
    dti_parameters: Optional[dict[str, Any]] = None
    saf_parameters: Optional[dict[str, Any]] = None
    mrtrix_bin_dir: Optional[Path] = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


def region_slug(label_ids: Iterable[int], region_names: Iterable[str] = ()) -> str:
    """Build a filesystem-safe, human-readable folder slug for the selected ROI region(s)."""
    cleaned_names: list[str] = []
    for raw in region_names:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "").strip()).strip("_.-")
        if token and token not in cleaned_names:
            cleaned_names.append(token)
    ids = [int(v) for v in dict.fromkeys(label_ids)]
    if cleaned_names:
        base = "__".join(cleaned_names[:3])
        if len(cleaned_names) > 3:
            base += f"_plus{len(cleaned_names) - 3}"
        if len(base) > 64:
            base = base[:64].rstrip("_.-")
        return base
    if ids:
        joined = "_".join(str(v) for v in ids[:4])
        if len(ids) > 4:
            joined += f"_plus{len(ids) - 4}"
        return f"roi_{joined}"
    return "roi_custom"


def load_label_names(
    parcellation_path: Path,
    color_table_path: Optional[Path] = None,
) -> dict[int, str]:
    """Read integer label -> region name mapping from sibling .ctbl or FreeSurferColorLUT.txt."""
    from shared.result_scene import FREESURFER_LUT

    names: dict[int, str] = {}
    candidates: list[Path] = []
    if color_table_path is not None and Path(color_table_path).is_file():
        candidates.append(Path(color_table_path))
    parc = Path(parcellation_path)
    stem = parc.name
    for suffix in (".nii.gz", ".nii", ".mgz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for sibling in (parc.with_name(stem + ".ctbl"), parc.with_name(stem + "_lut.txt")):
        if sibling.is_file() and sibling not in candidates:
            candidates.append(sibling)
    if FREESURFER_LUT.is_file() and FREESURFER_LUT not in candidates:
        candidates.append(FREESURFER_LUT)

    for lut_path in candidates:
        try:
            for raw_line in lut_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    label_id = int(parts[0])
                except ValueError:
                    continue
                if label_id not in names:
                    names[label_id] = parts[1]
        except OSError:
            continue
    return names


def resolve_ddparcel_outputs(
    subject_root: Optional[Path],
    subject_id: str,
    *,
    output_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Locate a completed DDParcel parcellation (+ optional color table) for ``subject_id``."""
    from shared.io_layout import subject_layout

    roots: list[Path] = []
    if output_root is not None:
        out_p = Path(output_root).expanduser()
        if out_p.is_dir():
            roots.append(out_p)
            if out_p.name.lower() not in {"output", "outputs"} and (out_p / "outputs").is_dir():
                roots.append(out_p / "outputs")
    if subject_root is not None:
        sub_p = Path(subject_root).expanduser()
        if sub_p.is_dir():
            layout = subject_layout(sub_p, output_root=output_root)
            for cand in (layout.outputs, sub_p / "outputs"):
                if cand.is_dir() and cand not in roots:
                    roots.append(cand)

    sid = str(subject_id or "").strip()
    candidates: list[Path] = []
    for root in roots:
        parc_dir = root / "01_anatomical_priors" / "brain_parcellation"
        if parc_dir.is_dir():
            if sid:
                exact = parc_dir / f"{sid}_parcellation.nii.gz"
                if exact.is_file():
                    candidates.append(exact)
            for f in sorted(parc_dir.glob("*_parcellation.nii.gz")):
                if f not in candidates:
                    candidates.append(f)
        # Fallback to stage cache if outputs folder didn't publish yet.
        workspace = root.parent if root.name.lower() in {"output", "outputs"} else root
        stage_dir = workspace / "cache" / "stages" / "01_anatomical_priors" / "brain_parcellation"
        if stage_dir.is_dir():
            for f in sorted(stage_dir.glob("*-DDSurfer-wmparc.nii.gz")) + sorted(stage_dir.glob("*wmparc*.nii.gz")):
                if f not in candidates:
                    candidates.append(f)

    if not candidates:
        return None
    parc = candidates[0].resolve()
    stem = parc.name[:-7] if parc.name.lower().endswith(".nii.gz") else parc.stem
    ctbl = parc.with_name(stem + ".ctbl")
    return {
        "parcellation_path": parc,
        "color_table_path": ctbl if ctbl.is_file() else None,
        "subject_id": sid or stem.removesuffix("_parcellation"),
        "output_root": parc.parents[2] if len(parc.parents) >= 3 and parc.parent.name == "brain_parcellation" else None,
    }


def build_roi_seed_mask(
    parcellation_path: Path,
    label_ids: Iterable[int],
    brain_mask_path: Optional[Path],
    output_path: Path,
    *,
    dilate_cortical_voxels: int = 1,
) -> dict[str, Any]:
    """Create a binary uint8 NIfTI ROI seed mask for ``label_ids`` in parcellation space."""
    import nibabel as nib
    import numpy as np

    ids = sorted({int(v) for v in label_ids if int(v) > 0})
    if not ids:
        raise ValueError("Select at least one non-zero DDParcel brain region.")

    parc_img = nib.load(str(Path(parcellation_path).expanduser().resolve()))
    parc_data = np.rint(np.asarray(parc_img.get_fdata())).astype(np.int32)
    roi_mask = np.isin(parc_data, ids)
    raw_voxels = int(roi_mask.sum())
    if raw_voxels == 0:
        raise ValueError(f"Selected DDParcel label(s) {ids} contain 0 voxels in {parcellation_path}.")

    # Cortical GM labels (>=1000) sit on the grey-matter ribbon; a 1-voxel 6-connected dilation
    # into the brain mask reaches the adjacent WM boundary so seeding yields anatomically faithful tracks.
    has_cortical = any(label_id >= 1000 for label_id in ids)
    dilated = False
    if has_cortical and int(dilate_cortical_voxels) > 0:
        grown = roi_mask.copy()
        for _ in range(int(dilate_cortical_voxels)):
            step = grown.copy()
            step[1:, :, :] |= grown[:-1, :, :]
            step[:-1, :, :] |= grown[1:, :, :]
            step[:, 1:, :] |= grown[:, :-1, :]
            step[:, :-1, :] |= grown[:, 1:, :]
            step[:, :, 1:] |= grown[:, :, :-1]
            step[:, :, :-1] |= grown[:, :, 1:]
            grown = step
        roi_mask = grown
        dilated = True

    if brain_mask_path is not None and Path(brain_mask_path).is_file():
        mask_img = nib.load(str(Path(brain_mask_path).expanduser().resolve()))
        mask_arr = np.asarray(mask_img.get_fdata()) > 0
        if mask_arr.shape == roi_mask.shape:
            clipped = roi_mask & mask_arr
            if np.any(clipped):
                roi_mask = clipped

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = parc_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), parc_img.affine, header), str(output_path))
    return {
        "roi_mask_path": output_path,
        "label_ids": ids,
        "raw_voxels": raw_voxels,
        "seed_voxels": int(roi_mask.sum()),
        "dilated_cortical": dilated,
    }


def _find_cached_wmfod(layout, subject_id: str) -> Optional[Path]:
    candidates: list[Path] = []
    csd_cache = layout.stage_cache("03_tractography/csd")
    if csd_cache.is_dir():
        candidates.extend(sorted(csd_cache.rglob(f"{subject_id}_wmfod_norm.mif")))
        candidates.extend(sorted(csd_cache.rglob("*_wmfod_norm.mif")))
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    return None


def _find_cached_5tt(layout, subject_id: str) -> Optional[Path]:
    for cand in (
        layout.act_prep / f"{subject_id}_5tt.mif",
        layout.act_prep / f"{subject_id}_5tt.nii.gz",
    ):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    return None


def _run_mrtrix_cmd(cmd: list[str], mrtrix_bin_dir: Optional[Path], log: Optional[Callable[[str], None]]) -> None:
    from shared.external_process import merged_startup_environment, subprocess_no_window_kwargs, text_subprocess_kwargs
    from shared.pipeline import build_cli_environment

    _emit("[ROI Tracking] Running: " + " ".join(cmd), log)
    env = merged_startup_environment(build_cli_environment(cmd[0]))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        env=env,
        **text_subprocess_kwargs(),
        **subprocess_no_window_kwargs(),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"MRtrix command failed ({Path(cmd[0]).name}, exit {proc.returncode}): {tail}")


def _ensure_wmfod_for_roi(
    subject_id: str,
    analysis_inputs: dict[str, Path],
    layout,
    mrtrix_bin_dir: Optional[Path],
    log: Optional[Callable[[str], None]],
) -> tuple[Path, Path]:
    from tractography.csd import _cmd

    cached = _find_cached_wmfod(layout, subject_id)
    shared_dir = layout.stage_cache("03_tractography/csd/roi_shared")
    shared_dir.mkdir(parents=True, exist_ok=True)
    mask_mif = shared_dir / f"{subject_id}_mask.mif"
    if not mask_mif.is_file():
        _run_mrtrix_cmd(
            [_cmd("mrconvert", mrtrix_bin_dir), analysis_inputs["mask"].as_posix(), mask_mif.as_posix(), "-force"],
            mrtrix_bin_dir,
            log,
        )
    if cached is not None:
        _emit(f"[ROI Tracking] Reusing cached WM FOD: {cached}", log)
        return cached, mask_mif

    dwi_mif = shared_dir / f"{subject_id}_dwi.mif"
    wm_resp = shared_dir / f"{subject_id}_response_wm.txt"
    gm_resp = shared_dir / f"{subject_id}_response_gm.txt"
    csf_resp = shared_dir / f"{subject_id}_response_csf.txt"
    wm_fod = shared_dir / f"{subject_id}_wmfod.mif"
    gm_fod = shared_dir / f"{subject_id}_gm.mif"
    csf_fod = shared_dir / f"{subject_id}_csf.mif"
    wm_fod_norm = shared_dir / f"{subject_id}_wmfod_norm.mif"
    gm_norm = shared_dir / f"{subject_id}_gm_norm.mif"
    csf_norm = shared_dir / f"{subject_id}_csf_norm.mif"

    _run_mrtrix_cmd(
        [
            _cmd("mrconvert", mrtrix_bin_dir),
            analysis_inputs["dwi"].as_posix(),
            dwi_mif.as_posix(),
            "-fslgrad",
            analysis_inputs["bvec"].as_posix(),
            analysis_inputs["bval"].as_posix(),
            "-force",
        ],
        mrtrix_bin_dir,
        log,
    )
    _run_mrtrix_cmd(
        [
            _cmd("dwi2response", mrtrix_bin_dir),
            "dhollander",
            dwi_mif.as_posix(),
            wm_resp.as_posix(),
            gm_resp.as_posix(),
            csf_resp.as_posix(),
            "-mask",
            mask_mif.as_posix(),
            "-force",
        ],
        mrtrix_bin_dir,
        log,
    )
    # Detect multi-shell vs single-shell so MRtrix3 msmt_csd works on clinical single-shell DWI too.
    is_multishell = True
    try:
        import numpy as np

        bvals = np.loadtxt(str(analysis_inputs["bval"]), dtype=float).ravel()
        nonzero = bvals[bvals > 100.0]
        if nonzero.size > 0:
            rounded = sorted({int(round(v / 200.0) * 200) for v in nonzero})
            is_multishell = len(rounded) >= 2
    except Exception:
        is_multishell = True

    if is_multishell:
        _run_mrtrix_cmd(
            [
                _cmd("dwi2fod", mrtrix_bin_dir),
                "msmt_csd",
                dwi_mif.as_posix(),
                wm_resp.as_posix(),
                wm_fod.as_posix(),
                gm_resp.as_posix(),
                gm_fod.as_posix(),
                csf_resp.as_posix(),
                csf_fod.as_posix(),
                "-mask",
                mask_mif.as_posix(),
                "-force",
            ],
            mrtrix_bin_dir,
            log,
        )
        _run_mrtrix_cmd(
            [
                _cmd("mtnormalise", mrtrix_bin_dir),
                wm_fod.as_posix(),
                wm_fod_norm.as_posix(),
                gm_fod.as_posix(),
                gm_norm.as_posix(),
                csf_fod.as_posix(),
                csf_norm.as_posix(),
                "-mask",
                mask_mif.as_posix(),
                "-force",
            ],
            mrtrix_bin_dir,
            log,
        )
    else:
        _run_mrtrix_cmd(
            [
                _cmd("dwi2fod", mrtrix_bin_dir),
                "msmt_csd",
                dwi_mif.as_posix(),
                wm_resp.as_posix(),
                wm_fod.as_posix(),
                csf_resp.as_posix(),
                csf_fod.as_posix(),
                "-mask",
                mask_mif.as_posix(),
                "-force",
            ],
            mrtrix_bin_dir,
            log,
        )
        _run_mrtrix_cmd(
            [
                _cmd("mtnormalise", mrtrix_bin_dir),
                wm_fod.as_posix(),
                wm_fod_norm.as_posix(),
                csf_fod.as_posix(),
                csf_norm.as_posix(),
                "-mask",
                mask_mif.as_posix(),
                "-force",
            ],
            mrtrix_bin_dir,
            log,
        )
    return wm_fod_norm, mask_mif


def _export_tck_to_trk_and_vtk(tck_path: Path, reference_dwi: Path, trk_path: Path, vtk_path: Path, mrtrix_bin_dir: Optional[Path], log) -> int:
    import nibabel as nib
    import numpy as np
    from nibabel.streamlines import Field
    from tractography.csd import _cmd

    _run_mrtrix_cmd(
        [_cmd("tckconvert", mrtrix_bin_dir), tck_path.as_posix(), vtk_path.as_posix(), "-force"],
        mrtrix_bin_dir,
        log,
    )
    loaded = nib.streamlines.load(str(tck_path), lazy_load=False)
    streamlines = loaded.streamlines
    count = int(len(streamlines))
    if count == 0:
        raise RuntimeError("ROI tractography produced 0 streamlines for the selected region.")
    ref = nib.load(str(reference_dwi))
    header = nib.streamlines.TrkFile.create_empty_header()
    header[Field.VOXEL_TO_RASMM] = ref.affine
    header[Field.VOXEL_SIZES] = ref.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = ref.shape[:3]
    header[Field.VOXEL_ORDER] = "RAS"
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram, header), str(trk_path))
    return count


def run_roi_tractography(
    config: ROITrackingConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Execute standalone ROI-seeded tractography for one algorithm and DDParcel region set."""
    from shared.io_layout import subject_layout
    from shared.orchestrator import _resolve_analysis_inputs
    from shared.pipeline import RawInputBundle

    alg = str(config.algorithm or "").strip().lower()
    if alg not in SUPPORTED_ROI_ALGORITHMS:
        raise ValueError(f"Unsupported ROI tractography algorithm: {alg!r}")

    started = time.monotonic()
    subject_root = Path(config.subject_root).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve() if config.output_root else None
    layout = subject_layout(subject_root, output_root=output_root)
    analysis_inputs = _resolve_analysis_inputs(subject_root, config.subject_id, output_root)
    for req_key in ("dwi", "bval", "bvec", "mask"):
        p = analysis_inputs.get(req_key)
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"ROI tracking requires input {req_key}, which is missing for {config.subject_id}.")

    parcellation = Path(config.parcellation_path).expanduser().resolve()
    if not parcellation.is_file():
        raise FileNotFoundError(f"DDParcel parcellation file not found: {parcellation}")

    label_names_map = load_label_names(parcellation)
    resolved_names = tuple(
        config.region_names[idx] if idx < len(config.region_names) and config.region_names[idx]
        else label_names_map.get(int(lid), f"Label_{int(lid)}")
        for idx, lid in enumerate(config.label_ids)
    )
    slug = region_slug(config.label_ids, resolved_names)
    roi_dir = layout.tractography / "roi_tracking" / alg / slug
    roi_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.subject_id}_{alg}_roi_{slug}"
    roi_mask_path = roi_dir / f"{config.subject_id}_roi_mask_{slug}.nii.gz"
    out_vtk = roi_dir / f"{stem}.vtk"
    out_trk = roi_dir / f"{stem}.trk"
    manifest_path = roi_dir / "roi_tracking_manifest.json"

    _emit(f"[ROI Tracking] Preparing ROI seed mask for {', '.join(resolved_names)} ({alg})", log)
    seed_info = build_roi_seed_mask(
        parcellation,
        config.label_ids,
        analysis_inputs["mask"],
        roi_mask_path,
        dilate_cortical_voxels=int(config.dilate_cortical_voxels),
    )
    _emit(
        f"[ROI Tracking] Seed ROI ready: {seed_info['seed_voxels']} voxels "
        f"(raw={seed_info['raw_voxels']}, dilated={seed_info['dilated_cortical']})",
        log,
    )

    raw_bundle = RawInputBundle(
        subject_root=subject_root,
        subject_id=config.subject_id,
        input_dir=analysis_inputs["dwi"].parent,
        dwi_path=analysis_inputs["dwi"],
        bval_path=analysis_inputs["bval"],
        bvec_path=analysis_inputs["bvec"],
        mask_path=analysis_inputs["mask"],
    )

    streamline_count = 0
    out_tck: Optional[Path] = None

    if alg == "ukf":
        from tractography.ukf import run_ukf
        from shared.output_contracts import polydata_streamline_count

        ukf_opts = dict(config.ukf_parameters or {})
        ukf_opts.setdefault("seedsPerVoxel", 2)
        stage_dir = layout.stage_cache(f"03_tractography/roi_tracking/ukf/{slug}")
        _emit(f"[ROI Tracking] Running UKF ROI tractography ({', '.join(resolved_names)})", log)
        res = run_ukf(
            raw_bundle,
            stage_dir,
            options=ukf_opts,
            executable=config.ukf_executable,
            converter=config.dwiconvert_executable,
            device=config.device,
            seed_mask_path=roi_mask_path,
            force=True,
            log=log,
        )
        src_vtk = Path(res["outputs"]["tractogram"])
        src_trk = Path(res["outputs"]["tractogram_trk"])
        import shutil

        shutil.copy2(src_vtk, out_vtk)
        if src_trk.is_file():
            shutil.copy2(src_trk, out_trk)
        streamline_count = int(polydata_streamline_count(out_vtk) or 0)

    elif alg == "dti_det":
        from tractography.conversion import ensure_vtk_tractogram
        from tractography.dti import DTITrackingConfig, run_dti_tracking

        dti_opts = dict(config.dti_parameters or {})
        stage_dir = layout.stage_cache(f"03_tractography/roi_tracking/dti_det/{slug}")
        _emit(f"[ROI Tracking] Running DTI deterministic ROI tractography ({', '.join(resolved_names)})", log)
        res = run_dti_tracking(
            DTITrackingConfig(
                subject_id=config.subject_id,
                input_root=subject_root,
                input_bundle=raw_bundle,
                output_dir=stage_dir,
                fit_method=str(dti_opts.get("fit_method", "wls")),
                tracking_method="deterministic",
                output_format="trk",
                step_size_mm=float(dti_opts.get("step_size_mm", 0.5)),
                max_angle_deg=float(dti_opts.get("max_angle_deg", 30.0)),
                seed_density=max(2, int(dti_opts.get("seed_density", 2))),
                max_seed_voxels=None,
                random_seed=int(dti_opts.get("random_seed", 12345)),
                min_fa=float(dti_opts.get("min_fa", 0.2)),
                min_length_mm=float(dti_opts.get("min_length_mm", 10.0)),
                max_length_mm=float(dti_opts.get("max_length_mm", 250.0)),
                device=config.device,
                seed_mask_path=roi_mask_path,
                force_recompute=True,
                execute=True,
            ),
            log=log,
        )
        import shutil

        shutil.copy2(res.streamlines_path, out_trk)
        conv = ensure_vtk_tractogram(out_trk, out_vtk)
        streamline_count = int(conv.get("streamline_count", 0))

    elif alg in {"csd_act", "saf"}:
        from tractography.csd import _cmd

        wm_fod_norm, mask_mif = _ensure_wmfod_for_roi(
            config.subject_id, analysis_inputs, layout, config.mrtrix_bin_dir, log
        )
        five_tt = _find_cached_5tt(layout, config.subject_id)
        out_tck = roi_dir / f"{stem}.tck"
        opts = dict((config.saf_parameters if alg == "saf" else config.csd_act_parameters) or {})
        target_select = int(
            min(max(1000, int(config.select_streamlines or 10000)), 20000)
            if alg == "csd_act"
            else min(max(1000, int(opts.get("select_streamlines", 5000))), 10000)
        )
        min_len = float(opts.get("min_length_mm", 10.0))
        max_len = float(opts.get("max_length_mm", 80.0 if alg == "saf" else 250.0))
        tckgen_cmd = [
            _cmd("tckgen", config.mrtrix_bin_dir),
            wm_fod_norm.as_posix(),
            out_tck.as_posix(),
            "-algorithm",
            str(opts.get("algorithm", "iFOD2")),
            "-seed_image",
            roi_mask_path.as_posix(),
            "-select",
            str(target_select),
            "-minlength",
            str(min_len),
            "-maxlength",
            str(max_len),
            "-force",
        ]
        if five_tt is not None:
            tckgen_cmd.extend(["-act", five_tt.as_posix(), "-backtrack", "-crop_at_gmwmi"])
        else:
            tckgen_cmd.extend(["-mask", mask_mif.as_posix()])
        if opts.get("step_size_mm") is not None:
            tckgen_cmd.extend(["-step", str(opts["step_size_mm"])])
        if opts.get("cutoff") is not None:
            tckgen_cmd.extend(["-cutoff", str(opts["cutoff"])])
        _emit(f"[ROI Tracking] Running {alg.upper()} ROI tractography ({', '.join(resolved_names)})", log)
        _run_mrtrix_cmd(tckgen_cmd, config.mrtrix_bin_dir, log)
        streamline_count = _export_tck_to_trk_and_vtk(
            out_tck, analysis_inputs["dwi"], out_trk, out_vtk, config.mrtrix_bin_dir, log
        )

    elapsed = round(time.monotonic() - started, 2)
    outputs: dict[str, str] = {
        "tractogram_vtk": out_vtk.as_posix(),
        "roi_mask": roi_mask_path.as_posix(),
    }
    if out_trk.is_file():
        outputs["tractogram_trk"] = out_trk.as_posix()
    if out_tck is not None and out_tck.is_file():
        outputs["tractogram_tck"] = out_tck.as_posix()

    manifest_payload = {
        "kind": ROI_TRACKING_MANIFEST_KIND,
        "version": 1,
        "status": "completed",
        "subject_id": config.subject_id,
        "subject_root": subject_root.as_posix(),
        "output_root": layout.outputs.as_posix(),
        "algorithm": alg,
        "label_ids": [int(v) for v in config.label_ids],
        "region_names": list(resolved_names),
        "region_slug": slug,
        "parcellation_path": parcellation.as_posix(),
        "streamline_count": int(streamline_count),
        "seed_voxels": int(seed_info["seed_voxels"]),
        "runtime_seconds": elapsed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "outputs": outputs,
    }
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    _emit(
        f"[ROI Tracking] Completed {alg} ROI tractography in {elapsed:.1f}s "
        f"({streamline_count} streamlines): {out_vtk}",
        log,
    )
    return {
        "manifest_path": manifest_path,
        "manifest": manifest_payload,
        "tractogram_vtk": out_vtk,
        "streamline_count": streamline_count,
    }


def discover_roi_tracking_results(output_root: Optional[Path]) -> list[dict[str, Any]]:
    """Scan ``outputs/03_tractography/roi_tracking/**/roi_tracking_manifest.json`` for Results tree display."""
    if output_root is None:
        return []
    roi_root = Path(output_root) / "03_tractography" / "roi_tracking"
    if not roi_root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for manifest_file in sorted(roi_root.glob("*/*/roi_tracking_manifest.json")):
        try:
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != ROI_TRACKING_MANIFEST_KIND:
            continue
        if payload.get("status") != "completed":
            continue
        vtk_raw = (payload.get("outputs") or {}).get("tractogram_vtk")
        if not isinstance(vtk_raw, str) or not vtk_raw:
            continue
        vtk_path = Path(vtk_raw)
        if not vtk_path.is_file():
            continue
        records.append({
            "manifest_path": manifest_file.resolve(),
            "vtk_path": vtk_path.resolve(),
            "algorithm": str(payload.get("algorithm") or ""),
            "region_names": list(payload.get("region_names") or []),
            "label_ids": list(payload.get("label_ids") or []),
            "streamline_count": int(payload.get("streamline_count") or 0),
            "created_at": str(payload.get("created_at") or ""),
            "payload": payload,
        })
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records
