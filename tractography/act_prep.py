"""ACT preparation: partial-volume 5TT from DDSeg + DDParcel, GMWMI via MRtrix3.

5TT channels (MRtrix convention): 0 cortical GM, 1 sub-cortical GM, 2 WM,
3 CSF, 4 pathology.

* Inside the DDSeg foreground, cGM/WM/CSF are the DDSeg tissue probabilities
  renormalised to sum to one (partial volume instead of hard 0/1 labels).
* DDParcel deep grey matter (thalamus, caudate, putamen, pallidum,
  hippocampus, amygdala, accumbens) goes to the sGM channel as
  ``1 - P(CSF)``; ventricles (lateral, inferior lateral, 3rd, 4th) become
  pure CSF. Brainstem, ventral DC and cerebellar WM go to WM as
  ``1 - P(CSF)``, following MRtrix FreeSurfer2ACT tissue identities.
  DDParcel is resampled into the DDSeg grid through both affines;
  it is never indexed by raw array position (the two are stored in different
  voxel orders).
* ``5ttcheck`` validates the image and ``5tt2gmwmi`` derives the seeding
  interface, replacing the former binary dilation band.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional
from shared.external_process import managed_tool_environment, resolve_external_command

ACT_PREP_VERSION = 3
ACT_PREP_METHOD = "ddseg_partial_volume_ddparcel_tissue_priors_5tt2gmwmi_v3"
# MRtrix FreeSurfer2ACT_sgm_amyg_hipp: ventral DC is WM, not deep GM.
DEFAULT_SGM_LABELS: tuple[int, ...] = (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
DEFAULT_WM_LABELS: tuple[int, ...] = (7, 16, 28, 46, 60)
# Lateral ventricles, inferior lateral ventricles, 3rd and 4th ventricle.
DEFAULT_CSF_LABELS: tuple[int, ...] = (4, 43, 5, 44, 14, 15)
PROBABILITY_FILENAMES = {
    "wm": "ProbabilisticMap_WM.nii.gz",
    "gm": "ProbabilisticMap_GM.nii.gz",
    "csf": "ProbabilisticMap_CSF.nii.gz",
}


@dataclass(frozen=True)
class ACTPrepConfig:
    subject_id: str
    segmentation_path: Path
    output_dir: Path
    wm_label: int = 1
    gm_label: int = 2
    csf_label: int = 3
    execute: bool = False
    probability_paths: Optional[Mapping[str, Path]] = None
    parcellation_path: Optional[Path] = None
    sgm_labels: tuple[int, ...] = DEFAULT_SGM_LABELS
    csf_labels: tuple[int, ...] = DEFAULT_CSF_LABELS
    mrtrix_bin_dir: Optional[Path] = None
    anatomy_source: Mapping[str, object] = field(default_factory=dict)
    wm_prior_labels: tuple[int, ...] = DEFAULT_WM_LABELS
    allow_ddseg_only: bool = False


@dataclass(frozen=True)
class ACTPrepResult:
    subject_id: str
    output_dir: Path
    five_tt_path: Path
    gmwmi_path: Path
    manifest_path: Path
    qc_path: Path
    executed: bool


def _file_info(path: Optional[Path]) -> dict[str, object]:
    if path is None:
        return {"path": None, "exists": False}
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _resolve_command(name: str, bin_dir: Optional[Path]) -> Optional[str]:
    return resolve_external_command(name, bin_dir)


def default_probability_paths(segmentation_path: Path) -> dict[str, Path]:
    return {key: segmentation_path.parent / name for key, name in PROBABILITY_FILENAMES.items()}


def build_partial_volume_5tt(
    labels,
    probabilities: Optional[Mapping[str, object]],
    parcellation_on_grid=None,
    *,
    wm_label: int = 1,
    gm_label: int = 2,
    csf_label: int = 3,
    sgm_labels: tuple[int, ...] = DEFAULT_SGM_LABELS,
    csf_labels: tuple[int, ...] = DEFAULT_CSF_LABELS,
    wm_prior_labels: tuple[int, ...] = DEFAULT_WM_LABELS,
):
    """Return (5TT float32 array, stats) on the DDSeg grid.

    ``parcellation_on_grid`` must already be resampled to the DDSeg grid.
    Without probabilities the hard labels are used (legacy fallback).
    """

    import numpy as np

    groups = (set(sgm_labels), set(csf_labels), set(wm_prior_labels))
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("ACT sGM, CSF and WM prior labels must be disjoint.")
    labels = np.rint(np.asarray(labels)).astype(np.int32)
    foreground = labels > 0
    five_tt = np.zeros((*labels.shape, 5), dtype=np.float32)
    hard = {"gm": labels == gm_label, "wm": labels == wm_label, "csf": labels == csf_label}
    if probabilities is not None:
        stacked = np.stack([np.asarray(probabilities[key], dtype=np.float64) for key in ("gm", "wm", "csf")], axis=-1)
        stacked = np.clip(np.nan_to_num(stacked, nan=0.0), 0.0, None)
        total = stacked.sum(axis=-1)
        usable = foreground & (total > 0)
        fallback = foreground & ~usable
        for channel, (index, key) in zip((0, 2, 3), enumerate(("gm", "wm", "csf"))):
            values = np.zeros(labels.shape, dtype=np.float64)
            values[usable] = stacked[..., index][usable] / total[usable]
            values[fallback] = hard[key][fallback]
            five_tt[..., channel] = values.astype(np.float32)
    else:
        five_tt[..., 0] = hard["gm"]
        five_tt[..., 2] = hard["wm"]
        five_tt[..., 3] = hard["csf"]

    sgm = np.zeros(labels.shape, dtype=bool)
    ventricles = np.zeros(labels.shape, dtype=bool)
    wm_prior = np.zeros(labels.shape, dtype=bool)
    if parcellation_on_grid is not None:
        parcellation = np.rint(np.asarray(parcellation_on_grid)).astype(np.int32)
        if parcellation.shape != labels.shape:
            raise ValueError(f"Parcellation grid {parcellation.shape} does not match segmentation grid {labels.shape}.")
        sgm = foreground & np.isin(parcellation, sgm_labels)
        ventricles = foreground & np.isin(parcellation, csf_labels)
        wm_prior = foreground & np.isin(parcellation, wm_prior_labels)
        non_csf = 1.0 - five_tt[..., 3]
        five_tt[..., 1][sgm] = non_csf[sgm]
        five_tt[..., 0][sgm] = 0.0
        five_tt[..., 2][sgm] = 0.0
        five_tt[..., 2][wm_prior] = non_csf[wm_prior]
        five_tt[..., 0][wm_prior] = 0.0
        five_tt[..., 1][wm_prior] = 0.0
        five_tt[ventricles] = 0.0
        five_tt[..., 3][ventricles] = 1.0

    tissue_sum = five_tt.sum(axis=-1)
    stats = {
        "foreground_voxels": int(foreground.sum()),
        "channel_voxels_ge_0p5": {
            name: int((five_tt[..., index] >= 0.5).sum())
            for index, name in enumerate(("cgm", "sgm", "wm", "csf", "pathology"))
        },
        "channel_volume_voxel_units": {
            name: float(five_tt[..., index].sum()) for index, name in enumerate(("cgm", "sgm", "wm", "csf", "pathology"))
        },
        "sgm_override_voxels": int(sgm.sum()),
        "ventricle_override_voxels": int(ventricles.sum()),
        "wm_prior_override_voxels": int(wm_prior.sum()),
        "wm_prior_label_voxels": {
            str(label): int((foreground & (parcellation == label)).sum())
            for label in wm_prior_labels
        } if parcellation_on_grid is not None else {},
        "partial_volume_voxels": int(((five_tt.max(axis=-1) < 0.999) & foreground).sum()),
        "foreground_sum_min": float(tissue_sum[foreground].min()) if foreground.any() else None,
        "foreground_sum_max": float(tissue_sum[foreground].max()) if foreground.any() else None,
        "background_nonzero_voxels": int((tissue_sum[~foreground] > 0).sum()),
    }
    return five_tt, stats


def _load_inputs(config: ACTPrepConfig, segmentation_path: Path):
    import nibabel as nib
    import numpy as np

    from shared.world_space import resample_nearest_to_grid

    img = nib.load(segmentation_path.as_posix())
    labels = np.rint(np.asarray(img.dataobj)).astype(np.int32)
    unique_labels = set(int(value) for value in np.unique(labels))
    expected_labels = {0, int(config.wm_label), int(config.gm_label), int(config.csf_label)}
    if not unique_labels.issubset(expected_labels):
        raise ValueError(
            "ACT prep expects a compact DDSeg-style segmentation with labels "
            f"{sorted(expected_labels)}; got unexpected labels {sorted(unique_labels - expected_labels)} "
            f"from {segmentation_path}"
        )
    probability_paths = dict(config.probability_paths or default_probability_paths(segmentation_path))
    probabilities = None
    if all(Path(path).is_file() for path in probability_paths.values()):
        probabilities = {}
        for key, path in probability_paths.items():
            prob_img = nib.load(Path(path).as_posix())
            if prob_img.shape[:3] != labels.shape or not np.allclose(prob_img.affine, img.affine, atol=1e-3):
                raise ValueError(f"DDSeg probability map {path} is not on the segmentation grid.")
            probabilities[key] = np.asarray(prob_img.dataobj, dtype=np.float64)
    parcellation_on_grid = None
    if config.parcellation_path is not None:
        parc_img = nib.load(Path(config.parcellation_path).as_posix())
        parcellation_on_grid = resample_nearest_to_grid(
            np.rint(np.asarray(parc_img.dataobj)).astype(np.int32),
            parc_img.affine,
            labels.shape,
            img.affine,
        )
    return img, labels, probabilities, probability_paths, parcellation_on_grid


def _save_like(img, data, path: Path) -> None:
    import nibabel as nib
    import numpy as np

    header = img.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(np.asarray(data, dtype=np.float32), img.affine, header)
    out.set_qform(img.affine, code=int(img.header["qform_code"]) or 1)
    out.set_sform(img.affine, code=int(img.header["sform_code"]) or 1)
    nib.save(out, path.as_posix())


def _run_mrtrix(
    command: list[str],
    log: Optional[Callable[[str], None]],
    *,
    mrtrix_bin_dir: Optional[Path] = None,
) -> dict[str, object]:
    if log:
        log(f"[ACT Prep] Running: {' '.join(command)}")
    effective_bin = mrtrix_bin_dir
    if effective_bin is None and command and Path(command[0]).is_file():
        effective_bin = Path(command[0]).resolve().parent
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=managed_tool_environment(mrtrix_bin_dir=effective_bin),
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"ACT prep command failed ({completed.returncode}): {' '.join(command)}\n{record['stderr']}")
    return record


def _gmwmi_voxels(gmwmi_path: Path) -> int:
    import nibabel as nib
    import numpy as np

    return int(np.count_nonzero(np.asarray(nib.load(gmwmi_path.as_posix()).dataobj) > 0))


def _base_manifest(config: ACTPrepConfig, segmentation_path: Path, five_tt_path: Path, gmwmi_path: Path, qc_path: Path):
    probability_paths = dict(config.probability_paths or default_probability_paths(segmentation_path))
    return {
        "kind": "act_prep",
        "version": ACT_PREP_VERSION,
        "method": ACT_PREP_METHOD,
        "subject_id": config.subject_id,
        "executed": bool(config.execute),
        "status": "planned" if not config.execute else "running",
        "labels": {"wm": config.wm_label, "gm": config.gm_label, "csf": config.csf_label},
        "sgm_labels": list(config.sgm_labels),
        "csf_override_labels": list(config.csf_labels),
        "wm_prior_labels": list(config.wm_prior_labels),
        "anatomy_source": dict(config.anatomy_source),
        "allow_ddseg_only": config.allow_ddseg_only,
        "pathology_channel": {"index": 4, "modelled": False, "value": 0.0},
        "inputs": {
            "segmentation": _file_info(segmentation_path),
            "probabilities": {key: _file_info(Path(path)) for key, path in probability_paths.items()},
            "parcellation": _file_info(Path(config.parcellation_path)) if config.parcellation_path else None,
        },
        "outputs": {"five_tt": five_tt_path.as_posix(), "gmwmi": gmwmi_path.as_posix(), "qc": qc_path.as_posix()},
        "parcellation_resampling": "world-space nearest neighbour through both affines (no raw array indexing)",
        "notes": [
            "cGM/WM/CSF are DDSeg softmax probabilities renormalised to sum to 1 inside the DDSeg foreground.",
            "sGM = 1 - P(CSF) on DDParcel deep GM labels (sgm_labels); ventricles (csf_override_labels) are pure CSF.",
            "Hippocampus/amygdala use the MRtrix sgm_amyg_hipp convention. Ventral DC must not be sGM.",
            "Brainstem, ventral DC and cerebellar WM: WM = 1 - P(CSF), preserving CSF partial volume and DDSeg foreground.",
            "Tissue identities: MRtrix3 share/mrtrix3/_5ttgen/FreeSurfer2ACT_sgm_amyg_hipp.txt.",
            "GMWMI is generated with MRtrix3 5tt2gmwmi; the 5TT is validated with 5ttcheck.",
            "The optional pathology channel is zero; lesion-specific tissue priors are not estimated.",
        ],
        "warnings": [],
    }


def run_act_prep(config: ACTPrepConfig, *, log: Optional[Callable[[str], None]] = None) -> ACTPrepResult:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    segmentation_path = config.segmentation_path.expanduser().resolve()
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Tissue segmentation not found: {segmentation_path}")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    five_tt_path = output_dir / f"{config.subject_id}_5tt_from_labels.nii.gz"
    gmwmi_path = output_dir / f"{config.subject_id}_gmwmi_from_labels.nii.gz"
    manifest_path = output_dir / "act_prep_manifest.json"
    qc_path = output_dir / "act_prep_qc.json"
    manifest = _base_manifest(config, segmentation_path, five_tt_path, gmwmi_path, qc_path)
    qc: dict[str, object] = {
        "kind": "act_prep_qc",
        "version": ACT_PREP_VERSION,
        "subject_id": config.subject_id,
        "status": "planned",
        "executed": bool(config.execute),
    }

    if config.execute:
        # Invalidate old completion records before checking any execution prerequisites.
        qc["status"] = "running"
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            if config.parcellation_path is None and not config.allow_ddseg_only:
                raise ValueError("ACT preparation requires DDParcel anatomical labels; select parcellation or provide existing output. DDSeg-only fallback requires explicit allow_ddseg_only=True.")
            commands = {name: _resolve_command(name, config.mrtrix_bin_dir) for name in ("5ttcheck", "5tt2gmwmi")}
            missing = [name for name, resolved in commands.items() if resolved is None]
            if missing:
                raise RuntimeError(f"ACT prep requires MRtrix3 command(s): {', '.join(missing)}")
            img, labels, probabilities, _, parcellation_on_grid = _load_inputs(config, segmentation_path)
            if probabilities is None:
                manifest["warnings"].append("DDSeg probability maps missing: hard labels used for the 5TT.")
            if parcellation_on_grid is None:
                manifest["warnings"].append("Explicit DDSeg-only ACT fallback: subcortical GM, ventricular and selected WM priors are unavailable; anatomical review is required.")
            five_tt, stats = build_partial_volume_5tt(
                labels,
                probabilities,
                parcellation_on_grid,
                wm_label=config.wm_label,
                gm_label=config.gm_label,
                csf_label=config.csf_label,
                sgm_labels=tuple(config.sgm_labels),
                csf_labels=tuple(config.csf_labels),
                wm_prior_labels=tuple(config.wm_prior_labels),
            )
            qc.update(
                mode="partial_volume" if probabilities is not None else "hard_labels",
                sgm_from_ddparcel=parcellation_on_grid is not None,
                anatomical_prior_status="complete" if parcellation_on_grid is not None else "ddseg_only",
                pathology_channel=manifest["pathology_channel"],
                warnings=list(manifest["warnings"]),
                five_tt_stats=stats,
            )
            if parcellation_on_grid is not None and stats["channel_volume_voxel_units"]["sgm"] <= 0:
                raise ValueError("DDParcel contributes no subcortical GM to ACT 5TT; check label compatibility, brain coverage and spatial alignment.")
            if stats["foreground_voxels"] == 0:
                raise ValueError("DDSeg foreground is empty; ACT preparation cannot proceed.")
            for warning in manifest["warnings"]:
                if log:
                    log(f"[ACT Prep] WARNING: {warning}")
            _save_like(img, five_tt, five_tt_path)
            check = _run_mrtrix(
                [str(commands["5ttcheck"]), five_tt_path.as_posix()],
                log,
                mrtrix_bin_dir=config.mrtrix_bin_dir,
            )
            gmwmi = _run_mrtrix(
                [str(commands["5tt2gmwmi"]), five_tt_path.as_posix(), gmwmi_path.as_posix(), "-force"],
                log,
                mrtrix_bin_dir=config.mrtrix_bin_dir,
            )
            manifest["commands"] = [check["command"], gmwmi["command"]]
            qc.update(
                status="completed",
                wm_voxels=stats["channel_voxels_ge_0p5"]["wm"],
                gm_voxels=stats["channel_voxels_ge_0p5"]["cgm"] + stats["channel_voxels_ge_0p5"]["sgm"],
                csf_voxels=stats["channel_voxels_ge_0p5"]["csf"],
                gmwmi_voxels=_gmwmi_voxels(gmwmi_path),
                format_validated=True,
                anatomical_review="required",
                **{"5ttcheck": {"returncode": check["returncode"], "output": (check["stdout"] + check["stderr"])[-2000:]}},
            )
            if qc["gmwmi_voxels"] == 0:
                raise ValueError("ACT GM-WM interface is empty; inspect tissue priors before tracking.")
            manifest["status"] = "completed"
        except Exception as exc:
            qc.update(status="failed", error=str(exc))
            manifest.update(status="failed", error=str(exc))
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if log:
        log(f"[ACT Prep] Manifest: {manifest_path}")
    return ACTPrepResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        five_tt_path=five_tt_path,
        gmwmi_path=gmwmi_path,
        manifest_path=manifest_path,
        qc_path=qc_path,
        executed=bool(config.execute),
    )
