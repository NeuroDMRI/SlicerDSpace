"""DTI tractography using a lightweight DIPY backend."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from shared import NativeDTIBundle, RawInputBundle, default_task_output_dir, find_subject_root


SUPPORTED_TRACKING_METHODS = {"deterministic"}
SUPPORTED_OUTPUT_FORMATS = {"trk", "tck"}


@dataclass(frozen=True)
class DTITrackingConfig:
    subject_id: str
    input_root: Path
    input_bundle: Optional[RawInputBundle] = None
    output_dir: Optional[Path] = None
    fit_method: str = "wls"
    tracking_method: str = "deterministic"
    output_format: str = "trk"
    step_size_mm: float = 0.5
    max_angle_deg: float = 30.0
    seed_density: int = 1
    max_seed_voxels: Optional[int] = None
    random_seed: int = 12345
    min_fa: float = 0.2
    min_length_mm: float = 10.0
    max_length_mm: float = 250.0
    device: str = "auto"
    force_recompute: bool = False
    execute: bool = False


@dataclass(frozen=True)
class DTITrackingResult:
    subject_id: str
    output_dir: Path
    streamlines_path: Path
    manifest_path: Path
    qc_path: Path
    native_bundle_dir: Path
    native_cache_available: bool
    executed: bool


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


def _file_info(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _native_bundle_from_expected_cache(subject_root: Path, subject_id: str, fit_method: str) -> NativeDTIBundle:
    fit_key = fit_method.lower()
    bundle_dir = subject_root / "cache" / "native_dti" / fit_key
    return NativeDTIBundle(
        subject_root=subject_root,
        subject_id=subject_id,
        fit_method=fit_key,
        bundle_dir=bundle_dir,
        dwi_nhdr_path=bundle_dir / "DWI.nhdr",
        mask_nhdr_path=bundle_dir / "Mask.nhdr",
        dti_nrrd_path=bundle_dir / "DTI.nrrd",
        b0_nrrd_path=bundle_dir / "b0.nrrd",
        fa_nrrd_path=bundle_dir / "DTI_FA.nrrd",
        trace_nrrd_path=bundle_dir / "DTI_Trace.nrrd",
        mean_diffusivity_nrrd_path=bundle_dir / "DTI_MeanDiffusivity.nrrd",
        min_eig_nrrd_path=bundle_dir / "DTI_MinEigenvalue.nrrd",
        mid_eig_nrrd_path=bundle_dir / "DTI_MidEigenvalue.nrrd",
        max_eig_nrrd_path=bundle_dir / "DTI_MaxEigenvalue.nrrd",
        ddseg_md_nrrd_path=bundle_dir / "DTI_MD.nrrd",
        ddseg_e1_nrrd_path=bundle_dir / "DTI_E1.nrrd",
        ddseg_e2_nrrd_path=bundle_dir / "DTI_E2.nrrd",
        ddseg_e3_nrrd_path=bundle_dir / "DTI_E3.nrrd",
        fa_nii_path=bundle_dir / f"{subject_id}-dti-FractionalAnisotropy.nii.gz",
        trace_nii_path=bundle_dir / f"{subject_id}-dti-Trace.nii.gz",
        mean_diffusivity_nii_path=bundle_dir / f"{subject_id}-dti-MeanDiffusivity.nii.gz",
        min_eig_nii_path=bundle_dir / f"{subject_id}-dti-MinEigenvalue.nii.gz",
        mid_eig_nii_path=bundle_dir / f"{subject_id}-dti-MidEigenvalue.nii.gz",
        max_eig_nii_path=bundle_dir / f"{subject_id}-dti-MaxEigenvalue.nii.gz",
        manifest_path=bundle_dir / "manifest.json",
        dec_nrrd_path=bundle_dir / f"{subject_id}-dti-DEC.nrrd",
    )


def _native_cache_available(bundle: NativeDTIBundle) -> bool:
    required = [
        bundle.manifest_path,
        bundle.dti_nrrd_path,
        bundle.fa_nii_path,
        bundle.mean_diffusivity_nii_path,
    ]
    return all(path.is_file() for path in required)


def _validate_config(config: DTITrackingConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.fit_method.lower() not in {"ls", "wls"}:
        raise ValueError("fit_method must be 'ls' or 'wls'.")
    if config.tracking_method.lower() not in SUPPORTED_TRACKING_METHODS:
        raise ValueError(f"tracking_method must be one of {sorted(SUPPORTED_TRACKING_METHODS)}.")
    if config.output_format.lower() not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}.")
    if config.step_size_mm <= 0:
        raise ValueError("step_size_mm must be > 0.")
    if config.max_angle_deg <= 0:
        raise ValueError("max_angle_deg must be > 0.")
    if config.seed_density < 1:
        raise ValueError("seed_density must be >= 1.")
    if config.max_seed_voxels is not None and config.max_seed_voxels < 1:
        raise ValueError("max_seed_voxels must be >= 1 when provided.")
    if config.min_fa < 0:
        raise ValueError("min_fa must be >= 0.")
    if config.min_length_mm < 0:
        raise ValueError("min_length_mm must be >= 0.")
    if config.max_length_mm <= config.min_length_mm:
        raise ValueError("max_length_mm must be greater than min_length_mm.")
    if str(config.device).strip().lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")


def _resolve_dti_device(requested: str) -> str:
    import os

    env_override = os.environ.get("SLICERDSPACE_DTI_TRACKING_DEVICE")
    mode = str(env_override if env_override else requested).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported DTI tracking device {mode!r}.")
    if mode == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if mode == "cuda" and not cuda_ok:
        raise RuntimeError("DTI tracking CUDA device requested, but PyTorch CUDA is unavailable.")
    return "cuda" if cuda_ok else "cpu"


def _streamline_lengths(streamlines: list[object]) -> list[float]:
    import numpy as np

    n_sl = len(streamlines)
    if n_sl == 0:
        return []
    try:
        from nibabel.streamlines.array_sequence import ArraySequence

        seq = streamlines if isinstance(streamlines, ArraySequence) else ArraySequence(streamlines)
        pts = np.asarray(seq._data, dtype=np.float64)
        lengths_arr = np.asarray(seq._lengths, dtype=np.int64)
        offsets_arr = np.asarray(seq._offsets, dtype=np.int64)
        if pts.ndim == 2 and pts.shape[0] >= 2 and len(lengths_arr) == n_sl:
            seg_norms = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            out = np.zeros(n_sl, dtype=np.float64)
            valid = lengths_arr >= 2
            if np.any(valid):
                v_off = offsets_arr[valid]
                v_len = lengths_arr[valid]
                # Prefix sum of segment norms gives O(1) exact sum per streamline without boundary bleed
                csum = np.empty(pts.shape[0], dtype=np.float64)
                csum[0] = 0.0
                np.cumsum(seg_norms, out=csum[1:])
                out[valid] = csum[v_off + v_len - 1] - csum[v_off]
            return out.tolist()
    except Exception:
        pass

    lengths: list[float] = []
    for streamline in streamlines:
        points = np.asarray(streamline, dtype=float)
        if points.shape[0] < 2:
            lengths.append(0.0)
            continue
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    return lengths


def _eudx_length_limits(
    dipy_version: str,
    step_size_mm: float,
    min_length_mm: float = 10.0,
    max_length_mm: float = 250.0,
) -> tuple[int, int]:
    """Translate millimetre limits across the DIPY 1.12 unit change."""

    release = tuple(int(value) for value in re.findall(r"\d+", str(dipy_version))[:2])
    if release >= (1, 12):
        return max(0, int(math.floor(min_length_mm))), max(1, int(math.ceil(max_length_mm)))
    minimum_points = max(2, int(math.floor(min_length_mm / step_size_mm)) + 1)
    maximum_points = max(minimum_points, int(math.ceil(max_length_mm / step_size_mm)) + 1)
    return minimum_points, maximum_points


def _filter_streamlines_by_length(
    streamlines: list[object],
    *,
    min_length_mm: float,
    max_length_mm: float,
) -> tuple[list[object], list[float], int, int]:
    lengths = _streamline_lengths(streamlines)
    kept: list[object] = []
    kept_lengths: list[float] = []
    rejected_short = 0
    rejected_long = 0
    for streamline, length in zip(streamlines, lengths):
        if length + 1e-6 < min_length_mm:
            rejected_short += 1
        elif length - 1e-6 > max_length_mm:
            rejected_long += 1
        else:
            kept.append(streamline)
            kept_lengths.append(length)
    return kept, kept_lengths, rejected_short, rejected_long


def _tensor_direction_getter(
    evecs,
    tracking_mask,
    sphere,
    *,
    max_angle_deg: float,
    chunk_size: int = 8192,
):
    """Build a deterministic DIPY direction getter from fitted tensor axes."""

    import numpy as np
    from dipy.direction.peaks import PeaksAndMetrics

    peak_indices = np.full(tracking_mask.shape + (1,), -1, dtype=np.int32)
    peak_values = np.zeros(tracking_mask.shape + (1,), dtype=np.float32)
    voxel_indices = np.argwhere(tracking_mask)
    principal = np.nan_to_num(evecs[..., :, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )
    sphere_vertices = np.asarray(sphere.vertices, dtype=np.float32)

    for start in range(0, voxel_indices.shape[0], int(chunk_size)):
        coords = voxel_indices[start : start + int(chunk_size)]
        directions = principal[coords[:, 0], coords[:, 1], coords[:, 2]]
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 0
        closest = np.full(coords.shape[0], -1, dtype=np.int32)
        if np.any(valid):
            normalized = directions[valid] / norms[valid, None]
            closest[valid] = np.argmax(np.abs(normalized @ sphere_vertices.T), axis=1).astype(np.int32)
        peak_indices[coords[:, 0], coords[:, 1], coords[:, 2], 0] = closest
        # Direction-getter peak values are confidence weights, not physical
        # diffusivities.  Unit weights avoid filtering valid tensor axes by
        # EuDX's default QA threshold; FA remains the stopping criterion.
        peak_values[coords[:, 0], coords[:, 1], coords[:, 2], 0] = (closest >= 0).astype(np.float32)

    peaks = PeaksAndMetrics()
    peaks.sphere = sphere
    peaks.peak_indices = peak_indices
    peaks.peak_values = peak_values
    peaks.ang_thr = float(max_angle_deg)
    peaks._initialize()
    return peaks


def _fit_dti_fa_and_peaks(
    data_in_mask,
    mask,
    gtab,
    sphere,
    *,
    fit_method: str,
    min_fa: float,
    max_angle_deg: float,
    device: str = "auto",
):
    """Hardware-adaptive vectorized CPU / PyTorch FP64 CUDA WLS & OLS DTI fitter + peak projector.

    Replaces DIPY's per-voxel ``(G, 7)`` SVD ``pinv`` with batched ``7 x 7`` normal
    equations (with exact ``pinv`` fallback on rank-deficient voxels), giving >50x
    faster tensor fitting while matching DIPY ``TensorModel`` eigenvalues/FA/peaks.
    """
    import numpy as np
    from dipy.direction.peaks import PeaksAndMetrics
    from dipy.reconst.dti import (
        MIN_POSITIVE_SIGNAL,
        design_matrix,
        fractional_anisotropy,
        from_lower_triangular,
    )
    from shared.hardware_adapt import (
        adaptive_cuda_scaled_batch,
        empty_cuda_cache_safely,
        run_cuda_chunked_with_oom_fallback,
    )

    resolved_device = _resolve_dti_device(device)
    explicit_cuda = str(device).strip().lower() == "cuda"
    w_mat = np.asarray(design_matrix(gtab), dtype=np.float64)
    w_pinv_t = np.linalg.pinv(w_mat).T
    w_outer = (w_mat[:, :, None] * w_mat[:, None, :]).reshape(w_mat.shape[0], 49)
    lo_tri_idx = from_lower_triangular(np.arange(6, dtype=np.int64))
    min_diff = 1e-6 / float(-w_mat.min())
    is_wls = fit_method.upper() in {"WLS", "WLLS"}
    sphere_verts = np.asarray(sphere.vertices, dtype=np.float64)
    n_vox = int(data_in_mask.shape[0])

    def _cpu_chunk(start: int, end: int):
        sig = np.maximum(np.asarray(data_in_mask[start:end], dtype=np.float64), MIN_POSITIVE_SIGNAL)
        log_s = np.log(sig)
        beta_ols = log_s @ w_pinv_t
        if is_wls:
            w = np.exp(np.clip(beta_ols @ w_mat.T, -50.0, 50.0))
            w2 = w * w
            lhs = (w2 @ w_outer).reshape(-1, 7, 7)
            rhs = (w2 * log_s) @ w_mat
            try:
                beta = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                beta = np.full_like(beta_ols, np.nan)
            bad = ~np.all(np.isfinite(beta), axis=1)
            if np.any(bad):
                beta[bad] = np.einsum(
                    "...ij,...j->...i",
                    np.linalg.pinv(w_mat * w[bad, :, None]),
                    w[bad] * log_s[bad],
                )
        else:
            beta = beta_ols
        tensors = beta[:, lo_tri_idx]
        evals_asc, evecs_asc = np.linalg.eigh(tensors)
        evals = np.maximum(evals_asc[:, ::-1], min_diff)
        principal = np.nan_to_num(evecs_asc[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
        fa_chunk = np.nan_to_num(fractional_anisotropy(evals), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(principal, axis=1)
        closest = np.full(end - start, -1, dtype=np.int32)
        valid_dir = (fa_chunk >= float(min_fa)) & (norms > 0)
        if np.any(valid_dir):
            normed = principal[valid_dir] / norms[valid_dir, None]
            closest[valid_dir] = np.argmax(np.abs(normed @ sphere_verts.T), axis=1).astype(np.int32)
        return fa_chunk, closest

    if resolved_device == "cuda" and n_vox > 0:
        import torch

        dev = torch.device("cuda")
        w_t = torch.as_tensor(w_mat, device=dev, dtype=torch.float64)
        w_pinv_t_gpu = torch.as_tensor(w_pinv_t, device=dev, dtype=torch.float64)
        w_outer_t = torch.as_tensor(w_outer, device=dev, dtype=torch.float64)
        lo_idx_t = torch.as_tensor(lo_tri_idx, device=dev, dtype=torch.long)
        sphere_t = torch.as_tensor(sphere_verts.T, device=dev, dtype=torch.float64)
        per_vox_bytes = max(2048, int(w_mat.shape[0]) * 8 * 8)
        init_chunk = adaptive_cuda_scaled_batch(
            per_item_bytes=per_vox_bytes,
            default_batch=16384,
            min_batch=2048,
            max_batch=16384,
        )

        def _cuda_chunk(start: int, end: int):
            sig_np = np.ascontiguousarray(data_in_mask[start:end], dtype=np.float64)
            sig_t = torch.clamp(torch.from_numpy(sig_np).to(dev), min=float(MIN_POSITIVE_SIGNAL))
            log_s = torch.log(sig_t)
            beta_ols = log_s @ w_pinv_t_gpu
            if is_wls:
                w = torch.exp(torch.clamp(beta_ols @ w_t.T, min=-50.0, max=50.0))
                w2 = w * w
                lhs = (w2 @ w_outer_t).view(-1, 7, 7)
                rhs = ((w2 * log_s) @ w_t).unsqueeze(-1)
                beta, info = torch.linalg.solve_ex(lhs, rhs)
                beta = beta.squeeze(-1)
                bad = (info != 0) | (~torch.all(torch.isfinite(beta), dim=1))
                if bool(torch.any(bad)):
                    bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(1)
                    wb = w.index_select(0, bad_idx)
                    lsb = log_s.index_select(0, bad_idx)
                    pinv_b = torch.linalg.pinv(w_t.unsqueeze(0) * wb.unsqueeze(-1))
                    beta_bad = torch.matmul(pinv_b, (wb * lsb).unsqueeze(-1)).squeeze(-1)
                    beta = beta.clone()
                    beta.index_copy_(0, bad_idx, beta_bad)
            else:
                beta = beta_ols
            tensors = beta[:, lo_idx_t]
            n_t = tensors.shape[0]
            if n_t <= 8192:
                evals_asc, evecs_asc = torch.linalg.eigh(tensors)
            else:
                ev_list = []
                ec_list = []
                for sub_s in range(0, n_t, 8192):
                    ev_s, ec_s = torch.linalg.eigh(tensors[sub_s : sub_s + 8192])
                    ev_list.append(ev_s)
                    ec_list.append(ec_s)
                evals_asc = torch.cat(ev_list, dim=0)
                evecs_asc = torch.cat(ec_list, dim=0)
            evals = torch.clamp(torch.flip(evals_asc, dims=[1]), min=min_diff)
            ev1, ev2, ev3 = evals[:, 0], evals[:, 1], evals[:, 2]
            denom = ev1 * ev1 + ev2 * ev2 + ev3 * ev3
            num = 0.5 * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2)
            fa_t = torch.where(denom > 0, torch.sqrt(num / torch.where(denom > 0, denom, torch.ones_like(denom))), torch.zeros_like(denom))
            fa_t = torch.nan_to_num(fa_t, nan=0.0, posinf=0.0, neginf=0.0)
            principal = torch.nan_to_num(evecs_asc[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
            norms = torch.linalg.norm(principal, dim=1)
            closest_t = torch.full((end - start,), -1, device=dev, dtype=torch.int32)
            valid_dir = (fa_t >= float(min_fa)) & (norms > 0)
            if bool(torch.any(valid_dir)):
                v_idx = torch.nonzero(valid_dir, as_tuple=False).squeeze(1)
                normed = principal.index_select(0, v_idx) / norms.index_select(0, v_idx).unsqueeze(1)
                proj = torch.abs(normed @ sphere_t)
                closest_t.index_copy_(0, v_idx, torch.argmax(proj, dim=1).to(torch.int32))
            return fa_t.detach().cpu().numpy(), closest_t.detach().cpu().numpy()

        try:
            results = run_cuda_chunked_with_oom_fallback(
                n_vox,
                initial_chunk_size=init_chunk,
                min_cuda_chunk_size=1024,
                cuda_fn=_cuda_chunk,
                cpu_fallback_fn=None if explicit_cuda else _cpu_chunk,
            )
        finally:
            del w_t, w_pinv_t_gpu, w_outer_t, lo_idx_t, sphere_t
            empty_cuda_cache_safely()
    else:
        results = [_cpu_chunk(s, min(n_vox, s + 32768)) for s in range(0, n_vox, 32768)] if n_vox > 0 else []

    fa_vol = np.zeros(mask.shape, dtype=np.float64)
    peak_indices = np.full(mask.shape + (1,), -1, dtype=np.int32)
    peak_values = np.zeros(mask.shape + (1,), dtype=np.float32)
    if results:
        fa_flat = np.concatenate([r[0] for r in results], axis=0)
        closest_flat = np.concatenate([r[1] for r in results], axis=0)
        fa_vol[mask] = fa_flat
        peak_indices[mask, 0] = closest_flat
        peak_values[mask, 0] = (closest_flat >= 0).astype(np.float32)

    tracking_mask = np.logical_and(mask, fa_vol >= float(min_fa))
    peaks = PeaksAndMetrics()
    peaks.sphere = sphere
    peaks.peak_indices = peak_indices
    peaks.peak_values = peak_values
    peaks.ang_thr = float(max_angle_deg)
    peaks._initialize()
    return fa_vol, tracking_mask, peaks, resolved_device


def _run_dipy_dti_tracking(
    config: DTITrackingConfig,
    *,
    streamlines_path: Path,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    try:
        from dipy import __version__ as dipy_version
        import nibabel as nib
        import numpy as np
        from dipy.core.gradients import gradient_table
        from dipy.data import default_sphere
        from dipy.io.stateful_tractogram import Space, StatefulTractogram
        from dipy.io.streamline import save_tck, save_trk
        from dipy.reconst.dti import TensorModel, fractional_anisotropy
        from dipy.tracking import utils
        from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
        try:
            from dipy.tracking.tracker import eudx_tracking
        except ImportError:  # DIPY 1.10 compatibility for Slicer 5.6 / Python 3.9
            eudx_tracking = None
            from dipy.tracking.local_tracking import LocalTracking
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("DTI tractography execution requires DIPY, nibabel, and numpy.") from exc

    raw = config.input_bundle or find_subject_root(
        config.input_root.expanduser().resolve(),
        config.subject_id,
    )
    _emit("[Tractography] Loading DWI, gradients, and mask for DIPY DTI tracking", log)
    dwi_img = nib.load(raw.dwi_path.as_posix())
    dwi_shape = tuple(int(dim) for dim in dwi_img.shape)
    mask = np.asarray(nib.load(raw.mask_path.as_posix()).get_fdata()) > 0
    bvals = np.loadtxt(raw.bval_path.as_posix(), ndmin=1)
    bvecs = np.loadtxt(raw.bvec_path.as_posix())
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if len(dwi_shape) != 4:
        raise ValueError(f"DTI tracking expects 4D DWI data, got shape {dwi_shape}")
    if dwi_shape[:3] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match DWI shape {dwi_shape[:3]}")
    if dwi_shape[3] != len(bvals):
        raise ValueError(f"DWI volume count {dwi_shape[3]} does not match bvals count {len(bvals)}")

    # Extract masked voxels in exact C-order (`dwi_full[mask]`): use fast single-pass
    # in-memory decode when ample host RAM is available, or bounded-RAM mmap z-slabs
    # on modest 8-16 GiB machines / heavily loaded cohort nodes.
    from shared.diffusion_io import diffusion_image
    from shared.hardware_adapt import GIB, adaptive_cpu_slab_z, query_host_available_ram_bytes

    n_mask_vox = int(mask.sum())
    data_in_mask = np.empty((n_mask_vox, dwi_shape[3]), dtype=np.float32)
    if n_mask_vox > 0:
        full_f32_bytes = int(np.prod(dwi_shape)) * 4
        avail_ram = query_host_available_ram_bytes()
        can_single_pass_ram = (
            avail_ram is not None
            and avail_ram >= max(int(4 * GIB), full_f32_bytes * 3)
        )
        if can_single_pass_ram:
            full_arr = np.asarray(dwi_img.dataobj, dtype=np.float32)
            data_in_mask[:] = full_arr[mask]
            del full_arr
        else:
            slab_z = adaptive_cpu_slab_z(dwi_shape, 16, min_step_z=4)
            mask_rank = np.full(mask.shape, -1, dtype=np.int32)
            mask_rank[mask] = np.arange(n_mask_vox, dtype=np.int32)
            with diffusion_image(raw.dwi_path, streamlines_path.parent) as mm_img:
                mm_dataobj = mm_img.dataobj
                for z0 in range(0, dwi_shape[2], slab_z):
                    z1 = min(dwi_shape[2], z0 + slab_z)
                    mask_slab = mask[:, :, z0:z1]
                    if not np.any(mask_slab):
                        continue
                    dest_idx = mask_rank[:, :, z0:z1][mask_slab]
                    slab_arr = np.asarray(mm_dataobj[:, :, z0:z1, :], dtype=np.float32)
                    data_in_mask[dest_idx] = slab_arr[mask_slab]
            del mask_rank

    gtab = gradient_table(bvals, bvecs=bvecs)
    _emit(f"[Tractography] Fitting DTI tensor and sphere peaks (method={config.fit_method.upper()}, device={config.device})", log)
    fa, tracking_mask, peaks, resolved_device = _fit_dti_fa_and_peaks(
        data_in_mask,
        mask,
        gtab,
        default_sphere,
        fit_method=config.fit_method.upper(),
        min_fa=float(config.min_fa),
        max_angle_deg=float(config.max_angle_deg),
        device=config.device,
    )
    del data_in_mask
    seed_mask = tracking_mask
    if config.max_seed_voxels is not None:
        voxel_indices = np.argwhere(seed_mask)
        if voxel_indices.shape[0] > int(config.max_seed_voxels):
            rng = np.random.default_rng(int(config.random_seed))
            selected = rng.choice(voxel_indices.shape[0], size=int(config.max_seed_voxels), replace=False)
            limited_mask = np.zeros_like(seed_mask, dtype=bool)
            limited_indices = voxel_indices[selected]
            limited_mask[limited_indices[:, 0], limited_indices[:, 1], limited_indices[:, 2]] = True
            seed_mask = limited_mask
    seed_voxel_count = int(seed_mask.sum())
    if seed_voxel_count == 0:
        raise RuntimeError("No voxels passed the DTI tracking FA/mask threshold.")

    _emit(f"[Tractography] Building seeds (tensor device={resolved_device})", log)
    seeds = utils.seeds_from_mask(
        seed_mask,
        dwi_img.affine,
        density=(int(config.seed_density), int(config.seed_density), int(config.seed_density)),
    )
    stopping_criterion = ThresholdStoppingCriterion(fa, float(config.min_fa))
    if eudx_tracking is not None:
        import sys

        min_length, max_length = _eudx_length_limits(
            dipy_version,
            config.step_size_mm,
            config.min_length_mm,
            config.max_length_mm,
        )
        local_tracking = eudx_tracking(
            seeds,
            stopping_criterion,
            dwi_img.affine,
            pam=peaks,
            step_size=float(config.step_size_mm),
            max_len=max_length,
            min_len=min_length,
            max_angle=float(config.max_angle_deg),
            nbr_threads=1 if sys.platform == "darwin" else 0,
            random_seed=int(config.random_seed),
            return_all=True,
        )
    else:
        min_length, max_length = _eudx_length_limits(
            "1.11",
            config.step_size_mm,
            config.min_length_mm,
            config.max_length_mm,
        )
        local_tracking = LocalTracking(
            peaks,
            stopping_criterion,
            seeds,
            dwi_img.affine,
            step_size=float(config.step_size_mm),
            maxlen=max_length,
            minlen=min_length,
            return_all=True,
        )
    candidate_streamlines = [streamline for streamline in local_tracking if len(streamline) >= 2]
    streamlines, lengths, rejected_short, rejected_long = _filter_streamlines_by_length(
        candidate_streamlines,
        min_length_mm=float(config.min_length_mm),
        max_length_mm=float(config.max_length_mm),
    )
    if not streamlines:
        raise RuntimeError(
            "DIPY tracking produced no streamlines within the configured length range "
            f"[{config.min_length_mm:g}, {config.max_length_mm:g}] mm."
        )
    streamlines_path.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)
    if streamlines_path.suffix.lower() == ".trk":
        save_trk(sft, streamlines_path.as_posix(), bbox_valid_check=False)
    elif streamlines_path.suffix.lower() == ".tck":
        save_tck(sft, streamlines_path.as_posix(), bbox_valid_check=False)
    else:
        raise ValueError(f"Unsupported streamline output format: {streamlines_path.suffix}")

    length_summary = {
        "mean_mm": float(np.mean(lengths)) if lengths else 0.0,
        "min_mm": float(np.min(lengths)) if lengths else 0.0,
        "max_mm": float(np.max(lengths)) if lengths else 0.0,
    }
    return {
        "streamline_count": len(streamlines),
        "candidate_streamline_count": len(candidate_streamlines),
        "rejected_short_streamlines": rejected_short,
        "rejected_long_streamlines": rejected_long,
        "seed_voxel_count": seed_voxel_count,
        "seed_count": int(len(seeds)),
        "fa_threshold": float(config.min_fa),
        "tracking_mask_voxels": int(tracking_mask.sum()),
        "length": length_summary,
    }


@scheduled_stage('dti_tracking')
def run_dti_tracking(config: DTITrackingConfig, *, log: Optional[Callable[[str], None]] = None) -> DTITrackingResult:
    """Prepare or execute DTI tractography outputs and manifests."""

    _validate_config(config)
    raw = config.input_bundle or find_subject_root(
        config.input_root.expanduser().resolve(),
        config.subject_id,
    )
    if raw.subject_id != config.subject_id:
        raise ValueError("input_bundle subject_id does not match DTITrackingConfig.subject_id.")
    fit_method = config.fit_method.lower()
    tracking_method = config.tracking_method.lower()
    output_format = config.output_format.lower()
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else default_task_output_dir(raw.subject_root, "tractography_dti")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    native_bundle = _native_bundle_from_expected_cache(raw.subject_root, raw.subject_id, fit_method)
    native_available = _native_cache_available(native_bundle)
    stem = f"{raw.subject_id}-dti-{tracking_method}"
    streamlines_path = output_dir / f"{stem}.{output_format}"
    manifest_path = output_dir / f"{stem}_manifest.json"
    qc_path = output_dir / f"{stem}_qc.json"

    manifest = {
        "kind": "dti_tracking",
        "version": 1,
        "subject_id": raw.subject_id,
        "backend": "dipy",
        "executed": bool(config.execute),
        "status": "running" if config.execute else "planned",
        "parameters": {
            "fit_method": fit_method,
            "tracking_method": tracking_method,
            "output_format": output_format,
            "step_size_mm": config.step_size_mm,
            "max_angle_deg": config.max_angle_deg,
            "seed_density": config.seed_density,
            "max_seed_voxels": config.max_seed_voxels,
            "random_seed": config.random_seed,
            "min_fa": config.min_fa,
            "min_length_mm": config.min_length_mm,
            "max_length_mm": config.max_length_mm,
            "force_recompute": bool(config.force_recompute),
        },
        "raw_inputs": {
            "dwi": _file_info(raw.dwi_path),
            "bval": _file_info(raw.bval_path),
            "bvec": _file_info(raw.bvec_path),
            "mask": _file_info(raw.mask_path),
        },
        "native_dti_cache": {
            "available": native_available,
            "bundle_dir": native_bundle.bundle_dir.as_posix(),
            "manifest": _file_info(native_bundle.manifest_path),
            "dti_nrrd": _file_info(native_bundle.dti_nrrd_path),
            "fa": _file_info(native_bundle.fa_nii_path),
            "md": _file_info(native_bundle.mean_diffusivity_nii_path),
        },
        "outputs": {
            "streamlines": streamlines_path.as_posix(),
            "qc": qc_path.as_posix(),
        },
    }
    qc = {
        "subject_id": raw.subject_id,
        "status": "running" if config.execute else "planned",
        "executed": bool(config.execute),
        "native_cache_available": native_available,
        "streamline_count": None,
        "length_mean_mm": None,
        "length_min_mm": None,
        "length_max_mm": None,
        "rejected_short_streamlines": None,
        "rejected_long_streamlines": None,
        "mask_coverage": None,
    }
    if config.execute:
        execution_summary = _run_dipy_dti_tracking(config, streamlines_path=streamlines_path, log=log)
        length = execution_summary["length"]  # type: ignore[index]
        manifest["status"] = "completed"
        qc.update(
            {
                "status": "completed",
                "streamline_count": execution_summary["streamline_count"],
                "candidate_streamline_count": execution_summary["candidate_streamline_count"],
                "rejected_short_streamlines": execution_summary["rejected_short_streamlines"],
                "rejected_long_streamlines": execution_summary["rejected_long_streamlines"],
                "seed_voxel_count": execution_summary["seed_voxel_count"],
                "seed_count": execution_summary["seed_count"],
                "tracking_mask_voxels": execution_summary["tracking_mask_voxels"],
                "length_mean_mm": length["mean_mm"],  # type: ignore[index]
                "length_min_mm": length["min_mm"],  # type: ignore[index]
                "length_max_mm": length["max_mm"],  # type: ignore[index]
                "streamlines_exists": streamlines_path.is_file(),
            }
        )
        manifest["execution_summary"] = execution_summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(f"[Tractography] {'Completed' if config.execute else 'Prepared'} DTI tracking: {manifest_path}", log)

    return DTITrackingResult(
        subject_id=raw.subject_id,
        output_dir=output_dir,
        streamlines_path=streamlines_path,
        manifest_path=manifest_path,
        qc_path=qc_path,
        native_bundle_dir=native_bundle.bundle_dir,
        native_cache_available=native_available,
        executed=bool(config.execute),
    )
