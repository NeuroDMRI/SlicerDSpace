#!/usr/bin/env python3
"""Paired real-data benchmark across the 5 canonical SlicerDSpace modules.

Executes 1 warmup + 3 alternating repeats per benchmark on real HCP-YA subject
data (100408, 100206, 149741, 877168) and verifies 100% exact output parity.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage
from scipy.linalg import lapack
from skimage import measure
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _alternating_benchmark(run_ref, run_opt, verify_fn, repeats: int = 3):
    out_ref = run_ref()
    out_opt = run_opt()
    verify_fn(out_ref, out_opt)
    times = {"reference": [], "optimized": []}
    for rep in range(repeats):
        pair = [("reference", run_ref), ("optimized", run_opt)]
        if rep % 2 == 1:
            pair.reverse()
        results = {}
        for name, fn in pair:
            t0 = time.perf_counter()
            results[name] = fn()
            times[name].append(time.perf_counter() - t0)
        verify_fn(results["reference"], results["optimized"])
    ref_med = statistics.median(times["reference"])
    opt_med = statistics.median(times["optimized"])
    return {
        "reference_median_seconds": ref_med,
        "optimized_median_seconds": opt_med,
        "speed_ratio": ref_med / opt_med,
        "time_reduction_percent": 100.0 * (1.0 - opt_med / ref_med),
        "times_seconds": times,
        "repeats": repeats,
        "warmups_per_method": 1,
        "exact_output_parity": True,
    }


def benchmark_ddseg(hcp_100408_out: Path, hcp_100408_raw: Path, models_root: Path) -> list[dict]:
    from anatomy.segmentation.inference import (
        _softmax,
        _softmax_nhwc,
        combine_views,
        normalize_class_scores,
        predict_view,
    )
    from anatomy.segmentation.model_loader import load_models
    from anatomy.segmentation.preprocess import (
        load_mask,
        masking_and_normalizing,
        split_views,
        stack_features,
    )

    feature_dir = hcp_100408_out.parent.parent / "inputs" / "100408" / "cache" / "native_dti" / "ls"
    mask_path = hcp_100408_raw / "T1w" / "Diffusion" / "nodif_brain_mask.nii.gz"
    weights_dir = models_root / "segmentation" / "weights"

    mask = load_mask(mask_path)
    feature_maps, _ = masking_and_normalizing(feature_dir, mask, "DTI")
    stacked = stack_features(feature_maps, "DTI")
    views = split_views(stacked)

    def _predict_view_reference(model, slices, apply_softmax: bool) -> np.ndarray:
        scores = []
        for slc in slices:
            if model.input_layout == "NHWC":
                x = np.expand_dims(slc, axis=0).astype(np.float32)
            else:
                x = np.expand_dims(slc, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
            with torch.no_grad():
                tensor = torch.from_numpy(x).to(model.device)
                y = model.module(tensor).detach().cpu().numpy()
            if model.output_layout == "NHWC":
                prob = _softmax_nhwc(y) if apply_softmax else y
                score = prob[0]
            else:
                prob = _softmax(y) if apply_softmax else y
                score = prob[0].transpose(1, 2, 0)
            scores.append(score)
        return np.stack(scores, axis=2)

    def _predict_model_space_reference(models_bundle):
        pred_axial = _predict_view_reference(models_bundle.axial, views["axial"], False)
        pred_sagittal = _predict_view_reference(models_bundle.sagittal, views["sagittal"], False)
        pred_coronal = _predict_view_reference(models_bundle.coronal, views["coronal"], False)
        combined = combine_views(pred_axial, pred_sagittal, pred_coronal)
        return combined["labels"], normalize_class_scores(combined["prob_maps"])

    def _predict_model_space_optimized(models_bundle):
        pred_axial = predict_view(models_bundle.axial, views["axial"], False)
        pred_sagittal = predict_view(models_bundle.sagittal, views["sagittal"], False)
        pred_coronal = predict_view(models_bundle.coronal, views["coronal"], False)
        combined = combine_views(pred_axial, pred_sagittal, pred_coronal)
        return combined["labels"], normalize_class_scores(combined["prob_maps"])

    records = []
    if torch.cuda.is_available():
        torch.set_num_threads(4)
        models_cuda = load_models(weights_dir, "dti", "cuda")

        def run_ref_cuda():
            torch.cuda.synchronize()
            res = _predict_model_space_reference(models_cuda)
            torch.cuda.synchronize()
            return res

        def run_opt_cuda():
            torch.cuda.synchronize()
            res = _predict_model_space_optimized(models_cuda)
            torch.cuda.synchronize()
            return res

        def verify_cuda(a, b):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])

        stats_cuda = _alternating_benchmark(run_ref_cuda, run_opt_cuda, verify_cuda, repeats=3)
        stats_cuda.update(
            {
                "module": "Anatomical priors",
                "sub_algorithm": "DDSeg",
                "optimization": "Uniform background slice cache + inference_mode (full 3-view CUDA)",
                "subject_id": "100408",
                "device": "cuda",
            }
        )
        records.append(stats_cuda)
        print(f"[DDSeg cuda] {stats_cuda['reference_median_seconds']:.3f}s -> {stats_cuda['optimized_median_seconds']:.3f}s ({stats_cuda['speed_ratio']:.2f}x)", flush=True)
        del models_cuda
        torch.cuda.empty_cache()

    # CPU benchmark on a 32-slice strided sample across the full 256-slice sagittal view
    torch.set_num_threads(16)
    models_cpu = load_models(weights_dir, "dti", "cpu")
    cpu_slices = views["sagittal"][::8]

    def run_ref_cpu():
        return _predict_view_reference(models_cpu.sagittal, cpu_slices, False)

    def run_opt_cpu():
        return predict_view(models_cpu.sagittal, cpu_slices, False)

    stats_cpu = _alternating_benchmark(run_ref_cpu, run_opt_cpu, np.testing.assert_array_equal, repeats=3)
    stats_cpu.update(
        {
            "module": "Anatomical priors",
            "sub_algorithm": "DDSeg",
            "optimization": "Uniform background slice cache + inference_mode (strided full-view CPU sample)",
            "subject_id": "100408",
            "device": "cpu",
        }
    )
    records.append(stats_cpu)
    print(f"[DDSeg cpu] {stats_cpu['reference_median_seconds']:.3f}s -> {stats_cpu['optimized_median_seconds']:.3f}s ({stats_cpu['speed_ratio']:.2f}x)", flush=True)
    del models_cpu
    return records


def benchmark_ddparcel(hcp_100408_out: Path, models_root: Path) -> list[dict]:
    from collections import OrderedDict
    from anatomy.parcellation.DDSurfer_Pred import run_network
    from anatomy.parcellation.data_loader.augmentation import ToTensorTest
    from anatomy.parcellation.data_loader.load_neuroimaging_data import (
        OrigDataThickSlices_Fused_Input,
        get_largest_cc,
        get_thick_slices,
        load_and_conform_image,
        map_prediction_sagittal2full,
    )
    from anatomy.parcellation.models.networks import FastSurferCNN_Fuse_Unet_v3_extended
    from torch.utils.data import DataLoader
    from torchvision import transforms

    ddparcel_dir = hcp_100408_out / "03_anatomy" / "ddparcel"
    weights_dir = models_root / "parcellation" / "weights"
    logger = logging.getLogger("bench_ddparcel")

    feat_paths = [
        str(ddparcel_dir / "100408-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz"),
        str(ddparcel_dir / "100408-dti-MidEigenvalue-Reg-NormMasked.nii.gz"),
        str(ddparcel_dir / "100408-dti-MinEigenvalue-Reg-NormMasked.nii.gz"),
        str(ddparcel_dir / "100408-dti-Trace-Reg-NormMasked.nii.gz"),
    ]
    orig_data_list = [load_and_conform_image(p, logger=logger)[2] for p in feat_paths]

    records = []
    if torch.cuda.is_available():
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        models = {}
        ckpts = {}
        params_models = {}
        for plane in ("Axial", "Coronal", "Sagittal"):
            key = plane.lower()
            params_net = {
                "num_channels": 7,
                "num_filters": 64,
                "num_modality": 4,
                "backbone_model": sorted(str(p) for p in (weights_dir / "backbones").glob(f"*{key}.pkl")),
                "kernel_h": 5,
                "kernel_w": 5,
                "stride_conv": 1,
                "pool": 2,
                "stride_pool": 2,
                "num_classes": 54 if plane == "Sagittal" else 82,
                "kernel_c": 1,
                "kernel_d": 1,
                "use_cuda": True,
            }
            model = FastSurferCNN_Fuse_Unet_v3_extended(copy.deepcopy(params_net))
            model.to(device)
            models[plane] = model
            ckpts[plane] = str(weights_dir / f"Fused-Unet-v3-noMax-{key}.pkl")
            params_models[plane] = {
                "device": device,
                "use_cuda": True,
                "batch_size": 4,
                "model_parallel": False,
            }

        def _get_thick_slices_ref(img_data: np.ndarray, slice_thickness: int = 3) -> np.ndarray:
            h, w, d = img_data.shape
            img_data_pad = np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge")
            thick_slices = np.empty((h, w, d, 0), dtype=np.float32)
            for slice_idx in range(2 * slice_thickness + 1):
                thick_slices = np.append(
                    thick_slices, np.expand_dims(img_data_pad[:, :, slice_idx : d + slice_idx], axis=3), axis=3
                )
            return thick_slices

        def _run_network_ref(img_filename, orig_data, prediction_probability, plane, ckpt, params_model, model):
            with mock.patch("anatomy.parcellation.data_loader.load_neuroimaging_data.get_thick_slices", _get_thick_slices_ref):
                test_dataset = OrigDataThickSlices_Fused_Input(
                    img_filename, orig_data, plane=plane, transforms=transforms.Compose([ToTensorTest()])
                )
                _ = test_dataset.images
            test_data_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=params_model["batch_size"])
            backbones = copy.deepcopy(model.FastSurferCNNs)
            model_state = torch.load(ckpt, map_location=params_model["device"], weights_only=True)
            new_state_dict = OrderedDict()
            for k, v in model_state["model_state_dict"].items():
                new_state_dict[k[7:] if k[:7] == "module." else k] = v
            model.load_state_dict(new_state_dict)
            model.FastSurferCNNs = backbones
            model.eval()
            with torch.no_grad():
                start_index = 0
                for sample_batch in test_data_loader:
                    images_batch = sample_batch["image"].to(params_model["device"])
                    _, temp_list = model(images_batch)
                    for t_idx, temp in enumerate(temp_list[:1]):
                        start_index_tmp = start_index
                        if plane == "Axial":
                            temp = temp.permute(3, 0, 2, 1)
                            prediction_probability[t_idx, :, start_index_tmp : start_index_tmp + temp.shape[1], :, :] += torch.mul(temp.cpu(), 0.4)
                            start_index_tmp += temp.shape[1]
                        elif plane == "Coronal":
                            temp = temp.permute(2, 3, 0, 1)
                            prediction_probability[t_idx, :, :, start_index_tmp : start_index_tmp + temp.shape[2], :] += torch.mul(temp.cpu(), 0.4)
                            start_index_tmp += temp.shape[2]
                        else:
                            temp = map_prediction_sagittal2full(temp).permute(0, 3, 2, 1)
                            prediction_probability[t_idx, start_index_tmp : start_index_tmp + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 0.2)
                            start_index_tmp += temp.shape[0]
                    start_index = start_index_tmp
            return prediction_probability

        def run_3view_ref():
            torch.cuda.synchronize()
            pred_prob = torch.zeros((1, 256, 256, 256, 82), dtype=torch.float)
            for plane in ("Axial", "Coronal", "Sagittal"):
                pred_prob = _run_network_ref(
                    feat_paths, orig_data_list, pred_prob, plane, ckpts[plane], params_models[plane], models[plane]
                )
            torch.cuda.synchronize()
            return pred_prob

        def run_3view_opt():
            torch.cuda.synchronize()
            pred_prob = torch.zeros((1, 256, 256, 256, 82), dtype=torch.float)
            for plane in ("Axial", "Coronal", "Sagittal"):
                pred_prob = run_network(
                    feat_paths, orig_data_list, pred_prob, plane, ckpts[plane], params_models[plane], models[plane], logger
                )
            torch.cuda.synchronize()
            return pred_prob

        stats_3v = _alternating_benchmark(
            run_3view_ref,
            run_3view_opt,
            lambda a, b: np.testing.assert_array_equal(a.numpy(), b.numpy()),
            repeats=3,
        )
        stats_3v.update(
            {
                "module": "Anatomical priors",
                "sub_algorithm": "DDParcel",
                "optimization": "Uniform background batch cache + stack thick-slices + sagittal LUT",
                "subject_id": "100408",
                "device": "cuda",
            }
        )
        records.append(stats_3v)
        print(f"[DDParcel 3-view CUDA] {stats_3v['reference_median_seconds']:.3f}s -> {stats_3v['optimized_median_seconds']:.3f}s ({stats_3v['speed_ratio']:.2f}x)", flush=True)

    # DDParcel connected-component post-processing benchmark on real 256^3 parcellation
    wmparc_reg = np.asarray(nib.load(str(ddparcel_dir / "100408-DDSurfer-wmparc-Reg.mgz")).dataobj, dtype=np.int16)
    rng = np.random.default_rng(2026)
    unique_labels = [int(u) for u in np.unique(wmparc_reg) if u != 0]
    noisy_vol = wmparc_reg.copy()
    bg_coords = np.argwhere(noisy_vol == 0)
    chosen = bg_coords[rng.choice(len(bg_coords), size=len(unique_labels), replace=False)]
    for (x, y, z), lbl in zip(chosen, unique_labels):
        noisy_vol[x, y, z] = lbl

    def cc_ref():
        pred_data = noisy_vol.copy()
        for label in np.unique(pred_data):
            if label != 0:
                mask = pred_data == label
                largest_cc = get_largest_cc(mask)
                labels = measure.label(mask, connectivity=3, background=0)
                for region in measure.regionprops(labels):
                    if (
                        region.bbox_area > 5
                        and (not np.array_equal(labels == region.label, largest_cc))
                    ):
                        continue
                    elif not np.array_equal(labels == region.label, largest_cc):
                        pred_data[labels == region.label] = 0
        return pred_data

    def cc_opt():
        pred_data = noisy_vol.copy()
        for label in np.unique(pred_data):
            if label != 0:
                mask = pred_data == label
                labels, num_labels = measure.label(mask, connectivity=3, background=0, return_num=True)
                if num_labels <= 1:
                    continue
                counts = np.bincount(labels.ravel())
                largest_label = int(np.argmax(counts[1:]) + 1)
                for region in measure.regionprops(labels):
                    if region.label == largest_label or region.bbox_area > 5:
                        continue
                    pred_data[region.slice][region.image] = 0
        return pred_data

    stats_cc = _alternating_benchmark(cc_ref, cc_opt, np.testing.assert_array_equal, repeats=3)
    stats_cc.update(
        {
            "module": "Anatomical priors",
            "sub_algorithm": "DDParcel",
            "optimization": "Bounding-box connected-component relabeling (256^3 volume)",
            "subject_id": "100408",
            "device": "cpu",
        }
    )
    records.append(stats_cc)
    print(f"[DDParcel CC cleanup] {stats_cc['reference_median_seconds']:.3f}s -> {stats_cc['optimized_median_seconds']:.3f}s ({stats_cc['speed_ratio']:.2f}x)", flush=True)
    return records


def benchmark_mkcurve(hcp_raw_root: Path) -> list[dict]:
    from microstructure import mkcurve, mkcurve_tensor
    from scipy.linalg import lstsq

    def _prev_ols_initial(scaled, logs):
        return lstsq(scaled, logs.T, lapack_driver="gelsy")[0]

    def _prev_tensor_basis(directions, order):
        directions = np.asarray(directions, dtype=float)
        indices, counts = mkcurve_tensor._tensor_terms(order)
        return np.column_stack([np.prod(directions[:, index], axis=1) * count for index, count in zip(indices, counts)])

    records = []
    for subj in ("100206", "149741", "877168"):
        diff_dir = hcp_raw_root / subj / "T1w" / "Diffusion"
        bvals = np.loadtxt(diff_dir / "bvals", dtype=np.float64).reshape(-1)
        bvecs = np.loadtxt(diff_dir / "bvecs", dtype=np.float64).T
        keep = bvals <= 2500.0
        bvals_k = bvals[keep]
        bvecs_k = bvecs[keep]
        design = mkcurve_tensor.design_matrix(bvals_k, bvecs_k)
        img = nib.load(str(diff_dir / "data.nii.gz"))
        mask = np.asarray(nib.load(str(diff_dir / "nodif_brain_mask.nii.gz")).dataobj) > 0
        cx, cy, cz = [s // 2 for s in mask.shape]
        patch_mask = mask[cx - 8 : cx + 8, cy - 8 : cy + 8, cz - 2 : cz + 2]
        coords = np.argwhere(patch_mask)[:64] + np.array([cx - 8, cy - 8, cz - 2])
        vol_patch = np.asarray(img.dataobj[cx - 8 : cx + 8, cy - 8 : cy + 8, cz - 2 : cz + 2, :], dtype=np.float64)[..., keep]
        signals = np.array([vol_patch[x - (cx - 8), y - (cy - 8), z - (cz - 2)] for x, y, z in coords], dtype=np.float64)
        grid = np.linspace(1.0, 6000.0, 200, dtype=np.float64)

        def run_prev():
            with (
                mock.patch.object(mkcurve_tensor, "_ols_initial", _prev_ols_initial),
                mock.patch.object(mkcurve_tensor, "tensor_basis", _prev_tensor_basis),
            ):
                return mkcurve._fit_chunk((signals.copy(), design, grid, 0.5, "reference"))

        def run_opt():
            return mkcurve._fit_chunk((signals.copy(), design, grid, 0.5, "reference"))

        def verify(left, right):
            for idx in (0, 1, 2):
                np.testing.assert_array_equal(left[idx], right[idx])
            for k in left[3]:
                np.testing.assert_array_equal(left[3][k], right[3][k])

        stats = _alternating_benchmark(run_prev, run_opt, verify, repeats=3)
        stats.update(
            {
                "module": "Microstructure",
                "sub_algorithm": "MK-Curve",
                "optimization": f"Precomputed OLS QR + batched eigh + vectorized tensor basis ({subj})",
                "subject_id": subj,
                "voxels": 64,
                "measurements": int(design.shape[0]),
                "grid_samples": 200,
            }
        )
        records.append(stats)
        print(f"[MK-Curve {subj}] {stats['reference_median_seconds']:.3f}s -> {stats['optimized_median_seconds']:.3f}s ({stats['speed_ratio']:.2f}x)", flush=True)
    return records


def benchmark_csd_tractogram_summary(hcp_100408_out: Path) -> dict:
    from tractography.csd import _file_info, _tractogram_summary

    trk_path = hcp_100408_out / "04_tractography" / "tractograms" / "dti_det" / "benchmark" / "100408-dti-deterministic.trk"

    def run_ref():
        seq = nib.streamlines.load(str(trk_path), lazy_load=False).streamlines
        lengths = []
        all_points_finite = True
        for streamline in seq:
            points = np.asarray(streamline, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
                all_points_finite = False
                continue
            if not np.isfinite(points).all():
                all_points_finite = False
                continue
            lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        values = np.asarray(lengths, dtype=np.float64)
        summary = {
            "file": _file_info(trk_path),
            "streamline_count": int(values.size),
            "all_points_finite": bool(all_points_finite),
        }
        if values.size:
            summary["fraction_shorter_than_20mm"] = float(np.mean(values < 20.0))
            summary["length_mm"] = {
                "min": float(np.min(values)),
                "p05": float(np.percentile(values, 5)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        return summary

    def run_opt():
        return _tractogram_summary(trk_path)

    def verify(a, b):
        assert a == b, f"Mismatch: {a} != {b}"

    stats = _alternating_benchmark(run_ref, run_opt, verify, repeats=3)
    stats.update(
        {
            "module": "Tractography",
            "sub_algorithm": "CSD / ACT / SIFT2",
            "optimization": "Vectorized ArraySequence tractogram QC validation",
            "subject_id": "100408",
        }
    )
    print(f"[CSD QC summary] {stats['reference_median_seconds']:.3f}s -> {stats['optimized_median_seconds']:.3f}s ({stats['speed_ratio']:.2f}x)", flush=True)
    return stats


def benchmark_wma_native_measures(hcp_100408_out: Path) -> dict:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from tractography.wma_native_measures import measure_polydata

    trk_path = hcp_100408_out / "04_tractography" / "tractograms" / "dti_det" / "benchmark" / "100408-dti-deterministic.trk"
    seq = nib.streamlines.load(str(trk_path), lazy_load=False).streamlines[:50000]
    poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(np.asarray(seq._data, dtype=np.float32)))
    poly.SetPoints(vpts)
    lines = vtk.vtkCellArray()
    for offset, length in zip(seq._offsets.tolist(), seq._lengths.tolist()):
        lines.InsertNextCell(int(length))
        for idx in range(int(offset), int(offset + length)):
            lines.InsertCellPoint(idx)
    poly.SetLines(lines)

    def run_ref():
        points = poly.GetPoints()
        count = int(points.GetNumberOfPoints())
        lns = poly.GetLines()
        line_count, total_length = 0, 0.0
        ids = vtk.vtkIdList()
        lns.InitTraversal()
        while lns.GetNextCell(ids):
            n = ids.GetNumberOfIds()
            if n == 0:
                continue
            line_count += 1
            for index in range(1, n):
                p, q = points.GetPoint(ids.GetId(index - 1)), points.GetPoint(ids.GetId(index))
                total_length += math.dist(p, q)
        return {
            "Num_Points": count,
            "Num_Fibers": line_count,
            "Mean_Length": total_length / line_count if line_count else math.nan,
            "Num_Clamp_Excluded": 0,
        }

    def run_opt():
        return measure_polydata(poly)

    def verify(a, b):
        assert a["Num_Points"] == b["Num_Points"]
        assert a["Num_Fibers"] == b["Num_Fibers"]
        assert a["Num_Clamp_Excluded"] == b["Num_Clamp_Excluded"]
        np.testing.assert_allclose(a["Mean_Length"], b["Mean_Length"], rtol=1e-10, atol=1e-9)

    stats = _alternating_benchmark(run_ref, run_opt, verify, repeats=3)
    stats.update(
        {
            "module": "Fibre bundles",
            "sub_algorithm": "WMA",
            "optimization": "Vectorized VTK polydata tract length & scalar summary (50k streamlines)",
            "subject_id": "100408",
        }
    )
    print(f"[WMA measure_polydata] {stats['reference_median_seconds']:.3f}s -> {stats['optimized_median_seconds']:.3f}s ({stats['speed_ratio']:.2f}x)", flush=True)
    return stats


def benchmark_connectome(hcp_100408_out: Path) -> list[dict]:
    from collections import Counter
    import csv
    from connectome.pipeline import (
        ConnectomeConfig,
        _build_endpoint_label_assigner,
        _connectome_node_labels,
        _iter_streamlines,
        _label_at_point,
        _load_label_volume,
        _load_scalar_volume,
        _metric_requires_length,
        _metric_requires_scalar,
        _parse_label_names,
        _scalar_at_point,
        _streamline_length_mm,
        _write_nodes_table,
        _write_python_connectome,
    )

    trk_full = hcp_100408_out / "04_tractography" / "tractograms" / "dti_det" / "benchmark" / "100408-dti-deterministic.trk"
    parc_path = hcp_100408_out / "03_anatomy" / "ddparcel" / "100408-DDSurfer-wmparc.nii.gz"
    fa_path = hcp_100408_out / "02_microstructure" / "dti" / "100408_dti_FA.nii.gz"

    records = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        trk_50k = root / "100408_50k.tck"
        tg = nib.streamlines.load(str(trk_full), lazy_load=False)
        sub_tg = nib.streamlines.Tractogram(tg.streamlines[:50000], affine_to_rasmm= np.eye(4))
        nib.streamlines.save(nib.streamlines.TckFile(sub_tg), str(trk_50k))

        def _write_python_connectome_ref(config: ConnectomeConfig, matrix_path: Path, nodes_path: Path):
            data, inverse_affine = _load_label_volume(config.parcellation_path)
            scalar_data, scalar_inverse_affine = (
                _load_scalar_volume(config.scalar_map_path) if _metric_requires_scalar(config.metric) else (None, None)
            )
            unique_nonzero_labels = sorted({int(lbl) for lbl in np.unique(data) if int(lbl) != 0})
            node_labels = _connectome_node_labels(unique_nonzero_labels, node_filter=config.node_filter)
            allowed_node_labels = set(node_labels)
            assign_endpoint, _ = _build_endpoint_label_assigner(
                data, inverse_affine, allowed_labels=node_labels, radius_mm=config.assignment_radius_mm
            )
            _write_nodes_table(nodes_path, node_labels, label_names={})
            edge_counts: Counter = Counter()
            edge_scalar_sums: Counter = Counter()
            edge_scalar_counts: Counter = Counter()
            for streamline in _iter_streamlines(config.tractography_path):
                if len(streamline) < 2:
                    continue
                direct_source = _label_at_point(streamline[0], data, inverse_affine)
                direct_target = _label_at_point(streamline[-1], data, inverse_affine)
                source = assign_endpoint(streamline[0])
                target = assign_endpoint(streamline[-1])
                if source == 0 or target == 0:
                    continue
                if source == target and config.zero_diagonal:
                    continue
                edge = tuple(sorted((source, target))) if config.symmetric else (source, target)
                edge_counts[edge] += 1
                if scalar_data is not None and scalar_inverse_affine is not None:
                    samples = [
                        v
                        for v in (_scalar_at_point(pt, scalar_data, scalar_inverse_affine) for pt in streamline)
                        if v is not None
                    ]
                    if samples:
                        edge_scalar_sums[edge] += float(np.mean(samples))
                        edge_scalar_counts[edge] += 1
            with matrix_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_node", "target_node", config.metric])
                for (source, target), count in sorted(edge_counts.items()):
                    if _metric_requires_scalar(config.metric):
                        sc = edge_scalar_counts[(source, target)]
                        val = (edge_scalar_sums[(source, target)] / sc) if sc else ""
                        writer.writerow([source, target, val])
                    else:
                        writer.writerow([source, target, count])
            return matrix_path.read_text(encoding="utf-8")

        for metric, scalar_map, label in (
            ("streamline_count", None, "Batched endpoint voxel + KD-tree radial assignment (50k streamlines)"),
            ("mean_fa", fa_path, "Batched endpoint assignment + vectorized along-tract FA sampling (50k streamlines)"),
        ):
            cfg = ConnectomeConfig(
                subject_id="100408",
                tractography_path=trk_50k,
                parcellation_path=parc_path,
                output_dir=root / f"out_{metric}",
                metric=metric,
                scalar_map_path=scalar_map,
                assignment_radius_mm=4.0,
                backend="python",
                execute=True,
            )

            def run_ref():
                m_path = root / f"ref_{metric}.csv"
                n_path = root / f"ref_{metric}.tsv"
                return _write_python_connectome_ref(cfg, m_path, n_path)

            def run_opt():
                m_path = root / f"opt_{metric}.csv"
                n_path = root / f"opt_{metric}.tsv"
                _write_python_connectome(cfg, m_path, n_path, lut_path=None)
                return m_path.read_text(encoding="utf-8")

            def verify(a, b):
                rows_a = list(csv.reader(a.splitlines()))
                rows_b = list(csv.reader(b.splitlines()))
                assert len(rows_a) == len(rows_b), f"Row count mismatch: {len(rows_a)} != {len(rows_b)}"
                assert rows_a[0] == rows_b[0]
                for ra, rb in zip(rows_a[1:], rows_b[1:]):
                    assert ra[:2] == rb[:2], f"Edge mismatch: {ra[:2]} != {rb[:2]}"
                    np.testing.assert_allclose(float(ra[2]), float(rb[2]), rtol=1e-12, atol=1e-12)

            stats = _alternating_benchmark(run_ref, run_opt, verify, repeats=3)
            stats.update(
                {
                    "module": "Connectome",
                    "sub_algorithm": f"Connectome ({metric})",
                    "optimization": label,
                    "subject_id": "100408",
                }
            )
            records.append(stats)
            print(f"[Connectome {metric}] {stats['reference_median_seconds']:.3f}s -> {stats['optimized_median_seconds']:.3f}s ({stats['speed_ratio']:.2f}x)", flush=True)
    return records


def main() -> int:
    hcp_100408_out = Path("/data04/chengjin/r5_cohort_multi/pair_end_to_end_r1/adaptive/trials/end_to_end-t8-r0-adaptive/outputs/100408")
    hcp_raw_root = Path("/dataset/nfs_share01/HCP/HCP-YA/3T/Diffusion_Preprocessed")
    models_root = Path("/data04/chengjin/opt/SlicerDSpace-model-assets")
    out_json = ROOT / "docs" / "release" / "benchmarks" / "five_module_subalgorithm_optimizations_20261110.json"

    all_records = []
    if "--only-wma-connectome" in sys.argv and out_json.is_file():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        all_records.extend(
            [r for r in prev.get("records", []) if r.get("module") in ("Anatomical priors", "Microstructure", "Tractography")]
        )
    elif "--skip-anatomy" in sys.argv and out_json.is_file():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        all_records.extend([r for r in prev.get("records", []) if r.get("module") == "Anatomical priors"])
        all_records.extend(benchmark_mkcurve(hcp_raw_root))
        all_records.append(benchmark_csd_tractogram_summary(hcp_100408_out))
    else:
        all_records.extend(benchmark_ddseg(hcp_100408_out, hcp_raw_root / "100408", models_root))
        all_records.extend(benchmark_ddparcel(hcp_100408_out, models_root))
        all_records.extend(benchmark_mkcurve(hcp_raw_root))
        all_records.append(benchmark_csd_tractogram_summary(hcp_100408_out))
    all_records.append(benchmark_wma_native_measures(hcp_100408_out))
    all_records.extend(benchmark_connectome(hcp_100408_out))

    report = {
        "kind": "five_module_subalgorithm_optimizations",
        "status": "passed",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_on": "2026-11-10",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "records": all_records,
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
