"""Computation- and RNG-preserving runtime acceleration for TractCloud.

Installs in-process vectorized replacements into ``tractcloud`` at launch time
without modifying any file inside the checksum-verified ``TractCloud`` source
repository:
  1. Vectorized CSR ``extract_ras_features`` with branchless binary-search
     arc-length resampling (bit-for-bit identical to ``np.searchsorted``).
  2. Device-resident RNG-identical ``RealDataDataset`` + batched tensor
     ``run_inference`` (running pairwise streamline distances and top-k KNN on
     GPU when CUDA is active, eliminating the 50,000-call Python DataLoader loop
     and per-batch CPU->GPU ``global_feat`` copies, and re-enabling cuDNN in
     strict IEEE FP32 mode with TF32 disabled).
  3. Cached CSR ``extract_fibers`` + multi-threaded fast-zlib ``.vtp`` export.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any

import numpy as np


_ACTIVE_DEVICE_STR: str = "cpu"
_POLYDATA_CSR_CACHE: dict[int, tuple[Any, dict[str, Any]]] = {}


def _worker_budget() -> int:
    env_val = os.environ.get("SLICERDSPACE_STAGE_CPUS") or os.environ.get("OMP_NUM_THREADS")
    if env_val:
        try:
            return max(1, min(16, int(env_val)))
        except ValueError:
            pass
    try:
        avail = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        avail = os.cpu_count() or 1
    return max(1, min(16, avail))


def _get_polydata_csr(polydata) -> dict[str, Any]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    key = id(polydata)
    cached = _POLYDATA_CSR_CACHE.get(key)
    if cached is not None and cached[0] is polydata:
        return cached[1]

    num_fibers = int(polydata.GetNumberOfLines())
    pts_vtk = polydata.GetPoints().GetData() if polydata.GetPoints() else None
    all_points = vtk_to_numpy(pts_vtk) if pts_vtk is not None else np.zeros((0, 3), dtype=np.float32)

    cell_array = polydata.GetLines()
    if num_fibers == 0 or cell_array is None:
        cell_offsets = np.zeros(1, dtype=np.int64)
        lengths = np.zeros(0, dtype=np.int64)
        connectivity = np.zeros(0, dtype=np.int64)
    elif hasattr(cell_array, "GetOffsetsArray") and cell_array.GetOffsetsArray() is not None:
        cell_offsets = np.asarray(vtk_to_numpy(cell_array.GetOffsetsArray()), dtype=np.int64)
        lengths = np.diff(cell_offsets)
        connectivity = np.asarray(vtk_to_numpy(cell_array.GetConnectivityArray()), dtype=np.int64)
    else:
        lengths = np.zeros(num_fibers, dtype=np.int64)
        cell_offsets = np.zeros(num_fibers + 1, dtype=np.int64)
        cell_array.InitTraversal()
        pt_ids = vtk.vtkIdList()
        conn_list = []
        offset = 0
        for i in range(num_fibers):
            cell_array.GetNextCell(pt_ids)
            n = pt_ids.GetNumberOfIds()
            lengths[i] = n
            cell_offsets[i] = offset
            for j in range(n):
                conn_list.append(pt_ids.GetId(j))
            offset += n
        cell_offsets[num_fibers] = offset
        connectivity = np.array(conn_list, dtype=np.int64)

    conn_is_arange = bool(
        connectivity.shape[0] == all_points.shape[0]
        and (
            connectivity.shape[0] == 0
            or (
                connectivity[0] == 0
                and connectivity[-1] == connectivity.shape[0] - 1
                and np.all(np.diff(connectivity) == 1)
            )
        )
    )

    in_pd = polydata.GetPointData()
    pt_arrays = []
    for i in range(in_pd.GetNumberOfArrays()):
        arr = in_pd.GetArray(i)
        if arr is not None:
            pt_arrays.append((arr.GetName(), vtk_to_numpy(arr)))

    in_cd = polydata.GetCellData()
    cell_arrays = []
    for i in range(in_cd.GetNumberOfArrays()):
        arr = in_cd.GetArray(i)
        if arr is not None:
            cell_arrays.append((arr.GetName(), vtk_to_numpy(arr)))

    entry = {
        "num_fibers": num_fibers,
        "all_points": all_points,
        "cell_offsets": cell_offsets,
        "lengths": lengths,
        "connectivity": connectivity,
        "conn_is_arange": conn_is_arange,
        "pt_arrays": pt_arrays,
        "cell_arrays": cell_arrays,
    }
    _POLYDATA_CSR_CACHE[key] = (polydata, entry)
    return entry


def fast_extract_ras_features(polydata, num_points: int = 15) -> np.ndarray:
    """Vectorized exact replacement for tractcloud.inference.extract_ras_features."""
    csr = _get_polydata_csr(polydata)
    num_fibers = csr["num_fibers"]
    all_points = csr["all_points"]
    cell_offsets = csr["cell_offsets"]
    lengths = csr["lengths"]
    connectivity = csr["connectivity"]
    conn_is_arange = csr["conn_is_arange"]

    feat = np.zeros((num_fibers, num_points, 3), dtype=np.float64)
    if num_fibers == 0:
        return feat

    ordered_pts = all_points if conn_is_arange else all_points[connectivity]
    t_grid = np.linspace(0.0, 1.0, num_points, dtype=np.float64)[None, :]
    unique_lengths = np.unique(lengths)

    for n_pts_raw in unique_lengths:
        n_pts = int(n_pts_raw)
        if n_pts < 2:
            continue
        fiber_indices = np.flatnonzero(lengths == n_pts)
        batch_size = fiber_indices.size
        starts = cell_offsets[fiber_indices]
        gather_2d = starts[:, None] + np.arange(n_pts, dtype=np.int64)[None, :]
        coords = np.asarray(ordered_pts[gather_2d], dtype=np.float64)

        diffs = np.diff(coords, axis=1)
        seg_lens = np.sqrt(np.sum(diffs * diffs, axis=2))
        cum_len = np.zeros((batch_size, n_pts), dtype=np.float64)
        cum_len[:, 1:] = np.cumsum(seg_lens, axis=1)
        total_len = cum_len[:, -1]

        degenerate = total_len < 1e-12
        if np.any(degenerate):
            feat[fiber_indices[degenerate], :, :] = coords[degenerate, 0:1, :]

        valid = ~degenerate
        if not np.any(valid):
            continue

        valid_coords = coords[valid]
        valid_cum_len = cum_len[valid]
        valid_total_len = total_len[valid]
        valid_fiber_idx = fiber_indices[valid]
        n_valid = valid_fiber_idx.size

        t = t_grid * valid_total_len[:, None]
        row_idx = np.arange(n_valid, dtype=np.intp)[:, None]

        # Exact vectorized binary search matching np.searchsorted(..., side='right') - 1
        lo = np.zeros((n_valid, num_points), dtype=np.intp)
        hi = np.full((n_valid, num_points), n_pts, dtype=np.intp)
        for _ in range(int(n_pts).bit_length() + 1):
            mid = (lo + hi) >> 1
            active = (hi - lo) > 1
            if not np.any(active):
                break
            mid_safe = np.minimum(mid, n_pts - 1)
            go_right = valid_cum_len[row_idx, mid_safe] <= t
            lo = np.where(active & go_right, mid, lo)
            hi = np.where(active & (~go_right), mid, hi)

        idx = np.clip(lo, 0, n_pts - 2)
        cum_lo = valid_cum_len[row_idx, idx]
        cum_hi = valid_cum_len[row_idx, idx + 1]
        seg_len = cum_hi - cum_lo
        seg_len = np.where(seg_len < 1e-12, 1.0, seg_len)
        frac = ((t - cum_lo) / seg_len)[:, :, None]

        coords_lo = valid_coords[row_idx, idx]
        coords_hi = valid_coords[row_idx, idx + 1]
        feat[valid_fiber_idx] = coords_lo + frac * (coords_hi - coords_lo)

    return feat


def fast_extract_fibers(polydata, fiber_indices):
    """Vectorized CSR replacement for tractcloud.vtk_io.extract_fibers."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    csr = _get_polydata_csr(polydata)
    f_idx = np.asarray(fiber_indices, dtype=np.int64)
    out_pd = vtk.vtkPolyData()
    if f_idx.size == 0:
        out_pd.SetPoints(vtk.vtkPoints())
        out_pd.SetLines(vtk.vtkCellArray())
        return out_pd

    offsets_np = csr["cell_offsets"]
    conn_np = csr["connectivity"]
    pts_np = csr["all_points"]

    lens = offsets_np[f_idx + 1] - offsets_np[f_idx]
    new_offsets = np.empty(f_idx.size + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(lens, out=new_offsets[1:])
    total_conn = int(new_offsets[-1])
    src_conn_pos = np.arange(total_conn, dtype=np.int64) + np.repeat(
        offsets_np[f_idx] - new_offsets[:-1], lens
    )
    old_pt_ids = conn_np[src_conn_pos]

    if csr["conn_is_arange"] and (f_idx.size == 1 or bool(np.all(f_idx[1:] > f_idx[:-1]))):
        old_ids = old_pt_ids
        new_conn = np.arange(total_conn, dtype=np.int64)
    else:
        uniq, first_pos, inv = np.unique(old_pt_ids, return_index=True, return_inverse=True)
        order = np.argsort(first_pos)
        old_ids = uniq[order]
        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)
        new_conn = remap[inv]

    out_points = vtk.vtkPoints()
    out_points.SetData(numpy_to_vtk(np.ascontiguousarray(pts_np[old_ids]), deep=True))
    out_pd.SetPoints(out_points)

    out_lines = vtk.vtkCellArray()
    if hasattr(out_lines, "SetData"):
        out_lines.SetData(
            numpy_to_vtkIdTypeArray(new_offsets, deep=True),
            numpy_to_vtkIdTypeArray(new_conn, deep=True),
        )
    else:  # pragma: no cover
        parts = [
            np.r_[lens[k], new_conn[new_offsets[k] : new_offsets[k + 1]]]
            for k in range(f_idx.size)
        ]
        out_lines.SetCells(int(f_idx.size), numpy_to_vtkIdTypeArray(np.hstack(parts), deep=True))
    out_pd.SetLines(out_lines)

    out_pt_data = out_pd.GetPointData()
    for arr_name, np_arr in csr["pt_arrays"]:
        out_arr = numpy_to_vtk(np.ascontiguousarray(np_arr[old_ids]), deep=True)
        out_arr.SetName(arr_name)
        out_pt_data.AddArray(out_arr)

    out_cell_data = out_pd.GetCellData()
    for arr_name, np_arr in csr["cell_arrays"]:
        out_arr = numpy_to_vtk(np.ascontiguousarray(np_arr[f_idx]), deep=True)
        out_arr.SetName(arr_name)
        out_cell_data.AddArray(out_arr)

    return out_pd


def fast_write_polydata(polydata, filepath: str) -> None:
    """Fast zlib level-1 VTP / binary VTK writer."""
    import vtk

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetCompressorTypeToZLib()
        comp = writer.GetCompressor()
        if comp is not None and hasattr(comp, "SetCompressionLevel"):
            comp.SetCompressionLevel(1)
    elif ext == ".vtk":
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    writer.SetFileName(filepath)
    writer.SetInputData(polydata)
    writer.Write()


def install_tractcloud_accelerations(pipeline_mod, resolved_device: str = "cpu") -> None:
    """Monkey-patch real tractcloud modules in-memory without touching source files."""
    global _ACTIVE_DEVICE_STR
    if os.environ.get("SLICERDSPACE_DISABLE_TRACTCLOUD_ACCEL") == "1":
        return
    if not getattr(pipeline_mod, "__file__", None):
        return

    import torch
    import torch.utils.data as data
    import tractcloud.inference as inference_mod
    import tractcloud.vtk_io as vtk_io_mod

    _ACTIVE_DEVICE_STR = "cuda:0" if str(resolved_device).startswith("cuda") else "cpu"

    if str(resolved_device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    fiber_dist_fn = inference_mod._fiber_distance_efficient

    class FastRealDataDataset(data.Dataset):
        """RNG-identical dataset keeping compact KNN indices on target compute device."""

        def __init__(
            self,
            feat: np.ndarray,
            k: int = 20,
            k_global: int = 80,
            k_ds_rate: float = 0.1,
            rough_num_fiber_per_iter: int = 10000,
            progress_callback=None,
        ) -> None:
            self.feat = np.ascontiguousarray(feat, dtype=np.float32)
            self.k = int(k)
            self.k_global = int(k_global)
            num_fiber, num_point, num_feat = self.feat.shape

            # Exact upstream RNG call #1: global feature indices
            if self.k_global == 0:
                self.global_feat = np.zeros((1, num_point, num_feat, 1), dtype=np.float32)
            else:
                rand_idx = np.random.randint(0, num_fiber, self.k_global)
                self.global_feat = (
                    self.feat[rand_idx].transpose(1, 2, 0)[None, :, :, :].astype(np.float32)
                )

            dev = torch.device(
                _ACTIVE_DEVICE_STR
                if (_ACTIVE_DEVICE_STR.startswith("cuda") and torch.cuda.is_available())
                else "cpu"
            )
            self._compute_device = dev
            # (N, 3, num_point) contiguous tensor on compute device
            self._feat_transposed = torch.as_tensor(
                np.ascontiguousarray(self.feat.transpose(0, 2, 1)),
                dtype=torch.float32,
                device=dev,
            )
            self._local_feat_np: np.ndarray | None = None

            if self.k == 0:
                self._knn_global_idx: torch.Tensor | None = None
            else:
                knn_idx = torch.empty((num_fiber, self.k), dtype=torch.int64, device=dev)
                num_iter = max(num_fiber // rough_num_fiber_per_iter, 1)
                num_per_iter = (num_fiber // num_iter) + 1
                with torch.no_grad():
                    for i_iter in range(num_iter):
                        start = i_iter * num_per_iter
                        end = min((i_iter + 1) * num_per_iter, num_fiber)
                        cur_len = end - start
                        if cur_len <= 0:
                            continue
                        cur = self._feat_transposed[start:end]
                        # Exact upstream RNG calls #2..N: per-chunk random subset
                        if 0 < k_ds_rate < 1:
                            num_ds = int(cur_len * k_ds_rate)
                            ds_indices_np = np.random.choice(cur_len, size=num_ds, replace=False)
                            ds_idx_t = torch.as_tensor(ds_indices_np, dtype=torch.int64, device=dev)
                            ds_feat = cur.index_select(0, ds_idx_t)
                            dist_mat = fiber_dist_fn(cur, ds_feat, num_points=num_point)
                            topk_local = dist_mat.topk(k=self.k, largest=False, dim=-1)[1]
                            knn_idx[start:end] = ds_idx_t[topk_local] + start
                        else:
                            dist_mat = fiber_dist_fn(cur, cur, num_points=num_point)
                            topk_local = dist_mat.topk(k=self.k, largest=False, dim=-1)[1]
                            knn_idx[start:end] = topk_local + start
                        if progress_callback:
                            progress_callback((i_iter + 1) / num_iter)
                self._knn_global_idx = knn_idx

        @property
        def local_feat(self) -> np.ndarray:
            if self._local_feat_np is None:
                n_f, n_p, n_c = self.feat.shape
                if self.k == 0 or self._knn_global_idx is None:
                    self._local_feat_np = np.zeros((n_f, n_p, n_c, 1), dtype=np.float32)
                else:
                    idx_cpu = self._knn_global_idx.cpu().numpy()
                    # feat[idx_cpu] is (N, k, P, 3) -> transpose to (N, P, 3, k)
                    self._local_feat_np = np.transpose(self.feat[idx_cpu], (0, 2, 3, 1)).astype(np.float32)
            return self._local_feat_np

        def __getitem__(self, index: int):
            if self.k == 0 or self._knn_global_idx is None:
                loc = np.zeros((self.feat.shape[1], self.feat.shape[2], 1), dtype=np.float32)
            else:
                nbrs = self._knn_global_idx[index].cpu().numpy()
                loc = np.transpose(self.feat[nbrs], (1, 2, 0))
            return torch.from_numpy(self.feat[index]), torch.from_numpy(loc)

        def __len__(self) -> int:
            return int(self.feat.shape[0])

    orig_run_inference = inference_mod.run_inference

    def fast_run_inference(
        model,
        data_loader,
        global_feat,
        num_classes,
        device,
        progress_callback=None,
    ):
        ds = getattr(data_loader, "dataset", None)
        if not isinstance(ds, FastRealDataDataset):
            return orig_run_inference(
                model,
                data_loader,
                global_feat,
                num_classes,
                device,
                progress_callback=progress_callback,
            )

        target_dev = device if isinstance(device, torch.device) else torch.device(device)
        if target_dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.enabled = True

        n_total = len(ds)
        if n_total == 0:
            return []
        batch_size = int(getattr(data_loader, "batch_size", 2048) or 2048)
        total_batches = (n_total + batch_size - 1) // batch_size

        feat_t = ds._feat_transposed.to(target_dev, non_blocking=True)
        knn_idx = ds._knn_global_idx.to(target_dev, non_blocking=True) if ds._knn_global_idx is not None else None
        # global_feat is (1, P, 3, k_global) -> transpose(2, 1) -> (1, 3, P, k_global)
        global_t = torch.as_tensor(global_feat, dtype=torch.float32, device=target_dev).transpose(2, 1).contiguous()

        preds_gpu: list[torch.Tensor] = []
        with torch.no_grad():
            for b_idx, start in enumerate(range(0, n_total, batch_size)):
                end = min(start + batch_size, n_total)
                cur_b = end - start
                points = feat_t[start:end]  # (B, 3, P)
                if knn_idx is not None:
                    # feat_t[knn_idx[start:end]] -> (B, k, 3, P) -> permute -> (B, 3, P, k)
                    k_local = feat_t[knn_idx[start:end]].permute(0, 2, 3, 1).contiguous()
                else:
                    k_local = torch.zeros(
                        (cur_b, points.shape[1], points.shape[2], 1),
                        dtype=torch.float32,
                        device=target_dev,
                    )
                k_global_b = global_t.expand(cur_b, -1, -1, -1)
                info = torch.cat((k_local, k_global_b), dim=3)
                logits = model(points, info).view(-1, num_classes)
                preds_gpu.append(logits.argmax(dim=1))
                if progress_callback:
                    progress_callback((b_idx + 1) / total_batches)

        return torch.cat(preds_gpu, dim=0).cpu().numpy().tolist()

    inference_mod.extract_ras_features = fast_extract_ras_features
    inference_mod.RealDataDataset = FastRealDataDataset
    inference_mod.run_inference = fast_run_inference

    vtk_io_mod.extract_fibers = fast_extract_fibers
    vtk_io_mod.write_polydata = fast_write_polydata

    pipeline_mod.extract_ras_features = fast_extract_ras_features
    pipeline_mod.RealDataDataset = FastRealDataDataset
    pipeline_mod.run_inference = fast_run_inference
    pipeline_mod.extract_fibers = fast_extract_fibers
    pipeline_mod.write_polydata = fast_write_polydata

    # Patch TractCloudPipeline methods to keep cuDNN enabled on CUDA, write VTPs in parallel,
    # and release cached CSR polydata buffers cleanly after each file.
    cls_pipe = getattr(pipeline_mod, "TractCloudPipeline", None)
    if cls_pipe is not None:
        orig_run_on_polydata = cls_pipe.run_on_polydata
        orig_run_on_file = cls_pipe.run_on_file

        def patched_run_on_polydata(self, polydata):
            global _ACTIVE_DEVICE_STR
            _ACTIVE_DEVICE_STR = str(self.device)
            return orig_run_on_polydata(self, polydata)

        def patched_run_on_file(self, input_path, output_dir, create_mrb=False):
            if create_mrb:
                try:
                    return orig_run_on_file(self, input_path, output_dir, create_mrb=True)
                finally:
                    _POLYDATA_CSR_CACHE.clear()

            polydata = vtk_io_mod.read_polydata(input_path)
            self.reporter.status(
                f"Loaded {polydata.GetNumberOfLines()} streamlines from {os.path.basename(input_path)}"
            )
            try:
                result = self.run_on_polydata(polydata)
                os.makedirs(output_dir, exist_ok=True)
                output_files: dict[str, dict[str, str]] = {}
                write_jobs: list[tuple[Any, str]] = []
                full_names = pipeline_mod.TRACT_FULL_NAMES

                for category, tracts in result.items():
                    cat_dir = os.path.join(output_dir, category)
                    os.makedirs(cat_dir, exist_ok=True)
                    output_files[category] = {}
                    for tract_name, tract_pd in tracts.items():
                        full_name = full_names.get(tract_name, tract_name)
                        filepath = os.path.join(cat_dir, f"{full_name} ({tract_name}).vtp")
                        output_files[category][tract_name] = filepath
                        write_jobs.append((tract_pd, filepath))

                if write_jobs:
                    workers = max(1, min(_worker_budget(), len(write_jobs), 8))
                    if workers == 1:
                        for pd_obj, dst in write_jobs:
                            fast_write_polydata(pd_obj, dst)
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            futs = [pool.submit(fast_write_polydata, pd_obj, dst) for pd_obj, dst in write_jobs]
                            for fut in futs:
                                fut.result()

                total_tracts = sum(len(t) for t in output_files.values())
                self.reporter.result(total_tracts, output_dir, total_time=None)
                return output_files
            finally:
                _POLYDATA_CSR_CACHE.pop(id(polydata), None)

        cls_pipe.run_on_polydata = patched_run_on_polydata
        cls_pipe.run_on_file = patched_run_on_file
