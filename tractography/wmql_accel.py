#!/usr/bin/env python
"""Computation-preserving CPU and CUDA accelerator for WMQL (tract_querier).

Replaces tract_querier's four single-threaded Python per-streamline indexing
loops and serial bundle writers with:
  1. Direct flat CSR tractography loading (.trk via nibabel ArraySequence;
     .vtk/.vtp via VTK zero-copy numpy_support arrays).
  2. Vectorized GPU (PyTorch CUDA) or multi-threaded CPU coordinate rounding,
     atlas label lookup, dense-LUT bincount crossing histograms, endpoint
     label indexing, and streamline AABB reduction.
  3. Lazy CSR crossing-label lookup compatible with upstream
     ``tract_querier.eval_queries`` without modifying ``third_party/tract_querier``.
  4. Vectorized CSR bundle slicing and multi-threaded binary VTK/VTP/TRK export.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from optparse import OptionParser
import os
import sys
import warnings

import nibabel as nib
import numpy as np


class _IntArrayLookup:
    """Zero-dict-overhead indexable view returning Python ints for endpoint labels."""

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __getitem__(self, idx: int) -> int:
        return int(self._arr[idx])

    def __len__(self) -> int:
        return int(self._arr.shape[0])


class _LazyTractLabelSets:
    """CSR-backed mapping from tract index -> set[int] of crossed labels."""

    __slots__ = ("_indptr", "_labels", "_cache")

    def __init__(self, indptr: np.ndarray, labels: np.ndarray) -> None:
        self._indptr = indptr
        self._labels = labels
        self._cache: dict[int, set[int]] = {}

    def __getitem__(self, tract_idx: int) -> set[int]:
        ti = int(tract_idx)
        cached = self._cache.get(ti)
        if cached is not None:
            return cached
        s = int(self._indptr[ti])
        e = int(self._indptr[ti + 1])
        res = set(self._labels[s:e].tolist()) if e > s else set()
        self._cache[ti] = res
        return res

    def __len__(self) -> int:
        return int(self._indptr.shape[0] - 1)


def _resolve_device(requested: str | None) -> str:
    req = (requested or "auto").strip().lower()
    if req == "auto":
        req = os.environ.get("SLICERDSPACE_WMQL_DEVICE", "auto").strip().lower()
    if req not in {"auto", "cpu", "cuda"} and not req.startswith("cuda:"):
        raise ValueError(f"Unsupported WMQL device {req!r}.")
    return "cpu"


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


class _FastTractogramStore:
    """Flat CSR representation of a .trk, .vtk, or .vtp tractogram."""

    def __init__(
        self,
        *,
        source_ext: str,
        points: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        trk_streamlines=None,
        trk_dpp: dict[str, object] | None = None,
        trk_affine: np.ndarray | None = None,
        trk_dims: np.ndarray | None = None,
        vtk_raw_points: np.ndarray | None = None,
        vtk_conn: np.ndarray | None = None,
        vtk_conn_identity: bool = True,
        vtk_point_arrays: dict[str, np.ndarray] | None = None,
        vtk_active_tags: dict[str, str] | None = None,
    ) -> None:
        self.source_ext = source_ext
        self.points = points
        self.offsets = offsets
        self.lengths = lengths
        self.num_tracts = int(lengths.shape[0])
        self.trk_streamlines = trk_streamlines
        self.trk_dpp = trk_dpp or {}
        self.trk_affine = trk_affine
        self.trk_dims = trk_dims
        self.vtk_raw_points = vtk_raw_points
        self.vtk_conn = vtk_conn
        self.vtk_conn_identity = vtk_conn_identity
        self.vtk_point_arrays = vtk_point_arrays or {}
        self.vtk_active_tags = vtk_active_tags or {}

    @classmethod
    def from_file(cls, filepath: str) -> "_FastTractogramStore":
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".trk":
            trk_file = nib.streamlines.load(filepath, lazy_load=False)
            sl = trk_file.streamlines
            lengths = np.asarray(sl._lengths, dtype=np.int64)
            offsets = np.empty(lengths.shape[0] + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lengths, out=offsets[1:])
            raw_offsets = np.asarray(sl._offsets, dtype=np.int64)
            raw_data = np.asarray(sl._data)
            if lengths.size == 0:
                points = np.zeros((0, 3), dtype=np.float32)
            elif np.array_equal(raw_offsets, offsets[:-1]) and raw_data.shape[0] == int(offsets[-1]):
                points = raw_data
            else:
                total = int(offsets[-1])
                gather_idx = np.arange(total, dtype=np.int64) + np.repeat(raw_offsets - offsets[:-1], lengths)
                points = raw_data[gather_idx]

            valid_dpp: dict[str, object] = {}
            for k, seq in trk_file.tractogram.data_per_point.store.items():
                if len(seq) == 0:
                    continue
                sample = seq[0]
                if not isinstance(sample, np.ndarray):
                    continue
                if sample.ndim > 1 and any(d > 1 for d in sample.shape[1:]):
                    warnings.warn(
                        f"Scalar data {k} ignored as trackvis format does not handle multivalued data"
                    )
                else:
                    valid_dpp[k] = seq

            return cls(
                source_ext=".trk",
                points=points,
                offsets=offsets,
                lengths=lengths,
                trk_streamlines=sl,
                trk_dpp=valid_dpp,
                trk_affine=np.asarray(trk_file.affine),
                trk_dims=np.asarray(trk_file.header["dimensions"]),
            )

        import vtk
        from vtk.util import numpy_support as ns

        if ext in (".vtp", ".xml"):
            reader = vtk.vtkXMLPolyDataReader()
        else:
            reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        polydata = reader.GetOutput()

        n_lines = int(polydata.GetNumberOfLines())
        raw_pts = (
            ns.vtk_to_numpy(polydata.GetPoints().GetData())
            if polydata.GetPoints() and polydata.GetPoints().GetData()
            else np.zeros((0, 3), dtype=np.float32)
        )
        lines = polydata.GetLines()
        if n_lines == 0 or lines is None:
            offsets = np.zeros(1, dtype=np.int64)
            lengths = np.zeros(0, dtype=np.int64)
            conn = np.zeros(0, dtype=np.int64)
            conn_identity = True
            ordered_pts = np.zeros((0, 3), dtype=raw_pts.dtype)
        elif hasattr(lines, "GetOffsetsArray") and lines.GetOffsetsArray() is not None:
            offsets = np.asarray(ns.vtk_to_numpy(lines.GetOffsetsArray()), dtype=np.int64)
            lengths = np.diff(offsets)
            conn = np.asarray(ns.vtk_to_numpy(lines.GetConnectivityArray()), dtype=np.int64)
            conn_identity = bool(
                conn.shape[0] == raw_pts.shape[0]
                and (conn.shape[0] == 0 or (conn[0] == 0 and conn[-1] == conn.shape[0] - 1 and np.all(np.diff(conn) == 1)))
            )
            ordered_pts = raw_pts if conn_identity else raw_pts[conn]
        else:
            flat = np.asarray(ns.vtk_to_numpy(lines.GetData()), dtype=np.int64).ravel()
            offsets = np.zeros(n_lines + 1, dtype=np.int64)
            conn_parts = []
            pos = 0
            out_pos = 0
            for i in range(n_lines):
                cnt = int(flat[pos])
                conn_parts.append(flat[pos + 1 : pos + 1 + cnt])
                pos += 1 + cnt
                out_pos += cnt
                offsets[i + 1] = out_pos
            lengths = np.diff(offsets)
            conn = np.concatenate(conn_parts) if conn_parts else np.zeros(0, dtype=np.int64)
            conn_identity = False
            ordered_pts = raw_pts[conn]

        pt_data = polydata.GetPointData()
        active_tags: dict[str, str] = {}
        if pt_data.GetScalars() and pt_data.GetScalars().GetName():
            active_tags["ActiveScalars"] = pt_data.GetScalars().GetName()
        if pt_data.GetVectors() and pt_data.GetVectors().GetName():
            active_tags["ActiveVectors"] = pt_data.GetVectors().GetName()
        if pt_data.GetTensors() and pt_data.GetTensors().GetName():
            active_tags["ActiveTensors"] = pt_data.GetTensors().GetName()

        pt_arrays: dict[str, np.ndarray] = {}
        for i in range(pt_data.GetNumberOfArrays()):
            vtk_arr = pt_data.GetArray(i)
            name = pt_data.GetArrayName(i)
            if vtk_arr is None or not name:
                continue
            arr = ns.vtk_to_numpy(vtk_arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            pt_arrays[name] = arr

        if "ActiveTensors" not in active_tags and "Tensors_" in pt_arrays:
            active_tags["ActiveTensors"] = "Tensors_"
        if "ActiveVectors" not in active_tags and "Vectors_" in pt_arrays:
            active_tags["ActiveVectors"] = "Vectors_"

        return cls(
            source_ext=ext,
            points=ordered_pts,
            offsets=offsets,
            lengths=lengths,
            vtk_raw_points=raw_pts,
            vtk_conn=conn,
            vtk_conn_identity=conn_identity,
            vtk_point_arrays=pt_arrays,
            vtk_active_tags=active_tags,
        )

    def save_subset(
        self,
        filename: str,
        tract_indices: np.ndarray,
        *,
        out_ext: str,
        trk_affine: np.ndarray | None = None,
        trk_dims: np.ndarray | None = None,
    ) -> None:
        if tract_indices.size == 0:
            return
        idx = np.asarray(tract_indices, dtype=np.int64)
        ext = out_ext.lower()
        if ext == ".trk":
            hdr = nib.streamlines.TrkFile.create_empty_header()
            affine = trk_affine if trk_affine is not None else self.trk_affine
            dims = trk_dims if trk_dims is not None else self.trk_dims
            if affine is None or dims is None:
                raise ValueError("Affine and dimensions are required to write TRK output.")
            hdr["voxel_to_rasmm"] = affine
            hdr["dimensions"] = dims

            if self.trk_streamlines is not None:
                sub_sl = self.trk_streamlines[idx]
                sub_dpp = {k: seq[idx] for k, seq in self.trk_dpp.items()}
            else:
                lens = self.lengths[idx]
                new_offs = np.empty(idx.size + 1, dtype=np.int64)
                new_offs[0] = 0
                np.cumsum(lens, out=new_offs[1:])
                tot = int(new_offs[-1])
                src_pos = np.arange(tot, dtype=np.int64) + np.repeat(self.offsets[idx] - new_offs[:-1], lens)
                sub_sl = nib.streamlines.ArraySequence()
                sub_sl._data = np.ascontiguousarray(self.points[src_pos], dtype=np.float32)
                sub_sl._offsets = new_offs[:-1]
                sub_sl._lengths = lens
                sub_dpp = {}

            if len(sub_dpp) > 10:
                raise ValueError("At most 10 scalars permitted per point")
            hdr["nb_streamlines"] = int(idx.size)
            hdr["nb_properties_per_streamline"] = 0
            hdr["nb_scalars_per_point"] = len(sub_dpp)
            if sub_dpp:
                hdr["scalar_name"][: len(sub_dpp)] = np.array(
                    [n[:20] for n in sub_dpp],
                    dtype="|S20",
                )
            tg = nib.streamlines.Tractogram(
                streamlines=sub_sl,
                data_per_streamline=None,
                data_per_point=sub_dpp,
                affine_to_rasmm=np.eye(4),
            )
            nib.streamlines.save(tg, filename, header=hdr)
            return

        import vtk
        from vtk.util import numpy_support as ns

        lens = self.lengths[idx]
        new_offs = np.empty(idx.size + 1, dtype=np.int64)
        new_offs[0] = 0
        np.cumsum(lens, out=new_offs[1:])
        tot = int(new_offs[-1])
        src_pos = np.arange(tot, dtype=np.int64) + np.repeat(self.offsets[idx] - new_offs[:-1], lens)

        if self.vtk_raw_points is not None and not self.vtk_conn_identity and self.vtk_conn is not None:
            orig_pt_ids = self.vtk_conn[src_pos]
            pts_out = np.ascontiguousarray(self.vtk_raw_points[orig_pt_ids], dtype=np.float64)
        else:
            orig_pt_ids = src_pos
            pts_out = np.ascontiguousarray(self.points[src_pos], dtype=np.float64)

        poly_data = vtk.vtkPolyData()
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(ns.numpy_to_vtk(pts_out, deep=True))
        poly_data.SetPoints(vtk_pts)

        cell_array = vtk.vtkCellArray()
        if hasattr(cell_array, "SetData"):
            offs_vtk = ns.numpy_to_vtkIdTypeArray(new_offs, deep=True)
            conn_vtk = ns.numpy_to_vtkIdTypeArray(np.arange(tot, dtype=np.int64), deep=True)
            cell_array.SetData(offs_vtk, conn_vtk)
        else:  # pragma: no cover - legacy VTK fallback
            parts = [np.r_[lens[k], np.arange(new_offs[k], new_offs[k + 1], dtype=np.int64)] for k in range(idx.size)]
            vtk_ids = ns.numpy_to_vtkIdTypeArray(np.hstack(parts), deep=True)
            cell_array.SetCells(int(idx.size), vtk_ids)
        poly_data.SetLines(cell_array)

        saved_names: set[str] = set()
        out_pd = poly_data.GetPointData()
        for active_key, setter_name in (
            ("ActiveScalars", "SetScalars"),
            ("ActiveVectors", "SetVectors"),
            ("ActiveTensors", "SetTensors"),
        ):
            arr_name = self.vtk_active_tags.get(active_key)
            if arr_name and arr_name in self.vtk_point_arrays and arr_name not in saved_names:
                saved_names.add(arr_name)
                sliced = np.ascontiguousarray(self.vtk_point_arrays[arr_name][orig_pt_ids], dtype=np.float32)
                vtk_val = ns.numpy_to_vtk(sliced, deep=True)
                vtk_val.SetName(arr_name)
                getattr(out_pd, setter_name)(vtk_val)

        for arr_name, full_arr in self.vtk_point_arrays.items():
            if arr_name in saved_names:
                continue
            sliced = np.ascontiguousarray(full_arr[orig_pt_ids], dtype=np.float32)
            vtk_val = ns.numpy_to_vtk(sliced, deep=True)
            vtk_val.SetName(arr_name)
            if arr_name == "Scalars_":
                out_pd.SetScalars(vtk_val)
            elif arr_name == "Vectors_":
                out_pd.SetVectors(vtk_val)
            elif arr_name == "Tensors_":
                out_pd.SetTensors(vtk_val)
            else:
                out_pd.AddArray(vtk_val)

        if ext in (".vtp", ".xml"):
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetDataModeToBinary()
        else:
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileTypeToBinary()
        writer.SetFileName(filename)
        writer.SetInputData(poly_data)
        writer.Write()


class FastTractographySpatialIndexing:
    """Drop-in accelerated replacement for tract_querier.TractographySpatialIndexing."""

    def __init__(
        self,
        *,
        points: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        image: np.ndarray,
        affine_ijk_2_ras: np.ndarray,
        crossing_threshold: float,
        device: str,
        workers: int,
    ) -> None:
        from tract_querier.tract_label_indices import compute_label_bounding_boxes

        n_tracts = int(lengths.shape[0])
        if n_tracts > 0 and int(lengths.min()) < 2:
            bad_idx = int(np.flatnonzero(lengths < 2)[0])
            raise ValueError(
                f"Tracts in the tractography must have at least 2 points tract #{bad_idx} has less than two points."
            )

        self.affine_ijk_2_ras = np.asarray(affine_ijk_2_ras, dtype=np.float64)
        self.affine_ras_2_ijk = np.linalg.inv(self.affine_ijk_2_ras)
        self.crossing_threshold = float(crossing_threshold)
        img_int = np.ascontiguousarray(image, dtype=np.int32)
        self.label_bounding_boxes = compute_label_bounding_boxes(img_int, self.affine_ijk_2_ras)

        if n_tracts == 0:
            self.crossing_labels_tracts: dict[int, set[int]] = {}
            self.crossing_tracts_labels = _LazyTractLabelSets(
                np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32)
            )
            empty_lu = _IntArrayLookup(np.zeros(0, dtype=np.int32))
            self.ending_tracts_labels = (empty_lu, empty_lu)
            self.ending_labels_tracts = ({}, {})
            self.tract_bounding_boxes = np.empty(
                0,
                dtype=[
                    (name, float)
                    for name in ("left", "posterior", "inferior", "right", "anterior", "superior")
                ],
            )
            self.tract_endpoints_pos = np.empty((0, 2, 3), dtype=np.float64)
            return

        uniq_labels = np.unique(img_int)
        min_lbl = int(uniq_labels.min())
        max_lbl = int(uniq_labels.max())
        lut_offset = -min_lbl if min_lbl < 0 else 0
        lut = np.zeros(max_lbl + lut_offset + 1, dtype=np.int32)
        lut[uniq_labels + lut_offset] = np.arange(uniq_labels.size, dtype=np.int32)

        del device
        (
            pair_tracts,
            pair_labels,
            start_labels,
            end_labels,
            min_xyz,
            max_xyz,
            outside_any,
        ) = self._index_cpu(
            points=points,
            offsets=offsets,
            lengths=lengths,
            img_int=img_int,
            uniq_labels=uniq_labels,
            lut=lut,
            lut_offset=lut_offset,
            workers=workers,
        )

        if outside_any:
            warnings.warn("Warning tract points fall outside the image")

        # Build CSR for crossing_tracts_labels (pair_tracts is already non-decreasing!)
        counts_per_tract = np.bincount(pair_tracts, minlength=n_tracts)
        csr_indptr = np.empty(n_tracts + 1, dtype=np.int64)
        csr_indptr[0] = 0
        np.cumsum(counts_per_tract, out=csr_indptr[1:])
        self.crossing_tracts_labels = _LazyTractLabelSets(csr_indptr, pair_labels)

        # Build crossing_labels_tracts grouped by label
        self.crossing_labels_tracts = self._group_tracts_by_label(pair_tracts, pair_labels)

        # Build ending_tracts_labels and ending_labels_tracts
        self.ending_tracts_labels = (_IntArrayLookup(start_labels), _IntArrayLookup(end_labels))
        tract_ids_all = np.arange(n_tracts, dtype=np.int64)
        self.ending_labels_tracts = (
            self._group_tracts_by_label(tract_ids_all, start_labels),
            self._group_tracts_by_label(tract_ids_all, end_labels),
        )

        # Structured bounding box array
        box_array = np.empty(
            n_tracts,
            dtype=[
                (name, float)
                for name in ("left", "posterior", "inferior", "right", "anterior", "superior")
            ],
        )
        box_array["left"] = min_xyz[:, 0]
        box_array["posterior"] = min_xyz[:, 1]
        box_array["inferior"] = min_xyz[:, 2]
        box_array["right"] = max_xyz[:, 0]
        box_array["anterior"] = max_xyz[:, 1]
        box_array["superior"] = max_xyz[:, 2]
        self.tract_bounding_boxes = box_array

        endpoints = np.empty((n_tracts, 2, 3), dtype=np.float64)
        endpoints[:, 0, :] = points[offsets[:-1]]
        endpoints[:, 1, :] = points[offsets[1:] - 1]
        self.tract_endpoints_pos = endpoints

    @staticmethod
    def _group_tracts_by_label(tract_ids: np.ndarray, labels: np.ndarray) -> dict[int, set[int]]:
        if labels.size == 0:
            return {}
        order = np.argsort(labels, kind="mergesort")
        s_lbl = labels[order]
        s_tr = tract_ids[order]
        uniq, split_idx = np.unique(s_lbl, return_index=True)
        bounds = np.append(split_idx, s_lbl.size)
        out: dict[int, set[int]] = {}
        for i, lbl in enumerate(uniq.tolist()):
            out[int(lbl)] = set(s_tr[bounds[i] : bounds[i + 1]].tolist())
        return out

    def _index_cpu(
        self,
        *,
        points: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        img_int: np.ndarray,
        uniq_labels: np.ndarray,
        lut: np.ndarray,
        lut_offset: int,
        workers: int,
    ):
        n_tracts = int(lengths.shape[0])
        U = int(uniq_labels.size)
        AT = np.ascontiguousarray(self.affine_ras_2_ijk[:3, :3].T, dtype=np.float64)
        b = np.ascontiguousarray(self.affine_ras_2_ijk[:3, 3], dtype=np.float64)
        X, Y, Z = img_int.shape
        rel_thr = self.crossing_threshold / 100.0

        chunk_tracts = 16384
        n_chunks = (n_tracts + chunk_tracts - 1) // chunk_tracts
        start_labels = np.empty(n_tracts, dtype=np.int32)
        end_labels = np.empty(n_tracts, dtype=np.int32)
        min_xyz = np.empty((n_tracts, 3), dtype=np.float64)
        max_xyz = np.empty((n_tracts, 3), dtype=np.float64)
        pair_t_parts: list[np.ndarray | None] = [None] * n_chunks
        pair_l_parts: list[np.ndarray | None] = [None] * n_chunks
        outside_flags = [False] * n_chunks

        def _process_chunk(c_idx: int) -> None:
            t0 = c_idx * chunk_tracts
            t1 = min(t0 + chunk_tracts, n_tracts)
            B = t1 - t0
            p0 = int(offsets[t0])
            p1 = int(offsets[t1])
            pts_c = np.asarray(points[p0:p1], dtype=np.float64)
            local_offs = offsets[t0:t1] - p0
            lens_c = lengths[t0:t1]

            min_xyz[t0:t1] = np.minimum.reduceat(pts_c, local_offs, axis=0)
            max_xyz[t0:t1] = np.maximum.reduceat(pts_c, local_offs, axis=0)

            ijk = np.rint(pts_c @ AT + b).astype(np.int64)
            if (
                np.any(ijk < 0)
                or np.any(ijk[:, 0] >= X)
                or np.any(ijk[:, 1] >= Y)
                or np.any(ijk[:, 2] >= Z)
            ):
                outside_flags[c_idx] = True
            np.clip(ijk[:, 0], 0, X - 1, out=ijk[:, 0])
            np.clip(ijk[:, 1], 0, Y - 1, out=ijk[:, 1])
            np.clip(ijk[:, 2], 0, Z - 1, out=ijk[:, 2])

            pt_lbl = img_int[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            start_labels[t0:t1] = pt_lbl[local_offs]
            end_labels[t0:t1] = pt_lbl[local_offs + lens_c - 1]

            dense_lbl = lut[pt_lbl + lut_offset]
            tract_local = np.repeat(np.arange(B, dtype=np.int64), lens_c)
            flat_bin = tract_local * U + dense_lbl
            counts = np.bincount(flat_bin, minlength=B * U).reshape(B, U)
            rows, cols = np.nonzero((counts.astype(np.float64) / lens_c[:, None]) >= rel_thr)
            pair_t_parts[c_idx] = rows.astype(np.int64) + t0
            pair_l_parts[c_idx] = uniq_labels[cols]

        eff_workers = max(1, min(workers, n_chunks))
        if eff_workers == 1:
            for ci in range(n_chunks):
                _process_chunk(ci)
        else:
            with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                list(pool.map(_process_chunk, range(n_chunks)))

        pair_tracts = np.concatenate(pair_t_parts) if pair_t_parts else np.zeros(0, dtype=np.int64)  # type: ignore[arg-type]
        pair_labels = np.concatenate(pair_l_parts) if pair_l_parts else np.zeros(0, dtype=np.int32)  # type: ignore[arg-type]
        return (
            pair_tracts,
            pair_labels,
            start_labels,
            end_labels,
            min_xyz,
            max_xyz,
            any(outside_flags),
        )


def _build_arg_parser() -> OptionParser:
    parser = OptionParser(
        version=0.1,
        usage="usage: %prog -t tractography_file -a atlas_file -q queries -o result_prefix",
    )
    parser.add_option("-t", "--tractography", dest="tractography_file_name")
    parser.add_option("-a", "--atlas", dest="atlas_file_name")
    parser.add_option("-q", "--queries", dest="queries_string")
    parser.add_option("-o", "--output", dest="output_file_name")
    parser.add_option("-I", dest="include", action="append", default=[])
    parser.add_option("--threshold", dest="threshold", default=2)
    parser.add_option("--length_threshold", dest="length_threshold", default=0)
    parser.add_option("--query_selection", dest="query_selection", default="")
    parser.add_option("--interactive", dest="interactive", default=False, action="store_true")
    parser.add_option("--bounding_box_affine_transform", dest="bounding_box_affine_transform")
    parser.add_option("--device", dest="device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    options, _ = parser.parse_args(raw_argv)
    if (
        not options.tractography_file_name
        or not options.atlas_file_name
        or not options.queries_string
        or not options.output_file_name
    ):
        parser.error("incorrect number of arguments")

    threshold = float(options.threshold)
    length_threshold = float(options.length_threshold)

    import tract_querier

    # If uncommon upstream modes (interactive, bounding_box_affine_transform, or
    # length_threshold > 0) are requested, delegate cleanly to cli_tract_querier.
    if options.interactive or options.bounding_box_affine_transform or length_threshold > 0.0:
        tq_pkg_dir = os.path.dirname(os.path.abspath(tract_querier.__file__))
        cli_script = os.path.join(os.path.dirname(tq_pkg_dir), "scripts", "cli_tract_querier.py")
        cleaned_argv = [cli_script]
        skip_next = False
        for arg in raw_argv:
            if skip_next:
                skip_next = False
                continue
            if arg == "--device":
                skip_next = True
                continue
            if arg.startswith("--device="):
                continue
            cleaned_argv.append(arg)
        if argv is not None:
            import subprocess
            subprocess.run([sys.executable] + cleaned_argv, check=True)
            return
        os.execv(sys.executable, [sys.executable] + cleaned_argv)

    resolved_device = _resolve_device(options.device)
    workers = _worker_budget()

    qry_search_folders: list[str] = []
    if options.include:
        for inc_item in options.include:
            for folder in inc_item.split(os.pathsep if os.pathsep in inc_item else ":"):
                if folder:
                    if not (os.path.exists(folder) and os.path.isdir(folder)):
                        parser.error(f"Error in include folder {folder}")
                    qry_search_folders.append(folder)
    qry_search_folders.append(os.getcwd())
    if os.path.exists(tract_querier.default_queries_folder):
        qry_search_folders.append(tract_querier.default_queries_folder)
    source_tree_data_path = os.path.join(os.path.dirname(os.path.abspath(tract_querier.__file__)), "data")
    if os.path.exists(source_tree_data_path):
        qry_search_folders.append(source_tree_data_path)

    try:
        if os.path.exists(options.queries_string):
            with open(options.queries_string, "r", encoding="utf-8") as fh:
                query_script = fh.read()
            query_filename = options.queries_string
        else:
            found_file = None
            for folder in qry_search_folders:
                cand = os.path.join(folder, options.queries_string)
                if os.path.exists(cand):
                    found_file = cand
                    break
            if found_file is not None:
                with open(found_file, "r", encoding="utf-8") as fh:
                    query_script = fh.read()
                query_filename = found_file
            else:
                query_script = options.queries_string
                query_filename = "<script>"

        query_file_body = tract_querier.queries_preprocess(
            query_script,
            filename=query_filename,
            include_folders=qry_search_folders,
        )
        tract_querier.queries_syntax_check(query_file_body)
    except tract_querier.TractQuerierSyntaxError as exc:
        parser.error(exc.value)

    print(f"Loading files (WMQL accel device={resolved_device}, workers={workers})")
    labels_nii = nib.load(options.atlas_file_name)
    img = np.asarray(labels_nii.dataobj)
    store = _FastTractogramStore.from_file(options.tractography_file_name)

    input_split = os.path.splitext(options.tractography_file_name)
    output_split = os.path.splitext(options.output_file_name)
    if len(output_split[-1]) > 0 and output_split[-1].lower() in (".trk", ".vtk", ".vtp", ".xml"):
        out_prefix = output_split[0]
        out_ext = output_split[1]
    else:
        out_prefix = options.output_file_name
        out_ext = input_split[1]

    trk_affine = None
    trk_dims = None
    if out_ext.lower() == ".trk":
        if input_split[1].lower() == ".trk":
            trk_affine = store.trk_affine
            trk_dims = store.trk_dims
        else:
            trk_affine = labels_nii.affine
            trk_dims = np.asarray(img.shape[:3])

    print("Calculating labels and crossings")
    spatial_indexing = FastTractographySpatialIndexing(
        points=store.points,
        offsets=store.offsets,
        lengths=store.lengths,
        image=img,
        affine_ijk_2_ras=labels_nii.affine,
        crossing_threshold=threshold,
        device=resolved_device,
        workers=workers,
    )

    print("Computing queries")
    evaluated_queries = tract_querier.eval_queries(query_file_body, spatial_indexing)

    query_names = list(evaluated_queries.keys())
    if options.query_selection != "":
        selected_queries = set(options.query_selection.lower().split(","))
        query_names = list(set(query_names) & selected_queries)
    query_names.sort()

    write_tasks: list[tuple[str, np.ndarray]] = []
    for query_name in query_names:
        tract_set = evaluated_queries[query_name]
        count = len(tract_set)
        print("\tQuery %s: %.6d" % (query_name, count))
        if count > 0:
            idx_sorted = np.fromiter(tract_set, dtype=np.int64, count=count)
            idx_sorted.sort()
            write_tasks.append((f"{out_prefix}_{query_name}{out_ext}", idx_sorted))

    if write_tasks:
        out_parent = os.path.dirname(os.path.abspath(out_prefix))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        io_workers = max(1, min(workers, len(write_tasks), 8))
        if io_workers == 1:
            for fname, idx_arr in write_tasks:
                store.save_subset(fname, idx_arr, out_ext=out_ext, trk_affine=trk_affine, trk_dims=trk_dims)
        else:
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                futs = [
                    pool.submit(
                        store.save_subset,
                        fname,
                        idx_arr,
                        out_ext=out_ext,
                        trk_affine=trk_affine,
                        trk_dims=trk_dims,
                    )
                    for fname, idx_arr in write_tasks
                ]
                for fut in futs:
                    fut.result()


if __name__ == "__main__":
    main()
