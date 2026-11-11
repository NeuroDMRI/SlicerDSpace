import os
import numpy as np

from .trintersect_fast import trintersect_fast
from .endpoint_vertices import closest_endpoint_vertices


def _expand_boxes_to_cells(lo, hi, ny, nz):
    """Expand 3D integer box bounds ``(N, 3)`` into flat ``(item_idx, cell_id)`` arrays."""
    counts_xyz = hi - lo + 1
    counts = counts_xyz[:, 0] * counts_xyz[:, 1] * counts_xyz[:, 2]
    total = int(np.sum(counts))
    if total == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    item_idx = np.repeat(np.arange(len(lo), dtype=np.int32), counts)
    starts = np.empty(len(lo), dtype=np.int64)
    starts[0] = 0
    if len(lo) > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    ny_i = counts_xyz[item_idx, 1].astype(np.int64)
    nz_i = counts_xyz[item_idx, 2].astype(np.int64)
    yz_i = ny_i * nz_i
    dx = local // yz_i
    rem = local - dx * yz_i
    dy = rem // nz_i
    dz = rem - dy * nz_i
    cx = lo[item_idx, 0].astype(np.int64) + dx
    cy = lo[item_idx, 1].astype(np.int64) + dy
    cz = lo[item_idx, 2].astype(np.int64) + dz
    cell_id = ((cx * int(ny) + cy) * int(nz) + cz).astype(np.int32)
    return item_idx, cell_id


def _build_face_spatial_grid(FBB_min, FBB_max, max_cells_per_face=64):
    mesh_min = np.min(FBB_min, axis=0)
    mesh_max = np.max(FBB_max, axis=0)
    span = np.maximum(mesh_max - mesh_min, 1e-6)
    med_ext = np.median(FBB_max - FBB_min, axis=0)
    cell_size = np.maximum(med_ext * 2.0, span / 96.0)
    cell_size = np.maximum(cell_size, 1e-4)
    dims = np.clip(np.floor(span / cell_size).astype(np.int32) + 1, 1, 96)
    cell_size = np.maximum(span / np.maximum(dims - 0.01, 1.0), 1e-4)

    f_lo = np.clip(np.floor((FBB_min - mesh_min) / cell_size).astype(np.int32), 0, dims - 1)
    f_hi = np.clip(np.floor((FBB_max - mesh_min) / cell_size).astype(np.int32), 0, dims - 1)
    f_counts = (
        (f_hi[:, 0] - f_lo[:, 0] + 1).astype(np.int64)
        * (f_hi[:, 1] - f_lo[:, 1] + 1).astype(np.int64)
        * (f_hi[:, 2] - f_lo[:, 2] + 1).astype(np.int64)
    )
    small_mask = f_counts <= int(max_cells_per_face)
    small_faces = np.flatnonzero(small_mask).astype(np.int32)
    large_faces = np.flatnonzero(~small_mask).astype(np.int32)

    n_cells = int(dims[0]) * int(dims[1]) * int(dims[2])
    if small_faces.size > 0:
        local_f, cell_ids = _expand_boxes_to_cells(f_lo[small_faces], f_hi[small_faces], dims[1], dims[2])
        face_ids = small_faces[local_f]
        order = np.argsort(cell_ids, kind="mergesort")
        sorted_cells = cell_ids[order]
        cell_faces = face_ids[order]
        counts = np.bincount(sorted_cells, minlength=n_cells)
        cell_offsets = np.empty(n_cells + 1, dtype=np.int64)
        cell_offsets[0] = 0
        np.cumsum(counts, out=cell_offsets[1:])
    else:
        cell_offsets = np.zeros(n_cells + 1, dtype=np.int64)
        cell_faces = np.empty(0, dtype=np.int32)

    return {
        "mesh_min": mesh_min,
        "mesh_max": mesh_max,
        "cell_size": cell_size,
        "dims": dims,
        "cell_offsets": cell_offsets,
        "cell_faces": cell_faces,
        "large_faces": large_faces,
    }


def _gather_csr_pairs(query_items, query_cells, cell_offsets, cell_faces):
    starts = cell_offsets[query_cells]
    stops = cell_offsets[query_cells + 1]
    lengths = stops - starts
    nonempty = lengths > 0
    if not np.any(nonempty):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    q_items = query_items[nonempty]
    starts = starts[nonempty]
    lengths = lengths[nonempty]
    total = int(np.sum(lengths))
    out_items = np.repeat(q_items, lengths)
    group_starts = np.empty(len(lengths), dtype=np.int64)
    group_starts[0] = 0
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=group_starts[1:])
    csr_pos = np.repeat(starts, lengths) + (np.arange(total, dtype=np.int64) - np.repeat(group_starts, lengths))
    return out_items, cell_faces[csr_pos]


def crop_streamlines_outside_mesh_fast(V, F, tract, inside=None, outer_intersection=False, device=None):
    if device is None:
        device = os.environ.get("SLICERDSPACE_SAF_DEVICE", "cpu")
    nsl = len(tract)
    rsl = [None] * nsl
    rxf = [None] * nsl
    osi = [None] * nsl
    slp = [None] * nsl

    if inside is None:
        inside = np.zeros((nsl,), dtype=bool)
    elif np.isscalar(inside):
        inside = np.full((nsl,), bool(inside), dtype=bool)
    else:
        inside = np.asarray(inside, dtype=bool).reshape(-1)
        if inside.size != nsl:
            raise ValueError("inside must be a scalar or contain one value per streamline")

    V = np.asarray(V)
    F = np.asarray(F)
    nf = int(F.shape[0])

    if nsl == 0:
        raise AssertionError('No streamlines have survived GWG filtering')

    if nf > 0:
        tri_verts = V[F]  # (nf, 3, 3)
        FBB_max = np.max(tri_verts, axis=1)
        FBB_min = np.min(tri_verts, axis=1)
        grid = _build_face_spatial_grid(FBB_min, FBB_max)
    else:
        FBB_max = np.zeros((0, 3), dtype=float)
        FBB_min = np.zeros((0, 3), dtype=float)
        grid = None

    chunk_sl = 4096
    for sl_start in range(0, nsl, chunk_sl):
        sl_stop = min(nsl, sl_start + chunk_sl)
        batch_tracks = tract[sl_start:sl_stop]
        batch_n = sl_stop - sl_start
        seg_counts = np.array([max(0, sl.shape[0] - 1) for sl in batch_tracks], dtype=np.int64)
        seg_offsets = np.empty(batch_n + 1, dtype=np.int64)
        seg_offsets[0] = 0
        np.cumsum(seg_counts, out=seg_offsets[1:])
        total_segs = int(seg_offsets[-1])

        hit_seg = np.empty(0, dtype=np.int64)
        hit_face = np.empty(0, dtype=np.int64)

        if nf > 0 and total_segs > 0:
            nonempty_tracks = [sl for sl in batch_tracks if sl.shape[0] > 1]
            Q1 = np.vstack([sl[:-1] for sl in nonempty_tracks])
            Q2 = np.vstack([sl[1:] for sl in nonempty_tracks])
            S_min = np.minimum(Q1, Q2)
            S_max = np.maximum(Q1, Q2)

            mesh_min = grid["mesh_min"]
            mesh_max = grid["mesh_max"]
            cell_size = grid["cell_size"]
            dims = grid["dims"]

            in_mesh_box = np.all((S_max >= mesh_min) & (S_min <= mesh_max), axis=1)
            active_segs = np.flatnonzero(in_mesh_box).astype(np.int32)

            pair_seg_parts = []
            pair_face_parts = []

            if active_segs.size > 0:
                s_lo = np.clip(np.floor((S_min[active_segs] - mesh_min) / cell_size).astype(np.int32), 0, dims - 1)
                s_hi = np.clip(np.floor((S_max[active_segs] - mesh_min) / cell_size).astype(np.int32), 0, dims - 1)
                s_cell_counts = (
                    (s_hi[:, 0] - s_lo[:, 0] + 1).astype(np.int64)
                    * (s_hi[:, 1] - s_lo[:, 1] + 1).astype(np.int64)
                    * (s_hi[:, 2] - s_lo[:, 2] + 1).astype(np.int64)
                )
                small_s_mask = s_cell_counts <= 125
                if np.any(small_s_mask):
                    small_segs = active_segs[small_s_mask]
                    q_local, q_cells = _expand_boxes_to_cells(s_lo[small_s_mask], s_hi[small_s_mask], dims[1], dims[2])
                    cand_s, cand_f = _gather_csr_pairs(small_segs[q_local], q_cells, grid["cell_offsets"], grid["cell_faces"])
                    if cand_s.size > 0:
                        ov = np.all((FBB_max[cand_f] >= S_min[cand_s]) & (FBB_min[cand_f] <= S_max[cand_s]), axis=1)
                        if np.any(ov):
                            pair_seg_parts.append(cand_s[ov])
                            pair_face_parts.append(cand_f[ov])

                if grid["large_faces"].size > 0 and np.any(small_s_mask):
                    lf = grid["large_faces"]
                    small_segs = active_segs[small_s_mask]
                    ov_lf = (
                        (FBB_max[lf, None, 0] >= S_min[None, small_segs, 0]) & (FBB_min[lf, None, 0] <= S_max[None, small_segs, 0])
                        & (FBB_max[lf, None, 1] >= S_min[None, small_segs, 1]) & (FBB_min[lf, None, 1] <= S_max[None, small_segs, 1])
                        & (FBB_max[lf, None, 2] >= S_min[None, small_segs, 2]) & (FBB_min[lf, None, 2] <= S_max[None, small_segs, 2])
                    )
                    if np.any(ov_lf):
                        ls_i, lf_i = np.where(ov_lf.T)
                        pair_seg_parts.append(small_segs[ls_i])
                        pair_face_parts.append(lf[lf_i])

                if not np.all(small_s_mask):
                    for big_seg in active_segs[~small_s_mask]:
                        ov_b = np.all((FBB_max >= S_min[big_seg]) & (FBB_min <= S_max[big_seg]), axis=1)
                        if np.any(ov_b):
                            bf = np.flatnonzero(ov_b).astype(np.int32)
                            pair_seg_parts.append(np.full(bf.size, big_seg, dtype=np.int32))
                            pair_face_parts.append(bf)

            if pair_seg_parts:
                all_cand_s = np.concatenate(pair_seg_parts).astype(np.int64)
                all_cand_f = np.concatenate(pair_face_parts).astype(np.int64)
                # Sorting 1D key (seg * nf + face) deduplicates multi-cell hits AND preserves
                # exact column-major MATLAB find(BBFX) ordering: ascending segment, then ascending face.
                unique_keys = np.unique(all_cand_s * nf + all_cand_f)
                u_seg = unique_keys // nf
                u_face = unique_keys % nf
                P = V[F[u_face]]
                I = trintersect_fast(
                    P[:, 0, :],
                    P[:, 1, :],
                    P[:, 2, :],
                    Q1[u_seg],
                    Q2[u_seg],
                    device=device,
                )
                hit_seg = u_seg[I]
                hit_face = u_face[I]

        split_Bounds = np.searchsorted(hit_seg, seg_offsets) if hit_seg.size > 0 else np.zeros(batch_n + 1, dtype=np.int64)

        for local_i in range(batch_n):
            i = sl_start + local_i
            sl = batch_tracks[local_i]
            a, b = int(split_Bounds[local_i]), int(split_Bounds[local_i + 1])
            if b > a:
                xS = (hit_seg[a:b] - seg_offsets[local_i]).astype(int)
                xF = hit_face[a:b].astype(int)
            else:
                xS = np.array([], dtype=int)
                xF = np.array([], dtype=int)

            xP = xS + 1
            if inside[i]:
                xP = np.concatenate([[0], xP])
                xF = np.concatenate([[np.nan], xF])
            if xP.size % 2 == 1:
                xP = np.concatenate([xP, [sl.shape[0] - 1]])
                xF = np.concatenate([xF, [np.nan]])
            if outer_intersection and xF.size > 0:
                if xP.size > 2:
                    xP = np.concatenate([[xP[0]], [xP[-1]]])
                    xF = np.concatenate([[xF[0]], [xF[-1]]])

            xI = xP.reshape(-1, 2)
            xI[xI[:, 1] < sl.shape[0] - 1, 1] -= 1
            xIF = xF.reshape(-1, 2).T

            tsl = [sl[int(xI[j, 0]):int(xI[j, 1]) + 1] for j in range(xI.shape[0])]
            lmask = [t.shape[0] > 1 for t in tsl]
            rsl[i] = [t for t, keep in zip(tsl, lmask) if keep]
            slp[i] = xI[lmask].T if xI.size else np.zeros((2, 0), dtype=int)
            rxf[i] = xIF[:, lmask] if xIF.size else np.zeros((2, 0))
            osi[i] = np.repeat(i, sum(lmask))

    rsl = [t for sub in rsl if sub for t in sub]
    if not rsl:
        raise AssertionError('No streamlines have survived GWG filtering')
    tract_out = rsl
    orig_sl_ind = _concat_1d(osi, dtype=int)
    end_faces = _concat_2d(rxf, dtype=float)
    xpoints = _concat_2d(slp, dtype=int)

    if len(end_faces.shape) == 1:
        end_faces = end_faces.reshape(2, -1)

    xverts = None
    if end_faces.size > 0:
        xverts = closest_endpoint_vertices(V, F, tract_out, end_faces)

    return tract_out, orig_sl_ind, xverts, xpoints


def _concat_1d(values, dtype=float):
    valid = [np.asarray(v) for v in values if v is not None and len(v) > 0]
    if not valid:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(valid).astype(dtype, copy=False)


def _concat_2d(values, dtype=float):
    valid = [np.asarray(v) for v in values if v is not None and np.asarray(v).size > 0]
    if not valid:
        return np.zeros((2, 0), dtype=dtype)
    return np.concatenate(valid, axis=1).astype(dtype, copy=False)
