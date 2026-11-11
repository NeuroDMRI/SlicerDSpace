import os
import numpy as np
import time

from Filters.snap2mesh.coordinate_clustering import coordinate_clustering
from Filters.snap2mesh.find_closest_vertex import find_closest_vertex
from Support.verb_msg import verb_msg
from Support.pts_to_tck import pts_to_tck


def _pair_streamline_endpoint_values(values: np.ndarray, nsl: int) -> np.ndarray:
    """Pair SAF endpoint vectors as ``[starts; ends]``.

    SAF stores all streamline starts first and all streamline ends second.
    MATLAB's ``reshape(..., [], 2)'`` respects that column-major layout.
    NumPy's default row-major reshape does not, so we pair explicitly.
    """

    return np.vstack([values[:nsl], values[nsl:]])


def filters_GG_HH(PS, SD):
    verb_msg(f"Preparing for streamline-cortex filters. Current time: {time.strftime('%H:%M')}", PS)

    lvec = np.array([sl.shape[0] > 2 for sl in SD.DWI.sl_tracks], dtype=bool)
    nsl = int(np.sum(lvec))

    if nsl == 0:
        SD.DWI.saf.t_gghh = np.zeros((2, len(lvec)), dtype=int)
        SD.DWI.saf.m = np.zeros((2, len(lvec)), dtype=int)
        SD.DWI.saf.h = np.zeros((2, len(lvec)), dtype=bool)
        return PS, SD

    if not hasattr(SD.DWI, 'cvk') or SD.DWI.cvk is None or len(np.atleast_1d(SD.DWI.cvk)) == 0 or PS.force:
        SD.DWI.cvk, elapsed = coordinate_clustering(SD.DWI.cv, nsl)
        PS.logt.append(elapsed)

    PS.logt.append(['', ''])
    PS.logt.append(['Extraction of SAF:', ''])
    verb_msg(f"Starting streamline-cortex filters. Current time: {time.strftime('%H:%M')}", PS)
    PS.logt.append(['Number of streamlines before filtering:', nsl])
    verb_msg(f"{PS.logt[-1][0]} {PS.logt[-1][1]}", PS)
    verb_msg('Starting GG & HH filtering', PS)
    filtert = time.time()

    pts = []
    for sl in [s for s, keep in zip(SD.DWI.sl_tracks, lvec) if keep]:
        pts.append(sl[[0, -1], :])
    if pts:
        pts = np.vstack(pts)
        odd = np.array([(i % 2) == 0 for i in range(nsl * 2)])
        pts = np.vstack([pts[odd], pts[~odd]])
    else:
        pts = np.zeros((0, 3))

    gg_idx, dd_idx = find_closest_vertex(SD.DWI.cv, pts, SD.DWI.cvk)

    ggdd = SD.DWI.cvt[gg_idx - 1] - dd_idx
    gg_idx[(ggdd + PS.gg_margin) < 0] = 0

    hh_idx = np.zeros_like(gg_idx, dtype=bool)
    valid = gg_idx > 0
    hh_idx[valid] = SD.DWI.cvh[gg_idx[valid] - 1]
    gg_idx[valid] = SD.DWI.cvi[gg_idx[valid] - 1]

    if PS.gg_margin > 0:
        opp_ends = np.concatenate([np.arange(nsl, nsl * 2), np.arange(0, nsl)])
        mm_idx = gg_idx > 0
        mm_idx = mm_idx & (ggdd < 0)

        valid_idx = (gg_idx > 0) & (gg_idx[opp_ends] > 0) & (hh_idx == hh_idx[opp_ends])
        keep_mgn = mm_idx & valid_idx

        margin_in = np.zeros_like(mm_idx, dtype=bool)
        hemi = ['rh', 'lh']
        for side in range(2):
            side_vec = keep_mgn & (hh_idx == (side == 1))
            if not np.any(side_vec):
                continue
            from_wsm_vec = pts[side_vec] - SD.DWI.__dict__[hemi[side]].white[gg_idx[side_vec] - 1]
            N = get_normals(SD.DWI.__dict__[hemi[side]].white, SD.DWI.__dict__[hemi[side]].faces)
            surf_norm_vec = N[gg_idx[side_vec] - 1]
            inside_pts = np.einsum('ij,ij->i', from_wsm_vec, surf_norm_vec) < 0
            margin_in[side_vec] = inside_pts

        keep_mgn = mm_idx & valid_idx & margin_in
        mm_idx = np.zeros_like(mm_idx, dtype=int)

        for side in range(2):
            side_vec = keep_mgn & (hh_idx == (side == 1))
            if not np.any(side_vec):
                continue
            wsm_idx = establish_closest_wsm(
                pts[side_vec],
                SD.DWI.__dict__[hemi[side]].ht[gg_idx[side_vec] - 1],
                SD.DWI.__dict__[hemi[side]].white,
                SD.DWI.__dict__[hemi[side]].usedv,
                PS.gg_margin,
            )
            side_vec_idx = np.where(side_vec)[0]
            ok = wsm_idx > 0
            mm_idx[side_vec_idx[ok]] = wsm_idx[ok]

        gg_idx[(mm_idx == 0) & (ggdd < 0)] = 0
    else:
        mm_idx = np.zeros_like(gg_idx, dtype=int)

    if PS.dgn:
        pts_to_tck(pts[gg_idx == 0], os.path.join(PS.work_dir, 'rej.tck'))
        pts_to_tck(pts[ggdd >= 0], os.path.join(PS.work_dir, 'mcc.tck'))
        pts_to_tck(pts[mm_idx > 0], os.path.join(PS.work_dir, 'mgn.tck'))

    # MATLAB stores the first n entries as streamline starts and the next n as
    # streamline ends. NumPy's default row-major reshape would interleave them,
    # corrupting start/end pairing for GG/HH bookkeeping.
    # Keep original track indices, including rejected one-/two-point tracks.
    # Later hemisphere, length and weight masks address the original list.
    for name, values in [('t_gghh', gg_idx), ('m', mm_idx), ('h', hh_idx)]:
        paired = np.zeros((2, len(lvec)), dtype=values.dtype)
        paired[:, lvec] = _pair_streamline_endpoint_values(values, nsl)
        setattr(SD.DWI.saf, name, paired)

    t = time.time() - filtert
    PS.logt.append(['Time spent on GG & HH filtering:', _format_elapsed(t)])
    PS.logt.append(['Number of streamlines rejected due to extremely short length:', int(np.sum(~lvec))])
    PS.logt.append(['Number of streamlines rejected during GG filtering:', int(np.sum(lvec) - np.sum(np.all(SD.DWI.saf.t_gghh, axis=0)))])
    PS.logt.append(['Number of streamlines rejected during HH filtering:', int(np.sum(np.all(SD.DWI.saf.t_gghh, axis=0) * np.abs(np.diff(SD.DWI.saf.h, axis=0))))])

    for entry in PS.logt[-4:]:
        verb_msg(f"{entry[0]} {entry[1]}", PS)

    return PS, SD


def establish_closest_wsm(pts, ht, V, usedv, mgn):
    BB_min = pts - (ht[:, None] + mgn)
    BB_max = pts + (ht[:, None] + mgn)
    wsm_idx = np.zeros((pts.shape[0],), dtype=int)
    for i in range(pts.shape[0]):
        VBB = np.all((V - BB_min[i]) >= 0, axis=1) & np.all((V - BB_max[i]) <= 0, axis=1) & usedv
        if np.any(VBB):
            vv = V[VBB]
            d = np.sum((vv - pts[i]) ** 2, axis=1)
            vi = np.where(VBB)[0][np.argmin(d)]
            wsm_idx[i] = vi + 1
    return wsm_idx


def get_normals(V, F):
    nv = V.shape[0]
    NF = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    NV = np.zeros((nv, 3), dtype=float)
    for j in range(3):
        np.add.at(NV, F[:, j], NF)
    D = np.linalg.norm(NV, axis=1)
    D[D < np.finfo(float).eps] = 1
    NV = NV / D[:, None]
    return NV


def _format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'
