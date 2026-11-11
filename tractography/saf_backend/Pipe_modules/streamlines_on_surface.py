import os
import numpy as np

from Support.verb_msg import verb_msg
from Support.write_curv import write_curv
from Support.file_ops import link_or_copy
from Support.Supp_gz_fileparts import Supp_gz_fileparts
from Data_modules.apply_tck_sample import apply_tck_sample
from Support.read_tsf import read_tsf


def streamlines_on_surface(PS, DWI):
    out_dir = os.path.join(PS.work_dir, 'surf_data')
    os.makedirs(out_dir, exist_ok=True)

    for hemi in ('rh', 'lh'):
        link_or_copy(
            os.path.join(PS.fs_dir, 'surf', f'{hemi}.inflated'),
            os.path.join(out_dir, f'{hemi}.inflated'),
        )

    PS.logt.append(['', ''])

    if PS.dgn:
        PS.logt.append(['Results for all streamlines (based on midcortical surface):', ''])
        lt = data_to_curv('t_gghh', 't_gghh', DWI, out_dir)
        PS.logt.append(['Cortical coverage (% of all vertices) - MCC:', lt[0]])
        PS.logt.append(['Termination density - streamlines per vertex (AVG) - MCC:', lt[1]])
        PS.logt.append(['Termination density - streamlines per vertex (SD) - MCC:', lt[2]])
        PS.logt.append(['', ''])

    PS.logt.append(['Results for subcortical streamlines (based on white matter surface):', ''])
    lt = data_to_curv('t', 't', DWI, out_dir)
    PS.logt.append(['Cortical coverage (% of all vertices):', lt[0]])
    PS.logt.append(['Termination density - streamlines per vertex (AVG):', lt[1]])
    PS.logt.append(['Termination density - streamlines per vertex (SD):', lt[2]])

    lt = data_to_curv('l', 't', DWI, out_dir)
    PS.logt.append(['Streamline length per vertex, mm (AVG):', lt[1]])
    PS.logt.append(['Streamline length per vertex, mm (SD):', lt[2]])

    if PS.fa:
        sl_data = apply_tck_sample(DWI.saf.saf_file, PS.fa, out_dir, PS)
        lt = data_to_curv(sl_data[0], 't', DWI, out_dir, 'FA')
        PS.logt.append(['Mean streamline FA per vertex (AVG):', lt[1]])
        PS.logt.append(['Mean streamline FA per vertex (SD):', lt[2]])

    if DWI.sl_weights:
        lt = data_to_curv(os.path.join(PS.work_dir, 'SAF.wts'), 't', DWI, out_dir, 'wts')
        PS.logt.append(['Mean streamline weights per vertex (AVG):', lt[1]])
        PS.logt.append(['Mean streamline weights per vertex (SD):', lt[2]])

    if PS.scalar2surf:
        for sc in PS.scalar2surf:
            sl_data, sc_name = apply_tck_sample(DWI.saf.saf_file, sc, out_dir, PS)
            lt = data_to_curv(sl_data, 't', DWI, out_dir, sc_name)
            PS.logt.append([f'Mean streamline {sc_name} per vertex (AVG):', lt[1]])
            PS.logt.append([f'Mean streamline {sc_name} per vertex (SD):', lt[2]])

    PS.logt.append(['', ''])
    return PS, DWI


def data_to_curv(indata, inverts, DWI, out_dir, fname=None):
    if fname is None:
        if hasattr(DWI.saf, indata):
            fname = indata
        elif isinstance(indata, str):
            fname = Supp_gz_fileparts(indata)[1]
        else:
            raise RuntimeError('wrong data_to_curv input type')

    cv_num = len(DWI.cvh)

    rh_outfile = os.path.join(out_dir, f'rh.{fname}.{inverts}')
    lh_outfile = os.path.join(out_dir, f'lh.{fname}.{inverts}')

    invs = getattr(DWI.saf, inverts).astype(float)
    invs[DWI.saf.h.astype(bool)] *= -1

    nz_mask = invs != 0
    if np.any(nz_mask):
        unique_verts, inv_idx, counts = np.unique(invs[nz_mask], return_inverse=True, return_counts=True)
    else:
        unique_verts = np.array([], dtype=float)
        inv_idx = np.array([], dtype=int)
        counts = np.array([], dtype=int)

    if hasattr(DWI.saf, indata) and indata == inverts:
        unique_verts = np.vstack([unique_verts, counts]) if unique_verts.size else np.zeros((2, 0))
    else:
        if hasattr(DWI.saf, indata):
            indt = getattr(DWI.saf, indata)
        else:
            if isinstance(indata, str):
                indt_raw = read_tsf(indata)
                indt_raw[np.isnan(indt_raw)] = 0
                sl_ind = np.where(invs[0, :] != 0)[0]
                indt = np.zeros_like(invs[0, :])
                indt[sl_ind] = indt_raw
            else:
                indt = indata

        if unique_verts.size:
            indt_arr = np.asarray(indt, dtype=float).reshape(1, -1)
            indt_2d = np.broadcast_to(indt_arr, invs.shape)
            sums = np.bincount(inv_idx, weights=indt_2d[nz_mask], minlength=unique_verts.size)
            unique_vals = sums / counts
            unique_verts = np.vstack([unique_verts, unique_vals])
        else:
            unique_verts = np.zeros((2, 0))

    rverts_out = unique_verts[:, unique_verts[0, :] > 0] if unique_verts.size else np.zeros((2, 0))
    lverts_out = unique_verts[:, unique_verts[0, :] < 0] if unique_verts.size else np.zeros((2, 0))
    if lverts_out.size:
        lverts_out[0, :] = -lverts_out[0, :]

    rc_list = np.zeros((len(DWI.rh.seeds),), dtype=float)
    if rverts_out.size:
        rc_list[rverts_out[0, :].astype(int) - 1] = rverts_out[1, :]
    write_curv(rh_outfile, rc_list, DWI.rh.nfaces)

    lc_list = np.zeros((len(DWI.lh.seeds),), dtype=float)
    if lverts_out.size:
        lc_list[lverts_out[0, :].astype(int) - 1] = lverts_out[1, :]
    write_curv(lh_outfile, lc_list, DWI.lh.nfaces)

    l_mean = np.sum(unique_verts[1, :]) / cv_num if unique_verts.size else 0
    l_std = np.sum((unique_verts[1, :] - l_mean) ** 2) if unique_verts.size else 0
    l_std = l_std + (0 - l_mean) ** 2 * (cv_num - (unique_verts.shape[1] if unique_verts.size else 0))
    l_std = (l_std / (cv_num - 1)) ** 0.5 if cv_num > 1 else 0

    logt = [0, round(l_mean, 2), round(l_std, 2)]
    if hasattr(DWI.saf, indata) and indata == inverts:
        logt[0] = round((unique_verts.shape[1] if unique_verts.size else 0) / cv_num * 100, 2)

    return logt
