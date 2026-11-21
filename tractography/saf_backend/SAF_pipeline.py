import os
import time
import importlib
import numpy as np
from scipy.io import loadmat, savemat

from Class_files.init_class import init_class
from Class_files.storage_class import storage_class
from Class_files.modality_class import modality_class
from Class_files.tx_class import tx_class
from Support.verb_msg import verb_msg
from Support.process_FS_surfaces import process_FS_surfaces
from Pipe_modules.SAF_native_space import SAF_native_space
from Pipe_modules.prepare_seeds import prepare_seeds
from Pipe_modules.tractography_mrtrix import tractography_mrtrix
from Pipe_modules.seed_curv import seed_curv
from Filters.filters_GG_HH import filters_GG_HH
from Filters.filters_PG import filters_PG
from Pipe_modules.wholebrain_operations import wholebrain_operations
from Filters.filters_GWG import filters_GWG
from Filters.filters_saf import filters_saf
from Pipe_modules.streamlines_on_surface import streamlines_on_surface


def SAF_pipeline(*args):
    pstart = time.time()
    parsed_input = list(args)
    PS = init_class(parsed_input)
    if PS.dgn:
        save_progress(None, PS)
    update_log(PS)

    old_SD = os.path.exists(os.path.join(PS.work_dir, 'SD.mat'))
    if (not old_SD) or PS.force:
        SD = storage_class(PS)
    else:
        SD = load_progress(PS)

    SD = SAF_native_space(PS, SD)

    if (not old_SD) or PS.force:
        verb_msg('Processing surfaces', PS)
        VR, VL, FR, FL, LR, LL = process_FS_surfaces(SD.T1.rh.white_file, SD.T1.lh.white_file, PS)
        SD.T1.rh.white = VR
        SD.T1.lh.white = VL
        SD.T1.rh.faces = FR
        SD.T1.lh.faces = FL
        SD.T1.rh.usedv = LR
        SD.T1.lh.usedv = LL

        for side in ['rh', 'lh']:
            getattr(SD.DWI, side).usedv = getattr(SD.T1, side).usedv
            getattr(SD.DWI, side).faces = getattr(SD.T1, side).faces
            getattr(SD.T1, side).usedv = None
            getattr(SD.T1, side).faces = None

    SD = SD.tx_coords('white', PS)
    SD.DWI = SD.DWI.combine_verts('white')
    SD.DWI = SD.DWI.calc_vandf()
    SD = prepare_seeds(PS, SD)

    PS, SD = tractography_mrtrix(PS, SD)

    PS, SD = seed_curv(PS, SD)

    PR, PL, FR, FL, _, _ = process_FS_surfaces(SD.T1.rh.pial_file, SD.T1.lh.pial_file, PS)
    SD.T1.rh.pial = PR
    SD.T1.lh.pial = PL
    SD = SD.tx_coords('pial', PS)
    SD.DWI = SD.DWI.mid_surf()
    SD.DWI = SD.DWI.combine_verts('mid')

    PS, SD = filters_GG_HH(PS, SD)
    update_log(PS)

    PS, SD.DWI = filters_PG(PS, SD.DWI)
    if PS.dgn and ((not old_SD) or PS.force):
        save_progress(SD, PS)
    update_log(PS)

    PS, SD = wholebrain_operations(PS, SD)
    update_log(PS)

    PS, SD.DWI = filters_GWG(PS, SD.DWI)
    if PS.dgn:
        save_progress(SD, PS)
    update_log(PS)

    PS, SD.DWI = filters_saf(PS, SD.DWI)
    PS.logt.append(['Final SAF tractogram:', SD.DWI.saf.saf_file])
    verb_msg(f"{PS.logt[-1][0]} {PS.logt[-1][1]}", PS)

    PS, SD.DWI = streamlines_on_surface(PS, SD.DWI)

    if PS.optmodules:
        for mod in PS.optmodules:
            o = time.time()
            module = importlib.import_module(mod)
            fn = getattr(module, mod.rsplit('.', 1)[-1], None) or getattr(module, 'main', None)
            if not callable(fn):
                raise RuntimeError(f'Optional module "{mod}" must expose a callable named "{mod.rsplit(".", 1)[-1]}" or "main"')
            PS = fn(PS, SD)
            ot = time.time() - o
            PS.logt.append([f'Module {mod} time:', _format_elapsed(ot)])
            print(f"{PS.logt[-1][0]} {PS.logt[-1][1]}")

    t = time.time() - pstart
    PS.logt.append(['Total time taken:', _format_elapsed(t)])
    print(f"{PS.logt[-1][0]} {PS.logt[-1][1]}")
    update_log(PS)

    SD.DWI.sl_tracks = []
    save_progress(SD, PS, final=True)

    if not PS.dgn:
        try:
            os.remove(os.path.join(PS.work_dir, 'surf_tracks.tck'))
        except FileNotFoundError:
            pass
        if os.path.exists(os.path.join(PS.work_dir, 'GG_tracks.tck')):
            os.remove(os.path.join(PS.work_dir, 'GG_tracks.tck'))

    return SD, PS


def save_progress(SD, PS, final=False):
    savemat(os.path.join(PS.work_dir, 'PS.mat'), {'PS': _obj_to_dict(PS)})
    if SD is not None:
        if final:
            if SD.DWI.saf.t_gghh is not None:
                SD.DWI.saf.t_gghh = SD.DWI.saf.t_gghh.astype(np.uint32)
            if SD.DWI.saf.t_gwg is not None:
                SD.DWI.saf.t_gwg = SD.DWI.saf.t_gwg.astype(np.uint32)
            if SD.DWI.saf.m is not None:
                SD.DWI.saf.m = SD.DWI.saf.m.astype(np.uint32)
            if SD.DWI.saf.pre_c is not None:
                SD.DWI.saf.pre_c = SD.DWI.saf.pre_c.astype(np.uint32)
            if SD.DWI.saf.pre_h is not None:
                SD.DWI.saf.pre_h = SD.DWI.saf.pre_h.astype(np.uint32)
            if SD.DWI.saf.pre_xp is not None:
                SD.DWI.saf.pre_xp = SD.DWI.saf.pre_xp.astype(np.uint32)
            if SD.DWI.cvi is not None:
                SD.DWI.cvi = SD.DWI.cvi.astype(np.uint32)
            if SD.DWI.cv is not None:
                SD.DWI.cv = SD.DWI.cv.astype(np.float32)
            if SD.DWI.cvt is not None:
                SD.DWI.cvt = SD.DWI.cvt.astype(np.float32)
            if SD.DWI.cvk is not None:
                SD.DWI.cvk = SD.DWI.cvk.astype(np.uint32)
            if SD.DWI.saf.l is not None:
                SD.DWI.saf.l = SD.DWI.saf.l.astype(np.float32)
            for hemi in ['rh', 'lh']:
                obj = getattr(SD.DWI, hemi)
                if obj.faces is not None:
                    obj.faces = obj.faces.astype(np.uint32)
                if obj.white is not None:
                    obj.white = obj.white.astype(np.float32)
                if obj.mid is not None:
                    obj.mid = obj.mid.astype(np.float32)
                if obj.pial is not None:
                    obj.pial = obj.pial.astype(np.float32)
                if obj.ht is not None:
                    obj.ht = obj.ht.astype(np.float32)
        savemat(os.path.join(PS.work_dir, 'SD.mat'), {'SD': _obj_to_dict(SD)})


def load_progress(PS):
    path = os.path.join(PS.work_dir, 'SD.mat')
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(f'Cannot load MATLAB v7.3 progress file with scipy: {path}') from exc
    if 'SD' not in data:
        raise RuntimeError(f'Progress file does not contain SD: {path}')

    # Recreate class instances so methods such as tx_coords remain available,
    # then copy saved fields from scipy mat_struct objects onto them.
    SD = object.__new__(storage_class)
    SD.T1 = modality_class(PS)
    SD.DWI = modality_class(PS)
    SD.tx = object.__new__(tx_class)
    _copy_saved_attrs(SD, data['SD'])
    _restore_numeric_types(SD)
    return SD


def _copy_saved_attrs(target, source):
    names = getattr(source, '_fieldnames', None)
    if not names:
        return _mat_value(source)
    for name in names:
        value = getattr(source, name)
        if hasattr(target, name) and getattr(getattr(target, name), '__dict__', None) is not None and getattr(value, '_fieldnames', None):
            _copy_saved_attrs(getattr(target, name), value)
        else:
            setattr(target, name, _mat_value(value))
    return target


def _mat_value(value):
    if getattr(value, '_fieldnames', None):
        return {name: _mat_value(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_mat_value(v) for v in value.flat]
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    return value


def _restore_numeric_types(SD):
    for obj, names in [
        (SD.DWI, ['cv', 'cvt']),
        (SD.DWI.saf, ['l']),
    ]:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                setattr(obj, name, np.asarray(value, dtype=float))

    for name in ['cvi', 'cvk']:
        value = getattr(SD.DWI, name, None)
        if value is not None:
            setattr(SD.DWI, name, np.asarray(value, dtype=int))

    for hemi in ['rh', 'lh']:
        obj = getattr(SD.DWI, hemi)
        for name in ['white', 'mid', 'pial', 'ht']:
            value = getattr(obj, name, None)
            if value is not None:
                setattr(obj, name, np.asarray(value, dtype=float))
        if obj.faces is not None:
            obj.faces = np.asarray(obj.faces, dtype=int)


def update_log(PS):
    logf = open(os.path.join(PS.work_dir, 'log.txt'), 'w', encoding='utf-8')
    for row in PS.logt:
        logf.write(f"{row[0]} {row[1]}\n")
    logf.close()


def _obj_to_dict(obj):
    if obj is None:
        return np.array([])
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, '__dict__'):
        return {k: _obj_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_obj_to_dict(v) for v in obj]
    return obj


def _format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs_dir', required=True)
    parser.add_argument('--fod', required=True)
    parser.add_argument('--fa', default='')
    args = parser.parse_args()
    opts = ['fs_dir', args.fs_dir, 'fod', args.fod]
    if args.fa:
        opts += ['fa', args.fa]
    SAF_pipeline(*opts)
