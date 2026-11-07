import os
import shlex
import time

from Support.verb_msg import verb_msg
from Support.sys_exec import sys_exec
from Support.pts_to_tck import pts_to_tck
from .read_prefilter_tracks import read_prefilter_tracks


def tractography_mrtrix(PS, SD):
    SD.DWI.seed_coords_file_out = os.path.join(PS.work_dir, 'DWI_seeds_out.txt')

    if PS.noseed and os.path.exists(PS.noseed):
        SD.DWI.sl_init = PS.noseed
        PS.final_init = PS.noseed
    elif os.path.exists(SD.DWI.sl_init) and not PS.force:
        PS.final_init = SD.DWI.sl_init
    elif (not os.path.exists(SD.DWI.sl_init)) or PS.force:
        verb_msg('Running mrtrix tckgen', PS)
        tstart = time.time()

        start = ["tckgen"]
        if PS.surfseed:
            start.extend(("-seed_coordinates_global", SD.DWI.seed_coords_file_in))
        start.extend(("-output_seeds", SD.DWI.seed_coords_file_out))
        end = [PS.fod, SD.DWI.sl_init, "-force"]
        if PS.mrtrix_prefix:
            # Explicit legacy wrappers can use shell syntax in their options.
            command = f"{PS.mrtrix_prefix} {shlex.join(start)} {PS.tckgen} {shlex.join(end)}"
            description = command
        else:
            command = [*start, *shlex.split(PS.tckgen), *end]
            description = shlex.join(command)
        verb_msg("tckgen command:", PS)
        verb_msg(description, PS)
        sys_exec(command, PS)

        t = time.time() - tstart
        PS.logt.append(['Time spent on initial tractogram generation:', _format_elapsed(t)])

        if PS.dgn:
            pts_to_tck(SD.DWI.seed_coords_file_out, os.path.join(PS.work_dir, 'DWI_seeds_out.tck'))

        PS.final_init = SD.DWI.sl_init
    else:
        raise RuntimeError('tckgen input wrong')

    PS, SD = read_prefilter_tracks(PS, SD)
    return PS, SD


def _format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'
