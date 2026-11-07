import os

from Support.sys_exec import sys_exec
from Support.Supp_gz_fileparts import Supp_gz_fileparts


def apply_tck_sample(tracks, scalar, out_dir, PS):
    sc_name = Supp_gz_fileparts(scalar)[1]
    out_name = os.path.join(out_dir, sc_name + '.tsf')
    if not os.path.exists(out_name) or PS.force:
        sys_exec(["tcksample", "-stat_tck", "mean", "-precise", "-force", tracks, scalar, out_name], PS)
    return out_name, sc_name
