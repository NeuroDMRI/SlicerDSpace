import os
from Support.sys_exec import sys_exec
from Support.read_tsf import read_tsf
from Support.verb_msg import verb_msg


def filters_length(PS):
    sl_lenfilt = None
    if PS.maxlen is not None and PS.maxlen != '':
        verb_msg('Filtering lengths', PS)
        lengths_file = os.path.join(PS.work_dir, 'lengths.txt')
        sys_exec(["tckstats", "-dump", lengths_file, PS.final_init, "-force"], PS)
        lengths = read_tsf(lengths_file)
        if isinstance(PS.maxlen, str):
            PS.maxlen = float(PS.maxlen)
        sl_lenfilt = lengths > PS.maxlen
        PS.logt.append(['Number of streamlines rejected due to maxlen:', int(sl_lenfilt.sum())])
        verb_msg(f"{PS.logt[-1][0]} {PS.logt[-1][1]}", PS)
    return PS, sl_lenfilt
