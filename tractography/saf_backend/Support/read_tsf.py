import numpy as np


def read_tsf(tsf):
    with open(tsf, 'r', encoding='utf-8') as f:
        pos = f.tell()
        line = f.readline()
        while line and line.startswith('#'):
            pos = f.tell()
            line = f.readline()
        f.seek(pos)
        data = np.loadtxt(f)
        return data.reshape(-1)
