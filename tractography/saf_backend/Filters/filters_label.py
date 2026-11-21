import numpy as np


def filters_label(vert, labs):
    with open(labs, 'r', encoding='utf-8') as f:
        f.readline()
        f.readline()
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data[None, :]
    vert_ind = data[:, 0].astype(int) + 1
    lab_ind = np.zeros((vert.shape[0],), dtype=bool)
    lab_ind[vert_ind - 1] = True
    return lab_ind
