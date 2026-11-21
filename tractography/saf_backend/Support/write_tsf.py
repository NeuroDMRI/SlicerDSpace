import numpy as np


def write_tsf(scalar_data, fname):
    scalar_data = np.asarray(scalar_data).reshape(-1)
    with open(fname, 'w', encoding='utf-8') as f:
        np.savetxt(f, scalar_data[None, :], fmt='%.6f')
