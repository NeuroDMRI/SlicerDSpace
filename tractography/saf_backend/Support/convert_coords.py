import numpy as np

from .process_coords import process_coords
from .pts_to_tck import pts_to_tck


def convert_coords(in_file, out_file):
    in_mat = process_coords(in_file)
    if out_file.lower().endswith('csv'):
        out = in_mat.copy()
        out[:, 0:2] = -out[:, 0:2]
        out = np.column_stack([out[:, 0], out[:, 1], out[:, 2], np.zeros(out.shape[0])])
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('x,y,z,t\n')
            np.savetxt(f, out, delimiter=',', fmt='%.6f')
    elif out_file.lower().endswith('tck'):
        pts_to_tck(in_mat, out_file)
    else:
        np.savetxt(out_file, in_mat, fmt='%.6f')
    return in_mat
