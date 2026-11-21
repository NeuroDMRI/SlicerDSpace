import numpy as np


def process_coords(ic):
    if isinstance(ic, str):
        with open(ic, 'r', encoding='utf-8') as f:
            first = f.readline().strip()
            if first == 'x,y,z,t':
                data = np.loadtxt(f, delimiter=',')
                if data.ndim == 1:
                    data = data[None, :]
                sc = np.column_stack([-data[:, 0], -data[:, 1], data[:, 2]])
                return sc
            if first.startswith('#'):
                pos = f.tell()
                line = f.readline()
                while line and line.startswith('#'):
                    pos = f.tell()
                    line = f.readline()
                f.seek(pos)
                raw = np.genfromtxt(f, delimiter=',', usecols=(2, 3, 4))
                if raw.ndim == 1:
                    raw = raw[None, :]
                return raw
            f.seek(0)
            sc = np.loadtxt(f)
            if sc.ndim == 1:
                sc = sc[None, :]
            return sc[:, :3]
    if isinstance(ic, np.ndarray):
        return ic
    raise ValueError('Wrong coordinate input')
