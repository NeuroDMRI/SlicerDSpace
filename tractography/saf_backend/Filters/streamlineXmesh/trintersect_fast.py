import numpy as np


def _trintersect_cpu(P1, P2, P3, Q1, Q2):
    def signed_volume(a, b, d, c):
        return (1.0 / 6.0) * np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)

    I = np.sign(signed_volume(Q1, P1, P2, P3)) != np.sign(signed_volume(Q2, P1, P2, P3))
    if not np.any(I):
        return I
    idx = np.flatnonzero(I)
    q1_i, q2_i = Q1[idx], Q2[idx]
    p1_i, p2_i, p3_i = P1[idx], P2[idx], P3[idx]
    repside = np.sign(signed_volume(q1_i, q2_i, p2_i, p3_i))
    keep = repside == np.sign(signed_volume(q1_i, q2_i, p1_i, p2_i))
    if np.any(keep):
        sub = np.flatnonzero(keep)
        keep[sub] = repside[sub] == np.sign(
            signed_volume(q1_i[sub], q2_i[sub], p3_i[sub], p1_i[sub])
        )
    out = np.zeros(len(P1), dtype=bool)
    out[idx[keep]] = True
    return out


def trintersect_fast(P1, P2, P3, Q1, Q2, *, device: str = "cpu"):
    del device
    return _trintersect_cpu(P1, P2, P3, Q1, Q2)
