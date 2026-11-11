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


def _trintersect_cuda(P1, P2, P3, Q1, Q2):
    import torch
    from shared.hardware_adapt import adaptive_cuda_scaled_batch, run_cuda_chunked_with_oom_fallback

    n = len(P1)
    if n == 0:
        return np.zeros(0, dtype=bool)
    dev = torch.device("cuda")
    out = np.zeros(n, dtype=bool)
    # ~5 tensors x 3 x 8 bytes + intermediates ~ 512 bytes/pair
    batch = adaptive_cuda_scaled_batch(512, 262144, min_batch=4096, max_batch=524288, device=dev)

    def _signed_vol_t(a, b, d, c):
        return (1.0 / 6.0) * torch.sum(torch.linalg.cross(b - a, c - a, dim=1) * (d - a), dim=1)

    def _step(start: int, stop: int) -> None:
        p1 = torch.as_tensor(np.ascontiguousarray(P1[start:stop], dtype=np.float64), device=dev)
        p2 = torch.as_tensor(np.ascontiguousarray(P2[start:stop], dtype=np.float64), device=dev)
        p3 = torch.as_tensor(np.ascontiguousarray(P3[start:stop], dtype=np.float64), device=dev)
        q1 = torch.as_tensor(np.ascontiguousarray(Q1[start:stop], dtype=np.float64), device=dev)
        q2 = torch.as_tensor(np.ascontiguousarray(Q2[start:stop], dtype=np.float64), device=dev)

        s1 = torch.sign(_signed_vol_t(q1, p1, p2, p3))
        s2 = torch.sign(_signed_vol_t(q2, p1, p2, p3))
        hit = s1 != s2
        if torch.any(hit):
            idx = torch.nonzero(hit, as_tuple=False).squeeze(1)
            q1_i, q2_i = q1[idx], q2[idx]
            p1_i, p2_i, p3_i = p1[idx], p2[idx], p3[idx]
            rep = torch.sign(_signed_vol_t(q1_i, q2_i, p2_i, p3_i))
            keep = rep == torch.sign(_signed_vol_t(q1_i, q2_i, p1_i, p2_i))
            if torch.any(keep):
                sub = torch.nonzero(keep, as_tuple=False).squeeze(1)
                keep[sub] = rep[sub] == torch.sign(
                    _signed_vol_t(q1_i[sub], q2_i[sub], p3_i[sub], p1_i[sub])
                )
            hit_out = torch.zeros(stop - start, dtype=torch.bool, device=dev)
            hit_out[idx[keep]] = True
            out[start:stop] = hit_out.cpu().numpy()

    def _fallback(start: int, stop: int) -> None:
        out[start:stop] = _trintersect_cpu(
            P1[start:stop], P2[start:stop], P3[start:stop], Q1[start:stop], Q2[start:stop]
        )

    run_cuda_chunked_with_oom_fallback(n, batch, _step, min_chunk=4096, cpu_fallback_fn=_fallback)
    return out


def trintersect_fast(P1, P2, P3, Q1, Q2, *, device: str = "cpu"):
    if str(device).lower().startswith("cuda") and len(P1) >= 2048:
        try:
            import torch

            if torch.cuda.is_available():
                return _trintersect_cuda(P1, P2, P3, Q1, Q2)
        except Exception:
            pass
    return _trintersect_cpu(P1, P2, P3, Q1, Q2)
