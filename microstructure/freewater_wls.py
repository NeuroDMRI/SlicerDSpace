# Derived from dipy/reconst/fwdti.py::wls_iter, DIPY 1.11.0/1.12.1.
# Copyright (c) 2008-2025, dipy developers. All rights reserved.
# Modified for SlicerDSpace, 2026: avoid a dense diagonal weight matrix.
# BSD-3-Clause; see licenses/DIPY.txt. Not an upstream DIPY release.
"""Author WLS with identical arithmetic after a diagonal-free column scaling."""

from dipy.reconst import fwdti
import numpy as np

_protocol_context = None


def wls_iter(design_matrix, sig, S0, *, Diso=3e-3, mdreg=2.7e-3,
             min_signal=1.0e-6, piterations=3):
    """Retain the author's grid refinement, pseudoinverse and tensor outputs.

    Dispatch must first verify the original author source. Unusual dtypes,
    nonfinite/zero weights or invalid protocols retain original behavior.
    """
    options = dict(Diso=Diso, mdreg=mdreg, min_signal=min_signal, piterations=piterations)
    W = design_matrix
    if (type(W) is not np.ndarray or type(sig) is not np.ndarray
            or W.ndim != 2 or sig.ndim != 1 or W.shape != (len(sig), 7)
            or W.dtype != np.float64 or sig.dtype not in (np.float32, np.float64)
            or not np.isfinite(W).all()):
        return fwdti.wls_iter(W, sig, S0, **options)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        weights = sig**2
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        return fwdti.wls_iter(W, sig, S0, **options)
    log_s = np.log(np.maximum(sig, min_signal))
    # W.T @ diag(sig**2): one nonzero term per output entry, not a change
    # to the normal matrix or pseudoinverse. Match dot's C-order/zero output.
    WTS2 = np.multiply(W.T, weights[None, :], order='C')
    WTS2[WTS2 == 0] = 0.
    inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
    invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    params = np.dot(invWTS2W_WTS2, log_s)
    md = (params[0] + params[2] + params[5]) / 3
    if md < mdreg and np.mean(sig) > min_signal and S0 > min_signal:
        protocol = _protocol_context if _protocol_context is not None and _protocol_context.matches(W, Diso) else None
        fwsig = protocol.fwsig if protocol else np.exp(np.dot(W, np.array([Diso, 0, Diso, 0, 0, Diso, 0])))
        df, flow, fhig, ns = 1, 0, 1, 9
        for _ in range(piterations):
            df = df * 0.1
            if protocol:
                fs, FS, SFW = protocol.grid(flow, fhig, df, ns)
                SI = np.broadcast_to(sig[:, None], FS.shape)
            else:
                fs = np.linspace(flow + df, fhig - df, num=ns)
                SFW = np.array([fwsig] * ns)
                FS, SI = np.meshgrid(fs, sig)
            SA = SI - FS * S0 * SFW.T
            SA[SA <= 0] = min_signal
            y = np.log(SA / (1 - FS))
            all_new_params = np.dot(invWTS2W_WTS2, y)
            SIpred = (1 - FS) * np.exp(np.dot(W, all_new_params)) + FS * S0 * SFW.T
            F2 = np.sum(np.square(SI - SIpred), axis=0)
            Mind = np.argmin(F2)
            params = all_new_params[:, Mind]
            f = fs[Mind]
            flow, fhig, ns = f - df, f + df, 19
        evals, evecs = fwdti.decompose_tensor(fwdti.from_lower_triangular(params))
        return np.concatenate((evals, evecs[0], evecs[1], evecs[2], np.array([f])))
    fw_params = np.zeros(13)
    if md > mdreg:
        fw_params[12] = 1.0
    return fw_params


def _resolve_fw_device(device: str) -> str:
    requested = (device or "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported Free Water device '{device}'; expected 'cpu', 'cuda', or 'auto'")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUDA device requested for Free Water WLS, but torch.cuda.is_available() is False")
    return "cuda" if has_cuda else "cpu"


def wls_batch_fit(model, signals, *, device: str = "cpu"):
    """Chunk-vectorized Free-Water WLS fusing protocol reuse, column scaling, and optional CUDA fp64."""
    W = model.design_matrix
    signals = np.asarray(signals)
    if (
        type(W) is not np.ndarray
        or signals.ndim != 2
        or W.ndim != 2
        or W.shape != (signals.shape[1], 7)
        or W.dtype != np.float64
        or signals.dtype not in (np.float32, np.float64)
        or not np.isfinite(W).all()
        or model.args
    ):
        return model.fit(signals), "dipy_fallback"

    allowed_keys = {"Diso", "mdreg", "min_signal", "piterations"}
    if any(k not in allowed_keys for k in model.kwargs):
        return model.fit(signals), "dipy_fallback"

    Diso = model.kwargs.get("Diso", 3e-3)
    mdreg = model.kwargs.get("mdreg", 2.7e-3)
    min_signal = model.kwargs.get("min_signal", 1.0e-6)
    piterations = model.kwargs.get("piterations", 3)

    resolved_device = _resolve_fw_device(device)
    fit_fn = model.fit_method
    fn_globals = getattr(fit_fn, "__globals__", {})
    protocol = fn_globals.get("_protocol_context")
    if protocol is not None and not protocol.matches(W, Diso):
        protocol = None
    fwdti_mod = fn_globals.get("fwdti", fwdti)
    fwsig = (
        protocol.fwsig
        if protocol is not None
        else np.exp(np.dot(W, np.array([Diso, 0.0, Diso, 0.0, 0.0, Diso, 0.0], dtype=np.float64)))
    )

    n_vox = len(signals)
    out = np.zeros((n_vox, 13), dtype=np.float64)
    if n_vox == 0:
        return fwdti.FreeWaterTensorFit(model, out), resolved_device

    b0_mask = model.gtab.b0s_mask
    S0_all = np.mean(signals[:, b0_mask], axis=1)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        weights_all = signals ** 2
    valid_fast = np.isfinite(weights_all).all(axis=1) & np.all(weights_all > 0, axis=1)

    if not np.all(valid_fast):
        for idx in np.flatnonzero(~valid_fast):
            out[idx] = fit_fn(W, signals[idx], S0_all[idx], **model.kwargs)

    fast_indices = np.flatnonzero(valid_fast)
    if len(fast_indices) == 0:
        return fwdti.FreeWaterTensorFit(model, out), resolved_device

    if resolved_device == "cuda":
        import torch
        from shared.hardware_adapt import adaptive_cuda_scaled_batch, run_cuda_chunked_with_oom_fallback

        torch_dev = torch.device("cuda")
        W_t = torch.as_tensor(np.ascontiguousarray(W), dtype=torch.float64, device=torch_dev)
        col_scale = torch.linalg.norm(W_t, dim=0).clamp(min=1e-30)
        W_eq = W_t / col_scale
        fwsig_t = torch.as_tensor(np.array(fwsig, dtype=np.float64, copy=True, order="C"), dtype=torch.float64, device=torch_dev)
        fs0_1d = (
            torch.as_tensor(np.array(protocol.grid(0, 1, 0.1, 9)[0], dtype=np.float64, copy=True, order="C"), dtype=torch.float64, device=torch_dev)
            if protocol is not None
            else None
        )
        per_voxel_bytes = max(4096, int(W.shape[0]) * 19 * 64)
        batch_size = adaptive_cuda_scaled_batch(per_voxel_bytes, 8192, min_batch=128, max_batch=16384, device=torch_dev)

        def _cuda_fw_step(start: int, stop: int) -> None:
            ids = fast_indices[start:stop]
            sig_np = signals[ids]
            w_np = weights_all[ids]
            s0_np = S0_all[ids]
            mean_sig_np = np.mean(sig_np, axis=1)

            sig_t = torch.as_tensor(np.ascontiguousarray(sig_np), dtype=torch.float64, device=torch_dev)
            w_t = torch.as_tensor(np.ascontiguousarray(w_np), dtype=torch.float64, device=torch_dev)
            s0_t = torch.as_tensor(np.ascontiguousarray(s0_np), dtype=torch.float64, device=torch_dev)
            log_s_t = torch.log(torch.clamp(sig_t, min=float(min_signal)))

            w_max = w_t.amax(dim=1, keepdim=True).clamp(min=1e-300)
            w_norm = w_t / w_max
            WTS2_eq = W_eq.T.unsqueeze(0) * w_norm.unsqueeze(1)
            G_eq = torch.matmul(WTS2_eq, W_eq)
            L_eq, info_eq = torch.linalg.cholesky_ex(G_eq)
            diag_L = torch.diagonal(L_eq, dim1=-2, dim2=-1)
            well_cond = (info_eq == 0) & (diag_L.amin(dim=-1) > 1e-6 * diag_L.amax(dim=-1))

            P_t = torch.empty((len(ids), 7, W.shape[0]), dtype=torch.float64, device=torch_dev)
            if torch.any(well_cond):
                P_eq = torch.cholesky_solve(WTS2_eq[well_cond], L_eq[well_cond])
                P_t[well_cond] = P_eq / col_scale.view(1, 7, 1)
            if not torch.all(well_cond):
                ill = ~well_cond
                WTS2_raw = W_t.T.unsqueeze(0) * w_t[ill].unsqueeze(1)
                G_raw = torch.matmul(WTS2_raw, W_t)
                inv_G_raw = torch.linalg.pinv(G_raw)
                P_t[ill] = torch.matmul(inv_G_raw, WTS2_raw)

            params_t = torch.matmul(P_t, log_s_t.unsqueeze(-1)).squeeze(-1)
            params_cpu = params_t.cpu().numpy()
            md_cpu = (params_cpu[:, 0] + params_cpu[:, 2] + params_cpu[:, 5]) / 3.0
            active_mask = (md_cpu < mdreg) & (mean_sig_np > min_signal) & (s0_np > min_signal)

            sub_out = np.zeros((len(ids), 13), dtype=np.float64)
            sub_out[(~active_mask) & (md_cpu > mdreg), 12] = 1.0

            if np.any(active_mask):
                act_idx_np = np.flatnonzero(active_mask)
                act_idx_t = torch.as_tensor(act_idx_np, dtype=torch.long, device=torch_dev)
                B_a = len(act_idx_np)
                sig_a = sig_t[act_idx_t]
                s0_a = s0_t[act_idx_t]
                P_a = P_t[act_idx_t]
                params_a = params_t[act_idx_t]

                df = 1.0
                flow = torch.zeros(B_a, dtype=torch.float64, device=torch_dev)
                fhig = torch.ones(B_a, dtype=torch.float64, device=torch_dev)
                ns = 9
                f_a = torch.zeros(B_a, dtype=torch.float64, device=torch_dev)
                b_arange = torch.arange(B_a, device=torch_dev)

                for it in range(piterations):
                    df = df * 0.1
                    if it == 0 and fs0_1d is not None and ns == 9:
                        fs = fs0_1d.unsqueeze(0).expand(B_a, ns)
                    else:
                        steps = torch.linspace(0.0, 1.0, ns, dtype=torch.float64, device=torch_dev)
                        lo_b = (flow + df).unsqueeze(1)
                        hi_b = (fhig - df).unsqueeze(1)
                        fs = lo_b + steps.unsqueeze(0) * (hi_b - lo_b)
                    fw_term = (s0_a.unsqueeze(1) * fs).unsqueeze(1) * fwsig_t.view(1, -1, 1)
                    SA = sig_a.unsqueeze(2) - fw_term
                    SA = torch.where(SA <= 0.0, torch.tensor(float(min_signal), dtype=torch.float64, device=torch_dev), SA)
                    y = torch.log(SA / (1.0 - fs.unsqueeze(1)))
                    all_new_params = torch.matmul(P_a, y)
                    SIpred = (1.0 - fs.unsqueeze(1)) * torch.exp(torch.matmul(W_t.unsqueeze(0), all_new_params)) + fw_term
                    F2 = torch.sum(torch.square(sig_a.unsqueeze(2) - SIpred), dim=1)
                    Mind = torch.argmin(F2, dim=1)
                    params_a = all_new_params[b_arange, :, Mind]
                    f_a = fs[b_arange, Mind]
                    flow = f_a - df
                    fhig = f_a + df
                    ns = 19

                params_a_np = params_a.cpu().numpy()
                f_a_np = f_a.cpu().numpy()
                evals, evecs = fwdti_mod.decompose_tensor(fwdti_mod.from_lower_triangular(params_a_np))
                sub_out[act_idx_np] = np.concatenate(
                    (evals, evecs[:, 0], evecs[:, 1], evecs[:, 2], f_a_np[:, None]), axis=-1
                )

            out[ids] = sub_out

        def _cpu_fw_fallback(start: int, stop: int) -> None:
            nonlocal resolved_device
            ids = fast_indices[start:stop]
            cpu_fit, _ = wls_batch_fit(model, signals[ids], device="cpu")
            out[ids] = np.asarray(cpu_fit.model_params)
            resolved_device = "cpu"

        run_cuda_chunked_with_oom_fallback(
            len(fast_indices),
            batch_size,
            _cuda_fw_step,
            min_chunk=128,
            cpu_fallback_fn=_cpu_fw_fallback,
        )
        return fwdti.FreeWaterTensorFit(model, out), resolved_device

    # CPU batched path
    WT = np.ascontiguousarray(W.T)
    W_c = np.ascontiguousarray(W)
    fs0_cpu = protocol.grid(0, 1, 0.1, 9)[0] if protocol is not None else None
    batch_size = 512
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        for start in range(0, len(fast_indices), batch_size):
            ids = fast_indices[start : start + batch_size]
            sig = signals[ids]
            w = weights_all[ids]
            s0 = S0_all[ids]
            mean_sig = np.mean(sig, axis=1)

            log_s = np.log(np.maximum(sig, min_signal))
            WTS2 = WT[None, :, :] * w[:, None, :]
            WTS2[WTS2 == 0] = 0.0
            G = np.matmul(WTS2, W_c)
            inv_WT_S2_W = np.linalg.pinv(G)
            P = np.matmul(inv_WT_S2_W, WTS2)
            params = np.matmul(P, log_s[:, :, None])[:, :, 0]
            md = (params[:, 0] + params[:, 2] + params[:, 5]) / 3.0
            active_mask = (md < mdreg) & (mean_sig > min_signal) & (s0 > min_signal)

            sub_out = np.zeros((len(ids), 13), dtype=np.float64)
            sub_out[(~active_mask) & (md > mdreg), 12] = 1.0

            if np.any(active_mask):
                act_idx = np.flatnonzero(active_mask)
                B_a = len(act_idx)
                sig_a = np.asarray(sig[act_idx], dtype=np.float64)
                s0_a = np.asarray(s0[act_idx], dtype=np.float64)
                P_a = P[act_idx]
                params_a = params[act_idx]

                df = 1.0
                flow = np.zeros(B_a, dtype=np.float64)
                fhig = np.ones(B_a, dtype=np.float64)
                ns = 9
                f_a = np.zeros(B_a, dtype=np.float64)
                b_arange = np.arange(B_a)

                for it in range(piterations):
                    df = df * 0.1
                    if it == 0 and fs0_cpu is not None and ns == 9:
                        fs = np.broadcast_to(fs0_cpu[None, :], (B_a, ns))
                    else:
                        fs = np.linspace(flow + df, fhig - df, num=ns, axis=1)
                    fw_term = (s0_a[:, None] * fs)[:, None, :] * fwsig[None, :, None]
                    SA = sig_a[:, :, None] - fw_term
                    SA[SA <= 0] = min_signal
                    y = np.log(SA / (1.0 - fs[:, None, :]))
                    all_new_params = np.matmul(P_a, y)
                    SIpred = (1.0 - fs[:, None, :]) * np.exp(np.matmul(W_c[None, :, :], all_new_params)) + fw_term
                    F2 = np.sum(np.square(sig_a[:, :, None] - SIpred), axis=1)
                    Mind = np.argmin(F2, axis=1)
                    params_a = all_new_params[b_arange, :, Mind]
                    f_a = fs[b_arange, Mind]
                    flow = f_a - df
                    fhig = f_a + df
                    ns = 19

                evals, evecs = fwdti_mod.decompose_tensor(fwdti_mod.from_lower_triangular(params_a))
                sub_out[act_idx] = np.concatenate(
                    (evals, evecs[:, 0], evecs[:, 1], evecs[:, 2], f_a[:, None]), axis=-1
                )

            out[ids] = sub_out

    return fwdti.FreeWaterTensorFit(model, out), resolved_device

