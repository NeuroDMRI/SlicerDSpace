"""Launch installed PNLBrainMask-Torch in the selected external Python.

No TensorFlow/Keras import, upstream AST repair, or Slicer Python installation.
The external console script owns inference and writes its full device receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _is_mps_available(torch_mod) -> bool:
    mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
    fn = getattr(mps, "is_available", None)
    return bool(callable(fn) and fn() is True)


def select_device(requested: str, torch=None) -> dict:
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("PNL device must be auto, cpu, cuda or mps")
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return {"requested": requested, "resolved": "cpu", "framework": "pytorch",
                "reason": "Explicit CPU selection"}
    if torch is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PNL requires PyTorch in the selected external Python") from exc
    available = bool(torch.cuda.is_available()) if requested != "mps" else False
    mps_available = _is_mps_available(torch) if (requested in ("auto", "mps") and not available) else False
    if requested == "cuda" and not available:
        raise RuntimeError("PNL GPU requested but external PyTorch has no visible CUDA GPU")
    if requested == "mps" and not mps_available:
        raise RuntimeError("PNL MPS requested but external PyTorch has no available Apple Metal (MPS) GPU")
    if available:
        # Never unmask scheduler-restricted devices; force initialization to fail
        # here rather than silently accepting a broken CUDA runtime.
        torch.empty(1, device="cuda:0")
        resolved = "cuda"
        reason = "First caller-visible CUDA GPU"
    elif mps_available:
        torch.empty(1, device="mps")
        resolved = "mps"
        reason = "Apple Silicon Metal (MPS) GPU"
    else:
        resolved = "cpu"
        reason = "No CUDA or MPS GPU visible to external PyTorch"
    return {"requested": requested, "resolved": resolved,
            "framework": "pytorch", "torch_version": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reason": reason}


def _install_pnl_runtime_accelerations() -> None:
    if os.environ.get("SLICERDSPACE_DISABLE_PNL_ACCEL", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        import nibabel as nib
        import numpy as np
        import torch
        import pnl_brain_mask.inference as pnl_inf
        import pnl_brain_mask.model as pnl_mod
        import pnl_brain_mask.pipeline as pnl_pipe
        import pnl_brain_mask.weights as pnl_weights
    except Exception:
        return
    if getattr(pnl_pipe, "_sdds_accelerated", False):
        return

    orig_forward = pnl_mod.CompNet.forward

    def _compile_plan(model_obj, segmentation_only: bool):
        plans = getattr(model_obj, "_sdds_plans", None)
        if plans is None:
            plans = {}
            model_obj._sdds_plans = plans
        key = bool(segmentation_only)
        if key in plans:
            return plans[key]
        targets = {"final_op"} if key else set(model_obj.output_names)
        needed = set()

        def require(name):
            if name not in needed:
                needed.add(name)
                for parent in model_obj.parents[name]:
                    require(parent)

        for name in targets:
            require(name)
        uses = {name: 0 for name in needed}
        for name in needed:
            for parent in model_obj.parents[name]:
                uses[parent] += 1
        steps = []
        for layer in model_obj.spec:
            name, kind = layer["name"], layer["class_name"]
            if name not in needed:
                continue
            parents = tuple(model_obj.parents[name])
            op = model_obj.ops[name] if kind not in ("InputLayer", "Concatenate", "Add", "Dropout") else None
            act = model_obj.activations.get(name) if op is not None else None
            dead = []
            for parent in parents:
                uses[parent] -= 1
                if uses[parent] == 0 and parent not in targets:
                    dead.append(parent)
            steps.append((name, kind, parents, op, act, tuple(dead)))
        out_names = ("final_op",) if key else tuple(model_obj.output_names)
        plans[key] = (tuple(steps), out_names, key)
        return plans[key]

    def _fast_forward(self, x, *, capture=(), segmentation_only=False):
        if capture or self.training or x.ndim != 4 or tuple(x.shape[1:]) != (1, 256, 256):
            return orig_forward(self, x, capture=capture, segmentation_only=segmentation_only)
        steps, out_names, single = _compile_plan(self, segmentation_only)
        values = {}
        for name, kind, parents, op, act, dead in steps:
            if kind == "InputLayer":
                val = x
            elif kind == "Concatenate":
                val = torch.cat([values[p] for p in parents], dim=1)
            elif kind == "Add":
                val = values[parents[0]] + values[parents[1]]
            elif kind == "Dropout":
                val = values[parents[0]]
            else:
                val = op(values[parents[0]])
                if act == "relu":
                    val = torch.relu(val)
                elif act == "sigmoid":
                    val = torch.sigmoid(val)
            values[name] = val
            for parent in dead:
                del values[parent]
        return values[out_names[0]] if single else tuple(values[n] for n in out_names)

    def _save_nii_fast(img, path: Path) -> None:
        out_s = str(path)
        if out_s.endswith(".nii.gz"):
            with gzip.GzipFile(filename=out_s, mode="wb", compresslevel=1) as gz:
                fh = nib.FileHolder(fileobj=gz)
                img.to_file_map({"image": fh, "header": fh})
        else:
            nib.save(img, out_s)

    def _load_verified_states(model_dir: Path) -> dict:
        model_dir = Path(model_dir)
        if hasattr(pnl_weights, "verify_model_dir"):
            meta = pnl_weights.verify_model_dir(model_dir)
        else:
            meta = json.loads((model_dir / "weights.json").read_text(encoding="utf-8"))
            if meta.get("format") != 1 or meta.get("architecture_sha256") != pnl_weights.ARCHITECTURE_SHA256:
                raise ValueError("Unrecognized converted model")

        def _load_one(view_name: str):
            rec = meta["views"][view_name]
            if rec["file"] != f"{view_name}.pt":
                raise ValueError("Unexpected model provenance")
            if hasattr(pnl_weights, "AUTHOR_FILES") and rec.get("source_sha256") != pnl_weights.AUTHOR_FILES[view_name][1]:
                raise ValueError("Unexpected model provenance")
            w_path = model_dir / rec["file"]
            pnl_weights.check(w_path, rec["sha256"])
            return torch.load(w_path, map_location="cpu", weights_only=True)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {v: pool.submit(_load_one, v) for v in pnl_inf.VIEWS}
            return {v: futs[v].result() for v in pnl_inf.VIEWS}

    @torch.inference_mode()
    def _fast_predict(volume, model_dir, *, device="cpu", batch_size=1, all_outputs=False, _preloaded=None):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if volume.shape != (256, 256, 256) or not np.isfinite(volume).all():
            raise ValueError("Expected a finite normalized IIT-grid volume")
        from shared.hardware_adapt import (
            adaptive_gpu_scaled_batch,
            empty_gpu_cache_safely,
            enable_cpu_flush_denormal,
            is_gpu_oom,
            should_enable_cudnn_benchmark,
        )

        active_dev = str(device)
        is_cuda = active_dev.startswith("cuda")
        is_mps = active_dev.startswith("mps")
        is_gpu = is_cuda or is_mps
        allow_cpu_fallback = (os.environ.get("SLICERDSPACE_PNL_REQUESTED_DEVICE", "auto").strip().lower() == "auto")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if is_cuda:
            torch.backends.cudnn.benchmark = should_enable_cudnn_benchmark(active_dev)
        eff_bs = max(int(batch_size), 32 if is_cuda else 16)
        if is_gpu:
            eff_bs = adaptive_gpu_scaled_batch(
                96 * 1024 * 1024, eff_bs, min_batch=1, max_batch=eff_bs, device=active_dev
            )

        if _preloaded is not None:
            states, model = _preloaded
        else:
            states = _load_verified_states(Path(model_dir))
            try:
                model = pnl_mod.CompNet().to(active_dev)
            except RuntimeError as exc:
                if not (is_gpu and allow_cpu_fallback and is_gpu_oom(exc)):
                    raise
                empty_gpu_cache_safely()
                active_dev = "cpu"
                is_cuda = is_mps = is_gpu = False
                eff_bs = min(eff_bs, 16)
                enable_cpu_flush_denormal(torch)
                model = pnl_mod.CompNet().to("cpu")
            model.eval()

        def _fallback_cpu() -> None:
            nonlocal active_dev, is_cuda, is_mps, is_gpu, eff_bs
            empty_gpu_cache_safely()
            active_dev = "cpu"
            is_cuda = is_mps = is_gpu = False
            eff_bs = min(eff_bs, 16)
            enable_cpu_flush_denormal(torch)
            model.to("cpu")

        probabilities = {}
        try:
            for view in pnl_inf.VIEWS:
                print("[PNL Torch] Predicting " + view, flush=True)
                model.load_state_dict(states.pop(view), strict=True)
                slices = pnl_inf.orient(volume, view)
                output = np.empty(volume.shape, dtype=np.float32)
                nonzero_mask = np.any(slices != 0.0, axis=(1, 2))
                zero_idx = np.flatnonzero(~nonzero_mask)
                nz_idx = np.flatnonzero(nonzero_mask)
                if len(zero_idx) > 0:
                    try:
                        z_in = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=active_dev)
                        z_res = model(z_in, segmentation_only=not all_outputs)
                    except RuntimeError as exc:
                        if not (is_gpu and allow_cpu_fallback and is_gpu_oom(exc)):
                            raise
                        _fallback_cpu()
                        z_in = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device="cpu")
                        z_res = model(z_in, segmentation_only=not all_outputs)
                    if all_outputs:
                        z_res = z_res[0]
                    output[zero_idx] = z_res[0, 0].cpu().numpy()
                if len(nz_idx) > 0:
                    nz_slices = np.ascontiguousarray(slices[nz_idx, None])
                    start = 0
                    cur_bs = eff_bs
                    while start < len(nz_idx):
                        batch_np = nz_slices[start:start + cur_bs]
                        try:
                            t_in = torch.from_numpy(batch_np)
                            if is_cuda and sys.platform != "win32":
                                t_in = t_in.pin_memory().to(active_dev, non_blocking=True)
                            else:
                                t_in = t_in.to(active_dev)
                            res = model(t_in, segmentation_only=not all_outputs)
                            if all_outputs:
                                res = res[0]
                            output[nz_idx[start:start + len(batch_np)]] = res[:, 0].cpu().numpy()
                            start += len(batch_np)
                        except RuntimeError as exc:
                            if not (is_gpu and is_gpu_oom(exc)):
                                raise
                            if cur_bs > 1:
                                empty_gpu_cache_safely()
                                cur_bs = max(1, cur_bs // 2)
                                continue
                            if allow_cpu_fallback:
                                _fallback_cpu()
                                cur_bs = min(eff_bs, 16)
                                continue
                            raise
                if not np.isfinite(output).all():
                    raise RuntimeError("CNN produced nonfinite probabilities")
                probabilities[view] = pnl_inf.orient(output, view)
        finally:
            del model
            if is_gpu:
                empty_gpu_cache_safely()
        fused, mask = pnl_inf.aggregate(probabilities)
        return probabilities, fused, mask

    def _fast_finish_native(template_mask, registered, baseline, matrix, output, *, ants_bin=None, threads=1):
        work_parent = Path(output).parent
        template_file = work_parent / "template_mask.nii"
        header = registered.header.copy()
        nib.save(nib.Nifti1Image(template_mask.astype(np.int16), registered.affine, header), template_file)
        inverse = work_parent / "inverse.nii"
        filled = work_parent / "filled.nii"
        pnl_pipe.run_ants(["antsApplyTransforms", "-d", "3", "-i", template_file, "-r", baseline,
                           "-o", inverse, "--transform", f"[{matrix},1]"], ants_bin=ants_bin, threads=threads)
        pnl_pipe.run_ants(["ImageMath", "3", filled, "FillHoles", inverse], ants_bin=ants_bin, threads=threads)
        original = nib.load(baseline)
        values = np.asarray(nib.load(filled).dataobj, dtype=np.float32).astype(np.uint8)
        if not np.isin(values, [0, 1]).all() or not values.any():
            raise RuntimeError("Native output must be binary and nonempty")
        out_hdr = original.header.copy()
        out_hdr.set_data_dtype(np.int16)
        _save_nii_fast(nib.Nifti1Image(values, original.affine, out_hdr), Path(output))

    orig_run = pnl_pipe.run

    def _fast_run(input_path, output_path, model_dir, template, *, bvals=None, threshold=50, percentile=99,
                  device="auto", threads=1, batch_size=1, ants_bin=None, work_dir=None, record=None):
        input_path, output_path = Path(input_path), Path(output_path)
        if output_path.is_symlink() or output_path.resolve() == input_path.resolve():
            return orig_run(input_path, output_path, model_dir, template, bvals=bvals, threshold=threshold,
                            percentile=percentile, device=device, threads=threads, batch_size=batch_size,
                            ants_bin=ants_bin, work_dir=work_dir, record=record)
        if not str(output_path).endswith(".nii.gz") or threads < 1:
            return orig_run(input_path, output_path, model_dir, template, bvals=bvals, threshold=threshold,
                            percentile=percentile, device=device, threads=threads, batch_size=batch_size,
                            ants_bin=ants_bin, work_dir=work_dir, record=record)
        protected = [input_path, Path(template), Path(model_dir) / "weights.json"]
        protected.extend(Path(model_dir) / (view + ".pt") for view in ("axial", "coronal", "sagittal"))
        if bvals:
            protected.append(Path(bvals))
        if any(output_path.resolve() == p.resolve() for p in protected):
            raise ValueError("Output must not overwrite an input or model resource")
        if record and (Path(record).is_symlink() or any(Path(record).resolve() == p.resolve()
                                                        for p in [*protected, output_path])):
            raise ValueError("Record must be separate from input, model resources, and output")
        pnl_weights.check(template, pnl_weights.TEMPLATE_SHA256)
        selection = pnl_inf.select_device(device)
        torch.set_num_threads(min(max(1, int(threads)), 32))
        if not str(selection["resolved"]).startswith(("cuda", "mps")):
            try:
                torch.set_flush_denormal(True)
            except Exception:
                pass
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = nib.load(input_path, keep_file_open=True)
        baseline_data = pnl_pipe.mean_b0(image, bvals, threshold)
        with tempfile.TemporaryDirectory(prefix="pnl-torch-", dir=work_dir or ("/tmp" if os.name == "posix" else None)) as temporary:
            work = Path(temporary)
            private_ants, private_template = pnl_pipe.ants_workspace(work, template, ants_bin)
            baseline = work / "baseline.nii.gz"
            _save_nii_fast(nib.Nifti1Image(baseline_data, image.affine, image.header), baseline)
            prefix = work / "rigid-"
            resolved_dev = selection["resolved"]

            @torch.inference_mode()
            def _preload_during_ants():
                from shared.hardware_adapt import (
                    adaptive_gpu_scaled_batch,
                    empty_gpu_cache_safely,
                    is_gpu_oom,
                    should_enable_cudnn_benchmark,
                )

                states = _load_verified_states(Path(model_dir))
                try:
                    mdl = pnl_mod.CompNet().to(resolved_dev)
                    mdl.eval()
                    if str(resolved_dev).startswith("cuda"):
                        torch.backends.cudnn.allow_tf32 = False
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.benchmark = should_enable_cudnn_benchmark(resolved_dev)
                        warm_bs = adaptive_gpu_scaled_batch(
                            96 * 1024 * 1024, 32, min_batch=1, max_batch=32, device=resolved_dev
                        )
                        dummy = torch.zeros((warm_bs, 1, 256, 256), dtype=torch.float32, device=resolved_dev)
                        _ = mdl(dummy, segmentation_only=True)
                        torch.cuda.synchronize()
                    elif str(resolved_dev).startswith("mps"):
                        warm_bs = adaptive_gpu_scaled_batch(
                            96 * 1024 * 1024, 16, min_batch=1, max_batch=16, device=resolved_dev
                        )
                        dummy = torch.zeros((warm_bs, 1, 256, 256), dtype=torch.float32, device=resolved_dev)
                        _ = mdl(dummy, segmentation_only=True)
                        mps_ns = getattr(torch, "mps", None)
                        sync_fn = getattr(mps_ns, "synchronize", None)
                        if callable(sync_fn):
                            sync_fn()
                except RuntimeError as exc:
                    allow_fb = (os.environ.get("SLICERDSPACE_PNL_REQUESTED_DEVICE", str(device)).strip().lower() == "auto")
                    if not (allow_fb and str(resolved_dev).startswith(("cuda", "mps")) and is_gpu_oom(exc)):
                        raise
                    empty_gpu_cache_safely()
                    mdl = pnl_mod.CompNet().to("cpu")
                    mdl.eval()
                    selection["resolved"] = "cpu"
                return states, mdl

            with ThreadPoolExecutor(max_workers=1) as preload_pool:
                preload_fut = preload_pool.submit(_preload_during_ants)
                pnl_pipe.run_ants(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", private_template, "-m", baseline,
                                   "-t", "r", "-o", prefix], ants_bin=private_ants, threads=threads)
                preloaded = preload_fut.result()

            registered = nib.load(str(prefix) + "Warped.nii.gz")
            data = pnl_inf.normalize(registered.get_fdata(), percentile)
            _, _, mask = _fast_predict(data, model_dir, device=selection["resolved"], batch_size=batch_size, _preloaded=preloaded)
            candidate = work / "native_mask.nii.gz"
            _fast_finish_native(mask, registered, baseline, str(prefix) + "0GenericAffine.mat", candidate,
                                ants_bin=private_ants, threads=threads)
            result = nib.load(candidate)
            if result.shape != image.shape[:3] or not np.allclose(result.affine, image.affine, rtol=0, atol=1e-4):
                raise RuntimeError("Native geometry changed")
            with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".nii.gz", delete=False) as stream:
                staged = Path(stream.name)
            try:
                shutil.copyfile(candidate, staged)
                staged.replace(output_path)
            finally:
                staged.unlink(missing_ok=True)
        receipt = {**selection, "package_version": pnl_pipe.__version__, "source_commit": pnl_weights.SOURCE_COMMIT,
                   "input_sha256": pnl_weights.sha256(input_path), "output_sha256": pnl_weights.sha256(output_path),
                   "weights": json.loads((Path(model_dir) / "weights.json").read_text(encoding="utf-8")),
                   "template_sha256": pnl_weights.TEMPLATE_SHA256, "percentile": percentile,
                   "view_weights": [0.1, 0.4, 0.5], "threshold": 0.45,
                   "inverse_interpolation": "ANTs default linear then FillHoles and uint8 cast",
                   "human_anatomical_review": "pending"}
        if record:
            Path(record).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt

    def _patched_select_device(requested_arg: str, torch_mod=None):
        eff = os.environ.get("SLICERDSPACE_PNL_REQUESTED_DEVICE") or requested_arg
        return select_device(eff, torch=torch_mod)

    pnl_inf.select_device = _patched_select_device
    pnl_mod.CompNet.forward = _fast_forward
    pnl_inf.predict = _fast_predict
    pnl_pipe.predict = _fast_predict
    pnl_pipe.finish_native = _fast_finish_native
    pnl_pipe.run = _fast_run
    pnl_pipe._sdds_accelerated = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    record = {"requested": args.device, "framework": "pytorch"}
    try:
        record.update(select_device(args.device))
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        script_args = args.upstream_args
        if script_args[:1] == ["--"]:
            script_args = script_args[1:]
        if script_args[:1] != ["mask"]:
            raise ValueError("PNL Torch launcher requires the standalone 'mask' subcommand")
        os.environ["SLICERDSPACE_PNL_REQUESTED_DEVICE"] = args.device
        _install_pnl_runtime_accelerations()
        cli_device = "auto" if args.device == "mps" else args.device
        sys.argv = [str(args.script), *script_args, "--device", cli_device, "--record", str(args.record)]
        runpy.run_path(str(args.script), run_name="__main__")
    except Exception as exc:
        record.update(status="blocked", error=str(exc))
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
