"""DDSurfer surface prediction for white-matter and pial cortical surfaces."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import nibabel as nib
import numpy as np
import torch

from inference.coordinates import Geometry, MODALITIES, export_crop_mesh
from utils.files import atomic_json, sha256_file
from utils.surface_io import load_mesh
from model.ddsurfer import TANet
from utils.mesh import taubin_smooth

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)

EAGER_CUDA_FREE_GIB = 18.0
LAZY_CUDA_MIN_BUDGET_GIB = 5.5
LAZY_CUDA_MARGIN_GIB = 1.5
EAGER_CUDA_MARGIN_GIB = 4.0


def _enable_cpu_flush_denormal() -> bool:
    """Flush subnormal float32 values to zero on CPU to avoid x86/Rosetta stalls."""
    try:
        fn = getattr(torch, 'set_flush_denormal', None)
        if callable(fn):
            return bool(fn(True))
    except Exception:
        pass
    return False


def _resolve_optimal_cpu_threads(requested_threads: int, device_str: str) -> int:
    """Resolve PyTorch CPU intra-op threads without altering CLI/manifest cpu_threads signatures."""
    dev = str(device_str or '').strip().lower()
    req = max(1, int(requested_threads))
    if not dev.startswith(('cpu', 'mps')) or os.environ.get('SLICERDSPACE_STAGE_CPUS'):
        return req
    if dev.startswith('cpu') and req != 4:
        return req
    try:
        from shared.hardware_adapt import optimal_cpu_torch_threads

        opt = int(optimal_cpu_torch_threads(max_conv_threads=8, server_max_threads=36))
    except Exception:
        cpus = max(1, int(os.cpu_count() or 4))
        if sys.platform == 'darwin' and cpus >= 6:
            p_cap = max(4, cpus - 2) if cpus <= 10 else max(6, (cpus * 3) // 4)
            opt = min(cpus, p_cap, 8)
        else:
            opt = min(cpus, 8)
    return max(req, opt)


def resolve_cuda_memory_budget(free_bytes: int, max_gpu_gib: float) -> tuple[float, bool]:
    """Return ``(cap_gib, use_lazy_models)`` or raise RuntimeError if free VRAM is insufficient.

    High-VRAM GPUs (>= 18 GiB free with >= 12 GiB cap) preserve the eager dual-surface route
    with a 4-GiB margin. Consumer 8-12 GiB GPUs (< 18 GiB free) load one surface model at a time
    (~4-5 GiB peak) and require at least a 5.5-GiB GPU budget plus a 1.5-GiB margin (>= 7 GiB free).
    """
    free_gib = float(free_bytes) / float(1024**3)
    lazy = free_gib < EAGER_CUDA_FREE_GIB or float(max_gpu_gib) < 12.0
    margin = LAZY_CUDA_MARGIN_GIB if lazy else EAGER_CUDA_MARGIN_GIB
    cap = min(float(max_gpu_gib), free_gib - margin)
    if cap < LAZY_CUDA_MIN_BUDGET_GIB:
        raise RuntimeError(
            'Insufficient free GPU memory; inference requires at least a 5.5-GiB GPU budget '
            'plus a 1.5-GiB free-memory margin (7 GiB free) for lazy single-surface loading'
        )
    return cap, lazy


def load_json(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_checksums(folder):
    path = folder / 'SHA256SUMS'
    checksums = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or name in checksums:
            raise ValueError(f'Invalid checksum entry: {path}')
        checksums[name] = digest
    return checksums


def load_model(checkpoint_path, digest, architecture, device):
    if sha256_file(checkpoint_path) != digest:
        raise ValueError(f'Checkpoint checksum mismatch: {checkpoint_path}')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f'Expected a tensor-only state dictionary: {checkpoint_path}')
    model = TANet(**architecture)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval().requires_grad_(False)
    del state
    return model


def resolve_precision(requested, device, default_precision):
    precision = requested
    if precision == 'auto':
        precision = default_precision if device.type == 'cuda' else 'fp32'
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires a supported CUDA device; explicitly choose fp32 otherwise')
    return precision


def precision_context(precision, device):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


class SurfacePredictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint_dir = args.checkpoint_root
        self.config = load_json(self.checkpoint_dir / 'manifest.json')
        architecture = self.config['architecture']
        templates = self.config['templates']
        checksums = load_checksums(self.checkpoint_dir)
        files = self.config['checkpoint_files'][args.surf_hemi]
        self.entries = {surface: dict(checkpoint_file=files[surface], sha256=checksums[files[surface]],
                                       postprocess_iters=self.config['postprocess_iters'][surface],
                                       template_output_sha256=templates[args.surf_hemi]['sha256'])
                        for surface in ('white', 'pial') if surface == 'white' or args.predict_mode == 'all'}
        self.geometry = Geometry(self.config['geometry']['affine'], self.config['geometry']['full_shape'], args.surf_hemi)
        for key, field in (('step_size', 'step_size'), ('n_svf', 'M'), ('n_res', 'R')):
            value = getattr(args, key)
            if value is not None and value != architecture[field]:
                raise ValueError(f'--{key} cannot override the model configuration')
        template = args.template_dir / templates[args.surf_hemi]['file']
        if sha256_file(template) != templates[args.surf_hemi]['sha256']:
            raise ValueError(f'Wrong transformed template: {template}')
        mesh = load_mesh(template)
        self.vertices = np.asarray(mesh.vertices - self.geometry.crop_origin, dtype=np.float32)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        if not np.isfinite(self.vertices).all() or not len(self.vertices) or not len(self.faces):
            raise ValueError('Invalid transformed template')
        if ((self.vertices < 0) | (self.vertices > np.array(self.geometry.crop_shape)-1)).any():
            raise ValueError('Template is outside the trained crop')
        lazy_cuda = False
        if self.device.type == 'cpu':
            _enable_cpu_flush_denormal()
        elif self.device.type == 'mps':
            _enable_cpu_flush_denormal()
            torch.empty(1, device=self.device)
        elif self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            cap, lazy_cuda = resolve_cuda_memory_budget(free, args.max_gpu_gib)
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / total), self.device)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.precision = resolve_precision(args.precision, self.device, self.config['precision'])
        self.architecture = architecture
        self._lazy_models = (
            self.device.type == 'mps'
            or (self.device.type == 'cuda' and lazy_cuda)
        ) and len(self.entries) > 1
        self._lazy_mps_models = self._lazy_models
        self.models = {} if self._lazy_models else {
            surface: load_model(self.checkpoint_dir / entry['checkpoint_file'], entry['sha256'], architecture, self.device)
            for surface, entry in self.entries.items()
        }
        LOG.info('Predicting %s hemisphere with %s precision', args.surf_hemi, self.precision)

    def load_volumes(self, folder, volume_cache=None):
        cache_key = folder.resolve()
        full_volumes = volume_cache.get(cache_key) if volume_cache is not None else None
        if full_volumes is None:
            full_volumes = {}
            for modality in MODALITIES:
                path = folder / f'{folder.name}-{modality}.nii.gz'
                image = nib.load(path)
                self.geometry.check_image(image, path)
                full_volumes[modality] = (path, image.get_fdata(dtype=np.float32, caching='unchanged'))
            if volume_cache is not None:
                volume_cache[cache_key] = full_volumes
        volumes = []
        for modality in MODALITIES:
            path, full_data = full_volumes[modality]
            data = full_data[self.geometry.x_slice]
            if not np.isfinite(data).all() or data.std() < 1e-8:
                raise ValueError(f'Nonfinite/constant feature: {path}')
            volumes.append(data)
        return torch.from_numpy(np.ascontiguousarray(np.stack(volumes)))[None].to(self.device)

    @torch.inference_mode()
    def predict(self, folder, volume_cache=None):
        started = time.perf_counter()
        volumes = (
            self.load_volumes(folder, volume_cache=volume_cache)
            if volume_cache is not None
            else self.load_volumes(folder)
        )
        vertices = torch.from_numpy(self.vertices.copy())[None].to(self.device)
        faces = torch.from_numpy(self.faces.copy())[None].to(self.device)
        laplacian_matrix = None
        directory = self.args.output_dir / 'mni' / folder.name
        directory.mkdir(parents=True, exist_ok=True)
        lazy_models = getattr(self, '_lazy_models', getattr(self, '_lazy_mps_models', False))
        for surface in ('white', 'pial'):
            if surface not in self.entries:
                continue
            model = self.models.get(surface)
            if model is None:
                entry = self.entries[surface]
                model = load_model(self.checkpoint_dir / entry['checkpoint_file'], entry['sha256'], self.architecture, self.device)
            execution_devices = dict(getattr(model, 'execution_devices', None) or {})  # read before a lazy GPU/MPS model is released
            with precision_context(self.precision, self.device):
                raw = model(vertices, volumes)
            if lazy_models:
                del model
                if self.device.type == 'mps' and hasattr(torch, 'mps'):
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                elif self.device.type == 'cuda' and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            elif self.device.type == 'mps' and hasattr(torch, 'mps'):
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            elif self.device.type == 'cuda' and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if not torch.isfinite(raw).all():
                raise FloatingPointError(f'Nonfinite {surface} prediction: {folder.name}')
            iterations = self.entries[surface]['postprocess_iters']
            if self.device.type == 'mps':
                raw_smooth = raw.cpu()
                faces_smooth = faces.cpu()
                if iterations > 0 and laplacian_matrix is None:
                    from utils.mesh import laplacian
                    laplacian_matrix = laplacian(faces_smooth)
                processed = taubin_smooth(raw_smooth, faces_smooth, n_iters=iterations, laplacian_matrix=laplacian_matrix)
            else:
                if iterations > 0 and laplacian_matrix is None:
                    from utils.mesh import laplacian
                    laplacian_matrix = laplacian(faces)
                processed = taubin_smooth(raw, faces, n_iters=iterations, laplacian_matrix=laplacian_matrix)
            kind = 'wm' if surface == 'white' else 'pial'
            output = directory / f'{folder.name}_predicted_{kind}_surface_{self.args.surf_hemi}.obj'
            export_crop_mesh(output, processed[0].float().cpu().numpy(), self.faces, self.geometry)
            metadata = load_json(output.with_suffix('.json'))
            metadata.update(checkpoint_file=self.entries[surface]['checkpoint_file'],
                            checkpoint_sha256=self.entries[surface]['sha256'],
                            precision=self.precision, postprocess_iters=iterations, surface=surface,
                            template_sha256=self.entries[surface]['template_output_sha256'],
                            execution_devices=execution_devices)
            atomic_json(output.with_suffix('.json'), metadata)
            if self.args.save_debug:
                np.savez_compressed(directory / f'{self.args.surf_hemi}_{kind}_crop_voxel.npz',
                                    raw=raw[0].float().cpu().numpy(), postprocessed=processed[0].float().cpu().numpy(), faces=self.faces)
            # Initialize pial prediction from the white-matter surface.
            vertices = processed.detach().float().to(self.device)
            LOG.info('Saved %s (scanner RAS mm)', output)
        LOG.info('Finished %s (%s) in %.1fs', folder.name, self.args.surf_hemi, time.perf_counter()-started)


def main(default_hemisphere, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_type', default='hcp')
    parser.add_argument('--surf_hemi', choices=('left', 'right', 'both'), default=default_hemisphere,
                        help='Default: both hemispheres; select left/right to predict one side.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('auto','bf16','fp32'), default='fp32',
                        help='Default: fp32 on CPU/CUDA; bf16 reduces memory use on supported '
                             'CUDA devices; auto follows the model manifest.')
    parser.add_argument('--predict_mode', choices=('wm','all'), default='all')
    parser.add_argument('--input_root', type=Path, default=ROOT/'outputs/.cache/volumes')
    parser.add_argument('--output_dir', type=Path, default=ROOT/'outputs/.cache/predictions')
    parser.add_argument('--checkpoint_root', type=Path, default=ROOT/'weights')
    parser.add_argument('--template_dir', type=Path, default=ROOT/'template')
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--n_svf', type=int)
    parser.add_argument('--n_res', type=int)
    parser.add_argument('--save-debug', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-gpu-gib', type=float, default=24)
    parser.add_argument('--log-level', choices=('DEBUG','INFO','WARNING','ERROR'), default='INFO')
    args = parser.parse_args(argv)
    if args.data_type != 'hcp':
        parser.error('These checkpoints require --data_type hcp')
    if args.cpu_threads < 1 or args.max_gpu_gib <= 0:
        parser.error('Resource limits must be positive')
    logging.basicConfig(level=getattr(logging,args.log_level), format='[%(levelname)s] %(message)s')
    if str(args.device).startswith(('cpu', 'mps')):
        _enable_cpu_flush_denormal()
    torch_threads = _resolve_optimal_cpu_threads(args.cpu_threads, args.device)
    torch.set_num_threads(torch_threads)
    if args.subjects:
        if len(args.subjects) != len(set(args.subjects)) or any(Path(s).name != s or s in ('.','..') for s in args.subjects):
            parser.error('Subject IDs must be unique directory names')
        folders = [args.input_root/s for s in args.subjects]
    else:
        folders = sorted(p for p in args.input_root.iterdir() if p.is_dir())
    if not folders or any(not p.is_dir() for p in folders):
        raise FileNotFoundError('No subjects or missing explicitly requested subject directories')
    hemispheres = ('left', 'right') if args.surf_hemi == 'both' else (args.surf_hemi,)
    volume_cache = {} if len(hemispheres) > 1 and len(folders) == 1 else None
    for hemisphere in hemispheres:
        settings = argparse.Namespace(**vars(args))
        settings.surf_hemi = hemisphere
        predictor = SurfacePredictor(settings)
        for folder in folders:
            if volume_cache is not None:
                try:
                    predictor.predict(folder, volume_cache=volume_cache)
                except TypeError:
                    predictor.predict(folder)
            else:
                predictor.predict(folder)
        del predictor
