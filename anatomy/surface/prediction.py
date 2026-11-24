"""Compatibility API delegating all inference to the pinned DDSurfer release."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from .backend_bridge import BACKEND_ROOT, run_backend, python_executable


@dataclass(frozen=True)
class PredictionStageConfig:
    subject_id: str
    hemisphere: str
    predict_mode: str
    device: str
    step_size: float
    n_svf: int
    n_res: int
    template_dir: Path
    checkpoint_root: Path
    input_root: Path
    output_root: Path
    precision: str = 'fp32'
    python_bin: Optional[Path] = None
    cpu_threads: int = 4
    max_gpu_gib: float = 24.


@dataclass(frozen=True)
class PredictionResult:
    subject_id: str
    hemisphere: str
    wm_mesh_path: Path
    pial_mesh_path: Optional[Path]


def discover_subject_directories(input_root, subjects=None):
    root = Path(input_root)
    names = subjects if subjects is not None else sorted(p.name for p in root.iterdir() if p.is_dir())
    return {name: root / name for name in names if (root / name).is_dir()}


def predict_subject(config):
    from shared.environment import resolve_auto_torch_device, surface_gpu_memory_decision
    requested = str(config.device or 'auto').strip().lower()
    surface_py = Path(python_executable(config.python_bin))
    resolved = resolve_auto_torch_device(config.device, surface_py)
    if requested in {'', 'auto'} and resolved.startswith('cuda'):
        gpu_action, _ = surface_gpu_memory_decision('auto', surface_py, resolved_device=resolved)
        if gpu_action == 'cpu':
            resolved = 'cpu'
    config = replace(config, device=resolved)
    command = [BACKEND_ROOT / 'DDSurfer_predict.py', '--subjects', config.subject_id,
        '--surf_hemi', config.hemisphere, '--predict_mode', config.predict_mode,
        '--device', config.device, '--precision', config.precision,
        '--cpu-threads', str(config.cpu_threads), '--max-gpu-gib', str(config.max_gpu_gib),
        '--input_root', config.input_root, '--output_dir', config.output_root,
        '--checkpoint_root', config.checkpoint_root, '--template_dir', config.template_dir,
        '--step_size', config.step_size, '--n_svf', config.n_svf, '--n_res', config.n_res]
    run_backend(command, python_bin=config.python_bin)
    directory = config.output_root / 'mni' / config.subject_id
    wm = directory / f'{config.subject_id}_predicted_wm_surface_{config.hemisphere}.obj'
    pial = directory / f'{config.subject_id}_predicted_pial_surface_{config.hemisphere}.obj'
    return PredictionResult(config.subject_id, config.hemisphere, wm, pial if config.predict_mode == 'all' else None)


class SurfacePredictor:
    def __init__(self, config):
        self.config = config

    def predict(self, subject_dir=None):
        config = self.config
        if subject_dir is not None:
            subject_dir = Path(subject_dir)
            config = replace(config, subject_id=subject_dir.name, input_root=subject_dir.parent)
        return predict_subject(config)
