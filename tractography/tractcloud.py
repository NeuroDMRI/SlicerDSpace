"""TractCloud integration for tractography parcellation.

This backend follows the SlicerDMRI TractCloud module design but keeps
dependency installation and execution explicit. It prepares a reproducible
``python -m tractcloud`` command, manifest, and QC record, and only runs when
``execute=True``.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from shared.external_process import python_tool_environment, text_subprocess_kwargs
from shared.output_contracts import vtp_streamline_count
from shared.product_staging import archive_run_records, new_product_staging, publish_products, retain_failed_products


TRACTCLOUD_REPOSITORY = "https://github.com/SlicerDMRI/TractCloud"
ADAPTER_REVISION = 'seeded-gpu-preferred-strict-runtime-20261013'
TRACTCLOUD_QC_VERSION = 4
MIN_BUNDLE_SAMPLING_SCREEN = 20  # Local sanity screen, not an author accuracy guarantee.
VALID_TRACTCLOUD_DEVICES = {"auto", "cpu", "cuda"}
TRACTCLOUD_COMPAT_BOOTSTRAP = (
    "import json, logging, os, random, runpy, sys; import numpy as np; import torch; "
    "seed = int(os.environ.get('SLICERDSPACE_TRACTCLOUD_SEED', '0')); "
    "random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); "
    "torch.cuda.manual_seed_all(seed); import tractcloud.pipeline as pipeline; "
    "pipeline.__dict__.setdefault('logging', logging); "
    "\nrequested = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'auto'\n"
    "available = torch.cuda.is_available() if requested != 'cpu' else False\n"
    "if requested == 'cuda' and not available:\n"
    "    raise RuntimeError('TractCloud CUDA requested but unavailable in its selected Python')\n"
    "resolved = 'cuda' if available else 'cpu'\n"
    "if resolved == 'cuda':\n"
    "    torch.empty(1, device='cuda:0')\n"
    "if '--device' in sys.argv:\n"
    "    sys.argv[sys.argv.index('--device') + 1] = resolved\n"
    "record = {'requested': requested, 'resolved': resolved, 'torch_version': str(torch.__version__)}\n"
    "print('TractCloud device: ' + json.dumps(record), flush=True)\n"
    "record_path = os.environ.get('SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD')\n"
    "if record_path:\n"
    "    with open(record_path, 'w') as stream:\n"
    "        json.dump(record, stream)\n"
    "if resolved == 'cuda' and hasattr(pipeline, 'run_inference'):\n"
    "    original_inference = pipeline.run_inference\n"
    "    def strict_gpu_inference(model, loader, global_feat, num_classes, device, **kwargs):\n"
    "        if str(device).startswith('cpu'):\n"
    "            raise RuntimeError('TractCloud GPU failed; automatic CPU retry disabled; fix CUDA or explicitly select CPU')\n"
    "        return original_inference(model, loader, global_feat, num_classes, device, **kwargs)\n"
    "    pipeline.run_inference = strict_gpu_inference\n"
    "runpy.run_module('tractcloud', run_name='__main__')"
)


def _file_info(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {"path": expanded.as_posix(), "exists": False}
    stat = expanded.stat()
    return {
        "path": expanded.resolve().as_posix(),
        "exists": True,
        "is_file": expanded.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@dataclass(frozen=True)
class TractCloudConfig:
    """Configuration for TractCloud whole-brain tractography parcellation."""

    subject_id: str
    input_tractography: Path
    output_dir: Path
    tractcloud_root: Optional[Path] = None
    python_executable: Optional[Path] = None
    include_other: bool = False
    device: str = "auto"
    batch_size: int = 2048
    data_dir: Optional[Path] = None
    create_mrb: bool = False
    execute: bool = False
    random_seed: int = 0
    model_assets_dir: Optional[Path] = None


def model_input_signature(data_dir: Optional[Path]) -> dict[str, object]:
    directory = (data_dir or Path(os.environ.get('TRACTCLOUD_DATA_DIR', str(Path.home() / '.cache/tractcloud')))).expanduser()
    result = {}
    for name in ('TrainedModel/best_tract_f1_model.pth', 'TrainedModel/cli_args.txt', 'TrainData_800clu800ol/HCP_mass_center.npy'):
        path = directory / name
        info = _file_info(path)
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            info['sha256'] = digest.hexdigest()
        result[name] = info
    return result


@dataclass(frozen=True)
class TractCloudResult:
    """Prepared or completed TractCloud parcellation run."""

    subject_id: str
    output_dir: Path
    manifest_path: Path
    qc_path: Path
    command: tuple[str, ...]
    executed: bool
    backend_status: dict[str, Any]
    return_code: Optional[int] = None
    status: str = 'planned'


def resolve_tractcloud_python(python_executable: Optional[Path] = None) -> Path:
    """Resolve the Python executable used to invoke ``python -m tractcloud``."""

    if python_executable is not None:
        candidate = python_executable.expanduser().absolute()
        if not candidate.is_file():
            raise FileNotFoundError(f"TractCloud Python executable not found: {candidate}")
        return candidate

    env_value = (
        os.environ.get("SLICERDSPACE_TRACTCLOUD_PYTHON_EXECUTABLE")
        or os.environ.get("TRACTCLOUD_PYTHON_EXECUTABLE")
        or os.environ.get("SLICERDSPACE_TRACTCLOUD_PYTHON")
        or os.environ.get("TRACTCLOUD_PYTHON")
    )
    if env_value:
        candidate = Path(env_value).expanduser().absolute()
        if not candidate.is_file():
            raise FileNotFoundError(f"TractCloud Python executable not found: {candidate}")
        return candidate

    from shared.backend_sources import available_darwin_arm64_tractcloud_python, needs_darwin_arm64_tractcloud_python
    if needs_darwin_arm64_tractcloud_python():
        arm64_py = available_darwin_arm64_tractcloud_python()
        if arm64_py is not None:
            return arm64_py

    for executable_name in ("PythonSlicer", "python3", "python"):
        found = shutil.which(executable_name)
        if found:
            return Path(found).absolute()

    from shared.external_process import _default_slicer_python, is_slicer_python
    if is_slicer_python(sys.executable):
        slicer_py = _default_slicer_python()
        if slicer_py is not None:
            return slicer_py.absolute()

    return Path(sys.executable).absolute()


def resolve_tractcloud_root(tractcloud_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve explicit/environment sources before the verified prepared cache."""

    candidate_value = tractcloud_root
    if candidate_value is None:
        env_value = os.environ.get("SLICERDSPACE_TRACTCLOUD_ROOT") or os.environ.get("TRACTCLOUD_ROOT")
        candidate_value = Path(env_value) if env_value else None
    if candidate_value is None:
        from shared.backend_sources import available_source
        candidate_value = available_source("tractcloud")
    if candidate_value is None:
        return None

    candidate = candidate_value.expanduser().resolve()
    package_candidates = [
        candidate / "tractcloud",
        candidate / "src" / "tractcloud",
    ]
    if not any(path.is_dir() for path in package_candidates):
        raise FileNotFoundError(
            f"TractCloud source checkout not found at {candidate}. Expected tractcloud/ or src/tractcloud/."
        )
    return candidate


def _tractcloud_pythonpath(root: Optional[Path]) -> Optional[str]:
    if root is None:
        return None
    src_dir = root / "src"
    return src_dir.as_posix() if (src_dir / "tractcloud").is_dir() else root.as_posix()


def _tractcloud_source_info(root: Optional[Path]) -> dict[str, Any] | None:
    if root is None:
        return None
    info: dict[str, Any] = {
        "root": root.as_posix(),
        "repository": TRACTCLOUD_REPOSITORY,
        "pythonpath": _tractcloud_pythonpath(root),
    }
    if (root / ".git").is_dir():
        try:
            commit = subprocess.check_output(
                ["git", "-C", root.as_posix(), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                **text_subprocess_kwargs(),
            ).strip()
            info["commit"] = commit
        except Exception:
            pass
    return info


def _tractcloud_env(config: "TractCloudConfig", backend_status: Optional[dict[str, Any]] = None) -> dict[str, str]:
    env: dict[str, str] = {}
    source = backend_status.get("source") if backend_status else None
    root_value = source.get("pythonpath") if isinstance(source, dict) else None
    if root_value:
        env["PYTHONPATH"] = root_value
    if config.data_dir is not None:
        env["TRACTCLOUD_DATA_DIR"] = config.data_dir.expanduser().resolve().as_posix()
    env['SLICERDSPACE_TRACTCLOUD_SEED'] = str(config.random_seed)
    env['SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD'] = str(config.output_dir.expanduser().resolve() / 'tractcloud_device.json')
    return env


def inspect_tractcloud_backend(
    python_executable: Optional[Path] = None,
    tractcloud_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Inspect TractCloud package availability without importing torch in-process."""

    python_path = resolve_tractcloud_python(python_executable)
    root = resolve_tractcloud_root(tractcloud_root)
    status: dict[str, Any] = {
        "available": False,
        "python_executable": python_path.as_posix(),
        "repository": TRACTCLOUD_REPOSITORY,
        "source": _tractcloud_source_info(root),
        "packages": {
            "tractcloud": False,
            "torch": False,
        },
        "error": None,
    }
    probe = (
        "import importlib.util, json; "
        "print(json.dumps({"
        "'tractcloud': importlib.util.find_spec('tractcloud') is not None, "
        "'torch': importlib.util.find_spec('torch') is not None"
        "}))"
    )
    try:
        env = python_tool_environment(python_path)
        source_pythonpath = _tractcloud_pythonpath(root)
        if source_pythonpath:
            env["PYTHONPATH"] = source_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [python_path.as_posix(), "-c", probe],
            check=True,
            capture_output=True,
            env=env,
            **text_subprocess_kwargs(),
        )
        packages = json.loads(completed.stdout.strip() or "{}")
        status["packages"].update({key: bool(packages.get(key)) for key in status["packages"]})
        status["available"] = bool(status["packages"]["tractcloud"] and status["packages"]["torch"])
    except Exception as exc:
        status["error"] = str(exc)
    return status


def validate_tractcloud_inputs(config: TractCloudConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.input_tractography.suffix.lower() not in {".vtk", ".vtp"}:
        raise ValueError("TractCloud input must be a VTK/VTP tractography file.")
    if not config.input_tractography.expanduser().is_file():
        raise FileNotFoundError(f"Input tractography not found: {config.input_tractography}")
    if config.device not in VALID_TRACTCLOUD_DEVICES:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if config.batch_size < 64:
        raise ValueError("batch_size must be >= 64.")
    if isinstance(config.random_seed, bool) or not isinstance(config.random_seed, int) or not 0 <= config.random_seed < 2**32:
        raise ValueError('random_seed must be an integer in [0, 2**32).')
    if config.data_dir is not None and not config.data_dir.expanduser().is_dir():
        raise FileNotFoundError(f"TractCloud data directory not found: {config.data_dir}")
    backend_status = inspect_tractcloud_backend(config.python_executable, config.tractcloud_root)
    if config.execute and not backend_status["available"]:
        missing = [name for name, available in backend_status["packages"].items() if not available]
        raise RuntimeError(f"TractCloud backend is incomplete. Missing package(s): {', '.join(missing)}")


def build_tractcloud_command(config: TractCloudConfig) -> tuple[str, ...]:
    python_path = resolve_tractcloud_python(config.python_executable)
    output_dir = config.output_dir.expanduser().resolve()
    cmd: list[str] = [
        python_path.as_posix(),
        "-m",
        "tractcloud",
        "--input",
        config.input_tractography.expanduser().resolve().as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--device",
        config.device,
        "--batch-size",
        str(config.batch_size),
    ]
    if config.include_other:
        cmd.append("--include-other")
    if config.create_mrb:
        cmd.append("--mrb")
    if config.data_dir is not None:
        cmd.extend(["--data-dir", config.data_dir.expanduser().resolve().as_posix()])
    return tuple(cmd)


def summarize_tractcloud_outputs(output_dir: Path) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    bundle_files = sorted(path for path in root.rglob("*.vtp")
                          if not any(part.startswith('.') for part in path.relative_to(root).parts)) if root.is_dir() else []
    category_dirs = sorted(
        item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith('.')
    ) if root.is_dir() else []
    mrb_files = sorted(root.glob("*.mrb")) if root.is_dir() else []

    line_counts: dict[str, Optional[int]] = {}
    for path in bundle_files:
        relative = path.relative_to(root).as_posix()
        line_counts[relative] = vtp_streamline_count(path)
    readable_line_counts = {
        name: count for name, count in line_counts.items() if count is not None
    }
    return {
        "root": root.as_posix(),
        "root_exists": root.is_dir(),
        "category_dirs": category_dirs,
        "bundle_count": len(bundle_files),
        "bundle_readable_count": len(readable_line_counts),
        "bundle_nonempty_count": sum(count > 0 for count in readable_line_counts.values()),
        "assigned_streamline_count": sum(readable_line_counts.values()),
        "bundle_streamline_counts": line_counts,
        "empty_bundle_files": sorted(
            name for name, count in readable_line_counts.items() if count == 0
        ),
        "unreadable_bundle_files": sorted(
            name for name, count in line_counts.items() if count is None
        ),
        "bundle_files": [path.as_posix() for path in bundle_files],
        "mrb_files": [path.as_posix() for path in mrb_files],
    }


def _tractcloud_output_contract_error(output_summary: dict[str, Any]) -> Optional[str]:
    bundle_count = int(output_summary.get("bundle_count", 0) or 0)
    readable_count = int(output_summary.get("bundle_readable_count", 0) or 0)
    nonempty_count = int(output_summary.get("bundle_nonempty_count", 0) or 0)
    assigned_count = int(output_summary.get("assigned_streamline_count", 0) or 0)
    if bundle_count <= 0:
        return "TractCloud did not produce any bundle files."
    if readable_count != bundle_count:
        return "TractCloud produced one or more VTP files that could not be decoded."
    if nonempty_count != bundle_count or assigned_count <= 0:
        return "TractCloud produced empty bundle outputs or assigned no streamlines."
    return None


@scheduled_stage('tractcloud', gpu=True)
def run_tractcloud_parcellation(
    config: TractCloudConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> TractCloudResult:
    """Prepare or execute TractCloud parcellation for one tractogram."""

    if config.data_dir is None:
        from shared.optional_assets import resolve_tractcloud_data_directory
        shared_data = resolve_tractcloud_data_directory(asset_root=config.model_assets_dir)
        if shared_data is not None:
            config = replace(config, data_dir=shared_data)
    validate_tractcloud_inputs(config)
    backend_status = inspect_tractcloud_backend(config.python_executable, config.tractcloud_root)
    command = build_tractcloud_command(config)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "tractcloud_manifest.json"
    qc_path = output_dir / "tractcloud_qc.json"
    previous_records = archive_run_records(output_dir, ('tractcloud_manifest.json', 'tractcloud_qc.json')) if config.execute else None
    output_summary = summarize_tractcloud_outputs(output_dir)
    resolved_root = resolve_tractcloud_root(config.tractcloud_root)

    manifest = {
        "kind": "tractcloud_parcellation",
        "version": TRACTCLOUD_QC_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "name": "TractCloud registration-free tractography parcellation",
        "subject_id": config.subject_id,
        "input_tractography": config.input_tractography.expanduser().resolve().as_posix(),
        "output_dir": output_dir.as_posix(),
        "tractcloud_root": resolved_root.as_posix() if resolved_root else None,
        "include_other": config.include_other,
        "device": config.device,
        "batch_size": config.batch_size,
        "random_seed": config.random_seed,
        "data_dir": config.data_dir.expanduser().resolve().as_posix() if config.data_dir else None,
        "create_mrb": config.create_mrb,
        "executed": bool(config.execute),
        "status": "planned" if not config.execute else "running",
        "command": list(command),
        "backend": backend_status,
        "previous_records": previous_records,
        "dependencies": {
            "input_tractography": _file_info(config.input_tractography),
            "data_dir": _file_info(config.data_dir) if config.data_dir else None,
            "model_inputs": model_input_signature(config.data_dir),
        },
        "outputs": {
            "qc": qc_path.as_posix(),
            "summary": output_summary,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if log:
        mode = "Executing" if config.execute else "Prepared dry-run"
        log(f"[TractCloud] {mode} command: {' '.join(command)}")
        log(f"[TractCloud] Manifest: {manifest_path}")

    start_time = time.monotonic()
    return_code: Optional[int] = None
    execution_error: Optional[str] = None
    execution_exception: Optional[Exception] = None
    staging = None
    published_archive = None
    if config.execute:
        env = python_tool_environment(backend_status['python_executable'], _tractcloud_env(config, backend_status))
        try:
            (output_dir / 'tractcloud_device.json').unlink(missing_ok=True)
            staging = new_product_staging(output_dir, 'tractcloud')
            runtime_command = list(command)
            runtime_command[runtime_command.index('--output-dir') + 1] = str(staging)
            # Preserve the official network/weights while managing devices in
            # its own Python and refusing the upstream CUDA-error CPU retry.
            runtime_command[1:3] = ['-c', TRACTCLOUD_COMPAT_BOOTSTRAP]
            manifest['executed_command'] = runtime_command
            manifest['compatibility_adjustments'] = ['Supply missing upstream logging; resolve device in the selected Python and refuse automatic CPU retries after CUDA failure; network and weights unchanged']
            completed = subprocess.run(runtime_command, check=True, env=env)
            return_code = completed.returncode
            device_record = output_dir / 'tractcloud_device.json'
            if device_record.is_file():
                manifest['device_execution'] = json.loads(device_record.read_text(encoding='utf-8'))
        except subprocess.CalledProcessError as exc:
            return_code = exc.returncode
            execution_error = str(exc)
            execution_exception = exc
        except OSError as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
            execution_exception = exc
        finally:
            output_summary = summarize_tractcloud_outputs(staging) if staging else {}
            if execution_error is None:
                execution_error = _tractcloud_output_contract_error(output_summary)
            if staging:
                try:
                    if execution_error is None:
                        previous_names = ['Association', 'Projection', 'Commissural', 'Cerebellar', 'Superficial', 'Other']
                        previous_names.extend(p.name for p in output_dir.glob('*.mrb'))
                        published_archive = publish_products(staging, output_dir, previous_names)
                        output_summary = summarize_tractcloud_outputs(output_dir)
                    else:
                        staging = retain_failed_products(staging, output_dir)
                        output_summary = summarize_tractcloud_outputs(staging)
                except OSError as exc:
                    execution_error = f'Failed to publish TractCloud products: {exc}'

    if config.execute and execution_error is None:
        execution_error = _tractcloud_output_contract_error(output_summary)

    elapsed_seconds = time.monotonic() - start_time
    qc = {
        "kind": "tractcloud_parcellation_qc",
        "version": TRACTCLOUD_QC_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "subject_id": config.subject_id,
        "status": "failed" if execution_error else ("completed" if config.execute else "planned"),
        "executed": bool(config.execute),
        "automatic_qc_passed": bool(config.execute and execution_error is None),
        "checks": {"decoded_nonempty_bundle_products": _tractcloud_output_contract_error(output_summary) is None},
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "human_review": "pending",
        "quantification_eligible": False,
        "sampling_screen": {
            "minimum_streamlines": MIN_BUNDLE_SAMPLING_SCREEN,
            "scope": "Local sampling sanity, separate from file/geometry QC and expert anatomical review",
            "excluded_from_quantification": [name for name, count in
                output_summary.get('bundle_streamline_counts', {}).items()
                if count is None or count < MIN_BUNDLE_SAMPLING_SCREEN],
        },
        "sampling_warnings": [f'{name}: {count} streamlines; below the local {MIN_BUNDLE_SAMPLING_SCREEN}-track screen'
            for name, count in output_summary.get('bundle_streamline_counts', {}).items()
            if count is None or count < MIN_BUNDLE_SAMPLING_SCREEN],
        "scientific_scope": "Fresh decoded bundle products, not clinical or expert anatomical certification; seed-controlled upstream neighborhood sampling, not cross-platform bitwise equivalence.",
        "random_seed": config.random_seed,
        "repeatability_scope": "Python/NumPy/PyTorch seeds are fixed; identical hardware/configuration repetition is tested separately, not guaranteed across devices or library versions",
        "previous_products": published_archive,
        "backend_available": bool(backend_status["available"]),
        "output_summary": output_summary,
        "error": execution_error,
    }
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["status"] = qc["status"]
    manifest["executed"] = bool(config.execute)
    manifest["return_code"] = return_code
    manifest["runtime_seconds"] = elapsed_seconds
    manifest["error"] = execution_error
    manifest["previous_products"] = published_archive
    manifest["outputs"]["summary"] = output_summary
    if config.execute:
        manifest['dependencies']['model_inputs'] = model_input_signature(config.data_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if execution_exception is not None:
        raise execution_exception

    return TractCloudResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        qc_path=qc_path,
        command=command,
        executed=bool(config.execute),
        backend_status=backend_status,
        return_code=return_code,
        status=qc['status'],
    )
