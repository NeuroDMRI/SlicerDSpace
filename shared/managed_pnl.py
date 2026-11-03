"""Pinned PyTorch PNL assets, with a separate author-compatible ANTs runtime."""

from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import urllib.request

from shared.external_process import external_python_environment

PNL_COMMIT = "d7fe0fd4976c2d5a3489526f75d2a00dae3e1fe1"
PNL_TORCH_COMMIT = "8d201a50447ebea8b10a4f252bfd262931603970"
PNL_TERMS_URL = "https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation"
PNL_MODEL_TERMS_URL = PNL_TERMS_URL + "/releases/tag/v0.3"
PNL_TEMPLATE_TERMS_URL = "https://www.nitrc.org/frs/?group_id=432"
PNL_TORCH_URL = "https://github.com/ChengjinLii/PNLBrainMask-Torch"
PNL_WHEEL = "pnl_brain_mask-0.1.0-py3-none-any.whl"
ASSETS = {
    PNL_WHEEL: (
        f"https://raw.githubusercontent.com/ChengjinLii/PNLBrainMask-Torch/{PNL_TORCH_COMMIT}/artifacts/{PNL_WHEEL}",
        "44f16ebe938db2ad4be0a5d4f1fe834b73ffdcdcdc1205c7ba965781a5916dac"),
    "IITmean_b0_256.nii.gz": (
        "https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz",
        "2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead"),
}
RESOURCE_HASHES = {
    "pnl-models/weights.json": "b1af9575253ec94aa5e52b22a14731745e07fac4fb9453c70074e9e2cae48baa",
    "pnl-models/IITmean_b0_256.nii.gz": ASSETS["IITmean_b0_256.nii.gz"][1],
    "pnl-models/axial.pt": "27bf8c38a38dfef44f1bb25ab2d180803d9a5794e5dcdb69c4ab0692f2744287",
    "pnl-models/coronal.pt": "a11da30afb41d8db1afa7c447ff94260e7421b0e00c4dd1018825408059a2feb",
    "pnl-models/sagittal.pt": "7dd553b55a4d1d108ddd3e0bad99d6275a4c3a9c833dd469a0ad026d393afdcc",
}


def pnl_python_environment(python, extra_env=None, path_prepend=()):
    """Expose only the selected Python's CUDA wheel libraries to its process."""
    import os
    prefix = Path(python).expanduser().absolute().parent.parent
    cuda_bins = sorted(prefix.glob("lib/python*/site-packages/nvidia/*/bin"))
    environment = external_python_environment(extra_env, path_prepend=[*path_prepend, prefix / "bin", *cuda_bins])
    libraries = [prefix / "lib", *sorted(prefix.glob("lib/python*/site-packages/nvidia/*/lib"))]
    paths = [str(path) for path in libraries if path.is_dir()]
    paths.extend(environment.get("LD_LIBRARY_PATH", "").split(os.pathsep))
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return environment


def verified_download(name, cache):
    from shared.managed_runtimes import file_hash
    url, expected = ASSETS[name]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if target.is_symlink():
        raise ValueError("Runtime download cannot be a symlink")
    if target.is_file() and file_hash(target) == expected:
        return target
    part = target.with_suffix(target.suffix + ".part")
    if part.is_symlink():
        raise ValueError("Runtime partial download cannot be a symlink")
    print("[PNL setup] Downloading " + name, flush=True)
    with urllib.request.urlopen(url, timeout=60) as source, part.open("wb") as destination:
        size = 0
        for block in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(block)
            size += len(block)
            if size > 512 * 1024 * 1024:
                raise ValueError("PNL download exceeds its bounded size")
            print(f"[PNL setup] {name}: {size // (1024 * 1024)} MiB", flush=True)
    if file_hash(part) != expected:
        raise RuntimeError("PNL checksum mismatch: " + name)
    part.replace(target)
    return target


def extract_regular_files(archive, destination):
    """Drop the one archive root; reject links, traversal and oversized data."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Runtime asset directory cannot be a symlink")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) > 10000 or sum(item.size for item in members) > 1024**3:
            raise ValueError("Runtime archive exceeds extraction limits")
        roots = set()
        for item in members:
            path = PurePosixPath(item.name)
            if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in item.name
                    or ":" in item.name or not (item.isfile() or item.isdir())):
                raise ValueError("Unsafe runtime archive member: " + item.name)
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("Runtime archive must have exactly one top-level folder")
        destination.mkdir(parents=True, exist_ok=True)
        for item in members:
            parts = PurePosixPath(item.name).parts[1:]
            if not parts or not item.isfile():
                continue
            target = destination.joinpath(*parts)
            if any(part.is_symlink() for part in (target, *target.parents)):
                raise ValueError("Refusing archive extraction through a symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(item) as data, target.open("wb") as output:
                shutil.copyfileobj(data, output)


def provision_assets(prefix, cache, *, repair=False, asset_root=None):
    from shared.managed_runtimes import _run, file_hash
    from shared.optional_assets import verified_resource_directory
    from shared.model_assets import configured_model_assets_root

    source = verified_download(PNL_WHEEL, cache)
    # Installing this package does not install TensorFlow, Keras, h5py or conversion.
    # Keep a compatible existing Torch build; new managed prefixes use the
    # CPU build covered by paired validation, not an unqualified CUDA upgrade.
    python = str(prefix / "bin/python")
    probe = subprocess.run([python, "-c", "import torch; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); assert (2,4)<=v<(3,0)"],
                           capture_output=True, env=pnl_python_environment(python), timeout=60)
    if probe.returncode:
        _run([python, "-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
    if repair:
        _run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(source)])
    _run([python, "-m", "pip", "install", str(source)])
    models = prefix / "pnl-models"
    if models.is_symlink():
        raise ValueError("PNL model directory cannot be a symlink")
    models.mkdir(parents=True, exist_ok=True)
    template = models / "IITmean_b0_256.nii.gz"
    shared = verified_resource_directory("pnl-cnn", asset_root=asset_root)
    if shared:
        for relative, expected in RESOURCE_HASHES.items():
            target, original = models / Path(relative).name, shared / Path(relative).name
            if not target.exists() and original.is_file() and file_hash(original) == expected:
                shutil.copyfile(original, target)
    # A historical install may already have the authorized IIT template. Its
    # known hash is sufficient; do not require or download the old HDF5 group.
    if not template.exists():
        root = configured_model_assets_root(asset_root)
        for folder in ("pnl_cnn_torch", "pnl_cnn"):
            original = root / "brain_mask" / folder / template.name
            if original.is_file() and file_hash(original) == ASSETS[template.name][1]:
                shutil.copyfile(original, template)
                break
    _run([str(prefix / "bin/python"), "-m", "pnl_brain_mask", "fetch",
          "--model-dir", str(models), "--accept-terms",
          *([] if template.is_file() and file_hash(template) == ASSETS[template.name][1] else ["--template"])])


def verify_assets(prefix, *, asset_root=None):
    from shared.managed_runtimes import file_hash
    import json

    models = prefix / "pnl-models"
    for relative, expected in RESOURCE_HASHES.items():
        resource = prefix / relative
        if resource.is_symlink() or not resource.is_file():
            raise RuntimeError("PNL Torch runtime resources are incomplete: " + relative)
        if file_hash(resource) != expected:
            raise RuntimeError("PNL resource checksum changed: " + relative)
    script = prefix / "bin/pnl-brain-mask"
    if not script.is_file():
        raise RuntimeError("PNL Torch console entry point is missing")
    probe = Path(__file__).with_name("pnl_runtime_probe.py")
    result = subprocess.run([str(prefix / "bin/python"), str(probe), str(models)], check=True,
                            capture_output=True, text=True,
                            env=pnl_python_environment(prefix / "bin/python"), timeout=180)
    details = json.loads(result.stdout)
    return {"external_dependencies": {"pnl_mask_python": str(prefix / "bin/python"),
            "pnl_mask_script": str(script), "pnl_mask_model_dir": str(models),
            "pnl_mask_ants_bin_dir": str(prefix / "bin")},
            "pnl_resource_sha256": {**{relative: file_hash(prefix / relative) for relative in RESOURCE_HASHES},
                                    **{"package/" + name: digest for name, digest in details["package_source_sha256"].items()}},
            "pnl_source_commit": PNL_COMMIT, "pnl_torch_commit": PNL_TORCH_COMMIT,
            "pnl_framework": "pytorch", "pnl_runtime_probe": details}
