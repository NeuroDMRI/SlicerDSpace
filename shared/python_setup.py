"""Version-aware Python installation shared by old and new Slicer runtimes."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import site
import subprocess
import sys
import sysconfig
import tempfile

from shared.external_process import subprocess_no_window_kwargs

PROTECTED_PACKAGES = ("numpy", "scipy", "vtk", "SimpleITK", "torch", "torchvision", "requests")


def _packaging_api():
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("packaging"):
            raise
        # A fresh external venv may have only pip, not a separate packaging
        # distribution. pip already bundles the parser needed to bootstrap it.
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    return InvalidRequirement, Requirement, canonicalize_name


def installed_versions():
    _, _, canonicalize_name = _packaging_api()
    return {canonicalize_name(item.metadata["Name"]): item.version
            for item in importlib.metadata.distributions() if item.metadata.get("Name")}


def requirement_satisfied(value):
    InvalidRequirement, Requirement, _ = _packaging_api()
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        return False
    if requirement.marker and not requirement.marker.evaluate():
        return True
    try:
        return not requirement.url and importlib.metadata.version(requirement.name) in requirement.specifier
    except importlib.metadata.PackageNotFoundError:
        return False


def _norm_path(value):
    return os.path.normcase(os.path.realpath(str(value)))


def _bundle_dirs_ahead_of_slicer(bundle_dirs):
    """P-f: the bundle dirs that come before Slicer's own site-packages on ``sys.path``.

    Only such a bundle shadows a package that pip installs into Slicer's site-packages. A bundle that is appended
    after it (Windows, where the bundle has no numpy) loses to the pip install, so no refusal applies. A bundle
    that is not on ``sys.path`` at all is not active.
    """
    order = {}
    for index, entry in enumerate(sys.path):
        if entry:
            order.setdefault(_norm_path(entry), index)
    slicer_dirs = set()
    try:
        slicer_dirs.update(_norm_path(item) for item in site.getsitepackages())
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            slicer_dirs.add(_norm_path(sysconfig.get_paths()[key]))
        except Exception:
            pass
    slicer_index = min((order[item] for item in slicer_dirs if item in order), default=None)
    ahead = []
    for item in bundle_dirs:
        index = order.get(_norm_path(item))
        if index is None:
            continue
        if slicer_index is None or index < slicer_index:
            ahead.append(item)
    return ahead


def bundled_shortfalls(requirements):
    """D5: messages for requests that the bundled copy of a package cannot satisfy.

    The check judges satisfaction from the first-found copy (the bundle, which comes first on ``sys.path``),
    but pip installs into Slicer's site-packages, where the bundled copy still wins. A request above the
    bundled version can therefore never pass; pip would download it and the post-check would then fail.
    """
    try:
        from shared.bundled_python import bundled_site_packages_dirs

        bundle_dirs = _bundle_dirs_ahead_of_slicer(str(path) for path in bundled_site_packages_dirs())
    except Exception:
        return []
    if not bundle_dirs:
        return []
    InvalidRequirement, Requirement, canonicalize_name = _packaging_api()
    bundled = {}
    for dist in importlib.metadata.distributions(path=bundle_dirs):
        name = dist.metadata.get("Name") if dist.metadata else None
        if name:
            bundled.setdefault(canonicalize_name(name), dist.version)
    messages = []
    for value in requirements:
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            continue
        if requirement.url or (requirement.marker and not requirement.marker.evaluate()):
            continue
        version = bundled.get(canonicalize_name(requirement.name))
        if version is None or requirement.specifier.contains(version, prereleases=True):
            continue
        messages.append(
            f"bundled {requirement.name} {version} is older than required ({value}); "
            "reinstall the SlicerDSpace package")
    return messages


def _refuse_bundled_shortfalls(requirements):
    shortfalls = bundled_shortfalls(requirements)
    if shortfalls:
        raise RuntimeError("; ".join(shortfalls))


def restart_packages(before, after, loaded):
    _, _, canonicalize_name = _packaging_api()
    return sorted({canonicalize_name(name) for name in loaded
                   if before.get(canonicalize_name(name)) != after.get(canonicalize_name(name))})


def _import_pytorch_utils():
    if "PyTorchUtils" not in sys.modules:
        from shared.external_process import find_pythonslicer
        exe = Path(sys.executable).resolve()
        for parent in tuple(exe.parents)[:5]:
            for pattern in ("Extensions-*/PyTorch/lib/Slicer-*/qt-scripted-modules",
                            "slicer.org/Extensions-*/PyTorch/lib/Slicer-*/qt-scripted-modules"):
                for candidate in sorted(parent.glob(pattern)):
                    if (candidate / "PyTorchUtils.py").is_file() and str(candidate) not in sys.path:
                        sys.path.append(str(candidate))
        try:
            import slicer
            import slicer.util
            if not hasattr(slicer, "app") and not (Path(sys.executable).parent / "PythonSlicer").is_file():
                launcher = next((find_pythonslicer(p) for p in tuple(exe.parents)[:5] if find_pythonslicer(p)), None)
                if launcher is not None:
                    def _macos_execute_python_module(module, args, _launcher=str(launcher), _util=slicer.util):
                        if sys.platform == "darwin" and module == "light_the_torch":
                            module = "pip"
                            args = [a for a in args if not str(a).startswith("--pytorch-computation-backend")]
                        proc = _util.launchConsoleProcess([_launcher, "-m", module, *args], useStartupEnvironment=False)
                        _util.logProcessOutput(proc)
                    slicer.util._executePythonModule = _macos_execute_python_module
        except Exception:
            pass
    import PyTorchUtils
    return PyTorchUtils


def _torch_cuda_build_satisfied() -> bool:
    """Return True when the installed torch distribution is a CUDA-enabled build."""
    try:
        version = str(importlib.metadata.version("torch") or "").strip().lower()
    except importlib.metadata.PackageNotFoundError:
        return False
    if not version or "+cpu" in version:
        return False
    if "+cu" in version:
        return True
    try:
        import torch

        return bool(getattr(getattr(torch, "version", None), "cuda", None))
    except Exception:
        return False


def install_requirements(
    requirements,
    *,
    protected=True,
    loaded=(),
    torch_minimum=None,
    torchvision_required=False,
    require_cuda_torch=False,
    replace_cpu_torch=False,
):
    """Called only by the explicitly authorized, cancellable setup worker."""
    requirements = tuple(dict.fromkeys(requirements))
    before = installed_versions()
    backend = "already_satisfied"
    need_cuda_torch = bool(require_cuda_torch or replace_cpu_torch) and not _torch_cuda_build_satisfied()
    if torch_minimum and (
        not requirement_satisfied("torch>=" + torch_minimum)
        or (torchvision_required and not requirement_satisfied("torchvision"))
        or need_cuda_torch
    ):
        _refuse_bundled_shortfalls(["torch>=" + torch_minimum])  # D5: the PyTorch extension installs outside the bundle
        try:
            PyTorchUtils = _import_pytorch_utils()
        except ModuleNotFoundError:
            PyTorchUtils = None
        if PyTorchUtils is not None:
            logic = PyTorchUtils.PyTorchUtilsLogic()
            install_kwargs = {"askConfirmation": False, "torchVersionRequirement": ">=" + torch_minimum}
            if need_cuda_torch:
                import inspect

                try:
                    params = inspect.signature(logic.installTorch).parameters
                except (TypeError, ValueError):
                    params = {}
                if "forceComputationBackend" in params:
                    backend_choice = "cu"
                    compat_fn = getattr(logic, "getCompatibleComputationBackend", None)
                    if callable(compat_fn):
                        try:
                            compat = compat_fn()
                            if isinstance(compat, str) and compat.strip().lower().startswith("cu"):
                                backend_choice = compat.strip()
                        except Exception:
                            pass
                    install_kwargs["forceComputationBackend"] = backend_choice
                if "forceReinstall" in params:
                    install_kwargs["forceReinstall"] = True
                elif "force" in params:
                    install_kwargs["force"] = True
            torch = logic.installTorch(**install_kwargs)
            importlib.invalidate_caches()
            if (
                torch is None
                or not requirement_satisfied("torch>=" + torch_minimum)
                or (torchvision_required and not requirement_satisfied("torchvision"))
                or (need_cuda_torch and not _torch_cuda_build_satisfied())
            ):
                raise RuntimeError(
                    "PyTorch extension could not prepare a compatible CUDA torch build"
                    if need_cuda_torch
                    else "PyTorch extension could not prepare a compatible torch build"
                )
            backend = "PyTorch_extension"
        elif need_cuda_torch:
            raise RuntimeError(
                "CPU-only PyTorch is installed and the Slicer PyTorch extension helper (PyTorchUtils) is unavailable; "
                "install a CUDA-enabled PyTorch wheel matching your NVIDIA driver."
            )
    missing = [
        item
        for item in requirements
        if (not requirement_satisfied(item))
        or (
            need_cuda_torch
            and (str(item).strip().lower() == "torch" or str(item).strip().lower().startswith(("torch>", "torch=", "torch<")))
        )
    ]
    if missing:
        _refuse_bundled_shortfalls(missing)  # D5: before pip, so no long download ends in a failed post-check
        with tempfile.TemporaryDirectory(prefix="sdds-pip-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            pinned = installed_versions() if protected else {}
            _, _, canonicalize_name = _packaging_api()
            unpinned = {"torch", "torchvision"} if need_cuda_torch else set()
            constraints.write_text(
                "".join(
                    f"{name}=={pinned[canonicalize_name(name)]}\n"
                    for name in PROTECTED_PACKAGES
                    if canonicalize_name(name) in pinned and canonicalize_name(name) not in unpinned
                ),
                encoding="utf-8",
            )
            try:
                packaging = importlib.import_module("slicer.packaging")
            except ModuleNotFoundError as exc:
                if exc.name not in {"slicer", "slicer.packaging"}:
                    raise
                packaging = None
            if packaging is not None and callable(getattr(packaging, "pip_ensure", None)):
                backend = "slicer.packaging"
                InvalidRequirement, Requirement, _ = _packaging_api()
                named = True
                for value in missing:
                    try:
                        named = named and not Requirement(value).url
                    except InvalidRequirement:
                        named = False
                if named:
                    packaging.pip_ensure(missing, constraints=str(constraints), prompt_install=False,
                                         prompt_restart=False, skip_in_testing=False)
                else:
                    # pip_ensure parses names as PEP 508 requirements and cannot
                    # accept bare VCS URLs/local projects used by WMQL adapters.
                    packaging.pip_install(missing, constraints=str(constraints), show_progress=False)
            else:
                backend = "python_subprocess_compatibility"
                subprocess.run([sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
                                "--", *missing], check=True, **subprocess_no_window_kwargs())
    importlib.invalidate_caches()
    after = installed_versions()
    # PEP 508 names have an exact metadata check; direct source URLs are checked
    # by pip's exit status, and the final environment doctor checks imports.
    InvalidRequirement, Requirement, _ = _packaging_api()
    for value in requirements:
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            continue
        if not requirement.url and not requirement_satisfied(value):
            raise RuntimeError("Installed package does not satisfy " + value)
    changed = restart_packages(before, after, loaded)
    return {"kind": "slicerdsurfer_python_setup", "status": "completed", "backend": backend,
            "requirements": list(requirements), "installed_before": before, "installed_after": after,
            "restart_required": bool(changed), "restart_packages": changed}
