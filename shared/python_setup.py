"""Version-aware Python installation shared by old and new Slicer runtimes."""

import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile

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


def install_requirements(requirements, *, protected=True, loaded=(), torch_minimum=None, torchvision_required=False):
    """Called only by the explicitly authorized, cancellable setup worker."""
    requirements = tuple(dict.fromkeys(requirements))
    before = installed_versions()
    backend = "already_satisfied"
    if torch_minimum and (
        not requirement_satisfied("torch>=" + torch_minimum)
        or (torchvision_required and not requirement_satisfied("torchvision"))
    ):
        try:
            PyTorchUtils = _import_pytorch_utils()
        except ModuleNotFoundError:
            PyTorchUtils = None
        if PyTorchUtils is not None:
            logic = PyTorchUtils.PyTorchUtilsLogic()
            torch = logic.installTorch(askConfirmation=False, torchVersionRequirement=">=" + torch_minimum)
            if (torch is None or not requirement_satisfied("torch>=" + torch_minimum)
                    or (torchvision_required and not requirement_satisfied("torchvision"))):
                raise RuntimeError("PyTorch extension could not prepare a compatible torch build")
            backend = "PyTorch_extension"
    missing = [item for item in requirements if not requirement_satisfied(item)]
    if missing:
        with tempfile.TemporaryDirectory(prefix="sdds-pip-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            pinned = installed_versions() if protected else {}
            _, _, canonicalize_name = _packaging_api()
            constraints.write_text("".join(f"{name}=={pinned[canonicalize_name(name)]}\n"
                                           for name in PROTECTED_PACKAGES if canonicalize_name(name) in pinned))
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
                                "--", *missing], check=True)
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
