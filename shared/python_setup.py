"""Version-aware Python installation shared by old and new Slicer runtimes."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import re
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


_CUDA_BACKEND_SPEC_RE = re.compile(r"^cu(?:da)?(?P<version>\d+(?:\.\d+)?)$", re.IGNORECASE)
_BLACKWELL_GPU_RE = re.compile(r"(\brtx\s*(?:pro\s*)?50\d\d\b|\bblackwell\b|\bg?b[12]00\b)", re.IGNORECASE)
_PYTORCH_CUDA_WHEEL_FLAVORS = ("cu128", "cu126", "cu124", "cu121", "cu118")
_TORCH_FAMILY_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})


def _parse_cuda_backend_spec(value):
    """Parse a versioned CUDA backend (e.g. ``'cu124'``, ``CUDABackend(12, 8)``) into ``((major, minor), 'cuXY')``."""
    raw = getattr(value, "local_specifier", value)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    match = _CUDA_BACKEND_SPEC_RE.match(text)
    if match is None:
        return None
    ver = match.group("version")
    if "." in ver:
        parts = ver.split(".", 1)
        if not (parts[0].isdigit() and parts[1].isdigit()):
            return None
        major, minor = int(parts[0]), int(parts[1])
    else:
        if len(ver) < 2 or not ver.isdigit():
            return None
        major, minor = int(ver[:-1]), int(ver[-1])
    if major <= 0:
        return None
    return (major, minor), f"cu{major}{minor}"


def _filter_modern_cuda_backends(parsed_items, *, blackwell_required=False):
    dedup = {}
    for ver_tuple, spec in parsed_items:
        dedup[ver_tuple] = spec
    ordered = sorted(dedup.items())
    if not ordered:
        return []
    if blackwell_required:
        bw = [spec for ver, spec in ordered if ver >= (12, 8)]
        if bw:
            return bw
    modern = [spec for ver, spec in ordered if ver >= (11, 8)]
    return modern if modern else [spec for _, spec in ordered]


def _query_nvidia_smi_driver_and_gpus():
    """Return ``(driver_tuple_or_None, blackwell_detected)`` from ``nvidia-smi``, best-effort."""
    import shutil

    exe = shutil.which("nvidia-smi")
    if not exe:
        try:
            from shared.hardware_adapt import find_nvidia_smi

            exe = find_nvidia_smi()
        except Exception:
            exe = None
    if not exe:
        return None, False
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            **subprocess_no_window_kwargs(),
        )
    except Exception:
        return None, False
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, False
    driver_tuple = None
    blackwell = False
    for line in completed.stdout.splitlines():
        parts = [cell.strip() for cell in line.split(",")]
        if not parts or not parts[0]:
            continue
        if driver_tuple is None:
            nums = [int(tok) for tok in re.findall(r"\d+", parts[0])[:3]]
            if nums:
                driver_tuple = tuple(nums + [0] * max(0, 2 - len(nums)))
        gpu_name = parts[1] if len(parts) > 1 else ""
        if gpu_name and _BLACKWELL_GPU_RE.search(gpu_name):
            blackwell = True
        cap_text = parts[2] if len(parts) > 2 else ""
        if "." in cap_text:
            try:
                if int(cap_text.split(".", 1)[0].strip()) >= 10:
                    blackwell = True
            except ValueError:
                pass
    return driver_tuple, blackwell


def _detect_host_cuda_backends():
    """Return compatible versioned CUDA backends (e.g. ``['cu118', 'cu124', 'cu126', 'cu128']``) for this host."""
    driver_tuple, blackwell = _query_nvidia_smi_driver_and_gpus()
    parsed = []
    try:
        import light_the_torch._cb as ltt_cb

        detect_fn = getattr(ltt_cb, "_detect_compatible_cuda_backends", None)
        if callable(detect_fn):
            for item in detect_fn() or ():
                spec = _parse_cuda_backend_spec(item)
                if spec is not None:
                    parsed.append(spec)
    except Exception:
        pass
    if not parsed and driver_tuple is not None:
        is_win = sys.platform == "win32"
        min_cu12 = (528, 33) if is_win else (525, 60)
        min_cu118 = (452, 39) if is_win else (450, 80)
        if driver_tuple[:2] >= min_cu12:
            parsed.extend([
                ((11, 8), "cu118"),
                ((12, 1), "cu121"),
                ((12, 4), "cu124"),
                ((12, 6), "cu126"),
                ((12, 8), "cu128"),
            ])
        elif driver_tuple[:2] >= min_cu118:
            parsed.append(((11, 8), "cu118"))
    return _filter_modern_cuda_backends(parsed, blackwell_required=blackwell)


def _resolve_pytorch_utils_cuda_backend(logic, torch_minimum):
    """Resolve a concrete versioned CUDA backend string for ``PyTorchUtilsLogic.installTorch``.

    ``Slicer/light-the-torch`` rejects bare ``'cu'`` before wheel resolution; always pass one or
    more concrete versioned CUDA backend specifiers (e.g. ``'cu124'`` or ``'cu118,cu124,cu126,cu128'``).
    """
    _, blackwell = _query_nvidia_smi_driver_and_gpus()
    plural_fn = getattr(logic, "getCompatibleComputationBackends", None)
    if callable(plural_fn):
        try:
            raw_backends = plural_fn(torchVersionRequirement=">=" + torch_minimum)
        except TypeError:
            raw_backends = plural_fn()
        parsed = []
        for item in raw_backends or ():
            spec = _parse_cuda_backend_spec(item)
            if spec is not None:
                parsed.append(spec)
        filtered = _filter_modern_cuda_backends(parsed, blackwell_required=blackwell)
        if not filtered:
            raise RuntimeError(
                "No CUDA-compatible NVIDIA GPU/driver was detected by Slicer PyTorch (light-the-torch); "
                "install or update the NVIDIA driver, or select CPU."
            )
        return ",".join(filtered)

    single_fn = getattr(logic, "getCompatibleComputationBackend", None)
    if callable(single_fn):
        parsed_single = _parse_cuda_backend_spec(single_fn())
        if parsed_single is not None:
            return parsed_single[1]

    detected = _detect_host_cuda_backends()
    if detected:
        return ",".join(detected)
    raise RuntimeError(
        "Could not determine a compatible versioned CUDA computation backend (e.g., cu124, cu126, cu128) "
        "for PyTorch installation; check your NVIDIA driver or select CPU."
    )


def _select_pytorch_wheel_index_flavor(detected_backends):
    available = set(detected_backends or ())
    for flavor in _PYTORCH_CUDA_WHEEL_FLAVORS:
        if flavor in available:
            return flavor
    return detected_backends[-1] if detected_backends else None


def _requirement_canonical_name(value):
    InvalidRequirement, Requirement, canonicalize_name = _packaging_api()
    try:
        return canonicalize_name(Requirement(str(value)).name)
    except InvalidRequirement:
        return str(value).strip().lower()


def _windows_safe_torch_requirements(requirements):
    """Apply Windows MAX_PATH exclusions (matching upstream SlicerPyTorch) to torch/torchvision specs."""
    if sys.platform != "win32":
        return list(requirements)
    exclusions = {"torch": "!=2.13.*", "torchvision": "!=0.28.*"}
    adjusted = []
    for raw in requirements:
        text = str(raw).strip()
        canon = _requirement_canonical_name(text)
        excl = exclusions.get(canon)
        if excl and excl not in text:
            if any(ch in text for ch in "<>=!~"):
                text = f"{text},{excl}"
            else:
                text = f"{text}{excl}"
        adjusted.append(text)
    return adjusted


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
    direct_cuda_wheel=False,
    cuda_index_url=None,
):
    """Called only by the explicitly authorized, cancellable setup worker."""
    requirements = tuple(dict.fromkeys(requirements))
    before = installed_versions()
    backend = "already_satisfied"
    want_cuda_torch = bool(require_cuda_torch or replace_cpu_torch or direct_cuda_wheel)
    need_cuda_torch = want_cuda_torch and not _torch_cuda_build_satisfied()
    if not direct_cuda_wheel and torch_minimum and (
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
            uninstall_before_reinstall = False
            if need_cuda_torch:
                import inspect

                try:
                    params = inspect.signature(logic.installTorch).parameters
                except (TypeError, ValueError):
                    params = {}
                if "forceComputationBackend" in params:
                    install_kwargs["forceComputationBackend"] = _resolve_pytorch_utils_cuda_backend(
                        logic, torch_minimum
                    )
                if "forceReinstall" in params:
                    install_kwargs["forceReinstall"] = True
                elif "force" in params:
                    install_kwargs["force"] = True
                elif "torch" in before and callable(getattr(logic, "uninstallTorch", None)):
                    # Upstream SlicerPyTorch installTorch delegates to light-the-torch without
                    # --force-reinstall; uninstall CPU torch first so pip replaces it with CUDA.
                    uninstall_before_reinstall = True
            if uninstall_before_reinstall:
                logic.uninstallTorch()
                importlib.invalidate_caches()
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
    cuda_torch_pending = want_cuda_torch and not _torch_cuda_build_satisfied()
    missing = [
        item
        for item in requirements
        if (not requirement_satisfied(item))
        or (cuda_torch_pending and _requirement_canonical_name(item) in _TORCH_FAMILY_PACKAGES)
    ]
    if cuda_torch_pending and not any(_requirement_canonical_name(item) == "torch" for item in missing):
        missing.insert(0, "torch>=" + torch_minimum if torch_minimum else "torch")
    if cuda_torch_pending and torchvision_required and not any(
        _requirement_canonical_name(item) == "torchvision" for item in missing
    ):
        missing.append("torchvision")
    if missing:
        torch_missing = (
            [item for item in missing if _requirement_canonical_name(item) in _TORCH_FAMILY_PACKAGES]
            if cuda_torch_pending
            else []
        )
        other_missing = [item for item in missing if item not in torch_missing]
        # D5: check before pip so no long download ends in a failed post-check.
        # Direct CUDA wheel installation into Slicer's site-packages supersedes bundled CPU torch on restart.
        shortfall_targets = other_missing if direct_cuda_wheel else missing
        if shortfall_targets:
            _refuse_bundled_shortfalls(shortfall_targets)
        with tempfile.TemporaryDirectory(prefix="sdds-pip-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            pinned = installed_versions() if protected else {}
            _, _, canonicalize_name = _packaging_api()
            unpinned = set(_TORCH_FAMILY_PACKAGES) if cuda_torch_pending else set()
            constraints.write_text(
                "".join(
                    f"{name}=={pinned[canonicalize_name(name)]}\n"
                    for name in PROTECTED_PACKAGES
                    if canonicalize_name(name) in pinned and canonicalize_name(name) not in unpinned
                ),
                encoding="utf-8",
            )
            if torch_missing:
                explicit_index = str(cuda_index_url or "").strip()
                if explicit_index:
                    index_url = explicit_index
                else:
                    detected_cuda = _detect_host_cuda_backends()
                    flavor = _select_pytorch_wheel_index_flavor(detected_cuda)
                    if not flavor:
                        raise RuntimeError(
                            "CUDA-enabled PyTorch was requested for this Python environment, "
                            "but no compatible NVIDIA GPU/driver was detected."
                        )
                    index_url = f"https://download.pytorch.org/whl/{flavor}"
                backend = "python_subprocess_cuda_wheel"
                if protected:
                    # Inside Slicer's Python, avoid --force-reinstall (which would reinstall Slicer's
                    # numpy/pillow) and disable pip's wheel cache (~2.7 GB savings). Uninstall any
                    # pre-existing torch/torchvision in Slicer's site-packages first, then install
                    # strictly from the CUDA index (never passing --extra-index-url PyPI).
                    existing_torch = [pkg for pkg in ("torch", "torchvision", "torchaudio") if pkg in before]
                    if existing_torch:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "uninstall", "-y", *existing_torch],
                            check=False,
                            **subprocess_no_window_kwargs(),
                        )
                    install_torch_reqs = _windows_safe_torch_requirements(torch_missing)
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-cache-dir",
                            "--upgrade-strategy",
                            "only-if-needed",
                            "--index-url",
                            index_url,
                            "--constraint",
                            str(constraints),
                            "--",
                            *install_torch_reqs,
                        ],
                        check=True,
                        **subprocess_no_window_kwargs(),
                    )
                else:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--force-reinstall",
                            "--index-url",
                            index_url,
                            "--constraint",
                            str(constraints),
                            "--",
                            *torch_missing,
                        ],
                        check=True,
                        **subprocess_no_window_kwargs(),
                    )
            if other_missing:
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
                    for value in other_missing:
                        try:
                            named = named and not Requirement(value).url
                        except InvalidRequirement:
                            named = False
                    if named:
                        packaging.pip_ensure(other_missing, constraints=str(constraints), prompt_install=False,
                                             prompt_restart=False, skip_in_testing=False)
                    else:
                        # pip_ensure parses names as PEP 508 requirements and cannot
                        # accept bare VCS URLs/local projects used by WMQL adapters.
                        packaging.pip_install(other_missing, constraints=str(constraints), show_progress=False)
                else:
                    if backend == "already_satisfied":
                        backend = "python_subprocess_compatibility"
                    subprocess.run([sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
                                    "--", *other_missing], check=True, **subprocess_no_window_kwargs())
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
    if want_cuda_torch and not _torch_cuda_build_satisfied():
        raise RuntimeError(
            "CUDA-enabled PyTorch was requested, but the installed torch distribution is still CPU-only or lacks CUDA support."
        )
    changed = restart_packages(before, after, loaded)
    restart_needed = bool(changed)
    if direct_cuda_wheel and backend == "python_subprocess_cuda_wheel":
        restart_needed = True
        if "torch" not in changed:
            changed = sorted({*changed, "torch"})
    return {"kind": "slicerdsurfer_python_setup", "status": "completed", "backend": backend,
            "requirements": list(requirements), "installed_before": before, "installed_after": after,
            "restart_required": restart_needed, "restart_packages": changed}
