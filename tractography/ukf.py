"""Invoke the installed official UKFTractography CLI, never a reimplementation."""

from dataclasses import asdict, dataclass, fields, replace
from shared.cohort_schedule import scheduled_stage
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


REFERENCE = "https://github.com/pnlbwh/ukftractography/blob/2d2b661539700aa825c1949823119601e11e1823/UKFTractography/UKFTractography.xml"
MASK_POLICY = "native-mask-intersect-positive-mean-b0-v1"
# A conservative input-integrity guard, not an anatomical-accuracy threshold.
MAX_EXCLUDED_MASK_FRACTION = 0.01
# Gradient frame (batch 2). DWIConvert FSLToNrrd writes world-correct gradients only for axis-aligned grids with
# det(A) < 0; these keep the converter's lines verbatim and the v1 policy, so finished runs stay valid. Every other
# grid gets its gradient directions from the bvec file by the FSL rule and the v2 policy, so old runs are redone.
CONVERSION_POLICY_DWICONVERT = "official-gradient-geometry-probe-with-lossless-native-payload-v1"
CONVERSION_POLICY_FSL_WORLD = "official-geometry-probe-lossless-payload-fsl-rule-gradients-v2"
INPUT_FRAME_DWICONVERT = "NIfTI RAS+, gradients converted by official DWIConvert"
INPUT_FRAME_FSL_WORLD = ("NIfTI RAS+; geometry from official DWIConvert; gradient directions from the bvec file by "
                         "the FSL rule g = R F b (MRtrix -fslgrad), DWIConvert b-value scaling kept")


@dataclass(frozen=True)
class UKFParameters:
    # Explicit author-module defaults, not a subject-tuned ORG protocol.
    numTensor: int = 2
    seedsPerVoxel: int = 1
    seedingThreshold: float = 0.18
    stoppingFA: float = 0.15
    stoppingThreshold: float = 0.1
    stepLength: float = 0.3
    recordLength: float = 0.9
    maxHalfFiberLength: float = 250.0
    numThreads: int = -1
    freeWater: bool = False
    recordFA: bool = True
    recordTensors: bool = True
    Qm: float = 0.0
    Ql: float = 0.0
    Qw: float = 0.0
    Rs: float = 0.0


GPU_OPTION_KEYS = ("useGPU", "gpuDevice", "gpuPrecision")
_GPU_EXECUTABLES = ("ukf-gpu", "UKFTractography_gpu.exe")


def parse_gpu_options(values=None):
    """Extract and validate optional NeuroDMRI/ukftractography_gpu CLI controls."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("UKF parameters must be a JSON object")
    gpu_options = {}
    if "useGPU" in values:
        if not isinstance(values["useGPU"], bool):
            raise ValueError("UKF useGPU must be boolean")
        gpu_options["useGPU"] = values["useGPU"]
    if "gpuDevice" in values:
        dev = values["gpuDevice"]
        if isinstance(dev, bool) or not isinstance(dev, int) or dev < -1:
            raise ValueError("UKF gpuDevice must be an integer >= -1")
        gpu_options["gpuDevice"] = dev
    if "gpuPrecision" in values:
        prec = values["gpuPrecision"]
        if isinstance(prec, bool) or not isinstance(prec, int) or prec not in (0, 1):
            raise ValueError("UKF gpuPrecision must be 0 (FP64) or 1 (guarded mixed precision)")
        gpu_options["gpuPrecision"] = prec
    return gpu_options


def parameters(values=None):
    if values is not None and not isinstance(values, dict):
        raise ValueError("UKF parameters must be a JSON object")
    parse_gpu_options(values)
    cpu_values = {k: v for k, v in (values or {}).items() if k not in GPU_OPTION_KEYS}
    unknown = set(cpu_values) - {field.name for field in fields(UKFParameters)}
    if unknown:
        raise ValueError("Unknown UKF parameters: " + ", ".join(sorted(unknown)))
    result = UKFParameters(**cpu_values)
    for key, value in asdict(result).items():
        if key in ("freeWater", "recordFA", "recordTensors"):
            if not isinstance(value, bool):
                raise ValueError(key + " must be boolean")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be finite numeric")
    if result.numTensor not in (1, 2) or type(result.numTensor) is not int:
        raise ValueError("UKF numTensor must be integer 1 or 2")
    if type(result.seedsPerVoxel) is not int or not 1 <= result.seedsPerVoxel <= 50:
        raise ValueError("UKF seedsPerVoxel must be integer in [1,50]")
    if type(result.numThreads) is not int or result.numThreads != -1 and result.numThreads < 1:
        raise ValueError("UKF numThreads must be -1 (author automatic) or positive integer")
    if not all(0 <= getattr(result, key) <= 1 for key in ("seedingThreshold", "stoppingFA", "stoppingThreshold")):
        raise ValueError("UKF thresholds must be in [0,1]")
    if (
        not 0.1 <= result.stepLength <= 1
        or not 0.1 <= result.recordLength <= 4
        or not 1 <= result.maxHalfFiberLength <= 500
    ):
        raise ValueError("UKF step/output lengths exceed documented author limits")
    for key, maximum in (("Qm", 0.25), ("Ql", 1000), ("Qw", 0.25), ("Rs", 0.25)):
        if not 0 <= getattr(result, key) <= maximum:
            raise ValueError(f"UKF {key} must be in [0,{maximum}]; zero selects author automatic noise")
    if not result.recordFA or not result.recordTensors:
        raise ValueError("UKF output must retain the author-default FA and tensor arrays")
    if os.environ.get('SLICERDSPACE_COHORT_SCHEDULE'):
        budget = int(os.environ['SLICERDSPACE_STAGE_CPUS'])
        if result.numThreads == -1:
            result = replace(result, numThreads=budget)
        elif result.numThreads > budget:
            raise ValueError('UKF thread request exceeds the admitted cohort CPU reservation')
    return result


def _extension_search_roots():
    roots = []
    module_root = Path(__file__).resolve().parents[1]
    if module_root.name == "qt-scripted-modules" and len(module_root.parents) >= 3:
        roots.append(module_root.parents[2])
    roots.append(module_root)
    roots.extend(Path(value) for key in ("SLICER_HOME", "Slicer_HOME") if (value := os.environ.get(key)))
    roots.append(Path(sys.executable).resolve().parent.parent)
    return roots


# CP-1: a UKF CLI built against another Slicer's VTK cannot load. Scan the CLI and its
# UKFBase library for versioned VTK library names (vtkcommon-9.5.dll, libvtkCommonCore-9.5.so).
_VTK_VERSIONED_NAME = re.compile(rb"vtk[A-Za-z]*-(\d+)\.(\d+)")
_VTK_SCAN_CACHE = {}


def _running_vtk_version():
    """(major, minor) of the VTK in this Slicer, or None when it cannot be read."""
    try:
        import vtk

        return int(vtk.vtkVersion.GetVTKMajorVersion()), int(vtk.vtkVersion.GetVTKMinorVersion())
    except Exception:
        return None


def _scan_vtk_versions(path):
    versions = set()
    tail = b""
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            data = tail + chunk
            for match in _VTK_VERSIONED_NAME.finditer(data):
                versions.add((int(match.group(1)), int(match.group(2))))
            tail = data[-128:]
    return versions


def _ukf_vtk_versions(executable):
    """VTK (major, minor) versions named by the CLI and its sibling UKFBase library (cached per path)."""
    executable = Path(executable)
    candidates = [executable]
    for directory in (executable.parent, executable.parent.parent):
        try:
            for sibling in sorted(directory.glob("*UKFBase*")):
                name = sibling.name.lower()
                if sibling.is_file() and (name.endswith((".dll", ".dylib", ".so")) or ".so." in name):
                    candidates.append(sibling)
        except OSError:
            continue
    versions = set()
    for path in candidates:
        try:
            info = path.stat()
            key = (str(path), info.st_mtime_ns, info.st_size)
            if key not in _VTK_SCAN_CACHE:
                _VTK_SCAN_CACHE[key] = frozenset(_scan_vtk_versions(path))
            versions |= _VTK_SCAN_CACHE[key]
        except OSError:
            continue
    return versions


def ukf_vtk_mismatch(executable):
    """Return (found, running) version strings when a Slicer-built UKF CLI targets another VTK, else None.

    Only a CLI inside a cli-modules folder is checked. A standalone runtime, an unreadable
    file, a file with no versioned VTK name or an unreadable running VTK never blocks.
    """
    executable = Path(executable)
    if executable.parent.name != "cli-modules":
        return None
    running = _running_vtk_version()
    if running is None:
        return None
    found = _ukf_vtk_versions(executable)
    if not found or running in found:
        return None
    return (
        ", ".join("%d.%d" % version for version in sorted(found)),
        "%d.%d" % running,
    )


def _ukf_vtk_error(rejected):
    lines = [
        "UKFTractography in this package was built for a different 3D Slicer version (its VTK differs from the "
        "running 3D Slicer, so the CLI cannot load). Install UKFTractography for this 3D Slicer from the Extension "
        "Manager, use 3D Slicer 5.10, or use a SlicerDSpace package rebuilt for this version."
    ]
    for path, (found, running) in rejected:
        lines.append("  %s links VTK %s; running Slicer uses VTK %s" % (path, found, running))
    return "\n".join(lines)


def _host_supports_avx2():
    """Return True/False when x86_64 AVX2 capability can be probed, or None on non-x86 / unknown hosts."""
    import platform

    machine = (platform.machine() or "").strip().lower()
    if machine in {"arm64", "aarch64"}:
        return None
    if sys.platform == "win32":
        try:
            import ctypes

            PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.IsProcessorFeaturePresent.argtypes = [ctypes.c_uint32]
            kernel32.IsProcessorFeaturePresent.restype = ctypes.c_bool
            return bool(kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            for line in cpuinfo.splitlines():
                if line.lower().startswith("flags") and ":" in line:
                    flags = set(line.split(":", 1)[1].strip().lower().split())
                    return "avx2" in flags
        except OSError:
            return None
        return None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.optional.avx2_0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                val = out.stdout.strip()
                if val in {"0", "1"}:
                    return val == "1"
        except Exception:
            return None
    return None


def _standalone_requires_avx2(executable):
    """True when a bundled standalone UKF binary was compiled with AVX2 instructions."""
    path = Path(executable)
    for runtime_json in (path.parent.parent / "runtime.json", path.parent / "runtime.json"):
        try:
            if runtime_json.is_file():
                payload = json.loads(runtime_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    if isinstance(payload.get("requires_avx2"), bool):
                        return payload["requires_avx2"]
                    arch = str(payload.get("arch") or "").strip().lower()
                    if arch in {"arm64", "aarch64"}:
                        return False
                    if arch in {"x86_64", "amd64", "x64"}:
                        return True
        except Exception:
            pass
    parts = [part.lower() for part in path.parts]
    if any(("arm64" in part or "aarch64" in part) for part in parts if part.startswith("ukf-")):
        return False
    if any(
        part in {"ukf-windows-x64", "ukf-win-x64", "ukf-linux-x64", "ukf-linux", "ukf-win32"}
        for part in parts
    ):
        return True
    return False


def _ukf_avx2_incompatible(executable):
    """True when ``executable`` requires AVX2 and the host CPU reports no AVX2 support."""
    return _standalone_requires_avx2(executable) and _host_supports_avx2() is False


def _ukf_avx2_error(rejected_avx2):
    lines = [
        "Selected standalone UKFTractography requires CPU AVX2 instructions, which this processor does not support. "
        "Install the standard UKFTractography extension through Slicer Extension Manager or pass --ukf-executable "
        "pointing to a non-AVX2 build."
    ]
    for path in rejected_avx2:
        lines.append("  %s (requires AVX2)" % path)
    return "\n".join(lines)


def resolve_executable(explicit=None, name="UKFTractography"):
    candidate = explicit
    if not candidate and name == "UKFTractography":
        candidate = os.environ.get("SLICERDSPACE_UKF_EXECUTABLE") or os.environ.get("UKF_GPU_EXECUTABLE")
    if not candidate and name == "DWIConvert":
        candidate = os.environ.get("SLICERDSPACE_DWICONVERT_EXECUTABLE")
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if name == "UKFTractography" and _ukf_avx2_incompatible(path):
            raise RuntimeError(_ukf_avx2_error([path]))
        return path
    rejected = []
    rejected_avx2 = []

    def usable(path):
        if name != "UKFTractography":
            return True
        if _ukf_avx2_incompatible(path):
            if str(path) not in [str(p) for p in rejected_avx2]:
                rejected_avx2.append(path)
            return False
        mismatch = ukf_vtk_mismatch(path)
        if mismatch is None:
            return True
        if (str(path), mismatch) not in [(str(p), m) for p, m in rejected]:
            rejected.append((path, mismatch))
        return False

    if name == "UKFTractography":
        roots = _extension_search_roots()
        gpu_names_lower = tuple(n.lower() for n in _GPU_EXECUTABLES)
        for gpu_name in _GPU_EXECUTABLES:
            found_gpu = shutil.which(gpu_name)
            if found_gpu and usable(Path(found_gpu).resolve()):
                return Path(found_gpu).resolve()
        for root in roots:
            for pattern in (
                "Runtime/bin/*",
                "lib/Slicer-*/qt-scripted-modules/Runtime/bin/*",
                "slicer.org/Extensions-*/*/lib/Slicer-*/qt-scripted-modules/Runtime/bin/*",
                "Extensions-*/*/lib/Slicer-*/qt-scripted-modules/Runtime/bin/*",
                "lib/Slicer-*/cli-modules/*",
                "slicer.org/Extensions-*/*/lib/Slicer-*/cli-modules/*",
            ):
                for path in root.glob(pattern):
                    if path.is_file() and path.name.lower() in gpu_names_lower and usable(path.resolve()):
                        return path.resolve()
        standalone_names_lower = (name.lower(), name.lower() + ".exe")
        for root in roots:
            for pattern in (
                "bundled-runtimes/ukf-*/bin/*",
                "Extensions-*/*/bundled-runtimes/ukf-*/bin/*",
                "slicer.org/Extensions-*/*/bundled-runtimes/ukf-*/bin/*",
            ):
                for path in sorted(root.glob(pattern)):
                    if path.is_file() and path.name.lower() in standalone_names_lower:
                        resolved_standalone = path.resolve()
                        if usable(resolved_standalone):
                            return resolved_standalone
        try:
            from shared.native_runtimes import runtime_root

            user_rt = runtime_root()
            if user_rt.is_dir():
                for path in sorted(user_rt.glob("ukf-*/bin/*")):
                    if path.is_file() and path.name.lower() in standalone_names_lower:
                        resolved_standalone = path.resolve()
                        if usable(resolved_standalone):
                            return resolved_standalone
        except Exception:
            pass
    found = shutil.which(name)
    if found and usable(Path(found).resolve()):
        return Path(found).resolve()
    cause = None
    try:
        from shared.pipeline import resolve_cli_executable

        cli = Path(resolve_cli_executable(name.lower())).resolve()
        if usable(cli):
            return cli
    except (RuntimeError, AttributeError, ImportError, FileNotFoundError) as exc:
        cause = exc
    # Background PythonSlicer does not instantiate the GUI module registry.
    roots = _extension_search_roots()
    for root in roots:
        for pattern in ("lib/Slicer-*/cli-modules/*", "slicer.org/Extensions-*/*/lib/Slicer-*/cli-modules/*"):
            for path in root.glob(pattern):
                if path.is_file() and path.name.lower() in (name.lower(), name.lower() + ".exe") and usable(path.resolve()):
                    return path.resolve()
    if rejected:
        raise RuntimeError(_ukf_vtk_error(rejected))
    if rejected_avx2:
        raise RuntimeError(_ukf_avx2_error(rejected_avx2))
    raise RuntimeError(
        "Install UKFTractography and SlicerDMRI through Slicer Extension Manager, restart Slicer, "
        "and select UKF again. Headless users may pass --ukf-executable. Missing CLI: " + name
    ) from cause


def paths(subject, directory):
    directory = Path(directory)
    return dict(
        tractogram=directory / (subject + "_ukf.vtk"),
        tractogram_trk=directory / (subject + "_ukf.trk"),
        qc=directory / "ukf_qc.json",
        manifest=directory / "ukf_manifest.json",
        schema=directory / "ukf_cli.xml",
        effective_mask=directory / (subject + "_ukf_valid_mask.nii.gz"),
    )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def executable_identity(executable):
    executable = Path(executable).resolve()
    libraries = {}
    roots = {executable.parent, executable.parent.parent / "lib", *list(executable.parents)[:3]}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*UKF*", "*ukf*", "*DWIConvert*", "libcudart*", "cudart64_*"):
            for path in root.glob(pattern):
                name = path.name.lower()
                if path.is_file() and (name.endswith((".dll", ".dylib", ".so")) or ".so." in name):
                    resolved = str(path.resolve())
                    if resolved not in libraries:
                        libraries[resolved] = digest(path)
    return dict(path=str(executable), sha256=digest(executable), companion_libraries=libraries)


def _dwiconvert_gradients_valid(affine):
    """True when DWIConvert FSLToNrrd already writes world-correct gradients: det(A) < 0 and no rotation."""
    from shared import gradient_frame

    return not gradient_frame.determinant_positive(affine) and gradient_frame.is_signed_diagonal(affine)


def _input_affine(raw):
    """The DWI affine; None when nibabel cannot read the file (then affected, as in the cache-key pin)."""
    import nibabel as nib

    try:
        return nib.load(str(raw.dwi_path)).affine
    except Exception:
        return None


def conversion_policy(affine):
    if affine is not None and _dwiconvert_gradients_valid(affine):
        return CONVERSION_POLICY_DWICONVERT
    return CONVERSION_POLICY_FSL_WORLD


def _fsl_gradient_table(raw):
    import numpy as np

    bvals = np.loadtxt(raw.bval_path).reshape(-1)
    bvecs = np.loadtxt(raw.bvec_path, ndmin=2)
    if bvecs.shape == (3, len(bvals)):
        bvecs = bvecs.T
    return bvals, bvecs


def _normalize_ukf_device_request(device: str | None) -> str:
    req = str(device or "auto").strip().lower() or "auto"
    if req.startswith("auto"):
        return "auto"
    if req.startswith("cuda"):
        return req.split()[0]
    return req


def _ukf_binary_has_compiled_cuda(binary: Path | str | None) -> bool | None:
    """Return False when ``binary`` is demonstrably a CPU-only UKF build despite declaring --useGPU XML.

    Returns True when metadata explicitly confirms CUDA support, False when metadata or
    native binary markers prove a CPU-only build, and None when unknown (e.g. test scripts).
    """
    if not binary:
        return None
    try:
        path = Path(binary).expanduser().resolve()
    except OSError:
        path = Path(binary)
    for meta_path in (path.parent / "runtime.json", path.parent.parent / "runtime.json"):
        try:
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    for key in ("cuda", "cuda_enabled", "gpu_enabled", "use_cuda"):
                        if key in meta and isinstance(meta[key], bool):
                            return bool(meta[key])
                    backend = str(meta.get("backend") or meta.get("compute_backend") or "").strip().lower()
                    if backend == "cpu":
                        return False
                    if backend in {"cuda", "gpu"}:
                        return True
        except Exception:
            pass
    try:
        if not path.is_file():
            return None
        with path.open("rb") as fh:
            magic = fh.read(4)
        is_native_binary = (
            magic[:2] == b"MZ"
            or magic == b"\x7fELF"
            or magic in (b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
        )
        if not is_native_binary:
            return None
        # Upstream ukftractography_gpu compiles this literal only in the `#else` branch of `#ifdef UKF_USE_CUDA`.
        data = path.read_bytes()
        if b"this build has no CUDA support" in data or b"built without CUDA" in data:
            return False
    except OSError:
        return None
    return None


def signature(raw, options, executable, converter=None, *, device="auto"):
    converter = converter or resolve_executable(name="DWIConvert")
    gpu_opts = parse_gpu_options(options)
    req = _normalize_ukf_device_request(device)
    payload = dict(
        version=5,
        subject_id=raw.subject_id,
        parameters=asdict(parameters(options)),
        inputs={
            key: dict(path=str(getattr(raw, key).resolve()), sha256=digest(getattr(raw, key)))
            for key in ("dwi_path", "bval_path", "bvec_path", "mask_path")
        },
        executable=executable_identity(executable),
        converter=executable_identity(converter),
        protocol="author-module-defaults-with-explicit-overrides",
        shell_policy="all supplied volumes; no implicit shell filtering",
        conversion_policy=conversion_policy(_input_affine(raw)),
        mask_policy=MASK_POLICY,
        max_excluded_mask_fraction=MAX_EXCLUDED_MASK_FRACTION,
    )
    gpu_sig = {}
    if gpu_opts.get("useGPU") is True or req.startswith("cuda"):
        gpu_sig["useGPU"] = True
    elif "useGPU" in gpu_opts and gpu_opts["useGPU"] is False:
        gpu_sig["useGPU"] = False
    if gpu_opts.get("gpuPrecision", 0) != 0:
        gpu_sig["gpuPrecision"] = gpu_opts["gpuPrecision"]
    if gpu_sig:
        payload["gpu"] = gpu_sig
    return payload


UKF_EXECUTION_ONLY_PARAMS = frozenset({"numThreads", "gpuDevice"})


def _ukf_executable_matches(recorded, expected) -> bool:
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        return False
    if recorded.get("sha256") != expected.get("sha256"):
        return False
    rec_libs = recorded.get("companion_libraries") or {}
    exp_libs = expected.get("companion_libraries") or {}
    if not isinstance(rec_libs, dict) or not isinstance(exp_libs, dict):
        return False
    rec_norm = sorted((os.path.normcase(Path(str(k)).name), str(v)) for k, v in rec_libs.items())
    exp_norm = sorted((os.path.normcase(Path(str(k)).name), str(v)) for k, v in exp_libs.items())
    return rec_norm == exp_norm


def _ukf_signature_matches(recorded, expected) -> bool:
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        return False
    for key in (
        "version", "subject_id", "protocol", "shell_policy",
        "conversion_policy", "mask_policy", "max_excluded_mask_fraction", "gpu",
    ):
        if recorded.get(key) != expected.get(key):
            return False
    rec_params = recorded.get("parameters")
    exp_params = expected.get("parameters")
    if not isinstance(rec_params, dict) or not isinstance(exp_params, dict):
        return False
    rec_sci = {k: v for k, v in rec_params.items() if k not in UKF_EXECUTION_ONLY_PARAMS}
    exp_sci = {k: v for k, v in exp_params.items() if k not in UKF_EXECUTION_ONLY_PARAMS}
    if rec_sci != exp_sci:
        return False
    rec_inputs = recorded.get("inputs")
    exp_inputs = expected.get("inputs")
    if not isinstance(rec_inputs, dict) or not isinstance(exp_inputs, dict):
        return False
    if set(rec_inputs.keys()) != set(exp_inputs.keys()):
        return False
    for key, exp_item in exp_inputs.items():
        rec_item = rec_inputs.get(key)
        if not isinstance(rec_item, dict) or not isinstance(exp_item, dict):
            return False
        if rec_item.get("sha256") != exp_item.get("sha256"):
            return False
    return (
        _ukf_executable_matches(recorded.get("executable"), expected.get("executable"))
        and _ukf_executable_matches(recorded.get("converter"), expected.get("converter"))
    )


def current(raw, directory, options=None, executable=None, converter=None, *, device="auto", _signature=None):
    try:
        products = paths(raw.subject_id, directory)
        record = json.loads(products["manifest"].read_text(encoding="utf-8"))
        expected_signature = (
            _signature
            if _signature is not None
            else signature(
                raw,
                options,
                resolve_executable(executable),
                resolve_executable(converter, name="DWIConvert"),
                device=device,
            )
        )
        if record["status"] != "completed" or not _ukf_signature_matches(record.get("signature"), expected_signature):
            return False
        gpu_opts = parse_gpu_options(options)
        req = _normalize_ukf_device_request(device)
        rec_gpu_enabled = bool((record.get("gpu") or {}).get("enabled"))
        if (req.startswith("cuda") or gpu_opts.get("useGPU") is True) and not rec_gpu_enabled:
            return False
        if (req in {"cpu", "mps"} or gpu_opts.get("useGPU") is False) and rec_gpu_enabled:
            return False
        qc = json.loads(products["qc"].read_text(encoding="utf-8"))
        return qc.get("automatic_qc_passed") is True and all(
            path.is_file() and record["artifacts"][key] == digest(path)
            for key, path in products.items()
            if key != "manifest"
        )
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError):
        return False


def _host_ram_gib() -> float | None:
    """Return physical host memory in GiB when observable, else None."""
    try:
        from shared.hardware_adapt import host_total_ram_gib

        return host_total_ram_gib()
    except Exception:
        pass
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total / float(1024 ** 3)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) and stat.ullTotalPhys > 0:
                return float(stat.ullTotalPhys) / float(1024 ** 3)
        except Exception:
            pass
    elif hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return float(pages * page_size) / float(1024 ** 3)
        except (OSError, ValueError):
            pass
    return None


def _is_optimized_standalone_executable(executable: Path | str | None) -> bool:
    """True when ``executable`` is the bundled standalone optimized CPU UKF binary."""
    if not executable:
        return False
    path = Path(executable)
    parts = [part.lower() for part in path.parts]
    if "bundled-runtimes" in parts and any(part.startswith("ukf-") for part in parts):
        return True
    try:
        runtime_json = path.parent.parent / "runtime.json"
        if runtime_json.is_file():
            payload = json.loads(runtime_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and str(payload.get("kind", "")).strip().lower() == "ukf-standalone":
                return True
    except Exception:
        pass
    return False


def _host_memory_snapshot() -> dict[str, float | None]:
    """Return cross-OS host memory snapshot (total, available, Windows commit headroom) in GiB."""
    try:
        from shared.hardware_adapt import query_host_memory_snapshot

        return query_host_memory_snapshot()
    except Exception:
        total = _host_ram_gib()
        return {
            "total_ram_gib": total,
            "available_ram_gib": None,
            "commit_limit_gib": None,
            "commit_total_gib": None,
            "commit_available_gib": None,
            "swap_total_gib": None,
            "swap_free_gib": None,
            "effective_available_ram_gib": None,
        }


def _estimate_ukf_peak_ram_gib(
    inputs: dict | None,
    *,
    executable: Path | str | None = None,
    selected: UKFParameters | None = None,
) -> float:
    """Estimate peak host RAM/commit working set (GiB) for CPU UKF loading + VTK output."""
    inp = inputs or {}
    volumes = int(inp.get("volumes") or 0)
    voxels = int(inp.get("effective_brain_voxels") or inp.get("brain_voxels") or 0)
    if volumes <= 0 or voxels <= 0:
        return 0.0
    seeds = max(1, int(getattr(selected, "seedsPerVoxel", 1) or 1))
    dwi_scale = (volumes / 288.0) * (voxels / 950_000.0)
    vox_scale = voxels / 950_000.0
    if _is_optimized_standalone_executable(executable):
        base = 0.5 if voxels >= 50_000 else 0.1
        return max(base, base + 6.5 * dwi_scale + 2.0 * vox_scale * seeds)
    base = 1.0 if voxels >= 50_000 else 0.1
    return max(base, base + 19.0 * dwi_scale + 5.0 * vox_scale * seeds)


def _resolve_ukf_memory_inputs(
    *,
    ram_gib: float | None,
    available_ram_gib: float | None,
    commit_available_gib: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve ``(host_ram, available_ram, commit_available, effective_available)`` in GiB."""
    if ram_gib is None and available_ram_gib is None and commit_available_gib is None:
        snap = _host_memory_snapshot()
        host_ram = snap.get("total_ram_gib")
        if host_ram is None:
            host_ram = _host_ram_gib()
        avail_ram = snap.get("available_ram_gib")
        commit_avail = snap.get("commit_available_gib")
    else:
        host_ram = float(ram_gib) if ram_gib is not None else _host_ram_gib()
        avail_ram = float(available_ram_gib) if available_ram_gib is not None else None
        commit_avail = float(commit_available_gib) if commit_available_gib is not None else None
    avail_parts = [v for v in (avail_ram, commit_avail) if v is not None]
    effective_avail = min(avail_parts) if avail_parts else None
    return host_ram, avail_ram, commit_avail, effective_avail


def resolve_adaptive_cpu_num_threads(
    selected: UKFParameters,
    inputs: dict | None,
    *,
    use_gpu: bool,
    executable: Path | str | None = None,
    ram_gib: float | None = None,
    available_ram_gib: float | None = None,
    commit_available_gib: float | None = None,
    cpu_count: int | None = None,
    platform_name: str | None = None,
    is_apple_silicon: bool | None = None,
) -> int:
    """Resolve effective CPU ``--numThreads`` at execution time without mutating cache keys.

    Preserves explicit user overrides, GPU runs, cohort-scheduled runs, and high-RAM (>48 GiB
    total and >32 GiB available/commit headroom) Linux/Windows servers unchanged, while capping
    memory-heavy CPU UKF runs on modest-RAM workstations, commit/RAM-pressured hosts, and
    unified-memory macOS hosts to prevent OOM/swap thrashing and E-core stalls.
    Bundled standalone optimized UKF binaries eliminate per-step heap allocations and flat-line
    peak RSS across thread counts, so idle Windows/Linux hosts with >12 GiB RAM and comfortable
    available/commit headroom can use all cores.
    """
    if use_gpu or int(selected.numThreads) != -1 or os.environ.get("SLICERDSPACE_COHORT_SCHEDULE"):
        return int(selected.numThreads)

    plat = (sys.platform if platform_name is None else str(platform_name)).lower()
    host_ram, _avail_ram, _commit_avail, effective_avail = _resolve_ukf_memory_inputs(
        ram_gib=ram_gib,
        available_ram_gib=available_ram_gib,
        commit_available_gib=commit_available_gib,
    )
    if cpu_count is None:
        try:
            cpus = max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            cpus = max(1, int(os.cpu_count() or 1))
    else:
        cpus = max(1, int(cpu_count))

    stage_budget: int | None = None
    stage_raw = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if stage_raw.isdigit() and int(stage_raw) > 0:
        stage_budget = int(stage_raw)

    if is_apple_silicon is None:
        if plat == "darwin" and platform_name is None:
            try:
                from shared.hardware_adapt import is_apple_silicon_host

                apple_silicon = bool(is_apple_silicon_host())
            except Exception:
                apple_silicon = False
        else:
            apple_silicon = False
    else:
        apple_silicon = bool(is_apple_silicon)

    # High-RAM Linux/Windows servers with ample available/commit headroom: zero regression.
    if (
        plat != "darwin"
        and (host_ram is None or host_ram > 48.0)
        and (effective_avail is None or effective_avail > 32.0)
    ):
        return min(cpus, stage_budget) if stage_budget is not None else -1

    inp = inputs or {}
    volumes = int(inp.get("volumes") or 0)
    voxels = int(inp.get("effective_brain_voxels") or inp.get("brain_voxels") or 0)
    heavy = volumes >= 120 or voxels >= 450_000
    optimized_standalone = _is_optimized_standalone_executable(executable)
    est_peak = _estimate_ukf_peak_ram_gib(inputs, executable=executable, selected=selected)

    cap: int | None = None
    if heavy and host_ram is not None and host_ram <= 48.0:
        if optimized_standalone:
            # Optimized standalone UKF keeps flat RSS (~7.5 GiB on 288-vol HCP) and
            # zero per-step allocator lock contention across threads; only guard <=12 GiB RAM.
            if host_ram <= 12.0:
                cap = min(cpus, 4)
        else:
            if host_ram <= 12.0:
                cap = min(cpus, 2)
            elif host_ram <= 20.0:
                cap = min(cpus, 4)
            elif host_ram <= 36.0:
                cap = min(cpus, 6)
            else:
                cap = min(cpus, 8)

    # Factor in current available physical RAM and Windows commit headroom so hosts whose
    # memory is already consumed by other applications throttle conservatively instead of
    # spawning unbounded threads into swap thrashing or VTK allocation failure.
    if effective_avail is not None and (heavy or (est_peak >= 1.0 and effective_avail < est_peak * 1.5)):
        headroom_ratio = effective_avail / max(est_peak, 1.0)
        avail_cap: int | None = None
        if effective_avail <= 6.0 or headroom_ratio < 0.75:
            avail_cap = min(cpus, 2 if optimized_standalone else 1)
        elif effective_avail <= 10.0 or headroom_ratio < 1.15:
            avail_cap = min(cpus, 4 if optimized_standalone else 2)
        elif effective_avail <= 16.0 and (not optimized_standalone or headroom_ratio < 1.5):
            avail_cap = min(cpus, 6 if optimized_standalone else 4)
        elif not optimized_standalone and effective_avail <= 24.0:
            avail_cap = min(cpus, 4)
        elif not optimized_standalone and effective_avail <= 36.0:
            avail_cap = min(cpus, 6)
        if avail_cap is not None:
            cap = avail_cap if cap is None else min(cap, avail_cap)

    if plat == "darwin":
        mac_cap: int | None = None
        if host_ram is not None:
            if host_ram <= 12.0:
                mac_cap = min(cpus, 4 if optimized_standalone else 2)
            elif host_ram <= 20.0:
                mac_cap = min(cpus, 8 if optimized_standalone else 4)
            elif host_ram <= 36.0:
                mac_cap = min(cpus, max(6, (cpus * 3) // 4), 12) if (optimized_standalone and apple_silicon) else min(cpus, 10 if optimized_standalone else 6)
            elif apple_silicon:
                mac_cap = min(cpus, max(6, (cpus * 3) // 4), 12 if optimized_standalone else 10)
        elif apple_silicon:
            mac_cap = min(cpus, max(6, (cpus * 3) // 4), 12 if optimized_standalone else 10)
        if mac_cap is not None:
            cap = mac_cap if cap is None else min(cap, mac_cap)

    if stage_budget is not None:
        cap = min(cpus, stage_budget) if cap is None else min(cap, stage_budget)

    if cap is None:
        return -1
    return max(1, int(cap))


def _ukf_cpu_memory_runtime_warning(
    inputs: dict,
    *,
    use_gpu: bool,
    executable: Path | str | None = None,
    selected: UKFParameters | None = None,
    ram_gib: float | None = None,
    available_ram_gib: float | None = None,
    commit_available_gib: float | None = None,
) -> str | None:
    """Warn when CPU UKF runs under tight installed RAM, available RAM, or Windows commit headroom."""
    if use_gpu:
        return None
    volumes = int(inputs.get("volumes") or 0)
    shells = inputs.get("approximate_shells") or ()
    voxels = int(inputs.get("effective_brain_voxels") or inputs.get("brain_voxels") or 0)
    host_ram, avail_ram, commit_avail, effective_avail = _resolve_ukf_memory_inputs(
        ram_gib=ram_gib,
        available_ram_gib=available_ram_gib,
        commit_available_gib=commit_available_gib,
    )
    est_peak = _estimate_ukf_peak_ram_gib(inputs, executable=executable, selected=selected)
    optimized_standalone = _is_optimized_standalone_executable(executable)

    # Dynamic available RAM / Windows commit headroom guard (triggers even on 32-64 GiB hosts
    # when other apps have consumed most physical RAM or Windows commit charge).
    if effective_avail is not None and est_peak >= 1.0 and effective_avail < est_peak * 1.25:
        mem_parts: list[str] = []
        if host_ram is not None:
            mem_parts.append(f"{host_ram:.1f} GiB installed RAM")
        if avail_ram is not None:
            mem_parts.append(f"{avail_ram:.1f} GiB available RAM")
        if commit_avail is not None:
            mem_parts.append(f"{commit_avail:.1f} GiB Windows commit headroom")
        mem_summary = ", ".join(mem_parts) if mem_parts else f"{effective_avail:.1f} GiB effective headroom"
        commit_tight = commit_avail is not None and commit_avail < est_peak * 1.15
        if commit_tight:
            risk = (
                "Windows commit headroom is critically tight relative to UKF's working set; "
                "VTK tractogram output allocation may fail (std::bad_alloc) or trigger severe pagefile/swap thrashing."
            )
            action = (
                "Close memory-heavy applications, increase the Windows paging file / commit limit, "
                "use GPU UKF (--useGPU) or CSD/ACT, or cap UKF output density/threads via "
                '--ukf-parameters \'{"numThreads": 2, "recordLength": 1.8}\'.'
            )
        else:
            risk = "Host available memory is tight relative to UKF's working set and may trigger swap thrashing or OOM."
            action = (
                "Close memory-heavy applications, use GPU UKF (--useGPU) or CSD/ACT, or pass conservative "
                '--ukf-parameters (e.g. \'{"numThreads": 2}\' or a larger "recordLength").'
            )
        return (
            f"CPU UKF memory headroom warning ({volumes} volumes, {voxels:,} mask voxels; "
            f"estimated peak ~{est_peak:.1f} GiB vs {mem_summary}): {risk} "
            f"Adaptive --numThreads throttling is applied automatically when unset (-1). {action}"
        )

    if volumes < 150 or len(shells) < 2 or voxels < 500_000:
        return None
    if host_ram is None:
        return None
    if optimized_standalone:
        if host_ram > 16.0:
            return None
        return (
            f"High-resolution multi-shell CPU UKF ({volumes} volumes, {len(shells)} shells, "
            f"{voxels:,} mask voxels on {host_ram:.1f} GiB RAM) uses ~8 GB peak RAM during loading; "
            "close large applications or consider CSD/ACT or GPU UKF (--useGPU) if memory pressure occurs."
        )
    if host_ram > 36.0:
        return None
    return (
        f"High-resolution multi-shell CPU UKF ({volumes} volumes, {len(shells)} shells, "
        f"{voxels:,} mask voxels on {host_ram:.1f} GiB RAM) can take >40 min and >25 GB RAM. "
        "Adaptive --numThreads capping is applied automatically when unset (-1); "
        "consider CSD/ACT, GPU UKF (--useGPU), or capping --numThreads explicitly if memory pressure occurs."
    )


def validate_inputs(raw):
    return _validated_inputs(raw)[0]


def _validated_inputs(raw):
    import nibabel as nib
    import numpy as np

    image = nib.load(raw.dwi_path)
    mask_image = nib.load(raw.mask_path)
    b = np.loadtxt(raw.bval_path).reshape(-1)
    g = np.loadtxt(raw.bvec_path, ndmin=2)
    if g.shape == (3, len(b)):
        g = g.T
    if len(image.shape) != 4 or image.shape[3] != len(b) or g.shape != (len(b), 3):
        raise ValueError("UKF DWI and gradient dimensions disagree")
    if not np.isfinite(b).all() or np.any(b < 0) or not np.isfinite(g).all():
        raise ValueError("UKF gradients must be finite, b-values nonnegative")
    nonzero = b > 50
    if (
        not np.any(~nonzero)
        or np.count_nonzero(nonzero) < 6
        or np.any(abs(np.linalg.norm(g[nonzero], axis=1) - 1) > 0.01)
    ):
        raise ValueError("UKF needs b0, at least six diffusion measurements and unit b-vectors")
    directions = g[nonzero]
    design = np.column_stack(
        [
            directions[:, 0] ** 2,
            directions[:, 1] ** 2,
            directions[:, 2] ** 2,
            directions[:, 0] * directions[:, 1],
            directions[:, 0] * directions[:, 2],
            directions[:, 1] * directions[:, 2],
        ]
    )
    if np.linalg.matrix_rank(design) < 6:
        raise ValueError("UKF diffusion directions do not span a tensor design (rank < 6)")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid UKF spatial affine")
    if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("UKF mask and DWI must share the native grid")
    mask = np.asanyarray(mask_image.dataobj)
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError("UKF mask must be nonempty, finite and binary")
    from shared.diffusion_io import ensure_cached_uncompressed_nifti

    dwi_path = Path(raw.dwi_path)
    is_gz = dwi_path.name.lower().endswith(".nii.gz")
    cached_nii = ensure_cached_uncompressed_nifti(dwi_path) if is_gz else None
    non_b0 = ~nonzero
    if cached_nii is not None or not is_gz:
        src_image = nib.load(str(cached_nii or dwi_path), mmap="r", keep_file_open=False) if cached_nii is not None else image
        nx, ny, nz = src_image.shape[:3]
        baseline = np.empty((nx, ny, nz), dtype=np.float32)
        for z0 in range(0, nz, 16):
            z1 = min(nz, z0 + 16)
            slab = np.asarray(src_image.dataobj[:, :, z0:z1, :], dtype=np.float32)
            if not np.isfinite(slab).all():
                raise ValueError("UKF DWI contains NaN or Inf; repair the source data before fitting")
            baseline[:, :, z0:z1] = slab[..., non_b0].mean(axis=3)
    else:
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("UKF DWI contains NaN or Inf; repair the source data before fitting")
        baseline = data[..., non_b0].mean(axis=3)
    effective = (mask > 0) & np.isfinite(baseline) & (baseline > 0)
    original_count = int(np.count_nonzero(mask))
    excluded = original_count - int(effective.sum())
    excluded_fraction = excluded / original_count
    if not effective.any() or excluded_fraction > MAX_EXCLUDED_MASK_FRACTION:
        raise ValueError(
            f"UKF brain mask has {excluded}/{original_count} voxels without finite positive b0 signal "
            f"({excluded_fraction:.2%}); exceeds the {MAX_EXCLUDED_MASK_FRACTION:.0%} input-integrity limit. "
            "Review the DWI and mask; source files are unchanged."
        )
    report = dict(
        volumes=len(b),
        b0_volumes=int(np.count_nonzero(~nonzero)),
        brain_voxels=original_count,
        effective_brain_voxels=int(effective.sum()),
        excluded_nonpositive_b0_voxels=excluded,
        excluded_mask_fraction=excluded_fraction,
        mask_policy=MASK_POLICY,
        max_excluded_mask_fraction=MAX_EXCLUDED_MASK_FRACTION,
        warnings=([f"UKF excludes {excluded} mask voxels without finite positive mean b0 signal; "
                   "the source DWI and mask are unchanged."] if excluded else []),
        approximate_shells=np.unique(np.round(b[nonzero] / 100) * 100).tolist(),
        input_coordinate_system=(INPUT_FRAME_DWICONVERT if _dwiconvert_gradients_valid(image.affine)
                                 else INPUT_FRAME_FSL_WORLD),
    )
    return report, effective


def write_effective_mask(raw, effective, target):
    import nibabel as nib
    import numpy as np

    target = Path(target)
    if target.resolve() in {Path(getattr(raw, key)).resolve() for key in
                            ('dwi_path', 'mask_path', 'bval_path', 'bvec_path')}:
        raise ValueError('UKF effective mask must not overwrite a source input')
    source = nib.load(raw.mask_path)
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    nib.save(source.__class__(effective.astype(np.uint8), source.affine, header), target)
    return replace(raw, mask_path=target)


def _load_validated_vtk_polydata(path):
    """Read and validate a legacy VTK streamline file in a single pass."""
    import re
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0 or path.suffix.lower() != ".vtk":
        raise ValueError("UKF output is empty or malformed")
    from contextlib import nullcontext
    from shared.external_process import link_or_copy_file, needs_windows_ascii_staging, windows_ascii_temp_dir

    stage_vtk = needs_windows_ascii_staging(path)
    with (windows_ascii_temp_dir(prefix="ukf_vtk_") if stage_vtk else nullcontext(None)) as stage_dir:
        reader_path = link_or_copy_file(path, stage_dir / "tracts.vtk") if stage_vtk else path
        reader = vtk.vtkPolyDataReader()
        errors = []
        reader.AddObserver("ErrorEvent", lambda *_: errors.append(True))
        reader.AddObserver("WarningEvent", lambda *_: errors.append(True))
        reader.SetFileName(str(reader_path))
        reader.Update()
        if errors or reader.GetErrorCode():
            raise ValueError("UKF output is empty or malformed")
        data = reader.GetOutput()
    with path.open("rb") as stream:
        head_bytes = stream.read(65536)
    header = head_bytes.split(b"\n", 4)
    if len(header) < 4 or header[2].strip().upper() not in {b"ASCII", b"BINARY"}:
        raise ValueError("UKF output is empty or malformed")
    is_ascii = header[2].strip().upper() == b"ASCII"
    raw_bytes = path.read_bytes() if is_ascii else head_bytes
    declared = re.search(rb"\bPOINTS\s+(\d+)\s+\w+\s", raw_bytes)
    if declared is None and not is_ascii:
        raw_bytes = path.read_bytes()
        declared = re.search(rb"\bPOINTS\s+(\d+)\s+\w+\s", raw_bytes)
    if declared is None or int(declared.group(1)) != data.GetNumberOfPoints():
        raise ValueError("UKF output is empty or malformed")
    if is_ascii:
        raw_tokens = raw_bytes[declared.end():].split(maxsplit=3 * data.GetNumberOfPoints())
        values = np.asarray(raw_tokens[: 3 * data.GetNumberOfPoints()], dtype=float)
        if values.size != 3 * data.GetNumberOfPoints() or not np.isfinite(values).all():
            raise ValueError("UKF output is empty or malformed")
    pts_obj = data.GetPoints()
    if pts_obj is None or data.GetNumberOfVerts() or data.GetNumberOfPolys() or data.GetNumberOfStrips():
        raise ValueError("UKF output is empty or malformed")
    points = vtk_to_numpy(pts_obj.GetData())
    if not np.isfinite(points).all():
        raise ValueError("UKF output has nonfinite coordinates")
    lines = data.GetLines()
    count = int(data.GetNumberOfLines())
    offsets = vtk_to_numpy(lines.GetOffsetsArray())
    connectivity = vtk_to_numpy(lines.GetConnectivityArray())
    if count < 1 or len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(connectivity):
        raise ValueError("UKF output is empty or malformed")
    if np.any(np.diff(offsets) < 2) or np.any(connectivity < 0) or np.any(connectivity >= data.GetNumberOfPoints()):
        raise ValueError("UKF output is empty or malformed")
    ordered = points[connectivity]
    if not np.isfinite(ordered).all():
        raise ValueError("UKF output has nonfinite coordinates")
    changes = np.any(np.diff(ordered, axis=0) != 0, axis=1)
    if count > 1:
        changes[offsets[1:-1] - 1] = False
    if np.any(np.add.reduceat(changes.astype(np.int64), offsets[:-1]) == 0):
        raise ValueError("UKF output is empty or malformed")
    return data, ordered, offsets


def output_qc(path, raw, selected=None, *, _polydata_bundle=None):
    import nibabel as nib
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    data, ordered, offsets = _polydata_bundle if _polydata_bundle is not None else _load_validated_vtk_polydata(path)
    count = int(len(offsets) - 1)
    image = nib.load(raw.mask_path)
    inverse = np.linalg.inv(image.affine)
    mask = np.asanyarray(image.dataobj) > 0
    seg_norms = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    csum = np.empty(len(ordered), dtype=np.float64)
    csum[0] = 0.0
    csum[1:] = np.cumsum(seg_norms, dtype=np.float64)
    lengths = csum[offsets[1:] - 1] - csum[offsets[:-1]]
    total = int(len(ordered))
    inside = 0
    inv_rot_t = np.ascontiguousarray(inverse[:3, :3].T, dtype=np.float64)
    inv_trans = np.asarray(inverse[:3, 3], dtype=np.float64)
    shape_arr = np.asarray(mask.shape, dtype=np.intp)
    chunk = 2_000_000
    for start in range(0, total, chunk):
        pts_chunk = np.asarray(ordered[start : start + chunk], dtype=np.float64)
        voxels = np.rint(pts_chunk @ inv_rot_t + inv_trans).astype(np.intp)
        valid = ((voxels >= 0) & (voxels < shape_arr)).all(axis=1)
        if np.any(valid):
            inside += int(mask[voxels[valid, 0], voxels[valid, 1], voxels[valid, 2]].sum())
    arrays_valid = True
    for index in range(1, (selected or parameters()).numTensor + 1):
        for name, components in ((f"tensor{index}", 9), (f"FA{index}", 1)):
            array = data.GetPointData().GetArray(name)
            valid_array = (
                array is not None
                and array.GetNumberOfTuples() == data.GetNumberOfPoints()
                and array.GetNumberOfComponents() == components
            )
            if valid_array:
                values = vtk_to_numpy(array)
                valid_array = bool(np.isfinite(values).all())
                if name.startswith("FA"):
                    valid_array = valid_array and bool(((values >= -1e-6) & (values <= 1 + 1e-6)).all())
            arrays_valid = arrays_valid and valid_array
    checks = dict(
        nonempty_geometry=count == len(lengths),
        positive_lengths=bool(np.all(lengths > 0)),
        native_mask_overlap=inside / max(1, total) >= 0.9,
        requested_tensor_and_fa_arrays=arrays_valid,
    )
    return dict(
        status="completed",
        automatic_qc_passed=all(checks.values()),
        checks=checks,
        streamline_count=count,
        length_mm_percentiles=np.percentile(lengths, [0, 25, 50, 75, 100]).tolist(),
        points_inside_brain_fraction=inside / max(1, total),
        human_review="pending",
        scientific_scope="Geometry/native-mask sanity only; not anatomical accuracy or reliable ORG sampling",
    )


def schema_supports_gpu(schema) -> bool:
    """Return True when the installed UKF CLI exposes NeuroDMRI/ukftractography_gpu flags."""
    names = {node.findtext("name") for node in schema.findall("./parameters/*") if node.find("name") is not None}
    return {"useGPU", "gpuDevice", "gpuPrecision"}.issubset(names)


def ukf_gpu_process_environment(startup_environment, executable):
    """Isolate bundled ukf-gpu and standalone ukf binaries from Slicer's Qt/Python/runtime libraries."""
    exe_path = Path(executable)
    parts_lower = [part.lower() for part in exe_path.parts]
    is_standalone_runtime = "bundled-runtimes" in parts_lower and any("ukf" in p for p in parts_lower)
    is_gpu_bundle = is_standalone_runtime or exe_path.name.lower().startswith(("ukf-gpu", "ukftractography_gpu")) or (
        exe_path.parent.name.lower() == "bin" and exe_path.parent.parent.name.lower() == "runtime"
    )
    env = dict(startup_environment)
    if not is_gpu_bundle:
        return env
    removed = {
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "UKF_GPU_DEVICES",
        "UKF_GPU_PROGRESS",
    }
    for key in list(env):
        if key.upper() in removed:
            del env[key]
    lib_dir = exe_path.parent.parent / "lib"
    if is_standalone_runtime and sys.platform.startswith("linux") and lib_dir.is_dir():
        # The Linux standalone UKF has no RUNPATH: point the loader at its own bundled libraries only.
        env["LD_LIBRARY_PATH"] = str(lib_dir)
    if sys.platform == "win32":
        from pathlib import PureWindowsPath

        windows = next((value for key, value in env.items() if key.upper() == "SYSTEMROOT"), None)
        cuda_path = next((value for key, value in env.items() if key.upper() == "CUDA_PATH"), None)
        if windows:
            for key in list(env):
                if key.upper() == "PATH":
                    del env[key]
            path_parts = [str(PureWindowsPath(str(exe_path)).parent), windows + r"\System32", windows]
            if cuda_path and (Path(cuda_path) / "bin").is_dir():
                path_parts.append(str(PureWindowsPath(cuda_path) / "bin"))
            env["PATH"] = ";".join(path_parts)
    return env


def _cuda_runtime_available_for_ukf(binary: Path) -> bool:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            if bool(torch_mod.cuda.is_available()):
                return True
        except Exception:
            pass
    try:
        import contextlib
        import ctypes
        import ctypes.util

        @contextlib.contextmanager
        def _win_dll_dir(folder: Path):
            handle = None
            adder = getattr(os, "add_dll_directory", None)
            if sys.platform == "win32" and callable(adder) and folder.is_dir():
                try:
                    handle = adder(str(folder))
                except OSError:
                    handle = None
            try:
                yield
            finally:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

        candidates = []
        lib_dir = Path(binary).parent.parent / "lib"
        if sys.platform == "win32":
            candidates.extend(sorted(Path(binary).parent.glob("cudart64_*.dll")))
            cuda_path = os.environ.get("CUDA_PATH", "").strip()
            if cuda_path:
                cuda_bin = Path(cuda_path) / "bin"
                if cuda_bin.is_dir():
                    candidates.extend(sorted(cuda_bin.glob("cudart64_*.dll")))
        elif sys.platform.startswith("linux"):
            if lib_dir.is_dir():
                candidates.extend(sorted(lib_dir.glob("libcudart.so*")))
        found_sys = ctypes.util.find_library("cudart")
        if found_sys:
            candidates.append(Path(found_sys))
        deduped = list(dict.fromkeys(candidates))
        for cand in deduped:
            try:
                with _win_dll_dir(cand.parent):
                    lib = ctypes.CDLL(str(cand))
                    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                    lib.cudaGetDeviceCount.restype = ctypes.c_int
                    count = ctypes.c_int(0)
                    if lib.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0:
                        return True
            except (OSError, AttributeError):
                continue
        if sys.platform == "win32":
            sys_nvcuda = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvcuda.dll"
            for drv_name in ("nvcuda.dll", str(sys_nvcuda)):
                try:
                    driver = ctypes.WinDLL(drv_name)
                    driver.cuInit.argtypes = [ctypes.c_uint]
                    driver.cuInit.restype = ctypes.c_int
                    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                    driver.cuDeviceGetCount.restype = ctypes.c_int
                    if driver.cuInit(0) == 0:
                        count = ctypes.c_int(0)
                        if driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0:
                            return True
                except (OSError, AttributeError):
                    continue
        if sys.platform.startswith("linux"):
            try:
                driver = ctypes.CDLL("libcuda.so.1")
                if driver.cuInit(0) == 0:
                    count = ctypes.c_int(0)
                    if driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0:
                        return True
            except (OSError, AttributeError):
                pass
    except Exception:
        pass
    return False


def resolve_ukf_gpu_dispatch(device: str, gpu_opts: dict, schema, binary: Path) -> tuple[bool, int, int]:
    """Decide whether to pass ``--useGPU --gpuDevice ... --gpuPrecision ...`` to UKF CLI."""
    req = _normalize_ukf_device_request(device)
    compiled_cuda = _ukf_binary_has_compiled_cuda(binary)
    supports_gpu = schema_supports_gpu(schema) and (compiled_cuda is not False)
    explicit_gpu = gpu_opts.get("useGPU") is True or req.startswith("cuda")
    if explicit_gpu and not supports_gpu:
        if compiled_cuda is False:
            raise RuntimeError(
                f"Installed UKF CLI ({binary}) is a CPU-only build without CUDA support (-DUKF_USE_CUDA=ON); "
                "install a CUDA-enabled UKF build or run with device='cpu'/'auto'."
            )
        raise RuntimeError(
            "Installed UKF CLI lacks --useGPU; install NeuroDMRI/ukftractography_gpu or omit useGPU."
        )
    if not supports_gpu or gpu_opts.get("useGPU") is False or req in {"cpu", "mps"}:
        return False, 0, 0

    if "gpuDevice" in gpu_opts:
        gpu_device = int(gpu_opts["gpuDevice"])
    elif req == "cuda:all":
        gpu_device = -1
    elif req.startswith("cuda:"):
        try:
            gpu_device = int(req.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Unsupported UKF CUDA device: {device}") from exc
    else:
        gpu_device = 0
    gpu_precision = int(gpu_opts.get("gpuPrecision", 0))

    if req.startswith("cuda") or gpu_opts.get("useGPU") is True:
        return True, gpu_device, gpu_precision
    if _cuda_runtime_available_for_ukf(binary):
        return True, gpu_device, gpu_precision
    return False, 0, 0


def build_command(
    binary,
    dwi,
    mask,
    target,
    selected,
    schema,
    *,
    use_gpu: bool = False,
    gpu_device: int = 0,
    gpu_precision: int = 0,
    seeds_file=None,
    seed_labels=None,
):
    """Use SEM switches explicitly: XML GUI defaults do not enable CLI switches."""
    nodes = {node.findtext("name"): node for node in schema.findall("./parameters/*") if node.find("name") is not None}
    required_names = set(asdict(selected)).union(("dwiFile", "maskFile", "tracts"))
    if seeds_file is not None:
        required_names.update(("seedsFile", "seedLabels"))
    if not required_names.issubset(nodes):
        raise RuntimeError("Installed UKF CLI lacks requested parameters; no fallback")
    command = [str(binary), "--dwiFile", str(dwi), "--maskFile", str(mask), "--tracts", str(target)]
    if seeds_file is not None:
        command.extend(("--seedsFile", str(seeds_file), "--seedLabels", str(seed_labels or "1")))
    for key, value in asdict(selected).items():
        if seeds_file is not None and key == "seedingThreshold":
            continue  # Option 3 (--seedsFile / --seedLabels) supersedes Option 1 (--seedingThreshold)
        if isinstance(value, bool):
            default = nodes[key].findtext("default", "").strip().lower()
            expected = "false" if key == "freeWater" else "true"
            if default != expected:
                raise RuntimeError(f"Installed UKF boolean default differs for {key}; inspect its protocol")
            if value:
                command.append("--" + key)
        else:
            command.extend(("--" + key, str(value)))
    if use_gpu:
        if not schema_supports_gpu(schema):
            raise RuntimeError("Installed UKF CLI lacks --useGPU; install NeuroDMRI/ukftractography_gpu.")
        command.extend(("--useGPU", "--gpuDevice", str(gpu_device), "--gpuPrecision", str(gpu_precision)))
    return command


def inspect_schema(binary, env):
    from shared.external_process import native_arm64_argv, subprocess_no_window_kwargs, text_subprocess_kwargs

    description = subprocess.run(
        native_arm64_argv([str(binary), "--xml"], env),
        capture_output=True,
        timeout=30,
        env=env,
        **text_subprocess_kwargs(),
        **subprocess_no_window_kwargs(),
    )
    start = description.stdout.find("<?xml")
    if start < 0:
        start = description.stdout.find("<executable")
    end = description.stdout.find("</executable>", max(0, start))
    if start < 0 or end < 0:
        raise RuntimeError("UKF CLI schema inspection failed: " + description.stderr[-1000:])
    xml = description.stdout[start : end + len("</executable>")]
    # Some upstream builds return from PARSE_ARGS on --xml but then continue
    # into LoadFiles in ModuleEntryPoint. Accept only complete, parseable XML;
    # conversion and tracking themselves must still return zero.
    return ET.fromstring(xml), xml, description.returncode


def export_trk(source, reference, target, *, _polydata_bundle=None):
    import nibabel as nib
    import numpy as np
    from nibabel.streamlines import ArraySequence, Field

    image = nib.load(reference)
    header = nib.streamlines.TrkFile.create_empty_header()
    header[Field.VOXEL_TO_RASMM] = image.affine
    header[Field.VOXEL_SIZES] = image.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = image.shape[:3]
    header[Field.VOXEL_ORDER] = "RAS"
    if _polydata_bundle is not None or Path(source).suffix.lower() == ".vtk":
        _, ordered, offsets = _polydata_bundle if _polydata_bundle is not None else _load_validated_vtk_polydata(source)
        seq = ArraySequence()
        seq._data = np.ascontiguousarray(ordered, dtype=np.float32)
        seq._offsets = np.ascontiguousarray(offsets[:-1], dtype=np.intp)
        seq._lengths = np.ascontiguousarray(np.diff(offsets), dtype=np.intp)
        tractogram = nib.streamlines.Tractogram(seq, affine_to_rasmm=np.eye(4))
    else:
        from connectome.pipeline import _iter_streamlines

        tractogram = nib.streamlines.LazyTractogram(
            streamlines=lambda: _iter_streamlines(source), affine_to_rasmm=np.eye(4)
        )
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram, header), str(target))


def verify_trk(source, target, *, _polydata_bundle=None):
    import itertools
    import nibabel as nib
    import numpy as np

    if _polydata_bundle is not None or Path(source).suffix.lower() == ".vtk":
        _, ordered, offsets = _polydata_bundle if _polydata_bundle is not None else _load_validated_vtk_polydata(source)
        expected_lengths = np.diff(offsets)
        count = int(len(expected_lengths))
        if count == 0:
            raise ValueError("Empty UKF TRK conversion")
        converted = nib.streamlines.load(str(target), lazy_load=False).streamlines
        if len(converted) != count or not np.array_equal(converted._lengths, expected_lengths):
            raise ValueError("UKF VTK/TRK streamline count or point count changed")
        if not np.allclose(ordered, converted._data, atol=1e-4, rtol=0):
            raise ValueError("UKF VTK/TRK native RAS coordinates changed beyond 0.0001 mm")
        return count
    from connectome.pipeline import _iter_streamlines

    count = 0
    converted = nib.streamlines.load(target, lazy_load=True).streamlines
    for original, restored in itertools.zip_longest(_iter_streamlines(source), converted):
        if original is None or restored is None or np.shape(original) != np.shape(restored):
            raise ValueError("UKF VTK/TRK streamline count or point count changed")
        if not np.allclose(original, restored, atol=1e-4, rtol=0):
            raise ValueError("UKF VTK/TRK native RAS coordinates changed beyond 0.0001 mm")
        count += 1
    if count == 0:
        raise ValueError("Empty UKF TRK conversion")
    return count


def write_conversion_probe(raw, target):
    """Ask the official converter about geometry/gradients without quantizing DWI."""
    import nibabel as nib
    import numpy as np

    image = nib.load(raw.dwi_path)
    values = np.arange(8 * image.shape[3], dtype=np.int64)
    if values[-1] > np.iinfo(np.int16).max:
        raise ValueError("Too many volumes for the lossless UKF conversion probe")
    probe = values.astype(np.int16).reshape((2, 2, 2, image.shape[3]))
    header = image.header.copy()
    header.set_data_dtype(np.int16)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(probe, image.affine, header), target)
    return probe


def restore_lossless_payload(raw, probe, converted, dwi, mask):
    """Validate a pure axis permutation/flip, then copy native signals bit-for-bit.

    DWIConvert supplies the geometry and the b-value scaling of each gradient line. Its
    short-only image IO is used only for a tiny integer probe. The actual DWI
    and mask share the same verified mapping and never pass through a short cast.

    Gradient frame (batch 2): the gradient lines must match the bval/bvec rows in count and order.
    For an axis-aligned det < 0 grid the DWIConvert lines are kept verbatim after a check against the
    FSL rule; for every other grid each line gets the direction of g = R F b in the header frame and
    keeps the norm DWIConvert wrote (zero lines stay zero).
    """
    import nibabel as nib
    import nrrd
    import numpy as np

    converted_probe, header = nrrd.read(str(converted))
    directions = np.asarray(header["space directions"], dtype=float)
    spatial = np.flatnonzero(np.isfinite(directions).all(axis=1))
    channels = np.flatnonzero(~np.isfinite(directions).all(axis=1))
    if len(spatial) != 3 or len(channels) != 1:
        raise ValueError("DWIConvert did not return three spatial axes plus one volume axis")
    flips = {
        "left-posterior-superior": [-1, -1, 1],
        "right-anterior-superior": [1, 1, 1],
        "left-anterior-superior": [-1, 1, 1],
    }
    if header.get("space") not in flips:
        raise ValueError("Unsupported DWIConvert coordinate frame")
    to_ras = np.diag(flips[header["space"]])
    affine = np.eye(4)
    affine[:3, :3] = to_ras @ directions[spatial].T
    affine[:3, 3] = to_ras @ header["space origin"]
    image = nib.load(raw.dwi_path)
    mapping = np.linalg.inv(image.affine) @ affine
    signed = np.rint(mapping[:3, :3]).astype(int)
    if (
        not np.allclose(mapping[:3, :3], signed, atol=1e-5, rtol=0)
        or not np.array_equal(abs(signed).sum(axis=0), np.ones(3))
        or not np.array_equal(abs(signed).sum(axis=1), np.ones(3))
    ):
        raise ValueError("DWIConvert requires resampling, not a lossless axis transformation")
    order = tuple(np.argmax(abs(signed), axis=0).tolist())
    negative = [bool(signed[old, new] < 0) for new, old in enumerate(order)]
    probe_shift = np.zeros(3)
    for new, old in enumerate(order):
        probe_shift[old] = int(negative[new])
    if not np.allclose(mapping[:3, 3], probe_shift, atol=1e-4, rtol=0):
        raise ValueError("DWIConvert origin does not match a lossless probe transformation")

    def reorder(array):
        axes = order + tuple(range(3, array.ndim))
        transformed = np.transpose(array, axes)
        for axis, reverse in enumerate(negative):
            if reverse:
                transformed = np.flip(transformed, axis)
        return transformed

    channel = int(channels[0])
    if not np.array_equal(np.moveaxis(converted_probe, channel, -1), reorder(probe)):
        raise ValueError("DWIConvert changed probe intensities or volume order")
    gradients = [value for key, value in header.items() if key.startswith("DWMRI_gradient_")]
    if len(gradients) != image.shape[3] or not all(
        np.isfinite(np.fromstring(value, sep=" ")).all() and len(np.fromstring(value, sep=" ")) == 3
        for value in gradients
    ):
        raise ValueError("Incomplete DWIConvert gradient metadata")
    gradient_report = _set_world_gradients(raw, image.affine, header)
    full_shift = np.zeros(3)
    for new, old in enumerate(order):
        full_shift[old] = image.shape[old] - 1 if negative[new] else 0
    origin_ras = image.affine[:3, :3] @ full_shift + image.affine[:3, 3]
    header["space origin"] = to_ras @ origin_ras
    for key in ("data file", "sizes", "type", "endian"):
        header.pop(key, None)
    header["encoding"] = "raw"

    import tempfile
    import nrrd.writer as _nrrd_writer
    from shared.diffusion_io import ensure_cached_uncompressed_nifti, open_fast_gzip_reader

    def _stream_write_raw_nrrd(target_path: Path, src_image, nrrd_header: dict) -> np.dtype:
        out_shape = reorder(np.broadcast_to(np.zeros((), dtype=np.uint8), src_image.shape)).shape
        # Move axis -1 (volumes) to `channel` for shape/dtype calculation.
        dummy_moved = np.moveaxis(np.broadcast_to(np.zeros((), dtype=np.uint8), out_shape), -1, channel)
        target_shape = dummy_moved.shape
        sample = np.asanyarray(src_image.dataobj[0:1, 0:1, 0:1, 0:1])
        sample_dtype = sample.dtype
        stub = np.broadcast_to(np.zeros((), dtype=sample_dtype), target_shape)
        formatted = _nrrd_writer._handle_header(stub, dict(nrrd_header), index_order="F")
        nrrd_header.update(formatted)

        # In NRRD Fortran order, the last axis varies slowest; write contiguous slabs along it.
        last_total = target_shape[-1]
        elem_per_slice = max(1, int(np.prod(target_shape[:-1], dtype=np.int64))) * max(1, sample_dtype.itemsize)
        step = max(1, min(last_total, (48 * 1024 * 1024) // elem_per_slice))
        last_src_axis = 3 if channel == 3 else order[2]
        reverse_last = False if channel == 3 else negative[2]

        with target_path.open("wb") as fh:
            _nrrd_writer._write_header(fh, formatted)
            for out_start in range(0, last_total, step):
                out_end = min(last_total, out_start + step)
                if reverse_last:
                    in_slice = slice(last_total - out_end, last_total - out_start)
                else:
                    in_slice = slice(out_start, out_end)
                selectors = [slice(None), slice(None), slice(None), slice(None)]
                selectors[last_src_axis] = in_slice
                raw_slab = np.asanyarray(src_image.dataobj[tuple(selectors)])
                moved_slab = np.moveaxis(reorder(raw_slab), -1, channel)
                fh.write(moved_slab.tobytes(order="F"))
        return sample_dtype

    dwi_path = Path(raw.dwi_path)
    cached_nii = ensure_cached_uncompressed_nifti(dwi_path) if dwi_path.name.lower().endswith(".nii.gz") else None
    if cached_nii is not None or not dwi_path.name.lower().endswith(".nii.gz"):
        src_img = nib.load(str(cached_nii or dwi_path), mmap="r", keep_file_open=False)
        native_dtype = _stream_write_raw_nrrd(Path(dwi), src_img, header)
    else:
        expected_uncompressed = int(np.prod(image.shape, dtype=np.int64)) * max(1, image.get_data_dtype().itemsize)
        if expected_uncompressed <= 128 * 1024 * 1024:
            data = np.asanyarray(image.dataobj)
            native_dtype = data.dtype
            nrrd.write(str(dwi), np.ascontiguousarray(np.moveaxis(reorder(data), -1, channel)), header)
        else:
            with tempfile.TemporaryDirectory(prefix=".ukf-nii-", dir=Path(dwi).parent) as tmp_dir:
                tmp_nii = Path(tmp_dir) / "data.nii"
                with open_fast_gzip_reader(dwi_path) as src_fh, tmp_nii.open("wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh, length=8 * 1024 * 1024)
                src_img = nib.load(str(tmp_nii), mmap="r", keep_file_open=False)
                native_dtype = _stream_write_raw_nrrd(Path(dwi), src_img, header)

    mask_header = {
        "space": header["space"],
        "space directions": directions[spatial],
        "space origin": header["space origin"],
        "kinds": ["domain"] * 3,
        "encoding": "raw",
    }
    nrrd.write(str(mask), np.ascontiguousarray(reorder(np.asanyarray(nib.load(raw.mask_path).dataobj)).astype(np.uint8)), mask_header)
    report = dict(
        policy="lossless-native-payload",
        probe_voxels=8,
        gradient_count=len(gradients),
        native_dtype=str(native_dtype),
        axis_order=list(order),
        reversed_axes=negative,
        output_origin_ras=origin_ras.tolist(),
        resampled=False,
        allow_lossy_conversion=False,
    )
    if gradient_report is not None:
        report["gradient_frame"] = gradient_report
    return report


def _set_world_gradients(raw, affine, header):
    """Check the DWIConvert gradient lines against the bvec file; rewrite their directions where DWIConvert is wrong.

    Returns None when the lines are kept verbatim (axis-aligned det < 0 grid), else a short receipt.
    """
    import numpy as np
    from shared import gradient_frame

    keys = ["DWMRI_gradient_%04d" % index for index in range(len(
        [key for key in header if key.startswith("DWMRI_gradient_")]))]
    if any(key not in header for key in keys):
        raise ValueError("DWIConvert gradient keys are not numbered 0000 to N-1")
    values = [header[key] for key in keys]
    bvals, bvecs = _fsl_gradient_table(raw)
    try:
        gradient_frame.check_nrrd_gradients_match_fsl(values, bvals, bvecs)
    except ValueError as error:
        raise ValueError(f"UKF gradient conversion: {error}") from error
    space = header["space"]
    frame = np.asarray(header.get("measurement frame", np.eye(3)), dtype=float).T  # pynrrd rows -> columns
    world = gradient_frame.fsl_to_world(bvecs, affine)
    written = gradient_frame.nrrd_to_world([gradient_frame.parse_nrrd_vector(value) for value in values], space, frame)
    moving = np.linalg.norm(written, axis=1) > gradient_frame.ZERO_GRADIENT_NORM
    before = gradient_frame.axis_angle_degrees(written[moving], world[moving]) if moving.any() else np.zeros(1)
    if _dwiconvert_gradients_valid(affine):
        if before.max() > gradient_frame.KEEP_TOLERANCE_DEG:
            raise ValueError(
                f"DWIConvert gradients differ from the FSL rule by {before.max():.3f} degrees on an axis-aligned "
                "grid; the converter behaves differently from the measured version. UKF was not run.")
        return None
    updated = gradient_frame.nrrd_gradient_values_for_world(values, world, space, frame)
    for key, value in zip(keys, updated):
        header[key] = value
    return dict(rule="g = R F b (FSL bvecs, MRtrix -fslgrad)", source="bvec file",
                kept="DWIConvert geometry and per-line norm (b-value scaling)",
                dwiconvert_max_error_deg=float(before.max()))


def _ukf_uses_gpu(*args, **kwargs) -> bool:
    """Return True when UKF execution resolves to CUDA so cohort admission reserves a GPU slot."""
    try:
        gpu_opts = parse_gpu_options(kwargs.get("options"))
    except Exception:
        return False
    req = _normalize_ukf_device_request(kwargs.get("device", "auto"))
    if gpu_opts.get("useGPU") is False or req in {"cpu", "mps"}:
        return False
    try:
        binary = resolve_executable(kwargs.get("executable"))
    except Exception:
        return False
    if _ukf_binary_has_compiled_cuda(binary) is False:
        return False
    explicit_gpu = req.startswith("cuda") or gpu_opts.get("useGPU") is True
    if not explicit_gpu and not _cuda_runtime_available_for_ukf(binary):
        return False
    try:
        from shared.external_process import merged_startup_environment
        from shared.pipeline import build_cli_environment

        env = ukf_gpu_process_environment(
            merged_startup_environment(build_cli_environment(str(binary))),
            binary,
        )
        schema, _, _ = inspect_schema(binary, env)
    except Exception:
        # Fail closed when the binary is not known CPU-only and GPU execution is requested
        # or CUDA is available: a transient --xml probe timeout must not admit a subsequent
        # retry inside run_ukf() with 0 reserved GPU slots.
        return True
    try:
        use_gpu, _, _ = resolve_ukf_gpu_dispatch(req, gpu_opts, schema, binary)
        return bool(use_gpu)
    except Exception:
        return False


@scheduled_stage('ukf', gpu=_ukf_uses_gpu)
def run_ukf(
    raw,
    directory,
    *,
    options=None,
    executable=None,
    converter=None,
    device="auto",
    force=False,
    seed_mask_path=None,
    log=None,
):
    from shared.external_process import merged_startup_environment
    from shared.pipeline import build_cli_environment

    gpu_opts = parse_gpu_options(options)
    selected = parameters(options)
    binary = resolve_executable(executable)
    converter = resolve_executable(converter, name="DWIConvert")
    products = paths(raw.subject_id, Path(directory).expanduser().resolve())
    products["manifest"].parent.mkdir(parents=True, exist_ok=True)
    before = signature(raw, options, binary, converter, device=device)
    if not force and current(raw, directory, options, binary, converter, device=device, _signature=before):
        return dict(status="cached", outputs=products)
    started = time.monotonic()
    env = ukf_gpu_process_environment(merged_startup_environment(build_cli_environment(str(binary))), binary)
    schema, xml, probe_code = inspect_schema(binary, env)
    use_gpu, gpu_device, gpu_precision = resolve_ukf_gpu_dispatch(device, gpu_opts, schema, binary)
    build_command(
        binary,
        "DWI.nrrd",
        "Mask.nrrd",
        "tracts.vtk",
        selected,
        schema,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_precision=gpu_precision,
    )
    inputs, effective = _validated_inputs(raw)
    mem_snapshot = _host_memory_snapshot()
    eff_threads = resolve_adaptive_cpu_num_threads(selected, inputs, use_gpu=use_gpu, executable=binary)
    execution_selected = replace(selected, numThreads=eff_threads) if eff_threads != selected.numThreads else selected
    record = dict(
        status="running",
        signature=before,
        inputs=inputs,
        reference=REFERENCE,
        schema_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        schema_probe_exit_code=probe_code,
        gpu=dict(enabled=use_gpu, device=gpu_device if use_gpu else None, precision=gpu_precision if use_gpu else None),
        effective_cpu_num_threads=None if use_gpu else eff_threads,
        resource_preflight=dict(
            estimated_peak_ram_gib=round(_estimate_ukf_peak_ram_gib(inputs, executable=binary, selected=execution_selected), 2),
            host_memory_snapshot=mem_snapshot,
        ),
        commands=[],
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    def save():
        temporary = products["manifest"].with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(products["manifest"])

    save()
    products["schema"].write_text(xml, encoding="utf-8")
    try:
        tracking_raw = write_effective_mask(raw, effective, products["effective_mask"])
        record["effective_mask"] = dict(path=str(products["effective_mask"]),
                                        sha256=digest(products["effective_mask"]))
        save()
        cpu_mem_warning = _ukf_cpu_memory_runtime_warning(
            inputs, use_gpu=use_gpu, executable=binary, selected=execution_selected
        )
        if cpu_mem_warning:
            inputs["warnings"] = [*inputs["warnings"], cpu_mem_warning]
        if log:
            if not use_gpu and eff_threads != selected.numThreads:
                ram_val = _host_ram_gib()
                ram_note = f" on {ram_val:.1f} GiB host RAM" if ram_val is not None else ""
                avail_val = mem_snapshot.get("available_ram_gib")
                commit_val = mem_snapshot.get("commit_available_gib")
                headroom_bits = []
                if avail_val is not None:
                    headroom_bits.append(f"{avail_val:.1f} GiB avail")
                if commit_val is not None:
                    headroom_bits.append(f"{commit_val:.1f} GiB commit avail")
                headroom_note = f" ({', '.join(headroom_bits)})" if headroom_bits else ""
                log(
                    f"[UKF] Adaptive CPU thread cap: --numThreads {eff_threads} "
                    f"(default -1{ram_note}{headroom_note}, {inputs.get('volumes', 0)} volumes, "
                    f"{inputs.get('effective_brain_voxels', 0):,} mask voxels)"
                )
            for warning in inputs['warnings']:
                log('[UKF] '+warning)
        converter_env = merged_startup_environment(build_cli_environment(str(converter)))
        from shared.external_process import (
            copy_file_atomic,
            needs_windows_ascii_staging,
            windows_ascii_temp_dir,
        )

        use_ascii_work = needs_windows_ascii_staging(
            products["manifest"].parent,
            raw.bval_path,
            raw.bvec_path,
        )
        work_ctx = (
            windows_ascii_temp_dir(prefix="ukf_")
            if use_ascii_work
            else tempfile.TemporaryDirectory(prefix=".ukf-", dir=products["manifest"].parent)
        )
        with work_ctx as temporary:
            work = Path(temporary)
            dwi, mask, tracts = work / "DWI.nrrd", work / "Mask.nrrd", work / "tracts.vtk"
            seeds_nrrd = (work / "Seeds.nrrd") if seed_mask_path is not None else None
            probe_path, converted_probe = work / "geometry_probe.nii.gz", work / "geometry_probe.nrrd"
            probe = write_conversion_probe(raw, probe_path)
            if use_ascii_work:
                cli_bval = work / "input.bval"
                cli_bvec = work / "input.bvec"
                shutil.copyfile(raw.bval_path, cli_bval)
                shutil.copyfile(raw.bvec_path, cli_bvec)
            else:
                cli_bval = raw.bval_path
                cli_bvec = raw.bvec_path
            conversion = [
                str(converter),
                "--conversionMode",
                "FSLToNrrd",
                "--inputVolume",
                str(probe_path),
                "--inputBValues",
                str(cli_bval),
                "--inputBVectors",
                str(cli_bvec),
                "--outputVolume",
                str(converted_probe),
            ]
            command = build_command(
                binary,
                dwi,
                mask,
                tracts,
                execution_selected,
                schema,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                gpu_precision=gpu_precision,
                seeds_file=seeds_nrrd,
                seed_labels="1" if seeds_nrrd is not None else None,
            )
            record["commands"] = [conversion, command]
            save()
            for index, cmd in enumerate((conversion, command)):
                log_path = products["manifest"].parent / ("ukf_" + str(index) + ".log")
                if log:
                    if index == 0:
                        log("[UKF] Converting gradients with DWIConvert")
                    elif use_gpu:
                        log(f"[UKF] Running UKFTractography (GPU device={gpu_device}, precision={gpu_precision})")
                    else:
                        log("[UKF] Running official UKFTractography")
                with log_path.open("wb") as stream:
                    from shared.external_process import native_arm64_argv, subprocess_no_window_kwargs

                    stage_env = converter_env if index == 0 else env
                    exec_cmd = native_arm64_argv(cmd, stage_env)
                    process = subprocess.Popen(exec_cmd, stdout=stream, stderr=subprocess.STDOUT,
                                               env=stage_env,
                                               **subprocess_no_window_kwargs())
                    try:
                        if process.wait():
                            raise RuntimeError("UKF stage failed; see " + str(log_path))
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                if index == 0:
                    record["conversion"] = restore_lossless_payload(tracking_raw, probe, converted_probe, dwi, mask)
                    if seeds_nrrd is not None:
                        import nibabel as _nib
                        import nrrd as _nrrd
                        import numpy as _np

                        _, mask_hdr = _nrrd.read(str(mask))
                        order = tuple(int(a) for a in record["conversion"]["axis_order"])
                        neg = list(record["conversion"]["reversed_axes"])
                        seed_arr = (_np.asanyarray(_nib.load(str(Path(seed_mask_path))).dataobj) > 0).astype(_np.uint8)
                        seed_arr = _np.transpose(seed_arr, order)
                        for ax_i, rev in enumerate(neg):
                            if rev:
                                seed_arr = _np.flip(seed_arr, ax_i)
                        seed_hdr = {
                            "space": mask_hdr["space"],
                            "space directions": mask_hdr["space directions"],
                            "space origin": mask_hdr["space origin"],
                            "kinds": ["domain"] * 3,
                            "encoding": "raw",
                        }
                        _nrrd.write(str(seeds_nrrd), _np.ascontiguousarray(seed_arr), seed_hdr)
                    save()
                elif use_gpu:
                    gpu_log = log_path.read_text(encoding="utf-8", errors="replace").lower()
                    cpu_fallback_markers = (
                        "falling back",
                        "this build has no cuda support",
                        "built without cuda",
                        "using cpu",
                    )
                    if any(marker in gpu_log for marker in cpu_fallback_markers) or (
                        binary.name.lower().startswith(("ukf-gpu", "ukftractography_gpu"))
                        and "tractography done on device" not in gpu_log
                    ):
                        record["gpu"] = dict(enabled=False, device=None, precision=None, fallback_to_cpu=True)
                        save()
                        raise RuntimeError("UKF GPU execution was not confirmed or fell back to CPU; see " + str(log_path))
            # Release multi-GB intermediate NRRD buffers immediately after UKF completes so
            # TRK export does not require double scratch disk space on modest system drives.
            dwi.unlink(missing_ok=True)
            mask.unlink(missing_ok=True)
            polydata_bundle = _load_validated_vtk_polydata(tracts)
            qc = output_qc(tracts, tracking_raw, selected, _polydata_bundle=polydata_bundle)
            qc['input_validation'] = inputs
            qc['effective_mask'] = record['effective_mask']
            if not qc["automatic_qc_passed"]:
                products["qc"].write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
                failed = products["manifest"].parent / (work.name + "-failed.vtk")
                shutil.copyfile(tracts, failed)
                record["failed_tractogram"] = str(failed)
                raise RuntimeError("UKF geometry/native-mask QC failed; see " + str(failed))
            export_trk(tracts, raw.dwi_path, work / "tracts.trk", _polydata_bundle=polydata_bundle)
            qc["vtk_trk_verified_streamlines"] = verify_trk(tracts, work / "tracts.trk", _polydata_bundle=polydata_bundle)
            if signature(raw, options, binary, converter, device=device) != before:
                raise RuntimeError("UKF inputs or executable changed during execution")
            if use_ascii_work:
                copy_file_atomic(tracts, products["tractogram"])
                copy_file_atomic(work / "tracts.trk", products["tractogram_trk"])
            else:
                shutil.move(tracts, products["tractogram"])
                shutil.move(work / "tracts.trk", products["tractogram_trk"])
        products["qc"].write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        record.update(
            status="completed",
            artifacts={key: digest(path) for key, path in products.items() if key != "manifest"},
            runtime_seconds=time.monotonic() - started,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        return dict(status="completed", outputs=products)
    except BaseException as exc:
        record.update(
            status="cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=str(exc),
            runtime_seconds=time.monotonic() - started,
        )
        save()
        raise
