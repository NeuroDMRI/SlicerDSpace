"""User-local optional CLI environments; never install into Slicer's Python."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request

from shared.external_process import external_python_environment
from shared.model_assets import default_user_model_assets_root
from shared.platform_info import host_machine

FSL_CHANNEL = "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/"
MINIFORGE_URL = "https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh"
MINIFORGE_SHA256 = "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"
MINIFORGE_MACOS_ARM64 = "Miniforge3-26.7.2-0-MacOSX-arm64.sh"
MINIFORGE_MACOS_ARM64_SHA256 = "d70bfa2e97afcda96927c9b9ca0e2316cb7750e4ce651c94388267cbe9588711"
MINIFORGE_BOOTSTRAP = "miniforge-26.7.2-0"
FSL_TERMS_URL = "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html"
# B4: on Darwin only these profiles have a qualified build, on both host architectures.
DARWIN_PROFILES = frozenset({"mrtrix", "fsl_cpu"})
# fsl-eddy 2602.0 build string per conda subdir (design fact sheet).
FSL_EDDY_BUILD = {"linux-64": "hb35a746_0", "osx-64": "h7ffe42b_0", "osx-arm64": "hed4507f_0"}
PROFILES = {
    "pnl_cnn": {"directory": "pnl-cnn-torch-v0.1.0", "packages": ("python=3.11", "pip=24.3.1", "ants=2.3.0"),
                "channels": ("pnlbwh", "conda-forge"),
                "commands": ("python", "antsRegistrationSyNQuick.sh", "antsRegistration", "antsApplyTransforms", "ImageMath", "PrintHeader"),
                "settings": ()},
    "mrtrix": {"directory": "mrtrix-3.0.8", "packages": ("mrtrix3=3.0.8=h07fc403_0", "python=3.12", "libclang13=22.1.6", "libclang-cpp22.1=22.1.6"),
               "channels": ("https://conda.anaconda.org/mrtrix3", "conda-forge"), "commands": ("mrconvert", "dwi2mask", "dwifslpreproc", "tckgen"),
               "settings": ("mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir")},
    "fsl_cpu": {"directory": "fsl-2602.0-cpu", "packages": ("fsl-eddy=2602.0=hb35a746_0", "fsl-topup=2203.5", "python=3.12"),
                "channels": (FSL_CHANNEL, "conda-forge"), "commands": ("eddy", "eddy_cpu", "topup"),
                "settings": ("preprocessing_fsl_bin_dir",)},
    "fsl_cuda": {"directory": "fsl-2602.0-cuda", "packages": ("fsl-eddy=2602.0=hb35a746_0", "fsl-eddy-cuda=2602.0=h1a432fe_1", "fsl-topup=2203.5", "python=3.12"),
                 "channels": (FSL_CHANNEL, "conda-forge"), "commands": ("eddy", "eddy_cpu", "eddy_cuda", "topup"),
                 "settings": ("preprocessing_fsl_bin_dir",)},
}


def runtime_root():
    return Path(os.environ.get("SLICERDSPACE_RUNTIME_ROOT") or
                default_user_model_assets_root().parent / "runtimes").expanduser().resolve()


def conda_subdir(system=None, machine=None):
    """Expected conda subdir from the OS and the host (not Slicer's) architecture; None when unqualified."""
    system, machine = system or platform.system(), (machine or host_machine()).lower()
    if system == 'Linux' and machine in {'x86_64', 'amd64'}:
        return 'linux-64'
    if system == 'Darwin' and machine in {'x86_64', 'amd64'}:
        return 'osx-64'
    if system == 'Darwin' and machine in {'arm64', 'aarch64'}:
        return 'osx-arm64'
    return None


def supported_platform(system=None, machine=None, *, profiles=None):
    system, machine = system or platform.system(), (machine or host_machine()).lower()
    if system == 'Linux' and machine in {'x86_64', 'amd64'}:
        return True
    if system == 'Darwin' and machine in {'x86_64', 'amd64', 'arm64', 'aarch64'} and profiles and set(profiles) <= DARWIN_PROFILES:
        version = platform.mac_ver()[0]
        return (tuple(map(int, version.split('.')[:2])) >= (14, 5)) if version else platform.system() != 'Darwin'
    return False


def profile_spec(name):
    spec = dict(PROFILES[name])
    if platform.system() == 'Darwin' and name == 'mrtrix':
        spec['packages'] = ('mrtrix3=3.0.8=hee5d243_0', 'python=3.12')
    build = FSL_EDDY_BUILD.get(conda_subdir())
    if name == 'fsl_cpu' and build:
        spec['packages'] = (f'fsl-eddy=2602.0={build}',) + tuple(spec['packages'][1:])
    return spec


def bootstrap_artifact():
    if platform.system() == 'Darwin':
        if host_machine().lower() in {'arm64', 'aarch64'}:
            return MINIFORGE_URL.rsplit('/', 1)[0] + '/' + MINIFORGE_MACOS_ARM64, MINIFORGE_MACOS_ARM64_SHA256
        filename = 'Miniforge3-26.7.2-0-MacOSX-x86_64.sh'
        return (MINIFORGE_URL.rsplit('/', 1)[0] + '/' + filename,
                'b00e7798658f92721a3ae2f6b9832695ffc6baf07758894d726268055359f6c5')
    return MINIFORGE_URL, MINIFORGE_SHA256


def cuda_visible():
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    try:
        check = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                               timeout=10, env=external_python_environment())
        return check.returncode == 0 and "GPU " in check.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def cuda_candidate():
    """Cheap GUI hint only; actual GPU usability is checked in the worker."""
    return os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"} and shutil.which("nvidia-smi") is not None


def selected_profiles(node, gpu_available):
    """Only provision external tools used by the current GUI selections."""
    preproc = bool(getattr(node, "targetPreprocessing", False)) and getattr(node, "preprocessingBackend", "dipy") == "mrtrix_fsl"
    from shared.algorithm_selection import algorithms
    tracts = bool(getattr(node, "targetTractography", False)) and bool(set(algorithms(getattr(node, "orchestrationTractographyBackend", "dti_det"))) & {"csd_act", "saf"})
    conn = bool(getattr(node, "targetConnectome", False)) and getattr(node, "connectomeBackend", "python") == "mrtrix"
    mask = getattr(node, "brainMaskBackend", "provided") == "mrtrix"
    result = ["mrtrix"] if preproc or tracts or conn or mask else []
    if preproc:
        result.append("fsl_cuda" if gpu_available else "fsl_cpu")
    return tuple(result)


def install_plan(profiles, root=None):
    selected = tuple(dict.fromkeys(profiles))
    if not selected or any(profile not in PROFILES for profile in selected):
        raise ValueError("Choose one or more known managed runtime profiles")
    if "fsl_cpu" in selected and "fsl_cuda" in selected:
        raise ValueError("Choose one FSL device profile, not both")
    base = (root or runtime_root()).expanduser().resolve()
    return {"kind": "slicerdsurfer_runtime_plan", "root": str(base),
            "platform_supported": supported_platform(profiles=selected), "profiles": list(selected),
            "prefixes": {name: str(base / profile_spec(name)["directory"]) for name in selected},
            "conda_subdir": conda_subdir(),
            "fsl_license_required": any(name.startswith("fsl_") for name in selected),
            "pnl_terms_required": "pnl_cnn" in selected,
            "fsl_terms_url": FSL_TERMS_URL, "changes_slicer_python": False,
            "changes_shell_or_driver": False, "bootstrap_url": bootstrap_artifact()[0],
            "bootstrap_sha256": bootstrap_artifact()[1]}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_prefix(profile, prefix):
    spec = profile_spec(profile)
    if prefix.is_symlink():
        raise ValueError("Managed runtime prefix cannot be a symlink")
    commands = {name: prefix / "bin" / name for name in spec["commands"]}
    missing = [name for name, path in commands.items() if not path.is_file() or not os.access(path, os.X_OK)]
    if missing:
        raise RuntimeError("Runtime is incomplete: " + ", ".join(missing))
    records = [json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")]
    expected_subdir = conda_subdir()
    if any(record.get("subdir") not in {expected_subdir, "noarch"} for record in records):
        raise RuntimeError(f"Runtime built for another architecture; use Repair (expected {expected_subdir})")
    installed = {record["name"]: record["version"] for record in records}
    for requested in spec["packages"]:
        name, version, *_ = requested.split("=")
        if not installed.get(name, "").startswith(version + ".") and installed.get(name) != version:
            raise RuntimeError("Runtime package version mismatch: " + requested)
    for name, command in commands.items():
        # ELF library discovery catches incomplete transfers and wrong ABI environments.
        with command.open("rb") as stream:
            elf = stream.read(4) == b"\x7fELF"
        if elf:
            probe = subprocess.run(["ldd", str(command)], capture_output=True, text=True,
                                   timeout=30, env=external_python_environment())
            if probe.returncode or "not found" in probe.stdout:
                raise RuntimeError(f"Runtime libraries unavailable for {name}: {probe.stdout} {probe.stderr}")
    functionality = []
    if profile == 'mrtrix':
        import struct
        import tempfile
        with tempfile.TemporaryDirectory(prefix='slicerdspace-mrtrix-test-') as tmp:
            original, output = Path(tmp) / 'input.mif', Path(tmp) / 'output.mif'
            header = b'mrtrix image\ndim: 2,2,2\nvox: 1,1,1\nlayout: +0,+1,+2\ndatatype: Float32LE\nfile: . 256\nEND\n'
            original.write_bytes(header.ljust(256, b'\0') + struct.pack('<8f', *range(8)))
            subprocess.run([str(prefix / 'bin/mrconvert'), str(original), str(output), '-quiet', '-datatype', 'Float32LE'],
                           check=True, capture_output=True, timeout=60, env=external_python_environment())
            data = output.read_bytes()
            end = data.index(b'\nEND\n')
            offset_line = next(line for line in data[:end].decode('ascii').splitlines() if line.startswith('file:'))
            offset = int(offset_line.split()[-1])
            if struct.unpack('<8f', data[offset:offset + 32]) != tuple(float(i) for i in range(8)):
                raise RuntimeError('MRtrix synthetic image round-trip failed')
            functionality.append('mrconvert:synthetic_image_roundtrip')
    if profile == 'pnl_cnn':
        import tempfile
        with tempfile.TemporaryDirectory(prefix='slicerdspace-ants-test-') as tmp:
            original, output = Path(tmp) / 'input.nii.gz', Path(tmp) / 'output.nii.gz'
            python = str(prefix / 'bin/python')
            environment = external_python_environment(path_prepend=[prefix / 'bin'])
            create = ('import sys, nibabel as n, numpy as np; '
                      'n.save(n.Nifti1Image(np.arange(27,dtype=np.float32).reshape(3,3,3),np.eye(4)),sys.argv[1])')
            subprocess.run([python, '-c', create, str(original)], check=True, capture_output=True,
                           timeout=60, env=environment)
            subprocess.run([str(prefix / 'bin/ImageMath'), '3', str(output), 'm', str(original), '1'],
                           check=True, capture_output=True, timeout=60, env=environment)
            compare = ('import sys, nibabel as n, numpy as np; a,b=[n.load(p) for p in sys.argv[1:]]; '
                       'np.testing.assert_array_equal(np.asanyarray(a.dataobj),np.asanyarray(b.dataobj)); '
                       'np.testing.assert_allclose(a.affine,b.affine,atol=1e-6)')
            subprocess.run([python, '-c', compare, str(original), str(output)], check=True, capture_output=True,
                           timeout=60, env=environment)
            functionality.append('ants:synthetic_image_roundtrip')
    return {"external_dependencies": {key: str(prefix / "bin") for key in spec["settings"]},
            "command_sha256": {name: file_hash(path) for name, path in commands.items()},
            "package_versions": installed, "functional_checks": functionality,
            "scientific_validation": "not_inferred_from_installation"}


def _run(argv, extra_env=None):
    print("[Runtime setup] " + repr(argv), flush=True)
    subprocess.run(argv, check=True, env=external_python_environment(extra_env))


@contextmanager
def setup_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another managed runtime installation is active") from exc
        yield


def _path_conda_platform(executable):
    """Conda platform reported by ``conda info --json``; None when it cannot be read."""
    try:
        check = subprocess.run([executable, "info", "--json"], capture_output=True, text=True,
                               timeout=60, env=external_python_environment())
        return json.loads(check.stdout).get("platform") if check.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
        return None


def ensure_conda(root, explicit=None):
    if explicit:
        executable = Path(explicit).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError("Configured conda executable is unavailable: " + str(executable))
        return str(executable)
    subdir = conda_subdir()
    if subdir is None:
        raise RuntimeError("No qualified conda architecture for this platform")
    existing = shutil.which("conda")
    # B4: a conda of another platform (for example osx-64 under Rosetta on Apple Silicon) would build the wrong packages.
    if existing and _path_conda_platform(existing) == subdir:
        return existing
    name = f"{MINIFORGE_BOOTSTRAP}-{subdir}"
    bootstrap = root / name
    marker = root / (name + ".bootstrap.json")
    executable = bootstrap / "bin" / "conda"
    if bootstrap.is_symlink():
        raise ValueError("Managed bootstrap prefix cannot be a symlink")
    if bootstrap.exists():
        receipt = json.loads(marker.read_text()) if marker.is_file() else {}
        if receipt.get("kind") != "slicerdsurfer_miniforge_bootstrap" or receipt.get("prefix") != str(bootstrap):
            raise RuntimeError("Refusing to repair an unmanaged bootstrap directory: " + str(bootstrap))
        if receipt.get("subdir") != subdir:
            raise RuntimeError(f"Managed bootstrap was not built for {subdir}; remove it or use Repair: " + str(bootstrap))
    if executable.is_file():
        check = subprocess.run([str(executable), "--version"], capture_output=True,
                               timeout=30, env=external_python_environment())
        if check.returncode == 0:
            return str(executable)
    bootstrap_url, bootstrap_sha256 = bootstrap_artifact()
    installer = root / bootstrap_url.rsplit('/', 1)[-1]
    if not installer.is_file() or file_hash(installer) != bootstrap_sha256:
        part = installer.with_suffix(".part")
        print("[Runtime setup] Downloading verified Miniforge bootstrap", flush=True)
        with urllib.request.urlopen(bootstrap_url, timeout=60) as response, part.open("wb") as output:
            count = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                count += len(chunk)
                print(f"[Runtime setup] Bootstrap download: {count // (1024 * 1024)} MiB", flush=True)
        if file_hash(part) != bootstrap_sha256:
            raise RuntimeError("Miniforge checksum mismatch; downloaded file will not be executed")
        part.replace(installer)
    record = {"kind": "slicerdsurfer_miniforge_bootstrap", "prefix": str(bootstrap), "subdir": subdir,
              "url": bootstrap_url, "sha256": bootstrap_sha256}
    # The installer's update mode repairs an interrupted plugin-owned prefix.
    write_json(marker, dict(record, status="installing"))
    _run(["bash", str(installer), "-b", *( ["-u"] if bootstrap.exists() else []), "-p", str(bootstrap)])
    if not executable.is_file():
        raise RuntimeError("Miniforge did not produce its expected conda executable")
    write_json(marker, dict(record, status="ready"))
    return str(executable)


def configured_profiles(config, candidates, root=None):
    """Recognize plugin-owned paths, not arbitrary administrator environments."""
    base = (root or runtime_root()).expanduser().resolve()
    selected = []
    for name in candidates:
        spec = profile_spec(name)
        keys = spec["settings"] if name != "pnl_cnn" else ("pnl_mask_python",)
        prefix = base / spec["directory"]
        if not (base / (spec["directory"] + ".json")).is_file():
            continue
        for key in keys:
            expected = prefix / "bin/python" if key == "pnl_mask_python" else prefix / "bin"
            value = config.get(key)
            if value and Path(value).expanduser().absolute() == expected:
                selected.append(name)
                break
    return tuple(dict.fromkeys(selected))


def provision(profiles, *, root=None, conda=None, accept_fsl_license=False, accept_pnl_terms=False, repair=False, asset_root=None):
    plan = install_plan(profiles, root)
    if not plan["platform_supported"]:
        raise RuntimeError("No qualified managed package for the selected platform/profile combination")
    if platform.system() == 'Darwin' and platform.mac_ver()[0] and tuple(map(int, platform.mac_ver()[0].split('.')[:2])) < (14, 5):
        raise RuntimeError('The qualified MRtrix macOS package requires macOS 14.5 or later')
    if plan["fsl_license_required"] and not accept_fsl_license:
        raise PermissionError("Read and explicitly accept the FSL license before installing: " + FSL_TERMS_URL)
    if "pnl_cnn" in profiles and not accept_pnl_terms:
        from shared.managed_pnl import PNL_TERMS_URL
        raise PermissionError("Review the PNL source/model/template terms before installing: " + PNL_TERMS_URL)
    base = Path(plan["root"])
    settings, records = {}, {}
    with setup_lock(base):
        manager = None
        for profile in plan["profiles"]:
            spec, prefix = profile_spec(profile), Path(plan["prefixes"][profile])
            if prefix.is_symlink():
                raise ValueError("Managed runtime prefix cannot be a symlink")
            receipt_path = base / (spec["directory"] + ".json")
            old = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            if prefix.exists() and (old.get("kind") != "slicerdsurfer_managed_runtime" or old.get("profile") != profile):
                raise RuntimeError("Refusing to modify an unmanaged environment: " + str(prefix))
            if old.get("status") == "ready" and not repair:
                record = verify_prefix(profile, prefix)
                if profile == "pnl_cnn":
                    from shared.managed_pnl import verify_assets
                    record.update(verify_assets(prefix, asset_root=asset_root))
                    if record["pnl_resource_sha256"] != old.get("pnl_resource_sha256"):
                        raise RuntimeError("PNL resources changed; inspect them before reuse")
                if record["command_sha256"] != old.get("command_sha256"):
                    raise RuntimeError("Installed runtime checksum changed; inspect it before reuse")
                print("[Runtime setup] Reusing verified " + profile, flush=True)
            else:
                manager = manager or ensure_conda(base, conda)
                pending = {"kind": "slicerdsurfer_managed_runtime", "profile": profile,
                           "status": "installing", "prefix": str(prefix),
                           "fsl_license_accepted": bool(accept_fsl_license),
                           "pnl_terms_accepted": bool(accept_pnl_terms), "repair": bool(repair), "plan": plan}
                if old.get("status") == "ready":
                    write_json(base / (spec["directory"] + f".before-repair-{time.time_ns()}.json"), old)
                write_json(receipt_path, pending)
                action = "install" if (prefix / "conda-meta" / "history").is_file() else "create"
                command = [manager, action, "-y", "--solver", "libmamba", "--override-channels",
                           "--strict-channel-priority", "-p", str(prefix)]
                for channel in spec["channels"]:
                    command.extend(["-c", channel])
                if repair and action == "install":
                    command.append("--force-reinstall")
                _run([*command, *spec["packages"]], {"CONDA_SUBDIR": conda_subdir()})
                if profile == "pnl_cnn":
                    from shared.managed_pnl import provision_assets
                    provision_assets(prefix, base / "downloads", repair=repair, asset_root=asset_root)
                record = verify_prefix(profile, prefix)
                if profile == "pnl_cnn":
                    from shared.managed_pnl import verify_assets
                    record.update(verify_assets(prefix, asset_root=asset_root))
                locked = subprocess.run([manager, "list", "--explicit", "-p", str(prefix)],
                                        capture_output=True, text=True, check=True,
                                        env=external_python_environment())
                (base / (spec["directory"] + ".explicit.txt")).write_text(locked.stdout)
                if profile == "pnl_cnn":
                    pip_lock = subprocess.run([str(prefix / "bin/python"), "-m", "pip", "freeze", "--all"],
                                              capture_output=True, text=True, check=True,
                                              env=external_python_environment())
                    (base / (spec["directory"] + ".pip-freeze.txt")).write_text(pip_lock.stdout)
                write_json(receipt_path, dict(pending, **record, status="ready", completed_at=time.time()))
            settings.update(record["external_dependencies"])
            records[profile] = str(receipt_path)
    return {"kind": "slicerdsurfer_runtime_setup", "status": "completed",
            "external_dependencies": settings, "receipts": records, "plan": plan}
