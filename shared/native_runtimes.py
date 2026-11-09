"""Platform-native tool bundle installer, independent of Conda/WSL/MSYS2.

A future release may supply *qualified* Windows, Linux, and macOS artifacts.
This layer never pretends that a missing/unpublished build is supported.
Bundles are pinned by SHA256, extracted without symlinks/path traversal, and
installed to a user-owned runtime root. It does not compile or change the host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile


class NativeRuntimeUnavailable(RuntimeError):
    """The requested native tool has no qualified build for this platform."""


def host_platform(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64", "x64": "x86_64"}.get(machine, machine)
    return system, arch


def runtime_root():
    from shared.model_assets import default_user_model_assets_root
    return Path(os.environ.get("SLICERDSPACE_RUNTIME_ROOT") or
                default_user_model_assets_root().parent / "runtimes").expanduser().resolve()


def installed_bundle_bin_dirs(name, *, system=None, machine=None, root=None):
    """Return bin directories of installed native runtime bundles matching name and host platform."""
    system, machine = host_platform(system, machine)
    base = Path(root or runtime_root()).expanduser().resolve()
    if not base.is_dir():
        return ()
    prefix = f"{name}-"
    suffix = f"-{system.lower()}-{machine}"
    dirs = []
    try:
        children = sorted(base.iterdir(), reverse=True)
    except OSError:
        return ()
    for child in children:
        if child.is_dir() and child.name.startswith(prefix) and child.name.endswith(suffix):
            bin_dir = child / "bin"
            if (child / "native-runtime.json").is_file() and bin_dir.is_dir():
                dirs.append(bin_dir.resolve())
    return tuple(dirs)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_bundle(catalog, name, *, system=None, machine=None):
    """Pick the explicit host build; never substitute a WSL or other-OS build."""
    system, machine = host_platform(system, machine)
    records = catalog.get("bundles", ())
    matches = [r for r in records if r.get("name") == name
               and r.get("system") == system and r.get("machine") == machine]
    if len(matches) != 1:
        raise NativeRuntimeUnavailable(
            f"No unique qualified native {name} runtime for {system}/{machine}. "
            "An external Linux/WSL runtime is not an acceptable substitute.")
    r = matches[0]
    if r.get("published") is not True:
        raise NativeRuntimeUnavailable(f"Native {name} runtime for {system}/{machine} has not been published.")
    if r.get("format") not in ("zip", "tar.gz"):
        raise ValueError("Unsupported native runtime archive format")
    if not isinstance(r.get("sha256"), str) or len(r["sha256"]) != 64:
        raise ValueError("Native runtime bundle lacks pinned SHA256")
    if not isinstance(r.get("version"), str) or not r["version"].strip():
        raise ValueError("Native runtime bundle lacks version")
    if not isinstance(r.get("commands"), list) or not r["commands"]:
        raise ValueError("Native runtime bundle lacks command list")
    if not isinstance(r.get("license_url"), str) or not r["license_url"]:
        raise ValueError("Native runtime bundle lacks license URL")
    return r


def _safe_relpath(raw):
    if "\\" in raw or ":" in raw:
        raise ValueError(f"Unsafe archive member: {raw}")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        raise ValueError(f"Unsafe archive member: {raw}")
    return path


def _extract(archive, dest, fmt):
    """Extract regular files only with exact paths and no symlink handling."""
    seen = set()
    total = 0
    maximum = 8 * 1024**3
    if fmt == "zip":
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                p = _safe_relpath(member.filename)
                mode = (member.external_attr >> 16)
                if stat.S_ISLNK(mode) or (not member.is_dir() and mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise ValueError("Native runtime archive must not contain links or special files")
                if p in seen:
                    raise ValueError("Duplicate native runtime archive member")
                seen.add(p)
                target = dest.joinpath(*p.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.file_size
                if total > maximum:
                    raise ValueError("Native runtime archive exceeds unpacked size limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out)
                if mode & 0o111:
                    target.chmod(target.stat().st_mode | 0o111)
    else:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                p = _safe_relpath(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError("Native runtime archive must contain regular files and directories only")
                if p in seen:
                    raise ValueError("Duplicate native runtime archive member")
                seen.add(p)
                target = dest.joinpath(*p.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.size
                if total > maximum:
                    raise ValueError("Native runtime archive exceeds unpacked size limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out)
                if member.mode & 0o111:
                    target.chmod(target.stat().st_mode | 0o111)


def install_bundle(record, *, archive=None, root=None, accept_terms=False):
    """Install one verified bundle. Never overwrite a ready or unknown prefix.

    A GUI caller should display the exact licence URL, size and SHA256 first.
    Passing accept_terms=True records user consent, not a licensing exemption.
    """
    if not accept_terms:
        raise PermissionError("Review and accept the tool licence before installing")
    for key in ("name", "version", "system", "machine"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", record[key]):
            raise ValueError(f"Invalid native runtime {key}")
    root = Path(root or runtime_root()).expanduser().resolve()
    dest = root / (record["name"] + "-" + record["version"] + "-" + record["system"].lower()
                   + "-" + record["machine"])
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Refusing to replace native runtime directory: {dest}")
    root.mkdir(parents=True, exist_ok=True)
    download = None
    try:
        if archive is None:
            if not record.get("url", "").startswith("https://"):
                raise NativeRuntimeUnavailable("No HTTPS download URL for native runtime")
            fd, download = tempfile.mkstemp(prefix="native-download-", suffix=".part", dir=root)
            os.close(fd)
            with urllib.request.urlopen(record["url"], timeout=90) as src, open(download, "wb") as out:
                shutil.copyfileobj(src, out)
            archive = download
        if _sha256(archive) != record["sha256"]:
            raise ValueError("Native runtime SHA256 mismatch")
        with tempfile.TemporaryDirectory(prefix="native-stage-", dir=root) as temp:
            stage = Path(temp) / "payload"
            stage.mkdir()
            _extract(archive, stage, record["format"])
            paths = []
            for command in record["commands"]:
                rel = _safe_relpath(command)
                path = stage.joinpath(*rel.parts)
                if not path.is_file():
                    raise ValueError(f"Missing native executable: {command}")
                paths.append(str(rel))
            receipt = {
                "name": record["name"], "version": record["version"],
                "system": record["system"], "machine": record["machine"],
                "archive_sha256": record["sha256"], "commands": paths,
                "license_url": record["license_url"],
            }
            (stage / "native-runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
            stage.rename(dest)
        return dest
    finally:
        if download is not None:
            Path(download).unlink(missing_ok=True)


CATALOG = Path(__file__).resolve().parents[1] / "models" / "native_runtime_bundles.json"
BUNDLE_SETTINGS = {
    "mrtrix": ("mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir"),
    "fsl": ("preprocessing_fsl_bin_dir",),
}


def load_catalog(path=None):
    return json.loads(Path(path or CATALOG).read_text(encoding="utf-8"))


def provision_bundles(names, *, root=None, archive_dir=None, accept_terms=False, execute=False, catalog_path=None):
    """Plan or install verified native runtime bundles (`mrtrix`, `fsl`) without WSL/Conda."""
    selected = tuple(dict.fromkeys(names))
    if not selected or any(name not in BUNDLE_SETTINGS for name in selected):
        raise ValueError("Choose one or more supported native runtime bundles (mrtrix, fsl)")
    catalog = load_catalog(catalog_path)
    base = Path(root or runtime_root()).expanduser().resolve()
    records = {name: select_bundle(catalog, name) for name in selected}
    prefixes = {
        name: str(base / f"{r['name']}-{r['version']}-{r['system'].lower()}-{r['machine']}")
        for name, r in records.items()
    }
    plan = {
        "kind": "slicerdsurfer_runtime_setup",
        "method": "native_bundle",
        "status": "planned",
        "root": str(base),
        "bundles": list(selected),
        "prefixes": prefixes,
        "download_bytes": sum(int(r.get("bytes", 0)) for r in records.values()),
        "external_dependencies": {},
        "changes_slicer_python": False,
    }
    if not execute:
        return plan
    if not accept_terms:
        raise PermissionError("Review and accept the native runtime licences before installing")
    settings = {}
    installed = {}
    for name, record in records.items():
        dest = Path(prefixes[name])
        receipt_file = dest / "native-runtime.json"
        reused = False
        if dest.is_dir() and receipt_file.is_file():
            try:
                existing = json.loads(receipt_file.read_text(encoding="utf-8"))
                reused = (
                    existing.get("archive_sha256") == record["sha256"]
                    and all(dest.joinpath(*_safe_relpath(cmd).parts).is_file() for cmd in record["commands"])
                )
            except (OSError, ValueError):
                reused = False
        if not reused:
            archive = None
            if archive_dir is not None:
                cand = Path(archive_dir) / record.get("artifact", "")
                if cand.is_file():
                    archive = cand
            dest = install_bundle(record, archive=archive, root=base, accept_terms=True)
        bin_dir = str((dest / "bin").resolve())
        for key in BUNDLE_SETTINGS[name]:
            settings[key] = bin_dir
        installed[name] = str(dest)
    plan.update(status="completed", external_dependencies=settings, installed_prefixes=installed)
    return plan
