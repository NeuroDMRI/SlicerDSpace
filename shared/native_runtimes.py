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


def bundled_runtime_roots():
    """Return existing bundled-runtimes directories shipped inside the installed extension."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    env_root = os.environ.get("SLICERDSPACE_BUNDLED_RUNTIMES_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend((
        repo_root / "SlicerDSpace" / "bundled-runtimes",
        repo_root / "bundled-runtimes",
    ))
    if len(repo_root.parents) >= 3:
        candidates.append(repo_root.parents[2] / "bundled-runtimes")
    roots = []
    seen = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            roots.append(resolved)
    return tuple(roots)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_tree_sha256(root_dir):
    """Compute a deterministic SHA256 over all regular files in a runtime directory (excluding native-runtime.json)."""
    root = Path(root_dir)
    entries = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "native-runtime.json":
            continue
        entries.append(f"{rel}:{_sha256(path)}\n")
    digest = hashlib.sha256()
    for line in entries:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def is_valid_installed_bundle_dir(bundle_dir, record=None):
    """Return True only if native-runtime.json and all declared command files exist."""
    bundle_dir = Path(bundle_dir)
    receipt_file = bundle_dir / "native-runtime.json"
    bin_dir = bundle_dir / "bin"
    if not bundle_dir.is_dir() or not receipt_file.is_file() or not bin_dir.is_dir():
        return False
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        commands = list(receipt.get("commands") or ())
        if record is not None:
            if receipt.get("archive_sha256") != record.get("sha256"):
                return False
            for cmd in record.get("commands") or ():
                if cmd not in commands:
                    commands.append(cmd)
        if commands:
            return all(bundle_dir.joinpath(*_safe_relpath(cmd).parts).is_file() for cmd in commands)
        return any(p.is_file() for p in bin_dir.iterdir())
    except (OSError, ValueError, TypeError):
        return False


def installed_bundle_bin_dirs(name, *, system=None, machine=None, root=None):
    """Return bin directories of installed native runtime bundles matching name and host platform."""
    system, machine = host_platform(system, machine)
    bases = (
        (Path(root).expanduser().resolve(),)
        if root is not None
        else (*bundled_runtime_roots(), runtime_root())
    )
    prefix = f"{name}-"
    suffix = f"-{system.lower()}-{machine}"
    dirs = []
    seen = set()
    for base in bases:
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), reverse=True)
        except OSError:
            continue
        for child in children:
            if child.is_dir() and child.name.startswith(prefix) and child.name.endswith(suffix):
                bin_dir = child / "bin"
                if is_valid_installed_bundle_dir(child):
                    resolved = bin_dir.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        dirs.append(resolved)
    return tuple(dirs)


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
    folder_name = (record["name"] + "-" + record["version"] + "-" + record["system"].lower()
                   + "-" + record["machine"])
    dest = root / folder_name
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Refusing to replace native runtime directory: {dest}")
    root.mkdir(parents=True, exist_ok=True)
    download = None
    try:
        if archive is None:
            if not record.get("url", "").startswith("https://"):
                raise NativeRuntimeUnavailable("No HTTPS download URL for native runtime")
            suffix = ".zip" if record["url"].lower().endswith(".zip") else ".part"
            fd, download = tempfile.mkstemp(prefix="native-download-", suffix=suffix, dir=root)
            os.close(fd)
            with urllib.request.urlopen(record["url"], timeout=90) as src, open(download, "wb") as out:
                shutil.copyfileobj(src, out)
            archive = download
        digest = _sha256(archive)
        valid_digests = {record["sha256"]}
        if isinstance(record.get("unified_sha256"), str) and len(record["unified_sha256"]) == 64:
            valid_digests.add(record["unified_sha256"])
        archive_matched = digest in valid_digests
        tree_sha256 = record.get("tree_sha256")
        has_tree_sha256 = isinstance(tree_sha256, str) and len(tree_sha256) == 64
        if not archive_matched and not has_tree_sha256:
            raise ValueError("Native runtime SHA256 mismatch")
        archive_str = str(archive).lower()
        fmt = "zip" if archive_str.endswith(".zip") else ("tar.gz" if archive_str.endswith((".tar.gz", ".tgz")) else record["format"])
        with tempfile.TemporaryDirectory(prefix="native-stage-", dir=root) as temp:
            stage = Path(temp) / "payload"
            stage.mkdir()
            _extract(archive, stage, fmt)
            first_cmd = _safe_relpath(record["commands"][0])
            if not stage.joinpath(*first_cmd.parts).is_file():
                for cand in stage.rglob(folder_name):
                    if cand.is_dir() and cand.joinpath(*first_cmd.parts).is_file():
                        stage = cand
                        break
            if not archive_matched:
                if not has_tree_sha256 or bundle_tree_sha256(stage) != tree_sha256:
                    raise ValueError("Native runtime SHA256 mismatch")
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
            if has_tree_sha256:
                receipt["tree_sha256"] = tree_sha256
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
    downloaded_by_url = {}
    try:
        for name, record in records.items():
            folder_name = f"{record['name']}-{record['version']}-{record['system'].lower()}-{record['machine']}"
            candidate_dirs = [Path(prefixes[name])]
            if root is None:
                candidate_dirs = [b / folder_name for b in bundled_runtime_roots()] + candidate_dirs
            dest = Path(prefixes[name])
            reused = False
            for cand_dest in candidate_dirs:
                if is_valid_installed_bundle_dir(cand_dest, record=record):
                    dest = cand_dest
                    reused = True
                    break
            if not reused:
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                archive = None
                if archive_dir is not None:
                    for art_key in ("artifact", "unified_artifact"):
                        art_name = record.get(art_key, "")
                        if art_name and (Path(archive_dir) / art_name).is_file():
                            archive = Path(archive_dir) / art_name
                            break
                if archive is None and record.get("url", "").startswith("https://"):
                    url = record["url"]
                    if url in downloaded_by_url and Path(downloaded_by_url[url]).is_file():
                        archive = downloaded_by_url[url]
                    else:
                        base.mkdir(parents=True, exist_ok=True)
                        suffix = ".zip" if url.lower().endswith(".zip") else (".tar.gz" if url.lower().endswith((".tar.gz", ".tgz")) else ".part")
                        fd, shared_dl = tempfile.mkstemp(prefix="native-shared-download-", suffix=suffix, dir=base)
                        os.close(fd)
                        with urllib.request.urlopen(url, timeout=90) as src, open(shared_dl, "wb") as out:
                            shutil.copyfileobj(src, out)
                        downloaded_by_url[url] = Path(shared_dl)
                        archive = downloaded_by_url[url]
                dest = install_bundle(record, archive=archive, root=base, accept_terms=True)
            bin_dir = str((dest / "bin").resolve())
            for key in BUNDLE_SETTINGS[name]:
                settings[key] = bin_dir
            installed[name] = str(dest)
    finally:
        for shared_dl in downloaded_by_url.values():
            Path(shared_dl).unlink(missing_ok=True)
    plan.update(status="completed", external_dependencies=settings, installed_prefixes=installed)
    return plan
