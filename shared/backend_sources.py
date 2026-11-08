"""Pinned optional algorithm sources in the user cache, without Git or pip VCS installs."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

SOURCES = {
    "wmql": dict(repository="demianw/tract_querier", revision="5c69c594d489859e08485110678dca7f6f4cba40",
                 sha256="63e84b6f99c8b992d72ea9ed965e8ac7da2323504122753354419cfc05965549",
                 bytes=34202710, setting="tract_querier_root", entry="scripts/cli_tract_querier.py",
                 paths=("tract_querier", "scripts", "pyproject.toml", "setup.py", "setup.cfg", "license.rst", "README.md"),
                 requirements=("numpy>=1.26", "nibabel>=4", "vtk>8.1")),
    "wma": dict(repository="SlicerDMRI/whitematteranalysis", revision="4532b911386d2d1bc7ebba63fdd13e65b1660623",
                sha256="51ce595ac4f0f8ae761a532e8893484bee3bb1b7810dbc937cb127c2228d4c19",
                bytes=2477247, setting="wma_root", entry="bin/wm_apply_ORG_atlas_to_subject.sh",
                paths=("whitematteranalysis", "bin", "utilities", "setup.py", "setup.cfg", "pyproject.toml", "requirements.txt", "License.txt", "README.md"),
                requirements=("numpy", "scipy", "nibabel", "vtk", "joblib", "xlrd", "matplotlib", "pandas", "Pillow", "statsmodels")),
    "tractcloud": dict(repository="SlicerDMRI/TractCloud", revision="94de6275d07d7b5eeb50e4cd4d6e85fde29d5fb0",
                       sha256="55d111c96945dee36b9aa1ff07508d0d2aa67b62656613365a3dacb474a76cda",
                       bytes=64992, setting="tractcloud_root", entry="src/tractcloud/cli.py",
                       paths=("src", "pyproject.toml", "LICENSE", "README.md"),
                       requirements=("numpy>=1.21", "vtk>=9", "torch>=2.3.1")),
}


def source_root():
    from shared.model_assets import default_user_model_assets_root
    return default_user_model_assets_root().parent / "algorithm-sources"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_path(name, root=None):
    return Path(root or source_root()) / (name + "-" + SOURCES[name]["revision"][:12])


def verify_source(name, directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Algorithm source directory must not be a symlink")
    receipt = directory / ".slicerdspace-source.json"
    if receipt.is_symlink():
        raise ValueError("Algorithm source receipt must not be a symlink")
    record = json.loads(receipt.read_text())
    if record.get("profile") != name or record.get("archive_sha256") != SOURCES[name]["sha256"]:
        raise ValueError("Unrecognized algorithm source receipt")
    if not record.get("files") or SOURCES[name]["entry"] not in record["files"]:
        raise ValueError("Algorithm entry point is missing")
    for relative, digest in record["files"].items():
        path = PurePosixPath(relative)
        target = directory.joinpath(*path.parts)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ':' in relative or not path.parts:
            raise ValueError("Invalid algorithm source path")
        if any(part.is_symlink() for part in (target, *list(target.parents)[:len(path.parts)])):
            raise ValueError("Algorithm source contains a linked file")
        if not target.is_file() or file_hash(target) != digest:
            raise ValueError("Algorithm source changed: " + relative)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != set(record["files"]) | {receipt.name}:
        raise ValueError("Algorithm source contains unrecorded files")
    return directory


def available_source(name, root=None):
    directory = profile_path(name, root)
    if not directory.exists():
        return None
    try:
        return verify_source(name, directory)
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def source_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".prepare.lock"
    if path.is_symlink():
        raise ValueError("Refusing linked source lock")
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare_sources(profiles, *, root=None, execute=False, accept_terms=False, offline_root=None):
    profiles = tuple(dict.fromkeys(profiles))
    if not profiles or any(name not in SOURCES for name in profiles):
        raise ValueError("Select known algorithm source profiles")
    root = Path(root or source_root()).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Algorithm source root must not be a symlink")
    result = dict(kind="slicerdspace_backend_setup", status="planned", root=str(root),
                  profiles=list(profiles), external_dependencies={},
                  download_bytes=sum(SOURCES[name]["bytes"] for name in profiles if available_source(name, root) is None),
                  terms=["https://github.com/" + SOURCES[name]["repository"] + "/blob/" +
                         SOURCES[name]["revision"] + ("/license.rst" if name == "wmql" else
                                                       "/License.txt" if name == "wma" else "/LICENSE") for name in profiles])
    if not execute:
        return result
    if not accept_terms:
        raise PermissionError("Review algorithm source licences before preparation")
    with source_lock(root):
        for name in profiles:
            spec, destination = SOURCES[name], profile_path(name, root)
            if destination.exists():
                verify_source(name, destination)
                print("[Sources] Reusing verified " + name, flush=True)
            else:
                downloads = root / "downloads"
                downloads.mkdir(exist_ok=True)
                archive = downloads / (name + ".tar.gz")
                if archive.is_symlink():
                    raise ValueError("Refusing linked source archive")
                if not archive.is_file() or file_hash(archive) != spec["sha256"]:
                    if offline_root:
                        source = Path(offline_root) / archive.name
                        if file_hash(source) != spec["sha256"]:
                            raise ValueError("Offline algorithm archive checksum mismatch")
                        shutil.copyfile(source, archive)
                    else:
                        url = "https://codeload.github.com/" + spec["repository"] + "/tar.gz/" + spec["revision"]
                        print("[Sources] Downloading " + name, flush=True)
                        with tempfile.NamedTemporaryFile(dir=downloads, delete=False) as stream:
                            partial = Path(stream.name)
                            try:
                                with urllib.request.urlopen(url, timeout=60) as response:
                                    shutil.copyfileobj(response, stream)
                                stream.close()
                                if file_hash(partial) != spec["sha256"]:
                                    raise ValueError("Downloaded algorithm archive checksum mismatch")
                                partial.replace(archive)
                            finally:
                                stream.close()
                                partial.unlink(missing_ok=True)
                with tempfile.TemporaryDirectory(dir=root, prefix=".extract-") as temporary:
                    staged = Path(temporary) / "source"
                    staged.mkdir()
                    total = 0
                    with tarfile.open(archive, "r:gz") as bundle:
                        for member in bundle:
                            parts = PurePosixPath(member.name).parts
                            if member.issym() or member.islnk() or member.isdev() or ".." in parts or member.name.startswith(("/", "\\")) or "\\" in member.name or ':' in member.name:
                                raise ValueError("Unsafe algorithm archive member")
                            if (not member.isfile() or len(parts) < 2 or parts[1] not in spec["paths"]
                                    or any(part in {"tests", "test", ".git", "__pycache__"} for part in parts[1:])):
                                continue
                            total += member.size
                            if total > 128 * 1024 * 1024:
                                raise ValueError("Algorithm source exceeds extraction budget")
                            target = staged.joinpath(*parts[1:])
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.extractfile(member) as source, target.open("wb") as output:
                                shutil.copyfileobj(source, output)
                            target.chmod(0o755 if member.mode & 0o111 else 0o644)
                    files = {p.relative_to(staged).as_posix(): file_hash(p) for p in staged.rglob("*") if p.is_file()}
                    record = dict(profile=name, revision=spec["revision"], archive_sha256=spec["sha256"], files=files)
                    (staged / ".slicerdspace-source.json").write_text(json.dumps(record, indent=2) + "\n")
                    verify_source(name, staged)
                    staged.rename(destination)
            result["external_dependencies"][spec["setting"]] = str(destination)
    result["status"] = "completed"
    return result
