#!/usr/bin/env python3
"""Create a clean, rooted archive for local Slicer Extension Manager and All-in-One releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_FILE_NAMES = {"extension_staging_manifest.json"}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
NON_EXEC_SUFFIXES = {".xml", ".json", ".txt", ".md", ".rst", ".pth", ".cfg", ".toml", ".yml", ".yaml", ".csv", ".tsv", ".ini"}


def _is_excluded(path: Path, extension_root: Path) -> bool:
    relative = path.relative_to(extension_root)
    return (
        any(part in EXCLUDED_DIR_NAMES for part in relative.parts)
        or path.name in EXCLUDED_FILE_NAMES
        or path.suffix.lower() in EXCLUDED_FILE_SUFFIXES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_mark_executable(path: Path, relative: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in NON_EXEC_SUFFIXES:
        return False
    if suffix == ".py":
        return bool(set(relative.parts) & {"bin", "Scripts", "utilities"})
    if suffix in {".exe", ".dll", ".so", ".dylib", ".sh"} or ".so." in path.name:
        return True
    parts = set(relative.parts)
    if parts & {"bin", "Scripts", "cli-modules", "utilities"}:
        return True
    return os.access(path, os.X_OK)


def _rewrite_s4ext_depends(extension_root: Path, depends_value: str) -> None:
    targets = [extension_root / "SlicerDSpace.s4ext", *sorted(extension_root.glob("share/Slicer-*/SlicerDSpace.s4ext"))]
    for s4ext in targets:
        if not s4ext.is_file():
            continue
        text = s4ext.read_text(encoding="utf-8")
        updated = re.sub(r"(?m)^depends\s+.*$", f"depends {depends_value}", text)
        s4ext.write_text(updated, encoding="utf-8")


def _stage_surface_runtime(surface_archive: Path, bundled_root: Path) -> dict[str, object]:
    catalog = json.loads((REPO_ROOT / "models" / "runtime_bundles.json").read_text(encoding="utf-8"))
    digest = _sha256(surface_archive)
    record = next((item for item in catalog.get("surface", ()) if item.get("sha256") == digest), None)
    if record is None:
        raise ValueError(f"Surface runtime archive SHA256 {digest} does not match models/runtime_bundles.json")
    system = str(record["system"]).lower()
    machine = str(record["machine"]).lower()
    dest = bundled_root / f"surface-runtime-{system}-{machine}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(surface_archive, "r:gz") as bundle:
        for member in bundle:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in member.name or ":" in member.name or member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe surface runtime archive member: {member.name}")
            if not member.isfile():
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            target.chmod(0o755 if (member.mode & 0o111) else 0o644)
    manifest_file = dest / "runtime.json"
    if not manifest_file.is_file():
        raise ValueError("Bundled surface runtime is missing runtime.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("activation_key_included") is not False:
        raise ValueError("Bundled surface runtime must have activation_key_included=false")
    return {
        "name": "surface",
        "system": record["system"],
        "machine": record["machine"],
        "archive_sha256": digest,
        "directory": f"bundled-runtimes/{dest.name}",
        "activation_key_included": False,
    }


def _stage_bundled_runtimes(
    extension_root: Path,
    *,
    mrtrix_archive: Path | None = None,
    fsl_archive: Path | None = None,
    surface_archive: Path | None = None,
    extra_runtime_dirs: list[Path] | None = None,
) -> list[dict[str, object]]:
    extra_dirs = list(extra_runtime_dirs or ())
    if not any((mrtrix_archive, fsl_archive, surface_archive, extra_dirs)):
        return []
    from shared.native_runtimes import install_bundle, load_catalog

    bundled_root = extension_root / "bundled-runtimes"
    bundled_root.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    staged: list[dict[str, object]] = []
    for tool_name, archive_path in (("mrtrix", mrtrix_archive), ("fsl", fsl_archive)):
        if archive_path is None:
            continue
        resolved_archive = archive_path.expanduser().resolve()
        digest = _sha256(resolved_archive)
        record = next(
            (r for r in catalog.get("bundles", ()) if r.get("name") == tool_name and r.get("sha256") == digest),
            None,
        )
        if record is None:
            raise ValueError(f"{tool_name} archive SHA256 {digest} does not match models/native_runtime_bundles.json")
        dest = install_bundle(record, archive=resolved_archive, root=bundled_root, accept_terms=True)
        staged.append({
            "name": tool_name,
            "version": record["version"],
            "system": record["system"],
            "machine": record["machine"],
            "archive_sha256": digest,
            "directory": f"bundled-runtimes/{dest.name}",
            "command_count": len(record["commands"]),
        })
    if surface_archive is not None:
        staged.append(_stage_surface_runtime(surface_archive.expanduser().resolve(), bundled_root))
    for extra_dir in extra_dirs:
        src = extra_dir.expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"Extra runtime directory does not exist: {src}")
        dst = bundled_root / src.name
        if dst != src:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)
        staged.append({
            "name": src.name,
            "directory": f"bundled-runtimes/{dst.name}",
        })
    return staged


def _is_macho_binary(path: Path) -> bool:
    try:
        magic = path.read_bytes()[:4]
    except OSError:
        return False
    return magic in {
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }


def _relink_macos_cli_libs(dest: Path, lib_dest: Path, copied_names: list[str]) -> None:
    if sys.platform != "darwin":
        return
    lib_ukf = lib_dest / "libUKFBase.dylib"
    install_name_tool = shutil.which("install_name_tool")
    otool = shutil.which("otool")
    if not lib_ukf.is_file() or not install_name_tool or not otool or not _is_macho_binary(lib_ukf):
        return
    codesign = shutil.which("codesign")
    subprocess.run([install_name_tool, "-id", "@rpath/libUKFBase.dylib", str(lib_ukf)], check=True)
    if codesign:
        subprocess.run([codesign, "--force", "--sign", "-", str(lib_ukf)], check=True)
    for name in copied_names:
        if name.lower().endswith(".xml"):
            continue
        target = dest / name
        if not target.is_file() or not _is_macho_binary(target):
            continue
        otool_out = subprocess.check_output([otool, "-L", str(target)], text=True, encoding="utf-8", errors="replace")
        old_refs: list[str] = []
        for line in otool_out.splitlines()[1:]:
            dep = line.strip().split(" (", 1)[0].strip()
            if dep.endswith("/libUKFBase.dylib") and dep not in old_refs:
                old_refs.append(dep)
        if not old_refs:
            continue
        for old_ref in old_refs:
            subprocess.run(
                [install_name_tool, "-change", old_ref, "@loader_path/../libUKFBase.dylib", str(target)],
                check=True,
            )
        if codesign:
            subprocess.run([codesign, "--force", "--sign", "-", str(target)], check=True)


def _stage_cli_modules(extension_root: Path, cli_modules_dir: Path | None, slicer_subdir: str) -> list[str]:
    if cli_modules_dir is None:
        return []
    src = cli_modules_dir.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"CLI modules directory does not exist: {src}")
    lib_dest = extension_root / "lib" / slicer_subdir
    dest = lib_dest / "cli-modules"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for item in sorted(src.iterdir()):
        if not item.is_file():
            continue
        target = dest / item.name
        if target.resolve() != item.resolve():
            shutil.copy2(item, target)
        if _should_mark_executable(target, Path("cli-modules") / target.name):
            target.chmod(0o755)
        copied.append(item.name)

    for search_dir in (src.parent, src):
        if not search_dir.is_dir():
            continue
        for pattern in ("lib*UKF*.dylib", "lib*UKF*.so*"):
            for lib_file in sorted(search_dir.glob(pattern)):
                if not lib_file.is_file():
                    continue
                lib_target = lib_dest / lib_file.name
                if lib_target.resolve() != lib_file.resolve():
                    shutil.copy2(lib_file, lib_target)
                lib_target.chmod(0o755)

    _relink_macos_cli_libs(dest, lib_dest, copied)
    return copied


def _parse_wma_atlases_arg(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ("org", "east-west", "naba")
    tokens = raw.split(",") if isinstance(raw, str) else list(raw)
    cleaned = [tok.strip().lower() for tok in tokens if str(tok).strip()]
    if not cleaned or cleaned == ["none"]:
        return ()
    valid = {"org", "east-west", "naba"}
    invalid = [tok for tok in cleaned if tok not in valid]
    if invalid:
        raise ValueError(f"Unsupported WMA atlas names: {invalid}")
    return tuple(dict.fromkeys(cleaned))


def _stage_model_assets(
    extension_root: Path,
    model_assets_root: Path | None,
    *,
    include_wma_atlases: tuple[str, ...] = ("org", "east-west", "naba"),
    include_upstream_only: bool = False,
) -> dict[str, object] | None:
    if model_assets_root is None:
        return None
    src = model_assets_root.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Model assets directory does not exist: {src}")

    core_manifest = json.loads((REPO_ROOT / "models" / "assets_manifest.json").read_text(encoding="utf-8"))
    ext_manifest = json.loads((REPO_ROOT / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))

    expected_hashes: dict[str, str] = {}
    for prof in ("surface", "segmentation", "parcellation"):
        for rel in core_manifest["profiles"][prof]["required_paths"]:
            expected_hashes[rel] = core_manifest["asset_records"][rel]["sha256"]
    ext_profiles = ("pnl-cnn", "pnl-cnn-weights", "tractcloud") if include_upstream_only else ("pnl-cnn-weights", "tractcloud")
    for prof in ext_profiles:
        for rel in ext_manifest["profiles"][prof].get("required_paths", ()):
            rec = ext_manifest["asset_records"][rel]
            if not include_upstream_only and rec.get("redistribution") == "upstream-only":
                continue
            expected_hashes[rel] = rec["sha256"]

    if (src / "surface/postprocessing/atlas.tar.gz").is_file():
        atlas_rel = "surface/postprocessing/atlas.tar.gz"
        expected_hashes[atlas_rel] = ext_manifest["asset_records"][atlas_rel]["sha256"]

    selected_wma = set(include_wma_atlases)
    wma_profiles = tuple(
        (family, prof_name)
        for family, prof_name in (
            ("org", "wma"),
            ("east-west", "wma-east-west"),
            ("naba", "wma-naba"),
        )
        if family in selected_wma
    )
    for _, prof_name in wma_profiles:
        prof_def = ext_manifest["profiles"][prof_name]
        prof_dir = prof_def["directory"]
        for rel in prof_def.get("required_paths", ()):
            if "/archives/" in rel:
                continue
            expected_hashes[rel] = ext_manifest["asset_records"][rel]["sha256"]
        index_rel = prof_def.get("extracted_index", "bundles/wma/extracted_resource_index.json")
        index_path = src / index_rel
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing WMA index file for {prof_name}: {index_path}")
        wma_index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in wma_index["files"]:
            rel = f"{prof_dir}/{entry['path']}"
            expected_hashes[rel] = entry["sha256"]

    dest_root = extension_root / "bundled-model-assets"
    if dest_root.resolve() != src:
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for rel, expected_digest in sorted(expected_hashes.items()):
        source_file = src / rel
        if not source_file.is_file():
            raise FileNotFoundError(f"Required model asset missing from {src}: {rel}")
        actual_digest = _sha256(source_file)
        if actual_digest != expected_digest:
            raise ValueError(f"Model asset SHA256 mismatch for {rel}: expected {expected_digest}, got {actual_digest}")
        target_file = dest_root / rel
        if not target_file.exists() or target_file.resolve() != source_file.resolve():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if target_file.exists():
                target_file.unlink()
            try:
                os.link(source_file, target_file)
            except OSError:
                shutil.copy2(source_file, target_file)
        total_bytes += target_file.stat().st_size

    return {
        "directory": "bundled-model-assets",
        "file_count": len(expected_hashes),
        "total_bytes": total_bytes,
        "wma_atlases": [name for name, _ in wma_profiles],
    }


def _stage_sources(extension_root: Path, sources_root: Path | None) -> list[str]:
    if sources_root is None:
        return []
    from shared.backend_sources import SOURCES, prepare_sources, profile_path, verify_source

    src = sources_root.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Algorithm sources directory does not exist: {src}")
    dest_root = extension_root / "bundled-sources"
    if dest_root.resolve() != src:
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)

    profiles = tuple(SOURCES.keys())
    all_extracted = all(profile_path(name, src).is_dir() for name in profiles)
    if all_extracted:
        for name in profiles:
            verified = verify_source(name, profile_path(name, src))
            target = profile_path(name, dest_root)
            if target.resolve() != verified.resolve():
                shutil.copytree(verified, target, symlinks=False)
            verify_source(name, target)
    else:
        downloads_dir = src / "downloads" if (src / "downloads").is_dir() else src
        prepare_sources(profiles, root=dest_root, execute=True, accept_terms=True, offline_root=downloads_dir)
        shutil.rmtree(dest_root / "downloads", ignore_errors=True)
        (dest_root / ".prepare.lock").unlink(missing_ok=True)

    staged_names: list[str] = []
    for name in profiles:
        target = profile_path(name, dest_root)
        verify_source(name, target)
        for sub in ("bin", "utilities", "scripts"):
            subdir = target / sub
            if subdir.is_dir():
                for item in subdir.rglob("*"):
                    if item.is_file() and item.suffix.lower() not in {".txt", ".md", ".rst", ".json"}:
                        item.chmod(0o755)
        staged_names.append(target.name)
    return sorted(staged_names)


def _normalize_runtime_arch(raw: str | None = None) -> str:
    value = str(raw or platform.machine() or "").strip().lower()
    return {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(value, value)


def _s4ext_arch(extension_root: Path) -> str:
    s4ext = extension_root / "SlicerDSpace.s4ext"
    if s4ext.is_file():
        for line in s4ext.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "arch":
                return _normalize_runtime_arch(parts[1])
    return _normalize_runtime_arch()


def _stage_python_site_packages(
    extension_root: Path,
    site_packages_dir: Path | None,
    *,
    python_version: str = "3.12",
    arch: str | None = None,
    slicer_numpy_minimum: str = "2.0.2",
) -> dict[str, object] | None:
    if site_packages_dir is None:
        return None
    src = site_packages_dir.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Python site-packages directory does not exist: {src}")

    dest_root = extension_root / "bundled-python"
    dest_sp = dest_root / "site-packages"
    if dest_sp.resolve() != src:
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_sp.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src,
            dest_sp,
            dirs_exist_ok=True,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    for pycache in list(dest_sp.rglob("__pycache__")):
        shutil.rmtree(pycache, ignore_errors=True)
    # Remove unused C++ headers and deeply nested test/third_party trees that exceed Windows 260-char MAX_PATH
    for prune_rel in (
        Path("torch") / "include",
        Path("statsmodels") / "tsa" / "vector_ar" / "tests" / "JMulTi_results",
    ):
        prune_target = dest_sp / prune_rel
        if prune_target.is_dir():
            shutil.rmtree(prune_target, ignore_errors=True)
    for dist_tp in list(dest_sp.glob("*.dist-info/licenses/third_party")):
        if dist_tp.is_dir():
            shutil.rmtree(dist_tp, ignore_errors=True)

    for script_sub in ("bin", "Scripts"):
        sdir = dest_sp / script_sub
        if sdir.is_dir():
            for item in sdir.iterdir():
                if item.is_file():
                    item.chmod(0o755)

    resolved_arch = _normalize_runtime_arch(arch) if arch else _s4ext_arch(extension_root)
    meta = {
        "python_version": python_version,
        "arch": resolved_arch,
        "site_packages": "site-packages",
        "slicer_numpy_minimum": slicer_numpy_minimum,
    }
    (dest_root / "python-runtime.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # Self-contained bundle includes torch inside bundled-python/site-packages, so Slicer Extension Manager
    # must not attempt to query/download the separate PyTorch extension over the network during offline install.
    _rewrite_s4ext_depends(extension_root, "NA")

    file_count = sum(1 for p in dest_sp.rglob("*") if p.is_file())
    return {
        "directory": "bundled-python/site-packages",
        "python_version": python_version,
        "file_count": file_count,
    }


def package_extension(
    extension_root: Path,
    output_path: Path,
    *,
    mrtrix_archive: Path | None = None,
    fsl_archive: Path | None = None,
    surface_archive: Path | None = None,
    extra_runtime_dirs: list[Path] | None = None,
    cli_modules_dir: Path | None = None,
    model_assets_root: Path | None = None,
    include_wma_atlases: tuple[str, ...] = ("org", "east-west", "naba"),
    include_upstream_only: bool = False,
    sources_root: Path | None = None,
    python_site_packages: Path | None = None,
    python_version: str = "3.12",
    zip_compression: str = "deflated",
    tar_compression: str = "gz",
) -> dict[str, object]:
    """Package a staged extension with the top-level directory Slicer expects."""

    root = extension_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Staged extension root does not exist: {root}")
    module_files = sorted(root.glob("lib/Slicer-*/qt-scripted-modules/SlicerDSpace.py"))
    if len(module_files) != 1:
        raise ValueError(f"Expected exactly one staged SlicerDSpace.py, found {len(module_files)}")
    if not (root / "SlicerDSpace.s4ext").is_file():
        raise FileNotFoundError(f"Extension metadata is missing: {root / 'SlicerDSpace.s4ext'}")

    slicer_subdir = module_files[0].parents[1].name
    bundled_runtimes = _stage_bundled_runtimes(
        root,
        mrtrix_archive=mrtrix_archive,
        fsl_archive=fsl_archive,
        surface_archive=surface_archive,
        extra_runtime_dirs=extra_runtime_dirs,
    )
    bundled_cli = _stage_cli_modules(root, cli_modules_dir, slicer_subdir)
    bundled_assets = _stage_model_assets(
        root,
        model_assets_root,
        include_wma_atlases=_parse_wma_atlases_arg(include_wma_atlases),
        include_upstream_only=include_upstream_only,
    )
    bundled_sources = _stage_sources(root, sources_root)
    bundled_python = _stage_python_site_packages(root, python_site_packages, python_version=python_version)

    # Also record pre-staged bundled-runtimes if already present in extension_root
    if not bundled_runtimes and (root / "bundled-runtimes").is_dir():
        bundled_runtimes = [
            {"name": child.name, "directory": f"bundled-runtimes/{child.name}"}
            for child in sorted((root / "bundled-runtimes").iterdir())
            if child.is_dir()
        ]

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    members = [root]
    members.extend(path for path in sorted(root.rglob("*")) if not _is_excluded(path, root))
    is_macos_archive = bool(
        any(r.get("system") == "Darwin" for r in bundled_runtimes)
        or "macos" in output.name.lower()
        or "darwin" in output.name.lower()
    )
    archive_prefix = (
        Path(root.name) / "Slicer.app" / "Contents" / "Extensions-34045" / root.name
        if is_macos_archive
        else Path(root.name)
    )

    if output.name.lower().endswith(".zip"):
        if zip_compression.lower() != "deflated":
            raise ValueError(
                f"Unsupported zip_compression={zip_compression!r}; 3D Slicer vtkArchive requires 'deflated'."
            )
        comp_mode = zipfile.ZIP_DEFLATED
        zip_kwargs: dict[str, object] = {
            "mode": "w",
            "compression": comp_mode,
            "allowZip64": True,
            "compresslevel": 6,
        }
        with zipfile.ZipFile(output, **zip_kwargs) as archive:
            for path in members:
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                archive_name = (archive_prefix / relative).as_posix()
                info = zipfile.ZipInfo.from_file(path, arcname=archive_name, strict_timestamps=False)
                info.compress_type = comp_mode
                mode = 0o755 if _should_mark_executable(path, relative) else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                with path.open("rb") as src, archive.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    else:
        def _write_tar_members(archive: tarfile.TarFile) -> None:
            for path in members:
                relative = path.relative_to(root)
                archive_name = archive_prefix if relative == Path(".") else (archive_prefix / relative)
                info = archive.gettarinfo(path.as_posix(), arcname=archive_name.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = "root"
                info.gname = "root"
                if path.is_dir():
                    info.mode = 0o755
                    archive.addfile(info)
                elif path.is_file():
                    info.mode = 0o755 if _should_mark_executable(path, relative) else 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)

        if tar_compression.lower() == "xz":
            xz_bin = shutil.which("xz")
            if xz_bin:
                with output.open("wb") as out_fh:
                    proc = subprocess.Popen([xz_bin, "-T0", "-6", "-c"], stdin=subprocess.PIPE, stdout=out_fh)
                    assert proc.stdin is not None
                    try:
                        with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                            _write_tar_members(archive)
                    finally:
                        proc.stdin.close()
                    rc = proc.wait()
                    if rc != 0:
                        raise RuntimeError(f"xz compression failed with exit code {rc}")
            else:
                with tarfile.open(output, mode="w:xz", format=tarfile.PAX_FORMAT, preset=6) as archive:
                    _write_tar_members(archive)
        else:
            with tarfile.open(output, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as archive:
                _write_tar_members(archive)

    payload = {
        "kind": "slicerdsurfer_extension_package",
        "version": 2,
        "extension_name": root.name,
        "extension_root": root.as_posix(),
        "archive": output.as_posix(),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": _sha256(output),
        "member_count": len(members),
        "module_file": module_files[0].relative_to(root).as_posix(),
        "bundled_runtimes": bundled_runtimes,
        "bundled_cli_modules": bundled_cli,
        "bundled_model_assets": bundled_assets,
        "bundled_sources": bundled_sources,
        "bundled_python": bundled_python,
        "excluded_directory_names": sorted(EXCLUDED_DIR_NAMES),
        "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
        "excluded_file_suffixes": sorted(EXCLUDED_FILE_SUFFIXES),
    }
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = manifest_path.as_posix()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mrtrix-archive", type=Path, help="Optional modular MRtrix3 runtime archive to bundle.")
    parser.add_argument("--fsl-archive", type=Path, help="Optional modular FSL runtime archive to bundle.")
    parser.add_argument("--surface-archive", type=Path, help="Optional modular FreeSurfer surface runtime archive to bundle.")
    parser.add_argument("--extra-runtime-dir", action="append", default=[], type=Path,
                        help="Additional runtime directory to copy into bundled-runtimes/ (repeatable).")
    parser.add_argument("--cli-modules-dir", type=Path, help="Companion CLI modules directory (e.g. UKFTractography).")
    parser.add_argument("--model-assets-root", type=Path, help="Verified model-assets root to bundle.")
    parser.add_argument(
        "--wma-atlases",
        default="org,east-west,naba",
        help="Comma-separated WMA atlases to bundle (org,east-west,naba) or 'none' to omit all WMA atlases.",
    )
    parser.add_argument(
        "--include-upstream-only",
        action="store_true",
        help="Include upstream-only files (such as IITmean_b0_256.nii.gz) when building local non-release bundles.",
    )
    parser.add_argument("--sources-root", type=Path, help="Verified algorithm-sources root (or downloads dir) to bundle.")
    parser.add_argument("--python-site-packages", type=Path, help="Prepared Slicer Python site-packages directory to bundle.")
    parser.add_argument("--python-version", default="3.12", help="Target Slicer Python major.minor version (default: 3.12).")
    parser.add_argument("--zip-compression", choices=("deflated",), default="deflated",
                        help="ZIP compression algorithm when --output ends with .zip (Slicer vtkArchive requires deflated).")
    parser.add_argument("--tar-compression", choices=("gz", "xz"), default="gz",
                        help="TAR compression algorithm when --output is a tar archive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = package_extension(
        args.extension_root,
        args.output,
        mrtrix_archive=args.mrtrix_archive,
        fsl_archive=args.fsl_archive,
        surface_archive=args.surface_archive,
        extra_runtime_dirs=args.extra_runtime_dir,
        cli_modules_dir=args.cli_modules_dir,
        model_assets_root=args.model_assets_root,
        include_wma_atlases=_parse_wma_atlases_arg(args.wma_atlases),
        include_upstream_only=args.include_upstream_only,
        sources_root=args.sources_root,
        python_site_packages=args.python_site_packages,
        python_version=args.python_version,
        zip_compression=args.zip_compression,
        tar_compression=args.tar_compression,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
