"""Incremental companion-component scanner and updater for SlicerDSpace.

Extends ``shared.code_updater`` so ``Check for updates`` (and the post-restart
handoff when upgrading from v0.1.1) inspects and replaces only the specific
model checkpoints, installed WMA atlases, bundled Python algorithm sources, and
standalone native runtimes whose SHA-256 digests actually changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "NeuroDMRI/SlicerDSpace"
ASSETS_RELEASE_BASE = f"https://github.com/{GITHUB_REPO}/releases/download/assets"

ProgressCallback = Callable[[int, int], None]
PhaseCallback = Callable[[str], None]

# Pinned standalone UKF archives hosted in the permanent ``assets`` release.
STANDALONE_UKF_SPECS: tuple[dict[str, Any], ...] = (
    {
        "system": "Darwin",
        "machine": "arm64",
        "dir_name": "ukf-darwin-arm64",
        "filename": "ukf-standalone-darwin-arm64.tar.gz",
        "format": "tar.gz",
        "bytes": 3699789,
        "sha256": "e68e8190f8d3e5ae70e5dbc68d55d99f6c6551abbebc3dd1a1696c43ada51f8f",
        "executable": "bin/UKFTractography",
        "source_commit": "3f03d002960863dcce69a4c9c7e3d466026d6057",
        "requires_avx2": False,
        "url": f"{ASSETS_RELEASE_BASE}/ukf-standalone-darwin-arm64.tar.gz",
    },
    {
        "system": "Linux",
        "machine": "x86_64",
        "dir_name": "ukf-linux-x64",
        "filename": "ukf-standalone-linux-x64.tar.gz",
        "format": "tar.gz",
        "bytes": 6031472,
        "sha256": "8631887f22ed6d79bf692f29280a824d9276dec3eb41702241f9382241dcffd",
        "executable": "bin/UKFTractography",
        "source_commit": "3f03d002960863dcce69a4c9c7e3d466026d6057",
        "requires_avx2": True,
        "url": f"{ASSETS_RELEASE_BASE}/ukf-standalone-linux-x64.tar.gz",
    },
    {
        "system": "Windows",
        "machine": "x86_64",
        "dir_name": "ukf-windows-x64",
        "filename": "ukf-standalone-windows-x64.zip",
        "format": "zip",
        "bytes": 2365153,
        "sha256": "a9359316af30163f4a2abd19a282d0595e311d0c2c0122e2c29cc70bf23ce47c",
        "executable": "bin/UKFTractography.exe",
        "source_commit": "3f03d002960863dcce69a4c9c7e3d466026d6057",
        "requires_avx2": True,
        "url": f"{ASSETS_RELEASE_BASE}/ukf-standalone-windows-x64.zip",
    },
)

WMA_PACK_PROFILES: dict[str, tuple[str, str, str]] = {
    "wma_org": ("wma", "bundles/wma", "bundles/wma/extracted_resource_index.json"),
    "wma_naba": ("wma-naba", "bundles/wma_naba", "bundles/wma_naba/extracted_resource_index.json"),
    "wma_east_west": ("wma-east-west", "bundles/wma_east_west", "bundles/wma_east_west/extracted_resource_index.json"),
}


def _sha256_file(path: Path) -> str:
    from shared.update_journal import win_fs_path

    digest = hashlib.sha256()
    with win_fs_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_writable_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    probe = directory / f".slicerdspace_comp_probe_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _ensure_writable_or_parent(path: Path) -> bool:
    curr = path
    while not curr.exists() and curr.parent != curr:
        curr = curr.parent
    return _is_writable_dir(curr)


def _extension_container_root(module_root: Path) -> Path | None:
    """Return ``.../SlicerDSpace`` when ``module_root`` is ``.../SlicerDSpace/lib/Slicer-X.Y/qt-scripted-modules``."""

    resolved = module_root.expanduser().resolve()
    if resolved.name == "qt-scripted-modules" and len(resolved.parents) >= 3:
        return resolved.parents[2]
    if (resolved / "bundled-runtimes").is_dir() or (resolved / "bundled-model-assets").is_dir() or (resolved / "bundled-sources").is_dir():
        return resolved
    return None


def _bundled_model_roots_for(module_root: Path) -> tuple[Path, ...]:
    env_override = os.environ.get("SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT")
    cands: list[Path] = []
    if env_override:
        cands.append(Path(env_override).expanduser().resolve())
    else:
        ext = _extension_container_root(module_root)
        if ext is not None:
            cands.append(ext / "bundled-model-assets")
        cands.append(module_root.expanduser().resolve() / "bundled-model-assets")
    out: list[Path] = []
    for c in cands:
        if c.is_dir():
            r = c.resolve()
            if r not in out:
                out.append(r)
    return tuple(out)


def _bundled_source_roots_for(module_root: Path) -> tuple[Path, ...]:
    env_override = os.environ.get("SLICERDSPACE_BUNDLED_SOURCES_ROOT")
    cands: list[Path] = []
    if env_override:
        cands.append(Path(env_override).expanduser().resolve())
    else:
        ext = _extension_container_root(module_root)
        if ext is not None:
            cands.append(ext / "bundled-sources")
        cands.append(module_root.expanduser().resolve() / "bundled-sources")
    out: list[Path] = []
    for c in cands:
        if c.is_dir():
            r = c.resolve()
            if r not in out:
                out.append(r)
    return tuple(out)


def _bundled_runtime_roots_for(module_root: Path) -> tuple[Path, ...]:
    env_override = os.environ.get("SLICERDSPACE_BUNDLED_RUNTIMES_ROOT")
    cands: list[Path] = []
    if env_override:
        cands.append(Path(env_override).expanduser().resolve())
    else:
        ext = _extension_container_root(module_root)
        if ext is not None:
            cands.append(ext / "bundled-runtimes")
        cands.append(module_root.expanduser().resolve() / "bundled-runtimes")
    out: list[Path] = []
    for c in cands:
        if c.is_dir():
            r = c.resolve()
            if r not in out:
                out.append(r)
    return tuple(out)


def _host_platform_pair(system: str | None = None, machine: str | None = None) -> tuple[str, str]:
    try:
        from shared.native_runtimes import host_platform

        return host_platform(system, machine)
    except Exception:
        sys_name = system or platform.system()
        mach = (machine or platform.machine()).lower()
        arch = {"amd64": "x86_64", "aarch64": "arm64", "x64": "x86_64"}.get(mach, mach)
        return sys_name, arch


def _asset_urls_for_record(relpath: str, record: dict[str, Any], *, assets_base: str = ASSETS_RELEASE_BASE) -> list[str]:
    sha = str(record.get("sha256") or "").strip().lower()
    fname = PurePosixPath(relpath).name
    urls: list[str] = []
    # Upstream-only IIT template is hosted on NITRC, not redistributed in GitHub assets.
    if fname != "IITmean_b0_256.nii.gz" and len(sha) == 64:
        urls.append(f"{assets_base}/{sha}-{fname}")
    for u in record.get("urls") or ():
        s = str(u or "").strip()
        if s and s not in urls:
            urls.append(s)
    return urls


def build_components_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    """Build deterministic companion-component metadata from the repository manifests."""

    root = (repo_root or REPO_ROOT).expanduser().resolve()
    assets_manifest_path = root / "models" / "assets_manifest.json"
    ext_manifest_path = root / "models" / "external_assets_manifest.json"
    packs_manifest_path = root / "models" / "release_packs_manifest.json"
    surface_bundles_path = root / "models" / "runtime_bundles.json"
    native_bundles_path = root / "models" / "native_runtime_bundles.json"

    if not assets_manifest_path.is_file() or not packs_manifest_path.is_file():
        return {}

    assets_data = json.loads(assets_manifest_path.read_text(encoding="utf-8"))
    ext_data = json.loads(ext_manifest_path.read_text(encoding="utf-8")) if ext_manifest_path.is_file() else {}
    packs_data = json.loads(packs_manifest_path.read_text(encoding="utf-8"))

    all_asset_records: dict[str, dict[str, Any]] = {}
    for src in (assets_data.get("asset_records") or {}, ext_data.get("asset_records") or {}):
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, dict):
                    all_asset_records[str(k)] = dict(v)

    assets_base = str(packs_data.get("assets_release_base_url") or ASSETS_RELEASE_BASE).rstrip("/")
    packs = packs_data.get("packs") or {}

    model_packs: dict[str, dict[str, Any]] = {}
    wma_packs: dict[str, dict[str, Any]] = {}

    for pack_id, pinfo in packs.items():
        if not isinstance(pinfo, dict):
            continue
        tier = str(pinfo.get("tier") or "")
        if tier == "core":
            files_spec: dict[str, dict[str, Any]] = {}
            for rel, fsha in (pinfo.get("files") or {}).items():
                rec = all_asset_records.get(rel) or {}
                sha = str(fsha or rec.get("sha256") or "").strip().lower()
                size_b = int(rec.get("size_bytes") or 0)
                urls = _asset_urls_for_record(rel, {**rec, "sha256": sha}, assets_base=assets_base)
                if len(sha) == 64 and urls:
                    files_spec[str(rel)] = {
                        "sha256": sha,
                        "size_bytes": size_b,
                        "urls": urls,
                    }
            model_packs[str(pack_id)] = {
                "id": str(pack_id),
                "filename": str(pinfo.get("filename") or ""),
                "url": str(pinfo.get("url") or ""),
                "sha256": str(pinfo.get("sha256") or "").lower(),
                "size_bytes": int(pinfo.get("size_bytes") or 0),
                "files": files_spec,
            }
        elif pack_id in WMA_PACK_PROFILES:
            _prof, directory, index_rel = WMA_PACK_PROFILES[pack_id]
            idx_rec = all_asset_records.get(index_rel) or {}
            wma_packs[str(pack_id)] = {
                "id": str(pack_id),
                "directory": directory,
                "index_relpath": index_rel,
                "index_sha256": str(idx_rec.get("sha256") or "").lower(),
                "filename": str(pinfo.get("filename") or ""),
                "url": str(pinfo.get("url") or ""),
                "sha256": str(pinfo.get("sha256") or "").lower(),
                "size_bytes": int(pinfo.get("size_bytes") or 0),
                "parts": list(pinfo.get("parts") or ()),
            }

    iit_rel = "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"
    iit_rec = all_asset_records.get(iit_rel) or {}
    iit_spec = None
    if iit_rec.get("sha256") and iit_rec.get("urls"):
        iit_spec = {
            "relpath": iit_rel,
            "sha256": str(iit_rec["sha256"]).lower(),
            "size_bytes": int(iit_rec.get("size_bytes") or 0),
            "urls": [str(u) for u in iit_rec.get("urls") or () if str(u).strip()],
        }

    algorithm_sources: dict[str, dict[str, Any]] = {}
    try:
        from shared.backend_sources import SOURCES

        for sname, sspec in SOURCES.items():
            algorithm_sources[str(sname)] = {
                "name": str(sname),
                "repository": str(sspec["repository"]),
                "revision": str(sspec["revision"]),
                "sha256": str(sspec["sha256"]).lower(),
                "size_bytes": int(sspec["bytes"]),
                "entry": str(sspec["entry"]),
                "paths": list(sspec["paths"]),
                "url": f"https://codeload.github.com/{sspec['repository']}/tar.gz/{sspec['revision']}",
            }
    except Exception:
        pass

    surface_runtimes: list[dict[str, Any]] = []
    if surface_bundles_path.is_file():
        try:
            sdata = json.loads(surface_bundles_path.read_text(encoding="utf-8"))
            for item in sdata.get("surface") or ():
                if isinstance(item, dict) and item.get("published"):
                    surface_runtimes.append(dict(item))
        except Exception:
            pass

    native_runtimes: list[dict[str, Any]] = []
    if native_bundles_path.is_file():
        try:
            ndata = json.loads(native_bundles_path.read_text(encoding="utf-8"))
            for item in ndata.get("bundles") or ():
                if not isinstance(item, dict) or not item.get("published"):
                    continue
                url = str(item.get("standalone_url") or item.get("url") or "").strip()
                unified = str(item.get("unified_artifact") or "").strip()
                # Only include standalone runtime tarballs/zips (~15-45 MB), never the 350-500 MB OS Base archive.
                if url and (not unified or not url.endswith("/" + unified)):
                    entry = dict(item)
                    entry["url"] = url
                    native_runtimes.append(entry)
        except Exception:
            pass

    return {
        "version": 1,
        "model_packs": model_packs,
        "iit_template": iit_spec,
        "wma_packs": wma_packs,
        "algorithm_sources": algorithm_sources,
        "runtimes": {
            "surface": surface_runtimes,
            "ukf_standalone": [dict(x) for x in STANDALONE_UKF_SPECS],
            "native": native_runtimes,
        },
    }


def _file_matches_cached(path: Path, expected_sha: str, expected_size: int, cache: dict[str, Any]) -> tuple[bool, bool]:
    """Return ``(matches, cache_dirty)`` using mtime/size-keyed SHA256 cache."""

    from shared.model_asset_integrity import cached_sha256_of_file

    try:
        if path.is_symlink() or not path.is_file():
            return False, False
        st = path.stat()
        if expected_size > 0 and st.st_size != expected_size:
            return False, False
        before_len = len(cache.get("entries", {})) if isinstance(cache, dict) else 0
        actual = cached_sha256_of_file(path, cache)
        after_len = len(cache.get("entries", {})) if isinstance(cache, dict) else 0
        return actual == expected_sha.lower(), after_len != before_len
    except OSError:
        return False, False


def _macos_version_tuple() -> tuple[int, int] | None:
    if platform.system() != "Darwin":
        return None
    ver = platform.mac_ver()[0]
    if not ver:
        return None
    try:
        parts = [int(x) for x in ver.split(".")[:2]]
        return (parts[0], parts[1] if len(parts) > 1 else 0)
    except ValueError:
        return None


def scan_component_updates(
    module_root: Path,
    components: dict[str, Any] | None = None,
    *,
    system: str | None = None,
    machine: str | None = None,
    explicit_model_root: Path | None = None,
    allow_git_checkout: bool = False,
) -> list[dict[str, Any]]:
    """Return ordered minimal list of outdated installed companion components."""

    resolved_module = module_root.expanduser().resolve()
    if (resolved_module / ".git").exists() and not allow_git_checkout:
        return []

    comp = components if isinstance(components, dict) and components else build_components_manifest(resolved_module)
    if not comp:
        return []

    from shared.model_asset_integrity import load_sha256_cache, save_sha256_cache
    from shared.model_assets import _sha256_cache_path, configured_model_assets_root

    cache_path = _sha256_cache_path()
    sha_cache = load_sha256_cache(cache_path)
    cache_dirty = False

    user_model_root = configured_model_assets_root(explicit_model_root)
    bundled_model_roots = _bundled_model_roots_for(resolved_module)
    model_roots = [user_model_root, *[r for r in bundled_model_roots if r != user_model_root]]

    updates: list[dict[str, Any]] = []

    # 1. Core model files (per-file granularity: only download files of installed packs whose sha256 differs).
    for pack_id, pinfo in (comp.get("model_packs") or {}).items():
        files_map = pinfo.get("files") or {}
        if not files_map:
            continue
        installed_anywhere = any(
            (root / rel).is_file()
            for root in model_roots
            for rel in files_map
        )
        if not installed_anywhere:
            continue
        for rel, fspec in files_map.items():
            exp_sha = str(fspec.get("sha256") or "").lower()
            exp_size = int(fspec.get("size_bytes") or 0)
            urls = list(fspec.get("urls") or ())
            if len(exp_sha) != 64 or not urls:
                continue
            # Follow runtime resolution priority: user_model_root first if present there, else first bundled root.
            effective_path: Path | None = None
            for root in model_roots:
                cand = root / rel
                if cand.is_file():
                    effective_path = cand
                    break
            if effective_path is not None:
                ok, dirty = _file_matches_cached(effective_path, exp_sha, exp_size, sha_cache)
                cache_dirty = cache_dirty or dirty
                if ok:
                    continue
            updates.append(
                {
                    "kind": "model_file",
                    "id": f"model:{rel}",
                    "pack_id": str(pack_id),
                    "relpath": str(rel),
                    "size_bytes": exp_size,
                    "sha256": exp_sha,
                    "urls": urls,
                }
            )

    # Check IIT template only if already present on disk with a mismatched digest.
    iit_spec = comp.get("iit_template")
    if isinstance(iit_spec, dict) and iit_spec.get("relpath") and iit_spec.get("sha256"):
        iit_rel = str(iit_spec["relpath"])
        iit_existing = next((r / iit_rel for r in model_roots if (r / iit_rel).is_file()), None)
        if iit_existing is not None:
            ok, dirty = _file_matches_cached(
                iit_existing,
                str(iit_spec["sha256"]).lower(),
                int(iit_spec.get("size_bytes") or 0),
                sha_cache,
            )
            cache_dirty = cache_dirty or dirty
            if not ok:
                updates.append(
                    {
                        "kind": "model_file",
                        "id": f"model:{iit_rel}",
                        "pack_id": "iit_template",
                        "relpath": iit_rel,
                        "size_bytes": int(iit_spec.get("size_bytes") or 0),
                        "sha256": str(iit_spec["sha256"]).lower(),
                        "urls": list(iit_spec.get("urls") or ()),
                    }
                )

    # 2. Optional WMA atlases (ONLY inspect atlases already installed on this machine).
    for pack_id, wspec in (comp.get("wma_packs") or {}).items():
        directory = str(wspec.get("directory") or "")
        index_rel = str(wspec.get("index_relpath") or "")
        index_sha = str(wspec.get("index_sha256") or "").lower()
        if not directory or not index_rel or len(index_sha) != 64:
            continue
        installed_root = next(
            (r for r in model_roots if (r / directory).is_dir() and any((r / directory).iterdir())),
            None,
        )
        if installed_root is None:
            continue
        idx_file = installed_root / index_rel
        ok, dirty = _file_matches_cached(idx_file, index_sha, 0, sha_cache)
        cache_dirty = cache_dirty or dirty
        wma_valid = ok
        if wma_valid:
            try:
                idx_payload = json.loads(idx_file.read_text(encoding="utf-8"))
                for item in idx_payload.get("files") or ():
                    f_rel = f"{directory}/{item['path']}"
                    f_path = installed_root / f_rel
                    if not f_path.is_file():
                        wma_valid = False
                        break
                    exp_b = int(item.get("size_bytes") or item.get("bytes") or 0)
                    if exp_b > 0 and f_path.stat().st_size != exp_b:
                        wma_valid = False
                        break
            except Exception:
                wma_valid = False
        if not wma_valid:
            updates.append(
                {
                    "kind": "wma_pack",
                    "id": f"wma:{pack_id}",
                    "pack_id": str(pack_id),
                    "directory": directory,
                    "filename": str(wspec.get("filename") or ""),
                    "url": str(wspec.get("url") or ""),
                    "sha256": str(wspec.get("sha256") or "").lower(),
                    "size_bytes": int(wspec.get("size_bytes") or 0),
                    "parts": list(wspec.get("parts") or ()),
                }
            )

    if cache_dirty:
        try:
            save_sha256_cache(cache_path, sha_cache)
        except OSError:
            pass

    # 3. Bundled Python algorithm sources (wmql, wma, tractcloud).
    try:
        from shared.backend_sources import source_root, verify_source

        src_roots = [source_root(), *_bundled_source_roots_for(resolved_module)]
        dedup_src_roots: list[Path] = []
        for r in src_roots:
            rp = Path(r).expanduser().resolve()
            if rp not in dedup_src_roots:
                dedup_src_roots.append(rp)

        for sname, sspec in (comp.get("algorithm_sources") or {}).items():
            prefix = f"{sname}-"
            any_installed = any(
                r.is_dir() and any(c.is_dir() and c.name.startswith(prefix) for c in r.iterdir())
                for r in dedup_src_roots
            )
            if not any_installed:
                continue
            target_dir_name = f"{sname}-{str(sspec['revision'])[:12]}"
            verified = False
            for r in dedup_src_roots:
                cand = r / target_dir_name
                if cand.is_dir():
                    try:
                        verify_source(sname, cand, spec=sspec)
                        verified = True
                        break
                    except Exception:
                        continue
            if not verified:
                updates.append(
                    {
                        "kind": "algorithm_source",
                        "id": f"source:{sname}",
                        "name": str(sname),
                        "size_bytes": int(sspec.get("size_bytes") or 0),
                        "sha256": str(sspec.get("sha256") or "").lower(),
                        "spec": dict(sspec),
                    }
                )
    except Exception:
        pass

    # 4. Standalone native runtimes (surface-runtime, ukf-standalone, standalone mrtrix/fsl).
    sys_name, arch = _host_platform_pair(system, machine)
    runtimes_spec = comp.get("runtimes") or {}
    bundled_rt_roots = _bundled_runtime_roots_for(resolved_module)

    # 4a. Surface runtime (only if no explicit external FREESURFER_HOME / SLICERDSPACE_SURFACE_RUNTIME is configured)
    if not os.environ.get("SLICERDSPACE_SURFACE_RUNTIME") and not os.environ.get("FREESURFER_HOME"):
        surf_rec = next(
            (
                r
                for r in (runtimes_spec.get("surface") or ())
                if r.get("system") == sys_name and r.get("machine") == arch and r.get("published")
            ),
            None,
        )
        if surf_rec is not None:
            min_macos = str(surf_rec.get("min_macos") or "").strip()
            macos_ok = True
            if sys_name == "Darwin" and min_macos:
                cur_mac = _macos_version_tuple()
                req_parts = [int(x) for x in min_macos.split(".")[:2]]
                req_mac = (req_parts[0], req_parts[1] if len(req_parts) > 1 else 0)
                if cur_mac is not None and cur_mac < req_mac:
                    macos_ok = False
            if macos_ok:
                surf_cands: list[Path] = []
                for rt_root in bundled_rt_roots:
                    for folder in (f"surface-runtime-{sys_name.lower()}-{arch}", f"{sys_name.lower()}-{arch}"):
                        if (rt_root / folder).is_dir():
                            surf_cands.append(rt_root / folder)
                try:
                    from shared.surface_runtime import default_surface_runtime, is_valid_surface_runtime_dir, runtime_dir_macos_block

                    def_pref = Path(default_surface_runtime())
                    if def_pref.is_dir() and def_pref not in surf_cands:
                        surf_cands.append(def_pref)
                except Exception:
                    is_valid_surface_runtime_dir = lambda p: False  # type: ignore[assignment]
                    runtime_dir_macos_block = lambda p: None  # type: ignore[assignment]

                if surf_cands:
                    surf_up_to_date = False
                    for cand in surf_cands:
                        if not is_valid_surface_runtime_dir(cand):
                            continue
                        if runtime_dir_macos_block(cand) is not None:
                            continue
                        try:
                            rt_json = json.loads((cand / "runtime.json").read_text(encoding="utf-8"))
                            if rt_json.get("recipe_sha256") != surf_rec.get("recipe_sha256"):
                                continue
                            files_dict = rt_json.get("files") or {}
                            unpacked = sum(int(v.get("bytes") or 0) for v in files_dict.values() if isinstance(v, dict))
                            if int(surf_rec.get("unpacked_bytes") or 0) and unpacked != int(surf_rec["unpacked_bytes"]):
                                continue
                            stamp = cand / ".slicerdspace_archive_sha256"
                            if stamp.is_file() and stamp.read_text(encoding="utf-8").strip().lower() != str(surf_rec["sha256"]).lower():
                                continue
                            surf_up_to_date = True
                            break
                        except Exception:
                            continue
                    if not surf_up_to_date:
                        updates.append(
                            {
                                "kind": "surface_runtime",
                                "id": "runtime:surface",
                                "size_bytes": int(surf_rec.get("bytes") or 0),
                                "sha256": str(surf_rec.get("sha256") or "").lower(),
                                "url": str(surf_rec.get("url") or ""),
                                "record": dict(surf_rec),
                            }
                        )

    # 4b. Standalone UKF runtime (for packaged OS Base installs that carry bundled-runtimes/)
    if bundled_rt_roots and not os.environ.get("SLICERDSPACE_UKF_EXECUTABLE"):
        ukf_rec = next(
            (
                r
                for r in (runtimes_spec.get("ukf_standalone") or ())
                if r.get("system") == sys_name and r.get("machine") == arch
            ),
            None,
        )
        if ukf_rec is not None:
            avx2_ok = True
            if ukf_rec.get("requires_avx2"):
                try:
                    from tractography.ukf import _host_supports_avx2

                    if _host_supports_avx2() is False:
                        avx2_ok = False
                except Exception:
                    pass
            if avx2_ok:
                dir_name = str(ukf_rec["dir_name"])
                exe_rel = str(ukf_rec["executable"])
                exp_commit = str(ukf_rec.get("source_commit") or "")
                ukf_roots = list(bundled_rt_roots)
                try:
                    from shared.native_runtimes import runtime_root

                    ukf_roots.append(runtime_root())
                except Exception:
                    pass
                ukf_ok = False
                for r in ukf_roots:
                    cand = r / dir_name
                    rt_file = cand / "runtime.json"
                    exe_file = cand / exe_rel
                    if rt_file.is_file() and exe_file.is_file():
                        try:
                            rt_data = json.loads(rt_file.read_text(encoding="utf-8"))
                            if not exp_commit or str(rt_data.get("source_commit") or "") == exp_commit:
                                ukf_ok = True
                                break
                        except Exception:
                            continue
                if not ukf_ok:
                    updates.append(
                        {
                            "kind": "ukf_runtime",
                            "id": "runtime:ukf",
                            "size_bytes": int(ukf_rec.get("bytes") or 0),
                            "sha256": str(ukf_rec.get("sha256") or "").lower(),
                            "url": str(ukf_rec.get("url") or ""),
                            "record": dict(ukf_rec),
                        }
                    )

    # 4c. Standalone native runtimes (mrtrix / fsl) when standalone archives are published and sha256 differs.
    for nrec in runtimes_spec.get("native") or ():
        if nrec.get("system") != sys_name or nrec.get("machine") != arch:
            continue
        name = str(nrec.get("name") or "")
        if not name:
            continue
        prefix = f"{name}-"
        installed_dirs = [
            child
            for r in bundled_rt_roots
            if r.is_dir()
            for child in r.iterdir()
            if child.is_dir() and child.name.startswith(prefix)
        ]
        if not installed_dirs:
            continue
        exp_sha = str(nrec.get("sha256") or "").lower()
        already_ok = False
        for d in installed_dirs:
            rcpt = d / "native-runtime.json"
            if rcpt.is_file():
                try:
                    if str(json.loads(rcpt.read_text(encoding="utf-8")).get("archive_sha256") or "").lower() == exp_sha:
                        already_ok = True
                        break
                except Exception:
                    pass
        if not already_ok:
            updates.append(
                {
                    "kind": "native_runtime",
                    "id": f"runtime:{name}",
                    "name": name,
                    "size_bytes": int(nrec.get("bytes") or 0),
                    "sha256": exp_sha,
                    "url": str(nrec.get("url") or ""),
                    "record": dict(nrec),
                }
            )

    return updates


def _download_verified_stream(
    urls: list[str],
    dest: Path,
    *,
    expected_sha256: str,
    expected_size: int = 0,
    offline_dir: str = "",
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
    timeout: int = 120,
    on_bytes: Callable[[int], None] | None = None,
) -> Path:
    from shared.code_updater import EXTENSION_VERSION, _candidate_urls_with_fallback
    from shared.network_config import find_local_pack_archive, open_url
    from shared.update_journal import win_fs_path

    exp_sha = expected_sha256.strip().lower()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if offline_dir and urlopen_fn is None:
        for cand_name in dict.fromkeys([dest.name, *[PurePosixPath(u.split("?")[0]).name for u in urls if u]]):
            local_hit = find_local_pack_archive(offline_dir, cand_name, expected_sha256=exp_sha)
            if local_hit is not None and local_hit.is_file():
                shutil.copy2(win_fs_path(local_hit), win_fs_path(dest))
                actual_size = win_fs_path(dest).stat().st_size
                if (expected_size <= 0 or actual_size == expected_size) and _sha256_file(dest) == exp_sha:
                    if on_bytes is not None:
                        on_bytes(actual_size)
                    return dest
                win_fs_path(dest).unlink(missing_ok=True)

    candidates: list[str] = []
    for raw_u in urls:
        for exp_u in _candidate_urls_with_fallback(raw_u, mirror_prefix=mirror_prefix):
            if exp_u not in candidates:
                candidates.append(exp_u)

    part_path = dest.with_name(f".{dest.name}.part.{os.getpid()}")
    part_fs = win_fs_path(part_path)
    errors: list[str] = []
    for candidate in candidates:
        part_fs.unlink(missing_ok=True)
        req = urllib.request.Request(
            candidate,
            headers={"User-Agent": f"SlicerDSpace-Updater/{EXTENSION_VERSION}"},
        )
        try:
            downloaded = 0
            digest = hashlib.sha256()
            with open_url(req, timeout=timeout, proxy_url=proxy_url, urlopen_fn=urlopen_fn) as resp:
                with part_fs.open("wb") as out_fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out_fh.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if expected_size > 0 and downloaded > expected_size + 1024 * 1024:
                            raise ValueError(f"Download exceeded expected size for {dest.name}")
                        if on_bytes is not None:
                            on_bytes(downloaded)
            if expected_size > 0 and downloaded != expected_size:
                raise ValueError(f"Size mismatch for {dest.name}: {downloaded} != {expected_size}")
            actual_sha = digest.hexdigest()
            if actual_sha != exp_sha:
                raise ValueError(f"SHA-256 mismatch for {dest.name}: {actual_sha} != {exp_sha}")
            os.replace(part_fs, win_fs_path(dest))
            return dest
        except Exception as exc:
            part_fs.unlink(missing_ok=True)
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(f"Failed to download {dest.name}: " + "; ".join(errors[-3:]))


def _atomic_swap_directory(staged_dir: Path, target_dir: Path) -> None:
    from shared.external_process import rename_with_retry
    from shared.update_journal import win_fs_path

    target_fs = win_fs_path(target_dir)
    staged_fs = win_fs_path(staged_dir)
    target_fs.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir.with_name(f".{target_dir.name}.old.{os.getpid()}")
    backup_fs = win_fs_path(backup_dir)
    shutil.rmtree(backup_fs, ignore_errors=True)

    had_existing = target_fs.exists()
    if had_existing:
        rename_with_retry(target_fs, backup_fs)
    try:
        rename_with_retry(staged_fs, target_fs)
    except BaseException:
        if had_existing and backup_fs.exists() and not target_fs.exists():
            try:
                rename_with_retry(backup_fs, target_fs)
            except Exception:
                pass
        raise
    if had_existing:
        shutil.rmtree(backup_fs, ignore_errors=True)


def _extract_archive_clean(archive_path: Path, staging_dir: Path, fmt: str) -> None:
    from shared.update_journal import win_fs_path

    staging_root = Path(os.path.abspath(os.fspath(staging_dir)))
    if fmt == "zip":
        with zipfile.ZipFile(win_fs_path(archive_path), "r") as zf:
            for member in zf.infolist():
                raw = member.filename.replace("\\", "/").strip("/")
                if not raw:
                    continue
                p = PurePosixPath(raw)
                if p.is_absolute() or ".." in p.parts or any(":" in x for x in p.parts):
                    raise ValueError(f"Unsafe zip member: {member.filename!r}")
                mode = (member.external_attr >> 16)
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Refusing symlink in zip archive: {member.filename!r}")
                target = staging_dir.joinpath(*p.parts)
                if not Path(os.path.abspath(os.fspath(target))).is_relative_to(staging_root):
                    raise ValueError(f"Archive member escapes staging root: {member.filename!r}")
                t_fs = win_fs_path(target)
                if member.is_dir():
                    t_fs.mkdir(parents=True, exist_ok=True)
                    continue
                t_fs.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, t_fs.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if os.name != "nt":
                    exec_bit = bool(mode & 0o111) or p.as_posix().startswith("bin/") or "/bin/" in p.as_posix()
                    t_fs.chmod(0o755 if exec_bit else 0o644)
    else:
        with tarfile.open(win_fs_path(archive_path), "r:*") as tf:
            for member in tf:
                if member.issym() or member.islnk() or member.isdev() or member.ischr() or member.isblk() or member.isfifo():
                    raise ValueError(f"Refusing special member in tar archive: {member.name!r}")
                raw = member.name.replace("\\", "/").strip("/")
                if not raw or raw == ".":
                    continue
                p = PurePosixPath(raw)
                if p.is_absolute() or ".." in p.parts or any(":" in x for x in p.parts):
                    raise ValueError(f"Unsafe tar member: {member.name!r}")
                target = staging_dir.joinpath(*p.parts)
                if not Path(os.path.abspath(os.fspath(target))).is_relative_to(staging_root):
                    raise ValueError(f"Archive member escapes staging root: {member.name!r}")
                t_fs = win_fs_path(target)
                if member.isdir():
                    t_fs.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                t_fs.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    raise ValueError(f"Cannot read tar member: {member.name!r}")
                with src, t_fs.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if os.name != "nt":
                    exec_bit = bool((member.mode or 0) & 0o111) or p.as_posix().startswith("bin/") or "/bin/" in p.as_posix()
                    t_fs.chmod(0o755 if exec_bit else 0o644)


def apply_component_updates(
    module_root: Path,
    updates: list[dict[str, Any]],
    *,
    timeout: int = 120,
    offline_dir: Path | str | None = None,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    phase_callback: PhaseCallback | None = None,
    explicit_model_root: Path | None = None,
    base_completed_bytes: int = 0,
    grand_total_bytes: int | None = None,
) -> dict[str, Any]:
    """Download, verify, and atomically replace each outdated companion component."""

    if not updates:
        return {"updated_components": [], "updated_count": 0}

    from shared.code_updater import _replace_with_retry
    from shared.model_asset_integrity import cached_sha256_of_file, load_sha256_cache, save_sha256_cache
    from shared.model_assets import _sha256_cache_path, configured_model_assets_root
    from shared.network_config import (
        extract_modular_pack,
        find_local_pack_archive,
        load_network_settings,
        merge_split_archive,
    )
    from shared.update_journal import win_fs_path

    resolved_module = module_root.expanduser().resolve()
    settings = load_network_settings()
    resolved_offline = str(offline_dir if offline_dir is not None else settings.get("offline_assets_dir") or "").strip()

    user_model_root = configured_model_assets_root(explicit_model_root)
    bundled_model_roots = _bundled_model_roots_for(resolved_module)
    bundled_src_roots = _bundled_source_roots_for(resolved_module)
    bundled_rt_roots = _bundled_runtime_roots_for(resolved_module)

    comp_total = sum(max(0, int(u.get("size_bytes") or 0)) for u in updates)
    total_bytes = grand_total_bytes if grand_total_bytes is not None else (base_completed_bytes + comp_total)
    done_so_far = int(base_completed_bytes)

    cache_path = _sha256_cache_path()
    sha_cache = load_sha256_cache(cache_path)
    updated_ids: list[str] = []

    tmp_work = Path(tempfile.mkdtemp(prefix="slicerdspace_comp_upd_"))
    try:
        for idx, item in enumerate(updates):
            kind = str(item.get("kind") or "")
            item_size = max(0, int(item.get("size_bytes") or 0))

            def _report_item_bytes(cur_bytes: int, _base: int = done_so_far) -> None:
                if progress_callback is not None and total_bytes > 0:
                    progress_callback(min(total_bytes, _base + max(0, int(cur_bytes))), total_bytes)

            if phase_callback is not None:
                phase_callback("downloading")

            if kind == "model_file":
                rel = str(item["relpath"])
                exp_sha = str(item["sha256"]).lower()
                dl_tmp = tmp_work / f"model_{idx}_{PurePosixPath(rel).name}"
                _download_verified_stream(
                    list(item["urls"]),
                    dl_tmp,
                    expected_sha256=exp_sha,
                    expected_size=item_size,
                    offline_dir=resolved_offline,
                    mirror_prefix=mirror_prefix,
                    proxy_url=proxy_url,
                    urlopen_fn=urlopen_fn,
                    timeout=timeout,
                    on_bytes=_report_item_bytes,
                )
                if phase_callback is not None:
                    phase_callback("applying")

                # Choose destination(s): replace in-place inside writable bundled-model-assets when it holds an outdated copy;
                # also update user_model_root if it already had a copy, or fall back to user_model_root if bundled is read-only.
                wrote_any = False
                for b_root in bundled_model_roots:
                    b_cand = b_root / rel
                    if b_cand.exists() and _ensure_writable_or_parent(b_cand.parent):
                        b_cand.parent.mkdir(parents=True, exist_ok=True)
                        tmp_dst = b_cand.with_name(f".{b_cand.name}.upd.{os.getpid()}")
                        shutil.copy2(win_fs_path(dl_tmp), win_fs_path(tmp_dst))
                        _replace_with_retry(tmp_dst, b_cand)
                        cached_sha256_of_file(b_cand, sha_cache)
                        wrote_any = True

                u_cand = user_model_root / rel
                if not wrote_any or u_cand.exists():
                    win_fs_path(u_cand.parent).mkdir(parents=True, exist_ok=True)
                    tmp_dst = u_cand.with_name(f".{u_cand.name}.upd.{os.getpid()}")
                    shutil.copy2(win_fs_path(dl_tmp), win_fs_path(tmp_dst))
                    _replace_with_retry(tmp_dst, u_cand)
                    cached_sha256_of_file(u_cand, sha_cache)

                win_fs_path(dl_tmp).unlink(missing_ok=True)
                updated_ids.append(str(item["id"]))

            elif kind == "wma_pack":
                directory = str(item["directory"])
                target_base = next(
                    (
                        r
                        for r in [user_model_root, *bundled_model_roots]
                        if (r / directory).exists() and _ensure_writable_or_parent(r / directory)
                    ),
                    user_model_root,
                )
                pack_filename = str(item["filename"] or f"{item['pack_id']}.tar.gz")
                pack_archive = tmp_work / pack_filename
                parts_list = list(item.get("parts") or ())
                local_wma = (
                    find_local_pack_archive(resolved_offline, pack_filename, expected_sha256=str(item["sha256"]))
                    if resolved_offline and urlopen_fn is None
                    else None
                )
                if local_wma is not None and local_wma.is_file():
                    shutil.copy2(win_fs_path(local_wma), win_fs_path(pack_archive))
                    _report_item_bytes(item_size)
                elif parts_list:
                    parts_dir = tmp_work / f"wma_parts_{idx}"
                    parts_dir.mkdir(parents=True, exist_ok=True)
                    part_done = 0
                    for pspec in parts_list:
                        pname = str(pspec["filename"])
                        psize = int(pspec.get("size_bytes") or 0)
                        _download_verified_stream(
                            [str(pspec["url"])],
                            parts_dir / pname,
                            expected_sha256=str(pspec["sha256"]),
                            expected_size=psize,
                            offline_dir=resolved_offline,
                            mirror_prefix=mirror_prefix,
                            proxy_url=proxy_url,
                            urlopen_fn=urlopen_fn,
                            timeout=max(timeout, 300),
                            on_bytes=lambda b, _pd=part_done: _report_item_bytes(_pd + b),
                        )
                        part_done += psize
                    merged = merge_split_archive(
                        parts_dir,
                        pack_filename,
                        expected_sha256=str(item["sha256"]),
                        output_dir=tmp_work,
                    )
                    shutil.rmtree(win_fs_path(parts_dir), ignore_errors=True)
                    if merged is None or not merged.is_file():
                        raise RuntimeError(f"Failed to merge split WMA archive {pack_filename}")
                    pack_archive = merged
                else:
                    _download_verified_stream(
                        [str(item["url"])],
                        pack_archive,
                        expected_sha256=str(item["sha256"]),
                        expected_size=item_size,
                        offline_dir=resolved_offline,
                        mirror_prefix=mirror_prefix,
                        proxy_url=proxy_url,
                        urlopen_fn=urlopen_fn,
                        timeout=max(timeout, 300),
                        on_bytes=_report_item_bytes,
                    )
                if phase_callback is not None:
                    phase_callback("applying")
                stage_extract = tmp_work / f"wma_stage_{idx}"
                stage_extract.mkdir(parents=True, exist_ok=True)
                extract_modular_pack(pack_archive, stage_extract, expected_sha256=str(item["sha256"]))
                staged_sub = stage_extract.joinpath(*PurePosixPath(directory).parts)
                if not staged_sub.is_dir():
                    raise RuntimeError(f"Extracted WMA pack missing expected directory {directory}")
                _atomic_swap_directory(staged_sub, target_base.joinpath(*PurePosixPath(directory).parts))
                shutil.rmtree(win_fs_path(stage_extract), ignore_errors=True)
                win_fs_path(pack_archive).unlink(missing_ok=True)
                updated_ids.append(str(item["id"]))

            elif kind == "algorithm_source":
                from shared.backend_sources import file_hash, source_root, verify_source

                sname = str(item["name"])
                sspec = dict(item["spec"])
                dl_arc = tmp_work / f"src_{sname}.tar.gz"
                _download_verified_stream(
                    [str(sspec["url"])],
                    dl_arc,
                    expected_sha256=str(sspec["sha256"]),
                    expected_size=item_size,
                    offline_dir=resolved_offline,
                    mirror_prefix=mirror_prefix,
                    proxy_url=proxy_url,
                    urlopen_fn=urlopen_fn,
                    timeout=timeout,
                    on_bytes=_report_item_bytes,
                )
                if phase_callback is not None:
                    phase_callback("applying")

                dest_root = next(
                    (r for r in bundled_src_roots if _is_writable_dir(r)),
                    Path(source_root()).expanduser().resolve(),
                )
                dest_root.mkdir(parents=True, exist_ok=True)
                dest_dir = dest_root / f"{sname}-{str(sspec['revision'])[:12]}"
                stage_src = dest_root / f".stage_{sname}_{os.getpid()}"
                shutil.rmtree(win_fs_path(stage_src), ignore_errors=True)
                stage_src.mkdir(parents=True, exist_ok=True)
                try:
                    allowed_paths = set(sspec.get("paths") or ())
                    with tarfile.open(win_fs_path(dl_arc), "r:gz") as bundle:
                        for member in bundle:
                            parts = PurePosixPath(member.name).parts
                            if (
                                member.issym()
                                or member.islnk()
                                or member.isdev()
                                or ".." in parts
                                or member.name.startswith(("/", "\\"))
                                or "\\" in member.name
                                or ":" in member.name
                            ):
                                raise ValueError("Unsafe algorithm archive member")
                            if (
                                not member.isfile()
                                or len(parts) < 2
                                or parts[1] not in allowed_paths
                                or any(p in {"tests", "test", ".git", "__pycache__"} for p in parts[1:])
                            ):
                                continue
                            target = stage_src.joinpath(*parts[1:])
                            win_fs_path(target.parent).mkdir(parents=True, exist_ok=True)
                            src_fh = bundle.extractfile(member)
                            if src_fh is None:
                                continue
                            with src_fh, win_fs_path(target).open("wb") as out_fh:
                                shutil.copyfileobj(src_fh, out_fh)
                            if os.name != "nt":
                                win_fs_path(target).chmod(0o755 if (member.mode or 0) & 0o111 else 0o644)
                    files_rec = {
                        p.relative_to(stage_src).as_posix(): file_hash(p)
                        for p in stage_src.rglob("*")
                        if p.is_file()
                    }
                    receipt = {
                        "profile": sname,
                        "revision": sspec["revision"],
                        "archive_sha256": sspec["sha256"],
                        "files": files_rec,
                    }
                    (stage_src / ".slicerdspace-source.json").write_text(
                        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
                    )
                    verify_source(sname, stage_src, spec=sspec)
                    _atomic_swap_directory(stage_src, dest_dir)
                finally:
                    shutil.rmtree(win_fs_path(stage_src), ignore_errors=True)
                    win_fs_path(dl_arc).unlink(missing_ok=True)

                # Remove superseded legacy siblings (e.g. wma-4532b911386d) in writable roots.
                for clean_root in [dest_root, *[r for r in bundled_src_roots if _is_writable_dir(r)]]:
                    if not clean_root.is_dir():
                        continue
                    for sibling in list(clean_root.iterdir()):
                        if sibling.is_dir() and sibling.name.startswith(f"{sname}-") and sibling.name != dest_dir.name:
                            shutil.rmtree(win_fs_path(sibling), ignore_errors=True)
                updated_ids.append(str(item["id"]))

            elif kind == "surface_runtime":
                rec = dict(item["record"])
                sys_name = str(rec["system"]).lower()
                arch = str(rec["machine"])
                dl_arc = tmp_work / PurePosixPath(str(rec.get("url") or "surface-runtime.tar.gz")).name
                _download_verified_stream(
                    [str(rec["url"])],
                    dl_arc,
                    expected_sha256=str(rec["sha256"]),
                    expected_size=item_size,
                    offline_dir=resolved_offline,
                    mirror_prefix=mirror_prefix,
                    proxy_url=proxy_url,
                    urlopen_fn=urlopen_fn,
                    timeout=timeout,
                    on_bytes=_report_item_bytes,
                )
                if phase_callback is not None:
                    phase_callback("applying")

                writable_bundled = next((r for r in bundled_rt_roots if _is_writable_dir(r)), None)
                if writable_bundled is not None:
                    target_dir = writable_bundled / f"surface-runtime-{sys_name}-{arch}"
                else:
                    from shared.surface_runtime import default_surface_runtime

                    target_dir = Path(default_surface_runtime()).expanduser().resolve()
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                stage_rt = target_dir.parent / f".stage_surface_rt_{os.getpid()}"
                shutil.rmtree(win_fs_path(stage_rt), ignore_errors=True)
                stage_rt.mkdir(parents=True, exist_ok=True)
                try:
                    _extract_archive_clean(dl_arc, stage_rt, "tar.gz")
                    # Unwrap single top-level folder if present, otherwise files are directly at root.
                    entries = [c for c in stage_rt.iterdir()]
                    inner = entries[0] if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "runtime.json").is_file() else stage_rt
                    if not (inner / "runtime.json").is_file():
                        raise RuntimeError("Extracted surface runtime is missing runtime.json")
                    (inner / ".slicerdspace_archive_sha256").write_text(str(rec["sha256"]).lower() + "\n", encoding="utf-8")
                    _atomic_swap_directory(inner, target_dir)
                finally:
                    shutil.rmtree(win_fs_path(stage_rt), ignore_errors=True)
                    win_fs_path(dl_arc).unlink(missing_ok=True)
                updated_ids.append(str(item["id"]))

            elif kind == "ukf_runtime":
                rec = dict(item["record"])
                dir_name = str(rec["dir_name"])
                fmt = str(rec.get("format") or "tar.gz")
                dl_arc = tmp_work / str(rec.get("filename") or f"{dir_name}.{fmt}")
                _download_verified_stream(
                    [str(rec["url"])],
                    dl_arc,
                    expected_sha256=str(rec["sha256"]),
                    expected_size=item_size,
                    offline_dir=resolved_offline,
                    mirror_prefix=mirror_prefix,
                    proxy_url=proxy_url,
                    urlopen_fn=urlopen_fn,
                    timeout=timeout,
                    on_bytes=_report_item_bytes,
                )
                if phase_callback is not None:
                    phase_callback("applying")

                writable_bundled = next((r for r in bundled_rt_roots if _is_writable_dir(r)), None)
                if writable_bundled is not None:
                    dest_parent = writable_bundled
                else:
                    from shared.native_runtimes import runtime_root

                    dest_parent = runtime_root()
                dest_parent.mkdir(parents=True, exist_ok=True)
                target_dir = dest_parent / dir_name
                stage_ukf = dest_parent / f".stage_ukf_{os.getpid()}"
                shutil.rmtree(win_fs_path(stage_ukf), ignore_errors=True)
                stage_ukf.mkdir(parents=True, exist_ok=True)
                try:
                    _extract_archive_clean(dl_arc, stage_ukf, fmt)
                    inner = stage_ukf / dir_name if (stage_ukf / dir_name).is_dir() else stage_ukf
                    if not (inner / str(rec["executable"])).is_file():
                        raise RuntimeError(f"Extracted UKF runtime missing {rec['executable']}")
                    _atomic_swap_directory(inner, target_dir)
                finally:
                    shutil.rmtree(win_fs_path(stage_ukf), ignore_errors=True)
                    win_fs_path(dl_arc).unlink(missing_ok=True)
                updated_ids.append(str(item["id"]))

            elif kind == "native_runtime":
                from shared.native_runtimes import install_bundle

                rec = dict(item["record"])
                dl_arc = tmp_work / str(rec.get("artifact") or f"{rec['name']}.{rec['format']}")
                _download_verified_stream(
                    [str(rec["url"])],
                    dl_arc,
                    expected_sha256=str(rec["sha256"]),
                    expected_size=item_size,
                    offline_dir=resolved_offline,
                    mirror_prefix=mirror_prefix,
                    proxy_url=proxy_url,
                    urlopen_fn=urlopen_fn,
                    timeout=timeout,
                    on_bytes=_report_item_bytes,
                )
                if phase_callback is not None:
                    phase_callback("applying")
                writable_bundled = next((r for r in bundled_rt_roots if _is_writable_dir(r)), None)
                target_root = writable_bundled
                if target_root is not None:
                    folder_name = f"{rec['name']}-{rec['version']}-{str(rec['system']).lower()}-{rec['machine']}"
                    if (target_root / folder_name).exists():
                        shutil.rmtree(win_fs_path(target_root / folder_name), ignore_errors=True)
                install_bundle(rec, archive=dl_arc, root=target_root, accept_terms=True)
                win_fs_path(dl_arc).unlink(missing_ok=True)
                updated_ids.append(str(item["id"]))

            done_so_far += item_size
            if progress_callback is not None and total_bytes > 0:
                progress_callback(min(total_bytes, done_so_far), total_bytes)

        try:
            save_sha256_cache(cache_path, sha_cache)
        except OSError:
            pass

        return {
            "updated_components": updated_ids,
            "updated_count": len(updated_ids),
        }
    finally:
        shutil.rmtree(tmp_work, ignore_errors=True)
