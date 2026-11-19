#!/usr/bin/env python3
"""Prepare only selected non-code resources, without installing any runtime."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_model_assets import download_asset_plan, load_manifest, resolve_asset_plan, verify_asset_plan
from scripts.import_external_resources import extract_org, import_file
from shared.managed_runtimes import write_json
from shared.model_assets import bundled_model_asset_roots, configured_model_assets_root, _sha256_cache
from shared.network_config import (
    download_file_verified,
    extract_modular_pack,
    find_local_pack_archive,
    load_network_settings,
    load_release_packs_manifest,
    merge_split_archive,
)
from shared.optional_assets import MANIFEST, verified_resource_directory
from shared.model_asset_integrity import verify_against_manifest

PROFILES = ("pnl-cnn", "tractcloud", "wma", "wma-east-west", "wma-naba", "surface-postprocess")
PATH_SETTINGS = {"pnl-cnn": "pnl_mask_model_dir", "tractcloud": "tractcloud_data_dir", "wma": "wma_atlas_dir",
                 "wma-east-west": "wma_atlas_dir", "wma-naba": "wma_atlas_dir"}
_RELEASE_PACK_BY_PROFILE = {
    "wma": "wma_org",
    "wma-naba": "wma_naba",
    "wma-east-west": "wma_east_west",
}


def _try_prepare_from_release_pack(profile: str, root: Path, *, offline_root=None) -> bool:
    """Install a WMA atlas from a published single/split GitHub Release archive or offline pack."""
    pack_id = _RELEASE_PACK_BY_PROFILE.get(profile)
    if not pack_id:
        return False
    repo_manifest = (Path(__file__).resolve().parents[1] / "models/external_assets_manifest.json").resolve()
    if Path(MANIFEST).resolve() != repo_manifest:
        return False
    pack = (load_release_packs_manifest().get("packs") or {}).get(pack_id)
    if not isinstance(pack, dict):
        return False
    filename = str(pack.get("filename") or "")
    expected_sha = str(pack.get("sha256") or "")
    if not filename or not expected_sha:
        return False

    search_dir = offline_root or load_network_settings().get("offline_assets_dir") or ""
    if search_dir:
        local_pack = find_local_pack_archive(search_dir, filename, expected_sha256=expected_sha)
        if local_pack is not None:
            print(f"[Resources] Extracting offline release pack {local_pack.name} -> {root}", flush=True)
            extract_modular_pack(local_pack, root, expected_sha256=expected_sha)
            return verified_resource_directory(profile, asset_root=root) is not None

    if offline_root or not pack.get("published"):
        return False

    cache_dir = root / ".release-pack-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = list(pack.get("parts") or ())
    downloaded_files: list[Path] = []
    try:
        if parts:
            for part in parts:
                part_name = str(part.get("filename") or "")
                part_url = str(part.get("url") or "")
                part_sha = str(part.get("sha256") or "")
                part_size = int(part.get("size_bytes") or 0)
                if not part_name or not part_url:
                    return False
                dest = cache_dir / part_name
                download_file_verified(
                    part_url,
                    dest,
                    expected_sha256=part_sha,
                    expected_size=part_size,
                )
                downloaded_files.append(dest)
            merged = merge_split_archive(cache_dir, filename, expected_sha256=expected_sha, output_dir=cache_dir)
            if merged is None:
                return False
            downloaded_files.append(merged)
            print(f"[Resources] Extracting merged atlas pack {merged.name} -> {root}", flush=True)
            extract_modular_pack(merged, root, expected_sha256=expected_sha)
        elif pack.get("url"):
            dest = cache_dir / filename
            download_file_verified(
                str(pack["url"]),
                dest,
                expected_sha256=expected_sha,
                expected_size=int(pack.get("size_bytes") or 0),
            )
            downloaded_files.append(dest)
            print(f"[Resources] Extracting atlas pack {dest.name} -> {root}", flush=True)
            extract_modular_pack(dest, root, expected_sha256=expected_sha)
        else:
            return False
        return verified_resource_directory(profile, asset_root=root) is not None
    except Exception as exc:
        print(f"[Resources] Release pack acquisition fallback ({filename}: {exc})", file=sys.stderr, flush=True)
        return False
    finally:
        for path in downloaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if cache_dir.is_dir() and not any(cache_dir.iterdir()):
                cache_dir.rmdir()
        except OSError:
            pass


def _stage_bundled_assets(root: Path, plan, *, repo_root: Path | None = None) -> int:
    """Copy or hardlink verified files from extension ``bundled-model-assets`` into ``root`` so only truly missing files are downloaded."""
    resolved_repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    repo_manifest = (resolved_repo / "models/external_assets_manifest.json").resolve()
    if Path(MANIFEST).resolve() != repo_manifest:
        return 0
    bundled_roots = bundled_model_asset_roots(resolved_repo)
    if not bundled_roots:
        return 0
    resolved_root = root.expanduser().resolve()
    cache = _sha256_cache()
    staged = 0
    for relative, _record in plan.asset_records.items():
        dst = resolved_root / relative
        if dst.is_file():
            continue
        rel_parts = Path(relative).parts
        cur = resolved_root
        has_symlink = cur.is_symlink()
        for part in rel_parts:
            cur = cur / part
            if cur.is_symlink():
                has_symlink = True
                break
        if has_symlink:
            continue
        for bundled_root in bundled_roots:
            try:
                if bundled_root.resolve() == resolved_root:
                    continue
            except OSError:
                pass
            cand = bundled_root / relative
            if not cand.is_file() or cand.is_symlink():
                continue
            ok, _reason = verify_against_manifest(cand, relative, plan.asset_records, cache=cache)
            if not ok:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(cand, dst)
            except OSError:
                shutil.copy2(cand, dst)
            print(f"[Resources] Reusing bundled {relative}", flush=True)
            staged += 1
            break
    return staged


def _sync_iit_to_bundled(root: Path, *, repo_root: Path | None = None) -> None:
    """Best-effort mirror of downloaded ``IITmean_b0_256.nii.gz`` into a writable ``bundled-model-assets`` dir that already holds PNL .pt weights."""
    rel = Path("brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz")
    src = root / rel
    if not src.is_file():
        return
    resolved_repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    for bundled_root in bundled_model_asset_roots(resolved_repo):
        try:
            if bundled_root.resolve() == root.resolve():
                continue
        except OSError:
            continue
        weights = bundled_root / "brain_mask/pnl_cnn_torch/axial.pt"
        dst = bundled_root / rel
        if weights.is_file() and not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except OSError:
                pass


def prepare(profiles, *, asset_root=None, execute=False, accept_terms=False, offline_root=None):
    profiles = list(dict.fromkeys(profiles))
    if not profiles or set(profiles) - set(PROFILES):
        raise ValueError("Select supported optional resource profiles")
    root = configured_model_assets_root(asset_root)
    manifest = load_manifest(MANIFEST)
    plans = [resolve_asset_plan(manifest, profile) for profile in profiles]
    records = {path: record for plan in plans for path, record in plan.asset_records.items()}
    result = {"kind": "slicerdsurfer_optional_asset_setup", "status": "planned", "profiles": profiles,
              "root": str(root), "size_bytes": sum(int(record["size_bytes"]) for record in records.values()),
              "terms": list(dict.fromkeys(notice for plan in plans for notice in plan.terms_notices)),
              "external_dependencies": {}, "scope": "Resource integrity only; runtimes and scientific QC are separate"}
    definitions = [manifest["profiles"][profile] for profile in profiles]
    expanded = sum(int(item.get("expanded_size_bytes", 0)) for item in definitions)
    scratch = max((sum(int(records[part]["size_bytes"]) for part in item.get("archive_parts", ()))
                   for item in definitions), default=0)
    result["expanded_size_bytes"] = expanded
    result["estimated_peak_disk_bytes"] = result["size_bytes"] + expanded + scratch
    if not execute:
        return result
    if not accept_terms and not offline_root:
        raise PermissionError("Review resource terms and explicitly accept them before download")
    private_mirror = any(source.get("access") == "authorized-private" for source in manifest.get("asset_sources", {}).values())
    if private_mirror and not offline_root:
        unavailable = [relative for relative, record in records.items() if not (root / relative).is_file() and not record.get("urls")]
        if unavailable:
            raise PermissionError("The resource mirror is private and does not support anonymous downloads. "
                                  "Use an authorized materialized Git LFS checkout or --from-directory to import selected resources. "
                                  "No partial public download was started. Missing private resources: " + ", ".join(unavailable))
    for relative, record in records.items():
        target = root / relative
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise ValueError("Refusing a symlink resource destination: " + relative)
        if target.exists() and (not target.is_file() or not verify_against_manifest(
                target, relative, records, cache=_sha256_cache())[0]):
            raise ValueError("Existing resource differs; inspect it or choose a clean asset root: " + relative)
    for profile, plan in zip(profiles, plans):
        print("[Resources] Preparing " + profile, flush=True)
        _stage_bundled_assets(root, plan)
        existing = verified_resource_directory(profile, asset_root=root)
        if existing is not None:
            print("[Resources] Reusing verified " + profile, flush=True)
            if profile == "pnl-cnn":
                _sync_iit_to_bundled(root)
            if profile in PATH_SETTINGS:
                result["external_dependencies"][PATH_SETTINGS[profile]] = str(existing)
            continue
        definition = manifest["profiles"][profile]
        if definition.get("expanded_size_bytes"):
            missing_bytes = sum(record["size_bytes"] for relative, record in plan.asset_records.items()
                                if not (root / relative).is_file())
            scratch_bytes = sum(plan.asset_records[part]["size_bytes"] for part in definition.get("archive_parts", ()))
            required = missing_bytes + int(definition["expanded_size_bytes"]) + scratch_bytes
            disk = root
            while not disk.exists():
                disk = disk.parent
            if shutil.disk_usage(disk).free < required:
                raise OSError(f"Insufficient atlas storage: allow {required} additional bytes for download, extraction and scratch files under {root}")
        if _try_prepare_from_release_pack(profile, root, offline_root=offline_root):
            directory = verified_resource_directory(profile, asset_root=root)
            if directory is None:
                raise RuntimeError("Resources are incomplete after preparation: " + profile)
            if profile in PATH_SETTINGS:
                result["external_dependencies"][PATH_SETTINGS[profile]] = str(directory)
            continue
        if offline_root:
            offline_path = Path(offline_root)
            packs = (load_release_packs_manifest().get("packs") or {}).values()
            extracted_from_pack = False
            for pack in packs:
                if profile not in (pack.get("profiles") or ()):
                    continue
                local_pack = find_local_pack_archive(
                    offline_path,
                    str(pack.get("filename") or ""),
                    expected_sha256=str(pack.get("sha256") or ""),
                )
                if local_pack is not None:
                    extract_modular_pack(local_pack, root, expected_sha256=str(pack.get("sha256") or ""))
                    extracted_from_pack = True
            for relative, record in plan.asset_records.items():
                if (offline_path / relative).is_file():
                    import_file(offline_path / relative, root / relative, record)
                elif not extracted_from_pack:
                    import_file(offline_path / relative, root / relative, record)
        else:
            download_asset_plan(root, plan, manifest, source="url",
                                progress_callback=lambda event: print(json.dumps(event), flush=True))
        checked = verify_asset_plan(root, plan, check_hashes=True)
        if not checked["ok"]:
            raise RuntimeError("Resource verification failed: " + json.dumps(checked))
        if profile == "wma":
            # ZIPs and index are pinned before extracting or trusting any content.
            for relative in plan.required_paths:
                if relative.endswith(".zip"):
                    print("[Resources] Extracting and verifying " + relative, flush=True)
                    extract_org(root / relative, root / "bundles/wma")
        elif manifest["profiles"][profile].get("archive_parts"):
            from scripts.import_external_resources import extract_indexed_archive
            extract_indexed_archive(root, manifest["profiles"][profile])
        directory = verified_resource_directory(profile, asset_root=root)
        if directory is None:
            raise RuntimeError("Resources are incomplete after preparation: " + profile)
        if profile == "pnl-cnn":
            _sync_iit_to_bundled(root)
        if profile in PATH_SETTINGS:
            result["external_dependencies"][PATH_SETTINGS[profile]] = str(directory)
    result["external_dependencies"]["model_assets_dir"] = str(root)
    result["status"] = "completed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--from-directory", type=Path, help="Import only selected verified files; no network")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-terms", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.profiles, asset_root=args.asset_root, execute=args.execute,
                         accept_terms=args.accept_terms, offline_root=args.from_directory)
    except Exception as exc:
        result = {"kind": "slicerdsurfer_optional_asset_setup", "status": "failed", "error": str(exc)}
    if args.result:
        write_json(args.result, result)
    print(json.dumps(result, indent=2), flush=True)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
