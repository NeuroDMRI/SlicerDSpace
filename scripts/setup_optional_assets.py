#!/usr/bin/env python3
"""Prepare only selected non-code resources, without installing any runtime."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_model_assets import download_asset_plan, load_manifest, resolve_asset_plan, verify_asset_plan
from scripts.import_external_resources import extract_org, import_file
from shared.managed_runtimes import write_json
from shared.model_assets import configured_model_assets_root, _sha256_cache
from shared.optional_assets import MANIFEST, verified_resource_directory
from shared.model_asset_integrity import verify_against_manifest

PROFILES = ("pnl-cnn", "tractcloud", "wma", "wma-east-west", "wma-naba", "surface-postprocess")
PATH_SETTINGS = {"pnl-cnn": "pnl_mask_model_dir", "tractcloud": "tractcloud_data_dir", "wma": "wma_atlas_dir",
                 "wma-east-west": "wma_atlas_dir", "wma-naba": "wma_atlas_dir"}


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
        existing = verified_resource_directory(profile, asset_root=root)
        if existing is not None:
            print("[Resources] Reusing verified " + profile, flush=True)
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
        if offline_root:
            for relative, record in plan.asset_records.items():
                import_file(Path(offline_root) / relative, root / relative, record)
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
