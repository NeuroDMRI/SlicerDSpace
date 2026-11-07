"""Verified optional resources under the explicitly configured user asset root."""

import json
from pathlib import Path

from shared.model_assets import configured_model_assets_root, _sha256_cache, _persist_sha256_cache
from shared.model_asset_integrity import ModelAssetIntegrityError, verify_against_manifest


MANIFEST = Path(__file__).resolve().parents[1] / "models/external_assets_manifest.json"


def verified_resource_directory(profile, *, asset_root=None):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    definition = manifest["profiles"][profile]
    root = configured_model_assets_root(asset_root)
    paths, visited = [], set()

    def collect(name):
        if name in visited:
            return
        visited.add(name)
        item = manifest["profiles"][name]
        for child in item.get("include_profiles", []):
            collect(child)
        paths.extend(item.get("required_paths", []))

    collect(profile)
    if not paths or any(not (root / path).is_file() for path in paths):
        return None
    cache = _sha256_cache()
    records = {relative: manifest["asset_records"][relative] for relative in dict.fromkeys(paths)}
    if profile == "wma" or definition.get("extracted_index"):
        index_relative = definition.get("extracted_index", "bundles/wma/extracted_resource_index.json")
        index_path = root / index_relative
        ok, reason = verify_against_manifest(index_path, index_relative,
                                            records, cache=cache)
        if index_path.is_symlink() or not ok:
            raise ModelAssetIntegrityError("ORG extraction index integrity check failed: " + reason)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for item in index["files"]:
            relative = definition["directory"] + "/" + item["path"]
            if (".." in Path(item["path"]).parts or Path(item["path"]).is_absolute()
                    or "\\" in item["path"] or ":" in item["path"]):
                raise ModelAssetIntegrityError("Unsafe ORG index path")
            if not (root / relative).is_file():
                return None
            records[relative] = item
    for relative, record in records.items():
        path = root / relative
        if not record.get("sha256"):
            raise ModelAssetIntegrityError('Optional resource has no trusted hash: ' + relative)
        ok, reason = verify_against_manifest(path, relative, records, cache=cache)
        if any(part.is_symlink() for part in (path, *path.parents)) or not ok:
            raise ModelAssetIntegrityError(f"Optional resource integrity check failed: {relative}: {reason}")
    _persist_sha256_cache()
    return root / definition["directory"]


def resolve_wma_atlas_directory(explicit=None, *, asset_root=None, atlas_family="org"):
    """Keep explicit/legacy paths authoritative; use the shared atlas otherwise."""
    import os
    from tractography.wma_atlases import atlas_definition
    definition = atlas_definition(atlas_family)
    value = explicit or os.environ.get("SLICERDSPACE_WMA_ATLAS_" + atlas_family.upper().replace("-", "_") + "_DIR")
    if not value and atlas_family == "org":
        value = os.environ.get("SLICERDSPACE_WMA_ATLAS_DIR") or os.environ.get("WMA_ATLAS_DIR")
    return Path(value).expanduser().absolute() if value else verified_resource_directory(definition["profile"], asset_root=asset_root)


def inherited_wma_atlas_directory(value, atlas_family, asset_root=None):
    """Do not carry a managed or different-family path into a new atlas run."""
    from tractography.wma_atlases import ATLAS_DEFINITIONS, atlas_definition
    selected = atlas_definition(atlas_family)
    if not value:
        return None
    value = Path(value).expanduser()
    if asset_root and any(value.resolve() == (Path(asset_root).expanduser() / item["directory"]).resolve()
                          for item in ATLAS_DEFINITIONS.values()):
        return None
    if all((value / child).is_dir() for child in selected["directories"]):
        return value
    if any(name != atlas_family and all((value / child).is_dir() for child in item["directories"])
           for name, item in ATLAS_DEFINITIONS.items()):
        return None
    return value


def resolve_tractcloud_data_directory(explicit=None, *, asset_root=None):
    import os
    value = explicit or os.environ.get("SLICERDSPACE_TRACTCLOUD_DATA_DIR") or os.environ.get("TRACTCLOUD_DATA_DIR")
    return Path(value).expanduser().absolute() if value else verified_resource_directory("tractcloud", asset_root=asset_root)


PNL_MODEL_GAPS = ("pnl_mask_model_dir", "weights.json", "IITmean_b0_256.nii.gz", ".pt ")


def missing_optional_profiles(result, brain_mask_backend, wma_atlas="org", pnl_route="managed"):
    """Infer resource actions from selected required blockers, never all backends."""
    profiles = []
    for item in result.statuses:
        if not item.required or item.status != "missing":
            continue
        if item.key.startswith("wma_org_atlas"):
            from tractography.wma_atlases import atlas_definition
            profiles.append(atlas_definition(wma_atlas)["profile"])
        elif item.key == "tractcloud_data_dir" or item.key.startswith("tractcloud_data:"):
            profiles.append("tractcloud")
        elif item.key == "brain_mask_backend" and brain_mask_backend == "synthstrip":
            if any("synthstrip.1.pt" in str(reason) for reason in (item.details or {}).get("missing_required", [])):
                profiles.append("synthstrip")
        elif item.key == "brain_mask_backend" and brain_mask_backend == "pnl_cnn" and pnl_route == "native":
            # SC-1c: the native route needs the brain_mask assets; it installs them only when nothing else is missing.
            reasons = [str(reason) for reason in (item.details or {}).get("missing_required", [])]
            if reasons and all(any(gap in reason + " " for gap in PNL_MODEL_GAPS) for reason in reasons):
                profiles.append("pnl-cnn")
    return list(dict.fromkeys(profiles))
