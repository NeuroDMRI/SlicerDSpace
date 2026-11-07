"""Run record for review and archiving (T1 design D6, C40).

At the end of every run, after all manifests are final, ``run_record.json`` is written in the
run folder (``00_run_metadata``). It lists the run manifest, the parameters, the software and asset
versions, the accepted licences, a QC summary, and the SHA-256 of every final product file that the
run manifest names in its task outputs. It excludes itself, work folders and intermediates. It copies
no output file. Pure Python: no Slicer, Qt or heavy imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

RUN_RECORD_NAME = "run_record.json"
RUN_RECORD_KIND = "slicerddsurfer_run_record"
RUN_RECORD_VERSION = 1

# Path parts that mark a work folder or an intermediate: such files are not final products.
EXCLUDED_PATH_PARTS = frozenset({"cache", "work", "tmp", "temp", "intermediate", "intermediates", "scratch"})
# Manifest keys that are bulky run telemetry, not parameters.
NON_PARAMETER_KEYS = frozenset({
    "tasks", "performance_profile", "performance_profile_path", "phase_timings_seconds", "elapsed_seconds",
    "wall_time_seconds", "wall_time_scope", "report_paths", "report_error", "output_tree_report_paths",
    "output_tree_report_error", "output_tree_audit_path", "output_tree_warnings", "output_tree_semantics_path",
    "output_tree_semantics_summary", "failure", "ended_at", "status",
})
PACKAGES = ("numpy", "scipy", "nibabel", "dipy", "SimpleITK", "torch", "amico")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, Optional[str]]:
    from importlib import metadata

    versions: dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _is_final_product(path: Path, run_record_path: Path, output_root: Optional[Path] = None) -> bool:
    """False for the record itself and for files under a work folder or intermediate of the run output root.

    Only the parts of the path below the run output root count (T-B1): a root that sits under /tmp, /work,
    /scratch or a folder named cache or temp must not hide its products. A path outside the root is kept.
    """
    if path.name == RUN_RECORD_NAME or path == run_record_path or path.name.startswith("run_record-"):
        return False  # the record itself and the earlier records kept by B4
    if output_root is None:
        return True
    try:
        parts = path.relative_to(output_root).parts
    except ValueError:
        try:  # the same folder reached through a symbolic link
            parts = path.resolve().relative_to(Path(output_root).resolve()).parts
        except (OSError, ValueError):
            return True
    return not (set(part.lower() for part in parts[:-1]) & EXCLUDED_PATH_PARTS)


def _output_root(manifest: dict, manifest_path: Path) -> Path:
    """The run output root: the ``output_root`` of the manifest, else the folder above ``00_run_metadata``."""
    declared = manifest.get("output_root")
    return Path(declared) if declared else Path(manifest_path).parent.parent


def _iter_output_paths(value: Any):
    """Yield every string path in a task ``outputs`` value (dict values, lists, strings)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_output_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_output_paths(item)


def product_inventory(manifest: dict, run_record_path: Path, output_root: Optional[Path] = None) -> list[dict]:
    """SHA-256 inventory of the final product files in the task outputs of the run manifest."""
    if output_root is None:
        output_root = _output_root(manifest, run_record_path)
    products: dict[str, dict] = {}
    for task in manifest.get("tasks") or []:
        for key, value in (task.get("outputs") or {}).items():
            for text in _iter_output_paths(value):
                path = Path(text)
                if not path.is_absolute() or not _is_final_product(path, run_record_path, output_root):
                    continue
                if path.is_file():
                    products.setdefault(path.as_posix(), {
                        "task": task.get("name"), "key": key, "path": path.as_posix(),
                        "size": path.stat().st_size, "sha256": sha256_file(path)})
                elif not path.exists():
                    products.setdefault(path.as_posix(), {
                        "task": task.get("name"), "key": key, "path": path.as_posix(), "size": None, "sha256": None})
    return [products[name] for name in sorted(products)]


def _qc_summary(manifest: dict) -> dict:
    tasks = []
    counts: dict[str, int] = {}
    for task in manifest.get("tasks") or []:
        status = str(task.get("status"))
        counts[status] = counts.get(status, 0) + 1
        entry = {"name": task.get("name"), "status": status, "reason": task.get("reason") or ""}
        qc_path = task.get("qc_path")
        if qc_path and Path(qc_path).is_file():
            try:
                qc = json.loads(Path(qc_path).read_text(encoding="utf-8"))
                entry["qc_path"] = qc_path
                entry["qc_status"] = qc.get("status") if isinstance(qc, dict) else None
            except (OSError, ValueError):
                entry["qc_status"] = "unreadable"
        tasks.append(entry)
    return {"manifest_status": manifest.get("status"), "task_status_counts": counts, "tasks": tasks}


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFESTS = ("models/assets_manifest.json", "models/external_assets_manifest.json")
# Task name (before "@") -> model asset profile that the task reads. WMA and the brain mask depend on options.
TASK_ASSET_PROFILES = {
    "segmentation.ddseg": ("segmentation",), "parcellation.ddparcel": ("parcellation",),
    "surface.ddsurfer": ("surface",), "surface.ddsurfer_postprocess": ("surface", "surface-postprocess"),
    "bundles.tractcloud": ("tractcloud",),
}
BRAIN_MASK_PROFILES = {"pnl_cnn": ("pnl-cnn-weights",), "synthstrip": ("synthstrip",)}
USED_STATUSES = frozenset({"completed", "skipped"})


def _load_asset_manifests(repo_root: Path) -> list[dict]:
    loaded = []
    for relative in ASSET_MANIFESTS:
        try:
            data = json.loads((repo_root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            loaded.append({"path": relative, "version": data.get("version"), "profiles": data.get("profiles") or {},
                           "records": data.get("asset_records") or {},
                           "upstream_commit": data.get("ddsurfer_upstream_commit")})
    return loaded


def used_asset_profiles(manifest: dict) -> list[str]:
    """Names of the model asset profiles that the finished tasks of the run read."""
    names: list[str] = []
    tasks = [task for task in manifest.get("tasks") or [] if task.get("status") in USED_STATUSES]
    for task in tasks:
        name = str(task.get("name", "")).split("@", 1)[0]
        names.extend(TASK_ASSET_PROFILES.get(name, ()))
        if name == "bundles.wma":
            try:
                from tractography.wma_atlases import atlas_definition
                names.append(atlas_definition(manifest.get("wma_atlas") or "org")["profile"])
            except (ImportError, ValueError):
                names.append("wma")
    if tasks:
        names.extend(BRAIN_MASK_PROFILES.get(str(manifest.get("brain_mask_backend")), ()))
    return sorted(dict.fromkeys(names))


def _custom_model_root(manifest: dict) -> Optional[Path]:
    """The model folder named by the run option or ``SLICERDSPACE_MODEL_ASSETS_DIR``, or None when none is named.

    C2: a folder that equals the per-user default root is returned like any other. It is not trusted by its path:
    ``asset_versions`` checks it with the resolver's own check (``_default_root_matches_catalog``). The GUI saves the
    managed root as ``model_assets_dir``, and the resolver does not verify an explicit root at run time.
    """
    import os

    explicit = manifest.get("model_assets_dir") or os.environ.get("SLICERDSPACE_MODEL_ASSETS_DIR")
    return Path(explicit).expanduser() if explicit else None


def _is_default_model_root(folder: Path) -> bool:
    try:
        from shared.model_assets import default_user_model_assets_root
        return folder.resolve() == default_user_model_assets_root().expanduser().resolve()
    except (OSError, ImportError):
        return False


def _default_root_matches_catalog(files: dict, records: dict) -> bool:
    """C2: True when the per-user default root holds every catalogued file of the profile with the catalogued SHA-256.

    This is the check of the model-asset resolver (``_has_verified_required_asset``), one file at a time, with the
    records of the manifest that lists the profile. It uses the same (mtime, size) digest cache and the same cache keys
    (the default root as the resolver spells it), so a file that the resolver verified before is not hashed again.
    The managed install (``download_model_assets.py``) hashes with its own function and writes no cache entry: the
    first record after a fresh install hashes each catalogued file once.
    """
    expected = [path for path, digest in files.items() if digest]
    if not expected:
        return False
    try:
        from shared.model_assets import _has_verified_required_asset, default_user_model_assets_root
        root = default_user_model_assets_root()
        return all(_has_verified_required_asset(root, (path,), manifest_records=records) for path in expected)
    except (OSError, ImportError):
        return False


def _custom_pnl_model_dir(manifest: dict) -> Optional[Path]:
    """B3 (4): the PNL weights folder given by the run option or ``SLICERDSPACE_PNL_MASK_MODEL_DIR``, else None."""
    import os

    configuration = manifest.get("brain_mask_configuration")
    explicit = (configuration.get("pnl_mask_model_dir") if isinstance(configuration, dict) else None) \
        or os.environ.get("SLICERDSPACE_PNL_MASK_MODEL_DIR")
    return Path(explicit).expanduser() if explicit else None


def _receipt_check_names(profile: str) -> list[str]:
    """C3: the keys under which ``verify_offline_assets`` can record a check that covers ``profile``.

    The writer names a model profile by itself (``segmentation``) and an optional resource profile with the prefix
    ``optional:`` (``optional:pnl-cnn``). A verified profile that includes this one covers it as well: the
    ``optional:pnl-cnn`` check verifies the ``pnl-cnn-weights`` files.
    """
    def includes(name: str, wanted: str, profiles: dict, seen: set) -> bool:
        if name in seen:
            return False
        seen.add(name)
        children = (profiles.get(name) or {}).get("include_profiles") or []
        return wanted in children or any(includes(child, wanted, profiles, seen) for child in children)

    names = [profile, "optional:" + profile]
    for found in _load_asset_manifests(REPO_ROOT):
        for parent in found["profiles"]:
            if parent != profile and includes(parent, profile, found["profiles"], set()):
                names.extend((parent, "optional:" + parent))
    return list(dict.fromkeys(names))


def _receipt_verifies(folder: Path, profile: str, files: dict) -> bool:
    """B3 (2): True when a completed offline-asset verification receipt covers this folder and profile.

    ``scripts/verify_offline_assets.py`` writes ``offline-assets-*.json`` beside the user config after it checked the
    profile against the manifest. C3: the check names are those that the writer really uses (``_receipt_check_names``).
    The receipt counts only if no catalogued file is newer than the receipt.
    """
    try:
        from shared.user_config import default_user_config_path
        receipts = sorted(default_user_config_path().parent.glob("offline-assets-*.json"))
        wanted = folder.resolve()
        names = _receipt_check_names(profile)
        for receipt in receipts:
            data = json.loads(receipt.read_text(encoding="utf-8"))
            checks = data.get("checks") if isinstance(data, dict) else None
            if (not isinstance(checks, dict) or data.get("kind") != "slicerdsurfer_offline_assets"
                    or data.get("status") != "completed" or Path(str(data.get("root"))).resolve() != wanted
                    or not any((checks.get(name) or {}).get("ok") for name in names)):
                continue
            stamp = receipt.stat().st_mtime_ns
            if all((folder / path).is_file() and (folder / path).stat().st_mtime_ns <= stamp for path in files):
                return True
    except (OSError, ValueError, ImportError, AttributeError):
        return False
    return False


def _folder_matches_catalog(folder: Path, files: dict, profile: Optional[str] = None) -> bool:
    """True when the folder holds every catalogued file of the profile with the catalogued SHA-256.

    A completed offline-asset verification receipt for this folder and profile (``_receipt_verifies``) answers without
    hashing. Otherwise the digests come from the stored (mtime and size) cache of the model-asset resolver; a
    catalogued file that is not in the cache is hashed once, and the cache keeps the digest.
    """
    expected = {path: digest for path, digest in files.items() if digest}
    if not expected:
        return False
    if profile and _receipt_verifies(folder, profile, expected):
        return True
    try:
        from shared.model_asset_integrity import cached_sha256_of_file
        from shared.model_assets import _persist_sha256_cache, _sha256_cache

        cache = _sha256_cache()
        matched = all((folder / path).is_file() and cached_sha256_of_file(folder / path, cache) == digest
                      for path, digest in expected.items())
        _persist_sha256_cache()
        return matched
    except (OSError, ImportError):
        return False


def asset_versions(manifest: dict, repo_root: Optional[Path] = None) -> dict:
    """Version and SHA-256 values that the asset manifests already store, per profile that the run used.

    When no model folder is named, the values come from the manifests and nothing is hashed. A named model folder
    (run option or environment variable) is checked against the catalog before the catalog values are used, and a
    folder that does not match records the version ``"unknown"``. C2: this holds for a folder that equals the
    per-user default root too. That folder goes through the resolver's check (digest cache, else one hash per
    uncached catalogued file). Any other named folder goes through ``_folder_matches_catalog`` (a verification
    receipt, else the digest cache, else one hash per uncached file). A profile that no manifest lists gets the model
    folder and the version ``"unknown"``.
    """
    manifests = _load_asset_manifests(Path(repo_root) if repo_root else REPO_ROOT)
    folder = manifest.get("model_assets_dir")

    def paths_of(name: str, found: dict, seen: set) -> list[str]:
        if name in seen or name not in found["profiles"]:
            return []
        seen.add(name)
        profile = found["profiles"][name]
        paths = [path for child in profile.get("include_profiles", []) for path in paths_of(child, found, seen)]
        return paths + list(profile.get("required_paths", []))

    custom_root = _custom_model_root(manifest)
    default_root = custom_root is not None and _is_default_model_root(custom_root)
    if custom_root is not None:
        folder = custom_root.as_posix()
    result: dict[str, Any] = {}
    for name in used_asset_profiles(manifest):
        for found in manifests:
            if name not in found["profiles"]:
                continue
            files = {path: (found["records"].get(path) or {}).get("sha256")
                     for path in dict.fromkeys(paths_of(name, found, set()))}
            # T-C7b: a user-supplied model folder is not verified by the resolver. Take the catalog values
            # only when every catalogued file in that folder has the catalogued SHA-256; else record "unknown".
            if custom_root is not None and not (
                    _default_root_matches_catalog(files, found["records"]) if default_root
                    else _folder_matches_catalog(custom_root, files, name)):
                result[name] = {"manifest": None, "version": "unknown", "folder": folder, "files": {}}
                break
            # B3 (4): a custom PNL weights folder is not the catalog folder. Its files sit beside each other, so the
            # catalogued names are looked up there; a folder that does not match records "unknown", not the catalog hashes.
            pnl_dir = _custom_pnl_model_dir(manifest) if name == "pnl-cnn-weights" else None
            if pnl_dir is not None and not _folder_matches_catalog(
                    pnl_dir, {Path(path).name: digest for path, digest in files.items()}, None):
                result[name] = {"manifest": None, "version": "unknown", "folder": pnl_dir.as_posix(), "files": {}}
                break
            entry = {"manifest": found["path"], "version": found["version"], "files": files,
                     "sha256_source": "asset manifest (stored value, not re-hashed at run time)"}
            if found["upstream_commit"] and name.startswith("surface"):
                entry["upstream_commit"] = found["upstream_commit"]
            result[name] = entry
            break
        else:
            result[name] = {"manifest": None, "version": "unknown", "folder": folder, "files": {}}
    return result


# Child keys that always differ between tracking variants and say nothing about the settings.
PER_CHILD_NOISE_KEYS = frozenset({"run_name", "run_plan_path", "run_plan_summary", "previous_run_records",
                                  "started_at", "input_validation_path", "archived_development_outputs",
                                  "run_manifest_path"})


# Keys that only a finished child manifest holds. A child that crashed has the checkpoint manifest shape without them
# (B3 (1)); the aggregate takes them from the children that hold them.
# C4: brain_mask_configuration is one of them. It holds the custom pnl_mask_model_dir that asset_versions must see.
SETTINGS_ONLY_IN_FINISHED_CHILDREN = ("wma_atlas", "brain_mask_backend", "model_assets_dir", "brain_mask_configuration")


def _read_children(manifest: dict) -> list[tuple[Any, dict, Path]]:
    """(backend, data, path) of every readable child manifest of an aggregate manifest."""
    children: list[tuple[Any, dict, Path]] = []
    for entry in manifest.get("child_runs") or []:
        try:
            path = Path(entry["manifest_path"])
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(data, dict):
            children.append((entry.get("backend"), data, path))
    return children


def child_manifest_records(manifest: dict) -> list[dict]:
    """B3 (3): the child manifests that the record read (backend, path, SHA-256)."""
    return [{"backend": backend, "path": path.as_posix(), "sha256": sha256_file(path)}
            for backend, _, path in _read_children(manifest)]


def merge_child_settings(manifest: dict) -> tuple[dict, list[dict]]:
    """T-C5b: effective settings of an aggregate manifest (several tracking backends).

    Keys that all readable child manifests hold with the same value go to the top level; the aggregate's own
    keys win. Keys that differ go to one entry per child (``tracking_backend`` plus the differing keys).
    ``accepted_licences`` is the union. A manifest without readable children comes back unchanged.
    For the keys in ``SETTINGS_ONLY_IN_FINISHED_CHILDREN`` a child without the key (a crashed child) is skipped.
    """
    children = [(backend, data) for backend, data, _ in _read_children(manifest)]
    if not children:
        return manifest, []
    keys = set().union(*(data.keys() for _, data in children))
    merged: dict[str, Any] = {}
    differing: set[str] = set()
    for key in keys:
        holders = [data for _, data in children if key in data] if key in SETTINGS_ONLY_IN_FINISHED_CHILDREN \
            else [data for _, data in children]
        values = [data.get(key) for data in holders]
        if all(value == values[0] for value in values) and all(key in data for data in holders):
            merged[key] = values[0]
        else:
            differing.add(key)
    merged.update(manifest)
    licences = []
    for source in [manifest] + [data for _, data in children]:
        for item in source.get("accepted_licences") or []:
            if item not in licences:
                licences.append(item)
    merged["accepted_licences"] = licences
    per_child = [dict({"tracking_backend": backend},
                      **{key: data[key] for key in sorted(differing - NON_PARAMETER_KEYS - PER_CHILD_NOISE_KEYS - {"accepted_licences"})
                         if key in data})
                 for backend, data in children]
    return merged, per_child


def _assets(manifest: dict) -> dict:
    assets: dict[str, Any] = {}
    atlas = manifest.get("atlas_t2")
    if atlas:
        path = Path(atlas)
        assets["atlas_t2"] = {"path": path.as_posix(), "sha256": sha256_file(path) if path.is_file() else None}
    if manifest.get("model_assets_dir"):
        assets["model_assets_dir"] = manifest["model_assets_dir"]
    assets["model_profiles"] = asset_versions(manifest)  # T-C7 (D6): asset versions of the run
    return assets


def _mask_provenance(effective: dict) -> dict:
    """F2b: route, status and reason of the mask the run consumed.

    The consumed mask is the one `_resolve_analysis_inputs` returns: the preprocessing final mask when preprocessing
    is complete, else the canonical input mask. For a final mask the preprocessing manifest is passed along.
    """
    from shared.brain_mask import mask_provenance
    from shared.orchestrator import _preprocessing_outputs, _resolve_analysis_inputs

    root, subject_id = effective.get("subject_root"), effective.get("subject_id")
    if not root or not subject_id:
        return {"route": "unknown"}
    output_root = Path(effective["output_root"]) if effective.get("output_root") else None
    try:
        mask = _resolve_analysis_inputs(Path(root), str(subject_id), output_root)["mask"]
        outputs = _preprocessing_outputs(Path(root), str(subject_id), output_root)
        manifest = None
        if mask == outputs["mask"]:
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        return mask_provenance(mask, preprocessing_manifest=manifest)
    except (OSError, ValueError, TypeError):
        return {"route": "unknown"}


def _licence_acceptances() -> list[dict]:
    """Licence acceptances that the setup step stored in the user config ({id, version, time}), sorted by id."""
    try:
        from shared.user_config import load_user_config
        accepted = load_user_config()["accepted_licences"]
    except (OSError, ValueError, KeyError, ImportError):
        return []
    return [dict(accepted[key]) for key in sorted(accepted)]


def build_run_record(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_path = manifest_path.parent / RUN_RECORD_NAME
    effective, per_child = merge_child_settings(manifest)
    parameters = {key: value for key, value in sorted(effective.items()) if key not in NON_PARAMETER_KEYS}
    if per_child:
        parameters["per_tracking_backend"] = per_child
    return {
        "kind": RUN_RECORD_KIND,
        "version": RUN_RECORD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": manifest.get("subject_id"),
        "run_name": manifest.get("run_name"),
        "run_manifest": {"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)},
        # B3 (3): an aggregate manifest names its child manifests; the record lists the ones it read.
        "child_manifests": child_manifest_records(manifest),
        "parameters": parameters,
        # T2 SC-1b / F2 / SC1-CARRY-1: the route of the mask that the run consumed, read from that mask's receipt
        # (a supplied mask records "supplied", one without a usable receipt "unknown"), for every requested backend;
        # `requested_backend` is the selection, not the origin of the mask. The requested route stays in the parameters.
        **({"brain_mask": {"requested_backend": effective.get("brain_mask_backend"), **_mask_provenance(effective)}}
           if effective.get("brain_mask_backend") else {}),
        "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "packages": _package_versions()},
        "assets": _assets(effective),
        # `accepted_licences` is what the manifest holds. B3 (5): the setup step (T2 B7) writes the acceptances (id,
        # licence-file SHA-256, time) into the user config; the record lists them as `licence_acceptances`.
        "accepted_licences": effective.get("accepted_licences", []),
        "licence_acceptances": _licence_acceptances(),
        "qc_summary": _qc_summary(manifest),
        "products": product_inventory(manifest, record_path, _output_root(manifest, manifest_path)),
        "excludes": "run_record.json itself, work folders (" + ", ".join(sorted(EXCLUDED_PATH_PARTS))
                    + ") and intermediates; no output file is copied",
    }


def _previous_record_name(path: Path) -> str:
    """B4: ``run_record-<started_at>.json`` for the record at `path`, from the start time that record holds."""
    import re

    stamp = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        parameters = data.get("parameters") if isinstance(data, dict) else None
        stamp = (parameters or {}).get("started_at") or data.get("created_at")
    except (OSError, ValueError, AttributeError):
        pass
    if not stamp:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    safe = re.sub(r"[^0-9A-Za-z._+-]+", "", str(stamp).replace("+00:00", "Z").replace(":", "")) or "unknown"
    return f"run_record-{safe}.json"


def write_run_record(manifest_path: Path) -> Path:
    """Write ``run_record.json`` beside the run manifest; return its path.

    B4: a record that already exists is kept under ``run_record-<started_at>.json`` (its own start time) before the new
    one is written; nothing is deleted.
    """
    record = build_run_record(manifest_path)
    path = Path(manifest_path).parent / RUN_RECORD_NAME
    if path.is_file():
        archived = path.with_name(_previous_record_name(path))
        number = 2
        while archived.exists():
            archived = archived.with_name(archived.stem.rsplit("~", 1)[0] + f"~{number}.json")
            number += 1
        os.replace(path, archived)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
