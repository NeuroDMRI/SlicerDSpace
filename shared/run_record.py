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
    if path.name == RUN_RECORD_NAME or path == run_record_path:
        return False
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


def asset_versions(manifest: dict, repo_root: Optional[Path] = None) -> dict:
    """Version and SHA-256 values that the asset manifests already store, per profile that the run used.

    Nothing is hashed here (the weights are large). A profile that no manifest lists gets the model
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

    result: dict[str, Any] = {}
    for name in used_asset_profiles(manifest):
        for found in manifests:
            if name not in found["profiles"]:
                continue
            files = {path: (found["records"].get(path) or {}).get("sha256")
                     for path in dict.fromkeys(paths_of(name, found, set()))}
            entry = {"manifest": found["path"], "version": found["version"], "files": files,
                     "sha256_source": "asset manifest (stored value, not re-hashed at run time)"}
            if found["upstream_commit"] and name.startswith("surface"):
                entry["upstream_commit"] = found["upstream_commit"]
            result[name] = entry
            break
        else:
            result[name] = {"manifest": None, "version": "unknown", "folder": folder, "files": {}}
    return result


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


def build_run_record(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_path = manifest_path.parent / RUN_RECORD_NAME
    return {
        "kind": RUN_RECORD_KIND,
        "version": RUN_RECORD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": manifest.get("subject_id"),
        "run_name": manifest.get("run_name"),
        "run_manifest": {"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)},
        "parameters": {key: value for key, value in sorted(manifest.items()) if key not in NON_PARAMETER_KEYS},
        "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "packages": _package_versions()},
        "assets": _assets(manifest),
        # The licence acceptance record belongs to the setup step of T2 (B7); this run records what the manifest holds.
        "accepted_licences": manifest.get("accepted_licences", []),
        "qc_summary": _qc_summary(manifest),
        "products": product_inventory(manifest, record_path, _output_root(manifest, manifest_path)),
        "excludes": "run_record.json itself, work folders (" + ", ".join(sorted(EXCLUDED_PATH_PARTS))
                    + ") and intermediates; no output file is copied",
    }


def write_run_record(manifest_path: Path) -> Path:
    """Write ``run_record.json`` beside the run manifest; return its path."""
    record = build_run_record(manifest_path)
    path = Path(manifest_path).parent / RUN_RECORD_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
