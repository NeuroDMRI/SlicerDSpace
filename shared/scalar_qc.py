"""Dependency-free acceptance gate for previously decoded scalar products."""

import json
from pathlib import Path
from typing import Optional

from shared.cache_reuse import (
    artifact_map_matches,
    file_record_matches,
    file_signature,
    resolve_relocated_file,
)


def _signature(path):
    return file_signature(path, include_sha256=True)


def _source_search_dirs(qc_path: Path) -> tuple[Path, ...]:
    """Candidate directories for resolving relocated source files next to a QC JSON."""
    parent = Path(qc_path).resolve().parent
    dirs = [parent]
    for ancestor in parent.parents[:5]:
        dirs.append(ancestor)
        for sub in ("input", "preprocessed"):
            candidate = ancestor / sub
            if candidate.is_dir():
                dirs.append(candidate)
        if (ancestor / "input").is_dir():
            break
    return tuple(dict.fromkeys(dirs))


def _infer_subject_root(qc_path: Path) -> Optional[Path]:
    parent = Path(qc_path).resolve().parent
    for ancestor in (parent, *parent.parents[:6]):
        if (ancestor / "input").is_dir() or (ancestor / "cache").is_dir():
            return ancestor
    return None


def scalar_qc_current(
    qc_path: Path,
    maps: list[Path],
    *,
    source_paths: Optional[dict[str, Path]] = None,
) -> bool:
    """Changed files or old/failed records require fresh numerical validation."""
    try:
        qc_file = Path(qc_path)
        record = json.loads(qc_file.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        checks = record.get("checks")
        if (record.get("status") != "completed" or record.get("automatic_qc_passed") is not True
                or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values())):
            return False
        subject_root = _infer_subject_root(qc_file)
        search_dirs = _source_search_dirs(qc_file)
        for key, source in (record.get("source_signature") or {}).items():
            if not isinstance(source, dict):
                return False
            explicit = source_paths.get(key) if isinstance(source_paths, dict) else None
            target = Path(explicit) if explicit is not None else resolve_relocated_file(
                source.get("path"),
                search_dirs,
                subject_root=subject_root,
            )
            if target is None or not file_record_matches(source, target, allow_relocated=True):
                return False
        return artifact_map_matches(
            record.get("output_artifacts"),
            maps,
            root=qc_file.resolve().parent,
            allow_relocated=True,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
