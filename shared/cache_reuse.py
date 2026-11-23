"""Shared file-signature and cache-reuse comparison helpers.

Keep scientific invalidation strict (input/output content, scientific
parameters, explicit gradient-frame and integrity checks) while tolerating
harmless non-scientific churn (mtime drift when size + sha256 match, subject
directory relocations, runtime install-path moves, and `.py` CodeOverlay edits).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Optional

# Files up to 32 MiB are always hashed on reuse checks so even same-size,
# mtime-preserving tampering is caught immediately. Larger files (e.g. 500 MB
# 4D DWIs) take the fast path when resolved path, size, and mtime_ns (plus
# ctime_ns when recorded) all match, and fall back to full SHA-256 verification
# whenever path or timestamps changed (such as after folder moves or copies).
ALWAYS_HASH_MAX_BYTES = 32 * 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA-256 of a regular file via bounded 1 MiB reads."""
    target = Path(path)
    with target.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def file_signature(path: Path | str, *, include_sha256: bool = True) -> dict[str, Any]:
    """Build a portable file signature including size, mtime_ns, and optional sha256."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256 and resolved.is_file():
        record["sha256"] = sha256_file(resolved)
    return record


def paths_equal(first: object, second: object) -> bool:
    """Compare two filesystem paths after resolution without crashing on bad strings."""
    try:
        a = str(first or "").strip()
        b = str(second or "").strip()
        if not a or not b:
            return False
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return False


def names_equal(first: object, second: object) -> bool:
    """Compare basenames using platform-appropriate case folding."""
    a = Path(str(first or "")).name
    b = Path(str(second or "")).name
    if not a or not b:
        return False
    return os.path.normcase(a) == os.path.normcase(b)


def file_record_matches(
    record: object,
    path: Path | str,
    *,
    allow_relocated: bool = True,
    require_same_name: bool = True,
    always_hash_max_bytes: int = ALWAYS_HASH_MAX_BYTES,
) -> bool:
    """Verify that ``path`` matches a recorded file metadata dict.

    - When ``sha256`` is recorded: verifies ``size`` and ``sha256`` (always
      for files <= ``always_hash_max_bytes``, and always whenever ``path`` or
      ``mtime_ns``/``ctime_ns`` changed). Ignores ``mtime_ns``/``ctime_ns``
      drift and directory relocation when content matches.
    - When ``sha256`` is absent (legacy record): requires ``size`` and
      ``mtime_ns`` to match, and requires either the same resolved path or
      (when ``allow_relocated`` is True) the same basename.
    """
    if not isinstance(record, dict):
        return False
    target = Path(path)
    if not target.is_file():
        return False
    try:
        stat = target.stat()
    except OSError:
        return False

    recorded_size = record.get("size", record.get("size_bytes"))
    if recorded_size is not None:
        try:
            if int(recorded_size) != stat.st_size:
                return False
        except (TypeError, ValueError):
            return False

    recorded_path = str(record.get("path") or "").strip()
    same_path = bool(recorded_path) and paths_equal(recorded_path, target)
    if recorded_path and not same_path:
        if not allow_relocated:
            return False
        if require_same_name and not names_equal(recorded_path, target):
            return False

    recorded_sha = record.get("sha256")
    if isinstance(recorded_sha, str) and recorded_sha:
        if stat.st_size <= always_hash_max_bytes or not same_path:
            try:
                return sha256_file(target) == recorded_sha
            except OSError:
                return False
        try:
            mtime_same = (
                "mtime_ns" in record and int(record["mtime_ns"]) == stat.st_mtime_ns
            )
            ctime_same = (
                "ctime_ns" not in record or int(record["ctime_ns"]) == stat.st_ctime_ns
            )
        except (TypeError, ValueError):
            mtime_same = False
            ctime_same = False
        if mtime_same and ctime_same:
            return True
        try:
            return sha256_file(target) == recorded_sha
        except OSError:
            return False

    # Legacy record without sha256: require size + mtime_ns so same-size edits
    # at the same path cannot slip past cache validation.
    if recorded_size is None:
        return False
    try:
        return int(record.get("mtime_ns", -1)) == stat.st_mtime_ns
    except (TypeError, ValueError):
        return False


def file_records_equivalent(
    recorded: object,
    expected: object,
    *,
    allow_relocated: bool = True,
    require_same_name: bool = True,
) -> bool:
    """Compare two recorded file-signature dicts (e.g. from two manifests).

    When ``expected`` points to an existing file on disk, verifies ``recorded``
    against that file via :func:`file_record_matches` (so a legacy ``recorded``
    entry without ``sha256`` still matches a newly computed ``expected`` entry
    with ``sha256`` when size and mtime match, and a ``sha256``-bearing
    ``recorded`` entry ignores mtime/path churn).
    """
    if recorded is None and expected is None:
        return True
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        return False
    exp_path_str = str(expected.get("path") or "").strip()
    if exp_path_str:
        exp_path = Path(exp_path_str)
        if exp_path.is_file():
            return file_record_matches(
                recorded,
                exp_path,
                allow_relocated=allow_relocated,
                require_same_name=require_same_name,
            )
    # Fallback for synthetic/mocked dicts where the path is absent or not on disk.
    rec_sha = recorded.get("sha256")
    exp_sha = expected.get("sha256")
    if isinstance(rec_sha, str) and rec_sha and isinstance(exp_sha, str) and exp_sha:
        if rec_sha != exp_sha:
            return False
        rec_size = recorded.get("size")
        exp_size = expected.get("size")
        if rec_size is not None and exp_size is not None and rec_size != exp_size:
            return False
        if require_same_name and recorded.get("path") and expected.get("path"):
            if not names_equal(recorded["path"], expected["path"]):
                return False
        return True
    return recorded == expected


def _rel_tail_key(path_str: str, depth: int = 2) -> tuple[str, ...]:
    parts = Path(path_str).parts
    tail = parts[-depth:] if len(parts) >= depth else parts
    return tuple(os.path.normcase(part) for part in tail)


def artifact_map_matches(
    recorded: object,
    expected_paths: Iterable[Path],
    *,
    root: Optional[Path] = None,
    allow_relocated: bool = True,
) -> bool:
    """Match an ``{abs_path: file_signature}`` mapping against ``expected_paths``.

    Requires an exact one-to-one match between every expected path and a
    recorded artifact entry. Exact resolved keys match first; when relocated,
    matches by relative path under ``root``, then by 2-segment trailing path
    ``(parent.name, name)``, then by unique basename.
    """
    if not isinstance(recorded, dict):
        return False
    expected = [Path(p).resolve() for p in expected_paths]
    if len(recorded) != len(expected):
        return False

    unused_keys = set(recorded.keys())
    resolved_root = Path(root).resolve() if root is not None else None

    for path in expected:
        if not path.is_file():
            return False
        exact_key = str(path)
        matched_key: Optional[str] = None
        if exact_key in unused_keys:
            matched_key = exact_key
        else:
            for key in unused_keys:
                if paths_equal(key, path):
                    matched_key = key
                    break
        if matched_key is None and allow_relocated:
            candidates: list[str] = []
            if resolved_root is not None:
                try:
                    rel_parts = tuple(
                        os.path.normcase(p) for p in path.relative_to(resolved_root).parts
                    )
                except ValueError:
                    rel_parts = ()
                if rel_parts:
                    candidates = [
                        key
                        for key in unused_keys
                        if _rel_tail_key(key, len(rel_parts)) == rel_parts
                    ]
            if len(candidates) != 1:
                tail2 = _rel_tail_key(str(path), 2)
                candidates = [
                    key for key in unused_keys if _rel_tail_key(key, 2) == tail2
                ]
            if len(candidates) != 1:
                candidates = [
                    key for key in unused_keys if names_equal(key, path)
                ]
            if len(candidates) == 1:
                matched_key = candidates[0]
        if matched_key is None:
            return False
        if not file_record_matches(
            recorded[matched_key],
            path,
            allow_relocated=allow_relocated,
            require_same_name=True,
        ):
            return False
        unused_keys.remove(matched_key)

    return not unused_keys


def resolve_relocated_file(
    recorded_path: object,
    candidate_dirs: Iterable[Path] = (),
    *,
    subject_root: Optional[Path] = None,
    anchor_markers: tuple[str, ...] = (
        "input",
        "preprocessed",
        "outputs",
        "cache",
        "pred_results_SlicerDSpace",
    ),
) -> Optional[Path]:
    """Resolve a previously recorded file path if it still exists or was relocated."""
    raw = str(recorded_path or "").strip()
    if not raw:
        return None
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()

    parts = direct.parts
    if subject_root is not None:
        base = Path(subject_root).expanduser().resolve()
        for marker in anchor_markers:
            for idx, part in enumerate(parts):
                if os.path.normcase(part) == os.path.normcase(marker):
                    candidate = base.joinpath(*parts[idx:])
                    if candidate.is_file():
                        return candidate.resolve()

    name = direct.name
    if not name:
        return None
    for directory in candidate_dirs:
        if directory is None:
            continue
        folder = Path(directory)
        direct_candidate = folder / name
        if direct_candidate.is_file():
            return direct_candidate.resolve()
        if len(parts) >= 2:
            nested_candidate = folder / parts[-2] / name
            if nested_candidate.is_file():
                return nested_candidate.resolve()
    return None
