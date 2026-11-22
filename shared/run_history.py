"""Pure helpers for discovering and describing SlicerDSpace run manifests.

Every way of starting a run (the Slicer GUI, ``run_subject_pipeline.py`` on
the command line, or ``scripts/run_batch_pipeline.py``) leaves a JSON manifest
on disk. This module finds those manifests under an output root, validates
their ``kind``/``version`` and turns them into small immutable records the
Slicer "Results" section can list across sessions. It never imports Qt or
Slicer so it can be unit tested with the plain Python test runner.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


SUBJECT_MANIFEST_KIND = "slicerddsurfer_product_orchestration"
# The batch runner writes this spelling (single "d"); match it verbatim.
BATCH_MANIFEST_KIND = "slicerdsurfer_batch_run"
SUPPORTED_MANIFEST_VERSIONS: dict[str, frozenset[int]] = {
    SUBJECT_MANIFEST_KIND: frozenset({1}),
    BATCH_MANIFEST_KIND: frozenset({1}),
}
RUN_METADATA_DIR = "00_run_metadata"
BATCH_METADATA_DIR = "00_batch_metadata"
SUBJECT_MANIFEST_SUFFIX = "_orchestration_manifest.json"
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_DIRS = 4000
TERMINAL_TASK_STATES = frozenset(
    {"completed", "passed", "skipped", "reused", "blocked", "failed", "partial_failed", "cancelled"}
)
FAILED_TASK_STATES = frozenset({"failed", "partial_failed"})


class ManifestError(ValueError):
    """Raised when a file is not a loadable SlicerDSpace run manifest.

    The message is written for end users: it says what is wrong and what to
    pick instead.
    """


@dataclass(frozen=True)
class RunRecord:
    """One discovered run, parsed from its manifest."""

    manifest_path: Path
    kind: str  # "subject" or "batch"
    run_name: str
    subject_id: str
    targets: tuple[str, ...]
    status: str
    timestamp: Optional[datetime]
    version: int
    subject_count: int = 0
    output_root: str = ""

    @property
    def is_batch(self) -> bool:
        return self.kind == "batch"

    @property
    def timestamp_text(self) -> str:
        return format_timestamp(self.timestamp)


@dataclass(frozen=True)
class RunDiscovery:
    """Discovery result: valid records (newest first) and skipped files."""

    records: tuple[RunRecord, ...]
    skipped: tuple[tuple[Path, str], ...] = field(default_factory=tuple)
    truncated: bool = False


@dataclass(frozen=True)
class ProgressSnapshot:
    """Per-stage progress for one run, derived from its manifest and plan."""

    done: int
    total: int
    current_stage: str
    status: str
    finished_tasks: tuple[tuple[str, str], ...] = ()

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.done / self.total))

    @property
    def percent(self) -> int:
        return int(round(self.fraction * 100))


def format_timestamp(value: Optional[datetime]) -> str:
    """Return a compact local-time label, or an em dash when unknown."""

    if value is None:
        return "—"
    try:
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return value.strftime("%Y-%m-%d %H:%M")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _batch_filename_timestamp(path: Path) -> Optional[datetime]:
    # scripts/run_batch_pipeline.py names manifests batch_YYYYMMDD_HHMMSS.json (local time).
    stem = path.stem
    if not stem.startswith("batch_"):
        return None
    try:
        return datetime.strptime(stem[len("batch_"):], "%Y%m%d_%H%M%S").astimezone()
    except ValueError:
        return None


def _mtime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def manifest_timestamp(manifest: dict[str, Any], path: Path) -> Optional[datetime]:
    """Pick the most meaningful time for a run.

    Preference: ended_at, cancelled_at, updated_at, the latest task end time,
    started_at, the batch file name, and finally the file modification time.
    """

    return recorded_manifest_timestamp(manifest) or _batch_filename_timestamp(path) or _mtime(path)


def recorded_manifest_timestamp(manifest: dict[str, Any]) -> Optional[datetime]:
    """The time that the manifest itself records (the time fields of :func:`manifest_timestamp`), or None."""

    for key in ("ended_at", "cancelled_at", "updated_at"):
        parsed = _parse_iso(manifest.get(key))
        if parsed is not None:
            return parsed
    task_times = [
        parsed
        for parsed in (
            _parse_iso(task.get("ended_at")) for task in manifest.get("tasks") or () if isinstance(task, dict)
        )
        if parsed is not None
    ]
    if task_times:
        return max(task_times)
    return _parse_iso(manifest.get("started_at"))


def _batch_status(manifest: dict[str, Any]) -> str:
    if manifest.get('experiment') and manifest.get('status') in {'planned', 'running', 'cancelled', 'failed', 'completed'}:
        return manifest['status']
    results = [item for item in manifest.get("results") or () if isinstance(item, dict)]
    if not manifest.get("execute", True) and not results:
        return "planned"
    failed = 0
    for item in results:
        try:
            if int(item.get("returncode", 1)) != 0:
                failed += 1
        except (TypeError, ValueError):
            failed += 1
    if failed:
        return "failed"
    if results:
        return "completed"
    return "no subjects run"


def _subject_status(manifest: dict[str, Any]) -> str:
    status = str(manifest.get("status") or "").strip()
    if status:
        return status
    tasks = [task for task in manifest.get("tasks") or () if isinstance(task, dict)]
    states = {str(task.get("status") or "") for task in tasks}
    if states & FAILED_TASK_STATES:
        return "failed"
    if "blocked" in states:
        return "completed_with_blocks"
    return "completed" if tasks else "unknown"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return ()


def validate_manifest_payload(payload: Any, path: Path) -> tuple[str, int]:
    """Check ``kind`` and ``version``; return them or raise :class:`ManifestError`."""

    if not isinstance(payload, dict):
        raise ManifestError(
            f"{path.name} is not a SlicerDSpace run manifest (expected a JSON object). "
            "Choose a run folder that contains 00_run_metadata/<run>_orchestration_manifest.json."
        )
    kind = payload.get("kind")
    if kind not in SUPPORTED_MANIFEST_VERSIONS:
        raise ManifestError(
            f"{path.name} has kind {kind!r}, which is not a SlicerDSpace run manifest. "
            f"Expected {SUBJECT_MANIFEST_KIND!r} (single subject) or {BATCH_MANIFEST_KIND!r} (batch). "
            "Choose the run's 00_run_metadata folder or its *_orchestration_manifest.json file."
        )
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestError(
            f"{path.name} has no integer 'version' field (found {version!r}); it may be truncated or edited. "
            "Re-run the pipeline to regenerate the manifest."
        )
    supported = SUPPORTED_MANIFEST_VERSIONS[kind]
    if version not in supported:
        supported_text = ", ".join(str(item) for item in sorted(supported))
        raise ManifestError(
            f"{path.name} uses manifest version {version}; this SlicerDSpace build reads version(s) "
            f"{supported_text}. Update the extension to load runs written by a newer release."
        )
    return str(kind), version


def load_manifest_json(path: Path) -> dict[str, Any]:
    """Read a manifest file, converting I/O and JSON errors to :class:`ManifestError`."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestError(
            f"Run manifest not found: {path}. The run folder may have been moved or deleted; "
            "open Results again to rescan, or choose “Add a run folder…” to pick its new location."
        ) from exc
    except OSError as exc:
        raise ManifestError(f"Cannot read run manifest {path}: {exc}. Check file permissions.") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ManifestError(
            f"{Path(path).name} is not valid JSON ({exc}). A run that is still writing its manifest can "
            "show this briefly; otherwise re-run the pipeline to regenerate it."
        ) from exc
    validate_manifest_payload(payload, Path(path))
    return payload


def record_from_manifest(payload: dict[str, Any], path: Path) -> RunRecord:
    """Build a :class:`RunRecord` from an already validated manifest payload."""

    kind, version = validate_manifest_payload(payload, path)
    path = Path(path)
    if kind == BATCH_MANIFEST_KIND:
        subjects = [item for item in payload.get("subjects") or () if isinstance(item, dict)]
        return RunRecord(
            manifest_path=path,
            kind="batch",
            run_name=str(payload.get("run_name") or path.stem),
            subject_id=f"batch ({len(subjects)} subject{'s' if len(subjects) != 1 else ''})",
            targets=_forwarded_targets(payload.get("forwarded_args")),
            status=_batch_status(payload),
            timestamp=manifest_timestamp(payload, path),
            version=version,
            subject_count=len(subjects),
            output_root=str(payload.get("output_root") or ""),
        )
    run_name = str(payload.get("run_name") or path.name.replace(SUBJECT_MANIFEST_SUFFIX, ""))
    return RunRecord(
        manifest_path=path,
        kind="subject",
        run_name=run_name,
        subject_id=str(payload.get("subject_id") or "?"),
        targets=_string_tuple(payload.get("run_targets") or payload.get("targets")),
        status=_subject_status(payload),
        timestamp=manifest_timestamp(payload, path),
        version=version,
        subject_count=1,
        output_root=str(payload.get("output_root") or ""),
    )


def _forwarded_targets(forwarded: Any) -> tuple[str, ...]:
    if not isinstance(forwarded, (list, tuple)):
        return ()
    items = [str(item) for item in forwarded]
    for index, item in enumerate(items):
        if item == "--targets" and index + 1 < len(items):
            return _string_tuple(items[index + 1])
        if item.startswith("--targets="):
            return _string_tuple(item.split("=", 1)[1])
    return ()


def parse_run_manifest(path: Path) -> RunRecord:
    """Read, validate and describe one run manifest file."""

    path = Path(path).expanduser()
    return record_from_manifest(load_manifest_json(path), path)


def _is_subject_manifest_name(name: str) -> bool:
    return name.endswith(SUBJECT_MANIFEST_SUFFIX)


def _manifest_files_in_metadata_dir(directory: Path, *, batch: bool) -> list[Path]:
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return []
    found = []
    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if batch and entry.name.endswith(".json"):
            found.append(Path(entry.path))
        elif not batch and _is_subject_manifest_name(entry.name):
            found.append(Path(entry.path))
    return found


def discover_manifest_paths(
    roots: Iterable[Path | str],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_dirs: int = DEFAULT_MAX_DIRS,
) -> tuple[list[Path], bool]:
    """Find candidate manifest files under ``roots``.

    Breadth-first, at most ``max_depth`` directory levels below each root and
    ``max_dirs`` directories in total. Only direct children of
    ``00_run_metadata`` / ``00_batch_metadata`` are collected, so archived
    ``history/`` copies and per-task ``tasks/`` manifests are never listed.
    A root may also be a manifest file. Returns ``(paths, truncated)``.
    """

    found: list[Path] = []
    seen_files: set[str] = set()
    visited: set[str] = set()
    truncated = False
    visited_count = 0

    def add(path: Path) -> None:
        key = os.path.realpath(path)
        if key not in seen_files:
            seen_files.add(key)
            found.append(path)

    for raw_root in roots:
        if raw_root is None or not str(raw_root).strip():
            continue
        root = Path(str(raw_root)).expanduser()
        if root.is_file():
            add(root)
            continue
        if not root.is_dir():
            continue
        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        while queue:
            directory, depth = queue.popleft()
            real = os.path.realpath(directory)
            if real in visited:
                continue
            visited.add(real)
            visited_count += 1
            if visited_count > max_dirs:
                truncated = True
                break
            if directory.name == RUN_METADATA_DIR:
                for path in _manifest_files_in_metadata_dir(directory, batch=False):
                    add(path)
                continue
            if directory.name == BATCH_METADATA_DIR:
                for path in _manifest_files_in_metadata_dir(directory, batch=True):
                    add(path)
                continue
            if depth >= max_depth:
                continue
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError:
                continue
            priority_dirs: list[Path] = []
            regular_dirs: list[Path] = []
            for entry in entries:
                if entry.name.startswith(".") or entry.name in {"cache", "input", "preprocessed"}:
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if not is_dir:
                    continue
                sub = Path(entry.path)
                if entry.name in {RUN_METADATA_DIR, BATCH_METADATA_DIR} or (
                    depth + 1 < max_depth
                    and (
                        (sub / RUN_METADATA_DIR).is_dir()
                        or (sub / "outputs" / RUN_METADATA_DIR).is_dir()
                        or (sub / BATCH_METADATA_DIR).is_dir()
                    )
                ):
                    priority_dirs.append(sub)
                else:
                    regular_dirs.append(sub)
            for sub in reversed(priority_dirs):
                queue.appendleft((sub, depth + 1))
            for sub in regular_dirs:
                queue.append((sub, depth + 1))
        if truncated:
            break
    return found, truncated


def _sort_key(record: RunRecord) -> tuple[float, str]:
    stamp = record.timestamp.timestamp() if record.timestamp is not None else float("-inf")
    return (stamp, record.manifest_path.as_posix())


def _pick_preferred_manifest_record(
    current: tuple[RunRecord, int, bool],
    candidate: tuple[RunRecord, int, bool],
) -> tuple[RunRecord, int, bool]:
    cur_rec, cur_tasks, cur_cancelled = current
    cand_rec, cand_tasks, cand_cancelled = candidate
    cur_stamp = cur_rec.timestamp.timestamp() if cur_rec.timestamp is not None else float("-inf")
    cand_stamp = cand_rec.timestamp.timestamp() if cand_rec.timestamp is not None else float("-inf")
    if cand_stamp > cur_stamp + 10.0:
        return candidate
    if cur_stamp > cand_stamp + 10.0:
        return current
    cur_rank = (1 if cur_tasks > 0 else 0, cur_tasks, 1 if cur_cancelled else 0, cur_stamp, cur_rec.manifest_path.as_posix())
    cand_rank = (1 if cand_tasks > 0 else 0, cand_tasks, 1 if cand_cancelled else 0, cand_stamp, cand_rec.manifest_path.as_posix())
    return candidate if cand_rank > cur_rank else current


CANCELLED_MANIFEST_SUFFIX = "_cancelled_orchestration_manifest.json"


def _group_key(kind: str, manifest_path: Path) -> tuple[str, ...]:
    """The slot of a run: a re-run, a resume or a cancel companion of the same run share it."""

    path = Path(manifest_path)
    if kind == "subject":
        name = path.name
        if name.endswith(CANCELLED_MANIFEST_SUFFIX):
            base = name[: -len(CANCELLED_MANIFEST_SUFFIX)]
        elif name.endswith(SUBJECT_MANIFEST_SUFFIX):
            base = name[: -len(SUBJECT_MANIFEST_SUFFIX)]
        else:
            base = name
        return ("subject", os.path.realpath(path.parent), base)
    return ("batch", os.path.realpath(path))


def run_manifest_key(manifest_path: Path | str, kind: str = "subject") -> str:
    """The ``discover_runs`` slot of a manifest path as one string (the Results hide list key)."""

    return "\t".join(_group_key(kind, Path(manifest_path)))


def run_record_key(record: RunRecord) -> str:
    """The ``discover_runs`` slot of a record as one string; not the manifest path, which a cancel switches."""

    return run_manifest_key(record.manifest_path, record.kind)


def hidden_stamp(record: RunRecord) -> Optional[float]:
    """The record's own time as stored in the hide list; None when the record has no time."""

    return record.timestamp.timestamp() if record.timestamp is not None else None


def _stamp_value(stamp: Optional[float]) -> float:
    return float("-inf") if stamp is None else float(stamp)


def split_hidden(records: Iterable[RunRecord], hidden: dict[str, Optional[float]]
                 ) -> tuple[tuple[RunRecord, ...], tuple[RunRecord, ...]]:
    """Split ``records`` into (shown, hidden), both in the given order.

    A record is hidden while its slot is in ``hidden`` and its own stamp is not newer than the stored one, so a
    later run, re-run, resume or cancel in the same slot shows again by itself.
    """

    shown: list[RunRecord] = []
    gone: list[RunRecord] = []
    for record in records:
        key = run_record_key(record)
        if key in hidden and _stamp_value(hidden_stamp(record)) <= _stamp_value(hidden[key]):
            gone.append(record)
        else:
            shown.append(record)
    return tuple(shown), tuple(gone)


_OUTPUT_FOLDER_NAMES = frozenset({"output", "outputs"})


def _workspace_of(output_root: Path) -> Path:
    """The folder that holds ``outputs`` (and ``input``); an output root of another name is its own workspace."""

    return output_root.parent if output_root.name.lower() in _OUTPUT_FOLDER_NAMES else output_root


def subject_folder_from_roots(output_root: Any, subject_root: Any) -> Optional[Path]:
    """The subject folder of a job from its roots (Results "Open folder" for a job without a manifest); None if unknown.

    Pure: nothing is read from disk and nothing is resolved, so the same text gives the same path.
    """

    output_text = str(output_root).strip() if output_root is not None else ""
    if output_text:
        return _workspace_of(Path(output_text))
    subject_text = str(subject_root).strip() if subject_root is not None else ""
    return Path(subject_text) if subject_text else None


def run_folder_path(record: RunRecord) -> Path:
    """The folder of a run on disk, for Results "Open folder". Pure: the caller checks that it exists.

    A subject run gives the folder that holds the subject's ``input/`` and ``outputs/``; a batch run gives its batch
    output folder (the stored ``output_root`` when that folder exists, else the parent of ``00_batch_metadata``).
    For a subject run the place of the manifest file wins over the stored ``output_root``, so a run folder that was
    moved still opens; a manifest outside the usual layout falls back to ``output_root``, then to its own folder.
    The only disk access is ``is_dir`` on a batch ``output_root``.
    """

    metadata = Path(record.manifest_path).parent
    if record.is_batch:
        # ``run_batch_pipeline`` writes the manifest to ``<output_base>/00_batch_metadata`` unless --manifest says
        # otherwise, and stores ``output_root`` only when --output-root was given: a stored root that exists wins.
        root = str(record.output_root or "").strip()
        if root and Path(root).is_dir():
            return Path(root)
        return metadata.parent if metadata.name == BATCH_METADATA_DIR else (Path(root) if root else metadata)
    if metadata.name == RUN_METADATA_DIR:
        return _workspace_of(metadata.parent)
    folder = subject_folder_from_roots(record.output_root, None)
    return folder if folder is not None else metadata


HIDDEN_RUNS_LIMIT = 500


def parse_hidden_runs(value: Any) -> dict[str, Optional[float]]:
    """Read the stored hide list (one JSON object string); anything broken reads as empty."""

    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    hidden: dict[str, Optional[float]] = {}
    for key, stamp in payload.items():
        if not isinstance(key, str) or not key:
            continue
        if stamp is None:
            hidden[key] = None
        elif isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            try:
                value = float(stamp)
            except (OverflowError, ValueError):
                continue  # a 400-digit integer: the broken entry reads as absent
            if math.isfinite(value):
                hidden[key] = value
    return hidden


def dump_hidden_runs(hidden: dict[str, Optional[float]], *, limit: int = HIDDEN_RUNS_LIMIT) -> str:
    """The hide list as one JSON string, capped at ``limit`` entries; the oldest stamps are dropped first."""

    items = list(hidden.items())
    if len(items) > limit:
        keep = {key for key, _stamp in sorted(items, key=lambda item: _stamp_value(item[1]), reverse=True)[:limit]}
        items = [item for item in items if item[0] in keep]
    return json.dumps(dict(items))


def discover_runs(
    roots: Iterable[Path | str],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_dirs: int = DEFAULT_MAX_DIRS,
) -> RunDiscovery:
    """Discover and validate runs under ``roots``; newest first."""

    paths, truncated = discover_manifest_paths(roots, max_depth=max_depth, max_dirs=max_dirs)
    grouped: dict[tuple[str, ...], tuple[RunRecord, int, bool]] = {}
    skipped: list[tuple[Path, str]] = []
    for path in paths:
        try:
            payload = load_manifest_json(path)
            record = record_from_manifest(payload, path)
        except ManifestError as exc:
            skipped.append((path, str(exc)))
            continue
        tasks = payload.get("tasks")
        task_count = len(tasks) if isinstance(tasks, dict) else 0
        is_cancelled_companion = record.manifest_path.name.endswith(CANCELLED_MANIFEST_SUFFIX)
        group_key = _group_key(record.kind, record.manifest_path)
        entry = (record, task_count, is_cancelled_companion)
        existing = grouped.get(group_key)
        grouped[group_key] = entry if existing is None else _pick_preferred_manifest_record(existing, entry)
    records = [item[0] for item in grouped.values()]
    records.sort(key=_sort_key, reverse=True)
    return RunDiscovery(records=tuple(records), skipped=tuple(skipped), truncated=truncated)


# Results (batch 2, archived runs): a run that reuses a run name moves the earlier manifest to
# ``<00_run_metadata>/history/<run name>/<UTC stamp>/<run name>_orchestration_manifest.json``
# (``orchestrator._archive_existing_run_records``). Such a manifest is not a run of its own: it adds the files it names
# to the Results tree of its owning record. ``discover_runs`` never lists it.
ARCHIVE_DIR_NAME = "history"
ARCHIVES_PER_RUN_LIMIT = 500  # a safety bound per run folder, not a display cap; logged once when it triggers
_ARCHIVE_LIMIT_LOGGED: set = set()
_ARCHIVE_CACHE: dict = {}  # path text -> (mtime_ns, size, (payload, timestamp) or None)


@dataclass(frozen=True)
class ArchivedManifest:
    """One archived manifest of a run record, with the ``stat`` values taken when it was listed."""

    path: Path
    folder_stamp: str
    mtime_ns: int
    size: int


def _stamp_folder_time(name: str) -> Optional[datetime]:
    try:
        return datetime.strptime(name, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _slot_name(record: RunRecord) -> str:
    """The run name of a record as its manifest file spells it (the slot base of ``_group_key``)."""

    return _group_key(record.kind, record.manifest_path)[2]


def archived_run_manifests(record: RunRecord, folder_records: Iterable[RunRecord] = ()) -> tuple[ArchivedManifest, ...]:
    """The archived manifests that belong to ``record``, newest folder stamp first.

    ``folder_records`` are the other current records (shown and hidden) that may share the record's metadata folder.
    An archive of ``history/<name>/`` belongs to the record of that run name; when no record of the folder has that
    name, it belongs to the newest record of the folder. A broken folder reads as empty.
    """

    if record.is_batch:
        return ()
    metadata = Path(record.manifest_path).parent
    history = metadata / ARCHIVE_DIR_NAME
    folder_key = os.path.realpath(metadata)
    siblings = [item for item in (record, *folder_records)
                if not item.is_batch and os.path.realpath(Path(item.manifest_path).parent) == folder_key]
    names = {_slot_name(item) for item in siblings}
    owns_orphans = os.path.realpath(max(siblings, key=_sort_key).manifest_path) == os.path.realpath(
        record.manifest_path)
    own_name = _slot_name(record)
    try:
        run_dirs = [entry for entry in os.scandir(history) if entry.is_dir()]
    except OSError:
        return ()
    found: list[ArchivedManifest] = []
    for run_dir in run_dirs:
        name = run_dir.name
        if name != own_name and (name in names or not owns_orphans):
            continue
        try:
            stamps = sorted((entry.name for entry in os.scandir(run_dir.path) if entry.is_dir()), reverse=True)
        except OSError:
            continue
        if len(stamps) > ARCHIVES_PER_RUN_LIMIT:
            if run_dir.path not in _ARCHIVE_LIMIT_LOGGED:
                _ARCHIVE_LIMIT_LOGGED.add(run_dir.path)
                logging.getLogger(__name__).warning(
                    "Results: %d archived runs in %s; only the newest %d are read.",
                    len(stamps), run_dir.path, ARCHIVES_PER_RUN_LIMIT)
            stamps = stamps[:ARCHIVES_PER_RUN_LIMIT]
        for stamp in stamps:
            path = Path(run_dir.path) / stamp / f"{name}{SUBJECT_MANIFEST_SUFFIX}"
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append(ArchivedManifest(path, stamp, stat.st_mtime_ns, stat.st_size))
    found.sort(key=lambda entry: entry.folder_stamp, reverse=True)
    return tuple(found)


_ARCHIVE_CACHE_MAX = 512


def read_archived_manifest(entry: ArchivedManifest) -> Optional[tuple[dict[str, Any], Optional[datetime]]]:
    """``(payload, time)`` of an archived subject manifest, or None when it is unreadable or not a subject manifest.

    Parsed once per (path, mtime, size). The time is the time the manifest records (its time fields), else the
    ``<stamp>`` folder name, and the file mtime only when both are missing (review R3: a touched or copied old
    archive must not look new). Read errors are never cached, and the cache is bounded.
    """

    key = str(entry.path)
    cached = _ARCHIVE_CACHE.get(key)
    if cached is not None and cached[0] == entry.mtime_ns and cached[1] == entry.size:
        return cached[2]
    result: Optional[tuple[dict[str, Any], Optional[datetime]]] = None
    try:
        payload = load_manifest_json(entry.path)
        parsed = record_from_manifest(payload, entry.path)
        if parsed.kind == "subject":
            result = (payload, recorded_manifest_timestamp(payload) or _stamp_folder_time(entry.folder_stamp)
                      or parsed.timestamp)
    except (ManifestError, OSError, ValueError, TypeError, AttributeError):
        result = None
    if result is not None:
        if len(_ARCHIVE_CACHE) >= _ARCHIVE_CACHE_MAX and key not in _ARCHIVE_CACHE:
            _ARCHIVE_CACHE.pop(next(iter(_ARCHIVE_CACHE)), None)
        _ARCHIVE_CACHE[key] = (entry.mtime_ns, entry.size, result)
    else:
        _ARCHIVE_CACHE.pop(key, None)
    return result


def clear_archived_manifest_cache() -> None:
    _ARCHIVE_CACHE.clear()


def candidate_run_roots(
    *,
    subject_root_text: str = "",
    output_root_text: str = "",
    batch_root_text: str = "",
    subject_id: str = "",
    batch_mode: bool = False,
    extra_dirs: Iterable[str] = (),
) -> tuple[Path, ...]:
    """Return de-duplicated directories to scan for runs.

    Covers the single-subject subfolder ``<output_root>/<subject_id>`` (when
    applicable) and GUI output directory (or ``<subject root>/outputs``), the
    subject root, the batch root and its default ``batch_outputs`` folder, and
    any folders the user added with “Add a run folder…” in this or an earlier session.
    """

    candidates: list[Path] = []
    output_text = (output_root_text or "").strip()
    subject_text = (subject_root_text or "").strip()
    batch_text = (batch_root_text or "").strip()
    if output_text:
        if not batch_mode and (subject_id or "").strip():
            from shared.ui_run_controller import (
                resolve_single_subject_output_root,
                resolve_single_subject_workspace_root,
            )

            resolved_output = resolve_single_subject_output_root(output_text, subject_id)
            if resolved_output is not None:
                candidates.append(resolved_output)
            resolved_workspace = resolve_single_subject_workspace_root(output_text, subject_id)
            if resolved_workspace is not None:
                candidates.append(resolved_workspace)
        candidates.append(Path(output_text).expanduser())
    if subject_text:
        subject_root = Path(subject_text).expanduser()
        if not output_text:
            candidates.append(subject_root / "outputs")
        candidates.append(subject_root)
    if batch_text:
        batch_root = Path(batch_text).expanduser()
        candidates.append(batch_root / "batch_outputs")
        candidates.append(batch_root)
    for extra in extra_dirs:
        if extra and str(extra).strip():
            candidates.append(Path(str(extra).strip()).expanduser())
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.realpath(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def remember_directory(existing: Iterable[str], directory: str, *, limit: int = 10) -> tuple[str, ...]:
    """Return a most-recent-first list with ``directory`` added, capped at ``limit``."""

    new_value = str(directory or "").strip()
    items = [str(item).strip() for item in existing if str(item or "").strip()]
    if new_value:
        items = [new_value] + [item for item in items if os.path.realpath(item) != os.path.realpath(new_value)]
    return tuple(items[: max(1, int(limit))])


STATUS_LABELS = {
    "completed": "completed",
    "completed_with_blocks": "completed, some blocked",
    "failed": "failed",
    "running": "running or interrupted",
    "cancelled": "cancelled",
    "planned": "planned only",
}


BATCH_HINT = "Batch run: pick one subject's run in the list to load its images."


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status or "unknown")


def run_record_label(record: RunRecord) -> str:
    """One-line combo-box label: time · subject · run · status."""

    return f"{record.timestamp_text} · {record.subject_id} · {record.run_name} · {status_label(record.status)}"


# Product names as the Products section shows them; every name in orchestrator.TARGET_ORDER needs one.
TARGET_LABELS = {
    "validation": "input validation",
    "preprocessing": "preprocessing",
    "segmentation": "DDSeg",
    "parcellation": "DDParcel",
    "surface": "DDSurfer surfaces",
    "surface_postprocess": "surface postprocess",
    "microstructure": "DTI",
    "microstructure_freewater": "free water",
    "microstructure_noddi": "NODDI",
    "microstructure_dki": "DKI",
    "microstructure_ifw": "iFW",
    "tractography": "tractography",
    "bundles": "bundles",
    "saf": "SAF",
    "wma": "WMA",
    "tractcloud": "TractCloud",
    "connectome": "connectome",
}


def run_record_brief(record: RunRecord) -> str:
    """One-line description for the details label; run_record_details stays the tooltip."""

    if record.is_batch:
        return BATCH_HINT
    if not record.targets:
        return "Products: not recorded"
    return "Products: " + ", ".join(TARGET_LABELS.get(target, target) for target in record.targets)


def run_record_details(record: RunRecord) -> str:
    """Multi-line description for the details label and tooltip."""

    targets = ", ".join(record.targets) if record.targets else "not recorded"
    lines = [
        f"Subject: {record.subject_id}    Run: {record.run_name}",
        f"Status: {status_label(record.status)}    Time: {record.timestamp_text}",
        f"Targets: {targets}",
        f"Manifest: {record.manifest_path}",
    ]
    if record.is_batch:
        lines.append(BATCH_HINT)
    return "\n".join(lines)


def _terminal_task_names(tasks: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    finished = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "")
        if status in TERMINAL_TASK_STATES:
            finished.append((str(task.get("name") or ""), status))
    return tuple(finished)


def progress_from_payloads(manifest: dict[str, Any], plan: Optional[dict[str, Any]]) -> ProgressSnapshot:
    """Compute stage progress from an orchestration checkpoint and its run plan.

    Denominator: plan items that will run (all items if none are flagged).
    Numerator: those plan items whose ``task_name`` (or target prefix) has a
    finished task in the manifest. Sub-stage task names that are not in the
    plan are tolerated and do not inflate the count.
    """

    finished = _terminal_task_names(manifest.get("tasks") or ())
    finished_names = {name for name, _status in finished}
    finished_prefixes = {name.split(".", 1)[0] for name in finished_names if "@" not in name}
    finished_variant_prefixes = {
        f"{name.split('.', 1)[0]}@{name.rsplit('@', 1)[1]}"
        for name in finished_names
        if "@" in name
    }
    items = [item for item in (plan or {}).get("items") or () if isinstance(item, dict)]
    runnable = [item for item in items if item.get("will_run")] or items
    total = len(runnable)
    done = 0
    pending: list[str] = []
    for item in runnable:
        task_name = str(item.get("task_name") or "")
        target = str(item.get("target") or "")
        if "@" in task_name:
            variant = task_name.rsplit("@", 1)[1]
            matched = task_name in finished_names or bool(
                target and f"{target}@{variant}" in finished_variant_prefixes
            )
        else:
            matched = task_name in finished_names or bool(target and target in finished_prefixes)
        if matched:
            done += 1
        else:
            pending.append(task_name or target)
    if not items:
        total = len(finished)
        done = len(finished)
    done = min(done, total)
    status = str(manifest.get("status") or "")
    current = pending[0] if pending and status == "running" else ""
    return ProgressSnapshot(done=done, total=total, current_stage=current, status=status, finished_tasks=finished)


def progress_from_manifest(manifest_path: Path, *, not_before: Optional[float] = None) -> Optional[ProgressSnapshot]:
    """Read a checkpoint manifest (and its run plan) from disk.

    ``not_before`` is a POSIX time: manifests last modified earlier belong to
    a previous run with the same name and are ignored (returns ``None``).
    Returns ``None`` while the manifest is missing, stale or half-written.
    """

    path = Path(manifest_path)
    try:
        if not_before is not None and path.stat().st_mtime + 1.0 < not_before:
            return None
        manifest = load_manifest_json(path)
    except (OSError, ManifestError):
        return None
    if manifest.get("kind") != SUBJECT_MANIFEST_KIND:
        return None
    child_runs = manifest.get("child_runs")
    if isinstance(child_runs, list) and child_runs and str(manifest.get("status") or "") == "running":
        from shared.multi_tracking import SHARED

        merged_tasks: list[dict[str, Any]] = [
            dict(t) for t in (manifest.get("tasks") or ()) if isinstance(t, dict)
        ]
        seen_names = {str(t.get("name") or "") for t in merged_tasks}
        for entry in child_runs:
            if not isinstance(entry, dict) or not entry.get("manifest_path"):
                continue
            child_path = Path(str(entry["manifest_path"])).expanduser()
            backend = str(entry.get("backend") or "")
            try:
                child_manifest = load_manifest_json(child_path)
            except (OSError, ManifestError):
                continue
            for task in child_manifest.get("tasks") or ():
                if not isinstance(task, dict):
                    continue
                tname = str(task.get("name") or "")
                is_shared = tname.split(".", 1)[0] in SHARED or tname.startswith(("microstructure.", "surface."))
                qual_name = tname if (is_shared or not backend or "@" in tname) else f"{tname}@{backend}"
                if qual_name not in seen_names:
                    seen_names.add(qual_name)
                    merged_tasks.append(dict(task, name=qual_name))
        if merged_tasks:
            manifest = dict(manifest, tasks=merged_tasks)
    plan: Optional[dict[str, Any]] = None
    plan_path = manifest.get("run_plan_path")
    if isinstance(plan_path, str) and plan_path:
        try:
            loaded = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            plan = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            plan = None
    return progress_from_payloads(manifest, plan)


def manifest_path_for_plan(run_plan_path: Path, run_name: str) -> Path:
    """Where the orchestrator writes the manifest for a plan (same metadata folder)."""

    return Path(run_plan_path).parent / f"{run_name}{SUBJECT_MANIFEST_SUFFIX}"
