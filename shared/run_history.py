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
    parsed = _parse_iso(manifest.get("started_at"))
    if parsed is not None:
        return parsed
    return _batch_filename_timestamp(path) or _mtime(path)


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
            "open Results and QC again to rescan, or choose “Add a run folder…” to pick its new location."
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
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if is_dir:
                    queue.append((Path(entry.path), depth + 1))
        if truncated:
            break
    return found, truncated


def _sort_key(record: RunRecord) -> tuple[float, str]:
    stamp = record.timestamp.timestamp() if record.timestamp is not None else float("-inf")
    return (stamp, record.manifest_path.as_posix())


def discover_runs(
    roots: Iterable[Path | str],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_dirs: int = DEFAULT_MAX_DIRS,
) -> RunDiscovery:
    """Discover and validate runs under ``roots``; newest first."""

    paths, truncated = discover_manifest_paths(roots, max_depth=max_depth, max_dirs=max_dirs)
    records: list[RunRecord] = []
    skipped: list[tuple[Path, str]] = []
    for path in paths:
        try:
            records.append(parse_run_manifest(path))
        except ManifestError as exc:
            skipped.append((path, str(exc)))
    records.sort(key=_sort_key, reverse=True)
    return RunDiscovery(records=tuple(records), skipped=tuple(skipped), truncated=truncated)


def candidate_run_roots(
    *,
    subject_root_text: str = "",
    output_root_text: str = "",
    batch_root_text: str = "",
    extra_dirs: Iterable[str] = (),
) -> tuple[Path, ...]:
    """Return de-duplicated directories to scan for runs.

    Covers the GUI's output directory (or ``<subject root>/outputs``), the
    subject root, the batch root and its default ``batch_outputs`` folder, and
    any folders the user added with “Add a run folder…” in this or an earlier session.
    """

    candidates: list[Path] = []
    output_text = (output_root_text or "").strip()
    subject_text = (subject_root_text or "").strip()
    batch_text = (batch_root_text or "").strip()
    if output_text:
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
    finished_prefixes = {name.split(".", 1)[0] for name in finished_names}
    items = [item for item in (plan or {}).get("items") or () if isinstance(item, dict)]
    runnable = [item for item in items if item.get("will_run")] or items
    total = len(runnable)
    done = 0
    pending: list[str] = []
    for item in runnable:
        task_name = str(item.get("task_name") or "")
        target = str(item.get("target") or "")
        if task_name in finished_names or (target and target in finished_prefixes):
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
