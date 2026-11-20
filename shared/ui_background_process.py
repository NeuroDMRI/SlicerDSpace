"""Small background-process helpers for the SlicerDSpace widget."""

from __future__ import annotations

import codecs
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def background_process_env_overrides() -> dict[str, str]:
    """Return environment overrides for the pipeline CLI's background QProcess.

    ``PYTHONUNBUFFERED=1`` keeps the child CLI's stdout unbuffered even
    though QProcess redirects it through a pipe (Python otherwise switches
    to full block buffering for a non-tty stdout), so the widget's status
    table keeps updating live instead of only after large buffered chunks
    or process exit.

    ``PYTHONUTF8=1`` and ``PYTHONIOENCODING=utf-8`` (X1) make the worker write UTF-8 to stdout and
    stderr. Without them, Slicer 5.10 on Windows uses the ANSI code page (cp1252) with strict errors,
    so a print of a non-ASCII path raises ``UnicodeEncodeError`` and ends the worker. The widget
    decodes the output as UTF-8 (see :class:`Utf8StreamDecoder`).
    """

    return {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _qbytearray_to_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    try:
        return bytes(value)
    except Exception:
        pass
    try:
        raw = value.data()
    except Exception:
        return b""
    if isinstance(raw, str):
        return raw.encode("utf-8", errors="replace")
    return bytes(raw) if isinstance(raw, (bytes, bytearray)) else b""


class Utf8StreamDecoder:
    """Decode worker output chunk by chunk as UTF-8 without corrupting a character split across reads (X1)."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, value: Any) -> str:
        return self._decoder.decode(_qbytearray_to_bytes(value))

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)


def qprocess_is_running(process: Any) -> bool:
    """Return True if a Qt process appears to be running."""

    if process is None:
        return False
    try:
        import qt

        return process.state() != qt.QProcess.NotRunning
    except Exception:
        return False


def build_background_run_context(
    common_run_kwargs: dict[str, Any],
    selected_workflows: list[str],
    *,
    batch_root: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build the context stored while a background run is active."""

    is_batch = batch_root is not None
    context = {
        "subject_id": "batch" if is_batch else common_run_kwargs["subject_id"],
        "subject_root": batch_root if is_batch else common_run_kwargs["subject_root"],
        "output_root": common_run_kwargs.get("output_root"),
        "run_name": common_run_kwargs["run_name"],
        "targets": common_run_kwargs["targets"],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "selected_workflows": selected_workflows,
    }
    if batch_size is not None:
        context["batch_size"] = batch_size
    return context
