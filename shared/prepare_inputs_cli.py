#!/usr/bin/env python3
"""Out-of-process CLI worker for input preparation (NIfTI / NRRD / DICOM)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.input_import import InputImportConfig, detect_import_source_kind, prepare_inputs


_OPTIONAL_PATH_FIELDS = (
    "bval_path",
    "bvec_path",
    "metadata_path",
    "mask_path",
    "dcm2niix_executable",
    "mrconvert_executable",
    "dwiconvert_executable",
)


def _optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def config_from_dict(payload: Mapping[str, Any]) -> InputImportConfig:
    """Deserialize an :class:`InputImportConfig` from a JSON-compatible mapping."""

    source_raw = str(payload.get("source") or "").strip()
    subject_root_raw = str(payload.get("subject_root") or "").strip()
    subject_id = str(payload.get("subject_id") or "").strip()
    if not source_raw or not subject_root_raw or not subject_id:
        raise ValueError("Input import request requires source, subject_root, and subject_id.")

    source = Path(source_raw).expanduser().resolve()
    subject_root = Path(subject_root_raw).expanduser()
    raw_kind = str(payload.get("source_kind") or "canonical").strip() or "canonical"
    source_kind = detect_import_source_kind(source) if raw_kind == "auto" else raw_kind
    optional_paths = {field: _optional_path(payload.get(field)) for field in _OPTIONAL_PATH_FIELDS}

    return InputImportConfig(
        source=source,
        subject_root=subject_root,
        subject_id=subject_id,
        source_kind=source_kind,  # type: ignore[arg-type]
        backend=str(payload.get("backend") or "auto").strip() or "auto",  # type: ignore[arg-type]
        output_kind=str(payload.get("output_kind") or "canonical-nifti").strip() or "canonical-nifti",  # type: ignore[arg-type]
        dry_run=bool(payload.get("dry_run", False)),
        force=bool(payload.get("force", False)),
        **optional_paths,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _choose_subject(
    config: InputImportConfig, request: Mapping[str, Any], mode: str, say: Callable[[str], None]
) -> tuple[InputImportConfig, Any]:
    """D3: choose the subject folder from the input identity inside the import job (thread or child process).

    ``request["subject_choice"]``: ``base_id``, ``output_dir`` and ``fallback_parent`` (the GUI root rule).
    Without it, or for a mode that imports nothing, the config is used as given.
    """
    choice_request = request.get("subject_choice")
    if not isinstance(choice_request, Mapping) or mode not in {"prepare", "detached_nrrd", "dicom"}:
        return config, None
    from dataclasses import replace

    from shared.input_import import choose_subject_for_source, import_source_signatures, subject_root_rule

    raw_path = None
    series = None
    if mode == "detached_nrrd":
        raw_val = str(request.get("raw_path") or request.get("raw") or "").strip()
        if not raw_val:
            raise ValueError("detached_nrrd mode requires 'raw_path'.")
        raw_path = Path(raw_val).expanduser().resolve()
    elif mode == "dicom":
        series = _series_from_request(request)
    signatures = import_source_signatures(config, raw_path=raw_path, dicom_series=series)
    root_for_id = subject_root_rule(str(choice_request.get("output_dir") or ""),
                                    str(choice_request.get("fallback_parent") or ""))
    choice = choose_subject_for_source(str(choice_request.get("base_id") or ""), root_for_id, signatures,
                                       source=config.source, source_kind=config.source_kind)
    say(f"[Prepare Inputs] Subject {choice.subject_id} ({choice.decision}): {choice.subject_root}")
    return replace(config, subject_root=choice.subject_root, subject_id=choice.subject_id,
                   source_signatures=signatures), choice


def _series_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    series_raw = request.get("series_entry") or request.get("series")
    if not isinstance(series_raw, Mapping):
        raise ValueError("dicom mode requires 'series_entry'.")
    series = dict(series_raw)
    series["paths"] = [Path(str(item)) for item in (series.get("paths") or ())]
    return series


def execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one input-import request dict and return its manifest payload.

    With ``subject_choice`` the job chooses the subject folder first (D3) and the result carries the chosen
    target under ``subject_choice``; that key is added fresh to every result and never saved in a manifest.
    """

    cfg_data = request.get("config") if isinstance(request.get("config"), Mapping) else request
    return execute_import(config_from_dict(cfg_data), request)


def _print_line(text: str) -> None:
    print(text, flush=True)


def execute_import(
    config: InputImportConfig,
    request: Mapping[str, Any],
    *,
    log: Callable[[str], None] | None = _print_line,
) -> dict[str, Any]:
    """:func:`execute_request` with a ready config: the GUI thread path calls it, so both job paths share the
    subject choice (D3) and the mode dispatch. ``log=None`` prints nothing (a GUI worker thread)."""

    say = log or (lambda _text: None)
    mode = str(request.get("mode") or "prepare").strip().lower()
    config, choice = _choose_subject(config, request, mode, say)
    result = _execute_mode(config, request, mode, say)
    if choice is not None:
        result = {**result, "subject_choice": choice.as_payload()}
    return result


def _execute_mode(
    config: InputImportConfig, request: Mapping[str, Any], mode: str, say: Callable[[str], None]
) -> dict[str, Any]:
    if mode == "prepare":
        say(f"[Prepare Inputs] Preparing {config.source_kind} inputs from {config.source}...")
        return prepare_inputs(config)

    if mode == "detached_nrrd":
        raw_val = request.get("raw_path") or request.get("raw") or ""
        if not str(raw_val).strip():
            raise ValueError("detached_nrrd mode requires 'raw_path'.")
        raw_path = Path(str(raw_val)).expanduser().resolve()
        from shared.ui_prepare_inputs import import_detached_nrrd

        say(f"[Prepare Inputs] Importing detached NRRD from {config.source}...")
        return import_detached_nrrd(config, raw_path)

    if mode == "dicom":
        series = _series_from_request(request)
        from shared.ui_prepare_inputs import import_dicom_series

        say(f"[Prepare Inputs] Importing DICOM series from {config.source}...")
        return import_dicom_series(config, series)

    if mode == "scan_dicom":
        from shared.dicom_input import scan_series

        say(f"[Prepare Inputs] Scanning DICOM series in {config.source}...")
        series_list = scan_series(config.source)
        serialized = [
            {
                **dict(entry),
                "paths": [Path(p).as_posix() for p in (entry.get("paths") or ())],
            }
            for entry in series_list
        ]
        return {"status": "completed", "series": serialized}

    raise ValueError(f"Unsupported prepare_inputs_cli mode: {mode!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON request path.")
    parser.add_argument("--result", type=Path, required=True, help="Atomic JSON result path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.expanduser().read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise ValueError("Prepare inputs request must be a JSON object.")
        result = execute_request(request)
    except Exception as exc:
        _write_json_atomic(args.result.expanduser(), {"status": "failed", "error": str(exc)})
        print(f"[Prepare Inputs] Failed: {exc}", file=sys.stderr, flush=True)
        return 1
    _write_json_atomic(args.result.expanduser(), result)
    print("[Prepare Inputs] Completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
