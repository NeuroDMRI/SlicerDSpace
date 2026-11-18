#!/usr/bin/env python3
"""Out-of-process CLI worker for input preparation (NIfTI / NRRD / DICOM)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
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


def execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one input-import request dict and return its manifest payload."""

    mode = str(request.get("mode") or "prepare").strip().lower()
    cfg_data = request.get("config") if isinstance(request.get("config"), Mapping) else request
    config = config_from_dict(cfg_data)

    if mode == "prepare":
        print(f"[Prepare Inputs] Preparing {config.source_kind} inputs from {config.source}...", flush=True)
        return prepare_inputs(config)

    if mode == "detached_nrrd":
        raw_val = request.get("raw_path") or request.get("raw") or ""
        if not str(raw_val).strip():
            raise ValueError("detached_nrrd mode requires 'raw_path'.")
        raw_path = Path(str(raw_val)).expanduser().resolve()
        from shared.ui_prepare_inputs import import_detached_nrrd

        print(f"[Prepare Inputs] Importing detached NRRD from {config.source}...", flush=True)
        return import_detached_nrrd(config, raw_path)

    if mode == "dicom":
        series_raw = request.get("series_entry") or request.get("series")
        if not isinstance(series_raw, Mapping):
            raise ValueError("dicom mode requires 'series_entry'.")
        series = dict(series_raw)
        series["paths"] = [Path(str(item)) for item in (series.get("paths") or ())]
        from shared.ui_prepare_inputs import import_dicom_series

        print(f"[Prepare Inputs] Importing DICOM series from {config.source}...", flush=True)
        return import_dicom_series(config, series)

    if mode == "scan_dicom":
        from shared.dicom_input import scan_series

        print(f"[Prepare Inputs] Scanning DICOM series in {config.source}...", flush=True)
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
