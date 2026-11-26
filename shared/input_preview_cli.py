#!/usr/bin/env python3
"""Out-of-process worker for View input preview so Slicer's PythonQt GIL never starves DWI reads."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_str = str(REPO_ROOT)
while _repo_str in sys.path:
    sys.path.remove(_repo_str)
sys.path.insert(0, _repo_str)

# Activate bundled-python (nibabel, pynrrd, etc.) before reading the DWI.
try:
    from shared.bundled_python import activate_bundled_python

    activate_bundled_python()
except Exception:
    pass

from shared.input_preview import (
    B0_THRESHOLD,
    PreviewError,
    PreviewSource,
    estimate_full_dwi_bytes,
    full_dwi_limit_bytes,
    gradient_table_unusable,
    mean_b0_image,
    read_full_dwi_to_shared_memory,
)

READY_SENTINEL = "SLICERDSPACE_INPUT_PREVIEW_READY"
_DEFAULT_HANDOFF_TIMEOUT_SECONDS = 60.0


def _optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def source_from_dict(payload: Mapping[str, Any]) -> PreviewSource:
    return PreviewSource(
        kind=str(payload.get("kind") or "nifti"),
        dwi=Path(str(payload["dwi"])).expanduser(),
        bval=_optional_path(payload.get("bval")),
        mask=_optional_path(payload.get("mask")),
        subject_id=str(payload.get("subject_id") or ""),
        origin=str(payload.get("origin") or ""),
        raw=_optional_path(payload.get("raw")),
        bvec=_optional_path(payload.get("bvec")),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _wait_for_ack(timeout_seconds: float) -> None:
    """Wait until the GUI writes a newline (or closes stdin) after copying the SharedMemory block."""
    done = threading.Event()

    def _reader() -> None:
        try:
            sys.stdin.readline()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_reader, daemon=True).start()
    done.wait(timeout=max(1.0, float(timeout_seconds)))


def run_request(request: Mapping[str, Any], result_path: Path, *, handoff_timeout: float = _DEFAULT_HANDOFF_TIMEOUT_SECONDS) -> int:
    source = source_from_dict(request["source"])
    cache_dir = Path(str(request["cache_dir"])).expanduser()
    limit_raw = request.get("limit")
    threshold = float(request.get("threshold", B0_THRESHOLD))
    shm = None
    try:
        if not Path(source.dwi).is_file():
            raise PreviewError("missing")
        limit = full_dwi_limit_bytes() if limit_raw is None else int(limit_raw)
        estimated = estimate_full_dwi_bytes(source)
        if gradient_table_unusable(source):
            mean_path = mean_b0_image(source, cache_dir, threshold)
            payload: dict[str, Any] = {
                "ok": True,
                "kind": "meanb0",
                "mean_b0": str(mean_path),
                "estimated_bytes": int(estimated),
                "limit_bytes": int(limit),
                "reason": "no_gradients",
            }
        elif estimated > limit:
            mean_path = mean_b0_image(source, cache_dir, threshold)
            payload = {
                "ok": True,
                "kind": "meanb0",
                "mean_b0": str(mean_path),
                "estimated_bytes": int(estimated),
                "limit_bytes": int(limit),
                "reason": "size",
            }
        else:
            try:
                shm, meta = read_full_dwi_to_shared_memory(source)
                payload = {
                    "ok": True,
                    "kind": "dwi",
                    "shm_name": meta["shm_name"],
                    "shape_ijk": list(meta["shape_ijk"]),
                    "count": int(meta["count"]),
                    "dtype": str(meta["dtype"]),
                    "ijk_to_ras": meta["ijk_to_ras"].tolist(),
                    "gradients": meta["gradients"].tolist(),
                    "bvalues": [float(b) for b in meta["bvalues"]],
                    "measurement_frame": meta["measurement_frame"].tolist(),
                    "voxel_bytes": int(meta["voxel_bytes"]),
                    "estimated_bytes": int(estimated),
                    "limit_bytes": int(limit),
                    "reason": "",
                }
            except (MemoryError, OSError):
                # Insufficient commit/SHM for full 4D DWI: fall back cleanly to mean b0.
                mean_path = mean_b0_image(source, cache_dir, threshold)
                payload = {
                    "ok": True,
                    "kind": "meanb0",
                    "mean_b0": str(mean_path),
                    "estimated_bytes": int(estimated),
                    "limit_bytes": int(limit),
                    "reason": "size",
                }
        _write_json_atomic(result_path, payload)
        print(READY_SENTINEL, flush=True)
        if shm is not None:
            _wait_for_ack(handoff_timeout)
        return 0
    except PreviewError as exc:
        _write_json_atomic(result_path, {"ok": False, "error_code": exc.code, "detail": exc.detail})
        print(READY_SENTINEL, flush=True)
        return 0
    except Exception as exc:
        _write_json_atomic(result_path, {"ok": False, "error_code": "bad_data", "detail": str(exc)})
        print(READY_SENTINEL, flush=True)
        return 1
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-process input preview worker for SlicerDSpace.")
    parser.add_argument("--request", required=True, help="JSON request file path.")
    parser.add_argument("--result", required=True, help="JSON result file path.")
    parser.add_argument("--handoff-timeout", type=float, default=_DEFAULT_HANDOFF_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    return run_request(request, Path(args.result), handoff_timeout=args.handoff_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
