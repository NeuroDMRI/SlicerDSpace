"""CUDA device-index helpers that do not import PyTorch."""

from __future__ import annotations

import os
from typing import Optional


def logical_cuda_index(requested_gpu: str, visible_devices: Optional[str] = None) -> int:
    """Resolve a requested CUDA device string to the PyTorch visible-device index.

    PyTorch (and DDSeg) index ``cuda:<idx>`` directly within the process's
    visible CUDA devices (after ``CUDA_VISIBLE_DEVICES`` filtering/reordering).
    When ``requested_gpu`` is already a valid visible-device ordinal
    (``0 <= idx < len(visible_ids)``), return it directly so
    ``CUDA_VISIBLE_DEVICES=1,0`` maps ``cuda:0 -> 0`` and ``cuda:1 -> 1``,
    matching DDSeg without double-translating ``CUDA_VISIBLE_DEVICES``.

    Only when ``requested_gpu`` is not a valid logical visible-device ordinal
    (for example a legacy physical GPU ID like ``"3"`` under
    ``CUDA_VISIBLE_DEVICES=1,3`` or a GPU UUID) do we match against
    ``visible_ids``.
    """

    requested = str(requested_gpu).split(",", 1)[0].strip()
    if requested.lower().startswith("cuda:"):
        requested = requested.split(":", 1)[1].strip()
    if not requested or requested == "-1":
        return 0
    visible = visible_devices if visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = (
        [value.strip() for value in str(visible).split(",") if value.strip()]
        if visible
        else []
    )
    try:
        index = int(requested)
    except ValueError:
        if requested in visible_ids:
            return visible_ids.index(requested)
        raise
    if index < 0:
        raise ValueError(f"CUDA device index must be non-negative, got {requested_gpu!r}")
    if visible_ids:
        if 0 <= index < len(visible_ids):
            return index
        if requested in visible_ids:
            return visible_ids.index(requested)
    return index


__all__ = ["logical_cuda_index"]
