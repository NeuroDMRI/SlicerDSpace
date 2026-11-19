"""Sample-data registry helpers without bundling restricted datasets."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAMPLE_DATA_REGISTRY_KIND = "slicerdsurfer_sample_data_registry"
SAMPLE_DATA_REGISTRY_VERSION = 1


@dataclass(frozen=True)
class SampleDataEntry:
    """One externally hosted or local sample-data entry."""

    sample_id: str
    label: str
    source: str
    source_type: str
    license: str
    description: str = ""
    sha256: str | None = None


def load_sample_data_registry(path: Path) -> tuple[SampleDataEntry, ...]:
    """Load a sample-data registry from JSON."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if payload.get("kind") != SAMPLE_DATA_REGISTRY_KIND:
        raise ValueError(f"Unsupported sample-data registry kind: {payload.get('kind')}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Sample-data registry must contain an entries list.")
    return tuple(
        SampleDataEntry(
            sample_id=str(entry["sample_id"]),
            label=str(entry["label"]),
            source=str(entry["source"]),
            source_type=str(entry.get("source_type") or "url"),
            license=str(entry.get("license") or "unknown"),
            description=str(entry.get("description") or ""),
            sha256=str(entry["sha256"]) if entry.get("sha256") else None,
        )
        for entry in entries
    )


def sample_data_registry_payload(entries: tuple[SampleDataEntry, ...]) -> dict[str, Any]:
    """Build a registry payload for local labs to publish their own examples."""

    return {
        "kind": SAMPLE_DATA_REGISTRY_KIND,
        "version": SAMPLE_DATA_REGISTRY_VERSION,
        "policy": "Registry entries must point to redistributable data with explicit license terms.",
        "entries": [
            {
                "sample_id": entry.sample_id,
                "label": entry.label,
                "source": entry.source,
                "source_type": entry.source_type,
                "license": entry.license,
                "description": entry.description,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
    }


def fetch_sample_source(entry: SampleDataEntry, destination: Path, *, overwrite: bool = False) -> Path:
    """Verify a staged sample before publishing it; preserve existing data on failure."""

    destination = destination.expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("Sample destination must not be a symbolic link")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sample destination: {destination}")
    expected = (entry.sha256 or "").lower()
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid sample SHA256")
    if entry.source_type == "url" and not expected:
        raise ValueError("Remote sample data requires an explicit SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sdds-sample-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "download"
        if entry.source_type == "local":
            shutil.copyfile(Path(entry.source).expanduser().resolve(), staged)
        elif entry.source_type == "url":
            with urllib.request.urlopen(entry.source, timeout=30) as response, staged.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        else:
            raise ValueError(f"Unsupported sample source_type: {entry.source_type}")
        digest = hashlib.sha256()
        with staged.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if expected and digest.hexdigest() != expected:
            raise ValueError("Sample SHA256 mismatch; no destination file was published")
        if overwrite:
            os.replace(staged, destination)
        else:
            # A second downloader must not replace a destination created meanwhile.
            try:
                os.link(staged, destination)
            except FileExistsError:
                raise
            except OSError:
                if destination.exists():
                    raise FileExistsError(f"Destination already exists: {destination}")
                shutil.copy2(staged, destination)
    return destination
