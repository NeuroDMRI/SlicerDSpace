"""Bounded, process-local cache for the author's registration length filter.

Only the attribute-free, deterministic preprocessing used by registration is
cached. Sampling, random state, objectives, transforms and clustering are not
intercepted. No on-disk product is reused.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import inspect
from pathlib import Path
import time


SUPPORTED_SOURCES = {
    "filter.py": "725f733c0c4396dccc563a314f3037cf3782c59dc4935819e224ef41ac08c4bb",
    "congeal_to_atlas.py": "0ae1682aa09f1b9d2ff6a2984255e0e59920babfcd1a216c8ee1744c64882822",
}


def default_cache_bytes() -> int:
    """Leave most currently available RAM to the author's other operations."""
    try:
        from shared.platform_info import available_host_ram_bytes
        available = available_host_ram_bytes()  # M1/F2: corrected for the Rosetta page-size mismatch
    except ImportError:
        available = None
    if available is None:
        return 256 * 1024 ** 2
    return max(1, min(4 * 1024 ** 3, available // 4))


def registration_cache_support(package_dir: Path) -> dict:
    hashes = {}
    for name in SUPPORTED_SOURCES:
        source = package_dir / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    supported = hashes == SUPPORTED_SOURCES
    return {"supported": supported, "source_sha256": hashes,
            "reason": "verified immutable registration preprocessing" if supported else "unverified author source; running unchanged"}


class RegistrationPreprocessCache:
    """Keep at most the atlas and subject length-filtered polydata in memory."""

    def __init__(self, original, max_entries: int = 2, max_bytes: int | None = None):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.original = original
        self.signature = inspect.signature(original)
        self.max_entries = max_entries
        self.max_bytes = default_cache_bytes() if max_bytes is None else max_bytes
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.entries = OrderedDict()
        self.hits = self.misses = self.bypassed = 0
        self.oversized = self.retained_bytes = self.peak_retained_bytes = 0
        self.filter_seconds = self.lookup_seconds = 0.0

    def __call__(self, *args, **kwargs):
        started = time.monotonic()
        values = self.signature.bind(*args, **kwargs)
        values.apply_defaults()
        values = values.arguments
        # Never retain caller-requested indices, attributes, or side-effectful
        # verbose operations. Unknown parameters disable caching, not execution.
        expected = {"inpd", "min_length_mm", "max_length_mm", "remove_u", "remove_u_endpoint_dist",
                    "remove_brainstem", "return_indices", "return_lengths", "preserve_point_data",
                    "preserve_cell_data", "verbose"}
        flags = ("remove_u", "remove_brainstem", "return_indices", "return_lengths",
                 "preserve_point_data", "preserve_cell_data", "verbose")
        if set(values) != expected or any(values[name] for name in flags):
            self.bypassed += 1
            return self.original(*args, **kwargs)
        source = values["inpd"]
        key = (id(source), source.GetMTime(), values["min_length_mm"], values["max_length_mm"])
        entry = self.entries.get(key)
        if entry is not None and entry[0] is source and entry[1].GetMTime() == entry[2]:
            self.hits += 1
            self.entries.move_to_end(key)
            self.lookup_seconds += time.monotonic() - started
            return entry[1]
        self.misses += 1
        filtered = self.original(*args, **kwargs)
        self.filter_seconds += time.monotonic() - started
        size = int(filtered.GetActualMemorySize()) * 1024
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.retained_bytes -= previous[3]
        if size > self.max_bytes or self.max_bytes == 0:
            self.oversized += 1
            return filtered
        while self.entries and (len(self.entries) >= self.max_entries or
                                self.retained_bytes + size > self.max_bytes):
            self.retained_bytes -= self.entries.popitem(last=False)[1][3]
        # Strong references prevent identity reuse. GetMTime also catches a
        # caller modifying cached points/cells through the supported VTK API.
        self.entries[key] = (source, filtered, filtered.GetMTime(), size)
        self.retained_bytes += size
        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        self.entries.move_to_end(key)
        return filtered

    def summary(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "bypassed": self.bypassed,
                "max_entries": self.max_entries, "retained_entries": len(self.entries),
                "max_bytes": self.max_bytes, "retained_bytes": self.retained_bytes,
                "peak_retained_bytes": self.peak_retained_bytes, "oversized_bypasses": self.oversized,
                "filter_seconds": self.filter_seconds, "lookup_seconds": self.lookup_seconds,
                "scope": "In-memory deterministic length filter only; author downsampling and optimization unchanged"}
