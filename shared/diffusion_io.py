"""Bounded-memory fast streaming decompression and verified uncompressed NIfTI stage cache."""

from contextlib import contextmanager
import contextvars
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Iterator, Optional

_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_CACHE_MIN_BYTES = 16 * 1024 * 1024
_CACHE_ELIGIBLE_PARENTS = frozenset({"input", "preprocessed", "01_preprocessing"})
_SUBJECT_CACHE_ENV_VAR = "SLICERDSPACE_SUBJECT_CACHE_ROOT"
_ACTIVE_SUBJECT_CACHE_ROOT: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "slicerdspace_active_subject_cache_root",
    default=None,
)
_SUBJECT_CACHE_ENV_LOCK = threading.RLock()


def active_subject_cache_root() -> Optional[Path]:
    """Return the active subject cache root from context or subprocess environment."""

    scoped = _ACTIVE_SUBJECT_CACHE_ROOT.get()
    if scoped is not None:
        return scoped
    env_value = (os.environ.get(_SUBJECT_CACHE_ENV_VAR) or "").strip()
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


@contextmanager
def subject_cache_root(cache_root: Optional[Path]) -> Iterator[Optional[Path]]:
    """Scope uncompressed NIfTI cache writes to ``cache_root`` for threads and subprocesses."""

    resolved = Path(cache_root).expanduser().resolve() if cache_root is not None else None
    token = _ACTIVE_SUBJECT_CACHE_ROOT.set(resolved)
    with _SUBJECT_CACHE_ENV_LOCK:
        previous_env = os.environ.get(_SUBJECT_CACHE_ENV_VAR)
        if resolved is None:
            os.environ.pop(_SUBJECT_CACHE_ENV_VAR, None)
        else:
            os.environ[_SUBJECT_CACHE_ENV_VAR] = resolved.as_posix()
    try:
        yield resolved
    finally:
        with _SUBJECT_CACHE_ENV_LOCK:
            if previous_env is None:
                os.environ.pop(_SUBJECT_CACHE_ENV_VAR, None)
            else:
                os.environ[_SUBJECT_CACHE_ENV_VAR] = previous_env
        _ACTIVE_SUBJECT_CACHE_ROOT.reset(token)


class _ProcessStreamReader:
    """Binary stream wrapper over a decompressor subprocess pipe with EOF exit checks."""

    def __init__(self, proc: subprocess.Popen, path: Path) -> None:
        assert proc.stdout is not None
        self._proc = proc
        self._stdout = proc.stdout
        self._path = path
        self.saw_eof = False
        self.verified = False

    def read(self, size: int = -1) -> bytes:
        chunk = self._stdout.read(size)
        if not chunk:
            self.saw_eof = True
            self.finish()
        return chunk

    def finish(self) -> None:
        if self.verified:
            return
        try:
            self._stdout.close()
        except Exception:
            pass
        rc = self._proc.wait()
        self.verified = True
        if rc != 0:
            raise EOFError(f"Corrupted or truncated gzip NIfTI stream: {self._path} (exit {rc})")


class _ThreadedGzipReader:
    """Background-thread gzip reader feeding an 8 MiB chunk queue (cross-platform fallback)."""

    def __init__(self, path: Path, chunk_bytes: int = _STREAM_CHUNK_BYTES) -> None:
        self._path = path
        self._chunk_bytes = chunk_bytes
        self._queue: queue.Queue[tuple[Optional[bytes], Optional[BaseException]]] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._buffer = b""
        self._offset = 0
        self._done = False
        self._thread = threading.Thread(target=self._produce, name="slicerdspace-gzip", daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            with gzip.open(self._path, "rb") as stream:
                while not self._stop.is_set():
                    data = stream.read(self._chunk_bytes)
                    if not data:
                        self._queue.put((b"", None))
                        return
                    while not self._stop.is_set():
                        try:
                            self._queue.put((data, None), timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:
            while not self._stop.is_set():
                try:
                    self._queue.put((None, exc), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size is None or size < 0:
            parts = []
            while True:
                piece = self.read(self._chunk_bytes)
                if not piece:
                    return b"".join(parts)
                parts.append(piece)
        out = bytearray()
        while len(out) < size:
            if self._offset >= len(self._buffer):
                if self._done:
                    break
                data, exc = self._queue.get()
                if exc is not None:
                    self._done = True
                    raise exc
                if not data:
                    self._done = True
                    break
                self._buffer = data
                self._offset = 0
            avail = len(self._buffer) - self._offset
            take = min(avail, size - len(out))
            out.extend(self._buffer[self._offset : self._offset + take])
            self._offset += take
        return bytes(out)

    def close(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


@contextmanager
def open_fast_gzip_reader(path: Path):
    """Yield a fast binary reader for ``.nii`` or ``.nii.gz``.

    On macOS/Linux, spawns native ``pigz -dc`` or ``gzip -dc`` in a separate OS
    process (escaping Python's GIL and Rosetta x86_64 emulation overhead on Apple
    Silicon when using system ``/usr/bin/gzip``). On Windows or when unavailable,
    runs ``gzip.open`` on a background producer thread.
    """

    source = Path(path)
    if not source.name.lower().endswith(".gz"):
        with source.open("rb", buffering=_STREAM_CHUNK_BYTES) as handle:
            yield handle
        return

    decompressor = shutil.which("pigz") or (shutil.which("gzip") if sys.platform != "win32" else None)
    if decompressor:
        from shared.external_process import external_python_environment, subprocess_no_window_kwargs

        try:
            proc = subprocess.Popen(
                [decompressor, "-dc", str(source)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=external_python_environment(),
                bufsize=_STREAM_CHUNK_BYTES,
                **subprocess_no_window_kwargs(),
            )
        except OSError:
            proc = None
        if proc is not None:
            reader = _ProcessStreamReader(proc, source)
            exc_active = False
            try:
                yield reader
            except BaseException:
                exc_active = True
                raise
            finally:
                if not reader.verified:
                    try:
                        if reader._stdout and not reader._stdout.closed:
                            reader._stdout.close()
                    except Exception:
                        pass
                    if exc_active or not reader.saw_eof:
                        try:
                            proc.terminate()
                            proc.wait(timeout=2.0)
                        except Exception:
                            try:
                                proc.kill()
                                proc.wait(timeout=2.0)
                            except Exception:
                                pass
                    else:
                        reader.finish()
            return

    reader = _ThreadedGzipReader(source)
    try:
        yield reader
    finally:
        reader.close()


def _cache_disabled() -> bool:
    return os.environ.get("SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE", "").strip().lower() in {"1", "true", "yes"}


def _cache_min_bytes() -> int:
    raw = os.environ.get("SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _DEFAULT_CACHE_MIN_BYTES


def _has_cache_disk_budget(probe_dir: Path, needed_bytes: int) -> bool:
    check_dir = probe_dir
    while not check_dir.exists() and check_dir != check_dir.parent:
        check_dir = check_dir.parent
    margin = 16 * 1024 * 1024 if os.environ.get("SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES") else 2 * (1024 ** 3)
    return shutil.disk_usage(check_dir).free >= max(0, int(needed_bytes)) * 2 + margin


def _cache_paths_for_source(source: Path) -> Optional[tuple[Path, Path, Path]]:
    resolved = Path(source).expanduser().resolve()
    if _cache_disabled():
        return None
    if not resolved.name.lower().endswith(".nii.gz"):
        return None
    if resolved.parent.name not in _CACHE_ELIGIBLE_PARENTS:
        return None
    active = active_subject_cache_root()
    cache_dir = (
        (active / "uncompressed_nifti")
        if active is not None
        else (resolved.parent.parent / "cache" / "uncompressed_nifti")
    )
    digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()[:20]
    return cache_dir, cache_dir / f"{digest}.nii", cache_dir / f"{digest}.json"


def _prune_old_cache_entries(cache_dir: Path, *, keep_name: str, max_entries: int = 2) -> None:
    try:
        entries = sorted(
            (p for p in cache_dir.glob("*.nii") if p.is_file() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return
    kept = 0
    for item in entries:
        if item.name == keep_name or kept < max_entries:
            kept += 1
            continue
        item.unlink(missing_ok=True)
        item.with_suffix(".json").unlink(missing_ok=True)


def get_cached_uncompressed_nifti(path: Path) -> Optional[Path]:
    """Return a validated cached uncompressed ``.nii`` path for ``path`` if one is fresh."""

    source = Path(path)
    targets = _cache_paths_for_source(source)
    if targets is None or not source.is_file():
        return None
    _, cached_nii, meta_path = targets
    try:
        src_stat = source.stat()
    except OSError:
        return None
    if src_stat.st_size < _cache_min_bytes():
        return None
    if not (cached_nii.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("source_size", -1)) != int(src_stat.st_size):
            return None
        if int(meta.get("source_mtime_ns", -1)) != int(src_stat.st_mtime_ns):
            return None
        expected_bytes = int(meta.get("expected_bytes", 0))
        if expected_bytes <= 0 or cached_nii.stat().st_size < expected_bytes:
            return None
        import nibabel as nib

        img = nib.load(str(cached_nii), mmap="r", keep_file_open=False)
        if not img.shape:
            return None
    except Exception:
        return None
    return cached_nii


def prepare_uncompressed_cache_target(target_gz_path: Path, expected_bytes: int) -> Optional[Path]:
    """Create a staging path under ``cache/uncompressed_nifti`` when caching is eligible and safe."""

    targets = _cache_paths_for_source(target_gz_path)
    if targets is None:
        return None
    cache_dir, _, _ = targets
    try:
        if not _has_cache_disk_budget(cache_dir.parent, expected_bytes):
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".staging-", suffix=".nii", dir=cache_dir)
        os.close(fd)
        return Path(name)
    except Exception:
        return None


def commit_predecoded_nifti_cache(
    target_gz_path: Path,
    staged_nii: Optional[Path],
    expected_bytes: int,
) -> Optional[Path]:
    """Atomically promote a pre-decoded ``.nii`` staging file after ``target_gz_path`` is committed."""

    if staged_nii is None or not staged_nii.is_file():
        return None
    targets = _cache_paths_for_source(target_gz_path)
    if targets is None or not target_gz_path.is_file():
        staged_nii.unlink(missing_ok=True)
        return None
    cache_dir, cached_nii, meta_path = targets
    try:
        src_stat = target_gz_path.stat()
        if src_stat.st_size < _cache_min_bytes() or staged_nii.stat().st_size < expected_bytes:
            staged_nii.unlink(missing_ok=True)
            return None
        import nibabel as nib

        nib.load(str(staged_nii), mmap="r", keep_file_open=False)
        os.replace(staged_nii, cached_nii)
        fd, tmp_meta_name = tempfile.mkstemp(prefix=".meta-", suffix=".json", dir=cache_dir)
        os.close(fd)
        tmp_meta = Path(tmp_meta_name)
        try:
            tmp_meta.write_text(
                json.dumps(
                    {
                        "source_path": str(target_gz_path.resolve()),
                        "source_size": int(src_stat.st_size),
                        "source_mtime_ns": int(src_stat.st_mtime_ns),
                        "expected_bytes": int(expected_bytes),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_meta, meta_path)
        finally:
            tmp_meta.unlink(missing_ok=True)
        _prune_old_cache_entries(cache_dir, keep_name=cached_nii.name)
        return cached_nii
    except Exception:
        staged_nii.unlink(missing_ok=True)
        return None


def nifti_data_offset(image) -> int:
    """Return the exact byte offset of the NIfTI voxel payload, preserving header extensions.

    Nibabel moves ``vox_offset`` from ``image.header`` into ``image.dataobj.offset``
    on load and resets ``header['vox_offset']`` to ``0``; reading only the header
    misreads any file with header extensions (``vox_offset > 352``).
    """

    proxy = getattr(image, "dataobj", None)
    offset = getattr(proxy, "offset", None)
    if offset is not None:
        return int(offset)
    hdr = getattr(image, "header", None)
    if hdr is not None and hasattr(hdr, "get_data_offset"):
        raw = int(hdr.get_data_offset() or 0)
        if raw > 0:
            return raw
    import nibabel as nib

    if isinstance(image, nib.Nifti2Image):
        return 544
    return 352


def nifti_scaling_params(image) -> tuple[float, float, bool]:
    """Return ``(slope, inter, needs_scaling)`` matching ``nibabel``'s ``get_fdata()`` semantics.

    Nibabel moves ``scl_slope`` and ``scl_inter`` from ``image.header`` into
    ``image.dataobj`` on load and resets the header fields to ``NaN``; reading only
    the header silently drops intensity scaling.
    """

    import math

    proxy = getattr(image, "dataobj", None)
    slope = getattr(proxy, "slope", None)
    inter = getattr(proxy, "inter", None)
    if slope is None and inter is None:
        hdr = getattr(image, "header", None)
        if hdr is not None and hasattr(hdr, "get_slope_inter"):
            slope, inter = hdr.get_slope_inter()
    if slope is None:
        return 1.0, 0.0, False
    try:
        slope_f = float(slope)
    except (TypeError, ValueError):
        return 1.0, 0.0, False
    if not math.isfinite(slope_f) or slope_f == 0.0:
        return 1.0, 0.0, False
    try:
        inter_f = 0.0 if inter is None else float(inter)
    except (TypeError, ValueError):
        inter_f = 0.0
    if not math.isfinite(inter_f):
        inter_f = 0.0
    needs = (slope_f != 1.0) or (inter_f != 0.0)
    return slope_f, inter_f, needs


def ensure_cached_uncompressed_nifti(path: Path) -> Optional[Path]:
    """Return an existing verified cached ``.nii`` or create one once for large stage inputs."""

    source = Path(path)
    cached = get_cached_uncompressed_nifti(source)
    if cached is not None:
        return cached
    targets = _cache_paths_for_source(source)
    if targets is None or not source.is_file():
        return None
    try:
        src_stat = source.stat()
        if src_stat.st_size < _cache_min_bytes():
            return None
        import nibabel as nib
        import numpy as np

        image = nib.load(str(source), mmap="r", keep_file_open=False)
        offset = nifti_data_offset(image)
        expected_bytes = int(np.prod(image.shape, dtype=np.int64)) * int(image.header.get_data_dtype().itemsize) + offset
        cache_dir, _, _ = targets
        if expected_bytes <= 0 or not _has_cache_disk_budget(cache_dir.parent, expected_bytes):
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".staging-", suffix=".nii", dir=cache_dir)
        os.close(fd)
        staged = Path(name)
        try:
            from shared.cohort_io import io_admission

            with io_admission("dwi_decompression"), open_fast_gzip_reader(source) as reader, staged.open("wb") as dst:
                while True:
                    chunk = reader.read(_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    dst.write(chunk)
            return commit_predecoded_nifti_cache(source, staged, expected_bytes)
        finally:
            staged.unlink(missing_ok=True)
    except Exception:
        return None


@contextmanager
def diffusion_image(path, work_dir):
    import nibabel as nib
    import numpy as np

    path = Path(path)
    image = nib.load(str(path), mmap="r", keep_file_open=False)
    if not path.name.endswith(".nii.gz"):
        yield image
        return
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = int(np.prod(image.shape)) * image.dataobj.dtype.itemsize + int(image.dataobj.offset)
    if shutil.disk_usage(work_dir).free < needed + 16 * 1024 * 1024:
        raise RuntimeError(f"Insufficient temporary disk space for one-pass DWI decompression ({needed} bytes required)")
    cached = ensure_cached_uncompressed_nifti(path)
    if cached is not None:
        yield nib.load(str(cached), mmap="r", keep_file_open=False)
        return
    with tempfile.TemporaryDirectory(prefix=".dwi-read-", dir=work_dir) as temporary:
        uncompressed = Path(temporary) / "data.nii"
        from shared.cohort_io import io_admission
        with io_admission('dwi_decompression'), open_fast_gzip_reader(path) as source, uncompressed.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=_STREAM_CHUNK_BYTES)
        loaded = nib.load(str(uncompressed), mmap=False if os.name == "nt" else "r", keep_file_open=False)
        try:
            yield loaded
        finally:
            try:
                loaded.uncache()
            except Exception:
                pass
            del loaded
            gc.collect()
