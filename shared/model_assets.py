"""Shared model-asset root resolution.

Model weights and templates are intentionally treated as user data, not as
files that must live inside the Slicer extension checkout.  Legacy checkout-
relative locations remain fallback candidates so older local installations do
not break.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from shared.model_asset_integrity import (
    DEFAULT_CACHE_FILENAME,
    ModelAssetIntegrityError,
    cached_sha256_of_file,
    load_sha256_cache,
    save_sha256_cache,
    verify_against_manifest,
)


MODEL_ASSETS_ENV = "SLICERDSPACE_MODEL_ASSETS_DIR"
BUNDLED_MODEL_ASSETS_ENV = "SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT"

# In-process cache of verified sha256 digests, lazily loaded from and
# persisted to disk (see _sha256_cache()) so a file is only re-hashed when
# its mtime/size actually change, not on every resolve call.
_SHA256_CACHE: dict[str, Any] | None = None


def _sha256_cache_path() -> Path:
    from shared.user_config import default_user_config_path

    return default_user_config_path().parent / DEFAULT_CACHE_FILENAME


def _sha256_cache() -> dict[str, Any]:
    global _SHA256_CACHE
    if _SHA256_CACHE is None:
        _SHA256_CACHE = load_sha256_cache(_sha256_cache_path())
    return _SHA256_CACHE


def _persist_sha256_cache() -> None:
    if _SHA256_CACHE is not None:
        try:
            save_sha256_cache(_sha256_cache_path(), _SHA256_CACHE)
        except OSError:
            pass


def _load_asset_manifest_records(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        return {}
    manifest_path = repo_root.expanduser().resolve() / "models" / "assets_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = manifest.get("asset_records")
    return records if isinstance(records, dict) else {}


_FALLBACK_HOME = Path.home()


class ModelAssetIncompleteError(ModelAssetIntegrityError):
    """LE-2: a model-asset folder exists but lacks files the manifest records (an interrupted download)."""


def incomplete_manifest_files(root: Path, relative: str, manifest_records: dict[str, Any]) -> list[str]:
    """LE-2: list the manifest-recorded files under the folder ``relative`` that are missing or have the wrong size.

    Every file the manifest records under the prefix must exist in ``root`` and, when the record holds
    ``size_bytes``, have exactly that size. Only a stat() is used; sha256 stays in the cached check of the caller.
    Returns an empty list when the manifest records nothing under the prefix (nothing to compare against).
    """

    keys = [key for key in _relative_manifest_keys(relative, manifest_records) if key != relative]
    prefix = relative.rstrip("/") + "/"
    problems: list[str] = []
    for key in sorted(keys):
        name = key[len(prefix):] if key.startswith(prefix) else key
        path = root / key
        try:
            if not path.is_file():
                problems.append(f"{name} (missing)")
                continue
            record = manifest_records.get(key)
            expected = record.get("size_bytes") if isinstance(record, dict) else None
            if isinstance(expected, int) and not isinstance(expected, bool) and path.stat().st_size != expected:
                problems.append(f"{name} (size {path.stat().st_size} B, expected {expected} B)")
        except OSError:
            problems.append(f"{name} (unreadable)")
    return problems


def incomplete_asset_message(directory: Path, problems: list[str]) -> str:
    shown = ", ".join(problems[:6]) + (f" and {len(problems) - 6} more" if len(problems) > 6 else "")
    return (
        f"{directory} is incomplete (an interrupted download?): {shown}. "
        "Use 'Install models' (or scripts/download_model_assets.py --profile core) to finish the download."
    )


def default_user_model_assets_root() -> Path:
    """Return the per-user default model-asset root."""

    from shared.branding import reuse_legacy_path
    from shared.environment import _safe_home
    home = _safe_home() or _FALLBACK_HOME
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        return reuse_legacy_path(base / "SlicerDSpace" / "model-assets")
    if system == "Darwin":
        return reuse_legacy_path(home / "Library" / "Application Support" / "SlicerDSpace" / "model-assets")
    return reuse_legacy_path(Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "SlicerDSpace" / "model-assets")


def configured_model_assets_root(explicit: Path | str | None = None) -> Path:
    """Return the explicit/env/default user asset root without legacy fallback."""

    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env_root = os.environ.get(MODEL_ASSETS_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return default_user_model_assets_root().expanduser().resolve()


def bundled_model_asset_roots(repo_root: Path | str | None = None) -> tuple[Path, ...]:
    """Return existing ``bundled-model-assets`` directories inside an installed extension."""

    env_override = os.environ.get(BUNDLED_MODEL_ASSETS_ENV)
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    else:
        resolved = Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
        if resolved.name == "qt-scripted-modules" and len(resolved.parents) >= 3:
            candidates.append(resolved.parents[2] / "bundled-model-assets")
        candidates.append(resolved / "bundled-model-assets")
    valid: list[Path] = []
    for cand in candidates:
        if cand.is_dir():
            real = cand.resolve()
            if real not in valid:
                valid.append(real)
    return tuple(valid)


def legacy_repo_asset_roots(repo_root: Path | str) -> tuple[Path, ...]:
    """Return bundled extension asset roots and checkout-relative roots."""

    root = Path(repo_root).expanduser().resolve()
    roots = [*bundled_model_asset_roots(root), root]
    deduped: list[Path] = []
    for item in roots:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def common_model_asset_roots() -> tuple[Path, ...]:
    """Return the roots to check automatically before falling back to the checkout.

    Deliberately narrow. Earlier versions scanned a long list of speculative
    shared/system locations (e.g. ``/data/shared/...``, ``/opt/...``, even a
    specific developer's home directory on the project's own server). Any
    local account -- or another user on a shared multi-user machine -- who
    could place a file at one of those well-known, guessable paths would
    have it picked up and loaded as a trusted model checkpoint. Only the
    current user's own configured default is checked automatically; any
    other location must be explicit (``--asset-root`` /
    ``SLICERDSPACE_MODEL_ASSETS_DIR``) or the extension's own checkout
    (see :func:`legacy_repo_asset_roots`), and even those are still
    sha256-verified against the manifest before being trusted (see
    :func:`resolve_model_assets_root`).
    """

    return (default_user_model_assets_root(),)


def _relative_manifest_keys(relative: str, manifest_records: dict[str, Any]) -> list[str]:
    if relative in manifest_records:
        return [relative]
    prefix = relative.rstrip("/") + "/"
    return [key for key in manifest_records if key.startswith(prefix)]


def _has_verified_required_asset(
    root: Path,
    required_paths: tuple[str, ...],
    *,
    manifest_records: dict[str, Any],
) -> bool:
    """Return True if ``root`` has at least one required asset that verifies.

    A required path that is itself a file is checked directly. A required
    path that is a directory (the common case: callers ask for e.g.
    ``"segmentation/weights"``) is accepted only if *every* file the
    manifest actually records under that prefix that is currently present
    on disk matches its recorded sha256 -- not just any one of them:
    otherwise a directory containing one genuine file alongside tampered or
    substituted ones would be accepted wholesale, and the caller only
    verifies at the directory level (individual files loaded from it later,
    e.g. one of several backbone checkpoints picked at runtime, are never
    re-checked). LE-2: a manifest-recorded file
    that is missing (or has the wrong size) fails the directory too: a folder
    left by an interrupted download is not accepted. A directory with no manifest coverage at all falls
    back to plain non-emptiness, since there is nothing recorded to verify
    it against.
    """

    cache = _sha256_cache()
    verified_anything = False
    accepted = False
    for relative in required_paths:
        if accepted:
            break
        candidate = root / relative
        if candidate.is_file():
            ok, _reason = verify_against_manifest(candidate, relative, manifest_records, cache=cache)
            verified_anything = True
            if ok:
                accepted = True
            continue
        if not candidate.is_dir():
            continue
        manifest_keys = _relative_manifest_keys(relative, manifest_records)
        if not manifest_keys:
            if any(candidate.iterdir()):
                accepted = True
            continue
        existing_keys = [key for key in manifest_keys if (root / key).is_file()]
        if not existing_keys:
            continue
        if incomplete_manifest_files(root, relative, manifest_records):
            # LE-2: a folder with only some of the recorded files (interrupted download) is not usable.
            continue
        all_verified = True
        for key in existing_keys:
            ok, _reason = verify_against_manifest(root / key, key, manifest_records, cache=cache)
            verified_anything = True
            if not ok:
                all_verified = False
        if all_verified:
            accepted = True
    if verified_anything:
        _persist_sha256_cache()
    return accepted


def resolve_model_assets_root(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
    required_paths: tuple[str, ...] = (),
    prefer_existing_legacy: bool = True,
) -> Path:
    """Return the asset root used for download/verify operations."""

    if (explicit is not None and str(explicit).strip()) or os.environ.get(MODEL_ASSETS_ENV):
        return configured_model_assets_root(explicit)
    manifest_records = _load_asset_manifest_records(repo_root)
    for candidate in common_model_asset_roots():
        if _has_verified_required_asset(candidate, required_paths, manifest_records=manifest_records):
            return candidate.expanduser().resolve()
    if prefer_existing_legacy and repo_root is not None:
        for legacy in legacy_repo_asset_roots(repo_root):
            if _has_verified_required_asset(legacy, required_paths, manifest_records=manifest_records):
                return legacy
    return configured_model_assets_root()


class ModelAssetOtherReleaseError(ModelAssetIntegrityError):
    """A bundled model file is a complete file of another release of the model than the manifest records.

    Install test of 26 November 2026, finding 4: the v0.1.1 base package ships the DDSurfer weights of that release,
    and a code update brings the v1.0.0 manifest. The copy is refused like every sha256 mismatch.
    """


# Callers that show the message (the environment check) find this case by the marker, like "is incomplete".
OTHER_RELEASE_MARKER = "is from another release"


def other_release_message(candidate: Path) -> str:
    return (
        f"{candidate} {OTHER_RELEASE_MARKER} of this model: it agrees with the SHA256SUMS list in its folder, but "
        "models/assets_manifest.json records another sha256 for this version of SlicerDSpace. The installed base "
        "package and the code are from different releases. Download the current model files (Install models in "
        "the Slicer UI, or scripts/download_model_assets.py). The bundled copy stays in place and is not used."
    )


def _matches_own_release_checksums(candidate: Path) -> bool:
    """True when the ``SHA256SUMS`` file next to ``candidate`` records it correctly.

    For ``SHA256SUMS`` itself: every file that it lists is present and matches. A release ships this list with its
    checkpoints (DDSurfer does), so a match means a complete file of that release, not a damaged or replaced one.
    The answer only chooses the message of an error that is raised anyway, so a list or a file that cannot be
    checked reads as "no match" and never raises.
    """

    try:
        sums_path = candidate.parent / "SHA256SUMS"
        if not candidate.is_file() or not sums_path.is_file():
            return False
        recorded: dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            name = parts[1].strip().lstrip("*") if len(parts) == 2 else ""
            if not name or Path(name).name != name or name in recorded:
                return False
            recorded[name] = parts[0].lower()
        names = tuple(recorded) if candidate.name == "SHA256SUMS" else (candidate.name,)
        if not names:
            return False
        cache = _sha256_cache()
        for name in names:
            path = candidate.parent / name
            if name not in recorded or path.is_symlink() or not path.is_file():
                return False
            if cached_sha256_of_file(path, cache) != recorded[name]:
                return False
        _persist_sha256_cache()
        return True
    except (OSError, ValueError):
        return False


def _unverified_asset_error(
    root: Path, relative: str, candidate: Path, manifest_records: dict[str, Any], *, bundled: bool = False
) -> ModelAssetIntegrityError:
    """LE-2: name the missing files of an incomplete folder; otherwise the sha256 mismatch.

    ``bundled``: the root is the extension's own ``bundled-model-assets``. Only there a complete file of another
    release gets its own message: the user cannot repair the base package, and the download goes to the user root.
    """

    if candidate.is_dir():
        problems = incomplete_manifest_files(root, relative, manifest_records)
        if problems:
            return ModelAssetIncompleteError(incomplete_asset_message(candidate, problems))
    if bundled and _matches_own_release_checksums(candidate):
        return ModelAssetOtherReleaseError(other_release_message(candidate))
    return ModelAssetIntegrityError(
        f"{candidate} exists but does not match its recorded sha256 in models/assets_manifest.json"
    )


def model_asset_path(relative: str, *, repo_root: Path, explicit_root: Path | None = None) -> Path:
    """Return the best existing, verified path for one model asset, or its default target.

    An explicitly user-supplied root (``explicit_root`` or the
    ``SLICERDSPACE_MODEL_ASSETS_DIR`` env var) is the user's own informed
    choice and is returned as-is, unverified -- same as
    :func:`resolve_model_assets_root`. When that root does not hold the
    asset, the copy in the extension's own ``bundled-model-assets`` is used
    instead, and it must pass the sha256 check. Otherwise, an existing candidate that
    fails sha256 verification against ``models/assets_manifest.json`` raises
    :class:`~shared.model_asset_integrity.ModelAssetIntegrityError` rather
    than being silently returned: callers here hand the result straight to
    ``torch.load()``, and have no other way to distinguish "not downloaded
    yet" from "wrong/tampered content".
    """

    if (explicit_root is not None and str(explicit_root).strip()) or os.environ.get(MODEL_ASSETS_ENV):
        root = resolve_model_assets_root(explicit=explicit_root, repo_root=repo_root, required_paths=(relative,))
        candidate = root / relative
        manifest_records = _load_asset_manifest_records(repo_root)
        # Opus P2 (bundled fallback): a folder left by an interrupted download is treated as absent when a complete,
        # verified bundled copy exists. When no such copy exists, the configured folder stays the answer, so the
        # caller reports it as incomplete.
        partial = candidate.is_dir() and bool(incomplete_manifest_files(root, relative, manifest_records))
        if candidate.exists() and not partial:
            return candidate
        # Setup saves the asset root after the first on-demand install. That root then lacks what the base package
        # ships in ``bundled-model-assets``: use the bundled copy, verified like every automatic root.
        for bundled_root in bundled_model_asset_roots(repo_root):
            bundled = bundled_root / relative
            if bundled.exists():
                if _has_verified_required_asset(bundled_root, (relative,), manifest_records=manifest_records):
                    return bundled
                if not partial:
                    raise _unverified_asset_error(bundled_root, relative, bundled, manifest_records, bundled=True)
        return candidate

    manifest_records = _load_asset_manifest_records(repo_root)
    root = resolve_model_assets_root(explicit=explicit_root, repo_root=repo_root, required_paths=(relative,))
    candidate = root / relative
    if candidate.exists():
        if _has_verified_required_asset(root, (relative,), manifest_records=manifest_records):
            return candidate
        raise _unverified_asset_error(root, relative, candidate, manifest_records)
    for legacy_root in legacy_repo_asset_roots(repo_root):
        legacy = legacy_root / relative
        if legacy.exists():
            if _has_verified_required_asset(legacy_root, (relative,), manifest_records=manifest_records):
                return legacy
            raise _unverified_asset_error(
                legacy_root, relative, legacy, manifest_records,
                bundled=legacy_root in bundled_model_asset_roots(repo_root))
    return candidate


def model_asset_path_or_none(
    relative: str, *, repo_root: Path, explicit_root: Path | None = None
) -> tuple[Path | None, str | None]:
    """Return ``(model_asset_path(...), None)``, or ``(None, message)`` on integrity failure.

    Callers that only use the result to *auto-populate a UI field* or to
    *report a doctor/dependency status* must not let
    :class:`~shared.model_asset_integrity.ModelAssetIntegrityError` propagate:
    doing so aborts the whole caller (module initialization, or the entire
    environment-doctor scan for every target, not just the one asset),
    which is worse than just reporting one item as unavailable. Callers that
    hand the result straight to ``torch.load()`` should keep calling
    :func:`model_asset_path` directly so a tampered checkpoint is a hard,
    unmissable error.
    """

    try:
        return model_asset_path(relative, repo_root=repo_root, explicit_root=explicit_root), None
    except ModelAssetIntegrityError as exc:
        return None, str(exc)


def model_asset_candidates(relative: str, *, repo_root: Path, explicit_root: Path | None = None) -> tuple[Path, ...]:
    """Return ordered candidate paths for diagnostics."""

    primary = resolve_model_assets_root(explicit=explicit_root, repo_root=repo_root, required_paths=(relative,))
    candidates = [primary / relative]
    for common_root in common_model_asset_roots():
        common = common_root / relative
        if common not in candidates:
            candidates.append(common)
    for legacy_root in legacy_repo_asset_roots(repo_root):
        legacy = legacy_root / relative
        if legacy not in candidates:
            candidates.append(legacy)
    return tuple(candidates)
