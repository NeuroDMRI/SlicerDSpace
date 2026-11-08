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
    load_sha256_cache,
    save_sha256_cache,
    verify_against_manifest,
)


MODEL_ASSETS_ENV = "SLICERDSPACE_MODEL_ASSETS_DIR"

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


def configured_model_assets_root(explicit: Path | None = None) -> Path:
    """Return the explicit/env/default user asset root without legacy fallback."""

    if explicit is not None:
        return explicit.expanduser().resolve()
    env_root = os.environ.get(MODEL_ASSETS_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return default_user_model_assets_root().expanduser().resolve()


def legacy_repo_asset_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return checkout-relative roots used by older SlicerDSpace installs."""

    root = repo_root.expanduser().resolve()
    return (root,)


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
    re-checked). A manifest-recorded file that is simply not present yet
    (not downloaded) does not by itself fail the directory -- but if none of
    the recorded files are present at all, the directory does not count as
    verified either. A directory with no manifest coverage at all falls
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

    if explicit is not None or os.environ.get(MODEL_ASSETS_ENV):
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


def model_asset_path(relative: str, *, repo_root: Path, explicit_root: Path | None = None) -> Path:
    """Return the best existing, verified path for one model asset, or its default target.

    An explicitly user-supplied root (``explicit_root`` or the
    ``SLICERDSPACE_MODEL_ASSETS_DIR`` env var) is the user's own informed
    choice and is returned as-is, unverified -- same as
    :func:`resolve_model_assets_root`. Otherwise, an existing candidate that
    fails sha256 verification against ``models/assets_manifest.json`` raises
    :class:`~shared.model_asset_integrity.ModelAssetIntegrityError` rather
    than being silently returned: callers here hand the result straight to
    ``torch.load()``, and have no other way to distinguish "not downloaded
    yet" from "wrong/tampered content".
    """

    if explicit_root is not None or os.environ.get(MODEL_ASSETS_ENV):
        root = resolve_model_assets_root(explicit=explicit_root, repo_root=repo_root, required_paths=(relative,))
        return root / relative

    manifest_records = _load_asset_manifest_records(repo_root)
    root = resolve_model_assets_root(explicit=explicit_root, repo_root=repo_root, required_paths=(relative,))
    candidate = root / relative
    if candidate.exists():
        if _has_verified_required_asset(root, (relative,), manifest_records=manifest_records):
            return candidate
        raise ModelAssetIntegrityError(
            f"{candidate} exists but does not match its recorded sha256 in models/assets_manifest.json"
        )
    for legacy_root in legacy_repo_asset_roots(repo_root):
        legacy = legacy_root / relative
        if legacy.exists():
            if _has_verified_required_asset(legacy_root, (relative,), manifest_records=manifest_records):
                return legacy
            raise ModelAssetIntegrityError(
                f"{legacy} exists but does not match its recorded sha256 in models/assets_manifest.json"
            )
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
