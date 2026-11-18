"""Network mirror, HTTP/SOCKS proxy, split-volume merge, and offline pack helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_PACKS_MANIFEST = REPO_ROOT / "models" / "release_packs_manifest.json"

MIRROR_PREFIX_ENV = "SLICERDSPACE_MIRROR_PREFIX"
PROXY_URL_ENV = "SLICERDSPACE_PROXY_URL"
OFFLINE_DIR_ENV = "SLICERDSPACE_OFFLINE_DIR"

PRESET_MIRRORS: tuple[tuple[str, str, str, str], ...] = (
    ("direct", "", "GitHub Official (Direct)", "GitHub 官方直连"),
    ("ghfast", "https://ghfast.top/", "China Mirror 1 (ghfast.top)", "国内加速镜像 1 (ghfast.top)"),
    ("ghproxy", "https://gh-proxy.com/", "China Mirror 2 (gh-proxy.com)", "国内加速镜像 2 (gh-proxy.com)"),
)

_GITHUB_ORIGIN_PREFIXES = (
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://objects.githubusercontent.com/",
    "https://media.githubusercontent.com/",
    "https://codeload.github.com/",
)

_PACK_LIMIT_FILES = 20000
_PACK_LIMIT_BYTES = 24 * 1024 * 1024 * 1024


def _clean_mirror_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not (text.startswith("https://") or text.startswith("http://")):
        return ""
    return text.rstrip("/") + "/"


def _clean_proxy_url(value: Any) -> str:
    return str(value or "").strip()


def _clean_offline_dir(value: Any) -> str:
    text = str(value or "").strip()
    return text


def load_network_settings(
    user_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return effective mirror, proxy, and offline-directory settings."""

    env = os.environ if environ is None else environ
    config = user_config
    if config is None:
        try:
            from shared.user_config import load_user_config

            config = load_user_config()
        except Exception:
            config = {}
    network = config.get("network") if isinstance(config, Mapping) else {}
    if not isinstance(network, Mapping):
        network = {}

    mirror_prefix = env.get(MIRROR_PREFIX_ENV)
    if mirror_prefix is None:
        mirror_prefix = network.get("mirror_prefix", "")

    proxy_url = (
        env.get(PROXY_URL_ENV)
        or env.get("HTTPS_PROXY")
        or env.get("https_proxy")
        or env.get("HTTP_PROXY")
        or env.get("http_proxy")
        or network.get("proxy_url", "")
    )

    offline_assets_dir = env.get(OFFLINE_DIR_ENV)
    if offline_assets_dir is None:
        offline_assets_dir = network.get("offline_assets_dir", "")

    return {
        "mirror_prefix": _clean_mirror_prefix(mirror_prefix),
        "proxy_url": _clean_proxy_url(proxy_url),
        "offline_assets_dir": _clean_offline_dir(offline_assets_dir),
    }


def save_network_settings(
    *,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    offline_assets_dir: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Persist mirror, proxy, and offline directory settings in ``user_config.json``."""

    from shared.user_config import update_user_config

    updates: dict[str, str] = {}
    if mirror_prefix is not None:
        updates["mirror_prefix"] = _clean_mirror_prefix(mirror_prefix)
    if proxy_url is not None:
        updates["proxy_url"] = _clean_proxy_url(proxy_url)
    if offline_assets_dir is not None:
        updates["offline_assets_dir"] = _clean_offline_dir(offline_assets_dir)
    if not updates:
        from shared.user_config import load_user_config

        return load_user_config(path)
    config, _ = update_user_config({"network": updates}, path=path)
    return config


def network_env_overrides(
    user_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return environment overrides so child setup jobs inherit mirror, proxy, and offline settings."""

    settings = load_network_settings(user_config=user_config, environ=environ)
    overrides: dict[str, str] = {}
    if settings["mirror_prefix"]:
        overrides[MIRROR_PREFIX_ENV] = settings["mirror_prefix"]
    if settings["proxy_url"]:
        overrides[PROXY_URL_ENV] = settings["proxy_url"]
        overrides["HTTP_PROXY"] = settings["proxy_url"]
        overrides["HTTPS_PROXY"] = settings["proxy_url"]
        overrides["http_proxy"] = settings["proxy_url"]
        overrides["https_proxy"] = settings["proxy_url"]
    if settings["offline_assets_dir"]:
        overrides[OFFLINE_DIR_ENV] = settings["offline_assets_dir"]
    return overrides


def expand_candidate_urls(
    urls: Iterable[str] | str,
    *,
    mirror_prefix: str | None = None,
    allow_mirror: bool = True,
) -> list[str]:
    """Prepend mirror-proxied GitHub URLs before direct fallback URLs."""

    raw_urls = [urls] if isinstance(urls, str) else list(urls or ())
    effective_prefix = (
        _clean_mirror_prefix(mirror_prefix)
        if mirror_prefix is not None
        else load_network_settings()["mirror_prefix"]
    )
    expanded: list[str] = []
    if allow_mirror and effective_prefix:
        for url in raw_urls:
            text = str(url or "").strip()
            if text.startswith(_GITHUB_ORIGIN_PREFIXES) and not text.startswith(effective_prefix):
                proxied = effective_prefix + text
                if proxied not in expanded:
                    expanded.append(proxied)
    for url in raw_urls:
        text = str(url or "").strip()
        if text and text not in expanded:
            expanded.append(text)
    return expanded


def open_url(
    request_or_url: str | urllib.request.Request,
    *,
    timeout: int = 60,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
):
    """Open a URL honoring any configured HTTP/HTTPS proxy without breaking mocked urlopen."""

    effective_proxy = (
        _clean_proxy_url(proxy_url)
        if proxy_url is not None
        else load_network_settings()["proxy_url"]
    )
    if effective_proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": effective_proxy, "https": effective_proxy})
        )
        return opener.open(request_or_url, timeout=timeout)
    return (urlopen_fn or urllib.request.urlopen)(request_or_url, timeout=timeout)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_packs_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or RELEASE_PACKS_MANIFEST
    if not path.is_file():
        return {"version": 1, "packs": {}, "bundles": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def find_local_pack_archive(
    directory: Path | str | None,
    filename: str,
    *,
    expected_sha256: str = "",
) -> Path | None:
    """Return verified single or split archive in ``directory`` if available."""

    if not directory or not filename:
        return None
    root = Path(directory).expanduser()
    if not root.is_dir():
        return None
    direct = root / filename
    if direct.is_file():
        if not expected_sha256 or _sha256_file(direct) == expected_sha256:
            return direct
    part_one = root / f"{filename}.001"
    if part_one.is_file():
        return merge_split_archive(root, filename, expected_sha256=expected_sha256)
    return None


def merge_split_archive(
    directory: Path | str,
    archive_name: str,
    *,
    expected_sha256: str = "",
    output_dir: Path | str | None = None,
) -> Path | None:
    """Stream-merge contiguous ``.001``, ``.002``, ... split volumes and verify SHA-256."""

    src_dir = Path(directory).expanduser().resolve()
    dst_dir = Path(output_dir).expanduser().resolve() if output_dir else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / archive_name

    if target.is_file() and (not expected_sha256 or _sha256_file(target) == expected_sha256):
        return target

    parts: list[Path] = []
    index = 1
    while True:
        candidate = src_dir / f"{archive_name}.{index:03d}"
        if not candidate.is_file():
            break
        parts.append(candidate)
        index += 1
    if not parts:
        return None

    staging = dst_dir / f"{archive_name}.merging"
    try:
        with staging.open("wb") as out_fh:
            for part in parts:
                with part.open("rb") as in_fh:
                    shutil.copyfileobj(in_fh, out_fh, length=16 * 1024 * 1024)
        if expected_sha256:
            actual = _sha256_file(staging)
            if actual != expected_sha256:
                raise ValueError(
                    f"Merged archive checksum mismatch for {archive_name}: expected {expected_sha256}, got {actual}"
                )
        os.replace(staging, target)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return target


def _safe_extract_member(relative_name: str) -> PurePosixPath | None:
    normalized = relative_name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if (
        not posix.parts
        or posix.is_absolute()
        or ".." in posix.parts
        or "\\" in relative_name
        or ":" in relative_name
    ):
        raise ValueError(f"Unsafe archive member path: {relative_name}")
    if posix.parts[0] == "__MACOSX" or any(part.startswith("._") for part in posix.parts):
        return None
    return posix


def extract_modular_pack(
    pack_archive: Path | str,
    asset_root: Path | str,
    *,
    expected_sha256: str = "",
) -> list[str]:
    """Verify and safely extract a modular asset pack into ``asset_root``."""

    archive = Path(pack_archive).expanduser().resolve()
    root = Path(asset_root).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Pack archive not found: {archive}")
    if expected_sha256:
        actual = _sha256_file(archive)
        if actual != expected_sha256:
            raise ValueError(f"Pack SHA256 mismatch for {archive.name}: expected {expected_sha256}, got {actual}")
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            if len(infos) > _PACK_LIMIT_FILES or sum(i.file_size for i in infos) > _PACK_LIMIT_BYTES:
                raise ValueError(f"Pack archive exceeds extraction limits: {archive.name}")
            for info in infos:
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Symlink entries are forbidden in pack archives: {info.filename}")
                posix = _safe_extract_member(info.filename)
                if info.is_dir() or posix is None:
                    continue
                target = root.joinpath(*posix.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".pack-", delete=False
                ) as tmp_fh:
                    tmp_path = Path(tmp_fh.name)
                    shutil.copyfileobj(src, tmp_fh, length=1024 * 1024)
                os.replace(tmp_path, target)
                extracted.append(posix.as_posix())
    else:
        with tarfile.open(archive, "r:*") as tf:
            file_count = 0
            total_bytes = 0
            for member in tf:
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Link/device entries are forbidden in pack archives: {member.name}")
                posix = _safe_extract_member(member.name)
                if not member.isfile() or posix is None:
                    continue
                file_count += 1
                total_bytes += max(0, int(member.size))
                if file_count > _PACK_LIMIT_FILES or total_bytes > _PACK_LIMIT_BYTES:
                    raise ValueError(f"Pack archive exceeds extraction limits: {archive.name}")
                target = root.joinpath(*posix.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".pack-", delete=False
                ) as tmp_fh:
                    tmp_path = Path(tmp_fh.name)
                    shutil.copyfileobj(src, tmp_fh, length=1024 * 1024)
                os.replace(tmp_path, target)
                extracted.append(posix.as_posix())

    # If ORG zips were unpacked as part of the WMA ORG pack, extract them cleanly too.
    wma_dir = root / "bundles" / "wma"
    for org_zip_rel in (
        "bundles/wma/archives/ORG-800FC-100HCP.zip",
        "bundles/wma/archives/ORG-RegAtlas-100HCP.zip",
    ):
        org_zip = root / org_zip_rel
        if org_zip.is_file() and org_zip_rel in extracted:
            from scripts.import_external_resources import extract_org

            extract_org(org_zip, wma_dir)
    return extracted


def download_file_verified(
    urls: Iterable[str] | str,
    destination: Path | str,
    *,
    expected_sha256: str = "",
    expected_size: int = 0,
    timeout: int = 90,
    proxy_url: str | None = None,
    mirror_prefix: str | None = None,
    allow_mirror: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Download ``destination`` with mirror fallback, Range resume, proxy support, and SHA-256 check."""

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and expected_sha256 and _sha256_file(target) == expected_sha256:
        return target

    candidates = expand_candidate_urls(urls, mirror_prefix=mirror_prefix, allow_mirror=allow_mirror)
    if not candidates:
        raise RuntimeError(f"No download URL provided for {target.name}")

    errors: list[str] = []
    part = target.with_name(target.name + ".part")
    for url in candidates:
        try:
            resume_from = part.stat().st_size if part.is_file() else 0
            if expected_size and resume_from > expected_size:
                part.unlink(missing_ok=True)
                resume_from = 0
            headers = {"User-Agent": "SlicerDSpace-downloader"}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"
            req = urllib.request.Request(url, headers=headers)
            with open_url(req, timeout=timeout, proxy_url=proxy_url) as resp:
                status = getattr(resp, "status", 200)
                append = bool(resume_from and status == 206)
                if resume_from and not append:
                    resume_from = 0
                clen = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
                total = (int(clen) + resume_from) if (clen and str(clen).isdigit()) else expected_size
                downloaded = resume_from
                mode = "ab" if append else "wb"
                with part.open(mode) as stream:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, total, url)
            if total and downloaded < total:
                raise RuntimeError(
                    f"Short download for {target.name}: got {downloaded} of {total} bytes (kept .part for resume)"
                )
            if expected_sha256:
                actual = _sha256_file(part)
                if actual != expected_sha256:
                    part.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch for {target.name}: expected {expected_sha256}, got {actual}")
            os.replace(part, target)
            return target
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                part.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Failed to download {target.name}: {errors}")
