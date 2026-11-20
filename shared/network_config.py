"""Network mirror, HTTP/SOCKS proxy, split-volume merge, and offline pack helpers."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

_ORIG_URLLIB_URLOPEN = urllib.request.urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_PACKS_MANIFEST = REPO_ROOT / "models" / "release_packs_manifest.json"

MIRROR_PREFIX_ENV = "SLICERDSPACE_MIRROR_PREFIX"
PROXY_URL_ENV = "SLICERDSPACE_PROXY_URL"
OFFLINE_DIR_ENV = "SLICERDSPACE_OFFLINE_DIR"

PRESET_MIRRORS: tuple[tuple[str, str, str, str], ...] = (
    ("direct", "", "GitHub Official (Direct)", "GitHub 官方直连"),
    ("ghfast", "https://ghfast.top/", "Mirror 1 (ghfast.top)", "加速镜像 1 (ghfast.top)"),
    ("ghproxy", "https://gh-proxy.com/", "Mirror 2 (gh-proxy.com)", "加速镜像 2 (gh-proxy.com)"),
)

_GITHUB_ORIGIN_PREFIXES = (
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://objects.githubusercontent.com/",
    "https://media.githubusercontent.com/",
    "https://codeload.github.com/",
)
_DIRECT_MIRROR_HOSTS = ("ghfast.top", "gh-proxy.com")
_PAR_CHUNK = 8 * 1024 * 1024
_PAR_WORKERS = 12
_MIRROR_WORKERS = 3
_DOWNLOAD_UA = "SlicerDSpace-downloader"

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
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw and "=" in raw:
        mapping: dict[str, str] = {}
        for item in raw.split(";"):
            part = item.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k_clean = k.strip().lower()
                v_clean = v.strip()
                if k_clean and v_clean:
                    mapping[k_clean] = v_clean
        raw = mapping.get("https") or mapping.get("http") or next(iter(mapping.values()), "")
        if not raw:
            return ""
    if "://" not in raw:
        return f"http://{raw}"
    return raw


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


def _detect_system_proxy() -> str:
    try:
        proxies = urllib.request.getproxies() or {}
        found = _clean_proxy_url(proxies.get("https") or proxies.get("http") or proxies.get("all") or "")
        if found:
            return found
    except Exception:
        pass
    if sys.platform == "darwin" and shutil.which("scutil"):
        try:
            proc = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=3)
            if proc.returncode == 0:
                kv: dict[str, str] = {}
                for line in (proc.stdout or "").splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        kv[k.strip()] = v.strip()
                if kv.get("HTTPSEnable") == "1" and kv.get("HTTPSProxy") and kv.get("HTTPSPort"):
                    return _clean_proxy_url(f"{kv['HTTPSProxy']}:{kv['HTTPSPort']}")
                if kv.get("HTTPEnable") == "1" and kv.get("HTTPProxy") and kv.get("HTTPPort"):
                    return _clean_proxy_url(f"{kv['HTTPProxy']}:{kv['HTTPPort']}")
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import winreg

            key_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                if int(enabled or 0) == 1 and server:
                    return _clean_proxy_url(server)
        except Exception:
            pass
    return ""


def _is_loopback_url(request_or_url: str | urllib.request.Request) -> bool:
    raw = str(getattr(request_or_url, "full_url", request_or_url) or "")
    try:
        host = (urllib.parse.urlparse(raw).hostname or "").lower()
    except Exception:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def _host_of(request_or_url: str | urllib.request.Request) -> str:
    raw = str(getattr(request_or_url, "full_url", request_or_url) or "")
    try:
        return (urllib.parse.urlparse(raw).hostname or "").lower()
    except Exception:
        return ""


def _is_mirror_host(request_or_url: str | urllib.request.Request) -> bool:
    host = _host_of(request_or_url)
    return any(host == d or host.endswith("." + d) for d in _DIRECT_MIRROR_HOSTS)


def _worker_count_for(url: str) -> int:
    return _MIRROR_WORKERS if _is_mirror_host(url) else _PAR_WORKERS


def _ssl_cafile() -> str:
    for env_key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        ca_file = (os.environ.get(env_key) or "").strip()
        if ca_file and Path(ca_file).is_file():
            return ca_file
    try:
        import certifi

        ca_file = certifi.where()
        if ca_file and Path(ca_file).is_file():
            return ca_file
    except Exception:
        pass
    return ""


def _ssl_context() -> ssl.SSLContext:
    ca_file = _ssl_cafile()
    if ca_file:
        try:
            return ssl.create_default_context(cafile=ca_file)
        except Exception:
            pass
    return ssl.create_default_context()


def _load_explicit_proxy() -> str:
    env_explicit = _clean_proxy_url(os.environ.get(PROXY_URL_ENV) or "")
    if env_explicit:
        return env_explicit
    try:
        from shared.user_config import load_user_config

        cfg = load_user_config()
    except Exception:
        cfg = {}
    network = cfg.get("network") if isinstance(cfg, Mapping) else {}
    if not isinstance(network, Mapping):
        return ""
    return _clean_proxy_url(network.get("proxy_url") or "")


def _resolve_effective_proxy(
    request_or_url: str | urllib.request.Request,
    *,
    proxy_url: str | None = None,
    bypass_system_proxy: bool = False,
    allow_system_detect: bool = True,
) -> str:
    if proxy_url is not None:
        effective_proxy = _clean_proxy_url(proxy_url)
    elif bypass_system_proxy:
        effective_proxy = _load_explicit_proxy()
    else:
        effective_proxy = load_network_settings()["proxy_url"]
        if (
            not effective_proxy
            and allow_system_detect
            and not _is_loopback_url(request_or_url)
        ):
            effective_proxy = _detect_system_proxy()
    if effective_proxy.lower().startswith("socks"):
        raise RuntimeError(
            f"SOCKS proxy {effective_proxy!r} is not supported without PySocks; "
            "please specify an HTTP/HTTPS proxy URL (e.g. http://127.0.0.1:7890)."
        )
    return effective_proxy


def open_url(
    request_or_url: str | urllib.request.Request,
    *,
    timeout: int = 60,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
    bypass_system_proxy: bool = False,
):
    """Open a URL honoring configured/system HTTP(S) proxy and CA bundle without breaking mocked urlopen."""

    effective_proxy = _resolve_effective_proxy(
        request_or_url,
        proxy_url=proxy_url,
        bypass_system_proxy=bypass_system_proxy,
        allow_system_detect=(urlopen_fn is None),
    )
    if effective_proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": effective_proxy, "https": effective_proxy}),
            urllib.request.HTTPSHandler(context=_ssl_context()),
        )
        return opener.open(request_or_url, timeout=timeout)
    if urlopen_fn is not None:
        return urlopen_fn(request_or_url, timeout=timeout)
    if bypass_system_proxy and urllib.request.urlopen is _ORIG_URLLIB_URLOPEN:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=_ssl_context()),
        )
        return opener.open(request_or_url, timeout=timeout)
    return urllib.request.urlopen(request_or_url, timeout=timeout, context=_ssl_context())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
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


class _RequestsResponseAdapter:
    """Uniform streaming response wrapper around ``requests.Response``."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.status = int(getattr(response, "status_code", 200))
        self.headers = response.headers
        self.url = str(getattr(response, "url", "") or "")
        self._iter = None

    def __enter__(self) -> "_RequestsResponseAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self._response.content
        if self._iter is None:
            self._iter = self._response.iter_content(chunk_size=max(65536, int(n)))
        try:
            return next(self._iter)
        except StopIteration:
            return b""

    def iter_chunks(self, chunk_size: int = 262144):
        for chunk in self._response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk


def _make_http_session(proxy: str, pool_size: int = 16):
    if urllib.request.urlopen is not _ORIG_URLLIB_URLOPEN:
        return None
    try:
        import requests
        from requests.adapters import HTTPAdapter

        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.trust_env = False
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}
        sess.verify = _ssl_cafile() or True
        sess.headers.update({"User-Agent": _DOWNLOAD_UA})
        return sess
    except Exception:
        return None


def _http_get_stream(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: int = 90,
    proxy_url: str | None = None,
    session: Any = None,
    bypass_system_proxy: bool = False,
):
    merged_headers = {"User-Agent": _DOWNLOAD_UA, "Accept-Encoding": "identity"}
    if headers:
        merged_headers.update(headers)
    if session is not None:
        resp = session.get(url, headers=merged_headers, stream=True, timeout=timeout, allow_redirects=True)
        if resp.status_code >= 400:
            code, reason = int(resp.status_code), str(getattr(resp, "reason", "") or "")
            raw_headers = dict(resp.headers or {})
            resp.close()
            raise urllib.error.HTTPError(url, code, reason, raw_headers, None)
        return _RequestsResponseAdapter(resp)
    req = urllib.request.Request(url, headers=merged_headers)
    return open_url(req, timeout=timeout, proxy_url=proxy_url, bypass_system_proxy=bypass_system_proxy)


def _probe_range(
    url: str,
    *,
    timeout: int = 60,
    proxy_url: str | None = None,
    session: Any = None,
    bypass_system_proxy: bool = False,
) -> tuple[bool, int, str]:
    try:
        with _http_get_stream(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            timeout=timeout,
            proxy_url=proxy_url,
            session=session,
            bypass_system_proxy=bypass_system_proxy,
        ) as resp:
            status = getattr(resp, "status", 200)
            resolved = str(getattr(resp, "url", "") or "") or (resp.geturl() if hasattr(resp, "geturl") else "")
            content_range = (resp.headers.get("Content-Range") or "") if hasattr(resp, "headers") else ""
            total = 0
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[1].strip()
                if tail.isdigit():
                    total = int(tail)
            return (status == 206 and total > 0), total, resolved
    except Exception:
        return False, 0, ""


def _download_attempts_for(candidates: list[str], proxy_url: str | None) -> list[tuple[str, bool]]:
    explicit = _clean_proxy_url(proxy_url) if proxy_url is not None else _load_explicit_proxy()
    ambient_proxy = "" if explicit else (load_network_settings()["proxy_url"] or _detect_system_proxy())
    attempts: list[tuple[str, bool]] = []
    for url in candidates:
        if _is_mirror_host(url) and not explicit and ambient_proxy and not _is_loopback_url(url):
            attempts.append((url, True))
            attempts.append((url, False))
        else:
            attempts.append((url, False))
    return attempts


def _download_parallel_verified(
    url: str,
    target: Path,
    part: Path,
    total_size: int,
    *,
    expected_sha256: str = "",
    timeout: int = 90,
    proxy_url: str | None = None,
    resolved_url: str = "",
    session: Any = None,
    bypass_system_proxy: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    meta = target.with_name(target.name + ".part.json")
    chunk_sz = _PAR_CHUNK
    n_chunks = (total_size + chunk_sz - 1) // chunk_sz
    done_flags = [False] * n_chunks

    if meta.is_file() and part.is_file() and part.stat().st_size == total_size:
        try:
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if int(saved.get("size") or 0) == total_size and int(saved.get("chunk_size") or 0) == chunk_sz:
                raw_done = saved.get("done") or ()
                if len(raw_done) == n_chunks:
                    done_flags = [bool(x) for x in raw_done]
        except Exception:
            pass
    elif part.is_file() and not meta.is_file():
        legacy_sz = part.stat().st_size
        if 0 < legacy_sz < total_size:
            full = min(n_chunks, legacy_sz // chunk_sz)
            for idx in range(full):
                done_flags[idx] = True
        else:
            part.unlink(missing_ok=True)

    todo = [idx for idx, ok in enumerate(done_flags) if not ok]
    done_bytes = sum(min(chunk_sz, total_size - idx * chunk_sz) for idx, ok in enumerate(done_flags) if ok)
    mode = "r+b" if part.is_file() else "wb"
    with part.open(mode) as pre:
        if pre.seek(0, os.SEEK_END) != total_size:
            pre.truncate(total_size)

    if todo:
        workers = max(1, min(_worker_count_for(url), len(todo)))
        url_lock = threading.Lock()
        state_lock = threading.Lock()
        write_lock = threading.Lock()
        abort_event = threading.Event()
        active_url = [resolved_url if (resolved_url and not _is_mirror_host(url)) else url]
        state = {"bytes": done_bytes}

        def _save_meta() -> None:
            tmp_meta = meta.with_name(meta.name + ".tmp")
            tmp_meta.write_text(
                json.dumps({"size": total_size, "chunk_size": chunk_sz, "done": done_flags}),
                encoding="utf-8",
            )
            os.replace(tmp_meta, meta)

        _save_meta()
        with part.open("r+b") as fh:
            def _fetch_chunk(idx: int) -> None:
                if abort_event.is_set():
                    return
                start = idx * chunk_sz
                end = min(total_size - 1, start + chunk_sz - 1)
                expected_len = end - start + 1
                last_exc: Exception | None = None
                for attempt in range(4):
                    if abort_event.is_set():
                        return
                    buf = bytearray()
                    with url_lock:
                        fetch_url = active_url[0]
                    try:
                        with _http_get_stream(
                            fetch_url,
                            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                            timeout=timeout,
                            proxy_url=proxy_url,
                            session=session,
                            bypass_system_proxy=bypass_system_proxy,
                        ) as resp:
                            status = getattr(resp, "status", 200)
                            if status != 206 and not (status == 200 and start == 0 and expected_len == total_size):
                                raise RuntimeError(f"Expected 206 for range {start}-{end}, got HTTP {status}")
                            if hasattr(resp, "iter_chunks"):
                                for blk in resp.iter_chunks(262144):
                                    if abort_event.is_set():
                                        return
                                    buf.extend(blk)
                            else:
                                while len(buf) < expected_len:
                                    if abort_event.is_set():
                                        return
                                    blk = resp.read(min(262144, expected_len - len(buf)))
                                    if not blk:
                                        break
                                    buf.extend(blk)
                        if abort_event.is_set():
                            return
                        if len(buf) != expected_len:
                            raise RuntimeError(f"Chunk {idx} short read {len(buf)}/{expected_len}")
                        with write_lock:
                            if abort_event.is_set():
                                return
                            fh.seek(start)
                            fh.write(buf)
                        with state_lock:
                            if abort_event.is_set():
                                return
                            done_flags[idx] = True
                            state["bytes"] += expected_len
                            _save_meta()
                            if progress_callback is not None:
                                progress_callback(state["bytes"], total_size, url)
                        return
                    except urllib.error.HTTPError as exc:
                        last_exc = exc
                        if exc.code in (403, 404, 410) and fetch_url != url:
                            _, _, fresh = _probe_range(
                                url,
                                timeout=min(60, timeout),
                                proxy_url=proxy_url,
                                session=session,
                                bypass_system_proxy=bypass_system_proxy,
                            )
                            with url_lock:
                                active_url[0] = fresh if (fresh and not _is_mirror_host(url)) else url
                        elif exc.code in (429, 503, 504):
                            if abort_event.wait(0.5 * (2 ** attempt)):
                                return
                    except Exception as exc:
                        last_exc = exc
                        if fetch_url != url:
                            with url_lock:
                                active_url[0] = url
                    if abort_event.wait(0.35 * (attempt + 1)):
                        return
                abort_event.set()
                raise RuntimeError(f"Chunk {idx} failed after retries: {last_exc}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_fetch_chunk, idx) for idx in todo]
                try:
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()
                except Exception:
                    abort_event.set()
                    for pending in futures:
                        pending.cancel()
                    raise

    if part.stat().st_size != total_size:
        raise RuntimeError(f"Short download for {target.name}: got {part.stat().st_size} of {total_size} bytes")
    if expected_sha256:
        actual = _sha256_file(part)
        if actual != expected_sha256:
            part.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {target.name}: expected {expected_sha256}, got {actual}")
    meta.unlink(missing_ok=True)
    os.replace(part, target)
    return target


def _download_streaming_verified(
    url: str,
    target: Path,
    part: Path,
    *,
    expected_sha256: str = "",
    expected_size: int = 0,
    timeout: int = 90,
    proxy_url: str | None = None,
    session: Any = None,
    bypass_system_proxy: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    resume_from = part.stat().st_size if part.is_file() else 0
    if expected_size and resume_from > expected_size:
        part.unlink(missing_ok=True)
        resume_from = 0
    headers = {"Accept-Encoding": "identity"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    with _http_get_stream(
        url,
        headers=headers,
        timeout=timeout,
        proxy_url=proxy_url,
        session=session,
        bypass_system_proxy=bypass_system_proxy,
    ) as resp:
        status = getattr(resp, "status", 200)
        append = bool(resume_from and status == 206)
        if resume_from and not append:
            resume_from = 0
        clen = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
        total = (int(clen) + resume_from) if (clen and str(clen).isdigit()) else expected_size
        downloaded = resume_from
        mode = "ab" if append else "wb"
        with part.open(mode) as stream:
            if hasattr(resp, "iter_chunks"):
                for chunk in resp.iter_chunks(262144):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, total, url)
            else:
                while True:
                    chunk = resp.read(262144)
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
    """Download ``destination`` with Keep-Alive, parallel Range chunks, mirror fallback, and SHA-256 check."""

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and expected_sha256 and _sha256_file(target) == expected_sha256:
        return target

    candidates = expand_candidate_urls(urls, mirror_prefix=mirror_prefix, allow_mirror=allow_mirror)
    if not candidates:
        raise RuntimeError(f"No download URL provided for {target.name}")

    errors: list[str] = []
    part = target.with_name(target.name + ".part")
    for url, bypass in _download_attempts_for(candidates, proxy_url):
        session = None
        try:
            proxy = _resolve_effective_proxy(
                url,
                proxy_url=proxy_url,
                bypass_system_proxy=bypass,
                allow_system_detect=True,
            )
            session = _make_http_session(proxy, pool_size=max(8, _worker_count_for(url) + 4))
            ok_range = False
            ptot = 0
            resolved = ""
            if session is not None or expected_size >= _PAR_CHUNK * 2 or expected_size == 0:
                ok_range, ptot, resolved = _probe_range(
                    url,
                    timeout=min(60, timeout),
                    proxy_url=proxy_url,
                    session=session,
                    bypass_system_proxy=bypass,
                )
            total_par = ptot if ok_range else 0
            if total_par >= _PAR_CHUNK * 2:
                return _download_parallel_verified(
                    url,
                    target,
                    part,
                    total_par,
                    expected_sha256=expected_sha256,
                    timeout=timeout,
                    proxy_url=proxy_url,
                    resolved_url=resolved,
                    session=session,
                    bypass_system_proxy=bypass,
                    progress_callback=progress_callback,
                )
            return _download_streaming_verified(
                url,
                target,
                part,
                expected_sha256=expected_sha256,
                expected_size=expected_size or ptot,
                timeout=timeout,
                proxy_url=proxy_url,
                session=session,
                bypass_system_proxy=bypass,
                progress_callback=progress_callback,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                part.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
    raise RuntimeError(f"Failed to download {target.name}: {errors}")
