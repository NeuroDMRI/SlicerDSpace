"""Generate OS-specific custom installer scripts (.bat / .command / .sh) for SlicerDSpace."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Iterable

from shared.network_config import RELEASE_PACKS_MANIFEST, load_release_packs_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_ARTIFACTS_JSON = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
RELEASE_TAG = "v0.1.1"
RELEASE_BASE_URL = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{RELEASE_TAG}"

SUPPORTED_OS = ("windows", "macos", "linux")
SUPPORTED_MODES = ("base", "modular", "all_in_one")
PACK_ORDER = ("wma_org", "wma_naba", "wma_east_west")

DEFAULT_IIT_TEMPLATE: dict[str, Any] = {
    "relpath": "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz",
    "sha256": "2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead",
    "size_bytes": 7830929,
    "urls": ["https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz"],
}

_BOOTSTRAP_BODY = textwrap.dedent(
    r'''
    import hashlib
    import json
    import os
    import platform
    import shutil
    import socket
    import ssl
    import subprocess
    import sys
    import tempfile
    import urllib.error
    import urllib.request
    from pathlib import Path

    _GITHUB_PREFIXES = (
        "https://github.com/",
        "https://raw.githubusercontent.com/",
        "https://objects.githubusercontent.com/",
        "https://media.githubusercontent.com/",
        "https://codeload.github.com/",
    )

    def _log(msg):
        print(f"[SlicerDSpace Installer] {msg}", flush=True)

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1048576), b""):
                h.update(chunk)
        return h.hexdigest()

    def _clean_mirror(prefix: str) -> str:
        prefix = str(prefix or "").strip()
        if not prefix or not (prefix.startswith("https://") or prefix.startswith("http://")):
            return ""
        return prefix.rstrip("/") + "/"

    def _expand_urls(urls, *, allow_mirror=True):
        raw = [urls] if isinstance(urls, str) else list(urls or ())
        mirror = _clean_mirror(CONFIG.get("mirror_prefix", ""))
        out = []
        if allow_mirror and mirror:
            for u in raw:
                if u.startswith(_GITHUB_PREFIXES) and not u.startswith(mirror):
                    m = mirror + u
                    if m not in out:
                        out.append(m)
        for u in raw:
            if u and u not in out:
                out.append(u)
        return out

    def _clean_proxy(raw):
        s = str(raw or "").strip()
        if not s:
            return ""
        if "://" not in s and "=" in s:
            mapping = {}
            for item in s.split(";"):
                part = item.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k.strip() and v.strip():
                        mapping[k.strip().lower()] = v.strip()
            s = mapping.get("https") or mapping.get("http") or next(iter(mapping.values()), "")
            if not s:
                return ""
        return s if "://" in s else ("http" + "://" + s)

    def _ssl_context():
        for env_k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            cf = str(os.environ.get(env_k) or "").strip()
            if cf and Path(cf).is_file():
                try:
                    return ssl.create_default_context(cafile=cf)
                except Exception:
                    pass
        try:
            import certifi

            cf = certifi.where()
            if cf and Path(cf).is_file():
                return ssl.create_default_context(cafile=cf)
        except Exception:
            pass
        return ssl.create_default_context()

    def _detect_system_proxy():
        try:
            proxies = urllib.request.getproxies() or {}
            found = _clean_proxy(proxies.get("https") or proxies.get("http") or proxies.get("all") or "")
            if found:
                return found
        except Exception:
            pass
        if sys.platform == "darwin" and shutil.which("scutil"):
            try:
                proc = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=3)
                if proc.returncode == 0:
                    kv = {}
                    for ln in (proc.stdout or "").splitlines():
                        if ":" in ln:
                            k, v = ln.split(":", 1)
                            kv[k.strip()] = v.strip()
                    if kv.get("HTTPSEnable") == "1" and kv.get("HTTPSProxy") and kv.get("HTTPSPort"):
                        return _clean_proxy(f"{kv['HTTPSProxy']}:{kv['HTTPSPort']}")
                    if kv.get("HTTPEnable") == "1" and kv.get("HTTPProxy") and kv.get("HTTPPort"):
                        return _clean_proxy(f"{kv['HTTPProxy']}:{kv['HTTPPort']}")
            except Exception:
                pass
        elif sys.platform == "win32":
            try:
                import winreg

                key_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as rk:
                    enabled, _ = winreg.QueryValueEx(rk, "ProxyEnable")
                    server, _ = winreg.QueryValueEx(rk, "ProxyServer")
                    if int(enabled or 0) == 1 and server:
                        return _clean_proxy(server)
            except Exception:
                pass
        return ""

    def _is_loopback_url(req):
        u = str(getattr(req, "full_url", req) or "").lower()
        host = u.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0] if "://" in u else ""
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _is_intel_mac():
        if sys.platform != "darwin":
            return False
        if str(platform.machine() or "").lower() not in ("x86_64", "amd64"):
            return False
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "sysctl.proc_translated"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            if out == "1":
                return False
        except Exception:
            pass
        return True

    def _pick_cache_dir(need_bytes=0):
        cands = [Path(tempfile.gettempdir()) / "slicerdspace-installer-cache"]
        if sys.platform == "win32":
            fb = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SlicerDSpace" / "installer-cache"
        else:
            fb = Path.home() / ".cache" / "slicerdspace-installer-cache"
        if fb not in cands:
            cands.append(fb)
        min_pref = max(int(need_bytes or 0), 2147483648)
        best = None
        best_free = -1
        for cand in cands:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                free = int(shutil.disk_usage(cand).free)
                if free >= min_pref:
                    return cand
                if free > best_free:
                    best, best_free = cand, free
            except OSError:
                pass
        if best is None:
            raise OSError("No writable installer cache directory available")
        if need_bytes and best_free < int(need_bytes):
            raise RuntimeError(
                f"Insufficient disk space in {best}: {best_free / 1073741824.0:.2f} GB free, "
                f"need {need_bytes / 1073741824.0:.2f} GB"
            )
        return best

    def _open_url(req, timeout=120):
        explicit = _clean_proxy(CONFIG.get("proxy_url", ""))
        proxy = explicit or ("" if _is_loopback_url(req) else _detect_system_proxy())
        if proxy.lower().startswith("socks"):
            raise RuntimeError(f"SOCKS proxy {proxy!r} is not supported without PySocks; use an HTTP/HTTPS proxy URL.")
        ctx = _ssl_context()
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx),
            )
            return opener.open(req, timeout=timeout)
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)

    def _download(urls, dest: Path, expected_sha256="", expected_size=0, allow_mirror=True):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            if expected_sha256 and _sha256(dest) == expected_sha256:
                _log(f"Reusing verified file: {dest.name}")
                return dest
            if not expected_sha256 and expected_size and dest.stat().st_size == expected_size:
                _log(f"Reusing file: {dest.name}")
                return dest
        part = dest.with_name(dest.name + ".part")
        errors = []
        for url in _expand_urls(urls, allow_mirror=allow_mirror):
            try:
                _log(f"Downloading {dest.name} <- {url}")
                resume_from = part.stat().st_size if part.is_file() else 0
                if expected_size and resume_from > expected_size:
                    part.unlink(missing_ok=True)
                    resume_from = 0
                headers = {"User-Agent": "SlicerDSpace-custom-installer"}
                if resume_from:
                    headers["Range"] = f"bytes={resume_from}-"
                req = urllib.request.Request(url, headers=headers)
                try:
                    resp_cm = _open_url(req, timeout=120)
                except urllib.error.HTTPError as range_exc:
                    if resume_from and range_exc.code in (400, 405, 501):
                        part.unlink(missing_ok=True)
                        resume_from = 0
                        req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-custom-installer"})
                        resp_cm = _open_url(req, timeout=120)
                    else:
                        raise
                with resp_cm as resp:
                    status = getattr(resp, "status", 200)
                    append = bool(resume_from and status == 206)
                    if resume_from and not append:
                        resume_from = 0
                    clen = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
                    total = (int(clen) + resume_from) if (clen and str(clen).isdigit()) else int(expected_size or 0)
                    done = resume_from
                    next_log = done + 32 * 1024 * 1024
                    with part.open("ab" if append else "wb") as fh:
                        while True:
                            chunk = resp.read(1048576)
                            if not chunk:
                                break
                            fh.write(chunk)
                            done += len(chunk)
                            if total and done >= next_log:
                                pct = 100.0 * done / max(1, total)
                                _log(f"  {dest.name}: {done / 1048576:.1f} / {total / 1048576:.1f} MB ({pct:.1f}%)")
                                next_log = done + 32 * 1024 * 1024
                if total and done < total:
                    raise RuntimeError(f"Short read {done}/{total} bytes")
                if expected_sha256:
                    actual = _sha256(part)
                    if actual != expected_sha256:
                        part.unlink(missing_ok=True)
                        raise ValueError(f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}")
                os.replace(part, dest)
                _log(f"Verified {dest.name}")
                return dest
            except urllib.error.HTTPError as exc:
                if exc.code == 416:
                    part.unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        raise RuntimeError(f"Unable to download {dest.name}: {errors}")

    def _search_dirs():
        dirs = []
        raw_offline = str(CONFIG.get("offline_dir", "") or "").strip()
        if raw_offline:
            p = Path(raw_offline).expanduser()
            if p.is_dir():
                dirs.append(p.resolve())
        if CONFIG.get("check_script_dir", True):
            here = Path(os.environ.get("SLICERDSPACE_SCRIPT_DIR") or Path(CONFIG["__script_file__"]).parent).resolve()
            if here.is_dir() and here not in dirs:
                dirs.append(here)
        return dirs

    def _acquire_spec(spec, cache_dir: Path):
        filename = spec["filename"]
        expected_sha = str(spec.get("sha256") or "")
        expected_size = int(spec.get("size_bytes") or 0)
        parts = spec.get("parts") or []
        dirs = _search_dirs()

        for d in dirs:
            cand = d / filename
            if cand.is_file() and (not expected_sha or _sha256(cand) == expected_sha):
                _log(f"Found verified offline archive: {cand}")
                return cand, []

        cached = cache_dir / filename
        if cached.is_file() and expected_sha and _sha256(cached) == expected_sha:
            _log(f"Reusing verified cache: {cached}")
            return cached, [cached]

        if parts:
            part_paths = []
            downloaded_parts = []
            for pspec in parts:
                pname = pspec["filename"]
                psha = str(pspec.get("sha256") or "")
                psize = int(pspec.get("size_bytes") or 0)
                found = None
                for d in dirs:
                    cand = d / pname
                    if cand.is_file() and (not psha or _sha256(cand) == psha):
                        _log(f"Using offline split part: {cand}")
                        found = cand
                        break
                if found is None:
                    found = _download(
                        pspec["url"],
                        cache_dir / pname,
                        expected_sha256=psha,
                        expected_size=psize,
                    )
                    downloaded_parts.append(found)
                part_paths.append(found)
            _log(f"Merging {len(part_paths)} volumes into {filename}...")
            merging = cache_dir / (filename + ".merging")
            with merging.open("wb") as out_fh:
                for pp in part_paths:
                    with pp.open("rb") as in_fh:
                        shutil.copyfileobj(in_fh, out_fh, length=16 * 1024 * 1024)
            if expected_sha and _sha256(merging) != expected_sha:
                merging.unlink(missing_ok=True)
                raise RuntimeError(f"Merged archive SHA-256 mismatch: {filename}")
            os.replace(merging, cached)
            _log(f"Merged and verified {filename}")
            return cached, [*downloaded_parts, cached]

        if not spec.get("url"):
            raise RuntimeError(f"Download URL is not configured yet for {filename}")
        archive = _download(
            spec["url"],
            cached,
            expected_sha256=expected_sha,
            expected_size=expected_size,
        )
        return archive, [archive]

    def _cleanup_temp_files(paths):
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    def _strip_quarantine_macos(path: Path):
        if sys.platform == "darwin" and shutil.which("xattr"):
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(path)], capture_output=True)

    def _install_extension_in_slicer(slicer_exe: str, archive_path: Path):
        _strip_quarantine_macos(archive_path)
        receipt = Path(tempfile.mkdtemp(prefix="slicerdspace-inst-")) / "receipt.json"
        helper = receipt.with_name("install_ext.py")
        helper.write_text(
            "import json, sys, traceback, qt, slicer\n"
            "from pathlib import Path\n"
            "def _run():\n"
            "    archive = sys.argv[1]\n"
            "    receipt = Path(sys.argv[2])\n"
            "    try:\n"
            "        em = slicer.app.extensionsManagerModel()\n"
            "        em.setInteractive(False)\n"
            "        meta = {'extensionname': 'SlicerDSpace', 'depends': 'NA', 'slicer_revision': str(em.slicerRevision), 'os': str(em.slicerOs), 'arch': str(em.slicerArch)}\n"
            "        ok = bool(em.installExtension('SlicerDSpace', meta, archive, False, True))\n"
            "        paths = [str(p) for p in em.extensionModulePaths('SlicerDSpace')]\n"
            "        receipt.write_text(json.dumps({'installed': ok, 'module_paths': paths}), encoding='utf-8')\n"
            "        slicer.app.exit(0 if ok else 1)\n"
            "    except Exception as exc:\n"
            "        receipt.write_text(json.dumps({'installed': False, 'error': str(exc), 'traceback': traceback.format_exc()}), encoding='utf-8')\n"
            "        slicer.app.exit(1)\n"
            "qt.QTimer.singleShot(500, _run)\n",
            encoding="utf-8",
        )
        cmd = [slicer_exe, "--no-splash", "--no-main-window", "--python-script", str(helper), str(archive_path), str(receipt)]
        env = os.environ.copy()
        if sys.platform.startswith("linux") and not env.get("DISPLAY"):
            xvfb = shutil.which("xvfb-run")
            if xvfb:
                cmd = [xvfb, "-a", *cmd]
            else:
                env.setdefault("QT_QPA_PLATFORM", "offscreen")
        _log("Installing SlicerDSpace base extension into 3D Slicer...")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0 or not receipt.is_file():
            raise RuntimeError(f"Slicer extension installation failed with exit code {proc.returncode}")
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if not data.get("installed"):
            raise RuntimeError(f"Slicer ExtensionManager rejected archive: {archive_path}")
        shutil.rmtree(receipt.parent, ignore_errors=True)
        return data

    def _find_module_root(slicer_exe: str, receipt_data: dict) -> Path:
        for mp in receipt_data.get("module_paths") or ():
            cand = Path(mp)
            if (cand / "SlicerDSpace.py").is_file():
                return cand.resolve()
        slicer_path = Path(slicer_exe).resolve()
        candidates = []
        if sys.platform == "darwin":
            home = Path.home()
            candidates.extend(slicer_path.parents[1].glob("Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))
            candidates.extend((home / "Library/Application Support/slicer.org").glob("Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))
        else:
            candidates.extend(slicer_path.parent.glob("slicer.org/Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))
        for c in candidates:
            if (c / "SlicerDSpace.py").is_file():
                return c.resolve()
        raise FileNotFoundError("Could not locate installed SlicerDSpace module directory")

    def _sync_iit_template(iit_spec: dict, work_dir: Path, bundled_assets: Path, asset_root: Path):
        if not iit_spec:
            return
        relpath = str(iit_spec.get("relpath") or "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz")
        expected_sha = str(iit_spec.get("sha256") or "")
        expected_size = int(iit_spec.get("size_bytes") or 0)
        targets = []
        if bundled_assets.is_dir():
            targets.append(bundled_assets / relpath)
        targets.append(asset_root / relpath)
        if all(t.is_file() and _sha256(t) == expected_sha for t in targets):
            return

        source_file = None
        for existing in targets:
            if existing.is_file() and _sha256(existing) == expected_sha:
                source_file = existing
                break
        if source_file is None:
            for d in _search_dirs():
                for cand in (d / relpath, d / Path(relpath).name):
                    if cand.is_file() and _sha256(cand) == expected_sha:
                        _log(f"Using local NITRC IIT template: {cand}")
                        source_file = cand
                        break
                if source_file is not None:
                    break
        temp_dl = None
        if source_file is None:
            try:
                temp_dl = work_dir / Path(relpath).name
                source_file = _download(
                    iit_spec.get("urls") or (),
                    temp_dl,
                    expected_sha256=expected_sha,
                    expected_size=expected_size,
                    allow_mirror=False,
                )
            except Exception as exc:
                _log(f"[WARN] Could not fetch optional NITRC IIT template ({exc}); PNL mask will fetch or prompt on first run.")
                return
        try:
            for dst in targets:
                if dst.is_file() and _sha256(dst) == expected_sha:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dst)
        finally:
            if temp_dl is not None:
                temp_dl.unlink(missing_ok=True)

    def main() -> int:
        slicer_exe = str(os.environ.get("SLICERDSPACE_RESOLVED_SLICER") or CONFIG.get("slicer_path") or "").strip()
        if not slicer_exe or not Path(slicer_exe).exists():
            raise FileNotFoundError(f"3D Slicer executable not found: {slicer_exe}")

        os_name = CONFIG["os"]
        if _is_intel_mac():
            raise RuntimeError("Intel x86_64 Macs are not supported by the macOS bundle; Apple Silicon (M1-M4) is required.")
        artifacts = CONFIG["artifacts"]
        bundle_key = {
            "windows": "windows_x64_base",
            "linux": "linux_x64_base",
            "macos": "macos_arm64_base",
        }[os_name]
        ext_spec = artifacts["bundles"][bundle_key]
        need_b = int(ext_spec.get("size_bytes") or 0) + int((artifacts.get("iit_template") or {}).get("size_bytes") or 0)
        work_dir = _pick_cache_dir(int(need_b * 1.15) + 536870912 if need_b else 0)
        ext_archive, ext_temps = _acquire_spec(ext_spec, work_dir)
        receipt = _install_extension_in_slicer(slicer_exe, ext_archive)
        _cleanup_temp_files(ext_temps)

        module_root = _find_module_root(slicer_exe, receipt)
        _log(f"SlicerDSpace extension installed at: {module_root}")

        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))

        from shared.network_config import extract_modular_pack, save_network_settings
        from shared.model_assets import configured_model_assets_root
        from shared.user_config import load_user_config, write_user_config

        save_network_settings(
            mirror_prefix=CONFIG.get("mirror_prefix", ""),
            proxy_url=CONFIG.get("proxy_url", ""),
            offline_assets_dir=CONFIG.get("offline_dir", ""),
        )

        if CONFIG.get("mirror_prefix"):
            os.environ["SLICERDSPACE_MIRROR_PREFIX"] = _clean_mirror(CONFIG["mirror_prefix"])
        if CONFIG.get("proxy_url"):
            os.environ["SLICERDSPACE_PROXY_URL"] = str(CONFIG["proxy_url"]).strip()
            os.environ["HTTPS_PROXY"] = str(CONFIG["proxy_url"]).strip()
            os.environ["HTTP_PROXY"] = str(CONFIG["proxy_url"]).strip()
        if CONFIG.get("offline_dir"):
            os.environ["SLICERDSPACE_OFFLINE_DIR"] = str(CONFIG["offline_dir"]).strip()

        custom_assets_dir = str(CONFIG.get("model_assets_dir") or "").strip()
        asset_root = configured_model_assets_root(custom_assets_dir or None)
        asset_root.mkdir(parents=True, exist_ok=True)
        bundled_assets = module_root.parents[2] / "bundled-model-assets"

        if custom_assets_dir and bundled_assets.is_dir() and bundled_assets.resolve() != asset_root.resolve():
            _log(f"Staging bundled base model weights -> {asset_root}")
            for src_file in bundled_assets.rglob("*"):
                if not src_file.is_file():
                    continue
                rel = src_file.relative_to(bundled_assets)
                dst = asset_root / rel
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src_file, dst)
                except OSError:
                    shutil.copy2(src_file, dst)

        _sync_iit_template(artifacts.get("iit_template") or {}, work_dir, bundled_assets, asset_root)

        selected_packs = [p for p in ("wma_org", "wma_naba", "wma_east_west") if p in set(CONFIG.get("selected_packs") or ())]
        packs = artifacts.get("packs") or {}
        for pack_id in selected_packs:
            pspec = packs.get(pack_id)
            if not pspec:
                continue
            archive, pack_temps = _acquire_spec(pspec, work_dir)
            _log(f"Extracting {pspec['filename']} -> {asset_root}")
            extract_modular_pack(archive, asset_root, expected_sha256=str(pspec.get("sha256") or ""))
            _cleanup_temp_files(pack_temps)

        cfg = load_user_config()
        ui_cfg = cfg.setdefault("ui", {})
        ui_cfg["language"] = ui_cfg.get("language") or str(CONFIG.get("language") or "en")
        deps = cfg.setdefault("external_dependencies", {})
        if custom_assets_dir:
            deps["model_assets_dir"] = str(asset_root)
        else:
            deps.pop("model_assets_dir", None)
        deps.pop("wma_atlas_dir", None)

        defaults = cfg.setdefault("run_defaults", {})
        targets = [
            "microstructure",
            "segmentation",
            "parcellation",
            "surface",
            "tractography",
            "saf",
            "bundles",
            "tractcloud",
            "connectome",
        ]
        if selected_packs:
            targets.append("wma")
            defaults["bundle_backend"] = "wma,tractcloud,wmql"
            defaults["wma_atlas"] = (
                "org" if "wma_org" in selected_packs else ("naba" if "wma_naba" in selected_packs else "east-west")
            )
        else:
            defaults["bundle_backend"] = "tractcloud,wmql"
        defaults["targets"] = list(dict.fromkeys(targets))
        write_user_config(cfg)

        try:
            from shared.environment import EnvironmentDoctorConfig, run_environment_doctor

            doc = run_environment_doctor(
                EnvironmentDoctorConfig(
                    repo_root=module_root,
                    targets=tuple(defaults["targets"]),
                    wma_atlas=defaults.get("wma_atlas", "org"),
                    brain_mask_backend="pnl_cnn",
                    model_assets_dir=asset_root if custom_assets_dir else None,
                    python_executable=Path(sys.executable),
                    slicer_executable=Path(slicer_exe),
                )
            )
            _log(f"Environment doctor check: {len(doc.statuses)} items, blockers={doc.blocker_count}")
        except Exception as exc:
            _log(f"Post-install check note: {exc}")

        _log("Installation completed successfully.")
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())
    '''
).strip()


def load_installer_artifacts(
    *,
    packs_manifest_path: Path | None = None,
    artifacts_path: Path | None = None,
) -> dict[str, Any]:
    """Load merged WMA atlas pack & OS base bundle metadata for script generation."""

    packs_manifest = load_release_packs_manifest(packs_manifest_path or RELEASE_PACKS_MANIFEST)
    packs = dict(packs_manifest.get("packs") or {})
    bundles = dict(packs_manifest.get("bundles") or {})

    art_file = artifacts_path or INSTALLER_ARTIFACTS_JSON
    if art_file.is_file():
        extra = json.loads(art_file.read_text(encoding="utf-8"))
        bundles.update(extra.get("bundles") or {})
        iit_template = dict(extra.get("iit_template") or DEFAULT_IIT_TEMPLATE)
    else:
        iit_template = dict(DEFAULT_IIT_TEMPLATE)

    bundle_defaults = {
        "windows_x64_base": {
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
            "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
            "size_bytes": 0,
            "sha256": "",
            "parts": [],
        },
        "linux_x64_base": {
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
            "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
            "size_bytes": 0,
            "sha256": "",
            "parts": [],
        },
        "macos_arm64_base": {
            "filename": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
            "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
            "size_bytes": 0,
            "sha256": "",
            "parts": [],
        },
    }
    for key, val in bundle_defaults.items():
        bundles.setdefault(key, val)

    return {
        "release_tag": packs_manifest.get("release_tag", RELEASE_TAG),
        "release_base_url": packs_manifest.get("release_base_url", RELEASE_BASE_URL),
        "iit_template": iit_template,
        "packs": packs,
        "bundles": bundles,
    }


def _compact_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Strip file-hash maps not required at runtime to keep generated installer scripts compact."""

    compact_packs = {}
    for pid, spec in (artifacts.get("packs") or {}).items():
        entry: dict[str, Any] = {
            "id": pid,
            "filename": spec["filename"],
            "url": spec.get("url", ""),
            "size_bytes": int(spec.get("size_bytes") or 0),
            "sha256": spec.get("sha256", ""),
        }
        if spec.get("parts"):
            entry["parts"] = list(spec["parts"])
        compact_packs[pid] = entry
    return {
        "release_tag": artifacts.get("release_tag", RELEASE_TAG),
        "release_base_url": artifacts.get("release_base_url", RELEASE_BASE_URL),
        "iit_template": dict(artifacts.get("iit_template") or DEFAULT_IIT_TEMPLATE),
        "packs": compact_packs,
        "bundles": artifacts.get("bundles") or {},
    }


def _build_python_payload(
    *,
    target_os: str,
    mode: str,
    slicer_path: str,
    model_assets_dir: str,
    selected_packs: Iterable[str],
    mirror_prefix: str,
    proxy_url: str,
    offline_dir: str,
    check_script_dir: bool,
    language: str,
    artifacts: dict[str, Any],
) -> str:
    cfg = {
        "os": target_os,
        "mode": mode,
        "slicer_path": slicer_path.strip(),
        "model_assets_dir": model_assets_dir.strip(),
        "selected_packs": [p for p in PACK_ORDER if p in set(selected_packs)],
        "mirror_prefix": mirror_prefix.strip(),
        "proxy_url": proxy_url.strip(),
        "offline_dir": offline_dir.strip(),
        "check_script_dir": bool(check_script_dir),
        "language": "zh" if str(language).startswith("zh") else "en",
        "artifacts": _compact_artifacts(artifacts),
    }
    config_json = json.dumps(cfg, ensure_ascii=True)
    return (
        "# __SLICERDSPACE_PY_BEGIN__\n"
        f"CONFIG = __import__('json').loads({config_json!r})\n"
        'CONFIG["__script_file__"] = __file__\n'
        f"{_BOOTSTRAP_BODY}\n"
        "# __SLICERDSPACE_PY_END__\n"
    )


def script_filename_for_os(target_os: str) -> str:
    return {
        "windows": "Install-SlicerDSpace.bat",
        "macos": "Install-SlicerDSpace.command",
        "linux": "Install-SlicerDSpace.sh",
    }[target_os]


def generate_installer_script(
    *,
    target_os: str,
    mode: str = "base",
    slicer_path: str = "",
    model_assets_dir: str = "",
    selected_packs: Iterable[str] = ("wma_org",),
    mirror_prefix: str = "",
    proxy_url: str = "",
    offline_dir: str = "",
    check_script_dir: bool = True,
    language: str = "en",
    artifacts: dict[str, Any] | None = None,
) -> str:
    """Build a self-contained .bat, .command, or .sh installer script string."""

    if target_os not in SUPPORTED_OS:
        raise ValueError(f"Unsupported target OS: {target_os}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported install mode: {mode}")
    normalized_mode = "base"

    art = artifacts or load_installer_artifacts()
    py_payload = _build_python_payload(
        target_os=target_os,
        mode=normalized_mode,
        slicer_path=slicer_path,
        model_assets_dir=model_assets_dir,
        selected_packs=selected_packs,
        mirror_prefix=mirror_prefix,
        proxy_url=proxy_url,
        offline_dir=offline_dir,
        check_script_dir=check_script_dir,
        language=language,
        artifacts=art,
    )

    clean_slicer = slicer_path.strip().replace('"', "")
    if target_os == "windows":
        header = textwrap.dedent(
            f'''\
            @echo off
            setlocal EnableExtensions
            title SlicerDSpace Installer
            set "SLICERDSPACE_SCRIPT_DIR=%~dp0"
            set "SLICER_INPUT={clean_slicer}"

            call :resolve_slicer "%SLICER_INPUT%"
            if not defined PYTHON_SLICER (
              for %%D in ("%LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0" "C:\\Program Files\\Slicer 5.10.0") do (
                if not defined PYTHON_SLICER call :resolve_slicer "%%~D"
              )
            )
            if not defined PYTHON_SLICER (
              set /p SLICER_INPUT="Enter path to 3D Slicer 5.10.0 folder or Slicer.exe: "
              call :resolve_slicer "%SLICER_INPUT%"
            )
            if not defined PYTHON_SLICER (
              echo [ERROR] Could not locate Slicer.exe and bin\\PythonSlicer.exe.
              pause
              exit /b 1
            )

            set "SLICERDSPACE_RESOLVED_SLICER=%SLICER_EXE%"
            "%PYTHON_SLICER%" -u -c "import sys; bmk='#'+' __SLICERDSPACE_PY_BEGIN__'; emk='#'+' __SLICERDSPACE_PY_END__'; t=open(sys.argv[1], encoding='utf-8').read(); b=t.split(bmk,1)[1].split(emk,1)[0]; exec(compile(b, sys.argv[1], 'exec'), {{'__name__': '__main__', '__file__': sys.argv[1]}})" "%~f0"
            set "SD_RC=%ERRORLEVEL%"
            echo.
            if %SD_RC%==0 (
              echo [OK] Setup complete. Open 3D Slicer and select Modules - Diffusion - SlicerDSpace.
            ) else (
              echo [ERROR] Installer exited with code %SD_RC%.
            )
            if not "%SLICERDSPACE_INSTALLER_NONINTERACTIVE%"=="1" pause
            exit /b %SD_RC%

            :resolve_slicer
            set "CAND=%~1"
            if "%CAND%"=="" goto :eof
            if exist "%CAND%\\bin\\PythonSlicer.exe" if exist "%CAND%\\Slicer.exe" (
              set "SLICER_EXE=%CAND%\\Slicer.exe"
              set "PYTHON_SLICER=%CAND%\\bin\\PythonSlicer.exe"
              goto :eof
            )
            if /I "%~nx1"=="Slicer.exe" if exist "%~dp1bin\\PythonSlicer.exe" (
              set "SLICER_EXE=%CAND%"
              set "PYTHON_SLICER=%~dp1bin\\PythonSlicer.exe"
              goto :eof
            )
            goto :eof
            '''
        )
        script = header + "\n" + py_payload
        return script.replace("\r\n", "\n").replace("\n", "\r\n")

    escaped_slicer = slicer_path.strip().replace('"', '\\"')
    if target_os == "macos":
        resolve_fn = textwrap.dedent(
            '''\
            resolve_slicer() {
              local p="${1:-}"
              [ -n "$p" ] || return 1
              if [ -f "$p/Contents/MacOS/Slicer" ] && [ -f "$p/Contents/bin/PythonSlicer" ]; then
                SLICER_BIN="$p/Contents/MacOS/Slicer"
                PYTHON_SLICER="$p/Contents/bin/PythonSlicer"
                return 0
              fi
              if [ -f "$p/MacOS/Slicer" ] && [ -f "$p/bin/PythonSlicer" ]; then
                SLICER_BIN="$p/MacOS/Slicer"
                PYTHON_SLICER="$p/bin/PythonSlicer"
                return 0
              fi
              if [ "$(basename "$p")" = "Slicer" ] && [ -f "$(dirname "$p")/../bin/PythonSlicer" ]; then
                SLICER_BIN="$p"
                PYTHON_SLICER="$(cd "$(dirname "$p")/.." && pwd)/bin/PythonSlicer"
                return 0
              fi
              return 1
            }
            '''
        )
        fallback_list = '"/Applications/Slicer.app" "$HOME/Applications/Slicer.app" "/Applications/Slicer-5.10.0.app"'
    else:
        resolve_fn = textwrap.dedent(
            '''\
            resolve_slicer() {
              local p="${1:-}"
              [ -n "$p" ] || return 1
              if [ -f "$p/Slicer" ] && [ -f "$p/bin/PythonSlicer" ]; then
                SLICER_BIN="$p/Slicer"
                PYTHON_SLICER="$p/bin/PythonSlicer"
                return 0
              fi
              if [ "$(basename "$p")" = "Slicer" ] && [ -f "$(dirname "$p")/bin/PythonSlicer" ]; then
                SLICER_BIN="$p"
                PYTHON_SLICER="$(dirname "$p")/bin/PythonSlicer"
                return 0
              fi
              return 1
            }
            '''
        )
        fallback_list = '"$HOME/Slicer-5.10.0-linux-amd64" "/opt/Slicer-5.10.0-linux-amd64" "/opt/slicer"'

    header = textwrap.dedent(
        f'''\
        #!/usr/bin/env bash
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        SCRIPT_PATH="${{SCRIPT_DIR}}/$(basename "$0")"
        export SLICERDSPACE_SCRIPT_DIR="${{SCRIPT_DIR}}"
        CONFIGURED_SLICER="{escaped_slicer}"
        SLICER_BIN=""
        PYTHON_SLICER=""

        {resolve_fn.rstrip()}

        if ! resolve_slicer "$CONFIGURED_SLICER"; then
          for cand in {fallback_list}; do
            if resolve_slicer "$cand"; then
              break
            fi
          done
        fi

        if [ -z "$PYTHON_SLICER" ]; then
          printf "Enter path to 3D Slicer 5.10.0 (e.g. /Applications/Slicer.app or Slicer dir): "
          read -r USER_SLICER
          resolve_slicer "$USER_SLICER" || {{
            echo "[ERROR] Could not find Slicer and bin/PythonSlicer at: $USER_SLICER"
            exit 1
          }}
        fi

        export SLICERDSPACE_RESOLVED_SLICER="$SLICER_BIN"
        set +e
        "$PYTHON_SLICER" -u -c "import sys; bmk='#'+' __SLICERDSPACE_PY_BEGIN__'; emk='#'+' __SLICERDSPACE_PY_END__'; t=open(sys.argv[1], encoding='utf-8').read(); b=t.split(bmk,1)[1].split(emk,1)[0]; exec(compile(b, sys.argv[1], 'exec'), {{'__name__': '__main__', '__file__': sys.argv[1]}})" "$SCRIPT_PATH"
        SD_RC=$?
        set -e
        if [ "$SD_RC" -eq 0 ]; then
          echo ""
          echo "[OK] Setup complete! Open 3D Slicer -> Modules -> Diffusion -> SlicerDSpace."
        else
          echo ""
          echo "[ERROR] Installer exited with code $SD_RC."
        fi
        if [ -t 0 ] && [ "${{SLICERDSPACE_INSTALLER_NONINTERACTIVE:-0}}" != "1" ]; then
          printf "Press Enter to close..."
          read -r _
        fi
        exit "$SD_RC"
        '''
    )
    return (header + "\n" + py_payload).replace("\r\n", "\n")
