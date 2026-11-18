/* SlicerDSpace Website Visual Installer Customizer. Vanilla JS, offline-safe. */
(function () {
  'use strict';

  var zh = document.documentElement.lang.toLowerCase().indexOf('zh') === 0;
  var RELEASE_TAG = 'v0.1.1';
  var RELEASE_BASE = 'https://github.com/NeuroDMRI/SlicerDSpace/releases/download/' + RELEASE_TAG;
  var IIT_BYTES = 7830929;

  var ARTIFACTS = {
    release_tag: RELEASE_TAG,
    release_base_url: RELEASE_BASE,
    iit_template: {
      relpath: 'brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz',
      sha256: '2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead',
      size_bytes: IIT_BYTES,
      urls: ['https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz']
    },
    packs: {
      wma_org: {
        id: 'wma_org',
        filename: 'SlicerDSpace-v0.1.1-WMA-ORG.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-ORG.tar.gz',
        size_bytes: 1500000000,
        sha256: '',
        parts: []
      },
      wma_naba: {
        id: 'wma_naba',
        filename: 'SlicerDSpace-v0.1.1-WMA-NABA.tar.gz',
        url: '',
        size_bytes: 2900000000,
        sha256: '',
        parts: []
      },
      wma_east_west: {
        id: 'wma_east_west',
        filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz',
        url: '',
        size_bytes: 9900000000,
        sha256: '',
        parts: []
      }
    },
    bundles: {
      windows_x64_base: {
        filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip',
        size_bytes: 1600000000,
        sha256: '',
        parts: []
      },
      linux_x64_base: {
        filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz',
        size_bytes: 1400000000,
        sha256: '',
        parts: []
      },
      macos_arm64_base: {
        filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz',
        size_bytes: 1400000000,
        sha256: '',
        parts: []
      }
    }
  };

  var OS_DEFAULTS = {
    macos: {
      label: 'macOS',
      bundleKey: 'macos_arm64_base',
      slicerPlaceholder: '/Applications/Slicer.app',
      modelPlaceholder: '~/Library/Application Support/SlicerDSpace/model-assets',
      scriptName: 'Install-SlicerDSpace.command',
      baseEstimateBytes: 1400000000
    },
    windows: {
      label: 'Windows',
      bundleKey: 'windows_x64_base',
      slicerPlaceholder: '%LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0',
      modelPlaceholder: '%LOCALAPPDATA%\\SlicerDSpace\\model-assets',
      scriptName: 'Install-SlicerDSpace.bat',
      baseEstimateBytes: 1600000000
    },
    linux: {
      label: 'Linux',
      bundleKey: 'linux_x64_base',
      slicerPlaceholder: '~/Slicer-5.10.0-linux-amd64',
      modelPlaceholder: '~/.local/share/SlicerDSpace/model-assets',
      scriptName: 'Install-SlicerDSpace.sh',
      baseEstimateBytes: 1400000000
    }
  };

  var BASE_MODULES = [
    {
      title: zh ? '基础预处理与 PNL 自动脑掩膜' : 'Preprocessing & PNL Brain Mask',
      tag: zh ? '内置' : 'Included'
    },
    {
      title: zh ? '解剖先验 · DDSeg / DDParcel / DDSurfer' : 'Anatomical Priors · DDSeg / DDParcel / DDSurfer',
      tag: zh ? '内置' : 'Included'
    },
    {
      title: zh ? '微结构与全脑追踪 · DTI / NODDI / DKI / CSD / UKF / SAF' : 'Microstructure & Tractography · DTI / NODDI / DKI / CSD / UKF / SAF',
      tag: zh ? '内置' : 'Included'
    },
    {
      title: zh ? '纤维束与脑连接组 · WMQL / TractCloud / 连接矩阵' : 'Bundles & Connectome · WMQL / TractCloud / Network',
      tag: zh ? '内置' : 'Included'
    }
  ];

  var WMA_ATLAS_META = [
    { id: 'wma_org', title: zh ? 'ORG · 标准 74 束图谱' : 'ORG · Standard 74-Tract' },
    { id: 'wma_naba', title: zh ? 'NABA · 跨年龄段图谱' : 'NABA · Lifespan Atlas' },
    { id: 'wma_east_west', title: zh ? 'East-West · 东西方人群图谱' : 'East-West · Cross-Population' }
  ];

  var BOOTSTRAP_PY = [
    'import hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.error, urllib.request',
    'from pathlib import Path',
    '_GITHUB_PREFIXES = ("https://github.com/", "https://raw.githubusercontent.com/", "https://objects.githubusercontent.com/", "https://media.githubusercontent.com/", "https://codeload.github.com/")',
    'def _log(msg): print(f"[SlicerDSpace Installer] {msg}", flush=True)',
    'def _sha256(path):',
    '    h = hashlib.sha256()',
    '    with path.open("rb") as fh:',
    '        for chunk in iter(lambda: fh.read(1048576), b""): h.update(chunk)',
    '    return h.hexdigest()',
    'def _clean_mirror(p):',
    '    p = str(p or "").strip()',
    '    return (p.rstrip("/") + "/") if (p.startswith("https://") or p.startswith("http://")) else ""',
    'def _expand_urls(urls, allow_mirror=True):',
    '    raw = [urls] if isinstance(urls, str) else list(urls or ())',
    '    mirror = _clean_mirror(CONFIG.get("mirror_prefix", ""))',
    '    out = []',
    '    if allow_mirror and mirror:',
    '        for u in raw:',
    '            if u.startswith(_GITHUB_PREFIXES) and not u.startswith(mirror):',
    '                m = mirror + u',
    '                if m not in out: out.append(m)',
    '    for u in raw:',
    '        if u and u not in out: out.append(u)',
    '    return out',
    'def _open_url(req, timeout=120):',
    '    proxy = str(CONFIG.get("proxy_url", "") or "").strip()',
    '    if proxy:',
    '        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})).open(req, timeout=timeout)',
    '    return urllib.request.urlopen(req, timeout=timeout)',
    'def _download(urls, dest, expected_sha256="", expected_size=0, allow_mirror=True):',
    '    dest.parent.mkdir(parents=True, exist_ok=True)',
    '    if dest.is_file():',
    '        if expected_sha256 and _sha256(dest) == expected_sha256:',
    '            _log(f"Reusing verified file: {dest.name}"); return dest',
    '        if not expected_sha256 and expected_size and dest.stat().st_size == expected_size:',
    '            _log(f"Reusing file: {dest.name}"); return dest',
    '    part = dest.with_name(dest.name + ".part")',
    '    errors = []',
    '    for url in _expand_urls(urls, allow_mirror=allow_mirror):',
    '        try:',
    '            _log(f"Downloading {dest.name} <- {url}")',
    '            resume_from = part.stat().st_size if part.is_file() else 0',
    '            if expected_size and resume_from > expected_size: part.unlink(missing_ok=True); resume_from = 0',
    '            headers = {"User-Agent": "SlicerDSpace-custom-installer"}',
    '            if resume_from: headers["Range"] = f"bytes={resume_from}-"',
    '            with _open_url(urllib.request.Request(url, headers=headers), timeout=120) as resp:',
    '                append = bool(resume_from and getattr(resp, "status", 200) == 206)',
    '                if resume_from and not append: resume_from = 0',
    '                clen = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None',
    '                total = (int(clen) + resume_from) if (clen and str(clen).isdigit()) else int(expected_size or 0)',
    '                done = resume_from; next_log = done + 33554432',
    '                with part.open("ab" if append else "wb") as fh:',
    '                    while True:',
    '                        chunk = resp.read(1048576)',
    '                        if not chunk: break',
    '                        fh.write(chunk); done += len(chunk)',
    '                        if total and done >= next_log:',
    '                            _log(f"  {dest.name}: {done/1048576:.1f}/{total/1048576:.1f} MB ({100.0*done/max(1,total):.1f}%)")',
    '                            next_log = done + 33554432',
    '            if total and done < total: raise RuntimeError(f"Short read {done}/{total}")',
    '            if expected_sha256:',
    '                actual = _sha256(part)',
    '                if actual != expected_sha256: part.unlink(missing_ok=True); raise ValueError(f"SHA-256 mismatch for {dest.name}: {actual}")',
    '            os.replace(part, dest); _log(f"Verified {dest.name}"); return dest',
    '        except urllib.error.HTTPError as exc:',
    '            if exc.code == 416: part.unlink(missing_ok=True)',
    '            errors.append(f"{url}: {exc}")',
    '        except Exception as exc:',
    '            errors.append(f"{url}: {exc}")',
    '    raise RuntimeError(f"Failed downloading {dest.name}: {errors}")',
    'def _search_dirs():',
    '    dirs = []',
    '    raw = str(CONFIG.get("offline_dir", "") or "").strip()',
    '    if raw and Path(raw).expanduser().is_dir(): dirs.append(Path(raw).expanduser().resolve())',
    '    if CONFIG.get("check_script_dir", True):',
    '        here = Path(os.environ.get("SLICERDSPACE_SCRIPT_DIR") or Path(CONFIG["__script_file__"]).parent).resolve()',
    '        if here.is_dir() and here not in dirs: dirs.append(here)',
    '    return dirs',
    'def _acquire_spec(spec, cache_dir):',
    '    filename = spec["filename"]; expected_sha = str(spec.get("sha256") or ""); expected_size = int(spec.get("size_bytes") or 0)',
    '    parts = spec.get("parts") or []; dirs = _search_dirs()',
    '    for d in dirs:',
    '        c = d / filename',
    '        if c.is_file() and (not expected_sha or _sha256(c) == expected_sha):',
    '            _log(f"Using offline archive: {c}"); return c, []',
    '    cached = cache_dir / filename',
    '    if cached.is_file() and expected_sha and _sha256(cached) == expected_sha:',
    '        _log(f"Reusing cache: {cached}"); return cached, [cached]',
    '    if parts:',
    '        part_paths = []; dl_parts = []',
    '        for ps in parts:',
    '            pn = ps["filename"]; psha = str(ps.get("sha256") or ""); psize = int(ps.get("size_bytes") or 0); found = None',
    '            for d in dirs:',
    '                c = d / pn',
    '                if c.is_file() and (not psha or _sha256(c) == psha): _log(f"Using offline part: {c}"); found = c; break',
    '            if found is None:',
    '                found = _download(ps["url"], cache_dir / pn, expected_sha256=psha, expected_size=psize)',
    '                dl_parts.append(found)',
    '            part_paths.append(found)',
    '        _log(f"Merging {len(part_paths)} volumes into {filename}...")',
    '        merging = cache_dir / (filename + ".merging")',
    '        with merging.open("wb") as out_fh:',
    '            for pp in part_paths:',
    '                with pp.open("rb") as in_fh: shutil.copyfileobj(in_fh, out_fh, length=16777216)',
    '        if expected_sha and _sha256(merging) != expected_sha: merging.unlink(missing_ok=True); raise RuntimeError(f"Merged SHA-256 mismatch: {filename}")',
    '        os.replace(merging, cached); _log(f"Merged and verified {filename}"); return cached, [*dl_parts, cached]',
    '    if not spec.get("url"): raise RuntimeError(f"Download URL is not configured yet for {filename}")',
    '    arc = _download(spec["url"], cached, expected_sha256=expected_sha, expected_size=expected_size)',
    '    return arc, [arc]',
    'def _cleanup_temp_files(paths):',
    '    for p in paths:',
    '        try: Path(p).unlink(missing_ok=True)',
    '        except OSError: pass',
    'def _install_extension(slicer_exe, archive_path):',
    '    if sys.platform == "darwin" and shutil.which("xattr"): subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(archive_path)], capture_output=True)',
    '    receipt = Path(tempfile.mkdtemp(prefix="slicerdspace-inst-")) / "receipt.json"',
    '    helper = receipt.with_name("install_ext.py")',
    '    helper.write_text("import json, sys, traceback, qt, slicer\\nfrom pathlib import Path\\ndef _run():\\n  a=sys.argv[1]; r=Path(sys.argv[2])\\n  try:\\n    em=slicer.app.extensionsManagerModel(); em.setInteractive(False)\\n    meta={\'extensionname\':\'SlicerDSpace\',\'depends\':\'NA\',\'slicer_revision\':str(em.slicerRevision),\'os\':str(em.slicerOs),\'arch\':str(em.slicerArch)}\\n    ok=bool(em.installExtension(\'SlicerDSpace\', meta, a, False, True))\\n    r.write_text(json.dumps({\'installed\':ok,\'module_paths\':[str(p) for p in em.extensionModulePaths(\'SlicerDSpace\')]}),encoding=\'utf-8\')\\n    slicer.app.exit(0 if ok else 1)\\n  except Exception as e:\\n    r.write_text(json.dumps({\'installed\':False,\'error\':str(e),\'traceback\':traceback.format_exc()}),encoding=\'utf-8\')\\n    slicer.app.exit(1)\\nqt.QTimer.singleShot(500, _run)\\n", encoding="utf-8")',
    '    cmd = [slicer_exe, "--no-splash", "--no-main-window", "--python-script", str(helper), str(archive_path), str(receipt)]',
    '    env = os.environ.copy()',
    '    if sys.platform.startswith("linux") and not env.get("DISPLAY"):',
    '        xvfb = shutil.which("xvfb-run")',
    '        if xvfb: cmd = [xvfb, "-a", *cmd]',
    '        else: env.setdefault("QT_QPA_PLATFORM", "offscreen")',
    '    _log("Installing SlicerDSpace base extension into 3D Slicer...")',
    '    proc = subprocess.run(cmd, env=env)',
    '    if proc.returncode != 0 or not receipt.is_file(): raise RuntimeError(f"Slicer install exited {proc.returncode}")',
    '    data = json.loads(receipt.read_text(encoding="utf-8")); shutil.rmtree(receipt.parent, ignore_errors=True)',
    '    if not data.get("installed"): raise RuntimeError("Slicer rejected extension archive")',
    '    return data',
    'def _find_module_root(slicer_exe, receipt_data):',
    '    for mp in receipt_data.get("module_paths") or ():',
    '        if (Path(mp) / "SlicerDSpace.py").is_file(): return Path(mp).resolve()',
    '    sp = Path(slicer_exe).resolve(); cands = []',
    '    if sys.platform == "darwin":',
    '        cands.extend(sp.parents[1].glob("Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))',
    '        cands.extend((Path.home() / "Library/Application Support/slicer.org").glob("Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))',
    '    else: cands.extend(sp.parent.glob("slicer.org/Extensions-*/SlicerDSpace/lib/Slicer-*/qt-scripted-modules"))',
    '    for c in cands:',
    '        if (c / "SlicerDSpace.py").is_file(): return c.resolve()',
    '    raise FileNotFoundError("Installed SlicerDSpace module directory not found")',
    'def _sync_iit_template(iit_spec, work_dir, bundled_assets, asset_root):',
    '    if not iit_spec: return',
    '    relpath = str(iit_spec.get("relpath") or "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz")',
    '    sha = str(iit_spec.get("sha256") or ""); sz = int(iit_spec.get("size_bytes") or 0)',
    '    targets = ([bundled_assets / relpath] if bundled_assets.is_dir() else []) + [asset_root / relpath]',
    '    if all(t.is_file() and _sha256(t) == sha for t in targets): return',
    '    src = next((t for t in targets if t.is_file() and _sha256(t) == sha), None)',
    '    if src is None:',
    '        for d in _search_dirs():',
    '            for cand in (d / relpath, d / Path(relpath).name):',
    '                if cand.is_file() and _sha256(cand) == sha: _log(f"Using local NITRC IIT template: {cand}"); src = cand; break',
    '            if src is not None: break',
    '    tmp_dl = None',
    '    if src is None:',
    '        try: tmp_dl = work_dir / Path(relpath).name; src = _download(iit_spec.get("urls") or (), tmp_dl, expected_sha256=sha, expected_size=sz, allow_mirror=False)',
    '        except Exception as exc: _log(f"[WARN] Optional NITRC IIT template download skipped ({exc})"); return',
    '    try:',
    '        for dst in targets:',
    '            if dst.is_file() and _sha256(dst) == sha: continue',
    '            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)',
    '    finally:',
    '        if tmp_dl is not None: tmp_dl.unlink(missing_ok=True)',
    'def main():',
    '    slicer_exe = str(os.environ.get("SLICERDSPACE_RESOLVED_SLICER") or CONFIG.get("slicer_path") or "").strip()',
    '    if not slicer_exe or not Path(slicer_exe).exists(): raise FileNotFoundError(f"Slicer not found: {slicer_exe}")',
    '    os_name, artifacts = CONFIG["os"], CONFIG["artifacts"]',
    '    work_dir = Path(tempfile.gettempdir()) / "slicerdspace-installer-cache"; work_dir.mkdir(parents=True, exist_ok=True)',
    '    bkey = {"windows":"windows_x64_base","linux":"linux_x64_base","macos":"macos_arm64_base"}[os_name]',
    '    ext_archive, ext_temps = _acquire_spec(artifacts["bundles"][bkey], work_dir)',
    '    receipt = _install_extension(slicer_exe, ext_archive); _cleanup_temp_files(ext_temps)',
    '    mod_root = _find_module_root(slicer_exe, receipt); _log(f"Extension installed at: {mod_root}")',
    '    if str(mod_root) not in sys.path: sys.path.insert(0, str(mod_root))',
    '    from shared.network_config import extract_modular_pack, save_network_settings',
    '    from shared.model_assets import configured_model_assets_root',
    '    from shared.user_config import load_user_config, write_user_config',
    '    save_network_settings(mirror_prefix=CONFIG.get("mirror_prefix",""), proxy_url=CONFIG.get("proxy_url",""), offline_assets_dir=CONFIG.get("offline_dir",""))',
    '    if CONFIG.get("mirror_prefix"): os.environ["SLICERDSPACE_MIRROR_PREFIX"] = _clean_mirror(CONFIG["mirror_prefix"])',
    '    if CONFIG.get("proxy_url"): os.environ["SLICERDSPACE_PROXY_URL"] = os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = str(CONFIG["proxy_url"]).strip()',
    '    if CONFIG.get("offline_dir"): os.environ["SLICERDSPACE_OFFLINE_DIR"] = str(CONFIG["offline_dir"]).strip()',
    '    custom_assets = str(CONFIG.get("model_assets_dir") or "").strip()',
    '    asset_root = configured_model_assets_root(custom_assets or None); asset_root.mkdir(parents=True, exist_ok=True)',
    '    bundled_assets = mod_root.parents[2] / "bundled-model-assets"',
    '    if custom_assets and bundled_assets.is_dir() and bundled_assets.resolve() != asset_root.resolve():',
    '        _log(f"Staging bundled base model weights -> {asset_root}")',
    '        for src_file in bundled_assets.rglob("*"):',
    '            if not src_file.is_file(): continue',
    '            dst = asset_root / src_file.relative_to(bundled_assets)',
    '            if dst.exists(): continue',
    '            dst.parent.mkdir(parents=True, exist_ok=True)',
    '            try: os.link(src_file, dst)',
    '            except OSError: shutil.copy2(src_file, dst)',
    '    _sync_iit_template(artifacts.get("iit_template") or {}, work_dir, bundled_assets, asset_root)',
    '    selected_packs = [p for p in ("wma_org", "wma_naba", "wma_east_west") if p in set(CONFIG.get("selected_packs") or ())]',
    '    packs = artifacts.get("packs") or {}',
    '    for pid in selected_packs:',
    '        pspec = packs.get(pid)',
    '        if not pspec: continue',
    '        arc, ptemps = _acquire_spec(pspec, work_dir); _log(f"Extracting {pspec[\'filename\']} -> {asset_root}")',
    '        extract_modular_pack(arc, asset_root, expected_sha256=str(pspec.get("sha256") or "")); _cleanup_temp_files(ptemps)',
    '    cfg = load_user_config(); cfg["ui"]["language"] = CONFIG.get("language", "en")',
    '    deps = cfg.setdefault("external_dependencies", {})',
    '    if custom_assets: deps["model_assets_dir"] = str(asset_root)',
    '    else: deps.pop("model_assets_dir", None)',
    '    deps.pop("wma_atlas_dir", None)',
    '    defaults = cfg.setdefault("run_defaults", {})',
    '    targets = ["microstructure", "segmentation", "parcellation", "surface", "tractography", "saf", "bundles", "tractcloud", "connectome"]',
    '    if selected_packs:',
    '        targets.append("wma"); defaults["bundle_backend"] = "wma,tractcloud,wmql"',
    '        defaults["wma_atlas"] = "org" if "wma_org" in selected_packs else ("naba" if "wma_naba" in selected_packs else "east-west")',
    '    else: defaults["bundle_backend"] = "tractcloud,wmql"',
    '    defaults["targets"] = list(dict.fromkeys(targets)); write_user_config(cfg)',
    '    try:',
    '        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor',
    '        doc = run_environment_doctor(EnvironmentDoctorConfig(repo_root=mod_root, targets=tuple(defaults["targets"]), wma_atlas=defaults.get("wma_atlas", "org"), brain_mask_backend="pnl_cnn", model_assets_dir=asset_root if custom_assets else None, python_executable=Path(sys.executable), slicer_executable=Path(slicer_exe)))',
    '        _log(f"Environment doctor check: {len(doc.statuses)} items, blockers={doc.blocker_count}")',
    '    except Exception as exc: _log(f"Post-install check note: {exc}")',
    '    _log("Installation completed successfully."); return 0',
    'if __name__ == "__main__": raise SystemExit(main())'
  ].join('\n');

  function formatBytes(n) {
    if (!n || n <= 0) { return '0 MB'; }
    var u = ['B', 'KB', 'MB', 'GB'];
    var i = 0;
    var val = Number(n);
    while (val >= 1024 && i < u.length - 1) { val /= 1024; i += 1; }
    return (i === 0 ? String(Math.round(val)) : val.toFixed(i >= 3 ? 2 : 1)) + ' ' + u[i];
  }

  function detectOS() {
    var ua = (navigator.platform || navigator.userAgent || '').toLowerCase();
    if (ua.indexOf('win') !== -1) { return 'windows'; }
    if (ua.indexOf('mac') !== -1) { return 'macos'; }
    return 'linux';
  }

  function asciiJsonLiteral(obj) {
    var s = JSON.stringify(obj);
    return "'" + s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/[\u007f-\uffff]/g, function (c) {
      return '\\u' + ('0000' + c.charCodeAt(0).toString(16)).slice(-4);
    }) + "'";
  }

  function buildScript(state) {
    var selectedPacks = state.wmaEnabled ? state.wmaAtlases.slice() : [];
    var cfg = {
      os: state.os,
      mode: 'base',
      slicer_path: state.slicerPath.trim(),
      model_assets_dir: state.modelAssetsDir.trim(),
      selected_packs: selectedPacks,
      mirror_prefix: state.mirrorPrefix.trim(),
      proxy_url: state.proxyUrl.trim(),
      offline_dir: state.offlineDir.trim(),
      check_script_dir: state.checkScriptDir,
      language: zh ? 'zh' : 'en',
      artifacts: ARTIFACTS
    };
    var pyPayload =
      '# __SLICERDSPACE_PY_BEGIN__\n' +
      'CONFIG = __import__("json").loads(' + asciiJsonLiteral(cfg) + ')\n' +
      'CONFIG["__script_file__"] = __file__\n' +
      BOOTSTRAP_PY + '\n' +
      '# __SLICERDSPACE_PY_END__\n';

    var cleanSlicer = state.slicerPath.trim().replace(/"/g, '');
    if (state.os === 'windows') {
      var bat = [
        '@echo off',
        'setlocal EnableExtensions',
        'title SlicerDSpace Installer',
        'set "SLICERDSPACE_SCRIPT_DIR=%~dp0"',
        'set "SLICER_INPUT=' + cleanSlicer + '"',
        'call :resolve_slicer "%SLICER_INPUT%"',
        'if not defined PYTHON_SLICER (',
        '  for %%D in ("%LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0" "C:\\Program Files\\Slicer 5.10.0") do (',
        '    if not defined PYTHON_SLICER call :resolve_slicer "%%~D"',
        '  )',
        ')',
        'if not defined PYTHON_SLICER (',
        '  set /p SLICER_INPUT="Enter path to 3D Slicer 5.10.0 folder or Slicer.exe: "',
        '  call :resolve_slicer "%SLICER_INPUT%"',
        ')',
        'if not defined PYTHON_SLICER (',
        '  echo [ERROR] Could not locate Slicer.exe and bin\\PythonSlicer.exe.',
        '  pause',
        '  exit /b 1',
        ')',
        'set "SLICERDSPACE_RESOLVED_SLICER=%SLICER_EXE%"',
        '"%PYTHON_SLICER%" -u -c "import sys; bmk=\'#\'+\' __SLICERDSPACE_PY_BEGIN__\'; emk=\'#\'+\' __SLICERDSPACE_PY_END__\'; t=open(sys.argv[1], encoding=\'utf-8\').read(); b=t.split(bmk,1)[1].split(emk,1)[0]; exec(compile(b, sys.argv[1], \'exec\'), {\'__name__\': \'__main__\', \'__file__\': sys.argv[1]})" "%~f0"',
        'set "SD_RC=%ERRORLEVEL%"',
        'echo.',
        'if %SD_RC%==0 (echo [OK] Setup complete. Open 3D Slicer -> Modules -> Diffusion -> SlicerDSpace.) else (echo [ERROR] Installer exited with code %SD_RC%.)',
        'if not "%SLICERDSPACE_INSTALLER_NONINTERACTIVE%"=="1" pause',
        'exit /b %SD_RC%',
        '',
        ':resolve_slicer',
        'set "CAND=%~1"',
        'if "%CAND%"=="" goto :eof',
        'if exist "%CAND%\\bin\\PythonSlicer.exe" if exist "%CAND%\\Slicer.exe" (set "SLICER_EXE=%CAND%\\Slicer.exe" & set "PYTHON_SLICER=%CAND%\\bin\\PythonSlicer.exe" & goto :eof)',
        'if /I "%~nx1"=="Slicer.exe" if exist "%~dp1bin\\PythonSlicer.exe" (set "SLICER_EXE=%CAND%" & set "PYTHON_SLICER=%~dp1bin\\PythonSlicer.exe" & goto :eof)',
        'goto :eof',
        '',
        pyPayload
      ].join('\n');
      return bat.replace(/\r?\n/g, '\r\n');
    }

    var escSlicer = cleanSlicer.replace(/\\/g, '\\\\');
    var resolveFn = state.os === 'macos'
      ? [
          'resolve_slicer() {',
          '  local p="${1:-}"; [ -n "$p" ] || return 1',
          '  if [ -f "$p/Contents/MacOS/Slicer" ] && [ -f "$p/Contents/bin/PythonSlicer" ]; then SLICER_BIN="$p/Contents/MacOS/Slicer"; PYTHON_SLICER="$p/Contents/bin/PythonSlicer"; return 0; fi',
          '  if [ -f "$p/MacOS/Slicer" ] && [ -f "$p/bin/PythonSlicer" ]; then SLICER_BIN="$p/MacOS/Slicer"; PYTHON_SLICER="$p/bin/PythonSlicer"; return 0; fi',
          '  if [ "$(basename "$p")" = "Slicer" ] && [ -f "$(dirname "$p")/../bin/PythonSlicer" ]; then SLICER_BIN="$p"; PYTHON_SLICER="$(cd "$(dirname "$p")/.." && pwd)/bin/PythonSlicer"; return 0; fi',
          '  return 1',
          '}'
        ].join('\n')
      : [
          'resolve_slicer() {',
          '  local p="${1:-}"; [ -n "$p" ] || return 1',
          '  if [ -f "$p/Slicer" ] && [ -f "$p/bin/PythonSlicer" ]; then SLICER_BIN="$p/Slicer"; PYTHON_SLICER="$p/bin/PythonSlicer"; return 0; fi',
          '  if [ "$(basename "$p")" = "Slicer" ] && [ -f "$(dirname "$p")/bin/PythonSlicer" ]; then SLICER_BIN="$p"; PYTHON_SLICER="$(dirname "$p")/bin/PythonSlicer"; return 0; fi',
          '  return 1',
          '}'
        ].join('\n');
    var fallbacks = state.os === 'macos'
      ? '"/Applications/Slicer.app" "$HOME/Applications/Slicer.app" "/Applications/Slicer-5.10.0.app"'
      : '"$HOME/Slicer-5.10.0-linux-amd64" "/opt/Slicer-5.10.0-linux-amd64" "/opt/slicer"';

    return [
      '#!/usr/bin/env bash',
      'set -euo pipefail',
      'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
      'SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"',
      'export SLICERDSPACE_SCRIPT_DIR="${SCRIPT_DIR}"',
      'CONFIGURED_SLICER="' + escSlicer + '"',
      'SLICER_BIN=""; PYTHON_SLICER=""',
      resolveFn,
      'if ! resolve_slicer "$CONFIGURED_SLICER"; then',
      '  for cand in ' + fallbacks + '; do if resolve_slicer "$cand"; then break; fi; done',
      'fi',
      'if [ -z "$PYTHON_SLICER" ]; then',
      '  printf "Enter path to 3D Slicer 5.10.0: "; read -r USER_SLICER',
      '  resolve_slicer "$USER_SLICER" || { echo "[ERROR] Slicer not found at: $USER_SLICER"; exit 1; }',
      'fi',
      'export SLICERDSPACE_RESOLVED_SLICER="$SLICER_BIN"',
      'set +e',
      '"$PYTHON_SLICER" -u -c "import sys; bmk=\'#\'+\' __SLICERDSPACE_PY_BEGIN__\'; emk=\'#\'+\' __SLICERDSPACE_PY_END__\'; t=open(sys.argv[1], encoding=\'utf-8\').read(); b=t.split(bmk,1)[1].split(emk,1)[0]; exec(compile(b, sys.argv[1], \'exec\'), {\'__name__\': \'__main__\', \'__file__\': sys.argv[1]})" "$SCRIPT_PATH"',
      'SD_RC=$?',
      'set -e',
      'if [ "$SD_RC" -eq 0 ]; then echo ""; echo "[OK] Setup complete! Open 3D Slicer -> Modules -> Diffusion -> SlicerDSpace."; else echo ""; echo "[ERROR] Installer exited with code $SD_RC."; fi',
      'if [ -t 0 ] && [ "${SLICERDSPACE_INSTALLER_NONINTERACTIVE:-0}" != "1" ]; then printf "Press Enter to close..."; read -r _; fi',
      'exit "$SD_RC"',
      '',
      pyPayload
    ].join('\n');
  }

  function initCustomizer() {
    var root = document.getElementById('installer-customizer');
    if (!root) { return; }

    var state = {
      os: detectOS(),
      slicerPath: '',
      modelAssetsDir: '',
      wmaEnabled: true,
      wmaAtlases: ['wma_org'],
      mirrorPreset: 'direct',
      mirrorPrefix: '',
      proxyUrl: '',
      offlineDir: '',
      checkScriptDir: true,
      moreOpen: false,
      previewOpen: false
    };

    function t(en, cn) { return zh ? cn : en; }

    function baseBytes() {
      var def = OS_DEFAULTS[state.os];
      var bundle = ARTIFACTS.bundles[def.bundleKey];
      return ((bundle && bundle.size_bytes) ? bundle.size_bytes : def.baseEstimateBytes) + IIT_BYTES;
    }

    function totalBytes() {
      var sum = baseBytes();
      if (state.wmaEnabled) {
        state.wmaAtlases.forEach(function (aid) {
          var p = ARTIFACTS.packs[aid];
          if (p) { sum += p.size_bytes || 0; }
        });
      }
      return sum;
    }

    function escapeAttr(v) {
      return String(v || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function render() {
      var def = OS_DEFAULTS[state.os];
      var html = [];
      html.push('<div class="ic-top">');
      html.push('<div class="ic-top-copy">');
      html.push('<h3 class="ic-heading">' + t('Custom Installer', '定制安装脚本') + '</h3>');
      html.push('<p class="ic-sub">' + t('Base workflows and anatomical priors are included; pick optional WMA atlases below.', '核心分析与解剖先验模型均默认内置，下方按需选择 WMA 白质图谱即可生成安装脚本。') + '</p>');
      html.push('</div></div>');

      html.push('<div class="ic-row">');
      html.push('<div class="ic-field"><span class="ic-k">' + t('System', '系统') + '</span><div class="ic-seg" role="group">');
      ['macos', 'windows', 'linux'].forEach(function (osKey) {
        html.push('<button type="button" class="ic-seg-btn" data-os="' + osKey + '" aria-pressed="' + (state.os === osKey ? 'true' : 'false') + '">' + OS_DEFAULTS[osKey].label + '</button>');
      });
      html.push('</div></div>');

      html.push('<div class="ic-field"><label class="ic-k" for="ic-slicer-path">' + t('3D Slicer 5.10.0 path (blank = auto)', 'Slicer 路径（默认位置可留空）') + '</label>');
      html.push('<input id="ic-slicer-path" class="ic-input" type="text" value="' + escapeAttr(state.slicerPath) + '" placeholder="' + escapeAttr(def.slicerPlaceholder) + '"></div>');
      html.push('</div>');

      html.push('<div class="ic-mod-head"><span class="ic-k">' + t('Base Package (Included · ', '统一底座套件（全部内置 · 约 ') + formatBytes(baseBytes()) + (zh ? '）' : ')') + '</span></div>');
      html.push('<div class="ic-packs">');
      BASE_MODULES.forEach(function (mod) {
        html.push('<div class="ic-pack is-locked"><input type="checkbox" checked disabled><span class="ic-pack-title">' + mod.title + '</span><span class="ic-pack-size">' + mod.tag + '</span></div>');
      });
      html.push('</div>');

      html.push('<div class="ic-wma-box ' + (state.wmaEnabled ? '' : 'is-disabled') + '">');
      html.push('<div class="ic-wma-head">');
      html.push('<label class="ic-wma-master"><input type="checkbox" id="ic-wma-toggle"' + (state.wmaEnabled ? ' checked' : '') + '><span>' + t('WMA White-Matter Atlas Clustering', 'WMA 白质纤维图谱分割') + '</span></label>');
      html.push('<span class="ic-wma-note">' + t('Uncheck WMA to skip all atlases; NABA / East-West optional', '取消 WMA 则关闭所有图谱；后两套大图谱默认不下载') + '</span>');
      html.push('</div>');
      html.push('<div class="ic-wma-grid">');
      WMA_ATLAS_META.forEach(function (atlas) {
        var pspec = ARTIFACTS.packs[atlas.id] || {};
        var checked = state.wmaEnabled && state.wmaAtlases.indexOf(atlas.id) !== -1;
        html.push('<label class="ic-pack ' + (checked ? 'is-selected' : '') + '"><input type="checkbox" data-atlas="' + atlas.id + '"' + (checked ? ' checked' : '') + '><span class="ic-pack-title">' + atlas.title + '</span><span class="ic-pack-size">' + formatBytes(pspec.size_bytes || 0) + '</span></label>');
      });
      html.push('</div></div>');

      html.push('<details id="ic-more" class="ic-more"' + (state.moreOpen ? ' open' : '') + '>');
      html.push('<summary class="ic-more-summary">' + t('Mirror, proxy & offline pack options', '镜像源 / 代理 / 本地离线目录') + '</summary>');
      html.push('<div class="ic-more-body">');
      html.push('<div class="ic-net-grid">');
      html.push('<div class="ic-field"><label class="ic-k" for="ic-mirror">' + t('Mirror', '镜像源') + '</label><select id="ic-mirror" class="ic-select">');
      html.push('<option value="direct"' + (state.mirrorPreset === 'direct' ? ' selected' : '') + '>' + t('GitHub Direct', 'GitHub 直连') + '</option>');
      html.push('<option value="https://ghfast.top/"' + (state.mirrorPreset === 'https://ghfast.top/' ? ' selected' : '') + '>ghfast.top</option>');
      html.push('<option value="https://gh-proxy.com/"' + (state.mirrorPreset === 'https://gh-proxy.com/' ? ' selected' : '') + '>gh-proxy.com</option>');
      html.push('<option value="custom"' + (state.mirrorPreset === 'custom' ? ' selected' : '') + '>' + t('Custom...', '自定义...') + '</option>');
      html.push('</select>');
      if (state.mirrorPreset === 'custom') {
        html.push('<input id="ic-custom-mirror" class="ic-input" style="margin-top:6px;" type="text" value="' + escapeAttr(state.mirrorPrefix) + '" placeholder="https://your-mirror.example/">');
      }
      html.push('</div>');
      html.push('<div class="ic-field"><label class="ic-k" for="ic-proxy">' + t('Proxy', 'HTTP/HTTPS 代理') + '</label><input id="ic-proxy" class="ic-input" type="text" value="' + escapeAttr(state.proxyUrl) + '" placeholder="http://127.0.0.1:7890"></div>');
      html.push('<div class="ic-field"><label class="ic-k" for="ic-offline-dir">' + t('Offline folder', '本地离线包目录') + '</label><input id="ic-offline-dir" class="ic-input" type="text" value="' + escapeAttr(state.offlineDir) + '" placeholder="' + t('Optional folder', '可选目录路径') + '"></div>');
      html.push('</div>');
      html.push('<div class="ic-more-sub">');
      html.push('<div class="ic-field"><label class="ic-k" for="ic-model-dir">' + t('Model directory (optional)', '模型存放目录（可选）') + '</label><input id="ic-model-dir" class="ic-input" type="text" value="' + escapeAttr(state.modelAssetsDir) + '" placeholder="' + escapeAttr(def.modelPlaceholder) + '"></div>');
      html.push('<label class="ic-check-inline"><input id="ic-check-script-dir" type="checkbox"' + (state.checkScriptDir ? ' checked' : '') + '><span>' + t('Check script folder for offline packs', '优先读取脚本同目录离线包') + '</span></label>');
      html.push('</div></div></details>');

      html.push('<div class="ic-bar">');
      html.push('<div class="ic-total">' + t('Download', '预计下载') + ' <strong>~' + formatBytes(totalBytes()) + '</strong></div>');
      html.push('<div class="ic-actions">');
      html.push('<button type="button" class="ic-btn-ghost" id="ic-toggle-preview">' + (state.previewOpen ? t('Hide code', '收起脚本') : t('Preview', '预览')) + '</button>');
      html.push('<button type="button" class="ic-btn-secondary" id="ic-copy-cmd-btn">' + t('Copy command', '复制命令') + '</button>');
      html.push('<button type="button" class="ic-btn-primary" id="ic-download-btn">' + t('Download ', '下载 ') + def.scriptName + '</button>');
      html.push('</div></div>');

      html.push('<div class="ic-preview code"' + (state.previewOpen ? '' : ' hidden') + '><div class="code-head"><span>' + escapeAttr(def.scriptName) + '</span><button type="button" class="ic-btn-secondary" id="ic-copy-script-btn">' + t('Copy', '复制') + '</button></div><pre><code id="ic-script-code"></code></pre></div>');

      root.innerHTML = html.join('');
      if (state.previewOpen) {
        document.getElementById('ic-script-code').textContent = buildScript(state);
      }
      bindEvents();
    }

    function bindEvents() {
      Array.prototype.forEach.call(root.querySelectorAll('[data-os]'), function (btn) {
        btn.addEventListener('click', function () { state.os = btn.getAttribute('data-os'); render(); });
      });

      var wmaToggle = document.getElementById('ic-wma-toggle');
      if (wmaToggle) {
        wmaToggle.addEventListener('change', function () {
          state.wmaEnabled = wmaToggle.checked;
          if (!state.wmaEnabled) {
            state.wmaAtlases = [];
          } else if (state.wmaAtlases.length === 0) {
            state.wmaAtlases = ['wma_org'];
          }
          render();
        });
      }

      Array.prototype.forEach.call(root.querySelectorAll('[data-atlas]'), function (cb) {
        cb.addEventListener('change', function () {
          var aid = cb.getAttribute('data-atlas');
          if (cb.checked) {
            state.wmaEnabled = true;
            if (state.wmaAtlases.indexOf(aid) === -1) { state.wmaAtlases.push(aid); }
          } else {
            state.wmaAtlases = state.wmaAtlases.filter(function (x) { return x !== aid; });
            if (state.wmaAtlases.length === 0) { state.wmaEnabled = false; }
          }
          render();
        });
      });

      var slicerInput = document.getElementById('ic-slicer-path');
      if (slicerInput) { slicerInput.addEventListener('input', function () { state.slicerPath = slicerInput.value; }); }
      var modelInput = document.getElementById('ic-model-dir');
      if (modelInput) { modelInput.addEventListener('input', function () { state.modelAssetsDir = modelInput.value; }); }
      var mirrorSel = document.getElementById('ic-mirror');
      if (mirrorSel) {
        mirrorSel.addEventListener('change', function () {
          state.mirrorPreset = mirrorSel.value;
          state.mirrorPrefix = mirrorSel.value === 'custom' ? state.mirrorPrefix : (mirrorSel.value === 'direct' ? '' : mirrorSel.value);
          render();
        });
      }
      var customMir = document.getElementById('ic-custom-mirror');
      if (customMir) { customMir.addEventListener('input', function () { state.mirrorPrefix = customMir.value; }); }
      var proxyEl = document.getElementById('ic-proxy');
      if (proxyEl) { proxyEl.addEventListener('input', function () { state.proxyUrl = proxyEl.value; }); }
      var offlineEl = document.getElementById('ic-offline-dir');
      if (offlineEl) { offlineEl.addEventListener('input', function () { state.offlineDir = offlineEl.value; }); }
      var checkHere = document.getElementById('ic-check-script-dir');
      if (checkHere) { checkHere.addEventListener('change', function () { state.checkScriptDir = checkHere.checked; }); }
      var moreEl = document.getElementById('ic-more');
      if (moreEl) { moreEl.addEventListener('toggle', function () { state.moreOpen = moreEl.open; }); }

      var dlBtn = document.getElementById('ic-download-btn');
      if (dlBtn) {
        dlBtn.addEventListener('click', function () {
          var scriptText = buildScript(state);
          var filename = OS_DEFAULTS[state.os].scriptName;
          var blob = new Blob([scriptText], { type: 'text/plain;charset=utf-8' });
          var a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
        });
      }
      var copyCmdBtn = document.getElementById('ic-copy-cmd-btn');
      if (copyCmdBtn) {
        copyCmdBtn.addEventListener('click', function () {
          var def = OS_DEFAULTS[state.os];
          var cmd = state.os === 'windows'
            ? '%USERPROFILE%\\Downloads\\' + def.scriptName
            : 'chmod +x ~/Downloads/' + def.scriptName + ' && ~/Downloads/' + def.scriptName;
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(cmd);
          }
          copyCmdBtn.textContent = zh ? '已复制' : 'Copied';
          setTimeout(function () { copyCmdBtn.textContent = zh ? '复制命令' : 'Copy command'; }, 1800);
        });
      }
      var prevBtn = document.getElementById('ic-toggle-preview');
      if (prevBtn) {
        prevBtn.addEventListener('click', function () { state.previewOpen = !state.previewOpen; render(); });
      }
      var copyScriptBtn = document.getElementById('ic-copy-script-btn');
      if (copyScriptBtn) {
        copyScriptBtn.addEventListener('click', function () {
          var txt = buildScript(state);
          if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(txt); }
          copyScriptBtn.textContent = zh ? '已复制' : 'Copied';
          setTimeout(function () { copyScriptBtn.textContent = zh ? '复制' : 'Copy'; }, 1600);
        });
      }
    }

    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCustomizer);
  } else {
    initCustomizer();
  }
})();
