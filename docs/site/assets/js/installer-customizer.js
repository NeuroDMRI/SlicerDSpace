/* SlicerDSpace Website Visual Installer Customizer. Vanilla JS, offline-safe. */
(function () {
  'use strict';

  var zh = typeof document !== 'undefined' && document.documentElement &&
    String(document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0;
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
        size_bytes: 1804334817,
        sha256: 'e1bb6f42218c80705746bc677ba7962468801585af31f9ce712b2164c298e6cd',
        parts: []
      },
      wma_naba: {
        id: 'wma_naba',
        filename: 'SlicerDSpace-v0.1.1-WMA-NABA.tar.gz',
        url: '',
        size_bytes: 3365003883,
        sha256: '88a810213c5c3a047e712eb2f1a2a49f3de98000c6e7ca056ffe9dac43dacbc8',
        parts: [
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001',
            size_bytes: 1900000000,
            sha256: 'a948e9d82eedcb3daec6ed7a0c5e5ec866d10d4070373aa7cbf4baeb2e3cf0c1',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.002',
            size_bytes: 1465003883,
            sha256: 'bdc60b267630cd737a6a9e8cf22db771147a5daba6c9e623998d0b5f4c5985c1',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.002'
          }
        ]
      },
      wma_east_west: {
        id: 'wma_east_west',
        filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz',
        url: '',
        size_bytes: 11405353584,
        sha256: '506cf84e1608a5975e72b83af4d5e25f7986849d12159d7a94d2def9681e33c1',
        parts: [
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001',
            size_bytes: 1900000000,
            sha256: '6daf10aba6b3bc530dc136c67e4da2d8b45f7b89d8cd4203de28bdf71b703d8f',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.002',
            size_bytes: 1900000000,
            sha256: '805e74a323dac63573759c362c3de81296ac1712482858d021dff8d986b2b670',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.002'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.003',
            size_bytes: 1900000000,
            sha256: '24c27591a46c54f387fd86045a564ee4241abf9a51d4cd8f30066a7b712e0071',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.003'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.004',
            size_bytes: 1900000000,
            sha256: '4a060647f641c12309953bb3ba887a23ba51ef4dac9d911ac42428fdda1f3932',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.004'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.005',
            size_bytes: 1900000000,
            sha256: '4fd4650b67bc51d49869e10363dba988fcad2e071ffce5fea6e1c3254680fb95',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.005'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.006',
            size_bytes: 1900000000,
            sha256: '5ad69210366044801eb3be30ad26a1744c3bf658f8b1388ec3dfe0a607039a44',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.006'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.007',
            size_bytes: 5353584,
            sha256: '2663479e6cd9fddaa0df5360646c86e7b2a31643c78c5a0eae3af715370cb455',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.007'
          }
        ]
      }
    },
    // __ARTIFACTS_BUNDLES_BEGIN__
    bundles: {
      windows_x64_base: {
        filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip',
        size_bytes: 1468710623,
        sha256: 'c6717c65235b060da35ab8c821a8b3032b8a9a29ef53257f2db34714e5b122cc',
        parts: []
      },
      linux_x64_base: {
        filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz',
        size_bytes: 1613619026,
        sha256: 'f6cfa723b320a7e1402d654d4d381c9e6f756aee41a011af07b9c1e34fe0b4db',
        parts: []
      },
      macos_arm64_base: {
        filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz',
        size_bytes: 1463950758,
        sha256: 'dfba026130532a4a13919d9dcec7dba9340148ed94501781a2d15e7ceeb27da6',
        parts: []
      }
    }
    // __ARTIFACTS_BUNDLES_END__
  };

  var OS_DEFAULTS = {
    macos: {
      label: 'macOS',
      s4extOs: 'macosx',
      bundleKey: 'macos_arm64_base',
      slicerPlaceholder: '/Applications/Slicer.app',
      modelPlaceholder: '~/Library/Application Support/SlicerDSpace/model-assets',
      setupZipName: 'SlicerDSpaceSetup-macOS.zip',
      scriptName: 'Install-SlicerDSpace.command',
      baseEstimateBytes: 1463950758
    },
    windows: {
      label: 'Windows',
      s4extOs: 'win',
      bundleKey: 'windows_x64_base',
      slicerPlaceholder: '%LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0',
      modelPlaceholder: '%LOCALAPPDATA%\\SlicerDSpace\\model-assets',
      setupZipName: 'SlicerDSpaceSetup-Windows.zip',
      scriptName: 'Install-SlicerDSpace.bat',
      baseEstimateBytes: 1468710623
    },
    linux: {
      label: 'Linux',
      s4extOs: 'linux',
      bundleKey: 'linux_x64_base',
      slicerPlaceholder: '~/Slicer-5.10.0-linux-amd64',
      modelPlaceholder: '~/.local/share/SlicerDSpace/model-assets',
      setupZipName: 'SlicerDSpaceSetup-Linux.zip',
      scriptName: 'Install-SlicerDSpace.sh',
      baseEstimateBytes: 1613619026
    }
  };

  // Shared parallel HTTP Range downloader + SHA-256 verification reused by GUI and CLI installers.
  var SHARED_DOWNLOAD_PY = [
    '_GITHUB_PREFIXES = ("https://github.com/", "https://raw.githubusercontent.com/", "https://objects.githubusercontent.com/", "https://media.githubusercontent.com/", "https://codeload.github.com/")',
    '_AUTO_FALLBACK_MIRRORS = ("https://ghfast.top/", "https://gh-proxy.com/")',
    '_PAR_CHUNK = 4194304',
    '_PAR_WORKERS = 12',
    '_SYS_PROXY_CACHE = [None]',
    'def _sha256(path, cancel_event=None):',
    '    h = hashlib.sha256()',
    '    with path.open("rb") as fh:',
    '        for chunk in iter(lambda: fh.read(1048576), b""):',
    '            if cancel_event is not None and cancel_event.is_set(): raise InterruptedError("Cancelled")',
    '            h.update(chunk)',
    '    return h.hexdigest()',
    'def _wait_cancel(cancel_event, secs):',
    '    if cancel_event is not None: cancel_event.wait(secs)',
    '    else: time.sleep(secs)',
    'def _clean_mirror(p):',
    '    p = str(p or "").strip()',
    '    return (p.rstrip("/") + "/") if (p.startswith("https://") or p.startswith("http://")) else ""',
    'def _expand_urls(urls, mirror_prefix="", allow_mirror=True):',
    '    raw = [urls] if isinstance(urls, str) else list(urls or ())',
    '    mirror = _clean_mirror(mirror_prefix)',
    '    out = []',
    '    if allow_mirror and mirror:',
    '        for u in raw:',
    '            if u.startswith(_GITHUB_PREFIXES) and not u.startswith(mirror):',
    '                m = mirror + u',
    '                if m not in out: out.append(m)',
    '    for u in raw:',
    '        if u and u not in out: out.append(u)',
    '    if allow_mirror and not mirror:',
    '        for fb in _AUTO_FALLBACK_MIRRORS:',
    '            for u in raw:',
    '                if u.startswith(_GITHUB_PREFIXES) and not u.startswith(fb):',
    '                    m = fb + u',
    '                    if m not in out: out.append(m)',
    '    return out',
    'def _clean_proxy(raw):',
    '    s = str(raw or "").strip()',
    '    if not s: return ""',
    '    if "://" not in s and "=" in s:',
    '        m = {}',
    '        for item in s.split(";"):',
    '            p = item.strip()',
    '            if "=" in p:',
    '                k, v = p.split("=", 1)',
    '                if k.strip() and v.strip(): m[k.strip().lower()] = v.strip()',
    '        s = m.get("https") or m.get("http") or next(iter(m.values()), "")',
    '        if not s: return ""',
    '    return s if "://" in s else ("http" + "://" + s)',
    'def _ssl_context():',
    '    for env_k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):',
    '        cf = str(os.environ.get(env_k) or "").strip()',
    '        if cf and Path(cf).is_file():',
    '            try: return ssl.create_default_context(cafile=cf)',
    '            except Exception: pass',
    '    try:',
    '        import certifi',
    '        cf = certifi.where()',
    '        if cf and Path(cf).is_file(): return ssl.create_default_context(cafile=cf)',
    '    except Exception: pass',
    '    return ssl.create_default_context()',
    'def _detect_system_proxy():',
    '    if _SYS_PROXY_CACHE[0] is not None: return _SYS_PROXY_CACHE[0]',
    '    found = ""',
    '    try:',
    '        px = urllib.request.getproxies() or {}',
    '        found = _clean_proxy(px.get("https") or px.get("http") or px.get("all") or "")',
    '    except Exception:',
    '        found = ""',
    '    if not found and sys.platform == "darwin" and shutil.which("scutil"):',
    '        try:',
    '            proc = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=3)',
    '            if proc.returncode == 0:',
    '                kv = {}',
    '                for ln in (proc.stdout or "").splitlines():',
    '                    if ":" in ln:',
    '                        k, v = ln.split(":", 1)',
    '                        kv[k.strip()] = v.strip()',
    '                if kv.get("HTTPSEnable") == "1" and kv.get("HTTPSProxy") and kv.get("HTTPSPort"):',
    '                    found = _clean_proxy(kv["HTTPSProxy"] + ":" + kv["HTTPSPort"])',
    '                elif kv.get("HTTPEnable") == "1" and kv.get("HTTPProxy") and kv.get("HTTPPort"):',
    '                    found = _clean_proxy(kv["HTTPProxy"] + ":" + kv["HTTPPort"])',
    '        except Exception:',
    '            found = ""',
    '    elif not found and sys.platform == "win32":',
    '        try:',
    '            import winreg',
    '            kp = r"Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings"',
    '            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, kp) as rk:',
    '                en, _ = winreg.QueryValueEx(rk, "ProxyEnable")',
    '                srv, _ = winreg.QueryValueEx(rk, "ProxyServer")',
    '                if int(en or 0) == 1 and srv: found = _clean_proxy(srv)',
    '        except Exception:',
    '            found = ""',
    '    _SYS_PROXY_CACHE[0] = found',
    '    return found',
    'def _is_loopback_url(req):',
    '    u = str(getattr(req, "full_url", req) or "").lower()',
    '    host = u.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0] if "://" in u else ""',
    '    return host in ("127.0.0.1", "localhost")',
    'def _open_url(req, proxy_url="", timeout=60):',
    '    explicit_proxy = _clean_proxy(proxy_url)',
    '    proxy = explicit_proxy or ("" if _is_loopback_url(req) else _clean_proxy(_detect_system_proxy()))',
    '    if proxy.lower().startswith("socks"):',
    '        raise RuntimeError(f"SOCKS proxy {proxy!r} requires PySocks; set an HTTP/HTTPS proxy.")',
    '    ctx = _ssl_context()',
    '    if proxy:',
    '        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}), urllib.request.HTTPSHandler(context=ctx)).open(req, timeout=timeout)',
    '    return urllib.request.urlopen(req, timeout=timeout, context=ctx)',
    'def _is_intel_mac():',
    '    if sys.platform != "darwin": return False',
    '    mach = str(platform.machine() or "").lower()',
    '    if mach not in ("x86_64", "amd64"): return False',
    '    try:',
    '        out = subprocess.check_output(["sysctl", "-n", "sysctl.proc_translated"], stderr=subprocess.DEVNULL, text=True, timeout=2).strip()',
    '        if out == "1": return False',
    '    except Exception: pass',
    '    return True',
    'def _pick_cache_dir(need_bytes=0):',
    '    cands = [Path(tempfile.gettempdir()) / "slicerdspace-installer-cache"]',
    '    fb = (Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SlicerDSpace" / "installer-cache") if sys.platform == "win32" else (Path.home() / ".cache" / "slicerdspace-installer-cache")',
    '    if fb not in cands: cands.append(fb)',
    '    min_pref = max(int(need_bytes or 0), 2147483648)',
    '    best = None; best_free = -1',
    '    for cand in cands:',
    '        try:',
    '            cand.mkdir(parents=True, exist_ok=True)',
    '            free = int(shutil.disk_usage(cand).free)',
    '            if free >= min_pref: return cand',
    '            if free > best_free: best, best_free = cand, free',
    '        except OSError: pass',
    '    if best is None: raise OSError("No writable installer cache directory available")',
    '    if need_bytes and best_free < int(need_bytes):',
    '        raise RuntimeError(f"Insufficient disk space in {best}: {best_free / 1073741824.0:.2f} GB free, need {need_bytes / 1073741824.0:.2f} GB")',
    '    return best',
    'def _probe_range(url, proxy_url="", expected_size=0, timeout=12):',
    '    total = int(expected_size or 0)',
    '    if expected_size and total <= _PAR_CHUNK: return False, total, None',
    '    try:',
    '        req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-installer", "Range": "bytes=0-0"})',
    '        with _open_url(req, proxy_url=proxy_url, timeout=timeout) as resp:',
    '            if getattr(resp, "status", 200) != 206: return False, total, None',
    '            cr = str((resp.headers.get("Content-Range") if hasattr(resp, "headers") else "") or "")',
    '            if "/" in cr:',
    '                tot_s = cr.rsplit("/", 1)[1].strip()',
    '                if tot_s.isdigit():',
    '                    ptot = int(tot_s)',
    '                    if total and ptot != total: return False, total, None',
    '                    total = ptot',
    '            return total > _PAR_CHUNK, total, None',
    '    except Exception as exc:',
    '        return False, total, exc',
    'def _download_parallel(url, part, meta_path, total, proxy_url="", progress_cb=None, cancel_event=None, log_cb=None, name=""):',
    '    num_chunks = (total + _PAR_CHUNK - 1) // _PAR_CHUNK',
    '    done_set = set()',
    '    def _save_meta(strict=False):',
    '        tmp_m = meta_path.with_name(meta_path.name + ".tmp")',
    '        try:',
    '            tmp_m.write_text(json.dumps({"total_size": total, "chunk_size": _PAR_CHUNK, "done": sorted(done_set)}), encoding="utf-8")',
    '            os.replace(tmp_m, meta_path)',
    '        except OSError:',
    '            if strict: raise',
    '    if part.is_file() and meta_path.is_file():',
    '        try:',
    '            m = json.loads(meta_path.read_text(encoding="utf-8"))',
    '            if int(m.get("total_size") or 0) == total and int(m.get("chunk_size") or 0) == _PAR_CHUNK and part.stat().st_size == total:',
    '                done_set = {int(i) for i in (m.get("done") or ()) if 0 <= int(i) < num_chunks}',
    '            else:',
    '                meta_path.unlink(missing_ok=True); part.unlink(missing_ok=True)',
    '        except Exception:',
    '            meta_path.unlink(missing_ok=True); part.unlink(missing_ok=True)',
    '    elif part.is_file():',
    '        legacy_sz = part.stat().st_size',
    '        if 0 < legacy_sz <= total:',
    '            done_set = set(range(num_chunks)) if legacy_sz == total else set(range(legacy_sz // _PAR_CHUNK))',
    '            if log_cb: log_cb(f"Migrating {len(done_set)}/{num_chunks} cached chunks from {part.name}")',
    '            with part.open("r+b") as fh: fh.truncate(total)',
    '            _save_meta(strict=True)',
    '        else:',
    '            part.unlink(missing_ok=True)',
    '    if not part.is_file() or part.stat().st_size != total:',
    '        with part.open("wb") as fh: fh.truncate(total)',
    '        done_set = set(); _save_meta(strict=True)',
    '    done_bytes = [sum(min(_PAR_CHUNK, total - c * _PAR_CHUNK) for c in done_set)]',
    '    todo_q = queue.Queue()',
    '    for cid in range(num_chunks):',
    '        if cid not in done_set: todo_q.put_nowait(cid)',
    '    if progress_cb: progress_cb(done_bytes[0], total, name)',
    '    if todo_q.empty(): return',
    '    n_w = min(_PAR_WORKERS, todo_q.qsize())',
    '    if log_cb: log_cb(f"Parallel download ({n_w} threads, {len(done_set)}/{num_chunks} chunks cached): {name} <- {url}")',
    '    state_lock = threading.Lock(); in_flight = {}; first_err = []; closed = [False]; last_meta_t = [time.monotonic()]',
    '    part_fh = part.open("r+b")',
    '    def _worker():',
    '        while True:',
    '            if (cancel_event is not None and cancel_event.is_set()) or first_err or closed[0]: return',
    '            try: cid = todo_q.get_nowait()',
    '            except queue.Empty: return',
    '            start = cid * _PAR_CHUNK; exp_len = min(_PAR_CHUNK, total - start); end = start + exp_len - 1',
    '            last_exc = None',
    '            for attempt in range(4):',
    '                if (cancel_event is not None and cancel_event.is_set()) or first_err or closed[0]: return',
    '                try:',
    '                    req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-installer", "Range": f"bytes={start}-{end}"})',
    '                    buf = bytearray()',
    '                    with _open_url(req, proxy_url=proxy_url, timeout=25) as resp:',
    '                        if getattr(resp, "status", 200) != 206: raise RuntimeError(f"Expected 206, got {getattr(resp, \'status\', 200)}")',
    '                        while len(buf) < exp_len:',
    '                            if (cancel_event is not None and cancel_event.is_set()) or first_err or closed[0]: return',
    '                            blk = resp.read(min(65536, exp_len - len(buf)))',
    '                            if not blk: break',
    '                            buf.extend(blk)',
    '                            with state_lock: in_flight[cid] = len(buf)',
    '                    if len(buf) != exp_len: raise RuntimeError(f"Short chunk {cid}: {len(buf)}/{exp_len}")',
    '                    with state_lock:',
    '                        if closed[0]: return',
    '                        part_fh.seek(start); part_fh.write(buf)',
    '                        done_set.add(cid); done_bytes[0] += exp_len; in_flight.pop(cid, None)',
    '                        now = time.monotonic()',
    '                        if now - last_meta_t[0] >= 1.0: _save_meta(strict=False); last_meta_t[0] = now',
    '                    last_exc = None; break',
    '                except Exception as exc:',
    '                    last_exc = exc',
    '                    with state_lock: in_flight.pop(cid, None)',
    '                    _wait_cancel(cancel_event, 0.6 * (attempt + 1))',
    '            if last_exc is not None:',
    '                with state_lock:',
    '                    if not first_err: first_err.append(last_exc)',
    '                return',
    '    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(n_w)]',
    '    try:',
    '        for t in threads: t.start()',
    '        while any(t.is_alive() for t in threads):',
    '            if cancel_event is not None and cancel_event.is_set(): raise InterruptedError("Cancelled")',
    '            with state_lock:',
    '                if first_err: raise first_err[0]',
    '                cur_done = min(total, done_bytes[0] + sum(in_flight.values()))',
    '            if progress_cb: progress_cb(cur_done, total, name)',
    '            _wait_cancel(cancel_event, 0.15)',
    '        for t in threads: t.join()',
    '        if cancel_event is not None and cancel_event.is_set(): raise InterruptedError("Cancelled")',
    '        if first_err: raise first_err[0]',
    '        if len(done_set) != num_chunks: raise RuntimeError(f"Incomplete parallel download {len(done_set)}/{num_chunks}")',
    '        if progress_cb: progress_cb(total, total, name)',
    '    finally:',
    '        with state_lock:',
    '            closed[0] = True',
    '            try: part_fh.flush(); part_fh.close()',
    '            except Exception: pass',
    '            _save_meta(strict=False)',
    'def _download_stream(url, part, meta_path, expected_size=0, proxy_url="", timeout=60, progress_cb=None, cancel_event=None, log_cb=None, name=""):',
    '    if meta_path.is_file():',
    '        try:',
    '            m = json.loads(meta_path.read_text(encoding="utf-8"))',
    '            ds = {int(i) for i in (m.get("done") or ())}; csz = int(m.get("chunk_size") or _PAR_CHUNK); contig = 0',
    '            while contig in ds: contig += 1',
    '            if contig > 0 and part.is_file():',
    '                with part.open("r+b") as fh: fh.truncate(min(int(expected_size or part.stat().st_size), contig * csz))',
    '            else: part.unlink(missing_ok=True)',
    '        except Exception: part.unlink(missing_ok=True)',
    '        meta_path.unlink(missing_ok=True)',
    '    if log_cb: log_cb(f"Downloading {name} <- {url}")',
    '    resume_from = part.stat().st_size if part.is_file() else 0',
    '    if expected_size and resume_from > expected_size: part.unlink(missing_ok=True); resume_from = 0',
    '    headers = {"User-Agent": "SlicerDSpace-installer"}',
    '    if resume_from: headers["Range"] = f"bytes={resume_from}-"',
    '    with _open_url(urllib.request.Request(url, headers=headers), proxy_url=proxy_url, timeout=timeout) as resp:',
    '        append = bool(resume_from and getattr(resp, "status", 200) == 206)',
    '        if resume_from and not append: resume_from = 0',
    '        clen = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None',
    '        total = (int(clen) + resume_from) if (clen and str(clen).isdigit()) else int(expected_size or 0)',
    '        done = resume_from; last_t = 0.0',
    '        if progress_cb: progress_cb(done, total, name)',
    '        with part.open("ab" if append else "wb") as fh:',
    '            while True:',
    '                if cancel_event is not None and cancel_event.is_set(): raise InterruptedError("Cancelled")',
    '                chunk = resp.read(65536)',
    '                if not chunk: break',
    '                fh.write(chunk); done += len(chunk)',
    '                now = time.monotonic()',
    '                if progress_cb and (now - last_t >= 0.15 or (total and done >= total)):',
    '                    progress_cb(done, total, name); last_t = now',
    '    if total and done < total: raise RuntimeError(f"Short read {done}/{total}")',
    'def _download(urls, dest, expected_sha256="", expected_size=0, mirror_prefix=None, proxy_url=None, allow_mirror=True, timeout=60, progress_cb=None, cancel_event=None, log_cb=None):',
    '    if mirror_prefix is None: mirror_prefix = CONFIG.get("mirror_prefix", "")',
    '    if proxy_url is None: proxy_url = CONFIG.get("proxy_url", "")',
    '    if log_cb is None and "_log" in globals(): log_cb = globals()["_log"]',
    '    dest.parent.mkdir(parents=True, exist_ok=True)',
    '    if dest.is_file():',
    '        if expected_sha256 and _sha256(dest, cancel_event=cancel_event) == expected_sha256:',
    '            if log_cb: log_cb(f"Reusing verified file: {dest.name}")',
    '            if progress_cb: progress_cb(int(expected_size or dest.stat().st_size), int(expected_size or dest.stat().st_size), dest.name)',
    '            return dest',
    '        if not expected_sha256 and expected_size and dest.stat().st_size == expected_size:',
    '            if log_cb: log_cb(f"Reusing file: {dest.name}")',
    '            if progress_cb: progress_cb(int(expected_size), int(expected_size), dest.name)',
    '            return dest',
    '    part = dest.with_name(dest.name + ".part"); meta_path = dest.with_name(dest.name + ".part.json"); errors = []',
    '    for url in _expand_urls(urls, mirror_prefix=mirror_prefix, allow_mirror=allow_mirror):',
    '        if cancel_event is not None and cancel_event.is_set(): raise InterruptedError("Cancelled")',
    '        try:',
    '            if log_cb: log_cb(f"Connecting: {url}")',
    '            probe_ok, total, probe_err = _probe_range(url, proxy_url=proxy_url, expected_size=expected_size, timeout=min(timeout, 12))',
    '            if probe_err is not None:',
    '                is_soft = (isinstance(probe_err, urllib.error.HTTPError) and probe_err.code in (400, 405, 501)) or isinstance(probe_err, (TimeoutError, socket.timeout)) or (isinstance(probe_err, urllib.error.URLError) and isinstance(getattr(probe_err, "reason", None), (TimeoutError, socket.timeout)))',
    '                if is_soft:',
    '                    if log_cb: log_cb(f"Range probe fallback to stream ({probe_err})")',
    '                    probe_ok = False',
    '                else:',
    '                    if isinstance(probe_err, urllib.error.HTTPError) and probe_err.code == 416:',
    '                        meta_path.unlink(missing_ok=True); part.unlink(missing_ok=True)',
    '                    raise probe_err',
    '            if probe_ok: _download_parallel(url, part, meta_path, total, proxy_url=proxy_url, progress_cb=progress_cb, cancel_event=cancel_event, log_cb=log_cb, name=dest.name)',
    '            else: _download_stream(url, part, meta_path, expected_size=expected_size, proxy_url=proxy_url, timeout=timeout, progress_cb=progress_cb, cancel_event=cancel_event, log_cb=log_cb, name=dest.name)',
    '            if expected_sha256:',
    '                if log_cb: log_cb(f"Verifying SHA-256: {dest.name}")',
    '                actual = _sha256(part, cancel_event=cancel_event)',
    '                if actual != expected_sha256: meta_path.unlink(missing_ok=True); part.unlink(missing_ok=True); raise ValueError(f"SHA-256 mismatch for {dest.name}")',
    '            meta_path.unlink(missing_ok=True); os.replace(part, dest)',
    '            if progress_cb: progress_cb(int(expected_size or dest.stat().st_size), int(expected_size or dest.stat().st_size), dest.name)',
    '            return dest',
    '        except InterruptedError: raise',
    '        except urllib.error.HTTPError as exc:',
    '            if exc.code == 416: meta_path.unlink(missing_ok=True); part.unlink(missing_ok=True)',
    '            if log_cb: log_cb(f"Source failed ({exc}); trying next...")',
    '            errors.append(f"{url}: {exc}")',
    '        except Exception as exc:',
    '            if log_cb: log_cb(f"Source failed ({exc}); trying next...")',
    '            errors.append(f"{url}: {exc}")',
    '    raise RuntimeError(f"Failed downloading {dest.name}: {errors}")'
  ].join('\n');

  // Shared graphical Slicer extension module injected into SlicerDSpaceSetup-<OS>.zip.
  var SETUP_MODULE_PY = [
    'import hashlib, json, os, platform, queue, shutil, socket, ssl, subprocess, sys, tempfile, threading, time, traceback, urllib.error, urllib.request',
    'from pathlib import Path',
    'import qt, slicer',
    'from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget',
    '_ACTIVE_DIALOG = None',
    SHARED_DOWNLOAD_PY,
    'def _search_dirs(offline_dir=""):',
    '    dirs = []',
    '    raw = str(offline_dir or "").strip()',
    '    if raw:',
    '        p = Path(raw).expanduser()',
    '        if p.is_dir(): dirs.append(p.resolve())',
    '    dl = Path.home() / "Downloads"',
    '    if dl.is_dir() and dl.resolve() not in dirs: dirs.append(dl.resolve())',
    '    return dirs',
    'def _acquire_spec(spec, cache_dir, offline_dir="", mirror_prefix="", proxy_url="", progress_cb=None, cancel_event=None, log_cb=None):',
    '    filename = spec["filename"]; expected_sha = str(spec.get("sha256") or ""); expected_size = int(spec.get("size_bytes") or 0)',
    '    parts = spec.get("parts") or []; dirs = _search_dirs(offline_dir)',
    '    for d in dirs:',
    '        c = d / filename',
    '        if c.is_file() and (not expected_sha or _sha256(c, cancel_event=cancel_event) == expected_sha):',
    '            if log_cb: log_cb(f"Using offline archive: {c}")',
    '            if progress_cb: progress_cb(expected_size, expected_size, filename)',
    '            return c, []',
    '    cached = cache_dir / filename',
    '    if cached.is_file() and expected_sha and _sha256(cached, cancel_event=cancel_event) == expected_sha:',
    '        if log_cb: log_cb(f"Reusing cache: {cached}")',
    '        if progress_cb: progress_cb(expected_size, expected_size, filename)',
    '        return cached, [cached]',
    '    if parts:',
    '        part_paths = []; dl_parts = []; done_parts = 0',
    '        for ps in parts:',
    '            pn = ps["filename"]; psha = str(ps.get("sha256") or ""); psize = int(ps.get("size_bytes") or 0); found = None',
    '            for d in dirs:',
    '                c = d / pn',
    '                if c.is_file() and (not psha or _sha256(c, cancel_event=cancel_event) == psha):',
    '                    if log_cb: log_cb(f"Using offline part: {c}")',
    '                    found = c; break',
    '            if found is None:',
    '                def _part_cb(d_bytes, _t, name, base=done_parts):',
    '                    if progress_cb: progress_cb(base + d_bytes, expected_size, name)',
    '                found = _download(ps["url"], cache_dir / pn, expected_sha256=psha, expected_size=psize, mirror_prefix=mirror_prefix, proxy_url=proxy_url, progress_cb=_part_cb, cancel_event=cancel_event, log_cb=log_cb)',
    '                dl_parts.append(found)',
    '            done_parts += int(psize or found.stat().st_size)',
    '            if progress_cb: progress_cb(min(done_parts, expected_size), expected_size, pn)',
    '            part_paths.append(found)',
    '        if log_cb: log_cb(f"Merging {len(part_paths)} volumes into {filename}...")',
    '        merging = cache_dir / (filename + ".merging")',
    '        with merging.open("wb") as out_fh:',
    '            for pp in part_paths:',
    '                if cancel_event is not None and cancel_event.is_set(): merging.unlink(missing_ok=True); raise InterruptedError("Cancelled")',
    '                with pp.open("rb") as in_fh: shutil.copyfileobj(in_fh, out_fh, length=16777216)',
    '        if expected_sha and _sha256(merging, cancel_event=cancel_event) != expected_sha:',
    '            merging.unlink(missing_ok=True); raise RuntimeError(f"Merged SHA-256 mismatch: {filename}")',
    '        os.replace(merging, cached)',
    '        return cached, [*dl_parts, cached]',
    '    if not spec.get("url"): raise RuntimeError(f"Download URL is not configured yet for {filename}")',
    '    arc = _download(spec["url"], cached, expected_sha256=expected_sha, expected_size=expected_size, mirror_prefix=mirror_prefix, proxy_url=proxy_url, progress_cb=progress_cb, cancel_event=cancel_event, log_cb=log_cb)',
    '    return arc, [arc]',
    'def _cleanup_paths(paths):',
    '    for p in paths:',
    '        try: Path(p).unlink(missing_ok=True)',
    '        except OSError: pass',
    'def _dequarantine(target):',
    '    if sys.platform == "darwin" and shutil.which("xattr"):',
    '        try: subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(target)], capture_output=True)',
    '        except Exception: pass',
    'def _resolve_mod_root(em):',
    '    for mp in (em.extensionModulePaths("SlicerDSpace") or ()):',
    '        if (Path(mp) / "SlicerDSpace.py").is_file(): return Path(mp).resolve()',
    '    raw_ep = em.extensionsInstallPath() if callable(getattr(em, "extensionsInstallPath", None)) else getattr(em, "extensionsInstallPath", "")',
    '    for base_s in (raw_ep, getattr(slicer.app, "extensionsInstallPath", "")):',
    '        if not base_s: continue',
    '        for cand in Path(str(base_s)).glob("SlicerDSpace/lib/Slicer-*/qt-scripted-modules"):',
    '            if (cand / "SlicerDSpace.py").is_file(): return cand.resolve()',
    '    raise FileNotFoundError("Installed SlicerDSpace module directory not found")',
    'def _is_slicer_510():',
    '    try: return int(getattr(slicer.app, "majorVersion", 0)) == 5 and int(getattr(slicer.app, "minorVersion", 0)) == 10',
    '    except Exception: return True',
    'def _host_bundle_key():',
    '    if sys.platform == "darwin": return "macos_arm64_base"',
    '    if sys.platform == "win32": return "windows_x64_base"',
    '    return "linux_x64_base"',
    'def show_setup_dialog(auto_start=False):',
    '    global _ACTIVE_DIALOG',
    '    if _ACTIVE_DIALOG is not None:',
    '        try: _ACTIVE_DIALOG.show(); _ACTIVE_DIALOG.raise_(); _ACTIVE_DIALOG.activateWindow(); return _ACTIVE_DIALOG',
    '        except Exception: _ACTIVE_DIALOG = None',
    '    parent_win = slicer.util.mainWindow() if hasattr(slicer.util, "mainWindow") else None',
    '    dlg = qt.QDialog(parent_win)',
    '    dlg.setWindowTitle("SlicerDSpace Setup")',
    '    dlg.setMinimumWidth(560)',
    '    _ACTIVE_DIALOG = dlg',
    '    layout = qt.QVBoxLayout(dlg)',
    '    layout.setSpacing(10)',
    '    title = qt.QLabel("<b>SlicerDSpace 5.10.0 One-Click Installer</b>")',
    '    subtitle = qt.QLabel("Installs the unified SlicerDSpace extension and selected white-matter atlases with resume and checksum verification.")',
    '    subtitle.setWordWrap(True)',
    '    layout.addWidget(title); layout.addWidget(subtitle)',
    '    slicer_ok = _is_slicer_510()',
    '    intel_mac = _is_intel_mac()',
    '    ver_ok = slicer_ok and (not intel_mac)',
    '    if not slicer_ok:',
    '        warn = qt.QLabel("This setup requires 3D Slicer 5.10.x. Please open 3D Slicer 5.10.0.")',
    '        warn.setWordWrap(True); warn.setStyleSheet("color: #f87171; font-weight: bold;")',
    '        layout.addWidget(warn)',
    '    if intel_mac:',
    '        warn_m = qt.QLabel("Intel x86_64 Macs are not supported by the Apple Silicon macOS bundle (requires M1/M2/M3/M4).")',
    '        warn_m.setWordWrap(True); warn_m.setStyleSheet("color: #f87171; font-weight: bold;")',
    '        layout.addWidget(warn_m)',
    '    _arts = CONFIG.get("artifacts") or {}',
    '    _b_bytes = int(((_arts.get("bundles") or {}).get(_host_bundle_key()) or {}).get("size_bytes") or 0) + int(((_arts.get("iit_template") or {})).get("size_bytes") or 0)',
    '    _b_gb = f"~{_b_bytes / 1073741824.0:.2f} GB" if _b_bytes > 0 else "~1.37 GB"',
    '    base_box = qt.QGroupBox("Core Extension")',
    '    base_lay = qt.QVBoxLayout(base_box)',
    '    base_cb = qt.QCheckBox(f"SlicerDSpace Base Features ({_b_gb})")',
    '    base_cb.setChecked(True); base_cb.setEnabled(False); base_lay.addWidget(base_cb)',
    '    layout.addWidget(base_box)',
    '    atlas_box = qt.QGroupBox("Optional WMA White-Matter Atlases")',
    '    atlas_lay = qt.QVBoxLayout(atlas_box)',
    '    preselected = set(CONFIG.get("selected_packs") or ())',
    '    atlas_checks = {}',
    '    for aid, alabel in (("wma_org", "ORG Standard 74-Tract Atlas (~1.68 GB, recommended)"), ("wma_naba", "NABA Lifespan Atlas (~3.13 GB)"), ("wma_east_west", "East-West Cross-Population Atlas (~10.62 GB)")):',
    '        cb = qt.QCheckBox(alabel); cb.setChecked(aid in preselected); atlas_checks[aid] = cb; atlas_lay.addWidget(cb)',
    '    layout.addWidget(atlas_box)',
    '    net_box = qt.QGroupBox("Network Mirror && Offline Directory")',
    '    net_form = qt.QFormLayout(net_box)',
    '    mirror_combo = qt.QComboBox()',
    '    mirror_items = [("GitHub Direct", ""), ("ghfast.top (Mirror)", "https://ghfast.top/"), ("gh-proxy.com (Mirror)", "https://gh-proxy.com/"), ("Custom...", "__custom__")]',
    '    for lbl, val in mirror_items: mirror_combo.addItem(lbl, val)',
    '    init_mirror = str(CONFIG.get("mirror_prefix") or "").strip()',
    '    custom_mirror_edit = qt.QLineEdit(init_mirror)',
    '    custom_mirror_edit.setPlaceholderText("https://your-mirror.example/")',
    '    idx = next((i for i, (_, v) in enumerate(mirror_items) if v and init_mirror.rstrip("/") + "/" == v), 0 if not init_mirror else 3)',
    '    mirror_combo.setCurrentIndex(idx)',
    '    custom_mirror_edit.setVisible(idx == 3)',
    '    def _on_mirror_change(index): custom_mirror_edit.setVisible(str(mirror_combo.itemData(index) or "") == "__custom__")',
    '    mirror_combo.currentIndexChanged.connect(_on_mirror_change)',
    '    mirror_box = qt.QWidget(); mirror_lay = qt.QVBoxLayout(mirror_box); mirror_lay.setContentsMargins(0, 0, 0, 0); mirror_lay.setSpacing(4)',
    '    mirror_lay.addWidget(mirror_combo); mirror_lay.addWidget(custom_mirror_edit)',
    '    net_form.addRow("Mirror", mirror_box)',
    '    proxy_edit = qt.QLineEdit(str(CONFIG.get("proxy_url") or "")); proxy_edit.setPlaceholderText("http://127.0.0.1:7890")',
    '    net_form.addRow("Proxy (opt.)", proxy_edit)',
    '    off_widget = qt.QWidget(); off_row = qt.QHBoxLayout(off_widget); off_row.setContentsMargins(0, 0, 0, 0)',
    '    off_edit = qt.QLineEdit(str(CONFIG.get("offline_dir") or "")); off_edit.setPlaceholderText("Optional folder with pre-downloaded packages")',
    '    off_btn = qt.QPushButton("Browse...")',
    '    def _browse_off():',
    '        picked = qt.QFileDialog.getExistingDirectory(dlg, "Select Offline Package Folder")',
    '        if picked: off_edit.setText(picked)',
    '    off_btn.clicked.connect(_browse_off)',
    '    off_row.addWidget(off_edit, 1); off_row.addWidget(off_btn)',
    '    net_form.addRow("Offline Dir", off_widget)',
    '    layout.addWidget(net_box)',
    '    status_lbl = qt.QLabel("Ready to install.")',
    '    status_lbl.setWordWrap(True)',
    '    pbar = qt.QProgressBar(); pbar.setRange(0, 1000); pbar.setValue(0)',
    '    log_box = qt.QPlainTextEdit(); log_box.setReadOnly(True); log_box.setMaximumHeight(115)',
    '    layout.addWidget(status_lbl); layout.addWidget(pbar); layout.addWidget(log_box)',
    '    btn_row = qt.QHBoxLayout()',
    '    start_btn = qt.QPushButton("Start Install"); start_btn.setDefault(True)',
    '    cancel_btn = qt.QPushButton("Cancel")',
    '    btn_row.addStretch(1); btn_row.addWidget(cancel_btn); btn_row.addWidget(start_btn)',
    '    layout.addLayout(btn_row)',
    '    if not ver_ok: start_btn.setEnabled(False)',
    '    state = {"running": False, "done": False, "cancel": threading.Event(), "queue": queue.Queue()}',
    '    poll_timer = qt.QTimer(dlg); poll_timer.setInterval(120)',
    '    def _append_log(line): log_box.appendPlainText(str(line))',
    '    def _set_controls_enabled(enabled):',
    '        for cb in atlas_checks.values(): cb.setEnabled(enabled)',
    '        mirror_combo.setEnabled(enabled); custom_mirror_edit.setEnabled(enabled); proxy_edit.setEnabled(enabled); off_edit.setEnabled(enabled); off_btn.setEnabled(enabled)',
    '        start_btn.setEnabled(enabled and ver_ok)',
    '    def _mirror_val():',
    '        idx_cur = mirror_combo.currentIndex() if callable(getattr(mirror_combo, "currentIndex", None)) else mirror_combo.currentIndex',
    '        d = str(mirror_combo.itemData(idx_cur) or "")',
    '        return custom_mirror_edit.text.strip() if d == "__custom__" else d',
    '    def _run_phase_a(sel_packs, mirror, proxy, offdir):',
    '        q = state["queue"]; cancel = state["cancel"]',
    '        def log_cb(msg): q.put(("log", msg))',
    '        try:',
    '            artifacts = CONFIG["artifacts"]; bkey = _host_bundle_key()',
    '            ext_spec = artifacts["bundles"][bkey]; iit_spec = artifacts.get("iit_template") or {}; packs = artifacts.get("packs") or {}',
    '            sel_specs = [packs[p] for p in sel_packs if p in packs]',
    '            total_all = int(ext_spec.get("size_bytes") or 0) + int(iit_spec.get("size_bytes") or 0) + sum(int(s.get("size_bytes") or 0) for s in sel_specs)',
    '            completed_base = [0]; t_prev = [time.monotonic()]; b_prev = [None]; spd_mb = [0.0]',
    '            def prog_cb(file_done, _file_tot, fname):',
    '                overall = min(total_all, completed_base[0] + int(file_done or 0))',
    '                now = time.monotonic()',
    '                if b_prev[0] is None: b_prev[0] = overall; t_prev[0] = now',
    '                dt = now - t_prev[0]',
    '                if dt >= 0.4:',
    '                    spd_mb[0] = max(0.0, (overall - b_prev[0]) / dt / 1048576.0)',
    '                    t_prev[0] = now; b_prev[0] = overall',
    '                pct_f = 100.0 * overall / max(1, total_all)',
    '                q.put(("progress", (int(900 * overall / max(1, total_all)), f"{fname} · {overall/1048576:.1f} / {total_all/1048576:.1f} MB ({pct_f:.1f}%, {spd_mb[0]:.2f} MB/s)")))',
    '            q.put(("progress", (5, "Connecting to download server...")))',
    '            log_cb("Checking network & local cache...")',
    '            if not str(proxy or "").strip():',
    '                detected_proxy = _detect_system_proxy()',
    '                if detected_proxy: log_cb(f"Using system proxy: {detected_proxy}")',
    '            if _is_intel_mac(): raise RuntimeError("Intel x86_64 Macs are not supported; Apple Silicon (M1-M4) is required.")',
    '            work_dir = _pick_cache_dir((int(total_all * 1.15) + 536870912) if total_all else 0)',
    '            ext_arc, ext_temps = _acquire_spec(ext_spec, work_dir, offline_dir=offdir, mirror_prefix=mirror, proxy_url=proxy, progress_cb=prog_cb, cancel_event=cancel, log_cb=log_cb)',
    '            completed_base[0] += int(ext_spec.get("size_bytes") or 0)',
    '            iit_file = None',
    '            if iit_spec:',
    '                try:',
    '                    iit_dl = work_dir / Path(str(iit_spec.get("relpath") or "IITmean_b0_256.nii.gz")).name',
    '                    for d in _search_dirs(offdir):',
    '                        for cand in (d / iit_spec["relpath"], d / iit_dl.name):',
    '                            if cand.is_file() and _sha256(cand, cancel_event=cancel) == iit_spec["sha256"]: iit_file = cand; break',
    '                        if iit_file is not None: break',
    '                    if iit_file is None:',
    '                        iit_file = _download(iit_spec.get("urls") or (), iit_dl, expected_sha256=str(iit_spec.get("sha256") or ""), expected_size=int(iit_spec.get("size_bytes") or 0), proxy_url=proxy, allow_mirror=False, timeout=20, progress_cb=prog_cb, cancel_event=cancel, log_cb=log_cb)',
    '                except InterruptedError: raise',
    '                except Exception as exc: log_cb(f"[WARN] NITRC IIT template skipped ({exc})")',
    '            completed_base[0] += int(iit_spec.get("size_bytes") or 0)',
    '            pack_arcs = []; asset_temps = []',
    '            for ps in sel_specs:',
    '                parc, ptemps = _acquire_spec(ps, work_dir, offline_dir=offdir, mirror_prefix=mirror, proxy_url=proxy, progress_cb=prog_cb, cancel_event=cancel, log_cb=log_cb)',
    '                pack_arcs.append((ps, parc)); asset_temps.extend(ptemps)',
    '                completed_base[0] += int(ps.get("size_bytes") or 0)',
    '            q.put(("downloads_ready", {"work_dir": str(work_dir), "ext_archive": str(ext_arc), "ext_temps": [str(x) for x in ext_temps], "iit_file": str(iit_file) if iit_file else "", "pack_archives": [(ps, str(pa)) for ps, pa in pack_arcs], "asset_temps": [str(x) for x in asset_temps], "sel_packs": sel_packs, "mirror": mirror, "proxy": proxy, "offdir": offdir}))',
    '        except InterruptedError: q.put(("cancelled", ""))',
    '        except Exception as exc: q.put(("error", f"{exc}\\n{traceback.format_exc()}"))',
    '    def _run_phase_c(info):',
    '        q = state["queue"]',
    '        def log_cb(msg): q.put(("log", msg))',
    '        try:',
    '            _cleanup_paths(info["ext_temps"])',
    '            mod_root = Path(info["mod_root"])',
    '            if str(mod_root) not in sys.path: sys.path.insert(0, str(mod_root))',
    '            from shared.network_config import extract_modular_pack, save_network_settings',
    '            from shared.model_assets import configured_model_assets_root',
    '            from shared.user_config import load_user_config, write_user_config',
    '            save_network_settings(mirror_prefix=info["mirror"], proxy_url=info["proxy"], offline_assets_dir=info["offdir"])',
    '            custom_assets = str(CONFIG.get("model_assets_dir") or "").strip()',
    '            asset_root = configured_model_assets_root(custom_assets or None); asset_root.mkdir(parents=True, exist_ok=True)',
    '            bundled_assets = mod_root.parents[2] / "bundled-model-assets"',
    '            if custom_assets and bundled_assets.is_dir() and bundled_assets.resolve() != asset_root.resolve():',
    '                log_cb(f"Staging bundled model weights -> {asset_root}")',
    '                for src_f in bundled_assets.rglob("*"):',
    '                    if not src_f.is_file(): continue',
    '                    dst = asset_root / src_f.relative_to(bundled_assets)',
    '                    if dst.exists(): continue',
    '                    dst.parent.mkdir(parents=True, exist_ok=True)',
    '                    try: os.link(src_f, dst)',
    '                    except OSError: shutil.copy2(src_f, dst)',
    '            iit_src = Path(info["iit_file"]) if info["iit_file"] else None',
    '            if iit_src and iit_src.is_file():',
    '                iit_rel = str((CONFIG["artifacts"].get("iit_template") or {}).get("relpath") or "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz")',
    '                iit_sha = str((CONFIG["artifacts"].get("iit_template") or {}).get("sha256") or "")',
    '                for base in ([bundled_assets] if bundled_assets.is_dir() else []) + [asset_root]:',
    '                    dst = base / iit_rel; dst.parent.mkdir(parents=True, exist_ok=True)',
    '                    if dst.is_file() and ((iit_sha and _sha256(dst) == iit_sha) or dst.resolve() == iit_src.resolve()): continue',
    '                    shutil.copy2(iit_src, dst)',
    '                if iit_src.parent == Path(info.get("work_dir") or (Path(tempfile.gettempdir()) / "slicerdspace-installer-cache")): _cleanup_paths([iit_src])',
    '            pack_list = info["pack_archives"]',
    '            for i, (ps, parc_str) in enumerate(pack_list):',
    '                parc = Path(parc_str)',
    '                q.put(("progress", (930 + int(60 * i / max(1, len(pack_list))), f"Extracting {ps[\'filename\']}...")))',
    '                log_cb(f"Extracting {ps[\'filename\']} -> {asset_root}")',
    '                extract_modular_pack(parc, asset_root, expected_sha256=str(ps.get("sha256") or ""))',
    '            _cleanup_paths(info["asset_temps"])',
    '            cfg = load_user_config(); ui_cfg = cfg.setdefault("ui", {}); ui_cfg["language"] = ui_cfg.get("language") or str(CONFIG.get("language") or "en")',
    '            deps = cfg.setdefault("external_dependencies", {})',
    '            if custom_assets: deps["model_assets_dir"] = str(asset_root)',
    '            else: deps.pop("model_assets_dir", None)',
    '            deps.pop("wma_atlas_dir", None)',
    '            defaults = cfg.setdefault("run_defaults", {})',
    '            targets = ["microstructure", "segmentation", "parcellation", "surface", "tractography", "saf", "bundles", "tractcloud", "connectome"]',
    '            sel_packs = info["sel_packs"]',
    '            if sel_packs:',
    '                targets.append("wma"); defaults["bundle_backend"] = "wma,tractcloud,wmql"',
    '                defaults["wma_atlas"] = "org" if "wma_org" in sel_packs else ("naba" if "wma_naba" in sel_packs else "east-west")',
    '            else: defaults["bundle_backend"] = "tractcloud,wmql"',
    '            defaults["targets"] = list(dict.fromkeys(targets)); write_user_config(cfg)',
    '            q.put(("all_done", {"mod_root": str(mod_root)}))',
    '        except Exception as exc: q.put(("error", f"{exc}\\n{traceback.format_exc()}"))',
    '    def _poll():',
    '        q = state["queue"]',
    '        while not q.empty():',
    '            item = q.get_nowait()',
    '            kind = item[0]',
    '            payload = item[1] if len(item) == 2 else item[1:]',
    '            if kind == "log": _append_log(payload)',
    '            elif kind == "progress":',
    '                pct, txt = payload',
    '                pbar.setValue(max(0, min(1000, int(pct))))',
    '                status_lbl.setText(txt)',
    '            elif kind == "downloads_ready":',
    '                try:',
    '                    pbar.setValue(910)',
    '                    status_lbl.setText("Installing SlicerDSpace into 3D Slicer...")',
    '                    qt.QApplication.processEvents()',
    '                    ext_arc = Path(payload["ext_archive"])',
    '                    _dequarantine(ext_arc)',
    '                    em = slicer.app.extensionsManagerModel(); em.setInteractive(False)',
    '                    if em.isExtensionInstalled("SlicerDSpace"):',
    '                        em.uninstallExtension("SlicerDSpace"); em.updateModel()',
    '                    meta = {"extensionname": "SlicerDSpace", "depends": "NA", "slicer_revision": str(em.slicerRevision), "os": str(em.slicerOs), "arch": str(em.slicerArch)}',
    '                    ok = bool(em.installExtension("SlicerDSpace", meta, str(ext_arc), False, True))',
    '                    if not ok: raise RuntimeError("Slicer Extension Manager rejected SlicerDSpace archive.")',
    '                    mod_root = _resolve_mod_root(em)',
    '                    _dequarantine(mod_root.parents[2])',
    '                    _append_log(f"Installed SlicerDSpace at: {mod_root}")',
    '                    payload["mod_root"] = str(mod_root)',
    '                    threading.Thread(target=_run_phase_c, args=(payload,), daemon=True).start()',
    '                except Exception as exc:',
    '                    state["running"] = False; poll_timer.stop(); _set_controls_enabled(True)',
    '                    status_lbl.setText(f"Install failed: {exc}")',
    '                    _append_log(traceback.format_exc())',
    '            elif kind == "all_done":',
    '                state["running"] = False; state["done"] = True; poll_timer.stop()',
    '                try:',
    '                    em = slicer.app.extensionsManagerModel()',
    '                    if em.isExtensionInstalled("SlicerDSpaceSetup"): em.scheduleExtensionForUninstall("SlicerDSpaceSetup")',
    '                except Exception: pass',
    '                pbar.setValue(1000)',
    '                status_lbl.setText("Setup complete! Restart 3D Slicer to open Modules -> Diffusion -> SlicerDSpace.")',
    '                _append_log("Setup finished successfully.")',
    '                cancel_btn.setText("Close")',
    '                start_btn.setText("Restart Slicer Now"); start_btn.setEnabled(True)',
    '            elif kind == "cancelled":',
    '                state["running"] = False; poll_timer.stop(); _set_controls_enabled(True)',
    '                status_lbl.setText("Cancelled. Adjust options and click Start Install.")',
    '                cancel_btn.setText("Close")',
    '            elif kind == "error":',
    '                state["running"] = False; poll_timer.stop(); _set_controls_enabled(True)',
    '                status_lbl.setText("Failed. Check log below or switch Mirror / Offline Dir.")',
    '                _append_log(payload); cancel_btn.setText("Close")',
    '    poll_timer.timeout.connect(_poll)',
    '    def _on_start():',
    '        if state["done"]: slicer.util.restart(); return',
    '        if state["running"] or not ver_ok: return',
    '        state["running"] = True; state["cancel"] = threading.Event()',
    '        _set_controls_enabled(False); cancel_btn.setText("Cancel")',
    '        sel = [aid for aid in ("wma_org", "wma_naba", "wma_east_west") if atlas_checks[aid].isChecked()]',
    '        pbar.setValue(0); status_lbl.setText("Starting downloads...")',
    '        poll_timer.start()',
    '        threading.Thread(target=_run_phase_a, args=(sel, _mirror_val(), proxy_edit.text.strip(), off_edit.text.strip()), daemon=True).start()',
    '    def _on_cancel():',
    '        if state["running"]: state["cancel"].set(); status_lbl.setText("Cancelling...")',
    '        else: dlg.close()',
    '    def _on_finished(_res):',
    '        global _ACTIVE_DIALOG',
    '        state["cancel"].set(); poll_timer.stop(); _ACTIVE_DIALOG = None',
    '    start_btn.clicked.connect(_on_start)',
    '    cancel_btn.clicked.connect(_on_cancel)',
    '    dlg.finished.connect(_on_finished)',
    '    dlg.show()',
    '    if auto_start and ver_ok: qt.QTimer.singleShot(600, _on_start)',
    '    return dlg',
    'class SlicerDSpaceSetup(ScriptedLoadableModule):',
    '    def __init__(self, parent):',
    '        ScriptedLoadableModule.__init__(self, parent)',
    '        self.parent.title = "SlicerDSpace Setup"',
    '        self.parent.categories = ["Diffusion"]',
    '        self.parent.dependencies = []',
    '        self.parent.contributors = ["Chengjin Li, Fan Zhang (Fan\'s Lab, UESTC)"]',
    '        self.parent.helpText = "Graphical one-click installer for SlicerDSpace."',
    '        if hasattr(slicer, "app") and slicer.app and not slicer.app.commandOptions().noMainWindow:',
    '            slicer.app.connect("startupCompleted()", self._onStartupCompleted)',
    '    def _onStartupCompleted(self):',
    '        try:',
    '            em = slicer.app.extensionsManagerModel()',
    '            if em and em.isExtensionInstalled("SlicerDSpace"): return',
    '        except Exception: pass',
    '        qt.QTimer.singleShot(600, lambda: show_setup_dialog(auto_start=bool(CONFIG.get("auto_start", False))))',
    'class SlicerDSpaceSetupWidget(ScriptedLoadableModuleWidget):',
    '    def setup(self):',
    '        ScriptedLoadableModuleWidget.setup(self)',
    '        btn = qt.QPushButton("Open SlicerDSpace Installer")',
    '        btn.clicked.connect(lambda: show_setup_dialog(auto_start=False))',
    '        self.layout.addWidget(btn); self.layout.addStretch(1)'
  ].join('\n');

  // Headless CLI script fallback (for HPC Linux / SSH servers without GUI).
  var BOOTSTRAP_PY = [
    'import hashlib, json, os, platform, queue, shutil, socket, ssl, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request',
    'from pathlib import Path',
    'def _log(msg): print(f"[SlicerDSpace Installer] {msg}", flush=True)',
    SHARED_DOWNLOAD_PY,
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
    '    helper.write_text("import json, sys, traceback, qt, slicer\\nfrom pathlib import Path\\ndef _run():\\n  a=sys.argv[1]; r=Path(sys.argv[2])\\n  try:\\n    em=slicer.app.extensionsManagerModel(); em.setInteractive(False)\\n    if em.isExtensionInstalled(\'SlicerDSpace\'): em.uninstallExtension(\'SlicerDSpace\'); em.updateModel()\\n    meta={\'extensionname\':\'SlicerDSpace\',\'depends\':\'NA\',\'slicer_revision\':str(em.slicerRevision),\'os\':str(em.slicerOs),\'arch\':str(em.slicerArch)}\\n    ok=bool(em.installExtension(\'SlicerDSpace\', meta, a, False, True))\\n    r.write_text(json.dumps({\'installed\':ok,\'module_paths\':[str(p) for p in em.extensionModulePaths(\'SlicerDSpace\')]}),encoding=\'utf-8\')\\n    slicer.app.exit(0 if ok else 1)\\n  except Exception as e:\\n    r.write_text(json.dumps({\'installed\':False,\'error\':str(e),\'traceback\':traceback.format_exc()}),encoding=\'utf-8\')\\n    slicer.app.exit(1)\\nqt.QTimer.singleShot(500, _run)\\n", encoding="utf-8")',
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
    '        try: tmp_dl = work_dir / Path(relpath).name; src = _download(iit_spec.get("urls") or (), tmp_dl, expected_sha256=sha, expected_size=sz, allow_mirror=False, timeout=20)',
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
    '    if _is_intel_mac(): raise RuntimeError("Intel x86_64 Macs are not supported; Apple Silicon (M1-M4) is required.")',
    '    bkey = {"windows":"windows_x64_base","linux":"linux_x64_base","macos":"macos_arm64_base"}[os_name]',
    '    need_b = int(((artifacts.get("bundles") or {}).get(bkey) or {}).get("size_bytes") or 0) + int((artifacts.get("iit_template") or {}).get("size_bytes") or 0)',
    '    work_dir = _pick_cache_dir(int(need_b * 1.15) + 536870912 if need_b else 0)',
    '    ext_archive, ext_temps = _acquire_spec(artifacts["bundles"][bkey], work_dir)',
    '    receipt = _install_extension(slicer_exe, ext_archive); _cleanup_temp_files(ext_temps)',
    '    mod_root = _find_module_root(slicer_exe, receipt); _log(f"Extension installed at: {mod_root}")',
    '    if sys.platform == "darwin" and shutil.which("xattr"): subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(mod_root.parents[2])], capture_output=True)',
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
    '    cfg = load_user_config(); ui_cfg = cfg.setdefault("ui", {}); ui_cfg["language"] = ui_cfg.get("language") or str(CONFIG.get("language") or "en")',
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

  var CRC_TABLE = (function () {
    var table = new Uint32Array(256);
    for (var i = 0; i < 256; i++) {
      var c = i;
      for (var k = 0; k < 8; k++) {
        c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      }
      table[i] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    var crc = 0xffffffff;
    for (var i = 0; i < bytes.length; i++) {
      crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function encodeUtf8(str) {
    return new TextEncoder().encode(String(str));
  }

  async function deflateRaw(bytes) {
    if (typeof CompressionStream === 'undefined') {
      throw new Error('This browser does not support Web CompressionStream.');
    }
    var cs = new CompressionStream('deflate-raw');
    var writer = cs.writable.getWriter();
    writer.write(bytes);
    writer.close();
    var ab = await new Response(cs.readable).arrayBuffer();
    return new Uint8Array(ab);
  }

  async function buildDeflatedZip(entries) {
    var dosTime = (12 << 11) | 0;
    var dosDate = ((2026 - 1980) << 9) | (11 << 5) | 18;
    var parts = [];
    var cdRecords = [];
    var offset = 0;

    for (var i = 0; i < entries.length; i++) {
      var nameBytes = encodeUtf8(entries[i].name);
      var rawBytes = entries[i].data instanceof Uint8Array ? entries[i].data : encodeUtf8(entries[i].data);
      var compBytes = await deflateRaw(rawBytes);
      var crc = crc32(rawBytes);

      var lfh = new Uint8Array(30 + nameBytes.length);
      var lv = new DataView(lfh.buffer);
      lv.setUint32(0, 0x04034b50, true);
      lv.setUint16(4, 20, true);
      lv.setUint16(6, 0x0800, true);
      lv.setUint16(8, 8, true);
      lv.setUint16(10, dosTime, true);
      lv.setUint16(12, dosDate, true);
      lv.setUint32(14, crc, true);
      lv.setUint32(18, compBytes.length, true);
      lv.setUint32(22, rawBytes.length, true);
      lv.setUint16(26, nameBytes.length, true);
      lv.setUint16(28, 0, true);
      lfh.set(nameBytes, 30);
      parts.push(lfh, compBytes);

      var cdh = new Uint8Array(46 + nameBytes.length);
      var cv = new DataView(cdh.buffer);
      cv.setUint32(0, 0x02014b50, true);
      cv.setUint16(4, 0x0314, true);
      cv.setUint16(6, 20, true);
      cv.setUint16(8, 0x0800, true);
      cv.setUint16(10, 8, true);
      cv.setUint16(12, dosTime, true);
      cv.setUint16(14, dosDate, true);
      cv.setUint32(16, crc, true);
      cv.setUint32(20, compBytes.length, true);
      cv.setUint32(24, rawBytes.length, true);
      cv.setUint16(28, nameBytes.length, true);
      cv.setUint16(30, 0, true);
      cv.setUint16(32, 0, true);
      cv.setUint16(34, 0, true);
      cv.setUint16(36, 0, true);
      cv.setUint32(38, 0x81a40000 >>> 0, true);
      cv.setUint32(42, offset >>> 0, true);
      cdh.set(nameBytes, 46);
      cdRecords.push(cdh);

      offset += lfh.length + compBytes.length;
    }

    var cdSize = cdRecords.reduce(function (sum, b) { return sum + b.length; }, 0);
    var eocd = new Uint8Array(22);
    var ev = new DataView(eocd.buffer);
    ev.setUint32(0, 0x06054b50, true);
    ev.setUint16(4, 0, true);
    ev.setUint16(6, 0, true);
    ev.setUint16(8, entries.length, true);
    ev.setUint16(10, entries.length, true);
    ev.setUint32(12, cdSize >>> 0, true);
    ev.setUint32(16, offset >>> 0, true);
    ev.setUint16(20, 0, true);

    var totalLen = offset + cdSize + eocd.length;
    var out = new Uint8Array(totalLen);
    var pos = 0;
    for (var j = 0; j < parts.length; j++) { out.set(parts[j], pos); pos += parts[j].length; }
    for (var k = 0; k < cdRecords.length; k++) { out.set(cdRecords[k], pos); pos += cdRecords[k].length; }
    out.set(eocd, pos);
    return out;
  }

  function asciiJsonLiteral(obj) {
    var s = JSON.stringify(obj);
    return "'" + s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/[\u007f-\uffff]/g, function (c) {
      return '\\u' + ('0000' + c.charCodeAt(0).toString(16)).slice(-4);
    }) + "'";
  }

  function buildConfig(state) {
    var selectedPacks = state.wmaEnabled ? (state.wmaAtlases || []).slice() : [];
    return {
      os: state.os || 'macos',
      mode: 'base',
      slicer_path: String(state.slicerPath || '').trim(),
      model_assets_dir: String(state.modelAssetsDir || '').trim(),
      selected_packs: selectedPacks,
      mirror_prefix: String(state.mirrorPrefix || '').trim(),
      proxy_url: String(state.proxyUrl || '').trim(),
      offline_dir: String(state.offlineDir || '').trim(),
      check_script_dir: state.checkScriptDir !== false,
      auto_start: Boolean(state.autoStart),
      language: String(state.language || ((typeof document !== 'undefined' && String(document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0) ? 'zh' : 'en')),
      artifacts: ARTIFACTS
    };
  }

  function buildSetupExtensionFiles(state) {
    var osKey = state.os || 'macos';
    var def = OS_DEFAULTS[osKey] || OS_DEFAULTS.macos;
    var cfg = buildConfig(state);
    var prefix = osKey === 'macos'
      ? 'SlicerDSpaceSetup/Slicer.app/Contents/Extensions-34045/SlicerDSpaceSetup'
      : 'SlicerDSpaceSetup';
    var s4ext = [
      'arch amd64',
      'category Diffusion',
      "contributors Chengjin Li, Fan Zhang (Fan's Lab, UESTC)",
      'depends NA',
      'description SlicerDSpace One-Click Graphical Installer',
      'enabled 1',
      'homepage https://neurodmri.github.io/SlicerDSpace/',
      'os ' + def.s4extOs,
      'scm git',
      'scmrevision ' + RELEASE_TAG,
      'scmurl https://github.com/NeuroDMRI/SlicerDSpace.git',
      'slicer_revision 34045',
      ''
    ].join('\n');
    var setupPy = [
      '# SlicerDSpace One-Click Graphical Setup Extension for 3D Slicer 5.10.0',
      'import json as _json',
      'CONFIG = _json.loads(' + asciiJsonLiteral(cfg) + ')',
      SETUP_MODULE_PY,
      ''
    ].join('\n');
    return [
      { name: prefix + '/share/Slicer-5.10/SlicerDSpaceSetup.s4ext', data: s4ext },
      { name: prefix + '/SlicerDSpaceSetup.s4ext', data: s4ext },
      { name: prefix + '/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpaceSetup.py', data: setupPy }
    ];
  }

  async function buildSetupZip(state) {
    return buildDeflatedZip(buildSetupExtensionFiles(state));
  }

  function buildScript(state) {
    var cfg = buildConfig(state);
    var pyPayload =
      '# __SLICERDSPACE_PY_BEGIN__\n' +
      'CONFIG = __import__("json").loads(' + asciiJsonLiteral(cfg) + ')\n' +
      'CONFIG["__script_file__"] = __file__\n' +
      BOOTSTRAP_PY + '\n' +
      '# __SLICERDSPACE_PY_END__\n';

    var cleanSlicer = String(state.slicerPath || '').trim().replace(/"/g, '');
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

  function detectOS() {
    if (typeof navigator === 'undefined') { return 'macos'; }
    var ua = (navigator.platform || navigator.userAgent || '').toLowerCase();
    if (ua.indexOf('win') !== -1) { return 'windows'; }
    if (ua.indexOf('mac') !== -1) { return 'macos'; }
    return 'linux';
  }

  function triggerBlobDownload(bytes, mimeType, filename) {
    var blob = new Blob([bytes], { type: mimeType });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  function initCustomizer() {
    var root = document.getElementById('installer-customizer');
    if (!root) { return; }

    var state = {
      os: detectOS(),
      slicerPath: '',
      modelAssetsDir: '',
      wmaEnabled: false,
      wmaAtlases: [],
      mirrorPreset: 'direct',
      mirrorPrefix: '',
      proxyUrl: '',
      offlineDir: '',
      checkScriptDir: true,
      autoStart: false
    };

    function t(en, cn) { return zh ? cn : en; }

    function escapeAttr(v) {
      return String(v || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function render() {
      var embedded = root.classList.contains('is-embedded');
      var html = [];
      if (!embedded) {
        html.push('<div class="ic-top">');
        html.push('<div class="ic-top-copy">');
        html.push('<h3 class="ic-heading">' + t('One-Click Graphical Setup', '一键图形化安装器') + '</h3>');
        html.push('<p class="ic-sub">' + t('Select your operating system and download the lightweight setup archive for 3D Slicer 5.10.0.', '选择操作系统并直接下载 3D Slicer 5.10.0 轻量安装引导包。') + '</p>');
        html.push('</div></div>');
      }

      html.push('<div class="ic-band ic-band-choose ic-band-os">');
      html.push('<div class="ic-band-head">');
      html.push('<span class="ic-band-title">' + t('Operating System', '操作系统') + '</span>');
      html.push('</div>');
      html.push('<div class="ic-os-row">');
      html.push('<div class="ic-seg" role="group" aria-label="' + escapeAttr(t('Operating System', '操作系统')) + '">');
      var OS_ARCH = { macos: 'arm64', windows: 'x86_64', linux: 'x86_64' };
      ['macos', 'windows', 'linux'].forEach(function (osKey) {
        html.push('<button type="button" class="ic-seg-btn" data-os="' + osKey + '" aria-pressed="' + (state.os === osKey ? 'true' : 'false') + '"><span>' + OS_DEFAULTS[osKey].label + '</span><span class="ic-seg-arch">' + OS_ARCH[osKey] + '</span></button>');
      });
      html.push('</div>');
      html.push('<div class="ic-actions">');
      html.push('<button type="button" class="ic-btn-primary" id="ic-download-btn">' + t('Download Slicer Setup', '下载 Slicer 一键安装包') + '</button>');
      html.push('</div>');
      html.push('</div>');
      html.push('</div>');

      root.innerHTML = html.join('');
      bindEvents();
    }

    function bindEvents() {
      Array.prototype.forEach.call(root.querySelectorAll('[data-os]'), function (btn) {
        btn.addEventListener('click', function () { state.os = btn.getAttribute('data-os'); render(); });
      });

      var dlBtn = document.getElementById('ic-download-btn');
      if (dlBtn) {
        dlBtn.addEventListener('click', function () {
          var filename = OS_DEFAULTS[state.os].setupZipName;
          var orig = dlBtn.textContent;
          dlBtn.disabled = true;
          dlBtn.textContent = zh ? '正在打包...' : 'Packaging...';
          buildSetupZip(state).then(function (zipBytes) {
            triggerBlobDownload(zipBytes, 'application/zip', filename);
            dlBtn.disabled = false;
            dlBtn.textContent = orig;
          }).catch(function () {
            dlBtn.disabled = false;
            dlBtn.textContent = orig;
            triggerBlobDownload(encodeUtf8(buildScript(state)), 'text/plain;charset=utf-8', OS_DEFAULTS[state.os].scriptName);
          });
        });
      }
    }

    render();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      ARTIFACTS: ARTIFACTS,
      OS_DEFAULTS: OS_DEFAULTS,
      buildSetupExtensionFiles: buildSetupExtensionFiles,
      buildSetupZip: buildSetupZip,
      buildScript: buildScript
    };
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initCustomizer);
    } else {
      initCustomizer();
    }
  }
})();
