/* SlicerDSpace Website Visual Installer Customizer. Vanilla JS, offline-safe. */
(function () {
  'use strict';

  var zh = document.documentElement.lang.toLowerCase().indexOf('zh') === 0;
  var RELEASE_TAG = 'v0.1.1';
  var RELEASE_BASE = 'https://github.com/NeuroDMRI/SlicerDSpace/releases/download/' + RELEASE_TAG;

  var ARTIFACTS = {
    release_tag: RELEASE_TAG,
    release_base_url: RELEASE_BASE,
    packs: {
      brain_mask_pnl: {
        id: 'brain_mask_pnl',
        filename: 'SlicerDSpace-v0.1.1-Pack-BrainMask-PNL.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Pack-BrainMask-PNL.zip',
        size_bytes: 200177953,
        sha256: '08129054bada6639b45228609389b857e84470eb6952eaa7e0a57471adc1b333',
        upstream_files: {
          'brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz': {
            sha256: '2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead',
            size_bytes: 7830929,
            urls: ['https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz'],
            redistribution: 'upstream-only'
          }
        }
      },
      anatomy: {
        id: 'anatomy',
        filename: 'SlicerDSpace-v0.1.1-Pack-Anatomy.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Pack-Anatomy.zip',
        size_bytes: 670899381,
        sha256: '5a81ee528a5485106e8537b71dce53659359d008404486353ae76edd6bcda9ae',
        upstream_files: {}
      },
      surface: {
        id: 'surface',
        filename: 'SlicerDSpace-v0.1.1-Pack-Surface.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Pack-Surface.zip',
        size_bytes: 222419412,
        sha256: '9432aea60a3919cda5a6361c77cc9dd9944b17a003cd7811973c5f8984ea1563',
        upstream_files: {}
      },
      tractcloud: {
        id: 'tractcloud',
        filename: 'SlicerDSpace-v0.1.1-Pack-TractCloud.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Pack-TractCloud.zip',
        size_bytes: 8217555,
        sha256: 'a34095caed89764c2043aa7598e60f14d9befdcb6fe46666ee1f4c599faa0078',
        upstream_files: {}
      },
      wma_org: {
        id: 'wma_org',
        filename: 'SlicerDSpace-v0.1.1-Pack-WMA-ORG.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Pack-WMA-ORG.zip',
        size_bytes: 1804782008,
        sha256: '531bc914f27298f82a87cfac03701ca79eb973ceec65e831498f0b2d9d684370',
        upstream_files: {}
      }
    },
    bundles: {
      core_scripted: {
        filename: 'SlicerDSpace-v0.1.1-Core-Scripted.zip',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Core-Scripted.zip',
        size_bytes: 4039964,
        sha256: '97089ee639a066cdd6dd14686ca8fabea8b9244a6f1c72c851999a44fc742b42'
      },
      macos_arm64_core: {
        filename: 'SlicerDSpace-v0.1.1-macOS-arm64-Core.tar.gz',
        url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64-Core.tar.gz',
        size_bytes: 367848476,
        sha256: 'c7366961ff42b1b42c387e0b7563d8de28215f1a972bcd9e6749ef7fdda72c79'
      },
      windows_x64_all_in_one: {
        filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip',
        size_bytes: 18124374419,
        sha256: 'b38d612ad000c6a86b78cc1db10c58c7422c3cd1ca506a65285d7d7598f2feae',
        parts: [
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.001',
            size_bytes: 1900000000,
            sha256: '816a9e730f75d7673dc2f1f4f2b9fc6caf13867939e8ab93feb94cd57838f7ee',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.001'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.002',
            size_bytes: 1900000000,
            sha256: '514f8cfe930ef54b3e7d63171eb2ffbf0ce4efc8ef0f888d5e2abe44685931e4',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.002'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.003',
            size_bytes: 1900000000,
            sha256: 'ccf65e829384e13ba1925bff1f87860030f406ed96b640d024ed22dc742df337',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.003'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.004',
            size_bytes: 1900000000,
            sha256: '20300c3bef02e9f4a79fafb76c8f4d46994a18d5ee9ce5745be8db2e90ed3a13',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.004'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.005',
            size_bytes: 1900000000,
            sha256: 'b14d457459979741583ba7139317394333dc2fb0cfc1c4c5e921d73a7da25d7b',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.005'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.006',
            size_bytes: 1900000000,
            sha256: '4021369d1e9433c02e9eb51efe7d417721781d6545672b059511d51e99e34600',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.006'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.007',
            size_bytes: 1900000000,
            sha256: 'ab317ff156cc9cd444f1c5f88e91ee886dfbef2f3b172973ff498ed93c2e5692',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.007'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.008',
            size_bytes: 1900000000,
            sha256: '9002830250981f09aecd9936142f6924e5dfb265daa82bc360ead30f297a8368',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.008'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.009',
            size_bytes: 1900000000,
            sha256: '5a7e8c8ce83c96b89f247a375c24fa0e0d23a10ee96c3da7c194cb31a1cd037c',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.009'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Windows-x64.zip.010',
            size_bytes: 1024374419,
            sha256: '97ef58d4da44ab62d28d597c17b8f3e7c0173df6f835852905206aca1050a7e2',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Windows-x64.zip.010'
          }
        ]
      },
      linux_x64_all_in_one: {
        filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz',
        size_bytes: 14416513720,
        sha256: '86cc9cb5ef34dbede12c9abdbe110b972ff0fc873d9cef34df3bc8cbad36d403',
        parts: [
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001',
            size_bytes: 1900000000,
            sha256: '01b7fd7c39941e8c3757042fbdce0a14e3ced752cd0672cbc847f7fbeb04f4f8',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.002',
            size_bytes: 1900000000,
            sha256: '476aefe2f555f629241f62b10636c8431272581ea2fb91c2b3606a036c5e3ee8',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.002'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.003',
            size_bytes: 1900000000,
            sha256: '0b0b8a9bd3a955b44cf9b3016f5e84dd5c3a2fdf45464fcbe9e872379b701639',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.003'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.004',
            size_bytes: 1900000000,
            sha256: '358c0b66de302e7b64fb63d3ada4a22bf690374852d8874e49552d813e562515',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.004'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.005',
            size_bytes: 1900000000,
            sha256: '3284c0b811ccf10e5c907fc152e1e49a98640a1a13be693fb3d94b7d35d5b436',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.005'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.006',
            size_bytes: 1900000000,
            sha256: '5a65b97b8f16bc88fef9826fc1e7f68804545d18ddf966a0bc5126e9a2ab9a50',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.006'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.007',
            size_bytes: 1900000000,
            sha256: 'c27bca4aafb1ccd4e5b6c50b78a374e5d4c49483d33bef03c401792678c45dd8',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.007'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-Linux-x64.tar.gz.008',
            size_bytes: 1116513720,
            sha256: 'd6488d61641cc5ea6c85db778b00266851378e77b012990ad28ddc095effe5c0',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.008'
          }
        ]
      },
      macos_arm64_all_in_one: {
        filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz',
        size_bytes: 14326384768,
        sha256: 'ca27a69813c9f988ad13dad0d01244ca907cfdfc981db5a445bf4f1ec1aa5d9a',
        parts: [
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.001',
            size_bytes: 1900000000,
            sha256: '719e3e6150a70291f44b20386c02f023e04edb9893160d7b2ff4935862fdb620',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.001'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.002',
            size_bytes: 1900000000,
            sha256: '51d69f43c3ed140d56ab7516d3791bc213b4a80d037045bdec107d400748b4c7',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.002'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.003',
            size_bytes: 1900000000,
            sha256: '2b3a00198601c9c6d7b6088388e508f788a4bf968a13ecf9f7bb969bc54e6cbc',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.003'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.004',
            size_bytes: 1900000000,
            sha256: '1aeed976ad2a100c5c98f35087c2ff14e3f1ee8851721187b89935c73e574079',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.004'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.005',
            size_bytes: 1900000000,
            sha256: '1d31624e6637ba2b6a2b8f6fcd9d7477c98c55c28320921cbc3a8e1d3d4044a9',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.005'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.006',
            size_bytes: 1900000000,
            sha256: '698e7c963a6fbd5004f14a5e30ceebd9deb665a4c2d702bd37ceaf45c643eec7',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.006'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.007',
            size_bytes: 1900000000,
            sha256: '82a779a2d9c86281bf1fb51207d0da6a2d0c630682bb06164491f63f67a938de',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.007'
          },
          {
            filename: 'SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.008',
            size_bytes: 1026384768,
            sha256: 'e78c500f2eab4102ab9d45d4d7f1a7506cd6cc3ed50841f0bab639a9c20f691b',
            url: RELEASE_BASE + '/SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.008'
          }
        ]
      }
    }
  };

  var OS_DEFAULTS = {
    macos: {
      label: 'macOS',
      slicerPlaceholder: '/Applications/Slicer.app',
      modelPlaceholder: '~/Library/Application Support/SlicerDSpace/model-assets',
      scriptName: 'Install-SlicerDSpace.command',
      coreEstimateBytes: 367848476
    },
    windows: {
      label: 'Windows',
      slicerPlaceholder: '%LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0',
      modelPlaceholder: '%LOCALAPPDATA%\\SlicerDSpace\\model-assets',
      scriptName: 'Install-SlicerDSpace.bat',
      coreEstimateBytes: 4039964
    },
    linux: {
      label: 'Linux',
      slicerPlaceholder: '~/Slicer-5.10.0-linux-amd64',
      modelPlaceholder: '~/.local/share/SlicerDSpace/model-assets',
      scriptName: 'Install-SlicerDSpace.sh',
      coreEstimateBytes: 4039964
    }
  };

  var PACK_META = [
    { id: 'brain_mask_pnl', title: zh ? 'PNL 自动脑掩膜' : 'PNL Brain Mask' },
    { id: 'anatomy', title: zh ? 'DDSeg 分割 + DDParcel 分区' : 'DDSeg + DDParcel' },
    { id: 'surface', title: zh ? 'DDSurfer 皮层表面 + SAF' : 'DDSurfer + SAF' },
    { id: 'tractcloud', title: zh ? 'TractCloud 纤维束分割' : 'TractCloud Bundles' },
    { id: 'wma_org', title: zh ? 'WMA ORG 白质图谱' : 'WMA ORG Atlas' }
  ];

  var PRESETS = {
    minimal: ['brain_mask_pnl'],
    clinical: ['brain_mask_pnl', 'anatomy', 'surface', 'tractcloud'],
    full: ['brain_mask_pnl', 'anatomy', 'surface', 'tractcloud', 'wma_org']
  };

  var BOOTSTRAP_PY = [
    'import hashlib, json, os, platform, shutil, stat, subprocess, sys, tarfile, tempfile, urllib.error, urllib.request, zipfile',
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
    '            _log(f"Using offline archive: {c}"); return c',
    '    cached = cache_dir / filename',
    '    if cached.is_file() and expected_sha and _sha256(cached) == expected_sha:',
    '        _log(f"Reusing cache: {cached}"); return cached',
    '    if parts:',
    '        part_paths = []',
    '        for ps in parts:',
    '            pn = ps["filename"]; psha = str(ps.get("sha256") or ""); psize = int(ps.get("size_bytes") or 0); found = None',
    '            for d in dirs:',
    '                c = d / pn',
    '                if c.is_file() and (not psha or _sha256(c) == psha): _log(f"Using offline part: {c}"); found = c; break',
    '            if found is None: found = _download(ps["url"], cache_dir / pn, expected_sha256=psha, expected_size=psize)',
    '            part_paths.append(found)',
    '        _log(f"Merging {len(part_paths)} volumes into {filename}...")',
    '        merging = cache_dir / (filename + ".merging")',
    '        with merging.open("wb") as out_fh:',
    '            for pp in part_paths:',
    '                with pp.open("rb") as in_fh: shutil.copyfileobj(in_fh, out_fh, length=16777216)',
    '        if expected_sha and _sha256(merging) != expected_sha: merging.unlink(missing_ok=True); raise RuntimeError(f"Merged SHA-256 mismatch: {filename}")',
    '        os.replace(merging, cached); _log(f"Merged and verified {filename}"); return cached',
    '    return _download(spec["url"], cached, expected_sha256=expected_sha, expected_size=expected_size)',
    'def _install_extension(slicer_exe, archive_path, extra_extensions=()):',
    '    if sys.platform == "darwin" and shutil.which("xattr"): subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(archive_path)], capture_output=True)',
    '    receipt = Path(tempfile.mkdtemp(prefix="slicerdspace-inst-")) / "receipt.json"',
    '    helper = receipt.with_name("install_ext.py")',
    '    helper.write_text("import json, sys, traceback, qt, slicer\\nfrom pathlib import Path\\ndef _run():\\n  a=sys.argv[1]; r=Path(sys.argv[2]); ex=[x for x in (sys.argv[3].split(\',\') if len(sys.argv)>3 else []) if x]\\n  try:\\n    em=slicer.app.extensionsManagerModel(); em.setInteractive(False)\\n    meta={\'extensionname\':\'SlicerDSpace\',\'depends\':\'NA\',\'slicer_revision\':str(em.slicerRevision),\'os\':str(em.slicerOs),\'arch\':str(em.slicerArch)}\\n    ok=bool(em.installExtension(\'SlicerDSpace\', meta, a, False, True))\\n    er={}\\n    for n in ex:\\n      try: er[n]=True if em.isExtensionInstalled(n) else bool(em.installExtensionFromServer(n, False))\\n      except Exception as e: er[n]=str(e)\\n    r.write_text(json.dumps({\'installed\':ok,\'module_paths\':[str(p) for p in em.extensionModulePaths(\'SlicerDSpace\')],\'extras\':er}),encoding=\'utf-8\')\\n    slicer.app.exit(0 if ok else 1)\\n  except Exception as e:\\n    r.write_text(json.dumps({\'installed\':False,\'error\':str(e),\'traceback\':traceback.format_exc()}),encoding=\'utf-8\')\\n    slicer.app.exit(1)\\nqt.QTimer.singleShot(500, _run)\\n", encoding="utf-8")',
    '    cmd = [slicer_exe, "--no-splash", "--no-main-window", "--python-script", str(helper), str(archive_path), str(receipt), ",".join(extra_extensions)]',
    '    env = os.environ.copy()',
    '    if sys.platform.startswith("linux") and not env.get("DISPLAY"):',
    '        xvfb = shutil.which("xvfb-run")',
    '        if xvfb: cmd = [xvfb, "-a", *cmd]',
    '        else: env.setdefault("QT_QPA_PLATFORM", "offscreen")',
    '    _log("Installing extension into 3D Slicer...")',
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
    'def main():',
    '    slicer_exe = str(os.environ.get("SLICERDSPACE_RESOLVED_SLICER") or CONFIG.get("slicer_path") or "").strip()',
    '    if not slicer_exe or not Path(slicer_exe).exists(): raise FileNotFoundError(f"Slicer not found: {slicer_exe}")',
    '    os_name, mode, artifacts = CONFIG["os"], CONFIG["mode"], CONFIG["artifacts"]',
    '    work_dir = Path(tempfile.gettempdir()) / "slicerdspace-installer-cache"; work_dir.mkdir(parents=True, exist_ok=True)',
    '    bkey = {"windows":"windows_x64_all_in_one","linux":"linux_x64_all_in_one","macos":"macos_arm64_all_in_one"}[os_name] if mode=="all_in_one" else ("macos_arm64_core" if os_name=="macos" else "core_scripted")',
    '    ext_archive = _acquire_spec(artifacts["bundles"][bkey], work_dir)',
    '    sel = set(CONFIG.get("selected_packs") or ())',
    '    extras = ("PyTorch",) if (mode=="modular" and os_name!="macos" and (sel & {"brain_mask_pnl","anatomy","surface","tractcloud"})) else ()',
    '    receipt = _install_extension(slicer_exe, ext_archive, extras)',
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
    '    selected_packs = list(CONFIG.get("selected_packs") or ()); dirs = _search_dirs()',
    '    if mode == "modular":',
    '        for pid in selected_packs:',
    '            pspec = (artifacts.get("packs") or {}).get(pid)',
    '            if not pspec: continue',
    '            arc = _acquire_spec(pspec, work_dir); _log(f"Extracting {pspec[\'filename\']} -> {asset_root}")',
    '            extract_modular_pack(arc, asset_root, expected_sha256=str(pspec.get("sha256") or ""))',
    '            for rel, urec in (pspec.get("upstream_files") or {}).items():',
    '                dst = asset_root / rel',
    '                if dst.is_file() and _sha256(dst) == urec["sha256"]: continue',
    '                got = False',
    '                for d in dirs:',
    '                    for cand in (d / rel, d / Path(rel).name):',
    '                        if cand.is_file() and _sha256(cand) == urec["sha256"]: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(cand, dst); got = True; break',
    '                    if got: break',
    '                if not got: _download(urec["urls"], dst, expected_sha256=urec["sha256"], expected_size=int(urec.get("size_bytes") or 0), allow_mirror=False)',
    '    cfg = load_user_config(); cfg["ui"]["language"] = CONFIG.get("language", "en")',
    '    deps = cfg.setdefault("external_dependencies", {})',
    '    if custom_assets: deps["model_assets_dir"] = str(asset_root)',
    '    if "tractcloud" in selected_packs: deps["tractcloud_data_dir"] = str((asset_root / "bundles/tractcloud").resolve())',
    '    if "wma_org" in selected_packs: deps["wma_atlas_dir"] = str((asset_root / "bundles/wma").resolve())',
    '    if "brain_mask_pnl" in selected_packs: deps["pnl_mask_model_dir"] = str((asset_root / "brain_mask/pnl_cnn_torch").resolve())',
    '    defaults = cfg.setdefault("run_defaults", {}); targets = ["microstructure", "tractography", "connectome"]',
    '    if "anatomy" in selected_packs: targets.extend(["segmentation", "parcellation"])',
    '    if "surface" in selected_packs: targets.extend(["surface", "saf"])',
    '    if "tractcloud" in selected_packs or "wma_org" in selected_packs:',
    '        targets.append("bundles"); bl = []',
    '        if "wma_org" in selected_packs: bl.append("wma")',
    '        if "tractcloud" in selected_packs: bl.append("tractcloud")',
    '        bl.append("wmql"); defaults["bundle_backend"] = ",".join(bl)',
    '    defaults["targets"] = list(dict.fromkeys(targets)); write_user_config(cfg)',
    '    if mode == "modular" and os_name != "macos":',
    '        ss = mod_root / "scripts" / "setup_selected.py"',
    '        if ss.is_file(): subprocess.run([sys.executable, str(ss), "--targets", ",".join(defaults["targets"]), "--execute", "--accept-licences"], check=True)',
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
    var cfg = {
      os: state.os,
      mode: state.mode,
      slicer_path: state.slicerPath.trim(),
      model_assets_dir: state.modelAssetsDir.trim(),
      selected_packs: state.packs.slice(),
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
      mode: 'modular',
      slicerPath: '',
      modelAssetsDir: '',
      packs: PRESETS.clinical.slice(),
      mirrorPreset: 'direct',
      mirrorPrefix: '',
      proxyUrl: '',
      offlineDir: '',
      checkScriptDir: true,
      moreOpen: false,
      previewOpen: false
    };

    function t(en, cn) { return zh ? cn : en; }

    function totalBytes() {
      if (state.mode === 'all_in_one') {
        var bkey = state.os === 'windows' ? 'windows_x64_all_in_one' : (state.os === 'linux' ? 'linux_x64_all_in_one' : 'macos_arm64_all_in_one');
        var b = ARTIFACTS.bundles[bkey];
        return (b && b.size_bytes) ? b.size_bytes : 3300000000;
      }
      var ckey = state.os === 'macos' ? 'macos_arm64_core' : 'core_scripted';
      var core = ARTIFACTS.bundles[ckey];
      var sum = (core && core.size_bytes) ? core.size_bytes : OS_DEFAULTS[state.os].coreEstimateBytes;
      state.packs.forEach(function (pid) {
        var p = ARTIFACTS.packs[pid];
        if (p) { sum += p.size_bytes || 0; }
      });
      return sum;
    }

    function render() {
      var def = OS_DEFAULTS[state.os];
      var coreBytes = (state.os === 'macos' ? (ARTIFACTS.bundles.macos_arm64_core.size_bytes || def.coreEstimateBytes) : (ARTIFACTS.bundles.core_scripted.size_bytes || def.coreEstimateBytes));
      var html = [];
      html.push('<div class="ic-top">');
      html.push('<div class="ic-top-copy">');
      html.push('<h3 class="ic-heading">' + t('Custom Installer', '定制安装脚本') + '</h3>');
      html.push('<p class="ic-sub">' + t('Pick your Slicer path and workflows; extra models can be added later inside Slicer.', '选好系统与常用模块即可下载脚本运行；未装功能日后可在 Slicer 内按需补齐。') + '</p>');
      html.push('</div>');
      html.push('<div class="ic-seg" role="group" aria-label="' + t('Install mode', '安装模式') + '">');
      html.push('<button type="button" class="ic-seg-btn" data-mode="modular" aria-pressed="' + (state.mode === 'modular' ? 'true' : 'false') + '">' + t('Modular', '按需精简') + '</button>');
      html.push('<button type="button" class="ic-seg-btn" data-mode="all_in_one" aria-pressed="' + (state.mode === 'all_in_one' ? 'true' : 'false') + '">' + t('All-in-One', '全量一体') + '</button>');
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

      if (state.mode === 'modular') {
        html.push('<div class="ic-mod-head"><span class="ic-k">' + t('Modules', '功能模块') + '</span>');
        html.push('<div class="ic-presets">');
        html.push('<button type="button" class="ic-link-btn" data-preset="minimal">' + t('Core only', '仅基础') + '</button>');
        html.push('<button type="button" class="ic-link-btn" data-preset="clinical">' + t('Standard', '常规') + '</button>');
        html.push('<button type="button" class="ic-link-btn" data-preset="full">' + t('All', '全选') + '</button></div></div>');

        html.push('<div class="ic-packs">');
        html.push('<div class="ic-pack is-locked"><input type="checkbox" checked disabled><span class="ic-pack-title">' + t('Core · Preprocessing, DTI/DKI/NODDI, CSD/UKF, Connectome', '核心套件（预处理 / 微结构 / 追踪 / 连接组）') + '</span><span class="ic-pack-size">' + formatBytes(coreBytes) + '</span></div>');

        PACK_META.forEach(function (meta) {
          var pspec = ARTIFACTS.packs[meta.id] || {};
          var checked = state.packs.indexOf(meta.id) !== -1;
          var extraIit = meta.id === 'brain_mask_pnl' ? 7830929 : 0;
          html.push('<label class="ic-pack ' + (checked ? 'is-selected' : '') + '"><input type="checkbox" data-pack="' + meta.id + '"' + (checked ? ' checked' : '') + '><span class="ic-pack-title">' + meta.title + '</span><span class="ic-pack-size">' + formatBytes((pspec.size_bytes || 0) + extraIit) + '</span></label>');
        });
        html.push('</div>');
      } else {
        html.push('<p class="ic-allinone-note">' + t('Automatically fetches and merges split archives (.001, .002, ...), verifies SHA-256, and installs the full offline extension into Slicer.', '自动拉取或合并全部分卷（.001, .002, ...），校验完整性后直接装入 3D Slicer。') + '</p>');
      }

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

    function escapeAttr(v) {
      return String(v || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function bindEvents() {
      Array.prototype.forEach.call(root.querySelectorAll('[data-os]'), function (btn) {
        btn.addEventListener('click', function () { state.os = btn.getAttribute('data-os'); render(); });
      });
      Array.prototype.forEach.call(root.querySelectorAll('[data-mode]'), function (btn) {
        btn.addEventListener('click', function () { state.mode = btn.getAttribute('data-mode'); render(); });
      });
      Array.prototype.forEach.call(root.querySelectorAll('[data-preset]'), function (btn) {
        btn.addEventListener('click', function () {
          state.packs = PRESETS[btn.getAttribute('data-preset')].slice();
          render();
        });
      });
      Array.prototype.forEach.call(root.querySelectorAll('[data-pack]'), function (cb) {
        cb.addEventListener('change', function () {
          var pid = cb.getAttribute('data-pack');
          if (cb.checked && state.packs.indexOf(pid) === -1) { state.packs.push(pid); }
          if (!cb.checked) { state.packs = state.packs.filter(function (x) { return x !== pid; }); }
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
