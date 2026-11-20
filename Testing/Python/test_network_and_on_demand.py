"""Tests for network mirror/proxy config, split merge/tar extract, on-demand detection, and installer generator."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class NetworkConfigTests(unittest.TestCase):
    def test_expand_candidate_urls_applies_mirror_only_to_github_and_keeps_direct_fallback(self) -> None:
        from shared.network_config import expand_candidate_urls

        gh_url = "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1/WMA-ORG.tar.gz"
        nitrc_url = "https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz"

        expanded = expand_candidate_urls([gh_url], mirror_prefix="https://ghfast.top", allow_mirror=True)
        self.assertEqual(expanded, [f"https://ghfast.top/{gh_url}", gh_url])

        nitrc_expanded = expand_candidate_urls([nitrc_url], mirror_prefix="https://ghfast.top/", allow_mirror=True)
        self.assertEqual(nitrc_expanded, [nitrc_url])

        disabled = expand_candidate_urls([gh_url], mirror_prefix="https://ghfast.top/", allow_mirror=False)
        self.assertEqual(disabled, [gh_url])

    def test_save_and_load_network_settings_and_env_overrides(self) -> None:
        from shared.network_config import load_network_settings, network_env_overrides, save_network_settings

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "user_config.json"
            save_network_settings(
                mirror_prefix="https://gh-proxy.com",
                proxy_url="http://127.0.0.1:7890",
                offline_assets_dir="/tmp/offline-packs",
                path=cfg_path,
            )
            from shared.user_config import load_user_config

            loaded_cfg = load_user_config(cfg_path)
            effective = load_network_settings(user_config=loaded_cfg, environ={})
            self.assertEqual(effective["mirror_prefix"], "https://gh-proxy.com/")
            self.assertEqual(effective["proxy_url"], "http://127.0.0.1:7890")
            self.assertEqual(effective["offline_assets_dir"], "/tmp/offline-packs")

            env_over = network_env_overrides(user_config=loaded_cfg, environ={})
            self.assertEqual(env_over["SLICERDSPACE_MIRROR_PREFIX"], "https://gh-proxy.com/")
            self.assertEqual(env_over["HTTPS_PROXY"], "http://127.0.0.1:7890")
            self.assertEqual(env_over["SLICERDSPACE_OFFLINE_DIR"], "/tmp/offline-packs")

    def test_merge_split_archive_and_extract_zip_and_tar_packs(self) -> None:
        from shared.network_config import extract_modular_pack, find_local_pack_archive, merge_split_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_src = root / "Pack-Sample.zip"
            with zipfile.ZipFile(pack_src, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("segmentation/weights/dti_axial.pt", b"weight-data-123")

            raw = pack_src.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            split_dir = root / "splits"
            split_dir.mkdir()
            half = len(raw) // 2
            (split_dir / "Pack-Sample.zip.001").write_bytes(raw[:half])
            (split_dir / "Pack-Sample.zip.002").write_bytes(raw[half:])

            merged = find_local_pack_archive(split_dir, "Pack-Sample.zip", expected_sha256=sha)
            self.assertIsNotNone(merged)
            self.assertEqual(merged.read_bytes(), raw)

            assets_dest = root / "assets"
            extracted = extract_modular_pack(merged, assets_dest, expected_sha256=sha)
            self.assertEqual(extracted, ["segmentation/weights/dti_axial.pt"])
            self.assertEqual((assets_dest / "segmentation/weights/dti_axial.pt").read_bytes(), b"weight-data-123")

            # Streaming .tar.gz extraction with split volumes
            tar_src = root / "WMA-Sample.tar.gz"
            payload = b"naba-cluster-payload"
            with tarfile.open(tar_src, "w:gz") as tf:
                info = tarfile.TarInfo("bundles/wma/naba/NABA-RegAtlas/atlas.vtp")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            tar_raw = tar_src.read_bytes()
            tar_sha = hashlib.sha256(tar_raw).hexdigest()
            tar_split = root / "tar_splits"
            tar_split.mkdir()
            t_half = len(tar_raw) // 2
            (tar_split / "WMA-Sample.tar.gz.001").write_bytes(tar_raw[:t_half])
            (tar_split / "WMA-Sample.tar.gz.002").write_bytes(tar_raw[t_half:])
            merged_tar = find_local_pack_archive(tar_split, "WMA-Sample.tar.gz", expected_sha256=tar_sha)
            self.assertIsNotNone(merged_tar)
            tar_dest = root / "tar_assets"
            extracted_tar = extract_modular_pack(merged_tar, tar_dest, expected_sha256=tar_sha)
            self.assertEqual(extracted_tar, ["bundles/wma/naba/NABA-RegAtlas/atlas.vtp"])
            self.assertEqual((tar_dest / "bundles/wma/naba/NABA-RegAtlas/atlas.vtp").read_bytes(), payload)

            # Reject path traversal zip member
            bad_zip = root / "bad.zip"
            with zipfile.ZipFile(bad_zip, "w") as zf:
                zf.writestr("../escape.txt", b"oops")
            with self.assertRaises(ValueError):
                extract_modular_pack(bad_zip, assets_dest)

            # Wrong checksum discards temporary merged file
            with self.assertRaises(ValueError):
                merge_split_archive(split_dir, "Pack-Sample.zip", expected_sha256="0" * 64, output_dir=root / "badmerge")


class OnDemandResourceDetectionTests(unittest.TestCase):
    def test_detect_missing_task_resources_flags_all_three_wma_atlases_and_clears(self) -> None:
        from shared.ui_on_demand_resources import detect_missing_task_resources

        with tempfile.TemporaryDirectory() as tmp:
            asset_root = Path(tmp) / "model-assets"
            asset_root.mkdir()
            node = SimpleNamespace(
                targetSurface=False,
                targetSurfacePostprocess=False,
                targetSegmentation=False,
                targetParcellation=False,
                targetBundles=True,
                orchestrationBundleBackend="wma",
                wmaAtlas="org",
                brainMaskBackend="provided",
                pnlMaskRoute="native",
            )
            owner = SimpleNamespace(_parameterNode=node)

            for family, expected_id, expected_dirs, index_rel in (
                ("org", "wma_org", ("ORG-800FC-100HCP", "ORG-RegAtlas-100HCP"), "bundles/wma/extracted_resource_index.json"),
                ("naba", "wma_naba", ("NABA-800FC", "NABA-RegAtlas"), "bundles/wma/naba/extracted_resource_index.json"),
                ("east-west", "wma_east_west", ("East-West_WM_Atlas_800FC", "East-West_WM_Atlas_Reg"), "bundles/wma/east-west/extracted_resource_index.json"),
            ):
                missing = detect_missing_task_resources(
                    owner,
                    targets=("bundles", "tractography"),
                    bundle_backends=("wma",),
                    wma_atlas=family,
                    brain_mask_backend="provided",
                    backend_paths={"model_assets_dir": str(asset_root)},
                )
                self.assertEqual([item["id"] for item in missing], [expected_id])
                self.assertEqual(missing[0]["pack_id"], expected_id)

                index_path = asset_root / index_rel
                index_path.parent.mkdir(parents=True, exist_ok=True)
                for child in expected_dirs:
                    (index_path.parent / child).mkdir(parents=True, exist_ok=True)
                index_path.write_text("{}", encoding="utf-8")

                still_missing = detect_missing_task_resources(
                    owner,
                    targets=("bundles", "tractography"),
                    bundle_backends=("wma",),
                    wma_atlas=family,
                    brain_mask_backend="provided",
                    backend_paths={"model_assets_dir": str(asset_root)},
                )
                self.assertEqual(still_missing, [])


class InstallerScriptBuilderTests(unittest.TestCase):
    def test_generates_parseable_scripts_for_base_with_and_without_wma(self) -> None:
        from shared.installer_script_builder import PACK_ORDER, generate_installer_script, script_filename_for_os

        self.assertEqual(PACK_ORDER, ("wma_org", "wma_naba", "wma_east_west"))
        self.assertEqual(script_filename_for_os("windows"), "Install-SlicerDSpace.bat")
        self.assertEqual(script_filename_for_os("macos"), "Install-SlicerDSpace.command")
        self.assertEqual(script_filename_for_os("linux"), "Install-SlicerDSpace.sh")

        for os_name, selected, expected_line_ending in (
            ("windows", ("wma_org",), "\r\n"),
            ("macos", ("wma_org", "wma_naba"), "\n"),
            ("linux", (), "\n"),
        ):
            with self.subTest(os_name=os_name, selected=selected):
                script = generate_installer_script(
                    target_os=os_name,
                    mode="base",
                    slicer_path="/custom/Slicer",
                    model_assets_dir="/custom/models",
                    selected_packs=selected,
                    mirror_prefix="https://ghfast.top/",
                    proxy_url="http://127.0.0.1:7890",
                    offline_dir="/custom/offline",
                    check_script_dir=True,
                    language="zh",
                )
                if expected_line_ending == "\r\n":
                    self.assertIn("\r\n", script)
                else:
                    self.assertNotIn("\r\n", script)
                py_block = script.split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
                ast.parse(py_block)
                self.assertIn("https://ghfast.top/", py_block)
                self.assertIn("http://127.0.0.1:7890", py_block)
                self.assertIn("/custom/offline", py_block)
                self.assertIn("iit_template", py_block)
                self.assertIn({"windows": "windows_x64_base", "macos": "macos_arm64_base", "linux": "linux_x64_base"}[os_name], py_block)
                for p in selected:
                    self.assertIn(p, py_block)

    def test_website_setup_zip_generator_produces_valid_slicer_extension_archives(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "(async () => {\n"
            "  const out = [];\n"
            "  for (const os of ['macos', 'windows', 'linux']) {\n"
            "    const z = await ic.buildSetupZip({\n"
            "      os,\n"
            "      wmaEnabled: true,\n"
            "      wmaAtlases: ['wma_org'],\n"
            "      mirrorPrefix: 'https://ghfast.top/',\n"
            "      proxyUrl: 'http://127.0.0.1:7890',\n"
            "      offlineDir: '/custom/offline',\n"
            "      language: 'zh',\n"
            "    });\n"
            "    const zGeneric = await ic.buildSetupZip({\n"
            "      os,\n"
            "      wmaEnabled: true,\n"
            "      wmaAtlases: ['wma_org'],\n"
            "      autoStart: false,\n"
            "    });\n"
            "    const script = ic.buildScript({\n"
            "      os,\n"
            "      wmaEnabled: true,\n"
            "      wmaAtlases: ['wma_org'],\n"
            "      mirrorPrefix: 'https://ghfast.top/',\n"
            "      proxyUrl: 'http://127.0.0.1:7890',\n"
            "      offlineDir: '/custom/offline',\n"
            "      language: 'zh',\n"
            "    });\n"
            "    out.push({\n"
            "      os,\n"
            "      b64: Buffer.from(z).toString('base64'),\n"
            "      b64Generic: Buffer.from(zGeneric).toString('base64'),\n"
            "      script,\n"
            "    });\n"
            "  }\n"
            "  process.stdout.write(JSON.stringify(out));\n"
            "})().catch((err) => { console.error(err); process.exit(1); });\n"
        )
        proc = subprocess.run(
            [node_bin, "-e", node_script, str(js_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = json.loads(proc.stdout)
        artifacts_doc = json.loads((REPO_ROOT / "docs" / "website" / "installer_artifacts.json").read_text(encoding="utf-8"))
        bundles_doc = artifacts_doc["bundles"]
        expected_hashes = {
            "macos": bundles_doc["macos_arm64_base"]["sha256"],
            "windows": bundles_doc["windows_x64_base"]["sha256"],
            "linux": bundles_doc["linux_x64_base"]["sha256"],
        }
        expected_s4ext_os = {"macos": "macosx", "windows": "win", "linux": "linux"}

        import base64

        for row in rows:
            os_key = row["os"]
            with self.subTest(os=os_key):
                raw_zip = base64.b64decode(row["b64"])
                with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
                    self.assertIsNone(zf.testzip())
                    for info in zf.infolist():
                        self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                    prefix = (
                        "SlicerDSpaceSetup/Slicer.app/Contents/Extensions-34045/SlicerDSpaceSetup"
                        if os_key == "macos"
                        else "SlicerDSpaceSetup"
                    )
                    names = set(zf.namelist())
                    s4ext_rel = f"{prefix}/share/Slicer-5.10/SlicerDSpaceSetup.s4ext"
                    py_rel = f"{prefix}/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpaceSetup.py"
                    self.assertIn(s4ext_rel, names)
                    self.assertIn(f"{prefix}/SlicerDSpaceSetup.s4ext", names)
                    self.assertIn(py_rel, names)

                    s4ext_text = zf.read(s4ext_rel).decode("utf-8")
                    self.assertIn("slicer_revision 34045", s4ext_text)
                    self.assertIn(f"os {expected_s4ext_os[os_key]}", s4ext_text)

                    py_src = zf.read(py_rel).decode("utf-8")
                    tree = ast.parse(py_src)
                    put_calls = [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "put"
                    ]
                    self.assertGreater(len(put_calls), 0)
                    for node in put_calls:
                        self.assertEqual(len(node.args), 1)
                        self.assertIsInstance(node.args[0], ast.Tuple)
                        self.assertEqual(len(node.args[0].elts), 2)
                    self.assertIn("Core Extension", py_src)
                    self.assertIn("SlicerDSpace Base Features (", py_src)
                    self.assertIn("base_cb.setChecked(True); base_cb.setEnabled(False)", py_src)
                    self.assertLess(py_src.index("Core Extension"), py_src.index("Optional WMA White-Matter Atlases"))
                    self.assertIn("Network Mirror && Offline Directory", py_src)
                    self.assertTrue(all(ord(ch) < 128 or ch == "·" for ch in py_src))
                    self.assertIn(expected_hashes[os_key], py_src)
                    self.assertIn("https://ghfast.top/", py_src)
                    self.assertIn('"auto_start":false', py_src)
                    self.assertIn('auto_start=bool(CONFIG.get("auto_start", False))', py_src)

                raw_generic = base64.b64decode(row["b64Generic"])
                with zipfile.ZipFile(io.BytesIO(raw_generic), "r") as zfg:
                    self.assertIsNone(zfg.testzip())
                    generic_py = zfg.read(py_rel).decode("utf-8")
                    self.assertIn('"auto_start":false', generic_py)
                    self.assertIn('"selected_packs":["wma_org"]', generic_py)
                    self.assertIn(expected_hashes[os_key], generic_py)

                cli_block = row["script"].split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
                ast.parse(cli_block)
                self.assertIn(expected_hashes[os_key], cli_block)

        js_text = js_path.read_text(encoding="utf-8")
        for forbidden in ("国内", "国外", "海外", "大陆"):
            self.assertNotIn(forbidden, js_text)

    def test_parallel_range_downloader_migrates_legacy_part_and_verifies_sha256(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        import http.server
        import threading
        import types

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "const files = ic.buildSetupExtensionFiles({ os: 'linux', wmaEnabled: true, wmaAtlases: ['wma_org'], language: 'zh' });\n"
            "const py = files.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data;\n"
            "process.stdout.write(py);\n"
        )
        py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout

        qt_stub = types.ModuleType("qt")
        slicer_stub = types.ModuleType("slicer")
        slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")
        slm_stub.ScriptedLoadableModule = object
        slm_stub.ScriptedLoadableModuleWidget = object
        slicer_stub.ScriptedLoadableModule = slm_stub
        saved_mods = {k: sys.modules.get(k) for k in ("qt", "slicer", "slicer.ScriptedLoadableModule")}
        ns: dict[str, object] = {}
        try:
            sys.modules["qt"] = qt_stub
            sys.modules["slicer"] = slicer_stub
            sys.modules["slicer.ScriptedLoadableModule"] = slm_stub
            exec(compile(py_src, "SlicerDSpaceSetup.py", "exec"), ns)
        finally:
            for k, orig in saved_mods.items():
                if orig is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = orig

        chunk_sz = int(ns["_PAR_CHUNK"])
        self.assertEqual(int(ns["_PAR_WORKERS"]), 12)
        payload = (b"SLICERDSPACE-TEST-BLOCK!" * ((chunk_sz * 5 // 2) // 24 + 1))[: chunk_sz * 5 // 2]
        expected_sha = hashlib.sha256(payload).hexdigest()
        recorded_ranges: list[str] = []
        range_lock = threading.Lock()

        class RangeHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                rng = str(self.headers.get("Range") or "")
                with range_lock:
                    recorded_ranges.append(rng)
                if not rng.startswith("bytes="):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                s_str, e_str = rng[6:].split("-", 1)
                start = int(s_str)
                end = int(e_str) if e_str else len(payload) - 1
                data = payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(data) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        srv_thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "pkg.bin"
                part = Path(td) / "pkg.bin.part"
                meta = Path(td) / "pkg.bin.part.json"
                # Seed legacy sequential partial download covering chunk 0 + partial chunk 1.
                part.write_bytes(payload[: chunk_sz + 524288])
                progress_events: list[tuple[int, int, str]] = []
                logs: list[str] = []
                got = ns["_download"](
                    [f"http://127.0.0.1:{srv.server_address[1]}/pkg.bin"],
                    dest,
                    expected_sha256=expected_sha,
                    expected_size=len(payload),
                    allow_mirror=False,
                    progress_cb=lambda d, t, n: progress_events.append((d, t, n)),
                    log_cb=logs.append,
                )
                self.assertEqual(got, dest)
                self.assertEqual(dest.read_bytes(), payload)
                self.assertFalse(part.exists())
                self.assertFalse(meta.exists())
                self.assertIn("bytes=0-0", recorded_ranges)
                self.assertIn(f"bytes={chunk_sz}-{chunk_sz * 2 - 1}", recorded_ranges)
                self.assertIn(f"bytes={chunk_sz * 2}-{len(payload) - 1}", recorded_ranges)
                self.assertNotIn(f"bytes=0-{chunk_sz - 1}", recorded_ranges)
                self.assertTrue(any("Migrating 1/3 cached chunks" in line for line in logs))
                self.assertTrue(progress_events and progress_events[-1][0] == len(payload))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_setup_extension_startup_opens_dialog_without_auto_starting(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        import types

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        for label, node_state_js in (
            ("website_customizer_defaults", "{ os: 'macos', wmaEnabled: true, wmaAtlases: ['wma_org'], mirrorPrefix: 'https://ghfast.top/' }"),
            ("release_generic_defaults", "{ os: 'macos', wmaEnabled: true, wmaAtlases: ['wma_org'], autoStart: false }"),
        ):
            with self.subTest(mode=label):
                node_script = (
                    "const ic = require(process.argv[1]);\n"
                    f"const files = ic.buildSetupExtensionFiles({node_state_js});\n"
                    "const py = files.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data;\n"
                    "process.stdout.write(py);\n"
                )
                py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout

                class _Signal:
                    def __init__(self) -> None:
                        self.callbacks: list[object] = []

                    def connect(self, fn: object) -> None:
                        self.callbacks.append(fn)

                    def emit(self, *a: object) -> None:
                        for cb in list(self.callbacks):
                            cb(*a)

                class _Widget:
                    def __init__(self, *args: object, **_kwargs: object) -> None:
                        self.args = args
                        self.text = str(args[0]) if args and isinstance(args[0], str) else ""
                        self.checked = False
                        self.enabled = True
                        self.visible = True
                        self.shown = False
                        self.currentIndex = 0
                        self.items: list[tuple[str, object]] = []
                        self.clicked = _Signal()
                        self.finished = _Signal()
                        self.currentIndexChanged = _Signal()

                    def __getattr__(self, name: str) -> object:
                        if name in {"setSpacing", "setWordWrap", "setStyleSheet", "setWindowTitle", "setMinimumWidth", "addWidget", "addLayout", "addStretch", "setContentsMargins", "addRow", "setRange", "setValue", "setReadOnly", "setMaximumHeight", "setPlaceholderText", "setDefault", "appendPlainText", "raise_", "activateWindow"}:
                            return lambda *a, **kw: None
                        raise AttributeError(name)

                    def show(self) -> None:
                        self.shown = True

                    def close(self) -> None:
                        self.shown = False

                    def setText(self, val: object) -> None:
                        self.text = str(val)

                    def setChecked(self, val: bool) -> None:
                        self.checked = bool(val)

                    def isChecked(self) -> bool:
                        return self.checked

                    def setEnabled(self, val: bool) -> None:
                        self.enabled = bool(val)

                    def setVisible(self, val: bool) -> None:
                        self.visible = bool(val)

                    def addItem(self, lbl: str, val: object = None) -> None:
                        self.items.append((lbl, val))

                    def itemData(self, idx: int) -> object:
                        return self.items[idx][1] if 0 <= idx < len(self.items) else ""

                    def setCurrentIndex(self, idx: int) -> None:
                        self.currentIndex = idx

                single_shots: list[tuple[int, object]] = []
                dialogs_created: list[_Widget] = []
                buttons_created: list[_Widget] = []

                class _Dialog(_Widget):
                    def __init__(self, *args: object, **kwargs: object) -> None:
                        super().__init__(*args, **kwargs)
                        dialogs_created.append(self)

                class _Button(_Widget):
                    def __init__(self, *args: object, **kwargs: object) -> None:
                        super().__init__(*args, **kwargs)
                        buttons_created.append(self)

                class _Timer:
                    def __init__(self, *_a: object) -> None:
                        self.timeout = _Signal()
                        self.running = False

                    def setInterval(self, _ms: int) -> None:
                        pass

                    def start(self) -> None:
                        self.running = True

                    def stop(self) -> None:
                        self.running = False

                    @staticmethod
                    def singleShot(ms: int, fn: object) -> None:
                        single_shots.append((ms, fn))

                qt_stub = types.ModuleType("qt")
                for cname in ("QVBoxLayout", "QHBoxLayout", "QFormLayout", "QLabel", "QGroupBox", "QCheckBox", "QComboBox", "QLineEdit", "QWidget", "QProgressBar", "QPlainTextEdit"):
                    setattr(qt_stub, cname, _Widget)
                qt_stub.QDialog = _Dialog
                qt_stub.QPushButton = _Button
                qt_stub.QTimer = _Timer
                qt_stub.QApplication = type("_App", (), {"processEvents": staticmethod(lambda: None)})
                qt_stub.QFileDialog = type("_FD", (), {"getExistingDirectory": staticmethod(lambda *a: "")})

                slicer_stub = types.ModuleType("slicer")
                slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")

                class _BaseSLM:
                    def __init__(self, parent: object) -> None:
                        self.parent = parent

                slm_stub.ScriptedLoadableModule = _BaseSLM
                slm_stub.ScriptedLoadableModuleWidget = _BaseSLM
                slicer_stub.ScriptedLoadableModule = slm_stub
                slicer_stub.util = type("_Util", (), {"mainWindow": staticmethod(lambda: None)})()
                em_stub = type("_EM", (), {"isExtensionInstalled": staticmethod(lambda _name: False)})()
                slicer_stub.app = type(
                    "_SlicerApp",
                    (),
                    {
                        "applicationVersion": "5.10.0",
                        "majorVersion": 5,
                        "minorVersion": 10,
                        "commandOptions": staticmethod(lambda: type("_Opt", (), {"noMainWindow": False})()),
                        "connect": staticmethod(lambda *_a: None),
                        "extensionsManagerModel": staticmethod(lambda: em_stub),
                    },
                )()

                saved_mods = {k: sys.modules.get(k) for k in ("qt", "slicer", "slicer.ScriptedLoadableModule")}
                ns: dict[str, object] = {}
                try:
                    sys.modules["qt"] = qt_stub
                    sys.modules["slicer"] = slicer_stub
                    sys.modules["slicer.ScriptedLoadableModule"] = slm_stub
                    exec(compile(py_src, "SlicerDSpaceSetup.py", "exec"), ns)
                    parent_obj = types.SimpleNamespace()
                    setup_mod = ns["SlicerDSpaceSetup"](parent_obj)
                    setup_mod._onStartupCompleted()
                    self.assertEqual(len(single_shots), 1)
                    _, startup_cb = single_shots.pop(0)
                    startup_cb()
                    self.assertEqual(len(dialogs_created), 1)
                    self.assertTrue(dialogs_created[0].shown)
                    self.assertEqual(single_shots, [])
                    start_btns = [b for b in buttons_created if b.text == "Start Install"]
                    self.assertEqual(len(start_btns), 1)
                    self.assertTrue(start_btns[0].enabled)

                    # Calling show_setup_dialog(auto_start=True) explicitly schedules _on_start.
                    ns["_ACTIVE_DIALOG"] = None
                    ns["show_setup_dialog"](auto_start=True)
                    self.assertEqual(len(single_shots), 1)
                finally:
                    for k, orig in saved_mods.items():
                        if orig is None:
                            sys.modules.pop(k, None)
                        else:
                            sys.modules[k] = orig

    def test_repackage_script_builds_three_release_setup_zips(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        import importlib.util

        script_path = REPO_ROOT / ".github" / "scripts" / "repackage_release_assets.py"
        spec = importlib.util.spec_from_file_location("repackage_release_assets_test", script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        artifacts = json.loads((REPO_ROOT / "docs" / "website" / "installer_artifacts.json").read_text(encoding="utf-8"))
        packs_manifest = json.loads((REPO_ROOT / "models" / "release_packs_manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            setup_meta, written = mod.build_setup_zips(Path(td), artifacts, published=False)
            self.assertEqual([p.name for p in written], [
                "SlicerDSpaceSetup-Windows.zip",
                "SlicerDSpaceSetup-Linux.zip",
                "SlicerDSpaceSetup-macOS.zip",
            ])
            self.assertEqual(set(setup_meta.keys()), {"windows", "linux", "macos"})
            merged_artifacts = dict(artifacts)
            merged_artifacts["setup_installers"] = setup_meta
            sums_text = mod._format_sha256sums(merged_artifacts, packs_manifest)
            for os_key in ("windows", "linux", "macos"):
                rec = setup_meta[os_key]
                self.assertIn(f"{rec['sha256']}  {rec['filename']}", sums_text)

    def test_detect_missing_pnl_template_only_reports_accurate_7_8mb_size(self) -> None:
        from shared.ui_on_demand_resources import detect_missing_task_resources

        with tempfile.TemporaryDirectory() as tmp:
            asset_root = Path(tmp) / "model-assets"
            pnl_dir = asset_root / "brain_mask" / "pnl_cnn_torch"
            pnl_dir.mkdir(parents=True)
            for fname in ("axial.pt", "coronal.pt", "sagittal.pt", "weights.json", "LICENSE.txt"):
                (pnl_dir / fname).write_bytes(b"weights")
            node = SimpleNamespace(
                targetSurface=False,
                targetSurfacePostprocess=False,
                targetSegmentation=False,
                targetParcellation=False,
                targetBundles=False,
                orchestrationBundleBackend="wmql",
                wmaAtlas="org",
                brainMaskBackend="pnl_cnn",
                pnlMaskRoute="native",
            )
            owner = SimpleNamespace(_parameterNode=node)
            missing = detect_missing_task_resources(
                owner,
                targets=("microstructure",),
                bundle_backends=("wmql",),
                brain_mask_backend="pnl_cnn",
                pnl_mask_route="native",
                backend_paths={"model_assets_dir": str(asset_root)},
            )
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["id"], "pnl_cnn")
            self.assertIn("IITmean_b0_256.nii.gz", missing[0]["title_en"])
            self.assertLess(missing[0]["size_bytes"], 10_000_000)

    def test_stage_bundled_assets_reuses_matching_extension_files(self) -> None:
        from unittest.mock import patch
        from scripts.setup_optional_assets import _stage_bundled_assets

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            bundled = repo / "bundled-model-assets" / "brain_mask" / "pnl_cnn_torch"
            bundled.mkdir(parents=True)
            sample_rel = "brain_mask/pnl_cnn_torch/weights.json"
            sample_payload = b"{}"
            sha = hashlib.sha256(sample_payload).hexdigest()
            (bundled / "weights.json").write_bytes(sample_payload)
            dest_root = (Path(tmp) / "user-assets").resolve()
            dest_root.mkdir()
            plan = SimpleNamespace(asset_records={sample_rel: {"sha256": sha, "size_bytes": len(sample_payload)}})
            with patch(
                "scripts.setup_optional_assets.bundled_model_asset_roots",
                return_value=(repo / "bundled-model-assets",),
            ):
                count = _stage_bundled_assets(dest_root, plan, repo_root=REPO_ROOT)
            self.assertEqual(count, 1)
            self.assertEqual((dest_root / sample_rel).read_bytes(), sample_payload)

    def test_setup_py_expand_urls_appends_auto_fallback_mirrors_for_github_only(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        import types

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "const files = ic.buildSetupExtensionFiles({ os: 'macos' });\n"
            "process.stdout.write(files.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data);\n"
        )
        py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout
        qt_stub = types.ModuleType("qt")
        slicer_stub = types.ModuleType("slicer")
        slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")
        slm_stub.ScriptedLoadableModule = object
        slm_stub.ScriptedLoadableModuleWidget = object
        slicer_stub.ScriptedLoadableModule = slm_stub
        saved = {k: sys.modules.get(k) for k in ("qt", "slicer", "slicer.ScriptedLoadableModule")}
        ns: dict[str, object] = {}
        try:
            sys.modules["qt"] = qt_stub
            sys.modules["slicer"] = slicer_stub
            sys.modules["slicer.ScriptedLoadableModule"] = slm_stub
            exec(compile(py_src, "SlicerDSpaceSetup.py", "exec"), ns)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        gh = "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1/a.tar.gz"
        nitrc = "https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz"
        self.assertEqual(
            ns["_expand_urls"]([gh], mirror_prefix="", allow_mirror=True),
            [gh, f"https://ghfast.top/{gh}", f"https://gh-proxy.com/{gh}"],
        )
        self.assertEqual(ns["_expand_urls"]([nitrc], mirror_prefix="", allow_mirror=True), [nitrc])
        self.assertEqual(ns["_expand_urls"]([gh], mirror_prefix="", allow_mirror=False), [gh])


class FastInstallerOptimizationTests(unittest.TestCase):
    """Cross-OS tests for Keep-Alive, single-redirect Range, smart mirror bypass, and worker CLI."""

    def _load_setup_module_ns(self, os_key: str = "windows") -> tuple[str, dict[str, object]]:
        import types

        node_bin = shutil.which("node")
        if not node_bin:
            self.skipTest("Node.js not installed")
        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            f"const files = ic.buildSetupExtensionFiles({{ os: {json.dumps(os_key)}, wmaEnabled: false }});\n"
            "process.stdout.write(files.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data);\n"
        )
        py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout
        qt_stub = types.ModuleType("qt")
        slicer_stub = types.ModuleType("slicer")
        slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")
        slm_stub.ScriptedLoadableModule = type("ScriptedLoadableModule", (), {})
        slm_stub.ScriptedLoadableModuleWidget = type("ScriptedLoadableModuleWidget", (), {})
        ns: dict[str, object] = {"__name__": "SlicerDSpaceSetup"}
        from unittest.mock import patch

        with patch.dict(
            sys.modules,
            {"qt": qt_stub, "slicer": slicer_stub, "slicer.ScriptedLoadableModule": slm_stub},
        ):
            exec(compile(py_src, "SlicerDSpaceSetup.py", "exec"), ns)
        return py_src, ns

    def _start_range_server(self, routes: dict[str, bytes], *, redirect_routes: dict[str, str] | None = None):
        import http.server
        import threading

        redirects = dict(redirect_routes or {})
        hits: list[tuple[str, str]] = []
        hit_lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def handle(self):
                try:
                    super().handle()
                except OSError:
                    pass

            def do_GET(self):
                with hit_lock:
                    hits.append((self.path, str(self.headers.get("Range") or "")))
                if self.path in redirects:
                    dest_path = redirects[self.path]
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}{dest_path}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path not in routes:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = routes[self.path]
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    spec = rng[6:].split(",", 1)[0].strip()
                    s_str, _, e_str = spec.partition("-")
                    start = int(s_str) if s_str else 0
                    end = int(e_str) if e_str else (len(data) - 1)
                    if start < 0 or end >= len(data) or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    chunk = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, hits

    def test_find_python_slicer_across_windows_macos_linux(self) -> None:
        _, ns = self._load_setup_module_ns("macos")
        find_fn = ns["_find_python_slicer"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            win_py = root / "win_slicer" / "bin" / "PythonSlicer.exe"
            win_py.parent.mkdir(parents=True)
            win_py.write_bytes(b"")
            self.assertEqual(find_fn(root / "win_slicer", is_win=True), win_py)

            mac_app = root / "Slicer.app"
            mac_py = mac_app / "Contents" / "bin" / "PythonSlicer"
            mac_py.parent.mkdir(parents=True)
            (mac_app / "Contents" / "MacOS").mkdir(parents=True)
            mac_py.write_bytes(b"")
            mac_py.chmod(0o755)
            for hint in (mac_app, mac_app / "Contents", mac_app / "Contents" / "MacOS"):
                self.assertEqual(find_fn(hint, is_win=False), mac_py)

            lin_py = root / "lin_slicer" / "bin" / "PythonSlicer"
            lin_py.parent.mkdir(parents=True)
            lin_py.write_bytes(b"")
            lin_py.chmod(0o755)
            self.assertEqual(find_fn(root / "lin_slicer", is_win=False), lin_py)

    def test_single_redirect_reuse_and_mirror_worker_cap(self) -> None:
        _, ns = self._load_setup_module_ns("linux")
        wcount = ns["_worker_count_for"]
        self.assertEqual(wcount("https://ghfast.top/https://github.com/a/b.zip", 12), 3)
        self.assertEqual(wcount("https://gh-proxy.com/https://github.com/a/b.zip", 12), 3)
        self.assertEqual(wcount("https://release-assets.githubusercontent.com/a/b.zip", 12), 12)
        self.assertEqual(wcount("http://127.0.0.1:8080/ext.zip", 5), 5)

        ns["_PAR_CHUNK"] = 64 * 1024
        payload = b"ABCD" * (40 * 1024)
        sha = hashlib.sha256(payload).hexdigest()
        server, hits = self._start_range_server(
            {"/cdn/ext.zip": payload},
            redirect_routes={"/releases/download/v0.1.1/ext.zip": "/cdn/ext.zip"},
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "ext.zip"
                ns["_download"](
                    f"http://127.0.0.1:{server.server_port}/releases/download/v0.1.1/ext.zip",
                    dest,
                    expected_sha256=sha,
                    expected_size=len(payload),
                    allow_mirror=False,
                )
                self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(), sha)
        finally:
            server.shutdown()
            server.server_close()

        origin_hits = [p for p, _ in hits if p == "/releases/download/v0.1.1/ext.zip"]
        cdn_hits = [p for p, _ in hits if p == "/cdn/ext.zip"]
        self.assertEqual(len(origin_hits), 1)
        self.assertGreaterEqual(len(cdn_hits), 3)

    def test_smart_direct_mirror_proxy_bypass_vs_explicit_proxy(self) -> None:
        from unittest.mock import patch
        from shared import network_config as nc

        _, ns = self._load_setup_module_ns("windows")
        ns["_SYS_PROXY_CACHE"][0] = None
        orig_detect = ns["_detect_system_proxy"]
        try:
            ns["_detect_system_proxy"] = lambda: "http://127.0.0.1:7897"
            self.assertEqual(ns["_resolve_proxy"]("https://ghfast.top/x", proxy_url="", bypass_sys_proxy=True), "")
            self.assertEqual(
                ns["_resolve_proxy"]("https://ghfast.top/x", proxy_url="", bypass_sys_proxy=False),
                "http://127.0.0.1:7897",
            )
            self.assertEqual(
                ns["_resolve_proxy"]("https://ghfast.top/x", proxy_url="http://127.0.0.1:9999", bypass_sys_proxy=True),
                "http://127.0.0.1:9999",
            )

            seen_attempts: list[tuple[str, bool]] = []
            orig_probe = ns["_probe_range"]

            def rec_probe(url, proxy_url="", expected_size=0, timeout=12, bypass_sys_proxy=False, sess=None):
                seen_attempts.append((url, bypass_sys_proxy))
                return False, 0, url, False, RuntimeError("stop")

            ns["_probe_range"] = rec_probe
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(RuntimeError):
                        ns["_download"](
                            "https://github.com/org/repo/releases/download/v0.1.1/a.zip",
                            Path(tmp) / "a.zip",
                            mirror_prefix="https://ghfast.top/",
                            expected_size=100,
                            proxy_url="",
                        )
                    self.assertEqual(
                        seen_attempts,
                        [
                            ("https://ghfast.top/https://github.com/org/repo/releases/download/v0.1.1/a.zip", True),
                            ("https://ghfast.top/https://github.com/org/repo/releases/download/v0.1.1/a.zip", False),
                            ("https://github.com/org/repo/releases/download/v0.1.1/a.zip", False),
                        ],
                    )
                    seen_attempts.clear()
                    ns["_detect_system_proxy"] = lambda: ""
                    with self.assertRaises(RuntimeError):
                        ns["_download"](
                            "https://github.com/org/repo/releases/download/v0.1.1/a.zip",
                            Path(tmp) / "a.zip",
                            mirror_prefix="https://ghfast.top/",
                            expected_size=100,
                            proxy_url="",
                        )
                    self.assertEqual(
                        seen_attempts,
                        [
                            ("https://ghfast.top/https://github.com/org/repo/releases/download/v0.1.1/a.zip", False),
                            ("https://github.com/org/repo/releases/download/v0.1.1/a.zip", False),
                        ],
                    )
            finally:
                ns["_probe_range"] = orig_probe
        finally:
            ns["_detect_system_proxy"] = orig_detect

        with (
            patch.dict(os.environ, {"HTTP_PROXY": "", "HTTPS_PROXY": "", "SLICERDSPACE_PROXY_URL": ""}, clear=False),
            patch.object(nc, "load_network_settings", return_value={"mirror_prefix": "", "proxy_url": "", "offline_assets_dir": ""}),
            patch.object(nc, "_load_explicit_proxy", return_value=""),
            patch.object(nc, "_detect_system_proxy", return_value="http://127.0.0.1:7897"),
        ):
            self.assertEqual(
                nc._download_attempts_for(["https://ghfast.top/x", "https://github.com/x"], None),
                [("https://ghfast.top/x", True), ("https://ghfast.top/x", False), ("https://github.com/x", False)],
            )
            self.assertEqual(
                nc._download_attempts_for(["https://ghfast.top/x", "https://github.com/x"], "http://127.0.0.1:9999"),
                [("https://ghfast.top/x", False), ("https://github.com/x", False)],
            )
            self.assertEqual(
                nc._resolve_effective_proxy("https://ghfast.top/x", proxy_url="http://127.0.0.1:9999", bypass_system_proxy=True),
                "http://127.0.0.1:9999",
            )
        with (
            patch.dict(os.environ, {"HTTP_PROXY": "", "HTTPS_PROXY": "", "SLICERDSPACE_PROXY_URL": ""}, clear=False),
            patch.object(nc, "load_network_settings", return_value={"mirror_prefix": "", "proxy_url": "", "offline_assets_dir": ""}),
            patch.object(nc, "_load_explicit_proxy", return_value=""),
            patch.object(nc, "_detect_system_proxy", return_value=""),
        ):
            self.assertEqual(
                nc._download_attempts_for(["https://ghfast.top/x", "https://github.com/x"], None),
                [("https://ghfast.top/x", False), ("https://github.com/x", False)],
            )

    def test_sha256sums_auto_refresh_and_fail_fast_on_size_mismatch(self) -> None:
        _, ns = self._load_setup_module_ns("windows")
        ns["_PAR_CHUNK"] = 64 * 1024
        payload = b"NWRK" * 37500
        new_sha = hashlib.sha256(payload).hexdigest()
        sums_bytes = f"{new_sha}  pkg.zip\n".encode("utf-8")

        srv1, _ = self._start_range_server(
            {
                "/releases/download/v0.1.1/pkg.zip": payload,
                "/releases/download/v0.1.1/SHA256SUMS.txt": sums_bytes,
            }
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "pkg.zip"
                ns["_download"](
                    f"http://127.0.0.1:{srv1.server_port}/releases/download/v0.1.1/pkg.zip",
                    dest,
                    expected_sha256="a" * 64,
                    expected_size=100000,
                    allow_mirror=False,
                )
                self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(), new_sha)
        finally:
            srv1.shutdown()
            srv1.server_close()

        srv2, hits2 = self._start_range_server({"/releases/download/v0.1.1/pkg.zip": payload})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    ns["_download"](
                        f"http://127.0.0.1:{srv2.server_port}/releases/download/v0.1.1/pkg.zip",
                        Path(tmp) / "pkg.zip",
                        expected_sha256="a" * 64,
                        expected_size=100000,
                        allow_mirror=False,
                    )
        finally:
            srv2.shutdown()
            srv2.server_close()

        pkg_hits = [r for p, r in hits2 if p == "/releases/download/v0.1.1/pkg.zip"]
        self.assertEqual(pkg_hits, ["bytes=0-0"])

    def test_external_download_worker_cli_emits_progress_and_ready(self) -> None:
        os_key = "windows" if sys.platform == "win32" else ("macos" if sys.platform == "darwin" else "linux")
        bundle_key = {"windows": "windows_x64_base", "macos": "macos_arm64_base", "linux": "linux_x64_base"}[os_key]
        py_src, _ = self._load_setup_module_ns(os_key)

        payload = b"WORKER_ZIP_" * (18 * 1024)
        sha = hashlib.sha256(payload).hexdigest()
        server, _ = self._start_range_server({"/ext.zip": payload})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                cfg = {
                    "os": os_key,
                    "mode": "base",
                    "slicer_path": "",
                    "model_assets_dir": "",
                    "selected_packs": [],
                    "mirror_prefix": "",
                    "proxy_url": "",
                    "offline_dir": "",
                    "check_script_dir": False,
                    "auto_start": False,
                    "language": "en",
                    "artifacts": {
                        "iit_template": {},
                        "packs": {},
                        "bundles": {
                            bundle_key: {
                                "filename": "ext.zip",
                                "url": f"http://127.0.0.1:{server.server_port}/ext.zip",
                                "size_bytes": len(payload),
                                "sha256": sha,
                                "parts": [],
                            }
                        },
                    },
                }
                lines = py_src.splitlines()
                idx = next(i for i, ln in enumerate(lines) if ln.startswith("CONFIG = _json.loads("))
                lines[idx] = f"CONFIG = _json.loads({json.dumps(json.dumps(cfg))})"
                worker_py = tdir / "SlicerDSpaceSetup.py"
                worker_py.write_text("\n".join(lines) + "\n", encoding="utf-8")
                job_file = tdir / "job.json"
                job_file.write_text(
                    json.dumps({"sel_packs": [], "mirror": "", "proxy": "", "offdir": ""}),
                    encoding="utf-8",
                )
                env = os.environ.copy()
                env.update({"TMP": str(tdir), "TEMP": str(tdir), "TMPDIR": str(tdir), "HOME": str(tdir), "USERPROFILE": str(tdir)})
                proc = subprocess.run(
                    [sys.executable, "-u", str(worker_py), "--download-worker", str(job_file)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
                events = []
                for raw in proc.stdout.splitlines():
                    s = raw.strip()
                    if s.startswith("{"):
                        events.append(json.loads(s))
                kinds = [ev["k"] for ev in events]
                self.assertIn("progress", kinds)
                self.assertIn("downloads_ready", kinds)
                ready = next(ev["p"] for ev in events if ev["k"] == "downloads_ready")
                self.assertTrue(ready.get("verified_packs"))
                ext_arc = Path(ready["ext_archive"])
                self.assertTrue(ext_arc.is_file())
                self.assertEqual(hashlib.sha256(ext_arc.read_bytes()).hexdigest(), sha)
        finally:
            server.shutdown()
            server.server_close()

    def test_network_config_download_file_verified_parallel_and_script_builder_all_os(self) -> None:
        from unittest.mock import patch
        from shared import network_config as nc
        from shared.installer_script_builder import generate_installer_script

        payload = b"XYZ1" * (40 * 1024)
        sha = hashlib.sha256(payload).hexdigest()
        server, hits = self._start_range_server({"/pack.tar.gz": payload})
        try:
            url = f"http://127.0.0.1:{server.server_port}/pack.tar.gz"
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ, {"HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": "", "SLICERDSPACE_PROXY_URL": ""}, clear=False),
                patch.object(nc, "_PAR_CHUNK", 64 * 1024),
            ):
                tdir = Path(tmp)
                part = tdir / "pack.tar.gz.part"
                meta = tdir / "pack.tar.gz.part.json"
                part.write_bytes(payload[: 64 * 1024] + b"\x00" * (len(payload) - 64 * 1024))
                meta.write_text(json.dumps({"size": len(payload), "chunk_size": 64 * 1024, "done": [True, False, False]}), encoding="utf-8")

                progress_seen: list[tuple[int, int]] = []
                out_resumed = nc.download_file_verified(
                    url,
                    tdir / "pack.tar.gz",
                    expected_sha256=sha,
                    expected_size=len(payload),
                    allow_mirror=False,
                    progress_callback=lambda d, tot, _u: progress_seen.append((int(d), int(tot))),
                )
                self.assertEqual(hashlib.sha256(out_resumed.read_bytes()).hexdigest(), sha)
                self.assertTrue(progress_seen)
                self.assertEqual(progress_seen[-1][0], len(payload))
                self.assertNotIn("bytes=0-65535", [r for _, r in hits])

                out_fresh = nc.download_file_verified(
                    url,
                    tdir / "fresh.tar.gz",
                    expected_sha256=sha,
                    expected_size=len(payload),
                    allow_mirror=False,
                )
                self.assertEqual(hashlib.sha256(out_fresh.read_bytes()).hexdigest(), sha)

            for target_os in ("windows", "macos", "linux"):
                with self.subTest(target_os=target_os):
                    script = generate_installer_script(target_os=target_os, mode="base", slicer_path="/custom/Slicer")
                    py_block = script.split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
                    ast.parse(py_block)
                    ns: dict[str, object] = {"__file__": str(REPO_ROOT / f"Install-SlicerDSpace-{target_os}.py")}
                    exec(compile(py_block, f"install_{target_os}.py", "exec"), ns)
                    ns["_PAR_CHUNK"] = 64 * 1024
                    with tempfile.TemporaryDirectory() as tmp:
                        out = Path(tmp) / f"ext_{target_os}.bin"
                        ns["_download"](url, out, expected_sha256=sha, expected_size=len(payload), allow_mirror=False)
                        self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), sha)
        finally:
            server.shutdown()
            server.server_close()

    def test_finalize_worker_cli_extracts_pack_stages_iit_and_writes_config(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            asset_dir = tdir / "custom_assets"
            node_script = (
                "const ic = require(process.argv[1]);\n"
                "const files = ic.buildSetupExtensionFiles({ "
                f"os: 'linux', wmaEnabled: true, wmaAtlases: ['wma_naba'], modelAssetsDir: {json.dumps(str(asset_dir))}, language: 'zh' "
                "});\n"
                "process.stdout.write(files.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data);\n"
            )
            py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout
            worker_py = tdir / "SlicerDSpaceSetup.py"
            worker_py.write_text(py_src, encoding="utf-8")

            ext_root = tdir / "SlicerDSpace"
            mod_root = ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
            mod_root.mkdir(parents=True, exist_ok=True)
            bundled = ext_root / "bundled-model-assets"
            (bundled / "sample").mkdir(parents=True, exist_ok=True)
            (bundled / "sample" / "w.bin").write_bytes(b"weights")

            iit_src = tdir / "IITmean_b0_256.nii.gz"
            iit_src.write_bytes(b"fake-iit")
            pack_zip = tdir / "SlicerDSpace-wma-naba-v0.1.1.zip"
            with zipfile.ZipFile(pack_zip, "w") as zf:
                zf.writestr("bundles/wma/naba/atlas.txt", "naba-ok")

            user_cfg = tdir / "user_config.json"
            job_file = tdir / "finalize_job.json"
            job_file.write_text(
                json.dumps({
                    "ext_temps": [],
                    "mod_root": str(mod_root),
                    "iit_file": str(iit_src),
                    "work_dir": str(tdir),
                    "pack_archives": [[{"filename": pack_zip.name}, str(pack_zip)]],
                    "asset_temps": [str(pack_zip)],
                    "sel_packs": ["wma_naba"],
                    "mirror": "",
                    "proxy": "",
                    "offdir": "",
                    "verified_packs": True,
                }),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(REPO_ROOT),
                "SLICERDSPACE_USER_CONFIG": str(user_cfg),
                "HOME": str(tdir),
                "USERPROFILE": str(tdir),
            })
            proc = subprocess.run(
                [sys.executable, "-u", str(worker_py), "--finalize-worker", str(job_file)],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
            kinds = [ev["k"] for ev in events]
            self.assertIn("progress", kinds)
            self.assertIn("all_done", kinds)
            self.assertTrue((asset_dir / "bundles" / "wma" / "naba" / "atlas.txt").is_file())
            self.assertTrue((asset_dir / "brain_mask" / "pnl_cnn_torch" / "IITmean_b0_256.nii.gz").is_file())
            self.assertFalse(pack_zip.exists())
            saved_cfg = json.loads(user_cfg.read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["run_defaults"]["wma_atlas"], "naba")
            self.assertIn("wma", saved_cfg["run_defaults"]["targets"])

    def test_parallel_fast_fail_zero_part_rejection_and_mirror_url_preservation(self) -> None:
        from unittest.mock import patch
        from shared import network_config as nc

        payload = b"REALDATA" * (24 * 1024)
        sha = hashlib.sha256(payload).hexdigest()
        server, hits = self._start_range_server(
            {"/mirror/pkg.bin": payload},
            redirect_routes={"/redir/pkg.bin": "/mirror/pkg.bin"},
        )
        try:
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ, {"HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": "", "SLICERDSPACE_PROXY_URL": ""}, clear=False),
                patch.object(nc, "_PAR_CHUNK", 64 * 1024),
            ):
                tdir = Path(tmp)
                url = f"http://127.0.0.1:{server.server_port}/mirror/pkg.bin"
                dest_nc = tdir / "nc.bin"
                # Seed a zero-filled pre-truncated .part without .part.json (must NOT be treated as completed!).
                dest_nc.with_name("nc.bin.part").write_bytes(b"\x00" * len(payload))
                nc.download_file_verified(url, dest_nc, expected_sha256="", expected_size=len(payload), allow_mirror=False)
                self.assertEqual(dest_nc.read_bytes(), payload)

                _, ns = self._load_setup_module_ns("linux")
                ns["_PAR_CHUNK"] = 64 * 1024
                dest_setup = tdir / "setup.bin"
                dest_setup.with_name("setup.bin.part").write_bytes(b"\x00" * len(payload))
                ns["_download"](url, dest_setup, expected_sha256="", expected_size=len(payload), allow_mirror=False)
                self.assertEqual(dest_setup.read_bytes(), payload)

                # When cap_ref is a mirror host, do not follow a 302 redirect target for Range chunks.
                hits.clear()
                ns["_DIRECT_MIRROR_HOSTS"] = ("127.0.0.1",)
                dest_mir = tdir / "mir.bin"
                ns["_download"](
                    f"http://127.0.0.1:{server.server_port}/redir/pkg.bin",
                    dest_mir,
                    expected_sha256=sha,
                    expected_size=len(payload),
                    allow_mirror=False,
                )
                self.assertEqual(dest_mir.read_bytes(), payload)
                origin_chunk_hits = [p for p, r in hits if p == "/redir/pkg.bin" and r != "bytes=0-0"]
                self.assertGreaterEqual(len(origin_chunk_hits), 3)

                # Fast-fail abort prevents queued remaining chunks from being retried repeatedly.
                import http.server
                import threading

                fail_hits: list[str] = []

                class FailChunkHandler(http.server.BaseHTTPRequestHandler):
                    def log_message(self, *_a):
                        return

                    def do_GET(self):
                        rng = str(self.headers.get("Range") or "")
                        fail_hits.append(rng)
                        if rng == "bytes=0-0":
                            self.send_response(206)
                            self.send_header("Content-Range", f"bytes 0-0/{8192 * 30}")
                            self.send_header("Content-Length", "1")
                            self.end_headers()
                            self.wfile.write(b"0")
                            return
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                fsrv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FailChunkHandler)
                threading.Thread(target=fsrv.serve_forever, daemon=True).start()
                try:
                    with (
                        patch.object(nc, "_PAR_CHUNK", 8192),
                        patch.object(nc, "_PAR_WORKERS", 2),
                    ):
                        with self.assertRaises(RuntimeError):
                            nc.download_file_verified(
                                f"http://127.0.0.1:{fsrv.server_port}/broken.bin",
                                tdir / "broken.bin",
                                expected_size=8192 * 30,
                                allow_mirror=False,
                            )
                    # Out of 30 chunks * 4 attempts = 120 possible requests, abort_event stops early.
                    chunk_attempts = [r for r in fail_hits if r != "bytes=0-0"]
                    self.assertLessEqual(len(chunk_attempts), 12)
                finally:
                    fsrv.shutdown()
                    fsrv.server_close()
        finally:
            server.shutdown()
            server.server_close()

    def test_cross_version_setup_zip_512_and_universal_scripts(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "const f512 = ic.buildSetupExtensionFiles({ os: 'macos', slicerVersion: '5.12.4', wmaEnabled: true, wmaAtlases: ['wma_org'] });\n"
            "const scripts = { win: ic.buildScript({ os: 'windows' }), mac: ic.buildScript({ os: 'macos' }), lin: ic.buildScript({ os: 'linux' }) };\n"
            "process.stdout.write(JSON.stringify({ f512, scripts }));\n"
        )
        out = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout
        payload = json.loads(out)
        names = {item["name"]: item["data"] for item in payload["f512"]}
        prefix = "SlicerDSpaceSetup/Slicer.app/Contents/Extensions-34645/SlicerDSpaceSetup"
        self.assertIn(f"{prefix}/share/Slicer-5.12/SlicerDSpaceSetup.s4ext", names)
        self.assertIn(f"{prefix}/SlicerDSpaceSetup.s4ext", names)
        self.assertIn(f"{prefix}/lib/Slicer-5.12/qt-scripted-modules/SlicerDSpaceSetup.py", names)
        self.assertIn("slicer_revision 34645", names[f"{prefix}/SlicerDSpaceSetup.s4ext"])
        py_src = names[f"{prefix}/lib/Slicer-5.12/qt-scripted-modules/SlicerDSpaceSetup.py"]
        self.assertTrue(all(ord(ch) < 128 or ch == "·" for ch in py_src))
        self.assertIn("/Applications/Slicer*.app", payload["scripts"]["mac"])
        self.assertIn("Slicer*", payload["scripts"]["win"])
        self.assertIn("Slicer*", payload["scripts"]["lin"])
        for scr in payload["scripts"].values():
            self.assertIn("_extract_base_archive_remapped", scr)

    def test_setup_module_version_check_and_remapped_tar_extraction(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        import io
        import tarfile
        import types
        import zipfile

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "const f = ic.buildSetupExtensionFiles({ os: 'macos', slicerVersion: '5.12.4' });\n"
            "process.stdout.write(f.find((x) => x.name.endsWith('SlicerDSpaceSetup.py')).data);\n"
        )
        py_src = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout

        qt_stub = types.ModuleType("qt")
        slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")

        class _BaseSLM:
            def __init__(self, parent: object = None) -> None:
                self.parent = parent

        slm_stub.ScriptedLoadableModule = _BaseSLM
        slm_stub.ScriptedLoadableModuleWidget = _BaseSLM
        slicer_stub = types.ModuleType("slicer")
        slicer_stub.ScriptedLoadableModule = slm_stub
        slicer_stub.app = type("_SlicerApp", (), {"majorVersion": 5, "minorVersion": 12, "patchVersion": 4})()
        saved = {k: sys.modules.get(k) for k in ("qt", "slicer", "slicer.ScriptedLoadableModule")}
        ns: dict[str, object] = {}
        try:
            sys.modules["qt"] = qt_stub
            sys.modules["slicer"] = slicer_stub
            sys.modules["slicer.ScriptedLoadableModule"] = slm_stub
            exec(compile(py_src, "SlicerDSpaceSetup.py", "exec"), ns)

            for maj, minv, expected in ((5, 8, True), (5, 10, True), (5, 12, True), (6, 0, True), (5, 6, False)):
                slicer_stub.app.majorVersion = maj
                slicer_stub.app.minorVersion = minv
                self.assertEqual(ns["_is_supported_slicer"](), expected, (maj, minv))

            self.assertEqual(
                ns["_remap_base_relpath"](
                    "SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py",
                    "Slicer-5.12",
                ),
                "lib/Slicer-5.12/qt-scripted-modules/SlicerDSpace.py",
            )
            self.assertEqual(
                ns["_remap_base_relpath"]("SlicerDSpace/Slicer.app/Contents/Extensions-34045", "Slicer-5.12"),
                "",
            )

            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                stub_zip = tdir / "stub.zip"
                ns["_build_stub_extension_archive"](stub_zip, "34645", "Slicer-5.12", "macosx", "amd64")
                with zipfile.ZipFile(stub_zip, "r") as zf:
                    znames = set(zf.namelist())
                    self.assertIn(
                        "SlicerDSpace/Slicer.app/Contents/Extensions-34645/SlicerDSpace/lib/Slicer-5.12/qt-scripted-modules/SlicerDSpace.py",
                        znames,
                    )

                src_tar = tdir / "base.tar.gz"
                with tarfile.open(src_tar, "w:gz") as tf:
                    files = [
                        ("SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py", b"# mod\n", 0o644),
                        ("SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace/bundled-runtimes/ukf-darwin-arm64/bin/UKFTractography", b"#!/bin/sh\n", 0o755),
                        ("SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace/SlicerDSpace.s4ext", b"slicer_revision 34045\nos macosx\narch amd64\n", 0o644),
                    ]
                    for arc, body, mode in files:
                        ti = tarfile.TarInfo(arc)
                        ti.size = len(body)
                        ti.mode = mode
                        tf.addfile(ti, io.BytesIO(body))
                    sym = tarfile.TarInfo("SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace/bundled-python/bin/python3")
                    sym.type = tarfile.SYMTYPE
                    sym.linkname = "python3.12"
                    tf.addfile(sym)

                ext_root = tdir / "Extensions-34645" / "SlicerDSpace"
                ns["_extract_base_archive_remapped"](src_tar, ext_root, "34645", "Slicer-5.12", slicer_os="macosx", slicer_arch="amd64")
                self.assertTrue((ext_root / "lib" / "Slicer-5.12" / "qt-scripted-modules" / "SlicerDSpace.py").is_file())
                ukf_bin = ext_root / "bundled-runtimes" / "ukf-darwin-arm64" / "bin" / "UKFTractography"
                self.assertTrue(ukf_bin.is_file())
                self.assertTrue(bool(ukf_bin.stat().st_mode & 0o100))
                sym_out = ext_root / "bundled-python" / "bin" / "python3"
                self.assertTrue(sym_out.is_symlink())
                self.assertEqual(os.readlink(sym_out), "python3.12")
                self.assertIn("slicer_revision 34645", (ext_root / "SlicerDSpace.s4ext").read_text(encoding="utf-8"))
        finally:
            for k, orig in saved.items():
                if orig is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = orig

    def test_bundled_python_macho_arch_and_compatibility(self) -> None:
        import struct
        from unittest.mock import patch
        from shared import bundled_python as bp

        def _macho64_le(cpu: int) -> bytes:
            return struct.pack(">I", 0xCFFAEDFE) + struct.pack("<I", cpu) + b"\x00" * 24

        def _fat(cpus: list[int]) -> bytes:
            out = struct.pack(">II", 0xCAFEBABE, len(cpus))
            for cpu in cpus:
                out += struct.pack(">IIIII", cpu, 0, 0, 0, 0)
            return out

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundled-python"
            sp = root / "site-packages"
            sp.mkdir(parents=True)
            (root / "python-runtime.json").write_text(
                json.dumps({"python_version": f"{sys.version_info[0]}.{sys.version_info[1]}"}),
                encoding="utf-8",
            )
            so_file = sp / "mod.so"
            so_file.write_bytes(_macho64_le(0x01000007))  # x86_64
            self.assertEqual(bp._macho_arches(so_file), {"x86_64"})
            with patch.object(bp.sys, "platform", "darwin"), patch.object(bp.platform, "machine", return_value="x86_64"):
                self.assertTrue(bp._runtime_compatible(root))
            with patch.object(bp.sys, "platform", "darwin"), patch.object(bp.platform, "machine", return_value="arm64"):
                self.assertFalse(bp._runtime_compatible(root))

            uni_file = sp / "uni.so"
            uni_file.write_bytes(_fat([0x01000007, 0x0100000C]))
            so_file.write_bytes(_fat([0x01000007, 0x0100000C]))
            self.assertEqual(bp._macho_arches(uni_file), {"x86_64", "arm64"})
            with patch.object(bp.sys, "platform", "darwin"), patch.object(bp.platform, "machine", return_value="arm64"):
                self.assertTrue(bp._runtime_compatible(root))

            # If a single-arch x86_64 .so accompanies universal2 .so, arm64 host still rejects it.
            (sp / "extra.so").write_bytes(_macho64_le(0x01000007))
            with patch.object(bp.sys, "platform", "darwin"), patch.object(bp.platform, "machine", return_value="arm64"):
                self.assertFalse(bp._runtime_compatible(root))

    def test_package_macos_ukf_standalone_tarball(self) -> None:
        import tarfile
        from scripts.build_macos_ukf_standalone import package_standalone_tarball

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            bin_p = tdir / "UKFTractography"
            xml_p = tdir / "UKFTractography.xml"
            bin_p.write_bytes(b"#!/bin/sh\n")
            xml_p.write_text("<executable><name>dwiFile</name></executable>\n", encoding="utf-8")
            arc, sha_p = package_standalone_tarball(
                binary=bin_p,
                xml_path=xml_p,
                source_repo="https://github.com/NeuroDMRI/ukftractography_gpu.git",
                source_commit="deadbeef",
                out_dir=tdir / "out",
            )
            self.assertTrue(arc.is_file())
            self.assertTrue(sha_p.is_file())
            self.assertIn(arc.name, sha_p.read_text(encoding="utf-8"))
            with tarfile.open(arc, "r:gz") as tf:
                names = {m.name: m for m in tf.getmembers()}
                self.assertIn("ukf-darwin-arm64/bin/UKFTractography", names)
                self.assertTrue(bool(names["ukf-darwin-arm64/bin/UKFTractography"].mode & 0o100))
                meta = json.loads(tf.extractfile("ukf-darwin-arm64/runtime.json").read().decode("utf-8"))
                self.assertIs(meta["activation_key_included"], False)
                self.assertIs(meta["slicer_independent"], True)
                self.assertEqual(meta["arch"], "arm64")


if __name__ == "__main__":
    unittest.main()
