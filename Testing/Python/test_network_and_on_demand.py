"""Tests for network mirror/proxy config, split merge/tar extract, on-demand detection, and installer generator."""

from __future__ import annotations

import ast
import hashlib
import io
import json
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


if __name__ == "__main__":
    unittest.main()
