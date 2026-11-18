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
            "    const script = ic.buildScript({\n"
            "      os,\n"
            "      wmaEnabled: true,\n"
            "      wmaAtlases: ['wma_org'],\n"
            "      mirrorPrefix: 'https://ghfast.top/',\n"
            "      proxyUrl: 'http://127.0.0.1:7890',\n"
            "      offlineDir: '/custom/offline',\n"
            "      language: 'zh',\n"
            "    });\n"
            "    out.push({ os, b64: Buffer.from(z).toString('base64'), script });\n"
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
        expected_hashes = {
            "macos": "aa634ea07b6e4708f8d53091d9dc726a6441e1b52cf74198031959bc28a4648b",
            "windows": "80999ea3caf471ea320ba3d39877ae6877cf12c0d364916243beb1f687d28cc6",
            "linux": "729895ff7c3103d75a9104d4c99ee86cf5173f936d81b39f42ae97c73c91cb59",
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
                    self.assertIn("Network Mirror && Offline Directory", py_src)
                    self.assertIn(expected_hashes[os_key], py_src)
                    self.assertIn("https://ghfast.top/", py_src)

                cli_block = row["script"].split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
                ast.parse(cli_block)
                self.assertIn(expected_hashes[os_key], cli_block)


if __name__ == "__main__":
    unittest.main()
