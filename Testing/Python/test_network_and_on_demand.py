"""Tests for network mirror/proxy config, offline split merge/extract, on-demand detection, and installer generator."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class NetworkConfigTests(unittest.TestCase):
    def test_expand_candidate_urls_applies_mirror_only_to_github_and_keeps_direct_fallback(self) -> None:
        from shared.network_config import expand_candidate_urls

        gh_url = "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1/Pack.zip"
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

    def test_merge_split_archive_and_extract_modular_pack(self) -> None:
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
    def test_detect_missing_task_resources_flags_and_clears_per_target(self) -> None:
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
                brainMaskBackend="pnl_cnn",
                pnlMaskRoute="native",
            )
            owner = SimpleNamespace(_parameterNode=node)
            missing = detect_missing_task_resources(
                owner,
                targets=("bundles", "tractography"),
                bundle_backends=("wma",),
                wma_atlas="org",
                brain_mask_backend="provided",
                backend_paths={"model_assets_dir": str(asset_root)},
            )
            self.assertEqual([item["id"] for item in missing], ["wma_org"])
            self.assertEqual(missing[0]["pack_id"], "wma_org")

            # Satisfy ORG directory layout + index file -> no longer reported missing
            wma_dir = asset_root / "bundles" / "wma"
            (wma_dir / "ORG-800FC-100HCP").mkdir(parents=True)
            (wma_dir / "ORG-RegAtlas-100HCP").mkdir(parents=True)
            (wma_dir / "extracted_resource_index.json").write_text("{}", encoding="utf-8")

            still_missing = detect_missing_task_resources(
                owner,
                targets=("bundles", "tractography"),
                bundle_backends=("wma",),
                wma_atlas="org",
                brain_mask_backend="provided",
                backend_paths={"model_assets_dir": str(asset_root)},
            )
            self.assertEqual(still_missing, [])


class InstallerScriptBuilderTests(unittest.TestCase):
    def test_generates_parseable_scripts_for_windows_macos_linux(self) -> None:
        from shared.installer_script_builder import generate_installer_script, script_filename_for_os

        self.assertEqual(script_filename_for_os("windows"), "Install-SlicerDSpace.bat")
        self.assertEqual(script_filename_for_os("macos"), "Install-SlicerDSpace.command")
        self.assertEqual(script_filename_for_os("linux"), "Install-SlicerDSpace.sh")

        for os_name, mode, expected_line_ending in (
            ("windows", "modular", "\r\n"),
            ("macos", "modular", "\n"),
            ("linux", "all_in_one", "\n"),
        ):
            with self.subTest(os_name=os_name, mode=mode):
                script = generate_installer_script(
                    target_os=os_name,
                    mode=mode,
                    slicer_path="/custom/Slicer",
                    model_assets_dir="/custom/models",
                    selected_packs=("brain_mask_pnl", "anatomy"),
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


if __name__ == "__main__":
    unittest.main()
