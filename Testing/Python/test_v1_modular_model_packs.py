"""Tests for SlicerDSpace v1.0.0 modular model-weight packs, updated DDSurfer weights, and installer parity."""

from __future__ import annotations

import hashlib
import importlib.util
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
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from scripts.package_modular_assets import (
    CORE_MODEL_PACK_ORDER,
    CORE_MODEL_PACK_SPECS,
    collect_model_pack_files,
    manifest_model_pack_entry,
    write_deterministic_tar_gz,
)
from shared.installer_script_builder import generate_installer_script

NEW_DDSURFER_COMMIT = "7d28dbddd73f339b9f1b97e475e35377df767672"
NEW_DDSURFER_HASHES = {
    "surface/weights/SHA256SUMS": "986fe52ba0341e76a6dc9cbc33bb6a52112beb02d70950104de2909c5ff419cc",
    "surface/weights/ddsurfer_lh_wm.pt": "b1ab6ae874e52c80fed4866c8aa03b586c4e7d463b2db688f703627db01a2454",
    "surface/weights/ddsurfer_rh_wm.pt": "da8277a627e55dc576886bebcc3637bee89c1e8a77ce0c77ab0a3163ba955fcd",
    "surface/weights/ddsurfer_lh_pial.pt": "58f93a8ea32ee0c0d59d9ca3875d8ef906fff15dc29c87d1fc313a7a25e28393",
    "surface/weights/ddsurfer_rh_pial.pt": "480961adf7730985b2c2288cac9b66067dcd72b722214ff0ed1183b6dcd0eec0",
}


def _load_repackage_module():
    path = REPO_ROOT / ".github" / "scripts" / "repackage_release_assets.py"
    spec = importlib.util.spec_from_file_location("repackage_release_assets_v1_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_mini_tar_gz(dest: Path, files: dict[str, bytes]) -> tuple[int, str]:
    root = dest.parent / f".src-{dest.stem}"
    hashes: dict[str, str] = {}
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        hashes[rel] = hashlib.sha256(content).hexdigest()
    size, sha = write_deterministic_tar_gz(root, hashes, dest)
    shutil.rmtree(root, ignore_errors=True)
    return size, sha


class V1ModularManifestAndPackTests(unittest.TestCase):
    def test_ddsurfer_manifest_uses_updated_commit_and_hashes(self) -> None:
        manifest = json.loads((REPO_ROOT / "models" / "assets_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["ddsurfer_upstream_commit"], NEW_DDSURFER_COMMIT)
        records = manifest["asset_records"]
        for rel, expected_sha in NEW_DDSURFER_HASHES.items():
            rec = records[rel]
            self.assertEqual(rec["sha256"], expected_sha, rel)
            urls = rec.get("urls") or []
            self.assertTrue(
                any(NEW_DDSURFER_COMMIT in u for u in urls),
                f"Expected pinned upstream URL at {NEW_DDSURFER_COMMIT} for {rel}: {urls}",
            )

        ext = json.loads((REPO_ROOT / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))
        atlas_urls = ext["asset_records"]["surface/postprocessing/atlas.tar.gz"]["urls"]
        self.assertTrue(any(NEW_DDSURFER_COMMIT in u for u in atlas_urls), atlas_urls)

    def test_five_core_model_packs_cover_all_38_bundled_assets_without_iit(self) -> None:
        self.assertEqual(CORE_MODEL_PACK_ORDER, ("ddsurfer", "ddseg", "ddparcel", "pnl_cnn", "tractcloud"))
        all_files: dict[str, str] = {}
        expected_counts = {"ddsurfer": 10, "ddseg": 3, "ddparcel": 16, "pnl_cnn": 5, "tractcloud": 4}
        for pid in CORE_MODEL_PACK_ORDER:
            files = collect_model_pack_files(pid, repo_root=REPO_ROOT)
            self.assertEqual(len(files), expected_counts[pid], pid)
            self.assertFalse(any(p.endswith("IITmean_b0_256.nii.gz") for p in files), pid)
            for rel, sha in files.items():
                self.assertEqual(len(sha), 64)
                self.assertNotIn(rel, all_files, f"Duplicate asset across packs: {rel}")
                all_files[rel] = sha
        self.assertEqual(len(all_files), 38)
        self.assertIn("surface/postprocessing/atlas.tar.gz", all_files)
        self.assertIn("Resources/Data/100HCP-population-mean-T2.nii.gz", all_files)
        for rel, sha in NEW_DDSURFER_HASHES.items():
            self.assertEqual(all_files[rel], sha)

    def test_v011_immutability_guard_blocks_uploads(self) -> None:
        mod = _load_repackage_module()
        with self.assertRaises(RuntimeError):
            mod._assert_mutable_release_tag("v0.1.1", dry_run=False)
        with self.assertRaises(RuntimeError):
            mod._assert_mutable_release_tag("model-assets-v0.1.1", dry_run=False)
        mod._assert_mutable_release_tag("v1.0.0", dry_run=False)
        mod._assert_mutable_release_tag("v0.1.1", dry_run=True)

    def test_repack_os_base_strips_bundled_model_assets_and_passes_audit(self) -> None:
        from scripts.audit_extension_package import audit_package
        from scripts.extension_runtime.archive_layout import inspect_archive_layout

        mod = _load_repackage_module()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src_zip = tmp / "SlicerDSpace-v0.1.1-Windows-x64.zip"
            with zipfile.ZipFile(src_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("SlicerDSpace/SlicerDSpace.s4ext", "arch amd64\nos win\nslicer_revision 34045\n")
                zf.writestr("SlicerDSpace/share/Slicer-5.10/SlicerDSpace.s4ext", "arch amd64\nos win\nslicer_revision 34045\n")
                zf.writestr("SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py", "# old code\n")
                zf.writestr("SlicerDSpace/bundled-python/python-runtime.json", '{"python_version": "3.12"}\n')
                zf.writestr("SlicerDSpace/bundled-python/site-packages/dummy.py", "X = 1\n")
                zf.writestr("SlicerDSpace/bundled-model-assets/surface/weights/ddsurfer_lh_wm.pt", b"old-weight")
                zf.writestr("SlicerDSpace/bundled-model-assets/segmentation/weights/dti_axial.pt", b"old-seg")
                zf.writestr("SlicerDSpace/bundled-model-assets/brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz", b"iit")

            args = SimpleNamespace(
                target="windows-x64",
                work_dir=tmp / "work",
                out_dir=tmp / "out",
                release_tag="v1.0.0",
                source_release_tag="v0.1.1",
                source_archive=src_zip,
                repo="NeuroDMRI/SlicerDSpace",
                keep_bundled_model_assets=False,
                macos_ukf_archive=None,
                standalone_ukf_archive=None,
                skip_macos_ukf=True,
                skip_standalone_ukf=True,
                dry_run=True,
            )
            with patch("shared.backend_sources.prepare_sources", side_effect=RuntimeError("offline-unit-test")):
                rc = mod.cmd_repack_os_base(args)
            self.assertEqual(rc, 0)
            out_zip = tmp / "out" / "SlicerDSpace-v1.0.0-Windows-x64.zip"
            self.assertTrue(out_zip.is_file())
            audit = audit_package(out_zip, REPO_ROOT)
            layout = inspect_archive_layout(out_zip)
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(layout["status"], "pass")
            self.assertEqual(audit["bundled_model_asset_file_count"], 0)
            self.assertGreater(audit["bundled_python_file_count"], 0)
            receipt = json.loads((tmp / "out" / "os-base-receipt-windows-x64.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["windows_x64_base"]["filename"], "SlicerDSpace-v1.0.0-Windows-x64.zip")
            self.assertIn("/releases/download/v1.0.0/", receipt["windows_x64_base"]["url"])

    def test_sync_release_tag_references_passes_chinese_locale_and_no_missing_keys(self) -> None:
        from scripts import build_website_locales as bwl

        mod = _load_repackage_module()
        with tempfile.TemporaryDirectory() as td:
            fake_repo = Path(td)
            for rel in (
                "docs/site/index.html",
                "docs/website/zh-CN.json",
                "shared/installer_script_builder.py",
                ".github/scripts/ci_install_dspace.py",
                "Testing/Python/test_website_editorial_copy.py",
            ):
                dst = fake_repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / rel, dst)

            with patch.object(mod, "REPO_ROOT", fake_repo):
                mod._sync_release_tag_references("v1.0.0")

            idx_html = (fake_repo / "docs" / "site" / "index.html").read_text(encoding="utf-8")
            self.assertIn("https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v1.0.0/SlicerDSpace-v1.0.0-Linux-x64.tar.gz", idx_html)
            self.assertIn("https://github.com/NeuroDMRI/SlicerDSpace/releases/download/assets/SlicerDSpace-Example-HCP-100307-NIfTI.zip", idx_html)
            self.assertIn("SlicerDSpace-v1.0.0-Windows-x64.zip", idx_html)
            self.assertIn("SlicerDSpace-v1.0.0-macOS-arm64.tar.gz", idx_html)
            self.assertNotIn("SlicerDSpace-v0.1.1-Linux-x64.tar.gz", idx_html)
            self.assertIn("download_model_assets.py", idx_html)

            translations = json.loads((fake_repo / "docs" / "website" / "zh-CN.json").read_text(encoding="utf-8"))
            parser = bwl.ChinesePage(translations)
            parser.feed(idx_html)
            parser.close()
            self.assertEqual(sorted(parser.missing), [], f"Missing Chinese translations for v1.0.0 HTML: {parser.missing}")

            # Verify no banned phrases were introduced.
            import Testing.Website.check_site as cs
            import re

            for pat in cs.BANNED:
                m = re.search(pat, idx_html, re.I)
                if m:
                    # Only references catalog paper titles may match; verify our new snippet doesn't.
                    start_cli = idx_html.split('<div class="start-cli"', 1)[1].split('</section>', 1)[0]
                    self.assertIsNone(re.search(pat, start_cli, re.I), f"Banned pattern {pat!r} in v1.0.0 CLI block")


class V1InstallerModularPackFlowTests(unittest.TestCase):
    def _build_fixture(self, td: Path) -> tuple[dict, Path, dict[str, bytes]]:
        offline = td / "offline"
        offline.mkdir(parents=True, exist_ok=True)
        payloads = {
            "ddsurfer": {"surface/weights/ddsurfer_lh_wm.pt": b"ddsurfer-new-v100", "surface/postprocessing/atlas.tar.gz": b"atlas"},
            "ddseg": {"segmentation/weights/dti_axial.pt": b"ddseg-v100"},
            "ddparcel": {"parcellation/weights/Fused-Unet-v3-noMax-axial.pkl": b"ddparcel-v100"},
            "pnl_cnn": {"brain_mask/pnl_cnn_torch/axial.pt": b"pnl-v100"},
            "tractcloud": {"bundles/tractcloud/TrainedModel/best_tract_f1_model.pth": b"tc-v100"},
        }
        model_packs = {}
        expected_files: dict[str, bytes] = {}
        for pid in CORE_MODEL_PACK_ORDER:
            fn = CORE_MODEL_PACK_SPECS[pid]["filename_template"].format(tag="v1.0.0")
            sz, sha = _make_mini_tar_gz(offline / fn, payloads[pid])
            model_packs[pid] = {
                "id": pid,
                "filename": fn,
                "url": f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/assets/{fn}",
                "size_bytes": sz,
                "sha256": sha,
                "published": True,
                "parts": [],
            }
            expected_files.update(payloads[pid])
        iit_bytes = b"iit-template-v100"
        iit_sha = hashlib.sha256(iit_bytes).hexdigest()
        (offline / "IITmean_b0_256.nii.gz").write_bytes(iit_bytes)
        expected_files["brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"] = iit_bytes

        ext_zip = offline / "SlicerDSpace-v1.0.0-Linux-x64.tar.gz"
        ext_zip.write_bytes(b"fake-base")
        artifacts = {
            "release_tag": "v1.0.0",
            "release_base_url": "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v1.0.0",
            "iit_template": {
                "relpath": "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz",
                "sha256": iit_sha,
                "size_bytes": len(iit_bytes),
                "urls": ["http://127.0.0.1:1/IITmean_b0_256.nii.gz"],
            },
            "model_packs": model_packs,
            "packs": {},
            "bundles": {
                "linux_x64_base": {
                    "filename": ext_zip.name,
                    "url": f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v1.0.0/{ext_zip.name}",
                    "size_bytes": ext_zip.stat().st_size,
                    "sha256": hashlib.sha256(ext_zip.read_bytes()).hexdigest(),
                    "parts": [],
                }
            },
        }
        return artifacts, offline, expected_files

    def test_python_cli_installer_installs_all_5_core_packs_by_default_and_supports_subset(self) -> None:
        import test_installer_prune_fallback as prune

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            artifacts, offline, expected_files = self._build_fixture(tmp)

            # 1. Default install: ALL 5 core model packs + IIT extracted automatically, WMA unselected.
            script = generate_installer_script(
                target_os="linux",
                mode="base",
                slicer_path="",
                model_assets_dir=str(tmp / "assets_full"),
                selected_packs=(),
                offline_dir=str(offline),
                check_script_dir=False,
                artifacts=artifacts,
            )
            payload = script.split("# __SLICERDSPACE_PY_BEGIN__\n", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
            ns = prune._load(payload)
            self.assertEqual(ns["CONFIG"]["selected_packs"], [])
            self.assertEqual(ns["CONFIG"]["selected_model_packs"], list(CORE_MODEL_PACK_ORDER))

            project = prune._Project(tmp / "proj_full", release_package=False)
            project.assets = tmp / "assets_full"
            slicer_exe = project.root / "Slicer"
            slicer_exe.write_text("", encoding="utf-8")
            logs: list[str] = []
            ns["_log"] = logs.append
            ns["_pick_supported_slicer"] = lambda exe, explicit: exe
            ns["_warn_linux_glibc_floor"] = lambda *a, **k: None
            ns["_is_intel_mac"] = lambda: False
            ns["_pick_cache_dir"] = lambda need=0: tmp / "cache_full"
            ns["_install_extension_in_slicer"] = lambda *a, **k: {}
            ns["_find_module_root"] = lambda exe, receipt: project.module_root
            with prune._Isolated(project, release_package=False), patch.dict(
                os.environ, {"SLICERDSPACE_RESOLVED_SLICER": str(slicer_exe)}
            ):
                rc = ns["main"]()
            self.assertEqual(rc, 0)
            for rel, content in expected_files.items():
                self.assertEqual((project.assets / rel).read_bytes(), content, rel)

            # 2. Future plugin-style selective install: only ["ddsurfer"].
            script_sub = generate_installer_script(
                target_os="linux",
                mode="base",
                slicer_path="",
                model_assets_dir=str(tmp / "assets_sub"),
                selected_packs=(),
                selected_model_packs=("ddsurfer",),
                offline_dir=str(offline),
                check_script_dir=False,
                artifacts=artifacts,
            )
            payload_sub = script_sub.split("# __SLICERDSPACE_PY_BEGIN__\n", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
            ns_sub = prune._load(payload_sub)
            self.assertEqual(ns_sub["CONFIG"]["selected_model_packs"], ["ddsurfer"])
            project2 = prune._Project(tmp / "proj_sub", release_package=False)
            project2.assets = tmp / "assets_sub"
            slicer_exe2 = project2.root / "Slicer"
            slicer_exe2.write_text("", encoding="utf-8")
            ns_sub["_log"] = lambda _m: None
            ns_sub["_pick_supported_slicer"] = lambda exe, explicit: exe
            ns_sub["_warn_linux_glibc_floor"] = lambda *a, **k: None
            ns_sub["_is_intel_mac"] = lambda: False
            ns_sub["_pick_cache_dir"] = lambda need=0: tmp / "cache_sub"
            ns_sub["_install_extension_in_slicer"] = lambda *a, **k: {}
            ns_sub["_find_module_root"] = lambda exe, receipt: project2.module_root
            with prune._Isolated(project2, release_package=False), patch.dict(
                os.environ, {"SLICERDSPACE_RESOLVED_SLICER": str(slicer_exe2)}
            ):
                rc2 = ns_sub["main"]()
            self.assertEqual(rc2, 0)
            self.assertTrue((project2.assets / "surface/weights/ddsurfer_lh_wm.pt").is_file())
            self.assertFalse((project2.assets / "segmentation/weights/dti_axial.pt").exists())

    def test_gui_setup_phase_a_and_phase_c_install_all_5_core_packs(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not installed in PATH")
        import test_installer_prune_fallback as prune
        from test_installer_gui_reinstall_gap import load_setup_module

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            artifacts, offline, expected_files = self._build_fixture(tmp)
            ns = load_setup_module()
            project = prune._Project(tmp / "proj_gui", release_package=False)
            project.assets = tmp / "assets_gui"
            ns["CONFIG"] = dict(
                ns["CONFIG"],
                os="linux",
                model_assets_dir=str(project.assets),
                selected_packs=[],
                selected_model_packs=list(CORE_MODEL_PACK_ORDER),
                artifacts=artifacts,
            )
            ns["_host_bundle_key"] = lambda: "linux_x64_base"
            ns["_is_intel_mac"] = lambda: False
            ns["_warn_linux_glibc_floor"] = lambda *a, **k: None
            ns["_pick_cache_dir"] = lambda need=0: tmp / "cache_gui"

            events_a: list[tuple[str, object]] = []
            import threading

            ns["_run_phase_a_core"]([], "", "", str(offline), lambda k, p: events_a.append((k, p)), threading.Event())
            ready = next((p for k, p in events_a if k == "downloads_ready"), None)
            self.assertIsNotNone(ready, events_a)
            assert isinstance(ready, dict)
            self.assertEqual(len(ready["model_archives"]), 5)
            self.assertEqual(ready["pack_archives"], [])

            ready["mod_root"] = str(project.module_root)
            events_c: list[tuple[str, object]] = []
            with prune._Isolated(project, release_package=False):
                ns["_run_phase_c_core"](ready, lambda k, p: events_c.append((k, p)))
            errors = [p for k, p in events_c if k == "error"]
            self.assertEqual(errors, [])
            self.assertEqual(events_c[-1][0], "all_done")
            for rel, content in expected_files.items():
                self.assertEqual((project.assets / rel).read_bytes(), content, rel)

    def test_on_demand_pack_matching_for_core_and_optional_models(self) -> None:
        from scripts.download_model_assets import _matching_release_packs, load_manifest, resolve_asset_plan
        from scripts import setup_optional_assets as soa

        self.assertEqual(soa._RELEASE_PACK_BY_PROFILE["pnl-cnn"], "pnl_cnn")
        self.assertEqual(soa._RELEASE_PACK_BY_PROFILE["tractcloud"], "tractcloud")
        self.assertEqual(soa._RELEASE_PACK_BY_PROFILE["surface-postprocess"], "ddsurfer")

        manifest = load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        fake_packs = {}
        for pid in CORE_MODEL_PACK_ORDER:
            files = collect_model_pack_files(pid, repo_root=REPO_ROOT)
            fake_packs[pid] = manifest_model_pack_entry(
                pid, size_bytes=12345, sha256="a" * 64, files=files, published=True, release_tag="v1.0.0"
            )
        with patch("scripts.download_model_assets.load_release_packs_manifest", return_value={"packs": fake_packs}):
            for profile, expected_pid in (("surface", "ddsurfer"), ("segmentation", "ddseg"), ("parcellation", "ddparcel")):
                plan = resolve_asset_plan(manifest, profile)
                matched = [p["id"] for p in _matching_release_packs(plan)]
                self.assertEqual(matched, [expected_pid], profile)
            core_plan = resolve_asset_plan(manifest, "core")
            self.assertEqual({p["id"] for p in _matching_release_packs(core_plan)}, {"ddsurfer", "ddseg", "ddparcel"})


if __name__ == "__main__":
    unittest.main()
