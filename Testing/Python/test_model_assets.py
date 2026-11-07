#!/usr/bin/env python3
"""Tests for manifest-driven model asset downloads."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_model_assets.py"


def _load_downloader_module():
    spec = importlib.util.spec_from_file_location("download_model_assets", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelAssetsTest(unittest.TestCase):
    def test_default_model_asset_root_is_user_data_directory(self) -> None:
        from shared.model_assets import MODEL_ASSETS_ENV, configured_model_assets_root

        previous = os.environ.pop(MODEL_ASSETS_ENV, None)
        try:
            root = configured_model_assets_root()
        finally:
            if previous is not None:
                os.environ[MODEL_ASSETS_ENV] = previous

        self.assertIn("SlicerDSpace", root.as_posix())
        self.assertNotEqual(root, REPO_ROOT)

    def test_explicit_model_asset_root_wins(self) -> None:
        from shared.model_assets import configured_model_assets_root

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(configured_model_assets_root(Path(tmp)), Path(tmp).resolve())

    def test_core_profile_resolves_manifest_includes_and_required_paths(self) -> None:
        module = _load_downloader_module()

        manifest = module.load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        plan = module.resolve_asset_plan(manifest, "core")

        self.assertEqual(plan.profile, "core")
        self.assertIn("surface", plan.resolved_profiles)
        self.assertIn("segmentation/weights/**", plan.include_patterns)
        self.assertIn("parcellation/weights/**", plan.include_patterns)
        self.assertIn("segmentation/weights/dti_axial.pt", plan.required_paths)
        self.assertIn("segmentation/weights/dti_axial.pt", plan.asset_records)
        self.assertIn("sha256", plan.asset_records["segmentation/weights/dti_axial.pt"])
        self.assertIn("github_lfs", plan.source_ids)
        self.assertIn("slicerdsurfer_models_git", plan.source_ids)
        self.assertIn("DDSURFER_CHECKPOINT_DIR", plan.external_overrides)
        self.assertIn("DDSurfer TANet surface model weights", plan.asset_groups)
        self.assertIn("surface/weights/ddsurfer_lh_wm.pt", plan.required_paths)
        self.assertIn("surface/template/100HCP-population-mean-T2-1mm.nii.gz", plan.required_paths)
        self.assertTrue(any("redistribution" in notice for notice in plan.terms_notices))

    def test_manifest_prefers_independent_model_repository_source(self) -> None:
        module = _load_downloader_module()

        manifest = module.load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        sources = manifest["asset_sources"]

        self.assertEqual(manifest["default_asset_source_ids"][0], "slicerdsurfer_models_git")
        self.assertEqual(sources["slicerdsurfer_models_git"]["status"], "active")
        self.assertIn(
            "https://github.com/ChengjinLii/SlicerDSpace-models/raw/main",
            sources["slicerdsurfer_models_git"]["url_template"],
        )

    def test_external_resource_mirror_uses_new_repository_and_retains_pinned_revision(self) -> None:
        manifest = json.loads((REPO_ROOT / 'models/external_assets_manifest.json').read_text(encoding='utf-8'))
        source = manifest['asset_sources']['optional_models_mirror']
        self.assertEqual(source['url_template'],
                         'https://github.com/ChengjinLii/SlicerDSpace-models/raw/'
                         '11167a58aaced67a69b3e0bc45656aea2abc1c8d/{path_urlencoded}')
        self.assertEqual(source['status'], 'active')

    def test_quick_gate_alias_resolves_to_core_profile(self) -> None:
        module = _load_downloader_module()

        manifest = module.load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        plan = module.resolve_asset_plan(manifest, "quick-gate")

        self.assertEqual(plan.profile, "quick-gate")
        self.assertIn("core", plan.resolved_profiles)
        self.assertIn("surface", plan.resolved_profiles)
        self.assertIn("parcellation/weights/**", plan.include_patterns)
        self.assertIn("segmentation/weights/dti_sagittal.pt", plan.required_paths)

    def test_verify_asset_plan_reports_missing_and_lfs_pointer_files(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "weights" / "model.pt"
            asset.parent.mkdir(parents=True)
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt", "weights/missing.pt"),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("github_lfs",),
                asset_records={},
            )

            asset.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n", encoding="utf-8")
            result = module.verify_asset_plan(root, plan)
            self.assertIn("weights/model.pt", result["lfs_pointers"])
            self.assertIn("weights/missing.pt", result["missing"])
            self.assertFalse(result["ok"])

            (root / "weights" / "missing.pt").write_text("downloaded", encoding="utf-8")
            asset.write_text("downloaded", encoding="utf-8")
            result = module.verify_asset_plan(root, plan)
            self.assertTrue(result["ok"])

    def test_asset_lockfile_records_sha256_and_terms(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "weights" / "model.pt"
            asset.parent.mkdir(parents=True)
            asset.write_text("downloaded", encoding="utf-8")
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=("unit weights",),
                terms_notices=("confirm terms",),
                source_ids=("mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["mirror"],
                        "size_bytes": len("downloaded"),
                    }
                },
            )

            lockfile = module.build_asset_lockfile(root, plan)

            self.assertEqual(lockfile["schema"], "slicerddsurfer.model_assets.lockfile.v1")
            self.assertEqual(lockfile["asset_groups"], ["unit weights"])
            self.assertEqual(lockfile["terms_notices"], ["confirm terms"])
            self.assertEqual(lockfile["records"][0]["path"], "weights/model.pt")
            self.assertFalse(lockfile["records"][0]["lfs_pointer"])
            self.assertIsNotNone(lockfile["records"][0]["sha256"])
            self.assertEqual(lockfile["records"][0]["source_ids"], ["mirror"])

    def test_verify_asset_plan_reports_size_and_sha256_mismatches(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "weights" / "model.pt"
            asset.parent.mkdir(parents=True)
            asset.write_text("bad", encoding="utf-8")
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["mirror"],
                        "size_bytes": 10,
                        "sha256": "0" * 64,
                    }
                },
            )

            result = module.verify_asset_plan(root, plan)

        self.assertFalse(result["ok"])
        self.assertEqual(result["hashes_checked"], 1)
        self.assertEqual(result["size_mismatches"][0]["path"], "weights/model.pt")
        self.assertEqual(result["sha256_mismatches"][0]["path"], "weights/model.pt")

    def test_verify_asset_plan_can_skip_hashes_for_ui_status(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "weights" / "model.pt"
            asset.parent.mkdir(parents=True)
            asset.write_text("downloaded", encoding="utf-8")
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["mirror"],
                        "size_bytes": asset.stat().st_size,
                        "sha256": "0" * 64,
                    }
                },
            )

            result = module.verify_asset_plan(root, plan, check_hashes=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["hashes_checked"], 0)

    def test_url_download_uses_base_url_and_verifies_checksum(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            mirror = tmp_path / "mirror"
            source_file = mirror / "weights" / "model.pt"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("downloaded", encoding="utf-8")
            expected_sha = module._sha256(source_file)
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("unit_mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["unit_mirror"],
                        "size_bytes": source_file.stat().st_size,
                        "sha256": expected_sha,
                    }
                },
            )
            manifest = {
                "asset_sources": {},
            }

            module.download_asset_plan(
                repo_root,
                plan,
                manifest,
                source="url",
                base_urls=(mirror.resolve().as_uri(),),
            )
            result = module.verify_asset_plan(repo_root, plan)

        self.assertTrue(result["ok"])

    def test_url_download_reports_file_and_byte_progress(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            mirror = tmp_path / "mirror"
            source_file = mirror / "weights" / "model.pt"
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"model-data" * 1024)
            source_size = source_file.stat().st_size
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("unit_mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["unit_mirror"],
                        "size_bytes": source_size,
                        "sha256": module._sha256(source_file),
                    }
                },
            )
            events = []

            module.download_asset_plan(
                repo_root,
                plan,
                {"asset_sources": {}},
                source="url",
                base_urls=(mirror.resolve().as_uri(),),
                progress_callback=events.append,
            )

        self.assertTrue(any(event["phase"] == "downloading" for event in events))
        complete = next(event for event in events if event["phase"] == "complete")
        self.assertEqual(complete["path"], "weights/model.pt")
        self.assertEqual(complete["downloaded_bytes"], source_size)
        self.assertEqual(complete["total_bytes"], source_size)
        self.assertEqual(complete["file_index"], 1)
        self.assertEqual(complete["file_count"], 1)

    def test_cancelled_download_keeps_partial_file_for_resume(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            target = root / "target.bin"
            source.write_bytes(b"x" * (3 * 1024 * 1024))
            cancel = {"requested": False}

            def on_progress(event) -> None:
                if event["phase"] == "downloading" and event["downloaded_bytes"] >= 1024 * 1024:
                    cancel["requested"] = True

            with self.assertRaises(module.DownloadCancelledError):
                module._download_url_to_path(
                    source.resolve().as_uri(),
                    target,
                    relative_path="weights/model.bin",
                    expected_size=source.stat().st_size,
                    progress_callback=on_progress,
                    cancel_check=lambda: cancel["requested"],
                )

            partial = target.with_name(target.name + ".part")
            self.assertTrue(partial.is_file())
            self.assertGreaterEqual(partial.stat().st_size, 1024 * 1024)
            self.assertFalse(target.exists())

    def test_content_range_starts_at_validates_rfc7233_header(self) -> None:
        module = _load_downloader_module()

        self.assertTrue(module._content_range_starts_at("bytes 100-999/1000", 100))
        self.assertFalse(module._content_range_starts_at("bytes 0-999/1000", 100))
        self.assertFalse(module._content_range_starts_at("", 100))
        self.assertFalse(module._content_range_starts_at("not-a-content-range", 100))

    def test_oversized_part_file_is_discarded_and_download_restarts(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(b"x" * 1000)  # larger than expected_size below
            source = root / "source.bin"
            source.write_bytes(b"y" * 10)

            module._download_url_to_path(
                source.resolve().as_uri(),
                target,
                relative_path="weights/model.bin",
                expected_size=10,
            )

            self.assertEqual(target.read_bytes(), b"y" * 10)
            self.assertFalse(partial.exists())

    def test_mismatched_content_range_on_resume_discards_the_part_file_and_raises(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(b"a" * 100)

            fake_response = mock.MagicMock()
            fake_response.status = 206
            fake_response.headers = {"Content-Range": "bytes 0-899/900", "Content-Length": "900"}
            fake_response.read.return_value = b""
            fake_response.__enter__.return_value = fake_response
            fake_response.__exit__.return_value = False

            with mock.patch("urllib.request.urlopen", return_value=fake_response):
                with self.assertRaises(module.DownloadIntegrityError):
                    module._download_url_to_path(
                        "https://example.invalid/weights/model.bin",
                        target,
                        relative_path="weights/model.bin",
                        expected_size=900,
                    )

            self.assertFalse(target.exists())
            # A Content-Range that disagrees with the resume offset we asked
            # for means the local .part is not a valid prefix of what the
            # server is serving now (e.g. the remote file changed); keeping
            # it around would just repeat this same mismatch on every future
            # resume attempt, so it must be discarded here (matching the 416
            # handling in test_416_response_discards_stale_part_and_raises).
            self.assertFalse(partial.exists())

    def test_416_response_discards_stale_part_and_raises(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(b"a" * 5)

            import io

            http_error = module.urllib.error.HTTPError(
                "https://example.invalid/weights/model.bin", 416, "Range Not Satisfiable", {}, io.BytesIO(b"")
            )
            self.addCleanup(http_error.close)

            with mock.patch("urllib.request.urlopen", side_effect=http_error):
                with self.assertRaises(module.DownloadIntegrityError):
                    module._download_url_to_path(
                        "https://example.invalid/weights/model.bin",
                        target,
                        relative_path="weights/model.bin",
                        expected_size=5,
                    )

            self.assertFalse(partial.exists())
            self.assertFalse(target.exists())

    def test_incomplete_read_keeps_part_file_for_resume(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            partial = target.with_name(target.name + ".part")

            fake_response = mock.MagicMock()
            fake_response.status = None
            fake_response.headers = {"Content-Length": "1000"}
            fake_response.read.side_effect = [
                b"first-chunk-bytes",
                http.client.IncompleteRead(b"partial", expected=500),
            ]
            fake_response.__enter__.return_value = fake_response
            fake_response.__exit__.return_value = False

            with mock.patch("urllib.request.urlopen", return_value=fake_response):
                with self.assertRaises(module.DownloadIntegrityError):
                    module._download_url_to_path(
                        "https://example.invalid/weights/model.bin",
                        target,
                        relative_path="weights/model.bin",
                        expected_size=1000,
                    )

            self.assertFalse(target.exists())
            self.assertTrue(partial.is_file())
            self.assertEqual(partial.read_bytes(), b"first-chunk-bytes")

    def test_short_read_without_server_error_does_not_promote_the_part_file(self) -> None:
        # Simulates a connection that closes cleanly (empty read = EOF) but
        # before delivering all the bytes the server itself declared -- no
        # exception is raised by http.client, so this must be caught by the
        # downloaded/expected_total size check, not by exception handling.
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.bin"
            partial = target.with_name(target.name + ".part")

            fake_response = mock.MagicMock()
            fake_response.status = None
            fake_response.headers = {"Content-Length": "1000"}
            fake_response.read.side_effect = [b"only-part-of-the-data", b""]
            fake_response.__enter__.return_value = fake_response
            fake_response.__exit__.return_value = False

            with mock.patch("urllib.request.urlopen", return_value=fake_response):
                with self.assertRaises(module.DownloadIntegrityError):
                    module._download_url_to_path(
                        "https://example.invalid/weights/model.bin",
                        target,
                        relative_path="weights/model.bin",
                        expected_size=1000,
                    )

            self.assertFalse(target.exists())
            self.assertTrue(partial.is_file())
            self.assertEqual(partial.read_bytes(), b"only-part-of-the-data")

    def test_force_discards_a_stale_part_file_before_redownloading(self) -> None:
        module = _load_downloader_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            mirror = tmp_path / "mirror"
            source_file = mirror / "weights" / "model.pt"
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"the-real-content")
            target = repo_root / "weights" / "model.pt"
            stale_part = target.with_name(target.name + ".part")
            stale_part.parent.mkdir(parents=True)
            stale_part.write_bytes(b"leftover-garbage-from-a-previous-attempt")
            plan = module.AssetPlan(
                profile="unit",
                resolved_profiles=("unit",),
                all_lfs=False,
                include_patterns=(),
                required_paths=("weights/model.pt",),
                external_overrides=(),
                asset_groups=(),
                terms_notices=(),
                source_ids=("unit_mirror",),
                asset_records={
                    "weights/model.pt": {
                        "path": "weights/model.pt",
                        "source_ids": ["unit_mirror"],
                        "size_bytes": source_file.stat().st_size,
                        "sha256": module._sha256(source_file),
                    }
                },
            )

            module.download_asset_urls(
                repo_root,
                plan,
                {"asset_sources": {}},
                source_ids=("unit_mirror",),
                base_urls=(mirror.resolve().as_uri(),),
                allow_planned_sources=True,
                force=True,
            )

            self.assertEqual(target.read_bytes(), b"the-real-content")
            self.assertFalse(stale_part.exists())

    def test_model_asset_setup_guidance_summarizes_missing_assets(self) -> None:
        module = _load_downloader_module()

        manifest = module.load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        plan = module.resolve_asset_plan(manifest, "core")
        verification = {
            "missing": ["segmentation/weights/dti_axial.pt"],
            "lfs_pointers": ["parcellation/weights/Fused-Unet-v3-noMax-axial.pkl"],
            "size_mismatches": [],
            "sha256_mismatches": [],
        }

        guidance = module.model_asset_setup_guidance(plan, manifest, verification)

        self.assertEqual(guidance["profile"], "core")
        self.assertGreater(guidance["required_file_count"], 0)
        self.assertIn("slicerdsurfer_models_git", guidance["active_url_sources"])
        self.assertIn("--source-id slicerdsurfer_models_git", guidance["recommended_command"])
        self.assertIn("--asset-root", guidance["recommended_command"])
        # S-N2 (guard): setup commands in messages run with PythonSlicer, not a system Python.
        self.assertTrue(guidance["recommended_command"].startswith("PythonSlicer scripts/download_model_assets.py "))
        self.assertTrue(guidance["verify_command"].startswith("PythonSlicer scripts/download_model_assets.py "))
        self.assertIn("target_asset_root", guidance)
        self.assertEqual(guidance["missing_count"], 1)
        self.assertEqual(guidance["lfs_pointer_count"], 1)
        self.assertIn("segmentation/weights/dti_axial.pt", guidance["examples"]["missing"])

    def test_sentinel_profile_is_tiny_url_download_check(self) -> None:
        module = _load_downloader_module()

        manifest = module.load_manifest(REPO_ROOT / "models" / "assets_manifest.json")
        plan = module.resolve_asset_plan(manifest, "sentinel")

        self.assertEqual(plan.required_paths, ("sentinel/slicerdsurfer_sentinel.txt",))
        self.assertEqual(plan.asset_groups, ("download sentinel",))
        self.assertLess(plan.asset_records["sentinel/slicerdsurfer_sentinel.txt"]["size_bytes"], 1024)
        self.assertEqual(plan.source_ids, ("slicerdsurfer_models_git",))


class ModelAssetRootSecurityTest(unittest.TestCase):
    """shared.model_assets: narrowed automatic scan roots + sha256 gating.

    Regression coverage for removing the hardcoded, guessable shared/dev
    scan paths (e.g. /data/shared/..., /opt/..., a specific developer's home
    directory) and for verifying resolved asset roots against
    models/assets_manifest.json before trusting them.
    """

    def setUp(self) -> None:
        import shared.model_assets as model_assets_module

        self.model_assets_module = model_assets_module
        # Isolate the in-process sha256 cache and its on-disk path so tests
        # never touch the real user config directory or leak state between
        # tests (the cache is a lazily-initialized module-level singleton).
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._cache_path = Path(self._cache_tmp.name) / "model_asset_sha256_cache.json"
        model_assets_module._SHA256_CACHE = None
        self._cache_path_patch = mock.patch.object(
            model_assets_module, "_sha256_cache_path", return_value=self._cache_path
        )
        self._cache_path_patch.start()

    def tearDown(self) -> None:
        self._cache_path_patch.stop()
        self.model_assets_module._SHA256_CACHE = None
        self._cache_tmp.cleanup()

    def _write_manifest(self, repo_root: Path, records: dict) -> None:
        manifest_dir = repo_root / "models"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "assets_manifest.json").write_text(
            json.dumps({"asset_records": records}), encoding="utf-8"
        )

    def test_common_model_asset_roots_has_no_hardcoded_shared_or_dev_paths(self) -> None:
        # The only automatically-scanned root is the current user's own
        # per-user default. This is deliberately checked by exact equality,
        # not by rejecting substrings like "chengjin" or "/data/": on the
        # project's own server the real (legitimate) account is named
        # chengjin and its home directory is under /data/, so those
        # substrings appear in a perfectly correct per-user default path
        # too and cannot be used to detect the removed, literal hardcoded
        # entries (e.g. "/data/chengjin/SlicerDDSurfer-model-assets",
        # distinct from that user's real default under ~/.local/share/...).
        roots = self.model_assets_module.common_model_asset_roots()

        self.assertEqual(roots, (self.model_assets_module.default_user_model_assets_root(),))
        removed_literal_roots = (
            Path("/data/shared/SlicerDSpace-model-assets"),
            Path("/data/shared/SlicerDDSurfer-models"),
            Path("/data/chengjin/SlicerDDSurfer-model-assets"),
            Path("/data/chengjin/SlicerDDSurfer-models"),
            Path("/opt/SlicerDSpace/model-assets"),
            Path("/usr/local/share/SlicerDSpace/model-assets"),
            Path("/Users/Shared/SlicerDSpace/model-assets"),
        )
        for removed in removed_literal_roots:
            self.assertNotIn(removed, roots)

    def test_model_asset_path_raises_on_a_file_that_fails_sha256_verification(self) -> None:
        # resolve_model_assets_root() still legitimately falls back to the
        # user's own default directory even when nothing there verifies
        # (that default is also the intended download target when nothing
        # has been fetched yet). The hard security boundary is
        # model_asset_path(): callers hand its result straight to
        # torch.load(), so an existing-but-tampered file there must raise
        # rather than be silently returned.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            planted = user_root / "segmentation" / "weights" / "dti_axial.pt"
            planted.parent.mkdir(parents=True)
            planted.write_bytes(b"a planted, untrusted checkpoint")
            # The manifest expects a different digest than what was planted.
            self._write_manifest(
                repo_root,
                {"segmentation/weights/dti_axial.pt": {"sha256": "0" * 64}},
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                with self.assertRaises(self.model_assets_module.ModelAssetIntegrityError):
                    self.model_assets_module.model_asset_path("segmentation/weights", repo_root=repo_root)

    def test_model_asset_path_or_none_reports_failure_instead_of_raising(self) -> None:
        # Callers that only auto-populate a UI field or report a doctor
        # status (module_widget.py's initializeParameterNode,
        # environment.py's _model_asset_spec_path) must not have a tampered
        # asset abort them entirely -- that would leave the user unable to
        # reach the environment-check/redownload UI at all.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            planted = user_root / "segmentation" / "weights" / "dti_axial.pt"
            planted.parent.mkdir(parents=True)
            planted.write_bytes(b"a planted, untrusted checkpoint")
            self._write_manifest(
                repo_root,
                {"segmentation/weights/dti_axial.pt": {"sha256": "0" * 64}},
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                path, error = self.model_assets_module.model_asset_path_or_none(
                    "segmentation/weights", repo_root=repo_root
                )

            self.assertIsNone(path)
            self.assertIsNotNone(error)
            self.assertIn("does not match", error)

    def test_model_asset_path_or_none_returns_the_path_on_success(self) -> None:
        from shared.model_asset_integrity import sha256_of_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            genuine = user_root / "segmentation" / "weights" / "dti_axial.pt"
            genuine.parent.mkdir(parents=True)
            genuine.write_bytes(b"the real, trusted checkpoint bytes")
            self._write_manifest(
                repo_root,
                {"segmentation/weights/dti_axial.pt": {"sha256": sha256_of_file(genuine)}},
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                path, error = self.model_assets_module.model_asset_path_or_none(
                    "segmentation/weights", repo_root=repo_root
                )

            self.assertIsNone(error)
            self.assertEqual(path, user_root.resolve() / "segmentation" / "weights")

    def test_resolve_accepts_a_root_whose_required_file_matches_the_manifest(self) -> None:
        from shared.model_asset_integrity import sha256_of_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            genuine = user_root / "segmentation" / "weights" / "dti_axial.pt"
            genuine.parent.mkdir(parents=True)
            genuine.write_bytes(b"the real, trusted checkpoint bytes")
            self._write_manifest(
                repo_root,
                {"segmentation/weights/dti_axial.pt": {"sha256": sha256_of_file(genuine)}},
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                resolved = self.model_assets_module.resolve_model_assets_root(
                    repo_root=repo_root, required_paths=("segmentation/weights",)
                )
                asset_path = self.model_assets_module.model_asset_path(
                    "segmentation/weights", repo_root=repo_root
                )

            self.assertEqual(resolved, user_root.resolve())
            self.assertEqual(asset_path, user_root.resolve() / "segmentation" / "weights")

    def test_explicit_root_bypasses_verification(self) -> None:
        # An explicitly user-supplied root is the user's own informed
        # choice, not an automatically-guessed path; it is trusted directly.
        with tempfile.TemporaryDirectory() as tmp:
            explicit_root = Path(tmp) / "explicit-assets"
            explicit_root.mkdir()

            resolved = self.model_assets_module.resolve_model_assets_root(
                explicit=explicit_root, required_paths=("segmentation/weights",)
            )

            self.assertEqual(resolved, explicit_root.resolve())

    def test_directory_with_no_manifest_coverage_falls_back_to_plain_existence(self) -> None:
        # A required directory the manifest does not track at all (nothing
        # recorded to verify against) keeps the old, best-effort behavior.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            unlisted_dir = user_root / "Resources" / "Data" / "unlisted"
            unlisted_dir.mkdir(parents=True)
            (unlisted_dir / "some_file.bin").write_bytes(b"not tracked by the manifest")
            self._write_manifest(repo_root, {})

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                resolved = self.model_assets_module.resolve_model_assets_root(
                    repo_root=repo_root, required_paths=("Resources/Data/unlisted",)
                )

            self.assertEqual(resolved, user_root.resolve())

    def test_directory_with_one_stale_file_among_several_is_rejected(self) -> None:
        # A directory containing several manifest-recorded files, all but
        # one of which verify, must be rejected outright -- not accepted
        # because "at least one" file matched. The caller only verifies at
        # the directory level; a substituted/tampered file the caller later
        # loads by itself (e.g. one of several backbone checkpoints picked
        # at runtime) would otherwise never be re-checked.
        from shared.model_asset_integrity import sha256_of_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            weights_dir = user_root / "parcellation" / "weights" / "backbones"
            weights_dir.mkdir(parents=True)
            genuine_a = weights_dir / "axial.pkl"
            genuine_b = weights_dir / "coronal.pkl"
            stale = weights_dir / "sagittal.pkl"
            genuine_a.write_bytes(b"genuine axial weights")
            genuine_b.write_bytes(b"genuine coronal weights")
            stale.write_bytes(b"a stale/substituted file")
            self._write_manifest(
                repo_root,
                {
                    "parcellation/weights/backbones/axial.pkl": {"sha256": sha256_of_file(genuine_a)},
                    "parcellation/weights/backbones/coronal.pkl": {"sha256": sha256_of_file(genuine_b)},
                    # Manifest records a *different* digest than what is on
                    # disk for this one file.
                    "parcellation/weights/backbones/sagittal.pkl": {"sha256": "0" * 64},
                },
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                with self.assertRaises(self.model_assets_module.ModelAssetIntegrityError):
                    self.model_assets_module.model_asset_path(
                        "parcellation/weights/backbones", repo_root=repo_root
                    )

    def test_directory_with_all_recorded_files_verified_is_accepted(self) -> None:
        # The positive counterpart: every recorded file present verifies,
        # so the directory as a whole is accepted.
        from shared.model_asset_integrity import sha256_of_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            weights_dir = user_root / "parcellation" / "weights" / "backbones"
            weights_dir.mkdir(parents=True)
            genuine_a = weights_dir / "axial.pkl"
            genuine_b = weights_dir / "coronal.pkl"
            genuine_a.write_bytes(b"genuine axial weights")
            genuine_b.write_bytes(b"genuine coronal weights")
            self._write_manifest(
                repo_root,
                {
                    "parcellation/weights/backbones/axial.pkl": {"sha256": sha256_of_file(genuine_a)},
                    "parcellation/weights/backbones/coronal.pkl": {"sha256": sha256_of_file(genuine_b)},
                },
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                resolved = self.model_assets_module.model_asset_path(
                    "parcellation/weights/backbones", repo_root=repo_root
                )

            self.assertEqual(resolved, weights_dir.resolve())

    def test_verification_always_persists_the_sha256_cache(self) -> None:
        # Every code path that computes a digest must persist it, not just
        # some of them -- otherwise a large weight file gets re-hashed on
        # every single call instead of only when it actually changes.
        from shared.model_asset_integrity import sha256_of_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            weights_dir = user_root / "segmentation" / "weights"
            weights_dir.mkdir(parents=True)
            asset = weights_dir / "dti_axial.pt"
            asset.write_bytes(b"genuine weights")
            self._write_manifest(
                repo_root,
                {"segmentation/weights/dti_axial.pt": {"sha256": sha256_of_file(asset)}},
            )

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                self.model_assets_module.model_asset_path("segmentation/weights", repo_root=repo_root)

            on_disk_cache = self.model_assets_module.load_sha256_cache(self._cache_path)
            self.assertIn(str(asset), on_disk_cache)
            self.assertEqual(on_disk_cache[str(asset)]["sha256"], sha256_of_file(asset))


if __name__ == "__main__":
    unittest.main()
