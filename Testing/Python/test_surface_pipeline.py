#!/usr/bin/env python3
"""Lightweight tests for the headless DDSurfer surface facade."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextlib.contextmanager
def _isolated_surface_asset_roots(base: Path):
    """Point every machine asset root at an empty folder under ``base`` (same patch targets as the late-errors tests)."""
    from shared import model_assets

    env = {key: value for key, value in os.environ.items()
           if key not in (model_assets.MODEL_ASSETS_ENV, "DDSURFER_TEMPLATE_DIR", "DDSURFER_CHECKPOINT_DIR",
                          "DDSURFER_REGISTRATION_ATLAS")}
    env[model_assets.BUNDLED_MODEL_ASSETS_ENV] = str(base / "bundled")
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(model_assets, "_SHA256_CACHE", None))
        stack.enter_context(mock.patch.object(model_assets, "_sha256_cache_path",
                                              return_value=base / "sha_cache.json"))
        stack.enter_context(mock.patch.object(model_assets, "default_user_model_assets_root",
                                              return_value=base / "user-assets"))
        stack.enter_context(mock.patch.object(model_assets, "legacy_repo_asset_roots", return_value=()))
        yield


class SurfacePipelineTest(unittest.TestCase):
    def test_dry_run_writes_manifest_and_qc(self) -> None:
        from anatomy.surface import SurfaceConfig, run_surface

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Opus (third-round check) P2: no machine asset root may reach this dry run.
            with _isolated_surface_asset_roots(root / "isolated"):
                result = run_surface(
                    SurfaceConfig(
                        subject_id="100307",
                        subject_root=root / "100307",
                        output_dir=root / "outputs" / "ddsurfer",
                        execute=False,
                    )
                )

            self.assertFalse(result.executed)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "ddsurfer_surface")
            self.assertEqual(manifest["status"], "planned")
            self.assertIn("available", manifest["resources"])
            self.assertEqual(qc["status"], "planned")

    def test_opus_third_round_p2_dry_run_ignores_the_machine_asset_roots(self) -> None:
        # Opus (third-round check) P2: the dry-run test passed no surface paths, so it raised
        # ModelAssetIntegrityError on a machine whose per-user, bundled or checkout root holds a surface asset that
        # fails its sha256 check. Simulate that machine in an OUTER patch, then run the dry-run test as an inner
        # test: its own isolation (applied inside this patch) must win.
        from shared import model_assets

        relative = "surface/template/100HCP-population-mean-T2-1mm.nii.gz"
        with tempfile.TemporaryDirectory() as tmp:
            dirty = Path(tmp).resolve() / "dirty"
            target = dirty / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(b"wrong bytes: an older release or an incomplete download")
            env = {key: value for key, value in os.environ.items()
                   if key not in (model_assets.MODEL_ASSETS_ENV, "DDSURFER_TEMPLATE_DIR", "DDSURFER_CHECKPOINT_DIR",
                                  "DDSURFER_REGISTRATION_ATLAS")}
            env[model_assets.BUNDLED_MODEL_ASSETS_ENV] = str(dirty)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
                stack.enter_context(mock.patch.object(model_assets, "_sha256_cache_path",
                                                      return_value=Path(tmp) / "outer_sha_cache.json"))
                stack.enter_context(mock.patch.object(model_assets, "_SHA256_CACHE", None))
                stack.enter_context(mock.patch.object(model_assets, "default_user_model_assets_root",
                                                      return_value=dirty))
                stack.enter_context(mock.patch.object(model_assets, "legacy_repo_asset_roots",
                                                      return_value=(dirty,)))
                result = unittest.TestResult()
                SurfacePipelineTest("test_dry_run_writes_manifest_and_qc").run(result)
        problems = [text for _case, text in result.errors + result.failures]
        self.assertEqual(problems, [], "\n".join(problems))
        self.assertEqual(result.testsRun, 1)

    def test_resource_inspection_rejects_legacy_assets(self) -> None:
        from anatomy.surface import SurfaceConfig, inspect_surface_resources

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            checkpoint = root / "ckpts"
            hcp = checkpoint / "hcp"
            template.mkdir()
            hcp.mkdir(parents=True)
            for name in ("100307-FA.nii.gz", "hcp_hemi-left_init_160k.obj", "hcp_hemi-right_init_160k.obj"):
                (template / name).write_text(name, encoding="utf-8")
            for name in (
                "DDCSR_white_lh_Full_DualStream_no_b0_Cascaded.pt",
                "DDCSR_white_rh_Full_DualStream_no_b0_Cascaded.pt",
                "DDCSR_pial_lh_Full_DualStream_no_b0_Cascaded.pt",
                "DDCSR_pial_rh_Full_DualStream_no_b0_Cascaded.pt",
            ):
                (hcp / name).write_text(name, encoding="utf-8")
            atlas = root / "atlas.nii.gz"
            atlas.write_text("atlas", encoding="utf-8")

            status = inspect_surface_resources(
                SurfaceConfig(
                    subject_id="100307",
                    subject_root=root / "100307",
                    output_dir=root / "outputs" / "ddsurfer",
                    template_dir=template,
                    checkpoint_root=checkpoint,
                    registration_atlas=atlas,
                )
            )

            self.assertFalse(status["available"])
            self.assertTrue(status["missing"])


if __name__ == "__main__":
    unittest.main()
