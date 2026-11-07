"""W-B8: setup receipts are validated and saved without a GUI owner."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.selected_setup import apply_setup_receipt
from shared.user_config import load_user_config


class SetupReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "user_config.json"
        patcher = patch.dict(os.environ, {"SLICERDSPACE_USER_CONFIG": str(self.config)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)

    def saved(self):
        return load_user_config(self.config)["external_dependencies"]

    def runtime_receipt(self, **settings):
        return {"kind": "slicerdsurfer_runtime_setup", "status": "completed", "external_dependencies": settings}

    def test_managed_runtime_receipt_is_saved_and_returned(self):
        folder = self.root / "bin"
        folder.mkdir()
        returned = apply_setup_receipt({"id": "managed_runtime"}, self.runtime_receipt(mrtrix_bin_dir=str(folder)))
        self.assertEqual(returned, {"mrtrix_bin_dir": str(folder)})
        self.assertEqual(self.saved(), returned)

    def test_surface_runtime_receipt_is_saved(self):
        folder = self.root / "surface"
        folder.mkdir()
        apply_setup_receipt({"id": "surface_runtime"}, self.runtime_receipt(postprocess_runtime=str(folder)))
        self.assertEqual(self.saved(), {"postprocess_runtime": str(folder)})

    def test_runtime_receipt_with_unexpected_key_or_missing_path_or_failure_is_refused(self):
        folder = self.root / "bin"
        folder.mkdir()
        step = {"id": "managed_runtime"}
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            apply_setup_receipt(step, self.runtime_receipt(python_executable=str(folder)))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            apply_setup_receipt(step, self.runtime_receipt(mrtrix_bin_dir=str(self.root / "missing")))
        with self.assertRaisesRegex(ValueError, "completed"):
            apply_setup_receipt(step, {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            apply_setup_receipt(step, self.runtime_receipt(pnl_mask_python=str(folder)))
        self.assertFalse(self.config.exists())

    def test_optional_resources_receipt(self):
        assets = self.root / "models"
        atlas = assets / "atlas"
        atlas.mkdir(parents=True)
        step = {"id": "optional_resources", "profiles": ["wma"], "root": str(assets)}
        receipt = {"kind": "slicerdsurfer_optional_asset_setup", "status": "completed", "profiles": ["wma"],
                   "root": str(assets), "external_dependencies": {"model_assets_dir": str(assets), "wma_atlas_dir": str(atlas)}}
        returned = apply_setup_receipt(step, receipt)
        self.assertEqual(self.saved(), returned)
        self.assertEqual(set(returned), {"model_assets_dir", "wma_atlas_dir"})
        for broken in (dict(receipt, profiles=["tractcloud"]), dict(receipt, root="/elsewhere"), dict(receipt, status="planned"),
                       dict(receipt, external_dependencies={"model_assets_dir": str(assets)}),
                       dict(receipt, external_dependencies={"model_assets_dir": str(assets), "wma_atlas_dir": str(assets / "gone")})):
            with self.assertRaises(ValueError):
                apply_setup_receipt(step, broken)

    def test_backend_sources_receipt_checks_the_source_with_verify_source(self):
        from shared.backend_sources import SOURCES, profile_path
        step = {"id": "backend_sources", "profiles": ["wmql"], "root": str(self.root / "src")}
        expected = {SOURCES["wmql"]["setting"]: str(profile_path("wmql", step["root"]))}
        receipt = {"status": "completed", "profiles": ["wmql"], "external_dependencies": expected}
        with patch("shared.backend_sources.verify_source") as verify:
            returned = apply_setup_receipt(step, receipt)
        verify.assert_called_once()
        self.assertEqual(self.saved(), returned)
        with patch("shared.backend_sources.verify_source", side_effect=ValueError("Algorithm source changed: x")):
            with self.assertRaisesRegex(ValueError, "changed"):
                apply_setup_receipt(step, receipt)
        with self.assertRaisesRegex(ValueError, "Invalid prepared"):
            apply_setup_receipt(step, dict(receipt, external_dependencies={"wma_root": str(self.root)}))

    def test_model_assets_step_saves_its_folder_and_refuses_a_missing_one(self):
        folder = self.root / "models"
        folder.mkdir()
        self.assertEqual(apply_setup_receipt({"id": "model_assets", "asset_root": str(folder)}, None),
                         {"model_assets_dir": str(folder)})
        self.assertEqual(self.saved(), {"model_assets_dir": str(folder)})
        with self.assertRaises(ValueError):
            apply_setup_receipt({"id": "model_assets", "asset_root": str(self.root / "gone")}, None)

    def test_steps_without_settings_save_nothing_and_unknown_steps_are_refused(self):
        self.assertEqual(apply_setup_receipt({"id": "python_packages"}, {"status": "completed"}), {})
        self.assertFalse(self.config.exists())
        with self.assertRaises(ValueError):
            apply_setup_receipt({"id": "mystery"}, {})


if __name__ == "__main__":
    unittest.main()
