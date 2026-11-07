"""Batch-backlog round, part B: run record items B3 and B4. Each test names its item."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "Testing" / "Python"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import user_data_isolation  # noqa: E402

user_data_isolation.install()

from shared import run_record  # noqa: E402
from shared.run_record import build_run_record, write_run_record  # noqa: E402


def _task(name, status="completed"):
    return {"name": name, "status": status, "outputs": {}}


def _aggregate(tmp, *, crashed_second=True):
    """Aggregate manifest with two children; the second one has the checkpoint shape (it crashed)."""
    meta = Path(tmp) / "out" / "00_run_metadata"
    meta.mkdir(parents=True)
    full = dict(kind="slicerddsurfer_product_orchestration", tractography_backend="csd_act", run_name="cmp__csd_act",
                wma_atlas="naba", brain_mask_backend="synthstrip", model_assets_dir=None, status="completed",
                tasks=[_task("bundles.wma")])
    checkpoint = dict(kind="slicerddsurfer_product_orchestration", version=1, status="running", subject_id="S1",
                      run_name="cmp__ukf", targets=["bundles"], execute=True, tasks=[_task("bundles.wma", "failed")])
    children = []
    for backend, data in (("csd_act", full), ("ukf", checkpoint if crashed_second else full)):
        path = meta / f"cmp__{backend}_orchestration_manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        children.append(dict(backend=backend, manifest_path=str(path), status="completed"))
    aggregate = meta / "cmp_orchestration_manifest.json"
    aggregate.write_text(json.dumps(dict(
        kind="slicerddsurfer_product_orchestration", subject_id="S1", run_name="cmp", status="failed",
        output_root=str(Path(tmp) / "out"), tractography_backends=["csd_act", "ukf"], child_runs=children,
        tasks=[_task("bundles.wma@csd_act")])), encoding="utf-8")
    return aggregate, [Path(child["manifest_path"]) for child in children]


class B3RecordTest(unittest.TestCase):
    def test_b3_1_a_crashed_child_does_not_drop_the_settings_of_the_finished_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate, _ = _aggregate(tmp)
            record = build_run_record(aggregate)
        parameters = record["parameters"]
        self.assertEqual(parameters["wma_atlas"], "naba")
        self.assertEqual(parameters["brain_mask_backend"], "synthstrip")
        self.assertIn("wma-naba", record["assets"]["model_profiles"])
        self.assertNotIn("wma", record["assets"]["model_profiles"])

    def test_b3_3_the_record_lists_the_child_manifests_it_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate, children = _aggregate(tmp)
            (children[1]).unlink()  # an unreadable child is not listed
            record = build_run_record(aggregate)
            expected = [{"backend": "csd_act", "path": children[0].as_posix(),
                         "sha256": hashlib.sha256(children[0].read_bytes()).hexdigest()}]
        self.assertEqual(record["child_manifests"], expected)

    def test_b3_3_a_single_run_has_no_child_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps({"subject_id": "s", "tasks": []}), encoding="utf-8")
            self.assertEqual(build_run_record(manifest)["child_manifests"], [])

    # ---- B3 (2): a verified folder counts as managed
    manifest = {"brain_mask_backend": "synthstrip", "tasks": [_task("segmentation.ddseg")]}

    def test_b3_2_the_default_managed_root_saved_by_the_gui_is_not_hashed(self) -> None:
        from shared.model_assets import default_user_model_assets_root

        root = default_user_model_assets_root()
        with patch("shared.model_asset_integrity.sha256_of_file", side_effect=AssertionError("must not hash")):
            profiles = run_record.asset_versions(dict(self.manifest, model_assets_dir=str(root)))
        self.assertEqual(profiles["segmentation"]["manifest"], "models/assets_manifest.json")
        self.assertTrue(profiles["segmentation"]["files"])

    def _catalog_folder(self, tmp):
        data = b"weights"
        digest = hashlib.sha256(data).hexdigest()
        catalog = {"manifest": "m.json", "version": "9", "records": {"seg/w.pt": {"sha256": digest}}, "path": "m.json",
                   "profiles": {"segmentation": {"required_paths": ["seg/w.pt"]}}, "upstream_commit": None}
        custom = Path(tmp) / "offline models"
        (custom / "seg").mkdir(parents=True)
        (custom / "seg" / "w.pt").write_bytes(data)
        return custom, catalog, digest

    def test_b3_2_a_folder_with_a_verifying_receipt_counts_as_managed_without_hashing(self) -> None:
        from shared.user_config import default_user_config_path

        with tempfile.TemporaryDirectory() as tmp:
            custom, catalog, digest = self._catalog_folder(tmp)
            receipt = default_user_config_path().parent / "offline-assets-b3test.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            self.addCleanup(lambda: receipt.unlink(missing_ok=True))
            guard = patch("shared.model_asset_integrity.sha256_of_file", side_effect=AssertionError("must not hash"))
            with patch.object(run_record, "_load_asset_manifests", return_value=[catalog]), \
                    patch("shared.model_assets._sha256_cache", return_value={}), guard:
                # Without a receipt the folder would be hashed (the guard raises): so no receipt, no claim.
                receipt.unlink(missing_ok=True)
                with self.assertRaises(AssertionError):
                    run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))
                receipt.write_text(json.dumps({
                    "kind": "slicerdsurfer_offline_assets", "status": "completed", "root": str(custom.resolve()),
                    "checks": {"segmentation": {"ok": True}}}), encoding="utf-8")
                profiles = run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))
                self.assertEqual(profiles["segmentation"]["files"], {"seg/w.pt": digest})
                # A catalogued file newer than the receipt: the receipt no longer covers it.
                future = receipt.stat().st_mtime + 100
                os.utime(custom / "seg" / "w.pt", (future, future))
                with self.assertRaises(AssertionError):
                    run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))

    def test_b3_4_a_custom_pnl_model_dir_is_not_recorded_with_the_catalog_hashes(self) -> None:
        manifest = {"brain_mask_backend": "pnl_cnn", "tasks": [_task("preprocessing.dwi")]}
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "my pnl weights"
            weights.mkdir()
            (weights / "axial.pt").write_bytes(b"other weights")
            custom = run_record.asset_versions(dict(manifest, brain_mask_configuration={"pnl_mask_model_dir": str(weights)}))
            from_env = None
            with patch.dict(os.environ, {"SLICERDSPACE_PNL_MASK_MODEL_DIR": str(weights)}):
                from_env = run_record.asset_versions(dict(manifest))
            default = run_record.asset_versions(dict(manifest, brain_mask_configuration={"pnl_mask_model_dir": None}))
        for found in (custom, from_env):
            self.assertEqual(found["pnl-cnn-weights"],
                             {"manifest": None, "version": "unknown", "folder": weights.as_posix(), "files": {}})
        self.assertEqual(default["pnl-cnn-weights"]["manifest"], "models/external_assets_manifest.json")

    def test_b3_5_accepted_licences_written_by_the_setup_step_are_read_into_the_record(self) -> None:
        from shared.user_config import empty_user_config

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "user_config.json"
            payload = empty_user_config()
            payload["accepted_licences"] = {"freesurfer": {"id": "freesurfer", "version": "ab" * 32, "time": "2026-11-07T10:00:00Z"}}
            config.write_text(json.dumps(payload), encoding="utf-8")
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps({"subject_id": "s", "tasks": []}), encoding="utf-8")
            with patch.dict(os.environ, {"SLICERDSPACE_USER_CONFIG": str(config)}):
                record = build_run_record(manifest)
        self.assertEqual(record["licence_acceptances"],
                         [{"id": "freesurfer", "version": "ab" * 32, "time": "2026-11-07T10:00:00Z"}])


if __name__ == "__main__":
    unittest.main()
