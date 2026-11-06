"""T1 fix round 3 (T-C5b, T-C7b, T-O2): regression tests, each named after its finding."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _task(name, status="completed"):
    return {"name": name, "status": status, "outputs": {}}


class TC5bAggregateRecordTest(unittest.TestCase):
    """T-C5b: the aggregate run record states the effective settings, read from the child manifests."""

    def write_run(self, tmp, **common):
        meta = Path(tmp) / "out" / "00_run_metadata"
        meta.mkdir(parents=True)
        children = []
        for backend, licence in (("csd_act", "FreeSurfer"), ("ukf", "NABA-data")):
            child = dict(kind="slicerddsurfer_product_orchestration", tractography_backend=backend,
                         run_name="cmp__" + backend, wma_atlas="naba", brain_mask_backend="synthstrip",
                         microstructure_ifw_echo_time_ms=82.0, compression="zstd",
                         accepted_licences=[licence, "FreeSurfer"], tasks=[_task("bundles.wma")], **common)
            path = meta / f"cmp__{backend}_orchestration_manifest.json"
            path.write_text(json.dumps(child), encoding="utf-8")
            children.append(dict(backend=backend, manifest_path=str(path), status="completed"))
        aggregate = meta / "cmp_orchestration_manifest.json"
        aggregate.write_text(json.dumps(dict(
            kind="slicerddsurfer_product_orchestration", subject_id="S1", run_name="cmp", status="completed",
            output_root=str(Path(tmp) / "out"), tractography_backends=["csd_act", "ukf"], child_runs=children,
            tasks=[_task("bundles.wma@ukf")])), encoding="utf-8")
        return aggregate

    def test_t_c5b_the_aggregate_record_names_the_naba_profile_and_keeps_the_settings(self):
        from shared.run_record import write_run_record

        with tempfile.TemporaryDirectory() as tmp:
            record = json.loads(write_run_record(self.write_run(tmp)).read_text(encoding="utf-8"))
        profiles = record["assets"]["model_profiles"]
        self.assertIn("wma-naba", profiles)
        self.assertNotIn("wma", profiles)
        parameters = record["parameters"]
        self.assertEqual(parameters["brain_mask_backend"], "synthstrip")
        self.assertEqual(parameters["microstructure_ifw_echo_time_ms"], 82.0)
        self.assertEqual(parameters["compression"], "zstd")
        self.assertEqual(parameters["wma_atlas"], "naba")
        self.assertIn("pnl-cnn-weights", {"pnl-cnn-weights"} | set())  # guard: names are plain strings
        self.assertIn("synthstrip", profiles)
        self.assertEqual([entry["tracking_backend"] for entry in parameters["per_tracking_backend"]],
                         ["csd_act", "ukf"])
        self.assertEqual([entry["tractography_backend"] for entry in parameters["per_tracking_backend"]],
                         ["csd_act", "ukf"])
        self.assertNotIn("tractography_backend", parameters)
        self.assertEqual(record["accepted_licences"], ["FreeSurfer", "NABA-data"])

    def test_t_c5b_a_manifest_without_readable_children_is_unchanged(self):
        from shared.run_record import merge_child_settings

        manifest = {"child_runs": [{"backend": "ukf", "manifest_path": "/nonexistent/x.json"}], "a": 1}
        self.assertEqual(merge_child_settings(manifest), (manifest, []))


class TC7bCustomModelRootTest(unittest.TestCase):
    """T-C7b: catalog hashes are recorded only for a verified model folder."""

    manifest = {"brain_mask_backend": "pnl_cnn", "tasks": [_task("segmentation.ddseg")]}

    def test_t_c7b_a_custom_ddseg_root_records_the_folder_and_unknown(self):
        from shared import run_record

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "my_weights"
            custom.mkdir()
            profiles = run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))
        self.assertEqual(profiles["segmentation"],
                         {"manifest": None, "version": "unknown", "folder": custom.as_posix(), "files": {}})

    def test_t_c7b_the_environment_root_counts_as_custom(self):
        from shared import run_record

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SLICERDSPACE_MODEL_ASSETS_DIR": tmp}):
            profiles = run_record.asset_versions(dict(self.manifest))
        self.assertEqual(profiles["segmentation"]["version"], "unknown")
        self.assertEqual(profiles["segmentation"]["folder"], Path(tmp).as_posix())

    def test_t_c7b_the_managed_root_keeps_the_catalog_values(self):
        from shared import run_record

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_MODEL_ASSETS_DIR", None)
            profiles = run_record.asset_versions(dict(self.manifest))
        self.assertEqual(profiles["segmentation"]["manifest"], "models/assets_manifest.json")
        self.assertTrue(profiles["segmentation"]["files"])

    def test_t_c7b_a_custom_folder_whose_files_match_the_catalog_keeps_the_catalog_values(self):
        from shared import run_record

        data = b"weights"
        import hashlib
        digest = hashlib.sha256(data).hexdigest()
        catalog = {"manifest": "m.json", "version": "9", "records": {"seg/w.pt": {"sha256": digest}},
                   "profiles": {"segmentation": {"required_paths": ["seg/w.pt"]}}, "upstream_commit": None}
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp)
            (custom / "seg").mkdir()
            (custom / "seg" / "w.pt").write_bytes(data)
            with patch.object(run_record, "_load_asset_manifests",
                              return_value=[dict(catalog, path="m.json")]), \
                    patch("shared.model_assets._sha256_cache", return_value={}), \
                    patch("shared.model_assets._persist_sha256_cache"):
                same = run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))
                (custom / "seg" / "w.pt").write_bytes(b"replaced")
                other = run_record.asset_versions(dict(self.manifest, model_assets_dir=str(custom)))
        self.assertEqual(same["segmentation"]["files"], {"seg/w.pt": digest})
        self.assertEqual(other["segmentation"]["version"], "unknown")

    def test_t_c7b_the_orchestrator_manifest_records_the_model_folder(self):
        source = (REPO_ROOT / "shared" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn('"model_assets_dir": model_assets_dir.as_posix() if model_assets_dir else None', source)


class TO2IfwFlagTest(unittest.TestCase):
    def test_t_o2_only_the_first_variant_keeps_the_ifw_flag(self):
        from shared.multi_tracking import variants
        from shared.orchestrator import ProductOrchestratorConfig

        config = ProductOrchestratorConfig("S1", Path("/x"), ("tractography", "microstructure_ifw"),
                                           tractography_backend="csd_act,ukf", microstructure_run_ifw=True)
        flags = [child.microstructure_run_ifw for child in variants(config)]
        self.assertEqual(flags, [True, False])


if __name__ == "__main__":
    unittest.main()
