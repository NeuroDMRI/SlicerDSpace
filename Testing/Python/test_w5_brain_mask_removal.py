"""W5 (D2, SC-2): the brain mask has two backends, pnl_cnn and provided. The removed ones leave setup and still read in old records."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Testing" / "Python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Testing" / "Python"))

import user_data_isolation  # the SHA-256 cache and the user config of these tests are temporary

user_data_isolation.install()

REMOVED = ("synthstrip", "mrtrix", "dipy")


def _status(key, *, required=True, details=None):
    return SimpleNamespace(key=key, kind="setting", status="missing", state="missing", required=required, targets=("validation",),
                           label=key, details=details or {}, install_group="brain_mask", install_hint="configure")


class W5SetupHasNoSynthStripTest(unittest.TestCase):
    def test_w5_no_setup_plan_contains_synthstrip_assets_or_components(self):
        """W5: with any backend name, a missing SynthStrip model adds no optional profile, asset or component."""
        from shared.optional_assets import missing_optional_profiles
        from shared.selected_setup import build_setup_plan

        mask = _status("brain_mask_backend", details={"missing_required": ["FreeSurfer models/synthstrip.1.pt"]})
        result = SimpleNamespace(statuses=[mask])
        for backend in ("pnl_cnn", "provided", *REMOVED):
            for system in ("Linux", "Darwin", "Windows"):
                with self.subTest(backend=backend, system=system):
                    self.assertNotIn("synthstrip", missing_optional_profiles(result, backend))
                    plan = build_setup_plan(result, brain_mask_backend=backend, system=system, machine="x86_64")
                    text = json.dumps(plan, default=str).lower()
                    self.assertNotIn("synthstrip", text)
                    self.assertNotIn("synthstrip", plan["optional_asset_profiles"])

    def test_w5_catalogs_and_profile_lists_hold_no_synthstrip(self):
        from scripts import setup_optional_assets

        manifest = json.loads((REPO_ROOT / "models/external_assets_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("synthstrip", setup_optional_assets.PROFILES)
        self.assertNotIn("synthstrip", manifest["profiles"])
        self.assertFalse([path for path in manifest["asset_records"] if "synthstrip" in path.lower()])
        self.assertNotIn("synthstrip", json.dumps(manifest["profiles"]).lower())
        self.assertFalse((REPO_ROOT / "shared" / "synthstrip_mask_runner.py").exists())
        resources = json.loads((REPO_ROOT / "scripts/extension_runtime/runtime_resources.json").read_text(encoding="utf-8"))
        self.assertNotIn("synthstrip", json.dumps(resources).lower())

    def test_w5_a_removed_mask_backend_no_longer_selects_the_mrtrix_runtime(self):
        """W5: MRtrix stays for preprocessing, tracking and connectomes. It is no longer a mask backend."""
        from shared.managed_runtimes import selected_profiles

        node = SimpleNamespace(targetPreprocessing=False, brainMaskBackend="mrtrix", targetConnectome=False,
                               connectomeBackend="auto")
        self.assertNotIn("mrtrix", selected_profiles(node, False))


class W5OldRecordsStillReadTest(unittest.TestCase):
    def test_w5_run_record_from_an_old_manifest_with_synthstrip_still_builds(self):
        """W5 (SC-2 rule 3): an old manifest that names synthstrip builds a run record and adds no asset profile for it."""
        from test_t1_fix2 import TC5bAggregateRecordTest
        from shared.run_record import write_run_record

        builder = TC5bAggregateRecordTest("write_run")  # builds two child manifests with brain_mask_backend "synthstrip"
        with tempfile.TemporaryDirectory() as tmp:
            record = json.loads(write_run_record(builder.write_run(tmp)).read_text(encoding="utf-8"))
        self.assertEqual(record["parameters"]["brain_mask_backend"], "synthstrip")
        self.assertEqual(record["brain_mask"]["requested_backend"], "synthstrip")
        self.assertNotIn("synthstrip", record["assets"]["model_profiles"])

    def test_w5_old_mask_receipt_naming_a_removed_backend_reads_as_its_route(self):
        """W5 (SC-2 rule 3): mask_provenance reads the receipt of a mask that a removed backend made."""
        import hashlib
        from shared.brain_mask import mask_provenance

        for backend in REMOVED:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                mask = Path(tmp) / "nodif_brain_mask.nii.gz"
                mask.write_bytes(b"old mask bytes")
                receipt = Path(tmp) / "nodif_brain_mask.manifest.json"
                receipt.write_text(json.dumps({"version": 2, "backend": backend,
                                               "output": {"sha256": hashlib.sha256(b"old mask bytes").hexdigest()}}))
                self.assertEqual(mask_provenance(mask), {"route": backend})


if __name__ == "__main__":
    unittest.main()
