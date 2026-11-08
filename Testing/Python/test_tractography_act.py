#!/usr/bin/env python3
"""Lightweight tests for ACT tractography planning."""

from __future__ import annotations

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


from tractography import ACTTrackingConfig, run_act_tracking


class ACTTractographyScaffoldTest(unittest.TestCase):
    def test_writes_manifest_with_ddparcel_dependency_and_missing_5tt_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parcellation = root / "wmparc.nii.gz"
            parcellation.write_text("parcel", encoding="utf-8")

            result = run_act_tracking(
                ACTTrackingConfig(
                    subject_id="149741",
                    parcellation_path=parcellation,
                    output_dir=root / "act",
                    bval_path=root / "bvals",
                    bvec_path=root / "bvecs",
                    mask_path=root / "mask.nii.gz",
                )
            )

            self.assertFalse(result.executed)
            self.assertFalse(result.standard_act_ready)
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "act_tracking")
            self.assertEqual(manifest["dependencies"]["ddparcel_parcellation"]["path"], parcellation.resolve().as_posix())
            self.assertFalse(manifest["act_readiness"]["standard_act_ready"])
            self.assertIn("DDParcel alone is not sufficient", manifest["act_readiness"]["reason"])
            self.assertFalse(qc["standard_act_ready"])

    def test_standard_act_ready_when_5tt_and_gmwmi_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parcellation = root / "wmparc.nii.gz"
            five_tt = root / "5tt.nii.gz"
            gmwmi = root / "gmwmi.nii.gz"
            for path in (parcellation, five_tt, gmwmi):
                path.write_text("placeholder", encoding="utf-8")

            result = run_act_tracking(
                ACTTrackingConfig(
                    subject_id="149741",
                    parcellation_path=parcellation,
                    output_dir=root / "act",
                    five_tt_path=five_tt,
                    gmwmi_path=gmwmi,
                )
            )

            self.assertTrue(result.standard_act_ready)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["act_readiness"]["standard_act_ready"])
            self.assertEqual(manifest["commands"], [])

    @unittest.skipIf(os.name == "nt", "Uses POSIX /bin/sh symlink fixtures")
    def test_execute_true_runs_fake_mrtrix_backend_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_backend = bin_dir / "fake_mrtrix.sh"
            fake_backend.write_text(
                "#!/bin/sh\n"
                "for arg in \"$@\"; do\n"
                "  case \"$arg\" in\n"
                "    *.tck|*.txt|*.mif) mkdir -p \"$(dirname \"$arg\")\"; touch \"$arg\" ;;\n"
                "  esac\n"
                "done\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_backend.chmod(0o755)
            for command in ("mrconvert", "dwi2response", "dwi2fod", "tckgen", "tcksift2", "tck2connectome"):
                link = bin_dir / command
                link.symlink_to(fake_backend)
            paths = {
                "parcellation": root / "wmparc.nii.gz",
                "dwi": root / "dwi.nii.gz",
                "mask": root / "mask.nii.gz",
                "five_tt": root / "5tt.nii.gz",
                "gmwmi": root / "gmwmi.nii.gz",
                "bval": root / "bvals",
                "bvec": root / "bvecs",
            }
            for path in paths.values():
                path.write_text("placeholder", encoding="utf-8")

            with mock.patch.dict(os.environ, {"PATH": bin_dir.as_posix() + os.pathsep + os.environ.get("PATH", "")}):
                result = run_act_tracking(
                    ACTTrackingConfig(
                        subject_id="149741",
                        parcellation_path=paths["parcellation"],
                        output_dir=root / "act",
                        dwi_path=paths["dwi"],
                        mask_path=paths["mask"],
                        five_tt_path=paths["five_tt"],
                        gmwmi_path=paths["gmwmi"],
                        bval_path=paths["bval"],
                        bvec_path=paths["bvec"],
                        execute=True,
                    )
                )

            self.assertTrue(result.executed)
            self.assertEqual(result.return_code, 0)
            self.assertTrue(result.streamlines_path.is_file())
            self.assertTrue(result.weights_path and result.weights_path.is_file())
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["status"], "completed")
            self.assertTrue(qc["streamlines_exists"])


if __name__ == "__main__":
    unittest.main()
