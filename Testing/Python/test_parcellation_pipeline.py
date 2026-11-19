#!/usr/bin/env python3
"""Focused tests for DDParcel pipeline glue."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DDParcelPipelineTest(unittest.TestCase):
    def test_run_ddsurfer_pred_emits_boundary_logs(self) -> None:
        captured_calls: list[tuple[list[str], str, str, bool]] = []
        log_messages: list[str] = []

        fake_module = types.ModuleType("anatomy.parcellation.DDSurfer_Pred")
        fake_nrrd = types.ModuleType("nrrd")

        def fake_options_parse():
            return types.SimpleNamespace(
                iname=["fake-input.nii.gz"],
                oname="fake-output.mgz",
                weights="fake-weights",
                subject_id="100307",
            )

        def fake_fastsurfercnn(iname, oname, logger, options):
            captured_calls.append((list(iname), oname, options.gpu, options.no_cuda))
            logger.info("mock inner progress")

        fake_module.options_parse = fake_options_parse
        fake_module.fastsurfercnn = fake_fastsurfercnn

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules,
            {
                "nrrd": fake_nrrd,
                "anatomy.parcellation.DDSurfer_Pred": fake_module,
            },
        ):
            from anatomy.parcellation import pipeline

            output_dir = Path(tmp) / "ddparcel"
            weights_dir = Path(tmp) / "weights"
            output_dir.mkdir()
            weights_dir.mkdir()

            pipeline._run_ddsurfer_pred(
                output_dir=output_dir,
                subject_id="100307",
                weights_dir=weights_dir,
                log=log_messages.append,
                device="cuda:1",
            )

        self.assertEqual(captured_calls, [(["fake-input.nii.gz"], "fake-output.mgz", "1", False)])
        self.assertEqual(sum("Launching DDSurfer prediction for 100307" in message for message in log_messages), 1)
        self.assertEqual(sum("mock inner progress" in message for message in log_messages), 1)
        self.assertEqual(sum("DDSurfer prediction finished for 100307" in message for message in log_messages), 1)

    def test_cuda_device_mapping_does_not_mutate_visible_devices(self) -> None:
        from anatomy.parcellation.device import logical_cuda_index

        self.assertEqual(logical_cuda_index("1", ""), 1)
        self.assertEqual(logical_cuda_index("1", "1"), 0)
        self.assertEqual(logical_cuda_index("3", "1,3"), 1)

    def test_inference_moves_batches_to_selected_device(self) -> None:
        source = (REPO_ROOT / "anatomy" / "parcellation" / "DDSurfer_Pred.py").read_text(encoding="utf-8")

        self.assertIn('device=params_model["device"]', source)
        self.assertIn("sub_batch.to(target_device", source)
        self.assertNotIn("images_batch = images_batch.cuda()", source)


if __name__ == "__main__":
    unittest.main()
