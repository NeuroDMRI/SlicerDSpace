#!/usr/bin/env python3
"""F1: the DDSurfer surface run must report where the volume network really runs.

With a torch build that has no MPS Conv3d, ``TANet.to('mps')`` keeps ``vf_net`` on the CPU.
The warning, the model attribute, the prediction metadata and the surface manifest must say so.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "anatomy/surface/ddsurfer_backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:  # make test-fast may run without torch
    torch = None


def _small_tanet(ddsurfer):
    return ddsurfer.TANet(C_hid=(8, 16, 24, 32, 48), inshape=(16, 32, 16), depths=(1, 1, 1, 1, 1),
                          kernels=(3, 3, 3, 3, 3), step_size=0.5)


@unittest.skipIf(torch is None, "torch is not installed")
class TANetPlacementReportTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(BACKEND_ROOT))
        self.addCleanup(self._unload)
        from model import ddsurfer
        self.ddsurfer = ddsurfer
        ddsurfer._VF_NET_CPU_WARNED = False  # the warning is once per process

    @staticmethod
    def _unload() -> None:
        sys.path.remove(str(BACKEND_ROOT))
        for name in [n for n in sys.modules if n == "model" or n.startswith("model.")]:
            del sys.modules[name]

    def test_f1_cpu_model_records_cpu_placement_without_warning(self) -> None:
        model = _small_tanet(self.ddsurfer).to("cpu")
        self.assertEqual(model.execution_devices, {"vf_net": "cpu", "att_net": "cpu"})

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "MPS is not available")
    def test_f1_mps_without_conv3d_warns_once_and_records_cpu_vf_net(self) -> None:
        model = _small_tanet(self.ddsurfer)
        with patch.object(self.ddsurfer, "_mps_supports_conv3d", return_value=False):
            with self.assertLogs(self.ddsurfer.LOG, level="WARNING") as logs:
                model.to("mps")
                model.to("mps")  # a second move in the same process must stay silent
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn("volume network", message)
        self.assertIn("CPU", message)
        self.assertIn("MPS Conv3d", message)
        self.assertEqual(model.execution_devices, {"vf_net": "cpu", "att_net": "mps"})
        self.assertEqual(next(model.vf_net.parameters()).device.type, "cpu")

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "MPS is not available")
    def test_f1_mps_with_conv3d_keeps_vf_net_on_mps_and_stays_silent(self) -> None:
        model = _small_tanet(self.ddsurfer)
        with patch.object(self.ddsurfer, "_mps_supports_conv3d", return_value=True):
            with self.assertNoLogs(self.ddsurfer.LOG, level="WARNING") if hasattr(self, "assertNoLogs") else _null():
                model.to("mps")
        self.assertEqual(model.execution_devices, {"vf_net": "mps", "att_net": "mps"})


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SurfaceManifestDeviceReportTests(unittest.TestCase):
    def test_f1_devices_are_merged_from_prediction_records(self) -> None:
        from anatomy.surface.pipeline import _merge_execution_devices
        same = {"vf_net": "cpu", "att_net": "mps"}
        self.assertEqual(_merge_execution_devices([{"execution_devices": same}, {"execution_devices": dict(same)}]), same)
        self.assertIsNone(_merge_execution_devices([{}, {"execution_devices": None}]))  # records from before F1
        self.assertEqual(_merge_execution_devices([{"execution_devices": {"vf_net": "cpu"}},
                                                   {"execution_devices": {"vf_net": "mps"}}]), {"vf_net": "cpu,mps"})

    def test_f1_log_note_only_when_the_volume_network_runs_elsewhere(self) -> None:
        from anatomy.surface.pipeline import _execution_device_note
        note = _execution_device_note("auto", "mps", {"vf_net": "cpu", "att_net": "mps"})
        self.assertIn("Requested auto; resolved mps", note)
        self.assertIn("volume network on cpu", note)
        self.assertIn("no MPS Conv3d in this torch build", note)
        self.assertEqual(_execution_device_note("auto", "mps", {"vf_net": "mps", "att_net": "mps"}), "")
        self.assertEqual(_execution_device_note("auto", "cuda:0", {"vf_net": "cuda", "att_net": "cuda"}), "")
        self.assertEqual(_execution_device_note("auto", "mps", None), "")

    def test_f1_run_registered_returns_devices_before_cleanup_deletes_the_predictions(self) -> None:
        from types import SimpleNamespace
        from anatomy.surface import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            mni = out / "pred_results_SlicerDSpace/mni/x"
            mni.mkdir(parents=True)
            devices = {"vf_net": "cpu", "att_net": "mps"}
            for hemi in ("left", "right"):
                for kind in ("wm", "pial"):
                    (mni / f"x_predicted_{kind}_surface_{hemi}.obj").write_text("mesh")
                    (mni / f"x_predicted_{kind}_surface_{hemi}.json").write_text(
                        json.dumps({"precision": "fp32", "execution_devices": devices}))
            prepared = SimpleNamespace(preprocessed_dir=root / "pre/x", raw_dir=root / "raw")
            prepared.preprocessed_dir.mkdir(parents=True)

            def predict(stage):
                return SimpleNamespace(wm_mesh_path=mni / f"x_predicted_wm_surface_{stage.hemisphere}.obj",
                                       pial_mesh_path=mni / f"x_predicted_pial_surface_{stage.hemisphere}.obj")

            config = pipeline.SurfaceConfig("x", root, out, device="mps", cleanup_intermediates=True, execute=True)
            resources = dict(registration_atlas=str(root / "atlas.nii.gz"), template_dir=str(root), checkpoint_root=str(root))
            with patch("anatomy.surface.preprocessing.preprocess_subject", return_value=prepared), \
                    patch("anatomy.surface.prediction.predict_subject", side_effect=lambda stage: predict(stage)), \
                    patch.object(pipeline, "run_backend"):
                result = pipeline._run_registered(config, resources, out, root / "subject")
            self.assertEqual(result, devices)
            self.assertFalse(mni.exists())  # cleanup ran, so the manifest value had to be read first

    def test_f1_run_surface_writes_execution_devices_and_logs_the_cpu_split(self) -> None:
        from anatomy.surface import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subj"
            subject.mkdir()
            for name in ("FractionalAnisotropy", "MinEigenvalue", "MidEigenvalue", "MaxEigenvalue", "MeanDiffusivity"):
                (subject / f"subj-dti-{name}-Reg.nii.gz").write_bytes(b"x")
            (subject / "subj-mask-Reg.nii.gz").write_bytes(b"x")
            (subject / "subj-b0ToAtlasT2.tfm").write_bytes(b"x")
            atlas, weights, templates = root / "atlas.nii.gz", root / "weights", root / "templates"
            weights.mkdir()
            templates.mkdir()
            atlas.write_bytes(b"x")
            files = {h: {"white": f"{h}.white.pt", "pial": f"{h}.pial.pt"} for h in ("left", "right")}
            for h in files:
                (weights / f"{h}.white.pt").write_bytes(b"x")
                (weights / f"{h}.pial.pt").write_bytes(b"x")
                (templates / f"{h}.obj").write_bytes(b"x")
            (weights / "manifest.json").write_text("{}")
            resources = dict(available=True, missing=[], upstream_commit="test", registration_atlas=str(atlas),
                             checkpoint_root=str(weights), template_dir=str(templates),
                             model_config=dict(checkpoint_files=files, templates={h: {"file": f"{h}.obj"} for h in files}))
            devices = {"vf_net": "cpu", "att_net": "mps"}
            lines = []
            config = pipeline.SurfaceConfig("subj", root, root / "out", device="mps", execute=True)
            with patch.object(pipeline, "inspect_surface_resources", return_value=resources), \
                    patch.object(pipeline, "runtime_info", return_value={}), \
                    patch.object(pipeline, "_run_registered", return_value=devices), \
                    patch.object(pipeline, "native_products", return_value=([], {})), \
                    patch("shared.environment.resolve_auto_torch_device", return_value="mps"):
                result = pipeline.run_surface(config, log=lines.append)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["resolved_device"], "mps")  # unchanged for compatibility
            self.assertEqual(manifest["execution_devices"], devices)
            self.assertTrue(any("resolved mps (volume network on cpu: no MPS Conv3d in this torch build)" in line
                                for line in lines), lines)


class PredictMetadataSourceTests(unittest.TestCase):
    def test_f1_prediction_metadata_and_pipeline_summary_carry_execution_devices(self) -> None:
        predict = (BACKEND_ROOT / "inference/predict.py").read_text(encoding="utf-8")
        run = (BACKEND_ROOT / "run_ddsurfer_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("execution_devices", predict.split("metadata.update(", 1)[1].split("atomic_json", 1)[0])
        self.assertIn("execution_devices", run.split("surfaces[name] = dict(", 1)[1].split("summary = dict(", 1)[0])


if __name__ == "__main__":
    unittest.main()
