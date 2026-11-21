from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_previous_slicer_module = sys.modules.get("slicer")
sys.modules["slicer"] = types.ModuleType("slicer")
try:
    import shared.module_logic as shared_module_logic
    from shared.module_logic import SharedLogicMixin, _ensure_torch_via_extension, _release_version
finally:
    if _previous_slicer_module is None:
        sys.modules.pop("slicer", None)
    else:
        sys.modules["slicer"] = _previous_slicer_module


class SharedModuleLogicTest(unittest.TestCase):
    def test_surface_loader_explicitly_distinguishes_native_and_ras_companion(self):
        from unittest.mock import Mock

        for filename, expected in (("s_predicted_wm_lh.stl", 1),
                                   ("s_predicted_wm_lh_orig.stl", 0)):
            with self.subTest(filename=filename):
                node = Mock()
                loader = Mock(return_value=(True, node))
                fake = types.SimpleNamespace(
                    util=types.SimpleNamespace(loadNodeFromFile=loader),
                    vtkMRMLStorageNode=types.SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
                )
                with patch.object(shared_module_logic, "slicer", fake):
                    self.assertTrue(SharedLogicMixin()._load_model_node(Path(filename), "wm", "s"))
                self.assertEqual(loader.call_args.kwargs["properties"]["coordinateSystem"], expected)
                node.SetName.assert_called_once_with("s_wm_lh")

    def test_release_version_handles_torch_build_suffix(self) -> None:
        self.assertEqual(_release_version("2.2.2+cpu"), (2, 2, 2))
        self.assertLess(_release_version("1.8.1+cpu"), _release_version("2.0.0"))

    def test_old_installed_torch_is_rejected_before_inference(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "1.8.1+cpu"

        fake_utils = types.ModuleType("PyTorchUtils")

        class FakeTorchLogic:
            @staticmethod
            def torchInstalled() -> bool:
                return True

        fake_utils.PyTorchUtilsLogic = FakeTorchLogic
        with patch.dict(sys.modules, {"torch": fake_torch, "PyTorchUtils": fake_utils}):
            with self.assertRaisesRegex(RuntimeError, r"requires torch>=2\.0\.0"):
                _ensure_torch_via_extension()

    def test_missing_model_assets_do_not_block_logic_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_root = Path(tmp) / "model-assets"
            candidates = {
                "Resources/Data/template": asset_root / "Resources" / "Data" / "template",
                "Resources/Data/ckpts": asset_root / "Resources" / "Data" / "ckpts",
                "Resources/Data/100HCP-population-mean-T2.nii.gz": (
                    asset_root / "Resources" / "Data" / "100HCP-population-mean-T2.nii.gz"
                ),
            }

            def fake_candidates(relative: str, **_kwargs):
                return (candidates[relative],)

            class DummyLogic(SharedLogicMixin):
                pass

            with (
                patch("shared.module_logic.model_asset_candidates", side_effect=fake_candidates),
                patch.dict(
                    os.environ,
                    {
                        "DDSURFER_TEMPLATE_DIR": "",
                        "DDSURFER_CHECKPOINT_DIR": "",
                        "DDSURFER_REGISTRATION_ATLAS": "",
                    },
                ),
            ):
                logic = DummyLogic()
                logic._initialize_shared_logic_state()

            self.assertEqual(logic.template_dir, logic.module_dir / 'surface/template')
            self.assertEqual(logic.checkpoint_root, logic.module_dir / 'surface/weights')
            self.assertEqual(
                logic.registration_atlas,
                logic.module_dir / 'surface/template/100HCP-population-mean-T2-1mm.nii.gz',
            )

    def test_vtk_tractogram_falls_back_to_model_node_without_slicerdmri(self) -> None:
        class FakeNode:
            def __init__(self) -> None:
                self.name = ""
                self.attributes: dict[str, str] = {}

            def SetName(self, name: str) -> None:
                self.name = name

            def SetAttribute(self, key: str, value: str) -> None:
                self.attributes[key] = value

        class DummyLogic(SharedLogicMixin):
            def __init__(self) -> None:
                self.messages: list[str] = []

            def log(self, message: str) -> None:
                self.messages.append(message)

        with tempfile.TemporaryDirectory() as tmp:
            tractogram = Path(tmp) / "bundle.vtk"
            tractogram.write_text("# vtk DataFile Version 3.0\n", encoding="ascii")
            node = FakeNode()
            fake_slicer = types.SimpleNamespace(
                util=types.SimpleNamespace(loadModel=lambda *_args, **_kwargs: (True, node)),
            )
            logic = DummyLogic()

            with patch.object(shared_module_logic, "slicer", fake_slicer):
                loaded = logic._load_fiber_bundle_node(tractogram, "bundle")

            self.assertTrue(loaded)
            self.assertEqual(node.name, "bundle")
            self.assertEqual(node.attributes["SlicerDDSurfer.SourcePath"], tractogram.as_posix())
            # B2 fiber display: the model node is the one native display, not a fallback for SlicerDMRI.
            self.assertNotIn("SlicerDDSurfer.DisplayFallback", node.attributes)


if __name__ == "__main__":
    unittest.main()
