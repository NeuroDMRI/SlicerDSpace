#!/usr/bin/env python3
"""Regression tests for manifest detection, scroll lock, quiet preflight, mask opacity,
DTI auto-dependency transparency, and DDParcel-guided ROI tractography.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _widget_method(name: str, **namespace):
    path = REPO_ROOT / "shared" / "module_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Bug1ManifestFallbackTests(unittest.TestCase):
    def test_extract_manifest_uses_candidate_recent_lines_and_expected_fallback(self) -> None:
        from shared.ui_run_controller import extract_background_manifest_path, parse_manifest_line

        extract = _widget_method(
            "_extract_background_manifest_path",
            Path=Path,
            parse_manifest_line=parse_manifest_line,
            extract_background_manifest_path=extract_background_manifest_path,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "Test1_orchestration_manifest.json"
            manifest.write_text('{"status":"completed"}\n', encoding="utf-8")

            # 1. Candidate recorded while streaming (even after buffer was cleared)
            owner = SimpleNamespace(
                _backgroundManifestPath=None,
                _backgroundManifestCandidate=manifest,
                _backgroundOutputBuffer="",
                _workerRecentLines=[],
                _backgroundExpectedManifestPath=None,
                _backgroundStartedWall=0.0,
            )
            self.assertEqual(extract(owner), manifest)

            # 2. Recent lines buffer fallback
            owner2 = SimpleNamespace(
                _backgroundManifestPath=None,
                _backgroundManifestCandidate=None,
                _backgroundOutputBuffer="",
                _workerRecentLines=[str(manifest)],
                _backgroundExpectedManifestPath=None,
                _backgroundStartedWall=0.0,
            )
            self.assertEqual(extract(owner2), manifest)

            # 3. Expected manifest path fallback when freshly written on exit
            owner3 = SimpleNamespace(
                _backgroundManifestPath=None,
                _backgroundManifestCandidate=None,
                _backgroundOutputBuffer="",
                _workerRecentLines=[],
                _backgroundExpectedManifestPath=manifest,
                _backgroundStartedWall=time.time() - 5.0,
                _manifest_is_final=lambda _p: True,
            )
            self.assertIsNone(extract(owner3, allow_expected_fallback=False))
            self.assertEqual(extract(owner3, allow_expected_fallback=True), manifest)


class Bug3QuietSetupJobTests(unittest.TestCase):
    def test_setup_job_show_dialog_false_creates_no_progress_dialog(self) -> None:
        from shared.ui_setup_jobs import SetupJob

        fake_qt = SimpleNamespace(
            QProgressDialog=mock.Mock(side_effect=AssertionError("QProgressDialog must not be created")),
            QProcess=mock.MagicMock(),
            QProcessEnvironment=SimpleNamespace(systemEnvironment=lambda: mock.MagicMock()),
            QTimer=mock.MagicMock(),
            Qt=SimpleNamespace(WindowModal=1),
        )
        fake_slicer = SimpleNamespace(util=SimpleNamespace(mainWindow=lambda: None))
        with mock.patch.dict(sys.modules, {"qt": fake_qt, "slicer": fake_slicer}):
            owner = SimpleNamespace(parent=None, appendLog=lambda *_a: None)
            job = SetupJob(
                owner,
                "Checking environment",
                [("Check", ["python", "-V"])],
                lambda *_a: None,
                show_dialog=False,
            )
            try:
                self.assertIsNone(job.dialog)
                fake_qt.QProgressDialog.assert_not_called()
            finally:
                if getattr(job, "log_file", None) is not None:
                    job.log_file.close()


class Bug4BrainMaskOpacityTests(unittest.TestCase):
    def test_brain_mask_opacity_and_soft_cyan_color_node(self) -> None:
        from shared.result_scene import BRAIN_MASK_LABEL_OPACITY, brain_mask_color_node

        self.assertLessEqual(BRAIN_MASK_LABEL_OPACITY, 0.25)
        self.assertGreater(BRAIN_MASK_LABEL_OPACITY, 0.1)

        colors: dict[int, tuple] = {}

        class _FakeLUT:
            def SetTableRange(self, _a, _b) -> None:
                pass

        class _FakeColorNode:
            def __init__(self) -> None:
                self.attrs: dict[str, str] = {}
                self._lut = _FakeLUT()

            def SetTypeToUser(self) -> None:
                pass

            def SetNumberOfColors(self, _n: int) -> None:
                pass

            def GetLookupTable(self):
                return self._lut

            def SetColor(self, idx: int, name: str, r: float, g: float, b: float, a: float) -> None:
                colors[idx] = (name, r, g, b, a)

            def SetAttribute(self, k: str, v: str) -> None:
                self.attrs[k] = v

            def GetAttribute(self, k: str):
                return self.attrs.get(k)

        fake_slicer = SimpleNamespace(
            util=SimpleNamespace(getNodesByClass=lambda _cls: []),
            mrmlScene=SimpleNamespace(AddNewNodeByClass=lambda _cls, _name: _FakeColorNode()),
        )
        node = brain_mask_color_node(fake_slicer)
        self.assertIsNotNone(node)
        self.assertEqual(colors[0][4], 0.0)
        self.assertEqual(colors[1][0], "Brain mask")
        self.assertGreater(colors[1][3], 0.8)  # cyan-ish blue channel


class Feature5DtiDependencyAndLiveStageTests(unittest.TestCase):
    def test_anatomical_priors_auto_require_dti_with_bilingual_notice(self) -> None:
        from shared.target_closure import closure_notice, expand_closure

        for target, expected_name in (
            ("segmentation", "DDSeg"),
            ("parcellation", "DDParcel"),
            ("surface", "DDSurfer"),
        ):
            closure = expand_closure((target,))
            self.assertEqual(closure.targets, ("microstructure", target))
            self.assertEqual(closure.required_by, {"microstructure": (target,)})
            self.assertIn("DTI", closure_notice(closure, language="en"))
            self.assertIn(expected_name, closure_notice(closure, language="en"))
            self.assertIn("同时自动运行", closure_notice(closure, language="zh"))
            self.assertIn("DTI", closure_notice(closure, language="zh"))

    def test_live_stage_reports_dti_fitting_and_atlas_registration(self) -> None:
        from shared.ui_run_state import parse_live_stage_from_log_line

        en_fit = parse_live_stage_from_log_line("[Shared] Building native DTI bundle (wls)", language="en")
        zh_fit = parse_live_stage_from_log_line("[Orchestrator] Running Microstructure DTI", language="zh")
        zh_reg = parse_live_stage_from_log_line(
            "[Shared] Running: BRAINSFit --fixedVolume 100HCP-population-mean-T2-1mm.nii.gz",
            language="zh",
        )
        self.assertIsNotNone(en_fit)
        self.assertIn("DTI", en_fit[0])
        self.assertIsNotNone(zh_fit)
        self.assertIn("DTI", zh_fit[0])
        self.assertIsNotNone(zh_reg)
        self.assertIn("DTI", zh_reg[0])


class Feature6RoiTrackingTests(unittest.TestCase):
    def test_region_slug_and_ukf_seeds_command(self) -> None:
        import xml.etree.ElementTree as ET
        from shared.roi_tracking import region_slug
        from tractography.ukf import UKFParameters, build_command

        self.assertEqual(region_slug([1024], ["ctx-lh-precentral"]), "ctx-lh-precentral")
        self.assertEqual(region_slug([1024, 2024], ["ctx-lh-precentral", "ctx-rh-precentral"]),
                         "ctx-lh-precentral__ctx-rh-precentral")
        self.assertEqual(region_slug([1024], []), "roi_1024")

        selected = UKFParameters()
        params_xml = "".join(
            f"<param><name>{k}</name><default>{'false' if k == 'freeWater' else 'true'}</default></param>"
            for k in (
                "dwiFile", "maskFile", "tracts", "seedsFile", "seedLabels",
                *selected.__dataclass_fields__.keys(),
            )
        )
        schema = ET.fromstring(f"<executable><parameters>{params_xml}</parameters></executable>")
        cmd = build_command(
            Path("/bin/UKFTractography"),
            Path("/tmp/dwi.nhdr"),
            Path("/tmp/mask.nrrd"),
            Path("/tmp/out.vtk"),
            selected,
            schema,
            seeds_file=Path("/tmp/seeds.nrrd"),
            seed_labels="1",
        )
        self.assertIn("--seedsFile", cmd)
        self.assertIn("--seedLabels", cmd)
        self.assertNotIn("--seedingThreshold", cmd)

    def test_build_roi_seed_mask_dilates_cortical_and_clips_to_brain_mask(self) -> None:
        import nibabel as nib
        import numpy as np
        from shared.roi_tracking import build_roi_seed_mask

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parc = np.zeros((7, 7, 7), dtype=np.int32)
            parc[3, 3, 3] = 1024  # cortical label (>= 1000)
            mask = np.zeros((7, 7, 7), dtype=np.uint8)
            mask[2:5, 2:5, 3] = 1  # single slice brain mask
            parc_path = root / "parc.nii.gz"
            mask_path = root / "mask.nii.gz"
            out_path = root / "roi_mask.nii.gz"
            nib.save(nib.Nifti1Image(parc, np.eye(4)), str(parc_path))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

            info = build_roi_seed_mask(parc_path, [1024], mask_path, out_path, dilate_cortical_voxels=1)
            self.assertEqual(info["raw_voxels"], 1)
            self.assertTrue(info["dilated_cortical"])
            # 6-connected dilation from (3,3,3) clipped to z=3 plane yields center + 4 in-plane neighbors = 5 voxels
            self.assertEqual(info["seed_voxels"], 5)
            out_arr = np.asarray(nib.load(str(out_path)).get_fdata())
            self.assertEqual(int((out_arr > 0).sum()), 5)

    def test_roi_manifest_discovery_result_candidates_and_bundle_isolation(self) -> None:
        from shared.orchestrator import _resolve_bundle_input_tractography
        from shared.result_scene import result_candidates
        from shared.roi_tracking import ROI_TRACKING_MANIFEST_KIND, discover_roi_tracking_results

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_root = root / "outputs"
            roi_dir = out_root / "03_tractography" / "roi_tracking" / "ukf" / "ctx-lh-precentral"
            roi_dir.mkdir(parents=True)
            vtk_file = roi_dir / "sub1_ukf_roi_ctx-lh-precentral.vtk"
            vtk_file.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
            manifest_file = roi_dir / "roi_tracking_manifest.json"
            payload = {
                "kind": ROI_TRACKING_MANIFEST_KIND,
                "version": 1,
                "status": "completed",
                "subject_id": "sub1",
                "algorithm": "ukf",
                "label_ids": [1024],
                "region_names": ["ctx-lh-precentral"],
                "region_slug": "ctx-lh-precentral",
                "streamline_count": 420,
                "created_at": "2026-11-27T12:00:00Z",
                "outputs": {"tractogram_vtk": vtk_file.as_posix()},
            }
            manifest_file.write_text(json.dumps(payload), encoding="utf-8")

            discovered = discover_roi_tracking_results(out_root)
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]["vtk_path"], vtk_file.resolve())

            cands, skipped = result_candidates(payload)
            self.assertEqual(skipped, [])
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0]["product"], "tractography.roi_tracking")
            self.assertEqual(cands[0]["path"], str(vtk_file.resolve()))

            # Side-task ROI tracking must NEVER be picked up by downstream bundle resolvers!
            bundle_in = _resolve_bundle_input_tractography(root, None, output_root=out_root, run_name="Test1")
            self.assertIsNone(bundle_in)

    def test_tractography_picker_roi_button_gating(self) -> None:
        from shared.ui_algorithm_picker import _roi_tracking_button

        class _Btn:
            def __init__(self) -> None:
                self.object_name = ""
                self.text = ""
                self.enabled = True
                self.tooltip = ""
                self.clicked = mock.MagicMock()

            def setObjectName(self, n): self.object_name = n
            def setText(self, t): self.text = t
            def setFixedSize(self, _w, _h): pass
            def setFocusPolicy(self, _p): pass
            def setEnabled(self, e): self.enabled = bool(e)
            def setCursor(self, _c): pass
            def setToolTip(self, t): self.tooltip = t
            def setAccessibleName(self, _a): pass
            def setStyleSheet(self, _s): pass

        fake_qt = SimpleNamespace(
            QToolButton=_Btn,
            Qt=SimpleNamespace(NoFocus=0, PointingHandCursor=13),
        )
        owner = SimpleNamespace(_uiWidget=None)

        btn_disabled = _roi_tracking_button(
            fake_qt, owner, "ukf",
            roi_spec={"enabled": False, "is_batch": False, "on_click": lambda _a: None},
            disabled=False, zh=True, dark=False,
        )
        self.assertEqual(btn_disabled.object_name, "algorithmRoi_ukf")
        self.assertFalse(btn_disabled.enabled)
        self.assertIn("DDParcel", btn_disabled.tooltip)

        btn_enabled = _roi_tracking_button(
            fake_qt, owner, "ukf",
            roi_spec={"enabled": True, "is_batch": False, "on_click": lambda _a: None},
            disabled=False, zh=False, dark=True,
        )
        self.assertTrue(btn_enabled.enabled)
        self.assertIn("DDParcel", btn_enabled.tooltip)


if __name__ == "__main__":
    unittest.main()
