"""Unit tests for non-destructive L/R inflated surface lateral separation (±35 mm display transform)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import vtk
except Exception:
    vtk = None

try:
    import nibabel as nib
except Exception:
    nib = None


VERTICES = [
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    [10.0, 10.0, 0.0],
    [10.0, 0.0, 10.0],
]
FACES = [[0, 1, 2], [0, 1, 3], [1, 4, 2], [1, 5, 3]]


class _FakeNode:
    def __init__(self, scene, identity: str, kind: str, name: str = ""):
        self.scene = scene
        self.identity = identity
        self.kind = kind
        self.name = name
        self.attributes: dict[str, str] = {}
        self.display_ids: list[str] = []
        self.storage_ids: list[str] = []
        self.polydata = None
        self.transform_id: str | None = None
        self.matrix = None
        self.hidden_from_editors = False
        self.visible = True

    def __getattr__(self, name):
        if name.startswith(("Set", "Get", "Create")):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def GetID(self):
        return self.identity

    def GetName(self):
        return self.name

    def SetName(self, name):
        self.name = name

    def GetClassName(self):
        return self.kind

    def IsA(self, name):
        if name == self.kind:
            return True
        if name == "vtkMRMLDisplayableNode" and self.kind == "vtkMRMLModelNode":
            return True
        return False

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def SetAndObservePolyData(self, data):
        self.polydata = data

    def GetPolyData(self):
        return self.polydata

    def CreateDefaultDisplayNodes(self):
        if not self.display_ids:
            self.display_ids.append(self.scene.AddNewNodeByClass("vtkMRMLModelDisplayNode").GetID())

    def GetNumberOfDisplayNodes(self):
        return len(self.display_ids)

    def GetNthDisplayNode(self, index):
        return self.scene.GetNodeByID(self.display_ids[index])

    def GetDisplayNode(self):
        return self.GetNthDisplayNode(0) if self.display_ids else None

    def GetNumberOfStorageNodes(self):
        return len(self.storage_ids)

    def GetNthStorageNode(self, index):
        return self.scene.GetNodeByID(self.storage_ids[index])

    def GetDisplayVisibility(self):
        return 1 if self.visible else 0

    def SetDisplayVisibility(self, value):
        self.visible = bool(value)

    def SetAndObserveTransformNodeID(self, node_id):
        self.transform_id = node_id

    def GetTransformNodeID(self):
        return self.transform_id

    def SetMatrixTransformToParent(self, matrix):
        self.matrix = matrix

    def GetMatrixTransformToParent(self):
        return self.matrix

    def SetHideFromEditors(self, flag):
        self.hidden_from_editors = bool(flag)

    def GetHideFromEditors(self):
        return self.hidden_from_editors


class _FakeScene:
    NodeRemovedEvent = 66001

    def __init__(self, builtin_ids=()):
        self.nodes: dict[str, _FakeNode] = {}
        self.observers: list[tuple[int, object]] = []
        self.closing = False
        self.count = 0
        self.builtin = {identity: _FakeNode(self, identity, "vtkMRMLColorTableNode") for identity in builtin_ids}

    def AddNewNodeByClass(self, kind: str, name: str = ""):
        self.count += 1
        node = _FakeNode(self, f"{kind}_{self.count}", kind, name)
        self.nodes[node.GetID()] = node
        return node

    def GetNodeByID(self, identity):
        return self.nodes.get(identity) or self.builtin.get(identity)

    def RemoveNode(self, node):
        if node is None or self.nodes.pop(node.GetID(), None) is None:
            return
        for event, callback in list(self.observers):
            if event == self.NodeRemovedEvent:
                callback(self, event, node)

    def AddObserver(self, event, callback):
        self.observers.append((event, callback))
        return len(self.observers)

    def IsClosing(self):
        return self.closing


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_obj(path: Path) -> None:
    lines = [f"v {x} {y} {z}" for x, y, z in VERTICES]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in FACES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@unittest.skipIf(vtk is None or nib is None, "VTK and nibabel required (run with PythonSlicer)")
class InflatedSeparationDisplayTest(unittest.TestCase):
    def setUp(self) -> None:
        from shared import surface_morphometry_display as smd

        self._prev_enabled = smd.inflated_separation_enabled()
        smd._INFLATED_SEPARATION_ENABLED = True
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene = _FakeScene(smd.SEQUENTIAL_TABLES[:1] + smd.DIVERGING_TABLES[:1])

        def load_model(path, _kind, properties=None, **kwargs):
            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", Path(path).name)
            node.SetAndObservePolyData(reader.GetOutput())
            node.CreateDefaultDisplayNodes()
            return True, node

        def add_legend(node):
            legend = self.scene.AddNewNodeByClass("vtkMRMLColorLegendDisplayNode")
            node.display_ids.append(legend.GetID())
            return legend

        self.slicer = SimpleNamespace(
            mrmlScene=self.scene,
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            vtkMRMLDisplayNode=SimpleNamespace(UseManualScalarRange=1, UseDataScalarRange=0),
            modules=SimpleNamespace(
                colors=SimpleNamespace(logic=lambda: SimpleNamespace(AddDefaultColorLegendDisplayNode=add_legend))
            ),
            util=SimpleNamespace(
                loadNodeFromFile=load_model,
                getNodesByClass=lambda kind: [n for n in self.scene.nodes.values() if n.IsA(kind)],
                showStatusMessage=lambda *_a, **_kw: None,
            ),
        )

        # Build a synthetic subject output with both hemispheres' white .obj, inflated, thickness, and aparc.annot.
        self.surf_dir = self.root / "outputs" / "01_anatomical_priors" / "cortical_surface"
        self.morph_dir = self.root / "outputs" / "01_anatomical_priors" / "surface_morphometry"
        self.surf_dir.mkdir(parents=True)
        self.morph_dir.mkdir(parents=True)
        for hemi in ("lh", "rh"):
            _write_obj(self.surf_dir / f"{hemi}.white.obj")
            nib.freesurfer.write_geometry(
                str(self.morph_dir / f"{hemi}.inflated"),
                np.array(VERTICES, dtype=np.float32),
                np.array(FACES, dtype=np.int32),
            )
            nib.freesurfer.write_morph_data(
                str(self.morph_dir / f"{hemi}.thickness"),
                np.linspace(1.5, 3.8, len(VERTICES)).astype(np.float32),
            )
            nib.freesurfer.write_annot(
                str(self.morph_dir / f"{hemi}.aparc.annot"),
                np.array([0, 1, -1, 0, 1, 1], dtype=np.int32),
                np.array([[220, 20, 10, 0, 0], [20, 220, 10, 0, 0]], dtype=np.int32),
                [b"one", b"two"],
                fill_ctab=True,
            )

        self.manifest = {
            "subject_root": str(self.root),
            "output_root": str(self.root / "outputs"),
            "tasks": [
                {
                    "name": "surface.ddsurfer",
                    "status": "completed",
                    "outputs": {
                        "lh_white": str(self.surf_dir / "lh.white.obj"),
                        "rh_white": str(self.surf_dir / "rh.white.obj"),
                    },
                },
                {
                    "name": "surface.ddsurfer_postprocess",
                    "status": "completed",
                    "outputs": {
                        "lh_inflated": str(self.morph_dir / "lh.inflated"),
                        "rh_inflated": str(self.morph_dir / "rh.inflated"),
                    },
                },
            ],
        }

    def tearDown(self) -> None:
        from shared import surface_morphometry_display as smd

        smd._INFLATED_SEPARATION_ENABLED = self._prev_enabled
        self.tmp.cleanup()

    def _load(self, filename: str, *, surfaces=None):
        from shared.surface_morphometry_display import PRODUCT, load_morphometry_row

        before = set(self.scene.nodes)
        candidate = {"path": str(self.morph_dir / filename), "product": PRODUCT, "key": filename, "label": filename}
        ok = load_morphometry_row(self.slicer, candidate, self.manifest, surfaces=surfaces)
        self.assertTrue(ok)
        created = [self.scene.nodes[nid] for nid in self.scene.nodes if nid not in before]
        model = next(n for n in created if n.kind == "vtkMRMLModelNode")
        return model, created

    def test_inflated_hemispheres_auto_separate_without_touching_disk_or_polydata(self) -> None:
        from shared import surface_morphometry_display as smd

        lh_path = self.morph_dir / "lh.inflated"
        rh_path = self.morph_dir / "rh.inflated"
        lh_sha_before, rh_sha_before = _file_sha256(lh_path), _file_sha256(rh_path)

        lh_model, _ = self._load("lh.inflated")
        rh_model, _ = self._load("rh.inflated")

        # Disk files are untouched.
        self.assertEqual(_file_sha256(lh_path), lh_sha_before)
        self.assertEqual(_file_sha256(rh_path), rh_sha_before)

        # PolyData vertices remain in exact unshifted scanner RAS coordinates.
        expected_pts = np.array(VERTICES, dtype=np.float32)
        for model in (lh_model, rh_model):
            pts = np.array([model.GetPolyData().GetPoint(i) for i in range(len(VERTICES))], dtype=np.float32)
            np.testing.assert_allclose(pts, expected_pts, atol=1e-5)

        # Owned linear transform nodes carry -35 mm (lh) and +35 mm (rh) on RAS X.
        lh_xform = self.scene.GetNodeByID(lh_model.GetTransformNodeID())
        rh_xform = self.scene.GetNodeByID(rh_model.GetTransformNodeID())
        self.assertIsNotNone(lh_xform)
        self.assertIsNotNone(rh_xform)
        self.assertEqual(lh_xform.kind, "vtkMRMLLinearTransformNode")
        self.assertTrue(lh_xform.GetHideFromEditors())
        self.assertAlmostEqual(lh_xform.GetMatrixTransformToParent().GetElement(0, 3), -smd.INFLATED_SEPARATION_OFFSET_MM)
        self.assertAlmostEqual(rh_xform.GetMatrixTransformToParent().GetElement(0, 3), +smd.INFLATED_SEPARATION_OFFSET_MM)
        self.assertAlmostEqual(lh_xform.GetMatrixTransformToParent().GetElement(1, 3), 0.0)
        self.assertAlmostEqual(lh_xform.GetMatrixTransformToParent().GetElement(2, 3), 0.0)

    def test_inflated_overlays_separate_while_white_surfaces_remain_unshifted(self) -> None:
        from shared import surface_morphometry_display as smd

        lh_thick_infl, _ = self._load("lh.thickness", surfaces=("inflated",))
        rh_annot_infl, _ = self._load("rh.aparc.annot", surfaces=("inflated",))
        lh_thick_white, _ = self._load("lh.thickness", surfaces=("white",))

        lh_xform = self.scene.GetNodeByID(lh_thick_infl.GetTransformNodeID())
        rh_xform = self.scene.GetNodeByID(rh_annot_infl.GetTransformNodeID())
        self.assertIsNotNone(lh_xform)
        self.assertIsNotNone(rh_xform)
        self.assertAlmostEqual(lh_xform.GetMatrixTransformToParent().GetElement(0, 3), -35.0)
        self.assertAlmostEqual(rh_xform.GetMatrixTransformToParent().GetElement(0, 3), +35.0)
        self.assertIsNone(lh_thick_white.GetTransformNodeID())
        self.assertIsNone(smd.inflated_hemisphere_of_node(lh_thick_white))

    def test_toggle_off_and_on_updates_existing_and_future_inflated_nodes_idempotently(self) -> None:
        from shared import surface_morphometry_display as smd

        lh_model, _ = self._load("lh.inflated")
        rh_overlay, _ = self._load("rh.thickness", surfaces=("inflated",))
        lh_xform_id = lh_model.GetTransformNodeID()
        rh_xform_id = rh_overlay.GetTransformNodeID()
        node_count_before = len(self.scene.nodes)

        # Toggle OFF -> both existing nodes drop to 0.0 mm without creating extra transform nodes.
        updated = smd.set_inflated_separation_enabled(self.slicer, False)
        self.assertEqual(updated, 2)
        self.assertFalse(smd.inflated_separation_enabled())
        self.assertEqual(len(self.scene.nodes), node_count_before)
        self.assertEqual(lh_model.GetTransformNodeID(), lh_xform_id)
        self.assertAlmostEqual(self.scene.GetNodeByID(lh_xform_id).GetMatrixTransformToParent().GetElement(0, 3), 0.0)
        self.assertAlmostEqual(self.scene.GetNodeByID(rh_xform_id).GetMatrixTransformToParent().GetElement(0, 3), 0.0)

        # Newly loaded rh.inflated while disabled starts at 0.0 mm.
        rh_model, _ = self._load("rh.inflated")
        rh_plain_xform_id = rh_model.GetTransformNodeID()
        self.assertAlmostEqual(
            self.scene.GetNodeByID(rh_plain_xform_id).GetMatrixTransformToParent().GetElement(0, 3), 0.0
        )

        # Toggle back ON -> all 3 inflated nodes move cleanly to -35 / +35 / +35 mm, reusing transforms.
        count_with_3 = len(self.scene.nodes)
        updated = smd.set_inflated_separation_enabled(self.slicer, True)
        self.assertEqual(updated, 3)
        self.assertTrue(smd.inflated_separation_enabled())
        self.assertEqual(len(self.scene.nodes), count_with_3)
        self.assertAlmostEqual(self.scene.GetNodeByID(lh_xform_id).GetMatrixTransformToParent().GetElement(0, 3), -35.0)
        self.assertAlmostEqual(self.scene.GetNodeByID(rh_xform_id).GetMatrixTransformToParent().GetElement(0, 3), +35.0)
        self.assertAlmostEqual(
            self.scene.GetNodeByID(rh_plain_xform_id).GetMatrixTransformToParent().GetElement(0, 3), +35.0
        )

    def test_removing_inflated_model_cleans_up_its_owned_transform_node(self) -> None:
        lh_model, lh_created = self._load("lh.inflated")
        rh_model, rh_created = self._load("rh.inflated")

        self.scene.RemoveNode(lh_model)
        self.assertEqual([n.GetID() for n in lh_created if n.GetID() in self.scene.nodes], [])
        self.assertEqual(set(self.scene.nodes), {n.GetID() for n in rh_created})

        self.scene.RemoveNode(rh_model)
        self.assertEqual(self.scene.nodes, {})

    def test_ui_contextual_bar_and_toggle_helper_sync_visibility_and_language(self) -> None:
        from shared import ui_run_history
        from shared.ui_onboarding import onboarding_text

        class _FakeCheckBox:
            def __init__(self, text=""):
                self.text = text
                self.tooltip = ""
                self.checked = False
                self._blocked = False

            def setText(self, text):
                self.text = text

            def setToolTip(self, tip):
                self.tooltip = tip

            def setChecked(self, val):
                self.checked = bool(val)

            def blockSignals(self, flag):
                prev = self._blocked
                self._blocked = bool(flag)
                return prev

        class _FakeBar:
            def __init__(self):
                self.visible = False

            def isVisible(self):
                return self.visible

            def setVisible(self, flag):
                self.visible = bool(flag)

        bar = _FakeBar()
        checkbox = _FakeCheckBox()
        owner = SimpleNamespace(
            _onboardingLanguage="zh",
            _resultsInflatedWidgets={"bar": bar, "checkbox": checkbox},
            _ui_text=lambda key, fallback="": onboarding_text("zh", key) or fallback,
        )

        # No inflated surfaces visible -> bar stays hidden.
        self.assertFalse(ui_run_history._refresh_inflated_bar(owner, slicer_module=self.slicer))
        self.assertFalse(bar.visible)

        # Load lh.inflated (visible) -> bar shows with checkbox checked.
        lh_model, _ = self._load("lh.inflated")
        self.assertTrue(ui_run_history._refresh_inflated_bar(owner, slicer_module=self.slicer))
        self.assertTrue(bar.visible)
        self.assertTrue(checkbox.checked)

        # One-click toggle OFF updates transform to 0 mm and unchecks box.
        self.assertFalse(ui_run_history.toggle_inflated_separation(owner, False, slicer_module=self.slicer))
        self.assertFalse(checkbox.checked)
        self.assertAlmostEqual(
            self.scene.GetNodeByID(lh_model.GetTransformNodeID()).GetMatrixTransformToParent().GetElement(0, 3), 0.0
        )

        # One-click toggle ON restores -35 mm and checks box.
        self.assertTrue(ui_run_history.toggle_inflated_separation(owner, True, slicer_module=self.slicer))
        self.assertTrue(checkbox.checked)
        self.assertAlmostEqual(
            self.scene.GetNodeByID(lh_model.GetTransformNodeID()).GetMatrixTransformToParent().GetElement(0, 3), -35.0
        )

        # Hiding lh.inflated hides the contextual bar again.
        lh_model.SetDisplayVisibility(False)
        self.assertFalse(ui_run_history._refresh_inflated_bar(owner, slicer_module=self.slicer))
        self.assertFalse(bar.visible)

        # Language strings exist in both EN and ZH.
        self.assertIn("±35 mm", onboarding_text("en", "results_inflated_separate"))
        self.assertIn("±35 mm", onboarding_text("zh", "results_inflated_separate"))
        self.assertIn("never modified", onboarding_text("en", "results_inflated_separate_tooltip"))
        self.assertIn("不会修改", onboarding_text("zh", "results_inflated_separate_tooltip"))


if __name__ == "__main__":
    unittest.main()
