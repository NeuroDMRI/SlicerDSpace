"""Batch 2, display lane, part 2: the DDParcel result loads as a named segmentation.

Request item: "DDParcel result should load as a Segmentation with region names and colours."
Each test imports the code under test inside the test, so a missing name fails per test.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import nibabel as nib
    import numpy as np
except ImportError:  # pragma: no cover - plain python3 has no numpy/nibabel
    nib = np = None

from shared.result_scene import source_stamp

with patch.dict(sys.modules, {"slicer": types.ModuleType("slicer")}):
    module_logic = importlib.import_module("shared.module_logic")

REPO = Path(__file__).resolve().parents[2]
BUNDLED_LUT = REPO / "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt"


def _read_lut(path):
    """Independent reader of a FreeSurfer-format table: {id: (name, (r, g, b, a))}."""
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 6 and fields[0].isdigit():
            entries.setdefault(int(fields[0]), (fields[1], tuple(int(value) for value in fields[2:6])))
    return entries


def _assigned_literal(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name for target in node.targets):
            value = node.value
            if isinstance(value, ast.Call):  # np.array([...])
                value = value.args[0]
            return ast.literal_eval(value)
    raise AssertionError(f"{name} not found")


def _ddparcel_emittable_ids():
    """Every label id DDParcel can write, read from its own source (not a hand-copied list)."""
    root = REPO / "anatomy/parcellation"
    loader = ast.parse((root / "data_loader/load_neuroimaging_data.py").read_text(encoding="utf-8"))
    mapping = next(node for node in ast.walk(loader)
                   if isinstance(node, ast.FunctionDef) and node.name == "map_label2aparc_aseg")
    ids = set(_assigned_literal(mapping, "labels"))
    predictor = ast.parse((root / "DDSurfer_Pred.py").read_text(encoding="utf-8"))
    split = _assigned_literal(predictor, "labels_list")  # left cortex ids; the right copy is id + 1000
    ids |= {value + 1000 for value in split}
    loop = next(node for node in ast.walk(predictor)
                if isinstance(node, ast.For) and getattr(node.target, "id", None) == "prob_class_lh")
    ids |= {value + offset for value in ast.literal_eval(loop.iter) for offset in (0, 1000)}
    ids |= set(_assigned_literal(predictor, "labels"))  # the clean-up list (77 is only ever a median of others)
    return sorted(int(value) for value in ids)


def _write_labels(path, values, shape=(6, 6, 6)):
    data = np.zeros(shape, dtype=np.int16)
    flat = data.reshape(-1)
    for index, value in enumerate(values):
        flat[index * 3:index * 3 + 3] = value
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@unittest.skipIf(np is None, "needs numpy and nibabel (run with PythonSlicer)")
class LabelTableTest(unittest.TestCase):
    def test_b2_parcellation_every_ddparcel_id_has_a_name_and_colour(self) -> None:
        from anatomy.parcellation.label_table import label_entries

        ids = _ddparcel_emittable_ids()
        self.assertGreaterEqual(len([value for value in ids if value]), 101)
        self.assertNotIn(1004, ids)  # Desikan-Killiany without corpus callosum cortex
        lut = _read_lut(BUNDLED_LUT)
        entries = {entry.label: entry for entry in label_entries(ids)}
        self.assertEqual(sorted(entries), sorted(set(ids) | {0}))
        generated = sorted(label for label, entry in entries.items() if entry.generated)
        print(f"\n[B2] DDParcel ids: {len(ids)}; ids with no LUT entry (Label_<id>): {generated}")
        self.assertEqual(generated, [])
        for label in ids:
            with self.subTest(label=label):
                self.assertEqual(entries[label].name, lut[label][0])
                self.assertEqual(entries[label].rgb, lut[label][1][:3])
        self.assertEqual(entries[1001].name, "ctx-lh-bankssts")

    def test_b2_parcellation_unknown_id_gets_label_name_and_generated_colour(self) -> None:
        from anatomy.parcellation.label_table import label_entries

        first = {entry.label: entry for entry in label_entries([2, 99999, 99998])}
        again = {entry.label: entry for entry in label_entries([99999])}
        self.assertEqual(first[99999].name, "Label_99999")
        self.assertTrue(first[99999].generated)
        self.assertFalse(first[2].generated)
        self.assertEqual(first[99999].rgb, again[99999].rgb)  # deterministic
        self.assertNotEqual(first[99999].rgb, first[99998].rgb)
        self.assertNotEqual(first[99999].rgb, (0, 0, 0))


@unittest.skipIf(np is None, "needs numpy and nibabel (run with PythonSlicer)")
class ParcellationSidecarTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "s"
        from shared.io_layout import subject_layout

        self.layout = subject_layout(self.root)
        self.stage = self.root / "stage"
        self.stage.mkdir(parents=True)
        self.ids = [0, 2, 41, 1001, 2035]
        self.source = _write_labels(self.stage / "s-DDSurfer-wmparc.nii.gz", self.ids[1:])

    def _publish(self):
        from shared.orchestrator import _publish_parcellation_outputs

        return _publish_parcellation_outputs({"wmparc": self.source}, self.layout, "s")

    def test_b2_parcellation_sidecars_hold_exactly_the_present_ids(self) -> None:
        before = self.source.read_bytes()
        published = self._publish()
        self.assertEqual(set(published), {"wmparc", "color_table", "freesurfer_lut"})
        nifti, ctbl, lut = (Path(published[key]) for key in ("wmparc", "color_table", "freesurfer_lut"))
        self.assertEqual(ctbl, self.layout.brain_parcellation / "s_parcellation.ctbl")
        self.assertEqual(lut, self.layout.brain_parcellation / "s_parcellation_lut.txt")
        self.assertEqual(nifti.read_bytes(), before)  # the label NIfTI is unchanged
        present = sorted(int(value) for value in np.unique(np.asarray(nib.load(str(nifti)).dataobj)))
        reference = _read_lut(BUNDLED_LUT)
        slicer_table = _read_lut(ctbl)
        self.assertEqual(sorted(slicer_table), present)
        self.assertEqual(slicer_table[0], (reference[0][0], (0, 0, 0, 0)))
        for label in present[1:]:
            self.assertEqual(slicer_table[label], (reference[label][0], reference[label][1][:3] + (255,)))
        freesurfer_table = _read_lut(lut)
        self.assertEqual(sorted(freesurfer_table), present)
        for label in present:
            self.assertEqual(freesurfer_table[label], reference[label])  # FreeSurfer's own A column (0 = opaque)
        self.assertEqual(slicer_table[1001][0], "ctx-lh-bankssts")

    def test_b2_parcellation_sidecars_are_also_written_on_the_reuse_branch(self) -> None:
        first = self._publish()
        for key in ("color_table", "freesurfer_lut"):
            Path(first[key]).unlink()
        second = self._publish()  # the resume branch publishes the same staged outputs again
        self.assertEqual(second, first)
        self.assertTrue(Path(second["color_table"]).is_file())
        self.assertTrue(Path(second["freesurfer_lut"]).is_file())

    def test_b2_parcellation_current_sidecars_are_not_rewritten_on_resume(self) -> None:
        first = self._publish()
        stamps = {key: os.stat(first[key]).st_mtime_ns for key in ("color_table", "freesurfer_lut")}
        self.assertEqual(self._publish(), first)
        # An unchanged .ctbl keeps its stamp, so a later load reuses its colour node instead of adding one.
        self.assertEqual({key: os.stat(first[key]).st_mtime_ns for key in stamps}, stamps)

    def test_b2_parcellation_sidecar_writer_never_blocks_the_run(self) -> None:
        self.source.write_text("not a NIfTI", encoding="utf-8")  # fixtures write text into .nii.gz files
        published = self._publish()
        self.assertEqual(set(published), {"wmparc"})

    def test_b2_parcellation_completeness_inputs_are_unchanged(self) -> None:
        from shared.orchestrator import _parcellation_outputs

        outputs = _parcellation_outputs(self.root, "s")
        self.assertEqual(set(outputs), {"wmparc", "wmparc_reg", "fa_reg_norm", "trace_reg_norm",
                                        "min_eig_reg_norm", "mid_eig_reg_norm"})
        published = self._publish()
        stage_dir = outputs["wmparc"].parent
        for key in ("color_table", "freesurfer_lut"):
            self.assertNotEqual(Path(published[key]).parent, stage_dir)
            self.assertNotIn(Path(published[key]), set(outputs.values()))


class _Color:
    def __init__(self, identity, name=""):
        self.identity, self.name, self.attributes, self.entries = identity, name, {}, {}
        self.count = 0
        self.kind = "vtkMRMLColorTableNode"

    def GetID(self):
        return self.identity

    def GetName(self):
        return self.name

    def IsA(self, name):
        return name == self.kind

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def SetTypeToUser(self):
        pass

    def GetNumberOfColors(self):
        return self.count

    def SetNumberOfColors(self, count):
        self.count = count

    def GetColorName(self, label):
        return self.entries.get(label, ("(none)", None))[0]

    def SetColor(self, label, name, r, g, b, a=1.0):
        self.entries[label] = (name, (r, g, b, a))
        self.count = max(self.count, label + 1)

    def GetLookupTable(self):
        return self

    def SetTableRange(self, *_args):
        pass

    def GetTableValue(self, label):
        return self.entries.get(label, ("", (0.0, 0.0, 0.0, 0.0)))[1]


class _Node:
    def __init__(self, identity, kind, name=""):
        self.identity, self.kind, self.name, self.attributes = identity, kind, name, {}
        self.display = Mock()
        self.display.GetColorNodeID.side_effect = lambda: self.color_id
        self.display.SetAndObserveColorNodeID.side_effect = self._set_color
        self.color_id = None
        self.visible = True
        self.CreateClosedSurfaceRepresentation = Mock()  # must stay uncalled: no 3D surface build on load
        self.ras_bounds = (139.0, 219.0, -131.0, -71.0, 105.0, 153.0)  # an off-origin node (b2fix D2)

    def _set_color(self, identity):
        self.color_id = identity

    def GetID(self):
        return self.identity

    def GetClassName(self):
        return self.kind

    def GetName(self):
        return self.name

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def SetName(self, name):
        self.name = name

    def IsA(self, name):
        volume = {"vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLVolumeNode"}
        if self.kind == "vtkMRMLLabelMapVolumeNode":
            return name in volume | {"vtkMRMLDisplayableNode"}
        if self.kind == "vtkMRMLScalarVolumeNode":
            return name in volume - {"vtkMRMLLabelMapVolumeNode"} | {"vtkMRMLDisplayableNode"}
        return name in {self.kind, "vtkMRMLDisplayableNode"}

    def GetDisplayNode(self):
        return self.display

    def GetRASBounds(self, bounds):
        bounds[:] = self.ras_bounds

    def CreateDefaultDisplayNodes(self):
        pass

    def GetDisplayVisibility(self):
        return self.visible

    def SetDisplayVisibility(self, value):
        self.visible = bool(value)


class _Scene:
    def __init__(self):
        self.nodes, self.next_id = [], 0

    def AddNewNodeByClass(self, kind, name=""):
        self.next_id += 1
        node = (_Color if kind == "vtkMRMLColorTableNode" else _Node)("node" + str(self.next_id), *(
            (name,) if kind == "vtkMRMLColorTableNode" else (kind, name)))
        self.nodes.append(node)
        return node

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def GetNodeByID(self, identity):
        return next((node for node in self.nodes if node.GetID() == identity), None)


class _Logic(module_logic.SharedLogicMixin):
    def log(self, message):
        pass


class SegmentationLoaderTest(unittest.TestCase):
    """Fake Slicer: real file reads stay out; the loader's Slicer calls are recorded."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.scene = _Scene()
        self.imports = []
        self.import_result = True

        def load_label(path, properties=None, **_kwargs):
            return self.scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", (properties or {}).get("name", ""))

        def load_volume(path, properties=None, **_kwargs):
            return True, self.scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", (properties or {}).get("name", ""))

        def load_color(path):
            color = self.scene.AddNewNodeByClass("vtkMRMLColorTableNode", Path(path).name)
            color.loaded_from = str(path)
            for label, (name, rgba) in _read_lut(path).items():
                color.SetColor(label, name, *(value / 255 for value in rgba))
            return color

        def import_labelmap(labelmap, segmentation, *_args):
            self.imports.append({"labelmap": labelmap, "color": labelmap.GetDisplayNode().GetColorNodeID(),
                                 "segmentation": segmentation, "labelmap_in_scene": labelmap in self.scene.nodes})
            if isinstance(self.import_result, Exception):
                raise self.import_result
            return self.import_result

        def nodes_by_class(kind):
            if kind == "vtkMRMLColorTableNode":
                return [node for node in self.scene.nodes if isinstance(node, _Color)]
            if kind == "vtkMRMLTableNode":
                return [node for node in self.scene.nodes if getattr(node, "kind", "") == "vtkMRMLTableNode"]
            return [node for node in self.scene.nodes if not isinstance(node, _Color)
                    and node.kind != "vtkMRMLTableNode" and node.IsA(kind)]

        self.logic_api = SimpleNamespace(ImportLabelmapToSegmentationNode=Mock(side_effect=import_labelmap))
        self.fake = SimpleNamespace(
            mrmlScene=self.scene,
            modules=SimpleNamespace(segmentations=SimpleNamespace(logic=lambda: self.logic_api)),
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            util=SimpleNamespace(
                getNodesByClass=nodes_by_class, loadLabelVolume=load_label, loadVolume=load_volume,
                loadColorTable=load_color, setSliceViewerLayers=Mock(), resetSliceViews=Mock(),
                resetThreeDViews=Mock(), errorDisplay=Mock(),
            ),
        )
        self.addCleanup(patch.stopall)
        patch.object(module_logic, "slicer", self.fake).start()

        def register(slicer, node, run, subject, run_name, product, source):
            node.SetAttribute("SlicerDDSurfer.Run", run)
            node.SetAttribute("SlicerDDSurfer.Product", product)
            node.SetAttribute("SlicerDDSurfer.SourcePath", source)
            node.SetAttribute("SlicerDDSurfer.SourceStamp", source_stamp(source))

        patch("shared.result_scene.register_node", side_effect=register).start()
        self.logic = _Logic()

    def _manifest(self, *, ctbl_age=None, extra_tasks=(), with_ctbl=True):
        self.parcellation = self.root / "s_parcellation.nii.gz"
        self.parcellation.write_bytes(b"fixture labels")
        self.ctbl = self.root / "s_parcellation.ctbl"
        self.ctbl.write_text("# Color table file s_parcellation.ctbl\n# 1002 values\n0 Unknown 0 0 0 0\n"
                             "2 Left-Cerebral-White-Matter 245 245 245 255\n1001 ctx-lh-bankssts 25 100 40 255\n",
                             encoding="utf-8")
        if ctbl_age is not None:
            stamp = self.parcellation.stat().st_mtime + ctbl_age
            os.utime(self.ctbl, (stamp, stamp))
        self.ddseg = self.root / "s_ddseg.nii.gz"
        self.ddseg.write_bytes(b"fixture tissues")
        self.fa = self.root / "s_FA.nii.gz"
        self.fa.write_bytes(b"fixture fa")
        outputs = {"wmparc": str(self.parcellation)}
        if with_ctbl:
            outputs["color_table"] = str(self.ctbl)
        path = self.root / "00_run_metadata" / "run_orchestration_manifest.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"subject_id": "s", "run_name": "run", "output_root": str(self.root), "tasks": [
            {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(self.fa)}},
            {"name": "segmentation.ddseg", "status": "completed", "outputs": {"segmentation": str(self.ddseg)}},
            {"name": "parcellation.ddparcel", "status": "completed", "outputs": outputs},
            *extra_tasks]}))
        return path

    def _nodes_for(self, path):
        return [node for node in self.scene.nodes
                if isinstance(node, _Node) and node.GetAttribute("SlicerDDSurfer.SourcePath") == str(path)]

    def _color(self, identity):
        return self.scene.GetNodeByID(identity)

    def _assert_segmentation(self, *, color_from):
        nodes = self._nodes_for(self.parcellation)
        self.assertEqual([node.GetClassName() for node in nodes], ["vtkMRMLSegmentationNode"])
        self.assertFalse([node for node in self.scene.nodes
                          if isinstance(node, _Node) and node.kind == "vtkMRMLLabelMapVolumeNode"
                          and node.GetAttribute("SlicerDDSurfer.SourcePath") in {None, str(self.parcellation)}
                          and node.name.endswith("wmparc")], "the temporary labelmap stays in the scene")
        record = self.imports[-1]
        self.assertTrue(record["labelmap_in_scene"])
        self.assertIs(record["segmentation"], nodes[0])
        color = self._color(record["color"])
        self.assertIsNotNone(color, "no colour node was set on the labelmap before the import")
        self.assertEqual(color.GetColorName(1001), "ctx-lh-bankssts")
        self.assertEqual(getattr(color, "loaded_from", None), color_from)
        display = nodes[0].GetDisplayNode()
        display.SetOpacity2DFill.assert_called_with(0.35)
        display.SetVisibility2DOutline.assert_called_with(True)
        nodes[0].CreateClosedSurfaceRepresentation.assert_not_called()
        self.assertEqual(nodes[0].GetAttribute("SlicerDDSurfer.Run"), str(self.manifest_path))
        return nodes[0], color

    def test_b2_parcellation_every_load_path_gives_a_named_segmentation(self) -> None:
        from shared.ui_run_history import _toggle_output

        self.manifest_path = self._manifest()
        colors = []
        for path_name, call in (
            ("automatic", lambda: self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=True)),
            ("load", lambda: self.logic.loadKeyResultsFromManifest(self.manifest_path)),
            ("replace", lambda: self.logic.loadKeyResultsFromManifest(self.manifest_path, mode="replace")),
            ("append", lambda: self.logic.loadKeyResultsFromManifest(self.manifest_path, mode="append")),
            ("eye", lambda: _toggle_output(self._owner(), self._item(self.parcellation))),
        ):
            with self.subTest(path=path_name):
                for node in list(self._nodes_for(self.parcellation)):
                    self.scene.RemoveNode(node)
                with patch.dict(sys.modules, {"slicer": self.fake, "qt": _qt()}):
                    result = call()
                if result is not None:
                    self.assertFalse([item for item in result["skipped"] if item.get("path") == str(self.parcellation)])
                node, color = self._assert_segmentation(color_from=str(self.ctbl))
                self.assertTrue(node.GetDisplayVisibility())
                colors.append(color.GetID())
        self.assertEqual(len(set(colors)), 1, "every path must reuse the one colour node of the recorded .ctbl")

    def test_b2_parcellation_stale_or_missing_ctbl_uses_the_freesurfer_table(self) -> None:
        from shared.result_scene import FREESURFER_LUT

        for case, kwargs in (("older ctbl", {"ctbl_age": -60}), ("no ctbl", {"with_ctbl": False})):
            with self.subTest(case=case):
                self.scene.nodes.clear()
                self.manifest_path = self._manifest(**kwargs)
                self.logic.loadKeyResultsFromManifest(self.manifest_path, mode="replace")
                self._assert_segmentation(color_from=str(FREESURFER_LUT))

    def test_b2fix_display_D3_unreadable_ctbl_falls_back_to_the_freesurfer_table(self) -> None:
        """D3: slicer.util.loadColorTable raises on a bad .ctbl; the parcellation must still load with the FreeSurfer table."""
        from shared.result_scene import FREESURFER_LUT

        self.manifest_path = self._manifest()
        real_load = self.fake.util.loadColorTable

        def load_color(path):
            if str(path).endswith(".ctbl"):
                raise RuntimeError("Could not load color table from " + str(path))
            return real_load(path)

        self.fake.util.loadColorTable = load_color
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path, mode="replace")
        self.assertFalse([item for item in result["skipped"] if item.get("path") == str(self.parcellation)])
        self._assert_segmentation(color_from=str(FREESURFER_LUT))

    def test_b2_parcellation_failed_import_removes_the_temporary_labelmap(self) -> None:
        self.manifest_path = self._manifest()
        self.import_result = RuntimeError("import failed")
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path, source_path=self.parcellation)
        self.assertEqual([item["path"] for item in result["skipped"]], [str(self.parcellation)])
        self.assertFalse([node for node in self.scene.nodes if isinstance(node, _Node)])

    def test_b2_manual_load_colour_error_leaves_no_orphan_labelmap(self) -> None:
        regions = self.root / "s_selected_regions.nii.gz"
        regions.write_bytes(b"fixture regions")
        self.manifest_path = self._manifest(extra_tasks=(
            {"name": "statistics.regions", "status": "completed", "outputs": {"selected_regions": str(regions)}},))
        manifest = json.loads(self.manifest_path.read_text())
        manifest.update(selected_region_ids=[0])  # regional colours reject id 0
        self.manifest_path.write_text(json.dumps(manifest))
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path, source_path=regions)
        self.assertEqual([item["path"] for item in result["skipped"]], [str(regions)])
        self.assertFalse([node for node in self.scene.nodes if isinstance(node, _Node)])

    def test_b2_parcellation_show_automatic_results_keeps_the_segmentation_out_of_the_label_layer(self) -> None:
        self.manifest_path = self._manifest()
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=True)
        segmentation = self._nodes_for(self.parcellation)[0]
        ddseg = self._nodes_for(self.ddseg)[0]
        self.assertEqual(result["view_node_ids"].get("label"), ddseg.GetID())
        self.assertNotIn(segmentation.GetID(), result["view_node_ids"].values())

    def _slice_node(self, width=600, height=400):
        """A fake axial slice view: identity SliceToRAS, a field of view that does not show the segmentation."""
        matrix = SimpleNamespace(GetElement=lambda row, column: 1.0 if row == column else 0.0)
        return SimpleNamespace(GetDimensions=lambda: (width, height, 1), GetSliceToRAS=lambda: matrix,
                               GetFieldOfView=lambda: (250.0, 250.0, 1.0), SetFieldOfView=Mock())

    def _fake_slice_views(self, slice_node):
        original = self.fake.util.getNodesByClass
        self.fake.util.getNodesByClass = lambda kind: [slice_node] if kind == "vtkMRMLSliceNode" else original(kind)
        self.fake.vtkMRMLSliceNode = SimpleNamespace(JumpAllSlices=Mock(), CenteredJumpSlice=1)

    def _parcellation_only_manifest(self):
        path = self._manifest()
        manifest = json.loads(path.read_text())
        manifest["tasks"] = [task for task in manifest["tasks"] if task["name"] == "parcellation.ddparcel"]
        path.write_text(json.dumps(manifest))
        return path

    def test_b2fix_display_D2_segmentation_only_result_fits_and_centres_the_slice_views(self) -> None:
        """D2: a result with only DDParcel has no volume layer; the slice views must still show the segmentation."""
        slice_node = self._slice_node()
        self._fake_slice_views(slice_node)
        self.manifest_path = self._parcellation_only_manifest()
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=True)
        segmentation = self._nodes_for(self.parcellation)[0]
        self.assertEqual(result["view_node_ids"], {})  # the segmentation never becomes a volume layer
        self.fake.vtkMRMLSliceNode.JumpAllSlices.assert_called_once_with(self.scene, 179.0, -101.0, 129.0, 1)
        (fov_x, fov_y, fov_z), = [call.args for call in slice_node.SetFieldOfView.call_args_list]
        self.assertGreaterEqual(fov_x, 80.0)  # the 80 x 60 mm in-plane extent fits ...
        self.assertGreaterEqual(fov_y, 60.0)
        self.assertLess(fov_x, 3 * 80.0)  # ... and the view is zoomed to it, not left at 250 mm
        self.assertAlmostEqual(fov_x / fov_y, 600 / 400)
        self.assertEqual(fov_z, 1.0)
        self.assertIs(self._nodes_for(self.parcellation)[0], segmentation)

    def test_b2fix_display_D2_a_label_volume_still_decides_the_fit_when_there_is_one(self) -> None:
        slice_node = self._slice_node()
        self._fake_slice_views(slice_node)
        self.manifest_path = self._manifest()
        self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=True)
        slice_node.SetFieldOfView.assert_not_called()  # a volume layer is fitted by resetSliceViews, as before

    def test_b2fix_display_D4_automatic_ddseg_view_hides_the_parcellation_in_2d(self) -> None:
        """D4: with a DDSeg label layer shown, the automatic path loads DDParcel but leaves its 2D display off."""
        self.manifest_path = self._manifest()
        self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=True)
        segmentation = self._nodes_for(self.parcellation)[0]
        segmentation.display.SetVisibility2D.assert_called_once_with(False)
        self.assertTrue(segmentation.GetDisplayVisibility())  # loaded and listed as shown in Results

    def test_b2fix_display_D4_without_ddseg_or_by_hand_the_parcellation_shows_in_2d(self) -> None:
        for case in ("no ddseg", "manual"):
            with self.subTest(case=case):
                self.scene.nodes.clear()
                self.manifest_path = self._manifest()
                if case == "no ddseg":
                    data = json.loads(self.manifest_path.read_text())
                    data["tasks"] = [task for task in data["tasks"] if task["name"] != "segmentation.ddseg"]
                    self.manifest_path.write_text(json.dumps(data))
                self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=(case == "no ddseg"), mode="replace")
                segmentation = self._nodes_for(self.parcellation)[0]
                self.assertNotIn(((False,), {}), [(c.args, c.kwargs) for c in segmentation.display.SetVisibility2D.call_args_list])
                segmentation.display.SetVisibility2DFill.assert_called_with(True)

    def test_b2_parcellation_eye_hides_and_shows_a_segmentation_node(self) -> None:
        from shared.ui_run_history import _toggle_output

        self.manifest_path = self._manifest()
        self.logic.loadKeyResultsFromManifest(self.manifest_path, source_path=self.parcellation)
        node = self._nodes_for(self.parcellation)[0]
        self.assertEqual(node.GetClassName(), "vtkMRMLSegmentationNode")
        owner, item = self._owner(), self._item(self.parcellation)
        with patch.dict(sys.modules, {"slicer": self.fake, "qt": _qt()}):
            _toggle_output(owner, item)
            self.assertFalse(node.GetDisplayVisibility())
            _toggle_output(owner, item)
            self.assertTrue(node.GetDisplayVisibility())
        self.fake.util.errorDisplay.assert_not_called()
        self.assertEqual(self._nodes_for(self.parcellation), [node])  # toggled, not reloaded

    def test_b2_parcellation_unload_removes_the_segmentation_node(self) -> None:
        from shared.ui_run_history import _output_menu

        self.manifest_path = self._manifest()
        self.logic.loadKeyResultsFromManifest(self.manifest_path, source_path=self.parcellation)
        self.assertEqual(len(self._nodes_for(self.parcellation)), 1)
        owner, item = self._owner(), self._item(self.parcellation)
        owner._runOutputFiles = Mock()
        owner._runOutputFiles.itemAt.return_value = item
        menu = Mock()
        menu.addAction.side_effect = lambda text: text
        menu.exec_.side_effect = lambda *_args: "Unload from scene"
        fake_qt = _qt()
        fake_qt.QMenu = Mock(return_value=menu)
        with patch.dict(sys.modules, {"slicer": self.fake, "qt": fake_qt}), \
                patch("shared.ui_run_history._refresh_output_visibility"):
            _output_menu(owner, object())
        self.assertEqual(self._nodes_for(self.parcellation), [])

    def test_b2_ddseg_tissue_names_on_both_paths(self) -> None:
        from shared.segmentation_review import TISSUES

        self.manifest_path = self._manifest()
        colors = []
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                self.logic.loadKeyResultsFromManifest(self.manifest_path, automatic=automatic, mode="replace")
                nodes = self._nodes_for(self.ddseg)
                self.assertEqual([node.GetClassName() for node in nodes], ["vtkMRMLLabelMapVolumeNode"])
                color = self._color(nodes[0].GetDisplayNode().GetColorNodeID())
                self.assertIsNotNone(color)
                for value, name, rgb in TISSUES:
                    self.assertEqual(color.GetColorName(value), name)
                    self.assertEqual(tuple(round(v, 3) for v in color.GetTableValue(value)[:3]), rgb)
                colors.append(color.GetID())
        self.assertEqual(len(set(colors)), 1)

    def test_b2_manual_load_applies_the_categorical_presets(self) -> None:
        from shared.result_scene import FREESURFER_LUT

        node_parcellation = self.root / "s_node_parcellation.nii.gz"
        node_parcellation.write_bytes(b"fixture nodes")
        regions = self.root / "s_selected_regions.nii.gz"
        regions.write_bytes(b"fixture regions")
        self.manifest_path = self._manifest(extra_tasks=(
            {"name": "connectome.streamline_count", "status": "completed",
             "outputs": {"node_parcellation_path": str(node_parcellation)}},
            {"name": "statistics.regions", "status": "completed",
             "outputs": {"selected_regions": str(regions)}},
        ))
        manifest = json.loads(self.manifest_path.read_text())
        manifest.update(selected_region_ids=[1001, 2035], atlas_lut_path=str(self.ctbl))
        self.manifest_path.write_text(json.dumps(manifest))
        result = self.logic.loadKeyResultsFromManifest(self.manifest_path)
        self.assertFalse(result["skipped"])
        nodes = self._nodes_for(node_parcellation)
        self.assertEqual([node.GetClassName() for node in nodes], ["vtkMRMLLabelMapVolumeNode"])
        self.assertEqual(getattr(self._color(nodes[0].GetDisplayNode().GetColorNodeID()), "loaded_from", None),
                         str(FREESURFER_LUT))
        nodes = self._nodes_for(regions)
        self.assertEqual([node.GetClassName() for node in nodes], ["vtkMRMLLabelMapVolumeNode"])
        color = self._color(nodes[0].GetDisplayNode().GetColorNodeID())
        self.assertEqual(color.GetColorName(1001), "ctx-lh-bankssts")
        self.assertEqual(color.GetColorName(2035), "ROI-2035")  # not in the atlas table: a generated name
        self.assertEqual(color.GetAttribute("SlicerDDSurfer.LUTSourcePath"), str(self.ctbl))

    # -- Results tree fakes for the eye --------------------------------------------------------------------------

    def _owner(self):
        return SimpleNamespace(logic=self.logic, parent=None, _runOutputFiles=_Tree([]),
                               _latestResultManifestPath=self.manifest_path, _onboardingLanguage="en",
                               _open_local_path=Mock())

    def _item(self, path):
        item = Mock()
        roles = {256: str(path), 258: str(self.manifest_path)}
        item.data.side_effect = lambda _col, role: roles.get(role)
        item.text.side_effect = lambda col: "" if col == 1 else Path(path).name
        return item


class _Tree:
    def __init__(self, items):
        self.items = items

    def topLevelItemCount(self):
        return len(self.items)

    def topLevelItem(self, index):
        return self.items[index]


def _qt():
    fake = types.ModuleType("qt")
    fake.Qt = SimpleNamespace(UserRole=256)
    fake.QPixmap = Mock()
    fake.QIcon = Mock()
    fake.QByteArray = Mock()
    return fake


if __name__ == "__main__":
    unittest.main()
