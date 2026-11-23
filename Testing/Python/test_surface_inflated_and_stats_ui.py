"""Tests for inflated surface subgroup placement, morphometry overlays on inflated surfaces, and in-Slicer .stats viewer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import vtk
except ImportError:  # pragma: no cover
    np = vtk = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

from test_b2_surface_morphometry_display import (
    DECIDED_ORDER,
    FACES,
    PRODUCT,
    VERTICES,
    AdaptiveSurfaceCarrierTest,
    RunFolder,
    TempRootTest,
    _Node,
    write_obj,
)

SAMPLE_FREESURFER_APARC_STATS = """# Title Segmentation Statistics
# generating_program mris_anatomical_stats
# subjectname sub-01
# hemi lh
# Measure Cortex, NumVert, Number of Vertices, 134567, vertices
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 89123.4, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.54321, mm
# NTableCols 10
# TableCol  1 ColHeader StructName
# TableCol  1 Units     NA
# TableCol  2 ColHeader NumVert
# TableCol  2 Units     unitless
# TableCol  3 ColHeader SurfArea
# TableCol  3 Units     mm^2
# TableCol  4 ColHeader GrayVol
# TableCol  4 Units     mm^3
# TableCol  5 ColHeader ThickAvg
# TableCol  5 Units     mm
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
bankssts                                 1182    842   2105  2.451 0.422     0.118     0.023       12     1.4
caudalanteriorcingulate                   905    618   1821  2.690 0.541     0.131     0.027        9     1.1
insula                                   3201   2194   6840  3.012 0.615     0.125     0.029       31     3.2
"""

SAMPLE_DDSURFER_A2009S_STATS = """# DDSurfer regional surface statistics
# Cortex label excludes non-cortical vertices.
# Area is allocated to incident vertices; thickness mean/std use unweighted vertices (ddof=0).
# No voxel-wise segmentation volumes or intracranial volume estimates are included.
# subjectname sub-02
# hemi lh
# annotation aparc.a2009s
# NRows 3
# NTableCols 8
# TableCol 1 ColHeader StructName
# TableCol 1 Units NA
# TableCol 2 ColHeader NumVert
# TableCol 2 Units unitless
# TableCol 3 ColHeader SurfArea
# TableCol 3 Units mm^2
# TableCol 4 ColHeader PialArea
# TableCol 4 Units mm^2
# TableCol 5 ColHeader ThickAvg
# TableCol 5 Units mm
# TableCol 6 ColHeader ThickStd
# TableCol 6 Units mm
# TableCol 7 ColHeader MeanAbsCurv
# TableCol 7 Units mm^-1
# TableCol 8 ColHeader MeanAbsGaussCurv
# TableCol 8 Units mm^-2
# ColHeaders StructName NumVert SurfArea PialArea ThickAvg ThickStd MeanAbsCurv MeanAbsGaussCurv
G_and_S_frontomargin 1250 812.345000 945.120000 2.410000 0.480000 0.142000 0.031000
G_insular_short 640 420.500000 510.250000 3.180000 0.590000 0.119000 0.024000
S_central 2980 2015.750000 2290.100000 1.985000 0.375000 0.155000 0.028000
"""


class _Signal:
    def __init__(self) -> None:
        self.callbacks: list = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, *args) -> None:
        for cb in list(self.callbacks):
            cb(*args)


class _FakeStatsQt:
    """Minimal Qt fake that records dialog structure, table contents, filtering, clipboard, and actions."""

    class Qt:
        EditRole = 2
        AlignRight = 0x0002
        AlignVCenter = 0x0080

    class QAbstractItemView:
        NoEditTriggers = 0
        SelectRows = 1

    class QUrl:
        @staticmethod
        def fromLocalFile(path: str) -> str:
            return f"file://{path}"

    def __init__(self) -> None:
        self.dialogs: list = []
        self.clipboard_text = ""
        self.desktop_opened: list[str] = []
        fake_qt = self

        class _Clipboard:
            def setText(self, text: str) -> None:
                fake_qt.clipboard_text = text

        class QApplication:
            @staticmethod
            def clipboard():
                return _Clipboard()

        class QDesktopServices:
            @staticmethod
            def openUrl(url: str) -> bool:
                fake_qt.desktop_opened.append(url)
                return True

        class _Widget:
            def __init__(self, parent=None) -> None:
                self.parent = parent
                self.visible = False
                self.closed = False
                self.children_widgets: list = []

            def show(self) -> None:
                self.visible = True

            def raise_(self) -> None:
                pass

            def activateWindow(self) -> None:
                pass

            def close(self) -> None:
                self.closed = True
                self.visible = False

            def accept(self) -> None:
                self.close()

            def resize(self, w: int, h: int) -> None:
                self.size = (w, h)

            def setMinimumSize(self, w: int, h: int) -> None:
                self.min_size = (w, h)

        class QDialog(_Widget):
            def __init__(self, parent=None) -> None:
                super().__init__(parent)
                self.title = ""
                self.labels: list = []
                self.group_boxes: list = []
                self.tables: list = []
                self.plain_texts: list = []
                self.line_edits: list = []
                self.buttons: list = []
                fake_qt.dialogs.append(self)

            def setWindowTitle(self, title: str) -> None:
                self.title = title

        class _Layout:
            def __init__(self, parent=None) -> None:
                self.parent = parent
                self.items: list = []

            def _root_dialog(self):
                cur = self.parent
                while cur is not None and not isinstance(cur, QDialog):
                    cur = getattr(cur, "parent", None)
                return cur

            def addWidget(self, widget, *_a) -> None:
                self.items.append(widget)
                dlg = self._root_dialog()
                if dlg is not None:
                    if isinstance(widget, QLabel):
                        dlg.labels.append(widget)
                    elif isinstance(widget, QGroupBox):
                        dlg.group_boxes.append(widget)
                    elif isinstance(widget, QTableWidget):
                        dlg.tables.append(widget)
                    elif isinstance(widget, QPlainTextEdit):
                        dlg.plain_texts.append(widget)
                    elif isinstance(widget, QLineEdit):
                        dlg.line_edits.append(widget)
                    elif isinstance(widget, QPushButton):
                        dlg.buttons.append(widget)

            def addLayout(self, child_layout) -> None:
                self.items.append(child_layout)
                dlg = self._root_dialog()
                if dlg is not None:
                    child_layout.parent = dlg
                    for item in child_layout.items:
                        if isinstance(item, QLabel):
                            dlg.labels.append(item)
                        elif isinstance(item, QLineEdit):
                            dlg.line_edits.append(item)
                        elif isinstance(item, QPushButton):
                            dlg.buttons.append(item)

            def addStretch(self, *_a) -> None:
                pass

        class QVBoxLayout(_Layout):
            pass

        class QHBoxLayout(_Layout):
            pass

        class QLabel(_Widget):
            def __init__(self, text: str = "", parent=None) -> None:
                super().__init__(parent)
                self._text = text

            def text(self) -> str:
                return self._text

            def setText(self, text: str) -> None:
                self._text = text

            def setWordWrap(self, flag: bool) -> None:
                self.word_wrap = flag

        class QGroupBox(_Widget):
            def __init__(self, title: str = "", parent=None) -> None:
                super().__init__(parent)
                self.title = title

        class QLineEdit(_Widget):
            def __init__(self, parent=None) -> None:
                super().__init__(parent)
                self.placeholder = ""
                self.textChanged = _Signal()

            def setPlaceholderText(self, text: str) -> None:
                self.placeholder = text

            def setClearButtonEnabled(self, flag: bool) -> None:
                self.clear_enabled = flag

        class QPushButton(_Widget):
            def __init__(self, text: str = "", parent=None) -> None:
                super().__init__(parent)
                self._text = text
                self.clicked = _Signal()

            def text(self) -> str:
                return self._text

        class QPlainTextEdit(_Widget):
            def __init__(self, parent=None) -> None:
                super().__init__(parent)
                self.read_only = False
                self.plain_text = ""

            def setReadOnly(self, flag: bool) -> None:
                self.read_only = flag

            def setPlainText(self, text: str) -> None:
                self.plain_text = text

        class QTableWidgetItem:
            def __init__(self, text: str = "") -> None:
                self._text = text
                self._data: dict = {}
                self.alignment = None

            def text(self) -> str:
                return self._text

            def setData(self, role: int, value) -> None:
                self._data[role] = value

            def data(self, role: int):
                return self._data.get(role)

            def setTextAlignment(self, align: int) -> None:
                self.alignment = align

        class QTableWidget(_Widget):
            def __init__(self, rows: int, cols: int, parent=None) -> None:
                super().__init__(parent)
                self._rows = rows
                self._cols = cols
                self.headers: list[str] = []
                self.cells: dict[tuple[int, int], QTableWidgetItem] = {}
                self.hidden_rows: set[int] = set()
                self.sorting_enabled = False
                self.alternating = False
                self.edit_triggers = None
                self.selection_behavior = None

            def rowCount(self) -> int:
                return self._rows

            def columnCount(self) -> int:
                return self._cols

            def setHorizontalHeaderLabels(self, labels: list[str]) -> None:
                self.headers = list(labels)

            def setEditTriggers(self, trig) -> None:
                self.edit_triggers = trig

            def setAlternatingRowColors(self, flag: bool) -> None:
                self.alternating = flag

            def setSelectionBehavior(self, behavior) -> None:
                self.selection_behavior = behavior

            def setSortingEnabled(self, flag: bool) -> None:
                self.sorting_enabled = flag

            def setItem(self, r: int, c: int, item: QTableWidgetItem) -> None:
                self.cells[(r, c)] = item

            def item(self, r: int, c: int):
                return self.cells.get((r, c))

            def resizeColumnsToContents(self) -> None:
                pass

            def setRowHidden(self, r: int, hidden: bool) -> None:
                if hidden:
                    self.hidden_rows.add(r)
                else:
                    self.hidden_rows.discard(r)

        self.QApplication = QApplication
        self.QDesktopServices = QDesktopServices
        self.QDialog = QDialog
        self.QVBoxLayout = QVBoxLayout
        self.QHBoxLayout = QHBoxLayout
        self.QLabel = QLabel
        self.QGroupBox = QGroupBox
        self.QLineEdit = QLineEdit
        self.QPushButton = QPushButton
        self.QPlainTextEdit = QPlainTextEdit
        self.QTableWidgetItem = QTableWidgetItem
        self.QTableWidget = QTableWidget


# ---------------------------------------------------------------------------- 1. Inflated in Surfaces subgroup
class InflatedSubgroupPlacementTest(TempRootTest):
    def _build_tree(self, run: RunFolder):
        from shared.result_scene import result_candidates as real_candidates
        from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest
        from test_ui_results_tree import Harness

        data = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        data.update({"kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed",
                     "ended_at": "2026-11-01T10:00:00Z"})
        run.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        with mock.patch("shared.result_scene.result_candidates", side_effect=real_candidates):
            h.refresh([parse_run_manifest(run.manifest_path)])
        return h

    def test_inflated_files_appear_under_cortical_surfaces_alongside_white_and_pial(self) -> None:
        from shared import ui_run_history
        from shared.surface_morphometry_display import MORPHOMETRY_ROW_KIND
        from test_ui_results_tree import USER_ROLE, find_child

        run = RunFolder(self.root)
        for hemi in ("lh", "rh"):
            pial = write_obj(run.surface_dir / f"{hemi}.pial.obj", shift=(0.5, 0.5, 0.5))
            run.manifest["tasks"][0]["outputs"][f"pial_{hemi}"] = str(pial)
        run.manifest_path.write_text(json.dumps(run.manifest), encoding="utf-8")

        h = self._build_tree(run)
        subject = next(item for item in h.tree.items if item.text(0).startswith("s"))
        anatomy = find_child(subject, "Anatomical priors")
        surf_grp = find_child(anatomy, "Cortical surfaces")
        post_grp = find_child(anatomy, "Surface postprocess")

        self.assertEqual(surf_grp.data(0, USER_ROLE + ui_run_history.ROLE_KEY), "anatomy::surfaces")
        self.assertEqual(post_grp.data(0, USER_ROLE + ui_run_history.ROLE_KEY), "anatomy::postprocess")

        surf_rows = [surf_grp.child(i) for i in range(surf_grp.childCount())]
        post_rows = [post_grp.child(i) for i in range(post_grp.childCount())]
        surf_names = [row.text(0) for row in surf_rows]
        post_names = [row.text(0) for row in post_rows]

        # white, pial, and inflated are all grouped together under Cortical surfaces
        for expected in ("lh.white.obj", "lh.pial.obj", "rh.white.obj", "rh.pial.obj", "lh.inflated", "rh.inflated"):
            self.assertIn(expected, surf_names)
        # inflated is NOT duplicated under Surface postprocess
        self.assertNotIn("lh.inflated", post_names)
        self.assertNotIn("rh.inflated", post_names)
        self.assertEqual(post_names, [name for name in DECIDED_ORDER if not name.endswith(".inflated")])

        # inflated rows keep MORPHOMETRY_ROW_KIND, manifest pointer, eye icon (not external arrow), and tooltips
        by_name = {row.text(0): row for row in surf_rows}
        for name in ("lh.inflated", "rh.inflated"):
            row = by_name[name]
            self.assertEqual(row.data(0, USER_ROLE + ui_run_history.ROLE_KIND), MORPHOMETRY_ROW_KIND)
            self.assertEqual(row.data(0, USER_ROLE + ui_run_history.ROLE_MANIFEST), str(run.manifest_path))
            self.assertEqual(row.text(1), "")
            self.assertIn(1, row.icons)
            self.assertIn("Show the inflated surface", row.tooltips[0])

    def test_postprocess_with_only_inflated_does_not_create_empty_postprocess_subgroup(self) -> None:
        from test_ui_results_tree import find_child

        run = RunFolder(self.root, morph_names=["lh.inflated", "rh.inflated"])
        run.manifest["tasks"] = [run.manifest["tasks"][1]]
        run.manifest_path.write_text(json.dumps(run.manifest), encoding="utf-8")

        h = self._build_tree(run)
        subject = next(item for item in h.tree.items if item.text(0).startswith("s"))
        anatomy = find_child(subject, "Anatomical priors")
        subgroups = [anatomy.child(i).text(0) for i in range(anatomy.childCount())]
        self.assertEqual(subgroups, ["Cortical surfaces (2)"])

    def test_surface_carrier_path_helpers_and_inflated_carrier_resolution(self) -> None:
        from shared.surface_morphometry_display import (
            SURFACES,
            carrier_surface_of_node,
            inflated_carrier,
            surface_from_carrier_path,
            surface_from_obj_path,
            surface_from_path,
        )

        self.assertEqual(SURFACES, ("white", "pial", "inflated"))
        self.assertEqual(surface_from_carrier_path("/x/lh.white.obj"), ("lh", "white"))
        self.assertEqual(surface_from_carrier_path("/x/rh.pial.obj"), ("rh", "pial"))
        self.assertEqual(surface_from_carrier_path("/x/lh.inflated"), ("lh", "inflated"))
        self.assertEqual(surface_from_path("/x/rh.inflated"), ("rh", "inflated"))
        self.assertIsNone(surface_from_carrier_path("/x/lh.thickness"))
        self.assertIsNone(surface_from_carrier_path("/x/lh.white.stl"))
        # Backwards compatibility of surface_from_obj_path
        self.assertEqual(surface_from_obj_path("/x/lh.white.obj"), ("lh", "white"))
        self.assertIsNone(surface_from_obj_path("/x/lh.inflated"))

        # carrier_surface_of_node infers inflated from either CarrierSurface or CarrierPath
        n1 = _Node("n1", {"SlicerDDSurfer.CarrierSurface": "inflated"})
        n2 = _Node("n2", {"SlicerDDSurfer.CarrierPath": "/run/surface_morphometry/rh.inflated"})
        n_plain = _Node("p", {"SlicerDDSurfer.SourcePath": "/run/surface_morphometry/lh.inflated"})
        self.assertEqual(carrier_surface_of_node(n1), "inflated")
        self.assertEqual(carrier_surface_of_node(n2), "inflated")
        self.assertIsNone(carrier_surface_of_node(n_plain))

        # inflated_carrier finds from task outputs and falls back to published layout (respecting M4 boundary)
        run = RunFolder(self.root)
        self.assertEqual(inflated_carrier(run.manifest, "lh"), run.morph("lh.inflated"))
        manifest_no_post = dict(run.manifest, tasks=[])
        self.assertEqual(inflated_carrier(manifest_no_post, "lh", run.morph("lh.thickness")), run.morph("lh.inflated"))
        self.assertEqual(inflated_carrier(manifest_no_post, "rh"), run.morph("rh.inflated"))
        other_layout = dict(manifest_no_post, output_root=str(self.root / "other" / "outputs"))
        self.assertIsNone(inflated_carrier(other_layout, "lh", run.morph("lh.thickness")))

    def test_sibling_hide_works_for_inflated_overlays_without_touching_white_or_other_hemi(self) -> None:
        from shared.surface_morphometry_display import PREFIX, hide_same_surface_siblings

        lh_inf = "/o/m/lh.inflated"
        rh_inf = "/o/m/rh.inflated"
        lh_white = "/o/s/lh.white.obj"
        run = "/o/00_run_metadata/m.json"

        def mk(nid, src, carrier=None):
            attrs = {PREFIX + "Run": run, PREFIX + "SourcePath": src}
            if carrier:
                attrs[PREFIX + "CarrierPath"] = carrier
            return _Node(nid, attrs)

        plain_lh_inf = mk("pi", lh_inf)
        thick_lh_inf = mk("ti", "/o/m/lh.thickness", lh_inf)
        aparc_lh_inf = mk("ai", "/o/m/lh.aparc.annot", lh_inf)
        plain_lh_white = mk("pw", lh_white)
        thick_lh_white = mk("tw", "/o/m/lh.thickness", lh_white)
        plain_rh_inf = mk("ri", rh_inf)
        scene = [plain_lh_inf, thick_lh_inf, aparc_lh_inf, plain_lh_white, thick_lh_white, plain_rh_inf]
        fake = SimpleNamespace(util=SimpleNamespace(getNodesByClass=lambda kind: scene if kind != "vtkMRMLTableNode" else []))

        hidden = hide_same_surface_siblings(fake, [thick_lh_inf])
        self.assertEqual(sorted(n.GetID() for n in hidden), ["ai", "pi"])
        self.assertTrue(thick_lh_inf.visible)
        self.assertFalse(plain_lh_inf.visible or aparc_lh_inf.visible)
        self.assertTrue(plain_lh_white.visible and thick_lh_white.visible and plain_rh_inf.visible)


# ------------------------------------------------------------ 2. Overlays on inflated surfaces (real nibabel/VTK)
@unittest.skipIf(np is None or vtk is None or nib is None, "numpy/VTK/nibabel required")
class InflatedOverlayDisplayTest(AdaptiveSurfaceCarrierTest):
    def _run_with_inflated(self):
        run = self._run_with_white_and_pial()
        count = len(VERTICES)
        cras = np.array([10.0, -5.0, 2.0])
        info = {
            "head": np.array([2, 0, 20], dtype=np.int32),
            "valid": "1  # volume info valid",
            "filename": "brain.mgz",
            "volume": np.array([256, 256, 256]),
            "voxelsize": np.array([1.0, 1.0, 1.0]),
            "xras": np.array([-1.0, 0.0, 0.0]),
            "yras": np.array([0.0, 0.0, -1.0]),
            "zras": np.array([0.0, 1.0, 0.0]),
            "cras": cras,
        }
        inflated_pts = np.array(VERTICES, dtype=np.float32) * 2.0
        for hemi in ("lh", "rh"):
            nib.freesurfer.write_geometry(
                str(run.morph(f"{hemi}.inflated")),
                inflated_pts,
                np.array(FACES, dtype=np.int32),
                volume_info=info,
            )
        nib.freesurfer.write_morph_data(str(run.morph("lh.curv")), np.linspace(-0.4, 0.4, count).astype(np.float32))
        nib.freesurfer.write_morph_data(str(run.morph("lh.area")), np.linspace(0.2, 1.5, count).astype(np.float32))
        ctab = np.array([[220, 20, 10, 0, 0], [20, 220, 10, 0, 0]], dtype=np.int32)
        nib.freesurfer.write_annot(
            str(run.morph("lh.aparc.a2009s.annot")),
            np.array([1, 0, -1, 1, 0, 1], dtype=np.int32),
            ctab,
            [b"G_front", b"S_central"],
            fill_ctab=True,
        )
        return run, inflated_pts + cras

    def _add_plain_inflated(self, run, hemi: str = "lh", *, visible: bool = True):
        from shared.surface_morphometry_display import PREFIX

        node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", f"{hemi}.inflated")
        node.SetAttribute(PREFIX + "Run", str(run.manifest_path.resolve()))
        node.SetAttribute(PREFIX + "SourcePath", str(run.morph(f"{hemi}.inflated").resolve()))
        node.visible = visible
        node.GetDisplayVisibility = lambda n=node: int(n.visible)
        node.SetDisplayVisibility = lambda v, n=node: setattr(n, "visible", bool(v))
        return node

    def test_measures_and_annotations_load_onto_open_inflated_surface_with_scanner_ras_geometry(self) -> None:
        from shared.surface_morphometry_display import PREFIX, active_carrier_surfaces, load_morphometry_row

        run, expected_pts = self._run_with_inflated()
        self._add_plain_inflated(run, "lh", visible=True)
        self.assertEqual(active_carrier_surfaces(self.fake, run.manifest, "lh", run.morph("lh.thickness")), ("inflated",))

        for name, array_name in (
            ("lh.thickness", "thickness"),
            ("lh.curv", "curv"),
            ("lh.area", "area"),
            ("lh.aparc.annot", "aparc"),
            ("lh.aparc.a2009s.annot", "aparc_a2009s"),
        ):
            with self.subTest(name=name):
                before = set(self.scene.nodes)
                cand = {"path": str(run.morph(name)), "product": PRODUCT, "key": name}
                self.assertTrue(load_morphometry_row(self.fake, cand, run.manifest))
                new_nodes = [n for nid, n in self.scene.nodes.items() if nid not in before]
                models = [n for n in new_nodes if n.kind == "vtkMRMLModelNode"]
                self.assertEqual(len(models), 1)
                model = models[0]
                self.assertEqual(model.name, f"{name} (inflated)")
                self.assertEqual(model.GetAttribute(PREFIX + "CarrierSurface"), "inflated")
                self.assertEqual(model.GetAttribute(PREFIX + "CarrierPath"), str(run.morph("lh.inflated").resolve()))
                self.assertEqual(model.GetNumberOfStorageNodes(), 0)
                poly = model.GetPolyData()
                self.assertEqual(poly.GetNumberOfPoints(), len(VERTICES))
                pts = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
                np.testing.assert_allclose(pts, expected_pts, atol=1e-4)
                self.assertIsNotNone(poly.GetPointData().GetArray(array_name))
                if name.endswith(".annot"):
                    self.assertTrue(any(n.kind == "vtkMRMLColorTableNode" for n in new_nodes))
                else:
                    self.assertTrue(any(n.kind == "vtkMRMLColorLegendDisplayNode" for n in new_nodes))

    def test_count_mismatch_on_inflated_surface_leaves_no_leaked_nodes(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        run, _ = self._run_with_inflated()
        self._add_plain_inflated(run, "lh", visible=True)
        before = set(self.scene.nodes)
        cand = {"path": str(run.morph("lh.sulc")), "product": PRODUCT, "key": "lh.sulc"}  # has len(VERTICES) - 1
        with self.assertRaises(MorphometryError):
            load_morphometry_row(self.fake, cand, run.manifest)
        self.assertEqual(set(self.scene.nodes), before)

    def test_white_and_inflated_both_open_loads_overlay_on_both_with_single_legend(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row

        run, _ = self._run_with_inflated()
        self._add_plain_surface(run, "lh.white.obj", visible=True)
        self._add_plain_inflated(run, "lh", visible=True)
        before = set(self.scene.nodes)
        cand = {"path": str(run.morph("lh.thickness")), "product": PRODUCT, "key": "lh.thickness"}
        self.assertTrue(load_morphometry_row(self.fake, cand, run.manifest))
        new_nodes = [n for nid, n in self.scene.nodes.items() if nid not in before]
        models = [n for n in new_nodes if n.kind == "vtkMRMLModelNode"]
        legends = [n for n in new_nodes if n.kind == "vtkMRMLColorLegendDisplayNode"]
        self.assertEqual([m.GetAttribute(PREFIX + "CarrierSurface") for m in models], ["white", "inflated"])
        self.assertEqual([m.name for m in models], ["lh.thickness (white)", "lh.thickness (inflated)"])
        self.assertEqual(len(legends), 1)

    def test_ui_toggle_lifecycle_for_inflated_and_overlays(self) -> None:
        from shared import ui_run_history
        from shared.result_scene import result_candidates as real_candidates, source_stamp
        from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row
        from test_ui_results_tree import Harness, find_child

        run, _ = self._run_with_inflated()
        data = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        data.update({"kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed",
                     "ended_at": "2026-11-01T10:00:00Z"})
        run.manifest_path.write_text(json.dumps(data), encoding="utf-8")

        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        h.slicer = self.fake
        h.slicer.util.errorDisplay = mock.MagicMock()
        h.owner.parent = None

        def _stamp_node(node, manifest_path, product, spath):
            node.SetAttribute(PREFIX + "Run", str(manifest_path.resolve()))
            node.SetAttribute(PREFIX + "Product", product)
            node.SetAttribute(PREFIX + "SourcePath", str(spath.resolve()))
            node.SetAttribute(PREFIX + "SourceStamp", source_stamp(spath))

        def fake_load(manifest_path, *, source_path=None, morphometry_surfaces=None, **_kw):
            spath = Path(source_path)
            before = set(self.scene.nodes)
            if spath.name.endswith(".obj"):
                node = self._add_plain_surface(run, spath.name, visible=True)
                _stamp_node(node, manifest_path, "surface.ddsurfer", spath)
                return {"loaded": [{"path": str(spath)}], "reused": [], "skipped": [], "deferred": []}
            cand = {"path": str(spath), "product": PRODUCT, "key": spath.name}
            load_morphometry_row(self.fake, cand, run.manifest, surfaces=morphometry_surfaces)
            for nid, node in list(self.scene.nodes.items()):
                if nid not in before and node.kind == "vtkMRMLModelNode":
                    node.visible = True
                    node.GetDisplayVisibility = lambda n=node: int(n.visible)
                    node.SetDisplayVisibility = lambda v, n=node: setattr(n, "visible", bool(v))
                    _stamp_node(node, manifest_path, PRODUCT, spath)
            return {"loaded": [{"path": str(spath)}], "reused": [], "skipped": [], "deferred": []}

        h.owner.logic = SimpleNamespace(loadKeyResultsFromManifest=fake_load)
        with mock.patch("shared.result_scene.result_candidates", side_effect=real_candidates), \
                mock.patch.dict(sys.modules, {"qt": h.qt, "slicer": h.slicer}):
            h.refresh([parse_run_manifest(run.manifest_path)])
            subject = next(item for item in h.tree.items if item.text(0).startswith("s"))
            anatomy = find_child(subject, "Anatomical priors")
            surf_grp = find_child(anatomy, "Cortical surface")
            post_grp = find_child(anatomy, "Surface postprocess")
            surf_rows = {surf_grp.child(i).text(0): surf_grp.child(i) for i in range(surf_grp.childCount())}
            post_rows = {post_grp.child(i).text(0): post_grp.child(i) for i in range(post_grp.childCount())}

            # 1. Open lh.inflated from Cortical surfaces
            ui_run_history._toggle_output(h.owner, surf_rows["lh.inflated"])
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Hide")
            plain_inf = next(
                n for n in self.scene.nodes.values()
                if n.kind == "vtkMRMLModelNode"
                and n.GetAttribute(PREFIX + "SourcePath") == str(run.morph("lh.inflated").resolve())
            )
            self.assertTrue(plain_inf.GetDisplayVisibility())

            # 2. Show lh.aparc.annot -> displays on lh.inflated and hides plain lh.inflated
            ui_run_history._toggle_output(h.owner, post_rows["lh.aparc.annot"])
            self.assertEqual(post_rows["lh.aparc.annot"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Show")
            self.assertFalse(plain_inf.GetDisplayVisibility())
            aparc_nodes = [
                n for n in self.scene.nodes.values()
                if n.kind == "vtkMRMLModelNode"
                and n.GetAttribute(PREFIX + "SourcePath") == str(run.morph("lh.aparc.annot").resolve())
                and n.GetDisplayVisibility()
            ]
            self.assertEqual([n.GetAttribute(PREFIX + "CarrierSurface") for n in aparc_nodes], ["inflated"])

            # 3. Switch overlay to lh.thickness while aparc is active on inflated -> detects inflated via CarrierPath
            ui_run_history._toggle_output(h.owner, post_rows["lh.thickness"])
            self.assertEqual(post_rows["lh.thickness"].tooltips[1], "Hide")
            self.assertEqual(post_rows["lh.aparc.annot"].tooltips[1], "Show")
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Hide")
            thick_nodes = [
                n for n in self.scene.nodes.values()
                if n.kind == "vtkMRMLModelNode"
                and n.GetAttribute(PREFIX + "SourcePath") == str(run.morph("lh.thickness").resolve())
                and n.GetDisplayVisibility()
            ]
            self.assertEqual([n.GetAttribute(PREFIX + "CarrierSurface") for n in thick_nodes], ["inflated"])

            # 4. Hide lh.thickness -> reveals plain lh.inflated again
            ui_run_history._toggle_output(h.owner, post_rows["lh.thickness"])
            self.assertEqual(post_rows["lh.thickness"].tooltips[1], "Show")
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Hide")
            self.assertTrue(plain_inf.GetDisplayVisibility())

            # 5. Show lh.aparc.a2009s.annot on lh.inflated, then turn OFF lh.inflated eye -> hides BOTH!
            ui_run_history._toggle_output(h.owner, post_rows["lh.aparc.a2009s.annot"])
            self.assertEqual(post_rows["lh.aparc.a2009s.annot"].tooltips[1], "Hide")
            ui_run_history._toggle_output(h.owner, surf_rows["lh.inflated"])
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Show")
            self.assertEqual(post_rows["lh.aparc.a2009s.annot"].tooltips[1], "Show")
            self.assertFalse(plain_inf.GetDisplayVisibility())

            # 6. Load lh.curv on white first, then switch to inflated -> loads missing inflated carrier
            ui_run_history._toggle_output(h.owner, post_rows["lh.curv"])  # defaults to white
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Hide")
            ui_run_history._toggle_output(h.owner, surf_rows["lh.white.obj"])  # hide white + curv(white)
            ui_run_history._toggle_output(h.owner, surf_rows["lh.inflated"])  # show inflated
            ui_run_history._toggle_output(h.owner, post_rows["lh.curv"])  # now switches to inflated
            visible_curv = [
                n for n in self.scene.nodes.values()
                if n.kind == "vtkMRMLModelNode"
                and n.GetAttribute(PREFIX + "SourcePath") == str(run.morph("lh.curv").resolve())
                and n.GetDisplayVisibility()
            ]
            self.assertEqual([n.GetAttribute(PREFIX + "CarrierSurface") for n in visible_curv], ["inflated"])
            self.assertEqual(surf_rows["lh.inflated"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Show")


# ------------------------------------------------------------------ 3. FreeSurfer .stats parsing and in-Slicer UI
class FreeSurferStatsParserAndUiTest(TempRootTest):
    def test_parse_freesurfer_aparc_stats_with_measures_and_colheaders(self) -> None:
        from shared.surface_morphometry_display import parse_freesurfer_stats, stats_to_tsv

        stats_file = self.root / "lh.aparc.stats"
        stats_file.write_text(SAMPLE_FREESURFER_APARC_STATS, encoding="utf-8")

        parsed = parse_freesurfer_stats(stats_file)
        self.assertEqual(
            parsed["measures"],
            [
                ("NumVert", "Number of Vertices", "134567", "vertices"),
                ("WhiteSurfArea", "White Surface Total Area", "89123.4", "mm^2"),
                ("MeanThickness", "Mean Thickness", "2.54321", "mm"),
            ],
        )
        self.assertEqual(
            parsed["columns"],
            ["StructName", "NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"],
        )
        self.assertEqual(len(parsed["rows"]), 3)
        self.assertEqual(parsed["rows"][0][0], "bankssts")
        self.assertEqual(parsed["rows"][0][1], "1182")
        self.assertEqual(parsed["rows"][2][0], "insula")
        self.assertEqual(parsed["units"]["SurfArea"], "mm^2")
        self.assertEqual(parsed["units"]["ThickAvg"], "mm")
        self.assertEqual(parsed["metadata"]["subjectname"], "sub-01")
        self.assertEqual(parsed["metadata"]["hemi"], "lh")

        tsv = stats_to_tsv(parsed)
        tsv_lines = tsv.splitlines()
        self.assertEqual(len(tsv_lines), 4)
        self.assertTrue(tsv_lines[0].startswith("StructName\tNumVert\tSurfArea"))
        self.assertTrue(tsv_lines[1].startswith("bankssts\t1182\t842"))

    def test_parse_ddsurfer_a2009s_stats(self) -> None:
        from shared.surface_morphometry_display import parse_freesurfer_stats

        stats_file = self.root / "lh.aparc.a2009s.stats"
        stats_file.write_text(SAMPLE_DDSURFER_A2009S_STATS, encoding="utf-8")

        parsed = parse_freesurfer_stats(stats_file)
        self.assertEqual(parsed["measures"], [])
        self.assertEqual(parsed["metadata"]["annotation"], "aparc.a2009s")
        self.assertEqual(parsed["metadata"]["NRows"], "3")
        self.assertEqual(
            parsed["columns"],
            ["StructName", "NumVert", "SurfArea", "PialArea", "ThickAvg", "ThickStd", "MeanAbsCurv", "MeanAbsGaussCurv"],
        )
        self.assertEqual(len(parsed["rows"]), 3)
        self.assertEqual(parsed["rows"][1][0], "G_insular_short")
        self.assertEqual(parsed["units"]["PialArea"], "mm^2")
        self.assertEqual(parsed["units"]["MeanAbsCurv"], "mm^-1")

    def test_parse_empty_and_malformed_stats_fallback(self) -> None:
        from shared.surface_morphometry_display import parse_freesurfer_stats, stats_to_tsv

        empty_file = self.root / "empty.stats"
        empty_file.write_text("", encoding="utf-8")
        empty_parsed = parse_freesurfer_stats(empty_file)
        self.assertEqual(empty_parsed["measures"], [])
        self.assertEqual(empty_parsed["columns"], [])
        self.assertEqual(empty_parsed["rows"], [])
        self.assertEqual(empty_parsed["raw_text"], "")

        malformed = self.root / "notes.stats"
        raw = "# Free-form notes without ColHeaders\nsome unstructured text\nanother line\n"
        malformed.write_text(raw, encoding="utf-8")
        parsed = parse_freesurfer_stats(malformed)
        self.assertEqual(parsed["columns"], [])
        self.assertEqual(parsed["rows"], [])
        self.assertEqual(parsed["raw_text"], raw)
        self.assertEqual(stats_to_tsv(parsed), raw)

    @unittest.skipIf(vtk is None, "VTK required")
    def test_load_stats_table_node_creates_and_reuses_typed_vtk_table(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_stats_table_node

        stats_file = self.root / "lh.aparc.a2009s.stats"
        stats_file.write_text(SAMPLE_DDSURFER_A2009S_STATS, encoding="utf-8")

        class _FakeTableNode:
            def __init__(self, nid: str, name: str) -> None:
                self.nid = nid
                self.name = name
                self.table = None
                self.attrs: dict[str, str] = {}
                self.units: dict[str, str] = {}
                self.header_flag = False
                self.locked = False

            def GetID(self): return self.nid
            def SetAndObserveTable(self, t): self.table = t
            def SetUseColumnNameAsColumnHeader(self, v): self.header_flag = bool(v)
            def SetLocked(self, v): self.locked = bool(v)
            def SetColumnUnitLabel(self, col, unit): self.units[col] = unit
            def SetAttribute(self, k, v): self.attrs[k] = v
            def GetAttribute(self, k): return self.attrs.get(k)

        tables: list[_FakeTableNode] = []

        def add_new(cls_name: str, name: str):
            self.assertEqual(cls_name, "vtkMRMLTableNode")
            node = _FakeTableNode(f"tbl_{len(tables) + 1}", name)
            tables.append(node)
            return node

        fake_slicer = SimpleNamespace(
            mrmlScene=SimpleNamespace(AddNewNodeByClass=add_new),
            util=SimpleNamespace(getNodesByClass=lambda cls: list(tables) if cls == "vtkMRMLTableNode" else []),
        )

        node = load_stats_table_node(fake_slicer, stats_file)
        self.assertIsNotNone(node)
        self.assertEqual(len(tables), 1)
        self.assertTrue(node.header_flag)
        self.assertTrue(node.locked)
        self.assertEqual(node.attrs[PREFIX + "StatsSourcePath"], str(stats_file.resolve()))
        self.assertEqual(node.units["SurfArea"], "mm^2")
        self.assertNotIn("StructName", node.units)
        self.assertEqual(node.table.GetNumberOfRows(), 3)
        self.assertEqual(node.table.GetNumberOfColumns(), 8)
        self.assertEqual(node.table.GetColumnByName("StructName").GetValue(1), "G_insular_short")
        self.assertEqual(node.table.GetColumnByName("NumVert").GetValue(0), 1250)
        self.assertAlmostEqual(node.table.GetColumnByName("ThickAvg").GetValue(2), 1.985, places=5)

        # Calling a second time reuses the same node instead of creating a duplicate
        node2 = load_stats_table_node(fake_slicer, stats_file)
        self.assertIs(node2, node)
        self.assertEqual(len(tables), 1)

        # Non-tabular file returns None
        raw_file = self.root / "plain.stats"
        raw_file.write_text("no table here\n", encoding="utf-8")
        self.assertIsNone(load_stats_table_node(fake_slicer, raw_file))

    def test_show_stats_dialog_renders_table_filter_copy_and_fallback(self) -> None:
        from shared.surface_morphometry_display import show_stats_dialog

        stats_file = self.root / "lh.aparc.stats"
        stats_file.write_text(SAMPLE_FREESURFER_APARC_STATS, encoding="utf-8")

        fake_qt = _FakeStatsQt()
        info_calls: list[str] = []
        ext_calls: list[Path] = []
        active_tables: list = []
        layout_calls: list = []
        fake_slicer = SimpleNamespace(
            vtkMRMLLayoutNode=SimpleNamespace(SlicerLayoutFourUpTableView=35),
            util=SimpleNamespace(
                mainWindow=lambda: None,
                infoDisplay=lambda msg, **_kw: info_calls.append(msg),
                errorDisplay=mock.MagicMock(),
            ),
            app=SimpleNamespace(
                applicationLogic=lambda: SimpleNamespace(
                    GetSelectionNode=lambda: SimpleNamespace(SetActiveTableID=lambda tid: active_tables.append(tid)),
                    PropagateTableSelection=lambda: None,
                ),
                layoutManager=lambda: SimpleNamespace(
                    tableViewCount=0,
                    setLayout=lambda lid: layout_calls.append(lid),
                ),
            ),
        )
        owner = SimpleNamespace(
            parent=None,
            _open_local_path=lambda p, **_kw: ext_calls.append(Path(p)),
        )

        with mock.patch.dict(sys.modules, {"qt": fake_qt, "slicer": fake_slicer}):
            ok = show_stats_dialog(owner, stats_file, zh=False)
            self.assertTrue(ok)
            self.assertEqual(len(fake_qt.dialogs), 1)
            dlg = fake_qt.dialogs[0]
            self.assertTrue(dlg.visible)
            self.assertEqual(dlg.title, "lh.aparc.stats")
            self.assertEqual(len(dlg.group_boxes), 1)
            self.assertEqual(dlg.group_boxes[0].title, "Summary Measures")
            self.assertEqual(len(dlg.tables), 1)
            table = dlg.tables[0]
            self.assertEqual(table.rowCount(), 3)
            self.assertEqual(table.columnCount(), 10)
            self.assertTrue(table.sorting_enabled)
            self.assertTrue(table.alternating)
            self.assertIn("SurfArea (mm^2)", table.headers)

            # Filter box narrows visible rows
            dlg.line_edits[0].textChanged.emit("insula")
            self.assertEqual(table.hidden_rows, {0, 1})
            dlg.line_edits[0].textChanged.emit("")
            self.assertEqual(table.hidden_rows, set())

            # Copy Table (TSV) populates clipboard
            btn_by_text = {btn.text(): btn for btn in dlg.buttons}
            btn_by_text["Copy Table (TSV)"].clicked.emit()
            self.assertIn("StructName\tNumVert\tSurfArea", fake_qt.clipboard_text)
            self.assertIn("insula\t3201\t2194", fake_qt.clipboard_text)

            # Open in Slicer Table View loads table node & switches layout
            fake_tbl_node = SimpleNamespace(GetID=lambda: "vtkMRMLTableNode1")
            with mock.patch("shared.surface_morphometry_display.load_stats_table_node", return_value=fake_tbl_node):
                btn_by_text["Open in Slicer Table View"].clicked.emit()
            self.assertEqual(active_tables, ["vtkMRMLTableNode1"])
            self.assertEqual(layout_calls, [35])

            # Open Externally delegates to owner._open_local_path
            btn_by_text["Open Externally"].clicked.emit()
            self.assertEqual(ext_calls, [stats_file])

            # Close closes dialog
            btn_by_text["Close"].clicked.emit()
            self.assertTrue(dlg.closed)

            # Non-tabular fallback uses QPlainTextEdit (and zh labels work)
            raw_file = self.root / "raw.stats"
            raw_file.write_text("# No columns here\nplain text line\n", encoding="utf-8")
            self.assertTrue(show_stats_dialog(owner, raw_file, zh=True))
            raw_dlg = fake_qt.dialogs[1]
            self.assertEqual(raw_dlg.tables, [])
            self.assertEqual(len(raw_dlg.plain_texts), 1)
            self.assertTrue(raw_dlg.plain_texts[0].read_only)
            self.assertIn("plain text line", raw_dlg.plain_texts[0].plain_text)
            self.assertIn("关闭", [b.text() for b in raw_dlg.buttons])

            # Missing file shows infoDisplay and returns False
            self.assertFalse(show_stats_dialog(owner, self.root / "missing.stats", zh=False))
            self.assertEqual(len(info_calls), 1)

    def test_clicking_stats_row_in_results_opens_in_slicer_stats_dialog_not_external_path_fallback(self) -> None:
        from shared import ui_run_history
        from shared.result_scene import result_candidates as real_candidates
        from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest
        from test_ui_results_tree import Harness, find_child

        run = RunFolder(self.root)
        run.morph("lh.aparc.a2009s.stats").write_text(SAMPLE_DDSURFER_A2009S_STATS, encoding="utf-8")
        data = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        data.update({"kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed",
                     "ended_at": "2026-11-01T10:00:00Z"})
        run.manifest_path.write_text(json.dumps(data), encoding="utf-8")

        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        opened_externally: list[Path] = []
        h.owner._open_local_path = lambda p, **_kw: opened_externally.append(Path(p))
        h.owner._onboardingLanguage = "zh"

        with mock.patch("shared.result_scene.result_candidates", side_effect=real_candidates):
            h.refresh([parse_run_manifest(run.manifest_path)])
        subject = next(item for item in h.tree.items if item.text(0).startswith("s"))
        anatomy = find_child(subject, "解剖先验")
        post = find_child(anatomy, "表面后处理")
        stats_item = next(post.child(i) for i in range(post.childCount()) if post.child(i).text(0) == "lh.aparc.a2009s.stats")
        self.assertEqual(stats_item.text(1), "↗")
        self.assertIn("在 Slicer 中查看统计表", stats_item.tooltips[0])

        with mock.patch("shared.surface_morphometry_display.show_stats_dialog", return_value=True) as show_mock, \
                mock.patch.dict(sys.modules, {"qt": h.qt, "slicer": h.slicer}):
            ui_run_history._toggle_output(h.owner, stats_item)
        show_mock.assert_called_once_with(h.owner, run.morph("lh.aparc.a2009s.stats"), zh=True)
        self.assertEqual(opened_externally, [])

        # Non-.stats external rows (like .txt) still use _open_local_path
        txt_file = self.root / "qc_note.txt"
        txt_file.write_text("ok\n", encoding="utf-8")
        txt_item = h.qt.QTreeWidgetItem(post, ["qc_note.txt", "↗"])
        txt_item.setData(0, 256 + ui_run_history.ROLE_PATH, str(txt_file))
        txt_item.setData(0, 256 + ui_run_history.ROLE_MANIFEST, str(run.manifest_path))
        with mock.patch("shared.surface_morphometry_display.show_stats_dialog") as show_mock, \
                mock.patch.dict(sys.modules, {"qt": h.qt, "slicer": h.slicer}):
            ui_run_history._toggle_output(h.owner, txt_item)
        show_mock.assert_not_called()
        self.assertEqual(opened_externally, [txt_file])


if __name__ == "__main__":
    unittest.main()
