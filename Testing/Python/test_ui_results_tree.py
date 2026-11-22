#!/usr/bin/env python3
"""Package D7: the merged Results tree (Subject -> Family -> File), earlier results that stay, collapse and pop-out.

The Results panel is built against a fake ``qt``. Its tree, combo box, layouts and signals are small working fakes,
not mocks, so that expansion, selection, re-parenting and signal order are really exercised.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from shared import ui_run_history
from shared.run_history import RunDiscovery, SUBJECT_MANIFEST_KIND, parse_run_manifest
from shared.ui_onboarding import onboarding_text
from test_b2_fix_round import widget_methods
from test_ui_run_history import fake_qt, object_name

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_ROLE = 256
WINDOW, MINMAX, CLOSE = 1, 2, 4


class Signal:
    def __init__(self) -> None:
        self.callbacks: list = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self.callbacks):
            callback(*args)


class FakeItem:
    def __init__(self, parent, labels) -> None:
        self.parent_obj = parent
        self.labels = list(labels)
        self.children: list[FakeItem] = []
        self._data: dict = {}
        self.expanded = False
        self.tooltip = ""
        self.icons: dict = {}  # column -> icon (Batch 2: the Open folder button is the icon of column 1)
        self.tooltips: dict = {}
        if isinstance(parent, FakeTree):
            parent.items.append(self)
        else:
            parent.children.append(self)

    @property
    def tree(self) -> "FakeTree | None":
        """The tree that holds this item, or None for an item that ``takeChildren`` detached (FD1)."""
        node = self.parent_obj
        while node is not None and not isinstance(node, FakeTree):
            node = node.parent_obj
        return node

    def text(self, col): return self.labels[col] if col < len(self.labels) else ""

    def setText(self, col, text) -> None:
        while len(self.labels) <= col:
            self.labels.append("")
        self.labels[col] = text

    def setData(self, col, role, value) -> None: self._data[(col, role)] = value
    def data(self, col, role): return self._data.get((col, role))
    def setToolTip(self, col, tip) -> None:
        self.tooltips[col] = tip
        if col == 0:
            self.tooltip = tip

    def setIcon(self, col, icon) -> None: self.icons[col] = icon
    def childCount(self): return len(self.children)
    def child(self, i): return self.children[i]
    def takeChildren(self):
        # FD1: like Qt 5.15, the taken items leave the tree (no parent, no view).
        taken, self.children = self.children, []
        for child in taken:
            child.parent_obj = None
        return taken

    def addChild(self, child) -> None:
        child.parent_obj = self
        self.children.append(child)

    def parent(self): return None if isinstance(self.parent_obj, FakeTree) else self.parent_obj
    def isExpanded(self): return self.expanded

    def setExpanded(self, value) -> None:
        value = bool(value)
        if value == self.expanded:
            return
        self.expanded = value  # real Qt also emits these two signals for a programmatic change
        tree = self.tree
        if tree is not None:
            (tree.itemExpanded if value else tree.itemCollapsed).emit(self)


class FakeScrollBar:
    def __init__(self) -> None:
        self.value = 0

    def setValue(self, value) -> None:
        self.value = value


class FakeTree:
    def __init__(self) -> None:
        self.items: list[FakeItem] = []
        self.current: FakeItem | None = None
        self.bar = FakeScrollBar()
        self.max_heights: list = []
        self.itemClicked = Signal()
        self.itemDoubleClicked = Signal()
        self.itemExpanded = Signal()
        self.itemCollapsed = Signal()
        self.customContextMenuRequested = Signal()
        self._header = SimpleNamespace(setStretchLastSection=lambda *_a: None, setSectionResizeMode=lambda *_a: None)

    def __getattr__(self, name):  # PythonQt-style setters that the tests do not care about
        if name.startswith("set"):
            return lambda *_a, **_k: None
        raise AttributeError(name)

    def setMaximumHeight(self, value) -> None: self.max_heights.append(value)
    def header(self): return self._header
    def clear(self) -> None:
        self.items.clear()
        self.current = None
        self.bar.value = 0

    def expandAll(self) -> None:
        raise AssertionError("expandAll must not be called")

    def setCurrentItem(self, item) -> None:
        # FD2: Qt 5.15 scrolls to the new current item and so expands every collapsed ancestor
        # (QAbstractItemView::currentChanged -> scrollTo); each expansion emits itemExpanded.
        ancestor = item.parent() if item is not None else None
        while ancestor is not None:
            if not ancestor.expanded:
                ancestor.setExpanded(True)
            ancestor = ancestor.parent()
        self.current = item

    def currentItem(self):
        # FD1: in Qt 5.15 the current index is gone once its item left the tree (takeChildren, clear).
        node = self.current
        while node is not None and not isinstance(node.parent_obj, FakeTree):
            node = node.parent_obj
        return self.current if node is not None and node.parent_obj is self and node in self.items else None
    def verticalScrollBar(self): return self.bar
    @property
    def topLevelItemCount(self): return len(self.items)
    def topLevelItem(self, i): return self.items[i]


class FakeCombo:
    """PythonQt-style combo box: ``count`` and ``currentIndex`` are properties, signals honour ``blockSignals``."""

    def __init__(self, *_a) -> None:
        self.items: list[dict] = []
        self.currentIndex = 0
        self.enabled = True
        self.blocked = False
        self.currentIndexChanged = Signal()

    def __getattr__(self, name):
        if name.startswith(("set", "hide", "show")):
            return lambda *_a, **_k: None
        raise AttributeError(name)

    @property
    def count(self) -> int: return len(self.items)
    def clear(self) -> None: self.items = []
    def addItem(self, text, data=None) -> None: self.items.append({"text": text, "data": data})
    def insertSeparator(self, index) -> None: self.items.insert(index, {"text": "", "data": None})
    def setItemData(self, index, value, role=0) -> None: pass
    def itemData(self, index): return self.items[index]["data"] if 0 <= index < len(self.items) else None
    def blockSignals(self, flag):
        previous, self.blocked = self.blocked, bool(flag)
        return previous

    def setCurrentIndex(self, index) -> None:
        changed = index != self.currentIndex
        self.currentIndex = index
        if changed and not self.blocked:
            self.currentIndexChanged.emit(index)


class FakeLayout:
    """A layout that really holds its widgets, so a test can see where a widget lives."""

    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.widgets: list = []

    def __getattr__(self, name):
        if name.startswith(("set", "add", "insert")):
            return lambda *_a, **_k: None
        raise AttributeError(name)

    def addWidget(self, widget, *_a) -> None:
        self.widgets.append(widget)

    def insertWidget(self, index, widget, *_a) -> None:
        self.widgets.insert(index, widget)

    def removeWidget(self, widget) -> None:
        if widget in self.widgets:
            self.widgets.remove(widget)

    def indexOf(self, widget) -> int:
        return self.widgets.index(widget) if widget in self.widgets else -1


class FakeSettings:
    """QSettings stand-in; two harnesses that share one store act like a Slicer restart."""

    def __init__(self, store: dict | None = None) -> None:
        self.store = {} if store is None else store

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value) -> None:
        self.store[key] = value


class FakeAction:
    def __init__(self, text) -> None:
        self.text = text
        self.enabled = True
        self.toolTip = ""
        self.slots: list = []

    def connect(self, signal, slot) -> None:
        self.slots.append((signal, slot))

    def trigger(self) -> None:
        for signal, slot in list(self.slots):
            if signal == "triggered()":
                slot()


class FakeMenu:
    """A QMenu that holds real actions; ``exec_`` asks the harness which action the user picks."""

    def __init__(self, harness, *args) -> None:
        self.harness = harness
        self.ctor_args = args
        self.actions: list[FakeAction] = []
        self.slots: list = []

    def addAction(self, text):
        self.actions.append(FakeAction(text))
        return self.actions[-1]

    def addSeparator(self) -> None: pass
    def setToolTipsVisible(self, *_a) -> None: pass
    def clear(self) -> None: self.actions = []

    def connect(self, signal, slot) -> None:
        self.slots.append((signal, slot))

    def open(self) -> list[FakeAction]:
        """What a click on a menu button does: aboutToShow, then the actions are shown."""
        for signal, slot in list(self.slots):
            if signal == "aboutToShow()":
                slot()
        return self.actions

    def exec_(self, *_a):
        return self.harness.pick(self)


class FakeNode:
    """A displayable scene node with the few MRML calls the Results code uses."""

    def __init__(self, name: str, class_name: str, *, shown: bool = True, hide_from_editors: bool = False,
                 attributes: dict | None = None, displays: int = 1,
                 display_class: str = "vtkMRMLModelDisplayNode") -> None:
        self.name, self.class_name = name, class_name
        self.hide_from_editors = hide_from_editors
        self.attributes = dict(attributes or {})
        self.displays = [FakeDisplay(shown, display_class) for _ in range(displays)]

    def GetName(self): return self.name
    def GetHideFromEditors(self): return 1 if self.hide_from_editors else 0
    def GetAttribute(self, key): return self.attributes.get(key)
    def GetNumberOfDisplayNodes(self): return len(self.displays)
    def GetNthDisplayNode(self, index): return self.displays[index]
    def GetDisplayNode(self): return self.displays[0] if self.displays else None
    def GetDisplayVisibility(self): return int(any(display.visibility for display in self.displays))

    def SetDisplayVisibility(self, value) -> None:
        for display in self.displays:
            display.SetVisibility(value)


class FakeDisplay:
    def __init__(self, shown: bool, class_name: str = "vtkMRMLModelDisplayNode") -> None:
        self.visibility = 1 if shown else 0
        self.class_name = class_name
        self.calls: list = []

    def IsA(self, class_name): return class_name == self.class_name

    def GetVisibility(self): return self.visibility

    def SetVisibility(self, value) -> None:
        self.calls.append(value)
        self.visibility = int(value)


class FakeComposite:
    def __init__(self, background="vol1", foreground="vol2", label="lab1") -> None:
        self.layers = {"Background": background, "Foreground": foreground, "Label": label}

    def GetBackgroundVolumeID(self): return self.layers["Background"]
    def GetForegroundVolumeID(self): return self.layers["Foreground"]
    def GetLabelVolumeID(self): return self.layers["Label"]
    def SetBackgroundVolumeID(self, value) -> None: self.layers["Background"] = value
    def SetForegroundVolumeID(self, value) -> None: self.layers["Foreground"] = value
    def SetLabelVolumeID(self, value) -> None: self.layers["Label"] = value


def make_manifest(root: Path, subject: str, run: str, ended_at: str, candidates: list[tuple[str, str]],
                  status: str = "completed"):
    path = root / subject / "outputs" / "00_run_metadata" / f"{run}_orchestration_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": status, "subject_id": subject, "run_name": run,
        "ended_at": ended_at, "tasks": [],
        "report_paths": {"report_html": str(path.parent / f"{run}_report.html")},
        "cands": [{"product": product, "path": file} for product, file in candidates],
    }), encoding="utf-8")
    return parse_run_manifest(path)


class Harness:
    """Install the Results panel on a fake owner and drive it through the real refresh path."""

    def __init__(self, test: unittest.TestCase, *, language: str = "en", settings: FakeSettings | None = None) -> None:
        self.test = test
        self.qt = qt = fake_qt()
        qt.Qt.UserRole = USER_ROLE
        self.tree = FakeTree()
        self.combo_holder: list[FakeCombo] = []
        self.menus: list[FakeMenu] = []
        self.pick = lambda _menu: None  # the action the user picks in a context menu
        self.disk: list = []  # what a rescan outside ``refresh`` finds

        def make_menu(*args):
            self.menus.append(FakeMenu(self, *args))
            return self.menus[-1]

        qt.QMenu.side_effect = make_menu
        qt.QTreeWidget.side_effect = lambda *_a: self.tree
        qt.QTreeWidgetItem = lambda parent, labels: FakeItem(parent, labels)
        qt.QComboBox.side_effect = lambda *_a: self.combo_holder.append(FakeCombo()) or self.combo_holder[-1]
        qt.Qt.Window, qt.Qt.WindowMinMaxButtonsHint, qt.Qt.WindowCloseButtonHint = WINDOW, MINMAX, CLOSE
        self.layouts: list[FakeLayout] = []

        def make_layout(*args):
            layout = FakeLayout(args[0] if args else None)
            self.layouts.append(layout)
            return layout

        qt.QVBoxLayout.side_effect = make_layout
        qt.QHBoxLayout.side_effect = make_layout
        self.main_window = object()
        self.slicer = mock.MagicMock(name="slicer")
        self.slicer.util.mainWindow.return_value = self.main_window
        self.settings = settings if settings is not None else FakeSettings()
        self.slicer.app.userSettings.return_value = self.settings
        self.layout = FakeLayout()
        self.report_paths: list = []
        self.fits = 0
        self.owner = SimpleNamespace(
            ui=SimpleNamespace(resultsLayout=self.layout, resultsCollapsibleButton=mock.MagicMock()),
            _onboardingLanguage=language,
            _parameterNode=SimpleNamespace(inputDirectory="", outputDirectory="", batchRoot=""),
            _on_browse_run_directory_clicked=lambda *a: None,
            _on_open_latest_report_clicked=lambda *a: None,
            _on_open_output_tree_report_clicked=lambda *a: None,
            _open_local_path=lambda *a, **k: None,
            _set_run_status_table_rows=lambda _r: None,
            _set_latest_report_path=lambda p: self.report_paths.append(p),
            _set_latest_output_tree_path=lambda _p: None,
            _set_run_summary_message=lambda *_a: None,
            _run_summary_from_manifest=lambda _m: ("ok", "completed"),
            _queueCompactFit=lambda: setattr(self, "fits", self.fits + 1),
        )
        self.owner._on_run_history_selection_changed = lambda: ui_run_history.on_run_combo_changed(self.owner)
        self.owner._refresh_run_history = lambda *a, **k: ui_run_history.refresh_run_history(self.owner, *a, **k)
        self.patches = [
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": self.slicer}),
            mock.patch("shared.result_scene.result_candidates",
                       side_effect=lambda manifest, **_kw: (manifest.get("cands", []), {})),
            mock.patch("shared.result_scene.owned_nodes", return_value=[]),
            mock.patch.object(ui_run_history, "history_roots", return_value=[]),
            mock.patch.object(ui_run_history, "discover_runs",
                              side_effect=lambda *_a, **_k: RunDiscovery(records=tuple(self.disk))),
        ]
        for patch in self.patches:
            patch.start()
            test.addCleanup(patch.stop)
        ui_run_history.install_results_panel(self.owner)
        self.combo = self.combo_holder[0]

    def refresh(self, records, select_path=None) -> None:
        self.disk = list(records)
        ui_run_history.refresh_run_history(self.owner, select_path=select_path)

    def labels(self) -> list[str]:
        return [item.text(0) for item in self.tree.items]

    def widget(self, name: str):
        return next(w for w in self.qt.created if object_name(w) == name)

    def click(self, name: str) -> None:
        self.widget(name).clicked.connect.call_args[0][0]()


def child_texts(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


def find_child(item, prefix: str):
    return next(item.child(i) for i in range(item.childCount()) if item.child(i).text(0).startswith(prefix))


def file_names(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


class MergedTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def three_runs(self):
        """Three runs of sub1 with three run names; the file lists overlap."""

        old = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [
            ("preprocessing.dwi", "/x/a_dwi.nii.gz"), ("microstructure.fa", "/x/fa.nii.gz")])
        mid = make_manifest(self.root, "sub1", "runB", "2026-11-02T10:00:00Z", [
            ("microstructure.fa", "/x/fa.nii.gz"), ("microstructure.md", "/x/md.nii.gz")])
        new = make_manifest(self.root, "sub1", "runC", "2026-11-03T10:00:00Z", [
            ("microstructure.md", "/x/md.nii.gz"), ("tractography.tck", "/x/tracts.tck")])
        return old, mid, new

    def test_three_runs_of_one_subject_make_one_subject_item_with_each_family_and_file_once(self) -> None:
        old, mid, new = self.three_runs()
        h = Harness(self)
        h.refresh([new, mid, old])  # the order discover_runs returns: newest first
        self.assertEqual(h.tree.topLevelItemCount, 1)
        subject = h.tree.topLevelItem(0)
        self.assertTrue(subject.text(0).startswith("sub1"))
        self.assertIn("completed", subject.text(0))
        families = child_texts(subject)
        self.assertEqual(families, ["Preprocessing (1)", "Microstructure (2)", "Tractography (1)"])
        micro = find_child(subject, "Microstructure")
        self.assertEqual(file_names(micro), ["md.nii.gz", "fa.nii.gz"])  # newest run first, each file once
        # A file belongs to the newest run that lists it; the roles stay valid for the existing readers.
        owner_of = {
            "a_dwi.nii.gz": old, "fa.nii.gz": mid, "md.nii.gz": new, "tracts.tck": new,
        }
        index = {str(new.manifest_path): 0, str(mid.manifest_path): 1, str(old.manifest_path): 2}
        leaves = [leaf for family in (subject.child(i) for i in range(subject.childCount()))
                  for leaf in (family.child(j) for j in range(family.childCount()))]
        self.assertEqual(len(leaves), 4)
        for leaf in leaves:
            record = owner_of[leaf.text(0)]
            self.assertEqual(leaf.data(0, USER_ROLE + 2), str(record.manifest_path), leaf.text(0))
            self.assertEqual(leaf.data(0, USER_ROLE + 3), index[str(record.manifest_path)], leaf.text(0))
        # A family carries the newest contributing run; the subject carries the newest run.
        self.assertEqual(find_child(subject, "Preprocessing").data(0, USER_ROLE + 2), str(old.manifest_path))
        self.assertEqual(micro.data(0, USER_ROLE + 2), str(new.manifest_path))
        self.assertEqual(subject.data(0, USER_ROLE + 2), str(new.manifest_path))
        self.assertEqual(subject.data(0, USER_ROLE + 3), 0)
        for name in ("runA", "runB", "runC"):
            self.assertIn(name, subject.tooltip)
        self.assertLess(subject.tooltip.index("runC"), subject.tooltip.index("runB"))
        self.assertLess(subject.tooltip.index("runB"), subject.tooltip.index("runA"))
        self.assertEqual(set(h.owner._runTreeItems), {0, 1, 2})
        self.assertTrue(all(item is subject for item in h.owner._runTreeItems.values()))

    def test_the_newest_run_is_found_by_time_not_by_list_order(self) -> None:
        old, mid, new = self.three_runs()
        h = Harness(self)
        with mock.patch("shared.result_scene.owned_nodes", return_value=[]):
            ui_run_history._populate_run_tree(h.owner, (old, new, mid), 0)
        subject = h.tree.topLevelItem(0)
        self.assertEqual(subject.data(0, USER_ROLE + 3), 1)  # runC sits at index 1 of this list
        self.assertEqual(subject.data(0, USER_ROLE + 2), str(new.manifest_path))

    def test_a_click_on_a_subject_selects_its_newest_run_and_load_and_report_use_that_run(self) -> None:
        old, mid, new = self.three_runs()
        other = make_manifest(self.root, "sub2", "runZ", "2026-11-09T10:00:00Z", [("microstructure.fa", "/y/fa.nii")])
        h = Harness(self)
        h.refresh([other, new, mid, old])
        self.assertEqual(h.combo.currentIndex, 0)  # sub2's run is selected after the refresh
        self.assertEqual(h.owner._latestResultManifestPath, other.manifest_path)
        sub1 = next(item for item in h.tree.items if item.text(0).startswith("sub1"))
        h.tree.itemClicked.emit(sub1, 0)
        self.assertEqual(h.combo.currentIndex, 1)  # runC, the newest run of sub1
        self.assertEqual(h.owner._latestResultManifestPath, new.manifest_path)
        manifest, error = ui_run_history.validated_selected_manifest(h.owner)
        self.assertEqual((manifest, error), (new.manifest_path, ""))
        self.assertEqual(h.report_paths[-1], str(new.manifest_path.parent / "runC_report.html"))
        # The click did not rebuild the merged subject: it still lists all families.
        self.assertEqual(len(child_texts(sub1)), 3)

    def test_a_refresh_after_selecting_a_run_keeps_the_merged_files_of_every_run(self) -> None:
        old, mid, new = self.three_runs()
        h = Harness(self)
        h.refresh([new, mid, old])
        h.refresh([new, mid, old], select_path=old.manifest_path)  # the user picked an older run
        subject = h.tree.topLevelItem(0)
        self.assertEqual(child_texts(subject), ["Preprocessing (1)", "Microstructure (2)", "Tractography (1)"])
        self.assertEqual(h.owner._latestResultManifestPath, old.manifest_path)

    def test_batch_records_are_not_merged_into_a_subject(self) -> None:
        old, mid, new = self.three_runs()
        batch_path = self.root / "b1" / "00_batch_metadata" / "batch_20261104_100000.json"
        batch_path.parent.mkdir(parents=True)
        batch_path.write_text(json.dumps({
            "kind": "slicerdsurfer_batch_run", "version": 1, "run_name": "batch1", "subjects": [{}, {}],
            "results": [{"returncode": 0}], "ended_at": "2026-11-04T10:00:00Z"}), encoding="utf-8")
        batch = parse_run_manifest(batch_path)
        h = Harness(self)
        h.refresh([batch, new, mid, old])
        self.assertEqual(h.tree.topLevelItemCount, 2)
        batch_item = h.tree.topLevelItem(0)
        self.assertIn("batch (2 subjects)", batch_item.text(0))
        self.assertEqual(batch_item.childCount(), 0)
        self.assertEqual(batch_item.data(0, USER_ROLE + 3), 0)


class OutputMenuTest(unittest.TestCase):
    def test_fd3_edit_tissue_copy_uses_the_run_of_the_right_clicked_file(self) -> None:
        """FD3: in a merged subject a file can belong to an older run than the one in the run selector.
        "Edit tissue copy in Segment Editor" must act on the file's run, not on the selected run."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = make_manifest(root, "sub1", "runA", "2026-11-01T10:00:00Z", [
                ("segmentation.tissue", "/x/tissue_seg.nii.gz")])
            new = make_manifest(root, "sub1", "runB", "2026-11-02T10:00:00Z", [("microstructure.fa", "/x/fa.nii.gz")])
            h = Harness(self)
            h.refresh([new, old])
            self.assertEqual(h.owner._latestResultManifestPath, new.manifest_path)  # the newest run is selected
            subject = h.tree.topLevelItem(0)
            seg_leaf = find_child(subject, "Anatomical").child(0)
            self.assertEqual(seg_leaf.data(0, USER_ROLE + 2), str(old.manifest_path))
            seen = []
            h.owner._manage_result_scene = lambda mode: seen.append(
                (mode, ui_run_history.validated_selected_manifest(h.owner)[0]))
            h.tree.itemAt = lambda _position: seg_leaf
            h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)

            def make_menu(*_a):
                actions: list = []
                menu = mock.MagicMock(name="QMenu")
                menu.addAction.side_effect = lambda text: actions.append(mock.sentinel.__getattr__(f"a{len(actions)}")) \
                    or actions[-1]
                menu.exec_.side_effect = lambda *_a: actions[0]  # the user picks the first action: Edit tissue
                return menu

            h.qt.QMenu.side_effect = make_menu
            ui_run_history._output_menu(h.owner, (5, 5))
            self.assertEqual(seen, [("edit_tissue", old.manifest_path)])
            self.assertEqual(h.combo.currentIndex, 1)  # the run selector follows the file's run
            self.assertEqual(len(child_texts(subject)), 2)  # the selection did not rebuild the merged subject

    def _edit_tissue_with_a_refresh_inside_the_menu(self, rebuilt):
        """FD8 setup: right-click the tissue file of the older run, and while the menu's nested event loop
        runs, a refresh rebuilds the tree from ``rebuilt(old, new)`` and deletes the old C++ item."""

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        old = make_manifest(root, "sub1", "runA", "2026-11-01T10:00:00Z", [
            ("segmentation.tissue", "/x/tissue_seg.nii.gz")])
        new = make_manifest(root, "sub1", "runB", "2026-11-02T10:00:00Z", [("microstructure.fa", "/x/fa.nii.gz")])
        h = Harness(self)
        h.refresh([new, old])
        seg_leaf = find_child(h.tree.topLevelItem(0), "Anatomical").child(0)
        self.assertEqual(seg_leaf.data(0, USER_ROLE + 2), str(old.manifest_path))
        seen = []
        h.owner._manage_result_scene = lambda mode: seen.append(
            (mode, ui_run_history.validated_selected_manifest(h.owner)[0]))
        h.tree.itemAt = lambda _position: seg_leaf
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)

        def deleted(*_a):
            raise RuntimeError("underlying C++ object has been deleted")

        def make_menu(*_a):
            actions: list = []
            menu = mock.MagicMock(name="QMenu")
            menu.addAction.side_effect = lambda text: actions.append(mock.sentinel.__getattr__(f"b{len(actions)}")) \
                or actions[-1]

            def exec_(*_a):
                h.refresh(rebuilt(old, new))  # a run-progress poll or a run end refreshes the tree
                seg_leaf.data = deleted  # tree.clear() deleted the right-clicked item
                return actions[0]  # the user picks "Edit tissue copy in Segment Editor"

            menu.exec_.side_effect = exec_
            return menu

        h.qt.QMenu.side_effect = make_menu
        ui_run_history._output_menu(h.owner, (5, 5))
        return h, old, new, seen

    def test_fd8_a_refresh_while_the_menu_is_open_does_not_touch_the_deleted_item(self) -> None:
        """FD8: menu.exec_ runs a nested event loop; a refresh can delete the item and reorder the records.
        Edit tissue must resolve the file's run by its manifest in the current records."""

        h, old, new, seen = self._edit_tissue_with_a_refresh_inside_the_menu(lambda old, new: [old, new])
        self.assertEqual(h.combo.currentIndex, 0)  # the file's run (runA) sits at index 0 after the reorder
        self.assertEqual(h.combo.itemData(h.combo.currentIndex), old.manifest_path.as_posix())
        self.assertEqual(seen, [("edit_tissue", old.manifest_path)])

    def test_fd8_no_edit_when_the_files_run_is_gone_after_the_refresh(self) -> None:
        """FD8: when no current record has the file's manifest, Edit tissue would act on another run; skip it."""

        h, old, new, seen = self._edit_tissue_with_a_refresh_inside_the_menu(lambda old, new: [new])
        self.assertEqual(seen, [])


class ExpansionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def records(self, *subjects: tuple[str, str]):
        return [make_manifest(self.root, subject, "run1", stamp, [
            ("preprocessing.dwi", f"/x/{subject}_dwi.nii.gz"), ("microstructure.fa", f"/x/{subject}_fa.nii.gz")])
            for subject, stamp in subjects]

    def test_default_state_expands_only_the_newest_subject_down_to_collapsed_families(self) -> None:
        h = Harness(self)
        h.refresh(self.records(("sub3", "2026-11-03T00:00:00Z"), ("sub2", "2026-11-02T00:00:00Z"),
                               ("sub1", "2026-11-01T00:00:00Z")))
        newest, middle, oldest = (h.tree.topLevelItem(i) for i in range(3))
        self.assertTrue(newest.text(0).startswith("sub3"))
        self.assertTrue(newest.expanded)
        self.assertFalse(any(newest.child(i).expanded for i in range(newest.childCount())))
        self.assertGreater(newest.childCount(), 0)
        self.assertFalse(middle.expanded)
        self.assertFalse(oldest.expanded)

    def test_a_refresh_does_not_turn_the_default_into_a_remembered_user_choice(self) -> None:
        h = Harness(self)
        h.refresh(self.records(("sub1", "2026-11-01T00:00:00Z")))
        self.assertTrue(h.tree.topLevelItem(0).expanded)  # sub1 is the newest: expanded by default
        h.refresh(self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z")))
        self.assertTrue(h.tree.topLevelItem(0).text(0).startswith("sub2"))
        self.assertTrue(h.tree.topLevelItem(0).expanded)
        self.assertFalse(h.tree.topLevelItem(1).expanded)  # sub1 was never expanded by the user

    def test_what_the_user_expands_or_collapses_survives_a_refresh(self) -> None:
        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        newest, older = h.tree.topLevelItem(0), h.tree.topLevelItem(1)
        older.setExpanded(True)  # the user opens the older subject
        find_child(older, "Microstructure").setExpanded(True)
        find_child(newest, "Preprocessing").setExpanded(True)
        newest.setExpanded(False)  # and closes the newest one
        h.refresh(records)
        newest, older = h.tree.topLevelItem(0), h.tree.topLevelItem(1)
        self.assertFalse(newest.expanded)
        self.assertTrue(older.expanded)
        self.assertTrue(find_child(older, "Microstructure").expanded)
        self.assertFalse(find_child(older, "Preprocessing").expanded)
        self.assertTrue(find_child(newest, "Preprocessing").expanded)
        self.assertFalse(find_child(newest, "Microstructure").expanded)

    def test_a_new_owner_starts_from_the_default_again(self) -> None:
        first = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        first.refresh(records)
        first.tree.topLevelItem(1).setExpanded(True)
        first.tree.topLevelItem(0).setExpanded(False)
        second = Harness(self)
        second.refresh(records)
        self.assertTrue(second.tree.topLevelItem(0).expanded)
        self.assertFalse(second.tree.topLevelItem(1).expanded)

    def test_a_refresh_keeps_the_current_item_and_the_scroll_position(self) -> None:
        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        family = find_child(h.tree.topLevelItem(0), "Microstructure")
        family.setExpanded(True)
        h.refresh(records)
        family = find_child(h.tree.topLevelItem(0), "Microstructure")
        leaf = family.child(0)
        h.tree.setCurrentItem(leaf)
        h.tree.bar.value = 37
        h.refresh(records)
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.data(0, USER_ROLE), "/x/sub2_fa.nii.gz")
        self.assertEqual(h.tree.bar.value, 37)

    def test_fd1_rebuilding_the_selected_subject_keeps_the_current_file(self) -> None:
        """FD1: ``_show_output_files`` clears the subject's children; Qt then has no current item. Decide from
        the snapshot taken before the clear, not from ``currentItem()`` after it."""

        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        family = find_child(h.tree.topLevelItem(0), "Microstructure")
        family.setExpanded(True)
        h.tree.setCurrentItem(family.child(0))
        manifest = json.loads(records[0].manifest_path.read_text(encoding="utf-8"))
        ui_run_history.show_record(h.owner, records[0], manifest)  # rebuilds sub2, the selected subject
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.data(0, USER_ROLE), "/x/sub2_fa.nii.gz")
        # A current family survives the rebuild too.
        h.tree.setCurrentItem(find_child(h.tree.topLevelItem(0), "Microstructure"))
        ui_run_history.show_record(h.owner, records[0], manifest)
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertTrue(current.text(0).startswith("Microstructure"))

    def test_fd2_a_refresh_does_not_expand_a_family_the_user_collapsed_over_the_current_file(self) -> None:
        """FD2: making a file current under a collapsed family expands the family in Qt. The refresh must make
        the highest collapsed ancestor current instead, so the user's collapse stays."""

        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        family = find_child(h.tree.topLevelItem(0), "Microstructure")
        family.setExpanded(True)
        h.tree.setCurrentItem(family.child(0))
        family.setExpanded(False)  # the user collapses the family; the file stays current
        h.refresh(records)
        family = find_child(h.tree.topLevelItem(0), "Microstructure")
        self.assertFalse(family.expanded)
        self.assertIs(h.tree.currentItem(), family)
        self.assertFalse(h.owner._resultsExpansion["families"][("sub2", "microstructure")])
        h.refresh(records)  # and again: still collapsed
        self.assertFalse(find_child(h.tree.topLevelItem(0), "Microstructure").expanded)

    def test_fd2_a_refresh_does_not_expand_a_subject_the_user_collapsed_over_the_current_file(self) -> None:
        """FD2: under a collapsed subject the subject itself (the highest collapsed ancestor) becomes current."""

        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        subject = h.tree.topLevelItem(0)
        family = find_child(subject, "Microstructure")
        family.setExpanded(True)
        h.tree.setCurrentItem(family.child(0))
        family.setExpanded(False)
        subject.setExpanded(False)  # both the family and the subject are collapsed now
        h.refresh(records)
        subject = h.tree.topLevelItem(0)
        self.assertFalse(subject.expanded)
        self.assertFalse(find_child(subject, "Microstructure").expanded)
        self.assertIs(h.tree.currentItem(), subject)
        self.assertFalse(h.owner._resultsExpansion["subjects"]["sub2"])

    def test_the_current_item_goes_to_the_selected_subject_when_it_no_longer_exists(self) -> None:
        h = Harness(self)
        records = self.records(("sub2", "2026-11-02T00:00:00Z"), ("sub1", "2026-11-01T00:00:00Z"))
        h.refresh(records)
        h.tree.setCurrentItem(h.tree.topLevelItem(1))
        h.refresh(records[:1])  # sub1 is gone
        self.assertTrue(h.tree.currentItem().text(0).startswith("sub2"))


class RunStartTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def start(self, h: Harness, subject: str) -> None:
        h.owner._backgroundRunContext = {"subject_id": subject, "run_name": "r", "targets": []}
        ui_run_history.set_run_state(h.owner, "running")

    def test_reset_result_scene_actions_keeps_the_tree_and_its_item_map(self) -> None:
        Widget = widget_methods("_reset_result_scene_actions")
        tree = mock.MagicMock()
        button = SimpleNamespace(enabled=True)
        review = SimpleNamespace(text="")
        items = {0: object()}
        widget = Widget()
        widget._latestResultManifestPath = Path("old.json")
        widget._runOutputFiles = tree
        widget._runTreeItems = items
        widget._resultSceneButtons = [button]
        widget._reviewStateLabel = review
        widget._reset_result_scene_actions()
        tree.clear.assert_not_called()
        self.assertIs(widget._runTreeItems, items)
        self.assertIsNone(widget._latestResultManifestPath)
        self.assertFalse(button.enabled)
        self.assertIn("not available yet", review.text)

    def test_a_run_start_keeps_earlier_subjects_and_shows_the_running_subject_first_without_a_manifest(self) -> None:
        earlier = make_manifest(self.root, "sub1", "run1", "2026-11-01T00:00:00Z", [("microstructure.fa", "/x/fa")])
        h = Harness(self)
        h.refresh([earlier])
        self.start(h, "sub9")  # no manifest on disk for sub9 yet
        self.assertEqual(len(h.labels()), 2)
        first, second = h.tree.topLevelItem(0), h.tree.topLevelItem(1)
        self.assertTrue(first.text(0).startswith("sub9"))
        self.assertIn("running", first.text(0))
        self.assertTrue(first.expanded)
        self.assertEqual(first.childCount(), 0)
        self.assertTrue(second.text(0).startswith("sub1"))
        self.assertTrue(h.widget("resultsTreeContainer").setVisible.call_args[0][0])

    def test_the_running_subject_is_expanded_again_even_if_the_user_collapsed_it(self) -> None:
        earlier = make_manifest(self.root, "sub1", "run1", "2026-11-01T00:00:00Z", [("microstructure.fa", "/x/fa")])
        h = Harness(self)
        h.refresh([earlier])
        h.tree.topLevelItem(0).setExpanded(False)
        self.start(h, "sub1")
        self.assertTrue(h.tree.topLevelItem(0).expanded)

    def test_the_running_status_is_gone_after_the_run_ends_fails_or_is_cancelled(self) -> None:
        for end_state in ("completed", "failed", "cancelled"):
            with self.subTest(end_state=end_state):
                earlier = make_manifest(self.root, "sub1", "run1", "2026-11-01T00:00:00Z",
                                        [("microstructure.fa", "/x/fa")])
                h = Harness(self)
                h.refresh([earlier])
                self.start(h, "sub1")
                self.assertIn("running", h.tree.topLevelItem(0).text(0))
                ui_run_history.set_run_state(h.owner, end_state)
                self.assertNotIn("running", h.tree.topLevelItem(0).text(0))
                self.assertIn("completed", h.tree.topLevelItem(0).text(0))

    def test_a_job_that_never_started_leaves_no_running_subject(self) -> None:
        h = Harness(self)
        self.start(h, "sub9")
        self.assertEqual(h.labels(), ["sub9 · running"])
        ui_run_history.set_run_state(h.owner, "failed")
        self.assertEqual(h.labels(), [])
        self.assertTrue(h.widget("resultsTreeContainer").setVisible.call_args[0][0] is False)

    def test_a_batch_run_adds_no_running_subject(self) -> None:
        h = Harness(self)
        h.owner._backgroundRunContext = {"subject_id": "batch", "batch_size": 4}
        ui_run_history.set_run_state(h.owner, "running")
        self.assertEqual(h.labels(), [])

    def test_fd6_an_experiment_campaign_marks_no_subject_as_running(self) -> None:
        """FD6: a campaign (run_context with experiment_manifest_path) runs many subjects, not the GUI's subject."""

        earlier = make_manifest(self.root, "sub1", "run1", "2026-11-01T00:00:00Z", [("microstructure.fa", "/x/fa")])
        h = Harness(self)
        h.refresh([earlier])
        h.owner._backgroundRunContext = {
            "subject_id": "sub1", "run_name": "r", "targets": [],
            "experiment_manifest_path": str(self.root / "campaign" / "experiment_manifest.json"),
        }
        ui_run_history.set_run_state(h.owner, "running")
        self.assertEqual(ui_run_history.running_subject_id(h.owner), "")
        self.assertEqual(h.labels(), ["sub1 · completed"])

    def test_an_empty_selection_does_not_wipe_the_running_subject(self) -> None:
        h = Harness(self)
        self.start(h, "sub9")
        h.refresh([])  # a refresh that finds no manifest yet
        self.assertEqual(h.labels(), ["sub9 · running"])


class RunEndTest(unittest.TestCase):
    """FD4: a job of this session that ended (failed, cancelled, killed) while its manifest still says running."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)  # under /var on macOS: a symlink to /private/var

    def checkpoint(self, status: str = "running"):
        return make_manifest(self.root, "sub1", "run1", "2026-11-01T00:00:00Z", [("microstructure.fa", "/x/fa")],
                             status=status)

    def run_and_end(self, h: Harness, record, end_state: str) -> None:
        h.owner._backgroundRunContext = {"subject_id": "sub1", "run_name": "run1", "targets": []}
        h.owner._runProgressManifestPath = record.manifest_path.resolve()  # the planned path, resolved
        ui_run_history.set_run_state(h.owner, "running")
        h.owner._backgroundManifestPath = record.manifest_path  # the worker printed the manifest path
        ui_run_history.set_run_state(h.owner, end_state)

    def test_fd4_an_ended_job_shows_its_end_state_not_running(self) -> None:
        for end_state in ("failed", "cancelled"):
            with self.subTest(end_state=end_state):
                record = self.checkpoint()
                h = Harness(self)
                h.refresh([record])
                self.assertEqual(h.labels(), ["sub1 · running or interrupted"])
                self.run_and_end(h, record, end_state)
                self.assertEqual(h.labels(), [f"sub1 · {end_state}"])  # the rebuild at the run end
                h.refresh([parse_run_manifest(record.manifest_path)])  # the rescan: the disk still says running
                self.assertEqual(h.labels(), [f"sub1 · {end_state}"])
                self.assertIn(f"Status: {end_state}", h.tree.topLevelItem(0).tooltip)
                self.assertIn(end_state, h.combo.items[0]["text"])
                self.assertNotIn("running", h.combo.items[0]["text"])

    def test_fd4_the_ended_state_goes_when_the_manifest_no_longer_says_running(self) -> None:
        record = self.checkpoint()
        h = Harness(self)
        h.refresh([record])
        self.run_and_end(h, record, "failed")
        completed = self.checkpoint(status="completed")  # e.g. a later CLI run finished the same manifest
        h.refresh([completed])
        self.assertEqual(h.labels(), ["sub1 · completed"])
        self.assertEqual(h.owner._resultsEndedRuns, {})
        self.checkpoint(status="running")  # running again on disk: the old session knowledge is gone
        h.refresh([parse_run_manifest(record.manifest_path)])
        self.assertEqual(h.labels(), ["sub1 · running or interrupted"])

    def test_fd4_the_ended_state_goes_when_a_new_job_starts_for_the_manifest(self) -> None:
        record = self.checkpoint()
        h = Harness(self)
        h.refresh([record])
        self.run_and_end(h, record, "failed")
        self.assertEqual(len(h.owner._resultsEndedRuns), 1)
        h.owner._backgroundManifestPath = None  # a new job: the path is reset, the planned path is known
        h.owner._runProgressManifestPath = record.manifest_path
        ui_run_history.set_run_state(h.owner, "running")
        self.assertEqual(h.owner._resultsEndedRuns, {})

    def test_fd4_a_completed_job_and_a_final_manifest_add_nothing(self) -> None:
        record = self.checkpoint(status="failed")
        h = Harness(self)
        h.refresh([record])
        self.run_and_end(h, record, "failed")  # the disk already says failed: nothing to remember
        self.assertEqual(getattr(h.owner, "_resultsEndedRuns", {}), {})
        running = self.checkpoint()
        self.run_and_end(h, running, "completed")
        self.assertEqual(getattr(h.owner, "_resultsEndedRuns", {}), {})

    def test_fd4_a_batch_progress_file_is_not_taken_for_a_manifest(self) -> None:
        progress = self.root / "batch_progress.json"
        progress.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        h = Harness(self)
        h.owner._backgroundRunContext = {"subject_id": "batch", "batch_size": 2}
        h.owner._runProgressManifestPath, h.owner._runProgressIsBatch = progress, True
        ui_run_history.set_run_state(h.owner, "running")
        ui_run_history.set_run_state(h.owner, "failed")
        self.assertEqual(getattr(h.owner, "_resultsEndedRuns", {}), {})


class RememberedFoldersTest(unittest.TestCase):
    def owner(self, **node):
        values = {"inputDirectory": "", "outputDirectory": "", "batchRoot": "", **node}
        return SimpleNamespace(_parameterNode=SimpleNamespace(**values), _manualRunHistoryDirs=())

    def test_remembered_folders_are_scanned_also_when_an_output_folder_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            current, earlier = Path(tmp) / "current", Path(tmp) / "earlier"
            current.mkdir()
            earlier.mkdir()
            with mock.patch.object(ui_run_history, "load_history_directories", return_value=(str(earlier),)):
                roots = ui_run_history.history_roots(self.owner(outputDirectory=str(current)))
            self.assertIn(current, roots)
            self.assertIn(earlier, roots)

    def test_a_remembered_folder_that_no_longer_exists_is_skipped_without_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gone = Path(tmp) / "gone"
            current = Path(tmp) / "current"
            current.mkdir()
            record = make_manifest(current, "sub1", "run1", "2026-11-01T00:00:00Z", [])
            with mock.patch.object(ui_run_history, "load_history_directories", return_value=(str(gone),)):
                roots = ui_run_history.history_roots(self.owner(outputDirectory=str(current)))
            self.assertIn(gone, roots)
            discovery = ui_run_history.discover_runs(roots)
            self.assertEqual([r.manifest_path for r in discovery.records], [record.manifest_path])
            self.assertEqual(discovery.skipped, ())

    def test_a_run_remembers_its_output_folder(self) -> None:
        owner = self.owner(outputDirectory="/data/out")
        owner._backgroundRunContext = {"subject_id": "sub1"}
        with mock.patch.object(ui_run_history, "save_history_directory") as save:
            ui_run_history.set_run_state(owner, "running")
        save.assert_called_once_with("/data/out")

    def test_no_folder_is_remembered_when_no_folder_is_known_or_the_state_is_not_running(self) -> None:
        with mock.patch.object(ui_run_history, "save_history_directory") as save:
            empty = self.owner()
            empty._backgroundRunContext = {"subject_id": "sub1"}
            ui_run_history.set_run_state(empty, "running")
            full = self.owner(outputDirectory="/data/out")
            ui_run_history.set_run_state(full, "preflight")
        save.assert_not_called()

    def test_fd5_with_an_empty_output_folder_field_the_run_remembers_its_subject_root(self) -> None:
        """FD5: with no Output Folder the results go to <subject root>/outputs; remember the subject root, so the
        subject stays in Results after the user picks another input folder."""

        owner = self.owner(inputDirectory="/data/sub1")
        owner._backgroundRunContext = {"subject_id": "sub1", "subject_root": Path("/data/sub1")}
        with mock.patch.object(ui_run_history, "save_history_directory") as save:
            ui_run_history.set_run_state(owner, "running")
        save.assert_called_once_with(str(Path("/data/sub1")))

    def test_fd5_an_experiment_campaign_does_not_remember_the_guis_subject_root(self) -> None:
        """FD5/FD6: a campaign does not run the GUI's subject; ``ui_experiments.submit`` remembers its own folder."""

        owner = self.owner(inputDirectory="/data/sub1")
        owner._backgroundRunContext = {"subject_id": "sub1", "subject_root": Path("/data/sub1"),
                                       "experiment_manifest_path": "/data/campaign/m.json"}
        with mock.patch.object(ui_run_history, "save_history_directory") as save:
            ui_run_history.set_run_state(owner, "running")
        save.assert_not_called()


class CollapseTest(unittest.TestCase):
    def test_results_collapse_button_is_removed_and_body_is_expanded_by_default(self) -> None:
        h = Harness(self)
        body = h.widget("resultsBody")
        self.assertIsNone(h.owner._resultsCollapseButton)
        self.assertNotIn("resultsCollapseButton", {object_name(w) for w in h.qt.created})
        self.assertFalse(h.owner._resultsCollapsed)
        body.setVisible.assert_not_called()

    def test_the_header_row_stays_and_the_group_box_keeps_its_name_and_class(self) -> None:
        h = Harness(self)
        self.assertEqual(h.layout.widgets[0], h.widget("resultsHeaderRow"))
        self.assertIs(h.layout.widgets[-1], h.widget("resultsBody"))
        h.widget("resultsHeaderRow").setVisible.assert_not_called()

    def test_fd7a_header_buttons_have_the_size_and_style_of_the_modules_ellipsis_buttons(self) -> None:
        """FD7(a): 28 x 22 buttons with 16 x 16 icons and the ``algorithmPickerButton`` style rules, right-aligned
        in a header row no higher than the buttons. Only the calls can be tested here."""

        h = Harness(self)
        for name in ("resultsHideAllButton", "resultsMenuButton", "resultsPopOutButton"):
            with self.subTest(button=name):
                button = h.widget(name)
                button.setFixedSize.assert_called_with(28, 22)
                button.setIconSize.assert_called_with(h.qt.QSize.return_value)
                button.setProperty.assert_any_call("algorithmPickerButton", True)
                button.setCursor.assert_called_with(h.qt.Qt.PointingHandCursor)
        h.qt.QSize.assert_any_call(16, 16)
        h.widget("resultsPopOutButton").setToolButtonStyle.assert_called_with(h.qt.Qt.ToolButtonIconOnly)
        header = h.widget("resultsHeaderRow")
        header.setFixedHeight.assert_called_with(22)
        header_layout = next(layout for layout in h.layouts if layout.parent is header)
        self.assertEqual(header_layout.widgets, [h.widget("resultsHideAllButton"), h.widget("resultsMenuButton"),
                                                 h.widget("resultsPopOutButton")])

    def test_a_refresh_keeps_the_body_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = make_manifest(Path(tmp), "sub1", "run1", "2026-11-01T00:00:00Z", [])
            h = Harness(self)
            body = h.widget("resultsBody")
            body.setVisible.reset_mock()
            h.refresh([record])
            body.setVisible.assert_not_called()

    def test_chinese_texts_follow_the_language(self) -> None:
        h = Harness(self, language="zh")
        button = h.widget("resultsPopOutButton")
        self.assertEqual(button.setToolTip.call_args[0][0], onboarding_text("zh", "results_pop_out"))
        self.assertNotEqual(onboarding_text("zh", "results_pop_out"), onboarding_text("en", "results_pop_out"))
        self.assertNotEqual(onboarding_text("zh", "results_in_window"), onboarding_text("en", "results_in_window"))
        self.assertNotEqual(onboarding_text("zh", "results_bring_back"), onboarding_text("en", "results_bring_back"))
        self.assertNotEqual(onboarding_text("zh", "results_window_title"), onboarding_text("en", "results_window_title"))
        self.assertNotEqual(onboarding_text("zh", "results_running"), onboarding_text("en", "results_running"))


class PopOutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(self)
        self.body = self.h.widget("resultsBody")
        self.row = self.h.widget("resultsPoppedOutRow")
        self.pop = self.h.widget("resultsPopOutButton")
        self.body_index = self.h.layout.widgets.index(self.body)

    def dialog(self):
        dialogs = [w for w in self.h.qt.created if w.kind == "QDialog"]
        self.assertEqual(len(dialogs), 1)
        return dialogs[0]

    def finished_slot(self):
        calls = [c for c in self.dialog().connect.call_args_list if c[0][0] == "finished(int)"]
        self.assertEqual(len(calls), 1)
        return calls[0][0][1]

    def window_layout(self):
        dialog = self.dialog()
        for layout in self.h.layouts:
            if layout.parent is dialog:
                return layout
        raise AssertionError("no layout was made for the dialog")

    def open_window(self) -> None:
        self.h.click("resultsPopOutButton")

    def test_a_click_opens_one_non_modal_window_under_the_main_window_and_moves_the_same_tree_into_it(self) -> None:
        self.open_window()
        dialog = self.dialog()
        self.assertIs(dialog.ctor_args[0], self.h.main_window)
        dialog.setModal.assert_called_with(False)
        flags = dialog.setWindowFlags.call_args[0][0]
        self.assertTrue(flags & WINDOW and flags & MINMAX and flags & CLOSE)
        self.assertEqual(dialog.setWindowTitle.call_args[0][0], "Results - SlicerDSpace")
        self.assertEqual(dialog.resize.call_args[0], (900, 640))
        self.assertTrue(dialog.show.called)
        self.assertIn(self.body, self.window_layout().widgets)
        self.assertNotIn(self.body, self.h.layout.widgets)
        self.assertIs(self.h.owner._runOutputFiles, self.h.tree)
        self.assertEqual([w for w in self.h.qt.created if w.kind == "QTreeWidget"], [])  # no second tree
        self.assertEqual(self.h.tree.max_heights[-1], 16777215)  # the 280 px limit is lifted
        self.row.setVisible.assert_called_with(True)
        self.assertIs(self.pop.enabled, False)
        self.assertTrue(self.h.owner._resultsPoppedOut)
        self.assertEqual(self.h.widget("resultsPoppedOutLabel").setText.call_args[0][0],
                         "Results are shown in a separate window.")
        self.assertEqual(self.h.widget("resultsDockBackButton").setText.call_args[0][0], "Bring back")

    def assert_docked(self) -> None:
        self.assertEqual(self.h.layout.widgets.index(self.body), self.body_index)
        self.assertNotIn(self.body, self.window_layout().widgets)
        self.assertEqual(self.h.tree.max_heights[-1], 280)
        self.row.setVisible.assert_called_with(False)
        self.assertIs(self.pop.enabled, True)
        self.assertFalse(self.h.owner._resultsPoppedOut)
        self.assertIs(self.body.setVisible.call_args[0][0], True)
        self.assertIs(self.h.owner._runOutputFiles, self.h.tree)

    def test_closing_the_window_returns_the_body_to_its_place(self) -> None:
        self.open_window()
        self.finished_slot()(0)  # title-bar close or Esc: finished(int)
        self.assert_docked()

    def test_the_bring_back_button_returns_the_body_and_hides_the_window(self) -> None:
        self.open_window()
        self.h.click("resultsDockBackButton")
        self.assert_docked()
        self.assertTrue(self.dialog().hide.called)

    def test_programmatic_collapse_while_popped_out_returns_the_body_without_a_button(self) -> None:
        self.open_window()
        ui_run_history.toggle_results_collapsed(self.h.owner)
        self.assertEqual(self.h.layout.widgets.index(self.body), self.body_index)
        self.assertNotIn(self.body, self.window_layout().widgets)
        self.assertFalse(self.h.owner._resultsPoppedOut)
        self.assertTrue(self.h.owner._resultsCollapsed)
        self.assertIs(self.body.setVisible.call_args[0][0], False)
        self.row.setVisible.assert_called_with(False)
        self.assertIs(self.pop.enabled, True)

    def test_dock_results_body_is_what_cleanup_calls(self) -> None:
        self.open_window()
        ui_run_history.dock_results_body(self.h.owner)
        self.assert_docked()

    def test_docking_twice_is_safe(self) -> None:
        self.open_window()
        ui_run_history.dock_results_body(self.h.owner)
        ui_run_history.dock_results_body(self.h.owner)
        self.finished_slot()(0)
        self.assertEqual(self.h.layout.widgets.count(self.body), 1)
        self.assert_docked()

    def test_dock_without_a_window_does_nothing(self) -> None:
        ui_run_history.dock_results_body(self.h.owner)
        self.assertEqual(self.h.layout.widgets.count(self.body), 1)

    def test_the_window_is_reused_and_keeps_its_last_size(self) -> None:
        self.open_window()
        dialog = self.dialog()
        dialog.width, dialog.height = 1000, 700
        self.finished_slot()(0)
        self.h.click("resultsPopOutButton")
        self.assertIs(self.dialog(), dialog)  # still one dialog object
        self.assertEqual(dialog.resize.call_args[0], (1000, 700))
        self.assertIn(self.body, self.window_layout().widgets)
        dialog.setAttribute.assert_not_called()  # no WA_DeleteOnClose

    def test_the_tree_still_works_after_the_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = make_manifest(Path(tmp), "sub1", "run1", "2026-11-01T00:00:00Z", [("microstructure.fa", "/x/fa")])
            self.open_window()
            self.finished_slot()(0)
            self.h.refresh([record])
            self.assertEqual(self.h.tree.topLevelItemCount, 1)
            self.h.tree.topLevelItem(0).setExpanded(False)
            self.h.refresh([record])
            self.assertFalse(self.h.tree.topLevelItem(0).expanded)

    def test_the_window_gets_the_module_style_and_the_icon_follows_the_theme(self) -> None:
        self.h.owner._uiWidget = SimpleNamespace(styleSheet="QWidget { color: red; }")
        self.open_window()
        self.dialog().setStyleSheet.assert_called_with("QWidget { color: red; }")
        light = ui_run_history._results_icon_svg("popout", dark=False)
        dark = ui_run_history._results_icon_svg("popout", dark=True)
        self.assertNotEqual(light, dark)
        self.pop.setIcon.reset_mock()
        ui_run_history.refresh_results_window(self.h.owner, dark=True)
        self.assertTrue(self.pop.setIcon.called)

    def test_fd7b_the_window_matches_the_module_root_rule_and_palette(self) -> None:
        """FD7(b): the copied stylesheet styles the module root by ``qMRMLWidget#DDSurfer``, which the dialog is
        not; and a window does not inherit the root's palette (``apply_placeholder_palette``). The dialog gets
        an object name in the root rule and a copy of the root palette, in both themes. Only the wiring can be
        tested here; how it draws needs real Slicer."""

        for dark in (False, True):
            with self.subTest(dark=dark):
                h = Harness(self)
                root_rule = "qMRMLWidget#DDSurfer {\n  background-color: %s;\n}\n" % ("#303030" if dark else "#FFFFFF")
                style = root_rule + "QDialog {\n  background: #22272E;\n}\n"
                root_palette = object()
                h.owner._uiWidget = SimpleNamespace(styleSheet=style, palette=lambda: root_palette)
                h.click("resultsPopOutButton")
                dialog = next(w for w in h.qt.created if w.kind == "QDialog")
                dialog.setObjectName.assert_called_with("resultsWindow")
                sheet = dialog.setStyleSheet.call_args[0][0]
                self.assertIn("qMRMLWidget#DDSurfer, QDialog#resultsWindow {", sheet)
                self.assertIn("QDialog {\n  background: #22272E;", sheet)  # the rest of the sheet is unchanged
                h.qt.QPalette.assert_called_with(root_palette)
                dialog.setPalette.assert_called_with(h.qt.QPalette.return_value)
                # A theme change while the window is open restyles it again.
                dialog.setPalette.reset_mock()
                ui_run_history.refresh_results_window(h.owner, dark=dark)
                self.assertTrue(dialog.setPalette.called)

    def test_compact_fit_is_queued_when_the_body_moves(self) -> None:
        before = self.h.fits
        self.open_window()
        mid = self.h.fits
        self.finished_slot()(0)
        self.assertGreater(mid, before)
        self.assertGreater(self.h.fits, mid)


def write_cancelled_companion(main_record, cancelled_at: str):
    """The file a GUI cancel writes next to the main manifest (ui_run_controller): same slot, newer stamp."""

    path = main_record.manifest_path.with_name(
        main_record.manifest_path.name.replace("_orchestration_manifest.json", "_cancelled_orchestration_manifest.json"))
    path.write_text(json.dumps({
        "kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "cancelled", "subject_id": main_record.subject_id,
        "run_name": main_record.run_name, "cancelled_at": cancelled_at, "tasks": [], "cands": []}), encoding="utf-8")
    return parse_run_manifest(path)


class HideAndClearTest(unittest.TestCase):
    """Batch 2, Results: "Remove from Results list" and "Clear". Only the list changes: no file, no scene node."""

    REMOVED_EN = "Removed {count} run(s) from the list; files are kept."
    CLEARED_EN = "Results list cleared. New runs will appear here."

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.settings = FakeSettings()

    def runs(self):
        a1 = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa.nii.gz")])
        b1 = make_manifest(self.root, "sub1", "runB", "2026-11-02T10:00:00Z", [("microstructure.md", "/x/md.nii.gz")])
        a2 = make_manifest(self.root, "sub2", "runA", "2026-11-03T10:00:00Z", [("microstructure.fa", "/y/fa.nii.gz")])
        return a1, b1, a2

    def harness(self, records=None, **kwargs) -> Harness:
        h = Harness(self, settings=self.settings, **kwargs)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        if records is not None:
            h.refresh(records)
        return h

    def subject_item(self, h: Harness, subject: str):
        return next(item for item in h.tree.items if item.text(0).startswith(subject))

    def remove_action(self, menu: FakeMenu) -> FakeAction:
        return next(action for action in menu.actions if action.text == "Remove from Results list")

    def remove_via_menu(self, h: Harness, item) -> FakeMenu:
        h.tree.itemAt = lambda _position: item
        h.pick = self.remove_action
        ui_run_history._output_menu(h.owner, (3, 3))
        return h.menus[-1]

    def header_menu(self, h: Harness) -> dict:
        menu = h.owner._resultsMenu
        return {action.text: action for action in menu.open()}

    def clear(self, h: Harness) -> None:
        self.header_menu(h)["Clear results list"].trigger()

    def start(self, h: Harness, subject: str) -> None:
        h.owner._backgroundRunContext = {"subject_id": subject, "run_name": "r", "targets": []}
        ui_run_history.set_run_state(h.owner, "running")

    def combo_runs(self, h: Harness) -> list:
        return [item["data"] for item in h.combo.items if item["data"]]

    def test_b2_results_remove_a_subject_hides_it_from_tree_combo_and_records(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        family = find_child(self.subject_item(h, "sub1"), "Microstructure")
        menu = self.remove_via_menu(h, family)  # on a family item: acts on the whole subject
        action = self.remove_action(menu)
        self.assertTrue(action.enabled)
        self.assertIn("kept", action.toolTip)
        self.assertEqual(h.labels(), ["sub2 · completed"])
        self.assertEqual(self.combo_runs(h), [a2.manifest_path.as_posix()])
        self.assertEqual(h.owner._runHistoryRecords, (a2,))
        self.assertEqual(h.owner._runTreeRecords, (a2,))
        self.assertEqual(set(h.owner._runTreeItems), {0})
        sub2 = h.tree.topLevelItem(0)
        self.assertEqual(sub2.data(0, USER_ROLE + 3), h.combo.currentIndex)  # combo index == ROLE_RUN
        self.assertEqual(h.owner._latestResultManifestPath, a2.manifest_path)
        h.slicer.util.showStatusMessage.assert_called_with(self.REMOVED_EN.format(count=2), 4000)
        stored = json.loads(self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"])
        self.assertEqual(len(stored), 2)

    def test_b2_results_remove_on_a_file_item_and_on_the_subject_item(self) -> None:
        for target in ("file", "subject"):
            with self.subTest(target=target):
                self.settings = FakeSettings()
                a1, b1, a2 = self.runs()
                h = self.harness([a2, b1, a1])
                subject = self.subject_item(h, "sub1")
                item = find_child(subject, "Microstructure").child(0) if target == "file" else subject
                self.remove_via_menu(h, item)
                self.assertEqual(h.labels(), ["sub2 · completed"])

    def test_b2_results_hidden_runs_stay_hidden_after_a_restart(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        self.remove_via_menu(h, self.subject_item(h, "sub1"))
        restarted = self.harness([a2, b1, a1])  # a new owner with the same QSettings store
        self.assertEqual(restarted.labels(), ["sub2 · completed"])
        self.assertEqual(restarted.owner._runHistoryRecords, (a2,))

    def test_b2_results_a_newer_run_in_the_same_slot_shows_again(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        self.remove_via_menu(h, self.subject_item(h, "sub1"))
        rerun = make_manifest(self.root, "sub1", "runA", "2026-11-05T10:00:00Z", [("microstructure.fa", "/x/fa.nii.gz")])
        self.assertEqual(rerun.manifest_path, a1.manifest_path)  # a re-run overwrites the same manifest path
        h.refresh([rerun, a2, b1])
        self.assertEqual(h.owner._runHistoryRecords, (rerun, a2))  # runB of sub1 stays hidden
        self.assertEqual(h.labels(), ["sub1 · completed", "sub2 · completed"])

    def test_b2_results_a_cancel_companion_with_a_newer_stamp_shows_again(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        self.remove_via_menu(h, self.subject_item(h, "sub1"))
        companion = write_cancelled_companion(a1, "2026-11-04T10:00:00Z")
        h.refresh([companion, a2, b1])
        self.assertEqual(h.owner._runHistoryRecords, (companion, a2))
        self.assertEqual(h.labels(), ["sub1 · cancelled", "sub2 · completed"])

    def test_b2_results_clear_leaves_the_running_subject(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        h.owner._runProgressManifestPath = a1.manifest_path  # the live job writes runA of sub1
        self.start(h, "sub1")
        self.clear(h)
        self.assertEqual(h.labels(), ["sub1 · running"])
        self.assertEqual(h.owner._runHistoryRecords, (b1, a1))
        stored = json.loads(self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"])
        self.assertEqual(list(stored), [ui_run_history.run_record_key(a2)])
        # Remove is disabled on the running subject, with the reason.
        menu = self.remove_via_menu(h, self.subject_item(h, "sub1"))
        action = self.remove_action(menu)
        self.assertFalse(action.enabled)
        self.assertEqual(action.toolTip, onboarding_text("en", "results_remove_running"))
        self.assertEqual(h.labels(), ["sub1 · running"])

    def test_b2_results_after_a_clear_only_new_runs_are_listed(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        self.clear(h)
        self.assertEqual(h.labels(), [])
        h.slicer.util.showStatusMessage.assert_called_with(self.REMOVED_EN.format(count=3), 4000)
        new = make_manifest(self.root, "sub3", "runA", "2026-11-06T10:00:00Z", [("microstructure.fa", "/z/fa.nii.gz")])
        # module_widget refreshes with select_path of the run that just ended.
        h.refresh([new, a2, b1, a1], select_path=new.manifest_path)
        self.assertEqual(h.labels(), ["sub3 · completed"])
        self.assertEqual(h.owner._runHistoryRecords, (new,))
        self.assertEqual(self.combo_runs(h), [new.manifest_path.as_posix()])
        self.assertEqual(h.owner._latestResultManifestPath, new.manifest_path)
        self.assertTrue(h.widget("resultsTreeContainer").setVisible.call_args[0][0])

    def test_b2_results_empty_state_after_a_clear_and_show_hidden_restores_all(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        button = h.widget("resultsShowHiddenButton")
        self.assertIs(button.setVisible.call_args[0][0], False)
        self.clear(h)
        self.assertIs(h.widget("resultsTreeContainer").setVisible.call_args[0][0], False)
        self.assertEqual(h.owner._resultsEmptyLabel.setText.call_args[0][0], self.CLEARED_EN)
        self.assertNotEqual(self.CLEARED_EN, onboarding_text("en", "no_runs_found"))
        self.assertIs(button.setVisible.call_args[0][0], True)
        self.assertEqual(button.setText.call_args[0][0], "Show hidden results (3)")
        h.click("resultsShowHiddenButton")
        self.assertEqual(h.labels(), ["sub2 · completed", "sub1 · completed"])
        self.assertEqual(len(h.owner._runHistoryRecords), 3)
        self.assertEqual(json.loads(self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"]), {})
        self.assertIs(button.setVisible.call_args[0][0], False)

    def test_b2_results_no_runs_at_all_keeps_the_no_runs_text(self) -> None:
        h = self.harness([])
        self.assertEqual(h.owner._resultsEmptyLabel.setText.call_args[0][0], onboarding_text("en", "no_runs_found"))
        self.assertIs(h.widget("resultsShowHiddenButton").setVisible.call_args[0][0], False)

    def test_b2_results_header_menu_entries_and_their_enabled_state(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([])
        entries = self.header_menu(h)
        self.assertEqual(list(entries), ["Clear results list", "Show hidden results (0)"])
        self.assertFalse(entries["Clear results list"].enabled)  # nothing to clear
        self.assertFalse(entries["Show hidden results (0)"].enabled)
        h.refresh([a2, b1, a1])
        entries = self.header_menu(h)  # rebuilt on each open
        self.assertTrue(entries["Clear results list"].enabled)
        self.remove_via_menu(h, self.subject_item(h, "sub1"))
        entries = self.header_menu(h)
        self.assertTrue(entries["Show hidden results (2)"].enabled)
        entries["Show hidden results (2)"].trigger()
        self.assertEqual(len(h.owner._runHistoryRecords), 3)
        button = h.widget("resultsMenuButton")
        button.setMenu.assert_called_with(h.owner._resultsMenu)
        button.setPopupMode.assert_called_with(h.qt.QToolButton.InstantPopup)
        self.assertFalse(object_name(button).startswith("runHistory"))

    def test_b2_results_clear_is_disabled_when_only_the_running_subject_is_listed(self) -> None:
        h = self.harness([])
        self.start(h, "sub9")
        self.assertFalse(self.header_menu(h)["Clear results list"].enabled)

    def test_b2_results_header_menu_and_empty_state_in_chinese(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1], language="zh")
        texts = [action.text for action in h.owner._resultsMenu.open()]
        self.assertEqual(texts, ["清空结果列表", "显示已隐藏的结果（0）"])
        h.owner._resultsMenu.actions[0].trigger()
        self.assertEqual(h.owner._resultsEmptyLabel.setText.call_args[0][0], "结果列表已清空，新的运行结果会显示在这里。")
        self.assertEqual(h.widget("resultsShowHiddenButton").setText.call_args[0][0], "显示已隐藏的结果（3）")
        self.assertNotEqual(onboarding_text("zh", "results_remove"), onboarding_text("en", "results_remove"))
        self.assertEqual(onboarding_text("zh", "results_remove"), "从结果列表中移除")

    def test_b2_results_the_show_hidden_button_works_in_the_pop_out_window_and_after_docking_back(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        h.click("resultsPopOutButton")
        self.clear(h)
        body = h.widget("resultsBody")
        window_layout = next(layout for layout in h.layouts if layout.parent is not None
                             and getattr(layout.parent, "kind", "") == "QDialog")
        self.assertIn(body, window_layout.widgets)  # the empty row and its button live in the body
        h.click("resultsShowHiddenButton")
        self.assertEqual(len(h.owner._runHistoryRecords), 3)
        h.click("resultsDockBackButton")
        self.clear(h)
        h.click("resultsShowHiddenButton")
        self.assertEqual(len(h.owner._runHistoryRecords), 3)

    def test_b2_results_add_a_run_folder_un_hides_what_it_finds(self) -> None:
        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        self.clear(h)
        folder = str(self.root / "sub1")
        h.owner.parent = None
        h.qt.QFileDialog.getExistingDirectory.return_value = folder
        h.disk = [b1, a1]  # what the folder holds; the next rescan finds all three again
        with mock.patch.object(ui_run_history, "save_history_directory"):
            ui_run_history.browse_run_directory(h.owner)
            h.disk = [a2, b1, a1]
            ui_run_history.refresh_run_history(h.owner)
        self.assertEqual(h.owner._runHistoryRecords, (b1, a1))  # sub2 stays hidden
        h.slicer.util.showStatusMessage.assert_called_with(f"Found 2 run(s) under {folder}; selected the newest.", 4000)

    def test_b2_results_a_broken_stored_value_reads_as_empty(self) -> None:
        a1, b1, a2 = self.runs()
        for broken in ("{not json", "[]", "17", ["x"]):
            with self.subTest(value=broken):
                self.settings = FakeSettings({"SlicerDSpace/UserSettings/resultsHiddenRuns": broken})
                h = self.harness([a2, b1, a1])
                self.assertEqual(len(h.owner._runHistoryRecords), 3)
        h.slicer.app.userSettings.side_effect = RuntimeError("no settings")
        self.assertEqual(ui_run_history.load_hidden_runs(), {})

    def test_b2_results_remove_and_clear_delete_no_file_and_touch_no_scene_node(self) -> None:
        a1, b1, a2 = self.runs()
        files = sorted(str(path) for path in self.root.rglob("*") if path.is_file())
        with mock.patch("pathlib.Path.unlink") as unlink, mock.patch("os.remove") as remove, \
                mock.patch("os.unlink") as os_unlink, mock.patch("shutil.rmtree") as rmtree:
            h = self.harness([a2, b1, a1])
            self.remove_via_menu(h, self.subject_item(h, "sub1"))
            self.clear(h)
            h.click("resultsShowHiddenButton")
            self.clear(h)
        for fake in (unlink, remove, os_unlink, rmtree):
            fake.assert_not_called()
        h.slicer.mrmlScene.RemoveNode.assert_not_called()
        self.assertEqual(sorted(str(path) for path in self.root.rglob("*") if path.is_file()), files)

    def test_b2_results_the_context_menu_resolves_its_keys_before_exec(self) -> None:
        """FD8 for Remove: exec_ runs a nested event loop; a refresh there deletes the right-clicked item."""

        a1, b1, a2 = self.runs()
        h = self.harness([a2, b1, a1])
        subject = self.subject_item(h, "sub1")
        family = find_child(subject, "Microstructure")
        h.tree.itemAt = lambda _position: family

        def deleted(*_a):
            raise RuntimeError("underlying C++ object has been deleted")

        def pick(menu):
            h.refresh([a1, b1, a2])  # a refresh reorders the records and rebuilds the tree
            for item in (family, subject):
                item.data = deleted
                item.parent = deleted
            return self.remove_action(menu)

        h.pick = pick
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual(h.owner._runHistoryRecords, (a2,))
        stored = json.loads(self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"])
        self.assertEqual(set(stored), {ui_run_history.run_record_key(a1), ui_run_history.run_record_key(b1)})

    def test_b2_results_file_actions_stay_in_the_context_menu(self) -> None:
        tissue = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z",
                               [("segmentation.tissue", "/x/tissue_seg.nii.gz")])
        h = self.harness([tissue])
        leaf = find_child(h.tree.topLevelItem(0), "Anatomical").child(0)
        h.tree.itemAt = lambda _position: leaf
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual([action.text for action in h.menus[-1].actions],
                         ["Edit tissue copy in Segment Editor", "Open folder", "Remove from Results list"])

    def test_b2_results_menu_icon_has_its_own_glyph(self) -> None:
        for dark in (False, True):
            self.assertNotEqual(ui_run_history._results_icon_svg("list", dark=dark),
                                ui_run_history._results_icon_svg("popout", dark=dark))
        self.assertNotEqual(ui_run_history._results_icon_svg("list", dark=False),
                            ui_run_history._results_icon_svg("list", dark=True))


class B2FixResultsTest(unittest.TestCase):
    """Fix round, lane results: items R1 to R5 (live batch key, running batch protection, language refresh, icons)."""

    STARTED = "2026-11-10T00:00:00Z"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.settings = FakeSettings()

    harness = HideAndClearTest.harness
    subject_item = HideAndClearTest.subject_item
    remove_action = HideAndClearTest.remove_action
    remove_via_menu = HideAndClearTest.remove_via_menu
    header_menu = HideAndClearTest.header_menu
    clear = HideAndClearTest.clear

    def stored(self) -> dict:
        return json.loads(self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"])

    def campaign_record(self, status: str = "completed"):
        path = self.root / "campaign" / "experiments.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": "slicerdsurfer_batch_run", "version": 1, "run_name": "camp", "experiment": True,
            "status": status, "subjects": [{}, {}], "ended_at": "2026-11-01T10:00:00Z"}), encoding="utf-8")
        return parse_run_manifest(path)

    def start_batch(self, h: Harness, **context) -> None:
        h.owner._backgroundRunContext = {"subject_id": "batch", "batch_size": 2, "started_at": self.STARTED, **context}
        ui_run_history.set_run_state(h.owner, "running")

    def test_b2fix_results_R1_a_running_campaign_is_not_cleared_nor_removable(self) -> None:
        # The campaign record is "completed" with an old stamp here, so only the live key can protect it.
        camp = self.campaign_record()
        old = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([camp, old])
        h.owner._backgroundManifestPath = camp.manifest_path
        h.owner._backgroundRunContext = {"subject_id": "sub1", "run_name": "r", "targets": [],
                                         "experiment_manifest_path": str(camp.manifest_path)}
        ui_run_history.set_run_state(h.owner, "running")
        self.assertEqual(ui_run_history.run_record_key(camp).split("\t")[0], "batch")
        self.clear(h)
        self.assertEqual(h.owner._runHistoryRecords, (camp,))
        self.assertNotIn(ui_run_history.run_record_key(camp), self.stored())
        batch_item = next(item for item in h.tree.items if "batch" in item.text(0))
        action = self.remove_action(self.remove_via_menu(h, batch_item))
        self.assertFalse(action.enabled)
        self.assertEqual(h.owner._runHistoryRecords, (camp,))

    def test_b2fix_results_R1_the_filter_shows_a_running_campaign_that_an_older_hide_holds(self) -> None:
        camp = self.campaign_record()
        key = ui_run_history.run_record_key(camp)
        self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"] = json.dumps({key: 9e12})
        h = self.harness()
        h.owner._backgroundManifestPath = camp.manifest_path
        h.owner._backgroundRunContext = {"subject_id": "sub1", "run_name": "r", "targets": [],
                                         "experiment_manifest_path": str(camp.manifest_path)}
        ui_run_history.set_run_state(h.owner, "running")
        h.refresh([camp])
        self.assertEqual(h.owner._runHistoryRecords, (camp,))

    def batch_runs(self):
        old = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        live = make_manifest(self.root, "sub2", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/y/fa")],
                             status="running")  # in progress: its own stamp is old, its status says running
        done = make_manifest(self.root, "sub3", "runA", "2026-11-10T00:05:00Z", [("microstructure.fa", "/z/fa")])
        return old, live, done

    def test_b2fix_results_R2_clear_during_a_gui_batch_keeps_the_runs_of_the_batch(self) -> None:
        old, live, done = self.batch_runs()
        h = self.harness([done, live, old])
        self.start_batch(h)
        self.clear(h)
        self.assertEqual(h.owner._runHistoryRecords, (done, live))  # running by status, finished by stamp
        self.assertEqual(list(self.stored()), [ui_run_history.run_record_key(old)])
        self.assertEqual([label.split(" · ")[0] for label in sorted(h.labels())], ["sub2", "sub3"])

    def test_b2fix_results_R2_remove_during_a_gui_batch_is_refused_for_a_run_of_the_batch(self) -> None:
        old, live, done = self.batch_runs()
        h = self.harness([done, live, old])
        self.start_batch(h)
        for subject in ("sub2", "sub3"):
            action = self.remove_action(self.remove_via_menu(h, self.subject_item(h, subject)))
            self.assertFalse(action.enabled, subject)
        self.assertEqual(h.owner._runHistoryRecords, (done, live, old))
        action = self.remove_action(self.remove_via_menu(h, self.subject_item(h, "sub1")))
        self.assertTrue(action.enabled)  # an older run of the same job is not protected
        self.assertEqual(h.owner._runHistoryRecords, (done, live))

    def test_b2fix_results_R2_the_filter_shows_a_batch_run_that_an_older_hide_holds(self) -> None:
        old, live, done = self.batch_runs()
        hidden = {ui_run_history.run_record_key(record): 9e12 for record in (old, live, done)}
        self.settings.store["SlicerDSpace/UserSettings/resultsHiddenRuns"] = json.dumps(hidden)
        h = self.harness()
        self.start_batch(h)
        h.refresh([done, live, old])
        self.assertEqual(h.owner._runHistoryRecords, (done, live))

    def test_b2fix_results_R2_nothing_is_protected_when_no_job_runs(self) -> None:
        old, live, done = self.batch_runs()
        h = self.harness([done, live, old])
        h.owner._backgroundRunContext = {"subject_id": "batch", "batch_size": 2, "started_at": self.STARTED}
        self.clear(h)
        self.assertEqual(h.owner._runHistoryRecords, ())

    def test_b2fix_results_R4_a_language_switch_keeps_the_count_and_renames_the_row_buttons(self) -> None:
        from shared.ui_onboarding import apply_onboarding_language

        a1 = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([a1])
        self.clear(h)
        button = h.widget("resultsShowHiddenButton")
        self.assertEqual(button.setText.call_args[0][0], "Show hidden results (1)")
        apply_onboarding_language(h.owner, "zh")
        self.assertEqual(button.setText.call_args[0][0], "显示已隐藏的结果（1）")
        apply_onboarding_language(h.owner, "en")
        self.assertEqual(button.setText.call_args[0][0], "Show hidden results (1)")
        h.click("resultsShowHiddenButton")
        subject = self.subject_item(h, "sub1")
        self.assertEqual(subject.tooltips[1], "Open folder")
        calls = h.slicer.util.showStatusMessage.call_count
        with mock.patch.object(ui_run_history, "discover_runs", side_effect=AssertionError("rescan")):
            apply_onboarding_language(h.owner, "zh")
        self.assertIs(self.subject_item(h, "sub1"), subject)  # no rebuild of the tree
        self.assertEqual(subject.tooltips[1], "打开文件夹")
        self.assertEqual(subject.data(1, h.qt.Qt.AccessibleTextRole), "打开文件夹")
        self.assertEqual(h.widget("resultsHideAllButton").setAccessibleName.call_args[0][0], "关闭所有显示")
        self.assertEqual(h.slicer.util.showStatusMessage.call_count, calls)

    def test_b2fix_results_R5_a_theme_change_gives_the_folder_icon_only_to_rows_with_a_folder(self) -> None:
        a1 = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([a1])
        subject = self.subject_item(h, "sub1")
        family_row = FakeItem(h.tree, ["Tractography (1)"])  # the flat fallback tree: a family row at the top level
        subject.icons.clear()
        ui_run_history._refresh_folder_icons(h.owner, True)
        self.assertIsNotNone(subject.icons.get(1))
        self.assertNotIn(1, family_row.icons)


class OpenFolderTest(unittest.TestCase):
    """Batch 2 add-on: the Subject ID row has an "Open folder" button at its right end (column 1, like the eye)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def harness(self, records, **kwargs) -> Harness:
        h = Harness(self, **kwargs)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        h.refresh(records)
        return h

    def subject_item(self, h: Harness, prefix: str):
        return next(item for item in h.tree.items if item.text(0).startswith(prefix))

    def assert_opened(self, h: Harness, folder: Path) -> None:
        h.qt.QUrl.fromLocalFile.assert_called_once_with(str(folder))
        h.qt.QDesktopServices.openUrl.assert_called_once_with(h.qt.QUrl.fromLocalFile.return_value)

    def assert_not_opened(self, h: Harness) -> None:
        h.qt.QDesktopServices.openUrl.assert_not_called()
        h.qt.QUrl.fromLocalFile.assert_not_called()

    def test_b2_open_folder_one_run_the_subject_row_has_the_button_and_a_click_opens_the_folder(self) -> None:
        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([record])
        subject = self.subject_item(h, "sub1")
        self.assertIsNotNone(subject.icons.get(1))  # the button: a code-drawn icon in the second column
        self.assertEqual(subject.tooltips[1], "Open folder")
        self.assertEqual(subject.data(1, h.qt.Qt.AccessibleTextRole), "Open folder")
        self.assertEqual(subject.data(0, USER_ROLE + 5), str(self.root / "sub1"))  # stored on the item
        h.tree.itemClicked.emit(subject, 1)
        self.assert_opened(h, self.root / "sub1")

    def test_b2_open_folder_only_the_subject_row_has_the_button(self) -> None:
        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([record])
        subject = self.subject_item(h, "sub1")
        family = find_child(subject, "Microstructure")
        leaf = family.child(0)
        self.assertNotIn(1, family.icons)
        self.assertIsNone(family.data(0, USER_ROLE + 5))
        self.assertIsNone(leaf.data(0, USER_ROLE + 5))
        h.tree.itemClicked.emit(subject, 0)  # a click on the text only selects
        h.tree.itemClicked.emit(family, 1)
        self.assert_not_opened(h)

    def test_b2_open_folder_two_runs_in_different_folders_the_newest_wins(self) -> None:
        old = make_manifest(self.root / "disk_a", "sub1", "runA", "2026-11-01T10:00:00Z", [])
        new = make_manifest(self.root / "disk_b", "sub1", "runB", "2026-11-03T10:00:00Z", [])
        for records in ([new, old], [old, new]):  # the list order does not decide
            with self.subTest(order=[r.run_name for r in records]):
                h = self.harness(records)
                self.assertEqual(h.tree.topLevelItemCount, 1)
                h.tree.itemClicked.emit(h.tree.topLevelItem(0), 1)
                self.assert_opened(h, self.root / "disk_b" / "sub1")

    def test_b2_open_folder_a_batch_record_opens_its_batch_output_folder(self) -> None:
        batch_path = self.root / "b1" / "00_batch_metadata" / "batch_20261104_100000.json"
        batch_path.parent.mkdir(parents=True)
        batch_path.write_text(json.dumps({
            "kind": "slicerdsurfer_batch_run", "version": 1, "run_name": "batch1", "subjects": [{}, {}],
            "results": [{"returncode": 0}], "ended_at": "2026-11-04T10:00:00Z"}), encoding="utf-8")
        h = self.harness([parse_run_manifest(batch_path)])
        item = h.tree.topLevelItem(0)
        self.assertIsNotNone(item.icons.get(1))
        h.tree.itemClicked.emit(item, 1)
        self.assert_opened(h, self.root / "b1")

    def test_b2_open_folder_a_running_subject_without_a_manifest_opens_its_subject_folder(self) -> None:
        h = self.harness([])
        folder = self.root / "ws" / "sub9"
        (folder / "outputs").mkdir(parents=True)
        h.owner._backgroundRunContext = {"subject_id": "sub9", "run_name": "r", "targets": [],
                                         "subject_root": self.root / "in" / "sub9", "output_root": folder / "outputs"}
        ui_run_history.set_run_state(h.owner, "running")
        item = self.subject_item(h, "sub9")
        self.assertIsNotNone(item.icons.get(1))
        h.tree.itemClicked.emit(item, 1)
        self.assert_opened(h, folder)

    def test_b2_open_folder_a_running_subject_without_a_folder_shows_the_status_message(self) -> None:
        h = self.harness([])
        h.owner._backgroundRunContext = {"subject_id": "sub9", "run_name": "r", "targets": []}
        ui_run_history.set_run_state(h.owner, "running")
        h.tree.itemClicked.emit(self.subject_item(h, "sub9"), 1)
        self.assert_not_opened(h)
        self.assertTrue(h.slicer.util.showStatusMessage.called)

    def test_b2_open_folder_a_missing_folder_gives_a_status_message_and_nothing_else(self) -> None:
        import shutil

        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [])
        h = self.harness([record])
        gone = self.root / "sub1"
        shutil.rmtree(gone)
        with mock.patch("os.remove") as remove, mock.patch("os.mkdir") as mkdir, mock.patch("os.makedirs") as makedirs, \
                mock.patch("shutil.rmtree") as rmtree, mock.patch("shutil.move") as move, \
                mock.patch.object(Path, "mkdir") as path_mkdir, mock.patch.object(Path, "unlink") as unlink:
            h.tree.itemClicked.emit(h.tree.topLevelItem(0), 1)
        self.assert_not_opened(h)
        h.slicer.util.showStatusMessage.assert_called_with(f"The folder does not exist: {gone}", 4000)
        for call in (remove, mkdir, makedirs, rmtree, move, path_mkdir, unlink):
            call.assert_not_called()
        self.assertFalse(gone.exists())  # nothing was created

    def test_b2_open_folder_the_status_message_is_chinese_in_chinese(self) -> None:
        import shutil

        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [])
        h = self.harness([record], language="zh")
        subject = h.tree.topLevelItem(0)
        self.assertEqual(subject.tooltips[1], "打开文件夹")
        self.assertEqual(subject.data(1, h.qt.Qt.AccessibleTextRole), "打开文件夹")
        shutil.rmtree(self.root / "sub1")
        h.tree.itemClicked.emit(subject, 1)
        h.slicer.util.showStatusMessage.assert_called_with(f"文件夹不存在：{self.root / 'sub1'}", 4000)

    def test_b2_open_folder_the_context_menu_has_the_action_and_resolves_the_path_before_exec(self) -> None:
        """FD8: exec_ runs a nested event loop; a refresh there deletes the right-clicked item."""

        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [("microstructure.fa", "/x/fa")])
        h = self.harness([record])
        for kind in ("subject", "family"):
            with self.subTest(target=kind):
                h.qt.reset_mock()
                subject = self.subject_item(h, "sub1")  # fresh items: the last subtest rebuilt the tree
                family = find_child(subject, "Microstructure")
                h.tree.itemAt = lambda _position, item=subject if kind == "subject" else family: item

                def deleted(*_a):
                    raise RuntimeError("underlying C++ object has been deleted")

                def pick(menu):
                    h.refresh([record])
                    for item in (subject, family):
                        item.data = deleted
                        item.parent = deleted
                    return next(action for action in menu.actions if action.text == "Open folder")

                h.pick = pick
                ui_run_history._output_menu(h.owner, (3, 3))
                self.assertIn("Open folder", [action.text for action in h.menus[-1].actions])
                self.assert_opened(h, self.root / "sub1")

    def test_b2_open_folder_the_context_menu_entry_with_a_missing_folder_does_nothing_else(self) -> None:
        import shutil

        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [])
        h = self.harness([record])
        shutil.rmtree(self.root / "sub1")
        h.tree.itemAt = lambda _position: h.tree.topLevelItem(0)
        h.pick = lambda menu: next(action for action in menu.actions if action.text == "Open folder")
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assert_not_opened(h)
        self.assertTrue(h.slicer.util.showStatusMessage.called)

    def test_b2_open_folder_the_button_survives_a_rebuild_and_follows_the_theme(self) -> None:
        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [])
        h = self.harness([record])
        h.refresh([record])  # a rebuild makes new items
        h.tree.itemClicked.emit(h.tree.topLevelItem(0), 1)
        self.assert_opened(h, self.root / "sub1")
        # A theme change redraws the icon of every subject row (the stroke colour differs).
        with mock.patch.object(ui_run_history, "_results_icon",
                               side_effect=lambda _qt, glyph, *, dark: (glyph, dark)) as icon:
            ui_run_history.refresh_results_window(h.owner, dark=True)
            self.assertEqual(h.tree.topLevelItem(0).icons[1], ("folder", True))
            ui_run_history.refresh_results_window(h.owner, dark=False)
            self.assertEqual(h.tree.topLevelItem(0).icons[1], ("folder", False))
        self.assertNotEqual(ui_run_history._results_icon_svg("folder", dark=False),
                            ui_run_history._results_icon_svg("folder", dark=True))
        for other in ("popout", "list"):
            self.assertNotEqual(ui_run_history._results_icon_svg("folder", dark=False),
                                ui_run_history._results_icon_svg(other, dark=False))

    def test_b2_open_folder_works_in_the_pop_out_window_and_the_tree_is_not_touched_by_it(self) -> None:
        record = make_manifest(self.root, "sub1", "runA", "2026-11-01T10:00:00Z", [])
        h = self.harness([record])
        h.click("resultsPopOutButton")
        subject = h.tree.topLevelItem(0)  # the body, with its tree, moves into the window: the same tree object
        h.tree.itemClicked.emit(subject, 1)
        self.assert_opened(h, self.root / "sub1")
        self.assertEqual(h.tree.topLevelItemCount, 1)


class HideAllTest(unittest.TestCase):
    """Batch 2 add-on: one header button that turns off everything shown (volumes, segmentations, models)."""

    def harness(self, **kwargs) -> Harness:
        h = Harness(self, **kwargs)
        self.nodes: dict = {}
        h.slicer.util.getNodesByClass.side_effect = lambda class_name: list(self.nodes.get(class_name, ()))
        return h

    def scene(self):
        self.model = FakeNode("tracts", "vtkMRMLModelNode")
        self.fiber_bundle = FakeNode("dmri_tracts", "vtkMRMLFiberBundleNode", displays=3)
        self.two_displays = FakeNode("surf", "vtkMRMLModelNode", displays=2)
        self.already_hidden = FakeNode("old", "vtkMRMLModelNode", shown=False)
        self.slice_models = [
            FakeNode("Red Volume Slice", "vtkMRMLModelNode", hide_from_editors=True,
                     display_class="vtkMRMLSliceDisplayNode"),
            FakeNode("Yellow Volume Slice", "vtkMRMLModelNode", display_class="vtkMRMLSliceDisplayNode"),
        ]
        self.internal = FakeNode("tool helper", "vtkMRMLModelNode", hide_from_editors=True)
        self.segmentation = FakeNode("seg", "vtkMRMLSegmentationNode", displays=2)
        self.composites = [FakeComposite(), FakeComposite("volX", None, None)]
        self.volume_rendering = [FakeNode("vr1", "x", display_class="vtkMRMLVolumeRenderingDisplayNode"),
                                 FakeNode("vr2", "x", shown=False, display_class="vtkMRMLVolumeRenderingDisplayNode")]
        self.nodes = {
            "vtkMRMLModelNode": [self.model, self.two_displays, self.already_hidden, *self.slice_models, self.internal],
            "vtkMRMLFiberBundleNode": [self.fiber_bundle],
            "vtkMRMLSegmentationNode": [self.segmentation],
            "vtkMRMLSliceCompositeNode": self.composites,
            "vtkMRMLVolumeRenderingDisplayNode": [d.displays[0] for d in self.volume_rendering],
        }

    def test_b2_hide_all_the_header_button_is_icon_only_with_a_tooltip_and_an_accessible_name(self) -> None:
        h = self.harness()
        button = h.widget("resultsHideAllButton")
        self.assertFalse(object_name(button).startswith("runHistory"))
        button.setFixedSize.assert_called_with(28, 22)
        button.setProperty.assert_any_call("algorithmPickerButton", True)
        button.setToolButtonStyle.assert_called_with(h.qt.Qt.ToolButtonIconOnly)
        button.setToolTip.assert_called_with("Hide everything shown")
        button.setAccessibleName.assert_called_with("Hide everything shown")
        self.assertTrue(all(call[0][0] == "" for call in button.setText.call_args_list))  # no text on the button
        self.assertIn((button, "tooltip", "results_hide_all"), h.owner._uiLanguageBindings)
        self.assertIs(h.owner._resultsHideAllButton, button)
        self.assertEqual(button.clicked.connect.call_count, 1)
        header = h.widget("resultsHeaderRow")
        header_layout = next(layout for layout in h.layouts if layout.parent is header)
        self.assertEqual(header_layout.widgets[:2], [button, h.widget("resultsMenuButton")])

    def test_b2_hide_all_the_texts_are_english_and_chinese_and_no_text_is_a_label(self) -> None:
        self.assertEqual(onboarding_text("en", "results_hide_all"), "Hide everything shown")
        self.assertEqual(onboarding_text("zh", "results_hide_all"), "关闭所有显示")
        self.assertEqual(onboarding_text("en", "results_open_folder"), "Open folder")
        self.assertEqual(onboarding_text("zh", "results_open_folder"), "打开文件夹")
        h = self.harness(language="zh")
        button = h.widget("resultsHideAllButton")
        button.setToolTip.assert_called_with("关闭所有显示")
        button.setAccessibleName.assert_called_with("关闭所有显示")

    def test_b2_hide_all_the_icon_has_its_own_glyph_in_both_themes(self) -> None:
        for dark in (False, True):
            glyph = ui_run_history._results_icon_svg("hideall", dark=dark)
            for other in ("popout", "list", "folder"):
                self.assertNotEqual(glyph, ui_run_history._results_icon_svg(other, dark=dark))
        self.assertNotEqual(ui_run_history._results_icon_svg("hideall", dark=False),
                            ui_run_history._results_icon_svg("hideall", dark=True))
        h = self.harness()
        with mock.patch.object(ui_run_history, "_results_icon",
                               side_effect=lambda _qt, glyph, *, dark: (glyph, dark)):
            ui_run_history.refresh_results_window(h.owner, dark=True)
        h.widget("resultsHideAllButton").setIcon.assert_called_with(("hideall", True))

    def test_b2_hide_all_a_click_turns_off_models_segmentations_slice_layers_and_volume_rendering(self) -> None:
        h = self.harness()
        self.scene()
        h.click("resultsHideAllButton")
        self.assertEqual([d.visibility for d in self.model.displays], [0])
        self.assertEqual([d.visibility for d in self.fiber_bundle.displays], [0, 0, 0])
        self.assertEqual([d.visibility for d in self.two_displays.displays], [0, 0])
        self.assertEqual([d.visibility for d in self.segmentation.displays], [0, 0])
        self.assertEqual([d.visibility for d in self.already_hidden.displays], [0])
        self.assertEqual([c.layers for c in self.composites],
                         [{"Background": None, "Foreground": None, "Label": None}] * 2)
        self.assertEqual([n.displays[0].visibility for n in self.volume_rendering], [0, 0])

    def test_b2_hide_all_slice_models_and_internal_nodes_are_left_alone(self) -> None:
        h = self.harness()
        self.scene()
        h.click("resultsHideAllButton")
        for node in (*self.slice_models, self.internal):
            self.assertEqual([d.visibility for d in node.displays], [1], node.name)
            self.assertEqual([d.calls for d in node.displays], [[]], node.name)  # not even touched

    def test_b2_hide_all_every_segmentation_is_hidden_also_one_that_is_hidden_from_editors(self) -> None:
        h = self.harness()
        self.scene()
        extra = FakeNode("seg2", "vtkMRMLSegmentationNode", hide_from_editors=True)
        self.nodes["vtkMRMLSegmentationNode"].append(extra)
        h.click("resultsHideAllButton")
        self.assertEqual([d.visibility for d in extra.displays], [0])

    def test_b2_hide_all_refreshes_the_eye_icons_after_the_hide(self) -> None:
        h = self.harness()
        self.scene()
        seen: list = []
        with mock.patch.object(ui_run_history, "_refresh_output_visibility",
                               side_effect=lambda owner: seen.append((owner, self.model.displays[0].visibility,
                                                                      self.composites[0].layers["Label"]))):
            h.click("resultsHideAllButton")
        self.assertEqual(seen, [(h.owner, 0, None)])  # called once, after the nodes were hidden

    def test_b2_hide_all_removes_no_node_and_touches_no_file(self) -> None:
        h = self.harness()
        self.scene()
        with mock.patch("os.remove") as remove, mock.patch("os.unlink") as unlink, mock.patch("shutil.rmtree") as rmtree, \
                mock.patch("shutil.move") as move, mock.patch.object(Path, "unlink") as path_unlink, \
                mock.patch("builtins.open") as opened:
            h.click("resultsHideAllButton")
        for call in (remove, unlink, rmtree, move, path_unlink, opened):
            call.assert_not_called()
        h.slicer.mrmlScene.RemoveNode.assert_not_called()
        h.slicer.mrmlScene.Clear.assert_not_called()
        h.slicer.util.loadVolume.assert_not_called()
        h.qt.QDesktopServices.openUrl.assert_not_called()
        self.assertEqual(h.slicer.util.errorDisplay.call_count, 0)

    def test_b2_hide_all_an_empty_scene_does_nothing_and_does_not_fail(self) -> None:
        h = self.harness()
        h.click("resultsHideAllButton")
        self.assertEqual(h.slicer.util.errorDisplay.call_count, 0)

    def test_b2_hide_all_one_broken_node_does_not_stop_the_rest(self) -> None:
        h = self.harness()
        self.scene()

        def broken(*_a):
            raise RuntimeError("underlying C++ object has been deleted")

        self.model.GetNumberOfDisplayNodes = broken
        h.click("resultsHideAllButton")
        self.assertEqual([d.visibility for d in self.segmentation.displays], [0, 0])
        self.assertEqual(self.composites[0].layers["Label"], None)

    def test_b2_hide_all_the_header_test_forbids_no_other_button(self) -> None:
        """The panel keeps no runHistory* push button and no refresh button (``test_ui_run_history``)."""

        h = self.harness()
        names = {object_name(widget) for widget in h.qt.created}
        self.assertNotIn("runHistoryRefreshButton", names)
        self.assertEqual([w for w in h.qt.created if w.kind == "QPushButton" and object_name(w).startswith("runHistory")],
                         [])


class ModuleWidgetWiringTest(unittest.TestCase):
    def test_cleanup_docks_the_results_body_and_the_import_exists(self) -> None:
        source = (REPO_ROOT / "shared" / "module_widget.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        cleanup = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "cleanup"
                       and any(isinstance(arg, ast.arg) and arg.arg == "self" for arg in node.args.args))
        calls = [node.func.id for node in ast.walk(cleanup)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("dock_results_body", calls)


if __name__ == "__main__":
    unittest.main()
