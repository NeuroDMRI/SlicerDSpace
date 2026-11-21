"""Batch 2 fix round: one test per finding (C1..C3, O1..O11, G-H1..G-H4). The finding ID is in the test name.

Each test failed before the matching change. Items C4 (platform-independent G4 tests) live in
``test_preproc_fsl_gate.py``.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]

from shared.input_preview import (  # noqa: E402
    PREVIEW_ATTRIBUTE,
    PreviewError,
    PreviewSource,
    check_preview_dependencies,
    mean_b0_image,
    preview_cache_path,
    resolve_pending_source,
    resolve_preview_for_ui,
    show_preview_in_scene,
)

try:
    import nibabel as nib
    import numpy as np
except ImportError:  # pragma: no cover - host python without the Slicer stack
    nib = None
    np = None
try:
    import nrrd
except ImportError:  # pragma: no cover
    nrrd = None

AFFINE = np.diag([2.0, 2.5, 3.0, 1.0]) if np is not None else None


def widget_methods(*names: str, **extra: Any):
    """A class holding the named ``SlicerDSpaceWidget`` methods, compiled from ``shared/module_widget.py`` source."""

    path = REPO_ROOT / "shared" / "module_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(found) == set(names), set(names) - set(found)
    namespace = {"Path": Path, "Optional": Optional, "Any": Any, "os": os, "sys": sys, **extra}
    body = [found[name] for name in names]
    for node in body:
        node.decorator_list = []
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return type("Widget", (), {name: namespace[name] for name in names})


def write_nifti_dwi(folder: Path, bvals, name="data.nii.gz", shape=(4, 5, 6)) -> PreviewSource:
    folder.mkdir(parents=True, exist_ok=True)
    volumes = [np.full(shape, 10.0 * (i + 1), dtype=np.float32) for i in range(len(bvals))]
    dwi = folder / name
    nib.save(nib.Nifti1Image(np.stack(volumes, axis=-1), AFFINE), str(dwi))
    bval = folder / "bvals"
    bval.write_text(" ".join(str(b) for b in bvals))
    return PreviewSource("nifti", dwi, bval, None, "sub")


def write_detached_nrrd(folder: Path, fill: float = 10.0, name="dwi"):
    """A detached NRRD (header + raw) with 5 volumes of constant value ``fill * (i + 1)``; returns (nhdr, raw)."""

    folder.mkdir(parents=True, exist_ok=True)
    volumes = [np.full((4, 5, 6), fill * (i + 1), dtype=np.float32) for i in range(5)]
    data = np.stack(volumes, axis=-1).astype(np.int16)
    none = [float("nan")] * 3
    header = {
        "space": "right-anterior-superior",
        "space directions": np.array([[2.0, 0, 0], [0, 2.5, 0], [0, 0, 3.0], none]),
        "kinds": ["domain", "domain", "domain", "list"],
        "space origin": np.array([0.0, 0.0, 0.0]),
        "DWMRI_b-value": "1000",
        "encoding": "raw",
    }
    for index, vector in enumerate(["0 0 0", "1 0 0", "0 0 0", "0 1 0", "0 0 0"]):
        header["DWMRI_gradient_%04d" % index] = vector
    nrrd.write(str(folder / (name + ".nhdr")), data, header=header, detached_header=True)
    return folder / (name + ".nhdr"), folder / (name + ".raw")


# --------------------------------------------------------------------------- C1, C2


@unittest.skipIf(nib is None or nrrd is None, "numpy/nibabel/pynrrd unavailable")
class DetachedNrrdTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cache = self.root / "cache"

    def mean_value(self, source) -> float:
        out = mean_b0_image(source, self.cache)
        return float(np.asarray(nib.load(str(out)).dataobj).mean())

    def test_c1_pending_nhdr_with_a_detached_data_tile_previews_that_file(self):
        nhdr, raw = write_detached_nrrd(self.root / "own", fill=10.0)
        _other_nhdr, other_raw = write_detached_nrrd(self.root / "other", fill=100.0)  # same layout, other values
        plain = resolve_pending_source("nrrd", nhdr, subject_id="s")
        self.assertIsNone(plain.raw)
        self.assertAlmostEqual(self.mean_value(plain), 30.0)  # volumes 0, 2, 4 hold 10, 30, 50
        chosen = resolve_pending_source("nrrd", nhdr, raw=other_raw, subject_id="s")
        self.assertEqual(chosen.raw, other_raw.resolve())  # R3-2: resolved like Prepare Inputs
        self.assertAlmostEqual(self.mean_value(chosen), 300.0)  # the override is read, like import_detached_nrrd
        # The header file itself is left alone.
        self.assertIn(raw.name, nhdr.read_text())

    def test_c1_ui_resolution_passes_the_detached_file_and_a_missing_one_is_reported(self):
        nhdr, _raw = write_detached_nrrd(self.root / "own")
        _n, other_raw = write_detached_nrrd(self.root / "other", fill=100.0)
        pending = {"kind": "nrrd", "source": str(nhdr), "bval": "", "mask": "", "raw": str(other_raw)}
        source = resolve_preview_for_ui(input_dir="", subject_id="s", pending=pending)
        self.assertEqual(source.raw, other_raw.resolve())
        pending["raw"] = str(self.root / "nope.raw")
        with self.assertRaises(PreviewError) as caught:
            resolve_preview_for_ui(input_dir="", subject_id="s", pending=pending)
        self.assertEqual(caught.exception.code, "bad_data")

    def test_c1_widget_pending_source_includes_the_selected_raw_only_for_an_nhdr(self):
        Widget = widget_methods("_input_preview_pending_source", "_input_preview_is_batch")
        texts = {"source": "/x/d.nhdr", "bval": "", "mask": "", "raw": "/x/d.raw"}

        def owner(kind, source):
            me = Widget()
            me._parameterNode = SimpleNamespace(brainMaskBackend="pnl_cnn", batchMode=False)
            me._prepareInputWidgets = {"source_kind": kind}
            me._prepare_inputs_widget_text = lambda key: {**texts, "source": source}.get(key, "")
            me._input_import_selection = lambda: ("now",)
            me._preparedInputSelection = ("before",)
            return me

        self.assertEqual(owner("nrrd", "/x/d.nhdr")._input_preview_pending_source()["raw"], "/x/d.raw")
        self.assertEqual(owner("nrrd", "/x/d.nrrd")._input_preview_pending_source()["raw"], "")
        self.assertEqual(owner("nifti", "/x/d.nii.gz")._input_preview_pending_source()["raw"], "")

    def test_c2_cache_key_includes_the_detached_payload_file(self):
        nhdr, raw = write_detached_nrrd(self.root / "own", fill=10.0)
        source = PreviewSource("nrrd", nhdr, None, None, "s")
        before_key = preview_cache_path(source, self.cache)
        self.assertAlmostEqual(self.mean_value(source), 30.0)
        # Same header bytes and mtime; only the payload changes (new values, new mtime).
        _n, new_raw = write_detached_nrrd(self.root / "new", fill=20.0)
        raw.write_bytes(new_raw.read_bytes())
        stamp = raw.stat().st_mtime_ns + 7_000_000_000
        os.utime(raw, ns=(stamp, stamp))
        self.assertNotEqual(preview_cache_path(source, self.cache), before_key)
        self.assertAlmostEqual(self.mean_value(source), 60.0)  # not the stale cache

    def test_c2_cache_key_follows_the_override_file_not_the_headers_own_data(self):
        nhdr, _raw = write_detached_nrrd(self.root / "own")
        _n, override = write_detached_nrrd(self.root / "other", fill=100.0)
        with_raw = PreviewSource("nrrd", nhdr, None, None, "s", raw=override)
        self.assertNotEqual(preview_cache_path(with_raw, self.cache),
                            preview_cache_path(PreviewSource("nrrd", nhdr, None, None, "s"), self.cache))
        before = preview_cache_path(with_raw, self.cache)
        override.write_bytes(override.read_bytes()[:-2] + b"\x01\x00")
        self.assertNotEqual(preview_cache_path(with_raw, self.cache), before)

    def test_r3_1_a_nrrd_header_with_detached_data_puts_the_payload_in_the_cache_key(self):
        from shared.input_preview import nrrd_payload_files

        nhdr, raw = write_detached_nrrd(self.root / "own", fill=10.0)
        nrrd_path = nhdr.with_suffix(".nrrd")  # a .nrrd file whose header names detached data
        nrrd_path.write_bytes(nhdr.read_bytes().rstrip(b"\n") + b"\n\n")
        self.assertEqual(nrrd_payload_files(nrrd_path), [raw])
        source = PreviewSource("nrrd", nrrd_path, None, None, "s")
        before_key = preview_cache_path(source, self.cache)
        _n, new_raw = write_detached_nrrd(self.root / "new", fill=20.0)
        raw.write_bytes(new_raw.read_bytes())
        stamp = raw.stat().st_mtime_ns + 7_000_000_000
        os.utime(raw, ns=(stamp, stamp))
        self.assertNotEqual(preview_cache_path(source, self.cache), before_key)

    def test_r3_1_an_attached_nrrd_is_read_only_up_to_its_header_end(self):
        from shared.input_preview import nrrd_payload_files

        attached = self.root / "attached.nrrd"
        header = b"NRRD0004\ntype: int16\ndimension: 1\nsizes: 4\nencoding: raw\n\n"
        attached.write_bytes(header + b"\x00\x01data file: not_a_header.raw\n" + b"\x00" * 200_000)
        self.assertEqual(nrrd_payload_files(attached), [])
        plain = self.root / "noheaderdata.nrrd"
        plain.write_bytes(b"NRRD0004\ntype: int16\n\n")
        self.assertEqual(nrrd_payload_files(plain), [])

    def test_r3_2_a_relative_detached_data_tile_is_made_absolute_before_staging(self):
        nhdr, _raw = write_detached_nrrd(self.root / "own", fill=10.0)
        _n, other_raw = write_detached_nrrd(self.root / "other", fill=100.0)
        previous = os.getcwd()
        os.chdir(str(self.root))
        self.addCleanup(os.chdir, previous)
        chosen = resolve_pending_source("nrrd", nhdr, raw=os.path.join("other", other_raw.name), subject_id="s")
        self.assertTrue(chosen.raw.is_absolute())
        self.assertEqual(chosen.raw.resolve(), other_raw.resolve())
        self.assertAlmostEqual(self.mean_value(chosen), 300.0)  # the staged header finds the file

    def test_r3_stage_detached_header_refuses_a_non_ascii_data_path_instead_of_writing_question_marks(self):
        from shared.input_preview import stage_detached_header

        nhdr, _raw = write_detached_nrrd(self.root / "own")
        folder = self.root / "\u6570\u636e"
        folder.mkdir()
        odd_raw = folder / "dwi.raw"
        odd_raw.write_bytes(b"\x00\x00")
        staging = self.root / "stage"
        staging.mkdir()
        with self.assertRaises(PreviewError) as caught:
            stage_detached_header(nhdr, odd_raw, staging)
        self.assertEqual(caught.exception.code, "bad_data")
        self.assertIn("non-ASCII", caught.exception.detail)
        self.assertIn("ASCII-only folder", caught.exception.detail)
        self.assertFalse((staging / nhdr.name).exists())

    def test_o2_a_nrrd_that_announces_too_much_voxel_data_is_refused_before_it_is_read(self):
        from shared import input_preview

        nhdr, _raw = write_detached_nrrd(self.root / "own")
        source = PreviewSource("nrrd", nhdr, None, None, "s")
        with mock.patch.object(input_preview, "MAX_NRRD_BYTES", 100), \
                mock.patch.object(nrrd, "read", side_effect=AssertionError("must not read the data")):
            with self.assertRaises(PreviewError) as caught:
                mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")
        self.assertIn("GiB", caught.exception.text())
        self.assertIn("NIfTI", caught.exception.text())
        self.assertEqual(list(self.cache.glob("*")), [])


# --------------------------------------------------------------------------- O1, O6


@unittest.skipIf(nib is None, "numpy/nibabel unavailable")
class NiftiReadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cache = self.root / "cache"

    def test_o1_nifti_is_loaded_with_keep_file_open(self):
        source = write_nifti_dwi(self.root / "a", [0, 1000, 5, 1000, 0])
        real_load = nib.load
        calls = []

        def spy(path, **kwargs):
            calls.append(kwargs)
            return real_load(path, **kwargs)

        with mock.patch.object(nib, "load", side_effect=spy):
            out = mean_b0_image(source, self.cache)
        self.assertTrue(calls and calls[0].get("keep_file_open") is True, calls)
        np.testing.assert_allclose(np.asarray(nib.load(str(out)).dataobj), 30.0)

    def test_o1_a_small_file_is_read_in_one_pass_and_a_large_one_volume_by_volume_in_ascending_order(self):
        from nibabel.arrayproxy import ArrayProxy
        from shared import input_preview

        source = write_nifti_dwi(self.root / "a", [0, 1000, 5, 1000, 0])
        real_getitem = ArrayProxy.__getitem__
        seen = []

        def spy(proxy, index):
            seen.append(index[-1])
            return real_getitem(proxy, index)

        with mock.patch.object(ArrayProxy, "__getitem__", spy):
            mean_b0_image(source, self.cache / "small")
        self.assertEqual(seen, [])  # one pass: the proxy is converted to an array, never indexed per volume
        with mock.patch.object(ArrayProxy, "__getitem__", spy), mock.patch.object(input_preview, "_ONE_PASS_BYTES", 0):
            out = mean_b0_image(source, self.cache / "large")
        self.assertEqual(seen, [0, 2, 4])  # ascending, so a .nii.gz stream is only read forward
        np.testing.assert_allclose(np.asarray(nib.load(str(out)).dataobj), 30.0)

    def test_o6_a_failed_save_leaves_no_staging_file_and_a_prune_error_does_not_fail_the_preview(self):
        from shared import input_preview

        source = write_nifti_dwi(self.root / "a", [0, 1000, 5, 1000, 0])
        with mock.patch.object(os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(PreviewError):
                mean_b0_image(source, self.cache)
        self.assertEqual([p.name for p in self.cache.glob("tmp_*")], [])
        with mock.patch.object(input_preview, "_prune_cache", side_effect=OSError("vanished")):
            out = mean_b0_image(source, self.cache)
        self.assertTrue(out.is_file())


# --------------------------------------------------------------------------- C3, O10


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def GetID(self):
        return "id-" + self.name


class Composite:
    def __init__(self, foreground="old-foreground"):
        self.background, self.foreground, self.label = None, foreground, None

    def SetBackgroundVolumeID(self, value):
        self.background = value

    def SetForegroundVolumeID(self, value):
        self.foreground = value

    def SetLabelVolumeID(self, value):
        self.label = value

    def SetLabelOpacity(self, value):
        pass


class NewApiSlicer:
    """slicer.util as in Slicer 5.10: loadVolume returns the node, raises on failure, returnNode is deprecated."""

    def __init__(self, fail_label=False, layers_raise=False):
        self.nodes, self.load_kwargs, self.layers = [], [], None
        self.composites = [Composite(), Composite()]
        self.fail_label, self.layers_raise = fail_label, layers_raise
        self.mrmlScene = SimpleNamespace(RemoveNode=lambda _n: None, GetNodeByID=lambda _i: None)
        self.app = SimpleNamespace(layoutManager=lambda: None)

        def classes(name):
            return list(self.composites) if name == "vtkMRMLSliceCompositeNode" else list(self.nodes)

        self.util = SimpleNamespace(
            getNodesByClass=classes, loadVolume=self.load_volume, loadLabelVolume=self.load_label,
            setSliceViewerLayers=self.set_layers, resetSliceViews=lambda: None,
        )

    def load_volume(self, path, properties=None):
        self.load_kwargs.append(("volume", properties))
        node = FakeNode(properties["name"])
        self.nodes.append(node)
        return node

    def load_label(self, path, properties=None):
        if self.fail_label:
            raise RuntimeError("Failed to load node from file")
        node = FakeNode(properties["name"])
        self.nodes.append(node)
        return node

    def set_layers(self, **kwargs):
        if self.layers_raise:
            raise RuntimeError("no layout manager")
        self.layers = kwargs


class SceneTest(unittest.TestCase):
    def test_c3_the_slice_layers_call_clears_the_foreground_explicitly(self):
        fake = NewApiSlicer()
        show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", subject_id="s")
        self.assertIn("foreground", fake.layers)
        self.assertIsNone(fake.layers["foreground"])  # the default 'keep-current' would keep an old foreground

    def test_c3_the_composite_fallback_clears_the_foreground_too(self):
        fake = NewApiSlicer(layers_raise=True)
        show_preview_in_scene(fake, "/tmp/m.nii.gz", None, subject_id="s")
        for composite in fake.composites:
            self.assertEqual(composite.background, "id-s_input_meanb0")
            self.assertIsNone(composite.foreground)

    def test_o10_loaders_are_called_without_the_deprecated_return_node_and_failure_is_an_exception(self):
        fake = NewApiSlicer()
        info = show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", subject_id="s")
        self.assertEqual(info, {"volume_id": "id-s_input_meanb0", "mask_id": "id-s_input_mask"})
        self.assertEqual(fake.nodes[0].GetAttribute(PREVIEW_ATTRIBUTE), "meanb0")
        # A mask that raises still leaves the volume on screen.
        messages = []
        info = show_preview_in_scene(NewApiSlicer(fail_label=True), "/tmp/m.nii.gz", "/tmp/mask.nii.gz",
                                     log=messages.append)
        self.assertIsNone(info["mask_id"])
        self.assertTrue(messages and "mask" in messages[0])
        # A volume that raises becomes a PreviewError.
        broken = NewApiSlicer()
        broken.util.loadVolume = mock.Mock(side_effect=RuntimeError("Failed to load node from file"))
        with self.assertRaises(PreviewError) as caught:
            show_preview_in_scene(broken, "/tmp/m.nii.gz")
        self.assertEqual(caught.exception.code, "bad_data")


# --------------------------------------------------------------------------- O8, O11, O9


class BatchAndDependencyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_o11_missing_nibabel_or_pynrrd_is_reported_with_the_package_name(self):
        nifti = PreviewSource("nifti", Path("a"), None, None, "s")
        nrrd_source = PreviewSource("nrrd", Path("a"), None, None, "s")
        have_all = lambda _name: object()  # noqa: E731
        check_preview_dependencies(nifti, have_all)
        check_preview_dependencies(nrrd_source, have_all)
        no_nrrd = lambda name: None if name == "nrrd" else object()  # noqa: E731
        check_preview_dependencies(nifti, no_nrrd)  # pynrrd is only needed for NRRD input
        with self.assertRaises(PreviewError) as caught:
            check_preview_dependencies(nrrd_source, no_nrrd)
        self.assertEqual(caught.exception.code, "missing_dependency")
        self.assertIn("pynrrd", caught.exception.text())
        self.assertIn("pynrrd", caught.exception.text(zh=True))
        with self.assertRaises(PreviewError) as caught:
            check_preview_dependencies(None, lambda name: None)
        self.assertIn("nibabel", caught.exception.text())

    def _widget(self, source, batch=False):
        Widget = widget_methods(
            "_refresh_input_preview_controls", "_input_preview_key", "_set_input_preview_status",
        )

        class Button:
            enabled, tooltip = True, ""

            def setEnabled(self, value):
                self.enabled = value

            def setToolTip(self, value):
                self.tooltip = value

        class Label:
            text, visible = "", False

            def setText(self, value):
                self.text = value

            def setVisible(self, value):
                self.visible = value

            def setStyleSheet(self, _value):
                pass

        me = Widget()
        me._inputPreviewWidgets = {"button": Button(), "status": Label()}
        me._onboardingLanguage = "en"
        me._parameterNode = object()
        me._current = source
        me._input_preview_is_batch = lambda: batch

        def resolve():
            if isinstance(me._current, Exception):
                raise me._current
            return me._current

        me._resolve_input_preview_source = resolve
        return me

    def test_o11_widget_disables_the_button_with_a_reason_when_a_package_is_missing(self):
        source = PreviewSource("nrrd", Path("/x/a.nrrd"), None, None, "s")
        me = self._widget(source)
        with mock.patch("importlib.util.find_spec", side_effect=lambda name: None if name == "nrrd" else object()):
            me._refresh_input_preview_controls()
        self.assertFalse(me._inputPreviewWidgets["button"].enabled)
        self.assertIn("pynrrd", me._inputPreviewWidgets["button"].tooltip)
        with mock.patch("importlib.util.find_spec", side_effect=lambda name: object()):
            me._refresh_input_preview_controls()
        self.assertTrue(me._inputPreviewWidgets["button"].enabled)

    def test_o9_status_line_is_cleared_when_the_resolved_source_changes(self):
        first = PreviewSource("nifti", Path("/x/a.nii.gz"), None, None, "a")
        second = PreviewSource("nifti", Path("/x/b.nii.gz"), None, None, "b")
        me = self._widget(first)
        with mock.patch("importlib.util.find_spec", return_value=object()):
            me._refresh_input_preview_controls()
            me._set_input_preview_status("Showing the mean b0 image of a.", key=me._input_preview_key(first))
            label = me._inputPreviewWidgets["status"]
            me._refresh_input_preview_controls()  # same source: the line stays
            self.assertEqual(label.text, "Showing the mean b0 image of a.")
            me._current = second
            me._refresh_input_preview_controls()
            self.assertEqual(label.text, "")
            self.assertFalse(label.visible)
            # A running preview keeps its "Building" line.
            me._set_input_preview_status("Building…", key=me._input_preview_key(second))
            me._inputPreviewPending = True
            me._current = first
            me._refresh_input_preview_controls()
            self.assertEqual(label.text, "Building…")


# --------------------------------------------------------------------------- O7


class MachoWalkTest(unittest.TestCase):
    def test_o7_the_file_cap_ends_the_whole_directory_walk(self):
        from shared import macho_min_os

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            for index in range(6):
                folder = bundle / "bin" / ("d%d" % index)
                folder.mkdir(parents=True)
                (folder / "f").write_bytes(b"x")
            (bundle / "lib").mkdir()
            (bundle / "lib" / "g").write_bytes(b"x")
            visited = []
            real_walk = os.walk

            def counting_walk(top, **kwargs):
                for entry in real_walk(top, **kwargs):
                    visited.append(entry[0])
                    yield entry

            with mock.patch.object(macho_min_os, "_MAX_FILES_PER_DIR", 2), \
                    mock.patch.object(macho_min_os.os, "walk", counting_walk):
                macho_min_os.min_macos_for_dir(bundle, "arm64")
        self.assertLessEqual(len(visited), 4, visited)  # bin, two folders, and the one that hits the cap; not lib
        self.assertFalse([path for path in visited if path.endswith("lib")])


# --------------------------------------------------------------------------- O3, O4, G-H1


class FakeBox:
    def __init__(self):
        self.toolTip = ""

    def setToolTip(self, value):
        self.toolTip = value


class PreprocessingTooltipTest(unittest.TestCase):
    def test_o3_a_failing_gate_lookup_counts_as_offered_and_is_logged_once(self):
        from shared import ui_parameter_sync as sync

        box = FakeBox()
        node = SimpleNamespace(targetPreprocessing=True, orchestrationPreprocessingStages="denoise,eddy,topup")
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", side_effect=RuntimeError("boom")), \
                mock.patch.object(sync, "_FSL_LOOKUP_LOGGED", False):
            with self.assertLogs(sync.__name__, level="WARNING") as logs:
                self.assertTrue(sync.apply_preprocessing_availability(box, node=node))
                self.assertTrue(sync.apply_preprocessing_availability(box, node=node))
            self.assertEqual(len(logs.records), 1)
        self.assertEqual(node.orchestrationPreprocessingStages, "denoise,eddy,topup")

    def test_o4_tooltip_follows_the_ui_language_and_the_normal_one_returns_when_offered(self):
        from shared.ui_onboarding import onboarding_text
        from shared.ui_parameter_sync import apply_preprocessing_availability

        box = FakeBox()
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            apply_preprocessing_availability(box, language="en")
            self.assertIn("Eddy and Topup need FSL", box.toolTip)
            apply_preprocessing_availability(box, language="zh")
            self.assertIn("需要 FSL", box.toolTip)
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True):
            apply_preprocessing_availability(box, language="zh")  # the zh reason is replaced by the zh normal tooltip
            self.assertEqual(box.toolTip, onboarding_text("zh", "preprocessing_tooltip"))
            custom = FakeBox()
            custom.toolTip = "a tooltip that never showed the FSL reason"
            apply_preprocessing_availability(custom, language="en")  # left alone
            self.assertEqual(custom.toolTip, "a tooltip that never showed the FSL reason")
            en_box = FakeBox()
            with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
                apply_preprocessing_availability(en_box, language="en")
            apply_preprocessing_availability(en_box, language="en")
            self.assertEqual(en_box.toolTip, onboarding_text("en", "preprocessing_tooltip"))

    def test_o4_sync_passes_the_widget_language_to_the_gate(self):
        source = (REPO_ROOT / "shared" / "ui_parameter_sync.py").read_text(encoding="utf-8")
        self.assertIn('language=getattr(widget, "_onboardingLanguage", "en")', source)

    def _owner(self, language):
        ui = SimpleNamespace(targetPreprocessingCheckBox=FakeBox(), targetMicrostructureCheckBox=FakeBox())
        return SimpleNamespace(
            ui=ui, _onboardingLanguage=language, _algorithmSummaryLabels={}, _algorithmButtons={},
            _parameterNode=SimpleNamespace(
                orchestrationPreprocessingStages="denoise", orchestrationMicrostructureStages="dti",
                orchestrationDkiFitMethod="wls",
            ),
        )

    def test_g_h1_the_language_refresh_keeps_the_fsl_reason_in_the_preprocessing_tooltip(self):
        from shared.ui_compact import refresh_compact_algorithms

        owner = self._owner("en")
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            for language, needle in (("zh", "需要 FSL"), ("en", "Eddy and Topup need FSL"), ("zh", "需要 FSL")):
                refresh_compact_algorithms(owner, language)
                tip = owner.ui.targetPreprocessingCheckBox.toolTip
                self.assertIn(needle, tip, language)
            self.assertNotIn("FSL", owner.ui.targetMicrostructureCheckBox.toolTip)
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True):
            refresh_compact_algorithms(owner, "en")
        self.assertNotIn("need FSL", owner.ui.targetPreprocessingCheckBox.toolTip)


class PreviewRowGuardTest(unittest.TestCase):
    def test_r3_4_a_failing_preview_row_refresh_does_not_break_language_switching(self):
        from shared import ui_compact

        names = ("headlineLabel", "configurationCollapsibleButton", "targetsCollapsibleButton",
                 "parametersCollapsibleButton", "environmentCollapsibleButton", "resultsCollapsibleButton",
                 "runSelectedButton", "logHintLabel", "targetPreprocessingCheckBox", "targetSurfaceCheckBox",
                 "targetSurfacePostprocessCheckBox", "targetParcellationCheckBox", "targetSegmentationCheckBox",
                 "targetMicrostructureCheckBox", "targetTractographyCheckBox", "targetBundlesCheckBox",
                 "targetConnectomeCheckBox", "outputDirectoryLabel", "outputDirectoryPath")

        class Text:
            def setText(self, value):
                self.text = value

            def setToolTip(self, value):
                self.tooltip = value

            def setAccessibleName(self, value):
                self.accessible_name = value

        owner = SimpleNamespace(
            ui=SimpleNamespace(**{name: Text() for name in names}), _advancedSettingsPanel=Text(),
            _advancedSettingsHintLabel=Text(), _onboardingWidgets={"language_combo": Text()},
            _parameterNode=SimpleNamespace(orchestrationSelectStreamlines=1), _onboardingLanguage="en",
            _prepareInputWidgets={"source_kind": "nifti", "drop_hint": Text(), "raw_tile": Text()},
        )
        for key in ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask"):
            owner._prepareInputWidgets.update({key: Text(), key + "_label": Text(), key + "_hint": Text()})
            owner._prepareInputWidgets[key + "_label"].setText(key)
        owner._prepareInputWidgets["raw"].toPlainText = lambda: ""
        with mock.patch.object(ui_compact, "refresh_input_preview_controls", side_effect=RuntimeError("row broke")):
            with self.assertLogs(level="WARNING") as logs:
                ui_compact.apply_compact_language(owner, "zh")
        self.assertIn("row broke", "\n".join(logs.output))
        self.assertEqual(owner.ui.runSelectedButton.text, "\u8fd0\u884c SlicerDSpace")  # later labels still switched

    def test_r3_4_the_preview_row_install_in_the_compact_layout_is_exception_guarded(self):
        tree = ast.parse((REPO_ROOT / "shared" / "ui_compact.py").read_text(encoding="utf-8"))
        layout = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "install_compact_layout")
        guarded = []
        for node in ast.walk(layout):
            if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "Exception" for h in node.handlers
            ):
                for inner in node.body:
                    for call in ast.walk(inner):
                        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "install_input_preview_controls":
                            guarded.append(call)
        self.assertEqual(len(guarded), 1)


# --------------------------------------------------------------------------- O5, G-H2, G-H3, G-H4 (surface runtime)


class SurfaceBlockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bundled = self.root / "bundled-runtimes"
        self.bundled.mkdir()
        (self.root / "runtimes").mkdir()
        env = {"SLICERDSPACE_BUNDLED_RUNTIMES_ROOT": str(self.bundled),
               "SLICERDSPACE_RUNTIME_ROOT": str(self.root / "runtimes")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("FS_LICENSE", "FREESURFER_LICENSE", "SLICERDSPACE_SURFACE_RUNTIME"):
            os.environ.pop(name, None)
        from Testing.Python.test_preproc_fsl_gate import make_surface_bundle

        self.bundle = make_surface_bundle(self.bundled, minos=(15, 0))
        self.message = "The bundled surface runtime needs macOS 15.0 or later (this Mac: 14.2)"

    def mac(self, version="14.2"):
        stack = contextlib_exit_stack()
        for patcher in (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")),
            mock.patch("shared.platform_info.host_machine", return_value="arm64"),
            mock.patch("shared.environment.resolve_freesurfer_home", return_value=None),
            mock.patch("shared.surface_runtime.default_surface_runtime", return_value=self.root / "no-prefix"),
        ):
            stack.enter_context(patcher)
        self.addCleanup(stack.close)

    def test_o5_blocked_mac_still_reports_the_licence_that_is_available(self):
        from shared.surface_runtime import inspect_surface_runtime

        licence = self.root / "license.txt"
        licence.write_text("name\nkey\n")
        self.mac("14.2")
        with mock.patch("shared.surface_license.resolve_surface_license", return_value=licence), \
                mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            record = inspect_surface_runtime()
        self.assertTrue(record["macos_blocked"])
        self.assertEqual(record["notes"], self.message)
        self.assertTrue(record["license_available"])

    def test_g_h2_on_demand_check_treats_a_macos_blocked_bundle_as_not_ready_with_the_reason(self):
        from shared import ui_on_demand_resources as od

        self.mac("14.2")
        self.assertFalse(od._surface_runtime_ready())
        self.mac("15.1")
        self.assertTrue(od._surface_runtime_ready())

    def test_g_h2_an_explicit_saved_runtime_path_is_checked_too(self):
        from shared import ui_on_demand_resources as od

        self.mac("14.2")
        self.assertFalse(od._surface_runtime_ready(str(self.bundle)))
        items = od.detect_missing_task_resources(targets=("surface_postprocess",), bundle_backends=("wmql",),
                                                 backend_paths={"postprocess_runtime": str(self.bundle)})
        self.assertTrue([item for item in items if item["id"] == "surface_postprocess"], items)
        self.mac("15.1")
        self.assertTrue(od._surface_runtime_ready(str(self.bundle)))

    def test_r3_3_a_blocked_explicit_runtime_is_not_ready_even_when_another_runtime_loads(self):
        from Testing.Python.test_preproc_fsl_gate import make_surface_bundle
        from shared import ui_on_demand_resources as od

        # An installed (bundled) runtime that loads on macOS 14.2 ...
        for child in self.bundled.glob("surface-runtime-*"):
            import shutil

            shutil.rmtree(str(child))
        make_surface_bundle(self.bundled, minos=(11, 0))
        # ... and an explicitly selected one that needs macOS 15.
        explicit = make_surface_bundle(self.root / "explicit", minos=(15, 0))
        self.mac("14.2")
        self.assertTrue(od._surface_runtime_ready())  # the fine runtime counts when nothing is selected
        self.assertFalse(od._surface_runtime_ready(str(explicit)))
        items = od.detect_missing_task_resources(targets=("surface_postprocess",), bundle_backends=("wmql",),
                                                 backend_paths={"postprocess_runtime": str(explicit)})
        surface = [item for item in items if item["id"] == "surface_postprocess"]
        self.assertTrue(surface, items)
        self.assertIn("needs macOS 15.0", surface[0].get("reason", ""))
        self.mac("15.1")
        self.assertTrue(od._surface_runtime_ready(str(explicit)))

    def test_g_h2_the_supplement_dialog_shows_the_reason(self):
        from shared import ui_on_demand_resources as od

        item = {"id": "surface_postprocess", "title_zh": "表面", "title_en": "Surface", "size_bytes": 1_000_000,
                "reason": self.message}
        plain = dict(item)
        del plain["reason"]
        self.assertIn(self.message, od._missing_bullets_html([item], zh=False))
        self.assertIn(self.message, od._missing_bullets_html([item], zh=True))
        self.assertNotIn("macOS", od._missing_bullets_html([plain], zh=False))

    def test_g_h2_missing_resource_item_carries_the_macos_reason(self):
        from shared import ui_on_demand_resources as od

        self.mac("14.2")
        items = od.detect_missing_task_resources(targets=("surface_postprocess",), bundle_backends=("wmql",),
                                                 backend_paths={})
        surface = [item for item in items if item["id"] == "surface_postprocess"]
        self.assertTrue(surface, items)
        self.assertEqual(surface[0].get("reason"), self.message)

    def test_g_h3_direct_surface_run_stops_with_the_macos_message_before_launching(self):
        from anatomy.surface import postprocess_release as release

        self.mac("14.2")
        config = SimpleNamespace(runtime=None, freesurfer_home=None, output_root=self.root / "out", mri_dir=None,
                                 mode="whole", atlases=None, force=False, subject_id="s", license_file=None,
                                 sphere_seed=None, python_bin=None)
        with mock.patch("anatomy.surface.backend_bridge.run_backend", side_effect=AssertionError("no launch")):
            with self.assertRaises(RuntimeError) as caught:
                release._run_current_postprocess(config, self.root / "subj", self.root / "native", None)
        self.assertEqual(str(caught.exception), self.message)

    def test_g_h3_a_saved_path_to_the_blocked_bundle_is_checked_too(self):
        from shared.surface_runtime import surface_runtime_block_reason

        self.mac("14.2")
        self.assertEqual(surface_runtime_block_reason("native", self.bundle), self.message)
        self.assertIsNone(surface_runtime_block_reason("freesurfer", self.bundle))
        self.mac("15.1")
        self.assertIsNone(surface_runtime_block_reason("native", self.bundle))

    def test_g_h4_a_saved_runtime_that_this_macos_cannot_load_is_not_applied_nor_rewritten(self):
        path = REPO_ROOT / "shared" / "module_widget.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_read_environment_path_setting")
        func.decorator_list = []
        writes, updates = [], []

        class Settings:
            def value(self, key, default=""):
                return str(self_saved[0])

            def setValue(self, key, value):
                writes.append((key, value))

        self_saved = [str(self.bundle)]
        namespace = {
            "Path": Path, "os": os, "sys": sys,
            "slicer": SimpleNamespace(app=SimpleNamespace(userSettings=lambda: Settings())),
            "load_user_config": lambda: {}, "update_user_config": lambda data: updates.append(data),
        }
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
        me = SimpleNamespace(
            _environment_settings_key=lambda dependency_id: "k/" + dependency_id,
            _is_other_slicer_installation_path=lambda value: False,
            logged=[],
        )
        me.appendLog = me.logged.append
        read = namespace["_read_environment_path_setting"]
        self.mac("14.2")
        self.assertEqual(read(me, "postprocess_runtime"), "")
        self.assertEqual(writes, [])  # the saved setting is left as it is
        self.assertEqual(updates, [])
        self.assertEqual(len(me.logged), 1)
        self.assertIn("needs macOS 15.0", me.logged[0])
        read(me, "postprocess_runtime")
        self.assertEqual(len(me.logged), 1)  # logged once per reason
        self.mac("15.1")
        self.assertEqual(read(me, "postprocess_runtime"), str(self.bundle))


def contextlib_exit_stack():
    import contextlib

    return contextlib.ExitStack()


class PythonSyntaxTest(unittest.TestCase):
    def test_fix_round_modules_stay_python_3_9_clean(self):
        for name in ("shared/input_preview.py", "shared/module_widget.py", "shared/ui_compact.py",
                     "shared/ui_parameter_sync.py", "shared/macho_min_os.py", "shared/surface_runtime.py",
                     "shared/ui_on_demand_resources.py", "anatomy/surface/postprocess_release.py"):
            ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
