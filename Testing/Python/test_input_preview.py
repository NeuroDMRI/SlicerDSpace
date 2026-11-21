"""Tests for the 'View input' preview (item H1): DWI resolution and mean-b0 files.

The numeric tests need numpy, nibabel and pynrrd (PythonSlicer has them; the host python3
may not) and are skipped without them.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared.input_preview import (
    PREVIEW_ATTRIBUTE,
    PreviewError,
    PreviewSource,
    first_batch_source,
    mean_b0_image,
    resolve_pending_source,
    resolve_preview_for_ui,
    resolve_preview_source,
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

REPO_ROOT = Path(__file__).resolve().parents[2]
AFFINE = np.array([[2.0, 0, 0, -10.0], [0, 2.5, 0, 5.0], [0, 0, 3.0, 7.5], [0, 0, 0, 1]]) if np is not None else None
BVALS = [0, 1000, 5, 1000, 0]


def write_dwi(folder: Path, name: str = "data.nii.gz", bvals=BVALS, shape=(4, 5, 6)) -> Path:
    volumes = [np.full(shape, 10.0 * (i + 1), dtype=np.float32) for i in range(len(bvals))]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    nib.save(nib.Nifti1Image(np.stack(volumes, axis=-1), AFFINE), str(path))
    return path


def write_text(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@unittest.skipIf(nib is None, "numpy/nibabel unavailable")
class MeanB0Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cache = self.root / "cache"

    def nifti_source(self, bvals=BVALS) -> PreviewSource:
        subject = self.root / "subject" / "input"
        subject.mkdir(parents=True)
        dwi = write_dwi(subject, bvals=bvals)
        bval = write_text(subject / "bvals", " ".join(str(b) for b in bvals))
        return PreviewSource("nifti", dwi, bval, None, "sub-01")

    def test_h1_nifti_mean_b0_uses_only_low_b_volumes_and_keeps_geometry(self):
        source = self.nifti_source()
        out = mean_b0_image(source, self.cache)
        image = nib.load(str(out))
        self.assertEqual(image.shape, (4, 5, 6))
        # volumes 0, 2, 4 (b <= 50) hold 10, 30 and 50.
        np.testing.assert_allclose(np.asarray(image.dataobj), 30.0)
        np.testing.assert_allclose(image.affine, AFFINE)
        self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
        self.assertEqual(out.parent, self.cache)
        self.assertEqual(sorted(p.name for p in source.dwi.parent.iterdir()), ["bvals", "data.nii.gz"])

    def test_h1_mean_b0_is_reused_and_rebuilt_when_the_dwi_changes(self):
        source = self.nifti_source()
        first = mean_b0_image(source, self.cache)
        self.assertEqual(mean_b0_image(source, self.cache), first)
        write_dwi(source.dwi.parent, bvals=BVALS, shape=(4, 5, 7))
        second = mean_b0_image(source, self.cache)
        self.assertNotEqual(second, first)
        self.assertEqual(nib.load(str(second)).shape, (4, 5, 7))

    def test_h1_nifti_errors_are_preview_errors(self):
        source = self.nifti_source()
        write_text(source.bval, "0 1000")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")
        self.assertIn("2 entries", caught.exception.detail)
        write_text(source.bval, "1000 1000 1000 1000 1000")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "no_b0")
        write_text(source.bval, "0 a 0 0 0")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nifti", source.dwi, None, None, "s"), self.cache)
        self.assertEqual(caught.exception.code, "no_bval")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nifti", self.root / "none.nii.gz", source.bval, None, "s"), self.cache)
        self.assertEqual(caught.exception.code, "missing")

    def test_h1_three_dimensional_dwi_is_rejected(self):
        source = self.nifti_source()
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), AFFINE), str(source.dwi))
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")

    def test_h1_cache_keeps_only_the_newest_previews(self):
        for index in range(8):
            source = self.nifti_source() if index == 0 else source
            source = PreviewSource(source.kind, source.dwi, source.bval, None, f"sub-{index}")
            mean_b0_image(source, self.cache)
        self.assertEqual(len(list(self.cache.glob("preview_*_meanb0.nii.gz"))), 6)

    def write_nrrd(self, axis_last: bool, space: str = "left-posterior-superior") -> Path:
        shape = (4, 5, 6)
        volumes = [np.full(shape, 10.0 * (i + 1), dtype=np.float32) for i in range(5)]
        data = np.stack(volumes, axis=-1 if axis_last else 0).astype(np.int16)
        spatial = [[2.0, 0, 0], [0, 2.5, 0], [0, 0, 3.0]]
        none = [float("nan")] * 3
        directions = spatial + [none] if axis_last else [none] + spatial
        kinds = ["domain"] * 3 + ["list"] if axis_last else ["list"] + ["domain"] * 3
        header = {
            "space": space,
            "space directions": np.array(directions),
            "kinds": kinds,
            "space origin": np.array([10.0, -5.0, 7.5]),
            "DWMRI_b-value": "1000",
        }
        for index, vector in enumerate(["0 0 0", "1 0 0", "0 0 0", "0 1 0", "0 0 0"]):
            header["DWMRI_gradient_%04d" % index] = vector
        path = self.root / ("dwi_last.nrrd" if axis_last else "dwi_first.nrrd")
        nrrd.write(str(path), data, header=header)
        return path

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_mean_b0_from_gradient_header_with_ras_affine(self):
        for axis_last in (True, False):
            path = self.write_nrrd(axis_last)
            out = mean_b0_image(PreviewSource("nrrd", path, None, None, f"nrrd-{axis_last}"), self.cache)
            image = nib.load(str(out))
            self.assertEqual(image.shape, (4, 5, 6))
            np.testing.assert_allclose(np.asarray(image.dataobj), 30.0)
            # LPS -> RAS flips x and y: voxel axes become -2, -2.5, +3 and the origin (-10, 5, 7.5).
            expected = np.array([[-2.0, 0, 0, -10.0], [0, -2.5, 0, 5.0], [0, 0, 3.0, 7.5], [0, 0, 0, 1]])
            np.testing.assert_allclose(image.affine, expected)

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_ras_space_is_not_flipped_and_missing_gradients_fail(self):
        path = self.write_nrrd(True, space="right-anterior-superior")
        out = mean_b0_image(PreviewSource("nrrd", path, None, None, "ras"), self.cache)
        np.testing.assert_allclose(nib.load(str(out)).affine[:3, :3], np.diag([2.0, 2.5, 3.0]))
        data, header = nrrd.read(str(path))
        del header["DWMRI_gradient_0003"]
        broken = self.root / "broken.nrrd"
        nrrd.write(str(broken), data, header=header)
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nrrd", broken, None, None, "broken"), self.cache)
        self.assertEqual(caught.exception.code, "bad_data")


@unittest.skipIf(nib is None, "numpy/nibabel unavailable")
class ResolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

    def canonical_subject(self, parent: Path, name: str, mask: bool = False) -> Path:
        subject = parent / name
        write_dwi(write_text(subject / "input" / "keep").parent)
        for item in ("bvals", "bvecs"):
            write_text(subject / "input" / item)
        if mask:
            write_text(subject / "input" / "nodif_brain_mask.nii.gz")
        return subject

    def test_h1_subject_root_uses_the_validated_canonical_files(self):
        subject = self.canonical_subject(self.root, "sub-01", mask=True)
        source = resolve_preview_for_ui(batch_mode=False, input_dir=str(subject), subject_id="sub-01")
        self.assertEqual(source.kind, "nifti")
        self.assertEqual(source.dwi, subject / "input" / "data.nii.gz")
        self.assertEqual(source.bval, subject / "input" / "bvals")
        self.assertEqual(source.mask, subject / "input" / "nodif_brain_mask.nii.gz")
        # A parent folder plus subject id resolves to the same subject.
        again = resolve_preview_for_ui(batch_mode=False, input_dir=str(self.root), subject_id="sub-01")
        self.assertEqual(again.dwi, source.dwi)

    def test_h1_generated_brain_mask_is_not_a_user_mask(self):
        subject = self.canonical_subject(self.root, "sub-01", mask=True)
        write_text(subject / "input" / "nodif_brain_mask.manifest.json", "{}")
        source = resolve_preview_for_ui(batch_mode=False, input_dir=str(subject), subject_id="sub-01")
        self.assertIsNone(source.mask)

    def test_h1_raw_nifti_file_with_gradients_is_previewed_in_place(self):
        folder = self.root / "raw"
        dwi = write_dwi(folder, "scan.nii.gz")
        write_text(folder / "scan.bval", "0 1000 5 1000 0")
        write_text(folder / "scan.bvec")
        write_text(folder / "scan_mask.nii.gz")
        for target in (folder, dwi):
            source = resolve_preview_for_ui(batch_mode=False, input_dir=str(target), subject_id="raw")
            self.assertEqual((source.kind, source.dwi.name, source.bval.name, source.mask.name),
                             ("nifti", "scan.nii.gz", "scan.bval", "scan_mask.nii.gz"))

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_folder_is_detected_as_nrrd(self):
        folder = self.root / "n"
        folder.mkdir()
        path = folder / "dwi.nrrd"
        nrrd.write(str(path), np.zeros((2, 2, 2, 1), dtype=np.int16))
        source = resolve_preview_for_ui(batch_mode=False, input_dir=str(folder), subject_id="n")
        self.assertEqual((source.kind, source.dwi, source.bval), ("nrrd", path.resolve(), None))

    def test_h1_dicom_and_other_layouts_are_not_previewable(self):
        for mode, code in (("dicom", "dicom_unconverted"), ("legacy_registered", "no_dwi"), ("missing", "missing")):
            with self.assertRaises(PreviewError) as caught:
                resolve_preview_source({"mode": mode, "subject_root": None}, self.root, "s")
            self.assertEqual(caught.exception.code, code)
        self.assertEqual(PreviewError("dicom_unconverted").text(), "Convert inputs first (Prepare Inputs)")
        self.assertIn("准备输入", PreviewError("dicom_unconverted").text(zh=True))

    def test_h1_empty_or_missing_input_reports_no_input(self):
        for value, code in (("", "no_input"), (str(self.root / "absent"), "missing")):
            with self.assertRaises(PreviewError) as caught:
                resolve_preview_for_ui(batch_mode=False, input_dir=value, subject_id="s")
            self.assertEqual(caught.exception.code, code)

    def test_h1_batch_uses_the_first_subject_in_runner_order(self):
        batch = self.root / "batch"
        (batch / "a-empty").mkdir(parents=True)
        write_text(batch / "notes.txt")
        self.canonical_subject(batch, "b-sub")
        self.canonical_subject(batch, "c-sub")
        source = first_batch_source(batch)
        self.assertEqual(source.subject_id, "b-sub")
        self.assertEqual(source.dwi, (batch / "b-sub" / "input" / "data.nii.gz").resolve())
        via_ui = resolve_preview_for_ui(batch_mode=True, batch_root=str(batch), input_dir="ignored")
        self.assertEqual(via_ui.subject_id, "b-sub")

    def test_h1_batch_without_a_readable_subject_fails_cleanly(self):
        batch = self.root / "batch"
        (batch / "a").mkdir(parents=True)
        for value in (str(batch), str(self.root / "none"), ""):
            with self.assertRaises(PreviewError):
                first_batch_source(value)

    def test_h1_pending_tile_sources_resolve_without_an_import(self):
        folder = self.root / "tiles"
        dwi = write_dwi(folder, "x.nii.gz")
        bval = write_text(folder / "other.bval", "0 1000 5 1000 0")
        pending = {"kind": "nifti", "source": str(dwi), "bval": str(bval), "mask": ""}
        source = resolve_preview_for_ui(batch_mode=False, input_dir="stale", subject_id="s", pending=pending)
        self.assertEqual((source.kind, source.bval, source.mask), ("nifti", bval, None))
        with self.assertRaises(PreviewError) as caught:
            resolve_pending_source("nifti", dwi, bval="")
        self.assertEqual(caught.exception.code, "no_bval")
        with self.assertRaises(PreviewError) as caught:
            resolve_pending_source("dicom", folder)
        self.assertEqual(caught.exception.code, "dicom_unconverted")
        self.assertEqual(resolve_pending_source("nrrd", dwi, subject_id="s").kind, "nrrd")


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


class FakeSlicer:
    def __init__(self, fail_label=False):
        self.nodes = []
        self.removed = []
        self.layers = None
        self.fail_label = fail_label
        scene = SimpleNamespace(RemoveNode=self.removed.append, GetNodeByID=lambda _i: None)
        self.mrmlScene = scene
        self.app = SimpleNamespace(layoutManager=lambda: None)
        self.util = SimpleNamespace(
            getNodesByClass=lambda _cls: list(self.nodes),
            loadVolume=self.load_volume,
            loadLabelVolume=self.load_label,
            setSliceViewerLayers=self.set_layers,
            resetSliceViews=lambda: None,
        )

    # O10: slicer.util.loadVolume returns the node and raises on failure (no returnNode argument).
    def load_volume(self, path, properties=None):
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
        self.layers = kwargs


class SceneTest(unittest.TestCase):
    def test_h1_scene_shows_mean_b0_with_mask_overlay_and_replaces_old_preview(self):
        fake = FakeSlicer()
        old = FakeNode("old")
        old.SetAttribute(PREVIEW_ATTRIBUTE, "meanb0")
        unrelated = FakeNode("user-volume")
        fake.nodes = [old, unrelated]
        info = show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", subject_id="sub-01")
        self.assertEqual(fake.removed, [old])
        self.assertEqual(info, {"volume_id": "id-sub-01_input_meanb0", "mask_id": "id-sub-01_input_mask"})
        self.assertEqual(fake.layers["background"], info["volume_id"])
        self.assertEqual(fake.layers["label"], info["mask_id"])
        self.assertTrue(fake.layers["fit"])
        self.assertEqual(fake.nodes[-2].GetAttribute(PREVIEW_ATTRIBUTE), "meanb0")

    def test_h1_scene_without_mask_or_with_unloadable_mask_still_shows_the_volume(self):
        fake = FakeSlicer(fail_label=True)
        messages = []
        info = show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", log=messages.append)
        self.assertIsNone(info["mask_id"])
        self.assertIsNone(fake.layers["label"])
        self.assertTrue(messages and "mask" in messages[0])
        self.assertIsNone(show_preview_in_scene(FakeSlicer(), "/tmp/m.nii.gz")["mask_id"])


class WiringTest(unittest.TestCase):
    def test_h1_module_is_python39_syntax_clean(self):
        source = (REPO_ROOT / "shared" / "input_preview.py").read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 9))

    def test_h1_compact_language_refresh_sets_bilingual_label_and_tooltip(self):
        from shared.ui_compact import refresh_input_preview_controls

        class Button:
            def setText(self, value):
                self.text = value

            def setToolTip(self, value):
                self.tooltip = value

            def setAccessibleName(self, value):
                self.accessible = value

        button = Button()
        owner = SimpleNamespace(_inputPreviewWidgets={"button": button}, _onboardingLanguage="en")
        refresh_input_preview_controls(owner)
        self.assertEqual(button.text, "View input")
        refresh_input_preview_controls(owner, "zh")
        self.assertEqual(button.text, "预览输入")
        self.assertIn("b0", button.tooltip)
        refresh_input_preview_controls(SimpleNamespace())  # no widgets: no-op

    def test_h1_widget_refresh_disables_button_with_reason_and_follows_pending_import(self):
        source_path = REPO_ROOT / "shared" / "module_widget.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        wanted = {"_refresh_input_preview_controls"}
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in wanted]
        self.assertEqual(len(methods), 1)
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
        refresh = namespace["_refresh_input_preview_controls"]

        class Button:
            enabled = True
            tooltip = ""

            def setEnabled(self, value):
                self.enabled = value

            def setToolTip(self, value):
                self.tooltip = value

        def owner(resolve, batch=False, **flags):
            button = Button()
            return SimpleNamespace(
                _inputPreviewWidgets={"button": button}, _onboardingLanguage="en", _parameterNode=object(),
                _input_preview_is_batch=lambda: batch, _input_preview_batch_root=lambda: str(REPO_ROOT),
                _resolve_input_preview_source=resolve, **flags), button

        def dicom():
            raise PreviewError("dicom_unconverted")

        me, button = owner(dicom)
        refresh(me)
        self.assertFalse(button.enabled)
        self.assertEqual(button.tooltip, "Convert inputs first (Prepare Inputs)")
        me, button = owner(lambda: None)
        with mock.patch("importlib.util.find_spec", return_value=object()):  # O11: the host may lack nibabel
            refresh(me)
        self.assertTrue(button.enabled)
        me, button = owner(lambda: None, _inputImportPending=True)
        refresh(me)
        self.assertFalse(button.enabled)
        me, button = owner(dicom, batch=True)  # batch mode checks only the folder; resolving waits for the click
        with mock.patch("importlib.util.find_spec", return_value=object()):
            refresh(me)
        self.assertTrue(button.enabled)

    def test_h1_button_is_installed_in_inputs_section_and_refreshed_from_run_state(self):
        compact = (REPO_ROOT / "shared" / "ui_compact.py").read_text(encoding="utf-8")
        self.assertIn("install_input_preview_controls(owner)", compact)
        self.assertIn("configurationLayout", compact)
        widget = (REPO_ROOT / "shared" / "module_widget.py").read_text(encoding="utf-8")
        self.assertIn("self._refresh_input_preview_controls()", widget)
        self.assertIn("def _on_input_preview_clicked", widget)

    def test_input_drop_routing_tile_border_and_preview_placement(self):
        from Testing.Python.test_ui_run_history import fake_qt
        from shared import ui_prepare_inputs as prep

        assigned: dict[str, str] = {}
        shown: list[str] = []

        class Owner:
            _parameterNode = SimpleNamespace(batchMode=False)

            def _prepare_inputs_widget_text(self, key):
                return assigned.get(key, "")

            def _set_prepare_inputs_widget_text(self, key, value):
                assigned[key] = value

        owner = Owner()
        with mock.patch.object(prep, "show_input_format", side_effect=lambda _o, k: shown.append(k)):
            ok = prep._assign_dropped_file_paths(
                owner, "nifti_source",
                [Path("/tmp/sub/dwi.nii.gz"), Path("/tmp/sub/dwi.bval"), Path("/tmp/sub/dwi.bvec")],
            )
            self.assertTrue(ok)
            self.assertEqual(shown, ["nifti"])
            self.assertEqual(assigned["source"], str(Path("/tmp/sub/dwi.nii.gz")))
            self.assertEqual(assigned["bval"], str(Path("/tmp/sub/dwi.bval")))
            self.assertEqual(assigned["bvec"], str(Path("/tmp/sub/dwi.bvec")))

            assigned.clear()
            ok_mask = prep._assign_dropped_file_paths(owner, "mask", [Path("/tmp/sub/mask.nii.gz")])
            self.assertTrue(ok_mask)
            self.assertEqual(assigned, {"mask": str(Path("/tmp/sub/mask.nii.gz"))})

            assigned.clear()
            ok_raw = prep._assign_dropped_file_paths(owner, "raw", [Path("/tmp/sub/volume.img")])
            self.assertTrue(ok_raw)
            self.assertEqual(assigned, {"raw": str(Path("/tmp/sub/volume.img"))})

        class FakeLayout:
            def __init__(self, initial=()):
                self.items = list(initial)

            def indexOf(self, w):
                return self.items.index(w) if w in self.items else -1

            def addWidget(self, w, *_a):
                self.items.append(w)

            def insertWidget(self, idx, w, *_a):
                self.items.insert(idx, w)

            def removeWidget(self, w):
                if w in self.items:
                    self.items.remove(w)

        qt = fake_qt()
        btn = object()
        header = FakeLayout()
        row_layout = FakeLayout()
        me = SimpleNamespace(
            _parameterNode=SimpleNamespace(batchMode=False),
            _prepareInputWidgets={"format_header": header},
            _inputPreviewWidgets={"button": btn, "layout": row_layout},
        )
        with mock.patch.dict(sys.modules, {"qt": qt}):
            prep.place_input_preview_button(me)
            self.assertEqual(header.items, [btn])
            self.assertEqual(row_layout.items, [])
            prep.place_input_preview_button(me, batch=True)
            self.assertEqual(header.items, [])
            self.assertEqual(row_layout.items, [btn])

        prepare_source = (REPO_ROOT / "shared" / "ui_prepare_inputs.py").read_text(encoding="utf-8")
        self.assertIn("border: 2px dashed", prepare_source)
        self.assertNotIn("border: 1px dashed", prepare_source)

        class DummyQFrame:
            NoFrame = 0

            def __init__(self):
                self._props = {}
                self.frame_shape = None
                self.updated = False

            def property(self, name):
                return self._props.get(name)

            def setProperty(self, name, value):
                self._props[name] = value

            def setFrameShape(self, shape):
                self.frame_shape = shape

            def update(self):
                self.updated = True

        class DummyEvent:
            def __init__(self, urls=()):
                self._urls = list(urls)
                self.accepted = False
                self.ignored = False

            def mimeData(self):
                return SimpleNamespace(
                    hasUrls=lambda: bool(self._urls),
                    urls=lambda: list(self._urls),
                )

            def acceptProposedAction(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        local_url = SimpleNamespace(isLocalFile=lambda: True, toLocalFile=lambda: "/tmp/sub/drop_dwi.nii.gz")
        assigned.clear()
        shown.clear()
        with mock.patch.object(prep, "show_input_format", side_effect=lambda _o, k: shown.append(k)):
            tile, handles_drops = prep._make_input_drop_tile(
                SimpleNamespace(QFrame=DummyQFrame), owner, "nifti_source"
            )
            self.assertTrue(handles_drops)
            self.assertEqual(tile.frame_shape, DummyQFrame.NoFrame)

            enter_ev = DummyEvent([local_url])
            tile.dragEnterEvent(enter_ev)
            self.assertTrue(enter_ev.accepted)
            self.assertTrue(tile.property("dragActive"))

            tile.dragLeaveEvent(DummyEvent())
            self.assertFalse(tile.property("dragActive"))

            drop_ev = DummyEvent([local_url])
            tile.dropEvent(drop_ev)
            self.assertTrue(drop_ev.accepted)
            self.assertFalse(tile.property("dragActive"))
            self.assertEqual(shown, ["nifti"])
            self.assertEqual(assigned["source"], str(Path("/tmp/sub/drop_dwi.nii.gz")))


if __name__ == "__main__":
    unittest.main()
