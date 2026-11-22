"""Batch 2, morphometry lane: surface post-processing results show natively in Results (no SlicerFreeSurfer).

Request: the task ``surface.ddsurfer_postprocess`` publishes ``{lh,rh}.thickness/.curv/.sulc/.area``, ``.inflated``,
``.aparc.annot``, ``.aparc.a2009s.annot`` and ``.stats``. None was listed in Results, so users dragged them into the
SlicerFreeSurfer extension (one flat red-green table). The plugin shows them itself: each measure or annotation on the
white ``.obj`` surface of the same run, with its own colours; the inflated surface as a model; ``.stats`` opens outside.
Each test imports the code under test inside the test, so a missing name fails per test.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import nibabel as nib
    import vtk
except ImportError:  # pragma: no cover - plain python3 has no numpy/vtk/nibabel
    np = nib = vtk = None

PRODUCT = "surface.ddsurfer_postprocess"
DECIDED_ORDER = [
    "lh.thickness", "lh.curv", "lh.sulc", "lh.area", "lh.aparc.annot", "lh.aparc.a2009s.annot", "lh.inflated",
    "lh.aparc.stats", "lh.aparc.a2009s.stats",
    "rh.thickness", "rh.curv", "rh.sulc", "rh.area", "rh.aparc.annot", "rh.aparc.a2009s.annot", "rh.inflated",
    "rh.aparc.stats", "rh.aparc.a2009s.stats",
]
# An octahedron: 6 vertices, 8 faces (0-based). The .obj below stores the same vertex order.
VERTICES = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
FACES = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]


def write_obj(path: Path, vertices=VERTICES, faces=FACES, shift=(0.0, 0.0, 0.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x + shift[0]} {y + shift[1]} {z + shift[2]}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunFolder:
    """A run folder with a surface task (white .obj) and a post-processing task, as the orchestrator records them."""

    def __init__(self, root: Path, *, morph_names=DECIDED_ORDER, with_obj=True, sha_override=None):
        self.root = root
        outputs = root / "outputs"
        self.surface_dir = outputs / "01_anatomical_priors" / "cortical_surface"
        self.morph_dir = outputs / "01_anatomical_priors" / "surface_morphometry"
        self.structural = root / "cache" / "stages" / "01_anatomical_priors" / "surface_morphometry" / "structural"
        self.structural.mkdir(parents=True)
        self.morph_dir.mkdir(parents=True)
        surface_outputs = {}
        hashes = {}
        for hemi in ("lh", "rh"):
            if with_obj:
                obj = write_obj(self.surface_dir / f"{hemi}.white.obj")
                surface_outputs[f"white_{hemi}"] = str(obj)
                hashes[f"{hemi}.white.obj"] = sha256(obj)
            else:  # an old run: only the LPS .stl
                stl = self.surface_dir / f"s_predicted_wm_{hemi}.stl"
                stl.parent.mkdir(parents=True, exist_ok=True)
                stl.write_bytes(b"solid x\nendsolid x\n")
                surface_outputs[f"model_{hemi}"] = str(stl)
            hashes[f"{hemi}.pial.obj"] = "0" * 64
        hashes.update(sha_override or {})
        (self.structural / "manifest.json").write_text(json.dumps({
            "kind": "ddsurfer_surface_only_postprocess", "native_sha256": hashes}), encoding="utf-8")
        morph_outputs = {}
        for name in reversed(list(morph_names)):  # the recorded order must not decide the row order
            path = self.morph_dir / name
            path.write_bytes(b"x")
            morph_outputs[name] = str(path)
        morph_outputs["structural_root"] = str(self.structural)
        morph_outputs["stats_root"] = str(self.structural.parent / "stats")
        self.manifest = {
            "subject_id": "s", "run_name": "run", "subject_root": str(root), "output_root": str(outputs),
            "tasks": [
                {"name": "surface.ddsurfer", "status": "completed", "outputs": surface_outputs},
                {"name": PRODUCT, "status": "completed", "manifest_path": str(self.structural / "manifest.json"),
                 "outputs": morph_outputs},
            ],
        }
        self.manifest_path = outputs / "00_run_metadata" / "run_orchestration_manifest.json"
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def morph(self, name: str) -> Path:
        return self.morph_dir / name


class TempRootTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()


# --------------------------------------------------------------------------------------------- rows (item 1, 2)
class RowSelectionTest(TempRootTest):
    def test_b2_morph_rows_only_for_the_product_from_recorded_outputs_in_the_decided_order(self) -> None:
        from shared.result_scene import result_candidates

        run = RunFolder(self.root)
        (run.morph_dir / "lh.jacobian_white").write_bytes(b"x")  # on disk, not recorded: never a row
        candidates, _ = result_candidates(run.manifest, include_surface_morphometry=True)
        names = [Path(item["path"]).name for item in candidates if item["product"] == PRODUCT]
        self.assertEqual(names, DECIDED_ORDER)

    def test_b2_morph_rows_are_off_by_default(self) -> None:
        from shared.result_scene import result_candidates

        run = RunFolder(self.root)
        candidates, _ = result_candidates(run.manifest)
        self.assertEqual([item for item in candidates if item["product"] == PRODUCT], [])

    def test_b2_morph_another_products_thickness_file_gives_no_row(self) -> None:
        from shared.result_scene import result_candidates

        run = RunFolder(self.root)
        other = self.root / "outputs" / "other" / "lh.thickness"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"x")
        stray = self.root / "outputs" / "other" / "x.thickness"
        stray.write_bytes(b"x")
        run.manifest["tasks"].append({"name": "surface.other", "status": "completed",
                                      "outputs": {"lh.thickness": str(other), "x": str(stray)}})
        # A post-processing output whose name is not one of the published names is not a row either.
        weird = run.morph_dir / "notes.thickness"
        weird.write_bytes(b"x")
        run.manifest["tasks"][1]["outputs"]["notes"] = str(weird)
        candidates, _ = result_candidates(run.manifest, include_surface_morphometry=True)
        paths = {item["path"] for item in candidates}
        self.assertNotIn(str(other), paths)
        self.assertNotIn(str(stray), paths)
        self.assertNotIn(str(weird), paths)

    def test_b2_morph_a_failed_postprocess_task_gives_no_rows(self) -> None:
        from shared.result_scene import result_candidates

        run = RunFolder(self.root)
        run.manifest["tasks"][1]["status"] = "failed"
        candidates, _ = result_candidates(run.manifest, include_surface_morphometry=True)
        self.assertEqual([item for item in candidates if item["product"] == PRODUCT], [])

    def test_b2_morph_the_automatic_result_set_is_unchanged(self) -> None:
        from shared.auto_results import automatic_result_candidates

        run = RunFolder(self.root)
        selected, skipped, deferred = automatic_result_candidates(run.manifest)
        everything = [item.get("path") for item in selected + skipped + deferred]
        self.assertFalse([path for path in everything if path and Path(path).parent == run.morph_dir])
        self.assertTrue(any(item["product"] == "surface.ddsurfer" for item in selected))  # the white .obj stays


class LoadGateTest(TempRootTest):
    """Item 2: bulk loading does not take these files; only the click on one row's eye (source_path) does."""

    def _logic(self):
        with mock.patch.dict(sys.modules, {"slicer": types.ModuleType("slicer")}):
            module_logic = importlib.import_module("shared.module_logic")

        class Logic(module_logic.SharedLogicMixin):
            def log(self, message):
                pass

        fake = SimpleNamespace(mrmlScene=mock.MagicMock(), util=SimpleNamespace(getNodesByClass=lambda kind: []))
        patcher = mock.patch.object(module_logic, "slicer", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Logic()

    def test_b2_morph_load_key_results_without_a_source_path_does_not_include_the_morphometry_rows(self) -> None:
        run = RunFolder(self.root)
        logic = self._logic()
        seen = []

        def spy(manifest, **kwargs):
            seen.append(kwargs)
            return [], []

        with mock.patch("shared.result_scene.result_candidates", side_effect=spy):
            logic.loadKeyResultsFromManifest(run.manifest_path)
            logic.loadKeyResultsFromManifest(run.manifest_path, source_path=run.morph("lh.thickness"))
        self.assertFalse(seen[0].get("include_surface_morphometry", False))
        self.assertTrue(seen[1].get("include_surface_morphometry"))

    def test_b2_morph_a_clicked_row_is_dispatched_to_the_morphometry_loader(self) -> None:
        run = RunFolder(self.root)
        logic = self._logic()
        calls = []
        with mock.patch("shared.surface_morphometry_display.load_morphometry_row",
                        side_effect=lambda slicer, candidate, manifest: calls.append(candidate) or False):
            result = logic.loadKeyResultsFromManifest(run.manifest_path, source_path=run.morph("lh.curv"))
        self.assertEqual([Path(item["path"]).name for item in calls], ["lh.curv"])
        self.assertEqual(result["loaded"], [])
        self.assertTrue(result["skipped"])  # the loader said no: an error, not a fiber-loader call


# -------------------------------------------------------------------------------------------- pure helpers
class RowKindTest(unittest.TestCase):
    def test_b2_morph_row_kinds_and_tooltips_in_both_languages(self) -> None:
        from shared.surface_morphometry_display import morphometry_row, row_tooltip

        expected = {
            "lh.thickness": ("measure", "lh", "thickness"), "rh.curv": ("measure", "rh", "curv"),
            "lh.sulc": ("measure", "lh", "sulc"), "lh.area": ("measure", "lh", "area"),
            "lh.aparc.annot": ("annotation", "lh", "aparc"), "rh.aparc.a2009s.annot": ("annotation", "rh", "aparc.a2009s"),
            "lh.inflated": ("inflated", "lh", "inflated"), "rh.aparc.stats": ("stats", "rh", "aparc"),
        }
        for name, kind in expected.items():
            with self.subTest(name=name):
                self.assertEqual(morphometry_row(PRODUCT, "/r/" + name), kind)
        self.assertIsNone(morphometry_row("surface.other", "/r/lh.thickness"))
        self.assertIsNone(morphometry_row(PRODUCT + "@x", "/r/x.thickness"))
        self.assertIsNone(morphometry_row(PRODUCT, "/r/lh.white"))
        self.assertEqual(row_tooltip("/r/lh.thickness", zh=False),
                         "Show thickness on the open cortical surface (defaults to white)")
        self.assertTrue(row_tooltip("/r/lh.thickness", zh=True))
        self.assertNotEqual(row_tooltip("/r/lh.thickness", zh=True), row_tooltip("/r/lh.thickness", zh=False))
        for name in expected:
            self.assertTrue(row_tooltip("/r/" + name, zh=False))
            self.assertTrue(row_tooltip("/r/" + name, zh=True))


@unittest.skipIf(np is None, "numpy/nibabel required (run with PythonSlicer)")
class RangeTest(unittest.TestCase):
    def test_b2_morph_ranges(self) -> None:
        from shared.surface_morphometry_display import measure_range

        self.assertEqual(measure_range("thickness", np.array([0.0, 9.0])), (1.0, 4.5))
        curv = np.concatenate([np.linspace(-1, 1, 1000), [50.0, np.nan, np.inf]])
        low, high = measure_range("curv", curv)
        self.assertAlmostEqual(high, -low)
        self.assertAlmostEqual(high, float(np.percentile(np.abs(np.concatenate([np.linspace(-1, 1, 1000), [50.0]])), 98)))
        low, high = measure_range("sulc", np.zeros(10))
        self.assertGreater(high, low)  # never 0 wide
        self.assertAlmostEqual(high, -low)
        area = np.linspace(0, 2, 1001)
        self.assertEqual(measure_range("area", area), (0.0, float(np.percentile(area, 98))))
        low, high = measure_range("area", np.zeros(5))
        self.assertGreater(high, low)


@unittest.skipIf(np is None, "numpy/nibabel required (run with PythonSlicer)")
class AnnotationTest(TempRootTest):
    def test_b2_morph_annotation_maps_unknown_and_non_sequential_ids_to_table_rows(self) -> None:
        from shared.surface_morphometry_display import annotation_colors, read_annotation

        path = self.root / "lh.aparc.annot"
        ctab = np.array([[220, 20, 10, 0, 0], [20, 220, 10, 0, 0], [10, 20, 220, 0, 0]], dtype=np.int32)
        names = [b"alpha", b"beta", b"gamma"]
        labels = np.array([0, 2, -1, 1, 2, -1], dtype=np.int32)  # -1: no label
        nib.freesurfer.write_annot(str(path), labels, ctab, names, fill_ctab=True)
        indices, colors = read_annotation(nib, path)
        self.assertEqual(colors[0][0], "unknown")
        self.assertEqual(colors[0][1:], (0.5, 0.5, 0.5))  # neutral grey
        self.assertEqual([name for name, *_ in colors[1:]], ["alpha", "beta", "gamma"])
        self.assertEqual(indices.tolist(), [1, 3, 0, 2, 3, 0])
        self.assertAlmostEqual(colors[2][1], 20 / 255)
        # Ids that are not sequential (a raw annotation value that is not in the table) are unknown too.
        mapped, table = annotation_colors(np.array([5, 9, 7, 3]), np.array([[1, 2, 3, 0, 9], [4, 5, 6, 0, 7]]),
                                          [b"a", b"b"])
        self.assertEqual(mapped.tolist(), [0, 1, 2, 0])
        self.assertEqual([name for name, *_ in table], ["unknown", "a", "b"])

    def test_b2_morph_the_tables_own_unknown_entry_is_grey_too(self) -> None:
        """Real aparc tables start with "unknown" (25, 5, 25); aparc.a2009s with "Unknown" (0, 0, 0). Both are grey."""

        from shared.surface_morphometry_display import UNKNOWN_GREY, annotation_colors

        ctab = np.array([[25, 5, 25, 0, 1639705], [25, 100, 40, 0, 2647065], [0, 0, 0, 0, 0]])
        mapped, table = annotation_colors(np.array([1639705, 2647065, -1]), ctab,
                                          [b"unknown", b"bankssts", b"Unknown"])
        self.assertEqual(mapped.tolist(), [1, 2, 0])
        self.assertEqual(table[1], ("unknown", *UNKNOWN_GREY))
        self.assertEqual(table[3], ("Unknown", *UNKNOWN_GREY))
        self.assertAlmostEqual(table[2][2], 100 / 255)

    @staticmethod
    def write_sparse_annot(path: Path, labels, entries, max_index: int) -> None:
        """A version-2 annotation whose colour table has only the given indices (nibabel's write_annot writes a
        contiguous table). ``entries``: (index, name, (r, g, b)). Big-endian int32, as FreeSurfer writes it."""

        def i32(*values):
            return np.asarray(values, dtype=">i4").tobytes()

        data = i32(len(labels))
        for vertex, value in enumerate(labels):
            data += i32(vertex, value)
        table_name = b"sparse_ctab.txt\x00"
        data += i32(1, -2, max_index, len(table_name)) + table_name + i32(len(entries))
        for index, name, (red, green, blue) in entries:
            encoded = name.encode("ascii") + b"\x00"
            data += i32(index, len(encoded)) + encoded + i32(red, green, blue, 0)
        path.write_bytes(data)

    def test_b2_morph_fix_M2_a_sparse_colour_table_gives_each_label_its_own_colour_and_name(self) -> None:
        """Review M2: nibabel returns one ctab row per index up to the maximum (unused rows are zero) and a compact
        names list; the names and colours must not be paired by position."""

        from shared.surface_morphometry_display import read_annotation

        two, five = (200, 30, 40), (10, 180, 90)
        packed = {index: red + green * 256 + blue * 65536 for index, (red, green, blue) in ((2, two), (5, five))}
        labels = [packed[2], packed[5], -1, 0, packed[5], packed[2]]
        path = self.root / "lh.aparc.annot"
        self.write_sparse_annot(path, labels, [(2, "two", two), (5, "five", five)], max_index=6)
        _raw, ctab, names = nib.freesurfer.read_annot(str(path), orig_ids=True)
        self.assertEqual((len(ctab), len(names)), (6, 2))  # the nibabel shape this item is about
        indices, colors = read_annotation(nib, path)
        table = {name: rgb for name, *rgb in colors}
        self.assertEqual(colors[0], ("unknown", 0.5, 0.5, 0.5))
        self.assertEqual(sorted(table), ["five", "two", "unknown"])  # no unused all-zero row became a label
        for name, rgb in (("two", two), ("five", five)):
            self.assertEqual(table[name], [value / 255 for value in rgb])
        rows = [colors[index][0] for index in indices.tolist()]
        self.assertEqual(rows, ["two", "five", "unknown", "unknown", "five", "two"])

    def test_b2_morph_fix_M2_without_the_file_table_a_sparse_table_gets_neutral_names_and_right_colours(self) -> None:
        from shared.surface_morphometry_display import annotation_colors

        ctab = np.zeros((6, 5), dtype=np.int64)
        ctab[2, :3], ctab[5, :3] = (200, 30, 40), (10, 180, 90)
        ctab[:, 4] = ctab[:, 0] + ctab[:, 1] * 256 + ctab[:, 2] * 65536
        indices, colors = annotation_colors(np.array([ctab[5, 4], ctab[2, 4], 0, -1]), ctab, [b"two", b"five"])
        self.assertEqual([name for name, *_ in colors], ["unknown", "label 2", "label 5"])
        self.assertEqual(colors[1][1:], (200 / 255, 30 / 255, 40 / 255))
        self.assertEqual(colors[2][1:], (10 / 255, 180 / 255, 90 / 255))
        self.assertEqual(indices.tolist(), [2, 1, 0, 0])  # raw 0 matches no unused zero row

    @unittest.skipIf(vtk is None, "VTK required")
    def test_b2_morph_annotation_lookup_with_the_manual_range_returns_each_rows_colour(self) -> None:
        from shared.surface_morphometry_display import annotation_scalar_range

        colors = [("unknown", .5, .5, .5)] + [(f"r{i}", i / 40, 1 - i / 40, (i * 7 % 40) / 40) for i in range(1, 36)]
        table = vtk.vtkLookupTable()
        table.SetNumberOfTableValues(len(colors))
        for index, (_, red, green, blue) in enumerate(colors):
            table.SetTableValue(index, red, green, blue, 1.0)
        low, high = annotation_scalar_range(len(colors))
        table.SetTableRange(low, high)
        for index, (_, red, green, blue) in enumerate(colors):
            rgb = [0.0, 0.0, 0.0]
            table.GetColor(float(index), rgb)
            for got, want in zip(rgb, (red, green, blue)):  # the table stores 8-bit colours; rows differ by >= 1/40
                self.assertAlmostEqual(got, want, delta=1.5 / 255, msg=index)


@unittest.skipIf(np is None, "numpy/nibabel required (run with PythonSlicer)")
class CarrierTest(TempRootTest):
    def test_b2_morph_the_carrier_is_the_white_obj_of_the_surface_task(self) -> None:
        from shared.surface_morphometry_display import white_carrier

        run = RunFolder(self.root)
        self.assertEqual(white_carrier(run.manifest, "lh"), run.surface_dir / "lh.white.obj")
        self.assertEqual(white_carrier(run.manifest, "rh"), run.surface_dir / "rh.white.obj")

    def test_b2_morph_an_old_run_with_only_an_stl_has_no_carrier(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, check_carrier, white_carrier

        run = RunFolder(self.root, with_obj=False)
        self.assertIsNone(white_carrier(run.manifest, "lh"))
        with self.assertRaises(MorphometryError) as caught:
            check_carrier(run.manifest, run.manifest["tasks"][1], "lh")
        self.assertIn(".obj", str(caught.exception))
        self.assertIn("白", str(caught.exception))  # bilingual

    def test_b2_morph_a_carrier_whose_checksum_differs_from_the_postprocess_manifest_is_refused(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, check_carrier

        run = RunFolder(self.root, sha_override={"lh.white.obj": "f" * 64})
        with self.assertRaises(MorphometryError) as caught:
            check_carrier(run.manifest, run.manifest["tasks"][1], "lh")
        self.assertIn("checksum", str(caught.exception))
        self.assertEqual(check_carrier(run.manifest, run.manifest["tasks"][1], "rh"), run.surface_dir / "rh.white.obj")

    def test_b2_morph_an_unreadable_postprocess_manifest_skips_only_the_checksum(self) -> None:
        from shared.surface_morphometry_display import check_carrier

        run = RunFolder(self.root)
        (run.structural / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(check_carrier(run.manifest, run.manifest["tasks"][1], "lh"), run.surface_dir / "lh.white.obj")

    def test_b2_morph_a_value_count_that_differs_from_the_point_count_is_refused(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, check_count

        check_count(6, 6, "lh.thickness", "lh.white.obj")
        with self.assertRaises(MorphometryError) as caught:
            check_count(5, 6, "lh.thickness", "lh.white.obj")
        self.assertIn("5", str(caught.exception))
        self.assertIn("6", str(caught.exception))


@unittest.skipIf(np is None, "numpy/nibabel required (run with PythonSlicer)")
class InflatedTest(TempRootTest):
    def test_b2_morph_inflated_coordinates_get_the_surface_to_scanner_shift_of_the_footer(self) -> None:
        from shared.surface_morphometry_display import read_inflated

        # An off-centre, oblique, non-conformed volume geometry (as the backend's write_surface records it).
        angle = np.deg2rad(20)
        cosines = np.array([[-np.cos(angle), 0, np.sin(angle)], [0, 0, -1], [np.sin(angle), np.cos(angle), 0]]).T
        zooms = np.array([1.2, 0.9, 1.5])
        shape = np.array([170, 210, 140])
        cras = np.array([12.5, -30.0, 41.0])
        info = {"head": np.array([2, 0, 20], dtype=np.int32), "valid": "1  # volume info valid",
                "filename": "brain.mgz", "volume": shape, "voxelsize": zooms,
                "xras": cosines[:, 0], "yras": cosines[:, 1], "zras": cosines[:, 2], "cras": cras}
        scanner = np.array(VERTICES, dtype=np.float64) * 30 + np.array([5.0, -8.0, 20.0])
        mat = cosines * zooms
        scanner_affine = np.eye(4)
        scanner_affine[:3, :3], scanner_affine[:3, 3] = mat, cras - mat @ (shape / 2)
        tkr = np.eye(4)
        tkr[:3, :3] = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]) * zooms
        tkr[:3, 3] = -tkr[:3, :3] @ (shape / 2)
        surface = nib.affines.apply_affine(tkr @ np.linalg.inv(scanner_affine), scanner)
        reflected = np.linalg.det((tkr @ np.linalg.inv(scanner_affine))[:3, :3]) < 0
        faces = np.array(FACES)[:, ::-1] if reflected else np.array(FACES)
        path = self.root / "lh.inflated"
        nib.freesurfer.write_geometry(str(path), surface.astype(np.float32), faces.astype(np.int32), volume_info=info)
        points, out_faces = read_inflated(nib, path)
        np.testing.assert_allclose(points, scanner, atol=1e-3)
        self.assertEqual(out_faces.tolist(), [list(face) for face in FACES])  # winding restored after a reflection

    def test_b2_morph_the_footer_transform_agrees_with_independent_anchors(self) -> None:
        """Anchors from outside this module: (1) a conformed 1 mm volume, where scanner RAS = surface RAS + c_ras
        (FreeSurfer's documented rule); (2) nibabel's own MGH header affines (vox2ras and vox2ras-tkr)."""

        from shared.surface_morphometry_display import read_inflated, surface_to_scanner

        cras = np.array([3.5, -17.25, 9.0])
        conformed = {"volume": np.array([256, 256, 256]), "voxelsize": np.array([1.0, 1.0, 1.0]),
                     "xras": np.array([-1.0, 0, 0]), "yras": np.array([0, 0, -1.0]), "zras": np.array([0, 1.0, 0]),
                     "cras": cras}
        np.testing.assert_allclose(surface_to_scanner(conformed), np.block([[np.eye(3), cras[:, None]],
                                                                            [np.zeros((1, 3)), np.ones((1, 1))]]),
                                   atol=1e-9)
        path = self.root / "lh.inflated"
        info = dict(conformed, head=np.array([2, 0, 20], dtype=np.int32), valid="1  # volume info valid",
                    filename="brain.mgz")
        nib.freesurfer.write_geometry(str(path), np.array(VERTICES, dtype=np.float32), np.array(FACES, dtype=np.int32),
                                      volume_info=info)
        points, faces = read_inflated(nib, path)
        np.testing.assert_allclose(points, np.array(VERTICES) + cras, atol=1e-5)
        self.assertEqual(faces.tolist(), [list(face) for face in FACES])  # no reflection: winding kept
        # Oblique, anisotropic, off-centre: compare with nibabel's MGHHeader affines.
        from nibabel.freesurfer.mghformat import MGHHeader

        angle = np.deg2rad(25)
        cosines = np.array([[-np.cos(angle), 0, np.sin(angle)], [0, 0, -1], [np.sin(angle), np.cos(angle), 0]]).T
        oblique = {"volume": np.array([160, 200, 120]), "voxelsize": np.array([1.1, 0.8, 1.6]),
                   "xras": cosines[:, 0], "yras": cosines[:, 1], "zras": cosines[:, 2],
                   "cras": np.array([-6.0, 22.0, 31.0])}
        header = MGHHeader()
        header["dims"][:3] = oblique["volume"]
        header["delta"] = oblique["voxelsize"]
        header["Mdc"] = cosines.T  # nibabel stores the direction cosines as rows
        header["Pxyz_c"] = oblique["cras"]
        expected = header.get_affine() @ np.linalg.inv(header.get_vox2ras_tkr())
        np.testing.assert_allclose(surface_to_scanner(oblique), expected, atol=1e-4)  # the MGH header is float32

    def test_b2_morph_inflated_without_a_footer_keeps_its_coordinates(self) -> None:
        from shared.surface_morphometry_display import read_inflated

        path = self.root / "lh.inflated"
        nib.freesurfer.write_geometry(str(path), np.array(VERTICES, dtype=np.float32), np.array(FACES, dtype=np.int32))
        points, faces = read_inflated(nib, path)
        np.testing.assert_allclose(points, np.array(VERTICES), atol=1e-6)
        self.assertEqual(faces.tolist(), [list(face) for face in FACES])


# ------------------------------------------------------------------------------------- sibling hide (item 7)
class _Node:
    def __init__(self, identity, attributes, visible=True):
        self.identity, self.attributes, self.visible = identity, dict(attributes), visible

    def GetID(self):
        return self.identity

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def GetDisplayVisibility(self):
        return int(self.visible)

    def SetDisplayVisibility(self, value):
        self.visible = bool(value)

    def IsA(self, name):
        return name in {"vtkMRMLModelNode", "vtkMRMLDisplayableNode"}


class SiblingHideTest(unittest.TestCase):
    def test_b2_morph_showing_an_overlay_hides_the_other_overlays_and_the_plain_white_model_of_that_hemisphere(self):
        from shared.surface_morphometry_display import PREFIX, hide_same_surface_siblings

        lh_white, rh_white = "/o/cortical_surface/lh.white.obj", "/o/cortical_surface/rh.white.obj"
        run = "/o/00_run_metadata/m.json"

        def node(identity, source, carrier=None, other_run=False):
            attributes = {PREFIX + "Run": "/other.json" if other_run else run, PREFIX + "SourcePath": source}
            if carrier:
                attributes[PREFIX + "CarrierPath"] = carrier
            return _Node(identity, attributes)

        thickness = node("t", "/o/m/lh.thickness", lh_white)
        curv = node("c", "/o/m/lh.curv", lh_white)
        aparc = node("a", "/o/m/lh.aparc.annot", lh_white)
        white = node("w", lh_white)
        inflated = node("i", "/o/m/lh.inflated")
        rh_curv = node("rc", "/o/m/rh.curv", rh_white)
        plain_rh = node("rw", rh_white)
        other_run = node("ow", lh_white, other_run=True)
        scene = [thickness, curv, aparc, white, inflated, rh_curv, plain_rh, other_run]
        fake = SimpleNamespace(util=SimpleNamespace(getNodesByClass=lambda kind: scene if kind != "vtkMRMLTableNode" else []))
        hidden = hide_same_surface_siblings(fake, [thickness])
        self.assertEqual(sorted(item.GetID() for item in hidden), ["a", "c", "w"])
        self.assertTrue(thickness.visible)
        self.assertFalse(curv.visible or aparc.visible or white.visible)
        self.assertTrue(inflated.visible and rh_curv.visible and plain_rh.visible and other_run.visible)
        # The plain white model and the inflated surface hide nothing.
        for node_ in scene:
            node_.visible = True
        self.assertEqual(hide_same_surface_siblings(fake, [white]), [])
        self.assertEqual(hide_same_surface_siblings(fake, [inflated]), [])

    def test_b2_morph_the_plain_white_model_is_hidden_when_its_recorded_path_goes_through_a_symlink(self):
        """CarrierPath is resolved; the white row keeps the manifest's spelling. Both name one file."""

        from shared.surface_morphometry_display import PREFIX, hide_same_surface_siblings

        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            (real / "cortical_surface").mkdir(parents=True)
            (real / "cortical_surface" / "lh.white.obj").write_text("v 0 0 0\n", encoding="utf-8")
            link = Path(tmp) / "link"
            link.symlink_to(real, target_is_directory=True)
            run = "/run.json"
            thickness = _Node("t", {PREFIX + "Run": run, PREFIX + "SourcePath": str(real / "lh.thickness"),
                                    PREFIX + "CarrierPath": str((real / "cortical_surface" / "lh.white.obj").resolve())})
            white = _Node("w", {PREFIX + "Run": run,
                                PREFIX + "SourcePath": str(link / "cortical_surface" / "lh.white.obj")})
            fake = SimpleNamespace(util=SimpleNamespace(
                getNodesByClass=lambda kind: [thickness, white] if kind != "vtkMRMLTableNode" else []))
            self.assertEqual([node.GetID() for node in hide_same_surface_siblings(fake, [thickness])], ["w"])
            self.assertFalse(white.visible)


# --------------------------------------------------------------------------------- nibabel missing (item 3)
class NibabelMissingTest(TempRootTest):
    def test_b2_morph_without_nibabel_one_bilingual_message_and_no_node(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        run = RunFolder(self.root)
        scene = mock.MagicMock(name="mrmlScene")
        fake = SimpleNamespace(mrmlScene=scene, util=mock.MagicMock(name="util"))
        real_import = builtins.__import__

        def no_nibabel(name, *args, **kwargs):
            if name == "nibabel" or name.startswith("nibabel."):
                raise ImportError("No module named 'nibabel'")
            return real_import(name, *args, **kwargs)

        candidate = {"path": str(run.morph("lh.thickness")), "product": PRODUCT, "key": "lh.thickness"}
        with mock.patch.dict(sys.modules, {key: None for key in list(sys.modules) if key.split(".")[0] == "nibabel"}), \
                mock.patch("builtins.__import__", side_effect=no_nibabel):
            with self.assertRaises(MorphometryError) as caught:
                load_morphometry_row(fake, candidate, run.manifest)
        message = str(caught.exception)
        self.assertIn("nibabel", message)
        self.assertIn("缺少", message)
        self.assertNotIn("Traceback", message)
        self.assertEqual(scene.method_calls, [])
        self.assertEqual(fake.util.method_calls, [])


# --------------------------------------------------------------------- builder with a fake scene (items 4-6)
@unittest.skipIf(np is None or vtk is None, "numpy/VTK/nibabel required (run with PythonSlicer)")
class BuilderTest(TempRootTest):
    def setUp(self) -> None:
        super().setUp()
        from test_b2_fiber_display import _Node as FiberNode, _Scene, _Storage

        self.scene = _Scene()
        self.color_nodes = []
        scene = self.scene

        def add_new(kind, name=""):
            if kind == "vtkMRMLColorTableNode":
                color = mock.MagicMock(name="color")
                color.GetID.return_value = "color%d" % len(self.color_nodes)
                self.color_nodes.append(color)
                return color
            return _Scene.AddNewNodeByClass(scene, kind, name)

        scene.AddNewNodeByClass = add_new
        original_remove = scene.RemoveNode

        def remove(node):
            if node in self.color_nodes:
                self.color_nodes.remove(node)
                return
            original_remove(node)

        scene.RemoveNode = remove
        scene.GetNodeByID = lambda identity: next(
            (node for node in scene.nodes + self.color_nodes if node.GetID() == identity),
            object() if str(identity).startswith("vtkMRMLColorTableNode") else None)

        def load_model(path, kind, properties=None, **kwargs):
            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()
            node = scene.AddNewNodeByClass("vtkMRMLModelNode", "model")
            node.SetAndObservePolyData(reader.GetOutput())
            storage = _Storage(path)
            scene.storage_nodes.append(storage)
            node.storage.append(storage)
            return True, node

        self.legend = mock.MagicMock(name="legend")
        self.fake = SimpleNamespace(
            mrmlScene=scene,
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            vtkMRMLDisplayNode=SimpleNamespace(UseManualScalarRange=1, UseDataScalarRange=0),
            modules=SimpleNamespace(colors=SimpleNamespace(logic=lambda: SimpleNamespace(
                AddDefaultColorLegendDisplayNode=lambda node: self.legend))),
            util=SimpleNamespace(loadNodeFromFile=load_model,
                                 getNodesByClass=lambda kind: list(scene.nodes)),
        )
        self.FiberNode = FiberNode

    def _run(self, **kwargs):
        run = RunFolder(self.root, **kwargs)
        count = len(VERTICES)
        nib.freesurfer.write_morph_data(str(run.morph("lh.thickness")), np.linspace(1, 4, count).astype(np.float32))
        nib.freesurfer.write_morph_data(str(run.morph("lh.curv")), np.linspace(-.3, .3, count).astype(np.float32))
        nib.freesurfer.write_morph_data(str(run.morph("lh.sulc")), np.linspace(-2, 2, count - 1).astype(np.float32))
        ctab = np.array([[220, 20, 10, 0, 0], [20, 220, 10, 0, 0]], dtype=np.int32)
        nib.freesurfer.write_annot(str(run.morph("lh.aparc.annot")), np.array([0, 1, -1, 0, 1, 1], dtype=np.int32),
                                   ctab, [b"one", b"two"], fill_ctab=True)
        return run

    def _candidate(self, run, name):
        return {"path": str(run.morph(name)), "product": PRODUCT, "key": name, "label": PRODUCT + "_" + name}

    def test_b2_morph_a_measure_gets_its_own_white_model_with_its_scalar_and_no_storage_node(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row

        run = self._run()
        self.assertTrue(load_morphometry_row(self.fake, self._candidate(run, "lh.thickness"), run.manifest))
        self.assertEqual(len(self.scene.nodes), 1)
        node = self.scene.nodes[0]
        self.assertEqual(node.name, "lh.thickness")
        self.assertEqual(node.GetPolyData().GetNumberOfPoints(), len(VERTICES))
        self.assertIsNotNone(node.GetPolyData().GetPointData().GetArray("thickness"))
        self.assertEqual(node.GetNumberOfStorageNodes(), 0)
        self.assertEqual(self.scene.storage_nodes, [])
        self.assertEqual(node.GetAttribute(PREFIX + "CarrierPath"), str(run.surface_dir / "lh.white.obj"))
        display = node.display
        display.SetActiveScalar.assert_called_with("thickness", vtk.vtkAssignAttribute.POINT_DATA)
        display.SetScalarRangeFlag.assert_called_with(1)
        display.SetScalarRange.assert_called_with(1.0, 4.5)
        display.SetScalarVisibility.assert_called_with(True)
        self.legend.SetTitleText.assert_called_with("Thickness (mm)")

    def test_b2_morph_a_count_mismatch_leaves_no_node_behind(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        run = self._run()
        with self.assertRaises(MorphometryError):
            load_morphometry_row(self.fake, self._candidate(run, "lh.sulc"), run.manifest)
        self.assertEqual(self.scene.nodes, [])
        self.assertEqual(self.scene.storage_nodes, [])
        self.assertEqual(self.color_nodes, [])

    def test_b2_morph_a_checksum_mismatch_or_a_missing_obj_leaves_no_node_behind(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        for kwargs in ({"sha_override": {"lh.white.obj": "f" * 64}}, {"with_obj": False}):
            with self.subTest(**{key: str(value) for key, value in kwargs.items()}):
                with tempfile.TemporaryDirectory() as other:
                    self.root = Path(other).resolve()
                    run = self._run(**kwargs)
                    with self.assertRaises(MorphometryError):
                        load_morphometry_row(self.fake, self._candidate(run, "lh.thickness"), run.manifest)
                    self.assertEqual(self.scene.nodes, [])
                    self.assertEqual(self.scene.storage_nodes, [])

    def test_b2_morph_an_annotation_gets_its_own_colour_table_tied_to_the_model(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row

        run = self._run()
        self.assertTrue(load_morphometry_row(self.fake, self._candidate(run, "lh.aparc.annot"), run.manifest))
        node = self.scene.nodes[0]
        self.assertEqual(len(self.color_nodes), 1)
        color = self.color_nodes[0]
        color.SetNumberOfColors.assert_called_with(3)
        color.SetColor.assert_any_call(0, "unknown", 0.5, 0.5, 0.5, 1.0)
        self.assertEqual(node.GetAttribute(PREFIX + "ColorNodeID"), color.GetID())
        node.display.SetAndObserveColorNodeID.assert_called_with(color.GetID())
        node.display.SetScalarRange.assert_called_with(0.0, 2.0)
        array = node.GetPolyData().GetPointData().GetArray("aparc")
        self.assertEqual([int(array.GetValue(i)) for i in range(array.GetNumberOfTuples())], [1, 2, 0, 1, 2, 2])

    def test_b2_morph_removing_the_model_removes_its_colour_table(self) -> None:
        from shared.surface_morphometry_display import PREFIX, remove_owned_color_node

        run = self._run()
        from shared.surface_morphometry_display import load_morphometry_row

        load_morphometry_row(self.fake, self._candidate(run, "lh.aparc.annot"), run.manifest)
        node = self.scene.nodes[0]
        self.assertEqual(len(self.color_nodes), 1)
        self.color_nodes[0].GetAttribute.side_effect = lambda key: node.GetID() if key == PREFIX + "OwnerNodeID" else None
        remove_owned_color_node(self.fake, node)
        self.assertEqual(self.color_nodes, [])


# ------------------------------------------------- review M3/M4: a scene that holds display, legend and colour nodes
class _SceneNode:
    """A scene node with an id, attributes, display node ids and storage node ids; other Set*/Get* calls are no-ops."""

    def __init__(self, scene, identity, kind):
        self.scene, self.identity, self.kind, self.attributes = scene, identity, kind, {}
        self.display_ids, self.storage_ids, self.polydata, self.name = [], [], None, ""

    def __getattr__(self, name):
        if name.startswith(("Set", "Get", "Create")):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def GetID(self):
        return self.identity

    def GetClassName(self):
        return self.kind

    def IsA(self, name):
        return name == self.kind or (name == "vtkMRMLDisplayableNode" and self.kind == "vtkMRMLModelNode")

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def SetName(self, name):
        self.name = name

    def SetAndObservePolyData(self, data):
        self.polydata = data

    def GetPolyData(self):
        return self.polydata

    def GetNumberOfDisplayNodes(self):
        return len(self.display_ids)

    def GetNthDisplayNode(self, index):
        return self.scene.GetNodeByID(self.display_ids[index])

    def GetDisplayNode(self):
        return self.GetNthDisplayNode(0) if self.display_ids else None

    def CreateDefaultDisplayNodes(self):
        if not self.display_ids:
            self.display_ids.append(self.scene.AddNewNodeByClass("vtkMRMLModelDisplayNode").GetID())

    def GetNumberOfStorageNodes(self):
        return len(self.storage_ids)

    def GetNthStorageNode(self, index):
        return self.scene.GetNodeByID(self.storage_ids[index])


class _OwningScene:
    """Like vtkMRMLScene: RemoveNode removes one node (never its display nodes) and then fires NodeRemovedEvent."""

    NodeRemovedEvent = 66001

    def __init__(self, builtin_ids=()):
        self.nodes, self.observers, self.closing, self.count = {}, [], False, 0
        self.builtin = {identity: _SceneNode(self, identity, "vtkMRMLColorTableNode") for identity in builtin_ids}

    def AddNewNodeByClass(self, kind, name=""):
        self.count += 1
        node = _SceneNode(self, f"{kind}_{self.count}", kind)
        node.name = name
        self.nodes[node.GetID()] = node
        return node

    def GetNodeByID(self, identity):
        return self.nodes.get(identity) or self.builtin.get(identity)

    def RemoveNode(self, node):
        if node is None or self.nodes.pop(node.GetID(), None) is None:
            return
        for owner in self.nodes.values():
            if node.GetID() in owner.storage_ids:
                owner.storage_ids.remove(node.GetID())
        for event, callback in list(self.observers):
            if event == self.NodeRemovedEvent:
                callback(self, event, node)

    def AddObserver(self, event, callback):
        self.observers.append((event, callback))
        return len(self.observers)

    def IsClosing(self):
        return self.closing

    def kinds(self):
        return sorted(node.kind for node in self.nodes.values())


@unittest.skipIf(np is None or vtk is None, "numpy/VTK/nibabel required (run with PythonSlicer)")
class OwnedNodesTest(TempRootTest):
    """Review M3: a row owns its model, its display node(s), its legend and its colour table. Review M4: a run whose
    manifest has no completed surface task finds the white surface in the published layout of the same output root."""

    def setUp(self) -> None:
        super().setUp()
        from shared.surface_morphometry_display import DIVERGING_TABLES, SEQUENTIAL_TABLES

        self.scene = scene = _OwningScene(SEQUENTIAL_TABLES[:1] + DIVERGING_TABLES[:1])

        def load_model(path, kind, properties=None, **kwargs):
            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()
            node = scene.AddNewNodeByClass("vtkMRMLModelNode", "model")
            node.SetAndObservePolyData(reader.GetOutput())
            node.CreateDefaultDisplayNodes()
            node.storage_ids.append(scene.AddNewNodeByClass("vtkMRMLModelStorageNode").GetID())
            return True, node

        def add_legend(node):  # Slicer adds the legend as one more display node of the model
            legend = scene.AddNewNodeByClass("vtkMRMLColorLegendDisplayNode")
            node.display_ids.append(legend.GetID())
            return legend

        self.fake = SimpleNamespace(
            mrmlScene=scene,
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            vtkMRMLDisplayNode=SimpleNamespace(UseManualScalarRange=1, UseDataScalarRange=0),
            modules=SimpleNamespace(colors=SimpleNamespace(logic=lambda: SimpleNamespace(
                AddDefaultColorLegendDisplayNode=add_legend))),
            util=SimpleNamespace(loadNodeFromFile=load_model, getNodesByClass=lambda kind: [
                node for node in scene.nodes.values() if node.IsA(kind)]),
        )

    def _run(self, **kwargs):
        run = RunFolder(self.root, **kwargs)
        count = len(VERTICES)
        nib.freesurfer.write_morph_data(str(run.morph("lh.thickness")), np.linspace(1, 4, count).astype(np.float32))
        nib.freesurfer.write_morph_data(str(run.morph("lh.sulc")), np.linspace(-2, 2, count - 1).astype(np.float32))
        nib.freesurfer.write_annot(str(run.morph("lh.aparc.annot")), np.array([0, 1, -1, 0, 1, 1], dtype=np.int32),
                                   np.array([[220, 20, 10, 0, 0], [20, 220, 10, 0, 0]], dtype=np.int32),
                                   [b"one", b"two"], fill_ctab=True)
        nib.freesurfer.write_geometry(str(run.morph("lh.inflated")), np.array(VERTICES, dtype=np.float32),
                                      np.array(FACES, dtype=np.int32))
        return run

    def _load(self, run, name, manifest=None):
        from shared.surface_morphometry_display import load_morphometry_row

        before = set(self.scene.nodes)
        candidate = {"path": str(run.morph(name)), "product": PRODUCT, "key": name, "label": PRODUCT + "_" + name}
        self.assertTrue(load_morphometry_row(self.fake, candidate, run.manifest if manifest is None else manifest))
        new = [self.scene.nodes[identity] for identity in self.scene.nodes if identity not in before]
        return next(node for node in new if node.kind == "vtkMRMLModelNode"), new

    def test_b2_morph_fix_M3_a_count_mismatch_leaves_no_model_display_legend_or_colour_node(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        run = self._run()
        with self.assertRaises(MorphometryError):
            load_morphometry_row(self.fake, {"path": str(run.morph("lh.sulc")), "product": PRODUCT, "key": "lh.sulc"},
                                 run.manifest)
        self.assertEqual(self.scene.kinds(), [])

    def test_b2_morph_fix_M3_removing_a_row_model_removes_its_display_legend_and_colour_nodes(self) -> None:
        run = self._run()
        rows = {name: self._load(run, name) for name in ("lh.thickness", "lh.aparc.annot", "lh.inflated")}
        thickness_kinds = sorted(node.kind for node in rows["lh.thickness"][1])
        self.assertEqual(thickness_kinds, ["vtkMRMLColorLegendDisplayNode", "vtkMRMLModelDisplayNode",
                                           "vtkMRMLModelNode"])
        self.assertIn("vtkMRMLColorTableNode", [node.kind for node in rows["lh.aparc.annot"][1]])
        for name in ("lh.thickness", "lh.aparc.annot", "lh.inflated"):
            model, created = rows.pop(name)
            self.scene.RemoveNode(model)
            self.assertEqual([node.GetID() for node in created if node.GetID() in self.scene.nodes], [], name)
            remaining = {node.GetID() for _model, nodes in rows.values() for node in nodes}
            self.assertEqual(set(self.scene.nodes), remaining, name)  # never a node of another row

    def test_b2_morph_fix_M3_nothing_is_removed_while_the_scene_is_closing(self) -> None:
        run = self._run()
        model, created = self._load(run, "lh.aparc.annot")
        thickness, created_thickness = self._load(run, "lh.thickness")
        self.scene.closing = True
        self.scene.RemoveNode(model)
        self.scene.RemoveNode(thickness)
        expected = {node.GetID() for node in created + created_thickness} - {model.GetID(), thickness.GetID()}
        self.assertEqual(set(self.scene.nodes), expected)

    def test_b2_morph_fix_M3_removing_a_plain_model_removes_nothing_else(self) -> None:
        run = self._run()
        _model, created = self._load(run, "lh.thickness")
        plain = self.scene.AddNewNodeByClass("vtkMRMLModelNode", "lh.white.obj")
        plain.CreateDefaultDisplayNodes()
        plain_display = plain.display_ids[0]
        self.scene.RemoveNode(plain)
        self.assertEqual(set(self.scene.nodes), {node.GetID() for node in created} | {plain_display})

    def test_b2_morph_fix_M4_without_a_recorded_surface_task_the_published_white_surface_carries_the_row(self) -> None:
        from shared.surface_morphometry_display import PREFIX

        run = self._run()
        for surface_task in (None, {"name": "surface.ddsurfer", "status": "blocked", "reason": "missing resources"}):
            with self.subTest(surface_task=surface_task):
                manifest = dict(run.manifest, tasks=[task for task in run.manifest["tasks"]
                                                     if task["name"] != "surface.ddsurfer"])
                if surface_task is not None:
                    manifest["tasks"].insert(0, surface_task)
                model, _created = self._load(run, "lh.thickness", manifest)
                self.assertEqual(model.GetAttribute(PREFIX + "CarrierPath"),
                                 str((run.surface_dir / "lh.white.obj").resolve()))

    def test_b2_morph_fix_M4_the_published_white_surface_keeps_the_checksum_and_layout_checks(self) -> None:
        from shared.surface_morphometry_display import MorphometryError, load_morphometry_row

        run = self._run(sha_override={"lh.white.obj": "f" * 64})
        manifest = dict(run.manifest, tasks=[run.manifest["tasks"][1]])
        candidate = {"path": str(run.morph("lh.thickness")), "product": PRODUCT, "key": "lh.thickness"}
        with self.assertRaises(MorphometryError) as caught:
            load_morphometry_row(self.fake, candidate, manifest)
        self.assertIn("checksum", str(caught.exception))
        # A row that is not in the layout of the manifest's output root gets no fallback carrier.
        other = dict(manifest, output_root=str(self.root / "elsewhere" / "outputs"))
        with self.assertRaises(MorphometryError) as caught:
            load_morphometry_row(self.fake, candidate, other)
        self.assertIn(".obj", str(caught.exception))
        self.assertEqual(self.scene.kinds(), [])


# ------------------------------------------------------------------------------------ Results tree (items 1, 9, 10)
class TreeTest(TempRootTest):
    def test_b2_morph_the_tree_lists_the_rows_in_order_with_tooltips_and_stats_open_outside(self) -> None:
        from shared import ui_run_history
        from shared.result_scene import result_candidates as real_candidates
        from test_ui_results_tree import USER_ROLE, Harness, find_child
        from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest

        run = RunFolder(self.root)
        data = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        data.update({"kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed",
                     "ended_at": "2026-11-01T10:00:00Z"})
        run.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        with mock.patch("shared.result_scene.result_candidates", side_effect=real_candidates):
            h.refresh([parse_run_manifest(run.manifest_path)])
        subject = next(item for item in h.tree.items if item.text(0).startswith("s"))
        anatomy = find_child(subject, "Anatomical priors")
        post = find_child(anatomy, "Surface postprocess")
        rows = [post.child(i) for i in range(post.childCount())]
        names = [row.text(0) for row in rows if row.data(0, USER_ROLE + ui_run_history.ROLE_PATH)]
        self.assertEqual(names, DECIDED_ORDER)
        by_name = {row.text(0): row for row in rows}
        self.assertIn("Show thickness on the open cortical surface (defaults to white)", by_name["lh.thickness"].tooltips[0])
        self.assertIn(str(run.morph("lh.thickness")), by_name["lh.thickness"].tooltips[0])
        self.assertEqual(by_name["lh.aparc.stats"].text(1), "↗")
        self.assertNotEqual(by_name["lh.thickness"].text(1), "↗")
        self.h, self.by_name = h, by_name

    def test_b2_morph_a_language_switch_retitles_the_row_tooltips(self) -> None:
        from shared.ui_onboarding import apply_onboarding_language

        self.test_b2_morph_the_tree_lists_the_rows_in_order_with_tooltips_and_stats_open_outside()
        apply_onboarding_language(self.h.owner, "zh")
        self.assertIn("在已打开的皮层表面上显示皮层厚度（默认白质表面）", self.by_name["lh.thickness"].tooltips[0])
        apply_onboarding_language(self.h.owner, "en")
        self.assertIn("Show thickness on the open cortical surface (defaults to white)", self.by_name["lh.thickness"].tooltips[0])

    def test_b2_morph_the_eye_that_shows_an_overlay_hides_its_siblings_and_the_eye_that_hides_does_not(self) -> None:
        from shared import ui_run_history
        from shared.result_scene import source_stamp

        self.test_b2_morph_the_tree_lists_the_rows_in_order_with_tooltips_and_stats_open_outside()
        leaf = self.by_name["lh.thickness"]
        node = mock.MagicMock(name="lh.thickness")
        node.IsA.return_value = False
        node.GetAttribute.side_effect = lambda key: source_stamp(self.root / "outputs" / "01_anatomical_priors"
                                                                 / "surface_morphometry" / "lh.thickness")
        for visible in (False, True):
            with self.subTest(visible=visible), \
                    mock.patch.object(ui_run_history, "_output_nodes", return_value=[node]), \
                    mock.patch.object(ui_run_history, "_output_visible", return_value=visible), \
                    mock.patch("shared.surface_morphometry_display.hide_same_surface_siblings") as hide, \
                    mock.patch("shared.fiber_display.is_fiber_bundle_node", return_value=False):
                ui_run_history._toggle_output(self.h.owner, leaf)
                node.SetDisplayVisibility.assert_called_with(not visible)
                if visible:
                    hide.assert_not_called()
                else:
                    hide.assert_called_once_with(self.h.slicer, [node])


@unittest.skipIf(np is None or vtk is None, "numpy/VTK/nibabel required (run with PythonSlicer)")
class AdaptiveSurfaceCarrierTest(OwnedNodesTest):
    """Post-processing overlays follow whichever cortical surface(s) are open (white, pial, or both), defaulting to white."""

    def _run_with_white_and_pial(self):
        run = self._run()
        pial_hashes = {}
        surface_task = run.manifest["tasks"][0]
        for hemi in ("lh", "rh"):
            pial = write_obj(run.surface_dir / f"{hemi}.pial.obj", shift=(0.5, 0.5, 0.5))
            surface_task["outputs"][f"pial_{hemi}"] = str(pial)
            pial_hashes[f"{hemi}.pial.obj"] = sha256(pial)
        struct_manifest = run.structural / "manifest.json"
        record = json.loads(struct_manifest.read_text(encoding="utf-8"))
        record["native_sha256"].update(pial_hashes)
        struct_manifest.write_text(json.dumps(record), encoding="utf-8")
        run.manifest_path.write_text(json.dumps(run.manifest), encoding="utf-8")
        return run

    def _add_plain_surface(self, run, filename: str, *, visible: bool = True):
        from shared.surface_morphometry_display import PREFIX

        node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", filename)
        node.SetAttribute(PREFIX + "Run", str(run.manifest_path.resolve()))
        node.SetAttribute(PREFIX + "SourcePath", str((run.surface_dir / filename).resolve()))
        node.visible = visible
        node.GetDisplayVisibility = lambda n=node: int(n.visible)
        node.SetDisplayVisibility = lambda v, n=node: setattr(n, "visible", bool(v))
        return node

    def test_only_pial_open_loads_overlay_onto_pial_surface(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row

        run = self._run_with_white_and_pial()
        self._add_plain_surface(run, "lh.white.obj", visible=False)
        self._add_plain_surface(run, "lh.pial.obj", visible=True)
        before = set(self.scene.nodes)
        candidate = {"path": str(run.morph("lh.thickness")), "product": PRODUCT, "key": "lh.thickness"}
        self.assertTrue(load_morphometry_row(self.fake, candidate, run.manifest))
        new_models = [n for nid, n in self.scene.nodes.items() if nid not in before and n.kind == "vtkMRMLModelNode"]
        self.assertEqual(len(new_models), 1)
        self.assertEqual(new_models[0].name, "lh.thickness (pial)")
        self.assertEqual(new_models[0].GetAttribute(PREFIX + "CarrierSurface"), "pial")
        self.assertEqual(new_models[0].GetAttribute(PREFIX + "CarrierPath"),
                         str((run.surface_dir / "lh.pial.obj").resolve()))

    def test_both_white_and_pial_open_loads_overlay_onto_both_surfaces_with_single_legend(self) -> None:
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row

        run = self._run_with_white_and_pial()
        self._add_plain_surface(run, "lh.white.obj", visible=True)
        self._add_plain_surface(run, "lh.pial.obj", visible=True)
        before = set(self.scene.nodes)
        candidate = {"path": str(run.morph("lh.thickness")), "product": PRODUCT, "key": "lh.thickness"}
        self.assertTrue(load_morphometry_row(self.fake, candidate, run.manifest))
        new_nodes = [n for nid, n in self.scene.nodes.items() if nid not in before]
        new_models = [n for n in new_nodes if n.kind == "vtkMRMLModelNode"]
        legends = [n for n in new_nodes if n.kind == "vtkMRMLColorLegendDisplayNode"]
        self.assertEqual([m.GetAttribute(PREFIX + "CarrierSurface") for m in new_models], ["white", "pial"])
        self.assertEqual([m.name for m in new_models], ["lh.thickness (white)", "lh.thickness (pial)"])
        self.assertEqual(len(legends), 1)

    def test_neither_surface_open_defaults_to_white_and_opens_white_eye_and_restores_plain_on_hide(self) -> None:
        from shared import ui_run_history
        from shared.result_scene import result_candidates as real_candidates, source_stamp
        from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest
        from shared.surface_morphometry_display import PREFIX, load_morphometry_row
        from test_ui_results_tree import USER_ROLE, Harness, find_child

        run = self._run_with_white_and_pial()
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

            # Step 1: Neither white nor pial is open; user clicks lh.aparc.annot -> defaults to white AND opens lh.white.obj eye
            ui_run_history._toggle_output(h.owner, post_rows["lh.aparc.annot"])
            self.assertEqual(post_rows["lh.aparc.annot"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.pial.obj"].tooltips[1], "Show")

            # Step 2: User turns OFF lh.aparc.annot -> plain lh.white.obj loads & stays visible (WM surface stays open)
            ui_run_history._toggle_output(h.owner, post_rows["lh.aparc.annot"])
            self.assertEqual(post_rows["lh.aparc.annot"].tooltips[1], "Show")
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Hide")

            # Step 3: User turns OFF lh.white.obj and turns ON lh.pial.obj, then turns ON lh.aparc.annot -> switches to pial!
            ui_run_history._toggle_output(h.owner, surf_rows["lh.white.obj"])
            ui_run_history._toggle_output(h.owner, surf_rows["lh.pial.obj"])
            ui_run_history._toggle_output(h.owner, post_rows["lh.aparc.annot"])
            visible_aparc = [
                n for n in self.scene.nodes.values()
                if n.kind == "vtkMRMLModelNode" and n.GetAttribute(PREFIX + "SourcePath") == str(run.morph("lh.aparc.annot").resolve())
                and n.GetDisplayVisibility()
            ]
            self.assertEqual([n.GetAttribute(PREFIX + "CarrierSurface") for n in visible_aparc], ["pial"])
            self.assertEqual(surf_rows["lh.pial.obj"].tooltips[1], "Hide")
            self.assertEqual(surf_rows["lh.white.obj"].tooltips[1], "Show")

            # Step 4: User turns OFF lh.pial.obj eye while lh.aparc.annot is on it -> hides both pial and aparc!
            ui_run_history._toggle_output(h.owner, surf_rows["lh.pial.obj"])
            self.assertEqual(surf_rows["lh.pial.obj"].tooltips[1], "Show")
            self.assertEqual(post_rows["lh.aparc.annot"].tooltips[1], "Show")


if __name__ == "__main__":
    unittest.main()
