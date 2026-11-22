"""Batch 2 (Results show every finished result): one published connectome folder per tractography source.

Before this change every tracking variant published ``outputs/05_connectome/<subject>_streamline_count.csv`` (and the
nodes, SIFT2 and node-parcellation files) at one fixed path, so a later variant replaced the earlier variant's files and
both task records pointed to the same file. Now each variant publishes into ``outputs/05_connectome/<source>/`` (also
with one tracking), with ``<source>`` the same id the bundle folders use. Old flat files stay where they are.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_product_orchestrator as T  # fixture helpers only (a module object; its tests do not run twice)

from shared.io_layout import (
    _migrate_qc_from_stage_caches,
    archive_development_outputs,
    materialize_canonical_output_tree,
    migrate_subject_output_tree,
    subject_layout,
)
from shared.orchestrator import (
    ProductOrchestratorConfig,
    _publish_connectome_outputs,
    _run_stage_dirs,
    run_subject_products,
)

SID = "100307"
CSD_LIKE = ("csd_act", "saf")  # the SAF child tracks with CSD/ACT in its own run folders


def _write_subject(root: Path) -> Path:
    """Raw subject plus a parcellation; returns the parcellation path."""

    T._write_valid_raw_subject(root, shape=(2, 2, 2), affine=T.UNAFFECTED_AFFINE)
    layout = subject_layout(root)
    parcel_dir = layout.stage_cache("01_anatomical_priors/brain_parcellation")
    parcel_dir.mkdir(parents=True, exist_ok=True)
    parcel = parcel_dir / f"{SID}-DDSurfer-wmparc.nii.gz"
    T._write_test_volume(parcel, affine=T.UNAFFECTED_AFFINE)
    return parcel


def _write_tracking(root: Path, run_name: str, backend: str) -> dict:
    """A finished tracking stage for ``backend`` in ``run_name``; returns the stage dirs plus tractogram and weights."""

    layout = subject_layout(root)
    dirs = dict(_run_stage_dirs(layout, run_name, root))
    if backend in CSD_LIKE:
        tract = dirs["tractogram_dir"] / f"{SID}_csd_act_ifod2.tck"
        for path in (tract, tract.with_suffix(".trk")):
            T._write_test_tractogram(path)
        tract.with_suffix(".vtk").write_text("# vtk\n", encoding="utf-8")
        weights = dirs["tractogram_dir"] / f"{SID}_csd_act_ifod2_sift2_weights.txt"
        weights.write_text("1.0\n", encoding="utf-8")
        dirs["csd_dir"].mkdir(parents=True, exist_ok=True)
    else:
        tract = dirs["dti_tractogram_dir"] / f"{SID}-dti-deterministic.trk"
        T._write_test_tractogram(tract)
        weights = None
        (dirs["dti_tractogram_dir"] / f"{SID}-dti-deterministic_qc.json").write_text('{"status": "completed"}\n')
        (dirs["dti_tractogram_dir"] / f"{SID}-dti-deterministic_manifest.json").write_text("{}\n")
    dirs["tractogram"] = tract
    dirs["weights"] = weights
    return dirs


def _write_connectome_stage(dirs: dict, parcel: Path, backend: str) -> None:
    """Finished connectome stage files whose content names ``backend`` (as the resume check needs them)."""

    qc = '{"status": "completed", "node_count": 2, "edge_count": 1}\n'
    cdir = dirs["connectome_dir"]
    cdir.mkdir(parents=True, exist_ok=True)
    deps = {"tractography": T._file_record(dirs["tractogram"]), "ddparcel_parcellation": T._file_record(parcel)}
    (cdir / f"{SID}-connectome-streamline_count_qc.json").write_text(qc)
    (cdir / f"{SID}-connectome-streamline_count_manifest.json").write_text(json.dumps({"dependencies": deps}) + "\n")
    (cdir / f"{SID}-connectome-streamline_count.csv").write_text(
        f"source_node,target_node,streamline_count\n10,20,1\n#FROM_{backend}\n")
    (cdir / f"{SID}-connectome-nodes.tsv").write_text(f"node_id\tlabel\tname\n10\t10\tLeft_{backend}\n20\t20\tRight\n")
    (cdir / f"{SID}-connectome-node-parcellation.nii.gz").write_text(f"parcellation FROM {backend}\n")
    if backend in CSD_LIKE:
        sdir = dirs["sift2_connectome_dir"]
        sdir.mkdir(parents=True, exist_ok=True)
        stem = f"{SID}-connectome-sift2_weighted_streamline_count"
        (sdir / f"{stem}_qc.json").write_text(qc)
        (sdir / f"{stem}.csv").write_text(
            f"source_node,target_node,sift2_weighted_streamline_count\n10,20,1.5\n#FROM_{backend}_sift2\n")
        (sdir / f"{SID}-connectome-nodes.tsv").write_text(
            f"node_id\tlabel\tname\n10\t10\tLeft_{backend}_sift2\n20\t20\tRight\n")
        (sdir / f"{stem}_manifest.json").write_text(json.dumps({"dependencies": {
            **deps, "sift2_weights": T._file_record(dirs["weights"])}}) + "\n")


def _run(root: Path, run_name: str, tracking: str, *, run_connectome=None, **extra):
    """Run the orchestrator (resume, execute) with the tracking stages reported finished.

    ``run_connectome`` defaults to a function that fails the test: a finished connectome must not be computed again.
    """

    def must_not_run(*args, **kwargs):
        raise AssertionError("the connectome must not be computed again")

    config = ProductOrchestratorConfig(subject_id=SID, subject_root=root, targets=("tractography", "connectome"),
                                       run_name=run_name, tractography_backend=tracking, connectome_backend="python",
                                       execute=True, resume=True, **extra)
    with mock.patch("shared.orchestrator._csd_act_complete", return_value=True), \
            mock.patch("shared.orchestrator._dti_tracking_frame_current", return_value=True), \
            mock.patch("shared.orchestrator._segmentation_complete", return_value=True), \
            mock.patch("shared.orchestrator._publish_segmentation_outputs", return_value={}), \
            mock.patch("connectome.run_connectome", side_effect=run_connectome or must_not_run), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = run_subject_products(config)
    return json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))


def _connectome_tasks(manifest: dict) -> dict:
    return {task["name"]: task for task in manifest.get("tasks", []) if task["name"].startswith("connectome.")}


def _flat_files(layout) -> list[str]:
    """Files directly in ``outputs/05_connectome`` (the old flat layout)."""

    if not layout.connectome.is_dir():
        return []
    return sorted(p.name for p in layout.connectome.iterdir() if p.is_file())


class _SubjectCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / SID
        self.parcel = _write_subject(self.root)
        self.layout = subject_layout(self.root)

    def _variants(self, run_name: str, backends) -> None:
        for backend in backends:
            dirs = _write_tracking(self.root, f"{run_name}__{backend}", backend)
            _write_connectome_stage(dirs, self.parcel, backend)


class TwoVariantTests(_SubjectCase):
    def test_b2_conn_two_variants_publish_into_their_own_source_folder(self) -> None:
        """(a) Reproduction: DTI and CSD/ACT variants no longer share one published matrix or nodes file."""

        self._variants("cmp", ("dti_det", "csd_act"))
        tasks = _connectome_tasks(_run(self.root, "cmp", "dti_det,csd_act"))
        first, second = tasks["connectome.streamline_count@dti_det"], tasks["connectome.streamline_count@csd_act"]
        self.assertEqual((first["status"], second["status"]), ("skipped", "skipped"))
        self.assertNotEqual(first["outputs"]["matrix_path"], second["outputs"]["matrix_path"])
        self.assertNotEqual(first["outputs"]["nodes_path"], second["outputs"]["nodes_path"])
        for backend, task in (("dti_det", first), ("csd_act", second)):
            matrix = Path(task["outputs"]["matrix_path"])
            self.assertEqual(matrix, self.layout.connectome / backend / f"{SID}_streamline_count.csv")
            self.assertIn(f"#FROM_{backend}\n", matrix.read_text(encoding="utf-8"))
            self.assertEqual(Path(task["outputs"]["nodes_path"]), self.layout.connectome / backend / f"{SID}_nodes.tsv")
            self.assertEqual(Path(task["outputs"]["node_parcellation_path"]),
                             self.layout.connectome / backend / f"{SID}_node_parcellation.nii.gz")
        self.assertIn("Left_dti_det", Path(first["outputs"]["nodes_path"]).read_text(encoding="utf-8"))
        sift2 = tasks["connectome.sift2_weighted@csd_act"]
        self.assertEqual(Path(sift2["outputs"]["matrix_path"]),
                         self.layout.connectome / "csd_act" / f"{SID}_sift2_weighted.csv")
        self.assertIn("#FROM_csd_act_sift2", Path(sift2["outputs"]["matrix_path"]).read_text(encoding="utf-8"))
        self.assertFalse((self.layout.connectome / f"{SID}_streamline_count.csv").exists())
        self.assertEqual(_flat_files(self.layout), [])

    def test_b2_conn_next_run_start_and_run_end_keep_the_source_folders(self) -> None:
        """(b) Start-of-run migration and end-of-run archive leave both source folders alone and add no flat file."""

        self._variants("cmp", ("dti_det", "csd_act"))
        _run(self.root, "cmp", "dti_det,csd_act")
        before = {p.relative_to(self.layout.connectome).as_posix(): p.read_bytes()
                  for p in self.layout.connectome.rglob("*") if p.is_file()}
        self.assertTrue(before)
        materialize_canonical_output_tree(self.root, archive_legacy=False)
        archive_development_outputs(self.root, current_run_name="cmp__csd_act", current_tractography_backend="csd_act",
                                    archive_alternate_tractograms=False)
        after = {p.relative_to(self.layout.connectome).as_posix(): p.read_bytes()
                 for p in self.layout.connectome.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertEqual(sorted(p.name for p in self.layout.connectome.iterdir()), ["csd_act", "dti_det"])
        self.assertEqual(_flat_files(self.layout), [])

    def test_b2_conn_saf_child_publishes_into_saf_next_to_csd_act(self) -> None:
        """(e) Flow test: the SAF child (CSD/ACT tracking in its own run) publishes into ``saf/``, not ``csd_act/``."""

        self._variants("cmp", ("csd_act", "saf"))
        tasks = _connectome_tasks(_run(self.root, "cmp", "csd_act,saf"))
        for backend in ("csd_act", "saf"):
            with self.subTest(backend=backend):
                matrix = Path(tasks[f"connectome.streamline_count@{backend}"]["outputs"]["matrix_path"])
                self.assertEqual(matrix, self.layout.connectome / backend / f"{SID}_streamline_count.csv")
                self.assertIn(f"#FROM_{backend}\n", matrix.read_text(encoding="utf-8"))
                sift2 = Path(tasks[f"connectome.sift2_weighted@{backend}"]["outputs"]["matrix_path"])
                self.assertEqual(sift2, self.layout.connectome / backend / f"{SID}_sift2_weighted.csv")
                self.assertIn(f"#FROM_{backend}_sift2", sift2.read_text(encoding="utf-8"))


class SingleTrackingTests(_SubjectCase):
    def test_b2_conn_one_tracking_publishes_into_its_source_folder_and_resume_does_not_recompute(self) -> None:
        """(c) One tracking still gets ``<source>/``; a resumed finished run publishes there again, no recompute."""

        dirs = _write_tracking(self.root, "single", "dti_det")
        _write_connectome_stage(dirs, self.parcel, "dti_det")
        expected = self.layout.connectome / "dti_det" / f"{SID}_streamline_count.csv"
        for attempt in ("first run", "resumed run"):
            with self.subTest(attempt=attempt):
                task = _connectome_tasks(_run(self.root, "single", "dti_det"))["connectome.streamline_count"]
                self.assertEqual((task["status"], task["reason"]), ("skipped", "completed outputs found"))
                self.assertEqual(Path(task["outputs"]["matrix_path"]), expected)
                self.assertEqual(Path(task["outputs"]["nodes_path"]), expected.with_name(f"{SID}_nodes.tsv"))
                self.assertIn("#FROM_dti_det", expected.read_text(encoding="utf-8"))
                self.assertEqual(_flat_files(self.layout), [])
        # The folder alone never makes a finished run compute again: the reuse check reads the stage files.
        expected.unlink()
        task = _connectome_tasks(_run(self.root, "single", "dti_det"))["connectome.streamline_count"]
        self.assertEqual(task["status"], "skipped")
        self.assertTrue(expected.is_file())

    def test_b2_conn_fresh_run_records_the_source_folder_paths(self) -> None:
        """Fresh-run branches (fiber count + SIFT2, and SIFT2 alone) publish into ``csd_act/`` too."""

        computed = []

        def fake_run_connectome(config, log=None):
            out = Path(config.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            matrix = out / f"{SID}-connectome-{config.metric}.csv"
            matrix.write_text(f"computed {config.metric}\n", encoding="utf-8")
            nodes = out / f"{SID}-connectome-nodes.tsv"
            nodes.write_text("node_id\tlabel\tname\n10\t10\tLeft\n", encoding="utf-8")
            computed.append(config.metric)
            return SimpleNamespace(subject_id=SID, output_dir=out, matrix_path=matrix, nodes_path=nodes,
                                   manifest_path=out / "m.json", qc_path=out / "q.json")

        _write_tracking(self.root, "fresh", "csd_act")
        for metrics, expected in ((None, {"connectome.streamline_count": "streamline_count",
                                          "connectome.sift2_weighted": "sift2_weighted"}),
                                  ("sift2", {"connectome.sift2_weighted": "sift2_weighted"})):
            with self.subTest(metrics=metrics):
                computed.clear()
                extra = {} if metrics is None else {"connectome_metrics": metrics}
                tasks = _connectome_tasks(_run(self.root, "fresh", "csd_act", run_connectome=fake_run_connectome,
                                               **extra))
                self.assertTrue(computed)
                self.assertEqual(set(tasks), set(expected))
                for name, short in expected.items():
                    self.assertEqual(tasks[name]["status"], "completed")
                    self.assertEqual(Path(tasks[name]["outputs"]["matrix_path"]),
                                     self.layout.connectome / "csd_act" / f"{SID}_{short}.csv")
                    self.assertEqual(Path(tasks[name]["outputs"]["nodes_path"]),
                                     self.layout.connectome / "csd_act" / f"{SID}_nodes.tsv")
                self.assertEqual(_flat_files(self.layout), [])

    def test_b2_conn_old_flat_files_stay_byte_identical(self) -> None:
        """(d) Results from before this layout are left in place: not moved, not deleted, not rewritten."""

        dirs = _write_tracking(self.root, "single", "dti_det")
        _write_connectome_stage(dirs, self.parcel, "dti_det")
        self.layout.connectome.mkdir(parents=True, exist_ok=True)
        old = {f"{SID}_streamline_count.csv": b"OLD FLAT MATRIX\n", f"{SID}_nodes.tsv": b"OLD FLAT NODES\n",
               f"{SID}_sift2_weighted.csv": b"OLD FLAT SIFT2\n"}
        for name, data in old.items():
            (self.layout.connectome / name).write_bytes(data)
        _run(self.root, "single", "dti_det")
        for name, data in old.items():
            self.assertEqual((self.layout.connectome / name).read_bytes(), data, name)
        self.assertIn("#FROM_dti_det",
                      (self.layout.connectome / "dti_det" / f"{SID}_streamline_count.csv").read_text(encoding="utf-8"))


class PublishFunctionTests(unittest.TestCase):
    def test_b2_conn_publish_needs_a_source(self) -> None:
        """(f) No silent flat publish: the source is required and must not be empty."""

        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp) / "S1")
            stage = Path(tmp) / "stage"
            stage.mkdir()
            (stage / "m.csv").write_text("m\n", encoding="utf-8")
            outputs = {"matrix_path": stage / "m.csv"}
            with self.assertRaises(TypeError):
                _publish_connectome_outputs("streamline_count", outputs, layout, "S1")
            for empty in ("", "  ", None):
                with self.subTest(source=empty), self.assertRaises(ValueError):
                    _publish_connectome_outputs("streamline_count", outputs, layout, "S1", source=empty)
            self.assertFalse(layout.connectome.exists() and any(layout.connectome.rglob("*")))
            published = _publish_connectome_outputs("streamline_count", outputs, layout, "S1", source="ukf")
            self.assertEqual(Path(published["matrix_path"]), layout.connectome / "ukf" / "S1_streamline_count.csv")


class ArchiveAndMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / SID
        self.layout = subject_layout(self.root)

    def test_b2_conn_archive_spares_source_folders_and_still_archives_the_rest(self) -> None:
        """(g) Run end: the four source folders stay; unknown folders and root QC/manifest JSON still go to cache."""

        conn = self.layout.connectome
        sources = ("ukf", "csd_act", "dti_det", "saf")
        for source in sources:
            (conn / source).mkdir(parents=True)
            (conn / source / f"{SID}_streamline_count.csv").write_text(source, encoding="utf-8")
        (conn / "csd_act_mean_fa_check").mkdir()
        (conn / "csd_act_mean_fa_check" / "marker.txt").write_text("x", encoding="utf-8")
        (conn / f"{SID}-connectome-streamline_count_qc.json").write_text("{}", encoding="utf-8")
        (conn / f"{SID}-connectome-streamline_count_manifest.json").write_text("{}", encoding="utf-8")
        (conn / f"{SID}_streamline_count.csv").write_text("old flat", encoding="utf-8")

        moved = archive_development_outputs(self.root)

        legacy = self.layout.cache / "legacy_outputs" / "05_connectome"
        for source in sources:
            self.assertEqual((conn / source / f"{SID}_streamline_count.csv").read_text(encoding="utf-8"), source)
            self.assertFalse((legacy / source).exists())
        self.assertTrue((legacy / "csd_act_mean_fa_check" / "marker.txt").is_file())
        self.assertTrue((legacy / f"{SID}-connectome-streamline_count_qc.json").is_file())
        self.assertTrue((legacy / f"{SID}-connectome-streamline_count_manifest.json").is_file())
        self.assertEqual((conn / f"{SID}_streamline_count.csv").read_text(encoding="utf-8"), "old flat")
        self.assertEqual(len(moved), 3)

    def test_b2_conn_migration_publishes_flat_only_from_a_legacy_tree(self) -> None:
        """Rule 6: current stage caches never become flat files; a legacy 04_tractography tree still does."""

        stage = self.layout.stage_cache("05_connectome")
        current = stage / "r__ukf"
        (current / "mean_fa").mkdir(parents=True)
        (current / f"{SID}-connectome-streamline_count.csv").write_text("CURRENT", encoding="utf-8")
        (current / f"{SID}-connectome-nodes.tsv").write_text("CURRENT NODES", encoding="utf-8")
        (current / "mean_fa" / f"{SID}-connectome-mean_fa.csv").write_text("CURRENT FA", encoding="utf-8")
        published_ukf = self.layout.connectome / "ukf"
        published_ukf.mkdir(parents=True)
        (published_ukf / f"{SID}_sift2_weighted.csv").write_text("PUBLISHED UKF", encoding="utf-8")

        materialize_canonical_output_tree(self.root, archive_legacy=False)
        self.assertEqual(_flat_files(self.layout), [])

        legacy = self.root / "outputs" / "04_tractography" / "connectome" / "csd_act"
        legacy.mkdir(parents=True)
        (legacy / f"{SID}-connectome-streamline_count.csv").write_text("LEGACY", encoding="utf-8")
        old = time.time() - 1000
        os.utime(legacy / f"{SID}-connectome-streamline_count.csv", (old, old))
        migrate_subject_output_tree(self.root, clean_legacy=False)
        self.assertEqual((self.layout.connectome / f"{SID}_streamline_count.csv").read_text(encoding="utf-8"), "LEGACY")
        self.assertEqual(_flat_files(self.layout), [f"{SID}_streamline_count.csv"])
        self.assertEqual((published_ukf / f"{SID}_sift2_weighted.csv").read_text(encoding="utf-8"), "PUBLISHED UKF")

    def test_b2_conn_qc_text_migration_sees_source_folder_files(self) -> None:
        """(h) The migration-time QC text for the connectome also counts files in ``<source>/``."""

        stage = self.layout.stage_cache("05_connectome") / "r__csd_act"
        stage.mkdir(parents=True)
        (stage / f"{SID}-connectome-streamline_count_qc.json").write_text(
            '{"status": "completed", "node_count": 2, "edge_count": 1}\n', encoding="utf-8")
        published = self.layout.connectome / "csd_act"
        published.mkdir(parents=True)
        (published / f"{SID}_streamline_count.csv").write_text("source_node,target_node,streamline_count\n10,20,1\n",
                                                                 encoding="utf-8")
        written = _migrate_qc_from_stage_caches(self.layout, SID)
        self.assertIn("connectome.streamline_count", written)
        self.assertTrue(Path(written["connectome.streamline_count"]).is_file())


if __name__ == "__main__":
    unittest.main()
