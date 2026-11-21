"""Batch 2, item 5: tractography source per bundle algorithm, and one published bundle folder per tractography.

Layout part: every tracking variant publishes its bundles into ``04_bundles/<product>/<backend>/`` (``supplied`` for a
user tractogram file), so a second variant no longer overwrites the first variant's files.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared.io_layout import subject_layout
from shared.orchestrator import ProductOrchestratorConfig, run_subject_products


def _stage(layout, product: str, run_name: str) -> Path:
    folder = {"wmql": "04_bundles/wmql", "wma": "04_bundles/wma_org_atlas", "tractcloud": "04_bundles/tract_cloud_dl"}
    return layout.stage_cache(folder[product]) / run_name


def _fill_wmql(stage: Path, text: str) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "S_cst_left.trk").write_text(text, encoding="utf-8")
    (stage / "wmql_qc.json").write_text("{}", encoding="utf-8")


def _fill_tractcloud(stage: Path, text: str) -> None:
    folder = stage / "S" / "tracts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tract_AF_left.vtp").write_text(text, encoding="utf-8")


def _fill_wma(stage: Path, text: str) -> None:
    folder = stage / "S" / "AnatomicalTracts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "T_AF_left.vtp").write_text(text, encoding="utf-8")
    (stage / "S" / "diffusion_measurements.csv").write_text("name\n" + text + "\n", encoding="utf-8")


def _resume_run(root: Path, *, targets, tracking, bundles, **extra):
    """Run the orchestrator in resume mode (no execution) with every bundle stage reported complete."""

    config = ProductOrchestratorConfig(subject_id="S", subject_root=root, targets=targets, run_name="Test1",
                                       tractography_backend=tracking, bundle_backend=bundles, execute=False,
                                       resume=True, wma_atlas_dir=root, **extra)
    calls = {"wmql": [], "wma": [], "tractcloud": []}

    def spy(name):
        def complete(path, **kwargs):
            calls[name].append(Path(path))
            return True
        return complete

    with mock.patch("shared.orchestrator._wmql_complete", side_effect=spy("wmql")), \
            mock.patch("shared.orchestrator._wma_complete", side_effect=spy("wma")), \
            mock.patch("shared.orchestrator._tractcloud_complete", side_effect=spy("tractcloud")), \
            mock.patch("tractography.run_wma_org_atlas", side_effect=AssertionError("WMA must not rerun")), \
            mock.patch("tractography.run_tractcloud_parcellation", side_effect=AssertionError("TractCloud must not rerun")), \
            mock.patch("tractography.wmql.run_wmql_bundle_segmentation",
                       side_effect=AssertionError("WMQL must not rerun")), \
            contextlib.redirect_stdout(io.StringIO()):
        result = run_subject_products(config)
    return result, calls


def _manifest_output_dirs(result) -> dict:
    """``{task name: output_dir}`` of the bundle tasks in the run manifest (the aggregate one for variants)."""

    record = json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))
    return {task["name"]: (task.get("outputs") or {}).get("output_dir") for task in record.get("tasks", [])
            if task["name"].startswith("bundles.")}


class PublishedLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "S"
        (self.root / "input").mkdir(parents=True)
        self.layout = subject_layout(self.root)

    def test_b2_two_variants_do_not_overwrite_each_others_published_bundles(self) -> None:
        """Item 5 reproduction: CSD/ACT and UKF variants publish WMQL and TractCloud files with the same names."""

        _fill_wmql(_stage(self.layout, "wmql", "Test1__csd_act"), "csd wmql")
        _fill_wmql(_stage(self.layout, "wmql", "Test1__ukf"), "ukf wmql")
        _fill_tractcloud(_stage(self.layout, "tractcloud", "Test1__csd_act"), "csd tc")
        _fill_tractcloud(_stage(self.layout, "tractcloud", "Test1__ukf"), "ukf tc")
        _resume_run(self.root, targets=("tractography", "bundles"), tracking="csd_act,ukf", bundles="wmql,tractcloud")
        for folder, suffix, expected in ((self.layout.query_bundle_extraction, ".trk", {"csd wmql", "ukf wmql"}),
                                         (self.layout.bundle_identification, ".vtp", {"csd tc", "ukf tc"})):
            with self.subTest(folder=folder.name):
                found = {path.read_text(encoding="utf-8") for path in folder.rglob("*" + suffix)}
                self.assertEqual(found, expected)

    def test_b2_bundles_publish_into_one_subfolder_per_tracking_backend(self) -> None:
        _fill_wmql(_stage(self.layout, "wmql", "Test1__csd_act"), "csd wmql")
        _fill_wmql(_stage(self.layout, "wmql", "Test1__ukf"), "ukf wmql")
        _fill_tractcloud(_stage(self.layout, "tractcloud", "Test1__csd_act"), "csd tc")
        _fill_tractcloud(_stage(self.layout, "tractcloud", "Test1__ukf"), "ukf tc")
        _fill_wma(_stage(self.layout, "wma", "Test1__ukf"), "ukf wma")
        result, _ = _resume_run(self.root, targets=("tractography", "bundles", "wma", "tractcloud"),
                                tracking="csd_act,ukf", bundles="wmql,wma,tractcloud")
        layout = self.layout
        self.assertEqual((layout.query_bundle_extraction / "csd_act" / "S_cst_left.trk").read_text(), "csd wmql")
        self.assertEqual((layout.query_bundle_extraction / "ukf" / "S_cst_left.trk").read_text(), "ukf wmql")
        self.assertEqual((layout.bundle_identification / "csd_act" / "tract_AF_left.vtp").read_text(), "csd tc")
        self.assertEqual((layout.bundle_identification / "ukf" / "tract_AF_left.vtp").read_text(), "ukf tc")
        self.assertEqual((layout.fiber_clustering / "ukf" / "T_AF_left.vtp").read_text(), "ukf wma")
        self.assertTrue((layout.fiber_clustering / "ukf" / "bundle_measurements.csv").is_file())
        self.assertEqual(sorted(p.name for p in layout.query_bundle_extraction.iterdir()), ["csd_act", "ukf"])
        self.assertEqual(sorted(p.name for p in layout.fiber_clustering.iterdir()), ["ukf"])
        # The task records point at the variant's own subfolder, so a Results row shows its own variant's files.
        self.assertEqual(_manifest_output_dirs(result), {
            "bundles.wmql@csd_act": (layout.query_bundle_extraction / "csd_act").as_posix(),
            "bundles.wmql@ukf": (layout.query_bundle_extraction / "ukf").as_posix(),
            "bundles.tractcloud@csd_act": (layout.bundle_identification / "csd_act").as_posix(),
            "bundles.tractcloud@ukf": (layout.bundle_identification / "ukf").as_posix(),
            "bundles.wma@ukf": (layout.fiber_clustering / "ukf").as_posix()})

    def test_b2_one_tracking_backend_still_gets_its_subfolder(self) -> None:
        _fill_wmql(_stage(self.layout, "wmql", "Test1"), "dti wmql")
        _resume_run(self.root, targets=("tractography", "bundles"), tracking="dti_det", bundles="wmql")
        self.assertEqual((self.layout.query_bundle_extraction / "dti_det" / "S_cst_left.trk").read_text(), "dti wmql")
        self.assertFalse((self.layout.query_bundle_extraction / "S_cst_left.trk").exists())

    def test_b2_supplied_tractogram_file_publishes_into_supplied(self) -> None:
        supplied = self.root / "mine.vtk"
        supplied.write_text("vtk", encoding="utf-8")
        _fill_tractcloud(_stage(self.layout, "tractcloud", "Test1"), "own tc")
        _fill_wma(_stage(self.layout, "wma", "Test1"), "own wma")
        _resume_run(self.root, targets=("wma", "tractcloud"), tracking="csd_act", bundles="wma,tractcloud",
                    tractcloud_input_tractography=supplied, wma_input_tractography=supplied)
        self.assertEqual((self.layout.bundle_identification / "supplied" / "tract_AF_left.vtp").read_text(), "own tc")
        self.assertEqual((self.layout.fiber_clustering / "supplied" / "T_AF_left.vtp").read_text(), "own wma")

    def test_b2_resume_after_layout_move_republishes_without_rerunning(self) -> None:
        """Resume: the three completion checks read only the stage folder; a finished stage is published again.

        ``_wmql_complete`` gets the stage QC path, ``_wma_complete`` and ``_tractcloud_complete`` get the stage folder.
        None of them reads ``outputs/04_bundles``; the old flat files stay where they are and are not recomputed.
        """

        layout = self.layout
        _fill_wmql(_stage(layout, "wmql", "Test1"), "new wmql")
        _fill_wma(_stage(layout, "wma", "Test1"), "new wma")
        _fill_tractcloud(_stage(layout, "tractcloud", "Test1"), "new tc")
        for folder, name in ((layout.query_bundle_extraction, "S_cst_left.trk"),
                             (layout.fiber_clustering, "T_AF_left.vtp"),
                             (layout.bundle_identification, "tract_AF_left.vtp")):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("old flat", encoding="utf-8")
        result, calls = _resume_run(self.root, targets=("tractography", "bundles", "wma", "tractcloud"),
                                    tracking="ukf", bundles="wmql,wma,tractcloud")
        stage_root = layout.stage_cache("04_bundles")
        self.assertEqual(calls["wmql"], [_stage(layout, "wmql", "Test1") / "wmql_qc.json"])
        self.assertTrue(calls["wma"] and all(path == _stage(layout, "wma", "Test1") for path in calls["wma"]))
        self.assertEqual(calls["tractcloud"], [_stage(layout, "tractcloud", "Test1")])
        for paths in calls.values():
            for path in paths:
                self.assertTrue(path.is_relative_to(stage_root), path)
        statuses = {task.name: task.status for task in result.tasks if task.name.startswith("bundles.")}
        self.assertEqual(statuses, {"bundles.wmql": "skipped", "bundles.wma": "skipped", "bundles.tractcloud": "skipped"})
        self.assertEqual((layout.query_bundle_extraction / "ukf" / "S_cst_left.trk").read_text(), "new wmql")
        self.assertEqual((layout.fiber_clustering / "ukf" / "T_AF_left.vtp").read_text(), "new wma")
        self.assertEqual((layout.bundle_identification / "ukf" / "tract_AF_left.vtp").read_text(), "new tc")
        self.assertEqual((layout.query_bundle_extraction / "S_cst_left.trk").read_text(), "old flat")


def _route(targets, tracking, bundles, sources, **extra):
    """Closure + tracking variants for one selection: (closed algorithms, requesters, notice, {backend: targets})."""

    from shared.multi_tracking import variants
    from shared.orchestrator import _uses_multi_tracking_variants, _with_closure, closure_notice_for
    config = ProductOrchestratorConfig(subject_id="S", subject_root=Path("/tmp/S"), targets=targets, run_name="Test1",
                                       tractography_backend=tracking, bundle_backend=bundles, bundle_sources=sources,
                                       **extra)
    closed = _with_closure(config)
    multi, closed2 = _uses_multi_tracking_variants(closed)
    children = ({child.tractography_backend: child.targets for child in variants(closed2)} if multi
                else {closed2.tractography_backend: closed2.targets})
    requesters = {key: tuple(value) for key, value in closed.closure_algorithm_requesters.items()}
    return closed.tractography_backend, requesters, closure_notice_for(closed), children


BUNDLE_TARGETS = {"bundles", "wma", "tractcloud"}


def _bundle_children(children) -> dict:
    return {backend: tuple(t for t in targets if t in BUNDLE_TARGETS) for backend, targets in children.items()}


class ExplicitSourceRoutingTests(unittest.TestCase):
    """Item 5 routing: which tracking variant runs which bundle algorithm for each explicit source."""

    def test_b2_wmql_source_runs_wmql_only_on_that_variant(self) -> None:
        algos, requesters, _, children = _route(("tractography", "bundles"), "csd_act,ukf,dti_det", "wmql",
                                                {"wmql": "ukf"})
        self.assertEqual(algos, "csd_act,ukf,dti_det")
        self.assertEqual(requesters, {})
        self.assertEqual(_bundle_children(children), {"csd_act": (), "ukf": ("bundles",), "dti_det": ()})

    def test_b2_tractcloud_source_runs_tractcloud_only_on_that_variant(self) -> None:
        _, _, _, children = _route(("tractography", "bundles"), "csd_act,ukf", "wmql,tractcloud",
                                   {"tractcloud": "csd_act"})
        self.assertEqual(_bundle_children(children), {"csd_act": ("bundles", "tractcloud"), "ukf": ("bundles",)})

    def test_b2_wma_source_csd_act_runs_wma_on_csd_without_adding_ukf(self) -> None:
        algos, requesters, notice, children = _route(("tractography", "bundles"), "csd_act", "wma", {"wma": "csd_act"})
        self.assertEqual(algos, "csd_act")
        self.assertEqual(requesters, {})
        self.assertNotIn("UKF", notice)
        self.assertEqual(_bundle_children(children), {"csd_act": ("wma",)})

    def test_b2_wma_source_dti_det_with_ukf_ticked_moves_wma_off_ukf(self) -> None:
        algos, requesters, notice, children = _route(("tractography", "bundles"), "csd_act,ukf", "wma",
                                                     {"wma": "dti_det"})
        self.assertEqual(algos, "csd_act,ukf,dti_det")
        self.assertEqual(requesters, {"dti_det": ("wma",)})
        self.assertIn("DTI tractography (needed by WMA)", notice)
        self.assertEqual(_bundle_children(children), {"csd_act": (), "ukf": (), "dti_det": ("wma",)})
        self.assertEqual(children["dti_det"], ("tractography", "wma"))  # the added variant runs only its requester

    def test_b2_wma_all_runs_once_per_ticked_tracking_and_does_not_add_ukf(self) -> None:
        algos, requesters, _, children = _route(("tractography", "bundles"), "csd_act,dti_det", "wma", {"wma": "all"})
        self.assertEqual(algos, "csd_act,dti_det")
        self.assertEqual(requesters, {})
        self.assertEqual(_bundle_children(children), {"csd_act": ("wma",), "dti_det": ("wma",)})
        _, _, _, children = _route(("tractography", "bundles"), "csd_act,ukf,dti_det", "wma", {"wma": "all"})
        self.assertEqual(_bundle_children(children), {"csd_act": ("wma",), "ukf": ("wma",), "dti_det": ("wma",)})

    def test_b2_wma_all_skips_the_saf_variant(self) -> None:
        _, _, _, children = _route(("tractography", "bundles"), "csd_act,saf", "wma", {"wma": "all"})
        self.assertIn("wma", children["csd_act"])
        self.assertNotIn("wma", children.get("saf", ()))

    def test_b2_not_ticked_source_is_added_with_its_requester_and_named_in_the_note(self) -> None:
        algos, requesters, notice, children = _route(("tractography", "bundles", "connectome"), "csd_act",
                                                     "wmql,tractcloud", {"wmql": "ukf", "tractcloud": "ukf"})
        self.assertEqual(algos, "csd_act,ukf")
        self.assertEqual(requesters, {"ukf": ("bundles", "tractcloud")})
        self.assertIn("UKF tractography (needed by WMQL bundles and TractCloud)", notice)
        self.assertEqual(children["ukf"], ("tractography", "bundles", "tractcloud"))
        self.assertEqual(_bundle_children(children)["csd_act"], ())
        self.assertIn("connectome", children["csd_act"])  # connectome keeps the ticked tracking only

    def test_b2_bundles_only_with_a_not_ticked_source_runs_that_tracking_alone(self) -> None:
        """Like WMA alone today (UKF alone): nothing else reads the ticked tracking, so only the source runs."""

        algos, requesters, notice, children = _route(("bundles",), "csd_act", "wmql", {"wmql": "dti_det"})
        self.assertEqual(algos, "dti_det")
        self.assertEqual(requesters, {"dti_det": ("bundles",)})
        self.assertIn("DTI tractography (needed by WMQL bundles)", notice)
        self.assertEqual(_bundle_children(children), {"dti_det": ("bundles",)})

    def test_b2_wma_auto_next_to_an_explicit_source_still_adds_ukf(self) -> None:
        algos, requesters, _, children = _route(("bundles",), "csd_act", "wmql,wma", {"wmql": "dti_det"})
        self.assertEqual(algos, "ukf,dti_det")
        self.assertEqual(requesters, {"ukf": ("wma",), "dti_det": ("bundles",)})
        self.assertEqual(_bundle_children(children), {"ukf": ("wma",), "dti_det": ("bundles",)})

    def test_b2_source_of_an_algorithm_that_is_not_selected_changes_nothing(self) -> None:
        self.assertEqual(_route(("tractography", "bundles"), "csd_act,ukf", "wmql", {"tractcloud": "dti_det"}),
                         _route(("tractography", "bundles"), "csd_act,ukf", "wmql", {}))

    def test_b2_source_is_ignored_for_a_supplied_tractogram_file(self) -> None:
        with_source = _route(("bundles",), "csd_act", "wma", {"wma": "dti_det"},
                             wma_input_tractography=Path("/tmp/mine.vtk"))
        self.assertEqual(with_source, _route(("bundles",), "csd_act", "wma", {},
                                             wma_input_tractography=Path("/tmp/mine.vtk")))


class WmaOnSelectedSourceTests(unittest.TestCase):
    """Item 5: WMA on a non-UKF tractogram only as an explicit opt-in; the input is the variant's own tractogram."""

    def _plan(self, root: Path, tracking: str, sources: dict):
        from shared.orchestrator import plan_subject_products
        inputs = root / "input"
        inputs.mkdir(exist_ok=True)
        for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
            (inputs / name).write_text("fixture")
        parcel = root / "parcel.nii.gz"
        parcel.write_text("fixture")
        config = ProductOrchestratorConfig("s", root, ("tractography", "bundles"), tractography_backend=tracking,
                                           bundle_backend="wma", bundle_sources=sources, wma_atlas_dir=root,
                                           resume=False)
        with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel), \
                mock.patch("shared.orchestrator._resolve_slicer_executable", return_value=root / "Slicer"):
            return {item.target: item for item in plan_subject_products(config, write=False).items}

    def test_b2_plan_wma_on_csd_act_reads_the_csd_tractogram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = self._plan(Path(tmp), "csd_act", {"wma": "csd_act"})
            self.assertEqual(items["wma"].status, "will_run")
            self.assertEqual(items["wma"].inputs["input_tractography"], items["tractography"].outputs["tractogram_vtk"])
            self.assertTrue(str(items["wma"].inputs["input_tractography"]).endswith("_csd_act_ifod2.vtk"))

    def test_b2_plan_wma_on_dti_det_reads_the_dti_tractogram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = self._plan(Path(tmp), "dti_det", {"wma": "dti_det"})
            self.assertEqual(items["wma"].status, "will_run")
            self.assertEqual(items["wma"].inputs["input_tractography"], items["tractography"].outputs["tractogram_vtk"])

    def test_b2_run_wma_on_dti_det_converts_and_reads_the_current_dti_tractogram(self) -> None:
        from test_product_orchestrator import _write_valid_raw_subject
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_valid_raw_subject(root)
            config = ProductOrchestratorConfig("s", root, ("tractography", "bundles"), tractography_backend="dti_det",
                                               bundle_backend="wma", bundle_sources={"wma": "dti_det"},
                                               wma_atlas_dir=root, execute=True, resume=False)
            received = {}

            def fake_tracking(options, log=None):
                options.output_dir.mkdir(parents=True, exist_ok=True)
                track = options.output_dir / "s-dti-deterministic.trk"
                track.write_bytes(b"tracks")
                manifest = options.output_dir / "s-dti-deterministic_manifest.json"
                qc = options.output_dir / "s-dti-deterministic_qc.json"
                manifest.write_text(json.dumps({"status": "completed"}))
                qc.write_text(json.dumps({"status": "completed", "streamline_count": 10, "streamlines_exists": True,
                                          "length_min_mm": options.min_length_mm,
                                          "length_max_mm": options.max_length_mm}))
                return SimpleNamespace(streamlines_path=track, manifest_path=manifest, qc_path=qc)

            def fake_convert(source, destination):
                received["converted"] = (Path(source), Path(destination))
                destination.write_bytes(b"vtk")
                return destination

            def fake_wma(options, log=None):
                received["wma"] = options.input_tractography
                return SimpleNamespace(status="completed")

            with mock.patch("tractography.dti.run_dti_tracking", side_effect=fake_tracking), \
                    mock.patch("tractography.conversion.ensure_vtk_tractogram", side_effect=fake_convert), \
                    mock.patch("tractography.run_wma_org_atlas", side_effect=fake_wma), \
                    mock.patch("shared.orchestrator._wma_complete", side_effect=lambda *a, **k: "wma" in received), \
                    mock.patch("shared.orchestrator._resolve_slicer_executable", return_value=root / "Slicer"), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_subject_products(config)
            tasks = {task.name: task for task in result.tasks}
            self.assertEqual(tasks["tractography.dti_det"].status, "completed")
            self.assertEqual(tasks["bundles.wma"].status, "completed", tasks["bundles.wma"].reason)
            self.assertTrue(received["wma"].name.endswith("dti-deterministic.vtk"))
            self.assertEqual(received["converted"][1], received["wma"])

    def test_b2_default_wma_on_csd_still_refuses_without_ukf(self) -> None:
        """The refusal stays for the default: WMA never reads CSD/ACT unless the user chose it."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig("s", root, ("bundles",), tractography_backend="csd_act",
                                               bundle_backend="wma", wma_atlas_dir=root, execute=True, resume=False,
                                               excluded_targets=("tractography",))
            with mock.patch("shared.orchestrator._resolve_slicer_executable", return_value=root / "Slicer"), \
                    contextlib.redirect_stdout(io.StringIO()):
                tasks = {task.name: task for task in run_subject_products(config).tasks}
            self.assertEqual(tasks["bundles.wma"].status, "blocked")
            self.assertIn("no silent CSD substitution", tasks["bundles.wma"].reason)


def _cli_plan(argv):
    """Run the subject CLI in plan mode and return the orchestrator config it built."""

    from shared.run_subject_pipeline_cli import main
    captured = []

    def fake_plan(config, write=True):
        captured.append(config)
        return SimpleNamespace(run_plan_path=Path("/tmp/plan.json"), to_dict=lambda: {})

    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("shared.run_subject_pipeline_cli.plan_subject_products", side_effect=fake_plan), \
            contextlib.redirect_stdout(io.StringIO()):
        rc = main(["--no-user-config", "--subject-root", tmp, "--plan", "--skip-environment-check", *argv])
    return rc, captured[0]


class CliBundleSourceTests(unittest.TestCase):
    def test_b2_cli_parses_repeated_bundle_source_into_the_config(self) -> None:
        rc, config = _cli_plan(["--targets", "tractography,bundles", "--tractography-backend", "csd_act",
                                "--bundle-backend", "wmql,wma,tractcloud", "--bundle-source", "wma=csd_act",
                                "--bundle-source", "tractcloud=ukf", "--bundle-source", "wmql=auto"])
        self.assertEqual(rc, 0)
        self.assertEqual(config.bundle_sources, {"wma": "csd_act", "tractcloud": "ukf"})
        self.assertEqual(config.tractography_backend, "csd_act,ukf")
        self.assertEqual(config.closure_algorithm_requesters, {"ukf": ("tractcloud",)})

    def test_b2_cli_without_bundle_source_keeps_an_empty_map(self) -> None:
        rc, config = _cli_plan(["--targets", "bundles", "--bundle-backend", "wma"])
        self.assertEqual((rc, config.bundle_sources, config.tractography_backend), (0, {}, "ukf"))

    def test_b2_cli_rejects_bad_or_conflicting_bundle_sources(self) -> None:
        from shared.run_subject_pipeline_cli import main
        cases = {
            "it cannot be a bundle source": ["--bundle-source", "wmql=saf"],
            "Unknown bundle algorithm 'afq'": ["--bundle-source", "afq=ukf"],
            "Unknown tractography source 'probabilistic' for wmql": ["--bundle-source", "wmql=probabilistic"],
            "names wma twice with different sources": ["--bundle-source", "wma=csd_act", "--bundle-source",
                                                       "wma=dti_det"],
            "expects ALG=SOURCE": ["--bundle-source", "wmql"],
            "Unknown tractography source 'all' for tractcloud": ["--bundle-source", "tractcloud=all"],
            "cannot be combined with --bundle-tractography-backend": ["--bundle-source", "wma=csd_act",
                                                                      "--bundle-tractography-backend", "ukf"],
            "cannot be combined with --wma-input-tractography": ["--bundle-source", "wma=csd_act",
                                                                 "--wma-input-tractography", "/old.vtk"],
            "cannot be combined with --tractcloud-input-tractography": ["--bundle-source", "tractcloud=ukf",
                                                                        "--tractcloud-input-tractography", "/old.vtk"],
            "requires tractography": ["--bundle-source", "wmql=ukf", "--skip-targets", "tractography"],
        }
        for expected, extra in cases.items():
            with self.subTest(case=expected), tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stderr(io.StringIO()) as errors, contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit):
                main(["--no-user-config", "--subject-root", tmp, "--targets", "tractography,bundles",
                      "--bundle-backend", "wmql,wma,tractcloud", "--plan", "--skip-environment-check", *extra])
            self.assertIn(expected, errors.getvalue())

    def test_b2_wma_ukf_is_the_default_and_sends_no_source(self) -> None:
        from shared.bundle_sources import cli_tokens, normalize_bundle_sources
        self.assertEqual(normalize_bundle_sources(["wma=ukf"]), {})
        self.assertEqual(cli_tokens({"wma": "ukf", "wmql": "auto", "tractcloud": "dti_det"}),
                         ["--bundle-source", "tractcloud=dti_det"])

    def test_b2_saved_config_does_not_feed_bundle_sources_to_the_cli(self) -> None:
        """No stale override: the CLI reads the source only from explicit tokens, never from the saved config."""

        from shared.user_config import RUN_DEFAULTS, user_config_cli_values
        self.assertIn("bundle_sources", RUN_DEFAULTS)
        values = user_config_cli_values({"run_defaults": {"bundle_sources": {"wma": "csd_act"}}})
        self.assertNotIn("bundle_sources", values)

    def test_b2_batch_script_forwards_repeated_bundle_source_tokens(self) -> None:
        from scripts.run_batch_pipeline import build_parser
        tokens = ["--bundle-source", "wma=csd_act", "--bundle-source", "wmql=ukf"]
        _, forwarded = build_parser().parse_known_args(["--batch-root", "/tmp/batch", *tokens])
        self.assertEqual(forwarded, tokens)

    def test_b2_experiments_accept_the_bundle_source_parameter(self) -> None:
        from shared.experiments import PARAMETERS
        self.assertIn("bundle_sources", PARAMETERS)

    def test_b2_cohort_planning_counts_the_tracking_a_source_adds(self) -> None:
        from shared.cohort_targets import selected_stages
        base = ["--targets", "tractography,bundles", "--tractography-backend", "csd_act", "--bundle-backend", "wmql,wma"]
        self.assertNotIn("dti_tracking", selected_stages(base))
        self.assertIn("dti_tracking", selected_stages([*base, "--bundle-source", "wmql=dti_det"]))

    def test_b2_experiment_plan_carries_repeated_bundle_sources(self) -> None:
        """A variant and the GUI base arguments both carry --bundle-source into the compiled plan."""

        from shared.experiments import compile_plan, make_request
        with tempfile.TemporaryDirectory() as temp:
            subject = Path(temp) / "subject"
            subject.mkdir()
            base = ["--targets", "validation,tractography,bundles", "--brain-mask-backend", "provided",
                    "--tractography-backend", "csd_act", "--bundle-backend", "wmql,wma",
                    "--bundle-source", "wmql=ukf"]
            variants = json.dumps([{"name": "base", "args": []},
                                   {"name": "wma_dti", "args": ["--bundle-source", "wma=dti_det"]}])
            request = make_request(base_args=base, subject_root=subject, subject_id="subject",
                                   output_root=Path(temp) / "out", run_name="b2", variants=variants)
            plan = compile_plan(request, python_executable=sys.executable, pipeline_script=Path("pipeline.py"))
        effective = {row["variant"]: row["effective_config"] for row in plan["subjects"]}
        self.assertEqual(effective["base"]["bundle_sources"], {"wmql": "ukf"})
        self.assertEqual(effective["wma_dti"]["bundle_sources"], {"wmql": "ukf", "wma": "dti_det"})
        argv = {row["variant"]: row["command"] for row in plan["subjects"]}
        self.assertEqual(_source_tokens(argv["wma_dti"]), ["wmql=ukf", "wma=dti_det"])


def _selection(**extra):
    from shared.ui_run_controller import RunSelection
    values = dict(subject_id="S", subject_root_text="/tmp/b2/S", output_root_text="/tmp/b2/out",
                  selected_targets=("validation", "tractography", "bundles"), run_name="Test1",
                  tractography_backend="csd_act,ukf", bundle_backend="wmql,wma,tractcloud",
                  microstructure_freewater=False, microstructure_noddi=False, select_streamlines=1000,
                  ddseg_device="cpu", surface_device="cpu", connectome_backend="auto",
                  connectome_compress_mrtrix_dense=None, requested_targets=("validation", "tractography", "bundles"))
    values.update(extra)
    return RunSelection(**values)


def _source_tokens(argv) -> list:
    return [argv[index + 1] for index, token in enumerate(argv) if token == "--bundle-source"]


class GuiBuilderTests(unittest.TestCase):
    def build(self, selection):
        from shared.ui_run_controller import (build_background_cli_args, build_batch_background_cli_args,
                                              build_batch_run_kwargs, build_single_subject_run_kwargs)
        root = Path("/tmp/b2")
        with mock.patch("shared.ui_run_controller._ukf_run_options", return_value={}):
            single = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
            batch = build_batch_run_kwargs(selection, batch_root=root / "batch")
            return (single, batch, build_background_cli_args(single, {}, repo_root=root),
                    build_batch_background_cli_args(batch, {}, repo_root=root, batch_root=root / "batch",
                                                    batch_size=1, arrange_inputs="none"))

    def test_b2_both_builders_send_only_explicit_sources(self) -> None:
        sources = {"wmql": "auto", "wma": "dti_det", "tractcloud": "csd_act"}
        single, batch, single_argv, batch_argv = self.build(_selection(bundle_sources=sources))
        self.assertEqual(single["bundle_sources"], {"wma": "dti_det", "tractcloud": "csd_act"})
        self.assertEqual(batch["bundle_sources"], {"wma": "dti_det", "tractcloud": "csd_act"})
        self.assertEqual(_source_tokens(single_argv), ["wma=dti_det", "tractcloud=csd_act"])
        self.assertEqual(_source_tokens(batch_argv), ["wma=dti_det", "tractcloud=csd_act"])

    def test_b2_builders_drop_sources_of_unselected_algorithms(self) -> None:
        _, _, single_argv, batch_argv = self.build(_selection(bundle_backend="wmql",
                                                              bundle_sources={"wma": "all", "wmql": "ukf"}))
        self.assertEqual(_source_tokens(single_argv), ["wmql=ukf"])
        self.assertEqual(_source_tokens(batch_argv), ["wmql=ukf"])

    def test_b2_builders_keep_sources_when_the_bundle_backend_is_all(self) -> None:
        """The node's bundle Choice also offers ``all``: every bundle algorithm is selected then."""

        _, _, single_argv, batch_argv = self.build(_selection(bundle_backend="all",
                                                              bundle_sources={"tractcloud": "ukf"}))
        self.assertEqual(_source_tokens(single_argv), ["tractcloud=ukf"])
        self.assertEqual(_source_tokens(batch_argv), ["tractcloud=ukf"])

    def test_b2_single_run_with_a_wma_input_file_drops_the_wma_source(self) -> None:
        single, _, single_argv, batch_argv = self.build(_selection(
            bundle_sources={"wma": "csd_act"}, wma_input_tractography_text="/tmp/b2/S/in.vtk"))
        self.assertNotIn("bundle_sources", single)
        self.assertEqual(_source_tokens(single_argv), [])
        self.assertIn("--wma-input-tractography", single_argv)
        self.assertEqual(_source_tokens(batch_argv), ["wma=csd_act"])  # batch does not forward the WMA file

    def test_b2_cli_accepts_the_gui_tokens(self) -> None:
        _, _, single_argv, _ = self.build(_selection(bundle_sources={"wma": "all", "tractcloud": "ukf"}))
        rc, config = _cli_plan(["--targets", "tractography,bundles", "--tractography-backend", "csd_act,dti_det",
                                "--bundle-backend", "wma,tractcloud", *single_argv[single_argv.index("--bundle-source"):
                                                                                   single_argv.index("--bundle-source") + 4]])
        self.assertEqual(rc, 0)
        self.assertEqual(config.bundle_sources, {"wma": "all", "tractcloud": "ukf"})

    def test_b2_selected_closure_uses_the_sources(self) -> None:
        from shared.ui_run_controller import ProductTargetFlags, selected_closure
        flags = ProductTargetFlags(**{name: name in {"tractography", "bundles"}
                                      for name in ProductTargetFlags.__dataclass_fields__})
        closure = selected_closure(flags, tractography_backend="csd_act", bundle_backend="wma",
                                   microstructure_freewater=False, microstructure_noddi=False,
                                   bundle_sources={"wma": "dti_det"})
        self.assertEqual(closure.tractography_algorithms, ("csd_act", "dti_det"))
        self.assertEqual(closure.added_algorithms, {"dti_det": ("wma",)})


class ManifestRecordTests(unittest.TestCase):
    def test_b2_run_manifest_records_the_explicit_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "S"
            (root / "input").mkdir(parents=True)
            result, _ = _resume_run(root, targets=("tractography", "bundles"), tracking="csd_act,ukf",
                                    bundles="wmql,tractcloud",
                                    bundle_sources={"tractcloud": "ukf", "wmql": "auto"})
            record = json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(record["bundle_sources"], {"tractcloud": "ukf"})
            for child in record["child_runs"]:
                child_record = json.loads(Path(child["manifest_path"]).read_text(encoding="utf-8"))
                self.assertEqual(child_record["bundle_sources"], {"tractcloud": "ukf"})


class FakeCombo:
    """A QComboBox stand-in: item text, item data, item tooltips and the current index."""

    PRESET: dict = {}  # objectName -> source id the "user" picks before Apply

    def __init__(self, *args):
        self.items, self.tooltips, self.name, self.toolTip, self._index = [], {}, "", "", -1

    def setObjectName(self, name):
        self.name = name

    def addItem(self, text, data=None):
        self.items.append((text, data))

    def setItemData(self, index, value, role=None):
        self.tooltips[index] = value

    def setCurrentIndex(self, index):
        self._index = index

    @property
    def currentIndex(self):
        preset = self.PRESET.get(self.name)
        return [data for _, data in self.items].index(preset) if preset else self._index

    def itemData(self, index):
        return self.items[index][1]

    def setToolTip(self, text):
        self.toolTip = text

    def __getattr__(self, name):  # style calls (setStyleSheet, view, ...) are ignored
        return mock.MagicMock()


def _combo_rows(algorithm, ticked, zh, node=None):
    from shared.ui_bundle_source import add_staged_source_field
    qt = SimpleNamespace(QComboBox=FakeCombo, Qt=SimpleNamespace(ToolTipRole=3))
    form = mock.MagicMock()
    node = node or SimpleNamespace(orchestrationWmqlTractographySource="auto", orchestrationWmaTractographySource="auto",
                                   orchestrationTractcloudTractographySource="auto")
    attribute, staged = add_staged_source_field(qt, form, node, algorithm, zh, ticked)
    label, combo = form.addRow.call_args[0]
    return label, combo, attribute, staged


class SourceComboTests(unittest.TestCase):
    def test_b2_wmql_and_tractcloud_entries_in_english_with_the_adds_suffix(self) -> None:
        for algorithm in ("wmql", "tractcloud"):
            with self.subTest(algorithm=algorithm):
                label, combo, attribute, staged = _combo_rows(algorithm, ("csd_act",), zh=False)
                self.assertEqual(label, "Tractography source")
                self.assertEqual(combo.items, [("All ticked tractography (default)", "auto"), ("CSD/ACT", "csd_act"),
                                               ("UKF (adds this tracking)", "ukf"),
                                               ("DTI (adds this tracking)", "dti_det")])
                self.assertEqual(combo._index, 0)
                self.assertEqual(staged(), "auto")
        self.assertEqual(attribute, "orchestrationTractcloudTractographySource")

    def test_b2_wma_entries_mark_the_non_ukf_route_in_the_tooltip(self) -> None:
        _, combo, attribute, _ = _combo_rows("wma", ("csd_act", "ukf"), zh=False)
        self.assertEqual(attribute, "orchestrationWmaTractographySource")
        self.assertEqual(combo.items, [("UKF (default)", "auto"), ("CSD/ACT", "csd_act"),
                                       ("DTI (adds this tracking)", "dti_det"), ("All ticked tractography", "all")])
        warning = "Not the reference route; UKF is the validated input for WMA."
        self.assertEqual([combo.tooltips[index] == warning for index in range(4)], [False, True, True, True])
        _, combo, _, _ = _combo_rows("wma", ("csd_act",), zh=False)
        self.assertEqual(combo.items[0], ("UKF (default) (adds this tracking)", "auto"))

    def test_b2_chinese_entries_carry_the_same_ids(self) -> None:
        _, combo, _, _ = _combo_rows("wma", ("csd_act",), zh=True)
        self.assertEqual(combo.items, [("UKF（默认）（将自动加入该追踪）", "auto"), ("CSD/ACT", "csd_act"),
                                       ("DTI（将自动加入该追踪）", "dti_det"), ("全部已选纤维追踪", "all")])
        self.assertEqual(combo.tooltips[1], "非参考路线；UKF 是 WMA 经过验证的输入。")
        label, combo, _, _ = _combo_rows("wmql", ("csd_act", "ukf", "dti_det"), zh=True)
        self.assertEqual(label, "纤维追踪来源")
        self.assertEqual(combo.items, [("全部已选纤维追踪（默认）", "auto"), ("CSD/ACT", "csd_act"), ("UKF", "ukf"),
                                       ("DTI", "dti_det")])

    def test_b2_every_caption_has_english_and_chinese(self) -> None:
        from shared import ui_bundle_source as ui
        for name in ("LABEL", "ALL_TICKED_DEFAULT", "ALL_TICKED", "UKF_DEFAULT", "ADDS_TRACKING", "WMA_NOT_REFERENCE"):
            english, chinese = getattr(ui, name)
            self.assertTrue(english and chinese and english != chinese, name)
        for algorithm in ("wmql", "wma", "tractcloud"):
            english, chinese = ui.TOOLTIPS[algorithm]
            self.assertTrue(english and chinese and english != chinese, algorithm)
            self.assertEqual([row[0] for row in ui.source_entries(algorithm, (), False)],
                             [row[0] for row in ui.source_entries(algorithm, (), True)])

    def test_b2_combo_shows_the_stored_value_and_stages_the_pick(self) -> None:
        node = SimpleNamespace(orchestrationWmqlTractographySource="auto", orchestrationWmaTractographySource="dti_det",
                               orchestrationTractcloudTractographySource="auto")
        _, combo, _, staged = _combo_rows("wma", ("csd_act",), zh=False, node=node)
        self.assertEqual(combo.itemData(combo._index), "dti_det")
        combo.setCurrentIndex(3)
        self.assertEqual(staged(), "all")
        self.assertEqual(node.orchestrationWmaTractographySource, "dti_det")  # staged only: Apply writes it

    def test_b2_popup_apply_writes_the_picked_source_to_the_node(self) -> None:
        from Testing.Python import test_ui_algorithm_availability as availability_tests
        original = availability_tests.fake_qt

        def qt_with_combo():
            qt = original()
            qt.QComboBox = FakeCombo
            return qt

        class Popup(availability_tests.PopupBase):
            def runTest(self):  # pragma: no cover - used as a helper only
                pass

        popup = Popup()
        popup.setUp()
        self.addCleanup(popup.doCleanups)
        owner = availability_tests.make_owner(
            targetBundles=True, orchestrationBundleBackend="wmql,wma,tractcloud",
            orchestrationWmqlTractographySource="auto", orchestrationWmaTractographySource="auto",
            orchestrationTractcloudTractographySource="ukf")
        with mock.patch.object(availability_tests, "fake_qt", qt_with_combo), \
                mock.patch.object(FakeCombo, "PRESET", {"bundleSource_wma": "dti_det"}):
            accepted = popup.popup(owner, "bundles", chosen={"wmql", "wma", "tractcloud"})
        self.assertTrue(accepted)
        node = owner._parameterNode
        self.assertEqual((node.orchestrationWmqlTractographySource, node.orchestrationWmaTractographySource,
                          node.orchestrationTractcloudTractographySource), ("auto", "dti_det", "ukf"))
        buttons = {availability_tests.object_name(w) for w in popup.qt.created if w.kind == "QToolButton"}
        self.assertTrue({"algorithmSettings_wmql", "algorithmSettings_wma",
                         "algorithmSettings_tractcloud"} <= buttons)


class PersistenceTests(unittest.TestCase):
    def test_b2_saved_sources_round_trip_and_auto_clears_an_older_save(self) -> None:
        from shared.ui_bundle_source import apply_bundle_sources_to_node, node_bundle_sources, stored_bundle_sources
        from shared.user_config import load_user_config, update_user_config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            update_user_config({"run_defaults": {"bundle_sources": stored_bundle_sources(
                {"wma": "csd_act", "tractcloud": "ukf"})}}, path)
            saved = load_user_config(path)["run_defaults"]["bundle_sources"]
            self.assertEqual(saved, {"wma": "csd_act", "tractcloud": "ukf"})
            node = SimpleNamespace()
            apply_bundle_sources_to_node(node, saved)
            self.assertEqual(node_bundle_sources(node), saved)
            self.assertEqual(node.orchestrationWmqlTractographySource, "auto")
            # The user sets every source back to the default: the older explicit values must not survive the merge.
            update_user_config({"run_defaults": {"bundle_sources": stored_bundle_sources({})}}, path)
            self.assertEqual(load_user_config(path)["run_defaults"]["bundle_sources"], {})

    def test_b2_a_malformed_saved_value_restores_as_auto(self) -> None:
        from shared.ui_bundle_source import apply_bundle_sources_to_node
        from shared.user_config import normalize_user_config
        for saved in ("wma=csd_act", {"wma": "saf", "wmql": "afq", "tractcloud": "ukf"}, None, ["x"]):
            with self.subTest(saved=saved):
                defaults = normalize_user_config({"run_defaults": {"bundle_sources": saved}})["run_defaults"]
                expected = {"tractcloud": "ukf"} if isinstance(saved, dict) else {}
                self.assertEqual(defaults["bundle_sources"], expected)
                node = SimpleNamespace()
                apply_bundle_sources_to_node(node, saved)
                self.assertEqual(node.orchestrationWmaTractographySource, "auto")

    def test_b2_an_old_config_without_the_key_loads_as_every_source_auto(self) -> None:
        from shared.user_config import normalize_user_config
        self.assertEqual(normalize_user_config({"run_defaults": {"bundle_backend": "wma"}})["run_defaults"]
                         ["bundle_sources"], {})

    def test_b2_cli_setup_plan_closure_adds_the_saved_source(self) -> None:
        from scripts import setup_selected
        captured = {}

        def fake_plan(report, **kwargs):
            captured.update(kwargs)
            return {}

        config = {"run_defaults": {"tractography_backend": "csd_act", "bundle_backend": "tractcloud",
                                   "bundle_sources": {"tractcloud": "ukf"}}, "external_dependencies": {}}
        with mock.patch("shared.selected_setup.build_setup_plan", side_effect=fake_plan), \
                mock.patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
            closure, _, _ = setup_selected.build_cli_plan(["tractography", "tractcloud"], config,
                                                          doctor=lambda _config: SimpleNamespace())
        self.assertEqual(closure.tractography_algorithms, ("csd_act", "ukf"))
        self.assertEqual(closure.added_algorithms, {"ukf": ("tractcloud",)})


class ResultsResolutionTests(unittest.TestCase):
    """``result_scene.bundle_outputs``: old flat records stay readable; new records read their own subfolder."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = self.root / "cache" / "stage"
        self.stage.mkdir(parents=True)
        self.stage_file = self.stage / "tract_AF_left.vtp"
        self.stage_file.write_text("own variant", encoding="utf-8")
        self.qc = self.stage / "tractcloud_qc.json"
        self.qc.write_text(json.dumps({"status": "completed", "output_summary": {
            "bundle_files": [self.stage_file.as_posix()]}}), encoding="utf-8")

    def resolve(self, output_dir: Path) -> str:
        from shared import result_scene
        task = {"name": "bundles.tractcloud@csd_act", "qc_path": self.qc.as_posix(),
                "outputs": {"output_dir": output_dir.as_posix()}}
        with mock.patch.object(result_scene, "bundle_record_error", return_value=None):
            return result_scene.bundle_outputs(task)["bundle_tract_AF_left.vtp"]

    def publish(self, folder: Path, *, copy_text=None) -> Path:
        from shared.io_layout import publish_output_file
        folder.mkdir(parents=True, exist_ok=True)
        if copy_text is None:
            return publish_output_file(self.stage_file, folder / self.stage_file.name)
        target = folder / self.stage_file.name
        target.write_text(copy_text, encoding="utf-8")
        return target

    def test_b2_old_flat_layout_record_still_resolves_to_the_flat_file(self) -> None:
        flat = self.root / "outputs" / "04_bundles" / "bundle_identification"
        published = self.publish(flat)
        self.assertEqual(self.resolve(flat), published.as_posix())

    def test_b2_old_flat_layout_without_stage_cache_still_resolves(self) -> None:
        flat = self.root / "outputs" / "04_bundles" / "bundle_identification"
        published = self.publish(flat, copy_text="only copy left")
        self.stage_file.unlink()
        self.assertEqual(self.resolve(flat), published.as_posix())

    def test_b2_new_layout_record_resolves_to_its_backend_subfolder(self) -> None:
        folder = self.root / "outputs" / "04_bundles" / "bundle_identification" / "csd_act"
        published = self.publish(folder)
        self.assertEqual(self.resolve(folder), published.as_posix())

    def test_b2_old_flat_file_overwritten_by_another_variant_falls_back_to_own_stage_file(self) -> None:
        """Item 5: an old flat record whose file a later variant replaced shows its own stage file, not the other one."""

        flat = self.root / "outputs" / "04_bundles" / "bundle_identification"
        self.publish(flat, copy_text="the other variant, longer text")
        self.assertEqual(self.resolve(flat), self.stage_file.as_posix())


class QcSummaryLayoutTests(unittest.TestCase):
    def test_b2_qc_summary_migration_sees_bundle_files_in_backend_subfolders(self) -> None:
        from shared.io_layout import _bundle_files
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "query_bundle_extraction"
            (folder / "ukf").mkdir(parents=True)
            (folder / "ukf" / "a.trk").write_text("x", encoding="utf-8")
            (folder / "old.trk").write_text("x", encoding="utf-8")
            self.assertEqual([path.relative_to(folder).as_posix() for path in _bundle_files(folder, "*.trk")],
                             ["old.trk", "ukf/a.trk"])
            self.assertEqual(_bundle_files(Path(tmp) / "missing", "*.trk"), [])


def _widget_run_case():
    from test_subject_suffix_gui import GuiCase

    class WidgetRunPathTests(GuiCase):
        def test_b2_the_widget_run_button_puts_the_node_sources_into_the_run_selection(self) -> None:
            """``_onRunSelectedClicked`` builds the RunSelection with the node's explicit sources."""

            import shared.ui_run_controller as controller
            from shared.topup_parameters import TopupParameters
            from test_subject_suffix_choice import write_nifti
            from test_ui_algorithm_availability import make_node

            widget = self.widget()
            node = make_node(None, "dti", subjectId="3167", outputDirectory=str(self.out), brainMaskBackend="pnl_cnn",
                             device="auto", ddsegDevice="auto", autoInstallPythonDependencies=False,
                             orchestrationRunName="Test1", orchestrationSelectStreamlines=1000,
                             connectomeBackend="auto", connectomeCompressMrtrixDenseState="unset",
                             brainMaskDevice="auto", pnlMaskRoute="auto", maxParallelModels=1,
                             orchestrationWmqlTractographySource="auto",
                             orchestrationWmaTractographySource="dti_det",
                             orchestrationTractcloudTractographySource="ukf")
            widget._parameterNode = node
            self.set_nifti(write_nifti(self.tmp / "a"))
            self.import_now(widget)
            started: list = []
            widget.logic = SimpleNamespace(planSelectedProductOrchestration=lambda **kw: SimpleNamespace(
                run_plan_path=Path("/x/plan.json"), run_name="Test1", items=[], warnings=[]))
            stubs = dict(
                validateInputs=lambda manual=False: True,
                _input_preview_pending_source=lambda: None,
                _save_common_user_settings=lambda: None,
                _topup_validation=lambda batch_mode=False: SimpleNamespace(errors=[], warnings=[],
                                                                           parameters=TopupParameters()),
                _environment_check_kwargs=lambda **kw: {},
                _environment_check_signature=lambda kw: "sig",
                _cached_environment_result=lambda sig: object(),
                _environment_doctor_result_for_run=lambda kw, sig: SimpleNamespace(statuses=[]),
                _update_run_plan_table=lambda plan: None,
                _summarise_missing=lambda items: "missing " + ",".join(items),
                _startBackgroundPipeline=lambda kwargs, paths, selected: started.append((kwargs, selected)),
                _ifw_echo_time_ms=lambda: None,
                _data_filtered_run=lambda batch_mode: None,
                _selected_workflows=lambda: ["Microstructure"],
            )
            for name, fn in stubs.items():
                setattr(widget, name, fn)
            widget._setupRestartRequired = False
            widget._setupFunctionalFailure = False
            widget._selectedSetupActive = False
            seen: list = []
            original = controller.run_bundle_sources

            def spy(selection, **kwargs):
                seen.append(dict(selection.bundle_sources))
                return original(selection, **kwargs)

            with mock.patch.object(self.mw, "preflight_blockers_for_targets", return_value={}), \
                    mock.patch("shared.setup_resume.pending_setup", return_value=None), \
                    mock.patch("shared.ui_input_inspection.inspect_input_directory",
                               return_value={"mode": "subject_root"}), \
                    mock.patch("shared.ui_compact.sync_detected_input_format"), \
                    mock.patch.object(controller, "run_bundle_sources", spy):
                widget._onRunSelectedClicked()
                self.drive()
            self.assertEqual(len(started), 1, (self.status, self.errors))
            self.assertTrue(seen)
            self.assertEqual(seen[-1], {"wma": "dti_det", "tractcloud": "ukf"})

    return WidgetRunPathTests


WidgetRunPathTests = _widget_run_case()


if __name__ == "__main__":
    unittest.main()
