"""Batch 2 item 5, decision 4: with every bundle source at ``auto`` the routing is today's, by construction.

The fixtures were written by the base commit (3c912ad2, before the source setting existed) with
``python Testing/Python/test_b2_bundle_source_default.py --write <dir>``. They hold, for the investigation report's
truth-table sets (3 ticked sets x 6 tracking sets x 4 bundle sets; 5 tracking x 4 bundle x with/without a WMA input
file for the command lines):

* the closure (targets, tracking algorithms, requesters, the "Also runs automatically" note),
* the tracking variants (run name, backend, targets) and the plan item names,
* the command tokens of both GUI builders (single subject and batch).

The tests rebuild them with the source field left out, with ``{}``, and with every source set to ``auto``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROUTING_FIXTURE = FIXTURES / "b2_bundle_source_default_routing.json"
ARGV_FIXTURE = FIXTURES / "b2_bundle_source_default_argv.json"

TRACK_SETS = ["csd_act", "ukf", "dti_det", "csd_act,ukf", "csd_act,ukf,dti_det", "csd_act,saf"]
BUNDLE_SETS = ["wmql", "wma", "tractcloud", "wmql,wma,tractcloud"]
TICKED = (("tractography", "bundles", "connectome"), ("bundles",), ("tractography", "bundles"))
ARGV_TRACK_SETS = ["csd_act", "ukf", "dti_det", "csd_act,ukf", "csd_act,ukf,dti_det"]
ALL_AUTO = {"wmql": "auto", "wma": "auto", "tractcloud": "auto"}


def routing_cases(source_kwargs: dict) -> list:
    from shared.multi_tracking import variants
    from shared.orchestrator import (ProductOrchestratorConfig, _uses_multi_tracking_variants, _with_closure,
                                     closure_notice_for, plan_subject_products)
    cases = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "S"
        root.mkdir()
        for ticked in TICKED:
            for tracking in TRACK_SETS:
                for bundles in BUNDLE_SETS:
                    config = ProductOrchestratorConfig(subject_id="S", subject_root=root, targets=ticked,
                                                       run_name="Test1", tractography_backend=tracking,
                                                       bundle_backend=bundles, output_root=Path(temp) / "out",
                                                       **source_kwargs)
                    closed = _with_closure(config)
                    multi, closed2 = _uses_multi_tracking_variants(closed)
                    children = ([[c.run_name, c.tractography_backend, list(c.targets), c.bundle_tractography_backend]
                                 for c in variants(closed2)] if multi else
                                [[closed2.run_name, closed2.tractography_backend, list(closed2.targets), None]])
                    items = sorted(item.task_name for item in plan_subject_products(config, write=False).items)
                    cases.append(dict(ticked=list(ticked), tracking=tracking, bundles=bundles,
                                      closed_targets=list(closed.targets),
                                      closed_algorithms=closed.tractography_backend,
                                      requesters={k: list(v) for k, v in closed.closure_algorithm_requesters.items()},
                                      notice=closure_notice_for(closed), multi=multi, children=children,
                                      plan_items=items))
    return cases


def argv_rows(selection_kwargs: dict) -> list:
    from shared.ui_run_controller import (RunSelection, build_background_cli_args, build_batch_background_cli_args,
                                          build_batch_run_kwargs, build_single_subject_run_kwargs)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        placeholders = sorted({root.as_posix(), root.resolve().as_posix()}, key=len, reverse=True)

        def text(token) -> str:
            value = str(token)
            for prefix in placeholders:
                value = value.replace(prefix, "<ROOT>")
            return value

        with mock.patch("shared.ui_run_controller._ukf_run_options", return_value={}), \
                mock.patch.dict("os.environ", {"SLICERDSPACE_PREPROCESSING_EXECUTION_BACKEND": ""}), \
                mock.patch.object(sys, "platform", "darwin"):
            for tracking in ARGV_TRACK_SETS:
                for bundles in BUNDLE_SETS:
                    for with_input in (False, True):
                        wma_text = (root / "S" / "in.vtk").as_posix() if with_input else ""
                        selection = RunSelection(
                            subject_id="S", subject_root_text=(root / "S").as_posix(),
                            output_root_text=(root / "out").as_posix(),
                            selected_targets=("validation", "tractography", "bundles", "connectome"), run_name="Test1",
                            tractography_backend=tracking, bundle_backend=bundles, microstructure_freewater=False,
                            microstructure_noddi=False, select_streamlines=1000, ddseg_device="cpu",
                            surface_device="cpu", connectome_backend="auto", connectome_compress_mrtrix_dense=None,
                            wma_input_tractography_text=wma_text,
                            requested_targets=("validation", "tractography", "bundles", "connectome"),
                            **selection_kwargs)
                        single = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
                        batch = build_batch_run_kwargs(selection, batch_root=root / "batch")
                        rows.append(dict(
                            tracking=tracking, bundles=bundles, wma_input=with_input,
                            single=[text(t) for t in build_background_cli_args(single, {}, repo_root=root)],
                            batch=[text(t) for t in build_batch_background_cli_args(
                                batch, {}, repo_root=root, batch_root=root / "batch", batch_size=1,
                                arrange_inputs="none")]))
    return rows


@unittest.skipIf(sys.platform == "win32", "The fixtures hold POSIX path tokens")
class DefaultEqualsTodayTests(unittest.TestCase):
    maxDiff = None

    def test_b2_default_closure_variants_and_plan_items_equal_the_base_commit(self) -> None:
        expected = json.loads(ROUTING_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(len(expected), 72)
        for label, kwargs in (("field left out", {}), ("empty map", {"bundle_sources": {}}),
                              ("every source auto", {"bundle_sources": dict(ALL_AUTO)})):
            with self.subTest(sources=label):
                self.assertEqual(routing_cases(kwargs), expected)

    def test_b2_default_gui_command_tokens_equal_the_base_commit(self) -> None:
        expected = json.loads(ARGV_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(len(expected), 40)
        for label, kwargs in (("field left out", {}), ("empty map", {"bundle_sources": {}}),
                              ("every source auto", {"bundle_sources": dict(ALL_AUTO)})):
            with self.subTest(sources=label):
                rows = argv_rows(kwargs)
                self.assertEqual(rows, expected)
                self.assertFalse(any("--bundle-source" in row["single"] + row["batch"] for row in rows))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        target = Path(sys.argv[2])
        target.mkdir(parents=True, exist_ok=True)
        (target / ROUTING_FIXTURE.name).write_text(json.dumps(routing_cases({}), indent=1, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        (target / ARGV_FIXTURE.name).write_text(json.dumps(argv_rows({}), indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
    else:
        unittest.main()
