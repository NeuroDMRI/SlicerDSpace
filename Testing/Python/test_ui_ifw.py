"""T1 D3 GUI route (T-B4 = T-C2) and the three-state connectome compression (T-N1 = T-C6).

Slicer is not available here. The tests run the real widget methods through ``ast`` extraction, the real
``snapshot_selection`` of the experiment panel and the real picker with a stub ``slicer`` module.
"""

from __future__ import annotations

import ast
import contextlib
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.topup_parameters import TopupParameters  # noqa: E402
from shared.ui_run_controller import (  # noqa: E402
    ProductTargetFlags,
    build_background_cli_args,
    build_single_subject_run_kwargs,
    compress_mrtrix_dense_value,
    microstructure_ifw_enabled,
    parse_ifw_echo_time_ms,
    selected_product_targets,
)

WIDGET = REPO_ROOT / "shared" / "module_widget.py"
UI_FILE = REPO_ROOT / "Resources" / "UI" / "SlicerDSpace.ui"
STAGE_COMBOS = [
    "dti", "dti,freewater", "dti,noddi", "dti,dki", "dti,freewater,noddi", "dti,freewater,dki", "dti,noddi,dki",
    "dti,freewater,noddi,dki", "dti,ifw", "dti,freewater,ifw", "dti,noddi,ifw", "dti,dki,ifw",
    "dti,freewater,noddi,ifw", "dti,freewater,dki,ifw", "dti,noddi,dki,ifw", "dti,freewater,noddi,dki,ifw",
]


def widget_methods(*names):
    """The named ``module_widget`` methods, compiled without importing Slicer or Qt."""
    from shared.ui_run_controller import microstructure_ifw_enabled as enabled, parse_ifw_echo_time_ms as parse
    from shared.user_config import canonical_microstructure_stages

    tree = ast.parse(WIDGET.read_text(encoding="utf-8"))
    found = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names}
    namespace = dict(Optional=Optional, microstructure_ifw_enabled=enabled, parse_ifw_echo_time_ms=parse,
                     canonical_microstructure_stages=canonical_microstructure_stages)
    exec(compile(ast.Module(body=[found[name] for name in names], type_ignores=[]), str(WIDGET), "exec"), namespace)
    return [namespace[name] for name in names]


def make_node(**override):
    values = dict(
        orchestrationMicrostructureStages="dti", orchestrationRunFreeWater=False, orchestrationRunNoddi=False,
        orchestrationRunDki=False, orchestrationRunIfw=False, orchestrationIfwEchoTimeMs="",
        inputDirectory="/data/sub-01", outputDirectory="/data/out", orchestrationRunName="Test1",
        orchestrationSelectStreamlines=1000, ddsegDevice="cpu", device="cpu", connectomeBackend="auto",
        connectomeCompressMrtrixDenseState="unset", brainMaskBackend="pnl_cnn",
        brainMaskDevice="auto", orchestrationDkiFitMethod="MKCURVE", orchestrationWmqlPreset="official",
        orchestrationWmqlSelection="", maxParallelModels=1, ukfParameters="{}", wmaInputTractography="",
        experimentScope="subject", preprocessingReversePhaseB0="")
    values.update(override)
    return SimpleNamespace(**values)


def make_owner(node):
    stages_text, ifw_enabled, ifw_te = widget_methods(
        "_microstructure_stages_text", "_microstructure_ifw_enabled", "_ifw_echo_time_ms")
    from shared.ui_run_controller import microstructure_stage_flags

    owner = SimpleNamespace(_parameterNode=node)
    owner._microstructure_stages_text = lambda: stages_text(owner)
    owner._microstructure_ifw_enabled = lambda: ifw_enabled(owner)
    owner._ifw_echo_time_ms = lambda: ifw_te(owner)
    owner._microstructure_dki_enabled = lambda: "dki" in owner._microstructure_stages_text()
    owner._microstructure_stage_flags = lambda: microstructure_stage_flags(owner._microstructure_stages_text())
    owner._topup_validation = lambda batch_mode=False: SimpleNamespace(errors=(), warnings=(), parameters=TopupParameters())
    owner._effective_subject_id = lambda: "sub-01"
    owner._selected_product_targets = lambda: ("validation", "microstructure")
    owner._requested_product_targets = lambda: ("microstructure",)
    owner._tractography_backend_text = lambda: "csd_act"
    owner._closure_tractography_backend_text = lambda: "csd_act"
    owner._bundle_backend_text = lambda: "wmql"
    return owner


def cli_args(selection):
    common = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
    repo = REPO_ROOT
    return build_background_cli_args(common, {}, repo_root=repo)


class StubSlicer:
    """Just enough of ``slicer`` for ``set_algorithms``."""

    @staticmethod
    @contextlib.contextmanager
    def _node_modify(_node):
        yield

    def __init__(self):
        self.util = SimpleNamespace(NodeModify=self._node_modify)


class IfwSelectionTest(unittest.TestCase):
    def test_t_b4_the_picker_offers_ifw_and_sets_the_stage_and_the_run_flag(self):
        from shared.algorithm_selection import LABELS
        from shared.ui_algorithm_picker import FAMILIES, selected_algorithms, set_algorithms

        self.assertIn("ifw", FAMILIES["microstructure"][1])
        self.assertEqual(LABELS["ifw"], "iFW")
        node = make_node()
        calls = []
        owner = SimpleNamespace(_parameterNode=node, _refresh_advanced_run_settings=lambda: calls.append("refresh"),
                                _schedule_common_user_settings_save=lambda: calls.append("save"))
        with patch.dict(sys.modules, {"slicer": StubSlicer()}):
            set_algorithms(owner, "microstructure", ["ifw", "noddi"])
        self.assertEqual(node.orchestrationMicrostructureStages, "dti,noddi,ifw")
        self.assertTrue(node.orchestrationRunIfw)
        self.assertTrue(node.orchestrationRunNoddi)
        self.assertFalse(node.orchestrationRunFreeWater)
        self.assertEqual(selected_algorithms(node, "microstructure"), ("dti", "noddi", "ifw"))
        self.assertEqual(calls, ["refresh", "save"])

    def test_t_b4_the_parameter_node_and_the_hidden_combo_accept_every_ifw_combination(self):
        tree = ast.parse((REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"))
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                      and n.target.id == "orchestrationMicrostructureStages")
        choice = next(c for c in ast.walk(assign.annotation) if isinstance(c, ast.Call))
        self.assertEqual(ast.literal_eval(choice.args[0]), STAGE_COMBOS)
        names = {n.target.id for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        self.assertTrue({"orchestrationRunIfw", "orchestrationIfwEchoTimeMs"} <= names)
        root = ET.parse(UI_FILE).getroot()
        combo = next(w for w in root.iter("widget") if w.get("name") == "orchestrationMicrostructureStagesComboBox")
        items = [i.find("property/string").text for i in combo.findall("item")]
        self.assertEqual(items, STAGE_COMBOS)

    def test_t_b4_the_ui_has_the_ifw_checkbox_the_echo_time_field_and_read_only_t2_values(self):
        from microstructure.ifw import IFW_T2_CSF_MS, IFW_T2_GM_MS

        root = ET.parse(UI_FILE).getroot()
        widgets = {w.get("name"): w for w in root.iter("widget")}
        for name in ("orchestrationIfwCheckBox", "orchestrationIfwEchoTimeLabel", "orchestrationIfwEchoTimeLineEdit",
                     "orchestrationIfwT2TissueLineEdit", "orchestrationIfwT2CsfLineEdit"):
            self.assertIn(name, widgets)
        label = widgets["orchestrationIfwEchoTimeLabel"].find("property/string").text
        self.assertEqual(label, "Echo time (ms)")
        for name, value in (("orchestrationIfwT2TissueLineEdit", IFW_T2_GM_MS),
                            ("orchestrationIfwT2CsfLineEdit", IFW_T2_CSF_MS)):
            properties = {p.get("name"): p for p in widgets[name].findall("property")}
            self.assertEqual(properties["readOnly"].find("bool").text, "true")
            self.assertEqual(float(properties["text"].find("string").text), value)
        self.assertEqual((IFW_T2_GM_MS, IFW_T2_CSF_MS), (95.0, 1500.0))

    def test_t_b4_the_echo_time_field_text_parses_with_empty_as_unset(self):
        self.assertIsNone(parse_ifw_echo_time_ms(""))
        self.assertIsNone(parse_ifw_echo_time_ms("   "))
        self.assertIsNone(parse_ifw_echo_time_ms(None))
        self.assertEqual(parse_ifw_echo_time_ms(" 89.5 "), 89.5)
        self.assertEqual(parse_ifw_echo_time_ms("0"), 0.0)
        for bad in ("abc", "-1", "nan", "inf", "8,9"):
            with self.assertRaises(ValueError, msg=bad):
                parse_ifw_echo_time_ms(bad)

    def test_t_b4_widget_methods_read_the_stage_and_the_echo_time_from_the_node(self):
        owner = make_owner(make_node(orchestrationMicrostructureStages="dti,ifw", orchestrationIfwEchoTimeMs="89.5"))
        self.assertTrue(owner._microstructure_ifw_enabled())
        self.assertEqual(owner._ifw_echo_time_ms(), 89.5)
        owner._parameterNode.orchestrationIfwEchoTimeMs = ""
        self.assertIsNone(owner._ifw_echo_time_ms())
        owner._parameterNode.orchestrationIfwEchoTimeMs = "abc"
        with self.assertRaisesRegex(ValueError, "Echo time"):
            owner._ifw_echo_time_ms()
        # iFW off: a stale value in the field is not sent and a bad one does not stop the run.
        owner._parameterNode.orchestrationMicrostructureStages = "dti,freewater"
        self.assertFalse(owner._microstructure_ifw_enabled())
        self.assertIsNone(owner._ifw_echo_time_ms())
        # Older scenes carry only the legacy flag.
        legacy = make_owner(make_node(orchestrationMicrostructureStages="", orchestrationRunIfw=True))
        self.assertTrue(legacy._microstructure_ifw_enabled())
        self.assertTrue(microstructure_ifw_enabled("", legacy_ifw=True))
        self.assertFalse(microstructure_ifw_enabled("dti,dki"))

    def test_t_b4_selecting_ifw_and_entering_te_reaches_the_run_arguments_of_a_gui_experiment(self):
        from shared.ui_experiments import snapshot_selection

        owner = make_owner(make_node(orchestrationMicrostructureStages="dti,freewater,ifw",
                                     orchestrationIfwEchoTimeMs="89.5"))
        selection = snapshot_selection(owner)
        self.assertTrue(selection.microstructure_ifw)
        self.assertEqual(selection.ifw_echo_time_ms, 89.5)
        args = cli_args(selection)
        self.assertEqual(args[args.index("--ifw-echo-time-ms") + 1], "89.5")
        self.assertEqual(args[args.index("--microstructure-stages") + 1], "dti,freewater,ifw")

    def test_t_b4_an_empty_echo_time_stays_unset_in_the_run_arguments(self):
        from shared.ui_experiments import snapshot_selection

        owner = make_owner(make_node(orchestrationMicrostructureStages="dti,freewater,ifw"))
        selection = snapshot_selection(owner)
        self.assertTrue(selection.microstructure_ifw)
        self.assertIsNone(selection.ifw_echo_time_ms)
        args = cli_args(selection)
        self.assertNotIn("--ifw-echo-time-ms", args)
        self.assertIn("ifw", args[args.index("--microstructure-stages") + 1].split(","))

    def test_t_b4_a_bad_echo_time_stops_the_experiment_snapshot_with_a_message(self):
        from shared.ui_experiments import snapshot_selection

        owner = make_owner(make_node(orchestrationMicrostructureStages="dti,ifw", orchestrationIfwEchoTimeMs="x"))
        with self.assertRaisesRegex(ValueError, "Echo time"):
            snapshot_selection(owner)

    def test_t_b4_the_closure_of_an_ifw_selection_adds_free_water_and_segmentation(self):
        flags = ProductTargetFlags(microstructure=True)
        targets = selected_product_targets(
            flags, tractography_backend="csd_act", bundle_backend="wmql", microstructure_freewater=False,
            microstructure_noddi=False, microstructure_ifw=True)
        for name in ("microstructure_ifw", "microstructure_freewater", "segmentation"):
            self.assertIn(name, targets)
        plain = selected_product_targets(flags, tractography_backend="csd_act", bundle_backend="wmql",
                                         microstructure_freewater=False, microstructure_noddi=False)
        self.assertNotIn("microstructure_ifw", plain)

    def test_t_b4_experiment_variants_may_carry_the_ifw_flags(self):
        from shared.experiments import parameter_args

        self.assertEqual(parameter_args(["--ifw-echo-time-ms", "60"]), ["--ifw-echo-time-ms", "60"])
        self.assertEqual(parameter_args(["--run-ifw"]), ["--run-ifw"])

    def test_t_b4_saved_settings_carry_the_echo_time(self):
        from shared.ui_settings import coerce_user_setting_value, common_user_setting_specs

        self.assertIn(("orchestrationIfwEchoTimeMs", str), common_user_setting_specs())
        self.assertEqual(coerce_user_setting_value(None, str), "")
        self.assertEqual(coerce_user_setting_value("89.5", str), "89.5")

    def test_t_b4_guard_the_run_click_passes_ifw_into_the_run_selection(self):
        # Guard (label): _onRunSelectedClicked needs Slicer; check the call site by its source.
        tree = ast.parse(WIDGET.read_text(encoding="utf-8"))
        click = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_onRunSelectedClicked")
        selection = next(c for c in ast.walk(click) if isinstance(c, ast.Call)
                         and getattr(c.func, "id", "") == "RunSelection")
        keywords = {k.arg for k in selection.keywords}
        self.assertTrue({"microstructure_ifw", "ifw_echo_time_ms", "connectome_compress_mrtrix_dense"} <= keywords)

    def test_t_b4_compact_label_lists_ifw(self):
        from shared.ui_compact import compact_algorithm_labels

        label = compact_algorithm_labels(make_node(orchestrationMicrostructureStages="dti,freewater,ifw"))
        self.assertEqual(label["targetMicrostructureCheckBox"], "DTI + FW + iFW")


class CompressionStateTest(unittest.TestCase):
    def test_t_n1_three_states_map_to_none_true_false(self):
        self.assertIsNone(compress_mrtrix_dense_value("unset"))
        self.assertIsNone(compress_mrtrix_dense_value(""))
        self.assertIsNone(compress_mrtrix_dense_value(None))
        self.assertIs(compress_mrtrix_dense_value("on"), True)
        self.assertIs(compress_mrtrix_dense_value("off"), False)

    def test_t_n1_each_state_reaches_the_run_arguments(self):
        from shared.ui_experiments import snapshot_selection

        expected = {"unset": (None, None), "on": (True, "--compress-mrtrix-dense"),
                    "off": (False, "--no-compress-mrtrix-dense")}
        for state, (value, flag) in expected.items():
            with self.subTest(state=state):
                selection = snapshot_selection(make_owner(make_node(connectomeCompressMrtrixDenseState=state)))
                self.assertIs(selection.connectome_compress_mrtrix_dense, value)
                args = cli_args(selection)
                self.assertEqual("--compress-mrtrix-dense" in args, flag == "--compress-mrtrix-dense")
                self.assertEqual("--no-compress-mrtrix-dense" in args, flag == "--no-compress-mrtrix-dense")

    def test_t_n1_node_and_saved_settings_store_the_state_as_text(self):
        from shared.ui_settings import common_user_setting_specs

        tree = ast.parse((REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"))
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                      and n.target.id == "connectomeCompressMrtrixDenseState")
        self.assertEqual(ast.literal_eval(assign.value), "unset")
        choice = next(c for c in ast.walk(assign.annotation) if isinstance(c, ast.Call))
        self.assertEqual(ast.literal_eval(choice.args[0]), ["unset", "on", "off"])
        self.assertIn(("connectomeCompressMrtrixDenseState", str), common_user_setting_specs())
        self.assertNotIn("connectomeCompressMrtrixDense", {name for name, _ in common_user_setting_specs()})

    def test_t_n1_guard_the_run_click_uses_the_state_not_a_boolean(self):
        # Guard (label): the old code forwarded True or None only, so "off" could not be sent.
        source = WIDGET.read_text(encoding="utf-8")
        self.assertIn("compress_mrtrix_dense_value(", source)
        self.assertNotIn("connectomeCompressMrtrixDense ", source)
        self.assertNotIn("True if self._parameterNode.connectomeCompressMrtrixDense", source)


if __name__ == "__main__":
    unittest.main()
