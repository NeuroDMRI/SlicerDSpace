"""SC-1a / SC-1b: the pnl_mask_route setting through the user config, CLI, GUI, logic layers, manifest and run record."""

from __future__ import annotations

import ast
import inspect
import json
import tempfile
import unittest
import unittest.mock
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from shared import run_subject_pipeline_cli as cli
from shared.brain_mask import PNL_NATIVE_EXPERIMENTAL_REASON, pnl_route_record
from shared.run_record import build_run_record
from shared.ui_onboarding import onboarding_text
from shared.ui_run_controller import RunSelection, build_background_cli_args, build_single_subject_run_kwargs
from shared.ui_workflow_text import PNL_MASK_ROUTE_TEXT_KEYS
from shared.user_config import (
    apply_user_config_to_args,
    empty_user_config,
    normalize_user_config,
    user_config_cli_values,
    user_config_from_cli_args,
)

ROOT = Path(__file__).resolve().parents[2]


class PnlMaskRouteWiringTest(unittest.TestCase):
    def test_user_config_default_roundtrip_and_unknown_value(self) -> None:
        self.assertEqual(empty_user_config()["run_defaults"]["pnl_mask_route"], "auto")
        config = normalize_user_config({"run_defaults": {"pnl_mask_route": "native"}})
        self.assertEqual(user_config_cli_values(config)["pnl_mask_route"], "native")
        args = SimpleNamespace(pnl_mask_route="managed")
        saved = user_config_from_cli_args(args)
        self.assertEqual(saved["run_defaults"]["pnl_mask_route"], "managed")
        bad = normalize_user_config({"run_defaults": {"pnl_mask_route": "gpu"}})
        self.assertEqual(bad["run_defaults"]["pnl_mask_route"], "auto")

    def test_saved_value_applies_unless_the_cli_option_is_explicit(self) -> None:
        config = normalize_user_config({"run_defaults": {"pnl_mask_route": "native"}})
        args = SimpleNamespace(pnl_mask_route="auto")
        apply_user_config_to_args(args, config, explicit_dests=set())
        self.assertEqual(args.pnl_mask_route, "native")
        args = SimpleNamespace(pnl_mask_route="managed")
        apply_user_config_to_args(args, config, explicit_dests={"pnl_mask_route"})
        self.assertEqual(args.pnl_mask_route, "managed")

    def test_cli_option_default_choices_and_help(self) -> None:
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args([]).pnl_mask_route, "auto")
        for route in ("auto", "managed", "native"):
            self.assertEqual(parser.parse_args(["--pnl-mask-route", route]).pnl_mask_route, route)
        with self.assertRaises(SystemExit), unittest.mock.patch("sys.stderr"):
            parser.parse_args(["--pnl-mask-route", "gpu"])
        action = next(a for a in parser._actions if "--pnl-mask-route" in a.option_strings)
        self.assertIn("native (experimental)", action.help)
        self.assertIn(PNL_NATIVE_EXPERIMENTAL_REASON, " ".join(action.help.split()))

    def test_run_controller_forwards_the_route_to_kwargs_and_background_cli(self) -> None:
        selection = RunSelection("s", "/subject", "/out", ("microstructure",), "t", "csd_act", "wmql", False, False,
                                 1000000, "cuda:0", "auto", "python", False, pnl_mask_route="native")
        kwargs = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
        self.assertEqual(kwargs["pnl_mask_route"], "native")
        args = build_background_cli_args(selection, {}, repository_root=ROOT) if "repository_root" in \
            inspect.signature(build_background_cli_args).parameters else None
        if args is not None:
            self.assertEqual(args[args.index("--pnl-mask-route") + 1], "native")

    def test_gui_route_selector_is_declared_and_texts_exist(self) -> None:
        tree = ET.parse(ROOT / "Resources/UI/SlicerDSpace.ui")
        classes = {w.get("name"): w.get("class") for w in tree.iter("widget")}
        self.assertEqual(classes["pnlMaskRouteComboBox"], "QComboBox")
        self.assertEqual(classes["pnlMaskRouteLabel"], "QLabel")
        self.assertEqual(list(PNL_MASK_ROUTE_TEXT_KEYS), ["auto", "managed", "native"])
        for language in ("en", "zh"):
            self.assertIn("native", onboarding_text(language, PNL_MASK_ROUTE_TEXT_KEYS["native"]))
        self.assertEqual(onboarding_text("en", "pnl_route_native"), "native (experimental)")
        self.assertIn("native (experimental)", onboarding_text("en", "pnl_mask_route_tooltip"))
        source = ast.parse((ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"))
        node = next(item for item in source.body if isinstance(item, ast.ClassDef)
                    and item.name == "SlicerDSpaceParameterNode")
        defaults = {item.target.id: ast.literal_eval(item.value) for item in node.body if isinstance(item, ast.AnnAssign)}
        self.assertEqual(defaults["pnlMaskRoute"], "auto")

    def test_run_record_carries_the_route_and_the_experimental_marker(self) -> None:
        for route, name, experimental in (("managed", "managed-ants", False), ("native", "native-sitk", True)):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                manifest = Path(tmp) / "run_manifest.json"
                manifest.write_text(json.dumps({
                    "kind": "slicerddsurfer_product_orchestration", "subject_id": "sub-01",
                    "brain_mask_backend": "pnl_cnn", "pnl_mask_route": "auto",
                    "brain_mask_route": pnl_route_record(route),
                }), encoding="utf-8")
                record = build_run_record(manifest)
                mask = record["brain_mask"]
                self.assertEqual(mask["route"], name)
                self.assertEqual(mask["backend"], "pnl_cnn")
                if experimental:
                    self.assertEqual(mask["status"], "experimental")
                    self.assertEqual(mask["reason"], PNL_NATIVE_EXPERIMENTAL_REASON)
                else:
                    self.assertNotIn("status", mask)

    def test_sc1d_doctor_config_selects_the_route_checks(self) -> None:
        from shared.brain_mask import brain_mask_config_from_options, inspect_brain_mask_backend
        from shared.environment import EnvironmentDoctorConfig
        for route, system, name, wants_conda in (("auto", "Linux", "managed-ants", True), ("managed", "Darwin", "managed-ants", True),
                                                 ("auto", "Darwin", "native-sitk", False), ("native", "Linux", "native-sitk", False)):
            with self.subTest(route=route, system=system):
                config = EnvironmentDoctorConfig(repo_root=ROOT, targets=("preprocessing",), brain_mask_backend="pnl_cnn",
                                                 brain_mask_required=True, pnl_mask_route=route)
                found = inspect_brain_mask_backend(brain_mask_config_from_options(config), system=system)
                self.assertEqual(found["settings"]["pnl_route"], name)
                missing = " ".join(str(item) for item in found["missing_required"])
                self.assertEqual("pnl_mask_python" in missing, wants_conda)

    def test_manifest_without_a_pnl_route_has_no_brain_mask_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "run_manifest.json"
            manifest.write_text(json.dumps({"subject_id": "s", "brain_mask_backend": "synthstrip"}), encoding="utf-8")
            self.assertNotIn("brain_mask", build_run_record(manifest))


if __name__ == "__main__":
    unittest.main()
