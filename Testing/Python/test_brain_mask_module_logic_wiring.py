"""Shared Slicer logic forwards brain-mask settings without doing inference."""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_previous_slicer = sys.modules.get("slicer")
try:
    sys.modules["slicer"] = types.ModuleType("slicer")
    logic_module = importlib.import_module("shared.module_logic")
finally:
    if _previous_slicer is None:
        sys.modules.pop("slicer", None)
    else:
        sys.modules["slicer"] = _previous_slicer


METHODS = (
    "runInputValidation", "runEnvironmentDoctor", "runEnvironmentDependencyCheck",
    "runSelectedProductOrchestration", "planSelectedProductOrchestration",
)
PATH_KEYS = ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir")


class DummyLogic(logic_module.SharedLogicMixin):
    def __init__(self, module_dir: Path):
        self.module_dir = module_dir

    def log(self, message: str) -> None:
        pass


class BrainMaskModuleLogicWiringTest(unittest.TestCase):
    def test_real_widget_run_keywords_bind_and_forward_device(self):
        from shared.ui_run_controller import RunSelection, build_single_subject_run_kwargs

        selection = RunSelection('fixture', '/subject', '/output', ('microstructure',), 'test', 'csd_act',
                                 'wmql', False, False, 1000000, 'cuda:0', 'auto', 'python', False)
        options = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
        logic = DummyLogic(Path('/repository'))
        for name, callee, value in (
            ('runSelectedProductOrchestration', 'run_subject_products', types.SimpleNamespace(run_manifest_path=Path('/manifest'), tasks=())),
            ('planSelectedProductOrchestration', 'plan_subject_products', types.SimpleNamespace(run_plan_path=Path('/plan'), items=())),
        ):
            with self.subTest(method=name), patch.object(logic_module, callee, return_value=value) as call:
                inspect.signature(getattr(logic, name)).bind(**options)
                getattr(logic, name)(**options)
                self.assertEqual(call.call_args.args[0].tractcloud_device, 'cuda')

    def test_all_entrypoints_have_consistent_typed_defaults(self) -> None:
        for name in METHODS:
            with self.subTest(method=name):
                parameters = inspect.signature(getattr(DummyLogic, name)).parameters
                self.assertEqual(parameters["brain_mask_backend"].default, "pnl_cnn")
                self.assertEqual(parameters["brain_mask_backend"].annotation, "str")
                self.assertEqual(parameters["brain_mask_device"].default, "auto")
                for key in PATH_KEYS:
                    self.assertIsNone(parameters[key].default)
                    self.assertEqual(parameters[key].annotation, "Optional[Path]")
        for name in ("runEnvironmentDoctor", "runEnvironmentDependencyCheck"):
            self.assertIs(inspect.signature(getattr(DummyLogic, name)).parameters["brain_mask_required"].default, False)

    def test_sc1a_every_entrypoint_takes_and_forwards_pnl_mask_route(self) -> None:
        for name in METHODS:
            with self.subTest(method=name):
                self.assertEqual(inspect.signature(getattr(DummyLogic, name)).parameters["pnl_mask_route"].default, "auto")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logic = DummyLogic(root)
            execution = types.SimpleNamespace(run_manifest_path=root / "m.json", tasks=())
            planning = types.SimpleNamespace(run_plan_path=root / "p.json", items=())
            with (patch.object(logic_module, "run_subject_products", return_value=execution) as run,
                  patch.object(logic_module, "plan_subject_products", return_value=planning) as plan):
                logic.runSelectedProductOrchestration(subject_id="s", subject_root=root, targets=("microstructure",),
                                                      pnl_mask_route="native")
                logic.planSelectedProductOrchestration(subject_id="s", subject_root=root, targets=("microstructure",),
                                                       pnl_mask_route="managed")
            self.assertEqual(run.call_args.args[0].pnl_mask_route, "native")
            self.assertEqual(plan.call_args.args[0].pnl_mask_route, "managed")

    def test_validation_plan_doctors_and_execution_forward_all_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logic = DummyLogic(root)
            paths = {key: root / "external tools" / key for key in (*PATH_KEYS, "postprocess_freesurfer_home")}
            validation = types.SimpleNamespace(report_path=None, status="passed", error_count=0, warning_count=0)
            execution = types.SimpleNamespace(run_manifest_path=root / "manifest.json", tasks=())
            planning = types.SimpleNamespace(run_plan_path=root / "plan.json", items=())
            doctor = types.SimpleNamespace()
            for backend in ("pnl_cnn", "synthstrip", "mrtrix", "provided"):
                for name in METHODS:
                    with (
                        self.subTest(method=name, backend=backend),
                        patch.object(logic_module, "validate_subject_inputs", return_value=validation) as validate,
                        patch.object(logic_module, "run_subject_products", return_value=execution) as run,
                        patch.object(logic_module, "plan_subject_products", return_value=planning) as plan,
                        patch.object(logic_module, "run_environment_doctor", return_value=doctor) as check,
                        patch.object(logic_module, "summarize_environment_doctor", return_value="Checked"),
                        patch.object(logic_module, "write_environment_report", return_value={
                            key: root / key for key in ("json", "python_sh", "external_notes")
                        }),
                    ):
                        options = {"brain_mask_backend": backend, "brain_mask_device": "cuda", **paths}
                        if name == "runInputValidation":
                            logic.runInputValidation(subject_id="sub-01", input_root=root, **options)
                            called = validate
                        elif name in ("runEnvironmentDoctor", "runEnvironmentDependencyCheck"):
                            getattr(logic, name)(
                                targets=("microstructure",), subject_root=root,
                                brain_mask_required=True, **options,
                            )
                            called = check
                            self.assertTrue(called.call_args.args[0].brain_mask_required)
                            validate.assert_not_called()
                            run.assert_not_called()
                            plan.assert_not_called()
                        else:
                            getattr(logic, name)(subject_id="sub-01", subject_root=root,
                                                 targets=("microstructure",), **options)
                            called = run if name == "runSelectedProductOrchestration" else plan
                        called.assert_called_once()
                        config = called.call_args.args[0]
                        self.assertEqual(config.brain_mask_backend, backend)
                        self.assertEqual(config.brain_mask_device, "cuda")
                        for key, path in paths.items():
                            self.assertEqual(getattr(config, key), path)

    def test_generic_dependency_check_defaults_to_optional_mask_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(logic_module, "run_environment_doctor") as check,
                patch.object(logic_module, "summarize_environment_doctor", return_value="Checked"),
                patch.object(logic_module, "write_environment_report", return_value={
                    key: root / key for key in ("json", "python_sh", "external_notes")
                }),
            ):
                DummyLogic(root).runEnvironmentDependencyCheck(targets=("microstructure",), output_root=root)
            config = check.call_args.args[0]
            self.assertFalse(config.brain_mask_required)
            self.assertEqual(config.brain_mask_backend, "pnl_cnn")
            for key in PATH_KEYS:
                self.assertIsNone(getattr(config, key))


if __name__ == "__main__":
    unittest.main()
