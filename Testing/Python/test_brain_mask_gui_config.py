"""Brain-mask GUI/config contracts without requiring a Slicer runtime."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Optional
import unittest
import xml.etree.ElementTree as ET

from shared.dependency_config import (
    UI_DEPENDENCY_PATHS,
    add_dependency_config_arguments,
    apply_dependency_config,
    load_dependency_config,
    write_dependency_config,
)
from shared.ui_environment_panel import environment_path_entries
from shared.ui_onboarding import apply_onboarding_language, dependency_ui_text, onboarding_text
from shared.ui_parameter_sync import sync_parameter_node_to_ui
from shared.ui_run_controller import (
    RunSelection,
    build_background_cli_args,
    build_batch_background_cli_args,
    build_batch_run_kwargs,
    build_single_subject_run_kwargs,
)
from shared.ui_settings import common_user_setting_specs
from shared.ui_input_inspection import inspect_input_directory
from shared.ui_workflow_text import BRAIN_MASK_BACKEND_TEXT_KEYS
from shared.user_config import (
    apply_user_config_to_args,
    empty_user_config,
    explicit_cli_destinations,
    load_user_config,
    update_user_config,
    user_config_cli_values,
    user_config_from_cli_args,
    write_user_config,
)


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("pnl_cnn", "provided")
PNL_MODES = {
    "pnl_mask_python": "file",
    "pnl_mask_script": "file",
    "pnl_mask_model_dir": "dir",
    "pnl_mask_ants_bin_dir": "dir",
}


def widget_method(name, **namespace):
    """Exercise widget methods in isolation, following the layout-contract tests."""
    import os
    import sys

    path = ROOT / "shared/module_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    from shared.subject_id import effective_subject_id
    from shared.ui_run_state import gui_brain_mask_required
    if "sys" in namespace and not hasattr(namespace["sys"], "platform"):
        namespace["sys"].platform = sys.platform
    namespace.update(Any=Any, Optional=Optional, Path=Path, os=os, effective_subject_id=effective_subject_id,
                     gui_brain_mask_required=gui_brain_mask_required)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FakeCombo:
    def __init__(self):
        self.items = [[onboarding_text("en", key), value] for value, key in BRAIN_MASK_BACKEND_TEXT_KEYS.items()]
        self.currentIndex = 1
        self.blocked = False

    def findData(self, value):
        return next((index for index, item in enumerate(self.items) if item[1] == value), -1)

    def itemData(self, index):
        return self.items[index][1]

    def setItemText(self, index, text):
        self.items[index][0] = text

    def blockSignals(self, blocked):
        previous = self.blocked
        self.blocked = blocked
        return previous

    def setToolTip(self, text):
        self.tooltip = text


class BrainMaskGuiConfigTest(unittest.TestCase):
    def test_parameter_node_default_and_choices(self):
        tree = ast.parse((ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"))
        node = next(item for item in ast.walk(tree) if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name) and item.target.id == "brainMaskBackend")
        self.assertEqual(ast.literal_eval(node.value), "pnl_cnn")
        choice = next(item for item in ast.walk(node.annotation) if isinstance(item, ast.Call)
                      and isinstance(item.func, ast.Name) and item.func.id == "Choice")
        self.assertEqual(ast.literal_eval(choice.args[0]), list(BACKENDS))

    def test_w5_gui_and_onboarding_offer_exactly_pnl_cnn_and_provided(self):
        """W5 (D2): the Slicer node, the .ui combo and the onboarding table offer only pnl_cnn and provided."""
        self.assertEqual(list(BRAIN_MASK_BACKEND_TEXT_KEYS), ["pnl_cnn", "provided"])
        root = ET.parse(ROOT / "Resources/UI/SlicerDSpace.ui").getroot()
        combo = root.find(".//widget[@name='brainMaskBackendComboBox']")
        labels = [item.find("property[@name='text']/string").text for item in combo.findall("item")]
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels, [onboarding_text("en", key) for key in BRAIN_MASK_BACKEND_TEXT_KEYS.values()])
        fake = FakeCombo()
        apply_onboarding_language(SimpleNamespace(ui=SimpleNamespace(brainMaskBackendComboBox=fake)), "zh")
        self.assertEqual([item[1] for item in fake.items], ["pnl_cnn", "provided"])
        for language in ("en", "zh"):
            for removed in ("synthstrip", "mrtrix", "dipy", "SynthStrip", "dwi2mask", "DIPY"):
                for key in BRAIN_MASK_BACKEND_TEXT_KEYS.values():
                    self.assertNotIn(removed, onboarding_text(language, key))
                self.assertNotIn(removed, onboarding_text(language, "brain_mask_backend_tooltip"))

    def test_w5_saved_removed_backend_loads_as_pnl_cnn_with_one_note(self):
        """W5 (SC-2 rule 2): a saved synthstrip, mrtrix or dipy loads as pnl_cnn and the note says so once."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            for backend in ("synthstrip", "mrtrix", "dipy"):
                with self.subTest(backend=backend):
                    path.write_text(json.dumps({"run_defaults": {"brain_mask_backend": backend}}), encoding="utf-8")
                    with self.assertLogs("shared.user_config", level="WARNING") as logged:
                        saved = load_user_config(path)
                    self.assertEqual(saved["run_defaults"]["brain_mask_backend"], "pnl_cnn")
                    self.assertEqual(len(saved["notes"]), 1)
                    self.assertIn(backend, saved["notes"][0])
                    self.assertIn("pnl_cnn", saved["notes"][0])
                    self.assertEqual(len(logged.records), 1)
                    # The note is not saved: the next load is silent.
                    write_user_config(saved, path)
                    again = load_user_config(path)
                    self.assertEqual(again["run_defaults"]["brain_mask_backend"], "pnl_cnn")
                    self.assertNotIn("notes", again)
                    self.assertEqual(user_config_cli_values(saved)["brain_mask_backend"], "pnl_cnn")

    def test_selector_lives_in_analysis_parameters_not_products_or_inputs(self):
        root = ET.parse(ROOT / "Resources/UI/SlicerDSpace.ui").getroot()
        section = root.find(".//widget[@name='parametersCollapsibleButton']")
        combo = section.find(".//widget[@name='brainMaskBackendComboBox']")
        self.assertIsNotNone(combo)
        labels = [item.find("property[@name='text']/string").text for item in combo.findall("item")]
        self.assertEqual(labels, [onboarding_text("en", key) for key in BRAIN_MASK_BACKEND_TEXT_KEYS.values()])
        self.assertEqual(labels[0], "PNL CNN (dMRI)")
        # Readable labels must not be bound as Choice values by connectGui.
        self.assertIsNone(combo.find("property[@name='SlicerParameterName']"))
        for name in ("configurationCollapsibleButton", "targetsCollapsibleButton"):
            self.assertIsNone(root.find(f".//widget[@name='{name}']//widget[@name='brainMaskBackendComboBox']"))
        layout = (ROOT / "shared/ui_compact.py").read_text(encoding="utf-8")
        self.assertIn('"parametersCollapsibleButton", "resultsCollapsibleButton"', layout)
        self.assertIn('layout.insertWidget(layout.indexOf(advanced) + 1, environment)', layout)
        self.assertIn('"parametersCollapsibleButton": "compact_parameters"', layout)

    def test_user_config_roundtrip_preserves_each_backend_and_external_paths(self):
        self.assertEqual(empty_user_config()["run_defaults"]["brain_mask_backend"], "pnl_cnn")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            external = {key: str(Path(tmp) / key) for key in PNL_MODES}
            for backend in BACKENDS:
                with self.subTest(backend=backend):
                    write_user_config({"run_defaults": {"brain_mask_backend": backend},
                                       "external_dependencies": external}, path)
                    update_user_config({"ui": {"language": "zh"}}, path)
                    saved = load_user_config(path)
                    self.assertEqual(saved["run_defaults"]["brain_mask_backend"], backend)
                    values = user_config_cli_values(saved)
                    self.assertEqual(values["brain_mask_backend"], backend)
                    for key, value in external.items():
                        self.assertEqual(values[key], Path(value))
                    args = argparse.Namespace(**values)
                    rebuilt = user_config_from_cli_args(args, base_config=saved)
                    self.assertEqual(rebuilt["run_defaults"]["brain_mask_backend"], backend)
                    self.assertEqual(rebuilt["external_dependencies"], external)

    def test_old_config_gets_pnl_default_and_unknown_backend_is_not_silently_replaced(self):
        self.assertEqual(user_config_cli_values({"run_defaults": {}})["brain_mask_backend"], "pnl_cnn")
        self.assertEqual(user_config_cli_values({"run_defaults": {"brain_mask_backend": "unknown"}})
                         ["brain_mask_backend"], "unknown")

    def test_explicit_cli_selector_and_paths_override_saved_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--brain-mask-backend", choices=BACKENDS, default="pnl_cnn")
        add_dependency_config_arguments(parser)
        argv = ["--brain-mask-backend", "provided", "--pnl-mask-python", "/explicit/python"]
        args = parser.parse_args(argv)
        apply_user_config_to_args(args, {"run_defaults": {"brain_mask_backend": "pnl_cnn"},
                                        "external_dependencies": {"pnl_mask_python": "/saved/python"}},
                                  explicit_dests=explicit_cli_destinations(parser, argv))
        self.assertEqual(args.brain_mask_backend, "provided")
        self.assertEqual(args.pnl_mask_python, Path("/explicit/python"))

    def test_dependency_paths_modes_environment_placement_and_config_roundtrip(self):
        rows = {item["id"]: item for item in UI_DEPENDENCY_PATHS}
        entries = {entry.item["id"] for entry in environment_path_entries(UI_DEPENDENCY_PATHS)
                   if entry.kind == "row"}
        with tempfile.TemporaryDirectory() as tmp:
            expected = {key: Path(tmp) / key for key in PNL_MODES}
            parser = argparse.ArgumentParser()
            add_dependency_config_arguments(parser)
            argv = [token for key, path in expected.items() for token in ("--" + key.replace("_", "-"), str(path))]
            args = parser.parse_args(argv)
            config = write_dependency_config(args, Path(tmp) / "dependencies.json")
            self.assertEqual(load_dependency_config(config), expected)
            restored = argparse.Namespace(pnl_mask_python=Path("/explicit/python"))
            apply_dependency_config(restored, config)
            self.assertEqual(restored.pnl_mask_python, Path("/explicit/python"))
            for key, mode in PNL_MODES.items():
                self.assertIn(key, entries)
                self.assertEqual(rows[key]["mode"], mode)
                self.assertEqual(rows[key]["section"], "core-runtime")
                self.assertNotIn("value_type", rows[key])
                for language in ("en", "zh"):
                    self.assertNotEqual(dependency_ui_text(language, key, "label"), key)
                    self.assertTrue(dependency_ui_text(language, key, "tooltip"))

    def test_language_switch_preserves_canonical_selection_and_blocked_signals(self):
        combo = FakeCombo()
        owner = SimpleNamespace(ui=SimpleNamespace(brainMaskBackendComboBox=combo))
        for language in ("zh", "en"):
            for initially_blocked in (False, True):
                combo.blocked = initially_blocked
                apply_onboarding_language(owner, language)
                self.assertEqual(combo.currentIndex, 1)
                self.assertEqual(combo.itemData(1), "provided")
                self.assertEqual(combo.items[1][0], onboarding_text(language, "brain_mask_provided"))
                self.assertEqual(combo.items[0][0], "PNL CNN (dMRI)")
                self.assertEqual(combo.blocked, initially_blocked)
        self.assertIn("supplied masks", onboarding_text("en", "brain_mask_backend_tooltip"))
        self.assertIn("no automatic fallback", combo.tooltip)

    def test_combo_handler_persists_item_data_not_display_text(self):
        combo = FakeCombo()
        updates = []
        shown = []
        owner = SimpleNamespace(ui=SimpleNamespace(brainMaskBackendComboBox=combo),
                                _update_parameter_node=lambda *args: updates.append(args),
                                _update_pnl_route_visibility=shown.append,
                                _lastEnvironmentResult=object(), _lastEnvironmentSignature=("old",))
        changed = widget_method("_onBrainMaskBackendChanged", BRAIN_MASK_BACKEND_TEXT_KEYS=BRAIN_MASK_BACKEND_TEXT_KEYS)
        for index, backend in enumerate(BACKENDS):
            changed(owner, index)
            self.assertEqual(updates[-1], ("brainMaskBackend", backend))
            self.assertEqual(shown[-1], backend)  # SC-1b: the route selector follows the backend
            self.assertIsNone(owner._lastEnvironmentResult)
            self.assertIsNone(owner._lastEnvironmentSignature)
        changed(owner, -1)
        self.assertEqual(len(updates), len(BACKENDS))
        specs = widget_method("_common_user_setting_specs", common_user_setting_specs=common_user_setting_specs)(owner)
        self.assertIn(("brainMaskBackend", str), specs)

    def test_parameter_sync_uses_data_and_defaults_old_nodes_to_pnl(self):
        # The masking data-sync is the first data sync, before unrelated fields.
        class Synced(Exception):
            pass

        calls = []
        combo = object()

        def sync_data(target, value):
            calls.append((target, value))
            raise Synced

        source = ast.parse((ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"))
        definition = next(item for item in source.body if isinstance(item, ast.ClassDef)
                          and item.name == "SlicerDSpaceParameterNode")
        defaults = {item.target.id: ast.literal_eval(item.value) for item in definition.body
                    if isinstance(item, ast.AnnAssign)}
        owner = SimpleNamespace(ui=SimpleNamespace(brainMaskBackendComboBox=combo),
                                _sync_combo_data=sync_data)
        for method in ("path_widget", "line_edit", "combo_text", "checkbox", "spinbox"):
            setattr(owner, "_sync_" + method, lambda *args, **kwargs: None)
        for backend in BACKENDS:
            owner._parameterNode = SimpleNamespace(**{**defaults, "brainMaskBackend": backend})
            with self.assertRaises(Synced):
                sync_parameter_node_to_ui(owner)
            self.assertEqual(calls[-1], (combo, backend))
        defaults.pop("brainMaskBackend")
        owner._parameterNode = SimpleNamespace(**defaults)
        with self.assertRaises(Synced):
            sync_parameter_node_to_ui(owner)
        self.assertEqual(calls[-1], (combo, "pnl_cnn"))

    def test_single_and_batch_run_kwargs_and_cli_preserve_all_backends(self):
        paths = {key: Path("/external") / key for key in PNL_MODES}
        paths["postprocess_freesurfer_home"] = Path("/external/freesurfer")
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                selection = RunSelection(
                    subject_id="sub-01", subject_root_text="/subject", output_root_text="/output",
                    selected_targets=("validation", "microstructure"), run_name="Test1",
                    tractography_backend="dti_det", bundle_backend="wmql",
                    microstructure_freewater=False, microstructure_noddi=False, select_streamlines=100,
                    ddseg_device="cpu", surface_device="cpu", connectome_backend="python",
                    connectome_compress_mrtrix_dense=False, brain_mask_backend=backend,
                )
                single = build_single_subject_run_kwargs(selection, paths, preflight_blocked_targets={})
                batch = build_batch_run_kwargs(selection, batch_root=Path("/batch"))
                self.assertEqual(single["brain_mask_backend"], backend)
                self.assertEqual(batch["brain_mask_backend"], backend)
                for key, path in paths.items():
                    self.assertEqual(single[key], path)
                cli_args = (
                    build_background_cli_args(single, paths, repo_root=ROOT),
                    build_batch_background_cli_args(batch, paths, repo_root=ROOT, batch_root=Path("/batch"),
                                                    batch_size=2, arrange_inputs="none"),
                )
                for args in cli_args:
                    self.assertEqual(args.count("--brain-mask-backend"), 1)
                    self.assertEqual(args[args.index("--brain-mask-backend") + 1], backend)
                    for key, path in paths.items():
                        flag = "--" + key.replace("_", "-")
                        self.assertEqual(args.count(flag), 1)
                        self.assertEqual(args[args.index(flag) + 1], str(path))

    def test_doctor_requires_mask_generator_for_preprocessing_and_known_missing_masks(self):
        doctor_kwargs = widget_method("_environment_check_kwargs", sys=SimpleNamespace(path=[]),
                                      inspect_input_directory=inspect_input_directory)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (input_dir / name).touch()
            node = SimpleNamespace(subjectId="subject", inputDirectory=str(root), outputDirectory="", brainMaskBackend="pnl_cnn",
                                   targetPreprocessing=False,
                                   connectomeBackend="python", ddsegDevice="cpu", device="cpu")
            owner = SimpleNamespace(_parameterNode=node, _selected_environment_targets=lambda: ("microstructure",),
                                    _tractography_backend_text=lambda: "dti_det", _microstructure_stage_flags=lambda: (False, False),
                                    _closure_tractography_backend_text=lambda: "dti_det")
            paths = {key: root / key for key in PNL_MODES}
            owner._microstructure_dki_enabled = lambda: False
            kwargs = doctor_kwargs(owner, external_paths=paths)
            self.assertTrue(kwargs["brain_mask_required"])
            self.assertEqual(kwargs["brain_mask_backend"], "pnl_cnn")
            for key, path in paths.items():
                self.assertEqual(kwargs[key], path)
            (input_dir / "nodif_brain_mask.nii.gz").touch()
            self.assertFalse(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"])
            node.targetPreprocessing = True
            for backend in BACKENDS:
                node.brainMaskBackend = backend
                self.assertEqual(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"],
                                 backend != "provided")
            node.targetPreprocessing = False
            node.brainMaskBackend = "pnl_cnn"
            node.inputDirectory = ""
            # C5: a mask-needing target with no input folder and no user mask still needs the mask runtime.
            self.assertTrue(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"])
            node.brainMaskBackend = "provided"
            self.assertFalse(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"])
            node.brainMaskBackend = "pnl_cnn"
            node.targetPreprocessing = True
            self.assertTrue(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"])
            node.targetPreprocessing = False
            node.inputDirectory = str(root)
            (input_dir / "nodif_brain_mask.nii.gz").unlink()
            owner._selected_environment_targets = lambda: ("validation",)
            self.assertFalse(doctor_kwargs(owner, external_paths=paths)["brain_mask_required"])

    def test_doctor_resolves_existing_mask_in_subject_below_parent(self):
        doctor_kwargs = widget_method("_environment_check_kwargs", sys=SimpleNamespace(path=[]),
                                      inspect_input_directory=inspect_input_directory)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "subject/input"
            folder.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (folder / name).touch()
            node = SimpleNamespace(subjectId="subject", inputDirectory=str(root), outputDirectory="",
                brainMaskBackend="pnl_cnn", targetPreprocessing=False,
                connectomeBackend="python", ddsegDevice="auto", device="auto")
            owner = SimpleNamespace(_parameterNode=node, _selected_environment_targets=lambda: ("microstructure",),
                _tractography_backend_text=lambda: "dti_det", _microstructure_stage_flags=lambda: (False, False),
                _microstructure_dki_enabled=lambda: False, _closure_tractography_backend_text=lambda: "dti_det")
            self.assertFalse(doctor_kwargs(owner, external_paths={})["brain_mask_required"])
            (folder / "nodif_brain_mask.nii.gz").unlink()
            self.assertTrue(doctor_kwargs(owner, external_paths={})["brain_mask_required"])


if __name__ == "__main__":
    unittest.main()
