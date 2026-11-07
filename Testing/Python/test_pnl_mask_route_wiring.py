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
from shared.ui_run_controller import (
    RunSelection, build_background_cli_args, build_batch_background_cli_args, build_single_subject_run_kwargs)
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
        paths = {}
        single = build_background_cli_args(kwargs, paths, repo_root=ROOT)
        self.assertEqual(single[single.index("--pnl-mask-route") + 1], "native")
        batch = build_batch_background_cli_args(kwargs, paths, repo_root=ROOT, batch_root=ROOT, batch_size=1,
                                                arrange_inputs="no")
        self.assertEqual(batch[batch.index("--pnl-mask-route") + 1], "native")
        # The batch script forwards unknown options to the subject CLI, whose parser knows this one.
        tail = batch[batch.index("--pnl-mask-route"):batch.index("--pnl-mask-route") + 2]
        self.assertEqual(cli.build_parser().parse_args(tail).pnl_mask_route, "native")

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
        """The run record takes the route from the receipt of the mask in the subject's input folder (F2)."""
        for route, name, experimental in (("managed", "managed-ants", False), ("native", "native-sitk", True)):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                inputs = Path(tmp) / "subject" / "input"
                inputs.mkdir(parents=True)
                (inputs / "nodif_brain_mask.nii.gz").write_bytes(b"mask")
                self._write_receipt(inputs / "nodif_brain_mask.nii.gz", route)
                manifest = Path(tmp) / "run_manifest.json"
                manifest.write_text(json.dumps({
                    "kind": "slicerddsurfer_product_orchestration", "subject_id": "sub-01", "subject_root": str(Path(tmp) / "subject"),
                    "brain_mask_backend": "pnl_cnn", "pnl_mask_route": "auto",
                    "brain_mask_route_requested": pnl_route_record("managed" if route == "native" else "native"),
                }), encoding="utf-8")
                record = build_run_record(manifest)
                mask = record["brain_mask"]
                self.assertEqual(mask["route"], name)  # the receipt wins over the requested route
                self.assertEqual(mask["requested_backend"], "pnl_cnn")
                if experimental:
                    self.assertEqual(mask["status"], "experimental")
                    self.assertEqual(mask["reason"], PNL_NATIVE_EXPERIMENTAL_REASON)
                else:
                    self.assertNotIn("status", mask)

    def test_f2_supplied_mask_with_the_default_pnl_selection_records_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "subject" / "input"
            inputs.mkdir(parents=True)
            (inputs / "nodif_brain_mask.nii.gz").write_bytes(b"user mask")
            manifest = Path(tmp) / "run_manifest.json"
            manifest.write_text(json.dumps({
                "subject_id": "sub-01", "subject_root": str(Path(tmp) / "subject"), "brain_mask_backend": "pnl_cnn",
                "brain_mask_route_requested": pnl_route_record("native")}), encoding="utf-8")
            mask = build_run_record(manifest)["brain_mask"]
            self.assertEqual(mask, {"requested_backend": "pnl_cnn", "route": "supplied"})

    # ---- F2b: with complete preprocessing the consumed mask is the preprocessing final mask
    def _preprocessed_subject(self, tmp: str, *, generation: dict) -> tuple[Path, Path, Path]:
        """A subject with a complete preprocessing layout; returns (run manifest, final mask, input mask)."""
        from shared.io_layout import subject_layout
        root = Path(tmp) / "subject"
        inputs = root / "input"
        inputs.mkdir(parents=True)
        input_mask = inputs / "nodif_brain_mask.nii.gz"
        for name in ("data.nii.gz", "bvals", "bvecs"):  # W2b: the shared reuse predicate checks the raw input files
            (inputs / name).write_bytes(b"raw " + name.encode())
        preproc = subject_layout(root).preproc
        (preproc / "05_qc").mkdir(parents=True)
        for name in ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec"):
            (preproc / ("sub-01_" + name)).write_bytes(b"x")
        final = preproc / "sub-01_brain_mask_final.nii.gz"
        (preproc / "sub-01_preproc_manifest.json").write_text(json.dumps({
            "status": "completed", "mask_generation": generation}), encoding="utf-8")
        (preproc / "05_qc" / "sub-01_preproc_qc.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        run = Path(tmp) / "run_manifest.json"
        run.write_text(json.dumps({
            "subject_id": "sub-01", "subject_root": str(root), "brain_mask_backend": "pnl_cnn",
            "brain_mask_route_requested": pnl_route_record("native")}), encoding="utf-8")
        return run, final, input_mask

    @staticmethod
    def _seal(run: Path) -> None:
        """W2b: record the raw input files in the preprocessing cache key (version 5), as a real run does."""
        from shared.io_layout import subject_layout
        from shared.pipeline import _file_signature
        root = Path(json.loads(run.read_text(encoding="utf-8"))["subject_root"])
        path = subject_layout(root).preproc / "sub-01_preproc_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        names = {"dwi": "data.nii.gz", "bval": "bvals", "bvec": "bvecs", "mask": "nodif_brain_mask.nii.gz"}
        manifest["cache_key"] = {"version": 5, "inputs": {key: _file_signature(root / "input" / name)
                                                           for key, name in names.items()}}
        path.write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def _write_receipt(mask: Path, route: str, *, sha: str | None = None, backend: str = "pnl_cnn") -> None:
        import hashlib
        digest = sha or hashlib.sha256(mask.read_bytes()).hexdigest()
        mask.with_name(mask.name.removesuffix(".nii.gz") + ".manifest.json").write_text(json.dumps({
            "kind": "generated_brain_mask", "version": 2, "status": "completed", "backend": backend,
            "output": {"path": str(mask), "sha256": digest}, "pnl_route": pnl_route_record(route)}), encoding="utf-8")

    def test_f2b_generated_native_final_mask_of_complete_preprocessing_gives_native_experimental(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, final, input_mask = self._preprocessed_subject(tmp, generation={"backend": "pnl_cnn", "status": "completed"})
            input_mask.write_bytes(b"user input mask")  # the input is a different, supplied mask
            final.write_bytes(b"generated on the corrected dwi")
            self._write_receipt(final, "native")
            self._seal(run)
            mask = build_run_record(run)["brain_mask"]
            self.assertEqual(mask["route"], "native-sitk")
            self.assertEqual(mask["status"], "experimental")
            self.assertEqual(mask["reason"], PNL_NATIVE_EXPERIMENTAL_REASON)

    def test_f2b_stale_receipt_does_not_label_a_later_final_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, final, input_mask = self._preprocessed_subject(tmp, generation={"backend": "pnl_cnn", "status": "completed"})
            input_mask.write_bytes(b"generated input mask")
            self._write_receipt(input_mask, "native")  # a valid native receipt on the input must not be used here
            final.write_bytes(b"replaced mask")
            self._write_receipt(final, "native", sha="0" * 64)
            self._seal(run)
            self.assertEqual(build_run_record(run)["brain_mask"]["route"], "unknown")

    def test_f2b_provided_final_mask_copied_from_a_generated_input_mask_keeps_the_input_route(self) -> None:
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            run, final, input_mask = self._preprocessed_subject(tmp, generation={"backend": "pnl_cnn"})
            input_mask.write_bytes(b"generated input mask")
            self._write_receipt(input_mask, "native")
            shutil.copy2(input_mask, final)
            manifest = final.with_name("sub-01_preproc_manifest.json")
            manifest.write_text(json.dumps({"status": "completed", "mask_generation": {
                "backend": "provided", "status": "copied", "source": input_mask.as_posix()}}), encoding="utf-8")
            self._seal(run)
            mask = build_run_record(run)["brain_mask"]
            self.assertEqual((mask["route"], mask["status"]), ("native-sitk", "experimental"))
            # A copy that no longer matches its source is not labelled with the source's route.
            final.write_bytes(b"edited after the copy")
            self.assertEqual(build_run_record(run)["brain_mask"]["route"], "supplied")

    def test_f2b_incomplete_preprocessing_and_input_mask_without_a_receipt_gives_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, final, input_mask = self._preprocessed_subject(tmp, generation={"backend": "provided"})
            for path in final.parent.glob("sub-01_dwi_final*"):
                path.unlink()  # preprocessing not complete: the input mask is consumed
            input_mask.write_bytes(b"user mask")
            self.assertEqual(build_run_record(run)["brain_mask"], {"requested_backend": "pnl_cnn", "route": "supplied"})

    def test_f2_orchestrator_manifest_keeps_the_setting_and_names_the_requested_route(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products
        for route, system, requested in (("auto", "Darwin", "native-sitk"), ("auto", "Linux", "managed-ants"),
                                         ("native", "Linux", "native-sitk")):
            with self.subTest(route=route, system=system), tempfile.TemporaryDirectory() as tmp, \
                    unittest.mock.patch("platform.system", return_value=system):
                subject_root = Path(tmp) / "100307"
                (subject_root / "input").mkdir(parents=True)
                result = run_subject_products(ProductOrchestratorConfig(
                    subject_id="100307", subject_root=subject_root, targets=("microstructure",), execute=False,
                    brain_mask_backend="pnl_cnn", pnl_mask_route=route))
                data = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(data["pnl_mask_route"], route)
                self.assertEqual(data["brain_mask_route_requested"]["route"], requested)
                self.assertNotIn("brain_mask_route", data)

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

    def test_sc1_carry_1_every_requested_backend_records_the_consumed_mask(self) -> None:
        """SC1-CARRY-1: a non-PNL selection records the consumed mask too; a synthstrip receipt gives its backend."""
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "subject" / "input"
            inputs.mkdir(parents=True)
            mask_path = inputs / "nodif_brain_mask.nii.gz"
            mask_path.write_bytes(b"synthstrip mask")
            manifest = Path(tmp) / "run_manifest.json"
            manifest.write_text(json.dumps({"subject_id": "s", "subject_root": str(Path(tmp) / "subject"),
                                            "brain_mask_backend": "synthstrip"}), encoding="utf-8")
            self.assertEqual(build_run_record(manifest)["brain_mask"],
                             {"requested_backend": "synthstrip", "route": "supplied"})
            self._write_receipt(mask_path, "native", backend="synthstrip")
            self.assertEqual(build_run_record(manifest)["brain_mask"],
                             {"requested_backend": "synthstrip", "route": "synthstrip"})
            self._write_receipt(mask_path, "native", backend="synthstrip", sha="0" * 64)  # stale receipt
            self.assertEqual(build_run_record(manifest)["brain_mask"]["route"], "unknown")

    def test_sc1_carry_1_real_provided_backend_reusing_a_native_receipt_records_native_experimental(self) -> None:
        import nibabel as nib
        import numpy as np
        from shared.brain_mask import BrainMaskConfig, generate_brain_mask
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "subject"
            inputs = root / "input"
            inputs.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), dtype="float32"), np.eye(4)), str(inputs / "data.nii.gz"))
            (inputs / "bvals").write_text("0 1000\n")
            (inputs / "bvecs").write_text("0 1\n0 0\n0 0\n")
            mask = inputs / "nodif_brain_mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype="uint8"), np.eye(4)), str(mask))
            self._write_receipt(mask, "native")
            result = generate_brain_mask(inputs / "data.nii.gz", inputs / "bvals", inputs / "bvecs", mask,
                                         config=BrainMaskConfig(backend="provided"))
            self.assertEqual(result["status"], "reused")
            manifest = Path(tmp) / "run_manifest.json"
            manifest.write_text(json.dumps({"subject_id": "s", "subject_root": str(root),
                                            "brain_mask_backend": "provided"}), encoding="utf-8")
            record = build_run_record(manifest)["brain_mask"]
            self.assertEqual(record["requested_backend"], "provided")
            self.assertEqual((record["route"], record["status"]), ("native-sitk", "experimental"))


if __name__ == "__main__":
    unittest.main()
