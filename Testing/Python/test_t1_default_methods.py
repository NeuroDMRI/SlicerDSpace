#!/usr/bin/env python3
"""T1 W1: the DKI default (D4, MK-Curve) and the connectome default (D5, auto).

Synthetic inputs only. Each test names the design decision it checks.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from connectome import ConnectomeConfig, run_connectome
from connectome.pipeline import TCK2CONNECTOME_MISSING, select_connectome_backend


def _signature_defaults(paths, names):
    """Parameter defaults of every function in the files (AST: the Slicer modules import `slicer`)."""
    found = {}
    for path in paths:
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = node.args
                positional = args.posonlyargs + args.args
                pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
                pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                for argument, default in pairs:
                    if argument.arg in names:
                        found[f"{path}:{node.name}.{argument.arg}"] = ast.literal_eval(default)
    return found


def _class_annotation_default(path, class_name, attribute):
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == attribute:
                    return ast.literal_eval(item.value)
    raise AssertionError(f"{class_name}.{attribute} not found in {path}")


class DkiDefaultSitesTest(unittest.TestCase):
    """D4: MKCURVE is the default at every site that holds a DKI default."""

    def test_cli_default(self):
        from shared.run_subject_pipeline_cli import build_parser
        args = build_parser().parse_args(["--subject-root", "/tmp/s"])
        self.assertEqual(args.dki_fit_method, "MKCURVE")
        for method in ("WLS", "OLS", "MKCURVE_MATLAB"):
            parsed = build_parser().parse_args(["--subject-root", "/tmp/s", "--dki-fit-method", method])
            self.assertEqual(parsed.dki_fit_method, method)

    def test_gui_parameter_node_default(self):
        self.assertEqual(_class_annotation_default("SlicerDSpace.py", "SlicerDSpaceParameterNode",
                                                   "orchestrationDkiFitMethod"), "MKCURVE")

    def test_dataclass_defaults(self):
        from microstructure.dki import DkiConfig
        from microstructure.pipeline import MicrostructureConfig
        from shared.environment import EnvironmentDoctorConfig
        from shared.orchestrator import ProductOrchestratorConfig
        from shared.ui_run_controller import RunSelection
        self.assertEqual(DkiConfig.__dataclass_fields__["fit_method"].default, "MKCURVE")
        self.assertEqual(MicrostructureConfig.__dataclass_fields__["dki_fit_method"].default, "MKCURVE")
        self.assertEqual(EnvironmentDoctorConfig.__dataclass_fields__["dki_fit_method"].default, "MKCURVE")
        self.assertEqual(ProductOrchestratorConfig.__dataclass_fields__["microstructure_dki_fit_method"].default,
                         "MKCURVE")
        self.assertEqual(RunSelection.__dataclass_fields__["dki_fit_method"].default, "MKCURVE")

    def test_logic_entry_point_defaults(self):
        found = _signature_defaults(("shared/module_logic.py", "microstructure/module_logic.py"),
                                    {"dki_fit_method", "microstructure_dki_fit_method"})
        self.assertGreaterEqual(len(found), 5, found)
        self.assertEqual({value for value in found.values()}, {"MKCURVE"}, found)

    def test_saved_settings_and_run_defaults(self):
        from shared import user_config
        self.assertEqual(user_config.RUN_DEFAULTS["dki_fit_method"], "MKCURVE")
        source = (REPO_ROOT / "shared/user_config.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r'dki_fit_method[^\n]*"WLS"')

    def test_widget_and_controller_fallbacks(self):
        for path in ("shared/module_widget.py", "shared/ui_run_controller.py"):
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
            fallbacks = re.findall(r'(?:orchestrationDkiFitMethod|dki_fit_method)"?,\s*"([A-Z_]+)"', text)
            self.assertTrue(fallbacks, path)
            self.assertEqual(set(fallbacks), {"MKCURVE"}, path)

    def test_doctor_and_autotune_defaults(self):
        from shared.compute_tuning import calibration_options, tuning_key
        for stage in ("microstructure_dki", "microstructure_mkcurve"):
            self.assertEqual(calibration_options(stage)["fit_method"], "MKCURVE")
        default_key = tuning_key("microstructure_dki", [8, 8, 8, 90], [])
        explicit_key = tuning_key("microstructure_dki", [8, 8, 8, 90], ["--dki-fit-method", "MKCURVE"])
        self.assertEqual(default_key, explicit_key)
        self.assertNotEqual(default_key, tuning_key("microstructure_dki", [8, 8, 8, 90], ["--dki-fit-method", "WLS"]))

    def test_mkcurve_stop_names_wls_rerun(self):
        """D4: no automatic fallback; the failure carries the MK-Curve reason and the WLS re-run option."""
        from dataclasses import replace
        from microstructure.dki import DkiConfig, run_dki
        from shared.pipeline import RawInputBundle
        from test_microstructure_dki import synthetic_case, write_case
        case = synthetic_case()
        case["data"] = case["data"] / 1000.0  # b0 mean about 1: the integer synthetic-b0 grid step is 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_case(root, case)
            raw = RawInputBundle(root, "synthetic", root / "input", paths["dwi"], paths["bval"], paths["bvec"],
                                 paths["mask"])
            config = DkiConfig("synthetic", raw, root / "outputs/dki")
            self.assertEqual(config.fit_method, "MKCURVE")
            with self.assertRaises(ValueError) as caught:
                run_dki(config)
            message = str(caught.exception)
            self.assertIn("Signal scale cannot support", message)
            self.assertIn("re-run with DKI fit method WLS (--dki-fit-method WLS)", message)
            self.assertIn("re-run with DKI fit method WLS",
                          json.loads((config.output_dir / "dki_qc.json").read_text())["error"])
            # WLS stays available and is not affected.
            result = run_dki(replace(config, fit_method="WLS", output_dir=root / "outputs/wls"))
            self.assertEqual(result.status, "completed")


class ConnectomeDefaultSitesTest(unittest.TestCase):
    """D5: auto is the default and accepted at every site; compression is tri-state."""

    def test_cli_default_choices_and_tristate(self):
        from shared.run_subject_pipeline_cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--subject-root", "/tmp/s"])
        self.assertEqual(args.connectome_backend, "auto")
        self.assertIsNone(args.compress_mrtrix_dense)
        for backend in ("auto", "python", "mrtrix"):
            self.assertEqual(parser.parse_args(["--subject-root", "/tmp/s", "--connectome-backend", backend])
                             .connectome_backend, backend)
        self.assertIs(parser.parse_args(["--subject-root", "/tmp/s", "--compress-mrtrix-dense"])
                      .compress_mrtrix_dense, True)
        self.assertIs(parser.parse_args(["--subject-root", "/tmp/s", "--no-compress-mrtrix-dense"])
                      .compress_mrtrix_dense, False)

    def test_config_and_signature_defaults(self):
        from shared.environment import EnvironmentDoctorConfig
        from shared.orchestrator import ProductOrchestratorConfig
        from shared import user_config
        self.assertEqual(ProductOrchestratorConfig.__dataclass_fields__["connectome_backend"].default, "auto")
        self.assertIsNone(ProductOrchestratorConfig.__dataclass_fields__["connectome_compress_mrtrix_dense"].default)
        self.assertEqual(EnvironmentDoctorConfig.__dataclass_fields__["connectome_backend"].default, "auto")
        self.assertEqual(user_config.RUN_DEFAULTS["connectome_backend"], "auto")
        self.assertIsNone(ConnectomeConfig.__dataclass_fields__["compress_mrtrix_dense"].default)
        backends = _signature_defaults(("shared/module_logic.py",), {"connectome_backend"})
        compress = _signature_defaults(("shared/module_logic.py", "connectome/module_logic.py"),
                                       {"connectome_compress_mrtrix_dense", "compress_mrtrix_dense"})
        self.assertEqual(len(backends), 4, backends)
        self.assertEqual(set(backends.values()), {"auto"})
        self.assertGreaterEqual(len(compress), 3, compress)
        self.assertEqual(set(compress.values()), {None})
        self.assertEqual(_class_annotation_default("SlicerDSpace.py", "SlicerDSpaceParameterNode",
                                                   "connectomeBackend"), "auto")

    def test_orchestrator_accepts_auto(self):
        source = (REPO_ROOT / "shared/orchestrator.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('not in {"auto", "python", "mrtrix"}'), 2)
        self.assertNotIn('not in {"python", "mrtrix"}', source)

    def test_gui_controller_passes_tristate(self):
        from shared.ui_run_controller import _append_compress_arg
        for value, expected in ((None, []), (True, ["--compress-mrtrix-dense"]),
                                (False, ["--no-compress-mrtrix-dense"])):
            args = []
            _append_compress_arg(args, {"connectome_compress_mrtrix_dense": value})
            self.assertEqual(args, expected)

    def test_installer_choices_accept_auto(self):
        for path in ("scripts/install_dependencies.py", "scripts/dependency_profile_matrix.py",
                     "scripts/release_validation.py"):
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
            self.assertIn('"--connectome-backend"', text, path)
            self.assertRegex(text, r'"--connectome-backend"[^\n]*choices=\("auto", "python", "mrtrix"\)', path)

    def test_dependency_rule_follows_backend_and_source(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs
        def needs(backend, tracking):
            config = EnvironmentDoctorConfig(targets=("connectome",), repo_root=REPO_ROOT,
                                             connectome_backend=backend, tractography_backend=tracking)
            return any(getattr(spec, "command", None) == "tck2connectome" for spec in build_dependency_specs(config))
        self.assertTrue(needs("mrtrix", "ukf"))
        self.assertTrue(needs("auto", "csd_act"))
        self.assertFalse(needs("auto", "ukf"))
        self.assertFalse(needs("auto", "dti_det"))
        self.assertFalse(needs("python", "csd_act"))


class AutoBackendChoiceTest(unittest.TestCase):
    """D5: auto picks MRtrix for a .tck count, Python for anything else, never a silent fallback."""

    def config(self, source, metric="streamline_count", backend="auto", **extra):
        return ConnectomeConfig(subject_id="s", tractography_path=Path(source), parcellation_path=Path("p.nii.gz"),
                                metric=metric, backend=backend, **extra)

    def test_choice_per_source_and_metric(self):
        tool = {"tck2connectome": "/x/tck2connectome"}
        rows = [("a.tck", "streamline_count", True), ("a.tck", "sift2_weighted_streamline_count", True),
                ("a.tck", "mean_scalar", False), ("a.tck", "mean_length", False),
                ("a.vtk", "streamline_count", False), ("a.vtp", "streamline_count", False),
                ("a.trk", "streamline_count", False), ("a.vtk", "sift2_weighted_streamline_count", False)]
        for source, metric, expect_mrtrix in rows:
            with self.subTest(source=source, metric=metric):
                use, used, reason = select_connectome_backend(self.config(source, metric), tool)
                self.assertEqual(use, expect_mrtrix)
                self.assertEqual(used, "mrtrix" if expect_mrtrix else "python")
                self.assertTrue(reason.startswith("auto"))

    def test_missing_tool_is_an_error_for_auto_with_tck_count(self):
        with self.assertRaises(RuntimeError) as caught:
            select_connectome_backend(self.config("a.tck"), {"tck2connectome": None})
        self.assertEqual(str(caught.exception), TCK2CONNECTOME_MISSING)
        self.assertIn("install MRtrix3 in Prepare", TCK2CONNECTOME_MISSING)
        self.assertIn("--connectome-backend python", TCK2CONNECTOME_MISSING)
        # Python stays available without the tool, also for auto with a non-MRtrix source.
        self.assertEqual(select_connectome_backend(self.config("a.vtk"), {"tck2connectome": None})[0:2],
                         (False, "python"))

    def test_explicit_backends_keep_their_meaning(self):
        tool = {"tck2connectome": "/x/tck2connectome"}
        self.assertEqual(select_connectome_backend(self.config("a.tck", backend="python"), tool)[0:2],
                         (False, "python"))
        self.assertEqual(select_connectome_backend(self.config("a.tck", backend="mrtrix"), tool)[0:2],
                         (True, "mrtrix"))
        with self.assertRaisesRegex(RuntimeError, "Requested MRtrix connectome backend is unavailable"):
            select_connectome_backend(self.config("a.vtk", backend="mrtrix"), tool)
        with self.assertRaisesRegex(RuntimeError, "Requested MRtrix connectome backend is unavailable"):
            select_connectome_backend(self.config("a.tck", backend="mrtrix"), {"tck2connectome": None})


class AutoBackendRunTest(unittest.TestCase):
    """D5 end to end with a fake tck2connectome: compression default and manifest fields."""

    def setUp(self):
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        fake = self.bin_dir / "tck2connectome"
        fake.write_text("#!/bin/sh\n"
                        "printf '0,0,0,0,0\\n0,2,0,0,7\\n0,0,0,0,0\\n0,0,0,0,0\\n0,0,0,0,3\\n' > \"$3\"\n"
                        "exit 0\n", encoding="utf-8")
        fake.chmod(0o755)
        self.tractography = self.root / "streamlines.tck"
        self.tractography.write_text("tck", encoding="utf-8")
        labels = np.zeros((6, 6, 6), dtype=np.int16)
        labels[1, 1, 1] = 2
        labels[5, 5, 5] = 5
        self.parcellation = self.root / "wmparc.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), self.parcellation.as_posix())

    def run_connectome(self, name, **options):
        config = ConnectomeConfig(subject_id=name, tractography_path=self.tractography,
                                  parcellation_path=self.parcellation, output_dir=self.root / name,
                                  mrtrix_bin_dir=self.bin_dir, execute=True, **options)
        with mock.patch("connectome.pipeline.inspect_connectome_backends",
                        return_value={"tck2connectome": (self.bin_dir / "tck2connectome").as_posix(), "python": {}}):
            return run_connectome(config)

    def test_auto_with_mrtrix_path_compresses_unless_set(self):
        result = self.run_connectome("auto_default", backend="auto")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["backend_used"], "mrtrix")
        self.assertTrue(manifest["backend_reason"].startswith("auto"))
        self.assertIs(manifest["compress_mrtrix_dense"], True)
        self.assertTrue(Path(manifest["outputs"]["dense_matrix"]).is_file())
        matrix = result.matrix_path.read_text(encoding="utf-8")
        self.assertIn("2,5,7", matrix)  # node-filtered sparse matrix, as the Python backend has
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["backend"], "mrtrix3.tck2connectome")

    def test_auto_respects_an_explicit_off(self):
        result = self.run_connectome("auto_off", backend="auto", compress_mrtrix_dense=False)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertIs(manifest["compress_mrtrix_dense"], False)
        self.assertNotIn("dense_matrix", manifest["outputs"])

    def test_explicit_mrtrix_keeps_compression_off_by_default(self):
        result = self.run_connectome("mrtrix_default", backend="mrtrix")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["backend_used"], "mrtrix")
        self.assertIs(manifest["compress_mrtrix_dense"], False)
        self.assertNotIn("dense_matrix", manifest["outputs"])

    def test_missing_tool_fails_with_message_and_records_reason(self):
        config = ConnectomeConfig(subject_id="missing", tractography_path=self.tractography,
                                  parcellation_path=self.parcellation, output_dir=self.root / "missing",
                                  backend="auto", execute=True)
        with mock.patch("connectome.pipeline.inspect_connectome_backends",
                        return_value={"tck2connectome": None, "python": {}}):
            with self.assertRaisesRegex(RuntimeError, "MRtrix tck2connectome is needed for this connectome"):
                run_connectome(config)
        manifest = json.loads((self.root / "missing/missing-connectome-streamline_count_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNone(manifest["backend_used"])
        self.assertIn("--connectome-backend python", manifest["backend_reason"])

    def test_auto_with_python_source_records_python(self):
        source = self.root / "streamlines.vtk"
        source.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")

        def write_outputs(config, matrix_path, nodes_path, *, lut_path):
            matrix_path.write_text("source_node,target_node,streamline_count\n10,20,1\n", encoding="utf-8")
            nodes_path.write_text("node_id\tlabel\tname\n10\t10\tL\n20\t20\tR\n", encoding="utf-8")
            return {"node_count": 2, "edge_count": 1}

        config = ConnectomeConfig(subject_id="py", tractography_path=source, parcellation_path=self.parcellation,
                                  output_dir=self.root / "py", backend="auto", execute=True)
        with mock.patch("connectome.pipeline._write_python_connectome", side_effect=write_outputs):
            result = run_connectome(config)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["backend_used"], "python")
        self.assertIs(manifest["compress_mrtrix_dense"], False)


if __name__ == "__main__":
    unittest.main()
