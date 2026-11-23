"""Interpreter paths keep a virtualenv symlink (macOS F1).

A venv ``bin/python`` is a symlink to a base interpreter. Resolving it drops the
venv's site-packages, so every path that names an interpreter must stay unresolved.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class VenvLinkCase(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.link = self.tmp / "venv/bin/python"
        self.link.parent.mkdir(parents=True)
        self.link.symlink_to(Path(sys.executable).resolve())
        self.assertNotEqual(self.link, self.link.resolve())


class InterpreterPathTests(VenvLinkCase):
    @skip_unless_symlinks
    def test_setup_keeps_an_external_virtualenv_symlink(self):
        from shared.python_environment import setup_python_command
        self.assertEqual(setup_python_command(self.link, slicer_home=self.tmp / 'Slicer', launcher='Slicer'),
                         (str(self.link), []))

    @skip_unless_symlinks
    def test_setup_uses_slicer_launcher_for_its_raw_interpreter_and_wrapper(self):
        from shared.python_environment import setup_python_command
        home = self.tmp / 'Slicer'
        for name in ('python-real', 'PythonSlicer', 'python-real.exe', 'PythonSlicer.exe'):
            self.assertEqual(setup_python_command(home / 'bin' / name, slicer_home=home, launcher='Slicer'),
                             ('Slicer', ['--launch', 'PythonSlicer']))
            self.assertEqual(setup_python_command(home / 'bin' / name, slicer_home=home / 'bin/..', launcher='Slicer'),
                             ('Slicer', ['--launch', 'PythonSlicer']))

    @skip_unless_symlinks
    def test_helper_keeps_symlink_and_expands_user(self) -> None:
        from shared.external_process import interpreter_path

        self.assertEqual(interpreter_path(self.link), self.link)
        with mock.patch.dict(os.environ, {"HOME": str(self.tmp)}):
            self.assertEqual(interpreter_path(Path("~/venv/bin/python")), self.link)

    @skip_unless_symlinks
    def test_python_environment_resolver_keeps_symlink(self) -> None:
        from shared.python_environment import resolve_python_executable

        self.assertEqual(resolve_python_executable(python_executable=self.link), self.link)
        self.assertEqual(resolve_python_executable(python_environment=str(self.link)), self.link)

    @skip_unless_symlinks
    def test_doctor_specs_keep_symlink(self) -> None:
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs

        config = EnvironmentDoctorConfig(
            targets=("tractcloud",),
            repo_root=REPO_ROOT,
            python_executable=self.link,
            tractcloud_python_executable=self.link,
        )
        specs = {spec.key: spec for spec in build_dependency_specs(config)}
        self.assertEqual(specs["python:tractcloud:torch"].python_executable, self.link)
        self.assertEqual(specs["python:tractcloud"].python_executable, self.link)

    @skip_unless_symlinks
    def test_doctor_tractcloud_env_interpreter_keeps_symlink(self) -> None:
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs

        config = EnvironmentDoctorConfig(targets=("tractcloud",), repo_root=REPO_ROOT)
        with mock.patch.dict(os.environ, {"TRACTCLOUD_PYTHON_EXECUTABLE": str(self.link)}):
            specs = {spec.key: spec for spec in build_dependency_specs(config)}
        self.assertEqual(specs["python:tractcloud:torch"].python_executable, self.link)

    @skip_unless_symlinks
    def test_postprocess_command_keeps_symlink(self) -> None:
        from anatomy.surface import postprocess_release
        from anatomy.surface.postprocess_release import DDSurferPostprocessConfig, run_ddsurfer_postprocess

        root = self.tmp
        native_dir, mri_dir, backend_root = root / "native", root / "mri", root / "backend"
        native_dir.mkdir()
        mri_dir.mkdir()
        (backend_root / "config").mkdir(parents=True)
        (backend_root / "run.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (backend_root / "run.sh").chmod(0o755)
        for name in ("brain.mgz", "DDSurfer-wmparc.nii.gz"):
            (mri_dir / name).write_text(name, encoding="utf-8")
        for name in ("877168_predicted_wm_lh_orig.stl", "877168_predicted_pial_lh_orig.stl"):
            (native_dir / name).write_text(name, encoding="utf-8")

        class Stop(Exception):
            pass

        seen = []

        def popen(cmd, *args, **kwargs):
            seen.append(cmd)
            raise Stop

        config = DDSurferPostprocessConfig(
            subject_id="877168", native_dir=native_dir, mri_dir=mri_dir, output_root=root / "outputs",
            backend_root=backend_root, mode="left", python_bin=self.link,
        )
        with mock.patch.object(postprocess_release.subprocess, "Popen", popen), self.assertRaises(Stop):
            run_ddsurfer_postprocess(config)
        cmd = seen[0]
        self.assertEqual(cmd[cmd.index("--python-bin") + 1], str(self.link))


class OrchestratorInterpreterPathTests(VenvLinkCase):
    @skip_unless_symlinks
    def test_run_record_keeps_symlink(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        root = self.tmp / "100307"
        (root / "input").mkdir(parents=True)
        for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
            (root / "input" / name).write_text("x", encoding="utf-8")
        run_subject_products(ProductOrchestratorConfig(
            subject_id="100307", subject_root=root, targets=("microstructure",), execute=False,
            surface_python_bin=self.link, postprocess_python_bin=self.link,
            tractcloud_python_executable=self.link,
        ))
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in root.rglob("*.json")
            if "surface_python_bin" in path.read_text(encoding="utf-8")
        ]
        self.assertTrue(records)
        for key in ("surface_python_bin", "postprocess_python_bin", "tractcloud_python_executable"):
            self.assertEqual(records[0][key], self.link.as_posix(), key)

    @skip_unless_symlinks
    def test_resume_accepts_recorded_symlink_interpreter(self) -> None:
        import trimesh
        from anatomy.surface.backend_bridge import native_products, sha256, upstream_record
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, _surface_complete

        root = self.tmp / "x"
        surface = subject_layout(root).ddsurfer
        native = surface / "pred_results_SlicerDSpace/native/x"
        native.mkdir(parents=True)
        for hemi in ("lh", "rh"):
            for kind in ("white", "pial"):
                trimesh.creation.icosphere(subdivisions=1).export(str(native / f"{hemi}.{kind}.obj"))
        products, checks = native_products(native, "x", "all")
        inputs = []
        for index in range(15):
            path = root / f"input-{index}"
            path.write_text("registered-input", encoding="utf-8")
            inputs.append(path)
        outputs = [*products, *(Path(c["indexed_master"]) for c in checks.values())]
        manifest = dict(
            version=3, status="completed", input_mode="registered", precision="fp32",
            subject_id="x", subject_root=str(root.resolve()), cpu_threads=4, max_gpu_gib=24.,
            upstream_commit=upstream_record()["commit"],
            runtime=dict(executable=str(self.link)),
            input_sha256={str(p): sha256(p) for p in inputs},
            output_sha256={str(p): sha256(p) for p in outputs},
        )
        (surface / "surface_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (surface / "surface_qc.json").write_text(json.dumps(dict(automatic_qc_passed=True)), encoding="utf-8")
        config = ProductOrchestratorConfig(
            subject_id="x", subject_root=root, targets=("surface",), surface_python_bin=self.link)
        self.assertTrue(_surface_complete(root, "x", config=config))


if __name__ == "__main__":
    unittest.main()
