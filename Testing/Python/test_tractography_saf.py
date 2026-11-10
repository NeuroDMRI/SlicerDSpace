#!/usr/bin/env python3
"""Tests for Python SAF integration facade."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SAF_BACKEND_ROOT = REPO_ROOT / "tractography" / "saf_backend"
if str(SAF_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(SAF_BACKEND_ROOT))


class SAFIntegrationTest(unittest.TestCase):
    def test_gg_hh_endpoint_pairing_matches_matlab_layout(self) -> None:
        from tractography.saf_backend.Filters.filters_GG_HH import _pair_streamline_endpoint_values

        starts_then_ends = [10, 20, 30, 40, 1, 2, 3, 4]
        paired = _pair_streamline_endpoint_values(
            values=np.array(starts_then_ends),
            nsl=4,
        )
        self.assertEqual(paired.tolist(), [[10, 20, 30, 40], [1, 2, 3, 4]])

    def test_complete_intersection_matches_manual_triangle_gold_cases(self) -> None:
        from tractography.saf_backend.Filters.streamlineXmesh.trintersect_complete import (
            trintersect_complete,
        )

        triangle = (
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        )
        q1 = np.array([
            [0.25, 0.25, -1.0],  # face interior
            [1.00, 1.00, -1.0],  # outside triangle
            [0.50, 0.00, -1.0],  # edge
            [0.00, 0.00, -1.0],  # vertex
            [0.20, 0.20, 1.0],   # parallel to face
        ])
        q2 = np.array([
            [0.25, 0.25, 1.0],
            [1.00, 1.00, 1.0],
            [0.50, 0.00, 1.0],
            [0.00, 0.00, 1.0],
            [0.80, 0.20, 1.0],
        ])
        count = q1.shape[0]
        result = trintersect_complete(
            *(np.tile(vertex, (count, 1)) for vertex in triangle),
            q1,
            q2,
            np.tile(np.array([[0, 1, 2]]), (count, 1)),
            np.zeros(count, dtype=int),
            np.arange(count, dtype=int) * 2,
        )
        np.testing.assert_array_equal(result, [True, False, True, True, False])

    def test_complete_intersection_counts_shared_streamline_vertex_once(self) -> None:
        from tractography.saf_backend.Filters.streamlineXmesh.trintersect_complete import (
            trintersect_complete,
        )

        p1 = np.tile([[0.0, 0.0, 0.0]], (2, 1))
        p2 = np.tile([[1.0, 0.0, 0.0]], (2, 1))
        p3 = np.tile([[0.0, 1.0, 0.0]], (2, 1))
        result = trintersect_complete(
            p1,
            p2,
            p3,
            np.array([[0.25, 0.25, -1.0], [0.25, 0.25, 0.0]]),
            np.array([[0.25, 0.25, 0.0], [0.25, 0.25, 1.0]]),
            np.tile([[0, 1, 2]], (2, 1)),
            np.zeros(2, dtype=int),
            np.array([0, 1]),
        )
        np.testing.assert_array_equal(result, [False, True])

    def test_complete_intersection_unique_mapping_uses_flagged_group(self) -> None:
        from tractography.saf_backend.Filters.streamlineXmesh.trintersect_complete import (
            _deduplicate_true_rows,
        )

        # The true duplicate is not the first row in its group. MATLAB's
        # la(lb(flags)) mapping still marks the group's representative.
        result = _deduplicate_true_rows(
            np.array([[1, 9], [1, 9], [2, 9]]),
            np.array([False, True, True]),
        )
        np.testing.assert_array_equal(result, [True, False, True])

    def test_crop_intersections_follow_streamline_order_and_pair_faces(self) -> None:
        from tractography.saf_backend.Filters.streamlineXmesh.crop_streamlines_outside_mesh_complete import (
            crop_streamlines_outside_mesh_complete,
        )
        from tractography.saf_backend.Filters.streamlineXmesh.crop_streamlines_outside_mesh_fast import (
            crop_streamlines_outside_mesh_fast,
        )

        tetra_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Put the exit face before the entry face. A row-major np.where then
        # returns crossings backwards, whereas MATLAB find follows segments.
        tetra_faces = np.array([
            [1, 2, 3],
            [0, 1, 3],
            [0, 2, 1],
            [0, 3, 2],
        ])
        vertices = np.vstack([tetra_vertices, tetra_vertices + [2.0, 0.0, 0.0]])
        faces = np.vstack([tetra_faces, tetra_faces + 4])
        x = np.array([-0.5, -0.1, 0.1, 0.4, 0.7, 0.9, 1.2, 1.8, 1.9, 2.1, 2.4, 2.7, 2.9, 3.2])
        streamline = np.column_stack([x, np.full_like(x, 0.1), np.full_like(x, 0.1)])

        for crop in (
            crop_streamlines_outside_mesh_fast,
            crop_streamlines_outside_mesh_complete,
        ):
            with self.subTest(crop=crop.__name__):
                cropped, original, endpoint_vertices, xpoints = crop(
                    vertices, faces, [streamline], False, False
                )
                np.testing.assert_array_equal(original, [0, 0])
                np.testing.assert_array_equal(endpoint_vertices, [[1, 5], [2, 6]])
                np.testing.assert_array_equal(xpoints, [[2, 9], [4, 11]])
                np.testing.assert_allclose(cropped[0][:, 0], [0.1, 0.4, 0.7])
                np.testing.assert_allclose(cropped[1][:, 0], [2.1, 2.4, 2.7])

    def test_pial_escape_mask_uses_each_surviving_original_index(self) -> None:
        from Filters.filters_PG import _pial_escape_mask

        tracks = [np.zeros((5, 3)), np.zeros((9, 3)), np.zeros((7, 3))]
        result = _pial_escape_mask(
            np.array([[0, 1], [6, 4]]),
            np.array([2, 0]),
            tracks,
        )
        np.testing.assert_array_equal(result, [False, True])

    def test_portable_file_operations_link_copy_and_remove_glob(self) -> None:
        from Support import file_ops

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("surface", encoding="utf-8")
            destination = root / "output" / "rh.inflated"
            with mock.patch.object(file_ops.Path, "symlink_to", side_effect=OSError("links disabled")):
                file_ops.link_or_copy(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "surface")
            self.assertFalse(destination.is_symlink())

            (root / "ants_file.txt").write_text("x", encoding="utf-8")
            (root / "ants_dir").mkdir()
            (root / "ants_dir" / "nested.txt").write_text("x", encoding="utf-8")
            keep = root / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            removed = file_ops.remove_matching(root, "ants_*")
            self.assertEqual({path.name for path in removed}, {"ants_file.txt", "ants_dir"})
            self.assertTrue(keep.is_file())

    def test_saf_intermediate_cache_replaces_forced_rerun_outputs(self) -> None:
        from tractography.saf import _cache_saf_intermediates

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "PS.mat").write_text("first", encoding="utf-8")
            first = _cache_saf_intermediates(output)
            self.assertEqual(Path(first["PS.mat"]).read_text(encoding="utf-8"), "first")

            (output / "PS.mat").write_text("second", encoding="utf-8")
            (output / "cache" / "PS.mat.1").write_text("legacy", encoding="utf-8")
            second = _cache_saf_intermediates(output)

            self.assertEqual(Path(second["PS.mat"]).read_text(encoding="utf-8"), "second")
            self.assertFalse((output / "cache" / "PS.mat.1").exists())

    def test_sys_exec_propagates_external_command_failure(self) -> None:
        from Support.sys_exec import sys_exec

        ps = types.SimpleNamespace(dgn=False)
        failure = subprocess.CalledProcessError(7, "broken-command")
        with mock.patch("Support.sys_exec.subprocess.run", side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                sys_exec("broken-command", ps)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_windows_saf_decodes_posix_quotes_without_cmd_shell(self) -> None:
        from Support.sys_exec import sys_exec
        ps = types.SimpleNamespace(dgn=False)
        command = "PATH='C:/Tools with spaces':$PATH tckstats 'C:/Data/a ; name.tck' -dump 'C:/Data/lengths.txt'"
        with mock.patch("sys.platform", "win32"), \
                mock.patch("Support.sys_exec.startup_environment", return_value={"PATH": "original"}), \
                mock.patch("Support.sys_exec.shutil.which", return_value="C:/Tools with spaces/tckstats.exe"), \
                mock.patch("Support.sys_exec.subprocess.run") as run:
            sys_exec(command, ps)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0], ["C:/Tools with spaces/tckstats.exe", "C:/Data/a ; name.tck", "-dump", "C:/Data/lengths.txt"])
        self.assertTrue(run.call_args.kwargs["env"]["PATH"].startswith("C:/Tools with spaces"))

    def test_windows_saf_rejects_shell_operators_without_launch(self) -> None:
        from Support.sys_exec import sys_exec
        with mock.patch("sys.platform", "win32"), mock.patch("Support.sys_exec.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "single executable"):
                sys_exec("tckstats input.tck && other-command", types.SimpleNamespace(dgn=False))
            run.assert_not_called()

    def test_write_curv_records_freesurfer_face_count(self) -> None:
        import importlib

        io_module = types.ModuleType("nibabel.freesurfer.io")
        io_module.write_morph_data = mock.Mock()
        io_module.read_morph_data = mock.Mock()
        freesurfer_module = types.ModuleType("nibabel.freesurfer")
        freesurfer_module.io = io_module
        nibabel_module = types.ModuleType("nibabel")
        nibabel_module.freesurfer = freesurfer_module
        module_name = "tractography.saf_backend.Support.write_curv"
        previous = sys.modules.pop(module_name, None)
        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "nibabel": nibabel_module,
                    "nibabel.freesurfer": freesurfer_module,
                    "nibabel.freesurfer.io": io_module,
                },
            ):
                module = importlib.import_module(module_name)
                module.write_curv("rh.test", [1.0, 2.0], nfaces=np.int64(37))
            io_module.write_morph_data.assert_called_once()
            self.assertEqual(io_module.write_morph_data.call_args.kwargs["fnum"], 37)
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

    def _make_subject(self, root: Path, subject_id: str = "100307") -> Path:
        subject = root / subject_id
        input_dir = subject / "input"
        input_dir.mkdir(parents=True)
        for name in ["data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"]:
            (input_dir / name).write_text("stub", encoding="utf-8")
        return subject

    def _make_fs_dir(self, root: Path) -> Path:
        fs_dir = root / "freesurfer"
        (fs_dir / "surf").mkdir(parents=True)
        (fs_dir / "mri").mkdir()
        for name in ["rh.pial", "lh.pial", "rh.white", "lh.white", "rh.inflated", "lh.inflated", "rh.sphere", "lh.sphere", "rh.curv", "lh.curv"]:
            (fs_dir / "surf" / name).write_text("stub", encoding="utf-8")
        (fs_dir / "mri" / "brain.mgz").write_text("stub", encoding="utf-8")
        return fs_dir

    def test_saf_backend_is_bundled(self) -> None:
        from tractography import bundled_saf_backend_dir, inspect_saf_backend

        backend_dir = bundled_saf_backend_dir()
        self.assertTrue((backend_dir / "SAF_pipeline.py").is_file())
        status = inspect_saf_backend()
        self.assertTrue(status["backend_dir_exists"])
        self.assertIn("numpy", status["packages"])
        self.assertNotIn("ants", status)

    def test_saf_dry_run_writes_manifest_and_qc(self) -> None:
        from shared import subject_layout
        from tractography import SAFConfig, run_saf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._make_subject(root)
            fs_dir = self._make_fs_dir(root)
            fod = root / "wmfod.nii.gz"
            fod.write_text("stub", encoding="utf-8")
            tractogram = root / "input.tck"
            tractogram.write_text("stub", encoding="utf-8")

            result = run_saf(
                SAFConfig(
                    subject_id=subject.name,
                    input_root=subject,
                    fs_dir=fs_dir,
                    fod_path=fod,
                    pregenerated_tractogram=tractogram,
                    maxlen=None,
                    execute=False,
                )
            )
            layout = subject_layout(subject)
            self.assertEqual(result.output_dir, layout.saf)
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.qc_path.is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "saf_short_association_fibres")
            self.assertEqual(manifest["parameters"]["surface_space"], "native_dMRI")
            self.assertEqual(manifest["parameters"]["random_seed"], 12345)
            self.assertIn("reproducibility_note", manifest)
            self.assertIn("NOT bit-reproducible", manifest["reproducibility_note"])
            self.assertNotIn("mrtrix:tckgen", manifest["backend"]["missing_required"])
            self.assertEqual(qc["status"], "planned")
            self.assertFalse(result.executed)
            self.assertIn("saf_vtk", manifest["outputs"])
            self.assertIn("saf_vtk", manifest["outputs"]["summary"])

    def test_effective_tckgen_options_does_not_inject_ignored_config_seed(self) -> None:
        # MRtrix3 ignores "-config RandomSeed <n>" on the command line, so
        # SAF must not inject it; the seed is applied via MRTRIX_RNG_SEED
        # (see test_saf_execute_sets_mrtrix_rng_seed_env) and still recorded
        # in the manifest for provenance.
        from tractography.saf import SAFConfig, _effective_tckgen_options

        base = dict(
            subject_id="100307",
            input_root=Path("/tmp/subject"),
            fs_dir=Path("/tmp/fs"),
            fod_path=Path("/tmp/fod.mif"),
        )
        default_options = _effective_tckgen_options(
            SAFConfig(**base, tckgen_options="-select 1000")
        )
        explicit_options = _effective_tckgen_options(
            SAFConfig(**base, tckgen_options="-select 1000 -config RandomSeed 7")
        )

        self.assertEqual(default_options, "-select 1000")
        self.assertNotIn("RandomSeed", default_options)
        # An explicitly user-supplied token is left untouched (not our concern
        # to strip), but SAF itself never adds one.
        self.assertEqual(explicit_options, "-select 1000 -config RandomSeed 7")

    def test_saf_execute_sets_mrtrix_rng_seed_env(self) -> None:
        # This must exercise the real Support.sys_exec (not just check
        # os.environ directly): sys_exec's env comes from
        # startup_environment(), which inside real Slicer is a fixed
        # snapshot from slicer.util.startupEnvironment() that does not
        # reflect anything set on os.environ afterward. A fake backend that
        # only reads os.environ.get('MRTRIX_RNG_SEED') itself would pass
        # even if sys_exec never actually forwarded the seed to the
        # subprocess it launches -- exactly the bug this test must catch.
        from tractography import saf as saf_module
        from tractography.saf import SAFConfig, run_saf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._make_subject(root)
            fs_dir = self._make_fs_dir(root)
            fod = root / "wmfod.mif"
            fod.write_text("stub", encoding="utf-8")
            tractogram = root / "input.tck"
            tractogram.write_text("stub", encoding="utf-8")
            output_dir = root / "seed-env-saf"
            backend = root / "fake_saf_backend"
            backend.mkdir()
            (backend / "SAF_pipeline.py").write_text(
                "from Support.sys_exec import sys_exec\n"
                "class _PS:\n"
                "    dgn = False\n"
                "def SAF_pipeline(*args):\n"
                "    sys_exec('echo mrtrix-seed-probe', _PS())\n",
                encoding="utf-8",
            )
            backend_status = {"available": True, "missing_required": []}

            # Simulate startup_environment() returning a real-Slicer-style
            # snapshot that predates the seed being set, i.e. one that does
            # NOT itself contain MRTRIX_RNG_SEED, to prove sys_exec forwards
            # the seed explicitly (via its own extra_env argument) rather
            # than happening to inherit it from a snapshot that already had
            # it (a fixed mocked env with no extra_env support would let a
            # broken sys_exec pass this test by accident).
            fixed_snapshot = {"PATH": "/usr/bin"}
            observed_envs: list[dict] = []

            def fake_startup_environment(extra_env=None):
                merged = dict(fixed_snapshot)
                if extra_env:
                    merged.update(extra_env)
                return merged

            def fake_run(cmd, **kwargs):
                observed_envs.append(dict(kwargs.get("env") or {}))
                return mock.Mock(returncode=0, stdout="", stderr="")

            self.assertNotIn("MRTRIX_RNG_SEED", os.environ)
            sys.modules.pop("SAF_pipeline", None)
            try:
                with mock.patch.object(
                    saf_module,
                    "inspect_saf_backend",
                    return_value=backend_status,
                ), mock.patch.object(
                    saf_module,
                    "bundled_saf_backend_dir",
                    return_value=backend,
                ), mock.patch(
                    "Support.sys_exec.startup_environment", side_effect=fake_startup_environment
                ), mock.patch(
                    "Support.sys_exec.subprocess.run", side_effect=fake_run
                ):
                    # The fake backend does not produce SAF.tck, so run_saf()
                    # still raises after the sys_exec call above; that
                    # failure is expected and orthogonal to this assertion.
                    with self.assertRaises(RuntimeError):
                        run_saf(
                            SAFConfig(
                                subject_id=subject.name,
                                input_root=subject,
                                fs_dir=fs_dir,
                                fod_path=fod,
                                output_dir=output_dir,
                                pregenerated_tractogram=tractogram,
                                maxlen=None,
                                random_seed=424242,
                                execute=True,
                            )
                        )
            finally:
                sys.modules.pop("SAF_pipeline", None)

            self.assertTrue(observed_envs, "sys_exec's subprocess.run was never called")
            self.assertEqual(observed_envs[0].get("MRTRIX_RNG_SEED"), "424242")
            # The rest of the (mocked) snapshot must still be present --
            # forwarding the seed must not replace the whole environment.
            self.assertEqual(observed_envs[0].get("PATH"), "/usr/bin")
            # The environment variable must not leak past the call.
            self.assertNotIn("MRTRIX_RNG_SEED", os.environ)

    def test_saf_failure_finalizes_manifest_and_qc_before_raising(self) -> None:
        from tractography import saf as saf_module
        from tractography.saf import SAFConfig, run_saf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._make_subject(root)
            fs_dir = self._make_fs_dir(root)
            fod = root / "wmfod.mif"
            fod.write_text("stub", encoding="utf-8")
            tractogram = root / "input.tck"
            tractogram.write_text("stub", encoding="utf-8")
            output_dir = root / "failed-saf"
            backend_status = {
                "available": False,
                "missing_required": ["mrtrix:tckstats"],
            }

            with mock.patch.object(
                saf_module,
                "inspect_saf_backend",
                return_value=backend_status,
            ):
                with self.assertRaisesRegex(RuntimeError, "backend is incomplete"):
                    run_saf(
                        SAFConfig(
                            subject_id=subject.name,
                            input_root=subject,
                            fs_dir=fs_dir,
                            fod_path=fod,
                            output_dir=output_dir,
                            pregenerated_tractogram=tractogram,
                            maxlen=None,
                            execute=True,
                        )
                    )

            manifest = json.loads((output_dir / "saf_manifest.json").read_text(encoding="utf-8"))
            qc = json.loads((output_dir / "saf_qc.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["return_code"], 1)
            self.assertIn("mrtrix:tckstats", manifest["error"])
            self.assertEqual(qc["status"], "failed")
            self.assertEqual(qc["return_code"], 1)
            self.assertIn("mrtrix:tckstats", qc["error"])

    def test_saf_backend_exception_is_persisted_before_raising(self) -> None:
        from tractography import saf as saf_module
        from tractography.saf import SAFConfig, run_saf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._make_subject(root)
            fs_dir = self._make_fs_dir(root)
            fod = root / "wmfod.mif"
            fod.write_text("stub", encoding="utf-8")
            tractogram = root / "input.tck"
            tractogram.write_text("stub", encoding="utf-8")
            output_dir = root / "failed-backend"
            backend = root / "fake_saf_backend"
            backend.mkdir()
            (backend / "SAF_pipeline.py").write_text(
                "def SAF_pipeline(*args):\n"
                "    raise ValueError('synthetic SAF failure')\n",
                encoding="utf-8",
            )
            backend_status = {"available": True, "missing_required": []}

            sys.modules.pop("SAF_pipeline", None)
            try:
                with mock.patch.object(
                    saf_module,
                    "inspect_saf_backend",
                    return_value=backend_status,
                ), mock.patch.object(
                    saf_module,
                    "bundled_saf_backend_dir",
                    return_value=backend,
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic SAF failure"):
                        run_saf(
                            SAFConfig(
                                subject_id=subject.name,
                                input_root=subject,
                                fs_dir=fs_dir,
                                fod_path=fod,
                                output_dir=output_dir,
                                pregenerated_tractogram=tractogram,
                                maxlen=None,
                                execute=True,
                            )
                        )
            finally:
                sys.modules.pop("SAF_pipeline", None)

            manifest = json.loads((output_dir / "saf_manifest.json").read_text(encoding="utf-8"))
            qc = json.loads((output_dir / "saf_qc.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(qc["status"], "failed")
            self.assertEqual(qc["return_code"], 1)
            self.assertEqual(qc["error"], "synthetic SAF failure")

    def test_saf_execute_records_streamline_count_and_vtk(self) -> None:
        from tractography import saf as saf_module
        from tractography.saf import SAFConfig, run_saf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._make_subject(root)
            fs_dir = self._make_fs_dir(root)
            fod = root / "wmfod.mif"
            fod.write_text("stub", encoding="utf-8")
            bin_dir = root / "mrtrix-bin"
            bin_dir.mkdir()
            for command, body in {
                "mrconvert": "#!/bin/sh\nexit 0\n",
                "tckgen": "#!/bin/sh\nexit 0\n",
                "tckstats": "#!/bin/sh\nprintf '42\\n'\n",
                "tckconvert": "#!/bin/sh\ncp \"$1\" \"$2\"\n",
            }.items():
                path = bin_dir / command
                path.write_text(body, encoding="utf-8")
                path.chmod(0o755)

            # Use a fake bundled backend so the facade can exercise its
            # post-processing contract without running the full SAF algorithm.
            backend = root / "fake_saf_backend"
            backend.mkdir()
            (backend / "SAF_pipeline.py").write_text(
                "from pathlib import Path\n"
                "def SAF_pipeline(*args):\n"
                "    out = Path(args[args.index('outf') + 1])\n"
                "    out.mkdir(parents=True, exist_ok=True)\n"
                "    (out / 'SAF.tck').write_text('tractogram')\n",
                encoding="utf-8",
            )

            def fake_export(_tck_path, vtk_path, **_kwargs):
                vtk_path.write_text("vtk", encoding="utf-8")
                return True, None

            # The fake MRtrix discovery files above are intentionally inert.
            # Mock post-processing itself so this facade test is portable to
            # Windows, where POSIX shebang scripts are not executable files.
            with mock.patch.object(
                saf_module,
                "bundled_saf_backend_dir",
                return_value=backend,
            ), mock.patch.object(
                saf_module,
                "_count_tck_streamlines",
                return_value=(42, None),
            ), mock.patch.object(
                saf_module,
                "_export_tck_to_vtk",
                side_effect=fake_export,
            ), mock.patch.object(
                saf_module,
                "_tractogram_summary",
                return_value={
                    "streamline_count": 42,
                    "all_points_finite": True,
                    "length_mm": {"min": 5.0, "median": 18.0, "max": 39.0},
                },
            ):
                result = run_saf(
                    SAFConfig(
                        subject_id=subject.name,
                        input_root=subject,
                        fs_dir=fs_dir,
                        fod_path=fod,
                        mrtrix_bin_dir=bin_dir,
                        execute=True,
                    )
                )
            sys.modules.pop("SAF_pipeline", None)

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(result.streamline_count, 42)
            self.assertEqual(manifest["streamline_count"], 42)
            self.assertEqual(qc["streamline_count"], 42)
            self.assertTrue((result.output_dir / "SAF.vtk").is_file())
            self.assertEqual(result.saf_vtk_path, result.output_dir / "SAF.vtk")

    def test_slicer_logic_prefers_loading_saf_vtk(self) -> None:
        from tractography.module_logic import TractographyLogicMixin
        from tractography.saf import SAFResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "saf"
            output_dir.mkdir()
            tck = output_dir / "SAF.tck"
            vtk = output_dir / "SAF.vtk"
            manifest = output_dir / "saf_manifest.json"
            qc = output_dir / "saf_qc.json"
            for path in (tck, vtk, manifest, qc):
                path.write_text(path.name, encoding="utf-8")

            class _Logic(TractographyLogicMixin):
                def __init__(self) -> None:
                    self.loaded = []

                def log(self, message: str) -> None:
                    pass

                def _load_fiber_bundle_node(self, fiber_path: Path, node_name: str) -> None:
                    self.loaded.append((fiber_path, node_name))

            fake_result = SAFResult(
                subject_id="100307",
                output_dir=output_dir,
                saf_tractogram=tck,
                manifest_path=manifest,
                qc_path=qc,
                executed=True,
                backend_status={},
                saf_vtk_path=vtk,
                streamline_count=63589,
                return_code=0,
            )
            logic = _Logic()
            with mock.patch("tractography.module_logic.run_saf", return_value=fake_result):
                logic.runSAFPipeline(
                    subject_id="100307",
                    input_root=root,
                    fs_dir=root,
                    fod_path=root / "fod.mif",
                    execute=True,
                )

            self.assertEqual(logic.loaded, [(vtk, "100307_SAF")])

    def test_mrtrix_bin_dir_is_used_for_backend_discovery(self) -> None:
        from tractography import inspect_saf_backend

        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "mrtrix-bin"
            bin_dir.mkdir()
            for command in ["mrconvert", "tckgen", "tckstats", "tcksample"]:
                path = bin_dir / command
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)

            status = inspect_saf_backend(
                mrtrix_bin_dir=bin_dir,
                require_mrconvert=True,
                require_tckgen=True,
                require_tckstats=True,
                require_tcksample=True,
            )
            self.assertEqual(status["mrtrix"]["bin_dir"], bin_dir.resolve().as_posix())
            self.assertEqual(status["missing_required"], [])
            self.assertTrue(status["available"])

    def test_mrtrix_env_bin_dir_is_used_for_backend_discovery(self) -> None:
        from tractography import inspect_saf_backend

        old_value = os.environ.get("SLICERDSPACE_MRTRIX_BIN")
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "mrtrix-env-bin"
            bin_dir.mkdir()
            mrconvert = bin_dir / "mrconvert"
            mrconvert.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mrconvert.chmod(0o755)
            os.environ["SLICERDSPACE_MRTRIX_BIN"] = bin_dir.as_posix()
            try:
                status = inspect_saf_backend(
                    require_mrconvert=True,
                    require_tckgen=False,
                    require_tckstats=False,
                )
            finally:
                if old_value is None:
                    os.environ.pop("SLICERDSPACE_MRTRIX_BIN", None)
                else:
                    os.environ["SLICERDSPACE_MRTRIX_BIN"] = old_value
            self.assertEqual(status["mrtrix_required"]["mrconvert"], mrconvert.resolve().as_posix())
            self.assertEqual(status["missing_required"], [])

    def test_mrtrix_prefix_mode_is_recorded_without_path_lookup(self) -> None:
        from tractography import inspect_saf_backend

        status = inspect_saf_backend(
            mrtrix_prefix="singularity exec mrtrix.sif ",
            require_mrconvert=True,
            require_tckgen=True,
            require_tckstats=False,
        )
        self.assertEqual(status["mrtrix"]["discovery"], "prefix_unchecked")
        self.assertEqual(status["mrtrix_required"]["tckgen"], "singularity exec mrtrix.sif tckgen")
        self.assertEqual(status["missing_required"], [])

    def test_configured_mrtrix_bin_is_passed_as_a_path_to_saf(self) -> None:
        from tractography.saf import SAFConfig, _build_saf_args

        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "MRtrix with spaces; ampersand &" / "bin"
            bin_dir.mkdir(parents=True)
            config = SAFConfig(
                subject_id="100307",
                input_root=Path(tmp),
                fs_dir=Path(tmp) / "fs",
                fod_path=Path(tmp) / "fod.mif",
                mrtrix_bin_dir=bin_dir,
            )
            args = _build_saf_args(config, Path(tmp) / "output")
            self.assertNotIn("mrtrix_prefix", args)
            self.assertEqual(args[args.index("mrtrix_bin_dir") + 1], bin_dir.resolve().as_posix())

    def test_argv_runner_resolves_configured_bin_and_preserves_seed(self) -> None:
        from Support.sys_exec import sys_exec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bin_dir = root / "MRtrix tools"
            executable = bin_dir / ("tckstats.exe" if sys.platform == "win32" else "tckstats")
            bin_dir.mkdir()
            executable.touch()
            executable.chmod(0o755)
            path_with_shell_characters = str(root / "literal; file.tck")
            ps = types.SimpleNamespace(dgn=False, mrtrix_bin_dir=str(bin_dir))
            with mock.patch.dict("os.environ", {"MRTRIX_RNG_SEED": "31"}), \
                    mock.patch("Support.sys_exec.startup_environment", side_effect=lambda extra_env: dict(
                        PATH="original", **(extra_env or {}))), \
                    mock.patch("Support.sys_exec.shutil.which", return_value=str(Path(tmp) / "other-tckstats")) as which, \
                    mock.patch("Support.sys_exec.subprocess.run") as run:
                sys_exec(["tckstats", path_with_shell_characters], ps)
            which.assert_not_called()
            self.assertEqual(run.call_args.kwargs["env"]["PATH"], str(bin_dir) + os.pathsep + "original")
            self.assertEqual(run.call_args.args[0], [str(executable), path_with_shell_characters])
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertEqual(run.call_args.kwargs["env"]["MRTRIX_RNG_SEED"], "31")

    def test_windows_argv_runner_supplies_home_only_to_its_child(self) -> None:
        from Support.sys_exec import sys_exec

        ps = types.SimpleNamespace(dgn=False)
        for environment, expected in (
            ({"HOME": "custom home", "USERPROFILE": "profile"}, "custom home"),
            ({"USERPROFILE": "profile"}, "profile"),
            ({}, "detected home"),
        ):
            with self.subTest(environment=environment), mock.patch("sys.platform", "win32"), \
                    mock.patch("Support.sys_exec.Path.home", return_value=Path("detected home")), \
                    mock.patch("Support.sys_exec.startup_environment", return_value=dict(environment)), \
                    mock.patch("Support.sys_exec.shutil.which", return_value="/absolute/tckstats.exe"), \
                    mock.patch("Support.sys_exec.subprocess.run") as run:
                inherited_home = os.environ.get("HOME")
                sys_exec(["tckstats", "a.tck"], ps)
                self.assertEqual(run.call_args.kwargs["env"]["HOME"], expected)
                self.assertEqual(os.environ.get("HOME"), inherited_home)

    @unittest.skipIf(sys.platform == "win32", "Legacy shell wrappers are tested on POSIX")
    def test_legacy_prefix_still_wraps_the_mrtrix_command(self) -> None:
        from Support.sys_exec import sys_exec

        ps = types.SimpleNamespace(dgn=False, mrtrix_prefix="singularity exec 'MRtrix tools.sif'")
        with mock.patch("Support.sys_exec.subprocess.run") as run:
            sys_exec(["tckstats", "/data/a; b.tck"], ps)
        self.assertTrue(run.call_args.kwargs["shell"])
        self.assertEqual(shlex.split(run.call_args.args[0]),
                         ["singularity", "exec", "MRtrix tools.sif", "tckstats", "/data/a; b.tck"])


if __name__ == "__main__":
    unittest.main()
