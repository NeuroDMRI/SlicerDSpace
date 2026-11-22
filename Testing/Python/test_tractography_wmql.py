#!/usr/bin/env python3
"""Tests for WMQL bundle segmentation integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class WMQLIntegrationTest(unittest.TestCase):
    def test_custom_include_precedence_and_stale_snapshot_refresh(self):
        from tractography.wmql import WMQLConfig, run_wmql_bundle_segmentation, wmql_query_cache_current

        query = self.tmp_path / 'custom.qry'
        query.write_text('import FreeSurfer.qry\nCST.left = Brain_Stem\n')
        local = self.tmp_path / 'FreeSurfer.qry'
        local.write_text('Brain_Stem |= 16\n# selected local definitions\n')
        config = WMQLConfig('100307', self.tractography, self.atlas, self.output_dir,
                            query_path=query, tract_querier_root=self.tract_querier_root, python_executable=self.python)
        result = run_wmql_bundle_segmentation(config)
        previous = json.loads(result.manifest_path.read_text())
        self.assertTrue(wmql_query_cache_current(previous))
        previous_query = Path(previous['staged_query'])
        previous_aliases = previous_query.parent / 'FreeSurfer.qry'
        previous_bytes = previous_aliases.read_bytes()
        self.assertEqual(previous_bytes, local.read_bytes())
        local.write_text('Brain_Stem |= 116\n# deliberately changed synthetic label\n')
        self.assertFalse(wmql_query_cache_current(previous))
        result = run_wmql_bundle_segmentation(config)
        current = json.loads(result.manifest_path.read_text())
        self.assertTrue(wmql_query_cache_current(current))
        current_query = Path(current['staged_query'])
        self.assertNotEqual(previous_query, current_query)
        self.assertEqual(previous_aliases.read_bytes(), previous_bytes)
        self.assertEqual((current_query.parent / 'FreeSurfer.qry').read_bytes(), local.read_bytes())
        self.assertEqual(current['query_provenance']['preset'], 'external_query')
        (current_query.parent / 'FreeSurfer.qry').write_text('Brain_Stem |= 999\n')
        self.assertFalse(wmql_query_cache_current(current))

    def test_import_mutation_during_execution_is_not_successful(self):
        from tractography.wmql import WMQLConfig, run_wmql_bundle_segmentation

        backend = {'available': True, 'commands': {'python': str(self.python), 'tract_querier_executable': None,
                   'root_script': str(self.tract_querier_root / 'scripts/cli_tract_querier.py')}}
        def mutate(command, **kwargs):
            (Path(command[command.index('-q') + 1]).parent / 'FreeSurfer.qry').write_text('Brain_Stem |= 116\n')
            return subprocess.CompletedProcess(command, 0)
        with mock.patch('tractography.wmql.inspect_wmql_backend', return_value=backend), mock.patch(
            'tractography.wmql.subprocess.run', side_effect=mutate
        ), mock.patch('tractography.wmql._wmql_output_contract_error', return_value=None):
            result = run_wmql_bundle_segmentation(WMQLConfig('100307', self.tractography, self.atlas, self.output_dir,
                 tract_querier_root=self.tract_querier_root, python_executable=self.python, execute=True))
        qc = json.loads(result.qc_path.read_text())
        self.assertEqual(qc['status'], 'failed')
        self.assertFalse(qc['automatic_qc_passed'])
        self.assertIn('query/imports changed', qc['error'])

    def test_recursive_query_imports_reject_cycles_and_missing_files(self):
        from tractography.wmql import wmql_query_dependencies

        query = self.tmp_path / 'root.qry'
        query.write_text('import absent.qry\n')
        with self.assertRaises(FileNotFoundError):
            wmql_query_dependencies(query, [self.tmp_path])
        query.write_text('import second.qry\n')
        (self.tmp_path / 'second.qry').write_text('import root.qry\n')
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            wmql_query_dependencies(query, [self.tmp_path])

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # Resolve so paths recorded by the code under test (which resolves
        # its cwd/output paths) compare equal under macOS's /tmp ->
        # /private/tmp symlink.
        self.tmp_path = Path(self.tmp.name).resolve()
        if sys.platform == "win32":
            from shared.external_process import find_pythonslicer
            real_py = find_pythonslicer(Path(sys.executable).parent) or Path(sys.executable)
            py_helper = self.tmp_path / "fake_python.py"
            py_helper.write_text(
                "import pathlib, sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
                "    print('{\"tract_querier\": true, \"nibabel\": true, \"numpy\": true}')\n"
                "    sys.exit(0)\n"
                "out = sys.argv[sys.argv.index('-o') + 1]\n"
                "pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)\n"
                "for name in ('AF.left', 'AF.right', 'CST.left', 'CST.right'):\n"
                "    pathlib.Path(f'{out}_{name}.vtk').write_text(\n"
                "        '# vtk DataFile Version 3.0\\nWMQL unit fixture\\nASCII\\nDATASET POLYDATA\\nPOINTS 2 float\\n0 0 0 1 0 0\\nLINES 1 3\\n2 0 1\\n',\n"
                "        encoding='utf-8',\n"
                "    )\n",
                encoding="utf-8",
            )
            self.python = self.tmp_path / "python.cmd"
            self.python.write_text(f'@echo off\r\n"{real_py}" "{py_helper}" %*\r\n', encoding="utf-8")
        else:
            self.python = self.tmp_path / "python"
            self.python.write_text(
                """#!/bin/sh
if [ "$1" = "-c" ]; then
  echo '{"tract_querier": true, "nibabel": true, "numpy": true}'
  exit 0
fi
script="$1"
shift
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    out="$arg"
  fi
  prev="$arg"
done
mkdir -p "$(dirname "$out")"
for name in AF.left AF.right CST.left CST.right; do
  printf '# vtk DataFile Version 3.0\\nWMQL unit fixture\\nASCII\\nDATASET POLYDATA\\nPOINTS 2 float\\n0 0 0 1 0 0\\nLINES 1 3\\n2 0 1\\n' > "${out}_${name}.vtk"
done
exit 0
""",
                encoding="utf-8",
            )
            self.python.chmod(0o755)
        self.tract_querier_root = self.tmp_path / "tract_querier_root"
        scripts = self.tract_querier_root / "scripts"
        data = self.tract_querier_root / "tract_querier" / "data"
        scripts.mkdir(parents=True)
        data.mkdir(parents=True)
        (scripts / "cli_tract_querier.py").write_text("# fake cli\n", encoding="utf-8")
        (data / "FreeSurfer.qry").write_text("brain_stem |= 16\n", encoding="utf-8")
        # Exercise the complete import graph without depending on an installed author checkout.
        # This fake CLI never interprets anatomical rules; no equivalence claim is made.
        (data / "freesurfer_queries.qry").write_text(
            "# Synthetic import fixture only\nimport FreeSurfer.qry\n", encoding="utf-8"
        )
        default_data = mock.patch('tractography.wmql._default_query_data_dir', return_value=None)
        self.addCleanup(default_data.stop)
        default_data.start()
        self.tractography = self.tmp_path / "wholebrain.vtk"
        self.tractography.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
        self.atlas = self.tmp_path / "wmparc.nii.gz"
        import nibabel as nib
        import numpy as np
        labels = np.array([0, 2, 41, 16, 1001, 2001, 1024, 2024], dtype=np.int16).reshape(2, 2, 2)
        nib.save(nib.Nifti1Image(labels, np.eye(4)), self.atlas)
        self.output_dir = self.tmp_path / "wmql_out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_query_is_bundled(self) -> None:
        from tractography import bundled_wmql_query_path

        query = bundled_wmql_query_path()
        self.assertTrue(query.is_file())
        text = query.read_text(encoding="utf-8")
        self.assertIn("import FreeSurfer.qry", text)
        self.assertIn('UF.side =', text)
        self.assertNotIn("endpoints_in(Brain_Stem)", text)
        self.assertIn("CST.side", bundled_wmql_query_path('published_endpoints').read_text())

    def test_default_import_graph_is_required_without_local_author_data(self):
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        config = WMQLConfig('100307', self.tractography, self.atlas, self.output_dir,
                            tract_querier_root=self.tract_querier_root, python_executable=self.python)
        result = run_wmql_bundle_segmentation(config)
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual({Path(path).name for path in manifest['query_dependencies']},
                         {'freesurfer_queries.qry', 'FreeSurfer.qry'})
        (self.tract_querier_root / 'tract_querier/data/freesurfer_queries.qry').unlink()
        # Installed official query resources must not depend on the backend checkout's data directory.
        run_wmql_bundle_segmentation(config)

    def test_builds_wmql_command_without_running(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        result = run_wmql_bundle_segmentation(
            WMQLConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_path=self.atlas,
                output_dir=self.output_dir,
                tract_querier_root=self.tract_querier_root,
                python_executable=self.python,
                execute=False,
            )
        )

        self.assertFalse(result.executed)
        self.assertTrue(result.backend_status["available"])
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.qc_path.is_file())
        command = list(result.command)
        self.assertEqual(command[0], self.python.resolve().as_posix())
        self.assertIn("-t", command)
        self.assertIn("-a", command)
        self.assertIn("-q", command)
        self.assertNotIn("--query_selection", command)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["kind"], "wmql_bundle_segmentation")
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["backend"]["repository"], "https://github.com/demianw/tract_querier")
        self.assertTrue(Path(manifest['staged_query']).is_file())
        self.assertEqual(Path(manifest['staged_query']).parent.parent, self.output_dir / 'queries')

    def test_execute_true_summarizes_created_bundles_with_fake_backend(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        result = run_wmql_bundle_segmentation(
            WMQLConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_path=self.atlas,
                output_dir=self.output_dir,
                tract_querier_root=self.tract_querier_root,
                python_executable=self.python,
                execute=True,
            )
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.return_code, 0)
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "completed")
        self.assertFalse(qc['automatic_qc_passed'])
        self.assertFalse(qc['quantification_eligible'])
        self.assertIsNone(qc['error'])
        self.assertTrue(qc['execution_passed'])
        self.assertEqual(qc['scientific_status'], 'partial_bundle_coverage')
        self.assertEqual(qc['bundle_assessment']['requested_count'], 48)
        self.assertEqual(qc["output_summary"]["bundle_count"], 4)
        self.assertEqual(qc["output_summary"]["bundle_nonempty_count"], 4)
        self.assertEqual(len(qc["output_summary"]["bundles"]), 4)
        self.assertIn("size", qc["output_summary"]["bundles"][0])
        self.assertEqual(len(list(self.output_dir.glob('100307_desc-wmql_*.vtk'))), 4)

    def test_sampling_policy_is_not_just_nonempty(self):
        from tractography.wmql import _wmql_output_contract_error, MIN_BUNDLE_STREAMLINES
        summary = dict(bundle_count=1, bundle_readable_count=1, bundle_nonempty_count=1,
                       bundles=[dict(query_name='af.left', streamline_count=1)])
        self.assertIn('insufficient', _wmql_output_contract_error(summary, 1, ('AF.left',)))
        summary['bundles'][0]['streamline_count'] = MIN_BUNDLE_STREAMLINES
        self.assertIsNone(_wmql_output_contract_error(summary, 1, ('AF.left',)))

    def test_official_invalid_atlas_writes_failed_qc_without_starting_backend(self):
        from tractography import WMQLConfig, run_wmql_bundle_segmentation
        import nibabel as nib
        import numpy as np
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int16), np.eye(4)), self.atlas)
        with self.assertRaisesRegex(ValueError, 'Atlas compatibility failed'):
            run_wmql_bundle_segmentation(WMQLConfig('100307', self.tractography, self.atlas, self.output_dir,
                tract_querier_root=self.tract_querier_root, python_executable=self.python, execute=True))
        qc = json.loads((self.output_dir / 'wmql_qc.json').read_text())
        self.assertEqual(qc['status'], 'failed')
        self.assertFalse(qc['execution_passed'])
        self.assertFalse(qc['automatic_qc_passed'])
        self.assertIn('regional label IDs', qc['error'])
        self.assertFalse(list(self.output_dir.glob('100307_desc-wmql_*')))

    def test_windows_stages_query_includes_without_drive_separated_I_argument(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        with mock.patch("tractography.wmql._requires_windows_query_staging", return_value=True):
            result = run_wmql_bundle_segmentation(
                WMQLConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_path=self.atlas,
                    output_dir=self.output_dir,
                    tract_querier_root=self.tract_querier_root,
                    python_executable=self.python,
                    execute=False,
                )
            )

        self.assertNotIn("-I", result.command)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        snapshot = Path(manifest['staged_query']).parent
        self.assertTrue((snapshot / 'FreeSurfer.qry').is_file())
        self.assertEqual(manifest["execution_cwd"], snapshot.as_posix())

    def test_failed_backend_writes_qc_before_reraising(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        backend = {
            "available": True,
            "commands": {
                "tract_querier_executable": None,
                "root_script": (self.tract_querier_root / "scripts" / "cli_tract_querier.py").as_posix(),
                "python": self.python.as_posix(),
            },
            "missing_required": [],
        }
        failure = subprocess.CalledProcessError(7, ["tract_querier"])
        with (
            mock.patch("tractography.wmql.inspect_wmql_backend", return_value=backend),
            mock.patch("tractography.wmql.subprocess.run", side_effect=failure),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            run_wmql_bundle_segmentation(
                WMQLConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_path=self.atlas,
                    output_dir=self.output_dir,
                    tract_querier_root=self.tract_querier_root,
                    python_executable=self.python,
                    execute=True,
                )
            )

        qc = json.loads((self.output_dir / "wmql_qc.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.output_dir / "wmql_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "failed")
        self.assertEqual(qc["return_code"], 7)
        self.assertEqual(manifest["status"], "failed")

    def test_backend_launch_error_writes_qc_before_reraising(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        backend = {
            "available": True,
            "commands": {
                "tract_querier_executable": None,
                "root_script": (self.tract_querier_root / "scripts" / "cli_tract_querier.py").as_posix(),
                "python": self.python.as_posix(),
            },
            "missing_required": [],
        }
        failure = OSError("backend could not be launched")
        with (
            mock.patch("tractography.wmql.inspect_wmql_backend", return_value=backend),
            mock.patch("tractography.wmql.subprocess.run", side_effect=failure),
            self.assertRaises(OSError),
        ):
            run_wmql_bundle_segmentation(
                WMQLConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_path=self.atlas,
                    output_dir=self.output_dir,
                    tract_querier_root=self.tract_querier_root,
                    python_executable=self.python,
                    execute=True,
                )
            )

        qc = json.loads((self.output_dir / "wmql_qc.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.output_dir / "wmql_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "failed")
        self.assertIsNone(qc["return_code"])
        self.assertIn("OSError", qc["error"])
        self.assertEqual(manifest["status"], "failed")

    def test_bundle_parcellation_dispatches_wmql_backend(self) -> None:
        from tractography import BundleParcellationConfig, WMQLConfig, run_bundle_parcellation

        result = run_bundle_parcellation(
            BundleParcellationConfig(
                backend="wmql",
                wmql=WMQLConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_path=self.atlas,
                    output_dir=self.output_dir,
                    tract_querier_root=self.tract_querier_root,
                    python_executable=self.python,
                ),
            )
        )

        self.assertEqual(result.manifest_path.name, "wmql_manifest.json")

    def test_rejects_backend_config_mix(self) -> None:
        from tractography import BundleParcellationConfig, run_bundle_parcellation

        with self.assertRaisesRegex(ValueError, "requires only the wmql config"):
            run_bundle_parcellation(BundleParcellationConfig(backend="wmql"))

    def test_wmql_accel_subprocess_sets_single_thread_blas_env(self) -> None:
        from tractography import WMQLConfig, run_wmql_bundle_segmentation

        backend = {
            "available": True,
            "commands": {
                "tract_querier_executable": None,
                "root_script": (self.tract_querier_root / "scripts" / "cli_tract_querier.py").as_posix(),
                "python": self.python.as_posix(),
            },
            "missing_required": [],
        }
        captured_env: dict[str, str] = {}

        def _fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            out_prefix = Path(cmd[cmd.index("-o") + 1])
            out_prefix.parent.mkdir(parents=True, exist_ok=True)
            (out_prefix.parent / f"{out_prefix.name}_af.left.vtk").write_bytes(b"fake")
            return mock.Mock(returncode=0)

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SLICERDSPACE_DISABLE_WMQL_ACCEL")
        }
        with (
            mock.patch.dict(os.environ, clean_env, clear=True),
            mock.patch("tractography.wmql.inspect_wmql_backend", return_value=backend),
            mock.patch("tractography.wmql.subprocess.run", side_effect=_fake_run),
        ):
            run_wmql_bundle_segmentation(
                WMQLConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_path=self.atlas,
                    output_dir=self.output_dir,
                    tract_querier_root=self.tract_querier_root,
                    python_executable=self.python,
                    execute=True,
                )
            )

        for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(captured_env.get(key), "1")

    def test_wmql_accel_multithreaded_affine_matches_single_thread(self) -> None:
        import numpy as np
        from tractography.wmql_accel import FastTractographySpatialIndexing

        rng = np.random.default_rng(42)
        img = np.zeros((16, 18, 14), dtype=np.int32)
        img[2:8, 2:10, 2:8] = 1001
        img[8:14, 6:15, 5:12] = 2002
        # Non-trivial rotated/scaled/translated affine
        theta = 0.25
        affine = np.array(
            [
                [1.25 * np.cos(theta), -1.25 * np.sin(theta), 0.0, -10.5],
                [1.25 * np.sin(theta), 1.25 * np.cos(theta), 0.0, -12.0],
                [0.0, 0.0, 1.5, -8.25],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # > 2 * 16384 streamlines so _index_cpu exercises ThreadPoolExecutor across multiple chunks
        n_tracts = 36000
        lengths = np.full(n_tracts, 3, dtype=np.int64)
        offsets = np.arange(0, (n_tracts + 1) * 3, 3, dtype=np.int64)
        ijk_pts = rng.uniform(-1.0, 17.0, size=(n_tracts * 3, 3))
        points = (ijk_pts @ affine[:3, :3].T + affine[:3, 3]).astype(np.float32)

        idx_obj = object.__new__(FastTractographySpatialIndexing)
        idx_obj.affine_ijk_2_ras = affine
        idx_obj.affine_ras_2_ijk = np.linalg.inv(affine)
        idx_obj.crossing_threshold = 2.0
        uniq_labels = np.unique(img)
        lut = np.zeros(int(uniq_labels.max()) + 1, dtype=np.int32)
        lut[uniq_labels] = np.arange(uniq_labels.size, dtype=np.int32)

        res_single = idx_obj._index_cpu(
            points=points,
            offsets=offsets,
            lengths=lengths,
            img_int=img,
            uniq_labels=uniq_labels,
            lut=lut,
            lut_offset=0,
            workers=1,
        )
        res_multi = idx_obj._index_cpu(
            points=points,
            offsets=offsets,
            lengths=lengths,
            img_int=img,
            uniq_labels=uniq_labels,
            lut=lut,
            lut_offset=0,
            workers=8,
        )

        for a, b in zip(res_single[:-1], res_multi[:-1]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(res_single[-1], res_multi[-1])
        self.assertTrue(res_multi[-1])


if __name__ == "__main__":
    unittest.main()
