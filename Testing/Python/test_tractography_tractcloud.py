#!/usr/bin/env python3
"""Lightweight tests for TractCloud bundle parcellation integration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import types

try:
    from vtp_fixture import vtp_lines
except ImportError:
    from Testing.Python.vtp_fixture import vtp_lines


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TractCloudIntegrationTest(unittest.TestCase):
    def test_bootstrap_supplies_upstream_fallback_logger_without_changing_cli(self):
        from tractography.tractcloud import TRACTCLOUD_COMPAT_BOOTSTRAP
        package = types.ModuleType('tractcloud')
        pipeline = types.ModuleType('tractcloud.pipeline')
        package.pipeline = pipeline
        with mock.patch.dict(sys.modules, {'tractcloud': package, 'tractcloud.pipeline': pipeline}), \
             mock.patch('runpy.run_module') as invoke, \
             mock.patch.dict(sys.modules, {'torch': mock.Mock(__version__='test-double')}):
            exec(TRACTCLOUD_COMPAT_BOOTSTRAP, {})
        import logging
        self.assertIs(pipeline.logging, logging)
        invoke.assert_called_once_with('tractcloud', run_name='__main__')

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        if sys.platform == "win32":
            from shared.external_process import find_pythonslicer
            real_py = find_pythonslicer(Path(sys.executable).parent) or Path(sys.executable)
            py_helper = self.tmp_path / "fake_pythonslicer.py"
            py_helper.write_text(
                "import pathlib, sys\n"
                "if len(sys.argv) >= 3 and sys.argv[1] == '-c' and 'importlib.util' in sys.argv[2]:\n"
                "    print('{\"tractcloud\": true, \"torch\": true}')\n"
                "    sys.exit(0)\n"
                "if '--output-dir' in sys.argv:\n"
                "    out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "else:\n"
                "    out = next(pathlib.Path(__file__).parent.glob('.tractcloud_out-tractcloud-*'))\n"
                "assoc = out / 'Association'\n"
                "assoc.mkdir(parents=True, exist_ok=True)\n"
                f"(assoc / 'arcuate_fasciculus.vtp').write_text({vtp_lines(3)!r}, encoding='utf-8')\n",
                encoding="utf-8",
            )
            self.python = self.tmp_path / "PythonSlicer.cmd"
            self.python.write_text(f'@echo off\r\n"{real_py}" "{py_helper}" %*\r\n', encoding="utf-8")
        else:
            self.python = self.tmp_path / "PythonSlicer"
            self.python.write_text(
                """#!/bin/sh
if [ "$1" = "-c" ] && [ "$#" -eq 2 ]; then
  echo '{"tractcloud": true, "torch": true}'
  exit 0
fi
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-dir" ]; then
    out="$arg"
  fi
  prev="$arg"
done
mkdir -p "$out/Association"
printf 'VTP_FIXTURE' > "$out/Association/arcuate_fasciculus.vtp"
exit 0
""".replace("VTP_FIXTURE", vtp_lines(3)),
                encoding="utf-8",
            )
            self.python.chmod(0o755)
        self.tractography = self.tmp_path / "subject.vtp"
        self.tractography.write_text("<VTKFile></VTKFile>\n", encoding="utf-8")
        self.output_dir = self.tmp_path / "tractcloud_out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builds_tractcloud_command_without_running(self) -> None:
        from tractography import TractCloudConfig, run_tractcloud_parcellation

        result = run_tractcloud_parcellation(
            TractCloudConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                output_dir=self.output_dir,
                python_executable=self.python,
                include_other=True,
                device="cpu",
                batch_size=512,
                create_mrb=True,
                execute=False,
            )
        )

        self.assertFalse(result.executed)
        self.assertTrue(result.backend_status["available"])
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.qc_path.is_file())
        command = list(result.command)
        self.assertEqual(command[:3], [self.python.absolute().as_posix(), "-m", "tractcloud"])
        self.assertIn("--include-other", command)
        self.assertIn("--mrb", command)
        self.assertIn("cpu", command)
        self.assertIn("512", command)

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["kind"], "tractcloud_parcellation")
        self.assertEqual(manifest["status"], "planned")
        self.assertFalse(manifest["executed"])
        self.assertEqual(manifest["backend"]["repository"], "https://github.com/SlicerDMRI/TractCloud")

    def test_external_source_root_is_added_to_backend_probe_and_manifest(self) -> None:
        from tractography import TractCloudConfig, run_tractcloud_parcellation

        tractcloud_root = self.tmp_path / "TractCloud"
        package_dir = tractcloud_root / "src" / "tractcloud"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

        result = run_tractcloud_parcellation(
            TractCloudConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                output_dir=self.output_dir,
                tractcloud_root=tractcloud_root,
                python_executable=self.python,
                execute=False,
            )
        )

        self.assertTrue(result.backend_status["available"])
        self.assertEqual(result.backend_status["source"]["root"], tractcloud_root.resolve().as_posix())
        self.assertEqual(result.backend_status["source"]["pythonpath"], (tractcloud_root / "src").resolve().as_posix())
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["tractcloud_root"], tractcloud_root.resolve().as_posix())

    def test_execute_true_summarizes_created_bundles_with_fake_backend(self) -> None:
        from tractography import TractCloudConfig, run_tractcloud_parcellation

        result = run_tractcloud_parcellation(
            TractCloudConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                output_dir=self.output_dir,
                python_executable=self.python,
                execute=True,
            )
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.return_code, 0)
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "completed")
        self.assertEqual(qc["output_summary"]["bundle_count"], 1)
        self.assertEqual(qc["output_summary"]["assigned_streamline_count"], 3)
        self.assertTrue(qc['automatic_qc_passed'])  # Decoded geometry only.
        self.assertTrue(qc['sampling_warnings'])
        self.assertFalse(qc['quantification_eligible'])

    def test_output_summary_reports_nonempty_bundle_streamlines(self) -> None:
        from tractography.tractcloud import summarize_tractcloud_outputs

        association = self.output_dir / "Association"
        association.mkdir(parents=True)
        (association / "AF.vtp").write_text(
            vtp_lines(3),
            encoding="utf-8",
        )
        (association / "UF.vtp").write_text(
            vtp_lines(0),
            encoding="utf-8",
        )

        summary = summarize_tractcloud_outputs(self.output_dir)

        self.assertEqual(summary["bundle_count"], 2)
        self.assertEqual(summary["bundle_readable_count"], 2)
        self.assertEqual(summary["bundle_nonempty_count"], 1)
        self.assertEqual(summary["assigned_streamline_count"], 3)
        self.assertEqual(summary["empty_bundle_files"], ["Association/UF.vtp"])
        self.assertEqual(summary["unreadable_bundle_files"], [])

    def test_zero_exit_without_new_products_cannot_reuse_old_success(self) -> None:
        from tractography import TractCloudConfig, run_tractcloud_parcellation
        old = self.output_dir / 'Association' / 'old.vtp'
        old.parent.mkdir(parents=True)
        old.write_text(vtp_lines(3))
        backend = {'available': True, 'python_executable': str(self.python), 'source': None}
        with mock.patch('tractography.tractcloud.inspect_tractcloud_backend', return_value=backend), \
             mock.patch('tractography.tractcloud.subprocess.run', return_value=mock.Mock(returncode=0)):
            result = run_tractcloud_parcellation(TractCloudConfig(
                subject_id='100307', input_tractography=self.tractography,
                output_dir=self.output_dir, python_executable=self.python, execute=True))
        qc = json.loads(result.qc_path.read_text())
        self.assertEqual(qc['status'], 'failed')
        self.assertEqual(qc['output_summary']['bundle_count'], 0)
        self.assertEqual(result.status, 'failed')
        from shared.orchestrator import _task_from_result
        self.assertEqual(_task_from_result('bundles.tractcloud', result).status, 'failed')
        self.assertTrue(old.exists())

    def test_bundle_parcellation_dispatches_tractcloud_backend(self) -> None:
        from tractography import BundleParcellationConfig, TractCloudConfig, run_bundle_parcellation

        result = run_bundle_parcellation(
            BundleParcellationConfig(
                backend="tractcloud",
                tractcloud=TractCloudConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    output_dir=self.output_dir,
                    python_executable=self.python,
                ),
            )
        )

        self.assertEqual(result.manifest_path.name, "tractcloud_manifest.json")

    def test_rejects_backend_config_mix(self) -> None:
        from tractography import BundleParcellationConfig, run_bundle_parcellation

        with self.assertRaisesRegex(ValueError, "requires only the tractcloud config"):
            run_bundle_parcellation(BundleParcellationConfig(backend="tractcloud"))


if __name__ == "__main__":
    unittest.main()
