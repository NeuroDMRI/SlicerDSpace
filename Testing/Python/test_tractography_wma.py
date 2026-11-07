#!/usr/bin/env python3
"""Lightweight tests for the WMA tractography integration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from vtp_fixture import vtp_lines


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class WMAIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name).resolve()
        self.wma_root = self.tmp_path / "wma"
        self.wma_bin = self.wma_root / "bin"
        self.wma_bin.mkdir(parents=True)
        self.wma_script = self.wma_bin / "wm_apply_ORG_atlas_to_subject.sh"
        self.wma_script.write_text(
            """#!/bin/sh
while [ "$#" -gt 0 ]; do
  case "$1" in
    -i) input="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case_id=$(basename "$input")
case_id=${case_id%.*}
mkdir -p "$out/$case_id/FiberClustering" "$out/$case_id/AnatomicalTracts"
index=1
while [ "$index" -le 73 ]; do
  case "$index" in
    1) name=T_AF_left ;; 2) name=T_AF_right ;; 3) name=T_CST_left ;;
    4) name=T_CST_right ;; 5) name=T_CC4 ;; *) name=T_$(printf '%03d' "$index") ;;
  esac
  printf 'VTP_FIXTURE\n' > "$out/$case_id/AnatomicalTracts/$name.vtp"
  index=$((index + 1))
done
exit 0
""".replace("VTP_FIXTURE", vtp_lines(20)),
            encoding="utf-8",
        )
        self.wma_script.chmod(0o755)
        (self.wma_root / "VENDORED_SOURCE.md").write_text(
            "# Vendored source\n\n"
            "- Repository: https://github.com/SlicerDMRI/whitematteranalysis\n"
            "- Commit: 0123456789abcdef\n"
            "- Date added: 2026-07-25\n",
            encoding="utf-8",
        )
        for script_name in (
            "wm_append_clusters_to_anatomical_tracts.py",
            "wm_assess_cluster_location_by_hemisphere.py",
            "wm_cluster_from_atlas.py",
            "wm_cluster_remove_outliers.py",
            "wm_harden_transform.py",
            "wm_register_to_atlas_new.py",
            "wm_separate_clusters_by_hemisphere.py",
            "wm_diffusion_measurements.py",
            "xvfb-run",
            "Xvfb",
            "xauth",
        ):
            script = self.wma_bin / script_name
            script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)
        self.tractography = self.tmp_path / "subject.vtk"
        self.tractography.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
        self.atlas_dir = self.tmp_path / "ORG-Atlases"
        self.atlas_dir.mkdir()
        (self.atlas_dir / "ORG-RegAtlas-100HCP").mkdir()
        (self.atlas_dir / "ORG-800FC-100HCP").mkdir()
        self.slicer_executable = self.tmp_path / "Slicer"
        self.slicer_executable.write_text("#!/bin/sh\n", encoding="utf-8")
        self.slicer_executable.chmod(0o755)
        self.output_dir = self.tmp_path / "out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_macos_app_discovers_python_libraries_and_dmri_modules(self):
        from tractography.wma import _slicerdmri_module_paths, _slicer_runtime_library_paths
        contents = self.tmp_path / 'Slicer.app' / 'Contents'
        slicer = contents / 'MacOS' / 'Slicer'
        slicer.parent.mkdir(parents=True)
        slicer.touch()
        lib = contents / 'lib' / 'Slicer-5.10'
        lib.mkdir(parents=True)
        modules = contents / 'Extensions-34045' / 'SlicerDMRI' / 'lib' / 'Slicer-5.10' / 'cli-modules'
        modules.mkdir(parents=True)
        obsolete = contents / 'Extensions-99999' / 'SlicerDMRI' / 'lib' / 'Slicer-4.11' / 'cli-modules'
        obsolete.mkdir(parents=True)
        self.assertIn(modules.resolve(), _slicerdmri_module_paths(slicer))
        self.assertNotIn(obsolete.resolve(), _slicerdmri_module_paths(slicer))
        self.assertIn(lib.resolve(), _slicer_runtime_library_paths(slicer))

    def test_wrapper_runtime_comes_from_its_launcher_not_sibling_guess(self):
        from tractography.wma import _resolve_slicer_runtime
        home = self.tmp_path / 'actual-runtime'
        (home / 'bin').mkdir(parents=True)
        (home / 'Slicer').touch()
        (home / 'bin' / 'PythonSlicer').touch()
        completed = mock.Mock(returncode=0, stdout=f'SLICER_HOME={home}\n')
        with mock.patch('tractography.wma.subprocess.run', return_value=completed) as run:
            self.assertEqual(_resolve_slicer_runtime(self.slicer_executable), home / 'Slicer')
            self.assertEqual(run.call_args.args[0][1], '--launcher-dump-environment')
            self.assertNotIn('PYTHONHOME', run.call_args.kwargs['env'])
        with mock.patch('tractography.wma.subprocess.run', return_value=mock.Mock(
            returncode=0, stdout=f'SLICER_HOME={self.tmp_path / "missing"}\n')):
            with self.assertRaisesRegex(RuntimeError, 'incomplete runtime'):
                _resolve_slicer_runtime(self.slicer_executable)

    def test_direct_runtime_does_not_probe_or_start_slicer(self):
        from tractography.wma import _resolve_slicer_runtime
        (self.tmp_path / 'bin').mkdir()
        (self.tmp_path / 'bin' / 'PythonSlicer').touch()
        with mock.patch('tractography.wma.subprocess.run') as run:
            self.assertEqual(_resolve_slicer_runtime(self.slicer_executable), self.slicer_executable)
            run.assert_not_called()

    def test_linux_headless_defaults_to_virtual_display(self):
        from tractography.wma import WMAOrgAtlasConfig, build_wma_org_atlas_command, _use_virtual_x
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root)
        with mock.patch('tractography.wma.sys.platform', 'linux'), mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_use_virtual_x())
            command = build_wma_org_atlas_command(config)
            self.assertEqual(command[command.index('-x') + 1], '1')

    def test_existing_display_and_other_platforms_do_not_auto_enable_xvfb(self):
        from tractography.wma import _use_virtual_x
        for platform, environment in (
            ('linux', {'DISPLAY': ':22'}),
            ('linux', {'SLICERDSPACE_DISPLAY': ':23'}),
            ('darwin', {}), ('win32', {}),
        ):
            with self.subTest(platform=platform, environment=environment), \
                    mock.patch('tractography.wma.sys.platform', platform), \
                    mock.patch.dict(os.environ, environment, clear=True):
                self.assertFalse(_use_virtual_x())
                self.assertTrue(_use_virtual_x(True))

    def test_headless_preflight_requires_the_complete_virtual_display_tools(self):
        from tractography.wma import WMAOrgAtlasConfig, inspect_wma_backend, validate_wma_org_inputs
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root)
        from tractography.wma import _command_path
        def without_xvfb(name, root):
            return None if name == 'Xvfb' else _command_path(name, root)
        with mock.patch('tractography.wma.sys.platform', 'linux'), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch('tractography.wma._command_path', side_effect=without_xvfb):
            report = inspect_wma_backend(self.wma_root)
            self.assertFalse(report['available'])
            self.assertIn('Xvfb', report['missing_required'])
            with self.assertRaisesRegex(RuntimeError, 'Missing required command.*Xvfb'):
                validate_wma_org_inputs(config)

    def test_execution_records_automatically_selected_virtual_display(self):
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        with mock.patch('tractography.wma.sys.platform', 'linux'), \
                mock.patch.dict(os.environ, {'PATH': os.defpath}, clear=True):
            result = run_wma_org_atlas(config)
        record = json.loads(result.manifest_path.read_text())
        self.assertEqual(result.status, 'completed')
        self.assertTrue(record['use_virtual_x'])
        self.assertFalse(record['virtual_x_requested'])
        command = record['executed_command']
        self.assertEqual(command[command.index('-x') + 1], '1')

    def test_environment_doctor_checks_headless_tools_only_for_linux_wma(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs
        for platform, environment, targets, required in (
            ('Linux', {}, ('wma',), True),
            ('Linux', {'DISPLAY': ':10'}, ('wma',), False),
            ('Darwin', {}, ('wma',), False),
            ('Linux', {}, ('connectome',), False),
        ):
            with self.subTest(platform=platform, environment=environment, targets=targets), \
                    mock.patch.dict(os.environ, environment, clear=True):
                config = EnvironmentDoctorConfig(targets=targets, repo_root=REPO_ROOT, platform_name=platform,
                                                  slicer_executable=self.slicer_executable)
                specs = {spec.key: spec for spec in build_dependency_specs(config)}
                for command in ('xvfb-run', 'Xvfb', 'xauth'):
                    self.assertEqual('command:' + command in specs, required)

    def test_coverage_alone_cannot_hide_sparse_major_tracts(self):
        from tractography.wma import _wma_output_contract_error, REQUIRED_MAJOR_TRACTS
        summary = dict(anatomical_tract_count=74, anatomical_tract_readable_count=74,
                       anatomical_tract_nonempty_count=74,
                       anatomical_tract_streamline_counts={name: 100 for name in REQUIRED_MAJOR_TRACTS})
        self.assertIsNone(_wma_output_contract_error(summary))
        summary['anatomical_tract_streamline_counts']['T_AF_left.vtp'] = 1
        self.assertIn('T_AF_left.vtp=1', _wma_output_contract_error(summary))

    def test_windows_rejects_raw_shell_backend_before_launch(self):
        from tractography.wma import WMAOrgAtlasConfig, validate_wma_org_inputs
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        with mock.patch('tractography.wma.sys.platform', 'win32'), \
                mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'legacy'}):
            with self.assertRaisesRegex(RuntimeError, 'managed WSL command bridge'):
                validate_wma_org_inputs(config)

    def test_org_route_respects_platform_bridge_and_measurements(self):
        from tractography.wma import resolve_wma_execution_backend
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_wma_execution_backend(self.wma_root, platform='win32'), 'native')
            self.assertEqual(resolve_wma_execution_backend(self.wma_root, platform='linux'), 'legacy')
            self.assertEqual(resolve_wma_execution_backend(
                self.wma_root, platform='win32', atlas_family='naba'), 'legacy')
            self.assertEqual(resolve_wma_execution_backend(
                self.wma_root, platform='win32', export_diffusion_measures=True), 'legacy')
            self.assertEqual(resolve_wma_execution_backend(
                self.wma_root, platform='win32', export_diffusion_measures=True,
                measurement_backend='native_vtk_point_scalar_summary'), 'native')
            bridge = self.wma_script.with_name(self.wma_script.name + '.exe')
            bridge.touch()
            Path(str(bridge) + '.slicerdsurfer-owner').touch()
            self.assertEqual(resolve_wma_execution_backend(self.wma_root, platform='win32'), 'legacy')
        with mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'native'}, clear=True):
            self.assertEqual(resolve_wma_execution_backend(self.wma_root, platform='darwin'), 'native')
            with self.assertRaisesRegex(ValueError, 'FiberTractMeasurements'):
                resolve_wma_execution_backend(self.wma_root, export_diffusion_measures=True)

    def test_native_source_does_not_need_executable_author_shell(self):
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        self.wma_script.unlink()
        package = self.wma_root / 'whitematteranalysis'
        package.mkdir()
        (package / '__init__.py').touch()
        config = WMAOrgAtlasConfig(subject_id='subject', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root)
        with mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'native', 'DISPLAY': ':1'}):
            result = run_wma_org_atlas(config)
        record = json.loads(result.manifest_path.read_text())
        self.assertEqual(result.status, 'planned')
        self.assertEqual(result.command, ())
        self.assertEqual(record['execution_backend'], 'native')
        self.assertEqual(record['stage_commands'], [])

    def test_native_source_receipt_is_verified_even_with_an_explicit_root(self):
        from tractography.wma import inspect_wma_backend
        (self.wma_root / '.slicerdspace-source.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unrecognized algorithm source receipt'):
            inspect_wma_backend(self.wma_root, execution_backend='native')

    def test_native_subprocess_failure_retains_stage_log_and_products(self):
        from shared.external_process import find_pythonslicer
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        package = self.wma_root / 'whitematteranalysis'
        package.mkdir()
        (package / '__init__.py').touch()
        (self.wma_bin / 'wm_register_to_atlas_new.py').write_text(
            'import sys\nprint("native registration failed", file=sys.stderr)\nsys.exit(7)\n',
            encoding='utf-8',
        )
        config = WMAOrgAtlasConfig(subject_id='subject', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        python = find_pythonslicer(Path(sys.executable).parent) or Path(sys.executable)
        with (
            mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'native', 'DISPLAY': ':1'}),
            mock.patch('tractography.wma._resolve_slicer_runtime', return_value=self.slicer_executable),
            mock.patch('tractography.wma.find_pythonslicer', return_value=python),
        ):
            result = run_wma_org_atlas(config)
        record = json.loads(result.manifest_path.read_text())
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.return_code, 7)
        self.assertEqual(len(record['stage_commands']), 1)
        stage = record['stage_commands'][0]
        self.assertEqual(stage['stage'], 'registration-rigid_affine_fast')
        self.assertEqual(stage['return_code'], 7)
        self.assertIn('native registration failed', Path(record['outputs']['stderr_log']).read_text())
        self.assertFalse((self.output_dir / 'subject').exists())
        self.assertTrue(list(self.output_dir.glob('.failed_products/*/subject')))

    @unittest.skipIf(sys.platform == 'win32', 'Uses the POSIX author shell fixture')
    def test_resume_does_not_reuse_legacy_output_for_native_org(self):
        from shared.orchestrator import _wma_complete
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        config = WMAOrgAtlasConfig(subject_id='subject', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        resume_args = dict(input_tractography=self.tractography, atlas_dir=self.atlas_dir, wma_root=self.wma_root)
        with mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'legacy', 'DISPLAY': ':1'}):
            self.assertEqual(run_wma_org_atlas(config).status, 'completed')
            self.assertTrue(_wma_complete(self.output_dir, **resume_args))
        with mock.patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'native'}):
            self.assertFalse(_wma_complete(self.output_dir, **resume_args))

    def test_windows_prefers_explicit_executable_bridge(self):
        from tractography.wma import _wma_script_path, _command_path, resolve_wma_root
        alias = self.wma_script.with_name(self.wma_script.name + '.exe')
        alias.touch()
        tool = self.wma_bin / 'wm_cluster_from_atlas.py.exe'
        tool.touch()
        with mock.patch('tractography.wma.sys.platform', 'win32'):
            self.assertEqual(_wma_script_path(self.wma_root), alias)
            self.assertEqual(_command_path('wm_cluster_from_atlas.py', self.wma_root), tool.resolve())
            self.assertEqual(resolve_wma_root(self.wma_root), self.wma_root.resolve())

    def test_rerun_zero_exit_cannot_reuse_old_anatomical_products(self):
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, use_virtual_x=False, execute=True)
        first = run_wma_org_atlas(config)
        self.assertEqual(first.status, 'completed')
        self.wma_script.write_text('#!/bin/sh\nexit 0\n')
        second = run_wma_org_atlas(config)
        self.assertNotEqual(second.status, 'completed')
        qc = json.loads(second.qc_path.read_text())
        self.assertNotEqual(qc['status'], 'completed')
        self.assertTrue((self.output_dir / 'subject' / 'AnatomicalTracts').is_dir())
        self.assertTrue(list((self.output_dir / '.previous_records').rglob('wma_org_atlas_qc.json')))

    def test_builds_org_atlas_command_without_running_wma(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                registration_mode="rig",
                num_threads=2,
                use_virtual_x=True,
                clean_files=1,
                execute=False,
            )
        )

        self.assertFalse(result.executed)
        self.assertEqual(result.status, "planned")
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.qc_path.is_file())
        self.assertTrue(result.backend_status["available"])
        self.assertTrue(result.atlas_status["available"])
        command = list(result.command)
        self.assertEqual(command[0], self.wma_script.resolve().as_posix())
        self.assertIn("-i", command)
        self.assertIn(self.tractography.resolve().as_posix(), command)
        self.assertIn("-a", command)
        self.assertIn(self.atlas_dir.resolve().as_posix(), command)
        self.assertIn("-s", command)
        self.assertIn(self.slicer_executable.resolve().as_posix(), command)
        self.assertIn("-x", command)
        self.assertIn("1", command)

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["subject_id"], "100307")
        self.assertFalse(manifest["executed"])
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["registration_mode"], "rig")
        self.assertEqual(manifest["num_threads"], 2)
        self.assertEqual(manifest["backend"]["source"]["repository"], "https://github.com/SlicerDMRI/whitematteranalysis.git")
        self.assertEqual(manifest["backend"]["source"]["commit"], "0123456789abcdef")
        self.assertEqual(manifest["backend"]["source"]["date_added"], "2026-07-25")
        self.assertTrue(manifest["atlas"]["available"])
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "planned")
        self.assertFalse(qc["executed"])

    def test_original_measurement_export_keeps_author_cli(self):
        from dataclasses import replace
        from tractography.wma import WMAOrgAtlasConfig, build_wma_org_atlas_command
        cli = self.tmp_path / 'FiberTractMeasurements'
        cli.write_text('fixture')
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, export_diffusion_measures=True, fiber_tract_measurements_cli=cli)
        command = build_wma_org_atlas_command(config)
        self.assertEqual(command[command.index('-m') + 1], str(cli))
        self.assertEqual(command[command.index('-d') + 1], '1')
        native = replace(config, measurement_backend='native_vtk_point_scalar_summary',
                         fiber_tract_measurements_cli=None)
        self.assertNotIn('-m', build_wma_org_atlas_command(native))
        self.assertNotIn('-d', build_wma_org_atlas_command(native))

    def test_native_measurement_provenance_and_failure_prevent_publish(self):
        from dataclasses import replace
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True, export_diffusion_measures=True,
            measurement_backend='native_vtk_point_scalar_summary')
        provenance = {'backend': 'native_vtk_point_scalar_summary',
                      'upstream_fibertractmeasurements_parity_certified': False}
        with mock.patch('tractography.wma_native_measures.export_wma_case_measurements',
                        return_value=provenance) as exporter:
            result = run_wma_org_atlas(config)
        self.assertEqual(result.status, 'completed')
        exporter.assert_called_once()
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest['measurement_provenance'], provenance)
        self.assertEqual(manifest['requested_measurement_backend'], config.measurement_backend)
        failed_config = replace(config, output_dir=self.tmp_path / 'failed-native')
        with mock.patch('tractography.wma_native_measures.export_wma_case_measurements',
                        side_effect=RuntimeError('tensor not supported')):
            failed = run_wma_org_atlas(failed_config)
        self.assertEqual(failed.status, 'partial_failed')
        self.assertFalse((failed_config.output_dir / 'subject').exists())

    def test_unknown_measurement_backend_is_rejected(self):
        from tractography.wma import WMAOrgAtlasConfig, validate_wma_org_inputs
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            measurement_backend='typo')
        with self.assertRaisesRegex(ValueError, 'measurement backend'):
            validate_wma_org_inputs(config)

    @unittest.skipIf(sys.platform == 'win32', 'POSIX shim; Windows uses the WSL bridge')
    def test_registration_dispatch_and_reports_are_fresh_per_run(self):
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        python = self.tmp_path / 'bin' / 'PythonSlicer'
        python.parent.mkdir()
        # PythonSlicer launches python-real with Slicer's shared-library paths.
        # A symlink to python-real cannot run in the deliberately cleaned WMA env.
        import shlex
        launcher = Path(sys.executable).parent / 'PythonSlicer'
        interpreter = launcher if launcher.is_file() else Path(sys.executable)
        python.write_text('#!/bin/sh\nexec ' + shlex.quote(str(interpreter)) + ' "$@"\n')
        python.chmod(0o755)
        package = self.wma_root / 'whitematteranalysis'
        package.mkdir()
        (package / '__init__.py').write_text('')
        self.wma_script.write_text(self.wma_script.read_text().replace(
            'while [ "$#" -gt 0 ]; do',
            'wm_register_to_atlas_new.py -mode affine || exit $?\nwhile [ "$#" -gt 0 ]; do'))
        (self.wma_bin / 'wm_register_to_atlas_new.py').write_text(
            '#!/usr/bin/env python\nimport sys\nassert sys.argv[1:] == ["-mode", "affine"]\n')
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        first = run_wma_org_atlas(config)
        self.assertEqual(first.status, 'completed')
        manifest = json.loads(first.manifest_path.read_text())
        first_reports = manifest['registration_preprocess_cache']['reports']
        self.assertTrue(manifest['registration_preprocess_cache']['installed'])
        self.assertEqual(len(first_reports), 1)
        report = json.loads(Path(first_reports[0]).read_text())
        self.assertFalse(report['enabled'])  # Unverified fixture source is never optimized.
        self.assertEqual(report['status'], 'completed')
        second = run_wma_org_atlas(config)
        second_reports = json.loads(second.manifest_path.read_text())['registration_preprocess_cache']['reports']
        self.assertEqual(len(second_reports), 1)
        self.assertNotEqual(first_reports, second_reports)

    @unittest.skipIf(sys.platform == 'win32', 'POSIX shim; Windows uses the WSL bridge')
    def test_harden_step_finds_the_fiber_bundle_save_shim_first_on_path(self):
        # macOS, Slicer 5.12.4 + SlicerDMRI: saveNode() chose the unregistered "TRXFile" writer, so
        # wm_harden_transform.py wrote no cluster ("800 resulting fiber clusters, but only 23 generated").
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        python = self.tmp_path / 'bin' / 'PythonSlicer'
        python.parent.mkdir()
        python.symlink_to(sys.executable)
        author = self.wma_bin / 'harden_transform_with_slicer.py'
        author.write_text('import slicer\n')
        author.chmod(0o755)
        self.wma_script.write_text(self.wma_script.read_text().replace(
            'while [ "$#" -gt 0 ]; do',
            'h=$(command -v harden_transform_with_slicer.py) || exit 7\n'
            '[ "$h" != "%s" ] || exit 8\n'
            'grep -q "run_wma_harden_transform.run(.%s.)" "$h" || exit 9\n'
            'while [ "$#" -gt 0 ]; do' % (author, author)))
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, execute=True)
        result = run_wma_org_atlas(config)
        self.assertEqual(result.status, 'completed', result.manifest_path.read_text())
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest['harden_transform_shim'], 'harden_transform_with_slicer.py')
        runner = Path(__file__).resolve().parents[2] / 'scripts' / 'run_wma_harden_transform.py'
        self.assertIn('def run(author=None)', runner.read_text())

    def test_cache_can_be_disabled_without_changing_author_command(self):
        from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
        config = WMAOrgAtlasConfig(subject_id='100307', input_tractography=self.tractography,
            atlas_dir=self.atlas_dir, output_dir=self.output_dir, slicer_executable=self.slicer_executable,
            wma_root=self.wma_root, registration_preprocess_cache=False)
        result = run_wma_org_atlas(config)
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest['registration_preprocess_cache'],
                         {'requested': False, 'installed': False, 'reports': []})

    def test_rejects_non_vtk_inputs(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        bad_input = self.tmp_path / "subject.trk"
        bad_input.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            run_wma_org_atlas(
                WMAOrgAtlasConfig(
                    subject_id="100307",
                    input_tractography=bad_input,
                    atlas_dir=self.atlas_dir,
                    output_dir=self.output_dir,
                    slicer_executable=self.slicer_executable,
                    wma_root=self.wma_root,
                )
            )

    def test_wma_auto_discovery_excludes_saf_and_bundle_outputs(self) -> None:
        from shared.orchestrator import _resolve_wma_input_tractography

        subject_root = self.tmp_path / "subject"
        tractography_root = subject_root / "outputs" / "04_tractography"
        saf = tractography_root / "saf" / "SAF.vtk"
        bundle = tractography_root / "bundles" / "tractcloud" / "bundle.vtp"
        saf.parent.mkdir(parents=True)
        bundle.parent.mkdir(parents=True)
        saf.write_text("", encoding="utf-8")
        bundle.write_text("", encoding="utf-8")

        self.assertIsNone(_resolve_wma_input_tractography(subject_root, None))

        whole_brain = tractography_root / "cache" / "exports" / "whole_brain.vtk"
        whole_brain.parent.mkdir(parents=True)
        whole_brain.write_text("", encoding="utf-8")
        self.assertIsNone(_resolve_wma_input_tractography(subject_root, None))

        ukf_whole_brain = tractography_root / "cache" / "exports" / "subject_ukf_whole_brain.vtk"
        ukf_whole_brain.write_text("", encoding="utf-8")
        self.assertEqual(
            _resolve_wma_input_tractography(subject_root, None),
            ukf_whole_brain.resolve(),
        )

    def test_tck_input_is_converted_before_wma_execution(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        tck_input = self.tmp_path / "whole_brain.tck"
        tck_input.write_text("tck", encoding="utf-8")
        mrtrix_bin = self.tmp_path / "mrtrix" / "bin"
        mrtrix_bin.mkdir(parents=True)
        tckconvert = mrtrix_bin / "tckconvert"
        tckconvert.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)\n"
            "pathlib.Path(sys.argv[2]).write_text('# vtk DataFile Version 3.0\\n')\n",
            encoding="utf-8",
        )
        tckconvert.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=tck_input,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                mrtrix_bin_dir=mrtrix_bin,
                execute=True,
            )
        )

        self.assertEqual(result.status, "completed")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["input_conversion"]["required"])
        prepared_input = Path(manifest["prepared_input_tractography"])
        self.assertEqual(prepared_input.suffix, ".vtk")
        self.assertTrue(prepared_input.is_file())
        self.assertIn(prepared_input.as_posix(), result.command)

    def test_wma_root_is_external_and_explicit_by_default(self) -> None:
        from tractography.wma import resolve_wma_root

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "optional external backend"):
                resolve_wma_root()

    def test_rejects_incomplete_org_atlas_layout(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        bad_atlas = self.tmp_path / "bad-atlas"
        bad_atlas.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "ORG atlas directory is incomplete"):
            run_wma_org_atlas(
                WMAOrgAtlasConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_dir=bad_atlas,
                    output_dir=self.output_dir,
                    slicer_executable=self.slicer_executable,
                    wma_root=self.wma_root,
                )
            )

    def test_execute_true_records_completed_qc_with_fake_backend(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.return_code, 0)
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "completed")
        self.assertEqual(qc["return_code"], 0)
        self.assertTrue(Path(qc["logs"]["stdout"]).is_file())
        self.assertTrue(Path(qc["logs"]["stderr"]).is_file())

    def test_execute_true_marks_failed_when_script_returns_zero_without_outputs(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        self.wma_script.write_text(
            """#!/bin/sh
exit 0
""",
            encoding="utf-8",
        )
        self.wma_script.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.return_code, 0)
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "failed")
        self.assertIn("complete AnatomicalTracts output", qc["error"])
        self.assertTrue(Path(qc["logs"]["stdout"]).is_file())
        self.assertTrue(Path(qc["logs"]["stderr"]).is_file())

    def test_execute_true_records_nonzero_backend_failure_before_returning(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        self.wma_script.write_text(
            "#!/bin/sh\n"
            "printf 'backend failed\\n' >&2\n"
            "exit 7\n",
            encoding="utf-8",
        )
        self.wma_script.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.return_code, 7)
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "failed")
        self.assertIn("returned non-zero exit status 7", qc["error"])
        self.assertIn("backend failed", Path(qc["logs"]["stderr"]).read_text(encoding="utf-8"))

    def test_keyboard_interrupt_is_persisted_as_failed_qc(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        with mock.patch("tractography.wma.subprocess.run", side_effect=KeyboardInterrupt):
            result = run_wma_org_atlas(
                WMAOrgAtlasConfig(
                    subject_id="100307",
                    input_tractography=self.tractography,
                    atlas_dir=self.atlas_dir,
                    output_dir=self.output_dir,
                    slicer_executable=self.slicer_executable,
                    wma_root=self.wma_root,
                    execute=True,
                )
            )

        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.return_code, 130)
        self.assertEqual(qc["error"], "Interrupted by user.")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["return_code"], 130)

    def test_execute_true_marks_partial_failed_when_clusters_exist_but_anatomical_tracts_do_not(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        self.wma_script.write_text(
            """#!/bin/sh
while [ "$#" -gt 0 ]; do
  case "$1" in
    -i) input="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case_id=$(basename "$input")
case_id=${case_id%.*}
mkdir -p "$out/$case_id/FiberClustering/InitialClusters/${case_id}_reg"
printf '' > "$out/$case_id/FiberClustering/InitialClusters/${case_id}_reg/cluster_00001.vtp"
exit 0
""",
            encoding="utf-8",
        )
        self.wma_script.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual(qc["status"], "partial_failed")
        self.assertEqual(qc["output_summary"]["initial_cluster_count"], 1)
        self.assertEqual(qc["output_summary"]["anatomical_tract_count"], 0)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "partial_failed")
        self.assertIn("runtime_seconds", manifest)

    def test_execute_true_marks_partial_failed_below_required_tract_count(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        script = self.wma_script.read_text(encoding="utf-8").replace(
            'while [ "$index" -le 73 ]; do',
            'while [ "$index" -le 72 ]; do',
        )
        self.wma_script.write_text(script, encoding="utf-8")
        self.wma_script.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual(qc["output_summary"]["anatomical_tract_count"], 72)
        self.assertEqual(qc["acceptance_criteria"]["minimum_anatomical_tract_count"], 73)
        self.assertIn("expected at least 73 files", qc["error"])
        self.assertEqual(manifest["error"], qc["error"])

    def test_execute_true_rejects_atlas_filenames_when_most_tracts_are_empty(self) -> None:
        from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

        script = self.wma_script.read_text(encoding="utf-8").replace(
            vtp_lines(20),
            vtp_lines(0),
        )
        self.wma_script.write_text(script, encoding="utf-8")
        self.wma_script.chmod(0o755)

        result = run_wma_org_atlas(
            WMAOrgAtlasConfig(
                subject_id="100307",
                input_tractography=self.tractography,
                atlas_dir=self.atlas_dir,
                output_dir=self.output_dir,
                slicer_executable=self.slicer_executable,
                wma_root=self.wma_root,
                execute=True,
            )
        )

        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual(qc["output_summary"]["anatomical_tract_nonempty_count"], 0)
        self.assertEqual(qc["acceptance_criteria"]["minimum_nonempty_anatomical_tract_count"], 55)
        self.assertIn("under-populated", qc["error"])

    def test_runtime_library_paths_exclude_slicerdmri_extension_dirs(self) -> None:
        from tractography.wma import _slicer_runtime_library_paths

        slicer_root = self.tmp_path / "SlicerRoot"
        slicer_exec = slicer_root / "Slicer"
        qt_loadable = (
            slicer_root
            / "slicer.org"
            / "Extensions-99999"
            / "SlicerDMRI"
            / "lib"
            / "Slicer-5.10"
            / "qt-loadable-modules"
        )
        qt_loadable.mkdir(parents=True)
        (slicer_root / "lib" / "Slicer-5.10").mkdir(parents=True)
        (slicer_root / "lib").mkdir(exist_ok=True)
        slicer_exec.write_text("#!/bin/sh\n", encoding="utf-8")
        slicer_exec.chmod(0o755)

        paths = _slicer_runtime_library_paths(slicer_exec)
        self.assertIn((slicer_root / "lib" / "Slicer-5.10").resolve(), paths)
        self.assertNotIn(qt_loadable.resolve(), paths)
        self.assertNotIn(qt_loadable.parent.resolve(), paths)

    def test_runtime_library_paths_discover_installed_slicer_version(self) -> None:
        from tractography.wma import _slicer_runtime_library_paths

        slicer_root = self.tmp_path / "SlicerRoot56"
        slicer_exec = slicer_root / "Slicer.exe"
        slicer_library = slicer_root / "lib" / "Slicer-5.6"
        qt_scripted = (
            slicer_root
            / "slicer.org"
            / "Extensions-32448"
            / "SlicerDMRI"
            / "lib"
            / "Slicer-5.6"
            / "qt-scripted-modules"
        )
        slicer_library.mkdir(parents=True)
        qt_scripted.mkdir(parents=True)
        slicer_exec.write_text("", encoding="utf-8")

        paths = _slicer_runtime_library_paths(slicer_exec)

        self.assertIn(slicer_library.resolve(), paths)
        self.assertNotIn(qt_scripted.resolve(), paths)
        self.assertNotIn(qt_scripted.parent.resolve(), paths)

    def test_resolve_slicer_launcher_prefers_run_slicer_sh_when_present(self) -> None:
        from tractography.wma import _resolve_slicer_launcher

        software_root = self.tmp_path / "software"
        slicer_exec = software_root / "Slicer-5.10.0-linux-amd64" / "Slicer"
        slicer_exec.parent.mkdir(parents=True)
        slicer_exec.write_text("#!/bin/sh\n", encoding="utf-8")
        launcher = software_root / "run-slicer.sh"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")

        self.assertEqual(_resolve_slicer_launcher(slicer_exec), launcher.resolve())

    def test_runtime_display_prefers_explicit_env_then_display(self) -> None:
        from tractography.wma import _resolve_runtime_display

        with mock.patch.dict(os.environ, {"DISPLAY": ":99"}, clear=True):
            self.assertEqual(_resolve_runtime_display(), ":99")
        with mock.patch.dict(os.environ, {"DISPLAY": ":99", "SLICERDSPACE_DISPLAY": ":88"}, clear=True):
            self.assertEqual(_resolve_runtime_display(), ":88")

    def test_summarize_wma_outputs_includes_registration_progress(self) -> None:
        from tractography.wma import summarize_wma_outputs

        case_id = self.tractography.stem
        registration_dir = self.output_dir / case_id / "TractRegistration" / case_id / "iteration_00001_sigma_00030"
        registration_dir.mkdir(parents=True)
        progress = registration_dir.parent / "progress.txt"
        progress.write_text(
            (
                "Beginning registration. Total iterations will be: 10\n"
                "Done iteration 1 / 10. Percent finished approx: 0.57. Time: 11:06:08. Minutes Elapsed: 2.37\n"
                "Done iteration 2 / 10. Percent finished approx: 1.14. Time: 11:08:31. Minutes Elapsed: 2.38\n"
            ),
            encoding="utf-8",
        )

        summary = summarize_wma_outputs(self.output_dir, self.tractography)

        self.assertTrue(summary["registration_dir_exists"])
        self.assertTrue(summary["progress_file_exists"])
        self.assertEqual(summary["registration_iteration_dir_count"], 1)
        self.assertEqual(summary["progress"]["completed_iterations"], 2)
        self.assertEqual(summary["progress"]["total_iterations"], 10)
        self.assertAlmostEqual(summary["progress"]["latest_percent_finished"], 1.14)

    def test_summarize_wma_outputs_distinguishes_empty_anatomical_tracts(self) -> None:
        from tractography.wma import summarize_wma_outputs

        anatomical_dir = self.output_dir / self.tractography.stem / "AnatomicalTracts"
        anatomical_dir.mkdir(parents=True)
        (anatomical_dir / "T_AF_left.vtp").write_text(
            vtp_lines(3),
            encoding="utf-8",
        )
        (anatomical_dir / "T_AF_right.vtp").write_text(
            vtp_lines(0),
            encoding="utf-8",
        )

        summary = summarize_wma_outputs(self.output_dir, self.tractography)

        self.assertEqual(summary["anatomical_tract_count"], 2)
        self.assertEqual(summary["anatomical_tract_readable_count"], 2)
        self.assertEqual(summary["anatomical_tract_nonempty_count"], 1)
        self.assertEqual(summary["anatomical_streamline_count"], 3)
        self.assertEqual(summary["anatomical_empty_tracts"], ["T_AF_right.vtp"])


if __name__ == "__main__":
    unittest.main()
