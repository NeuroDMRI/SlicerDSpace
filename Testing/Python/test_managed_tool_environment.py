"""B3: the managed python shim and the FSL environment for MRtrix scripts, eddy, topup and dwifslpreproc."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared import external_process as ep
from preprocessing import pipeline


REAL_DISCOVER = ep._discover_fsl_bin_dir


class ToolEnvironmentBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.shims = self.root / 'shims'
        self.mrtrix = self.make_prefix('mrtrix', with_python=False)
        self.fsl = self.make_prefix('fsl', with_python=False)
        # A5 N9: keep these tests independent of an FSL on the developer machine.
        patcher = patch.object(ep, '_discover_fsl_bin_dir', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_prefix(self, name, with_python):
        bin_dir = self.root / name / 'bin'
        bin_dir.mkdir(parents=True)
        if with_python:
            (bin_dir / 'python').write_text('#!/bin/sh\n')
            (bin_dir / 'python').chmod(0o755)
        return bin_dir

    def environment(self, system, **kwargs):
        return ep.managed_tool_environment(system=system, shim_root=self.shims, **kwargs)

    def path_entries(self, env):
        return env['PATH'].split(os.pathsep)


class ShimTests(ToolEnvironmentBase):
    def test_linux_shim_points_to_the_runtime_conda_python(self):
        python = self.make_prefix('conda', with_python=True) / 'python'
        env = self.environment('linux', mrtrix_bin_dir=python.parent)
        folder = Path(self.path_entries(env)[0])
        self.assertEqual(folder.parent, self.shims)
        for name in ('python', 'python3'):
            script = folder / name
            self.assertTrue(os.access(script, os.X_OK))
            self.assertEqual(script.read_text(), f'#!/bin/sh\n# SlicerDSpace managed shim\nexec {python} "$@"\n')

    def test_linux_without_a_runtime_python_adds_no_shim(self):
        env = self.environment('linux', mrtrix_bin_dir=self.mrtrix)
        self.assertEqual(self.path_entries(env)[0], str(self.mrtrix))
        self.assertFalse(self.shims.exists())

    def test_macos_shim_uses_the_fsl_prefix_python_when_fsl_is_installed(self):
        fsl_python = self.make_prefix('fslpy', with_python=True) / 'python'
        slicer = self.root / 'PythonSlicer'
        slicer.write_text('#!/bin/sh\n')
        env = self.environment('darwin', mrtrix_bin_dir=self.mrtrix, fsl_bin_dir=fsl_python.parent, slicer_python=slicer)
        script = Path(self.path_entries(env)[0]) / 'python'
        self.assertIn(f'exec {fsl_python} "$@"', script.read_text())

    def test_macos_shim_falls_back_to_pythonslicer_without_fsl(self):
        slicer = self.root / 'PythonSlicer'
        slicer.write_text('#!/bin/sh\n')
        env = self.environment('darwin', mrtrix_bin_dir=self.mrtrix, slicer_python=slicer)
        script = Path(self.path_entries(env)[0]) / 'python3'
        self.assertIn(f'exec {slicer} "$@"', script.read_text())
        self.assertNotIn('FSLDIR', env)

    def test_macos_fsl_prefix_without_python_also_falls_back_to_pythonslicer(self):
        slicer = self.root / 'PythonSlicer'
        slicer.write_text('#!/bin/sh\n')
        env = self.environment('darwin', mrtrix_bin_dir=self.mrtrix, fsl_bin_dir=self.fsl, slicer_python=slicer)
        self.assertIn(f'exec {slicer} "$@"', (Path(self.path_entries(env)[0]) / 'python').read_text())

    def test_a5_n9_macos_shim_uses_the_fsl_python_found_through_fsldir_or_path(self):
        """A5 N9: no fsl_bin_dir passed; FSL found the usual way. Fails before: the shim pointed to PythonSlicer."""
        fsl_python = self.make_prefix('fslhome', with_python=True) / 'python'
        slicer = self.root / 'PythonSlicer'
        slicer.write_text('#!/bin/sh\n')
        with patch.object(ep, '_discover_fsl_bin_dir', REAL_DISCOVER):
            with patch.dict(os.environ, {'FSLDIR': str(fsl_python.parent.parent)}):
                env = self.environment('darwin', mrtrix_bin_dir=self.mrtrix, slicer_python=slicer)
            self.assertIn(f'exec {fsl_python} "$@"', (Path(self.path_entries(env)[0]) / 'python').read_text())
            (fsl_python.parent / 'eddy').write_text('#!/bin/sh\n')
            (fsl_python.parent / 'eddy').chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(fsl_python.parent)}):
                os.environ.pop('FSLDIR', None)
                self.assertEqual(ep.shim_target_python(system='darwin', slicer_python=slicer), fsl_python)
            with patch.dict(os.environ, {'PATH': str(self.mrtrix)}):
                os.environ.pop('FSLDIR', None)
                self.assertEqual(ep.shim_target_python(system='darwin', slicer_python=slicer), slicer)

    def test_macos_without_any_python_adds_no_shim(self):
        with patch.object(ep, '_default_slicer_python', return_value=None):
            env = self.environment('darwin', mrtrix_bin_dir=self.mrtrix)
        self.assertEqual(self.path_entries(env)[0], str(self.mrtrix))

    def test_windows_gets_no_shim(self):
        python = self.make_prefix('conda', with_python=True) / 'python'
        env = self.environment('win32', mrtrix_bin_dir=python.parent)
        self.assertEqual(self.path_entries(env)[0], str(python.parent))
        self.assertFalse(self.shims.exists())

    def test_shim_is_refreshed_atomically_and_reused(self):
        python = self.make_prefix('conda', with_python=True) / 'python'
        first = ep.python_shim_folder(python, self.shims)
        (first / 'python').write_text('tampered')
        again = ep.python_shim_folder(python, self.shims)
        self.assertEqual(first, again)
        self.assertIn('exec', (again / 'python').read_text())
        self.assertEqual(sorted(path.name for path in again.iterdir()), ['python', 'python3'])

    def test_shim_folder_is_first_then_mrtrix_then_fsl(self):
        python = self.make_prefix('conda', with_python=True) / 'python'
        env = self.environment('linux', mrtrix_bin_dir=python.parent, fsl_bin_dir=self.fsl)
        entries = self.path_entries(env)
        self.assertEqual(entries[1:3], [str(python.parent), str(self.fsl)])
        self.assertEqual(entries[0], str(ep.python_shim_folder(python, self.shims)))


class ShimWriterTests(ToolEnvironmentBase):
    def target(self):
        return self.make_prefix('conda', with_python=True) / 'python'

    def test_a_second_writer_between_write_and_chmod_cannot_break_the_first(self):
        """R-B6: with one shared python.tmp the second writer's replace made the first writer's chmod raise
        FileNotFoundError (the first parallel `run_batch_pipeline.py --jobs` run)."""
        target, state, real_chmod = self.target(), {'racing': False}, os.chmod

        def racing_chmod(path, mode, *args, **kwargs):
            if not state['racing']:
                state['racing'] = True
                ep.python_shim_folder(target, self.shims)  # another job finishes its whole write here
            return real_chmod(path, mode, *args, **kwargs)

        with patch('os.chmod', racing_chmod):
            folder = ep.python_shim_folder(target, self.shims)
        for name in ('python', 'python3'):
            self.assertTrue(os.access(folder / name, os.X_OK), name)
            self.assertIn(f'exec {target} "$@"', (folder / name).read_text())
        self.assertEqual(sorted(path.name for path in folder.iterdir()), ['python', 'python3'])

    def test_a_failed_write_leaves_no_temporary_file(self):
        target = self.target()
        with patch('os.replace', side_effect=OSError('disk full')), self.assertRaises(OSError):
            ep.python_shim_folder(target, self.shims)
        folder = next(self.shims.iterdir())
        self.assertEqual(list(folder.iterdir()), [])

    def test_an_identical_shim_is_not_rewritten(self):
        target = self.target()
        folder = ep.python_shim_folder(target, self.shims)
        before = {name: (folder / name).stat().st_ino for name in ('python', 'python3')}
        with patch('os.replace', side_effect=AssertionError('rewrote an identical shim')):
            self.assertEqual(ep.python_shim_folder(target, self.shims), folder)
        self.assertEqual(before, {name: (folder / name).stat().st_ino for name in ('python', 'python3')})

    def test_unwritable_runtime_root_gives_no_shim_and_one_log_line(self):
        """R-N2: the environment is the same as without a shim."""
        python = self.target()
        with patch.object(ep, 'python_shim_folder', side_effect=PermissionError('read-only')), \
                self.assertLogs(ep.__name__, level='WARNING') as logs:
            env = self.environment('linux', mrtrix_bin_dir=python.parent)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('shim skipped', logs.output[0])
        self.assertEqual(self.path_entries(env)[0], str(python.parent))

    def test_real_unwritable_shim_root_is_survived(self):
        python = self.target()
        blocker = self.root / 'blocker'
        blocker.write_text('a file where the shim root should be')
        with self.assertLogs(ep.__name__, level='WARNING'):
            env = ep.managed_tool_environment(system='linux', shim_root=blocker / 'shims', mrtrix_bin_dir=python.parent)
        self.assertEqual(self.path_entries(env)[0], str(python.parent))


class ShimInterpreterTests(ToolEnvironmentBase):
    def test_python3_is_preferred_over_python_in_a_bin_dir(self):
        """R-N5: for the MRtrix bin dir (Linux) and for the FSL bin dir (macOS)."""
        for system, kwargs in (('linux', 'mrtrix_bin_dir'), ('darwin', 'fsl_bin_dir')):
            bin_dir = self.make_prefix(f'both-{system}', with_python=True)
            (bin_dir / 'python3').write_text('#!/bin/sh\n')
            self.assertEqual(ep.shim_target_python(system=system, **{kwargs: bin_dir}), bin_dir / 'python3')

    def test_python_is_used_when_python3_is_absent_and_fsl_is_searched_after_mrtrix(self):
        with_python = self.make_prefix('only-python', with_python=True)
        self.assertEqual(ep.shim_target_python(system='linux', mrtrix_bin_dir=with_python), with_python / 'python')
        none = self.make_prefix('none', with_python=False)
        self.assertEqual(ep.shim_target_python(system='linux', mrtrix_bin_dir=none, fsl_bin_dir=with_python),
                         with_python / 'python')
        self.assertIsNone(ep.shim_target_python(system='linux', mrtrix_bin_dir=none))


class FslEnvironmentTests(ToolEnvironmentBase):
    def test_fsl_keys_are_present_and_output_type_is_fixed(self):
        with patch.dict(os.environ, {'FSLOUTPUTTYPE': 'NIFTI', 'FSLDIR': '/somewhere/else'}):
            env = self.environment('linux', fsl_bin_dir=self.fsl)
        self.assertEqual(env['FSLDIR'], str(self.fsl.parent))
        self.assertEqual(env['FSLOUTPUTTYPE'], 'NIFTI_GZ')
        self.assertIn(str(self.fsl), self.path_entries(env))

    def test_share_fsl_layout_gives_the_conda_prefix_as_fsldir(self):
        bin_dir = self.root / 'prefix/share/fsl/bin'
        bin_dir.mkdir(parents=True)
        self.assertEqual(ep.fsl_root_from_bin_dir(bin_dir), self.root / 'prefix')

    def test_no_fsl_dir_means_no_fsl_keys(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('FSLDIR', None)
            env = self.environment('linux', mrtrix_bin_dir=self.mrtrix)
        self.assertNotIn('FSLDIR', env)

    def test_extra_env_wins_and_none_is_dropped(self):
        env = self.environment('linux', fsl_bin_dir=self.fsl, extra_env={'MRTRIX_RNG_SEED': '7', 'X': None})
        self.assertEqual(env['MRTRIX_RNG_SEED'], '7')
        self.assertNotIn('X', env)

    def test_preprocessing_environment_helper_routes_through_the_shared_builder(self):
        config = SimpleNamespace(mrtrix_bin_dir=self.mrtrix, fsl_bin_dir=self.fsl)
        with patch.object(pipeline, 'managed_tool_environment', return_value={'ROUTED': '1'}) as builder:
            self.assertEqual(pipeline._external_command_environment(config), {'ROUTED': '1'})
        builder.assert_called_once_with(mrtrix_bin_dir=self.mrtrix, fsl_bin_dir=self.fsl)

    def test_eddy_and_dwifslpreproc_calls_both_get_the_fsl_environment(self):
        config = SimpleNamespace(mrtrix_bin_dir=self.mrtrix, fsl_bin_dir=self.fsl)
        seen = []
        commands = [[str(self.mrtrix / 'dwifslpreproc'), 'in.nii.gz', 'out.nii.gz'],
                    [str(self.fsl / 'eddy'), '--imain=in'], [str(self.fsl / 'topup'), '--imain=in']]

        def fake_launcher(command, record, environment):
            return command

        def fake_run(command, env, log):
            seen.append((Path(command[0]).name, dict(env)))

        raw = SimpleNamespace(subject_id='sub')
        with patch.dict(os.environ, {'FSLOUTPUTTYPE': 'ANALYZE'}), \
                patch.object(pipeline, 'preprocessing_launcher', side_effect=fake_launcher), \
                patch.object(pipeline, '_run_command', side_effect=fake_run), \
                patch.object(pipeline, '_mrtrix_backend_details', return_value={}):
            pipeline._execute_preprocessing_backend(raw, config, {'qc_dir': self.root}, {'backend': 'dipy_mppca+gibbs+fsl_eddy'},
                                                    commands, log=None)
        self.assertEqual([name for name, _ in seen], ['dwifslpreproc', 'eddy', 'topup'])
        for name, env in seen:
            self.assertEqual(env['FSLDIR'], str(self.fsl.parent), name)
            self.assertEqual(env['FSLOUTPUTTYPE'], 'NIFTI_GZ', name)
            self.assertIn(str(self.fsl), env['PATH'].split(os.pathsep), name)


if __name__ == '__main__':
    unittest.main()
