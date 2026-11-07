"""Equivalent volume processing, minimal DTI output and selective resumption."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from threading import Lock
import time
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing import dti, volumes
from preprocessing.cache import related_files
from preprocessing.conversion.nhdr_write import write_nhdr
from preprocessing.mask import skull_strip
from preprocessing.normalize import normalize_volume
from preprocessing.resample import resample_image_to_target_space
from test_raw_pipeline import make_bundle


class PreprocessingTests(unittest.TestCase):
    def test_batched_volumes_equal_original_steps_and_do_not_repeat_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'dti'; inputs.mkdir()
            affine = np.eye(4); affine[:3, 3] = [-88.29999542236328, -129., -69.]
            shape = (12, 20, 16)
            rng = np.random.default_rng(15)
            mask = (rng.random(shape) > .2).astype(np.uint8)
            mask_path = inputs / 'x-mask-Reg.nii.gz'
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            for scalar, suffix in volumes.CHANNELS:
                nib.save(nib.Nifti1Image(rng.random(shape).astype(np.float32), affine),
                         inputs / f'x-dti-{scalar}-Reg.nii.gz')
            sitk.WriteTransform(sitk.TranslationTransform(3), str(inputs / 'x-b0ToAtlasT2.tfm'))
            geometry = dict(volumes.GEOMETRY, target_size=shape)
            with patch.dict(volumes.GEOMETRY, geometry):
                volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(root / 'volumes'), '--minimal'])
                output = root / 'volumes'
                original = root / 'original'; original.mkdir()
                reference_mask = original / 'mask.nii.gz'
                resample_image_to_target_space(mask_path, reference_mask, **geometry)
                reference_mask_data = nib.load(reference_mask).get_fdata().astype(bool)
                for scalar, suffix in volumes.CHANNELS:
                    masked = original / f'{suffix}.masked.nii.gz'
                    reference = original / f'{suffix}.nii.gz'
                    skull_strip(inputs / f'x-dti-{scalar}-Reg.nii.gz', mask_path, masked)
                    resample_image_to_target_space(masked, reference, **geometry)
                    normalize_volume(reference, reference, mask=reference_mask_data)
                    left, right = nib.load(reference), nib.load(output / f'x-{suffix}.nii.gz')
                    np.testing.assert_array_equal(left.get_fdata(), right.get_fdata())
                    np.testing.assert_array_equal(left.affine, right.affine)
                    self.assertEqual(left.header.binaryblock, right.header.binaryblock)
                mtimes = {p: p.stat().st_mtime_ns for p in output.rglob('*.nii.gz')}
                volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(output), '--minimal'])
                self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
                missing = output / 'x-MD.nii.gz'; missing.unlink()
                with patch('preprocessing.volumes.normalize_volume', wraps=normalize_volume) as normalize:
                    volumes.main(['--subject', 'x', '--input-dir', str(inputs), '--output-dir', str(output), '--minimal'])
                    self.assertEqual(normalize.call_count, 1)
                np.testing.assert_array_equal(nib.load(missing).get_fdata(), nib.load(original / 'MD.nii.gz').get_fdata())

    def test_minimal_dti_keeps_commands_parallelizes_only_independent_stages_and_repairs_one_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = make_bundle(root / 'inputs')
            slicer = root / 'slicer'; (slicer / 'bin').mkdir(parents=True)
            core = root / 'core'; core.mkdir()
            for path in (slicer / 'Slicer', slicer / 'bin/PythonSlicer',
                         core / 'BRAINSFit', core / 'ResampleScalarVectorDWIVolume'):
                path.write_text('fixture')
            folder = root / 'dti'; folder.mkdir()
            # W4: no dmri_cli; PythonSlicer and the Teem script replace the two SlicerDMRI CLIs.
            args = argparse.Namespace(subject='x', output_dir=folder, slicer_path=slicer,
                                      core_cli=core, reference_image=files['mask'], mask_flip=1, jobs=2,
                                      minimal=True, **files)
            lock = Lock(); activity = [0, 0]; teem_calls = []
            def command(argv, **kwargs):
                if Path(argv[1]).name == 'teem_dti.py':
                    tool = 'teem_dti.py'
                    teem_calls.append(list(argv))
                    self.assertNotIn('PYTHONHOME', kwargs['env'])
                else:
                    tool = Path(argv[2]).name if argv[0] != sys.executable else Path(argv[1]).name
                if tool == 'nhdr_write.py':
                    values = {argv[i]: argv[i+1] for i in range(2, len(argv), 2)}
                    write_nhdr(values['--nifti'], values.get('--bval'), values.get('--bvec'), values['--nhdr'])
                    return
                if tool == 'normalize_dti.py':
                    sitk.WriteImage(sitk.GetImageFromArray(np.ones((5,6,7), np.float32)),
                                    argv[argv.index('--output')+1], True)
                    return
                with lock:
                    activity[0] += 1; activity[1] = max(activity)
                time.sleep(.01)
                if tool == 'BRAINSFit':
                    self.assertEqual(argv[-2:], ['--useRigid', '--useAffine'])
                    sitk.WriteTransform(sitk.TranslationTransform(3), argv[argv.index('--linearTransform')+1])
                else:
                    if tool == 'teem_dti.py':
                        self.assertEqual(argv[0], str(slicer / 'bin/PythonSlicer'))
                        self.assertEqual(argv[2:5], ['estimate', '--enumeration', 'WLS'])
                        outputs = argv[-2:] + [item.partition('=')[2] for i, item in enumerate(argv)
                                               if argv[i-1] == '--scalar']
                        self.assertEqual(len(outputs), 8)
                    else:
                        outputs = [argv[-1]]
                    for output in outputs:
                        sitk.WriteImage(sitk.GetImageFromArray(np.ones((5,6,7), np.float32)), output, True)
                with lock:
                    activity[0] -= 1
            with patch('preprocessing.dti.subprocess.run', side_effect=command) as commands:
                dti.run(args)
                self.assertEqual(len(teem_calls), 1)  # W4: one tensor process, not one per scalar
                self.assertEqual(activity[1], 2)
                self.assertTrue((folder / 'x-dti-Trace.nhdr').is_file())
                self.assertFalse((folder / 'x-dti-Trace-Reg.nii.gz').exists())
                self.assertFalse(list(folder.glob('*NormMasked*')))
                commands.reset_mock()
                dti.run(args)
                commands.assert_not_called()
                missing = folder / 'x-dti-MinEigenvalue-Reg.nii.gz'; missing.unlink()
                dti.run(args)
                self.assertEqual(commands.call_count, 1)
                commands.reset_mock()
                args.minimal = False
                dti.run(args)
                self.assertEqual(commands.call_count, 5)
                self.assertTrue((folder / 'x-dti-Trace-Reg.nii.gz').exists())
                self.assertEqual(len(list(folder.glob('*NormMasked*'))), 4)
                commands.reset_mock()
                related_files(folder / 'x-dti-MidEigenvalue.nhdr')[1].unlink()
                dti.run(args)
                self.assertEqual(commands.call_count, 1)
                # W4: the cache hashes the Teem script, so a changed script refits the tensor.
                custom = root / 'custom/teem_dti.py'
                custom.parent.mkdir(); custom.write_text('# v1')
                with patch.dict(os.environ, {'SLICERDSPACE_TEEM_DTI_SCRIPT': str(custom)}):
                    dti.run(args)
                    fitted = len(teem_calls)
                    dti.run(args)
                    self.assertEqual(len(teem_calls), fitted)
                    custom.write_text('# v2')
                    dti.run(args)
                    self.assertEqual(len(teem_calls), fitted + 1)


ROOT = Path(__file__).resolve().parents[1]


class PreflightToolTests(unittest.TestCase):
    def slicer(self, root, pythonslicer=True):
        home = root / 'slicer'
        core = home / 'lib/Slicer-5.2/cli-modules'
        core.mkdir(parents=True)
        (home / 'bin').mkdir()
        names = [home / 'Slicer', core / 'BRAINSFit', core / 'ResampleScalarVectorDWIVolume']
        names += [home / 'bin/PythonSlicer'] if pythonslicer else []
        for path in names:
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o755)
        return home

    def test_w6_slicer_files_need_no_slicerdmri(self):
        from utils.preflight import slicer_files, tensor_files
        with tempfile.TemporaryDirectory() as temp:
            home = self.slicer(Path(temp))
            with patch.dict(os.environ, {'SLICER_PATH': str(home)}, clear=True):
                files = slicer_files()
                self.assertIn(home / 'bin/PythonSlicer', files)
                self.assertFalse([path for path in files if 'DMRI' in str(path) or 'DWIToDTI' in str(path)])
                self.assertEqual([path.name for path in tensor_files()], ['teem_dti.py'])

    def test_w6_missing_pythonslicer_is_reported(self):
        from utils.preflight import slicer_files
        with tempfile.TemporaryDirectory() as temp:
            home = self.slicer(Path(temp), pythonslicer=False)
            with patch.dict(os.environ, {'SLICER_PATH': str(home)}, clear=True):
                with self.assertRaisesRegex(FileNotFoundError, 'PythonSlicer'):
                    slicer_files()
                self.assertNotIn('PythonSlicer', ' '.join(map(str, slicer_files(required=False))))


class PythonSlicerLayoutTests(unittest.TestCase):
    """F2: the backend finds PythonSlicer in the Linux, macOS .app and Windows layouts."""

    def tree(self, root, relative):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text('#!/bin/sh\n')
        return path

    def test_f2_resolver_handles_linux_macos_app_and_windows_trees(self):
        from preprocessing.dti import pythonslicer
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linux = self.tree(root / 'linux', 'bin/PythonSlicer')
            mac = self.tree(root / 'Slicer.app', 'Contents/bin/PythonSlicer')
            windows = self.tree(root / 'windows', 'bin/PythonSlicer.exe')
            self.assertEqual(pythonslicer(root / 'linux'), linux)
            self.assertEqual(pythonslicer(root / 'Slicer.app'), mac)  # the .app itself
            self.assertEqual(pythonslicer(root / 'Slicer.app/Contents'), mac)
            (root / 'Slicer.app/Contents/MacOS').mkdir()
            self.assertEqual(pythonslicer(root / 'Slicer.app/Contents/MacOS'), mac)
            self.assertEqual(pythonslicer(root / 'windows'), windows)
            with self.assertRaisesRegex(FileNotFoundError, 'PythonSlicer'):
                pythonslicer(root / 'empty')

    def test_b9_launcher_resolver_handles_linux_macos_app_and_windows_trees(self):
        from preprocessing.dti import slicer_launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            linux = self.tree(root / 'linux', 'Slicer')
            mac = self.tree(root / 'Slicer.app', 'Contents/MacOS/Slicer')
            windows = self.tree(root / 'windows', 'Slicer.exe')
            for path in (linux, mac, windows):
                path.chmod(0o755)
            self.assertEqual(slicer_launcher(root / 'linux'), linux)
            self.assertEqual(slicer_launcher(root / 'Slicer.app'), mac)  # the .app itself
            self.assertEqual(slicer_launcher(root / 'Slicer.app/Contents'), mac)
            self.assertEqual(slicer_launcher(root / 'Slicer.app/Contents/MacOS'), mac)
            self.assertEqual(slicer_launcher(root / 'windows'), windows)
            self.assertEqual(slicer_launcher(root / 'empty'), root / 'empty/Slicer')

    def test_b9_run_launches_the_registration_cli_with_the_macos_app_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = make_bundle(root / 'inputs')
            app = root / 'Slicer.app'
            launcher = self.tree(app, 'Contents/MacOS/Slicer')
            launcher.chmod(0o755)
            self.tree(app, 'Contents/bin/PythonSlicer')
            core = root / 'core'; core.mkdir()
            for name in ('BRAINSFit', 'ResampleScalarVectorDWIVolume'):
                (core / name).write_text('fixture')
            folder = root / 'dti'; folder.mkdir()
            args = argparse.Namespace(subject='x', output_dir=folder, slicer_path=app, core_cli=core,
                                      reference_image=files['mask'], mask_flip=1, jobs=1, minimal=True, **files)
            launched = []

            def command(argv, **kwargs):
                if argv[1:2] == ['--launch']:
                    launched.append(argv[0])
                    raise RuntimeError('stop after the first Slicer command')
                if Path(argv[1]).name == 'nhdr_write.py':
                    values = {argv[i]: argv[i + 1] for i in range(2, len(argv), 2)}
                    write_nhdr(values['--nifti'], values.get('--bval'), values.get('--bvec'), values['--nhdr'])
                    return
                outputs = [argv[-1]]
                if Path(argv[1]).name == 'teem_dti.py':
                    outputs = argv[-2:] + [item.partition('=')[2] for i, item in enumerate(argv) if argv[i - 1] == '--scalar']
                for output in outputs:
                    sitk.WriteImage(sitk.GetImageFromArray(np.ones((5, 6, 7), np.float32)), output, True)

            with patch('preprocessing.dti.subprocess.run', side_effect=command):
                with self.assertRaises(RuntimeError):
                    dti.run(args)
            self.assertEqual(launched[:1], [str(launcher)])

    def test_f2_dti_sh_finds_the_macos_app_layout_of_pythonslicer(self):
        text = (ROOT / 'preprocessing/dti.sh').read_text()
        # T3 moved the macOS rule to SLICER_ROOT (= Contents for a .app); the text no longer names Contents/bin.
        self.assertIn('*.app) SLICER_ROOT=$SLICER_PATH/Contents', text)
        self.assertIn('$SLICER_ROOT/bin/PythonSlicer', text)
        self.assertIn('bin/PythonSlicer.exe', text)


class DtiShellTests(unittest.TestCase):
    def setup(self, root, pythonslicer):
        files = make_bundle(root / 'inputs')
        slicer = root / 'slicer'
        core = slicer / 'lib/Slicer-5.2/cli-modules'
        core.mkdir(parents=True)
        (slicer / 'bin').mkdir()
        # W5: a Slicer without any SlicerDMRI folder.
        tools = [slicer / 'Slicer', core / 'BRAINSFit', core / 'ResampleScalarVectorDWIVolume']
        tools += [slicer / 'bin/PythonSlicer'] if pythonslicer else []
        for path in tools:
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o755)
        record = root / 'dti-command.txt'
        helper = root / 'python-helper'
        helper.write_text('#!/bin/sh\ncase "$(basename "$1")" in\n'
                          f'  dti.py) printf "%s\\n" "$@" > "{record}"; exit 0;;\n'
                          f'esac\nexec "{sys.executable}" "$@"\n')
        helper.chmod(0o755)
        command = ['bash', str(ROOT / 'preprocessing/dti.sh'), '--subject', 'x', '--dwi', files['dwi'],
                   '--bval', files['bval'], '--bvec', files['bvec'], '--mask', files['mask'],
                   '--output-root', root / 'out', '--slicer-path', slicer, '--python-bin', helper,
                   '--reference-image', files['mask']]
        result = subprocess.run(list(map(str, command)), cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        return result, record, slicer

    def test_w5_dti_sh_needs_no_slicerdmri_and_passes_no_dmri_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            result, record, slicer = self.setup(Path(temp), True)
            self.assertEqual(result.returncode, 0, result.stdout)
            arguments = record.read_text().splitlines()
            self.assertNotIn('--dmri-cli', arguments)
            self.assertEqual(Path(arguments[arguments.index('--slicer-path') + 1]).resolve(), slicer.resolve())

    def test_w5_dti_sh_reports_a_missing_pythonslicer(self):
        with tempfile.TemporaryDirectory() as temp:
            result, record, _ = self.setup(Path(temp), False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PythonSlicer', result.stdout)
            self.assertFalse(record.exists())


if __name__ == '__main__':
    unittest.main()
