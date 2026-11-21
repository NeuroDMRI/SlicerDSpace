"""Batch 2 runtime lane: rebuild the surface runtime for macOS 13 in GitHub Actions.

Covers the deployment-target parameter of the vendored build.py, the CI verifier script and the
static shape of .github/workflows/build-surface-runtime.yml. No compiler, network or GitHub is needed.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / 'anatomy/surface/ddsurfer_backend'
WORKFLOW = REPO / '.github/workflows/build-surface-runtime.yml'
VERIFIER = REPO / '.github/scripts/verify_surface_runtime.py'
COMMANDS = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register',
            'mris_place_surface', 'mri_label2label', 'mri_surf2surf')


def load_verifier():
    spec = importlib.util.spec_from_file_location('verify_surface_runtime_b2', str(VERIFIER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBuild:
    """Run build.build() with every external step replaced; collect the commands it would run."""

    def __init__(self, core_build, folder, system='Darwin', machine='arm64'):
        self.core_build, self.folder = core_build, Path(folder)
        self.system, self.machine = system, machine
        self.commands = []

    def check_output(self, command, **kwargs):
        command = [str(value) for value in command]
        if command[-1] == '--version' and command[0].endswith('cmake'):
            return 'cmake version 3.31.6\n'
        if command[-1] == '--version':
            return 'Apple clang version 17.0.0 (clang-1700.0.13.5)\n'
        if command[:2] == ['xcrun', '--show-sdk-version']:
            return '15.5\n'
        raise AssertionError('unexpected check_output: %r' % (command,))

    def run(self, command, log, env):
        self.commands.append(([str(value) for value in command], dict(env)))

    def directory(self, name):
        path = self.folder / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def install_assets(self, destination):
        (Path(destination) / 'docs').mkdir(parents=True, exist_ok=True)
        (Path(destination) / 'LICENSE.txt').write_text('notice\n')

    def copy2(self, source, destination):
        Path(destination).write_bytes(b'\xcf\xfa\xed\xfe binary')

    def __call__(self, argv, environ=None):
        b = self.core_build
        args = b.argparse.Namespace(jobs=2, prefix=self.folder / 'prefix', work_dir=self.folder / 'work',
                                    source_root=None, cmake=None, offline=False,
                                    macos_deployment_target=argv.get('target'), build_commit=argv.get('commit'))
        env = {key: value for key, value in os.environ.items() if key != 'MACOSX_DEPLOYMENT_TARGET'}
        env.update(environ or {})
        with patch.dict(os.environ, env, clear=True), \
                patch.object(b, 'host', return_value=(self.system, self.machine)), \
                patch.object(b, 'require_tool', side_effect=lambda value, name: '/usr/bin/' + name), \
                patch.object(b.subprocess, 'check_output', side_effect=self.check_output), \
                patch.object(b, 'prepare_source', return_value=self.directory('source')), \
                patch.object(b, 'prepare_dependency', side_effect=lambda name, work, offline: self.directory(name)), \
                patch.object(b, 'stage_source', return_value=self.directory('stage')), \
                patch.object(b, 'run', side_effect=self.run), \
                patch.object(b, 'install_assets', side_effect=self.install_assets), \
                patch.object(b.shutil, 'copy2', side_effect=self.copy2), \
                patch.object(b, 'validate_runtime', return_value={}):
            b.build(args)
        return json.loads((self.folder / 'prefix/runtime.json').read_text())

    def configures(self):
        return [(command, env) for command, env in self.commands if '-S' in command]


class BuildDeploymentTargetTests(unittest.TestCase):
    def setUp(self):
        context = patch.dict(sys.modules)
        context.start()
        self.addCleanup(context.stop)
        path = patch.object(sys, 'path', [str(BACKEND), *sys.path])
        path.start()
        self.addCleanup(path.stop)
        from postprocessing.core import build as core_build
        self.core_build = core_build

    def test_b2_runtime_target_reaches_both_cmake_configures_and_the_record(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = FakeBuild(self.core_build, folder)
            record = fake(dict(target='13.0', commit='0123abc'))
            configures = fake.configures()
            self.assertEqual(len(configures), 2)  # ITK, then the surface tools
            for command, env in configures:
                self.assertIn('-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0', command)
                self.assertEqual(env['MACOSX_DEPLOYMENT_TARGET'], '13.0')
            for _command, env in fake.commands:
                self.assertEqual(env['MACOSX_DEPLOYMENT_TARGET'], '13.0')
            build = record['build']
            self.assertEqual(build['macos_deployment_target'], '13.0')
            self.assertEqual(build['sdk'], '15.5')
            self.assertEqual(build['builder_commit'], '0123abc')
            self.assertEqual(build['cmake_target_options'], ['-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0'])
            self.assertIn('-DARM64', build['environment_flags']['CFLAGS'].split())
            self.assertIn('-DARM64', build['environment_flags']['CXXFLAGS'].split())
            self.assertEqual(record['recipe_sha256'], self.core_build.recipe_digest())

    def test_b2_runtime_without_target_keeps_the_local_build_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = FakeBuild(self.core_build, folder)
            record = fake(dict())
            for command, env in fake.configures():
                self.assertFalse(any(part.startswith('-DCMAKE_OSX_DEPLOYMENT_TARGET') for part in command))
                self.assertNotIn('MACOSX_DEPLOYMENT_TARGET', env)
            self.assertEqual(record['build']['macos_deployment_target'], '')
            self.assertEqual(record['build']['builder_commit'], '')

    def test_b2_runtime_target_from_environment_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = FakeBuild(self.core_build, folder)
            record = fake(dict(), environ={'MACOSX_DEPLOYMENT_TARGET': '14.0'})
            for command, _env in fake.configures():
                self.assertIn('-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0', command)
            self.assertEqual(record['build']['macos_deployment_target'], '14.0')

    def test_b2_runtime_target_rejects_bad_values_and_other_systems(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'deployment target'):
                FakeBuild(self.core_build, folder)(dict(target='thirteen'))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'macOS'):
                FakeBuild(self.core_build, folder, system='Linux', machine='x86_64')(dict(target='13.0'))

    def test_b2_runtime_command_line_accepts_target_and_commit(self):
        seen = []
        with patch.object(self.core_build, 'build', side_effect=seen.append):
            self.core_build.main(['--macos-deployment-target', '13.0', '--build-commit', 'abc'])
        self.assertEqual(seen[0].macos_deployment_target, '13.0')
        self.assertEqual(seen[0].build_commit, 'abc')


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = load_verifier()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / 'runtime'
        (self.root / 'bin').mkdir(parents=True)
        for name in COMMANDS:
            path = self.root / 'bin' / name
            path.write_bytes(b'\xcf\xfa\xed\xfe')
            path.chmod(0o755)
        (self.root / 'LICENSE.txt').write_text('notice\n')
        self.minos = {name: (13, 0, 0) for name in COMMANDS}
        self.libraries = {name: ['/usr/lib/libSystem.B.dylib', '/usr/lib/libc++.1.dylib'] for name in COMMANDS}
        self.started = {name: (0, name + ' freesurfer 7.4.1-ddsurfer-local') for name in COMMANDS}

    def check(self, **kwargs):
        options = dict(expected_min_macos='13.0', start=True,
                       min_os_reader=lambda path: self.minos[Path(path).name],
                       library_reader=lambda path: self.libraries[Path(path).name],
                       starter=lambda path: self.started[Path(path).name])
        options.update(kwargs)
        return self.verifier.check_runtime(self.root, **options)

    def test_b2_runtime_verifier_passes_a_clean_13_runtime(self):
        report = self.check()
        self.assertTrue(report['ok'], report['failures'])
        self.assertEqual(sorted(row['tool'] for row in report['tools']), sorted(COMMANDS))
        self.assertTrue(all(row['min_macos'] == '13.0' and row['started'] for row in report['tools']))

    def test_b2_runtime_verifier_fails_on_a_15_binary(self):
        self.minos['mris_sphere'] = (15, 0, 0)
        report = self.check()
        self.assertFalse(report['ok'])
        self.assertTrue(any('mris_sphere' in item and '15.0' in item for item in report['failures']))

    def test_b2_runtime_verifier_fails_on_a_non_system_library(self):
        self.libraries['mris_register'].append('/opt/homebrew/opt/libomp/lib/libomp.dylib')
        self.libraries['mri_surf2surf'].append('@rpath/libitk.dylib')
        report = self.check()
        self.assertFalse(report['ok'])
        joined = '\n'.join(report['failures'])
        self.assertIn('libomp', joined)
        self.assertIn('@rpath/libitk.dylib', joined)

    def test_b2_runtime_verifier_fails_on_a_tool_list_change(self):
        (self.root / 'bin/mris_inflate').unlink()
        (self.root / 'bin/mris_extra').write_bytes(b'\xcf\xfa\xed\xfe')
        self.minos['mris_extra'] = (13, 0, 0)
        self.libraries['mris_extra'] = ['/usr/lib/libSystem.B.dylib']
        self.started['mris_extra'] = (0, 'x')
        report = self.check()
        self.assertFalse(report['ok'])
        joined = '\n'.join(report['failures'])
        self.assertIn('mris_inflate', joined)
        self.assertIn('mris_extra', joined)

    def test_b2_runtime_verifier_fails_when_a_tool_does_not_start(self):
        self.started['mri_label2label'] = (134, 'dyld: Symbol not found')
        report = self.check()
        self.assertFalse(report['ok'])
        self.assertTrue(any('mri_label2label' in item for item in report['failures']))

    def test_b2_runtime_verifier_fails_on_newer_object_warning_in_build_logs(self):
        logs = Path(self.folder.name) / 'work'
        logs.mkdir()
        (logs / 'surface-build.log').write_text(
            "ld: warning: object file (libITKCommon.a(x.o)) was built for newer 'macOS' version (15.0) than being linked (13.0)\n")
        report = self.check(build_logs=logs)
        self.assertFalse(report['ok'])
        self.assertTrue(any('built for newer' in item for item in report['failures']))

    def test_b2_runtime_verifier_compares_data_files_with_a_reference(self):
        reference = Path(self.folder.name) / 'reference'
        (reference / 'bin').mkdir(parents=True)
        for name in COMMANDS:  # binaries may differ in bytes; only the names must match
            (reference / 'bin' / name).write_bytes(b'other build')
        (reference / 'LICENSE.txt').write_text('different notice\n')
        report = self.check(reference=reference)
        self.assertFalse(report['ok'])
        self.assertTrue(any('LICENSE.txt' in item for item in report['failures']))
        (reference / 'LICENSE.txt').write_text('notice\n')
        self.assertTrue(self.check(reference=reference)['ok'])

    def check_elf(self, versions, needed):
        return self.verifier.check_runtime(
            self.root, expected_min_macos=None, max_glibc='2.28', start=True,
            elf_reader=lambda path: (versions[Path(path).name], needed[Path(path).name]),
            starter=lambda path: self.started[Path(path).name])

    def test_b2_runtime_verifier_elf_mode_checks_glibc_and_libstdcxx(self):
        versions = {name: {'GLIBC_2.2.5', 'GLIBC_2.17'} for name in COMMANDS}
        # libcrypt.so.1 (FreeSurfer's activation check uses crypt) is part of glibc / libxcrypt on every target.
        needed = {name: ['libm.so.6', 'libc.so.6', 'libpthread.so.0', 'libcrypt.so.1'] for name in COMMANDS}
        report = self.check_elf(versions, needed)
        self.assertTrue(report['ok'], report['failures'])
        self.assertTrue(all(row['max_glibc'] == '2.17' for row in report['tools']))
        versions['mris_sphere'] = {'GLIBC_2.34', 'GLIBC_2.2.5'}
        versions['mris_register'] = {'GLIBC_2.17', 'GLIBCXX_3.4.29', 'CXXABI_1.3.9'}
        needed['mri_surf2surf'] = ['libc.so.6', 'libstdc++.so.6', 'libgomp.so.1']
        joined = '\n'.join(self.check_elf(versions, needed)['failures'])
        self.assertIn('mris_sphere: needs GLIBC_2.34', joined)
        self.assertIn('GLIBCXX_3.4.29', joined)
        self.assertIn('libstdc++.so.6', joined)
        self.assertIn('libgomp.so.1', joined)

    def test_b2_runtime_verifier_parses_objdump_and_readelf_output(self):
        objdump = ('0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.2.5) memcpy\n'
                   '0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.17  clock_gettime\n'
                   '0000000000000000      DF *UND*\t0000000000000000  GLIBCXX_3.4 _ZNSo3putEc\n')
        self.assertEqual(self.verifier.parse_symbol_versions(objdump), {'GLIBC_2.2.5', 'GLIBC_2.17', 'GLIBCXX_3.4'})
        readelf = (' 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]\n'
                   ' 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]\n')
        self.assertEqual(self.verifier.parse_needed(readelf), ['libm.so.6', 'libc.so.6'])

    def test_b2_runtime_verifier_parses_otool_output(self):
        text = ('/x/bin/mris_sphere:\n'
                '\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)\n'
                '\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1900.180.0)\n')
        self.assertEqual(self.verifier.parse_otool_libraries(text),
                         ['/usr/lib/libSystem.B.dylib', '/usr/lib/libc++.1.dylib'])


def job_blocks(text):
    jobs = text.split('\njobs:\n', 1)[1]
    names = list(re.finditer(r'(?m)^  ([A-Za-z0-9_-]+):\s*$', jobs))
    return {match.group(1): jobs[match.end():(names[i + 1].start() if i + 1 < len(names) else len(jobs))]
            for i, match in enumerate(names)}


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf-8')
        cls.jobs = job_blocks(cls.text)
        cls.head = cls.text.split('\njobs:\n', 1)[0]

    def test_b2_runtime_workflow_triggers(self):
        self.assertRegex(self.head, r'(?m)^  workflow_dispatch:\s*$')
        self.assertRegex(self.head, r'(?s)publish:.*?type: boolean.*?default: false')
        self.assertRegex(self.head, r'(?s)release_tag:.*?default: ""')
        self.assertRegex(self.head, r'(?s)  push:\s*\n    branches:\s*\n      - ci/b2-surface-runtime\s*\n')
        self.assertEqual(re.findall(r'(?m)^      - (\S+)', self.head), ['ci/b2-surface-runtime'])
        self.assertNotRegex(self.head, r'(?m)^  (pull_request|schedule|release):')
        self.assertRegex(self.head, r'(?m)^permissions:\s*\n  contents: read\s*$')

    def test_b2_runtime_build_job(self):
        job = self.jobs['build-darwin-arm64']
        self.assertIn('runs-on: macos-15', job)
        self.assertIn('postprocessing.core.build', job)
        self.assertIn('--macos-deployment-target 13.0', job)
        self.assertIn('MACOSX_DEPLOYMENT_TARGET: "13.0"', job)
        self.assertIn('verify_surface_runtime.py', job)
        self.assertIn('--expect-min-macos 13.0', job)
        self.assertIn('scripts/package_surface_runtime.py', job)
        self.assertRegex(self.head, r'(?m)^  ASSET: surface-runtime-darwin-arm64-macos13\.tar\.gz\s*$')
        self.assertIn('--output "$OUT/$ASSET"', job)
        self.assertIn('shasum -a 256', job)
        self.assertIn('actions/upload-artifact@v4', job)
        self.assertIn('if: failure()', job)  # build logs are printed when a step fails
        self.assertNotIn('gh release', job)

    def test_b2_runtime_publish_job_is_gated_and_never_overwrites(self):
        job = self.jobs['publish']
        self.assertIn("github.event_name == 'workflow_dispatch'", job)
        self.assertIn('inputs.publish', job)
        self.assertIn('needs: build-darwin-arm64', job)
        self.assertIn('contents: write', job)
        self.assertIn('gh release upload', job)
        self.assertIn('already has', job)  # the asset-absent check
        self.assertNotIn('--clobber', self.text)
        self.assertEqual(self.text.count('contents: write'), 1)

    def test_b2_runtime_publish_can_take_the_verified_artifact_of_an_earlier_run(self):
        # Each build embeds __DATE__/__TIME__, so a new run gives a new sha256. The catalog patch names one run.
        self.assertRegex(self.head, r'(?s)artifact_run_id:.*?type: string.*?default: ""')
        job = self.jobs['publish']
        self.assertIn('run-id: ${{ inputs.artifact_run_id || github.run_id }}', job)
        self.assertIn('github-token: ${{ github.token }}', job)
        self.assertIn('actions: read', job)

    def test_b2_runtime_linux_job_builds_in_manylinux_2_28_and_checks_glibc(self):
        job = self.jobs['build-linux-x86_64']
        self.assertIn('image: quay.io/pypa/manylinux_2_28_x86_64', job)
        self.assertIn('-static-libstdc++', job)
        self.assertIn('postprocessing.core.build', job)
        self.assertIn('--max-glibc 2.28', job)
        self.assertIn('surface-runtime-linux-x86_64-glibc228', job)
        self.assertNotIn('gh release', job)
        older = self.jobs['run-on-older-linux']
        self.assertIn('needs: build-linux-x86_64', older)
        self.assertIn('image: ubuntu:20.04', older)
        self.assertIn('--version', older)

    def test_b2_runtime_older_macos_job_may_fail(self):
        job = self.jobs['run-on-older-macos']
        self.assertIn('runs-on: macos-14', job)
        self.assertIn('continue-on-error: true', job)
        self.assertIn('needs: build-darwin-arm64', job)
        self.assertIn('actions/download-artifact@v4', job)
        self.assertIn('--version', job)
        self.assertIn('timeout-minutes:', job)


if __name__ == '__main__':
    unittest.main()
