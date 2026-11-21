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
import struct
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

    def __call__(self, argv, environ=None, existing=None):
        b = self.core_build
        if existing is not None:  # an already installed prefix; validate_runtime returns its record
            (self.folder / 'prefix').mkdir(parents=True, exist_ok=True)
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
                patch.object(b, 'validate_runtime', return_value=existing if existing is not None else {}):
            b.build(args)
        if existing is not None:
            return existing
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

    def test_b2fix_runtime_T3_no_target_unsets_a_cached_cmake_target(self):
        """T3 (Codex P1, Opus P2-4): a CMakeCache.txt target of an earlier build must not survive a no-target build."""
        with tempfile.TemporaryDirectory() as folder:
            fake = FakeBuild(self.core_build, folder)
            record = fake(dict())
            self.assertEqual(len(fake.configures()), 2)
            for command, _env in fake.configures():
                self.assertIn('-UCMAKE_OSX_DEPLOYMENT_TARGET', command)
            self.assertEqual(record['build']['macos_deployment_target'], '')
            self.assertEqual(record['build']['cmake_target_options'], ['-UCMAKE_OSX_DEPLOYMENT_TARGET'])
        with tempfile.TemporaryDirectory() as folder:  # the unset applies to macOS builds only
            fake = FakeBuild(self.core_build, folder, system='Linux', machine='x86_64')
            fake(dict())
            for command, _env in fake.configures():
                self.assertNotIn('-UCMAKE_OSX_DEPLOYMENT_TARGET', command)

    def existing_record(self, target):
        build = {} if target is None else dict(macos_deployment_target=target)
        return dict(recipe_sha256=self.core_build.recipe_digest(), build=build)

    def test_b2fix_runtime_T3_refuses_a_prefix_built_for_another_target(self):
        """T3 (Codex P1): an installed prefix with a different or unrecorded target is not reused silently."""
        for recorded in ('', None, '15.0', '14.0'):
            with tempfile.TemporaryDirectory() as folder:
                fake = FakeBuild(self.core_build, folder)
                with self.assertRaisesRegex(ValueError, 'deployment target'):
                    fake(dict(target='13.0'), existing=self.existing_record(recorded))
                self.assertEqual(fake.commands, [], recorded)

    def test_b2fix_runtime_T3_reuses_a_prefix_with_the_requested_target(self):
        for requested, recorded in (('13.0', '13.0'), ('13', '13.0'), ('13.0', '13.0.0')):
            with tempfile.TemporaryDirectory() as folder:
                fake = FakeBuild(self.core_build, folder)
                fake(dict(target=requested), existing=self.existing_record(recorded))
                self.assertEqual(fake.commands, [], (requested, recorded))

    def test_b2fix_runtime_T3_no_requested_target_reuses_any_prefix(self):
        for recorded in ('', None, '13.0'):
            with tempfile.TemporaryDirectory() as folder:
                fake = FakeBuild(self.core_build, folder)
                fake(dict(), existing=self.existing_record(recorded))
                self.assertEqual(fake.commands, [], recorded)

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


def macho(cputype=0x0100000C):
    """A thin 64-bit Mach-O header (magic, cputype, zeros) of 32 bytes."""
    return struct.pack('<II', 0xFEEDFACF, cputype) + b'\0' * 24


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = load_verifier()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / 'runtime'
        (self.root / 'bin').mkdir(parents=True)
        for name in COMMANDS:
            path = self.root / 'bin' / name
            path.write_bytes(macho())
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

    def test_b2fix_runtime_T2_verifier_fails_on_a_symlink_in_bin(self):
        """T2 (Codex P1, Opus P2-3): a symlinked tool skipped every check and still passed."""
        (self.root / 'bin/mris_sphere').unlink()
        (self.root / 'bin/mris_sphere').symlink_to('mris_inflate')
        report = self.check()
        self.assertFalse(report['ok'])
        self.assertTrue(any('mris_sphere' in item and 'symlink' in item for item in report['failures']), report['failures'])

    def test_b2fix_runtime_T2_verifier_fails_when_no_tool_was_checked(self):
        """T2 (Codex P1): all seven tools symlinks gave ok=true with tools=[]."""
        real = self.root / 'real'
        real.write_bytes(b'not a mach-o')
        for name in COMMANDS:
            (self.root / 'bin' / name).unlink()
            (self.root / 'bin' / name).symlink_to('../real')
        report = self.check()
        self.assertFalse(report['ok'])
        self.assertEqual(report['tools'], [])
        self.assertTrue(any('no tool was checked' in item for item in report['failures']), report['failures'])
        empty = Path(self.folder.name) / 'empty'
        (empty / 'bin').mkdir(parents=True)
        self.assertTrue(any('no tool was checked' in item
                            for item in self.verifier.check_runtime(empty, expected_min_macos='13.0')['failures']))

    def test_b2fix_runtime_T2_verifier_fails_on_a_binary_that_is_not_arm64(self):
        """T2 (Opus P2-3): minos 13.0 on an x86_64 file passed; the CPU type is checked now."""
        (self.root / 'bin/mris_sphere').write_bytes(macho(0x01000007))  # x86_64
        (self.root / 'bin/mri_surf2surf').write_bytes(b'\xcf\xfa\xed\xfe')  # header cut short
        fat = struct.pack('>IIIiiII', 0xCAFEBABE, 2, 0x0100000C, 0, 4096, 4096, 14) \
            + struct.pack('>iiIII', 0x01000007, 3, 8192, 4096, 14)
        (self.root / 'bin/mris_register').write_bytes(fat)  # arm64 + x86_64
        report = self.check()
        self.assertFalse(report['ok'])
        joined = '\n'.join(report['failures'])
        self.assertRegex(joined, r'mris_sphere: .*x86_64')
        self.assertRegex(joined, r'mri_surf2surf: .*CPU type')
        self.assertRegex(joined, r'mris_register: .*arm64.*x86_64')

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
        # T1: manual only on main; the push trigger of the CI branch is gone.
        self.assertNotRegex(self.head, r'(?m)^  (push|pull_request|schedule|workflow_run|release):')
        self.assertNotIn('ci/b2-surface-runtime', self.text)
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

    def test_b2fix_runtime_T5_publish_has_its_own_concurrency_group_that_is_never_cancelled(self):
        job = self.jobs['publish']
        self.assertRegex(job, r'(?s)\n    concurrency:\s*\n      group: \S+\s*\n      cancel-in-progress: false\s*\n')
        group = re.search(r'\n      group: (\S+)', job).group(1)
        self.assertNotEqual(group, re.search(r'(?m)^  group: (\S+)', self.head).group(1))

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

    def test_b2fix_runtime_T4_no_older_macos_job(self):
        """T4 (Opus P2-1): the macos-14 job always ended cancelled and made the run conclusion cancelled."""
        self.assertNotIn('run-on-older-macos', self.jobs)
        self.assertNotIn('macos-14', self.text)
        self.assertEqual(sorted(self.jobs), ['build-darwin-arm64', 'build-linux-x86_64', 'publish', 'run-on-older-linux'])


if __name__ == '__main__':
    unittest.main()
