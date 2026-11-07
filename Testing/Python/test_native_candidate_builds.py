"""No-network checks for release-candidate packaging scaffolding."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import ntpath
import os
import posixpath
import shlex
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('candidate_builder', ROOT / 'scripts/native/prepare_mrtrix_candidate.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

WINDOWS_TEST_PROGRAMS = (
    'mrconvert', 'dwi2fod', 'tckgen', 'tcksift2', 'tck2connectome', 'tckconvert',
    '5ttcheck', '5tt2gmwmi', 'mtnormalise',
    'amp2response', 'dwi2mask', 'dwi2tensor', 'dwiextract', 'maskfilter', 'mrcalc',
    'mrcat', 'mrinfo', 'mrmath', 'mrstats', 'mrthreshold', 'mrtransform', 'peaks2amp', 'sh2peaks',
    'tensor2metric', 'tckstats', 'tcksample',
)


def windows_source(root):
    source = root / 'upstream'
    (source / 'bin').mkdir(parents=True)
    (source / 'LICENCE.txt').write_text('upstream license fixture\n')
    for name in WINDOWS_TEST_PROGRAMS:
        binary = source / 'bin' / f'{name}.exe'
        binary.write_text('#!/bin/sh\n[ "$1" = "-version" ] && echo "' + name + ' 3.0.8"\n')
        if os.name != 'nt':
            binary.chmod(0o755)
    lib = source / 'lib/mrtrix3'
    (lib / 'dwi2response').mkdir(parents=True)
    (lib / '__init__.py').write_text('from ._version import __version__\n')
    (lib / '_version.py').write_text('__version__ = "3.0.8"\n')
    (lib / 'path.py').write_text(
        'import os\n'
        'try:\n'
        '  from shlex import quote\n'
        'except ImportError:\n'
        '  from pipes import quote #pylint: disable=deprecated-module\n'
    )
    (lib / 'dwi2response/dhollander.py').write_text('algorithm = "dhollander"\n')
    (source / 'bin/dwi2response').write_text(
        '#!/no/shebang/python\n'
        'import os, sys\n'
        'from pathlib import Path\n'
        'from mrtrix3 import __version__\n'
        'from mrtrix3.dwi2response.dhollander import algorithm\n'
        'root = Path(__file__).resolve().parents[1]\n'
        'assert root == Path.cwd().resolve() and root.name != "upstream"\n'
        'assert " " in root.name\n'
        'assert os.environ["PATH"].split(os.pathsep)[0] == str(root / "bin")\n'
        'assert "build-tools" not in os.environ["PATH"]\n'
        'assert "MSYSTEM" not in os.environ and "PYTHONHOME" not in os.environ\n'
        'assert os.environ["PYTHONPATH"] == str(root / "lib")\n'
        'assert os.environ["HOME"] == str(root)\n'
        'assert sys.argv[1:] == [algorithm, "__print_full_usage__"]\n'
        'print(algorithm, __version__)\n'
    )
    return source


class NativeCandidateTest(unittest.TestCase):
    def test_linux_candidate_has_unpublished_catalog_and_required_binaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'upstream'
            (source / 'bin').mkdir(parents=True)
            (source / 'LICENCE.txt').write_text('upstream license fixture\n')
            for name in module.NATIVE_COMMANDS:
                (source / 'bin' / name).write_text('#!/bin/sh\nexit 0\n')
            result = module.build(source, root / 'dist', system='Linux', machine='x86_64', dll_search=[], objdump='objdump')
            archive, record_file = Path(result['archive']), Path(result['record'])
            record = json.loads(record_file.read_text())
            self.assertEqual(record['sha256'], module.sha256(archive))
            self.assertFalse(record['published'])
            self.assertFalse(record['scientifically_accepted'])
            self.assertEqual(len(record['commands']), len(module.NATIVE_COMMANDS))
            from shared.native_runtimes import select_bundle, NativeRuntimeUnavailable
            with self.assertRaises(NativeRuntimeUnavailable):
                select_bundle({'bundles': [record]}, 'mrtrix', system='Linux', machine='x86_64')

    @unittest.skipIf(os.name == 'nt', 'Synthetic shell binaries need a POSIX host')
    def test_candidate_smoke_cli_runs_outside_the_source_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'upstream'
            (source / 'bin').mkdir(parents=True)
            (source / 'LICENCE.txt').write_text('license fixture\n')
            for name in module.NATIVE_COMMANDS:
                (source / 'bin' / name).write_text('#!/bin/sh\nexit 0\n')
            from shared.native_runtimes import host_platform
            system, machine = host_platform()
            result = module.build(source, root / 'dist', system=system, machine=machine,
                                  dll_search=[], objdump='objdump')
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            report = root / 'smoke.json'
            completed = subprocess.run([sys.executable, str(ROOT / 'scripts/native/smoke_mrtrix_candidate.py'),
                '--archive', result['archive'], '--record', result['record'], '--output', str(report)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            checks = json.loads(report.read_text())
            self.assertEqual(checks['status'], 'passed')
            self.assertEqual(len(checks['checks']), len(module.NATIVE_COMMANDS))

    def test_missing_executable_refuses_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'upstream'
            (source / 'bin').mkdir(parents=True)
            (source / 'LICENCE.txt').write_text('upstream license fixture\n')
            with self.assertRaisesRegex(FileNotFoundError, 'MRtrix3 program'):
                module.build(source, Path(temp) / 'dist', system='Linux', machine='x86_64',
                             dll_search=[], objdump='objdump')

    def test_windows_requires_runtime_dll_search_path(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'upstream'
            (source / 'bin').mkdir(parents=True)
            (source / 'LICENCE.txt').write_text('upstream license fixture\n')
            for name in module.NATIVE_COMMANDS:
                (source / 'bin' / (name + '.exe')).write_bytes(b'MZ\x00fixture')
            with self.assertRaisesRegex(ValueError, 'MSYS2 MinGW DLL search'):
                module.build(source, Path(temp) / 'dist', system='Windows', machine='x86_64',
                             dll_search=[], objdump='objdump')

    def test_windows_archive_contains_csd_act_and_response_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = windows_source(root)
            deps = root / 'mingw'
            deps.mkdir()
            (deps / 'libact.dll').write_bytes(b'ACT dependency')
            def fixture_imports(path, **kwargs):
                if path.suffix not in ('.exe', '.dll'):
                    raise ValueError(f'objdump expected a PE file: {path}')
                return {'libact.dll'} if path.name == '5tt2gmwmi.exe' else set()

            with patch.object(module, '_dll_dependencies', side_effect=fixture_imports):
                result = module.build(source, root / 'dist', system='Windows', machine='x86_64',
                                      dll_search=[deps], objdump='objdump')
            expected_commands = {'bin/dwi2response', *(f'bin/{name}.exe' for name in WINDOWS_TEST_PROGRAMS)}
            record = json.loads(Path(result['record']).read_text())
            self.assertEqual(set(record['commands']), expected_commands)
            self.assertFalse(record['published'])
            with zipfile.ZipFile(result['archive']) as archive:
                receipt = json.loads(archive.read('native-runtime.json'))
                self.assertEqual(set(receipt['commands']), expected_commands)
                self.assertTrue(expected_commands <= set(archive.namelist()))
                self.assertEqual(archive.read('bin/dwi2response'), (source / 'bin/dwi2response').read_bytes())
                self.assertIn('lib/mrtrix3/__init__.py', archive.namelist())
                self.assertIn('lib/mrtrix3/_version.py', archive.namelist())
                self.assertIn('lib/mrtrix3/dwi2response/dhollander.py', archive.namelist())
                self.assertIn('bin/libact.dll', archive.namelist())

    def test_windows_response_paths_round_trip_upstream_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = windows_source(root)
            path_helper = source / 'lib/mrtrix3/path.py'
            original = path_helper.read_bytes()
            with patch.object(module, '_dll_dependencies', return_value=set()):
                result = module.build(source, root / 'dist', system='Windows', machine='x86_64',
                                      dll_search=[source / 'bin'], objdump='objdump')
            with zipfile.ZipFile(result['archive']) as archive:
                packaged = archive.read('lib/mrtrix3/path.py')
                receipt = json.loads(archive.read('native-runtime.json'))
            namespace = {}
            exec(compile(packaged, 'lib/mrtrix3/path.py', 'exec'), namespace)
            namespace['os'] = SimpleNamespace(sep='\\', path=ntpath)
            quote = namespace['quote']
            for filename in (
                r'C:\subject\dwi.mif',
                r'C:\diffusion inputs\mask.mif',
                r'C:\研究数据\导出.mif',
                r"C:\safe & more\O'Neil %TEMP%.mif",
                r'\\server\share\work dir\subject.mif',
            ):
                with self.subTest(filename=filename):
                    target = r'D:\scratch dir\dwi.mif'
                    command = f'mrconvert {quote(filename)} - | dwiextract - {quote(target)}'
                    # MRtrix 3.0.8 run.command uses exactly this Windows split.
                    tokens = [entry.strip('"') for entry in shlex.split(command, posix=False)]
                    self.assertEqual(tokens, ['mrconvert', filename, '-', '|', 'dwiextract', '-', target])
            with self.assertRaisesRegex(ValueError, 'Windows command path'):
                quote('C:\\data" && external-command')
            self.assertEqual(quote('log argument'), shlex.quote('log argument'))
            namespace['os'] = SimpleNamespace(sep='/', path=posixpath)
            self.assertEqual(quote("/subject's data/dwi.mif"), shlex.quote("/subject's data/dwi.mif"))
            self.assertEqual(path_helper.read_bytes(), original)
            self.assertNotEqual(packaged, original)
            [patch_record] = receipt['patches']
            self.assertEqual(patch_record['file'], 'lib/mrtrix3/path.py')
            self.assertEqual(patch_record['upstream_commit'], '464173c5294e021b2d508e557085d90c09d642fd')
            self.assertEqual(patch_record['original_sha256'], module.sha256(path_helper))
            self.assertEqual(patch_record['patched_sha256'], hashlib.sha256(packaged).hexdigest())

    def test_windows_refuses_to_patch_unexpected_quote_implementation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = windows_source(root)
            (source / 'lib/mrtrix3/path.py').write_text('from shlex import quote as alternate\n')
            with patch.object(module, '_dll_dependencies', return_value=set()):
                with self.assertRaisesRegex(ValueError, 'unexpected upstream quote implementation'):
                    module.build(source, root / 'dist', system='Windows', machine='x86_64',
                                 dll_search=[source / 'bin'], objdump='objdump')

    def test_windows_requires_built_response_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = windows_source(root)
            (source / 'lib/mrtrix3/_version.py').unlink()
            with patch.object(module, '_dll_dependencies', return_value=set()):
                with self.assertRaisesRegex(FileNotFoundError, '_version.py'):
                    module.build(source, root / 'dist', system='Windows', machine='x86_64',
                                 dll_search=[source / 'bin'], objdump='objdump')

    @unittest.skipIf(os.name == 'nt', 'Synthetic shell binaries need a POSIX host')
    def test_windows_smoke_runs_response_with_explicit_python_and_checks_act_tools(self):
        from scripts.native import smoke_mrtrix_candidate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = windows_source(root)
            archive = root / 'candidate.zip'
            module.make_archive(source, archive, 'Windows')
            record = root / 'record.json'
            record.write_text(json.dumps({
                'system': 'Windows', 'machine': 'x86_64', 'format': 'zip', 'sha256': module.sha256(archive),
                'commands': [*(f'bin/{name}.exe' for name in WINDOWS_TEST_PROGRAMS), 'bin/dwi2response'],
            }))
            report = root / 'smoke.json'
            env = {'WINDIR': str(root / 'Windows'), 'PATH': str(root / 'build-tools'),
                   'MSYSTEM': 'MINGW64', 'PYTHONHOME': str(root / 'build-tools')}
            # A host Python with a custom loader (such as PythonSlicer)
            # may need LD_LIBRARY_PATH, which this test intentionally strips.
            python = shutil.which('python3', path=os.defpath) or sys.executable
            with (
                patch('shared.native_runtimes.host_platform', return_value=('Windows', 'x86_64')),
                patch.dict(os.environ, env),
                patch.object(smoke_mrtrix_candidate.sys, 'executable', python),
            ):
                smoke_mrtrix_candidate.main(['--archive', str(archive), '--record', str(record),
                                              '--output', str(report)])
            checks = {item['binary']: item for item in json.loads(report.read_text())['checks']}
            self.assertIn('dhollander 3.0.8', checks['bin/dwi2response']['output'])
            for name in ('5ttcheck', '5tt2gmwmi', 'mtnormalise', 'mrtransform'):
                self.assertIn(f'{name} 3.0.8', checks[f'bin/{name}.exe']['version_output'])

    def test_windows_smoke_rejects_incomplete_manifest(self):
        from scripts.native import smoke_mrtrix_candidate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'candidate.zip'
            with zipfile.ZipFile(archive, 'w'):
                pass
            record = root / 'record.json'
            record.write_text(json.dumps({
                'system': 'Windows', 'machine': 'x86_64', 'format': 'zip', 'sha256': module.sha256(archive),
                'commands': [],
            }))
            with patch('shared.native_runtimes.host_platform', return_value=('Windows', 'x86_64')):
                with patch.dict(os.environ, {'WINDIR': str(root / 'Windows')}):
                    with self.assertRaisesRegex(ValueError, 'bin/dwi2response'):
                        smoke_mrtrix_candidate.main(['--archive', str(archive), '--record', str(record),
                                                      '--output', str(root / 'smoke.json')])

    @unittest.skipIf(os.name == 'nt', 'Synthetic build tools need a POSIX host')
    def test_windows_build_searches_upstream_bin_for_dlls_before_toolchain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'upstream source'
            tools = root / 'tools'
            tools.mkdir()
            (source / 'lib').mkdir(parents=True)
            fixtures = {
                tools / 'git': 'echo 464173c5294e021b2d508e557085d90c09d642fd',
                tools / 'python': 'printf "%s\\0" "$@"',
                source / 'configure': 'exit 0',
                source / 'build': 'mkdir bin && touch bin/mrtrix.dll',
            }
            for path, body in fixtures.items():
                path.write_text('#!/bin/sh\n' + body + '\n')
                path.chmod(0o755)
            environment = {**os.environ, 'MSYSTEM': 'MINGW64',
                           'PATH': str(tools) + os.pathsep + os.defpath}
            script = ROOT / 'scripts/native/build_mrtrix_windows.sh'
            completed = subprocess.run(['bash', str(script), str(source), str(root / 'dist')],
                                       cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = completed.stdout.split('\0')
            search_dirs = [value for flag, value in zip(arguments, arguments[1:]) if flag == '--dll-search']
            self.assertEqual(search_dirs[:3], [str(source / 'bin'), '/mingw64/bin', '/usr/bin'])

    def test_windows_recursively_collects_transitive_dlls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deps = root / 'mingw'
            target = root / 'out'
            deps.mkdir(); target.mkdir()
            exe = root / 'prog.exe'
            exe.write_bytes(b'MZfixture')
            (deps / 'libfoo.dll').write_bytes(b'first')
            (deps / 'libbar.dll').write_bytes(b'second')
            from subprocess import CompletedProcess
            def fake_objdump(cmd, **kwargs):
                lookup = {'prog.exe': 'libfoo.dll', 'libfoo.dll': 'libbar.dll', 'libbar.dll': 'KERNEL32.dll'}
                return CompletedProcess(cmd, 0, stdout='DLL Name: ' + lookup[Path(cmd[-1]).name] + '\n', stderr='')
            with patch.object(module.subprocess, 'run', side_effect=fake_objdump):
                copied = module.copy_windows_dll_closure([exe], target, [deps], objdump='objdump')
            self.assertEqual(copied, ['libbar.dll','libfoo.dll'])

if __name__ == '__main__':
    unittest.main()
