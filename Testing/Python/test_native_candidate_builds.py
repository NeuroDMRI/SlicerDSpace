"""No-network checks for release-candidate packaging scaffolding."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('candidate_builder', ROOT / 'scripts/native/prepare_mrtrix_candidate.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


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
