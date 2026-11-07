#!/usr/bin/env python3
"""Relocation test of native CLI candidate with build-system PATH stripped.

This smoke is deliberately *not* a scientific-validation test. Exit nonzero on
missing commands, loader failures or nonzero version checks. Runner's OS and
architecture must match the candidate record.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    from shared.native_runtimes import _extract, _sha256, host_platform
    record = json.loads(args.record.read_text())
    actual_system, actual_machine = host_platform()
    if (record['system'], record['machine']) != (actual_system, actual_machine):
        raise RuntimeError(f'Cannot smoke a {record["system"]}/{record["machine"]} runtime '
                           f'on {actual_system}/{actual_machine}')
    if _sha256(args.archive) != record['sha256']:
        raise ValueError('Candidate archive checksum mismatch')
    result = {'system': actual_system, 'machine': actual_machine, 'checks': [],
              'status': 'running', 'scope': 'loader/--version only (not scientific validation)'}
    with tempfile.TemporaryDirectory(prefix='native-relocation-') as folder:
        root = Path(folder)
        _extract(args.archive, root, record['format'])
        # Remove MSYS2/miniconda/compilers from PATH; Windows system DLLs are
        # found by the OS loader, and bundled DLLs sit beside each EXE.
        if actual_system == 'Windows':
            windows = Path(os.environ['WINDIR'])
            base_path = str(windows / 'System32') + os.pathsep + str(windows)
        else:
            base_path = '/usr/bin:/bin'
        for command in record['commands']:
            executable = (root / command).resolve()
            if not executable.is_file() or not executable.is_relative_to(root.resolve()):
                raise FileNotFoundError(command)
            environment = dict(os.environ)
            environment['PATH'] = str(executable.parent) + os.pathsep + base_path
            # Set a clean runtime lib env. Do not rely on the build runner's
            # installed Qt/MSYS2/Homebrew dependencies.
            for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',
                        'PYTHONHOME', 'PYTHONPATH', 'MSYSTEM', 'MSYS2_PATH_TYPE'):
                environment.pop(key, None)
            completed = subprocess.run([str(executable), '-version'], cwd=root, env=environment,
                                       capture_output=True, text=True, errors='replace', timeout=45)
            output = ((completed.stdout or '') + '\n' + (completed.stderr or '')).strip()
            result['checks'].append({'binary': command, 'exit_code': completed.returncode,
                                     'version_output': output[:4000]})
            if completed.returncode != 0:
                raise RuntimeError(f'Portable command failed: {command}: {output[-600:]}')
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'commands': len(result['checks'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
