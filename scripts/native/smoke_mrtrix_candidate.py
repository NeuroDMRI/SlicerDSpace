#!/usr/bin/env python3
"""Relocation test of native CLI candidate with build-system PATH stripped.

This smoke is deliberately *not* a scientific-validation test. Exit nonzero on
missing commands, loader failures or a failed Dhollander script startup.
The runner's OS and architecture must match the candidate record.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    from scripts.native.prepare_mrtrix_candidate import WINDOWS_COMMANDS, WINDOWS_RESPONSE_SCRIPT

    record = json.loads(args.record.read_text())
    actual_system, actual_machine = host_platform()
    if (record['system'], record['machine']) != (actual_system, actual_machine):
        raise RuntimeError(f'Cannot smoke a {record["system"]}/{record["machine"]} runtime '
                           f'on {actual_system}/{actual_machine}')
    if _sha256(args.archive) != record['sha256']:
        raise ValueError('Candidate archive checksum mismatch')
    is_windows = actual_system == 'Windows'
    if is_windows:
        required = {WINDOWS_RESPONSE_SCRIPT, *(f'bin/{name}.exe' for name in WINDOWS_COMMANDS)}
        missing = required.difference(record['commands'])
        if missing:
            raise ValueError(f'Incomplete Windows candidate command list: {", ".join(sorted(missing))}')
    scope = 'loader/-version and Dhollander script startup' if is_windows else 'loader/-version only'
    result = {'system': actual_system, 'machine': actual_machine, 'checks': [],
              'status': 'running', 'scope': scope + ' (not scientific validation)'}
    prefix = 'native relocation ' if is_windows else 'native-relocation-'
    with tempfile.TemporaryDirectory(prefix=prefix) as folder:
        root = Path(folder).resolve()
        _extract(args.archive, root, record['format'])
        if is_windows:
            windows = Path(os.environ['WINDIR'])
            base_path = os.pathsep.join((str(windows / 'System32'), str(windows)))
            # Use only OS paths and the extracted payload. Isolate user MRtrix
            # config and scratch files from the runner's profile.
            environment = {
                'WINDIR': str(windows), 'SYSTEMROOT': str(windows),
                'HOME': str(root), 'USERPROFILE': str(root), 'TEMP': str(root), 'TMP': str(root),
                'MRTRIX_CONFIGFILE': str(root / 'mrtrix.conf'),
            }
        else:
            base_path = '/usr/bin:/bin'
            environment = dict(os.environ)
            for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',
                        'PYTHONHOME', 'PYTHONPATH', 'MSYSTEM', 'MSYS2_PATH_TYPE'):
                environment.pop(key, None)
        for command in record['commands']:
            executable = (root / command).resolve()
            if not executable.is_file() or not executable.is_relative_to(root):
                raise FileNotFoundError(command)
            child_env = {**environment, 'PATH': str(executable.parent) + os.pathsep + base_path}
            invocation = [str(executable), '-version']
            output_key = 'version_output'
            if is_windows and command == WINDOWS_RESPONSE_SCRIPT:
                child_env['PYTHONPATH'] = str(root / 'lib')
                # Ask the pinned upstream script to print the specific algorithm's
                # usage. This loads its Python modules without a help pager.
                invocation = [sys.executable, '-B', '-S', str(executable), 'dhollander', '__print_full_usage__']
                output_key = 'output'
            completed = subprocess.run(invocation, cwd=root, env=child_env,
                                       capture_output=True, text=True, errors='replace', timeout=45)
            output = ((completed.stdout or '') + '\n' + (completed.stderr or '')).strip()
            result['checks'].append({'binary': command, 'exit_code': completed.returncode,
                                     output_key: output[:4000]})
            if completed.returncode != 0:
                raise RuntimeError(f'Portable command failed: {command}: {output[-600:]}')
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'commands': len(result['checks'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
