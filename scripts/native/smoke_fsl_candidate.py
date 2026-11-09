#!/usr/bin/env python3
"""Relocation smoke test of a native FSL candidate archive with build PATH stripped."""
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    from shared.native_runtimes import _extract, _sha256, host_platform
    from scripts.native.prepare_fsl_candidate import FSL_COMMANDS, REQUIRED_CONFIG

    record = json.loads(args.record.read_text(encoding='utf-8'))
    actual_system, actual_machine = host_platform()
    if (record['system'], record['machine']) != (actual_system, actual_machine):
        raise RuntimeError(f'Cannot smoke a {record["system"]}/{record["machine"]} runtime '
                           f'on {actual_system}/{actual_machine}')
    if _sha256(args.archive) != record['sha256']:
        raise ValueError('Candidate archive checksum mismatch')
    is_windows = actual_system == 'Windows'
    suffix = '.exe' if is_windows else ''
    required = {f'bin/{name}{suffix}' for name in FSL_COMMANDS}
    missing = required.difference(record['commands'])
    if missing:
        raise ValueError(f'Incomplete FSL candidate command list: {", ".join(sorted(missing))}')

    result = {
        'system': actual_system,
        'machine': actual_machine,
        'checks': [],
        'status': 'running',
        'scope': 'loader/--help and b02b0.cnf inspection',
    }
    prefix = 'fsl relocation ' if is_windows else 'fsl-relocation-'
    with tempfile.TemporaryDirectory(prefix=prefix) as folder:
        root = Path(folder).resolve()
        _extract(args.archive, root, record['format'])
        config_path = root / REQUIRED_CONFIG
        if not config_path.is_file() or not config_path.stat().st_size:
            raise FileNotFoundError(f'Missing required FSL config: {REQUIRED_CONFIG}')
        if is_windows:
            windows = Path(os.environ['WINDIR'])
            base_path = os.pathsep.join((str(windows / 'System32'), str(windows)))
            environment = {
                'WINDIR': str(windows),
                'SYSTEMROOT': str(windows),
                'HOME': str(root),
                'USERPROFILE': str(root),
                'TEMP': str(root),
                'TMP': str(root),
                'FSLDIR': str(root),
                'FSLOUTPUTTYPE': 'NIFTI_GZ',
            }
        else:
            base_path = '/usr/bin:/bin'
            environment = dict(os.environ)
            for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',
                        'PYTHONHOME', 'PYTHONPATH', 'MSYSTEM', 'MSYS2_PATH_TYPE'):
                environment.pop(key, None)
            environment['FSLDIR'] = str(root)
            environment['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
            if (root / 'lib').is_dir():
                environment['LD_LIBRARY_PATH'] = str(root / 'lib')
        for command in record['commands']:
            executable = (root / command).resolve()
            if not executable.is_file() or not executable.is_relative_to(root):
                raise FileNotFoundError(command)
            child_env = {**environment, 'PATH': str(executable.parent) + os.pathsep + base_path}
            completed = subprocess.run([str(executable), '--help'], cwd=root, env=child_env,
                                       capture_output=True, text=True, errors='replace', timeout=45)
            output = ((completed.stdout or '') + '\n' + (completed.stderr or '')).strip()
            # FSL utilities print usage and return 0 or 1 on --help; loader errors return negative or >1 or empty output.
            if completed.returncode not in (0, 1) or 'Part of FSL' not in output and '--imain' not in output:
                raise RuntimeError(f'Portable FSL command failed ({completed.returncode}): {command}: {output[-600:]}')
            result['checks'].append({
                'binary': command,
                'exit_code': completed.returncode,
                'help_output': output[:2000],
            })
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'commands': len(result['checks'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
