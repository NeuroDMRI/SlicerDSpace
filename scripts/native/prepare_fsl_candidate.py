#!/usr/bin/env python3
"""Stage a candidate FSL runtime bundle (eddy_cpu, topup, applytopup + b02b0.cnf).

This developer/CI packaging script creates a relocatable zip/tar.gz archive and
draft catalog record for `shared/native_runtimes.py` without requiring WSL,
MSYS2, or Conda on the end user's machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native.prepare_mrtrix_candidate import (
    _copy_with_mode,
    _is_macho,
    copy_darwin_dylib_closure,
    copy_linux_so_closure,
    copy_windows_dll_closure,
    make_archive,
    sha256,
)

VERSION = '2602.0'
FSL_LICENSE_URL = 'https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html'
FSL_COMMANDS = ('eddy_cpu', 'topup', 'applytopup')
REQUIRED_CONFIG = 'etc/flirtsch/b02b0.cnf'
OPTIONAL_CONFIGS = (
    'etc/flirtsch/b02b0_1.cnf',
    'etc/flirtsch/b02b0_2.cnf',
    'etc/flirtsch/b02b0_4.cnf',
    'etc/flirtsch/b02b0_7T.cnf',
)


def _find_fsl_command(source: Path, command: str, system: str) -> Path:
    name = command + ('.exe' if system == 'Windows' else '')
    for root in (source / 'bin', source / 'prefix' / 'bin', source):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Missing built FSL program {name} under {source}')


def _find_fsl_config(source: Path, relpath: str) -> Path | None:
    for root in (source, source / 'prefix', source / 'share' / 'fsl'):
        candidate = root / relpath
        if candidate.is_file():
            return candidate
    return None


def build(source: Path, output: Path, *, system: str, machine: str, dll_search: list[Path], objdump: str | None = None) -> dict:
    if system not in ('Windows', 'Darwin', 'Linux'):
        raise ValueError('Unsupported host system')
    if machine not in ('x86_64', 'arm64') or (system == 'Windows' and machine != 'x86_64'):
        raise ValueError('Unsupported architecture; do not claim a build for another CPU')
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    name = f'fsl-{VERSION}-{system.lower()}-{machine}'
    extension = '.zip' if system == 'Windows' else '.tar.gz'
    artifact = output / (name + extension)
    record_path = output / (name + '.draft-record.json')
    if artifact.exists() or record_path.exists():
        raise FileExistsError('Refusing to overwrite previous FSL build artifacts')

    topup_cnf = _find_fsl_config(source, REQUIRED_CONFIG)
    if topup_cnf is None:
        raise FileNotFoundError(f'Missing required FSL Topup configuration {REQUIRED_CONFIG}')

    with tempfile.TemporaryDirectory(prefix='fsl-stage-', dir=output) as temp:
        stage = Path(temp)
        license_src = next((p for p in (source / 'LICENCE.txt', source / 'LICENSE.txt', source / 'etc/fslversion') if p.is_file()), None)
        if license_src is not None and license_src.name.lower().startswith('licen'):
            shutil.copy2(license_src, stage / 'LICENSE.txt')
        else:
            (stage / 'LICENSE.txt').write_text(
                f'FSL {VERSION} runtime components (topup, applytopup, eddy_cpu).\n'
                f'Subject to the FSL Software License: {FSL_LICENSE_URL}\n',
                encoding='utf-8',
            )
        destination = stage / 'bin'
        destination.mkdir()
        command_paths = []
        staged_binaries = []
        for command in FSL_COMMANDS:
            binary = _find_fsl_command(source, command, system)
            target = destination / binary.name
            _copy_with_mode(binary, target)
            command_paths.append(f'bin/{target.name}')
            staged_binaries.append(target)

        _copy_with_mode(topup_cnf, stage / REQUIRED_CONFIG)
        for optional_rel in OPTIONAL_CONFIGS:
            found = _find_fsl_config(source, optional_rel)
            if found is not None:
                _copy_with_mode(found, stage / optional_rel)

        copied_dlls = []
        if system == 'Windows':
            search_dirs = list(dll_search)
            for candidate_dir in (source / 'bin', source / 'prefix' / 'bin'):
                if candidate_dir.is_dir() and candidate_dir not in search_dirs:
                    search_dirs.insert(0, candidate_dir)
            if not search_dirs:
                raise ValueError('Windows requires DLL search directories')
            if objdump and shutil.which(objdump):
                copied_dlls = copy_windows_dll_closure(staged_binaries, destination, search_dirs, objdump=objdump)
            else:
                for directory in search_dirs:
                    for dll in sorted(directory.glob('*.dll')):
                        target_dll = destination / dll.name
                        if not target_dll.exists():
                            shutil.copy2(dll, target_dll)
                            copied_dlls.append(dll.name)
                if not copied_dlls:
                    raise ValueError('No Windows runtime DLLs found for FSL executables')
        else:
            elf_binaries = [p for p in staged_binaries if p.read_bytes()[:4] == b'\x7fELF']
            macho_binaries = [p for p in staged_binaries if _is_macho(p)]
            if system == 'Linux' and elf_binaries:
                copied_dlls = copy_linux_so_closure(elf_binaries, stage / 'lib', [source / 'lib', *dll_search])
            elif system == 'Darwin' and macho_binaries:
                copied_dlls = copy_darwin_dylib_closure(macho_binaries, stage / 'lib', [source / 'lib', *dll_search])
            else:
                for lib_dir in (source / 'lib',):
                    if lib_dir.is_dir():
                        shutil.copytree(lib_dir, stage / 'lib', symlinks=False)

        receipt = {
            'kind': 'slicerdspace_native_candidate',
            'name': 'fsl',
            'version': VERSION,
            'system': system,
            'machine': machine,
            'commands': command_paths,
            'configs': [REQUIRED_CONFIG],
            'copied_dlls': sorted(copied_dlls),
            'scientifically_accepted': False,
            'end_user_install_requires_build_tools': False,
            'license_url': FSL_LICENSE_URL,
        }
        (stage / 'native-runtime.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        make_archive(stage, artifact, system)

    record = {
        'name': 'fsl',
        'version': VERSION,
        'system': system,
        'machine': machine,
        'format': 'zip' if system == 'Windows' else 'tar.gz',
        'sha256': sha256(artifact),
        'bytes': artifact.stat().st_size,
        'commands': command_paths,
        'license_url': FSL_LICENSE_URL,
        'published': False,
        'url': None,
        'scientifically_accepted': False,
        'artifact': artifact.name,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {
        'archive': str(artifact),
        'record': str(record_path),
        'sha256': record['sha256'],
        'files': len(command_paths),
        'dlls': len(copied_dlls),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--system', required=True, choices=('Windows', 'Darwin', 'Linux'))
    parser.add_argument('--machine', required=True, choices=('x86_64', 'arm64'))
    parser.add_argument('--dll-search', action='append', type=Path, default=[])
    parser.add_argument('--objdump', default='objdump')
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.output, system=args.system, machine=args.machine,
                           dll_search=args.dll_search, objdump=args.objdump), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
