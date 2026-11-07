#!/usr/bin/env python3
"""Stage a *candidate*, not an approved MRtrix3 runtime, on a build runner.

This script is intentionally a developer/CI packaging tool; it never installs
compilers or MSYS2 on an end user's computer. It produces a zip/tar.gz archive
and a draft catalog record (published=false). Scientific certification and
redistribution checks are separate release gates.

Examples:
    python scripts/native/prepare_mrtrix_candidate.py --source mrtrix3 \
        --system Windows --machine x86_64 --output dist --dll-search C:/msys64/mingw64/bin
    python scripts/native/prepare_mrtrix_candidate.py --source mrtrix3 \
        --system Darwin --machine arm64 --output dist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

VERSION = '3.0.8'
# These are C++ CLI binaries, not helper scripts that implicitly require FSL.
# 5ttgen, dwi2response and dwifslpreproc are *not* certified by this step.
NATIVE_COMMANDS = ('mrconvert', 'dwi2fod', 'tckgen', 'tcksift2', 'tck2connectome', 'tckconvert')
SYSTEM_DLL = frozenset({
    'kernel32.dll','ntdll.dll','advapi32.dll','user32.dll','shell32.dll',
    'gdi32.dll','comdlg32.dll','ole32.dll','oleaut32.dll','ws2_32.dll',
    'secur32.dll','bcrypt.dll','crypt32.dll','rpcrt4.dll','shlwapi.dll',
    'version.dll','msvcrt.dll','ucrtbase.dll','combase.dll','imm32.dll',
    'setupapi.dll','winmm.dll','cfgmgr32.dll','psapi.dll','iphlpapi.dll',
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _find_command(source: Path, command: str, system: str) -> Path:
    name = command + ('.exe' if system == 'Windows' else '')
    for root in (source / 'bin', source / 'release' / 'bin', source):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Missing built MRtrix3 program {name}. A source checkout is not a build.')


def _dll_dependencies(binary: Path, *, objdump: str) -> set[str]:
    done = subprocess.run([objdump, '-p', str(binary)], capture_output=True, text=True,
                          errors='replace', timeout=90, check=True)
    return set(re.findall(r'^\s*DLL Name:\s*(\S+)', done.stdout, flags=re.I | re.M))


def copy_windows_dll_closure(programs: list[Path], destination: Path, search_dirs: list[Path], *, objdump: str) -> list[str]:
    """Bundle transitive MinGW DLLs. Unresolved non-system DLLs fail closed."""
    indexed: dict[str, Path] = {}
    for directory in search_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f'MSYS2 DLL directory not found: {directory}')
        for file in directory.glob('*.dll'):
            indexed.setdefault(file.name.lower(), file)
    seen = set()
    queue = list(programs)
    copied = []
    while queue:
        binary = queue.pop()
        for name in _dll_dependencies(binary, objdump=objdump):
            key = name.lower()
            if key in SYSTEM_DLL or key.startswith(('api-ms-win-', 'ext-ms-win-')) or key in seen:
                continue
            seen.add(key)
            source = indexed.get(key)
            if source is None:
                raise FileNotFoundError(f'Unresolved portable Windows DLL {name} required by {binary.name}')
            target = destination / source.name
            if not target.exists():
                shutil.copy2(source, target)
                queue.append(target)
                copied.append(target.name)
    return sorted(copied)


def _copy_with_mode(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if os.name != 'nt' and target.suffix != '.exe':
        target.chmod(target.stat().st_mode | 0o111)


def make_archive(folder: Path, archive: Path, system: str) -> None:
    if system == 'Windows':
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as out:
            for entry in sorted(folder.rglob('*')):
                if entry.is_file():
                    out.write(entry, entry.relative_to(folder).as_posix())
    else:
        with tarfile.open(archive, 'w:gz', format=tarfile.PAX_FORMAT) as out:
            for entry in sorted(folder.rglob('*')):
                if entry.is_file():
                    out.add(entry, arcname=entry.relative_to(folder).as_posix(), recursive=False)


def build(source: Path, output: Path, *, system: str, machine: str, dll_search: list[Path], objdump: str) -> dict:
    if system not in ('Windows', 'Darwin', 'Linux'):
        raise ValueError('Unsupported host system')
    if machine not in ('x86_64', 'arm64') or (system == 'Windows' and machine != 'x86_64'):
        raise ValueError('Unsupported architecture; do not claim a build for another CPU')
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    name = f'mrtrix-{VERSION}-{system.lower()}-{machine}'
    extension = '.zip' if system == 'Windows' else '.tar.gz'
    artifact = output / (name + extension)
    record_path = output / (name + '.draft-record.json')
    if artifact.exists() or record_path.exists():
        raise FileExistsError('Refusing to overwrite previous build artifacts')
    license_file = source / 'LICENCE.txt'
    if not license_file.is_file():
        raise FileNotFoundError('MRtrix3 LICENCE.txt is required in the candidate archive')
    with tempfile.TemporaryDirectory(prefix='mrtrix-stage-', dir=output) as temp:
        stage = Path(temp)
        shutil.copy2(license_file, stage / 'LICENCE.txt')
        destination = stage / 'bin'
        destination.mkdir()
        command_paths = []
        for command in NATIVE_COMMANDS:
            binary = _find_command(source, command, system)
            target = destination / binary.name
            _copy_with_mode(binary, target)
            command_paths.append(f'bin/{target.name}')
        copied_dlls = []
        if system == 'Windows':
            if not dll_search:
                raise ValueError('Windows requires MSYS2 MinGW DLL search directories; never publish unbundled executables')
            copied_dlls = copy_windows_dll_closure([stage / rel for rel in command_paths], destination,
                                                   dll_search, objdump=objdump)
        else:
            # Only bundle a platform's *own* library tree. Toolchain-provided
            # .dylib/.so portability must be verified on a clean test host.
            for library_root in (source / 'lib', source / 'share'):
                if library_root.is_dir():
                    shutil.copytree(library_root, stage / library_root.name, symlinks=False)
        receipt = {
            'kind': 'slicerdspace_native_candidate', 'version': VERSION,
            'system': system, 'machine': machine, 'commands': command_paths,
            'copied_dlls': copied_dlls, 'scientifically_accepted': False,
            'end_user_install_requires_build_tools': False,
            'limitations': [
                'This package only qualifies the listed C++ CLI programs.',
                'MRtrix Python scripts/ACT prior generation may require additional dependencies.',
                'FSL Eddy/Topup and FreeSurfer are not bundled or approved.',
            ],
        }
        (stage / 'native-runtime.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        make_archive(stage, artifact, system)
    record = {
        'name': 'mrtrix', 'version': VERSION, 'system': system, 'machine': machine,
        'format': 'zip' if system == 'Windows' else 'tar.gz',
        'sha256': sha256(artifact), 'bytes': artifact.stat().st_size,
        'commands': command_paths, 'license_url': 'https://github.com/MRtrix3/mrtrix3/blob/master/LICENCE.txt',
        'published': False, 'url': None, 'scientifically_accepted': False,
        'artifact': artifact.name,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return {'archive': str(artifact), 'record': str(record_path), 'sha256': record['sha256'],
            'files': len(command_paths), 'dlls': len(copied_dlls)}


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
