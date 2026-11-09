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
# Keep the existing macOS/Linux command set. The Windows candidate adds ACT
# preparation, normalisation and tools used by FSL, Dhollander and SAF.
NATIVE_COMMANDS = ('mrconvert', 'dwi2fod', 'tckgen', 'tcksift2', 'tck2connectome', 'tckconvert')
WINDOWS_COMMANDS = (
    *NATIVE_COMMANDS, '5ttcheck', '5tt2gmwmi', 'mtnormalise', 'mrtransform', 'tckstats', 'tcksample',
    'amp2response', 'dwi2mask', 'dwi2tensor', 'dwiextract', 'maskfilter', 'mrcalc',
    'mrcat', 'mrinfo', 'mrmath', 'mrstats', 'mrthreshold', 'peaks2amp', 'sh2peaks',
    'tensor2metric',
)
WINDOWS_OPTIONAL_ACT_COMMANDS = ('labelconvert', 'mrgrid')
EXTRA_BUILT_COMMANDS = (
    'tckedit', 'tckresample', 'tckmap', 'mrcrop',
    'tckinfo', 'tcktransform', 'tcksift', 'connectome2tck',
    '5ttedit', '5tt2vis', 'mredit', 'mrdump', 'maskdump',
    'mesh2voxel', 'meshconvert', 'meshfilter', 'voxel2mesh',
    'label2mesh', 'label2colour', 'labelstats',
    'fod2fixel', 'fixel2voxel', 'fixel2peaks', 'fixel2sh',
    'sh2amp', 'sh2power', 'sh2response', 'shconv',
    'dwidenoise', 'mrdegibbs', 'mrfilter', 'mrregister',
    'transformcalc', 'transformcompose', 'transformconvert',
    'warpinit', 'warpconvert', 'warpinvert',
)
WINDOWS_RESPONSE_SCRIPT = 'bin/dwi2response'
WINDOWS_ACT_SCRIPT = 'bin/5ttgen'
WINDOWS_PATH_HELPER = 'lib/mrtrix3/path.py'
ACT_FREESURFER_HELPER = 'lib/mrtrix3/_5ttgen/freesurfer.py'
REPO_FREESURFER_LUT = Path(__file__).resolve().parents[2] / 'anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt'
SYSTEM_DLL = frozenset({
    'kernel32.dll','ntdll.dll','advapi32.dll','user32.dll','shell32.dll',
    'gdi32.dll','comdlg32.dll','ole32.dll','oleaut32.dll','ws2_32.dll',
    'secur32.dll','bcrypt.dll','crypt32.dll','rpcrt4.dll','shlwapi.dll',
    'version.dll','msvcrt.dll','ucrtbase.dll','combase.dll','imm32.dll',
    'setupapi.dll','winmm.dll','cfgmgr32.dll','psapi.dll','iphlpapi.dll',
})
SYSTEM_SO_PREFIXES = (
    'linux-vdso.so', 'ld-linux', 'libc.so.', 'libm.so.', 'libdl.so.',
    'libpthread.so.', 'librt.so.', 'libutil.so.', 'libresolv.so.',
    'libnsl.so.', 'libcrypt.so.',
)


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


def copy_linux_so_closure(programs: list[Path], lib_dest: Path, search_dirs: list[Path] | None = None) -> list[str]:
    """Bundle non-glibc shared libraries required by ELF binaries as regular files in lib/."""
    elf_programs = [p for p in programs if p.is_file() and p.read_bytes()[:4] == b'\x7fELF']
    if not elf_programs or not shutil.which('ldd'):
        return []
    lib_dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if search_dirs:
        existing = env.get('LD_LIBRARY_PATH', '')
        extra = os.pathsep.join(str(d) for d in search_dirs if d.is_dir())
        env['LD_LIBRARY_PATH'] = f'{extra}{os.pathsep}{existing}' if existing else extra
    copied: list[str] = []
    seen: set[str] = set()
    queue = list(elf_programs)
    while queue:
        binary = queue.pop()
        done = subprocess.run(['ldd', str(binary)], env=env, capture_output=True, text=True,
                              errors='replace', timeout=60, check=True)
        for line in done.stdout.splitlines():
            line = line.strip()
            if 'not found' in line:
                raise FileNotFoundError(f'Unresolved Linux shared library for {binary.name}: {line}')
            match = re.match(r'^(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)$', line)
            if not match:
                continue
            soname, resolved_raw = match.group(1), match.group(2)
            if soname in seen or soname.startswith(SYSTEM_SO_PREFIXES):
                continue
            seen.add(soname)
            resolved = Path(resolved_raw).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f'Missing resolved shared library {resolved_raw} for {soname}')
            target = lib_dest / soname
            if not target.exists():
                shutil.copy2(resolved, target)
                target.chmod(0o755)
                queue.append(target)
                copied.append(soname)
    return sorted(copied)


def _copy_with_mode(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if os.name != 'nt' and target.suffix != '.exe':
        target.chmod(target.stat().st_mode | 0o111)


def _patch_windows_path_helper(stage: Path) -> dict[str, str]:
    """Match the pinned path helper to the upstream Windows command tokenizer."""
    path = stage / WINDOWS_PATH_HELPER
    upstream = path.read_bytes()
    newline = b'\r\n' if b'\r\n' in upstream else b'\n'
    anchor = (
        b'try:\n'
        b'  from shlex import quote\n'
        b'except ImportError:\n'
        b'  from pipes import quote #pylint: disable=deprecated-module\n'
    ).replace(b'\n', newline)
    if upstream.count(anchor) != 1:
        raise ValueError(f'Cannot patch {WINDOWS_PATH_HELPER}: unexpected upstream quote implementation')
    replacement = anchor.replace(b'import quote', b'import quote as _posix_quote')
    replacement += (
        b'\n'
        b'# SlicerDSpace: native Windows run.command uses shlex in non-POSIX mode.\n'
        b'# It strips double quotes from path tokens before launching an argv list.\n'
        b'def quote(value):\n'
        b"  if os.sep == '/' or not os.path.isabs(value):\n"
        b'    return _posix_quote(value)\n'
        b'  if any(character == \'"\' or ord(character) < 32 for character in value):\n'
        b"    raise ValueError('Invalid character in Windows command path')\n"
        b'  return \'"\' + value + \'"\'\n'
        b'\n'
    ).replace(b'\n', newline)
    patched = upstream.replace(anchor, replacement, 1)
    path.write_bytes(patched)
    return {
        'file': WINDOWS_PATH_HELPER,
        'upstream_commit': '464173c5294e021b2d508e557085d90c09d642fd',
        'change': 'Use double quotes for absolute paths parsed by native Windows run.command',
        'original_sha256': hashlib.sha256(upstream).hexdigest(),
        'patched_sha256': hashlib.sha256(patched).hexdigest(),
    }


def _patch_act_freesurfer_helper(stage: Path) -> None:
    """Allow 5ttgen freesurfer to use bundled FreeSurferColorLUT.txt and quote LUT paths on Windows."""
    helper = stage / ACT_FREESURFER_HELPER
    lut_dir = stage / 'share/mrtrix3/_5ttgen'
    if (lut_dir.is_dir() or helper.is_file()) and REPO_FREESURFER_LUT.is_file():
        lut_dir.mkdir(parents=True, exist_ok=True)
        if not (lut_dir / 'FreeSurferColorLUT.txt').is_file():
            shutil.copy2(REPO_FREESURFER_LUT, lut_dir / 'FreeSurferColorLUT.txt')
    if not helper.is_file():
        return
    text = helper.read_text(encoding='utf-8')
    old_env = (
        "    freesurfer_home = os.environ.get('FREESURFER_HOME', '')\n"
        "    if not freesurfer_home:\n"
    )
    new_env = (
        "    freesurfer_home = os.environ.get('FREESURFER_HOME', '')\n"
        "    if not freesurfer_home:\n"
        "      bundled_dir = os.path.join(path.shared_data_path(), path.script_subdir_name())\n"
        "      if os.path.isfile(os.path.join(bundled_dir, 'FreeSurferColorLUT.txt')):\n"
        "        freesurfer_home = bundled_dir\n"
        "    if not freesurfer_home:\n"
    )
    if old_env in text and 'bundled_dir' not in text:
        text = text.replace(old_env, new_env, 1)
    old_cmd = "run.command('labelconvert input.mif ' + lut_input_path + ' ' + lut_output_path + ' indices.mif')"
    new_cmd = "run.command('labelconvert input.mif ' + path.quote(lut_input_path) + ' ' + path.quote(lut_output_path) + ' indices.mif')"
    if old_cmd in text:
        text = text.replace(old_cmd, new_cmd, 1)
    helper.write_text(text, encoding='utf-8')


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
    if system == 'Windows' and not dll_search:
        raise ValueError('Windows requires MSYS2 MinGW DLL search directories; never publish unbundled executables')
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
        commands = WINDOWS_COMMANDS if system == 'Windows' else NATIVE_COMMANDS
        for command in commands:
            binary = _find_command(source, command, system)
            target = destination / binary.name
            _copy_with_mode(binary, target)
            command_paths.append(f'bin/{target.name}')
        copied_dlls = []
        patches = []
        if system == 'Windows':
            for relpath in (WINDOWS_RESPONSE_SCRIPT, 'lib/mrtrix3/__init__.py', 'lib/mrtrix3/_version.py',
                            'lib/mrtrix3/dwi2response/dhollander.py', WINDOWS_PATH_HELPER):
                if not (source / relpath).is_file():
                    raise FileNotFoundError(f'Missing built MRtrix3 response payload: {relpath}')
            _copy_with_mode(source / WINDOWS_RESPONSE_SCRIPT, stage / WINDOWS_RESPONSE_SCRIPT)
            has_act_script = (source / WINDOWS_ACT_SCRIPT).is_file()
            if has_act_script:
                for command in WINDOWS_OPTIONAL_ACT_COMMANDS:
                    binary = _find_command(source, command, system)
                    target = destination / binary.name
                    _copy_with_mode(binary, target)
                    command_paths.append(f'bin/{target.name}')
                _copy_with_mode(source / WINDOWS_ACT_SCRIPT, stage / WINDOWS_ACT_SCRIPT)
                if (source / 'share/mrtrix3').is_dir():
                    shutil.copytree(source / 'share/mrtrix3', stage / 'share/mrtrix3', symlinks=False)
            for extra_cmd in EXTRA_BUILT_COMMANDS:
                try:
                    binary = _find_command(source, extra_cmd, system)
                except FileNotFoundError:
                    continue
                target = destination / binary.name
                _copy_with_mode(binary, target)
                command_paths.append(f'bin/{target.name}')
            shutil.copytree(source / 'lib/mrtrix3', stage / 'lib/mrtrix3',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            patches.append(_patch_windows_path_helper(stage))
            _patch_act_freesurfer_helper(stage)
            copied_dlls = copy_windows_dll_closure([stage / rel for rel in command_paths], destination,
                                                   dll_search, objdump=objdump)
            command_paths.append(WINDOWS_RESPONSE_SCRIPT)
            if has_act_script:
                command_paths.append(WINDOWS_ACT_SCRIPT)
        else:
            for extra_cmd in (*WINDOWS_COMMANDS, *WINDOWS_OPTIONAL_ACT_COMMANDS, *EXTRA_BUILT_COMMANDS):
                if extra_cmd in commands:
                    continue
                try:
                    binary = _find_command(source, extra_cmd, system)
                except FileNotFoundError:
                    continue
                target = destination / binary.name
                _copy_with_mode(binary, target)
                command_paths.append(f'bin/{target.name}')
            elf_binaries = [stage / rel for rel in command_paths if (stage / rel).read_bytes()[:4] == b'\x7fELF']
            if system == 'Linux' and elf_binaries:
                copied_dlls = copy_linux_so_closure(elf_binaries, stage / 'lib', [source / 'lib'])
                if (source / 'lib/mrtrix3').is_dir():
                    shutil.copytree(source / 'lib/mrtrix3', stage / 'lib/mrtrix3',
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                if (source / 'share/mrtrix3').is_dir():
                    shutil.copytree(source / 'share/mrtrix3', stage / 'share/mrtrix3', symlinks=False)
                _patch_act_freesurfer_helper(stage)
                for script_rel in (WINDOWS_RESPONSE_SCRIPT, WINDOWS_ACT_SCRIPT):
                    if (source / script_rel).is_file():
                        _copy_with_mode(source / script_rel, stage / script_rel)
                        command_paths.append(script_rel)
            else:
                # Only bundle a platform's *own* library tree. Toolchain-provided
                # .dylib/.so portability must be verified on a clean test host.
                for library_root in (source / 'lib', source / 'share'):
                    if library_root.is_dir():
                        shutil.copytree(library_root, stage / library_root.name, symlinks=False)
        if system == 'Windows':
            limitations = [
                'The Windows response script requires a native Python interpreter.',
                'This candidate targets Dhollander response, CSD, normalisation and ACT with prepared priors.',
            ]
        else:
            limitations = [
                'This package only qualifies the listed C++ CLI programs.',
                'MRtrix Python scripts/ACT prior generation may require additional dependencies.',
            ]
        limitations.append('FSL Eddy/Topup and FreeSurfer are not bundled or approved.')
        receipt = {
            'kind': 'slicerdspace_native_candidate', 'version': VERSION,
            'system': system, 'machine': machine, 'commands': command_paths,
            'copied_dlls': copied_dlls, 'scientifically_accepted': False,
            'end_user_install_requires_build_tools': False,
            'limitations': limitations,
        }
        if patches:
            receipt['patches'] = patches
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
