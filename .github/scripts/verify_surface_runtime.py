#!/usr/bin/env python3
"""Verify a built surface runtime before it is packaged or published.

Checks, per binary in bin/ (and lib/ when present):
  * macOS (--expect-min-macos): the Mach-O minimum macOS equals the value (read with shared/macho_min_os.py),
    and every linked library is a system library (/usr/lib or /System/Library);
  * Linux (--max-glibc): no GLIBC_ symbol version above the value, no GLIBCXX_/CXXABI_ symbol version
    (libstdc++ linked statically), and only C-library objects are NEEDED;
  * the tool list equals the seven surface commands of the build recipe (the shipped runtime's list);
  * with --start, each tool starts with --version in a clean environment.
Optional: --build-logs fails on a linker note that an object was built for a newer macOS;
--reference compares every non-binary file (bytes) and the file list with a reference runtime.

Writes a JSON report (--json) and exits 1 on any failure.
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
import sys

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / 'anatomy/surface/ddsurfer_backend'
SYSTEM_PREFIXES = ('/usr/lib/', '/System/Library/')
# glibc objects. libcrypt.so.1 (crypt() in the activation check) is glibc's on RHEL 8 and libcrypt1 on
# Debian/Ubuntu; distributions with libcrypt.so.2 only (recent Fedora, Arch) need libxcrypt-compat.
LINUX_SYSTEM_LIBRARIES = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1', 'libcrypt.so.1',
                          'ld-linux-x86-64.so.2', 'ld-linux-aarch64.so.1'}
NEWER_OBJECT = re.compile(r'built for newer .{0,12}macOS.{0,12} version', re.IGNORECASE)


def recipe_commands():
    """The command names of the pinned build recipe (postprocessing/core/config.py COMMANDS)."""
    text = (BACKEND / 'postprocessing/core/config.py').read_text(encoding='utf-8')
    match = re.search(r'COMMANDS = \(([^)]*)\)', text)
    return tuple(re.findall(r"'([^']+)'", match.group(1)))


def read_min_os(path):
    sys.path.insert(0, str(REPO))
    try:
        from shared.macho_min_os import macho_min_macos
    finally:
        sys.path.remove(str(REPO))
    return macho_min_macos(path, 'arm64' if os.uname().machine == 'arm64' else None)


def parse_otool_libraries(text):
    """Library paths from ``otool -L`` output (the first line names the file itself)."""
    libraries = []
    for line in text.splitlines()[1:]:
        line = line.strip()
        if line:
            libraries.append(line.split(' (compatibility', 1)[0].strip())
    return libraries


def read_libraries(path):
    tool = shutil.which('otool')
    if tool is None:
        raise FileNotFoundError('otool is needed to list linked libraries')
    result = subprocess.run([tool, '-L', str(path)], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('otool -L failed for %s: %s' % (path, result.stderr.strip()))
    return parse_otool_libraries(result.stdout)


def parse_symbol_versions(text):
    """Symbol version names (GLIBC_2.17, GLIBCXX_3.4, ...) from ``objdump -T`` output."""
    return set(re.findall(r'\b((?:GLIBC|GLIBCXX|CXXABI|GCC)_[0-9][0-9.]*)\b', text))


def parse_needed(text):
    """NEEDED shared objects from ``readelf -d`` output."""
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', text)


def read_elf(path):
    versions = subprocess.run(['objdump', '-T', str(path)], capture_output=True, text=True, timeout=60)
    dynamic = subprocess.run(['readelf', '-d', str(path)], capture_output=True, text=True, timeout=60)
    if versions.returncode or dynamic.returncode:
        raise RuntimeError('objdump/readelf failed for %s: %s%s' % (path, versions.stderr, dynamic.stderr))
    return parse_symbol_versions(versions.stdout), parse_needed(dynamic.stdout)


def version_key(text):
    return tuple(int(part) for part in text.split('.') if part.isdigit())


def start_tool(path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DYLD_', 'LD_')) and key not in ('FREESURFER_HOME', 'SUBJECTS_DIR')}
    try:
        result = subprocess.run([str(path), '--version'], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return result.returncode, (result.stdout + result.stderr).strip()


def format_version(version):
    if not version:
        return ''
    major, minor, patch = (tuple(version) + (0, 0, 0))[:3]
    return '%d.%d.%d' % (major, minor, patch) if patch else '%d.%d' % (major, minor)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binaries(root):
    found = []
    for sub in ('bin', 'lib'):
        top = root / sub
        if top.is_dir():
            found += sorted(path for path in top.rglob('*') if path.is_file() and not path.is_symlink())
    return found


def compare_reference(root, reference):
    failures = []
    ours = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    theirs = {path.relative_to(reference).as_posix() for path in reference.rglob('*') if path.is_file()}
    for name in sorted(ours ^ theirs):
        if name != 'runtime.json':
            failures.append('file list differs from the reference: %s (%s)' % (name, 'only here' if name in ours else 'missing'))
    for name in sorted(ours & theirs):
        if name.startswith(('bin/', 'lib/')) or name == 'runtime.json':
            continue
        if sha256(root / name) != sha256(reference / name):
            failures.append('data file differs from the reference: ' + name)
    return failures


def check_elf(relative, row, versions, needed, max_glibc):
    failures = []
    glibc = sorted((name.split('_', 1)[1] for name in versions if name.startswith('GLIBC_')), key=version_key)
    row['max_glibc'] = glibc[-1] if glibc else ''
    row['libraries'] = list(needed)
    if glibc and version_key(glibc[-1]) > version_key(max_glibc):
        failures.append('%s: needs GLIBC_%s, limit %s' % (relative, glibc[-1], max_glibc))
    cxx = sorted(name for name in versions if name.startswith(('GLIBCXX_', 'CXXABI_')))
    if cxx:
        failures.append('%s: needs host libstdc++ versions %s' % (relative, ', '.join(cxx)))
    outside = [name for name in needed if name not in LINUX_SYSTEM_LIBRARIES]
    if outside:
        failures.append('%s: needs non-C-library objects: %s' % (relative, ', '.join(outside)))
    return failures


def check_runtime(root, expected_min_macos=None, start=False, build_logs=None, reference=None,
                  min_os_reader=read_min_os, library_reader=read_libraries, starter=start_tool, commands=None,
                  max_glibc=None, elf_reader=read_elf):
    root = Path(root)
    if bool(expected_min_macos) == bool(max_glibc):
        raise ValueError('Give exactly one of expected_min_macos (macOS) or max_glibc (Linux)')
    expected = (expected_min_macos or '').strip()
    failures, rows = [], []
    commands = tuple(commands or recipe_commands())
    present = sorted(path.name for path in (root / 'bin').iterdir() if path.is_file()) if (root / 'bin').is_dir() else []
    for name in sorted(set(commands) - set(present)):
        failures.append('tool missing from bin/: ' + name)
    for name in sorted(set(present) - set(commands)):
        failures.append('unexpected tool in bin/: ' + name)
    for path in binaries(root):
        relative = path.relative_to(root).as_posix()
        row = dict(tool=path.name, path=relative)
        if max_glibc:
            try:
                versions, needed = elf_reader(path)
                failures += check_elf(relative, row, versions, needed, max_glibc)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                row['libraries'] = []
                failures.append('%s: cannot read ELF symbol versions: %s' % (relative, error))
        else:
            version = format_version(min_os_reader(path))
            row['min_macos'] = version
            if version != expected:
                failures.append('%s: minimum macOS %s, expected %s' % (relative, version or 'unknown', expected))
            try:
                libraries = list(library_reader(path))
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                libraries = []
                failures.append('%s: cannot list libraries: %s' % (relative, error))
            row['libraries'] = libraries
            outside = [item for item in libraries if not item.startswith(SYSTEM_PREFIXES)]
            if outside:
                failures.append('%s: links non-system libraries: %s' % (relative, ', '.join(outside)))
        if start and relative.startswith('bin/'):
            code, output = starter(path)
            row['started'] = code == 0 and bool(output)
            row['version_output'] = output.splitlines()[0] if output else ''
            if not row['started']:
                failures.append('%s: --version failed (exit %s): %s' % (relative, code, output[:300]))
        rows.append(row)
    if build_logs is not None:
        for log in sorted(Path(build_logs).rglob('*.log')):
            for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
                if NEWER_OBJECT.search(line):
                    failures.append('%s: %s' % (log.name, line.strip()[:300]))
    if reference is not None:
        failures += compare_reference(root, Path(reference))
    return dict(runtime=str(root), expected_min_macos=expected, max_glibc=max_glibc or '', ok=not failures,
                failures=failures, tools=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--expect-min-macos', help='macOS runtime: required minimum macOS of every binary.')
    target.add_argument('--max-glibc', help='Linux runtime: highest allowed GLIBC_ symbol version.')
    parser.add_argument('--start', action='store_true', help='Start every tool with --version.')
    parser.add_argument('--build-logs', type=Path, help='Folder with the build logs (*.log) to scan.')
    parser.add_argument('--reference', type=Path, help='Reference runtime for the file list and data bytes.')
    parser.add_argument('--json', type=Path, help='Write the report here.')
    args = parser.parse_args(argv)
    report = check_runtime(args.runtime, args.expect_min_macos, start=args.start, max_glibc=args.max_glibc,
                           build_logs=args.build_logs, reference=args.reference)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + '\n', encoding='utf-8')
    for row in report['tools']:
        floor = 'glibc=' + row['max_glibc'] if 'max_glibc' in row else 'minos=' + row['min_macos']
        print('%-32s %-12s started=%-5s libs=%s' % (row['path'], floor, row.get('started', '-'), ','.join(row['libraries'])))
    for failure in report['failures']:
        print('FAIL: ' + failure)
    print('OK' if report['ok'] else 'FAILED (%d)' % len(report['failures']))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
