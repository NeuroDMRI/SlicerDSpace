#!/usr/bin/env python3
"""Verify a built macOS surface runtime before it is packaged or published.

Checks, per binary in bin/ (and lib/ when present):
  * the Mach-O minimum macOS equals --expect-min-macos (read with shared/macho_min_os.py);
  * every linked library is a system library (/usr/lib or /System/Library);
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


def check_runtime(root, expected_min_macos, start=False, build_logs=None, reference=None,
                  min_os_reader=read_min_os, library_reader=read_libraries, starter=start_tool, commands=None):
    root = Path(root)
    expected = expected_min_macos.strip()
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
    return dict(runtime=str(root), expected_min_macos=expected, ok=not failures, failures=failures, tools=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--expect-min-macos', required=True)
    parser.add_argument('--start', action='store_true', help='Start every tool with --version.')
    parser.add_argument('--build-logs', type=Path, help='Folder with the build logs (*.log) to scan.')
    parser.add_argument('--reference', type=Path, help='Reference runtime for the file list and data bytes.')
    parser.add_argument('--json', type=Path, help='Write the report here.')
    args = parser.parse_args(argv)
    report = check_runtime(args.runtime, args.expect_min_macos, start=args.start,
                           build_logs=args.build_logs, reference=args.reference)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + '\n', encoding='utf-8')
    for row in report['tools']:
        print('%-32s minos=%-6s started=%-5s libs=%s' % (row['path'], row['min_macos'], row.get('started', '-'),
                                                       ','.join(row['libraries'])))
    for failure in report['failures']:
        print('FAIL: ' + failure)
    print('OK' if report['ok'] else 'FAILED (%d)' % len(report['failures']))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
