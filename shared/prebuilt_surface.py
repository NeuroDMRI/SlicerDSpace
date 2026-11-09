"""Install hash-qualified native components without a compiler or full FreeSurfer."""

import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from shared.backend_sources import file_hash, source_lock

CATALOG = Path(__file__).resolve().parents[1] / 'models/runtime_bundles.json'


def bundle_for_platform(system=None, machine=None):
    payload = json.loads(CATALOG.read_text())
    selected_machine = (machine or platform.machine()).lower()
    machine = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(selected_machine, selected_machine)
    record = next((record for record in payload['surface'] if record['system'] == (system or platform.system())
                 and record['machine'] == machine), None)
    if record and record.get('min_macos') and platform.system() == 'Darwin':
        version = platform.mac_ver()[0]
        if not version or tuple(map(int, version.split('.')[:2])) < tuple(map(int, record['min_macos'].split('.')[:2])):
            return None
    return record


def prepare(*, prefix=None, asset_root=None, execute=False, accept_terms=False, offline=False):
    from shared.surface_runtime import bundled_surface_runtime, default_surface_runtime
    from shared.model_assets import configured_model_assets_root
    from anatomy.surface.backend_bridge import backend_environment, python_executable
    from anatomy.surface.ddsurfer_backend.postprocessing.core.config import recipe_digest
    record = bundle_for_platform()
    if record is None:
        raise ValueError('No qualified prebuilt surface bundle for this platform; use a supported configured runtime')
    if record['recipe_sha256'] != recipe_digest():
        raise ValueError('Prebuilt bundle catalog does not match the current source recipe')
    if prefix is None:
        default_pref = default_surface_runtime()
        bundled_pref = bundled_surface_runtime()
        prefix = (bundled_pref if bundled_pref is not None and not default_pref.exists() else default_pref)
    prefix = Path(prefix).expanduser().absolute()
    root = configured_model_assets_root(asset_root)
    relative = PurePosixPath(record['asset'])
    if relative.is_absolute() or '..' in relative.parts or '\\' in record['asset'] or ':' in record['asset']:
        raise ValueError('Invalid prebuilt bundle asset path')
    archive = root.joinpath(*relative.parts)
    result = dict(kind='slicerdsurfer_runtime_setup', status='planned', method='prebuilt',
                  prefix=str(prefix), archive=str(archive), download_bytes=record['bytes'],
                  external_dependencies={}, changes_slicer_python=False)
    if not execute:
        return result
    if not accept_terms:
        raise PermissionError('Review native-component source/data terms before installation')
    script = Path(__file__).resolve().parents[1] / 'scripts/probe_surface_runtime.py'

    def validate(path):
        probe = subprocess.run([python_executable(sys.executable), str(script), '--backend', 'native', '--home', str(path)],
                               env=backend_environment(), capture_output=True, text=True, check=True, timeout=180)
        checked = json.loads(probe.stdout.strip().splitlines()[-1])
        if not checked.get('components_available'):
            raise ValueError(checked.get('error', 'Prebuilt runtime functionality check failed'))
        manifest = json.loads((path / 'runtime.json').read_text())
        if manifest.get('recipe_sha256') != record['recipe_sha256'] or manifest.get('activation_key_included') is not False:
            raise ValueError('Prebuilt runtime recipe or activation policy differs')
        checked_commands = []
        for relative in sorted(manifest['files']):
            if not relative.startswith('bin/') or relative.lower().endswith('.dll'):
                continue
            command = subprocess.run([str(path / relative), '--version'], env=backend_environment(),
                                     capture_output=True, text=True, timeout=30)
            if command.returncode or not (command.stdout or command.stderr).strip():
                raise RuntimeError('Prebuilt native command could not start: ' + relative)
            checked_commands.append(relative + ':version')
        return checked_commands

    if prefix.is_symlink() or archive.is_symlink():
        raise ValueError('Prebuilt runtime paths must not be symbolic links')
    with source_lock(prefix.parent):
        if prefix.exists():
            checks = validate(prefix)
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.is_file() or file_hash(archive) != record['sha256']:
                if offline or not record.get('published'):
                    raise FileNotFoundError('Verified prebuilt bundle is unavailable in the model directory; import its offline archive first')
                partial = archive.with_suffix(archive.suffix + '.part')
                if partial.is_symlink():
                    raise ValueError('Refusing linked runtime download')
                with urllib.request.urlopen(record['url'], timeout=60) as response, partial.open('wb') as output:
                    shutil.copyfileobj(response, output)
                if file_hash(partial) != record['sha256']:
                    raise ValueError('Prebuilt runtime archive checksum mismatch')
                partial.replace(archive)
            with tempfile.TemporaryDirectory(dir=prefix.parent, prefix='.native-install-') as temporary:
                staged = Path(temporary) / 'runtime'
                staged.mkdir()
                total = 0
                with tarfile.open(archive, 'r:gz') as bundle:
                    seen = set()
                    for member in bundle:
                        path = PurePosixPath(member.name)
                        if (path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name or
                                member.issym() or member.islnk() or member.isdev()):
                            raise ValueError('Unsafe prebuilt runtime archive member')
                        if not member.isfile():
                            continue
                        if member.name in seen:
                            raise ValueError('Duplicate prebuilt runtime archive member')
                        seen.add(member.name)
                        total += member.size
                        if total > record['unpacked_bytes'] + 1024 * 1024 or len(seen) > record['files'] + 1:
                            raise ValueError('Prebuilt runtime exceeds declared extraction budget')
                        target = staged.joinpath(*path.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as stream, target.open('wb') as output:
                            shutil.copyfileobj(stream, output)
                        target.chmod(0o755 if member.mode & 0o111 else 0o644)
                checks = validate(staged)
                staged.rename(prefix)
    result.update(status='completed', external_dependencies={'postprocess_runtime': str(prefix)}, functional_checks=checks)
    return result
