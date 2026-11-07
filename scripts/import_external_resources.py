#!/usr/bin/env python3
"""Import explicitly selected resources with pinned hashes; never remove originals.

This is a maintainer/offline acquisition tool, not an installer for Python,
executables, source trees, subject data or arbitrary filesystem contents.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/external_assets_manifest.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def import_file(source, destination, expected):
    if source.is_symlink() or not source.is_file():
        raise ValueError('A regular, explicitly provided source is required: ' + str(source))
    if source.stat().st_size != expected['size_bytes'] or digest(source) != expected['sha256']:
        raise ValueError('Source checksum mismatch: ' + str(source))
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError('Refusing a symlink destination: ' + str(destination))
    if destination.exists():
        if destination.is_file() and digest(destination) == expected['sha256']:
            return 'reused'
        raise FileExistsError('Refusing to overwrite a different resource: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.resource-import-', dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        shutil.copyfile(source, staged)
        if digest(staged) != expected['sha256']:
            raise RuntimeError('Resource changed while copying: ' + str(source))
        staged.replace(destination)
    return 'imported'


def _symlink_inside(target, base):
    """True when ``target`` or a folder between it and ``base`` is a symlink.

    ``base`` is a resolved folder. A symlink at or above ``base`` (for example ``/var`` on
    macOS) is the user's own folder layout and is not checked.
    """
    path = Path(target)
    while path != base and base in path.parents:
        if path.is_symlink():
            return True
        path = path.parent
    return False


def extract_org(archive, destination):
    """Extract only the pinned atlas root; omit macOS resource-fork metadata."""
    root_name = archive.stem
    records = []
    destination = Path(destination)
    # Resolve the folder that holds the destination once: an ancestor symlink (/var on macOS) is
    # allowed, but the destination itself and everything below it must not be a symlink.
    anchor = destination.parent.resolve()
    destination = anchor / destination.name
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > 4 * 1024**3:
            raise ValueError('ORG archive exceeds extraction limits')
        for item in entries:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                    or ':' in item.filename or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe ORG archive member: ' + item.filename)
            if not path.parts or path.parts[0] == '__MACOSX' or any(part.startswith('._') for part in path.parts) or item.is_dir():
                continue
            if path.parts[0] != root_name:
                raise ValueError('Unexpected ORG archive root: ' + item.filename)
            target = destination.joinpath(*path.parts)
            if _symlink_inside(target, anchor):
                raise ValueError('Refusing symlink extraction target')
            with zipped.open(item) as stream:
                expected = hashlib.file_digest(stream, 'sha256').hexdigest()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.org-import-', dir=target.parent) as temporary:
                    staged = Path(temporary) / target.name
                    with zipped.open(item) as stream, staged.open('wb') as output:
                        shutil.copyfileobj(stream, output)
                    if digest(staged) != expected:
                        raise RuntimeError('Extracted ORG checksum mismatch')
                    staged.replace(target)
            elif digest(target) != expected:
                raise FileExistsError('Existing ORG file differs: ' + str(target))
            records.append({'path': path.as_posix(), 'sha256': expected, 'size_bytes': item.file_size})
    return records


def extract_indexed_archive(root, definition):
    """Reassemble verified mirror parts and extract only the pinned file inventory."""
    root = Path(root).resolve()  # resolve the root once: an ancestor symlink (/var on macOS) is allowed
    destination = root / definition["directory"]
    index = json.loads((root / definition["extracted_index"]).read_text(encoding="utf-8"))
    files = {item["path"]: item for item in index["files"]}
    if not files or len(files) > 5000 or sum(item["size_bytes"] for item in files.values()) > 64 * 1024**3:
        raise ValueError("Atlas index exceeds extraction limits")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe atlas index member: " + name)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".atlas-import-", dir=destination) as temporary:
        archive = Path(temporary) / "runtime.zip"
        with archive.open("wb") as output:
            for part in definition["archive_parts"]:
                with (root / part).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if digest(archive) != definition["archive_sha256"]:
            raise ValueError("Reassembled atlas archive checksum mismatch")
        with zipfile.ZipFile(archive) as zipped:
            entries = zipped.infolist()
            if len(entries) != len(files) or {item.filename for item in entries} != set(files):
                raise ValueError("Atlas archive differs from its pinned extraction inventory")
            for item in entries:
                record = files[item.filename]
                if item.file_size != record["size_bytes"] or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Atlas member size/type mismatch: " + item.filename)
                target = destination.joinpath(*PurePosixPath(item.filename).parts)
                if _symlink_inside(target, root):
                    raise ValueError("Refusing symlink extraction target")
                if target.exists():
                    if not target.is_file() or digest(target) != record["sha256"]:
                        raise FileExistsError("Existing atlas file differs: " + str(target))
                    continue
                staged = Path(temporary) / "member"
                with zipped.open(item) as source, staged.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if digest(staged) != record["sha256"]:
                    raise ValueError("Atlas member checksum mismatch: " + item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(target)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--pnl-prefix', type=Path)
    parser.add_argument('--tractcloud-data', type=Path)
    parser.add_argument('--tractcloud-license', type=Path)
    parser.add_argument('--synthstrip-model', type=Path)
    parser.add_argument('--wma-downloads', type=Path)
    parser.add_argument('--wma-license', type=Path)
    parser.add_argument('--execute', action='store_true', help='Copy verified resources; otherwise print a plan only')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    # Resolve the chosen root once (macOS keeps /tmp and /var behind symlinks).
    # import_file then refuses any link in the destination path. extract_org allows a link at or above
    # the destination's parent folder and refuses a link at or below the destination.
    root = args.asset_root.expanduser().resolve()
    manifest = json.loads(MANIFEST.read_text())
    mapping = {}
    pnl_torch = bool(args.pnl_prefix and (args.pnl_prefix / 'pnl-models/weights.json').is_file())
    pnl_directory = 'brain_mask/pnl_cnn_torch/' if pnl_torch else 'brain_mask/pnl_cnn/'
    for relative in manifest['asset_records']:
        source = None
        if args.pnl_prefix and relative.startswith(pnl_directory):
            if pnl_torch and relative.endswith('/LICENSE.txt'):
                licenses = list(args.pnl_prefix.glob('lib/python*/site-packages/pnl_brain_mask-*.dist-info/licenses/LICENSE'))
                if len(licenses) != 1:
                    parser.error('A single installed PNLBrainMask-Torch license is required for import')
                source = licenses[0]
            else:
                source = args.pnl_prefix / ('pnl-source/LICENSE' if relative.endswith('/LICENSE.txt') else 'pnl-models/' + Path(relative).name)
        elif args.tractcloud_data and relative.startswith('bundles/tractcloud/'):
            source = args.tractcloud_license if relative.endswith('/LICENSE.txt') else args.tractcloud_data / relative.removeprefix('bundles/tractcloud/')
        elif args.synthstrip_model and relative == 'brain_mask/synthstrip/synthstrip.1.pt':
            source = args.synthstrip_model
        elif args.wma_downloads and relative.startswith('bundles/wma/'):
            source = args.wma_license if relative.endswith('/LICENSE.txt') else args.wma_downloads / Path(relative).name
        if source is not None:
            mapping[relative] = source
    if args.tractcloud_data and not args.tractcloud_license or args.wma_downloads and not args.wma_license:
        parser.error('A complete upstream license is required for TractCloud and WMA imports')
    record = {'kind': 'slicerdsurfer_optional_resource_import', 'execute': args.execute,
              'source_files_retained': True, 'records': [], 'scope': 'Byte-preserving resource organization, not scientific QC or completed remote upload'}
    for relative, source in mapping.items():
        expected = manifest['asset_records'][relative]
        state = import_file(source.expanduser(), root / relative, expected) if args.execute else 'planned'
        record['records'].append({'path': relative, 'sha256': expected['sha256'], 'size_bytes': expected['size_bytes'],
                                  'status': state, 'redistribution': expected['redistribution']})
        print(state + ': ' + relative, flush=True)
    if args.execute and args.wma_downloads:
        files = []
        for relative in mapping:
            if relative.startswith('bundles/wma/archives/'):
                files.extend(extract_org(root / relative, root / 'bundles/wma'))
        lock = root / 'bundles/wma/extracted_resource_index.json'
        if lock.is_symlink():
            raise ValueError('Refusing a symlink resource index')
        lock.write_text(json.dumps({'kind': 'org_archive_extraction_index', 'files': files}, indent=2) + '\n')
        record['org_extracted_files'] = len(files)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
