#!/usr/bin/env python3
"""Vendor an exact DDSurfer snapshot and refresh its external asset records."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy_if_changed(source, destination):
    """Keep identical large assets and their timestamps intact during source sync."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size and digest(destination) == digest(source):
        return False
    shutil.copy2(source, destination)
    return True


def synchronize_backend(upstream):
    upstream = upstream.resolve()
    recorded = ROOT / 'anatomy/surface/ddsurfer_backend/UPSTREAM.json'
    patched = sorted(json.loads(recorded.read_text()).get('local_patches', {})) if recorded.is_file() else []
    if patched:
        # C4: a sync would overwrite these files with upstream text and drop the record.
        raise ValueError('The vendored snapshot has local patches that a sync would overwrite: '
                         + ', '.join(patched) + '. Re-apply them by hand after syncing, or remove them from '
                         'UPSTREAM.json local_patches once upstream carries the change.')
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('Commit the upstream changes before creating a reproducible snapshot')
    backend = ROOT / 'anatomy/surface/ddsurfer_backend'
    files = []
    for directory in ('model', 'inference', 'preprocessing', 'postprocessing', 'utils', 'tools', 'tests'):
        files.extend(p for p in (upstream / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.txt', '.json', '.cmake'))
    files.extend(upstream / name for name in
                 ('DDSurfer_predict.py', 'run_ddsurfer_pipeline.py', 'run_ddsurfer_pipeline.sh',
                  'environment.yml', 'requirements.txt', 'README.md'))
    files.extend(upstream / name for name in ('README.zh-CN.md', 'CITATION.cff', 'LICENSE', 'utils/LICENSE.pytorch3d')
                 if (upstream / name).is_file())
    overview = upstream / 'assets/ddsurfer-overview.png'
    if overview.is_file():
        files.append(overview)
    hashes = {}
    for source in sorted(files):
        relative = source.relative_to(upstream)
        if relative == Path('LICENSE'):
            relative = Path('LICENSE.txt')
        destination = backend / relative
        copy_if_changed(source, destination)
        hashes[str(relative)] = digest(source)
    # Remove only files recorded by our previous snapshot, never unrelated files.
    record = backend / 'UPSTREAM.json'
    if record.exists():
        for obsolete in set(json.loads(record.read_text())['files']) - hashes.keys():
            (backend / obsolete).unlink(missing_ok=True)
    record.write_text(json.dumps({'repository': 'DDSurfer', 'commit': revision, 'files': hashes}, indent=2) + '\n')
    return revision, len(hashes)


def synchronize(upstream, assets, *, source_only=False):
    upstream = upstream.resolve()
    revision, source_count = synchronize_backend(upstream)

    manifest_path = ROOT / 'models/assets_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if source_only:
        manifest['ddsurfer_upstream_commit'] = revision
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(commit=revision, source_files=source_count, assets_changed=False), indent=2))
        return
    assets = assets.resolve()
    legacy_prefixes = ('Resources/Data/ckpts/', 'Resources/Data/template/', 'surface/template/')
    manifest['asset_records'] = {key: value for key, value in manifest['asset_records'].items()
                                 if not key.startswith(legacy_prefixes)}
    required = []
    for folder, names in (
        ('weights', ('manifest.json', 'SHA256SUMS', 'ddsurfer_lh_wm.pt', 'ddsurfer_rh_wm.pt',
                     'ddsurfer_lh_pial.pt', 'ddsurfer_rh_pial.pt')),
        ('template', ('ddsurfer_lh_template.obj', 'ddsurfer_rh_template.obj',
                      '100HCP-population-mean-T2-1mm.nii.gz')),
    ):
        for name in names:
            source = upstream / folder / name
            relative = f'surface/{folder}/{name}'
            destination = assets / relative
            copy_if_changed(source, destination)
            required.append(relative)
            manifest['asset_records'][relative] = {
                'sha256': digest(source), 'size_bytes': source.stat().st_size,
                'source_ids': ['slicerdsurfer_models_release'],
            }
    manifest['ddsurfer_upstream_commit'] = revision
    profile = manifest['profiles']['surface']
    profile.update(description='Manifest-configured DDSurfer TANet weights, transformed templates and 1-mm atlas.',
                   asset_groups=['DDSurfer TANet surface model weights', '1-mm registration atlas',
                                 'transformed surface initialization meshes'],
                   terms_notices=['DDSurfer weights, transformed templates and the 1-mm atlas are mirrored from '
                                  'the recorded upstream revision. Public redistribution and DOI-archive terms '
                                  'require author confirmation before publication.',
                                  'Subject-level diffusion data are not bundled with this profile.'],
                   include_patterns=['surface/weights/**', 'surface/template/**'], required_paths=required)
    # The old registration atlas remains available for DDParcel, which is not migrated here.
    parcel = manifest['profiles']['parcellation']
    atlas = 'Resources/Data/100HCP-population-mean-T2.nii.gz'
    for field in ('required_paths', 'include_patterns'):
        if atlas not in parcel[field]:
            parcel[field].append(atlas)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    from download_model_assets import load_manifest, resolve_asset_plan
    from datetime import datetime, timezone
    plan = resolve_asset_plan(load_manifest(manifest_path), 'core')
    records = [dict(path=p, size_bytes=(assets / p).stat().st_size, sha256=digest(assets / p),
                    source_ids=manifest['asset_records'][p].get('source_ids', list(plan.source_ids))) for p in plan.required_paths]
    mirror = dict(kind='slicerdsurfer_model_asset_mirror', version=1,
                  schema='slicerddsurfer.model_assets.mirror.v1',
                  created_utc=datetime.now(timezone.utc).isoformat(), profile='core',
                  resolved_profiles=list(plan.resolved_profiles), asset_groups=list(plan.asset_groups),
                  terms_notices=list(plan.terms_notices), mirror_layout='repository-relative paths',
                  copy_mode='copy', dry_run=False, ddsurfer_upstream_commit=revision, records=records)
    (assets / 'asset_mirror_manifest.json').write_text(json.dumps(mirror, indent=2) + '\n')
    (assets / 'asset_mirror_index.tsv').write_text('path\tsize_bytes\tsha256\n' +
        ''.join(f"{r['path']}\t{r['size_bytes']}\t{r['sha256']}\n" for r in records))
    print(json.dumps({'commit': revision, 'source_files': source_count, 'assets': len(required),
                      'asset_root': str(assets)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--source-only', action='store_true', help='Do not modify model files or mirror indexes')
    args = parser.parse_args()
    if not args.source_only and args.asset_root is None:
        parser.error('--asset-root is required unless --source-only is selected')
    synchronize(args.upstream, args.asset_root, source_only=args.source_only)
