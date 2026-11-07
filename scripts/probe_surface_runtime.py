#!/usr/bin/env python3
"""Read-only native/FreeSurfer checks in an isolated Python import namespace."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'anatomy/surface/ddsurfer_backend'
sys.path.insert(0, str(ROOT))


def probe(args):
    sys.path.insert(0, str(BACKEND))
    from postprocessing.core.runtime import validate_runtime, selection
    from shared.surface_license import resolve_surface_license, usable_license

    home = args.home.expanduser().resolve()
    license_file = resolve_surface_license(
        args.license_file, freesurfer_home=home if args.backend == 'freesurfer' else None,
    )
    result = dict(backend=args.backend, home=str(home), components_available=False,
                  license_available=usable_license(license_file),
                  build_required=False, available=False)
    try:
        if args.backend == 'native':
            record = validate_runtime(home, args.hemi, args.atlases)
            result['source_revision'] = record['source_revision']
        else:
            hemis, atlases = selection(args.hemi, args.atlases)
            metric = 'mris_place_surface' if os.access(home / 'bin/mris_place_surface', os.X_OK) else 'mris_thickness'
            names = ('mris_curvature', 'mris_inflate', 'mris_sphere', 'mris_register', metric,
                     'mri_label2label', 'mri_surf2surf')
            required = [home / 'bin' / name for name in names]
            for hemi in hemis:
                required += [home / 'subjects/fsaverage/label' / f'{hemi}.cortex.label',
                             home / 'subjects/fsaverage/surf' / f'{hemi}.sphere.reg',
                             home / 'average' / f'{hemi}.average.curvature.filled.buckner40.tif']
                required += [home / 'subjects/fsaverage/label' / f'{hemi}.{atlas}.annot' for atlas in atlases]
            if any(not path.is_file() or not path.stat().st_size for path in required):
                raise FileNotFoundError('Configured FreeSurfer commands or selected atlas files are incomplete')
            if any(not os.access(home / 'bin' / name, os.X_OK) for name in names):
                raise ValueError('Configured FreeSurfer commands are not executable')
        result['components_available'] = True
    except (OSError, ValueError, RuntimeError) as exc:
        result['error'] = str(exc)
        result['build_required'] = args.backend == 'native' and not home.exists()
    result['available'] = result['components_available'] and result['license_available']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--backend', choices=('native', 'freesurfer'), required=True)
    parser.add_argument('--license-file', type=Path)
    parser.add_argument('--hemi', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--atlases', default='aparc,aparc.a2009s')
    args = parser.parse_args()
    try:
        record = probe(args)
    except Exception as exc:
        record = dict(components_available=False, license_available=False, available=False,
                      backend=args.backend, build_required=False, error=str(exc))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
