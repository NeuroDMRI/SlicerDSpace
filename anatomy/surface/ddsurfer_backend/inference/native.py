"""Transform scanner-RAS atlas meshes to native RAS using the registration pull map."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import SimpleITK as sitk

from utils.files import atomic_json, sha256_file
from utils.surface_io import load_mesh, write_obj


def atlas_ras_to_native_ras(vertices, transform):
    # BRAINSFit: fixed=atlas, moving=b0. The saved ITK map is atlas LPS -> native LPS.
    # Both mesh interfaces use RAS, so flip x/y before AND after TransformPoint.
    flip = np.array([-1., -1., 1.], dtype=np.float64)
    pts_lps = np.asarray(vertices, dtype=np.float64) * flip
    if hasattr(transform, 'IsLinear') and transform.IsLinear():
        origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=np.float64)
        basis = np.column_stack([
            np.asarray(transform.TransformPoint(axis), dtype=np.float64) - origin
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ])
        native_lps = pts_lps @ basis.T + origin
    else:
        native_lps = np.asarray([transform.TransformPoint(tuple(v)) for v in pts_lps], dtype=np.float64)
    return native_lps * flip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--pred-root', required=True, type=Path)
    parser.add_argument('--predict-mode', choices=('wm','all'), default='all')
    parser.add_argument('--output-dir', type=Path, help='Final native surface directory.')
    parser.add_argument('--metadata-dir', type=Path, help='Directory for detailed conversion records.')
    parser.add_argument('--dti-dir', type=Path, help='Native DTI directory for optional surface refinement.')
    parser.add_argument('--mask', type=Path, help='Native brain mask for optional surface refinement.')
    parser.add_argument('--refine', action='store_true', help='Enable native-space surface refinement.')
    parser.add_argument('--threads', type=int, default=8, help='CPU thread budget for surface refinement.')
    args = parser.parse_args(argv)
    transform_path = args.data_root/args.subject/f'{args.subject}-b0ToAtlasT2.tfm'
    transform = sitk.ReadTransform(str(transform_path))
    directory = args.output_dir if args.output_dir is not None else args.pred_root/'native'/args.subject
    directory.mkdir(parents=True,exist_ok=True)
    if args.output_dir is not None:
        logs = directory.parent / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        saved_transform = logs / 'b0ToAtlasT2.tfm'
        shutil.copyfile(transform_path, saved_transform)
    else:
        saved_transform = transform_path
    entries = []
    for hemi,short in [('left','lh'),('right','rh')]:
        for surface in (['wm','pial'] if args.predict_mode=='all' else ['wm']):
            source=args.pred_root/'mni'/args.subject/f'{args.subject}_predicted_{surface}_surface_{hemi}.obj'
            mesh=load_mesh(source)
            vertices=atlas_ras_to_native_ras(mesh.vertices,transform)
            if not np.isfinite(vertices).all():raise ValueError('Nonfinite native vertices')
            stem = f'{short}.{"white" if surface == "wm" else "pial"}'
            entries.append((stem, surface, short, source, vertices, np.asarray(mesh.faces)))
    refinement_info = dict(requested=bool(args.refine), applied=False, backend='disabled')
    if args.refine and args.dti_dir is not None and args.mask is not None:
        from inference.surface_refine import refine_bilateral_surfaces, refinement_library_source, resolve_dti_paths
        if resolve_dti_paths(args.dti_dir, args.subject, args.mask) is None:
            refinement_info = dict(requested=True, applied=False, backend='missing_dti_inputs')
            print(f'[DDSurfer] Native-space surface refinement skipped: missing native DTI inputs in {args.dti_dir}', flush=True)
        else:
            before_map = {stem: vertices.copy() for stem, _, _, _, vertices, _ in entries}
            refined = refine_bilateral_surfaces(
                {stem: (vertices, faces) for stem, _, _, _, vertices, faces in entries},
                args.dti_dir, args.subject, args.mask, nthreads=args.threads,
            )
            lib_src = refinement_library_source()
            applied = lib_src != 'unavailable' and any(
                not np.array_equal(before_map[stem], refined[stem]) for stem in before_map
            )
            refinement_info = dict(requested=True, applied=bool(applied), backend=lib_src)
            entries = [(stem, surface, short, source, refined[stem], faces)
                       for stem, surface, short, source, _, faces in entries]
            print(f'[DDSurfer] Native-space surface refinement applied={applied} (engine={lib_src}, threads={args.threads})', flush=True)
    for stem, surface, short, source, vertices, faces in entries:
        output=directory/(f'{stem}.obj' if args.output_dir is not None else f'{args.subject}_predicted_{surface}_{short}.obj')
        error = write_obj(output, vertices, faces)
        metadata_path = (args.metadata_dir / f'{stem}.json' if args.metadata_dir is not None else
                         logs / f'{stem}.json' if args.output_dir is not None else output.with_suffix('.json'))
        atomic_json(metadata_path,dict(coordinate_space='native_scanner_RAS_mm',
                    transform=str(saved_transform),transform_sha256=sha256_file(saved_transform),
                    transform_direction='atlas_LPS_to_native_LPS_pull_map',source_sha256=sha256_file(source),
                    output_sha256=sha256_file(output), max_export_error_mm=error,
                    surface_refinement=refinement_info))
        print(f'Saved {output} (native RAS mm)',flush=True)


if __name__=='__main__':
    main()
