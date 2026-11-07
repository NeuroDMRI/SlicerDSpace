"""Execute DTI stages (native Teem tensor fit, unchanged Slicer registration) with per-output receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.cache import related_files, validate_output
from utils.stages import StageCache, subject_lock

ROOT = Path(__file__).resolve().parents[1]
SCALARS = ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')
# Trace is retained in native space even when atlas inference needs only five maps.
TENSOR_SCALARS = SCALARS + ('Trace',)
TEEM_SCRIPT_VARIABLE = 'SLICERDSPACE_TEEM_DTI_SCRIPT'


def teem_script():
    """The shared Teem tensor-fit script: an explicit override, else the copy in this repository."""
    configured = os.environ.get(TEEM_SCRIPT_VARIABLE)
    script = Path(configured).expanduser() if configured else ROOT.parents[2] / 'shared/teem_dti.py'
    if not script.is_file():
        raise FileNotFoundError(f'Teem tensor script not found: {script} (set {TEEM_SCRIPT_VARIABLE})')
    return script


def pythonslicer(slicer_path):
    """PythonSlicer of the selected installation (F2).

    Linux and Windows: ``<root>/bin/PythonSlicer[.exe]``. macOS: ``Slicer.app/Contents/bin/PythonSlicer``, reached from
    the .app, its ``Contents`` folder or ``Contents/MacOS``. The backend runs standalone, so it does not import the
    repository's shared resolver; this list mirrors ``shared.external_process.find_pythonslicer``.
    """
    home = Path(slicer_path)
    for folder in (home / 'bin', home / 'Contents/bin', home.parent / 'bin', home, home / 'MacOS'):
        for name in ('PythonSlicer', 'PythonSlicer.exe'):
            if (folder / name).is_file():
                return folder / name
    raise FileNotFoundError(f'PythonSlicer not found in the Slicer installation: {home}')


def slicer_root(slicer_path):
    """Folder that holds ``lib/`` and ``bin/``: the path itself, or ``Contents`` for a macOS .app or ``Contents/MacOS``."""
    home = Path(slicer_path)
    return home.parent if home.name == 'MacOS' else home / 'Contents' if home.suffix == '.app' else home


def slicer_launcher(slicer_path):
    """Slicer executable of the selected installation (B9), as ``dti.sh`` finds it for the same folders.

    Linux and Windows: ``<root>/Slicer[.exe]``. macOS: ``Slicer.app/Contents/MacOS/Slicer``, reached from the .app,
    its ``Contents`` folder or ``Contents/MacOS``.
    """
    home = Path(slicer_path)
    root = slicer_root(home)
    for candidate in (home / 'Slicer', home / 'Slicer.exe', root / 'MacOS/Slicer', root / 'Slicer', root / 'Slicer.exe'):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return home / 'Slicer'  # not found: the command fails with this path in its error message


def substitute(part, replacements):
    """Swap a staged output path into one argument, also inside an ``Operation=path`` pair."""
    text = str(part)
    if text in replacements:
        return replacements[text]
    name, separator, value = text.partition('=')
    if separator and value in replacements:
        return f'{name}={replacements[value]}'
    return text


def run(args):
    folder = args.output_dir.resolve()
    stem = args.subject
    cache = StageCache(folder / '.stages.json', related_files, validate_output)
    slicer = [str(slicer_launcher(args.slicer_path)), '--launch']
    helper_env = dict(os.environ)
    for name, value in (('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'),
                        ('MKL_NUM_THREADS', '1'), ('NUMEXPR_NUM_THREADS', '1'),
                        ('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1'),
                        ('KMP_AFFINITY', 'disabled'), ('KMP_BLOCKTIME', '0'), ('OMP_WAIT_POLICY', 'PASSIVE')):
        helper_env.setdefault(name, value)

    def execute(key, command, inputs, outputs, helper=False, link=None, tool=None):
        def action(staged):
            replacements = dict(zip(map(str, outputs), map(str, staged)))
            if link is not None:
                (staged[0].parent / link.name).symlink_to(link.resolve())
            actual = [substitute(part, replacements) for part in command]
            subprocess.run(actual, check=True, env=tool_env if tool else helper_env if helper else None)
        implementation = [Path(__file__), ROOT / 'preprocessing/cache.py', ROOT / 'utils/stages.py']
        if tool:
            implementation += tool
        else:
            implementation += [Path(command[1])] if helper else [Path(command[0]), Path(command[2])]
        cache.run(key, list(inputs) + implementation, outputs, action, settings=list(map(str, command)))

    # PythonSlicer builds its own runtime; it must not inherit an outer Python home.
    tool_env = {name: value for name, value in os.environ.items() if name not in ('PYTHONHOME', 'PYTHONPATH')}
    dwi_header = folder / f'{stem}.nhdr'
    mask_header = folder / f'{stem}-mask.nhdr'
    converter = ROOT / 'preprocessing/conversion/nhdr_write.py'
    execute('dwi.header', [sys.executable, converter, '--nifti', args.dwi, '--bval', args.bval,
                          '--bvec', args.bvec, '--nhdr', dwi_header],
            [args.dwi, args.bval, args.bvec, ROOT / 'preprocessing/conversion/bval_bvec_io.py'],
            [dwi_header], helper=True, link=args.dwi)
    execute('mask.header', [sys.executable, converter, '--nifti', args.mask, '--nhdr', mask_header],
            [args.mask], [mask_header], helper=True, link=args.mask)
    tensor, b0 = (folder / f'{stem}-{kind}.nhdr' for kind in ('dti', 'b0'))
    scalars = SCALARS + (() if args.minimal else ('Trace',))
    scalar_outputs = [(kind, folder / f'{stem}-dti-{kind}.nhdr') for kind in TENSOR_SCALARS]
    # One native Teem process fits the tensor and writes every scalar map; the cache hashes the script.
    python, script = pythonslicer(args.slicer_path), teem_script()
    execute('tensor', [python, script, 'estimate', '--enumeration', 'WLS']
            + [item for kind, path in scalar_outputs for item in ('--scalar', f'{kind}={path}')]
            + [dwi_header, tensor, b0],
            [dwi_header], [tensor, b0] + [path for _, path in scalar_outputs], tool=[python, script])
    transform = folder / f'{stem}-b0ToAtlasT2.tfm'
    execute('registration', slicer + [args.core_cli / 'BRAINSFit', '--fixedVolume', args.reference_image,
                                     '--movingVolume', b0, '--linearTransform', transform, '--useRigid', '--useAffine'],
            [b0, args.reference_image], [transform])

    def registered(kind):
        source = args.mask if kind == 'mask' else folder / f'{stem}-dti-{kind}.nhdr'
        output = folder / (f'{stem}-mask-Reg.nii.gz' if kind == 'mask' else f'{stem}-dti-{kind}-Reg.nii.gz')
        execute(f'resample.{kind}', slicer + [args.core_cli / 'ResampleScalarVectorDWIVolume',
                '-i', 'nn' if kind == 'mask' else 'linear', source, '--Reference', args.reference_image,
                '--transformationFile', transform, output], [source, args.reference_image, transform], [output])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(registered, scalars + ('mask',)))
    if not args.minimal:
        for kind in ('FractionalAnisotropy', 'Trace', 'MinEigenvalue', 'MidEigenvalue'):
            source = folder / f'{stem}-dti-{kind}-Reg.nii.gz'
            mask = folder / f'{stem}-mask-Reg.nii.gz'
            output = folder / f'{stem}-dti-{kind}-Reg-NormMasked.nii.gz'
            execute(f'normalize.{kind}', [sys.executable, ROOT / 'preprocessing/normalize_dti.py',
                    '--input', source, '--mask', mask, '--output', output, '--flip', str(args.mask_flip)],
                    [source, mask], [output], helper=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    for name in ('output-dir', 'dwi', 'bval', 'bvec', 'mask', 'slicer-path', 'core-cli', 'reference-image'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dmri-cli', type=Path, help='deprecated; the tensor fit no longer uses SlicerDMRI')
    parser.add_argument('--mask-flip', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--minimal', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    with subject_lock(args.output_dir / '.dti.lock'):
        run(args)


if __name__ == '__main__':
    main()
