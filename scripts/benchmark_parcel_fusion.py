"""Paired full-resolution trained DDParcel batches, not whole-volume timing."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plane', choices=('Axial', 'Coronal', 'Sagittal'), action='append')
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or args.threads < 1 or not 1 <= args.batch_size <= 16 or args.output.exists():
        parser.error('Use >=3 repeats, positive threads, batch 1-16 and a fresh output path')

    import numpy as np
    import torch
    from anatomy.parcellation.models import networks
    from anatomy.parcellation.data_loader.augmentation import ToTensorTest
    from anatomy.parcellation.data_loader.load_neuroimaging_data import (
        load_and_conform_image, transform_axial, transform_sagittal)

    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU substitution')
    spec = importlib.util.spec_from_file_location('anatomy.parcellation.models._fusion_reference', args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    inputs = sorted(args.input.glob('*-dti-*-NormMasked.nii.gz'))
    if len(inputs) != 4:
        raise ValueError('Expected exactly the four real registered features')
    sources = [args.reference, Path(networks.__file__), Path(__file__),
               Path(networks.sm.__file__), *inputs, *sorted(args.weights.rglob('*.pkl'))]
    hashes = {str(path.resolve()): digest(path) for path in sources}
    volumes = [load_and_conform_image(str(path), logger=logging.getLogger('parcel.benchmark'))[2]
               for path in inputs]
    records = []
    report = dict(status='running', measured_at=datetime.now(timezone.utc).isoformat(),
                  scope='Complete trained network forward on selected full-resolution thick-slice batches; excludes loading, full-volume fusion, cleanup and export',
                  comparison='Same trained weights and arithmetic, preceding implementation versus single concatenation',
                  source_sha256=hashes, environment=dict(python=sys.version, torch=str(torch.__version__),
                  platform=platform.platform(), device=args.device, threads=args.threads,
                  cpu=platform.processor(), shared_host=True,
                  gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  thread_env={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
                  records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output)

    save()
    try:
        for plane in args.plane or ('Axial', 'Coronal', 'Sagittal'):
            key = plane.lower()
            params = dict(num_channels=7, num_filters=64, num_modality=4,
                          backbone_model=sorted(str(p) for p in (args.weights/'backbones').glob('*'+key+'.pkl')),
                          kernel_h=5, kernel_w=5, stride_conv=1, pool=2, stride_pool=2,
                          num_classes=54 if plane == 'Sagittal' else 82, kernel_c=1,
                          use_cuda=args.device == 'cuda')
            if len(params['backbone_model']) != 4:
                raise ValueError('Expected four backbone checkpoints per view')
            checkpoint = torch.load(args.weights/f'Fused-Unet-v3-noMax-{key}.pkl',
                                    map_location='cpu', weights_only=True)['model_state_dict']
            state = {name.removeprefix('module.'): value for name, value in checkpoint.items()}
            models = {}
            for name, module in (('reference', reference), ('optimized', networks)):
                model = module.FastSurferCNN_Fuse_Unet_v3_extended(copy.deepcopy(params))
                backbones = copy.deepcopy(model.FastSurferCNNs)
                model.load_state_dict(state)
                model.FastSurferCNNs = backbones
                models[name] = model.to(args.device).eval()

            batches = []
            for offset in range(args.batch_size):
                channels = []
                for volume in volumes:
                    view = transform_axial(volume) if plane == 'Axial' else (
                        transform_sagittal(volume) if plane == 'Sagittal' else volume)
                    center = view.shape[2]//2 + offset
                    indices = np.clip(np.arange(center-3, center+4), 0, view.shape[2]-1)
                    channels.append(view[:, :, indices])
                batches.append(ToTensorTest()(np.concatenate(channels, axis=2)))
            data = torch.from_numpy(np.stack(batches)).to(args.device)

            def run(name):
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.no_grad():
                    output = models[name](data)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                # Transfer/hash outside the timed region, identically for both.
                arrays = [value.detach().cpu().numpy() for value in (output[0], *output[1])]
                fingerprints = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
                if not all(np.isfinite(array).all() for array in arrays):
                    raise ValueError('Nonfinite trained-network outputs')
                return elapsed, fingerprints

            _, expected = run('reference')
            _, actual = run('optimized')
            assert actual == expected, 'Warmup logits differ'
            times = dict(reference=[], optimized=[])
            for iteration in range(args.repeats):
                order = ('reference', 'optimized') if iteration % 2 == 0 else ('optimized', 'reference')
                for name in order:
                    elapsed, actual = run(name)
                    assert actual == expected, 'Trained network logits differ'
                    times[name].append(elapsed)
            old, new = (statistics.median(times[name]) for name in ('reference', 'optimized'))
            row = dict(plane=plane, input_shape=list(data.shape), warmups_per_variant=1,
                       repeats=args.repeats, times_seconds=times,
                       reference_median_seconds=old, optimized_median_seconds=new,
                       speed_ratio=old/new, time_reduction_percent=100*(1-new/old),
                       exact_output_parity=True, all_returned_logits_sha256=expected)
            records.append(row)
            print(json.dumps(row), flush=True)
            save()
            del models, model, backbones, checkpoint, state, data
        assert all(digest(path) == value for path, value in hashes.items()), 'Code/inputs/weights changed'
        report.update(status='passed', source_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
