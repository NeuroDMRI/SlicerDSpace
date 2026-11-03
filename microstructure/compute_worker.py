"""Private subprocess entry for independent scalar fits, never ROI publication."""

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def encode(value):
    if isinstance(value, Path):
        return {'__path__': str(value)}
    if is_dataclass(value):
        return {'__dataclass__': type(value).__name__,
                'fields': {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def decode(value):
    from shared.pipeline import RawInputBundle, NativeDTIBundle
    from microstructure.pipeline import MicrostructureConfig

    if not isinstance(value, dict):
        return value
    if '__path__' in value:
        return Path(value['__path__'])
    if '__dataclass__' in value:
        classes = {item.__name__: item for item in (RawInputBundle, NativeDTIBundle, MicrostructureConfig)}
        return classes[value['__dataclass__']](**{key: decode(item) for key, item in value['fields'].items()})
    return {key: decode(item) for key, item in value.items()}


def run(stage, config):
    from microstructure import pipeline as p
    paths = p._output_paths(config.subject_id, config.output_dir)
    if stage == 'freewater':
        if config.force_recompute or not p._free_water_outputs_current(config, paths):
            p._compute_free_water_maps(config=config, paths=paths, log=print)
            p._write_json(paths['free_water_manifest'], p._expected_free_water_manifest(config))
    elif stage == 'noddi':
        if config.force_recompute or not p._noddi_outputs_current(config, paths):
            p._compute_noddi_maps(config=config, paths=paths, log=print)
            p._write_json(paths['noddi_manifest'], p._expected_noddi_manifest(config))
    elif stage == 'dki':
        from microstructure.dki import DkiConfig, run_dki
        run_dki(DkiConfig(subject_id=config.subject_id, raw_input_bundle=config.raw_input_bundle,
            output_dir=config.output_dir/'dki', fit_method=config.dki_fit_method,
            matlab_executable=config.dki_matlab_executable, mkcurve_root=config.dki_mkcurve_root,
            force_recompute=config.force_recompute, **(config.dki_execution or {})), log=print)
    else:
        raise ValueError('Unknown private scalar stage')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('freewater','noddi','dki'), required=True)
    args = parser.parse_args()
    run(args.stage, decode(json.loads(args.config.read_text(encoding='utf-8'))))
